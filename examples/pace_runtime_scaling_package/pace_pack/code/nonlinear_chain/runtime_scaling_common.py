"""
Shared helpers for the nonlinear_chain fault-scenario runtime-scaling
scripts (unrefined_scenario_scaling.py, refined_scenario_scaling.py) and
their companion plot script (plot_scenario_runtime_scaling.py).

Unlike integrator_chain's runtime sweeps (which fix the scenario set and
vary the state dimension N), these scripts fix N and vary the NUMBER OF
FAULT SCENARIOS instead, via nonlinear_chain_separating_input.create_scenarios's
independently-configurable num_actuator_faults / num_sensor_faults.

Resource guard rails
---------------------
The all-pairs separation loss makes JIT-compile cost grow roughly cubically
in the number of scenario pairs (measured: N=10, single-segment grad_fn
compile alone went 0.7s @ 3 scenarios -> 2.7s @ 7 -> 42s @ 11 -> did not
finish in 90s @ 15 -- and a follow-on probe at 21 scenarios pushed a single
process to >7.7GB RSS and climbing before being killed after ~10 minutes,
nearly freezing the host machine). To let the 3-21 sweep run to completion
anyway without risking that again, `run_worker_with_limits` executes each
scenario count as its OWN subprocess under a wall-clock timeout AND an
RSS-polling memory watchdog -- a point that would blow past either bound
gets killed and recorded as "timeout"/"oom" in the CSV instead of taking the
whole machine down with it. (Linux's RLIMIT_RSS is a no-op on modern
kernels and RLIMIT_AS is unreliable for JAX/XLA processes, which reserve
large virtual address ranges independent of actual physical usage -- hence
polling actual RSS from /proc rather than using an rlimit.)
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jax

_HERE = Path(__file__).resolve().parent

#: Prefix a worker subprocess must print before its one JSON result line, so
#: the driver can pick that line out of stdout even if JAX/XLA has emitted
#: other (non-JSON) chatter to stdout.
RESULT_JSON_PREFIX = "RESULT_JSON:"


def split_fault_budget(total_scenarios: int, N: int) -> Tuple[int, int]:
    """Split a target TOTAL scenario count into (num_actuator_faults,
    num_sensor_faults) for create_scenarios, so that
    `1 + Ka + Ks == total_scenarios` (1 for the Nominal scenario).

    Splits the remaining budget `total_scenarios - 1` as evenly as possible
    between Ka and Ks (Ka gets the extra one when the budget is odd), which
    is what lets every integer total from `3` (Ka=Ks=1) to `1 + 2*N`
    (Ka=Ks=N) be hit exactly for a fixed N -- e.g. N=10 covers the full
    3..21 range used by the scenario-scaling sweep scripts.

    Raises ValueError if the requested total isn't achievable with this N
    (i.e. Ka or Ks would need to fall outside [1, N]).
    """
    budget = total_scenarios - 1
    if budget < 2:
        raise ValueError(
            f"total_scenarios={total_scenarios} needs budget>=2 (Ka>=1 and Ks>=1); "
            f"got budget={budget}"
        )
    Ka = -(-budget // 2)   # ceil(budget / 2)
    Ks = budget - Ka
    if not (1 <= Ka <= N and 1 <= Ks <= N):
        raise ValueError(
            f"total_scenarios={total_scenarios} not achievable with N={N}: "
            f"would need Ka={Ka}, Ks={Ks}, both must be in [1, {N}]"
        )
    return Ka, Ks


def timeit_run(jitted_fn, args, kwargs=None, num_repeats: int = 100) -> float:
    """Average steady-state run time (seconds/call) over num_repeats calls.

    `jitted_fn` must already be compiled (i.e. this is not its first call)
    so that no compilation cost leaks into the measurement. Ported from
    integrator_chain/refinement_demo.py's `_timeit_run` -- see that module's
    docstring for why calls are dispatched back-to-back with a single
    block_until_ready at the end rather than blocking per-call.
    """
    kwargs = kwargs or {}
    t0 = time.perf_counter()
    out = None
    for _ in range(num_repeats):
        out = jitted_fn(*args, **kwargs)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / num_repeats


def memory_metric_kb(mem: Dict[str, float]) -> float:
    """Reduce a time_jit-style memory snapshot dict to one scalar (KB) for
    a CSV column: prefer GPU peak/current bytes-in-use if present, else fall
    back to the CPU RSS snapshot (`_memory_snapshot`'s only key when no GPU
    device stats are available -- the common case in this project, which
    defaults to JAX_PLATFORMS=cpu)."""
    for key in ("peak_bytes_in_use", "bytes_in_use"):
        if key in mem:
            return mem[key] / 1024.0
    if "ru_maxrss_kb" in mem:
        return mem["ru_maxrss_kb"]
    return float("nan")


def append_csv_row(csv_path: Path, row: dict):
    """Append one row to csv_path, writing the header on first write.

    Every call for a given csv_path must pass a dict with the SAME set of
    keys in the SAME order (the scaling scripts guarantee this by always
    building rows through one canonical field list, `ok` and failure rows
    alike) -- csv.DictWriter fixes its fieldnames from the first row it
    sees and errors on any later row with a key outside that set.
    """
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _read_rss_kb(pid: int) -> float:
    """Current resident set size (KB) of `pid`, via /proc -- 0.0 if the
    process has already exited or /proc is unavailable (non-Linux)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, OSError):
        pass
    return 0.0


def run_worker_with_limits(
    cmd: List[str],
    timeout_s: float,
    mem_limit_mb: float,
    poll_interval_s: float = 0.5,
) -> dict:
    """Run `cmd` as a subprocess, killing it the moment it exceeds
    `timeout_s` wall-clock or `mem_limit_mb` resident memory (polled from
    /proc every `poll_interval_s`) -- see module docstring for why.

    `cmd` must be a worker invocation that prints exactly one
    `RESULT_JSON:<json>` line to stdout on success (see the scaling
    scripts' `--worker` mode) and exits 0.

    Returns a dict:
      status        : 'ok' | 'timeout' | 'oom' | 'error'
      elapsed_s     : wall-clock time until the process ended/was killed
      peak_rss_mb   : highest RSS observed during polling
      data          : the parsed RESULT_JSON payload (status == 'ok' only)
      error         : short diagnostic string (status != 'ok' only)
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.perf_counter()
    peak_rss_kb = 0.0
    status = "ok"

    while proc.poll() is None:
        rss_kb = _read_rss_kb(proc.pid)
        peak_rss_kb = max(peak_rss_kb, rss_kb)
        elapsed = time.perf_counter() - t0
        if rss_kb > mem_limit_mb * 1024:
            status = "oom"
            proc.kill()
            break
        if elapsed > timeout_s:
            status = "timeout"
            proc.kill()
            break
        time.sleep(poll_interval_s)

    try:
        stdout, stderr = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=15)

    elapsed = time.perf_counter() - t0
    result = {"status": status, "elapsed_s": elapsed, "peak_rss_mb": peak_rss_kb / 1024.0, "data": None}

    if status == "ok":
        if proc.returncode != 0:
            result["status"] = "error"
            result["error"] = f"worker exited {proc.returncode}: {stderr[-500:]}"
        else:
            payload = None
            for line in stdout.splitlines():
                if line.startswith(RESULT_JSON_PREFIX):
                    payload = json.loads(line[len(RESULT_JSON_PREFIX):])
            if payload is None:
                result["status"] = "error"
                result["error"] = "no RESULT_JSON line in worker stdout"
            else:
                result["data"] = payload
    else:
        result["error"] = (
            f"killed ({status}): elapsed={elapsed:.1f}s peak_rss={peak_rss_kb / 1024:.1f}MB"
        )

    return result
