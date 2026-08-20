"""
Unrefined-multistep fault-scenario runtime/memory scaling sweep for the
decoupled cubic-drift chain (nonlinear_chain_separating_input.py).

Fixes the state dimension at N=10 and sweeps the TOTAL number of fault
scenarios from 3 to 21 (Nominal + Ka ActuatorFault_i + Ks SensorFault_i,
Ka=Ks split as evenly as possible via runtime_scaling_common.split_fault_budget
-- see that module's docstring: N=10 makes every integer total in [3, 21]
exactly achievable, from Ka=Ks=1 up to Ka=Ks=N=10).

For each scenario count, times the fully GPU-vmapped multistart optimizer
(`optimize_multistep_gpu` -- the "heavily runtime-optimized" path: jitted,
multi-restart via vmap, checkpointed loop bodies) via `time_jit` (separates
first-call JIT compile time from steady-state run time) plus a timeit-style
average over many post-compilation calls.

Each scenario count runs in its OWN subprocess (`--worker` mode below) under
a wall-clock timeout and an RSS memory watchdog (see
runtime_scaling_common.run_worker_with_limits) -- the all-pairs separation
loss makes compile cost explode combinatorially with scenario count (see
that module's docstring for measured numbers), so points near the top of
the 3-21 range are EXPECTED to time out or hit the memory cap rather than
complete; those get recorded with status='timeout'/'oom' in the CSV instead
of being silently skipped or, worse, taking down the host process.

See refined_scenario_scaling.py for the intersection-refinement counterpart
and plot_scenario_runtime_scaling.py for the companion plot over both CSVs.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from runtime_scaling_common import (
    split_fault_budget, timeit_run, memory_metric_kb, append_csv_row,
    run_worker_with_limits, RESULT_JSON_PREFIX,
)

N = 10
CSV_PATH = _HERE / "unrefined_scenario_scaling_v2.csv"
RUN_TIME_REPEATS = 100

DT = 0.02
STEPS_PER_SEGMENT = 1
NUM_SEGMENTS = 4
# Multi-start width is close to free here: the optimiser is one vmap over
# restarts, and at N=10 even 256 of them is a few hundred KB per step --
# far below the width that would make this GPU work for its living (the
# measured device peak at 20 restarts was 20-25MB on a 32GB V100).
# Iteration count, by contrast, is a scan the runtime is linear in. So
# spend on restarts and take it back on iterations: 12.8x the coverage
# for ~1.9x fewer iterations. Check the CSV `loss` column stays in the
# same ballpark as the 20/15 rows before trusting a re-run.
NUM_RESTARTS = 256
NUM_ITERS = 8
LEARNING_RATE = 0.1

# Per-point subprocess guard rails -- see run_worker_with_limits and this
# module's docstring. No self-imposed timeout/RSS cap: every point is left
# to run to completion, however long that takes, and the only thing that
# can still end a point early is the real OS/cgroup OOM killer (in which
# case the point is recorded as an 'error' row) -- each completed point's
# CSV row is flushed immediately, so a later point's failure never loses
# earlier results.
WORKER_TIMEOUT_S = math.inf
WORKER_MEM_LIMIT_MB = math.inf

# Every CSV row (success or failure) has exactly these keys, in this order,
# so csv.DictWriter's fieldnames stay consistent regardless of which totals
# succeed vs. time out/OOM (see append_csv_row's docstring).
CSV_FIELDS = [
    "num_scenarios", "N", "status", "elapsed_s", "peak_rss_mb",
    "num_actuator_faults", "num_sensor_faults",
    "compile_time_ms", "run_time_single_ms", "run_time_avg_ms", "run_time_repeats",
    "memory_kb", "num_restarts", "num_iters", "steps_per_segment", "num_segments", "loss",
]


def _compute_point(total_scenarios: int) -> dict:
    """The actual work for one scenario count. Only ever called inside the
    `--worker` subprocess (see __main__) -- never by the driver directly --
    so a crash or OOM here only takes down that one subprocess."""
    import jax.numpy as jnp
    import immrax as irx
    from nonlinear_chain_separating_input import (
        default_channel_params, create_scenarios, MultistepSequenceOptimizer,
        optimize_multistep_gpu, time_jit,
    )

    Ka, Ks = split_fault_budget(total_scenarios, N)
    a, b = default_channel_params(N)
    scenarios = create_scenarios(N, a, b, num_actuator_faults=Ka, num_sensor_faults=Ks)
    assert len(scenarios) == total_scenarios

    x0_ivl = irx.icentpert(jnp.zeros(N), jnp.full(N, 0.02))
    ms_opt = MultistepSequenceOptimizer(scenarios, x0_ivl, DT, STEPS_PER_SEGMENT, NUM_SEGMENTS)

    def full_multistart(seed):
        return optimize_multistep_gpu(
            ms_opt, num_restarts=NUM_RESTARTS, learning_rate=LEARNING_RATE,
            num_iters=NUM_ITERS, seed=seed,
        )

    jitted_fn, compile_t, run_t_single, mem = time_jit(full_multistart, 42)
    avg_run_t = timeit_run(jitted_fn, (42,), num_repeats=RUN_TIME_REPEATS)

    _, loss_opt, _, _ = jitted_fn(42)

    return {
        "num_actuator_faults": Ka,
        "num_sensor_faults": Ks,
        "compile_time_ms": compile_t * 1e3,
        "run_time_single_ms": run_t_single * 1e3,
        "run_time_avg_ms": avg_run_t * 1e3,
        "run_time_repeats": RUN_TIME_REPEATS,
        "memory_kb": memory_metric_kb(mem),
        "num_restarts": NUM_RESTARTS,
        "num_iters": NUM_ITERS,
        "steps_per_segment": STEPS_PER_SEGMENT,
        "num_segments": NUM_SEGMENTS,
        "loss": float(loss_opt),
    }


def _row_from_result(total_scenarios: int, result: dict) -> dict:
    row = {k: "" for k in CSV_FIELDS}
    row["num_scenarios"] = total_scenarios
    row["N"] = N
    row["status"] = result["status"]
    row["elapsed_s"] = round(result["elapsed_s"], 3)
    row["peak_rss_mb"] = round(result["peak_rss_mb"], 1)
    if result["status"] == "ok":
        row.update(result["data"])
    return row


def run_for_scenario_count(total_scenarios: int):
    print(f"\n{'=' * 70}\nUNREFINED MULTISTEP -- {total_scenarios} scenarios (N={N})\n{'=' * 70}")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(total_scenarios)]
    result = run_worker_with_limits(cmd, timeout_s=WORKER_TIMEOUT_S, mem_limit_mb=WORKER_MEM_LIMIT_MB)

    if result["status"] == "ok":
        d = result["data"]
        print(f"OK      compile {d['compile_time_ms']:8.2f} ms   "
              f"run(avg) {d['run_time_avg_ms']:7.3f} ms   "
              f"mem {d['memory_kb']:9.1f} KB   loss {d['loss']:.6f}   "
              f"(elapsed {result['elapsed_s']:.1f}s, peak_rss {result['peak_rss_mb']:.0f} MB)")
    else:
        print(f"{result['status'].upper():7s} {result.get('error', '')}")

    append_csv_row(CSV_PATH, _row_from_result(total_scenarios, result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=None,
                        help="internal: compute one scenario count and print RESULT_JSON to stdout")
    args = parser.parse_args()

    if args.worker is not None:
        payload = _compute_point(args.worker)
        print(RESULT_JSON_PREFIX + json.dumps(payload))
        sys.exit(0)

    import jax
    print("Devices:", jax.devices())
    print(f"Per-point limits: timeout={WORKER_TIMEOUT_S}s, mem_limit={WORKER_MEM_LIMIT_MB}MB")

    # Start each sweep from a clean CSV so old rows (possibly from a
    # different config) don't get mixed into the scaling plot.
    if CSV_PATH.exists():
        CSV_PATH.unlink()

    for total in range(3, 22):
        run_for_scenario_count(total)

    print(f"\nDone. Scenario-count runtime/memory data written to {CSV_PATH}")
