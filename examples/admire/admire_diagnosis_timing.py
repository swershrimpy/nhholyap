"""
Single-config ADMIRE fault-diagnosis timing probe (unrefined multistep vs.
refined multistep), aimed at finding an operating point whose POST-COMPILE
run time is under 50ms.

Why this is a separate script from admire_success_rate_analysis.py
--------------------------------------------------------------------
That script's "run time" is for an entire 27-config x NUM_RESTARTS x
NUM_ITERS batch compiled as ONE jax.jit call -- hundreds of ms to whole
seconds (measured: multistep_unrefined @ 0.5s horizon = 2.55s total /
9 configs = 283ms/config, at NUM_ITERS=100; see that script's saved .npz
files), not a single-diagnosis latency number. This script instead calls
that same run_batched_sweep with num_configs=1 (the folder's default
input_limit=0.05, x0_width=0.01) -- reusing that exact function and the
same _METHOD_LOSS_BUILDERS, so this differs from the success-rate sweep
only in HOW MANY configs are batched together and how many GD iterations
are run, not in what's being computed or which fault scenarios are used
(all 11, unchanged).

Horizon is fixed at 0.5s (5 Euler steps at dt=0.1) -- the cheapest and
best-precedented horizon in this folder (see
admire_success_rate_analysis.py's module docstring); a longer horizon only
makes hitting a latency target harder, and there is no reason to pay that
cost while searching for a sub-50ms operating point.

NUM_ITERS is swept (each value recompiles fresh -- it is a Python-level
jax.lax.scan TRIP COUNT baked into the trace at the loop-body level, but
note compile time in practice did NOT scale with it in the first run of
this script -- jax.lax.scan traces its body once regardless of length, so
NUM_ITERS mainly affects RUNTIME, not compile time) to find, per method,
the largest iteration count that still clears the 50ms target.

Per-config subprocess isolation (fixes a real measurement bug)
-----------------------------------------------------------------
The first version of this script called run_one() directly, in-process,
for all (method, num_iters) points in one Python process. JAX's
memory_stats()['peak_bytes_in_use'] is a PROCESS-LIFETIME running max (see
integrator_chain's HOST_RAM_INSTRUMENTATION work for the same caveat
elsewhere in this project) -- it is NEVER reset between calls. Run
12030802's log shows this directly: all 6 (method, num_iters) points
reported the EXACT SAME peak_bytes_in_use (49412608.0 bytes) regardless of
method or iteration count, which is only possible if every reading after
the first was just echoing an already-frozen historical peak rather than
that config's own peak. Each config now runs in its OWN subprocess
(`--worker <loss_kind> <num_iters>`) so it gets a fresh XLA allocator, and
peak_bytes_in_use becomes a genuine per-config measurement.
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

HORIZON_S = 0.5                            # cheapest/best-precedented horizon in this folder
INPUT_LIMIT = 0.05                         # folder default (admire_separating_input.py's hardcoded box)
X0_WIDTH = 0.01                            # folder default
NUM_RESTARTS = 20                          # matches admire_success_rate_analysis.py's default
NUM_ITERS_CANDIDATES = [50, 20, 5, 3, 1]   # descending -- report the largest that clears the target
LEARNING_RATE = 0.05
SEED = 42
TARGET_MS = 50.0

METHODS = ["multistep_unrefined", "multistep_refined"]

RESULT_JSON_PREFIX = "RESULT_JSON:"


def run_one(loss_kind: str, num_iters: int) -> dict:
    """The actual work for one (method, num_iters) point. Only ever called
    inside the `--worker` subprocess (see __main__) -- never by the driver
    directly -- so this process's JAX allocator (and its peak_bytes_in_use)
    reflects ONLY this one config, not whatever ran before it."""
    import jax.numpy as jnp
    from admire_success_rate_analysis import (
        run_batched_sweep, _METHOD_LOSS_BUILDERS,
        DT, NUM_STEPS_FOR_HORIZON, NUM_SEGMENTS_FOR_HORIZON,
    )

    num_steps = NUM_STEPS_FOR_HORIZON[HORIZON_S]
    num_segments = NUM_SEGMENTS_FOR_HORIZON[HORIZON_S]
    loss_fn_single, u_shape = _METHOD_LOSS_BUILDERS[loss_kind](num_steps, num_segments, DT)
    config_arrays = (jnp.array([INPUT_LIMIT]), jnp.array([X0_WIDTH]))

    u_final, losses_final, compile_t, run_t, mem = run_batched_sweep(
        loss_fn_single, u_shape, config_arrays,
        num_restarts=NUM_RESTARTS, num_iters=num_iters,
        learning_rate=LEARNING_RATE, seed=SEED,
    )
    best_loss = float(jnp.min(losses_final[0]))
    return {"compile_t": compile_t, "run_t": run_t, "mem": mem, "best_loss": best_loss}


def run_one_subprocess(loss_kind: str, num_iters: int) -> dict:
    """Runs run_one(loss_kind, num_iters) in a fresh subprocess and returns
    its result dict, parsed from the one RESULT_JSON line it prints."""
    import subprocess
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", loss_kind, str(num_iters)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker ({loss_kind}, num_iters={num_iters}) exited {proc.returncode}:\n"
            f"{proc.stderr[-2000:]}"
        )
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_JSON_PREFIX):
            payload = json.loads(line[len(RESULT_JSON_PREFIX):])
    if payload is None:
        raise RuntimeError(
            f"worker ({loss_kind}, num_iters={num_iters}) printed no RESULT_JSON line:\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
        )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("LOSS_KIND", "NUM_ITERS"), default=None,
                        help="internal: compute one (method, num_iters) point and print RESULT_JSON to stdout")
    args = parser.parse_args()

    if args.worker is not None:
        loss_kind, num_iters = args.worker[0], int(args.worker[1])
        payload = run_one(loss_kind, num_iters)
        print(RESULT_JSON_PREFIX + json.dumps(payload))
        sys.exit(0)

    import jax
    from admire_success_rate_analysis import SCENARIOS, NUM_STEPS_FOR_HORIZON, _METHOD_DISPLAY

    print("Devices:", jax.devices())
    print(f"Fixed: horizon={HORIZON_S}s ({NUM_STEPS_FOR_HORIZON[HORIZON_S]} steps), "
          f"input_limit={INPUT_LIMIT}, x0_width={X0_WIDTH}, "
          f"{len(SCENARIOS)} fault scenarios (unchanged), num_restarts={NUM_RESTARTS}")
    print(f"Target: post-compile run time < {TARGET_MS} ms")
    print("Each (method, num_iters) point runs in its own subprocess, so peak_bytes_in_use "
          "below is a genuine per-config measurement, not a stale process-lifetime max.\n")

    results = {}
    for loss_kind in METHODS:
        print(f"\n{'=' * 78}\n{_METHOD_DISPLAY[loss_kind]}\n{'=' * 78}")
        for num_iters in NUM_ITERS_CANDIDATES:
            r = run_one_subprocess(loss_kind, num_iters)
            run_ms = r["run_t"] * 1e3
            verdict = "PASS" if run_ms < TARGET_MS else "FAIL"
            print(f"  num_iters={num_iters:3d}  compile {r['compile_t'] * 1e3:9.2f} ms   "
                  f"run {run_ms:8.3f} ms   [{verdict}]   best_loss={r['best_loss']:.6f}")
            print(f"    mem={r['mem']}")
            results[(loss_kind, num_iters)] = (run_ms, verdict, r["best_loss"])

    print(f"\n{'=' * 78}\nSUMMARY (target < {TARGET_MS} ms)\n{'=' * 78}")
    for loss_kind in METHODS:
        passing = [n for n in NUM_ITERS_CANDIDATES if results[(loss_kind, n)][1] == "PASS"]
        if passing:
            best_n = max(passing)
            run_ms = results[(loss_kind, best_n)][0]
            print(f"  {_METHOD_DISPLAY[loss_kind]:22s}: PASS at num_iters<={best_n} ({run_ms:.3f} ms)")
        else:
            fastest_n = min(NUM_ITERS_CANDIDATES, key=lambda n: results[(loss_kind, n)][0])
            run_ms = results[(loss_kind, fastest_n)][0]
            print(f"  {_METHOD_DISPLAY[loss_kind]:22s}: FAIL at every tested num_iters "
                  f"(fastest {run_ms:.3f} ms @ num_iters={fastest_n})")

    print("\nDone.")
