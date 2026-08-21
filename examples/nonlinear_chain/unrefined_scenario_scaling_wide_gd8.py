"""
Standalone unrefined-multistep SCENARIO-COUNT runtime/RAM/VRAM sweep at the
wide-restart, 8-GD-iteration config (see run_unrefined_scaling_wide_gd8.sbatch
for the rationale: matches n256i8, the horizon config that won on runtime in
plot_horizon_runtime_scaling_best.py, at the wide 1.2M-restart budget used in
the horizon zeroth-order sweep).

This is deliberately a STANDALONE copy of unrefined_scenario_scaling.py's
sweep logic rather than an env-var override of that shared script. While this
job's first attempt was running (as run_unrefined_scaling_wide_gd8.sbatch
pointed at unrefined_scenario_scaling.py with NLCHAIN_NUM_RESTARTS/_NUM_ITERS
env overrides), another process editing that exact file on this shared
filesystem reverted it mid-run -- back to a hardcoded NUM_RESTARTS=20,
NUM_ITERS=15, no env-var support at all. Because each scenario-count point
spawns a FRESH subprocess that re-imports the .py file from disk, later
points silently picked up the reverted config instead of erroring, producing
a CSV with a real wide-restart point 1 followed by wrong-config points 2+
with no obvious sign anything was wrong (see the killed job's CSV,
preserved for reference, for exactly this pattern). Depending only on
nonlinear_chain_separating_input.py and runtime_scaling_common.py -- both
confirmed git-clean and unchanged by that other process -- and owning this
file exclusively removes that race entirely: nothing else on this machine
should be editing a file with this name.

Same per-point subprocess isolation as unrefined_horizon_scaling.py's
zeroth_wide config: WORKER_TIMEOUT_S / WORKER_MEM_LIMIT_MB are math.inf (not
the original script's 150s / 4096MB guard rails, which were sized for the
20-restart/15-iter config and would kill every point of a 1.2M-restart sweep
almost immediately) -- a point is left to run to completion or hit a real
device/OS OOM, recorded as status='error', while every completed point's row
is flushed to CSV immediately so a later failure or walltime kill never loses
earlier points.

See refined_scenario_scaling_wide_gd8.py for the intersection-refinement
counterpart.
"""

import json
import sys
import math
from pathlib import Path

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from runtime_scaling_common import (
    split_fault_budget, timeit_run, memory_metric_kb, append_csv_row,
    run_worker_with_limits, RESULT_JSON_PREFIX,
)

N = 10
CSV_PATH = _HERE / "unrefined_scenario_scaling_wide_gd8.csv"
RUN_TIME_REPEATS = 100

DT = 0.02
STEPS_PER_SEGMENT = 1
NUM_SEGMENTS = 4
NUM_RESTARTS = 1_200_000
NUM_ITERS = 8
LEARNING_RATE = 0.1

# No self-imposed timeout/RSS cap -- see module docstring. Only a real
# device/OS OOM ends a point early.
WORKER_TIMEOUT_S = math.inf
WORKER_MEM_LIMIT_MB = math.inf

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
    print(f"\n{'=' * 70}\nUNREFINED MULTISTEP (wide, gd8) -- {total_scenarios} scenarios (N={N})\n{'=' * 70}")
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
    import argparse
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
    print(f"num_restarts={NUM_RESTARTS}  num_iters={NUM_ITERS}")
    print(f"Per-point limits: timeout={WORKER_TIMEOUT_S}s, mem_limit={WORKER_MEM_LIMIT_MB}MB")

    if CSV_PATH.exists():
        CSV_PATH.unlink()

    for total in range(3, 22):
        run_for_scenario_count(total)

    print(f"\nDone. Scenario-count runtime/memory data written to {CSV_PATH}")
