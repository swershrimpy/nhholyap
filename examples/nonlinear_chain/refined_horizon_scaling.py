"""
Intersection-refinement PLANNING-HORIZON runtime/RAM/VRAM scaling sweep for
the decoupled cubic-drift chain (nonlinear_chain_separating_input.py).

Companion to refined_scenario_scaling.py: that script fixes the planning
horizon (NUM_STEPS=3) and sweeps the number of fault scenarios; this one
does the opposite -- fixes N=10 and the fault-scenario count at
TOTAL_SCENARIOS and sweeps the planning horizon NUM_STEPS directly
(horizon_s = NUM_STEPS * DT). Same sweep-axis swap as
unrefined_horizon_scaling.py -- see that module's docstring for why
runtime_scaling_common.py needs no changes to yield independent RAM (host,
externally-polled peak_rss_mb) and VRAM (device peak_bytes_in_use,
memory_kb) numbers per point.

This path's per-step cost is O(n_pairs) -- fixed here at C(7,2)=21 pairs by
TOTAL_SCENARIOS=7, the same value unrefined_horizon_scaling.py uses, so the
two sweeps are directly comparable at every horizon. Each horizon point
still runs in its OWN subprocess with no self-imposed timeout/RSS cap (see
unrefined_horizon_scaling.py's docstring for the full rationale); every
completed point's row is flushed to CSV immediately, so a walltime kill or
crash only ever loses the one point that was in flight, never the points
before it.

See unrefined_horizon_scaling.py for the unrefined counterpart and
plot_horizon_runtime_scaling.py for the companion plot over both CSVs.
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
TOTAL_SCENARIOS = 7   # fixed scenario count (Ka=3, Ks=3 via split_fault_budget) -- matches unrefined_horizon_scaling.py
CSV_PATH = _HERE / "refined_horizon_scaling_v2.csv"
RUN_TIME_REPEATS = 100

DT = 0.02
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

# Same horizon sweep (in Euler steps) as unrefined_horizon_scaling.py, so the
# two CSVs are directly comparable point-for-point in plot_horizon_runtime_scaling.py.
HORIZON_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50]

# See unrefined_scenario_scaling.py's docstring for why these are math.inf.
WORKER_TIMEOUT_S = math.inf
WORKER_MEM_LIMIT_MB = math.inf

CSV_FIELDS = [
    "num_steps", "horizon_s", "N", "num_scenarios", "status", "elapsed_s", "peak_rss_mb",
    "num_actuator_faults", "num_sensor_faults",
    "compile_time_ms", "run_time_single_ms", "run_time_avg_ms", "run_time_repeats",
    "memory_kb", "num_restarts", "num_iters", "loss",
]


def _compute_point(num_steps: int) -> dict:
    """The actual work for one horizon point. Only ever called inside the
    `--worker` subprocess (see __main__) -- never by the driver directly --
    so a crash or OOM here only takes down that one subprocess."""
    import jax.numpy as jnp
    import immrax as irx
    from nonlinear_chain_separating_input import (
        default_channel_params, create_scenarios, optimize_refined_gpu, time_jit,
    )

    Ka, Ks = split_fault_budget(TOTAL_SCENARIOS, N)
    a, b = default_channel_params(N)
    scenarios = create_scenarios(N, a, b, num_actuator_faults=Ka, num_sensor_faults=Ks)
    assert len(scenarios) == TOTAL_SCENARIOS

    x0_ivl = irx.icentpert(jnp.zeros(N), jnp.full(N, 0.02))

    def full_multistart(seed):
        return optimize_refined_gpu(
            x0_ivl, scenarios, dt=DT, num_steps=num_steps,
            num_restarts=NUM_RESTARTS, learning_rate=LEARNING_RATE,
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
        "loss": float(loss_opt),
    }


def _row_from_result(num_steps: int, result: dict) -> dict:
    row = {k: "" for k in CSV_FIELDS}
    row["num_steps"] = num_steps
    row["horizon_s"] = round(num_steps * DT, 4)
    row["N"] = N
    row["num_scenarios"] = TOTAL_SCENARIOS
    row["status"] = result["status"]
    row["elapsed_s"] = round(result["elapsed_s"], 3)
    row["peak_rss_mb"] = round(result["peak_rss_mb"], 1)
    if result["status"] == "ok":
        row.update(result["data"])
    return row


def run_for_horizon(num_steps: int):
    horizon_s = num_steps * DT
    print(f"\n{'=' * 70}\nREFINED MULTISTEP -- horizon {num_steps} steps "
          f"({horizon_s:.2f}s, N={N}, {TOTAL_SCENARIOS} scenarios)\n{'=' * 70}")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(num_steps)]
    result = run_worker_with_limits(cmd, timeout_s=WORKER_TIMEOUT_S, mem_limit_mb=WORKER_MEM_LIMIT_MB)

    if result["status"] == "ok":
        d = result["data"]
        print(f"OK      compile {d['compile_time_ms']:8.2f} ms   "
              f"run(avg) {d['run_time_avg_ms']:7.3f} ms   "
              f"VRAM {d['memory_kb']:9.1f} KB   RAM {result['peak_rss_mb']:7.1f} MB   "
              f"loss {d['loss']:.6f}   (elapsed {result['elapsed_s']:.1f}s)")
    else:
        print(f"{result['status'].upper():7s} {result.get('error', '')}")

    append_csv_row(CSV_PATH, _row_from_result(num_steps, result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=None,
                        help="internal: compute one horizon (num_steps) and print RESULT_JSON to stdout")
    args = parser.parse_args()

    if args.worker is not None:
        payload = _compute_point(args.worker)
        print(RESULT_JSON_PREFIX + json.dumps(payload))
        sys.exit(0)

    import jax
    print("Devices:", jax.devices())
    print(f"Fixed: N={N}, total_scenarios={TOTAL_SCENARIOS}, dt={DT}")
    print(f"Per-point limits: timeout={WORKER_TIMEOUT_S}s, mem_limit={WORKER_MEM_LIMIT_MB}MB")

    # Start each sweep from a clean CSV so old rows (possibly from a
    # different config) don't get mixed into the scaling plot.
    if CSV_PATH.exists():
        CSV_PATH.unlink()

    for num_steps in HORIZON_STEPS:
        run_for_horizon(num_steps)

    print(f"\nDone. Planning-horizon runtime/RAM/VRAM data written to {CSV_PATH}")
