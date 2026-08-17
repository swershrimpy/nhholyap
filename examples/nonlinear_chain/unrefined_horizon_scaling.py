"""
Unrefined-multistep PLANNING-HORIZON runtime/RAM/VRAM scaling sweep for the
decoupled cubic-drift chain (nonlinear_chain_separating_input.py).

Companion to unrefined_scenario_scaling.py: that script fixes the planning
horizon (NUM_SEGMENTS=4) and sweeps the number of fault scenarios; this one
does the opposite -- fixes N=10 and the fault-scenario count at
TOTAL_SCENARIOS (Nominal + Ka ActuatorFault_i + Ks SensorFault_i, split via
runtime_scaling_common.split_fault_budget) and sweeps the planning horizon,
i.e. NUM_SEGMENTS (steps_per_segment=1 throughout, so num_segments IS the
number of Euler steps and horizon_s = num_segments * DT is the look-ahead
time the multistep optimizer plans over).

Reuses runtime_scaling_common.py unchanged -- in particular
run_worker_with_limits already yields TWO independent memory numbers per
point, which is what makes a "RAM vs. VRAM" (not just "memory") comparison
possible without touching nonlinear_chain_separating_input.py's
_memory_snapshot()/time_jit() at all:
  - peak_rss_mb : the worker SUBPROCESS's own peak resident memory, polled
                  from /proc by the DRIVER from outside the process. This is
                  genuine host RAM, always present regardless of platform,
                  and needs no in-process reset trick (unlike
                  integrator_chain's HOST_PEAK_RSS) because each point is a
                  fresh subprocess -- there is nothing for it to have
                  accumulated from a previous point.
  - memory_kb   : memory_metric_kb(mem) from *inside* the worker's own
                  time_jit call. Under JAX_PLATFORMS=cuda (which the sbatch
                  script forces), jax.devices()[0].memory_stats() succeeds,
                  so this is the XLA allocator's peak_bytes_in_use --
                  genuine GPU VRAM, not host memory (CPU RSS is used only as
                  a fallback when no GPU device stats are available).

Each horizon point runs in its OWN subprocess (`--worker` mode below), same
as the scenario-count sweeps, with no self-imposed timeout/RSS cap -- every
point runs to completion however long that takes, and only the real
OS/cgroup OOM killer can end one early (recorded as status='error'). Each
completed point's row is flushed to CSV immediately (append_csv_row), so a
later point's failure -- or the whole job getting walltime-killed -- never
loses earlier points' data; only the point that was in flight when the job
died is missing, and everything before it is already durably on disk.

See refined_horizon_scaling.py for the intersection-refinement counterpart
and plot_horizon_runtime_scaling.py for the companion plot over both CSVs.
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
TOTAL_SCENARIOS = 7   # fixed scenario count (Ka=3, Ks=3 via split_fault_budget)
CSV_PATH = _HERE / "unrefined_horizon_scaling.csv"
RUN_TIME_REPEATS = 100

DT = 0.02
STEPS_PER_SEGMENT = 1
NUM_RESTARTS = 20
NUM_ITERS = 15
LEARNING_RATE = 0.1

# Planning-horizon sweep, in Euler steps (steps_per_segment=1, so this is
# also num_segments); horizon_s = HORIZON_STEPS[i] * DT ranges 0.02s-1.0s.
# The combinatorial blowup documented in runtime_scaling_common.py's
# docstring was driven by scenario PAIRS (fixed here at
# C(7,2)=21, far below the 210 pairs at total_scenarios=21 that motivated
# the subprocess/watchdog design in the first place), not by horizon length,
# so this range is expected to complete end to end -- but the no-timeout,
# flush-every-point design means it doesn't matter if it doesn't.
HORIZON_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50]

# See unrefined_scenario_scaling.py's docstring for why these are math.inf.
WORKER_TIMEOUT_S = math.inf
WORKER_MEM_LIMIT_MB = math.inf

CSV_FIELDS = [
    "num_segments", "horizon_s", "N", "num_scenarios", "status", "elapsed_s", "peak_rss_mb",
    "num_actuator_faults", "num_sensor_faults",
    "compile_time_ms", "run_time_single_ms", "run_time_avg_ms", "run_time_repeats",
    "memory_kb", "num_restarts", "num_iters", "steps_per_segment", "loss",
]


def _compute_point(num_segments: int) -> dict:
    """The actual work for one horizon point. Only ever called inside the
    `--worker` subprocess (see __main__) -- never by the driver directly --
    so a crash or OOM here only takes down that one subprocess."""
    import jax.numpy as jnp
    import immrax as irx
    from nonlinear_chain_separating_input import (
        default_channel_params, create_scenarios, MultistepSequenceOptimizer,
        optimize_multistep_gpu, time_jit,
    )

    Ka, Ks = split_fault_budget(TOTAL_SCENARIOS, N)
    a, b = default_channel_params(N)
    scenarios = create_scenarios(N, a, b, num_actuator_faults=Ka, num_sensor_faults=Ks)
    assert len(scenarios) == TOTAL_SCENARIOS

    x0_ivl = irx.icentpert(jnp.zeros(N), jnp.full(N, 0.02))
    ms_opt = MultistepSequenceOptimizer(scenarios, x0_ivl, DT, STEPS_PER_SEGMENT, num_segments)

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
        "loss": float(loss_opt),
    }


def _row_from_result(num_segments: int, result: dict) -> dict:
    row = {k: "" for k in CSV_FIELDS}
    row["num_segments"] = num_segments
    row["horizon_s"] = round(num_segments * DT, 4)
    row["N"] = N
    row["num_scenarios"] = TOTAL_SCENARIOS
    row["status"] = result["status"]
    row["elapsed_s"] = round(result["elapsed_s"], 3)
    row["peak_rss_mb"] = round(result["peak_rss_mb"], 1)
    if result["status"] == "ok":
        row.update(result["data"])
    return row


def run_for_horizon(num_segments: int):
    horizon_s = num_segments * DT
    print(f"\n{'=' * 70}\nUNREFINED MULTISTEP -- horizon {num_segments} steps "
          f"({horizon_s:.2f}s, N={N}, {TOTAL_SCENARIOS} scenarios)\n{'=' * 70}")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(num_segments)]
    result = run_worker_with_limits(cmd, timeout_s=WORKER_TIMEOUT_S, mem_limit_mb=WORKER_MEM_LIMIT_MB)

    if result["status"] == "ok":
        d = result["data"]
        print(f"OK      compile {d['compile_time_ms']:8.2f} ms   "
              f"run(avg) {d['run_time_avg_ms']:7.3f} ms   "
              f"VRAM {d['memory_kb']:9.1f} KB   RAM {result['peak_rss_mb']:7.1f} MB   "
              f"loss {d['loss']:.6f}   (elapsed {result['elapsed_s']:.1f}s)")
    else:
        print(f"{result['status'].upper():7s} {result.get('error', '')}")

    append_csv_row(CSV_PATH, _row_from_result(num_segments, result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=None,
                        help="internal: compute one horizon (num_segments) and print RESULT_JSON to stdout")
    args = parser.parse_args()

    if args.worker is not None:
        payload = _compute_point(args.worker)
        print(RESULT_JSON_PREFIX + json.dumps(payload))
        sys.exit(0)

    import jax
    print("Devices:", jax.devices())
    print(f"Fixed: N={N}, total_scenarios={TOTAL_SCENARIOS}, dt={DT}, steps_per_segment={STEPS_PER_SEGMENT}")
    print(f"Per-point limits: timeout={WORKER_TIMEOUT_S}s, mem_limit={WORKER_MEM_LIMIT_MB}MB")

    # Start each sweep from a clean CSV so old rows (possibly from a
    # different config) don't get mixed into the scaling plot.
    if CSV_PATH.exists():
        CSV_PATH.unlink()

    for num_segments in HORIZON_STEPS:
        run_for_horizon(num_segments)

    print(f"\nDone. Planning-horizon runtime/RAM/VRAM data written to {CSV_PATH}")
