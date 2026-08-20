"""
Unrefined-only counterpart to refinement_demo_stable.py's N=2..100 sweep.

refinement_demo_stable.py's run_for_order runs BOTH optimizers every order,
but only ever times the REFINED one through time_jit -- the unrefined pass
there exists solely to get a u_seq for a correctness sanity check (refined
loss at the unrefined-optimal u_seq stays <= the unrefined loss), and its
own compile/run time and memory are never measured. This script is the
missing other half: times ONLY the unrefined multistart optimizer
(`optimize_multistep_gpu`, the same jitted/vmapped-restarts/checkpointed
path `optimize_multistep`'s GPU branch uses -- see that function in
integrator_separating_input.py) via time_jit, across the same N=2..100
range, with the same per-order resettable host-RAM peak
(HOST_PEAK_RSS -- see integrator_separating_input.py's _HostPeakRSS) and
the same "Memory snapshot: {...}" line format plot_unrefined_runtime_scaling.py
parses, alongside device VRAM (peak_bytes_in_use).

Deliberately does NOT also run the refined optimizer -- this is the
unrefined-only mirror of refinement_demo_stable.py's refined-only
measurement, not a combined sweep. No known NaN-at-high-N issue exists on
this path (that bug was specific to propagate_with_refinement's
jnp.where-branch gradient gotcha -- see refinement_demo_stable.py's module
docstring), so there is no "_stable" fix needed here beyond reusing the
same scenarios/hyperparameters for a direct, apples-to-apples comparison
against the refined sweep's numbers.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp

from integrator_separating_input import (
    Scenario,
    create_scenarios,
    MultistepSequenceOptimizer,
    optimize_multistep_gpu,
    time_jit,
    HOST_PEAK_RSS,
)

import immrax as irx


def run_for_order(N: int, num_restarts: int = 30, num_iters: int = 20, learning_rate: float = 0.15):
    HOST_PEAK_RSS.reset()
    print(f"\n{'=' * 70}\nUNREFINED MULTISTEP (only) -- order N={N}\n{'=' * 70}")

    scenarios = create_scenarios(N)
    x0_ivl = irx.icentpert(jnp.zeros(N), jnp.full(N, 0.05))
    dt = 0.15
    num_steps = 6

    ms_opt = MultistepSequenceOptimizer(
        scenarios=scenarios, x0_ivl=x0_ivl, dt=dt, steps_per_segment=1, num_segments=num_steps,
    )

    def full_unrefined_multistart(seed):
        return optimize_multistep_gpu(
            ms_opt, num_restarts=num_restarts, learning_rate=learning_rate,
            num_iters=num_iters, seed=seed,
        )

    _, compile_t, run_t, mem = time_jit(full_unrefined_multistart, 42)
    print(f"\nunrefined multistart: compile {compile_t * 1e3:8.2f} ms   run {run_t * 1e3:7.3f} ms"
          f"   ({num_restarts} restarts x {num_iters} iters)")

    _, loss_opt, _, all_losses = full_unrefined_multistart(42)
    num_nan_restarts = int(jnp.sum(jnp.isnan(all_losses)))
    print(f"  unrefined best loss: {float(loss_opt):.6f}"
          f"   ({num_nan_restarts}/{num_restarts} restarts went nan)")

    print(f"\nMemory snapshot: {mem}")


if __name__ == "__main__":
    print("Devices:", jax.devices())
    HOST_PEAK_RSS.start()
    for N in range(2, 101):
        run_for_order(N)
    print("\nDone.")
