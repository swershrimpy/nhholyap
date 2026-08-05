"""
Multistep ("unrefined" / "uninformed") separating-input demo for the
Nth-order integrator chain.

Optimizes a control *sequence* (rather than a single constant input) to
maximize separation of the fault-scenario reachable sets at some point along
the horizon, propagating every scenario independently (no cross-scenario
observation information used mid-horizon -- see refinement_demo.py for the
version that does use it).

Reports JIT compile time vs. steady-state run time for both the multistep
loss function and the full GPU-vmapped multistart optimization.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from functools import partial

import jax
import jax.numpy as jnp
import immrax as irx

from integrator_separating_input import (
    create_scenarios,
    separation_loss_multistep,
    MultistepSequenceOptimizer,
    optimize_multistep_gpu,
    time_jit,
)


def run_for_order(N: int, num_restarts: int = 50, num_iters: int = 20, learning_rate: float = 0.15):
    print(f"\n{'=' * 70}\nMULTISTEP (UNREFINED) SEPARATING INPUT -- order N={N}\n{'=' * 70}")

    scenarios = create_scenarios(N)
    x0_ivl = irx.icentpert(jnp.zeros(N), jnp.full(N, 0.05))
    dt, steps_per_segment, num_segments = 0.1, 2, 10
    print(f"Horizon: {num_segments} segments x {steps_per_segment} steps x {dt} s "
          f"= {num_segments * steps_per_segment * dt} s total")

    # ── Loss-function compile vs. run time ────────────────────────────────
    loss_fn = partial(separation_loss_multistep, x0_ivl=x0_ivl, scenarios=scenarios,
                      dt=dt, steps_per_segment=steps_per_segment)
    u_seq_probe = jnp.tile(jnp.array([0.5]), (num_segments, 1))
    _, compile_t, run_t, mem = time_jit(loss_fn, u_seq_probe)
    print(f"\nloss_fn(u_seq):    compile {compile_t * 1e3:8.2f} ms   run {run_t * 1e3:7.3f} ms")

    # ── Full multistart optimization compile vs. run time ─────────────────
    ms_opt = MultistepSequenceOptimizer(scenarios, x0_ivl, dt, steps_per_segment, num_segments)

    def full_multistart(seed):
        return optimize_multistep_gpu(
            ms_opt, num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed
        )

    _, compile_t_opt, run_t_opt, mem_opt = time_jit(full_multistart, 42)
    print(f"full multistart:   compile {compile_t_opt * 1e3:8.2f} ms   run {run_t_opt * 1e3:7.3f} ms"
          f"   ({num_restarts} restarts x {num_iters} iters)")

    # ── Results ────────────────────────────────────────────────────────────
    u_seq_opt, loss_opt, _, _ = full_multistart(42)
    print(f"\nBest loss: {float(loss_opt):.6f}")
    print(f"Optimal control sequence (first 5 segments): "
          f"{[round(float(v), 3) for v in u_seq_opt[:5, 0]]}")
    print(f"\nMemory snapshot: {mem_opt}")


if __name__ == "__main__":
    print("Devices:", jax.devices())
    for N in (2, 3):
        run_for_order(N)
    print("\nDone.")
