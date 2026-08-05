"""
Single-step separating-input demo for the Nth-order integrator chain.

Runs the single-step (constant-u) separating-input optimizer for a couple of
concrete orders (N=2, N=3), reporting:
  - JIT compile time vs. steady-state run time for the loss function AND for
    the full GPU-vmapped multistart optimization (via time_jit).
  - The optimized separating input, its pairwise output-space overlaps, and
    the output-interval volumes per scenario.
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
    separation_loss,
    SeparatingInputOptimizer,
    optimize_parallel_gpu,
    time_jit,
)


def run_for_order(N: int, num_restarts: int = 50, num_iters: int = 20, learning_rate: float = 0.15):
    print(f"\n{'=' * 70}\nSINGLE-STEP SEPARATING INPUT -- order N={N}\n{'=' * 70}")

    scenarios = create_scenarios(N)
    print(f"{len(scenarios)} fault scenarios:")
    for s in scenarios:
        print(f"  - {s.name}")

    x0_ivl = irx.icentpert(jnp.zeros(N), jnp.full(N, 0.05))
    dt, num_steps = 0.2, 8
    print(f"\nPropagation: {num_steps} x {dt} s = {num_steps * dt} s total")

    # ── Loss-function compile vs. run time ────────────────────────────────
    loss_fn = partial(separation_loss, x0_ivl=x0_ivl, scenarios=scenarios, dt=dt, num_steps=num_steps)
    u_probe = jnp.array([0.5])
    _, compile_t, run_t, mem = time_jit(loss_fn, u_probe)
    print(f"\nloss_fn(u):        compile {compile_t * 1e3:8.2f} ms   run {run_t * 1e3:7.3f} ms")

    # ── Full multistart optimization compile vs. run time ─────────────────
    opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt, num_steps)

    def full_multistart(seed):
        return optimize_parallel_gpu(
            opt, num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed
        )

    _, compile_t_opt, run_t_opt, mem_opt = time_jit(full_multistart, 42)
    print(f"full multistart:   compile {compile_t_opt * 1e3:8.2f} ms   run {run_t_opt * 1e3:7.3f} ms"
          f"   ({num_restarts} restarts x {num_iters} iters)")

    # ── Results ────────────────────────────────────────────────────────────
    u_opt, loss_opt, _, _ = full_multistart(42)
    stats = opt.evaluate(u_opt)

    print(f"\nOptimal separating input: u = {float(u_opt[0]):+.4f}")
    print(f"Best loss: {float(loss_opt):.6f}")
    print("\nPairwise output-space overlaps:")
    for k, v in stats['pairwise_overlaps'].items():
        print(f"  {k}: {v:.6f}")
    print("\nOutput-interval volumes:")
    for k, v in stats['volumes'].items():
        print(f"  {k}: {v:.6f}")
    print(f"\nMemory snapshot: {mem_opt}")


if __name__ == "__main__":
    print("Devices:", jax.devices())
    for N in (2, 3):
        run_for_order(N)
    print("\nDone.")
