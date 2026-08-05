"""
Intersection-refinement demo for the Nth-order integrator chain.

Runs both the "unrefined"/"uninformed" multistep separating-input optimizer
(separation_loss_multistep -- each scenario propagated independently) and the
intersection-refinement optimizer (refined_overlap_loss -- at each step,
overlapping scenario pairs have their state tightened using the observed
output before propagating forward) on the *same* scenario set and horizon,
and prints a side-by-side loss comparison.

Since refinement can only tighten reachable sets (never loosen them), the
refined loss should never exceed the unrefined loss for a comparable horizon.

Also reports JIT compile time vs. steady-state run time for the refined
multistart optimization.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import immrax as irx

from integrator_separating_input import (
    create_scenarios,
    optimize_multistep,
    optimize_refined_gpu,
    refined_overlap_loss,
    time_jit,
)


def run_for_order(N: int, num_restarts: int = 30, num_iters: int = 20, learning_rate: float = 0.15):
    print(f"\n{'=' * 70}\nINTERSECTION REFINEMENT vs. UNREFINED -- order N={N}\n{'=' * 70}")

    scenarios = create_scenarios(N)
    x0_ivl = irx.icentpert(jnp.zeros(N), jnp.full(N, 0.05))
    dt = 0.15
    num_steps = 6   # refinement horizon (1 initial step + (num_steps-1) refined steps)

    # ── Unrefined multistep baseline (steps_per_segment=1 so num_segments == num_steps) ──
    print("\nOptimizing UNREFINED (uninformed) separating input sequence...")
    u_seq_unrefined, loss_unrefined, _ = optimize_multistep(
        scenarios, x0_ivl, dt=dt, steps_per_segment=1, num_segments=num_steps,
        learning_rate=learning_rate, num_iters=num_iters, num_restarts=num_restarts, verbose=False,
    )
    print(f"  unrefined best loss: {loss_unrefined:.6f}")

    # ── Refinement compile vs. run time + optimization ─────────────────────
    def full_refined_multistart(seed):
        return optimize_refined_gpu(
            x0_ivl, scenarios, dt=dt, num_steps=num_steps,
            num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
        )

    _, compile_t, run_t, mem = time_jit(full_refined_multistart, 42)
    print(f"\nrefined multistart: compile {compile_t * 1e3:8.2f} ms   run {run_t * 1e3:7.3f} ms"
          f"   ({num_restarts} restarts x {num_iters} iters)")

    u_seq_refined, loss_refined, _, _ = full_refined_multistart(42)
    print(f"  refined best loss:   {float(loss_refined):.6f}")

    # ── Apples-to-apples comparison: refined loss for the UNREFINED optimum's
    #    control sequence must be <= the unrefined loss for that same sequence
    #    (refinement can only tighten, never loosen). ──────────────────────
    refined_loss_at_unrefined_u = float(
        refined_overlap_loss(u_seq_unrefined, x0_ivl, scenarios, dt=dt, num_steps=num_steps)
    )
    print(f"\nSame control sequence (unrefined-optimal u_seq):")
    print(f"  unrefined loss: {loss_unrefined:.6f}")
    print(f"  refined loss:   {refined_loss_at_unrefined_u:.6f}"
          f"  ({'<=' if refined_loss_at_unrefined_u <= loss_unrefined + 1e-5 else '>'} unrefined, as expected)")

    print(f"\nMemory snapshot: {mem}")


if __name__ == "__main__":
    print("Devices:", jax.devices())
    for N in range(2, 100):
        run_for_order(N)
    print("\nDone.")
