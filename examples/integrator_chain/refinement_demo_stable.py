"""
Numerically-stable variant of refinement_demo.py's N=2..100 refinement sweep.

Diagnosis (from run 11754504's log, logs/integrator_refinement_scaling_11754504.out)
--------------------------------------------------------------------------------------
`optimize_refined_gpu`'s reported best loss goes to `nan` for every N from 65
through 99, while three things stay clean at those same N:
  - `optimize_multistep`'s (unrefined) reported best loss: exactly 0.0.
  - `refined_overlap_loss` called directly (no grad, no optimizer) on the
    unrefined-optimal control sequence: exactly 0.0, even at N=99.
  - `optimize_refined_gpu` uses the *same* multi-restart / learning-rate /
    iteration-count / control-box hyperparameters as the unrefined optimizer.

That the bug appears only when `jax.grad` is taken through
`propagate_with_refinement` -- never in a bare forward call -- is the
signature of JAX's documented "NaN from an unselected `where` branch"
gotcha (https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#nans):
`jax.grad(jnp.where(cond, a, b))` still traces and differentiates *both* `a`
and `b`, even though only one is selected in the forward pass. In
`propagate_with_refinement`'s per-pair step body, `_invert_observation` (a
division by `beta`) is called unconditionally on `y_int`, and only *after*
that is `has_overlap` used to select between the refined and un-refined
state via `jnp.where`. At high N, per-pair output intervals shrink toward
float32 machine epsilon (consistent with the unrefined loss's clean
underflow to exactly 0.0 in the same N range), so `has_overlap` flips False
far more often than at low N -- and whenever it does, `_invert_observation`
is still differentiated on that step's (now provably degenerate/borderline)
`y_int`. If that local gradient is ever `inf` (or the branch's forward value
ever is, from upstream interval-wrapping growth), `inf * 0` -- the zeroed-out
upstream cotangent from the *untaken* branch -- is `nan`, and it poisons the
whole restart's gradient from then on regardless of which branch forward
selected. `jnp.argmin`/`jnp.min` (used to pick the best of the 30 restarts)
then treat that one poisoned restart's `nan` as the "minimum", clobbering the
whole reported result even when other restarts converged fine.

(A log/log1p reparameterization of the *final* scalar loss, suggested as an
option to explore, would not have fixed this: the nan is manufactured
upstream, inside a single pair's gradient, before the per-step `jnp.min`
reduction the final loss is built from -- log1p(nan) is still nan. Rescaling
the loss doesn't change where or whether that nan is produced, so this file
does not use it.)

Fix
---
1. `propagate_with_refinement_stable`: the standard "double `where`" guard --
   before calling `_invert_observation`, replace `y_int` with a safe,
   well-conditioned dummy interval whenever `has_overlap` is False, so the
   untaken branch's local gradient stays finite (JAX's own recommended
   pattern for this exact gotcha). When `has_overlap` is True this is
   mathematically a no-op: the safe substitution is only used in the branch
   whose result gets discarded anyway.
2. `optimize_refined_gpu_stable`: two cheap, unconditionally-correct
   defense-in-depth additions on top of (1), in case some other numerical
   path (e.g. genuine interval overflow from the wrapping effect at very
   high N, unrelated to the where-gotcha) still produces a stray nan/inf
   gradient for a restart:
     - `jnp.nan_to_num` on each restart's gradient before the parameter
       update, so a poisoned restart freezes at its last finite iterate
       instead of the nan permanently contaminating all of its future
       iterates (`u - lr*nan == nan` forever after, otherwise).
     - `jnp.nanargmin`/select over final losses instead of `jnp.argmin`, so
       one restart still going nan (despite (1) and the gradient guard)
       can't mask an otherwise-valid best among the other restarts.

Only the refined path is touched -- `optimize_multistep` (unrefined) already
runs clean at every N per the diagnosis above, so it's imported unchanged.
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
    Scenario,
    create_scenarios,
    optimize_multistep,
    euler_step,
    observed_output,
    _invert_observation,
    _overlap_volume,
    _project_u,
    _run_unrolled_or_loop,
    _run_unrolled_or_loop_nocheckpoint,
    time_jit,
    HOST_PEAK_RSS,
)

from typing import List


def propagate_with_refinement_stable(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                     scenarios: List[Scenario], dt: float,
                                     num_steps: int = 2) -> jnp.ndarray:
    """Same as integrator_separating_input.propagate_with_refinement, with a
    double-`where` guard around `_invert_observation`'s input -- see module
    docstring for why. Only the per-pair step body differs from the
    original; step 1 and the overall loop structure are unchanged."""
    n = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    N = x0_ivl.lower.shape[0]

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:N], upper=arr[N:])

    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x1_batch = jax.vmap(lambda p: euler_step(emb_sys, x0_ivl, u_seq[0], p, dt))(p_batch)
    x1_ivls = [irx.Interval(lower=x1_batch.lower[i], upper=x1_batch.upper[i]) for i in range(n)]
    y1_ivls = [observed_output(x1, s) for x1, s in zip(x1_ivls, scenarios)]
    step1_cost = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            step1_cost = step1_cost + _overlap_volume(y1_ivls[i], y1_ivls[j])

    pxi_arr = jnp.stack([ivl_to_arr(x1_ivls[i]) for i, j in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x1_ivls[j]) for i, j in pairs])

    def step_body(carry, k):
        pxi_arr, pxj_arr, min_cost = carry
        u_k = u_seq[k + 1]
        step_cost = jnp.array(0.0)
        new_pxi, new_pxj = [], []

        for idx, (i, j) in enumerate(pairs):
            x_curr_i = arr_to_ivl(pxi_arr[idx])
            x_curr_j = arr_to_ivl(pxj_arr[idx])

            y_i = observed_output(x_curr_i, scenarios[i])
            y_j = observed_output(x_curr_j, scenarios[j])

            y_lo = jnp.maximum(y_i.lower, y_j.lower)
            y_hi = jnp.minimum(y_i.upper, y_j.upper)
            has_overlap = jnp.all(y_hi >= y_lo)

            # Double-`where` guard (see module docstring): feed
            # _invert_observation a safe, bounded interval whenever
            # has_overlap is False, so its (about-to-be-discarded) branch
            # can't manufacture an inf/nan local gradient. When has_overlap
            # is True this is exactly (y_lo, y_hi) -- no behavior change.
            y_lo_safe = jnp.where(has_overlap, y_lo, jnp.zeros_like(y_lo))
            y_hi_safe = jnp.where(has_overlap, y_hi, jnp.ones_like(y_hi))
            y_int_safe = irx.Interval(lower=y_lo_safe, upper=y_hi_safe)

            x_ref_i_overlap = _invert_observation(y_int_safe, scenarios[i])
            x_ref_j_overlap = _invert_observation(y_int_safe, scenarios[j])

            x_ref_i = irx.Interval(
                lower=jnp.where(has_overlap, x_ref_i_overlap.lower, x_curr_i.lower),
                upper=jnp.where(has_overlap, x_ref_i_overlap.upper, x_curr_i.upper),
            )
            x_ref_j = irx.Interval(
                lower=jnp.where(has_overlap, x_ref_j_overlap.lower, x_curr_j.lower),
                upper=jnp.where(has_overlap, x_ref_j_overlap.upper, x_curr_j.upper),
            )

            x_next_i = euler_step(scenarios[i].emb_system, x_ref_i, u_k, scenarios[i].p_interval, dt)
            x_next_j = euler_step(scenarios[j].emb_system, x_ref_j, u_k, scenarios[j].p_interval, dt)

            raw_cost = _overlap_volume(
                observed_output(x_next_i, scenarios[i]),
                observed_output(x_next_j, scenarios[j]),
            )
            step_cost = step_cost + jnp.where(has_overlap, raw_cost, jnp.array(0.0))
            new_pxi.append(ivl_to_arr(x_next_i))
            new_pxj.append(ivl_to_arr(x_next_j))

        return (jnp.stack(new_pxi), jnp.stack(new_pxj), jnp.minimum(min_cost, step_cost))

    init_carry = (pxi_arr, pxj_arr, step1_cost)
    _, _, min_cost_final = _run_unrolled_or_loop(step_body, init_carry, num_steps - 1)
    return min_cost_final


def refined_overlap_loss_stable(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                                scenarios: List[Scenario], dt: float,
                                num_steps: int = 2) -> jnp.ndarray:
    return propagate_with_refinement_stable(x0_ivl, u_seq, scenarios, dt, num_steps)


def optimize_refined_gpu_stable(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                                num_steps: int = 2, num_restarts: int = 50,
                                learning_rate: float = 0.05, num_iters: int = 200,
                                seed: int = 42):
    """Same as integrator_separating_input.optimize_refined_gpu, but on
    refined_overlap_loss_stable and with the nan-defense described in the
    module docstring (gradient nan_to_num + nanargmin restart selection)."""
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, num_steps, 1)) * 0.3 + jnp.array([0.5])

    def loss_fn_refined(u_seq):
        return refined_overlap_loss_stable(u_seq, x0_ivl, scenarios, dt, num_steps)

    batched_loss = jax.vmap(loss_fn_refined)
    batched_grad = jax.vmap(jax.grad(loss_fn_refined))

    def body(u_batch, _i):
        g = batched_grad(u_batch)
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return _project_u(u_batch - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.nanargmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def run_for_order(N: int, num_restarts: int = 30, num_iters: int = 20, learning_rate: float = 0.15):
    HOST_PEAK_RSS.reset()
    print(f"\n{'=' * 70}\nINTERSECTION REFINEMENT (stable) vs. UNREFINED -- order N={N}\n{'=' * 70}")

    scenarios = create_scenarios(N)
    x0_ivl = irx.icentpert(jnp.zeros(N), jnp.full(N, 0.05))
    dt = 0.15
    num_steps = 6

    print("\nOptimizing UNREFINED (uninformed) separating input sequence...")
    u_seq_unrefined, loss_unrefined, _ = optimize_multistep(
        scenarios, x0_ivl, dt=dt, steps_per_segment=1, num_segments=num_steps,
        learning_rate=learning_rate, num_iters=num_iters, num_restarts=num_restarts, verbose=False,
    )
    print(f"  unrefined best loss: {loss_unrefined:.6f}")

    def full_refined_multistart(seed):
        return optimize_refined_gpu_stable(
            x0_ivl, scenarios, dt=dt, num_steps=num_steps,
            num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
        )

    _, compile_t, run_t, mem = time_jit(full_refined_multistart, 42)
    print(f"\nrefined multistart: compile {compile_t * 1e3:8.2f} ms   run {run_t * 1e3:7.3f} ms"
          f"   ({num_restarts} restarts x {num_iters} iters)")

    u_seq_refined, loss_refined, _, all_losses = full_refined_multistart(42)
    num_nan_restarts = int(jnp.sum(jnp.isnan(all_losses)))
    print(f"  refined best loss:   {float(loss_refined):.6f}"
          f"   ({num_nan_restarts}/{num_restarts} restarts went nan)")

    refined_loss_at_unrefined_u = float(
        refined_overlap_loss_stable(u_seq_unrefined, x0_ivl, scenarios, dt=dt, num_steps=num_steps)
    )
    print(f"\nSame control sequence (unrefined-optimal u_seq):")
    print(f"  unrefined loss: {loss_unrefined:.6f}")
    print(f"  refined loss:   {refined_loss_at_unrefined_u:.6f}"
          f"  ({'<=' if refined_loss_at_unrefined_u <= loss_unrefined + 1e-5 else '>'} unrefined, as expected)")

    print(f"\nMemory snapshot: {mem}")


if __name__ == "__main__":
    print("Devices:", jax.devices())
    HOST_PEAK_RSS.start()
    for N in range(2, 101):
        run_for_order(N)
    print("\nDone.")
