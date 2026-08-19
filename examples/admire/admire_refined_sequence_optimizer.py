"""
ADMIRE Refinement-Based Sequence Optimisation  (memory-optimised)
=================================================================
Optimises a control sequence u_seq of shape (num_steps, 10) using an
observation-based state refinement loss.

Key idea (mirrors faulty_car.propagate_with_refinement)
-------------------------------------------------------
At each step k the shared observed output — state[3:6] = (pb, qb, rb) — is
intersected across every scenario pair (i, j).  The intersection tightens
both scenarios' state intervals to the region that is consistent with the
common observation before propagating to step k+1.

ADMIRE-specific simplification: output = state[3:6] with no sensor offset or
scale, so the inverse observation map is the identity — the refined state
simply replaces indices 3,4,5 with the intersection bounds.

Memory-optimisation strategy
-----------------------------
All 10 scenarios share one embedding system (_ADMIRE_EMB); only p_interval
differs per scenario.  Exploiting this:

* vmap over n scenarios for Step 1           → 1 euler_step graph (not 10)
* vmap over n_pairs for the scan body        → 2 euler_step graphs (not 90)
* lax.scan over time steps                   → O(1) graph in num_steps
* jax.checkpoint on scan body                → O(√num_steps) backward memory
* Flat UT arrays in carry; no Python for-loops inside JAX-traced functions

This reduces the XLA HLO program from O(n_pairs × euler_step) to
O(euler_step), bringing compilation memory from ~30 GB down to ~1–3 GB.

UT convention (irx.i2ut / irx.ut2i)
-------------------------------------
For a 9-D state interval:  x_ut = [lower (9,) | upper (9,)] — shape (18,).
  x_ut[0:9]  = lower,   x_ut[9:18]  = upper
  x_ut[3:6]  = lower[3:6] (pb,qb,rb lower)
  x_ut[12:15]= upper[3:6] (pb,qb,rb upper)

For a 10-D param interval: p_ut = [lower (10,) | upper (10,)] — shape (20,).

Public API
----------
propagate_with_refinement_admire(u_seq, x0_ivl, scenarios, dt, num_steps)
    -> scalar JAX-traceable loss (differentiable w.r.t. u_seq)

optimize_refined_sequence_gpu(x0_ivl, scenarios, dt, ...)
    -> (best_u_seq, best_loss, all_u_seq_final, all_losses)
"""

import sys
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import immrax as irx

HERE     = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
for p in (str(HERE), str(EXAMPLES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from admire_separating_input import _project_u   # clips control to [-0.05, 0.05]^10


# ── Loss ──────────────────────────────────────────────────────────────────────

def propagate_with_refinement_admire(
    u_seq: jnp.ndarray,
    x0_ivl: irx.Interval,
    scenarios: list,
    dt: float,
    num_steps: int,
) -> jnp.ndarray:
    """Memory-optimised refinement-based sequence loss for ADMIRE.

    Algorithm
    ---------
    Step 1:
        vmap one Euler step over all n scenarios simultaneously (1 graph).
        cost[0] = Σ_{i<j} overlap_volume(output(xᵢ), output(xⱼ))

    Steps 2 … num_steps (lax.scan, body vmapped over n_pairs):
        For each pair (i, j) — handled with a single vmap, not a Python loop:
        1. Intersect output intervals:  y = state[3:6]ᵢ ∩ state[3:6]ⱼ
        2. Refine: replace lower/upper[3:6] with intersection bounds.
        3. Euler step with shared emb_sys but pair-specific p_interval.
        4. cost[step] = Σ overlap_volume, masked 0 if already separated.

    Returns min cost over all steps.

    XLA graph size
    --------------
    Original (Python for-loops): O(n_pairs × euler_step) ≈ 90 euler_steps.
    This version (vmap):         O(euler_step)            — 2 graphs total.
    Compilation memory: ~1–3 GB instead of ~30 GB.

    Parameters
    ----------
    u_seq     : (num_steps, 10) control sequence
    x0_ivl    : initial 9-D state interval
    scenarios : list of Scenario from create_scenarios(); all share emb_system
    dt        : Euler step size (s)
    num_steps : total propagation steps (≥ 1)

    Returns
    -------
    Scalar (rad²) — differentiable w.r.t. u_seq.
    """
    n      = len(scenarios)
    pairs  = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Shared embedding system (all scenarios use the same one)
    emb_sys = scenarios[0].emb_system
    _t      = jnp.zeros(())   # JAX scalar t=0 (must not be a Python int)

    # Stack parameter intervals once: (n, 20) and (n_pairs, 20)
    p_stack  = jnp.stack([irx.i2ut(s.p_interval) for s in scenarios])  # (n, 20)
    pair_i   = jnp.array([i for i, j in pairs])                         # (P,)
    pair_j   = jnp.array([j for i, j in pairs])                         # (P,)
    pi_stack = p_stack[pair_i]   # (P, 20) — scenario i param per pair
    pj_stack = p_stack[pair_j]   # (P, 20) — scenario j param per pair

    # ── Step 1: vmap over all n scenarios (1 euler_step graph) ────────────
    x0_ut = irx.i2ut(x0_ivl)  # (18,) — shared starting state

    def one_scenario_step(p_ut, u):
        """Single Euler step from x0_ut under parameter interval p_ut."""
        dx_ut = emb_sys.f(_t, x0_ut, u, irx.ut2i(p_ut))
        return dx_ut * dt + x0_ut

    x1_stack = jax.vmap(one_scenario_step, in_axes=(0, None))(
        p_stack, u_seq[0]
    )  # (n, 18)

    # Initial pair states — index into x1_stack (no copy needed for XLA)
    xi_stack = x1_stack[pair_i]  # (P, 18)
    xj_stack = x1_stack[pair_j]  # (P, 18)

    # Initial cost: vmap overlap computation over all pairs (1 graph)
    def pair_overlap_cost(xi_ut, xj_ut):
        ov_lo = jnp.maximum(xi_ut[3:6],  xj_ut[3:6])   # lower of intersection
        ov_hi = jnp.minimum(xi_ut[12:15], xj_ut[12:15]) # upper of intersection
        return jnp.prod(jnp.maximum(ov_hi - ov_lo, 0.0))

    cost0 = jnp.sum(jax.vmap(pair_overlap_cost)(xi_stack, xj_stack))

    # ── Steps 2 … num_steps: lax.scan + vmap over pairs ──────────────────
    def step_one_pair(xi_ut, xj_ut, pi_ut, pj_ut, u_step):
        """Refine + Euler step for one pair.  All args are flat UT arrays."""
        # Output intersection (state[3:6] = pb, qb, rb)
        y_lo = jnp.maximum(xi_ut[3:6],  xj_ut[3:6])
        y_hi = jnp.minimum(xi_ut[12:15], xj_ut[12:15])
        has_overlap = jnp.all(y_hi >= y_lo)

        fallback  = (xi_ut[3:6] + xi_ut[12:15]) / 2   # output centre of xi
        y_lo_safe = jnp.where(has_overlap, y_lo, fallback)
        y_hi_safe = jnp.where(has_overlap, y_hi, fallback)

        # Refine: overwrite output indices 3:6 in lower and upper halves
        xi_ref_ut = jnp.concatenate([
            xi_ut[:9].at[3:6].set(y_lo_safe),
            xi_ut[9:].at[3:6].set(y_hi_safe),
        ])
        xj_ref_ut = jnp.concatenate([
            xj_ut[:9].at[3:6].set(y_lo_safe),
            xj_ut[9:].at[3:6].set(y_hi_safe),
        ])

        # Euler step with shared dynamics but pair-specific parameter intervals
        xn_i_ut = emb_sys.f(_t, xi_ref_ut, u_step, irx.ut2i(pi_ut)) * dt + xi_ref_ut
        xn_j_ut = emb_sys.f(_t, xj_ref_ut, u_step, irx.ut2i(pj_ut)) * dt + xj_ref_ut

        # Output overlap of propagated states
        ov_lo = jnp.maximum(xn_i_ut[3:6],  xn_j_ut[3:6])
        ov_hi = jnp.minimum(xn_i_ut[12:15], xn_j_ut[12:15])
        raw   = jnp.prod(jnp.maximum(ov_hi - ov_lo, 0.0))
        pair_cost = jnp.where(has_overlap, raw, 0.0)

        return xn_i_ut, xn_j_ut, pair_cost

    # vmap over P pairs at once: O(1) XLA graph (2 euler_step graphs total)
    step_all_pairs = jax.vmap(step_one_pair, in_axes=(0, 0, 0, 0, None))

    def scan_body(carry, u_step):
        xi_stack, xj_stack, min_cost = carry
        xn_i_stack, xn_j_stack, pair_costs = step_all_pairs(
            xi_stack, xj_stack, pi_stack, pj_stack, u_step
        )
        step_cost    = jnp.sum(pair_costs)
        new_min_cost = jnp.minimum(min_cost, step_cost)
        return (xn_i_stack, xn_j_stack, new_min_cost), None

    init_carry = (xi_stack, xj_stack, cost0)

    # checkpoint → O(√num_steps) backward-pass activation memory
    (_, _, scan_min_cost), _ = jax.lax.scan(
        jax.checkpoint(scan_body), init_carry, u_seq[1:]
    )

    return jnp.minimum(cost0, scan_min_cost)


# ── Optimizer ─────────────────────────────────────────────────────────────────

def optimize_refined_sequence_gpu(
    x0_ivl: irx.Interval,
    scenarios: list,
    dt: float,
    num_steps: int = 5,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent on the refinement loss.

    Batches all restarts with vmap and runs gradient descent inside a single
    lax.fori_loop so the entire optimisation executes as one XLA kernel.
    Designed for GPU; on CPU compilation is slow for large num_steps.

    Parameters
    ----------
    x0_ivl        : initial 9-D state interval
    scenarios     : list of Scenario from create_scenarios()
    dt            : Euler step size (s)
    num_steps     : control sequence length (horizon = num_steps × dt s)
    num_restarts  : parallel restarts (batch dimension for vmap)
    learning_rate : gradient-descent step size
    num_iters     : gradient steps per restart (lax.fori_loop iterations)
    seed          : PRNG seed for random initialisation

    Returns
    -------
    best_u_seq      : (num_steps, 10) best control sequence found
    best_loss       : scalar refined overlap loss at best_u_seq
    all_u_seq_final : (num_restarts, num_steps, 10) final sequences per restart
    all_losses      : (num_restarts,) final loss per restart
    """
    key = jax.random.PRNGKey(seed)
    u0  = jax.random.uniform(
        key, (num_restarts, num_steps, 10), minval=-0.1, maxval=0.1
    )

    def loss_fn(u_seq):
        return propagate_with_refinement_admire(
            u_seq, x0_ivl=x0_ivl, scenarios=scenarios,
            dt=dt, num_steps=num_steps,
        )

    batched_loss = jax.vmap(loss_fn)            # (R, S, 10) → (R,)
    batched_grad = jax.vmap(jax.grad(loss_fn))  # (R, S, 10) → (R, S, 10)

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final  = jax.lax.fori_loop(0, num_iters, body, u0)
    losses   = batched_loss(u_final)

    # Replace NaN losses with +inf so argmin ignores invalid restarts
    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx  = jnp.argmin(losses_valid)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_refined_sequence_gpu_fused(
    x0_ivl: irx.Interval,
    scenarios: list,
    dt: float,
    num_steps: int = 5,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
    init_scale: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Same as optimize_refined_sequence_gpu but fuses batched_loss/
    batched_grad into one jax.vmap(jax.value_and_grad(...)) call instead of
    separate tracings (same fix as admire_separating_input.py's
    optimize_multistep_gpu_fused / faulty_car_output_feedback_cbf.py's CBF
    optimizer). propagate_with_refinement_admire itself was already
    vmapped over scenarios/pairs (see its docstring), so this only fixes
    the outer optimizer's redundant forward-graph tracing.

    Caller should wrap this in an outer jax.jit before timing/deploying it.

    `init_scale`: half-width of the uniform u0 sampling range (was
    hardcoded to 0.1). _project_u's actual clip bound is +-0.5 rad; see
    optimize_multistep_gpu_fused's init_scale docstring for why widening
    this (this session's finding, mirroring faulty_car_output_feedback_cbf.py)
    matters for letting GD actually move instead of stalling near a narrow
    near-zero start.

    IMPORTANT (see faulty_car_output_feedback_cbf.py's session history):
    a refined loss of ~0 is a genuine diagnosability certificate ONLY if
    checked against what the refinement mechanism actually computes; it is
    NOT automatically the same as the raw (independent per-scenario) boxes
    being disjoint. Always verify a low loss against raw per-segment
    overlap (see this module's __main__ / multistep_refined_demo.py) before
    trusting it, especially when init_scale/learning_rate/num_restarts are
    pushed aggressively to hit a runtime budget.
    """
    key = jax.random.PRNGKey(seed)
    u0  = jax.random.uniform(
        key, (num_restarts, num_steps, 10), minval=-init_scale, maxval=init_scale
    )

    def loss_fn(u_seq):
        return propagate_with_refinement_admire(
            u_seq, x0_ivl=x0_ivl, scenarios=scenarios,
            dt=dt, num_steps=num_steps,
        )

    batched_value_and_grad = jax.vmap(jax.value_and_grad(loss_fn))

    def body(_, carry):
        u_batch, _prev_losses = carry
        losses, g = batched_value_and_grad(u_batch)
        return (_project_u(u_batch - learning_rate * g), losses)

    init_losses = jnp.zeros(num_restarts)
    u_final, losses = jax.lax.fori_loop(0, num_iters, body, (u0, init_losses))
    losses, _ = batched_value_and_grad(u_final)

    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx  = jnp.argmin(losses_valid)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


# ── Example usage (run on GPU) ─────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    from admire_separating_input import create_scenarios

    scenarios = create_scenarios(fault_effectiveness=0.0)[:10]
    x0_nom   = jnp.zeros(9).at[0].set(343.0 * 0.3)
    x0_ivl   = irx.icentpert(x0_nom, jnp.ones(9) * 0.01)

    best_u, best_loss, _, all_losses = optimize_refined_sequence_gpu(
        x0_ivl=x0_ivl,
        scenarios=scenarios,
        dt=0.1,
        num_steps=5,
        num_restarts=50,
        learning_rate=0.05,
        num_iters=200,
        seed=42,
    )
    print(f"Best refined-sequence loss: {float(best_loss):.6f} rad²")
    print(f"Best u_seq shape: {best_u.shape}")
    print(f"Loss distribution — min: {float(jnp.min(all_losses)):.4f}  "
          f"mean: {float(jnp.mean(all_losses)):.4f}  "
          f"max: {float(jnp.max(all_losses)):.4f}")
