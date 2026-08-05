"""
ADMIRE Refinement-Based Sequence Optimisation — RK4 ODE variant
================================================================
Drop-in replacement for admire_refined_sequence_optimizer.py that
substitutes every Euler step with a fixed-step 4th-order Runge-Kutta
integrator over [0, dt] with n_substeps internal steps.

Why fixed-step RK4 instead of jax.experimental.ode.odeint
----------------------------------------------------------
odeint uses an adaptive Dormand-Prince RK45 solver implemented with
lax.while_loop.  When nested inside vmap → lax.scan → vmap(grad) →
fori_loop, XLA must compile through the while_loop's control flow at
every level, producing a massive HLO program that can take 10–60 min
to JIT on CPU.

Fixed-step RK4 with a Python for loop over n_substeps is fully
unrolled at trace time: XLA sees a flat sequence of matrix-vector
products with no branches or loops.  This gives a graph of size
O(n_substeps × 4 × emb_sys.f) — small and fast to compile.

Tradeoff: accuracy is O(h^4) with h = dt / n_substeps, so increase
n_substeps if dt is large (default 4 substeps works well for dt ≤ 1 s).

Algorithm is otherwise identical to the Euler version:
  - vmap over n scenarios for the first step
  - lax.scan + vmap over pairs for subsequent steps, with
    output-intersection refinement before each propagation
  - min overlap cost over all steps is returned

Public API
----------
propagate_with_refinement_ode(u_seq, x0_ivl, scenarios, dt, num_steps,
                               n_substeps)
    -> scalar JAX-traceable loss (differentiable w.r.t. u_seq)

optimize_refined_sequence_ode(x0_ivl, scenarios, dt, ...)
    -> (best_u_seq, best_loss, all_u_seq_final, all_losses)

UT convention (same as Euler version)
--------------------------------------
For a 9-D state interval:  x_ut = [lower (9,) | upper (9,)] — shape (18,)
  indices 3:6   = lower[3:6] = (pb, qb, rb) lower bounds
  indices 12:15 = upper[3:6] = (pb, qb, rb) upper bounds
"""

import sys
from pathlib import Path
from functools import partial
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


# ── RK4 step helper ────────────────────────────────────────────────────────────

def _make_rk4_step(emb_sys, dt: float, n_substeps: int):
    """Return  (x_ut, u, p_ut) -> x_next_ut  using fixed-step RK4 over [0, dt].

    The Python for-loop over n_substeps is unrolled at trace time, so XLA
    sees a flat, branch-free graph — no while_loop, fast to JIT.
    """
    h   = dt / n_substeps
    _t  = jnp.zeros(())   # ADMIRE dynamics are autonomous; t is unused

    def rk4_step(x_ut, u, p_ut):
        p = irx.ut2i(p_ut)   # convert once; p doesn't change across substeps
        for _ in range(n_substeps):
            k1 = emb_sys.f(_t, x_ut,                u, p)
            k2 = emb_sys.f(_t, x_ut + (h / 2) * k1, u, p)
            k3 = emb_sys.f(_t, x_ut + (h / 2) * k2, u, p)
            k4 = emb_sys.f(_t, x_ut +  h       * k3, u, p)
            x_ut = x_ut + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_ut

    return rk4_step


# ── Loss ───────────────────────────────────────────────────────────────────────

def propagate_with_refinement_ode(
    u_seq: jnp.ndarray,
    x0_ivl: irx.Interval,
    scenarios: list,
    dt: float,
    num_steps: int,
    n_substeps: int = 4,
) -> jnp.ndarray:
    """Refinement-based sequence loss using fixed-step RK4 for ADMIRE.

    Parameters
    ----------
    u_seq      : (num_steps, 10) control sequence
    x0_ivl     : initial 9-D state interval (shared across all scenarios)
    scenarios  : list of Scenario from create_scenarios(); all share emb_system
    dt         : integration duration per control step (s)
    num_steps  : number of control steps (>= 1)
    n_substeps : RK4 substeps per control step; increase for larger dt

    Returns
    -------
    Scalar (rad²) — min separation cost over all steps, differentiable w.r.t. u_seq.
    """
    n      = len(scenarios)
    pairs  = [(i, j) for i in range(n) for j in range(i + 1, n)]

    emb_sys  = scenarios[0].emb_system
    rk4_step = _make_rk4_step(emb_sys, dt, n_substeps)

    # Stack parameter intervals: (n, 20)
    p_stack  = jnp.stack([irx.i2ut(s.p_interval) for s in scenarios])
    pair_i   = jnp.array([i for i, j in pairs])   # (P,)
    pair_j   = jnp.array([j for i, j in pairs])   # (P,)
    pi_stack = p_stack[pair_i]   # (P, 20)
    pj_stack = p_stack[pair_j]   # (P, 20)

    # ── Step 1: vmap over all n scenarios ─────────────────────────────────
    x0_ut = irx.i2ut(x0_ivl)  # (18,)

    def one_scenario_step(p_ut, u):
        return rk4_step(x0_ut, u, p_ut)

    x1_stack = jax.vmap(one_scenario_step, in_axes=(0, None))(
        p_stack, u_seq[0]
    )  # (n, 18)

    xi_stack = x1_stack[pair_i]  # (P, 18)
    xj_stack = x1_stack[pair_j]  # (P, 18)

    def pair_overlap_cost(xi_ut, xj_ut):
        ov_lo = jnp.maximum(xi_ut[3:6],   xj_ut[3:6])
        ov_hi = jnp.minimum(xi_ut[12:15], xj_ut[12:15])
        return jnp.prod(jnp.maximum(ov_hi - ov_lo, 0.0))

    cost0 = jnp.sum(jax.vmap(pair_overlap_cost)(xi_stack, xj_stack))

    # ── Steps 2 … num_steps: lax.scan + vmap over pairs ──────────────────
    def step_one_pair(xi_ut, xj_ut, pi_ut, pj_ut, u_step):
        # Output intersection: indices 3:6 (lower) and 12:15 (upper)
        y_lo = jnp.maximum(xi_ut[3:6],   xj_ut[3:6])
        y_hi = jnp.minimum(xi_ut[12:15], xj_ut[12:15])
        has_overlap = jnp.all(y_hi >= y_lo)

        fallback  = (xi_ut[3:6] + xi_ut[12:15]) / 2
        y_lo_safe = jnp.where(has_overlap, y_lo, fallback)
        y_hi_safe = jnp.where(has_overlap, y_hi, fallback)

        # Refine state intervals
        xi_ref_ut = jnp.concatenate([
            xi_ut[:9].at[3:6].set(y_lo_safe),
            xi_ut[9:].at[3:6].set(y_hi_safe),
        ])
        xj_ref_ut = jnp.concatenate([
            xj_ut[:9].at[3:6].set(y_lo_safe),
            xj_ut[9:].at[3:6].set(y_hi_safe),
        ])

        # Fixed-step RK4 (replaces odeint)
        xn_i_ut = rk4_step(xi_ref_ut, u_step, pi_ut)
        xn_j_ut = rk4_step(xj_ref_ut, u_step, pj_ut)

        # Overlap cost of propagated states
        ov_lo = jnp.maximum(xn_i_ut[3:6],   xn_j_ut[3:6])
        ov_hi = jnp.minimum(xn_i_ut[12:15], xn_j_ut[12:15])
        raw       = jnp.prod(jnp.maximum(ov_hi - ov_lo, 0.0))
        pair_cost = jnp.where(has_overlap, raw, 0.0)

        return xn_i_ut, xn_j_ut, pair_cost

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

    # No jax.checkpoint: removing it reduces backward-pass tracing overhead.
    # Re-add if GPU memory becomes a bottleneck (O(num_steps) activations).
    (_, _, scan_min_cost), _ = jax.lax.scan(scan_body, init_carry, u_seq[1:])

    return jnp.minimum(cost0, scan_min_cost)


# ── Optimizer ──────────────────────────────────────────────────────────────────

def optimize_refined_sequence_ode(
    x0_ivl: irx.Interval,
    scenarios: list,
    dt: float,
    num_steps: int = 5,
    num_restarts: int = 8,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
    n_substeps: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Multi-start gradient descent on the RK4-based refinement loss.

    Parameters
    ----------
    x0_ivl        : initial 9-D state interval
    scenarios     : list of Scenario from create_scenarios()
    dt            : integration duration per step (s)
    num_steps     : control sequence length
    num_restarts  : parallel random restarts (vmap batch dimension).
                    Smaller values reduce JIT time; 8 is a safe default.
    learning_rate : gradient-descent step size
    num_iters     : gradient steps per restart (lax.fori_loop iterations)
    seed          : PRNG seed for random initialisation
    n_substeps    : RK4 substeps per control step (accuracy vs graph-size tradeoff)

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
        return propagate_with_refinement_ode(
            u_seq, x0_ivl=x0_ivl, scenarios=scenarios,
            dt=dt, num_steps=num_steps, n_substeps=n_substeps,
        )

    batched_loss = jax.vmap(loss_fn)
    batched_grad = jax.vmap(jax.grad(loss_fn))

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final  = jax.lax.fori_loop(0, num_iters, body, u0)
    losses   = batched_loss(u_final)

    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx  = jnp.argmin(losses_valid)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from admire_separating_input import create_scenarios

    scenarios = create_scenarios(fault_effectiveness=0.0)[:10]
    x0_nom   = jnp.zeros(9).at[0].set(343.0 * 0.3)
    x0_ivl   = irx.icentpert(x0_nom, jnp.ones(9) * 0.01)

    best_u, best_loss, _, all_losses = optimize_refined_sequence_ode(
        x0_ivl=x0_ivl,
        scenarios=scenarios,
        dt=1.0,
        num_steps=5,
        num_restarts=8,
        learning_rate=0.05,
        num_iters=200,
        seed=42,
        n_substeps=4,
    )
    print(f"Best refined-sequence (RK4) loss: {float(best_loss):.6f} rad²")
    print(f"Best u_seq shape: {best_u.shape}")
    print(f"Loss distribution — min: {float(jnp.min(all_losses)):.4f}  "
          f"mean: {float(jnp.mean(all_losses)):.4f}  "
          f"max: {float(jnp.max(all_losses)):.4f}")
