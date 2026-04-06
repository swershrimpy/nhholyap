"""
Faulty Car Output-Feedback Controller with Collision-Avoidance CBF
==================================================================
Solves for a linear output-feedback controller K ∈ R^{2×2}, r ∈ R^2 that:

  1. Maximally separates the reachable sets of three fault scenarios in the
     (px, py) position space — enabling active fault diagnosis.
  2. Maintains the collision-avoidance Control Barrier Function (CBF)

         h(x) = (px − cx)² + (py − cy)² − r_obs²  ≥  0

     for all states in the reachable interval across every scenario.

Controller
----------
    u_k = clip(K @ y_k + r,  u_lo, u_hi)

where y_k is the robot's *observed* position:
  Nominal / Actuator-fault scenario  : y = [px, py]
  Sensor-fault scenario              : y = [0.95·px + 0.2,  0.95·py + 0.2]

Each closed-loop system encodes K and r *inside* its f(), so passing
theta = [K.flatten(), r] as the 'u' argument to the natural embedding
lets the closed-loop vector field be bounded by interval arithmetic.

Intersection Refinement
-----------------------
At each step k the shared observed output of every scenario pair (i, j) is
intersected.  The tighter intersection is used as the refined initial interval
for step k+1, producing tighter reachable bounds and a more informative loss.

CBF Safety Penalty
------------------
Collision-avoidance is enforced as a soft penalty on the *worst-case*
(unrefined, per-scenario) reachable interval:

    cbf_penalty(x_ivl) = max(0, − min_{x ∈ x_ivl} h(x))

The minimum of h over a position-interval box [pxl, pxu]×[pyl, pyu] is
achieved at the closest point to (cx, cy) — a convex minimisation:

    px* = clip(cx, pxl, pxu),  py* = clip(cy, pyl, pyu)
    h_min = (px* − cx)² + (py* − cy)² − r_obs²

Decision variable
-----------------
    theta ∈ R^6 = [K.flatten(), r]    K : (2,2) gain,  r : (2,) feedforward

Optimization
------------
    min_{theta}  separation_loss_refined(theta)  +  cbf_weight · cbf_penalty(theta)

solved with batched gradient descent (vmap + lax.fori_loop).
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
for _p in (str(_HERE), str(_EXAMPLES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np
from functools import partial
from typing import List, Tuple, Sequence, Dict

from faulty_car_separating_input import (
    Scenario,
    _obs_interval,
    _U_LO,
    _U_HI,
)
from interval_functions import overlap_size_lax


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Controller parameterisation
# ══════════════════════════════════════════════════════════════════════════════

_K_MAX = 5.0
# theta = [K00, K01, K10, K11,  r0, r1]   (6-D)
_THETA_LO = jnp.concatenate([jnp.full(4, -_K_MAX), _U_LO])
_THETA_HI = jnp.concatenate([jnp.full(4,  _K_MAX), _U_HI])

_OBS_OFFSET_SENSOR = jnp.array([0.2, 0.2])
_OBS_SCALE_SENSOR  = jnp.array([0.95])


def _project_theta(theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(theta, _THETA_LO, _THETA_HI)


def theta_to_K_r(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return theta[:4].reshape(2, 2), theta[4:6]


def K_r_to_theta(K: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([K.flatten(), r])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Closed-loop system classes
# ══════════════════════════════════════════════════════════════════════════════

class CarNomActCLSystem(irx.System):
    """Nominal / actuator-fault nonholonomic car under linear output feedback.

    The controller receives the true position as observation:
        y = [px, py]

    immrax convention: u carries controller parameters theta.
        u = theta = [K.flatten(), r]  ∈ R^6     (concrete, optimisation variable)
        p = [alpha]                    ∈ R^1     (abstract; 1=nominal, <1=fault)

    Dynamics:
        ṗx = v · cos φ
        ṗy = v · sin φ
        φ̇  = alpha · ω
    where   [v, ω] = clip(K @ [px, py] + r,  u_lo, u_hi)
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        K    = u[:4].reshape(2, 2)
        r_ff = u[4:6]
        y    = x[0:2]                            # true position observation
        ctrl = jnp.clip(K @ y + r_ff, _U_LO, _U_HI)
        v, omega = ctrl[0], ctrl[1]
        alpha = p[0]
        phi   = x[2]
        return jnp.array([
            v * jnp.cos(phi),
            v * jnp.sin(phi),
            alpha * omega,
        ])


class CarSensorFaultCLSystem(irx.System):
    """Sensor-fault nonholonomic car under linear output feedback.

    The sensor reports a biased, scaled position:
        y = scale · [px, py] + offset   (scale = 0.95, offset = [0.2, 0.2])

    The robot applies the controller to this corrupted measurement, so it
    drives as if the bias did not exist.

    immrax convention: u carries controller parameters theta.
        u = theta = [K.flatten(), r]  ∈ R^6
        p = [alpha]                    ∈ R^1   (= 1 for pure sensor fault)

    Dynamics (same as nominal but with biased control):
        ṗx = v · cos φ
        ṗy = v · sin φ
        φ̇  = alpha · ω
    where   [v, ω] = clip(K @ (scale·[px,py] + offset) + r,  u_lo, u_hi)
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        K    = u[:4].reshape(2, 2)
        r_ff = u[4:6]
        y    = _OBS_SCALE_SENSOR[0] * x[0:2] + _OBS_OFFSET_SENSOR  # biased
        ctrl = jnp.clip(K @ y + r_ff, _U_LO, _U_HI)
        v, omega = ctrl[0], ctrl[1]
        alpha = p[0]
        phi   = x[2]
        return jnp.array([
            v * jnp.cos(phi),
            v * jnp.sin(phi),
            alpha * omega,
        ])


# Module-level singletons
_NOM_ACT_CL_SYS = CarNomActCLSystem()
_NOM_ACT_CL_EMB = irx.natemb(_NOM_ACT_CL_SYS)

_SF_CL_SYS = CarSensorFaultCLSystem()
_SF_CL_EMB = irx.natemb(_SF_CL_SYS)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Closed-loop fault scenarios
# ══════════════════════════════════════════════════════════════════════════════

def create_cl_scenarios(
    actuator_alpha_lo: float = 0.0,
    actuator_alpha_hi: float = 0.5,
) -> List[Scenario]:
    """Three closed-loop fault scenarios for output-feedback optimisation.

    Uses the CL embeddings so that theta (K, r) is the 'u' argument during
    propagation.  obs_offset and obs_scale mirror faulty_car_separating_input.py.

    Returns
    -------
    [Nominal, Actuator-Fault, Sensor-Fault]
    """
    return [
        Scenario(
            name="Nominal",
            emb_system=_NOM_ACT_CL_EMB,
            p_interval=irx.icentpert(jnp.array([1.0]), jnp.zeros(1)),
            obs_offset=jnp.zeros(2),
            obs_scale=jnp.ones(1),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=_NOM_ACT_CL_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo]),
                upper=jnp.array([actuator_alpha_hi]),
            ),
            obs_offset=jnp.zeros(2),
            obs_scale=jnp.ones(1),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=_SF_CL_EMB,
            p_interval=irx.icentpert(jnp.array([1.0]), jnp.zeros(1)),
            obs_offset=_OBS_OFFSET_SENSOR,
            obs_scale=_OBS_SCALE_SENSOR,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Interval propagation helpers
# ══════════════════════════════════════════════════════════════════════════════

def cl_euler_step(
    emb_sys,
    x_ivl: irx.Interval,
    theta: jnp.ndarray,
    p_ivl: irx.Interval,
    dt: float,
) -> irx.Interval:
    """One forward-Euler interval step under the CL natural embedding.

    theta plays the role of 'u' in the CL system's f(t, x, u, p).
    t must be a JAX array (not a Python scalar) to avoid jaxpr arg-count
    mismatch in eqx.filter_make_jaxpr.
    """
    _t    = jnp.zeros(())
    x_ut  = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, theta, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


def cl_euler_multistep(
    emb_sys,
    x_ivl: irx.Interval,
    theta: jnp.ndarray,
    p_ivl: irx.Interval,
    dt: float,
    num_substeps: int,
) -> irx.Interval:
    """Apply num_substeps forward-Euler steps of size dt/num_substeps.

    Each sub-step holds theta constant (zero-order hold within the control
    interval).  When num_substeps == 1 this is identical to cl_euler_step.
    Uses jax.lax.fori_loop so XLA sees a fixed-size graph regardless of
    num_substeps.
    """
    sub_dt = dt / num_substeps
    xlen   = x_ivl.lower.shape[0]

    def sub_step(_, x_arr: jnp.ndarray) -> jnp.ndarray:
        x     = irx.Interval(lower=x_arr[:xlen], upper=x_arr[xlen:])
        x_nxt = cl_euler_step(emb_sys, x, theta, p_ivl, sub_dt)
        return jnp.concatenate([x_nxt.lower, x_nxt.upper])

    x_arr0   = jnp.concatenate([x_ivl.lower, x_ivl.upper])
    x_arr_f  = jax.lax.fori_loop(0, num_substeps, sub_step, x_arr0)
    return irx.Interval(lower=x_arr_f[:xlen], upper=x_arr_f[xlen:])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CBF helpers
# ══════════════════════════════════════════════════════════════════════════════

def cbf_min_over_interval(
    x_ivl: irx.Interval,
    obstacles: jnp.ndarray,  # (N, 3) — each row [cx, cy, r_obs]
) -> jnp.ndarray:
    """Minimum of h(x) = (px-cx)²+(py-cy)²-r² over the position interval.

    Since h is convex in (px, py), the minimum over a box is achieved at the
    closest point to (cx, cy):
        px* = clip(cx, pxl, pxu),  py* = clip(cy, pyl, pyu)

    Returns
    -------
    (N,) array — one h_min value per obstacle (negative means violation).
    """
    def _single(obs):
        cx, cy, r_obs = obs[0], obs[1], obs[2]
        px_star = jnp.clip(cx, x_ivl.lower[0], x_ivl.upper[0])
        py_star = jnp.clip(cy, x_ivl.lower[1], x_ivl.upper[1])
        return (px_star - cx) ** 2 + (py_star - cy) ** 2 - r_obs ** 2

    return jax.vmap(_single)(obstacles)   # (N,)


def cbf_penalty_interval(
    x_ivl: irx.Interval,
    obstacles: jnp.ndarray,  # (N, 3)
) -> jnp.ndarray:
    """Sum of CBF penalties over all obstacles for one state interval.

    penalty_i = max(0, -h_min_i)   (positive ↔ interval penetrates obstacle i)
    """
    h_mins = cbf_min_over_interval(x_ivl, obstacles)
    return jnp.sum(jnp.maximum(0.0, -h_mins))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Combined separation + CBF loss with intersection refinement
# ══════════════════════════════════════════════════════════════════════════════

def separation_cbf_loss_refined(
    theta: jnp.ndarray,
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,  # (N, 3) — each row [cx, cy, r_obs]
    cbf_weight: float = 1.0,
) -> jnp.ndarray:
    """Combined separation + CBF loss with observation-based state refinement.

    Algorithm
    ---------
    Step 1
        Propagate each scenario one Euler step with the CL controller theta.
        cost[1] = Σ_{i<j} overlap(obs_i, obs_j)      (output-space overlaps)
        cbf[1]  = Σ_i Σ_obs cbf_penalty(x_i)

    Steps 2 … num_steps  (intersection refinement)
        For each pair (i, j):
          1. Intersect observed outputs:  y = obs_i ∩ obs_j.
          2. Invert the obs model to get refined state intervals:
                 x_ref_i[0:2] = (y − offset_i) / scale_i
                 x_ref_i[2]   = phi unchanged from propagated interval
          3. Propagate x_ref one Euler step with theta → x_next_{i,j}.
          4. cost[k] += overlap(obs(x_next_i), obs(x_next_j))
             (masked to 0 if pair is already separated)
        CBF penalty is computed on the *unrefined* per-scenario intervals
        (conservative; ensures safety over the full reachable set).

    Returns
    -------
    min_k cost[k]  +  cbf_weight · Σ_k cbf[k]

    The minimum-over-steps separation loss is zero when the scenarios are
    fully separated at least once during the horizon.  The CBF term enforces
    safety at every step.

    Parameters
    ----------
    theta       : (6,) controller params [K.flatten(), r]
    x0_ivl      : initial state interval  [px, py, phi]
    cl_scenarios: list from create_cl_scenarios()
    dt          : Euler step size (s)
    num_steps   : total propagation steps (≥ 1)
    obstacles   : (N, 3) array — each row [cx, cy, r_obs]
    cbf_weight  : weight on the CBF penalty term

    Implementation note
    -------------------
    Steps 2…num_steps are executed via jax.lax.fori_loop so that the XLA
    computation graph stays O(1) in num_steps regardless of horizon length.
    A Python loop would unroll all num_steps copies into the graph at trace
    time, causing O(num_steps) compilation memory and graph size.
    """
    n     = len(cl_scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs)
    xlen = x0_ivl.lower.shape[0]   # 3

    # ── helpers: flatten/unflatten an Interval to a (2*xlen,) array ──────
    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

    # ── Step 1 ────────────────────────────────────────────────────────────
    x_ivls = [
        cl_euler_step(s.emb_system, x0_ivl, theta, s.p_interval, dt)
        for s in cl_scenarios
    ]
    obs_ivls = [_obs_interval(x, s) for x, s in zip(x_ivls, cl_scenarios)]

    sep_cost1 = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            sep_cost1 = sep_cost1 + overlap_size_lax(obs_ivls[i], obs_ivls[j])
    min_sep_cost = sep_cost1

    cbf_pen = jnp.array(0.0)
    for x in x_ivls:
        cbf_pen = cbf_pen + cbf_penalty_interval(x, obstacles)

    # Stack into arrays for the fori_loop carry
    #   x_arr    : (n,       2*xlen) — unrefined per-scenario states
    #   pxi_arr  : (n_pairs, 2*xlen) — xi state for each pair
    #   pxj_arr  : (n_pairs, 2*xlen) — xj state for each pair
    x_arr   = jnp.stack([ivl_to_arr(x)        for x      in x_ivls])
    pxi_arr = jnp.stack([ivl_to_arr(x_ivls[i]) for i, j  in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x_ivls[j]) for i, j  in pairs])

    # ── Steps 2 … num_steps via fori_loop ────────────────────────────────
    # The loop body is traced ONCE by JAX regardless of num_steps, keeping
    # the XLA graph size constant (O(1) in num_steps).
    # The inner Python loops over n=3 scenarios and n_pairs=3 pairs are
    # fine to leave unrolled because they don't grow with num_steps.
    def step_body(_, carry):
        x_arr, pxi_arr, pxj_arr, cbf_pen, min_sep_cost = carry

        # Propagate unrefined scenarios for CBF
        x_next_list = [
            cl_euler_step(
                cl_scenarios[si].emb_system,
                arr_to_ivl(x_arr[si]),
                theta,
                cl_scenarios[si].p_interval,
                dt,
            )
            for si in range(n)
        ]
        x_next_arr = jnp.stack([ivl_to_arr(x) for x in x_next_list])
        for x in x_next_list:
            cbf_pen = cbf_pen + cbf_penalty_interval(x, obstacles)

        # Intersection refinement per pair
        step_sep_cost = jnp.array(0.0)
        new_pxi_list  = []
        new_pxj_list  = []

        for idx, (i, j) in enumerate(pairs):
            xi = arr_to_ivl(pxi_arr[idx])
            xj = arr_to_ivl(pxj_arr[idx])
            obs_i = _obs_interval(xi, cl_scenarios[i])
            obs_j = _obs_interval(xj, cl_scenarios[j])

            y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
            y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
            has_overlap = jnp.all(y_hi >= y_lo)

            fallback  = (xi.lower[:2] + xi.upper[:2]) / 2
            y_lo_safe = jnp.where(has_overlap, y_lo, fallback)
            y_hi_safe = jnp.where(has_overlap, y_hi, fallback)

            si_s = cl_scenarios[i].obs_scale[0]
            sj_s = cl_scenarios[j].obs_scale[0]

            xi_ref = irx.Interval(
                lower=jnp.array([
                    (y_lo_safe[0] - cl_scenarios[i].obs_offset[0]) / si_s,
                    (y_lo_safe[1] - cl_scenarios[i].obs_offset[1]) / si_s,
                    xi.lower[2],
                ]),
                upper=jnp.array([
                    (y_hi_safe[0] - cl_scenarios[i].obs_offset[0]) / si_s,
                    (y_hi_safe[1] - cl_scenarios[i].obs_offset[1]) / si_s,
                    xi.upper[2],
                ]),
            )
            xj_ref = irx.Interval(
                lower=jnp.array([
                    (y_lo_safe[0] - cl_scenarios[j].obs_offset[0]) / sj_s,
                    (y_lo_safe[1] - cl_scenarios[j].obs_offset[1]) / sj_s,
                    xj.lower[2],
                ]),
                upper=jnp.array([
                    (y_hi_safe[0] - cl_scenarios[j].obs_offset[0]) / sj_s,
                    (y_hi_safe[1] - cl_scenarios[j].obs_offset[1]) / sj_s,
                    xj.upper[2],
                ]),
            )

            xn_i = cl_euler_step(
                cl_scenarios[i].emb_system, xi_ref, theta, cl_scenarios[i].p_interval, dt
            )
            xn_j = cl_euler_step(
                cl_scenarios[j].emb_system, xj_ref, theta, cl_scenarios[j].p_interval, dt
            )

            raw_cost = overlap_size_lax(
                _obs_interval(xn_i, cl_scenarios[i]),
                _obs_interval(xn_j, cl_scenarios[j]),
            )
            step_sep_cost = step_sep_cost + jnp.where(has_overlap, raw_cost, 0.0)
            new_pxi_list.append(ivl_to_arr(xn_i))
            new_pxj_list.append(ivl_to_arr(xn_j))

        min_sep_cost = jnp.minimum(min_sep_cost, step_sep_cost)
        new_pxi_arr  = jnp.stack(new_pxi_list)
        new_pxj_arr  = jnp.stack(new_pxj_list)
        return (x_next_arr, new_pxi_arr, new_pxj_arr, cbf_pen, min_sep_cost)

    init_carry = (x_arr, pxi_arr, pxj_arr, cbf_pen, min_sep_cost)
    _, _, _, cbf_pen_final, min_sep_cost_final = jax.lax.fori_loop(
        0, num_steps - 1, step_body, init_carry
    )

    return min_sep_cost_final + cbf_weight * cbf_pen_final


# ══════════════════════════════════════════════════════════════════════════════
# 7.  GPU-parallel optimisation
# ══════════════════════════════════════════════════════════════════════════════

def optimize_output_feedback_cbf_gpu(
    cl_scenarios: List[Scenario],
    x0_ivl: irx.Interval,
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,          # (N, 3) — [cx, cy, r_obs] per row
    cbf_weight: float = 1.0,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 200,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent over theta ∈ R^6.

    Initialises num_restarts random controller parameters, runs gradient
    descent on all in parallel via jax.vmap + jax.lax.fori_loop, and returns
    the best result.

    Initialisation: K ≈ 0 (small Gaussian), r ≈ [0.5, 0.0].

    Returns
    -------
    (best_theta, best_loss, all_theta_final, all_losses)
    """
    key          = jax.random.PRNGKey(seed)
    theta_mean   = jnp.concatenate([jnp.zeros(4), jnp.array([0.5, 0.0])])
    theta0       = jax.random.normal(key, (num_restarts, 6)) * 0.1 + theta_mean

    def loss_fn(theta):
        return separation_cbf_loss_refined(
            theta=theta,
            x0_ivl=x0_ivl,
            cl_scenarios=cl_scenarios,
            dt=dt,
            num_steps=num_steps,
            obstacles=obstacles,
            cbf_weight=cbf_weight,
        )

    batched_loss = jax.vmap(loss_fn)            # (R, 6) → (R,)
    batched_grad = jax.vmap(jax.grad(loss_fn))  # (R, 6) → (R, 6)

    def body(_, theta_batch):
        g = batched_grad(theta_batch)
        return _project_theta(theta_batch - learning_rate * g)

    theta_final = jax.lax.fori_loop(0, num_iters, body, theta0)
    losses      = batched_loss(theta_final)

    best_idx   = jnp.argmin(losses)
    best_theta = theta_final[best_idx]
    best_loss  = losses[best_idx]
    return best_theta, best_loss, theta_final, losses


def optimize_output_feedback_cbf(
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,         # (N, 3) — [cx, cy, r_obs] per row
    cbf_weight: float = 1.0,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 200,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Solve for output-feedback controller maximising fault separation + CBF.

    Returns
    -------
    (K, r, loss)
    K    : (2, 2) optimal output-feedback gain matrix
    r    : (2,)   optimal feedforward term
    loss : final combined loss value
    """
    best_theta, best_loss, _, _ = optimize_output_feedback_cbf_gpu(
        cl_scenarios=cl_scenarios,
        x0_ivl=x0_ivl,
        dt=dt,
        num_steps=num_steps,
        obstacles=obstacles,
        cbf_weight=cbf_weight,
        num_restarts=num_restarts,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=seed,
    )
    K, r = theta_to_K_r(best_theta)
    return K, r, best_loss


# ══════════════════════════════════════════════════════════════════════════════
# 7b.  Tracking output-feedback controller
# ══════════════════════════════════════════════════════════════════════════════

def tracking_cbf_loss(
    theta_seq: jnp.ndarray,     # (num_steps, 6) — per-step [K.flat, r]
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    obstacles: jnp.ndarray,     # (N, 3) — [cx, cy, r_obs]
    cbf_weight: float = 1.0,
    num_substeps: int = 1,
) -> jnp.ndarray:
    """Combined separation + CBF loss for a tracking output-feedback controller.

    Control law at step k
    ---------------------
        u_k = clip(K_k @ y_k + r_k,  u_lo, u_hi)

    where theta_seq[k] = [K_k.flat, r_k] contains a per-step gain K_k ∈ R^{2×2}
    and per-step feedforward r_k ∈ R^2.  The reference trajectory enters only
    through the initialisation in the outer optimiser, not through this loss.

    Decision variable
    -----------------
    theta_seq ∈ R^{num_steps × 6}

    Parameters
    ----------
    num_substeps : int
        Number of forward-Euler sub-steps per control interval.  Each control
        input theta_k is held constant for num_substeps steps of size
        dt/num_substeps.  Default 1 (original behaviour).
    """
    n         = len(cl_scenarios)
    pairs     = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_steps = theta_seq.shape[0]
    xlen      = x0_ivl.lower.shape[0]

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

    def make_theta(k: int) -> jnp.ndarray:
        """Return the controller parameters for step k."""
        return theta_seq[k]

    def prop(emb_sys, x_ivl, theta, p_ivl):
        """Propagate one control interval (num_substeps Euler sub-steps)."""
        return cl_euler_multistep(emb_sys, x_ivl, theta, p_ivl, dt, num_substeps)

    # ── Step 1: propagate with theta_0 ────────────────────────────────────
    theta_0 = make_theta(0)
    x_ivls  = [
        prop(s.emb_system, x0_ivl, theta_0, s.p_interval)
        for s in cl_scenarios
    ]
    obs_ivls = [_obs_interval(x, s) for x, s in zip(x_ivls, cl_scenarios)]

    sep_cost0 = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            sep_cost0 = sep_cost0 + overlap_size_lax(obs_ivls[i], obs_ivls[j])
    min_sep_cost = sep_cost0

    cbf_pen = jnp.array(0.0)
    for x in x_ivls:
        cbf_pen = cbf_pen + cbf_penalty_interval(x, obstacles)

    x_arr   = jnp.stack([ivl_to_arr(x) for x in x_ivls])
    pxi_arr = jnp.stack([ivl_to_arr(x_ivls[i]) for i, j in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x_ivls[j]) for i, j in pairs])

    # ── Steps 2..num_steps via fori_loop with time-varying theta ──────────
    def step_body(k, carry):
        x_arr, pxi_arr, pxj_arr, cbf_pen, min_sep_cost = carry

        # theta changes every step: r_eff_k = u_ol[k+1] − K @ ŷ[k+1]
        theta_k = make_theta(k + 1)

        # Unrefined propagation (for CBF)
        x_next_list = [
            prop(cl_scenarios[si].emb_system, arr_to_ivl(x_arr[si]),
                 theta_k, cl_scenarios[si].p_interval)
            for si in range(n)
        ]
        x_next_arr = jnp.stack([ivl_to_arr(x) for x in x_next_list])
        for x in x_next_list:
            cbf_pen = cbf_pen + cbf_penalty_interval(x, obstacles)

        # Intersection refinement per pair
        step_sep_cost = jnp.array(0.0)
        new_pxi_list  = []
        new_pxj_list  = []

        for idx, (i, j) in enumerate(pairs):
            xi = arr_to_ivl(pxi_arr[idx])
            xj = arr_to_ivl(pxj_arr[idx])
            obs_i = _obs_interval(xi, cl_scenarios[i])
            obs_j = _obs_interval(xj, cl_scenarios[j])

            y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
            y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
            has_overlap = jnp.all(y_hi >= y_lo)

            fallback  = (xi.lower[:2] + xi.upper[:2]) / 2
            y_lo_safe = jnp.where(has_overlap, y_lo, fallback)
            y_hi_safe = jnp.where(has_overlap, y_hi, fallback)

            si_s = cl_scenarios[i].obs_scale[0]
            sj_s = cl_scenarios[j].obs_scale[0]

            xi_ref = irx.Interval(
                lower=jnp.array([
                    (y_lo_safe[0] - cl_scenarios[i].obs_offset[0]) / si_s,
                    (y_lo_safe[1] - cl_scenarios[i].obs_offset[1]) / si_s,
                    xi.lower[2],
                ]),
                upper=jnp.array([
                    (y_hi_safe[0] - cl_scenarios[i].obs_offset[0]) / si_s,
                    (y_hi_safe[1] - cl_scenarios[i].obs_offset[1]) / si_s,
                    xi.upper[2],
                ]),
            )
            xj_ref = irx.Interval(
                lower=jnp.array([
                    (y_lo_safe[0] - cl_scenarios[j].obs_offset[0]) / sj_s,
                    (y_lo_safe[1] - cl_scenarios[j].obs_offset[1]) / sj_s,
                    xj.lower[2],
                ]),
                upper=jnp.array([
                    (y_hi_safe[0] - cl_scenarios[j].obs_offset[0]) / sj_s,
                    (y_hi_safe[1] - cl_scenarios[j].obs_offset[1]) / sj_s,
                    xj.upper[2],
                ]),
            )

            xn_i = prop(cl_scenarios[i].emb_system, xi_ref, theta_k, cl_scenarios[i].p_interval)
            xn_j = prop(cl_scenarios[j].emb_system, xj_ref, theta_k, cl_scenarios[j].p_interval)

            raw_cost = overlap_size_lax(
                _obs_interval(xn_i, cl_scenarios[i]),
                _obs_interval(xn_j, cl_scenarios[j]),
            )
            step_sep_cost = step_sep_cost + jnp.where(has_overlap, raw_cost, 0.0)
            new_pxi_list.append(ivl_to_arr(xn_i))
            new_pxj_list.append(ivl_to_arr(xn_j))

        min_sep_cost = jnp.minimum(min_sep_cost, step_sep_cost)
        return (x_next_arr, jnp.stack(new_pxi_list), jnp.stack(new_pxj_list), cbf_pen, min_sep_cost)

    init_carry = (x_arr, pxi_arr, pxj_arr, cbf_pen, min_sep_cost)
    _, _, _, cbf_pen_f, min_sep_f = jax.lax.fori_loop(
        0, num_steps - 1, step_body, init_carry
    )
    return min_sep_f + cbf_weight * cbf_pen_f


def optimize_tracking_cbf_gpu(
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    y_hat_seq: jnp.ndarray,     # (num_steps, 2) — reference positions (init only)
    u_ol_seq: jnp.ndarray,      # (num_steps, 2) — open-loop controls (init only)
    obstacles: jnp.ndarray,     # (N, 3)
    cbf_weight: float = 1.0,
    num_restarts: int = 100,
    learning_rate: float = 0.1,
    num_iters: int = 200,
    seed: int = 42,
    num_substeps: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent for a per-step tracking controller.

    The decision variable is theta_seq ∈ R^{num_steps × 6}, where each row
    theta_k = [K_k.flat, r_k] contains an independent gain K_k ∈ R^{2×2} and
    feedforward r_k ∈ R^2 for step k.  The control law is:

        u_k = clip(K_k @ y_k + r_k,  u_lo, u_hi)

    Initialisation: K_k ≈ 0,  r_k = u_ol_seq[k]  (tracks the reference
    open-loop trajectory with zero feedback correction at startup).

    Parameters
    ----------
    y_hat_seq : (num_steps, 2)
        Reference positions — used only to determine num_steps and for a
        warm-start initialisation of r_k.  Not used in the loss itself.
    u_ol_seq  : (num_steps, 2)
        Open-loop reference controls — used to initialise r_k = u_ol_k.
    num_substeps : int
        Number of forward-Euler sub-steps per control interval (default 1).

    Returns
    -------
    (best_theta_seq, best_loss, all_theta_seq_final, all_losses)
    best_theta_seq      : (num_steps, 6)  optimal per-step controller params
    best_loss           : scalar          combined loss at best_theta_seq
    all_theta_seq_final : (num_restarts, num_steps, 6)
    all_losses          : (num_restarts,)
    """
    num_steps = y_hat_seq.shape[0]
    key       = jax.random.PRNGKey(seed)

    # Initialise: K_k ≈ 0, r_k = u_ol_k  →  theta_mean[k] = [0,0,0,0, u_ol_k]
    theta_mean = jnp.concatenate(
        [jnp.zeros((num_steps, 4)), u_ol_seq], axis=-1
    )  # (num_steps, 6)

    noise   = jax.random.normal(key, (num_restarts, num_steps, 6)) * 0.1
    theta0  = noise + theta_mean[None]   # (num_restarts, num_steps, 6)

    def loss_fn(theta_seq):
        return tracking_cbf_loss(
            theta_seq,
            x0_ivl=x0_ivl,
            cl_scenarios=cl_scenarios,
            dt=dt,
            obstacles=obstacles,
            cbf_weight=cbf_weight,
            num_substeps=num_substeps,
        )

    batched_loss = jax.vmap(loss_fn)
    batched_grad = jax.vmap(jax.grad(loss_fn))

    def body(_, theta_batch):
        g = batched_grad(theta_batch)
        return jnp.clip(theta_batch - learning_rate * g, _THETA_LO, _THETA_HI)

    theta_final  = jax.lax.fori_loop(0, num_iters, body, theta0)
    losses       = batched_loss(theta_final)
    best_idx     = jnp.argmin(losses)
    best_theta   = theta_final[best_idx]   # (num_steps, 6)
    best_loss    = losses[best_idx]
    return best_theta, best_loss, theta_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_controller(
    theta: jnp.ndarray,
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,
) -> Dict:
    """Compute per-scenario position intervals, CBF values, and pairwise overlaps.

    Returns a dict with keys:
        'K', 'r'                   : unpacked controller
        'position_intervals'       : list of 2-D position intervals per scenario
        'cbf_min_values'           : (S, N) array — min h per scenario per obstacle
        'pairwise_overlaps'        : dict of (scenario_i vs scenario_j) → m²
    """
    K, r = theta_to_K_r(theta)

    # Propagate num_steps under the CL controller (unrefined)
    x_final = []
    for s in cl_scenarios:
        x = x0_ivl
        for _ in range(num_steps):
            x = cl_euler_step(s.emb_system, x, theta, s.p_interval, dt)
        x_final.append(x)

    pos_ivls = [
        irx.Interval(lower=x.lower[:2], upper=x.upper[:2])
        for x in x_final
    ]

    # CBF values at final step
    cbf_mins = np.array([
        [float(v) for v in cbf_min_over_interval(x, obstacles)]
        for x in x_final
    ])   # shape (S, N)

    n = len(cl_scenarios)
    obs_final = [_obs_interval(x, s) for x, s in zip(x_final, cl_scenarios)]
    pairwise = {}
    for i in range(n):
        for j in range(i + 1, n):
            key_ij = f"{cl_scenarios[i].name} vs {cl_scenarios[j].name}"
            pairwise[key_ij] = float(overlap_size_lax(obs_final[i], obs_final[j]))

    return {
        'K':                   np.array(K),
        'r':                   np.array(r),
        'position_intervals':  pos_ivls,
        'cbf_min_values':      cbf_mins,
        'pairwise_overlaps':   pairwise,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("FAULTY CAR OUTPUT-FEEDBACK CONTROLLER + COLLISION-AVOIDANCE CBF")
    print("=" * 70)

    cl_scenarios = create_cl_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
    print(f"\n{len(cl_scenarios)} closed-loop fault scenarios:")
    for s in cl_scenarios:
        print(f"  • {s.name}")
        print(f"    p ∈ [{np.array(s.p_interval.lower)}, {np.array(s.p_interval.upper)}]")
        print(f"    obs_offset = {np.array(s.obs_offset)},  obs_scale = {np.array(s.obs_scale)}")

    # Initial state interval
    x0_ivl = irx.icentpert(
        jnp.array([0.1, 0.1, 0.0]),
        jnp.array([0.1, 0.1, 0.1]),
    )
    print(f"\nInitial state interval:")
    print(f"  px ∈ [{float(x0_ivl.lower[0]):.3f}, {float(x0_ivl.upper[0]):.3f}] m")
    print(f"  py ∈ [{float(x0_ivl.lower[1]):.3f}, {float(x0_ivl.upper[1]):.3f}] m")
    print(f"  φ  ∈ [{float(x0_ivl.lower[2]):.3f}, {float(x0_ivl.upper[2]):.3f}] rad")

    # Obstacle: a 0.3-m-radius disk centred at (1.5, 0.8)
    obstacles = jnp.array([[1.5, 0.8, 0.3]])   # (1, 3)
    print(f"\nObstacle: centre ({float(obstacles[0,0]):.2f}, {float(obstacles[0,1]):.2f}) m,"
          f"  radius {float(obstacles[0,2]):.2f} m")

    dt, num_steps = 0.05, 20
    cbf_weight    = 2.0
    print(f"\nHorizon: {num_steps} × {dt} s = {num_steps * dt:.2f} s")
    print(f"CBF weight: {cbf_weight}")

    print("\nRunning output-feedback + CBF optimisation …\n")
    K, r, loss = optimize_output_feedback_cbf(
        x0_ivl=x0_ivl,
        cl_scenarios=cl_scenarios,
        dt=dt,
        num_steps=num_steps,
        obstacles=obstacles,
        cbf_weight=cbf_weight,
        num_restarts=100,
        learning_rate=0.01,
        num_iters=300,
        seed=42,
    )

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOptimal gain matrix K:")
    for row in K:
        print(f"  [{row[0]:+.4f}  {row[1]:+.4f}]")
    print(f"\nFeedforward r: [{r[0]:+.4f},  {r[1]:+.4f}]")
    print(f"\nCombined loss: {loss:.6f}")

    theta = K_r_to_theta(jnp.array(K), jnp.array(r))
    stats = evaluate_controller(theta, x0_ivl, cl_scenarios, dt, num_steps, obstacles)

    print(f"\nPairwise output overlaps (at final step, unrefined):")
    for k, v in stats['pairwise_overlaps'].items():
        print(f"  {k}: {v:.6f} m²")

    print(f"\nCBF min h(x) values at final step (negative = violation):")
    for i, s in enumerate(cl_scenarios):
        for j in range(obstacles.shape[0]):
            h = stats['cbf_min_values'][i, j]
            status = "SAFE" if h >= 0 else "VIOLATED"
            print(f"  {s.name} / obs {j}: h_min = {h:.4f} m²  [{status}]")

    print(f"\nFinal position intervals:")
    for s, iv in zip(cl_scenarios, stats['position_intervals']):
        wx = float(iv.upper[0] - iv.lower[0])
        wy = float(iv.upper[1] - iv.lower[1])
        print(f"  {s.name}:")
        print(f"    px ∈ [{float(iv.lower[0]):+.4f}, {float(iv.upper[0]):+.4f}] m  (width {wx:.4f})")
        print(f"    py ∈ [{float(iv.lower[1]):+.4f}, {float(iv.upper[1]):+.4f}] m  (width {wy:.4f})")

    print("\nDone.")
