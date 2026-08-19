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


def _obs_interval_raw(
    x_ivl: irx.Interval,
    obs_scale: jnp.ndarray,
    obs_offset: jnp.ndarray,
) -> irx.Interval:
    """Observed output interval with explicit scale/offset (vmappable)."""
    return irx.Interval(
        lower=obs_scale * x_ivl.lower[:2] + obs_offset,
        upper=obs_scale * x_ivl.upper[:2] + obs_offset,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Controller parameterisation
# ══════════════════════════════════════════════════════════════════════════════

_K_MAX = 2.0
# History this session, all at 5s horizon (dt=0.5, num_steps=10):
#   5.0  @ 100 restarts/6 iters : loss=0.0 but box blew up 20x (0.3m->6.1m) --
#        refined-loss=0 was real (see below) but the RAW boxes overlapped
#        1.5-2.0 m^2, i.e. genuinely too wide to be a useful/checkable
#        certificate in practice, and the CBF margin nearly hit 0.
#   1.0  @ 100/6  : box 4x growth, loss=0.0011, K saturated at bound on 2/4 entries.
#   1.5  @ 100/6  : box 1.5x growth, loss=0.00081, all 3 pairs raw-separated
#        to ~0.0008 m^2, K[0,0] saturated at -1.5.
#   2.0  @ 300/10 : box 1.28x growth (flattest yet), loss=0.00013, all 3 pairs
#        raw-separated to a few 1e-3 m^2, K[1,0] saturated at -2.0, but
#        runtime rose to ~76ms (300 restarts/10 iters vs the ~40ms budget's
#        100/6).
#   5.0  @ 500/15 (retried with much more search): loss=0.0 again, but box
#        STILL blew up 16.5x (0.3m->4.93m) and raw overlaps were 0.28-1.70
#        m^2 -- confirms this is a STRUCTURAL problem with allowing K this
#        large under the coarse (no-substep) Euler integration, not a
#        search-budget problem; more restarts/iters does not fix it.
#   3.0  @ 1000/15 : SAME blow-up signature as 5.0 (box 0.3m->2.98m, 10x;
#        raw overlap up to 6.65 m^2; r saturated at both bounds [-1,1] both
#        times K_MAX>=3.0 was tried). Confirms 2.0 is the genuine/blow-up
#        threshold, not noise -- settling on K_MAX=2.0, now pushing restarts
#        higher to try to close the last ~0.00013 gap to exact 0.
# back in. Found (this session) that with init_std=1.0 and a large learning
# rate, GD can drive K entries up to ~2.4 in magnitude -- combined with the
# coarse single-Euler-step-per-segment integration (no sub-stepping) used by
# separation_cbf_loss_refined_vmapped, this compounds into severe wrapping-
# effect box growth (observed: 0.3m -> 6.1m over a 10-step/5s horizon).
# Tightening the bound keeps K in a range where the natural embedding's
# per-step amplification stays modest enough not to blow up the reachable
# boxes into physical meaninglessness, while (empirically, see this
# session's re-test) still being wide enough for GD to find genuine
# (refined-loss=0) diagnosability solutions.
# theta = [K00, K01, K10, K11,  r0, r1]   (6-D)
_THETA_LO = jnp.concatenate([jnp.full(4, -_K_MAX), _U_LO])
_THETA_HI = jnp.concatenate([jnp.full(4,  _K_MAX), _U_HI])

_OBS_OFFSET_SENSOR = jnp.array([0.2, 0.2])
_OBS_SCALE_SENSOR  = jnp.array([0.95])

# ── Tracking controller actuation limits ──────────────────────────────────────
# v ∈ [-.15, .15] m/s,  ω ∈ [1.5, 1.5] rad/s
_CL_U_LO_TRACK = jnp.array([-0.1, -1.])
_CL_U_HI_TRACK = jnp.array([ 0.1,  1.])
# theta = [K.flat (4-D), u_ff (2-D)] — both K and feedforward are optimised
_THETA_LO_TRACK = jnp.concatenate([jnp.full(4, -_K_MAX), _CL_U_LO_TRACK])
_THETA_HI_TRACK = jnp.concatenate([jnp.full(4,  _K_MAX), _CL_U_HI_TRACK])


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


class CarUnifiedCLSystem(irx.System):
    """Unified CL system for all fault scenarios (theta=[K.flat,r], 6-D 'u').

    Mirrors CarUnifiedTrackCLSystem (below) but for the plain (non-tracking)
    output-feedback controller used by separation_cbf_loss_refined -- encodes
    the observation model (scale, offset) in p so ALL scenarios share ONE
    embedding, enabling a single jax.vmap over scenarios/pairs instead of a
    Python for-loop that separately traces+differentiates one call site per
    scenario/pair. See separation_cbf_loss_refined_vmapped's docstring for
    why that matters: an un-vmapped Python loop over N structurally-identical
    calls fed DIFFERENT runtime p_interval bounds cannot be merged by XLA's
    CSE (CSE only merges calls with identical inputs), so it costs roughly N
    times the compile work of one vmapped call.

    p = [alpha, obs_scale, obs_off_x, obs_off_y]  (4-D)
      Nominal        : p = [1.0,       1.0,  0.0, 0.0]
      Actuator Fault : p in [alpha_lo, 1.0,  0.0, 0.0] x [alpha_hi, 1.0, 0.0, 0.0]
      Sensor Fault   : p = [1.0,       0.95, 0.2, 0.2]
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        K = u[:4].reshape(2, 2)
        r_ff = u[4:6]
        alpha = p[0]
        obs_scale = p[1]
        obs_off = jnp.array([p[2], p[3]])
        y = obs_scale * x[0:2] + obs_off
        ctrl = jnp.clip(K @ y + r_ff, _U_LO, _U_HI)
        v, omega = ctrl[0], ctrl[1]
        phi = x[2]
        return jnp.array([v * jnp.cos(phi), v * jnp.sin(phi), alpha * omega])


_UNIFIED_CL_SYS = CarUnifiedCLSystem()
_UNIFIED_CL_EMB = irx.natemb(_UNIFIED_CL_SYS)


def create_unified_cl_scenarios(
    actuator_alpha_lo: float = 0.0,
    actuator_alpha_hi: float = 0.5,
) -> List[Scenario]:
    """Three closed-loop fault scenarios sharing ONE embedding (_UNIFIED_CL_EMB)
    -- see CarUnifiedCLSystem's docstring. Used by
    separation_cbf_loss_refined_vmapped / optimize_output_feedback_cbf_vmapped_gpu."""
    return [
        Scenario(
            name="Nominal",
            emb_system=_UNIFIED_CL_EMB,
            p_interval=irx.icentpert(jnp.array([1.0, 1.0, 0.0, 0.0]), jnp.zeros(4)),
            obs_offset=jnp.zeros(2),
            obs_scale=jnp.ones(1),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=_UNIFIED_CL_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo, 1.0, 0.0, 0.0]),
                upper=jnp.array([actuator_alpha_hi, 1.0, 0.0, 0.0]),
            ),
            obs_offset=jnp.zeros(2),
            obs_scale=jnp.ones(1),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=_UNIFIED_CL_EMB,
            p_interval=irx.icentpert(
                jnp.array([1.0, _OBS_SCALE_SENSOR[0],
                            _OBS_OFFSET_SENSOR[0], _OBS_OFFSET_SENSOR[1]]),
                jnp.zeros(4),
            ),
            obs_offset=_OBS_OFFSET_SENSOR,
            obs_scale=_OBS_SCALE_SENSOR,
        ),
    ]


# ── Tracking CL systems  (u = [K.flat, y_hat, u_ol], 8-D) ────────────────────

class CarNomActTrackCLSystem(irx.System):
    """Nominal / actuator-fault car under error-feedback tracking control.

    immrax convention:
        u = [K.flat, y_hat, u_ff]  ∈ R^8
            K     : (2,2) feedback gain       (optimisation variable)
            y_hat : (2,)  reference position  (fixed per step, from traj planner)
            u_ff  : (2,)  learned feedforward (optimisation variable)
        p = [alpha]  ∈ R^1

    Control law:
        u_k = clip(K @ (y − ŷ) + u_ff,  _CL_U_LO_TRACK, _CL_U_HI_TRACK)
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        K     = u[:4].reshape(2, 2)
        y_hat = u[4:6]
        u_ff  = u[6:8]
        y     = x[0:2]
        ctrl  = jnp.clip(K @ (y - y_hat) + u_ff, _CL_U_LO_TRACK, _CL_U_HI_TRACK)
        v, omega = ctrl[0], ctrl[1]
        alpha = p[0]
        phi   = x[2]
        return jnp.array([v * jnp.cos(phi), v * jnp.sin(phi), alpha * omega])


class CarSensorFaultTrackCLSystem(irx.System):
    """Sensor-fault car under error-feedback tracking control.

    Observation is biased:  y = scale · [px, py] + offset.
    The robot uses its (corrupted) observation to compute the tracking error.
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        K     = u[:4].reshape(2, 2)
        y_hat = u[4:6]
        u_ff  = u[6:8]
        y     = _OBS_SCALE_SENSOR[0] * x[0:2] + _OBS_OFFSET_SENSOR
        ctrl  = jnp.clip(K @ (y - y_hat) + u_ff, _CL_U_LO_TRACK, _CL_U_HI_TRACK)
        v, omega = ctrl[0], ctrl[1]
        alpha = p[0]
        phi   = x[2]
        return jnp.array([v * jnp.cos(phi), v * jnp.sin(phi), alpha * omega])


_NOM_ACT_TRACK_CL_SYS = CarNomActTrackCLSystem()
_NOM_ACT_TRACK_CL_EMB = irx.natemb(_NOM_ACT_TRACK_CL_SYS)

_SF_TRACK_CL_SYS = CarSensorFaultTrackCLSystem()
_SF_TRACK_CL_EMB = irx.natemb(_SF_TRACK_CL_SYS)


class CarUnifiedTrackCLSystem(irx.System):
    """Unified tracking CL system for all fault scenarios.

    Encodes the observation model in p so that one embedding covers all three
    fault scenarios (nominal, actuator fault, sensor fault).

    p = [alpha, obs_scale, obs_off_x, obs_off_y]  (4-D)
      Nominal        : p = [1.0,      1.0,  0.0, 0.0]
      Actuator Fault : p ∈ [alpha_lo, 1.0,  0.0, 0.0] × [alpha_hi, 1.0, 0.0, 0.0]
      Sensor Fault   : p = [1.0,      0.95, 0.2, 0.2]

    u = [K.flat (4), y_hat (2), u_ff (2)]  (8-D) — same layout as before.

    Control law:
        y    = obs_scale * [px, py] + [obs_off_x, obs_off_y]
        u_k  = clip(K @ (y − ŷ) + u_ff,  _CL_U_LO_TRACK, _CL_U_HI_TRACK)
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        K     = u[:4].reshape(2, 2)
        y_hat = u[4:6]
        u_ff  = u[6:8]
        alpha     = p[0]
        obs_scale = p[1]
        obs_off   = jnp.array([p[2], p[3]])
        y    = obs_scale * x[0:2] + obs_off
        ctrl = jnp.clip(K @ (y - y_hat) + u_ff, _CL_U_LO_TRACK, _CL_U_HI_TRACK)
        v, omega = ctrl[0], ctrl[1]
        phi = x[2]
        return jnp.array([v * jnp.cos(phi), v * jnp.sin(phi), alpha * omega])


# Single embedding shared by all three tracking scenarios
_UNIFIED_TRACK_CL_SYS = CarUnifiedTrackCLSystem()
_UNIFIED_TRACK_CL_EMB = irx.natemb(_UNIFIED_TRACK_CL_SYS)


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


def create_track_cl_scenarios(
    actuator_alpha_lo: float = 0.0,
    actuator_alpha_hi: float = 0.5,
) -> List[Scenario]:
    """Three closed-loop fault scenarios for the *tracking* optimiser.

    All three scenarios share _UNIFIED_TRACK_CL_EMB; the observation model
    (scale, offset) is encoded in the 4-D p interval:
        p = [alpha, obs_scale, obs_off_x, obs_off_y]

    This lets tracking_cbf_loss use a single vmap over scenarios instead of
    three separate embedding-system traces, drastically reducing GPU memory.
    """
    return [
        Scenario(
            name="Nominal",
            emb_system=_UNIFIED_TRACK_CL_EMB,
            p_interval=irx.icentpert(
                jnp.array([1.0, 1.0, 0.0, 0.0]), jnp.zeros(4)
            ),
            obs_offset=jnp.zeros(2),
            obs_scale=jnp.ones(1),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=_UNIFIED_TRACK_CL_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo, 1.0, 0.0, 0.0]),
                upper=jnp.array([actuator_alpha_hi, 1.0, 0.0, 0.0]),
            ),
            obs_offset=jnp.zeros(2),
            obs_scale=jnp.ones(1),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=_UNIFIED_TRACK_CL_EMB,
            p_interval=irx.icentpert(
                jnp.array([1.0, _OBS_SCALE_SENSOR[0],
                            _OBS_OFFSET_SENSOR[0], _OBS_OFFSET_SENSOR[1]]),
                jnp.zeros(4),
            ),
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

    Uses lax.scan for the sub-steps (not fori_loop) so the backward pass is
    a clean scan that XLA can vectorise efficiently when vmap is applied.
    The sub-step count is a compile-time constant, but kept as a scan rather
    than a Python loop to avoid OOM when num_substeps is large (e.g. 10+).
    """
    sub_dt = dt / num_substeps
    xlen   = x_ivl.lower.shape[0]

    def sub_step(x_arr: jnp.ndarray, _) -> tuple:
        x     = irx.Interval(lower=x_arr[:xlen], upper=x_arr[xlen:])
        x_nxt = cl_euler_step(emb_sys, x, theta, p_ivl, sub_dt)
        return jnp.concatenate([x_nxt.lower, x_nxt.upper]), None

    x_arr0     = jnp.concatenate([x_ivl.lower, x_ivl.upper])
    x_arr_f, _ = jax.lax.scan(sub_step, x_arr0, None, length=num_substeps)
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
# 6b.  Vectorized (vmapped) refined loss -- runtime optimization
# ══════════════════════════════════════════════════════════════════════════════

def separation_cbf_loss_refined_vmapped(
    theta: jnp.ndarray,
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,  # (N, 3) — each row [cx, cy, r_obs]
    cbf_weight: float = 1.0,
) -> jnp.ndarray:
    """Numerically equivalent to separation_cbf_loss_refined, but replaces
    every Python for-loop over scenarios/pairs (which separately traces and
    reverse-mode-differentiates one call to cl_euler_step per scenario/pair
    -- N un-vmapped calls cost roughly N times the compile work of one
    vmapped call, since XLA's CSE cannot merge calls fed different runtime
    p_interval bounds) with a SINGLE jax.vmap'd cl_euler_step call per
    propagation site:
      - Step 1 and each fori_loop iteration's "unrefined" propagation:
        one vmap over the n scenarios (requires cl_scenarios to share ONE
        embedding -- see create_unified_cl_scenarios / CarUnifiedCLSystem).
      - Each iteration's refined pair propagation: one vmap over the
        concatenated (2*n_pairs,) batch of both pair sides (xi_ref and
        xj_ref stacked), instead of 2*n_pairs separate calls.
    Pairwise overlap (overlap_size_lax, which internally uses lax.cond) is
    likewise vmapped over the n_pairs axis rather than Python-looped -- JAX
    auto-batches lax.cond under vmap.

    Requires cl_scenarios[*].emb_system to all be the SAME object (checked
    below) -- pass create_unified_cl_scenarios(...)'s output, not
    create_cl_scenarios(...)'s.
    """
    n = len(cl_scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs)
    xlen = x0_ivl.lower.shape[0]
    emb_sys = cl_scenarios[0].emb_system
    if any(s.emb_system is not emb_sys for s in cl_scenarios):
        raise ValueError(
            "separation_cbf_loss_refined_vmapped requires all cl_scenarios to "
            "share one embedding -- use create_unified_cl_scenarios(...), not "
            "create_cl_scenarios(...)."
        )

    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in cl_scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in cl_scenarios]),
    )
    obs_offset_batch = jnp.stack([s.obs_offset for s in cl_scenarios])  # (n, 2)
    obs_scale_batch  = jnp.stack([s.obs_scale for s in cl_scenarios])   # (n, 1)
    i_idx = jnp.array([i for i, j in pairs])
    j_idx = jnp.array([j for i, j in pairs])

    def step_all(x_ivl_single, p_ivl_single):
        return cl_euler_step(emb_sys, x_ivl_single, theta, p_ivl_single, dt)

    def obs_of(x_lower, x_upper, offset, scale):
        return offset + scale * x_lower[:2], offset + scale * x_upper[:2]

    def cbf_of(x_lower, x_upper):
        return cbf_penalty_interval(irx.Interval(lower=x_lower, upper=x_upper), obstacles)

    def arr_to_ivl_batch(arr):
        return irx.Interval(lower=arr[:, :xlen], upper=arr[:, xlen:])

    # ── Step 1: propagate all n scenarios in ONE vmapped call ─────────────
    x1_batch = jax.vmap(lambda p: step_all(x0_ivl, p))(p_batch)
    obs1_lo, obs1_hi = jax.vmap(obs_of)(x1_batch.lower, x1_batch.upper,
                                        obs_offset_batch, obs_scale_batch)
    obs1_i = irx.Interval(lower=obs1_lo[i_idx], upper=obs1_hi[i_idx])
    obs1_j = irx.Interval(lower=obs1_lo[j_idx], upper=obs1_hi[j_idx])
    sep_cost1 = jnp.sum(jax.vmap(overlap_size_lax)(obs1_i, obs1_j))
    min_sep_cost = sep_cost1

    cbf_pen = jnp.sum(jax.vmap(cbf_of)(x1_batch.lower, x1_batch.upper))

    x_arr   = jnp.concatenate([x1_batch.lower, x1_batch.upper], axis=-1)   # (n, 2*xlen)
    pxi_arr = x_arr[i_idx]   # (n_pairs, 2*xlen)
    pxj_arr = x_arr[j_idx]

    # ── Steps 2 … num_steps via fori_loop (O(1) graph size in num_steps) ──
    def step_body(_, carry):
        x_arr, pxi_arr, pxj_arr, cbf_pen, min_sep_cost = carry

        # Unrefined propagation for CBF -- one vmapped call over n scenarios
        x_next_batch = jax.vmap(step_all)(arr_to_ivl_batch(x_arr), p_batch)
        x_next_arr = jnp.concatenate([x_next_batch.lower, x_next_batch.upper], axis=-1)
        cbf_pen = cbf_pen + jnp.sum(jax.vmap(cbf_of)(x_next_batch.lower, x_next_batch.upper))

        # Intersection refinement, vectorized over n_pairs
        xi_lo, xi_hi = pxi_arr[:, :xlen], pxi_arr[:, xlen:]
        xj_lo, xj_hi = pxj_arr[:, :xlen], pxj_arr[:, xlen:]

        obs_offset_i, obs_scale_i = obs_offset_batch[i_idx], obs_scale_batch[i_idx]
        obs_offset_j, obs_scale_j = obs_offset_batch[j_idx], obs_scale_batch[j_idx]

        obs_i_lo = obs_offset_i + obs_scale_i * xi_lo[:, :2]
        obs_i_hi = obs_offset_i + obs_scale_i * xi_hi[:, :2]
        obs_j_lo = obs_offset_j + obs_scale_j * xj_lo[:, :2]
        obs_j_hi = obs_offset_j + obs_scale_j * xj_hi[:, :2]

        y_lo = jnp.maximum(obs_i_lo, obs_j_lo)
        y_hi = jnp.minimum(obs_i_hi, obs_j_hi)
        has_overlap = jnp.all(y_hi >= y_lo, axis=-1)   # (n_pairs,)

        fallback  = (xi_lo[:, :2] + xi_hi[:, :2]) / 2
        y_lo_safe = jnp.where(has_overlap[:, None], y_lo, fallback)
        y_hi_safe = jnp.where(has_overlap[:, None], y_hi, fallback)

        xi_ref_lo = jnp.concatenate([(y_lo_safe - obs_offset_i) / obs_scale_i, xi_lo[:, 2:3]], axis=-1)
        xi_ref_hi = jnp.concatenate([(y_hi_safe - obs_offset_i) / obs_scale_i, xi_hi[:, 2:3]], axis=-1)
        xj_ref_lo = jnp.concatenate([(y_lo_safe - obs_offset_j) / obs_scale_j, xj_lo[:, 2:3]], axis=-1)
        xj_ref_hi = jnp.concatenate([(y_hi_safe - obs_offset_j) / obs_scale_j, xj_hi[:, 2:3]], axis=-1)

        # Propagate BOTH pair sides in one vmap over the concatenated (2*n_pairs,) batch
        xref_lo = jnp.concatenate([xi_ref_lo, xj_ref_lo], axis=0)
        xref_hi = jnp.concatenate([xi_ref_hi, xj_ref_hi], axis=0)
        p_i = irx.Interval(lower=p_batch.lower[i_idx], upper=p_batch.upper[i_idx])
        p_j = irx.Interval(lower=p_batch.lower[j_idx], upper=p_batch.upper[j_idx])
        p_ref = irx.Interval(
            lower=jnp.concatenate([p_i.lower, p_j.lower], axis=0),
            upper=jnp.concatenate([p_i.upper, p_j.upper], axis=0),
        )
        xref = irx.Interval(lower=xref_lo, upper=xref_hi)
        xn_batch = jax.vmap(step_all)(xref, p_ref)
        xn_i = irx.Interval(lower=xn_batch.lower[:n_pairs], upper=xn_batch.upper[:n_pairs])
        xn_j = irx.Interval(lower=xn_batch.lower[n_pairs:], upper=xn_batch.upper[n_pairs:])

        obs_ni = irx.Interval(lower=obs_offset_i + obs_scale_i * xn_i.lower[:, :2],
                              upper=obs_offset_i + obs_scale_i * xn_i.upper[:, :2])
        obs_nj = irx.Interval(lower=obs_offset_j + obs_scale_j * xn_j.lower[:, :2],
                              upper=obs_offset_j + obs_scale_j * xn_j.upper[:, :2])
        raw_cost = jax.vmap(overlap_size_lax)(obs_ni, obs_nj)   # (n_pairs,)
        step_sep_cost = jnp.sum(jnp.where(has_overlap, raw_cost, 0.0))

        min_sep_cost = jnp.minimum(min_sep_cost, step_sep_cost)
        new_pxi_arr = jnp.concatenate([xn_i.lower, xn_i.upper], axis=-1)
        new_pxj_arr = jnp.concatenate([xn_j.lower, xn_j.upper], axis=-1)
        return (x_next_arr, new_pxi_arr, new_pxj_arr, cbf_pen, min_sep_cost)

    init_carry = (x_arr, pxi_arr, pxj_arr, cbf_pen, min_sep_cost)
    _, _, _, cbf_pen_final, min_sep_cost_final = jax.lax.fori_loop(
        0, num_steps - 1, step_body, init_carry
    )

    return min_sep_cost_final + cbf_weight * cbf_pen_final


def optimize_output_feedback_cbf_vmapped_gpu(
    cl_scenarios: List[Scenario],
    x0_ivl: irx.Interval,
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,
    cbf_weight: float = 1.0,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 200,
    seed: int = 42,
    init_std: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Same as optimize_output_feedback_cbf_gpu but (a) uses
    separation_cbf_loss_refined_vmapped (requires cl_scenarios from
    create_unified_cl_scenarios) and (b) fuses the per-iteration loss+grad
    into ONE jax.vmap(jax.value_and_grad(loss_fn)) call instead of separate
    batched_loss=vmap(loss_fn) and batched_grad=vmap(grad(loss_fn)) tracings
    -- jax.grad already reruns the forward pass internally, so tracing the
    (expensive interval-propagation) forward graph a second time for
    batched_loss alone was duplicated compile work. value_and_grad traces
    the forward pass once and reuses it for both outputs; the final
    per-restart loss used to pick the best restart now comes from the last
    GD iteration's value_and_grad call instead of a separate closing
    batched_loss(theta_final) call.

    `init_std`: stddev of the per-component Gaussian used to sample theta0
    around theta_mean (K=0, r=[0.5,0]). Originally hardcoded to 0.1 -- found
    (this session) to keep essentially every restart's K within a tiny
    neighborhood of 0, and combined with a near-vanishing gradient signal
    (loss's min-over-steps + has_overlap masking only backprops through one
    active step/pair at a time) and plain (non-normalized) GD at
    learning_rate=0.01, the winning restart moved ||K_final-K0||=0.0006 over
    20 iterations -- i.e. GD did essentially nothing and the reported
    "optimum" was just the best of 100 random draws. Widening init_std
    (e.g. to 1.0, still well inside the [-5,5] K clip range) spreads
    restarts across a much larger part of theta-space up front, so more of
    them start already exciting meaningful K@y feedback (and hence omega)
    rather than relying on GD to discover it from a near-zero start.
    """
    key        = jax.random.PRNGKey(seed)
    theta_mean = jnp.concatenate([jnp.zeros(4), jnp.array([0.5, 0.0])])
    theta0     = jax.random.normal(key, (num_restarts, 6)) * init_std + theta_mean

    def loss_fn(theta):
        return separation_cbf_loss_refined_vmapped(
            theta=theta, x0_ivl=x0_ivl, cl_scenarios=cl_scenarios, dt=dt,
            num_steps=num_steps, obstacles=obstacles, cbf_weight=cbf_weight,
        )

    batched_value_and_grad = jax.vmap(jax.value_and_grad(loss_fn))

    def body(_, carry):
        theta_batch, _prev_losses = carry
        losses, g = batched_value_and_grad(theta_batch)
        return (_project_theta(theta_batch - learning_rate * g), losses)

    init_losses = jnp.zeros(num_restarts)
    theta_final, losses = jax.lax.fori_loop(0, num_iters, body, (theta0, init_losses))
    # One more value_and_grad call so `losses` reflects theta_final (the
    # fori_loop's carried `losses` are evaluated at the PRE-update theta of
    # the final iteration, one step stale) -- matches optimize_output_
    # feedback_cbf_gpu's semantics of returning batched_loss(theta_final).
    losses, _ = batched_value_and_grad(theta_final)

    best_idx   = jnp.argmin(losses)
    best_theta = theta_final[best_idx]
    best_loss  = losses[best_idx]
    return best_theta, best_loss, theta_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 7b.  Tracking output-feedback controller
# ══════════════════════════════════════════════════════════════════════════════

def tracking_cbf_loss(
    theta_seq: jnp.ndarray,     # (num_steps, 6)
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    y_hat_seq: jnp.ndarray,     # (num_steps, 2)
    obstacles: jnp.ndarray,     # (N, 3)
    cbf_weight: float = 1.0,
    num_substeps: int = 1,
) -> jnp.ndarray:
    """Combined separation + CBF tracking loss.

    All three scenarios share _UNIFIED_TRACK_CL_EMB (observation model is
    encoded in the 4-D p interval).  Propagation and pair refinement are
    implemented as jax.vmap over scenarios / pairs rather than unrolled Python
    loops over heterogeneous embeddings.  This allows XLA to compile a single
    kernel instead of three separate subgraphs, dramatically reducing peak GPU
    memory during vmap(grad(loss_fn)) in the outer optimiser.
    """
    n          = len(cl_scenarios)
    pairs_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    P          = len(pairs_list)
    xlen       = x0_ivl.lower.shape[0]
    ylen       = 2   # output is [px, py]

    # ── Per-scenario constants (stacked for vmap) ─────────────────────────
    p_lowers    = jnp.stack([s.p_interval.lower for s in cl_scenarios])   # (n, plen)
    p_uppers    = jnp.stack([s.p_interval.upper for s in cl_scenarios])   # (n, plen)
    obs_scales  = jnp.stack([s.obs_scale         for s in cl_scenarios])  # (n, 1)
    obs_offsets = jnp.stack([s.obs_offset         for s in cl_scenarios]) # (n, 2)

    # ── Per-pair constants (compile-time, Python lists → JAX arrays) ──────
    pair_ia = [ia for ia, ib in pairs_list]
    pair_ib = [ib for ia, ib in pairs_list]
    pair_pl_a  = jnp.stack([p_lowers[ia]    for ia in pair_ia])  # (P, plen)
    pair_pu_a  = jnp.stack([p_uppers[ia]    for ia in pair_ia])
    pair_pl_b  = jnp.stack([p_lowers[ib]    for ib in pair_ib])
    pair_pu_b  = jnp.stack([p_uppers[ib]    for ib in pair_ib])
    pair_os_a  = jnp.stack([obs_scales[ia]  for ia in pair_ia])  # (P, 1)
    pair_oo_a  = jnp.stack([obs_offsets[ia] for ia in pair_ia])  # (P, 2)
    pair_os_b  = jnp.stack([obs_scales[ib]  for ib in pair_ib])
    pair_oo_b  = jnp.stack([obs_offsets[ib] for ib in pair_ib])

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

    def obs_arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:ylen], upper=arr[ylen:])

    def pack(k: int) -> jnp.ndarray:
        return jnp.concatenate([theta_seq[k, :4], y_hat_seq[k], theta_seq[k, 4:6]])

    # ── Propagate one scenario (all share _UNIFIED_TRACK_CL_EMB) ─────────
    def prop_one(x_arr: jnp.ndarray, p_lower: jnp.ndarray,
                  p_upper: jnp.ndarray, full_theta: jnp.ndarray) -> jnp.ndarray:
        x_ivl = arr_to_ivl(x_arr)
        p_ivl = irx.Interval(lower=p_lower, upper=p_upper)
        x_next = cl_euler_multistep(
            _UNIFIED_TRACK_CL_EMB, x_ivl, full_theta, p_ivl, dt, num_substeps
        )
        return ivl_to_arr(x_next)

    # ── vmap over n scenarios (one XLA kernel, not n separate subgraphs) ──
    def prop_all(x_arr_all: jnp.ndarray, full_theta: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda xa, pl, pu: prop_one(xa, pl, pu, full_theta))(
            x_arr_all, p_lowers, p_uppers
        )   # (n, 2*xlen)

    # ── CBF penalty vectorised over n scenarios ───────────────────────────
    def cbf_all(x_arr_all: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jax.vmap(
            lambda xa: cbf_penalty_interval(arr_to_ivl(xa), obstacles)
        )(x_arr_all))

    # ── Obs interval (flat) for one scenario ──────────────────────────────
    def obs_one(x_arr, obs_scale, obs_offset):
        return ivl_to_arr(_obs_interval_raw(arr_to_ivl(x_arr), obs_scale, obs_offset))

    # ── Refine + propagate ONE pair ───────────────────────────────────────
    def refine_and_prop_pair(
        pxi_arr: jnp.ndarray, pxj_arr: jnp.ndarray,
        pl_a: jnp.ndarray, pu_a: jnp.ndarray,
        pl_b: jnp.ndarray, pu_b: jnp.ndarray,
        os_a: jnp.ndarray, oo_a: jnp.ndarray,
        os_b: jnp.ndarray, oo_b: jnp.ndarray,
        full_theta: jnp.ndarray,
    ):
        xi = arr_to_ivl(pxi_arr)
        xj = arr_to_ivl(pxj_arr)

        obs_i = obs_arr_to_ivl(obs_one(pxi_arr, os_a, oo_a))
        obs_j = obs_arr_to_ivl(obs_one(pxj_arr, os_b, oo_b))

        y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
        y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
        has_overlap = jnp.all(y_hi >= y_lo)

        fallback  = (xi.lower[:2] + xi.upper[:2]) / 2
        y_lo_safe = jnp.where(has_overlap, y_lo, fallback)
        y_hi_safe = jnp.where(has_overlap, y_hi, fallback)

        si_s = os_a[0]
        sj_s = os_b[0]

        xi_ref = irx.Interval(
            lower=jnp.array([
                (y_lo_safe[0] - oo_a[0]) / si_s,
                (y_lo_safe[1] - oo_a[1]) / si_s,
                xi.lower[2],
            ]),
            upper=jnp.array([
                (y_hi_safe[0] - oo_a[0]) / si_s,
                (y_hi_safe[1] - oo_a[1]) / si_s,
                xi.upper[2],
            ]),
        )
        xj_ref = irx.Interval(
            lower=jnp.array([
                (y_lo_safe[0] - oo_b[0]) / sj_s,
                (y_lo_safe[1] - oo_b[1]) / sj_s,
                xj.lower[2],
            ]),
            upper=jnp.array([
                (y_hi_safe[0] - oo_b[0]) / sj_s,
                (y_hi_safe[1] - oo_b[1]) / sj_s,
                xj.upper[2],
            ]),
        )

        # Propagate both halves of the pair via a single vmap of 2
        both_arrs = jnp.stack([ivl_to_arr(xi_ref), ivl_to_arr(xj_ref)])  # (2, 2*xlen)
        both_pl   = jnp.stack([pl_a, pl_b])                               # (2, plen)
        both_pu   = jnp.stack([pu_a, pu_b])
        both_next = jax.vmap(
            lambda xa, pl, pu: prop_one(xa, pl, pu, full_theta)
        )(both_arrs, both_pl, both_pu)                                     # (2, 2*xlen)
        xn_i_arr, xn_j_arr = both_next[0], both_next[1]

        obs_ni = obs_arr_to_ivl(obs_one(xn_i_arr, os_a, oo_a))
        obs_nj = obs_arr_to_ivl(obs_one(xn_j_arr, os_b, oo_b))

        raw_cost = overlap_size_lax(obs_ni, obs_nj)
        overlap  = jnp.where(has_overlap, raw_cost, 0.0)
        return overlap, xn_i_arr, xn_j_arr

    # ── vmap over P pairs (one kernel for all pairs) ──────────────────────
    def refine_and_prop_all_pairs(
        pxi_arr_all: jnp.ndarray,  # (P, 2*xlen)
        pxj_arr_all: jnp.ndarray,
        full_theta: jnp.ndarray,
    ):
        return jax.vmap(
            lambda pxi, pxj, pla, pua, plb, pub, osa, ooa, osb, oob:
                refine_and_prop_pair(
                    pxi, pxj, pla, pua, plb, pub, osa, ooa, osb, oob, full_theta
                )
        )(
            pxi_arr_all, pxj_arr_all,
            pair_pl_a, pair_pu_a, pair_pl_b, pair_pu_b,
            pair_os_a, pair_oo_a, pair_os_b, pair_oo_b,
        )   # → overlaps (P,), xn_i_all (P, 2*xlen), xn_j_all (P, 2*xlen)

    # ── Step 0: propagate x0 → step 1 ────────────────────────────────────
    full_0 = pack(0)
    x0_arr = ivl_to_arr(x0_ivl)
    x_arr  = prop_all(jnp.stack([x0_arr] * n), full_0)   # (n, 2*xlen)
    cbf_pen = cbf_all(x_arr)
    sep_cost_acc = jnp.array(0.0)

    # ── lax.scan over steps 1..num_steps-1 ───────────────────────────────
    # Accumulate the SUM of refined overlaps over all (pair, step) so that
    # sep_cost > 0 whenever any (pair, step) has positive refined overlap.
    # This matches the oracle semantics: the loss is positive iff the oracle
    # returns True (there exists a step where a pair fails to separate after
    # refinement and one-step-ahead propagation).
    #
    # Fresh full-accumulated x_arr is used at each step (not a pairwise-
    # refined carry) so that each step's refinement starts from the true
    # reachable set.
    def scan_fn(carry, full_theta_k):
        x_arr, cbf_pen, sep_cost_acc = carry

        # 1. Propagate all n scenarios for CBF (single vmap)
        x_next_arr = prop_all(x_arr, full_theta_k)
        cbf_pen    = cbf_pen + cbf_all(x_next_arr)

        # 2. Refine + propagate all P pairs using full accumulated states
        pxi_now = jnp.stack([x_arr[ia] for ia, ib in pairs_list])  # (P, 2*xlen)
        pxj_now = jnp.stack([x_arr[ib] for ia, ib in pairs_list])
        overlaps, _, _ = refine_and_prop_all_pairs(pxi_now, pxj_now, full_theta_k)

        sep_cost_acc = sep_cost_acc + jnp.sum(overlaps)
        return (x_next_arr, cbf_pen, sep_cost_acc), None

    full_theta_seq = jax.vmap(pack)(jnp.arange(1, theta_seq.shape[0]))  # (T-1, 8)

    carry = (x_arr, cbf_pen, sep_cost_acc)
    (_, cbf_pen_f, sep_cost_f), _ = jax.lax.scan(
        scan_fn, carry, full_theta_seq
    )

    return sep_cost_f + cbf_weight * cbf_pen_f


def optimize_tracking_cbf_gpu(
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    y_hat_seq: jnp.ndarray,     # (num_steps, 2) — reference positions (fixed)
    obstacles: jnp.ndarray,     # (N, 3)
    cbf_weight: float = 1.0,
    num_restarts: int = 100,
    learning_rate: float = 0.1,
    num_iters: int = 200,
    seed: int = 42,
    num_substeps: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent for the error-feedback tracking controller.

    Decision variable:  theta_seq ∈ R^{num_steps × 6}
        Each row theta_k = [K_k.flat (4-D), u_ff_k (2-D)]
        Both the feedback gain and feedforward are jointly optimised.

    Control law:
        u_k = clip(K_k @ (y_k − ŷ_k) + u_ff_k,  [-1,-0.15], [1, 0.15])

    Actuation limits enforced inside the CL system:
        v  ∈ [-1.0,  1.0]  m/s
        ω  ∈ [-0.15, 0.15] rad/s

    Initialisation: K_k = 0, u_ff_k = 0  (small Gaussian noise around zero).

    Returns
    -------
    (best_theta_seq, best_loss, all_theta_seq_final, all_losses)
    best_theta_seq      : (num_steps, 6)  optimal per-step [K_k.flat, u_ff_k]
    best_loss           : scalar
    all_theta_seq_final : (num_restarts, num_steps, 6)
    all_losses          : (num_restarts,)
    """
    num_steps = y_hat_seq.shape[0]
    key       = jax.random.PRNGKey(seed)

    # Initialise K ≈ 0, u_ff ≈ 0  (small Gaussian noise around zero)
    noise   = jax.random.normal(key, (num_restarts, num_steps, 6)) * 0.1
    theta0  = noise  # (num_restarts, num_steps, 6)

    def loss_fn(theta_seq):
        return tracking_cbf_loss(
            theta_seq,
            x0_ivl=x0_ivl,
            cl_scenarios=cl_scenarios,
            dt=dt,
            y_hat_seq=y_hat_seq,
            obstacles=obstacles,
            cbf_weight=cbf_weight,
            num_substeps=num_substeps,
        )

    batched_loss = jax.vmap(loss_fn)
    batched_grad = jax.vmap(jax.grad(loss_fn))

    def body(_, theta_batch):
        g = batched_grad(theta_batch)
        return jnp.clip(theta_batch - learning_rate * g, _THETA_LO_TRACK, _THETA_HI_TRACK)

    theta_final  = jax.lax.fori_loop(0, num_iters, body, theta0)
    losses       = batched_loss(theta_final)
    best_idx     = jnp.argmin(losses)
    best_theta   = theta_final[best_idx]   # (num_steps, 4)
    best_loss    = losses[best_idx]
    return best_theta, best_loss, theta_final, losses


def optimize_tracking_cbf_vmapped_gpu(
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    y_hat_seq: jnp.ndarray,
    obstacles: jnp.ndarray,
    cbf_weight: float = 1.0,
    num_restarts: int = 100,
    learning_rate: float = 0.1,
    num_iters: int = 200,
    seed: int = 42,
    num_substeps: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Same as optimize_tracking_cbf_gpu, but fuses batched_loss/batched_grad
    into one jax.vmap(jax.value_and_grad(...)) call -- tracking_cbf_loss
    itself was already vmapped/scanned (see its docstring), but the OUTER
    optimizer still traced the forward loss graph twice (once for
    batched_loss, once inside batched_grad), the same redundancy fixed for
    optimize_output_feedback_cbf_vmapped_gpu. Caller should wrap this in an
    outer jax.jit before timing/deploying it -- see that function's
    docstring for why (avoids the eager-fori_loop+vmap near-OOM found
    earlier this session)."""
    num_steps = y_hat_seq.shape[0]
    key       = jax.random.PRNGKey(seed)
    theta0    = jax.random.normal(key, (num_restarts, num_steps, 6)) * 0.1

    def loss_fn(theta_seq):
        return tracking_cbf_loss(
            theta_seq, x0_ivl=x0_ivl, cl_scenarios=cl_scenarios, dt=dt,
            y_hat_seq=y_hat_seq, obstacles=obstacles, cbf_weight=cbf_weight,
            num_substeps=num_substeps,
        )

    batched_value_and_grad = jax.vmap(jax.value_and_grad(loss_fn))

    def body(_, carry):
        theta_batch, _prev_losses = carry
        losses, g = batched_value_and_grad(theta_batch)
        return (jnp.clip(theta_batch - learning_rate * g, _THETA_LO_TRACK, _THETA_HI_TRACK), losses)

    init_losses = jnp.zeros(num_restarts)
    theta_final, losses = jax.lax.fori_loop(0, num_iters, body, (theta0, init_losses))
    losses, _ = batched_value_and_grad(theta_final)   # re-evaluate at theta_final (see
                                                       # optimize_output_feedback_cbf_vmapped_gpu's
                                                       # docstring for why)

    best_idx   = jnp.argmin(losses)
    best_theta = theta_final[best_idx]
    best_loss  = losses[best_idx]
    return best_theta, best_loss, theta_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Open-loop separating controller with CBF
# ══════════════════════════════════════════════════════════════════════════════

def create_ol_scenarios(
    actuator_alpha_lo: float = 0.0,
    actuator_alpha_hi: float = 0.5,
) -> List[Scenario]:
    """Three open-loop fault scenarios (u = [v, ω] directly, no output feedback).

    Reuses the open-loop embeddings from faulty_car_separating_input:
      Nominal       : alpha = 1,             y = [px, py]
      Actuator Fault: alpha ∈ [lo, hi],      y = [px, py]
      Sensor Fault  : alpha = 1,             y = [0.95·px + 0.2, 0.95·py + 0.2]
    """
    from faulty_car_separating_input import create_scenarios
    return create_scenarios(actuator_alpha_lo, actuator_alpha_hi)


def ol_cbf_loss(
    theta_seq: jnp.ndarray,
    x0_ivl: irx.Interval,
    ol_scenarios: List[Scenario],
    dt: float,
    obstacles: jnp.ndarray,
    cbf_weight: float = 1.0,
    num_substeps: int = 1,
) -> jnp.ndarray:
    n         = len(ol_scenarios)
    pairs     = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    num_steps = theta_seq.shape[0]
    xlen      = x0_ivl.lower.shape[0]

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

    def _prop(emb_sys, x_ivl, u_k, p_ivl):
        return cl_euler_multistep(emb_sys, x_ivl, u_k, p_ivl, dt, num_substeps)

    # ── Step 1 ────────────────────────────────────────────────────────────
    u0 = theta_seq[0]
    x_ivls = [
        _prop(s.emb_system, x0_ivl, u0, s.p_interval)
        for s in ol_scenarios
    ]
    obs_ivls = [_obs_interval(x, s) for x, s in zip(x_ivls, ol_scenarios)]

    # Per-pair overlap at step 1 — shape (num_pairs,)
    pair_overlaps_step1 = jnp.stack([
        overlap_size_lax(obs_ivls[i], obs_ivls[j])
        for i, j in pairs
    ])
    # Each pair independently tracks its minimum overlap seen so far
    min_sep_per_pair = pair_overlaps_step1  # shape: (num_pairs,)

    cbf_pen = jnp.array(0.0)
    for x in x_ivls:
        cbf_pen = cbf_pen + cbf_penalty_interval(x, obstacles)

    x_arr   = jnp.stack([ivl_to_arr(x)         for x     in x_ivls])
    pxi_arr = jnp.stack([ivl_to_arr(x_ivls[i]) for i, j  in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x_ivls[j]) for i, j  in pairs])

    # ── Steps 2..num_steps ───────────────────────────────────────────────
    def step_fn(carry, u_k):
        x_arr, pxi_arr, pxj_arr, cbf_pen, min_sep_per_pair = carry

        # Unrefined propagation (for CBF only)
        x_next_list = [
            _prop(ol_scenarios[si].emb_system, arr_to_ivl(x_arr[si]),
                  u_k, ol_scenarios[si].p_interval)
            for si in range(n)
        ]
        x_next_arr = jnp.stack([ivl_to_arr(x) for x in x_next_list])
        for x in x_next_list:
            cbf_pen = cbf_pen + cbf_penalty_interval(x, obstacles)

        # Intersection refinement per pair
        new_pxi_list      = []
        new_pxj_list      = []
        pair_overlaps_now = []  # per-pair overlap at this timestep

        for idx, (i, j) in enumerate(pairs):
            xi = arr_to_ivl(pxi_arr[idx])
            xj = arr_to_ivl(pxj_arr[idx])
            obs_i = _obs_interval(xi, ol_scenarios[i])
            obs_j = _obs_interval(xj, ol_scenarios[j])

            y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
            y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
            has_overlap = jnp.all(y_hi >= y_lo)

            fallback  = (xi.lower[:2] + xi.upper[:2]) / 2
            y_lo_safe = jnp.where(has_overlap, y_lo, fallback)
            y_hi_safe = jnp.where(has_overlap, y_hi, fallback)

            si_s = ol_scenarios[i].obs_scale[0]
            sj_s = ol_scenarios[j].obs_scale[0]

            xi_ref = irx.Interval(
                lower=jnp.array([
                    (y_lo_safe[0] - ol_scenarios[i].obs_offset[0]) / si_s,
                    (y_lo_safe[1] - ol_scenarios[i].obs_offset[1]) / si_s,
                    xi.lower[2],
                ]),
                upper=jnp.array([
                    (y_hi_safe[0] - ol_scenarios[i].obs_offset[0]) / si_s,
                    (y_hi_safe[1] - ol_scenarios[i].obs_offset[1]) / si_s,
                    xi.upper[2],
                ]),
            )
            xj_ref = irx.Interval(
                lower=jnp.array([
                    (y_lo_safe[0] - ol_scenarios[j].obs_offset[0]) / sj_s,
                    (y_lo_safe[1] - ol_scenarios[j].obs_offset[1]) / sj_s,
                    xj.lower[2],
                ]),
                upper=jnp.array([
                    (y_hi_safe[0] - ol_scenarios[j].obs_offset[0]) / sj_s,
                    (y_hi_safe[1] - ol_scenarios[j].obs_offset[1]) / sj_s,
                    xj.upper[2],
                ]),
            )

            xn_i = _prop(ol_scenarios[i].emb_system, xi_ref, u_k, ol_scenarios[i].p_interval)
            xn_j = _prop(ol_scenarios[j].emb_system, xj_ref, u_k, ol_scenarios[j].p_interval)

            raw_cost = overlap_size_lax(
                _obs_interval(xn_i, ol_scenarios[i]),
                _obs_interval(xn_j, ol_scenarios[j]),
            )
            # Each pair contributes its own overlap (0.0 if no overlap)
            pair_overlaps_now.append(jnp.where(has_overlap, raw_cost, 0.0))
            new_pxi_list.append(ivl_to_arr(xn_i))
            new_pxj_list.append(ivl_to_arr(xn_j))

        # Update each pair's running minimum independently
        pair_overlaps_now = jnp.stack(pair_overlaps_now)           # (num_pairs,)
        min_sep_per_pair  = jnp.minimum(min_sep_per_pair, pair_overlaps_now)

        new_carry = (x_next_arr, jnp.stack(new_pxi_list), jnp.stack(new_pxj_list),
                     cbf_pen, min_sep_per_pair)
        return new_carry, None

    carry = (x_arr, pxi_arr, pxj_arr, cbf_pen, min_sep_per_pair)
    for k in range(1, num_steps):
        carry, _ = step_fn(carry, theta_seq[k])

    _, _, _, cbf_pen_f, min_sep_per_pair_f = carry

    # Sum of per-pair minima — matches sum_q( min_t( overlap_q(t) ) )
    sep_cost = jnp.sum(min_sep_per_pair_f)
    return sep_cost + cbf_weight * cbf_pen_f


def optimize_openloop_cbf_gpu(
    x0_ivl: irx.Interval,
    ol_scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,          # (N, 3) — [cx, cy, r_obs] per row
    cbf_weight: float = 1.0,
    num_restarts: int = 100,
    learning_rate: float = 0.1,
    num_iters: int = 200,
    seed: int = 42,
    num_substeps: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent for the open-loop separating controller.

    Decision variable:  theta_seq ∈ R^{num_steps × 2}
        Each row theta_k = [v_k, ω_k] — direct open-loop control input.
        The same command is broadcast to all fault scenarios.

    Actuation limits enforced by projection at each gradient step.

    Initialisation: uniform over the full admissible range
        [_CL_U_LO_TRACK, _CL_U_HI_TRACK] = [[-1, -0.15], [1, 0.15]]

    num_substeps : each 1-second control interval is integrated with
        num_substeps forward-Euler sub-steps of size dt/num_substeps,
        using lax.scan internally (O(1) graph size, minimal memory).

    Returns
    -------
    (best_theta_seq, best_loss, all_theta_seq_final, all_losses)
    best_theta_seq      : (num_steps, 2)  optimal per-step [v_k, ω_k]
    best_loss           : scalar
    all_theta_seq_final : (num_restarts, num_steps, 2)
    all_losses          : (num_restarts,)
    """
    key    = jax.random.PRNGKey(seed)
    # Uniform init over full admissible input range — better coverage than
    # small Gaussian noise, costs no extra memory vs normal sampling.
    theta0 = jax.random.uniform(
        key, (num_restarts, num_steps, 2),
        minval=_CL_U_LO_TRACK, maxval=_CL_U_HI_TRACK,
    )  # (num_restarts, num_steps, 2)

    def loss_fn(theta_seq):
        return ol_cbf_loss(
            theta_seq,
            x0_ivl=x0_ivl,
            ol_scenarios=ol_scenarios,
            dt=dt,
            obstacles=obstacles,
            cbf_weight=cbf_weight,
            num_substeps=num_substeps,
        )

    batched_loss = jax.vmap(loss_fn)
    batched_grad = jax.vmap(jax.grad(loss_fn))

    def body(_, theta_batch):
        g = batched_grad(theta_batch)
        return jnp.clip(theta_batch - learning_rate * g, _CL_U_LO_TRACK, _CL_U_HI_TRACK)

    theta_final = jax.lax.fori_loop(0, num_iters, body, theta0)
    losses      = batched_loss(theta_final)
    best_idx    = jnp.argmin(losses)
    best_theta  = theta_final[best_idx]   # (num_steps, 2)
    best_loss   = losses[best_idx]
    return best_theta, best_loss, theta_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Diagnostics
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

    # Unified embedding (all 3 scenarios share ONE traced/differentiated
    # embedding -- see CarUnifiedCLSystem's docstring) + the vmapped,
    # value_and_grad-fused loss/optimizer -- both added this session to fix
    # a >3x compile-time regression from the original Python-loop-over-
    # scenarios/pairs + separate loss/grad tracing implementation. Verified
    # bit-identical output to the original on the pre-fix config before
    # any of the settings below were tuned.
    cl_scenarios = create_unified_cl_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
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

    # 5s horizon / 10 steps (up from the original 1s/20 steps -- the short
    # horizon never let the actuator-fault pair excite omega enough to
    # separate; see this module's history comment on _K_MAX for the full
    # tuning trace). dt=0.5s per step means ONE big Euler step per segment
    # (no sub-stepping) -- this is why _K_MAX is bounded (see below): large
    # K under this coarse integration compounds into wrapping-effect box
    # blow-up that produces a misleadingly-low loss without genuine
    # (raw-independent-box) separation.
    dt, num_steps = 0.5, 10
    cbf_weight    = 2.0
    print(f"\nHorizon: {num_steps} × {dt} s = {num_steps * dt:.2f} s")
    print(f"CBF weight: {cbf_weight}")

    # Tuned this session (see _K_MAX's history comment for the full sweep):
    # K_MAX=2.0 is the empirically-verified threshold between genuine
    # (raw-box, not just refined-metric) separation and wrapping-effect
    # blow-up; init_std=1.0 (up from an original 0.1, which left GD unable
    # to move off its near-zero-K starting point) plus a larger
    # learning_rate=2.0 let 1000 restarts search effectively in just 2 GD
    # iterations, holding steady-state run time to ~32ms (<40ms budget) on
    # the GPU present on this machine. Best verified result at this exact
    # config: loss~4.7e-5, all 3 pairs' raw independent-box overlap
    # ~1e-5-1e-4 m^2 (vs. ~0.06 m^2 box size) and stable/non-growing --
    # negligible relative to a coarser (3 iters/LR=1.5) alternative that
    # LOOKED similar by loss value (0.000044) but left a persistent, real
    # 0.053 m^2 gap on the Nominal-vs-Sensor-Fault pair. Always check the
    # raw per-pair overlap directly (see plot/diagnostic scripts), not just
    # the scalar loss, before trusting a result from this optimizer.
    print("\nRunning output-feedback + CBF optimisation (unified/vmapped, GPU) …\n")
    best_theta, best_loss, theta_final, losses = optimize_output_feedback_cbf_vmapped_gpu(
        x0_ivl=x0_ivl,
        cl_scenarios=cl_scenarios,
        dt=dt,
        num_steps=num_steps,
        obstacles=obstacles,
        cbf_weight=cbf_weight,
        num_restarts=1000,
        learning_rate=2.0,
        num_iters=2,
        seed=42,
        init_std=1.0,
    )
    K, r = theta_to_K_r(best_theta)
    loss = float(best_loss)

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
