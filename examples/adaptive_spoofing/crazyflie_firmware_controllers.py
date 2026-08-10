"""
Crazyflie Firmware Controller Bank -- immrax System Definitions
====================================================================
Replaces `crazyflie_chain_controllers.py`'s fictional 4-candidate bank
(qps_snap_chain/pd_pos_vel/pid_pos_vel_i/indi_jerk, invented placeholders
for 3 of the 4 gains -- see that module's docstring and PLAN.md Sec 7) with
the REAL Crazyflie firmware controllers, transcribed from
`~/adaptive_spoofing/experiments/RQ3_transferability_pipeline/rq3_crazyflie_surrogates.py`:

    cf_pid           position_controller_pid.c    -- nested, saturated,
                     integral of VELOCITY error
    cf_mellinger     controller_mellinger.c        -- one-shot force vector,
                     integral of POSITION error
    cf_indi          controller_indi.c +
                     position_controller_indi.c    -- incremental, memory
                     is the previous command + a low-passed acceleration
    cf_brescianini   controller_brescianini.c      -- second-order reference
                     model, NO memory

Why four SEPARATE systems, not one shared masked-theta system
------------------------------------------------------------------
`crazyflie_chain_controllers.py`'s bank was unified into one masked-theta
family because all 4 fictional candidates really were subsets of one
5-term linear law. These four are not: they have different theta COUNTS
(15/12/11/10), different MEANINGS even where indices align, and different
memory kinds (velocity-error integral / position-error integral /
incremental-with-filtered-acceleration / none). Forcing them through one
traced function would require a much larger union theta and lose the
one-to-one correspondence with the source's own affine regressor -- exactly
the "more implementation work, no masking-convention risk" alternative
flagged (and not taken, because it wasn't needed) in PLAN.md Sec 6 Q2. Each
candidate below gets its OWN `irx.System`/embedding; propagation code loops
over the (small, fixed) candidate list in plain Python rather than
vmapping over a stacked shared parameter -- fine here because every
candidate's dynamics are LINEAR/AFFINE (no trig, no compile-cost concern),
unlike the nonlinear rigid-body plant this project's other systems worry
about.

State (21, unified/padded across all 4 candidates)
-----------------------------------------------------
    x[0:3]    position            (observable)
    x[3:6]    velocity            (observable)
    x[6:9]    attitude (roll,pitch,yaw), hover-linearized     (observable)
    x[9:12]   body rates (p,q,r)  (observable)
    x[12:15]  integ    -- cf_pid: integral of VELOCITY error
                          cf_mellinger: integral of POSITION error
                          cf_indi/cf_brescianini: unused, stays at 0
    x[15:18]  prev_cmd -- cf_indi only: the accumulated attitude/thrust
                          increment. Unused elsewhere.
    x[18:21]  accel_filt -- cf_indi only: low-passed reconstructed
                            acceleration. Unused elsewhere.

Control u (3,) = the attacker's position-spoof bias (same convention as
`crazyflie_chain_controllers.py`: the controller reads `p + bias`, the
plant integrates the true `p`).

Params p = each candidate's OWN `theta_true` (fixed, degenerate interval --
these are real, published-in-source firmware gains, not something the
optimizer searches over, matching the R4 decision in PLAN.md).

Reference r (15,) = `[pos_sp(3), vel_sp(3), acc_sp(3), jerk_sp(3), snap_sp(3)]`,
same layout and `hover_reference()` convention as before and as the source
module.

Why `evolution='discrete'`, not `continuous`+`euler_step`
-------------------------------------------------------------
`HoverPlant.matrices()` in the source is ALREADY a one-tick discrete affine
map (`x_next = A@x + B@u`, semi-implicit Euler baked into A,B -- see that
class's docstring for why semi-implicit, not explicit, matters here). There
is no continuous ODE to approximate with a separate Euler step; wrapping it
in `continuous`+`euler_step` would just add a redundant, incorrect
extra discretization. `evolution='discrete'` makes `f(t,x,u,p)` return
`x_next` directly, exactly matching the source's own `step()`.
(Note: immrax's discrete-evolution embedding needs an explicit
`refine=lambda z: z` kwarg passed to `emb.f(...)` -- its default is `None`,
which the discrete branch calls directly and crashes on. Not needed for
`continuous` evolution, which guards against `None` itself.)

Two primitives immrax's `natif` doesn't register, worked around
---------------------------------------------------------------------
`jnp.abs` and any direct comparison (`>`, `<`) are NOT in
`immrax.inclusion.nif.inclusion_registry` (only `min`/`max`/`eq`/etc.) --
needed for cf_pid's velocity-setpoint clamp and its saturation-survival
fraction (`_clip_survival` in the source). Two fixes, both verified against
interval inputs that straddle zero before trusting them here:
  - `abs(x)` -> `jnp.sqrt(x**2)`. Using `jnp.maximum(x, -x)` looks
    equivalent pointwise but is UNSOUND under natif's natural (not
    mean-value) extension: for an interval that straddles zero, natif
    computes elementwise max of `x` and `-x` independently, which just
    reproduces the same straddling interval instead of `[0, max(|lo|,|hi|)]`
    -- the classic interval-arithmetic dependency problem. Squaring then
    sqrt-ing IS computed soundly by natif (verified: a straddling-zero
    interval correctly collapses to `[0, ...]`, not blown up).
  - The source's `_clip_survival` (`clipped/raw` guarded by a hard
    `|raw|>1e-12` boolean) is replaced by the equivalent smooth form
    `min(1, L/(|raw|+eps))`, which needs no comparison at all and limits
    to exactly 1 as raw->0 (verified against the source's discrete formula
    at several concrete points: identical to 1e-6).
  - `_wrap_to_pi` (`%`, unsupported) is simply omitted: `yaw_sp - yaw`
    directly, valid because yaw_sp=0 and yaw stays near 0 (small-angle
    hover regime) throughout every scenario this module runs -- angle
    wrapping only matters for near-+-pi yaw, never reached here.

Cross-checked against the source module directly (not just eyeballed) in
tests/test_crazyflie_firmware_controllers.py, by importing
`rq3_crazyflie_surrogates` and comparing `.affine_step_terms()`/`.step()`
against this module's JAX implementation at random states -- same pattern
used for `crazyflie_12d.py` vs QPS's `forward_model()`.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

_RQ3_DIR = Path.home() / "adaptive_spoofing" / "experiments" / "RQ3_transferability_pipeline"
if str(_RQ3_DIR) not in sys.path:
    sys.path.insert(0, str(_RQ3_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import immrax as irx

import rq3_crazyflie_surrogates as ref   # the source of truth for A, B, theta_true

QPS_DT = 0.02
REFERENCE_DIM = 15
IDX_P, IDX_V = slice(0, 3), slice(3, 6)
IDX_ATT, IDX_RATE = slice(6, 9), slice(9, 12)
IDX_INTEG, IDX_PREV_CMD, IDX_ACCEL_FILT = slice(12, 15), slice(15, 18), slice(18, 21)
# `mem` (as passed to the per-candidate functions below) is ALREADY the 9-dim
# slice x21[12:21], so indexing it needs LOCAL offsets, not the x21-relative
# ones above (used only when slicing the full 21-dim state, i.e. nowhere in
# this file except conceptually in FirmwareControllerSystem.f's `x21[12:21]`).
_M_INTEG, _M_PREV_CMD, _M_ACCEL_FILT = slice(0, 3), slice(3, 6), slice(6, 9)

CANDIDATE_NAMES = ("cf_pid", "cf_mellinger", "cf_indi", "cf_brescianini")


def _safe_abs(x, grad_eps=1e-18):
    """Sound interval abs -- see module docstring. NOT jnp.maximum(x,-x).

    `jnp.sqrt(x**2)` alone is sound in VALUE but has a NaN GRADIENT at
    exactly x=0 (d/dx sqrt(x^2) = x/sqrt(x^2) = 0/0 there) -- a classic
    autodiff gotcha, found the hard way when a random-restart optimizer's
    initial bias landed a tracking error at exactly 0 and every gradient
    from that restart onward was NaN. `grad_eps` smooths this out (adds a
    completely negligible ~1e-9 floor to the returned value at x=0, verified
    against interval inputs straddling zero -- still sound, bounds shift by
    <1e-9) while making the gradient well-defined everywhere."""
    return jnp.sqrt(x ** 2 + grad_eps)


def _clip_survival(raw, limit, eps=1e-9):
    """Smooth, comparison-free equivalent of the source's _clip_survival."""
    return jnp.minimum(1.0, limit / (_safe_abs(raw) + eps))


def hover_reference(pos_sp: jnp.ndarray) -> jnp.ndarray:
    ref15 = jnp.zeros(REFERENCE_DIM)
    return ref15.at[0:3].set(jnp.asarray(pos_sp, dtype=jnp.float32))


# ══════════════════════════════════════════════════════════════════════════════
# Per-candidate virtual-control-terms + memory_step, transcribed to JAX.
# Each returns (u_offset (6,), phi_u (6, n_theta)) and mem_next (9,).
# All take: x (12, true or None-biased physical state as needed -- see each
# candidate below for exactly which rows get the bias added, matching the
# source's affine_step_terms/_virtual_control_terms split EXACTLY), r (15,),
# mem (9,), bias (3,).
# ══════════════════════════════════════════════════════════════════════════════

def _pid_virtual_control(x, r, mem, bias):
    x_ctrl_p = x[IDX_P] + bias
    e_p = r[0:3] - x_ctrl_p
    v_sp = jnp.array(ref.PID_POS_KP) * e_p + r[3:6]
    L = ref.PID_POS_VEL_MAX
    v_sp_c = jnp.clip(v_sp, -L, L)
    surv = _clip_survival(v_sp, L)

    roll, pitch, yaw = x[IDX_ATT]
    p_rate, q_rate, r_rate = x[IDX_RATE]
    integ = mem[_M_INTEG]
    n_theta = 15
    phi_u = jnp.zeros((6, n_theta))
    phi_u = phi_u.at[0, 1].set(surv[1] * e_p[1])
    phi_u = phi_u.at[0, 3].set(-x[IDX_V][1])
    phi_u = phi_u.at[0, 5].set(integ[1])
    phi_u = phi_u.at[0, 6].set(-roll)
    phi_u = phi_u.at[0, 9].set(-p_rate)
    phi_u = phi_u.at[1, 0].set(-surv[0] * e_p[0])
    phi_u = phi_u.at[1, 2].set(x[IDX_V][0])
    phi_u = phi_u.at[1, 4].set(-integ[0])
    phi_u = phi_u.at[1, 7].set(-pitch)
    phi_u = phi_u.at[1, 10].set(-q_rate)
    phi_u = phi_u.at[2, 8].set(0.0 - yaw)          # yaw_sp=0, no wrap needed (see docstring)
    phi_u = phi_u.at[2, 11].set(-r_rate)
    kt_z = ref.HoverPlant(dt=QPS_DT).thrust_accel_gain[2]
    phi_u = phi_u.at[5, 12].set(surv[2] * e_p[2] / kt_z)
    phi_u = phi_u.at[5, 13].set(-x[IDX_V][2] / kt_z)
    phi_u = phi_u.at[5, 14].set(integ[2] / kt_z)

    u_offset = jnp.zeros(6).at[5].set(r[8] / kt_z)
    e_v_clamped = v_sp_c - x[IDX_V]
    return u_offset, phi_u, e_v_clamped   # e_v_clamped needed by memory_step


def _pid_memory_step(x, r, mem, bias):
    _, _, e_v_clamped = _pid_virtual_control(x, r, mem, bias)
    integ_next = mem[_M_INTEG] + QPS_DT * e_v_clamped
    return jnp.concatenate([integ_next, mem[_M_PREV_CMD], mem[_M_ACCEL_FILT]])


def _mellinger_virtual_control(x, r, mem, bias):
    x_ctrl_p = x[IDX_P] + bias
    e_p = r[0:3] - x_ctrl_p
    e_v = r[3:6] - x[IDX_V]
    roll, pitch, yaw = x[IDX_ATT]
    p_rate, q_rate, r_rate = x[IDX_RATE]
    integ = mem[_M_INTEG]
    g = ref.GRAVITY
    n_theta = 12
    phi_u = jnp.zeros((6, n_theta))
    phi_u = phi_u.at[0, 0].set(e_p[1] / g)
    phi_u = phi_u.at[0, 1].set(e_v[1] / g)
    phi_u = phi_u.at[0, 2].set(integ[1] / g)
    phi_u = phi_u.at[0, 6].set(r[7] / g - roll)
    phi_u = phi_u.at[0, 9].set(-p_rate)
    phi_u = phi_u.at[1, 0].set(-e_p[0] / g)
    phi_u = phi_u.at[1, 1].set(-e_v[0] / g)
    phi_u = phi_u.at[1, 2].set(-integ[0] / g)
    phi_u = phi_u.at[1, 7].set(-r[6] / g - pitch)
    phi_u = phi_u.at[1, 10].set(-q_rate)
    phi_u = phi_u.at[2, 8].set(0.0 - yaw)
    phi_u = phi_u.at[2, 11].set(-r_rate)
    kt_z = ref.HoverPlant(dt=QPS_DT).thrust_accel_gain[2]
    phi_u = phi_u.at[5, 3].set(e_p[2] / kt_z)
    phi_u = phi_u.at[5, 4].set(e_v[2] / kt_z)
    phi_u = phi_u.at[5, 5].set(integ[2] / kt_z)

    u_offset = jnp.zeros(6).at[5].set(r[8] / kt_z)
    return u_offset, phi_u


def _mellinger_memory_step(x, r, mem, bias):
    p = x[IDX_P] + bias
    integ_next = mem[_M_INTEG] + QPS_DT * (r[0:3] - p)   # integral of POSITION error
    return jnp.concatenate([integ_next, mem[_M_PREV_CMD], mem[_M_ACCEL_FILT]])


def _indi_accel_ref(x_ctrl_p, v, r):
    e_p = r[0:3] - x_ctrl_p
    v_ref = jnp.array(ref.INDI_K_XI) * e_p + r[3:6]
    return jnp.array(ref.INDI_K_DXI) * (v_ref - v)


def _indi_virtual_control(x, r, mem, bias):
    x_ctrl_p = x[IDX_P] + bias
    e_p = r[0:3] - x_ctrl_p
    e_v = r[3:6] - x[IDX_V]
    roll, pitch, yaw = x[IDX_ATT]
    p_rate, q_rate, r_rate = x[IDX_RATE]
    prev_cmd = mem[_M_PREV_CMD]
    n_theta = 11
    phi_u = jnp.zeros((6, n_theta))
    phi_u = phi_u.at[0, 0].set(e_p[1])
    phi_u = phi_u.at[0, 1].set(e_v[1])
    phi_u = phi_u.at[0, 2].set(prev_cmd[1])
    phi_u = phi_u.at[0, 5].set(-roll)
    phi_u = phi_u.at[0, 8].set(-p_rate)
    phi_u = phi_u.at[1, 0].set(-e_p[0])
    phi_u = phi_u.at[1, 1].set(-e_v[0])
    phi_u = phi_u.at[1, 2].set(-prev_cmd[0])
    phi_u = phi_u.at[1, 6].set(-pitch)
    phi_u = phi_u.at[1, 9].set(-q_rate)
    phi_u = phi_u.at[2, 7].set(0.0 - yaw)
    phi_u = phi_u.at[2, 10].set(-r_rate)
    kt_z = ref.HoverPlant(dt=QPS_DT).thrust_accel_gain[2]
    phi_u = phi_u.at[5, 3].set(e_p[2] / kt_z)
    phi_u = phi_u.at[5, 4].set(e_v[2] / kt_z)

    u_offset = jnp.zeros(6).at[5].set(r[8] / kt_z)
    return u_offset, phi_u


_INDI_TAU = 1.0 / (2.0 * np.pi * ref.INDI_FILT_CUTOFF)
_INDI_FILT_ALPHA = float(np.clip(QPS_DT / (_INDI_TAU + QPS_DT), 0.0, 1.0))


def _indi_memory_step(x, r, mem, bias):
    x_ctrl_p = x[IDX_P] + bias
    a_meas = jnp.array([-ref.GRAVITY * x[IDX_ATT][1], ref.GRAVITY * x[IDX_ATT][0], 0.0])
    accel_filt_next = mem[_M_ACCEL_FILT] + _INDI_FILT_ALPHA * (a_meas - mem[_M_ACCEL_FILT])
    a_ref = _indi_accel_ref(x_ctrl_p, x[IDX_V], r)
    prev_cmd_next = ((1.0 - _INDI_FILT_ALPHA) * mem[_M_PREV_CMD]
                     + QPS_DT * (a_ref - accel_filt_next))
    return jnp.concatenate([mem[_M_INTEG], prev_cmd_next, accel_filt_next])


def _brescianini_virtual_control(x, r, mem, bias):
    x_ctrl_p = x[IDX_P] + bias
    e_p = r[0:3] - x_ctrl_p
    e_v = r[3:6] - x[IDX_V]
    roll, pitch, yaw = x[IDX_ATT]
    p_rate, q_rate, r_rate = x[IDX_RATE]
    g = ref.GRAVITY
    n_theta = 10
    phi_u = jnp.zeros((6, n_theta))
    phi_u = phi_u.at[0, 0].set(e_p[1] / g)
    phi_u = phi_u.at[0, 1].set(e_v[1] / g)
    phi_u = phi_u.at[0, 4].set(r[7] / g - roll)
    phi_u = phi_u.at[0, 7].set(-p_rate)
    phi_u = phi_u.at[1, 0].set(-e_p[0] / g)
    phi_u = phi_u.at[1, 1].set(-e_v[0] / g)
    phi_u = phi_u.at[1, 5].set(-r[6] / g - pitch)
    phi_u = phi_u.at[1, 8].set(-q_rate)
    phi_u = phi_u.at[2, 6].set(0.0 - yaw)
    phi_u = phi_u.at[2, 9].set(-r_rate)
    kt_z = ref.HoverPlant(dt=QPS_DT).thrust_accel_gain[2]
    phi_u = phi_u.at[5, 2].set(e_p[2] / kt_z)
    phi_u = phi_u.at[5, 3].set(e_v[2] / kt_z)

    u_offset = jnp.zeros(6).at[5].set(r[8] / kt_z)
    return u_offset, phi_u


def _brescianini_memory_step(x, r, mem, bias):
    return mem   # memoryless


_VIRTUAL_CONTROL = {
    "cf_pid": lambda x, r, mem, bias: _pid_virtual_control(x, r, mem, bias)[:2],
    "cf_mellinger": _mellinger_virtual_control,
    "cf_indi": _indi_virtual_control,
    "cf_brescianini": _brescianini_virtual_control,
}
_MEMORY_STEP = {
    "cf_pid": _pid_memory_step,
    "cf_mellinger": _mellinger_memory_step,
    "cf_indi": _indi_memory_step,
    "cf_brescianini": _brescianini_memory_step,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. System definition (one per candidate)
# ══════════════════════════════════════════════════════════════════════════════

class FirmwareControllerSystem(irx.System):
    """21-state discrete system for ONE named firmware controller candidate.

    State x (21) = [physical(12), integ(3), prev_cmd(3), accel_filt(3)].
    Control u (3) = attacker's position-spoof bias.
    Params p = this candidate's theta (n_theta,) -- see module docstring.
    """

    def __init__(self, name: str, ref15: jnp.ndarray, A: np.ndarray, B: np.ndarray):
        self.name = name
        self.evolution = 'discrete'
        self.xlen = 21
        self.ref15 = jnp.asarray(ref15)
        self.A = jnp.asarray(A)
        self.B = jnp.asarray(B)
        self._virtual_control = _VIRTUAL_CONTROL[name]
        self._memory_step = _MEMORY_STEP[name]

    def f(self, t, x21, u, p):
        bias = u
        x = x21[:12]
        mem = x21[12:21]
        u_offset, phi_u = self._virtual_control(x, self.ref15, mem, bias)
        u_virtual = u_offset + phi_u @ p
        x_next = self.A @ x + self.B @ u_virtual
        mem_next = self._memory_step(x, self.ref15, mem, bias)
        return jnp.concatenate([x_next, mem_next])


@dataclass
class Scenario:
    """One controller hypothesis. Unlike crazyflie_chain_controllers.py's
    Scenario, each has its OWN emb_system (see module docstring)."""
    name: str
    emb_system: object
    p_interval: irx.Interval


_BANK_CACHE: Dict[Tuple[float, ...], List[Scenario]] = {}


def create_scenarios(ref15: jnp.ndarray = None, names=CANDIDATE_NAMES) -> List[Scenario]:
    """Build the 4 real firmware-controller scenarios, using the reference
    module's own A/B/theta_true (see module docstring: imported, not
    re-derived, to avoid transcription drift)."""
    ref15 = hover_reference(jnp.array([0.0, 0.0, 1.0])) if ref15 is None else jnp.asarray(ref15)
    key = (tuple(np.asarray(ref15).tolist()), tuple(names))
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]

    ref_bank = {c.name: c for c in ref.get_bank(QPS_DT)}
    scenarios = []
    for name in names:
        cand = ref_bank[name]
        sys_ = FirmwareControllerSystem(name, ref15, cand.A, cand.B)
        emb = irx.natemb(sys_)
        theta = jnp.array(cand.theta_true, dtype=jnp.float32)
        scenarios.append(Scenario(name, emb, irx.Interval(lower=theta, upper=theta)))
    _BANK_CACHE[key] = scenarios
    return scenarios


def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray, p_ivl: irx.Interval) -> irx.Interval:
    """One discrete step via the natural embedding. Named `euler_step` for
    call-site parity with crazyflie_chain_controllers.py even though this is
    a genuine one-tick discrete map, not a forward-Euler approximation of a
    continuous ODE -- see module docstring."""
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    x_next_ut = emb_sys.f(_t, x_ut, u, p_ivl, refine=lambda z: z)
    return irx.ut2i(x_next_ut)


# ══════════════════════════════════════════════════════════════════════════════
# Observability: indices 0:12 (position, velocity, attitude, rates) are what a
# real Crazyflie's state estimator actually reports (unlike
# crazyflie_chain_controllers.py's flat-output chain, ALL of position through
# rates are plausible telemetry here, not just position/vel/acc/jerk).
# Indices 12:21 (integ, prev_cmd, accel_filt) are controller-internal memory
# -- no external observer sees a Crazyflie's own PID integrator or INDI
# filter state. Every discrimination signal below is computed on this
# observed projection, never on the hidden memory block.
# ══════════════════════════════════════════════════════════════════════════════

_BIAS_LIM = 0.3   # meters, per axis -- same box as crazyflie_chain_controllers.py


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(u, -_BIAS_LIM, _BIAS_LIM)


def observed_output(x_ivl: irx.Interval) -> irx.Interval:
    return irx.Interval(lower=x_ivl.lower[:12], upper=x_ivl.upper[:12])


def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    widths = jnp.maximum(
        jnp.minimum(ivl1.upper, ivl2.upper) - jnp.maximum(ivl1.lower, ivl2.lower),
        0.0,
    )
    return jnp.prod(widths)


def _output_overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    return _overlap_volume(observed_output(ivl1), observed_output(ivl2))


# ══════════════════════════════════════════════════════════════════════════════
# Propagation. Each scenario has its OWN emb_system (module docstring), so
# unlike crazyflie_chain_controllers.py there is no shared-trace
# vmap-over-stacked-params trick to apply -- just a plain Python loop over
# the (always exactly 4) scenarios / (at most 6) pairs. Fine here: every
# candidate's dynamics are linear/affine, so 4 (or 6, for pairs) separate
# JIT traces is cheap, unlike the nonlinear rigid-body systems elsewhere in
# this repo where retracing per scenario/pair was the whole compile-cost
# problem.
# ══════════════════════════════════════════════════════════════════════════════

def propagate_scenario(x0_ivl: irx.Interval, u_seq: jnp.ndarray, scenario: Scenario) -> irx.Interval:
    """Propagate x0_ivl through a sequence of spoof biases (T,3); returns the
    final state interval."""
    x_ivl = x0_ivl
    for k in range(u_seq.shape[0]):
        x_ivl = euler_step(scenario.emb_system, x_ivl, u_seq[k], scenario.p_interval)
    return x_ivl


def _propagate_all_scenarios(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                             scenarios: List[Scenario]) -> List[irx.Interval]:
    return [propagate_scenario(x0_ivl, u_seq, s) for s in scenarios]


def propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray, scenario: Scenario) -> irx.Interval:
    """Like propagate_scenario, but returns an Interval whose lower/upper
    have shape (T+1, 21) -- the full history, for separation_loss's
    min-over-time objective and for plotting."""
    T = u_seq.shape[0]
    lowers = [x0_ivl.lower]
    uppers = [x0_ivl.upper]
    x_ivl = x0_ivl
    for k in range(T):
        x_ivl = euler_step(scenario.emb_system, x_ivl, u_seq[k], scenario.p_interval)
        lowers.append(x_ivl.lower)
        uppers.append(x_ivl.upper)
    return irx.Interval(lower=jnp.stack(lowers), upper=jnp.stack(uppers))


def separation_loss(u_seq: jnp.ndarray, x0_ivl: irx.Interval, scenarios: List[Scenario]) -> jnp.ndarray:
    """Min over POST-STEP time (k=1..T, never k=0) of the sum of pairwise
    OBSERVED-OUTPUT overlaps (C(4,2)=6 pairs). Minimising this over the
    spoof-bias sequence `u_seq` maximises reachable-output separation across
    the four REAL controllers -- the discrimination signal. Subsumes the
    constant-bias case at T=1.

    k=0 (the shared prior x0_ivl, before ANY step/bias is applied) is
    deliberately EXCLUDED from the min: it's identical across every scenario
    by construction and completely bias-independent, so including it makes
    the whole objective degenerate -- verified empirically, optimizing with
    k=0 included returned "best loss" bit-for-bit equal to the zero-bias
    loss, because the min always just picked the untouched prior."""
    histories = [propagate_history(x0_ivl, u_seq, s) for s in scenarios]
    T1 = histories[0].lower.shape[0]
    n = len(scenarios)

    def overlap_at_k(k):
        total = jnp.array(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                ivl_i = irx.Interval(lower=histories[i].lower[k], upper=histories[i].upper[k])
                ivl_j = irx.Interval(lower=histories[j].lower[k], upper=histories[j].upper[k])
                total = total + _output_overlap_volume(ivl_i, ivl_j)
        return total

    overlaps = jnp.stack([overlap_at_k(k) for k in range(1, T1)])
    return jnp.min(overlaps)


class SeparatingInputOptimizer:
    """Gradient-descent optimizer for a controller-discriminating spoof-bias
    sequence over the 4 REAL firmware controllers."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.num_steps = num_steps

        from functools import partial
        _loss = partial(separation_loss, x0_ivl=x0_ivl, scenarios=scenarios)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def evaluate(self, u_seq: jnp.ndarray) -> Dict:
        histories = [propagate_history(self.x0_ivl, u_seq, s) for s in self.scenarios]
        n = len(self.scenarios)
        T1 = histories[0].lower.shape[0]
        final = [irx.Interval(lower=h.lower[-1], upper=h.upper[-1]) for h in histories]
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_output_overlap_volume(final[i], final[j]))
        volumes = {s.name: float(jnp.prod(observed_output(f).upper - observed_output(f).lower))
                  for s, f in zip(self.scenarios, final)}
        return {'state_intervals': final, 'pairwise_overlaps': overlaps, 'volumes': volumes,
               'histories': histories}


def optimize_parallel_gpu(opt: 'SeparatingInputOptimizer', num_restarts: int = 32,
                          learning_rate: float = 0.02, num_iters: int = 150, seed: int = 42):
    """Multi-start gradient descent for a discriminating spoof-bias sequence.

    Loops over restarts in plain Python (jax.lax.scan per restart over
    iterations), NOT jax.vmap over restarts -- found empirically that
    vmapping opt.grad_fn (which traces through natif's interval-embedded
    FirmwareControllerSystem.f) silently returns NaN gradients for SOME
    restarts that are perfectly finite when the exact same input is run
    individually (verified: restarts 9 and 28 of an identical seed-0 draw
    both broke only under vmap). This looks like a real vmap/natif
    interaction bug in immrax, not something fixable from here -- avoided
    rather than chased further."""
    key = jax.random.PRNGKey(seed)
    noise_scale = _BIAS_LIM * 0.5
    u0 = jax.random.normal(key, (num_restarts, opt.num_steps, 3)) * noise_scale

    @jax.jit
    def run_one(u_init):
        def scan_body(u, _i):
            g = opt.grad_fn(u)
            return _project_u(u - learning_rate * g), None
        u_final, _ = jax.lax.scan(scan_body, u_init, xs=jnp.arange(num_iters))
        return u_final

    u_finals = jnp.stack([run_one(u0[r]) for r in range(num_restarts)])
    losses = jnp.stack([opt.loss_fn(u_finals[r]) for r in range(num_restarts)])
    best_idx = jnp.argmin(losses)
    return u_finals[best_idx], losses[best_idx], u_finals, losses


# ══════════════════════════════════════════════════════════════════════════════
# Output-anticipating intersection refinement (design time) -- see
# crazyflie_chain_controllers.py's Section 5 docstring for the full
# rationale; identical principle here, just a plain double loop over pairs
# instead of vmap-over-stacked-pairs (module docstring: no shared trace to
# vmap across, and it's unnecessary given how cheap these systems are).
# ══════════════════════════════════════════════════════════════════════════════

def propagate_with_refinement(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                              scenarios: List[Scenario]) -> jnp.ndarray:
    """Multi-step propagation with per-pair OBSERVED-OUTPUT refinement.
    Returns min over steps of the per-step pairwise observed-output overlap
    sum (scalar)."""
    n = len(scenarios)
    T = u_seq.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    x_ivls = [x0_ivl for _ in scenarios]
    min_cost = jnp.array(np.inf)
    for k in range(T):
        # refine each pair's OBSERVABLE overlap before advancing
        refined = list(x_ivls)
        step_cost = jnp.array(0.0)
        for i, j in pairs:
            xi, xj = x_ivls[i], x_ivls[j]
            obs_lo_i, obs_hi_i = xi.lower[:12], xi.upper[:12]
            obs_lo_j, obs_hi_j = xj.lower[:12], xj.upper[:12]
            x_lo = jnp.maximum(obs_lo_i, obs_lo_j)
            x_hi = jnp.minimum(obs_hi_i, obs_hi_j)
            has_overlap = jnp.all(x_hi >= x_lo)

            new_lo_i = jnp.concatenate([jnp.where(has_overlap, x_lo, obs_lo_i), xi.lower[12:21]])
            new_hi_i = jnp.concatenate([jnp.where(has_overlap, x_hi, obs_hi_i), xi.upper[12:21]])
            new_lo_j = jnp.concatenate([jnp.where(has_overlap, x_lo, obs_lo_j), xj.lower[12:21]])
            new_hi_j = jnp.concatenate([jnp.where(has_overlap, x_hi, obs_hi_j), xj.upper[12:21]])
            refined[i] = irx.Interval(lower=new_lo_i, upper=new_hi_i)
            refined[j] = irx.Interval(lower=new_lo_j, upper=new_hi_j)

            xi_next = euler_step(scenarios[i].emb_system, refined[i], u_seq[k], scenarios[i].p_interval)
            xj_next = euler_step(scenarios[j].emb_system, refined[j], u_seq[k], scenarios[j].p_interval)
            pair_cost = jnp.where(has_overlap, _output_overlap_volume(xi_next, xj_next), 0.0)
            step_cost = step_cost + pair_cost

        # advance every scenario (not just pair members) using its own refined-if-touched interval
        new_x_ivls = []
        for idx, s in enumerate(scenarios):
            new_x_ivls.append(euler_step(s.emb_system, refined[idx], u_seq[k], s.p_interval))
        x_ivls = new_x_ivls
        min_cost = jnp.minimum(min_cost, step_cost)

    return min_cost


def refined_overlap_loss(u_seq: jnp.ndarray, x0_ivl: irx.Interval, scenarios: List[Scenario]) -> jnp.ndarray:
    return propagate_with_refinement(x0_ivl, u_seq, scenarios)


def optimize_refined_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], num_steps: int,
                         num_restarts: int = 24, learning_rate: float = 0.03,
                         num_iters: int = 150, seed: int = 42):
    """See optimize_parallel_gpu's docstring: plain Python loop over restarts,
    not jax.vmap, for the same verified vmap/natif NaN-gradient reason."""
    key = jax.random.PRNGKey(seed)
    noise_scale = _BIAS_LIM * 0.5
    u0 = jax.random.normal(key, (num_restarts, num_steps, 3)) * noise_scale

    def loss_fn(u_seq):
        return refined_overlap_loss(u_seq, x0_ivl, scenarios)
    grad_fn = jax.grad(loss_fn)

    @jax.jit
    def run_one(u_init):
        def scan_body(u, _i):
            g = grad_fn(u)
            return _project_u(u - learning_rate * g), None
        u_final, _ = jax.lax.scan(scan_body, u_init, xs=jnp.arange(num_iters))
        return u_final

    u_finals = jnp.stack([run_one(u0[r]) for r in range(num_restarts)])
    losses = jnp.stack([loss_fn(u_finals[r]) for r in range(num_restarts)])
    best_idx = jnp.argmin(losses)
    return u_finals[best_idx], losses[best_idx], u_finals, losses


# ══════════════════════════════════════════════════════════════════════════════
# React: online discrimination from real (or realistically simulated)
# observations -- see crazyflie_chain_controllers.py Section 6 for the full
# rationale (falsification = reachable-set analog of rq3_sme.py's "Theta is
# empty -> rejected"; w_bar = bounded measurement/model-mismatch budget,
# NOT an exact-point refinement, for the same float-precision reason
# documented there).
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_W_BAR = 1e-4


def simulate_true_trajectory(x0_point: jnp.ndarray, u_seq: jnp.ndarray, scenario: Scenario) -> jnp.ndarray:
    """Point (non-interval) rollout of ONE named controller, for generating
    a ground-truth observed trajectory in tests/demos."""
    f = scenario.emb_system.sys.f
    theta = scenario.p_interval.lower   # degenerate interval -> point value
    x = jnp.asarray(x0_point)
    traj = []
    for k in range(u_seq.shape[0]):
        x = f(jnp.zeros(()), x, u_seq[k], theta)
        traj.append(x[:12])
    return jnp.stack(traj)


def discriminate_controller(x0_ivl: irx.Interval, u_seq: jnp.ndarray, observed_traj: jnp.ndarray,
                            scenarios: List[Scenario], w_bar: float = _DEFAULT_W_BAR) -> Dict:
    """Run each of the 4 REAL controllers forward under the injected bias
    sequence, falsifying any whose predicted observed-output interval fails
    to contain the true observation, and refining survivors' observable
    dims to a w_bar-wide box around the truth each step they remain
    consistent."""
    T = u_seq.shape[0]
    falsified = {}
    fail_step = {}
    contained_history = {}
    for s in scenarios:
        x_ivl = x0_ivl
        f_falsified = False
        f_step = -1
        hist = []
        for k in range(T):
            x_next = euler_step(s.emb_system, x_ivl, u_seq[k], s.p_interval)
            y_pred = observed_output(x_next)
            y_true = observed_traj[k]
            contained = bool(jnp.all((y_true >= y_pred.lower - w_bar) & (y_true <= y_pred.upper + w_bar)))
            hist.append(contained)
            if not contained and not f_falsified:
                f_falsified, f_step = True, k
            obs_lo = y_true - w_bar if contained else y_pred.lower
            obs_hi = y_true + w_bar if contained else y_pred.upper
            x_ivl = irx.Interval(lower=jnp.concatenate([obs_lo, x_next.lower[12:21]]),
                                 upper=jnp.concatenate([obs_hi, x_next.upper[12:21]]))
        falsified[s.name] = f_falsified
        fail_step[s.name] = f_step
        contained_history[s.name] = np.array(hist)

    return {
        'falsified': falsified,
        'fail_step': fail_step,
        'contained_history': contained_history,
        'survivors': [s.name for s in scenarios if not falsified[s.name]],
    }
