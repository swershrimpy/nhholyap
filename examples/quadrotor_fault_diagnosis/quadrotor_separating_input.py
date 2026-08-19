"""
12-State Rigid-Body Quadrotor -- Active Fault Diagnosis with immrax
=====================================================================
Reachable-set-based active fault diagnosis for the standard body-frame
Newton-Euler quadrotor model, using immrax for interval arithmetic and
natural-embedding propagation. See PLAN.md in this directory for the full
dynamics derivation, the hover-equilibrium sanity check, and the design
decisions from the clarification round this module was built from.

System
------
  State   x = [x, y, z, phi, theta, psi, u, v, w, p, q, r]   (12 states)
            position (inertial) / Euler angles (roll,pitch,yaw) /
            body-frame linear velocity / body-frame angular velocity
  Control u = [U1, U2, U3, U4]   thrust (N) + roll/pitch/yaw moment (N*m) --
            already-mixed "virtual" actuator commands, no rotor-speed
            mixing matrix modeled.
  Params  p = alpha = [a1,a2,a3,a4]   per-channel actuator-fault authority,
            applied as u_eff = alpha * u (elementwise), ai=1 -> nominal.

  Dynamics: standard body-frame Newton-Euler quadrotor model, ZYX Euler
  angles. Physical parameters (m, g, Ixx, Iyy, Izz) are standard literature
  defaults (NOT fetched from the cited thesis PDF -- see PLAN.md "Decisions"
  for the exact values and why).

  KNOWN SINGULARITY (documented, not runtime-guarded): theta_dot/psi_dot
  blow up at theta=+-90 deg (Euler-angle gimbal lock). The default control
  box and demo/test initial intervals are chosen to keep theta well under
  that range over the horizons used here -- see PLAN.md Sec 1.

Fault scenarios (5 total, fixed -- control dimension is fixed at 4, so
scenario count is not configurable the way nonlinear_chain's is)
--------------------------------------------------------------------
  Nominal          alpha = [1,1,1,1]
  ActuatorFault_1  alpha[0] in [lo,hi], others 1   (thrust)
  ActuatorFault_2  alpha[1] in [lo,hi], others 1   (roll moment)
  ActuatorFault_3  alpha[2] in [lo,hi], others 1   (pitch moment)
  ActuatorFault_4  alpha[3] in [lo,hi], others 1   (yaw moment)

No sensor-fault layer (per the prompt): separation loss operates directly
on the propagated 12-state interval, not on a beta/xi-faulted observation
like integrator_chain/nonlinear_chain -- there is no observed_output /
_invert_observation here, only the raw state.

Three algorithmic layers (mirroring nonlinear_chain_separating_input.py /
admire_separating_input.py):
  1. Single-step separating input (Section 3)
  2. Multistep "unrefined"/"uninformed" separating input sequence (Section 4)
  3. Multistep intersection-refinement separating input sequence (Section 5)
     -- simpler than nonlinear_chain's version: with no sensor fault, the
     observation map is the identity, so refining a scenario pair's state
     estimate from their observed-output intersection collapses to a
     direct STATE-interval intersection (both scenarios refine to the SAME
     tightened state interval -- see PLAN.md Sec 3 for why this is the
     correct simplification, not an approximation).

Performance note: this system's trig-heavy embedding -- specifically
tan(theta) and 1/cos(theta) in the Euler-angle kinematics, which require
sign/pole case-splitting for a sound interval extension -- makes gradient
compile time grow steeply with the number of unrolled Euler steps (measured
single-restart: num_steps=1 ~2s, num_steps=5 ~56s/4.3GB). Keep num_steps
(single-step layer) and num_steps (refinement layer) small in practice;
num_segments (multistep-unrefined layer) is cheap to grow since segments
are scanned, not unrolled. See PLAN.md's "Compile-cost finding" and
propagate_with_refinement's docstring for the full story, including the
vmap-over-pairs fix that made the refinement layer usable at all.
"""

import sys
import time
import resource
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Optional, Dict

# File is at examples/quadrotor_fault_diagnosis/<name>.py -> parents[1] = examples/
_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Physical parameter defaults -- REAL Crazyflie 2.x values (updated from the
# original generic-literature-quadrotor defaults, m=0.468kg/I~1e-3, so this
# module's separating controllers can be validated against QPS's real
# rigid-body Crazyflie plant (crazyflie_12d.py's CrazyflieSystem, itself
# verified bit-for-bit against QPS's own forward_model()) -- see that
# module's docstring for the side-by-side parameter table this is sourced
# from (quadcopter_model.py:39-41 in QPS). g unchanged (mass-independent).
# ══════════════════════════════════════════════════════════════════════════════
_M = 0.03589     # mass, kg  (35.89 g -- QPS quadcopter_model.py)
_G = 9.81        # gravity, m/s^2
_IXX = 2.3951e-5  # roll-axis moment of inertia, kg*m^2  (QPS)
_IYY = 2.3951e-5  # pitch-axis moment of inertia, kg*m^2  (QPS)
_IZZ = 3.2346e-5  # yaw-axis moment of inertia, kg*m^2  (QPS)

_HOVER_THRUST = _M * _G   # ~0.352 N

# Control input box: u1 (thrust) centered on hover thrust, u2/u3/u4 (moments)
# a small symmetric range -- see PLAN.md Sec 3 for the reasoning (keeps
# attitude-rate excursions modest over the short horizons used here, well
# clear of the theta=+-90 deg gimbal-lock singularity). The moment bound
# itself was always "a tunable default, not a physical spec" (PLAN.md); it
# is RE-DERIVED here, not just copied, to preserve that same design intent
# under the real Crazyflie's ~200x smaller inertia -- the original 0.02 N*m
# bound implied a modest ~4.1/2.3 rad/s^2 (roll-pitch/yaw) peak angular
# acceleration against the OLD literature inertia; holding that SAME
# angular-acceleration bound and re-multiplying by the real Crazyflie's
# inertia gives the values below (an unchanged 0.02 N*m bound would instead
# imply peak angular accelerations of ~800+ rad/s^2 against the real,
# much smaller inertia -- wildly outside "modest," and would blow past the
# gimbal-lock-avoidance assumption almost immediately).
_U_LO = jnp.array([0.5 * _HOVER_THRUST, -9.8645e-5, -9.8645e-5, -7.3505e-5])
_U_HI = jnp.array([1.5 * _HOVER_THRUST, 9.8645e-5, 9.8645e-5, 7.3505e-5])

# Multi-start GD restart noise, per channel -- 10% of each channel's box
# half-width, same convention as the thrust channel's existing
# "0.1 * _HOVER_THRUST". MUST be re-derived alongside _U_LO/_U_HI, not left
# at the old moment bound's hardcoded 0.005: that value is ~50-70x LARGER
# than the entire new moment box (~2e-4 wide), so every restart's initial
# moment component would land far outside the box and get clipped to the
# same edge by _project_u, collapsing restart diversity to nothing instead
# of spreading restarts across the feasible region.
_U_NOISE_SCALE = jnp.array([0.1 * _HOVER_THRUST, 0.1 * 9.8645e-5, 0.1 * 9.8645e-5, 0.1 * 7.3505e-5])

# Loops (Euler steps, GD iterations) with a static length <= this are fully
# unrolled into straight-line code instead of lax.fori_loop/scan -- see
# `_run_unrolled_or_loop` below. Ported unchanged from nonlinear_chain.
_UNROLL_THRESHOLD = 64


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    """Project a control input (or batch) onto the feasible box.

    Works for any leading batch dimensions -- _U_LO/_U_HI (shape (4,))
    broadcast against the last axis.
    """
    return jnp.clip(u, _U_LO, _U_HI)


def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    """Branchless pairwise axis-aligned-box overlap volume. Ported verbatim
    from nonlinear_chain_separating_input.py (fully generic over dimension).

    Caveat (see RESULTS.md "Important caveat found while building this
    experiment"): this is a PRODUCT of 12 per-dimension overlap widths. For
    a short-horizon, near-hover state box, every dimension's ABSOLUTE width
    is small (~0.02-0.06 in this module's units) regardless of how much the
    two boxes actually overlap RELATIVE to their own size -- so the product
    can read as ~1e-19 (numerically indistinguishable from "separated") even
    when every single dimension still overlaps 60-90%. Driving this loss to
    ~0 via gradient descent is therefore not reliable evidence of true
    disjointness, and its gradient vanishes in exactly the regime where
    genuine separation still needs to be found (many small-but-nonzero
    per-dimension widths whose product is already tiny). `_soft_separation_loss`
    below is a proxy built to avoid both problems; kept alongside (not a
    replacement) since existing callers/tests pin this exact function."""
    widths = jnp.maximum(
        jnp.minimum(ivl1.upper, ivl2.upper) - jnp.maximum(ivl1.lower, ivl2.lower),
        0.0,
    )
    return jnp.prod(widths)


def _signed_separation_margin(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    """Per-dimension signed separation margin between two axis-aligned boxes.
    margin[d] > 0 means the boxes are DISJOINT along dimension d by that
    much; margin[d] < 0 means dimension d still overlaps by |margin[d]|.

    This is the quantity that actually determines disjointness: two
    axis-aligned boxes are disjoint iff AT LEAST ONE dimension is disjoint
    (max_d margin[d] >= 0), regardless of how the other dimensions overlap.
    `_overlap_volume` instead multiplies overlap widths across ALL 12
    dimensions, so it can look near-zero without any single dimension ever
    reaching genuine separation -- exactly the failure mode found when
    checking solve_and_plot.py's boxes by hand (ActuatorFault_2/3/4 overlap
    Nominal 60-88% per-dimension despite ~1e-19 volume)."""
    return jnp.maximum(ivl2.lower - ivl1.upper, ivl1.lower - ivl2.upper)


def _soft_separation_loss(ivl1: irx.Interval, ivl2: irx.Interval, tau: float = 0.01) -> jnp.ndarray:
    """Smooth proxy for 'these two boxes are not yet disjoint', built from
    the per-dimension margin instead of a volume product -- a drop-in
    alternative to `_overlap_volume` wherever that function is used as a
    per-pair separation cost (same signature, same "0 means separated"
    convention), intended to alleviate the vanishing-gradient/false-early-
    convergence problem documented on `_overlap_volume`.

    True disjointness only needs max_d margin[d] >= 0. A hard max routes
    gradient through a single dimension and gives exactly zero gradient to
    every other dimension -- including near-competitive runners-up -- which
    is its own vanishing-gradient trap early in optimization, before any
    one dimension is close to separating. `tau` softens the hard max into a
    logsumexp so every dimension with a competitive margin contributes
    gradient, while still converging to max_d margin[d] as tau -> 0.

    Bias correction (found empirically -- see git history/RESULTS.md "second
    caveat"): plain `tau*logsumexp(margins/tau)` is an UPPER bound on the
    true max, `max(margins) <= logsumexp*tau <= max(margins) + tau*log(D)`
    for D dimensions. At D=12, tau=0.01 that slack is `tau*log(12)~=0.025` --
    comparable to this module's actual margins (~0.02-0.03), so the
    UNCORRECTED loss hit exactly 0 (falsely claiming separation) while the
    true hard-max margin was still negative by about that much (confirmed:
    an optimizer run using the uncorrected version converged to loss=0 at
    every checked step while every pair was still genuinely overlapping by
    ~0.018-0.026 in every dimension). Subtracting `tau*log(D)` makes this a
    LOWER bound on the true max instead (`logsumexp*tau - tau*log(D) <=
    max(margins)`), so loss=0 here is a SAFE (never falsely-optimistic)
    certificate of true disjointness -- clipped at 0 once genuinely
    separated, matching `_overlap_volume`'s own convention, so optimization
    pressure stops there rather than pushing boxes further apart than
    necessary."""
    margins = _signed_separation_margin(ivl1, ivl2)
    n_dims = margins.shape[-1]
    soft_best_margin = tau * (jax.scipy.special.logsumexp(margins / tau) - jnp.log(n_dims))
    return jnp.maximum(-soft_best_margin, 0.0)


def _run_unrolled_or_loop(step_fn, init, n: int, unroll_threshold: int = _UNROLL_THRESHOLD):
    """Apply step_fn (carry, i) -> carry exactly n times, starting from init.
    Ported verbatim from nonlinear_chain_separating_input.py."""
    if n <= unroll_threshold:
        def body(carry):
            for i in range(n):
                carry = step_fn(carry, i)
            return carry
        return jax.checkpoint(body)(init)
    else:
        @jax.checkpoint
        def scan_body(carry, i):
            return step_fn(carry, i), None
        carry, _ = jax.lax.scan(scan_body, init, xs=jnp.arange(n))
        return carry


def _run_unrolled_or_loop_nocheckpoint(step_fn, init, n: int):
    """Advance `init` through step_fn exactly n times via jax.lax.scan.
    Ported verbatim from nonlinear_chain_separating_input.py (see its
    docstring for why the outer GD loop deliberately doesn't unroll)."""
    def body(carry, i):
        return step_fn(carry, i), None
    carry, _ = jax.lax.scan(body, init, xs=jnp.arange(n))
    return carry


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Timing / Memory Helper
# ══════════════════════════════════════════════════════════════════════════════

def _memory_snapshot() -> Dict[str, float]:
    """Best-effort memory snapshot: GPU device stats if available, else CPU RSS."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            return {k: float(v) for k, v in stats.items() if 'bytes' in k}
    except Exception:
        pass
    return {'ru_maxrss_kb': float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def time_jit(fn, *args, **kwargs):
    """JIT-compile `fn` and separately measure compile time vs. run time."""
    jitted_fn = jax.jit(fn)

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted_fn(*args, **kwargs))
    compile_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted_fn(*args, **kwargs))
    run_time_s = time.perf_counter() - t0

    del out
    return jitted_fn, compile_time_s, run_time_s, _memory_snapshot()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  System Definition
# ══════════════════════════════════════════════════════════════════════════════

class QuadrotorSystem(irx.System):
    """12-state rigid-body quadrotor with per-channel actuator-fault authority.

    State   x = [x, y, z, phi, theta, psi, u, v, w, p, q, r]
    Control u = [U1, U2, U3, U4]   (thrust N, roll/pitch/yaw moment N*m)
    Params  p = alpha               (shape (4,), alpha_i=1 -> nominal)

    Body-frame Newton-Euler quadrotor model, ZYX Euler angles -- see
    PLAN.md Sec 1 for the full derivation and the hover-equilibrium check.
    """

    def __init__(self, m: float = _M, g: float = _G,
                 Ixx: float = _IXX, Iyy: float = _IYY, Izz: float = _IZZ):
        self.evolution = 'continuous'
        self.xlen = 12
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

    def f(self, t, x, u, p):
        alpha = p
        U1, U2, U3, U4 = alpha * u

        phi, theta, psi = x[3], x[4], x[5]
        ub, vb, wb = x[6], x[7], x[8]
        pr, qr, rr = x[9], x[10], x[11]

        sphi, cphi = jnp.sin(phi), jnp.cos(phi)
        stheta, ctheta, ttheta = jnp.sin(theta), jnp.cos(theta), jnp.tan(theta)
        spsi, cpsi = jnp.sin(psi), jnp.cos(psi)

        # position kinematics: inertial velocity = R_body->inertial @ [u,v,w]
        xdot = ctheta * cpsi * ub + (sphi * stheta * cpsi - cphi * spsi) * vb \
            + (cphi * stheta * cpsi + sphi * spsi) * wb
        ydot = ctheta * spsi * ub + (sphi * stheta * spsi + cphi * cpsi) * vb \
            + (cphi * stheta * spsi - sphi * cpsi) * wb
        zdot = -stheta * ub + sphi * ctheta * vb + cphi * ctheta * wb

        # Euler-angle kinematics
        phidot = pr + sphi * ttheta * qr + cphi * ttheta * rr
        thetadot = cphi * qr - sphi * rr
        psidot = (sphi / ctheta) * qr + (cphi / ctheta) * rr

        # body-frame translational dynamics (with Coriolis coupling omega x V)
        ubdot = self.g * stheta - qr * wb + rr * vb
        vbdot = -self.g * sphi * ctheta - rr * ub + pr * wb
        wbdot = U1 / self.m - self.g * cphi * ctheta - pr * vb + qr * ub

        # rotational dynamics (Euler's equations, no rotor gyroscopic term --
        # see module docstring / PLAN.md for why)
        prdot = ((self.Iyy - self.Izz) / self.Ixx) * qr * rr + U2 / self.Ixx
        qrdot = ((self.Izz - self.Ixx) / self.Iyy) * pr * rr + U3 / self.Iyy
        rrdot = ((self.Ixx - self.Iyy) / self.Izz) * pr * qr + U4 / self.Izz

        return jnp.array([
            xdot, ydot, zdot, phidot, thetadot, psidot,
            ubdot, vbdot, wbdot, prdot, qrdot, rrdot,
        ])


# Module-level cache: one QuadrotorSystem + one irx.natemb(...) per distinct
# (m, g, Ixx, Iyy, Izz) tuple.
_EMB_CACHE: Dict[Tuple[float, float, float, float, float], Tuple[QuadrotorSystem, object]] = {}


def get_system_and_embedding(m: float = _M, g: float = _G, Ixx: float = _IXX,
                             Iyy: float = _IYY, Izz: float = _IZZ) -> Tuple[QuadrotorSystem, object]:
    """Return (system, natural_embedding) for the given physical parameters, cached."""
    key = (m, g, Ixx, Iyy, Izz)
    if key not in _EMB_CACHE:
        sys_ = QuadrotorSystem(m, g, Ixx, Iyy, Izz)
        emb = irx.natemb(sys_)
        _EMB_CACHE[key] = (sys_, emb)
    return _EMB_CACHE[key]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Fault Scenarios (no sensor-fault layer -- see module docstring)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One fault scenario: shared dynamics structure, differing only in the
    actuator-effectiveness interval alpha. No beta/xi -- there is no
    sensor-fault model in this module (see PLAN.md Sec 2)."""
    name: str
    emb_system: object
    p_interval: irx.Interval   # alpha interval, shape (4,)


def create_scenarios(
    m: float = _M, g: float = _G, Ixx: float = _IXX, Iyy: float = _IYY, Izz: float = _IZZ,
    alpha_lo: float = 0.5, alpha_hi: float = 0.9,
) -> List[Scenario]:
    """Return the 5 fault scenarios used throughout this module: Nominal +
    one ActuatorFault_i per control channel (thrust, roll, pitch, yaw).

    Parameters
    ----------
    m, g, Ixx, Iyy, Izz : physical parameters (see module-level defaults)
    alpha_lo / alpha_hi  : range of the per-channel actuator-fault authority
    """
    _, emb = get_system_and_embedding(m, g, Ixx, Iyy, Izz)

    ones_4 = jnp.ones(4)
    channel_names = ["ActuatorFault_1", "ActuatorFault_2", "ActuatorFault_3", "ActuatorFault_4"]

    scenarios = [
        Scenario("Nominal", emb, irx.Interval(lower=ones_4, upper=ones_4)),
    ]
    for i, name in enumerate(channel_names):
        alpha_lo_vec = ones_4.at[i].set(alpha_lo)
        alpha_hi_vec = ones_4.at[i].set(alpha_hi)
        scenarios.append(Scenario(name, emb, irx.Interval(lower=alpha_lo_vec, upper=alpha_hi_vec)))

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Single-Step Path
# ══════════════════════════════════════════════════════════════════════════════

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    """One forward-Euler interval step via the natural embedding.

    Note: t must be a JAX array (not a Python scalar) so that
    eqx.filter_make_jaxpr traces it as an abstract input and the jaxpr
    invar count matches the natif_jaxpr arg count.
    """
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


def _propagate_by_params(x0_ivl: irx.Interval, u: jnp.ndarray, emb_sys,
                         p_ivl: irx.Interval, dt: float, num_steps: int) -> irx.Interval:
    """Propagate *num_steps* Euler steps given raw (emb_sys, p_ivl) rather than
    a Scenario -- the vmappable primitive behind propagate_scenario and the
    scenario-batched loss functions."""
    def step(x_carry, _i):
        return euler_step(emb_sys, x_carry, u, p_ivl, dt)
    return _run_unrolled_or_loop(step, x0_ivl, num_steps)


def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    """Propagate the initial interval *num_steps* Euler steps under constant u."""
    return _propagate_by_params(x0_ivl, u, scenario.emb_system, scenario.p_interval, dt, num_steps)


def _propagate_all_scenarios(x0_ivl: irx.Interval, u: jnp.ndarray,
                             scenarios: List[Scenario], dt: float, num_steps: int) -> List[irx.Interval]:
    """Propagate every scenario from the same x0_ivl/u and return their final
    state intervals -- a single vmap over stacked p_intervals since all
    scenarios share one emb_system."""
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x_final_batch = jax.vmap(lambda p: _propagate_by_params(x0_ivl, u, emb_sys, p, dt, num_steps))(p_batch)
    return [
        irx.Interval(lower=x_final_batch.lower[i], upper=x_final_batch.upper[i])
        for i in range(len(scenarios))
    ]


def separation_loss(u: jnp.ndarray,
                    x0_ivl: irx.Interval,
                    scenarios: List[Scenario],
                    dt: float,
                    num_steps: int) -> jnp.ndarray:
    """Sum of pairwise state-interval overlaps (C(5,2)=10 pairs).

    Minimising this loss maximises the separation of the scenarios'
    reachable sets in state space (no sensor fault -> no observed-output
    layer, unlike integrator_chain/nonlinear_chain).
    """
    x_ivls = _propagate_all_scenarios(x0_ivl, u, scenarios, dt, num_steps)
    n = len(x_ivls)
    total = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            total = total + _overlap_volume(x_ivls[i], x_ivls[j])
    return total


class SeparatingInputOptimizer:
    """Gradient-descent optimizer for a fault-separating constant control input."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.num_steps = num_steps

        _loss = partial(separation_loss, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def optimize(self, u_init: Optional[jnp.ndarray] = None,
                 learning_rate: float = 0.01, num_iters: int = 150,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        if u_init is None:
            u_init = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
        u = u_init
        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)
            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = float(self.loss_fn(u))
                print(f"  Iter {i:4d}  loss={loss:.6f}  u={np.array(u).round(4)}  |g|={float(jnp.linalg.norm(g)):.4f}")
        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        x_ivls = _propagate_all_scenarios(self.x0_ivl, u, self.scenarios, self.dt, self.num_steps)
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_overlap_volume(x_ivls[i], x_ivls[j]))
        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, x_ivls)
        }
        return {'state_intervals': x_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes}


def optimize_parallel_gpu(opt: 'SeparatingInputOptimizer', num_restarts: int = 100,
                          learning_rate: float = 0.01, num_iters: int = 150,
                          seed: int = 42):
    """GPU-parallel multi-start gradient descent for a constant separating input."""
    key = jax.random.PRNGKey(seed)
    u_init = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    noise_scale = _U_NOISE_SCALE
    u0 = jax.random.normal(key, (num_restarts, 4)) * noise_scale + u_init

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u, _i):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def optimize_parallel_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                dt: float, num_steps: int, num_restarts: int = 100,
                                learning_rate: float = 0.01, num_iters: int = 150,
                                seed: int = 42):
    return optimize_parallel_gpu(
        SeparatingInputOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt, num_steps=num_steps),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Multistep ("Unrefined" / "Uninformed") Path
# ══════════════════════════════════════════════════════════════════════════════

def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate and record the state interval at the end of every segment.
    Returns an irx.Interval whose lower/upper have shape (num_segments, 12).

    Compile-cost note: the within-segment step loop uses jax.lax.scan, NOT
    a Python-unrolled loop the way nonlinear_chain_separating_input.py's
    version does (`_plain_unroll`). That's fine there since dynamics are
    cheap to unroll; here, this system's trig-heavy embedding makes even a
    handful of unrolled steps expensive to compile (measured: 5 unrolled
    steps ~56s/4.3GB elsewhere in this module -- see module docstring
    "Performance note"), so `steps_per_segment` -- e.g. 50 steps for a
    0.5s segment at dt=0.01 -- MUST be scanned, not unrolled, to stay
    compile-tractable. Segments themselves were already scanned (the outer
    jax.lax.scan below); this makes the whole function's compile cost
    ~independent of both steps_per_segment and num_segments.
    """
    def segment(x_ivl, u_k):
        def step(x_carry, _):
            return euler_step(emb_sys, x_carry, u_k, p_ivl, dt), None
        x_end, _ = jax.lax.scan(step, x_ivl, xs=None, length=steps_per_segment)
        return x_end, x_end

    _, x_hist = jax.lax.scan(jax.checkpoint(segment), x0_ivl, u_seq)
    return x_hist


def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                 scenario: Scenario, dt: float,
                                 steps_per_segment: int) -> irx.Interval:
    """Propagate x0_ivl through a sequence of control inputs; return final interval."""
    x_hist = _propagate_history(
        x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment
    )
    return irx.Interval(lower=x_hist.lower[-1], upper=x_hist.upper[-1])


def separation_loss_multistep(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                              scenarios: List[Scenario], dt: float,
                              steps_per_segment: int,
                              margin_tau: Optional[float] = None) -> jnp.ndarray:
    """Min over segments of the pairwise state-interval overlap sum.

    `margin_tau`: if None (default), the pairwise cost is the volume-product
    metric (`_overlap_volume`'s formula, inlined/vectorized below) --
    preserves prior behavior/tests exactly. If set to a float, uses the
    vectorized form of `_soft_separation_loss` (temperature `margin_tau`)
    instead: targets max-over-dimensions signed margin rather than a
    12-dimensional product, which is the metric that actually determines
    box disjointness and doesn't vanish just because the boxes are small in
    absolute terms -- see `_soft_separation_loss`'s docstring for why this
    matters at this module's short, near-hover horizons.

    Compile-cost note: the pairwise-overlap-at-every-segment computation is
    fully vectorized (gather over precomputed pair indices + jax.vmap over
    the segment axis), NOT a Python double loop over (segments x pairs) the
    way nonlinear_chain_separating_input.py's version is. That Python-loop
    form is cheap for nonlinear_chain's polynomial dynamics but, combined
    with this system's expensive-to-differentiate trig terms, made even
    num_segments=10 fail to compile in 90s here. Vectorizing brings
    num_segments=100 (a 1s horizon at dt=0.01) down to compile in line with
    num_segments=5 -- see module docstring "Performance note".
    """
    num_segments = u_seq.shape[0]
    n = len(scenarios)
    emb_sys = scenarios[0].emb_system

    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def prop_one(p_ivl_single):
        return _propagate_history(x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment)

    # x_hist_batch: Interval with lower/upper shape (n, num_segments, 12)
    x_hist_batch = jax.vmap(prop_one)(p_batch)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    i_idx = jnp.array([i for i, j in pairs])
    j_idx = jnp.array([j for i, j in pairs])

    def overlap_at_k(k):
        # Gather every pair's (i, j) state interval at segment k at once
        # (shape (n_pairs, 12)) instead of a Python loop over pairs.
        lower_i, upper_i = x_hist_batch.lower[i_idx, k], x_hist_batch.upper[i_idx, k]
        lower_j, upper_j = x_hist_batch.lower[j_idx, k], x_hist_batch.upper[j_idx, k]
        if margin_tau is None:
            widths = jnp.maximum(jnp.minimum(upper_i, upper_j) - jnp.maximum(lower_i, lower_j), 0.0)
            return jnp.sum(jnp.prod(widths, axis=-1))
        else:
            # log(n_dims) bias correction -- see _soft_separation_loss's
            # docstring "Bias correction": uncorrected logsumexp is an
            # upper bound on the true max, by up to margin_tau*log(12),
            # which is large enough at this module's margin scale to make
            # the loss falsely hit 0 before any dimension is truly disjoint.
            margins = jnp.maximum(lower_j - upper_i, lower_i - upper_j)   # (n_pairs, 12)
            n_dims = margins.shape[-1]
            soft_best_margin = margin_tau * (jax.scipy.special.logsumexp(
                margins / margin_tau, axis=-1) - jnp.log(n_dims))   # (n_pairs,)
            return jnp.sum(jnp.maximum(-soft_best_margin, 0.0))

    # vmap over segments instead of a Python loop over range(num_segments).
    segment_overlaps = jax.vmap(overlap_at_k)(jnp.arange(num_segments))
    return jnp.min(segment_overlaps)


class MultistepSequenceOptimizer:
    """Container for multistep loss/grad callables and sequence shape."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, steps_per_segment: int, num_segments: int,
                 margin_tau: Optional[float] = None):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments

        _loss = partial(separation_loss_multistep, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, steps_per_segment=steps_per_segment, margin_tau=margin_tau)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))


def optimize_multistep_gpu(opt: 'MultistepSequenceOptimizer', num_restarts: int = 100,
                           learning_rate: float = 0.01, num_iters: int = 150, seed: int = 42,
                           normalize_grad: bool = False):
    """GPU-parallel multi-start optimization for control sequences.

    `normalize_grad`: if True, each GD step moves `learning_rate` in the
    NORMALIZED gradient direction (`g / ||g||`) instead of `learning_rate *
    g`. Needed at longer horizons (large `opt.num_segments`) with
    `margin_tau` set: gradient norm through `separation_loss_multistep`'s
    unrolled Euler propagation compounds across segments (a standard
    RNN-like effect), reaching O(1e3) at num_segments=20 in this module's
    dynamics, while the control box itself is only O(1e-4) wide in the
    moment channels -- an unnormalized step at any learning_rate that isn't
    absurdly small either overshoots the box every iteration (clipped back
    to the same boundary point, permanently stuck -- confirmed: this is
    exactly what happened when this feature was added, `learning_rate` in
    [3e-6, 0.02] with the plain update all converged to the SAME stuck
    non-zero loss after a handful of iterations) or moves imperceptibly
    slowly. Normalizing decouples step SIZE from the gradient's (highly
    horizon- and iterate-dependent) magnitude; `learning_rate` in this mode
    should be set near the control box's own width, not a generic GD rate."""
    key = jax.random.PRNGKey(seed)
    u_init = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    noise_scale = _U_NOISE_SCALE
    u0 = jax.random.normal(key, (num_restarts, opt.num_segments, 4)) * noise_scale + u_init

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u_batch, _i):
        g = batched_grad(u_batch)
        if normalize_grad:
            flat = g.reshape(g.shape[0], -1)
            g = g / (jnp.linalg.norm(flat, axis=-1).reshape(-1, 1, 1) + 1e-12)
        return _project_u(u_batch - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def optimize_multistep_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                 dt: float, steps_per_segment: int, num_segments: int,
                                 num_restarts: int = 100, learning_rate: float = 0.01,
                                 num_iters: int = 150, seed: int = 42,
                                 margin_tau: Optional[float] = None,
                                 normalize_grad: bool = False):
    return optimize_multistep_gpu(
        MultistepSequenceOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
                                   steps_per_segment=steps_per_segment, num_segments=num_segments,
                                   margin_tau=margin_tau),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
        normalize_grad=normalize_grad,
    )


def optimize_multistep(scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float,
                       steps_per_segment: int, num_segments: int,
                       learning_rate: float = 0.01, num_iters: int = 300,
                       num_restarts: int = 100, verbose: bool = False,
                       seed: int = 42) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent over a sequence of control inputs (unrefined loss)."""
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    ms_opt = MultistepSequenceOptimizer(
        scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
        steps_per_segment=steps_per_segment, num_segments=num_segments,
    )

    _t0 = time.perf_counter()
    u_seq, loss_opt_jax, _, final_losses = optimize_multistep_gpu(
        opt=ms_opt, num_restarts=num_restarts, learning_rate=learning_rate,
        num_iters=num_iters, seed=seed,
    )
    elapsed = time.perf_counter() - _t0

    best_idx = int(jnp.argmin(final_losses))
    loss_opt = float(loss_opt_jax)

    if verbose:
        mean_loss = float(jnp.mean(final_losses))
        print(f"Multistep GPU multistart complete: best_loss={loss_opt:.6f}  "
              f"mean_final_loss={mean_loss:.6f}  best_restart={best_idx+1}/{num_restarts}")

    x_ivls = [
        propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment)
        for s in scenarios
    ]
    n = len(scenarios)
    overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            overlaps[f"{scenarios[i].name} vs {scenarios[j].name}"] = float(_overlap_volume(x_ivls[i], x_ivls[j]))
    volumes = {s.name: float(jnp.prod(iv.upper - iv.lower)) for s, iv in zip(scenarios, x_ivls)}
    stats = {
        'state_intervals': x_ivls,
        'pairwise_overlaps': overlaps,
        'volumes': volumes,
        'optimization_time_s': elapsed,
        'best_restart_idx': best_idx,
        'all_restart_losses': np.array(final_losses),
        'num_restarts': num_restarts,
    }
    return u_seq, loss_opt, stats


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Intersection-Refinement Path
# ══════════════════════════════════════════════════════════════════════════════

def propagate_with_refinement(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                              scenarios: List[Scenario], dt: float,
                              num_steps: int = 2,
                              pair_cost_fn=_overlap_volume) -> jnp.ndarray:
    """Multi-step propagation with per-pair state-interval refinement.

    `pair_cost_fn(ivl1, ivl2) -> scalar` computes the per-pair separation
    cost at each step (default `_overlap_volume`, preserving prior
    behavior/tests exactly). Pass `_soft_separation_loss` to optimize the
    margin-based proxy instead -- see that function's docstring for why:
    `_overlap_volume`'s product-of-widths can read as ~0 (and its gradient
    vanish) while every dimension still substantially overlaps, once the
    boxes themselves are small in absolute terms (as they are at this
    module's short, near-hover horizons).

    No sensor fault in this module -> the observation map is the identity,
    so refining a scenario pair's estimate from their intersection collapses
    to a direct STATE-interval intersection (both scenarios refine to the
    SAME tightened interval when they currently overlap; carried forward
    unrefined when they don't) -- see PLAN.md Sec 3.

    Compile-cost note (departs from nonlinear_chain_separating_input.py's
    version here): that module's per-pair refinement loop is a plain Python
    loop over pairs, tracing one euler_step call per (scenario_i,
    scenario_j) branch -- cheap there because its dynamics are polynomial.
    This system's trig-heavy embedding (tan(theta), 1/cos(theta)) makes each
    traced euler_step call expensive enough that a `len(pairs)*2`-call
    Python loop doesn't finish compiling in any reasonable time (measured:
    still compiling after 90s at the minimum useful num_steps=2, for only
    5 scenarios / 10 pairs). Since every scenario shares one emb_system
    (only p_interval differs), the fix mirrors the vmap-over-scenarios trick
    already used by `_propagate_all_scenarios`/`separation_loss_multistep`:
    stack each pair's (p_i, p_j) and vmap euler_step over the pair axis, so
    XLA traces the per-pair refinement step ONCE and reuses it across all
    pairs, instead of tracing it len(pairs) times.

    Returns
    -------
    min over steps of per-step pairwise overlap sum (scalar)
    """
    n = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    N = x0_ivl.lower.shape[0]
    emb_sys = scenarios[0].emb_system

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:N], upper=arr[N:])

    # Stacked per-pair (p_i, p_j) intervals -- shared emb_system across every
    # scenario is what makes vmapping this valid (see docstring above).
    p_i_batch = irx.Interval(
        lower=jnp.stack([scenarios[i].p_interval.lower for i, j in pairs]),
        upper=jnp.stack([scenarios[i].p_interval.upper for i, j in pairs]),
    )
    p_j_batch = irx.Interval(
        lower=jnp.stack([scenarios[j].p_interval.lower for i, j in pairs]),
        upper=jnp.stack([scenarios[j].p_interval.upper for i, j in pairs]),
    )

    # ── Step 1: propagate all scenarios one Euler step with u_seq[0] ──────
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x1_batch = jax.vmap(lambda p: euler_step(emb_sys, x0_ivl, u_seq[0], p, dt))(p_batch)
    x1_ivls = [irx.Interval(lower=x1_batch.lower[i], upper=x1_batch.upper[i]) for i in range(n)]
    step1_cost = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            step1_cost = step1_cost + pair_cost_fn(x1_ivls[i], x1_ivls[j])

    pxi_arr = jnp.stack([ivl_to_arr(x1_ivls[i]) for i, j in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x1_ivls[j]) for i, j in pairs])

    def refine_one_pair(xi_arr, xj_arr, p_i, p_j, u_k):
        """One pair's refine-then-propagate step. Traced ONCE, vmapped over
        the pair axis by step_body below."""
        x_curr_i = arr_to_ivl(xi_arr)
        x_curr_j = arr_to_ivl(xj_arr)

        x_lo = jnp.maximum(x_curr_i.lower, x_curr_j.lower)
        x_hi = jnp.minimum(x_curr_i.upper, x_curr_j.upper)
        has_overlap = jnp.all(x_hi >= x_lo)
        x_int = irx.Interval(lower=x_lo, upper=x_hi)

        x_ref_i = irx.Interval(
            lower=jnp.where(has_overlap, x_int.lower, x_curr_i.lower),
            upper=jnp.where(has_overlap, x_int.upper, x_curr_i.upper),
        )
        x_ref_j = irx.Interval(
            lower=jnp.where(has_overlap, x_int.lower, x_curr_j.lower),
            upper=jnp.where(has_overlap, x_int.upper, x_curr_j.upper),
        )

        x_next_i = euler_step(emb_sys, x_ref_i, u_k, p_i, dt)
        x_next_j = euler_step(emb_sys, x_ref_j, u_k, p_j, dt)

        raw_cost = pair_cost_fn(x_next_i, x_next_j)
        pair_cost = jnp.where(has_overlap, raw_cost, jnp.array(0.0))
        return ivl_to_arr(x_next_i), ivl_to_arr(x_next_j), pair_cost

    def step_body(carry, k):
        pxi_arr, pxj_arr, min_cost = carry
        u_k = u_seq[k + 1]
        new_pxi, new_pxj, pair_costs = jax.vmap(
            refine_one_pair, in_axes=(0, 0, 0, 0, None)
        )(pxi_arr, pxj_arr, p_i_batch, p_j_batch, u_k)
        step_cost = jnp.sum(pair_costs)
        return (new_pxi, new_pxj, jnp.minimum(min_cost, step_cost))

    init_carry = (pxi_arr, pxj_arr, step1_cost)
    _, _, min_cost_final = _run_unrolled_or_loop(step_body, init_carry, num_steps - 1)
    return min_cost_final


def refined_overlap_loss(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                         scenarios: List[Scenario], dt: float,
                         num_steps: int = 2,
                         pair_cost_fn=_overlap_volume) -> jnp.ndarray:
    """Separation loss using multi-step propagation with state refinement.
    See `propagate_with_refinement`'s docstring for `pair_cost_fn`."""
    return propagate_with_refinement(x0_ivl, u_seq, scenarios, dt, num_steps, pair_cost_fn)


def optimize_refined_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                         num_steps: int = 2, num_restarts: int = 50,
                         learning_rate: float = 0.05, num_iters: int = 200,
                         seed: int = 42,
                         pair_cost_fn=_overlap_volume) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent minimising refined_overlap_loss.

    `pair_cost_fn` defaults to `_overlap_volume` (unchanged prior behavior).
    Pass `_soft_separation_loss` to optimize the margin-based proxy instead
    -- see that function's docstring for why this matters at this module's
    short, near-hover horizons (the volume product can look ~0, and its
    gradient vanish, well before any dimension is genuinely disjoint)."""
    key = jax.random.PRNGKey(seed)
    u_init = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    noise_scale = _U_NOISE_SCALE
    u0 = jax.random.normal(key, (num_restarts, num_steps, 4)) * noise_scale + u_init

    def loss_fn_refined(u_seq):
        return refined_overlap_loss(u_seq, x0_ivl, scenarios, dt, num_steps, pair_cost_fn)

    batched_loss = jax.vmap(loss_fn_refined)
    batched_grad = jax.vmap(jax.grad(loss_fn_refined))

    def body(u_batch, _i):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses
