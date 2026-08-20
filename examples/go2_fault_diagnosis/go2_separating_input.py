"""
GO2 Separating Input -- Active Fault Diagnosis with immrax
==========================================================
Standalone extraction of the immrax-based separating-input core from
examples/go2/go2_separating_input_immrax.py (the "canonical template" for
this project's active-fault-diagnosis modules) into its own dedicated
folder, mirroring examples/unicycle/'s split off of examples/faulty_car/.

Reachable-set-based active fault diagnosis for the Unitree Go2 using
immrax for interval arithmetic and natural-embedding propagation.

Scenarios (4 total, fixed)
---------------------------
  Nominal               alpha=1,              perfect sensors
  Actuator Fault        alpha in [0.6, 0.8]   reduced yaw-rate authority
  Sensor Fault          vy_meas = 0 + noise in [-eps, +eps]
                        (faulty lateral-velocity sensor; robot's dead-reckoned
                         position drifts because it integrates vy_meas ~ 0
                         instead of the true commanded vy)
  Simultaneous Fault    both actuator and sensor faults active at once

Two algorithmic layers (see PLAN.md for why there is no third,
intersection-refinement layer here, unlike unicycle/admire):
  1. Single-step separating input
  2. Multistep unrefined (each scenario propagated independently)

What was changed relative to the original
-------------------------------------------
- `overlap_size_lax` is inlined as `_overlap_volume` (identical `lax.cond`
  branching logic, just no longer imported from
  `examples/faulty_car/interval_functions.py` -- this folder has no
  dependency on `examples/faulty_car` at all). This is INTENTIONALLY NOT
  the branchless clip-and-product `_overlap_volume` used by
  unicycle/admire/nonlinear_chain/quadrotor_fault_diagnosis -- that form
  was tried first and found to introduce NaN gradients (13/50 restarts at
  this module's default multistep demo config) that the `lax.cond`-branched
  form does not; see `_overlap_volume`'s own docstring below for the
  mechanism, and PLAN.md for the empirical isolation.
- Every `jax.lax.fori_loop` is replaced by `jax.lax.scan` (see `_scan_loop`
  below) -- scan has lower per-iteration dispatch overhead than fori_loop
  for this kind of loop, the same finding already applied throughout
  nonlinear_chain_separating_input.py / unicycle's car_separating_input.py.
  The `jax.checkpoint` gradient-memory wrapping on the per-Euler-step and
  per-segment bodies is preserved exactly (checkpoint composes the same way
  under scan as under fori_loop).
- The CBF/obstacle-avoidance closed-loop controller (original module's
  Section 6) is dropped, matching this project's established pattern of
  keeping the separating-input core free of CBF/output-feedback machinery
  (see unicycle/PLAN.md).
- `time_jit` / `_memory_snapshot` helpers are added (ported from
  unicycle/admire's convention) so the demo scripts can report compile vs.
  run time.

System
------
  State   x = [px, py, theta]         (position + heading, m / rad)
  Control u = [vx, vy, omega]         (body-frame velocities + yaw rate)
  Params  p = [alpha, beta] (Nom/Actuator) or [vy_noise, alpha, beta] (Sensor)

  Nom/Actuator dynamics:
      px_dot = vx*cos(theta) - alpha*vy*sin(theta)
      py_dot = vx*sin(theta) + alpha*vy*cos(theta)
      theta_dot = beta * omega

  Sensor Fault dynamics (dead-reckoned pose; vy_cmd is ignored, replaced by
  the faulty sensor's vy_noise reading):
      vy_corrupted = (1 - omega^2)*alpha*vy + omega^2*vy_noise
      px_dot = vx*cos(theta) - vy_corrupted*sin(theta)
      py_dot = vx*sin(theta) + vy_corrupted*cos(theta)
      theta_dot = beta * omega
"""

import os
import sys
import time
import resource
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
from jax import lax
import immrax as irx
import numpy as np

# Control input box constraints: vx, vy in [-.6, .6]; omega in [-.6, .6].
_U_LO = jnp.array([-.6, -.6, -.6])
_U_HI = jnp.array([ .6,  .6,  .6])


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    """Project a control input (or batch) onto the feasible box.

    Works for any leading batch dimensions -- clips only the last axis.
    """
    return jnp.clip(u, _U_LO, _U_HI)


def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    """Pairwise axis-aligned-box overlap volume; 0 when any axis fails to
    overlap.

    NOTE: this is intentionally the original module's `lax.cond`-branching
    `overlap_size_lax` (inlined here so this folder has no dependency on
    examples/faulty_car/interval_functions.py), NOT the branchless
    clip-and-product form used by unicycle/car_separating_input.py /
    nonlinear_chain_separating_input.py / quadrotor_separating_input.py.
    Verified empirically (see PLAN.md) that for THIS system's dynamics, the
    branchless `jnp.maximum(diff, 0.0)`-then-`jnp.prod` form back-propagates
    a NaN gradient through a meaningful fraction of multi-start GD restarts
    (13/50 at this module's default multistep demo config) that the
    `lax.cond`-branched form does not: `lax.cond` never differentiates
    through the untaken "has overlap" branch's `jnp.prod` when the boxes
    are actually disjoint, so a same-forward-value multi-axis-zero
    intersection width never reaches `jnp.prod`'s gradient at all. The
    forward VALUE is identical either way; only the gradient differs.
    """
    lo = jnp.maximum(ivl1.lower, ivl2.lower)
    hi = jnp.minimum(ivl1.upper, ivl2.upper)

    def has_overlap(operands):
        lower, upper = operands
        return jnp.prod(upper - lower)

    def no_overlap(operands):
        del operands
        return jnp.array(0.0)

    has_no_overlap = jnp.any(hi < lo)
    return lax.cond(jnp.logical_not(has_no_overlap), has_overlap, no_overlap, (lo, hi))


def _scan_loop(step_fn, init, n: int):
    """Advance `init` through step_fn exactly n times via jax.lax.scan.
    Used for the outer GD iteration loops -- scan has lower per-iteration
    dispatch overhead than fori_loop for this kind of loop (same finding
    documented in nonlinear_chain_separating_input.py / unicycle's
    car_separating_input.py)."""
    def body(carry, _):
        return step_fn(carry), None
    carry, _ = jax.lax.scan(body, init, xs=None, length=n)
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
# 1.  System Definitions
# ══════════════════════════════════════════════════════════════════════════════

class Go2NomActSystem(irx.System):
    """Unicycle dynamics for the nominal and actuator-fault scenarios.

    State   x = [px, py, theta]     (position + heading, m / rad)
    Control u = [vx, vy, omega]     (body-frame velocities + yaw rate)
    Params  p = [alpha, beta]       (actuator effectiveness; alpha = 1 -> nominal)

    Dynamics:
        px_dot = vx*cos(theta) - alpha*vy*sin(theta)
        py_dot = vx*sin(theta) + alpha*vy*cos(theta)
        theta_dot = beta * omega
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        vx, vy, omega = u[0], u[1], u[2]
        alpha = p[0]
        beta = p[1]
        theta = x[2]
        return jnp.array([
            vx * jnp.cos(theta) - alpha * vy * jnp.sin(theta),
            vx * jnp.sin(theta) + alpha * vy * jnp.cos(theta),
            beta * omega,
        ])


class Go2SensorFaultSystem(irx.System):
    """Dead-reckoned pose under a vy = 0 + noise sensor fault.

    When the lateral-velocity sensor fails it reports vy_meas ~ 0 + noise
    instead of the true commanded vy. The robot's onboard odometry integrates
    this corrupted measurement, so the *estimated* position diverges from the
    true position.

    State   x = [px_hat, py_hat, theta_hat]  (robot's dead-reckoned pose estimate)
    Control u = [vx, vy_cmd, omega]          (commanded; vy_cmd is *ignored* here)
    Params  p = [vy_noise, alpha, beta]      (vy reading from faulty sensor,
                                               vy_noise in [-eps, +eps])
    vy_corrupted = (1 - omega**2) * alpha * vy + omega**2 * vy_noise
    Dynamics (what the odometry integrates):
        px_hat_dot = vx*cos(theta_hat) - vy_corrupted*sin(theta_hat)
        py_hat_dot = vx*sin(theta_hat) + vy_corrupted*cos(theta_hat)
        theta_hat_dot = beta * omega
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        vx       = u[0]
        vy       = u[1]
        vy_noise = p[0]   # override commanded vy with the (faulty) sensor reading
        alpha = p[1]
        beta = p[2]
        omega    = u[2]
        theta    = x[2]
        vy_corrupted = (1 - omega ** 2) * alpha * vy + omega ** 2 * vy_noise
        return jnp.array([
            vx * jnp.cos(theta) - vy_corrupted * jnp.sin(theta),
            vx * jnp.sin(theta) + vy_corrupted * jnp.cos(theta),
            beta * omega,
        ])


# Module-level singletons (created once, reused across calls)
_NOM_ACT_SYS = Go2NomActSystem()
_NOM_ACT_EMB = irx.natemb(_NOM_ACT_SYS)

_SF_SYS = Go2SensorFaultSystem()
_SF_EMB = irx.natemb(_SF_SYS)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Fault Scenarios
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One fault scenario: a named (embedding, parameter-interval) pair."""
    name: str
    emb_system: object        # irx.natemb(...) result
    p_interval: irx.Interval  # parameter interval for this scenario


def create_scenarios(
    actuator_alpha_lo: float = 0.60,
    actuator_alpha_hi: float = 0.80,
    actuator_beta_low: float = 0.60,
    actuator_beta_high: float = .80,
    sensor_noise_bound: float = 0.25,
) -> List[Scenario]:
    """Return the four fault scenarios used throughout this module.

    Parameters
    ----------
    actuator_alpha_lo / hi:
        Range of the actuator effectiveness parameter alpha (nominal = 1).
    sensor_noise_bound:
        Half-width of the vy-sensor noise interval; vy_noise in [-eps, +eps].
    """
    return [
        Scenario(
            name="Nominal",
            emb_system=_NOM_ACT_EMB,
            p_interval=irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2)),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=_NOM_ACT_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo, actuator_beta_low]),
                upper=jnp.array([actuator_alpha_hi, actuator_beta_high]),
            ),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=_SF_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([-sensor_noise_bound, 1.0, 1.0]),
                upper=jnp.array([ sensor_noise_bound, 1.0, 1.0]),
            ),
        ),
        Scenario(
            name="Simultaneous Fault",
            emb_system=_SF_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([-sensor_noise_bound, actuator_alpha_lo / 2, actuator_beta_low / 2]),
                upper=jnp.array([ sensor_noise_bound, actuator_alpha_hi / 2, actuator_beta_high / 2]),
            ),
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Interval Propagation
# ══════════════════════════════════════════════════════════════════════════════

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    """One forward-Euler interval step via the natural embedding.

    Note: t must be a JAX array (not a Python scalar) so that
    eqx.filter_make_jaxpr traces it as an abstract input and the
    jaxpr invar count matches the natif_jaxpr arg count.
    """
    _t    = jnp.zeros(())   # t = 0 as a JAX scalar (so it's a jaxpr invar)
    x_ut  = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    """Propagate the initial interval *num_steps* Euler steps under a constant u.

    Returns the full 3-D state interval [px, py, theta] at the end of the horizon.

    jax.checkpoint prevents the embedding's internal intermediate values from
    being stored across loop steps during reverse-mode AD -- only the carry
    state (6 floats) is kept at each step boundary; the embedding ops
    (~200 floats) are recomputed on the backward sweep.
    """
    emb = scenario.emb_system
    p   = scenario.p_interval

    @jax.checkpoint
    def body(x_carry, _):
        return euler_step(emb, x_carry, u, p, dt), None

    x_final, _ = jax.lax.scan(body, x0_ivl, xs=None, length=num_steps)
    return x_final


def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate and record the state interval at the end of every segment.

    Returns an irx.Interval whose lower/upper have shape
    (num_segments, state_dim) -- one slice per segment end.
    Designed to be vmapped over p_ivl to parallelise across scenarios that
    share the same emb_sys.

    Memory note: two levels of gradient checkpointing are applied.
    - jax.checkpoint on euler_body: prevents storing ~200 embedding
      intermediates per Euler step; only the 6-float carry is retained at
      each step boundary.
    - jax.checkpoint on segment (scan body): prevents lax.scan from
      accumulating K copies of each segment's inner-scan residuals; instead
      each segment is recomputed once during the backward sweep.
    Combined, peak gradient memory scales as O(N x state_dim) per (restart,
    scenario) instead of O(K x N x embedding_ops).
    """
    def segment(x_ivl, u_k):
        @jax.checkpoint
        def euler_body(x, _): return euler_step(emb_sys, x, u_k, p_ivl, dt), None
        x_end, _ = jax.lax.scan(euler_body, x_ivl, xs=None, length=steps_per_segment)
        return x_end, x_end   # carry, stacked output

    _, x_hist = jax.lax.scan(jax.checkpoint(segment), x0_ivl, u_seq)
    return x_hist   # Interval: lower/upper shape (num_segments, state_dim)


def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                 scenario: Scenario,
                                 dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate x0_ivl through a sequence of control inputs.

    u_seq has shape (num_segments, 3). u_seq[k] is held constant for
    *steps_per_segment* Euler steps, then u_seq[k+1] takes over, etc.
    Total horizon = num_segments x steps_per_segment x dt seconds.
    Returns the final state interval only.
    """
    x_hist = _propagate_history(
        x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment
    )
    # x_hist.lower has shape (num_segments, state_dim); take the last row
    return irx.Interval(lower=x_hist.lower[-1], upper=x_hist.upper[-1])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Loss Function & Gradient
# ══════════════════════════════════════════════════════════════════════════════

def position_interval(x_ivl: irx.Interval) -> irx.Interval:
    """Extract the 2-D position sub-interval [px, py] from a 3-D state."""
    return irx.Interval(lower=x_ivl.lower[:2], upper=x_ivl.upper[:2])


def separation_loss(u: jnp.ndarray,
                    x0_ivl: irx.Interval,
                    scenarios: List[Scenario],
                    dt: float,
                    num_steps: int) -> jnp.ndarray:
    """Sum of pairwise position-interval overlaps.

    Minimising this loss maximises the separation of the reachable sets
    in the (px, py) output space, making fault diagnosis easier.

    Parameters
    ----------
    u         : constant control input [vx, vy, omega]
    x0_ivl    : initial state interval
    scenarios : list of Scenario objects
    dt        : Euler step size (s)
    num_steps : number of Euler steps

    Returns
    -------
    Scalar overlap volume (m^2); lower is better.
    """
    pos_ivls = [
        position_interval(propagate_scenario(x0_ivl, u, s, dt, num_steps))
        for s in scenarios
    ]
    n = len(pos_ivls)
    total = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            total = total + _overlap_volume(pos_ivls[i], pos_ivls[j])
    return total


def separation_loss_multistep(u_seq: jnp.ndarray,
                              x0_ivl: irx.Interval,
                              scenarios: List[Scenario],
                              dt: float,
                              steps_per_segment: int) -> jnp.ndarray:
    """Min over segments of the pairwise position-interval overlap sum.

    For each segment end k, computes the sum of pairwise position-interval
    overlaps across all scenarios. Returns the minimum over all K segment
    ends -- i.e., the loss is zero if the scenarios are fully separated at
    ANY point in the trajectory.

    Scenarios that share the same emb_system are propagated in parallel via
    jax.vmap over their stacked p_intervals (only 2 distinct emb_systems
    here -- Go2NomActSystem and Go2SensorFaultSystem -- so this Python loop
    is over emb-system groups, not per-scenario).

    Parameters
    ----------
    u_seq            : (num_segments, 3) sequence of control inputs
    x0_ivl           : initial state interval
    scenarios        : list of Scenario objects
    dt               : Euler step size (s)
    steps_per_segment: Euler steps each control input is held for

    Returns
    -------
    Scalar (m^2); lower is better.
    """
    num_segments = u_seq.shape[0]
    n = len(scenarios)

    # Group scenarios by emb_system identity: emb_id -> (emb_sys, [global_indices], [p_ivl])
    emb_groups: Dict[int, tuple] = {}
    for idx, s in enumerate(scenarios):
        eid = id(s.emb_system)
        if eid not in emb_groups:
            emb_groups[eid] = (s.emb_system, [], [])
        emb_groups[eid][1].append(idx)
        emb_groups[eid][2].append(s.p_interval)

    # Propagate each group in parallel (vmap over p_ivl)
    x_hist_all: List[irx.Interval] = [None] * n
    for emb_sys, indices, p_ivls in emb_groups.values():
        p_batch = irx.Interval(
            lower=jnp.stack([p.lower for p in p_ivls]),
            upper=jnp.stack([p.upper for p in p_ivls]),
        )

        def prop_one(p_ivl_single):
            return _propagate_history(
                x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment
            )

        x_hist_batch = jax.vmap(prop_one)(p_batch)

        for local_i, global_i in enumerate(indices):
            x_hist_all[global_i] = irx.Interval(
                lower=x_hist_batch.lower[local_i],
                upper=x_hist_batch.upper[local_i],
            )

    def overlap_at_k(k):
        pos_ivls_k = [
            irx.Interval(
                lower=x_hist_all[i].lower[k, :2],
                upper=x_hist_all[i].upper[k, :2],
            )
            for i in range(n)
        ]
        total = jnp.array(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                total = total + _overlap_volume(pos_ivls_k[i], pos_ivls_k[j])
        return total

    segment_overlaps = jnp.stack([overlap_at_k(k) for k in range(num_segments)])
    return jnp.min(segment_overlaps)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Optimizer
# ══════════════════════════════════════════════════════════════════════════════

class SeparatingInputOptimizer:
    """Gradient-descent optimizer for fault-separating control inputs."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl    = x0_ivl
        self.dt        = dt
        self.num_steps = num_steps

        _loss = partial(separation_loss,
                        x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def optimize(self,
                 u_init: Optional[jnp.ndarray] = None,
                 learning_rate: float = 0.01,
                 num_iters: int = 150,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        """Plain gradient descent. Returns (u_optimal, final_loss)."""
        if u_init is None:
            u_init = jnp.array([0.5, 0.0, 0.3])
        u = u_init

        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)

            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss  = float(self.loss_fn(u))
                gnorm = float(jnp.linalg.norm(g))
                print(f"  Iter {i:4d}  loss={loss:.6f}  "
                      f"u=[{float(u[0]):+.3f},{float(u[1]):+.3f},{float(u[2]):+.3f}]"
                      f"  |g|={gnorm:.4f}")

        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        """Return position intervals, pairwise overlaps, and volumes for *u*."""
        pos_ivls = [
            position_interval(
                propagate_scenario(self.x0_ivl, u, s, self.dt, self.num_steps)
            )
            for s in self.scenarios
        ]
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_overlap_volume(pos_ivls[i], pos_ivls[j]))

        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, pos_ivls)
        }
        return {
            'position_intervals': pos_ivls,
            'pairwise_overlaps':  overlaps,
            'volumes':            volumes,
        }


class MultistepSequenceOptimizer:
    """Container for multistep loss/grad callables and sequence shape."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, steps_per_segment: int, num_segments: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments

        _loss = partial(
            separation_loss_multistep,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            steps_per_segment=steps_per_segment,
        )
        self.loss_fn = jax.jit(_loss)            # (S,3) -> scalar
        self.grad_fn = jax.jit(jax.grad(_loss))  # (S,3) -> (S,3)


def optimize_multistart(opt: 'SeparatingInputOptimizer',
                        num_restarts: int = 3,
                        learning_rate: float = 0.01,
                        num_iters: int = 150,
                        verbose: bool = False,
                        ) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent using a pre-built optimizer.

    The caller creates the SeparatingInputOptimizer once (which compiles the
    JIT-ted loss and grad functions) and passes it here. Re-using the same
    optimizer object across multiple calls avoids recompilation.

    Returns (best_u, best_loss, best_stats).
    best_stats includes a 'restart_times_s' key: list of wall-clock seconds
    per restart.
    """
    best_u, best_loss, best_stats = None, float('inf'), None
    key  = jax.random.PRNGKey(42)
    restart_times: List[float] = []

    for r in range(num_restarts):
        key, subkey = jax.random.split(key)
        u_init = (jax.random.normal(subkey, (3,)) * 0.3
                  + jnp.array([0.5, 0.0, 0.3]))

        if verbose:
            print(f"\n{'-'*60}\nRestart {r+1}/{num_restarts}\n{'-'*60}")

        _t0 = time.perf_counter()
        u_opt, loss = opt.optimize(u_init, learning_rate, num_iters, verbose)
        restart_times.append(time.perf_counter() - _t0)

        stats = opt.evaluate(u_opt)

        if loss < best_loss:
            best_loss, best_u, best_stats = loss, u_opt, stats
            if verbose:
                print(f"  -> New best: {loss:.6f}")
        if loss == 0:
            break

    best_stats['restart_times_s'] = restart_times
    return best_u, best_loss, best_stats


def optimize_parallel(
    opt: 'SeparatingInputOptimizer',
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    max_workers: Optional[int] = None,
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[jnp.ndarray, float, Dict]:
    """Parallel multi-start gradient descent using a pre-built optimizer.

    Same API/outputs as optimize_multistart, but evaluates restarts in parallel.
    Returns (best_u, best_loss, best_stats), where best_stats includes:
      - restart_times_s: per-restart elapsed seconds (index-aligned)
    """
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    # Deterministic restart seeds.
    subkeys = jax.random.split(jax.random.PRNGKey(seed), num_restarts)

    # Trigger JIT compilation once on caller thread to avoid compile races.
    _u_warm = jnp.array([0.5, 0.0, 0.3])
    _ = opt.loss_fn(_u_warm)
    _ = opt.grad_fn(_u_warm)

    if max_workers is None:
        max_workers = min(num_restarts, max(1, (os.cpu_count() or 1)))
    else:
        max_workers = max(1, min(max_workers, num_restarts))

    restart_times: List[float] = [0.0] * num_restarts
    best_u, best_loss, best_stats = None, float("inf"), None

    def _run_one(restart_idx: int):
        key = subkeys[restart_idx]
        u_init = (jax.random.normal(key, (3,)) * 0.3
                  + jnp.array([0.5, 0.0, 0.3]))
        t0 = time.perf_counter()
        u_opt, loss = opt.optimize(u_init, learning_rate, num_iters, verbose=False)
        elapsed = time.perf_counter() - t0
        stats = opt.evaluate(u_opt)
        return restart_idx, u_opt, float(loss), stats, elapsed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, r): r
            for r in range(num_restarts)
        }

        for fut in as_completed(futures):
            r, u_opt, loss, stats, elapsed = fut.result()
            restart_times[r] = elapsed

            if verbose:
                print(f"Restart {r+1}/{num_restarts}: loss={loss:.6f} time={elapsed:.3f}s")

            if loss < best_loss:
                best_u, best_loss, best_stats = u_opt, loss, stats
                if verbose:
                    print(f"  -> New best from restart {r+1}: {best_loss:.6f}")

    best_stats['restart_times_s'] = restart_times
    return best_u, best_loss, best_stats


def optimize_parallel_gpu(opt, num_restarts=100, learning_rate=0.01, num_iters=150, seed=42):
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, 3)) * 0.3 + jnp.array([0.5, 0.0, 0.3])

    # Vectorize loss/grad across restart axis.
    batched_loss = jax.vmap(opt.loss_fn)             # (R,3) -> (R,)
    batched_grad = jax.vmap(opt.grad_fn)             # (R,3) -> (R,3)

    def body(u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    # One compiled loop on device.
    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)

    best_idx = jnp.argmin(losses)
    best_u = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_parallel_gpu_rejit(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    seed: int = 42,
):
    """Revised version of GPU-parallel multistart optimization for a single
    controller (rebuilds the optimizer, forcing a fresh JIT compile)."""
    return optimize_parallel_gpu(
        SeparatingInputOptimizer(
            scenarios=scenarios,
            x0_ivl=x0_ivl,
            dt=dt,
            num_steps=num_steps,
        ),
        num_restarts=num_restarts,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=seed,
    )


def optimize_multistep_gpu(
    opt: 'MultistepSequenceOptimizer',
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    seed: int = 42,
):
    """GPU-parallel multi-start optimization for control sequences.

    Mirrors optimize_parallel_gpu but for u_seq with shape (num_segments, 3).
    Returns:
      best_u_seq, best_loss, all_u_seq_final, all_final_losses
    """
    key = jax.random.PRNGKey(seed)
    u0 = (
        jax.random.normal(key, (num_restarts, opt.num_segments, 3)) * 0.1
        + jnp.array([0.5, 0.0, 0.3])
    )

    # Vectorize loss/grad across restart axis.
    batched_loss = jax.vmap(opt.loss_fn)   # (R,S,3) -> (R,)
    batched_grad = jax.vmap(opt.grad_fn)   # (R,S,3) -> (R,S,3)

    def body(u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    # One compiled loop on device.
    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)

    best_idx = jnp.argmin(losses)
    best_u = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_multistep_gpu_rejit(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    steps_per_segment: int,
    num_segments: int,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    seed: int = 42,
):
    key = jax.random.PRNGKey(seed)
    u0 = (
        jax.random.normal(key, (num_restarts, num_segments, 3)) * 0.1
        + jnp.array([0.5, 0.0, 0.3])
    )

    def loss_fn_multistep(u):
        return separation_loss_multistep(
            u,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            steps_per_segment=steps_per_segment,
        )

    # Vectorize loss/grad across restart axis.
    batched_loss = jax.vmap(loss_fn_multistep)              # (R,S,3) -> (R,)
    batched_grad = jax.vmap(jax.grad(loss_fn_multistep))    # (R,S,3) -> (R,S,3)

    def body(u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    # One compiled loop on device.
    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)

    best_idx = jnp.argmin(losses)
    best_u = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_multistep(scenarios: List[Scenario],
                       x0_ivl: irx.Interval,
                       dt: float,
                       steps_per_segment: int,
                       num_segments: int,
                       learning_rate: float = 0.01,
                       num_iters: int = 300,
                       num_restarts: int = 100,
                       verbose: bool = False,
                       seed: int = 42,
                       ) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent over a sequence of control inputs.

    Optimises u_seq of shape (num_segments, 3), where u_seq[k] is applied
    to all scenarios for *steps_per_segment* Euler steps before switching to
    u_seq[k+1]. Total horizon = num_segments x steps_per_segment x dt s.

    Uses separation_loss_multistep as the objective, which sums pairwise
    position-interval overlaps at the end of the full horizon.
    Performs batched multi-start optimisation over *num_restarts* random
    initial guesses and returns the best sequence.

    Parameters
    ----------
    scenarios         : list of Scenario objects
    x0_ivl            : initial state interval
    dt                : Euler step size (s)
    steps_per_segment : number of Euler steps per control segment
    num_segments      : number of control segments (length of sequence)
    learning_rate     : gradient descent step size
    num_iters         : number of gradient steps
    num_restarts      : number of random initial control sequences
    verbose           : print loss every 20 iterations
    seed              : PRNG seed for random initialisation

    Returns
    -------
    (u_seq_opt, loss_opt, stats)
    u_seq_opt : (num_segments, 3) optimal control sequence
    loss_opt  : final overlap loss (m^2)
    stats     : dict with 'position_intervals', 'pairwise_overlaps',
                'volumes', 'optimization_time_s', 'best_restart_idx',
                'all_restart_losses', 'num_restarts'
    """
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    ms_opt = MultistepSequenceOptimizer(
        scenarios=scenarios,
        x0_ivl=x0_ivl,
        dt=dt,
        steps_per_segment=steps_per_segment,
        num_segments=num_segments,
    )

    _t0 = time.perf_counter()
    u_seq, loss_opt_jax, _, final_losses = optimize_multistep_gpu(
        opt=ms_opt,
        num_restarts=num_restarts,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=seed,
    )
    elapsed = time.perf_counter() - _t0

    best_idx = int(jnp.argmin(final_losses))
    loss_opt = float(loss_opt_jax)

    if verbose:
        mean_loss = float(jnp.mean(final_losses))
        print(
            f"Multistep GPU multistart complete: best_loss={loss_opt:.6f}  "
            f"mean_final_loss={mean_loss:.6f}  best_restart={best_idx+1}/{num_restarts}"
        )

    # Build stats dict (same schema as SeparatingInputOptimizer.evaluate).
    pos_ivls = [
        position_interval(
            propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment)
        )
        for s in scenarios
    ]
    n = len(scenarios)
    overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            key_ij = f"{scenarios[i].name} vs {scenarios[j].name}"
            overlaps[key_ij] = float(_overlap_volume(pos_ivls[i], pos_ivls[j]))
    volumes = {
        s.name: float(jnp.prod(iv.upper - iv.lower))
        for s, iv in zip(scenarios, pos_ivls)
    }
    stats = {
        'position_intervals':  pos_ivls,
        'pairwise_overlaps':   overlaps,
        'volumes':             volumes,
        'optimization_time_s': elapsed,
        'best_restart_idx':    best_idx,
        'all_restart_losses':  np.array(final_losses),
        'num_restarts':        num_restarts,
    }
    return u_seq, loss_opt, stats
