"""
Faulty Nonholonomic Car -- Separating Input, Active Fault Diagnosis with immrax
=================================================================================
Clean rewrite of examples/faulty_car/faulty_car_separating_input.py. Same
system and scenarios; fixes several bugs found in the original (see PLAN.md
in this directory for the full writeup) and drops the surrounding sprawl
(7 near-duplicate animation scripts, CBF/closed-loop module, dead 4-state
car model) that had accreted around that one clean core.

System
------
  State   x = [px, py, phi]     (2-D position + heading)
  Control u = [v, omega]         (forward velocity + steering rate), box [-1,1]^2
  Params  p = [alpha]             (steering effectiveness; alpha=1 -> nominal)

  Dynamics:
      px_dot = v * cos(phi)
      py_dot = v * sin(phi)
      phi_dot = alpha * omega

Scenarios (3 total, fixed)
---------------------------
  Nominal          alpha=1,           y = [px, py]
  Actuator Fault   alpha in [0,0.5],  y = [px, py]
  Sensor Fault     alpha=1,           y = obs_scale*[px, py] + obs_offset
                                       (obs_scale=0.95, obs_offset=[0.2,0.2])

Bugs fixed relative to the original (see PLAN.md for the full analysis):
  1. Sensor Fault is now actually separable: `observed_output` is applied
     before computing overlap in ALL THREE layers (the original only
     applied it in the refined layer -- Sensor Fault's state is literally
     identical to Nominal's, so the single-step/multistep-unrefined losses
     were comparing two always-identical intervals).
  2. `_invert_observation` always uses its OWN scenario's obs_scale (the
     original's refinement step divided both scenarios in a pair by the
     second scenario's obs_scale).
  3. The refinement step is factored into one `_refine_and_step_pair`
     helper, used by both the jittable loss (`propagate_with_refinement`)
     and the animation's history collector (`collect_refinement_history`)
     -- in the original these were two independently-maintained
     reimplementations that had already drifted apart (the animation's
     didn't apply obs_scale at all).
  4. No-overlap fallback: on no-overlap, a pair's state now falls back to
     its UNCHANGED current interval (`jnp.where` gating the state itself),
     matching the already-proven pattern in nonlinear_chain_separating_input.py
     / quadrotor_separating_input.py -- the original derived a fallback
     observation from one scenario's center alone and inverted BOTH
     scenarios from it, corrupting the carried-forward state.
  5. One branchless `_overlap_volume` (clip-and-product, no lax.cond),
     no dependency on interval_functions.py's six near-duplicate variants.

Three algorithmic layers (mirroring nonlinear_chain_separating_input.py):
  1. Single-step separating input
  2. Multistep unrefined (each scenario propagated independently)
  3. Multistep intersection-refinement (observed-output intervals
     intersected mid-horizon to tighten state estimates)
"""

import sys
import time
import resource
from pathlib import Path
from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple, Optional, Dict

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

# Control input box: v, omega in [-1, 1].
_U_LO = jnp.array([-1.0, -1.0])
_U_HI = jnp.array([1.0, 1.0])


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    """Project a control input (or batch) onto the feasible box."""
    return jnp.clip(u, _U_LO, _U_HI)


def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    """Branchless pairwise axis-aligned-box overlap volume (clip-then-product;
    0 exactly when any axis fails to overlap). Same helper as
    nonlinear_chain_separating_input.py / quadrotor_separating_input.py --
    replaces interval_functions.py's lax.cond-based overlap_size_lax."""
    widths = jnp.maximum(
        jnp.minimum(ivl1.upper, ivl2.upper) - jnp.maximum(ivl1.lower, ivl2.lower),
        0.0,
    )
    return jnp.prod(widths)


def _scan_loop(step_fn, init, n: int):
    """Advance `init` through step_fn exactly n times via jax.lax.scan.
    Used for the outer GD iteration loops -- scan has lower per-iteration
    dispatch overhead than fori_loop for this kind of loop (same finding
    documented in nonlinear_chain_separating_input.py)."""
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
# 1.  System Definition
# ══════════════════════════════════════════════════════════════════════════════

class CarNomActSystem(irx.System):
    """Nonholonomic car with actuator-fault authority alpha.

    State   x = [px, py, phi]
    Control u = [v, omega]
    Params  p = [alpha]   (alpha=1 -> full steering authority)

    Dynamics:
        px_dot = v * cos(phi)
        py_dot = v * sin(phi)
        phi_dot = alpha * omega
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        v, omega = u[0], u[1]
        alpha = p[0]
        phi = x[2]
        return jnp.array([
            v * jnp.cos(phi),
            v * jnp.sin(phi),
            alpha * omega,
        ])


# Module-level singleton: one system + one embedding, reused everywhere.
_NOM_ACT_SYS = CarNomActSystem()
_NOM_ACT_EMB = irx.natemb(_NOM_ACT_SYS)


def get_system_and_embedding() -> Tuple[CarNomActSystem, object]:
    return _NOM_ACT_SYS, _NOM_ACT_EMB


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Fault Scenarios & Output Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One fault scenario: shared dynamics (via alpha) + an affine output map.

    obs_offset, obs_scale : the observed output is
        y = obs_scale * [px, py] + obs_offset
    obs_scale is a POINT value here (not interval-uncertain, unlike
    integrator_chain/nonlinear_chain's beta) -- every scenario's obs_scale
    is a known constant, so `_invert_observation` is a plain division, no
    4-corner min/max needed.
    """
    name: str
    emb_system: object
    p_interval: irx.Interval
    obs_offset: jnp.ndarray = field(default_factory=lambda: jnp.zeros(2))
    obs_scale: jnp.ndarray = field(default_factory=lambda: jnp.ones(1))


def create_scenarios(
    actuator_alpha_lo: float = 0.0,
    actuator_alpha_hi: float = 0.5,
    sensor_obs_offset: Tuple[float, float] = (0.2, 0.2),
    sensor_obs_scale: float = 0.95,
) -> List[Scenario]:
    """Return the three fault scenarios used throughout this module.

    All three share CarNomActSystem's dynamics/embedding. Actuator Fault is
    distinguished by alpha; Sensor Fault by an affine offset/scale on the
    observed [px, py] output (dynamics identical to Nominal).
    """
    _, emb = get_system_and_embedding()
    return [
        Scenario(
            name="Nominal",
            emb_system=emb,
            p_interval=irx.icentpert(jnp.array([1.0]), jnp.zeros(1)),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=emb,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo]),
                upper=jnp.array([actuator_alpha_hi]),
            ),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=emb,
            p_interval=irx.icentpert(jnp.array([1.0]), jnp.zeros(1)),
            obs_offset=jnp.array(sensor_obs_offset),
            obs_scale=jnp.array([sensor_obs_scale]),
        ),
    ]


def observed_output(x_ivl: irx.Interval, scenario: Scenario) -> irx.Interval:
    """y = obs_scale * [px, py] + obs_offset, via interval arithmetic.

    obs_scale is a point value (see Scenario docstring), so this is a
    straightforward elementwise affine map on both bounds.
    """
    pos_lower, pos_upper = x_ivl.lower[:2], x_ivl.upper[:2]
    scale = scenario.obs_scale[0]
    return irx.Interval(
        lower=scale * pos_lower + scenario.obs_offset,
        upper=scale * pos_upper + scenario.obs_offset,
    )


def _invert_observation(y_ivl: irx.Interval, scenario: Scenario,
                        phi_source: irx.Interval) -> irx.Interval:
    """Outer-enclose [px, py, phi] from y = obs_scale*[px,py] + obs_offset.

    obs_scale is a known positive point constant for `scenario` (not
    interval-uncertain), so inversion is exact division -- no 4-corner
    min/max needed (contrast integrator_chain/nonlinear_chain, whose beta
    IS interval-valued). phi is not observed; passed through unchanged
    from `phi_source` (the pre-refinement state interval for this scenario).
    """
    scale = scenario.obs_scale[0]
    pos_lower = (y_ivl.lower - scenario.obs_offset) / scale
    pos_upper = (y_ivl.upper - scenario.obs_offset) / scale
    return irx.Interval(
        lower=jnp.concatenate([pos_lower, phi_source.lower[2:3]]),
        upper=jnp.concatenate([pos_upper, phi_source.upper[2:3]]),
    )


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


def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    """Propagate the initial interval *num_steps* Euler steps under a constant u."""
    def body(i, x_carry):
        return euler_step(scenario.emb_system, x_carry, u, scenario.p_interval, dt)
    return jax.lax.fori_loop(0, num_steps, body, x0_ivl)


def separation_loss(u: jnp.ndarray,
                    x0_ivl: irx.Interval,
                    scenarios: List[Scenario],
                    dt: float,
                    num_steps: int) -> jnp.ndarray:
    """Sum of pairwise OBSERVED-output-interval overlaps.

    Minimising this loss maximises the separation of the scenarios'
    reachable sets in observed output space y -- fixes the original's bug
    of comparing raw state (see module docstring, fix #1).
    """
    y_ivls = [
        observed_output(propagate_scenario(x0_ivl, u, s, dt, num_steps), s)
        for s in scenarios
    ]
    n = len(y_ivls)
    total = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            total = total + _overlap_volume(y_ivls[i], y_ivls[j])
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
                 learning_rate: float = 0.05, num_iters: int = 200,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        if u_init is None:
            u_init = jnp.array([0.5, 0.3])
        u = u_init
        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)
            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = float(self.loss_fn(u))
                print(f"  Iter {i:4d}  loss={loss:.6f}  "
                      f"u=[{float(u[0]):+.3f},{float(u[1]):+.3f}]  "
                      f"|g|={float(jnp.linalg.norm(g)):.4f}")
        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        y_ivls = [
            observed_output(propagate_scenario(self.x0_ivl, u, s, self.dt, self.num_steps), s)
            for s in self.scenarios
        ]
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_overlap_volume(y_ivls[i], y_ivls[j]))
        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, y_ivls)
        }
        return {'output_intervals': y_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes}


def optimize_parallel_gpu(opt: 'SeparatingInputOptimizer', num_restarts: int = 100,
                          learning_rate: float = 0.05, num_iters: int = 200,
                          seed: int = 42):
    """GPU-parallel multi-start gradient descent for a constant separating input."""
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, 2)) * 0.3 + jnp.array([0.5, 0.3])

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def optimize_parallel_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                dt: float, num_steps: int, num_restarts: int = 100,
                                learning_rate: float = 0.05, num_iters: int = 200,
                                seed: int = 42):
    return optimize_parallel_gpu(
        SeparatingInputOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt, num_steps=num_steps),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Multistep (Unrefined) Path
# ══════════════════════════════════════════════════════════════════════════════

def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate and record the state interval at the end of every segment.
    Returns an irx.Interval whose lower/upper have shape (num_segments, 3)."""
    def segment(x_ivl, u_k):
        def euler_body(_, x): return euler_step(emb_sys, x, u_k, p_ivl, dt)
        x_end = jax.lax.fori_loop(0, steps_per_segment, euler_body, x_ivl)
        return x_end, x_end

    _, x_hist = jax.lax.scan(segment, x0_ivl, u_seq)
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
                              steps_per_segment: int) -> jnp.ndarray:
    """Min over segments of the pairwise OBSERVED-output-interval overlap sum.

    All scenarios share one emb_system (Sensor Fault only changes the
    output map, never the dynamics), so every scenario is propagated in a
    single vmap over stacked p_intervals.
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

    # x_hist_batch: Interval with lower/upper shape (n, num_segments, 3)
    x_hist_batch = jax.vmap(prop_one)(p_batch)

    def overlap_at_k(k):
        y_ivls_k = [
            observed_output(
                irx.Interval(lower=x_hist_batch.lower[i, k], upper=x_hist_batch.upper[i, k]),
                scenarios[i],
            )
            for i in range(n)
        ]
        total = jnp.array(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                total = total + _overlap_volume(y_ivls_k[i], y_ivls_k[j])
        return total

    segment_overlaps = jnp.stack([overlap_at_k(k) for k in range(num_segments)])
    return jnp.min(segment_overlaps)


class MultistepSequenceOptimizer:
    """Container for multistep loss/grad callables and sequence shape."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, steps_per_segment: int, num_segments: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments

        _loss = partial(separation_loss_multistep, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, steps_per_segment=steps_per_segment)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))


def optimize_multistep_gpu(opt: 'MultistepSequenceOptimizer', num_restarts: int = 100,
                           learning_rate: float = 0.05, num_iters: int = 200, seed: int = 42):
    """GPU-parallel multi-start optimization for control sequences."""
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, opt.num_segments, 2)) * 0.3 + jnp.array([0.5, 0.3])

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def optimize_multistep_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                 dt: float, steps_per_segment: int, num_segments: int,
                                 num_restarts: int = 100, learning_rate: float = 0.05,
                                 num_iters: int = 200, seed: int = 42):
    return optimize_multistep_gpu(
        MultistepSequenceOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
                                   steps_per_segment=steps_per_segment, num_segments=num_segments),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
    )


def optimize_multistep(scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float,
                       steps_per_segment: int, num_segments: int,
                       learning_rate: float = 0.05, num_iters: int = 300,
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

    y_ivls = [
        observed_output(propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment), s)
        for s in scenarios
    ]
    n = len(scenarios)
    overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            overlaps[f"{scenarios[i].name} vs {scenarios[j].name}"] = float(_overlap_volume(y_ivls[i], y_ivls[j]))
    volumes = {s.name: float(jnp.prod(iv.upper - iv.lower)) for s, iv in zip(scenarios, y_ivls)}
    stats = {
        'output_intervals': y_ivls,
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

def _refine_and_step_pair(
    x_curr_i: irx.Interval, x_curr_j: irx.Interval,
    scenario_i: Scenario, scenario_j: Scenario,
    u_k: jnp.ndarray, dt: float,
):
    """One scenario pair, one refine-then-propagate step.

    THE canonical per-pair refinement computation -- used by BOTH
    `propagate_with_refinement` (the jittable loss, inside fori_loop) and
    `collect_refinement_history` (the animation's plain-Python history
    collector), so the optimized loss and the visualized animation can
    never drift apart (fix #3 -- see module docstring).

    On no-overlap, falls back to the UNCHANGED current state (fix #4).

    Returns (x_next_i, x_next_j, obs_next_i, obs_next_j, pair_cost, has_overlap).
    """
    obs_i = observed_output(x_curr_i, scenario_i)
    obs_j = observed_output(x_curr_j, scenario_j)

    y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
    y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
    has_overlap = jnp.all(y_hi >= y_lo)
    y_int = irx.Interval(lower=y_lo, upper=y_hi)

    x_ref_i_overlap = _invert_observation(y_int, scenario_i, x_curr_i)
    x_ref_j_overlap = _invert_observation(y_int, scenario_j, x_curr_j)

    x_ref_i = irx.Interval(
        lower=jnp.where(has_overlap, x_ref_i_overlap.lower, x_curr_i.lower),
        upper=jnp.where(has_overlap, x_ref_i_overlap.upper, x_curr_i.upper),
    )
    x_ref_j = irx.Interval(
        lower=jnp.where(has_overlap, x_ref_j_overlap.lower, x_curr_j.lower),
        upper=jnp.where(has_overlap, x_ref_j_overlap.upper, x_curr_j.upper),
    )

    x_next_i = euler_step(scenario_i.emb_system, x_ref_i, u_k, scenario_i.p_interval, dt)
    x_next_j = euler_step(scenario_j.emb_system, x_ref_j, u_k, scenario_j.p_interval, dt)

    obs_next_i = observed_output(x_next_i, scenario_i)
    obs_next_j = observed_output(x_next_j, scenario_j)
    raw_cost = _overlap_volume(obs_next_i, obs_next_j)
    pair_cost = jnp.where(has_overlap, raw_cost, jnp.array(0.0))

    return x_next_i, x_next_j, obs_next_i, obs_next_j, pair_cost, has_overlap


def propagate_with_refinement(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                              scenarios: List[Scenario], dt: float,
                              num_steps: int = 2) -> jnp.ndarray:
    """Multi-step propagation with per-pair observation-based state refinement.

    Step 1 (computed once): propagate every scenario one Euler step from
    x0_ivl using u_seq[0]; cost = sum of pairwise observed-output overlaps.

    Steps 2..num_steps (one fori_loop iteration per step, all pairs inside):
    each pair independently refines-then-propagates via
    `_refine_and_step_pair`; state is carried as (n_pairs, 2*3) flattened-
    interval arrays.

    Returns
    -------
    min over steps of per-step pairwise overlap sum (scalar)
    """
    n = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    xlen = x0_ivl.lower.shape[0]

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

    # ── Step 1: propagate all scenarios one Euler step with u_seq[0] ──────
    x1_ivls = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
    y1_ivls = [observed_output(x1, s) for x1, s in zip(x1_ivls, scenarios)]
    step1_cost = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            step1_cost = step1_cost + _overlap_volume(y1_ivls[i], y1_ivls[j])

    pxi_arr = jnp.stack([ivl_to_arr(x1_ivls[i]) for i, j in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x1_ivls[j]) for i, j in pairs])

    def step_body(k, carry):
        pxi_arr, pxj_arr, min_cost = carry
        u_k = u_seq[k + 1]
        step_cost = jnp.array(0.0)
        new_pxi, new_pxj = [], []

        for idx, (i, j) in enumerate(pairs):
            x_curr_i = arr_to_ivl(pxi_arr[idx])
            x_curr_j = arr_to_ivl(pxj_arr[idx])
            x_next_i, x_next_j, _, _, pair_cost, _ = _refine_and_step_pair(
                x_curr_i, x_curr_j, scenarios[i], scenarios[j], u_k, dt
            )
            step_cost = step_cost + pair_cost
            new_pxi.append(ivl_to_arr(x_next_i))
            new_pxj.append(ivl_to_arr(x_next_j))

        return (jnp.stack(new_pxi), jnp.stack(new_pxj), jnp.minimum(min_cost, step_cost))

    init_carry = (pxi_arr, pxj_arr, step1_cost)
    _, _, min_cost_final = jax.lax.fori_loop(0, num_steps - 1, step_body, init_carry)
    return min_cost_final


def refined_overlap_loss(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                         scenarios: List[Scenario], dt: float,
                         num_steps: int = 2) -> jnp.ndarray:
    """Separation loss using multi-step propagation with state refinement."""
    return propagate_with_refinement(x0_ivl, u_seq, scenarios, dt, num_steps)


def optimize_refined_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                         num_steps: int = 2, num_restarts: int = 50,
                         learning_rate: float = 0.05, num_iters: int = 200,
                         seed: int = 42) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent minimising refined_overlap_loss."""
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, num_steps, 2)) * 0.3 + jnp.array([0.5, 0.3])

    def loss_fn_refined(u_seq):
        return refined_overlap_loss(u_seq, x0_ivl, scenarios, dt, num_steps)

    batched_loss = jax.vmap(loss_fn_refined)
    batched_grad = jax.vmap(jax.grad(loss_fn_refined))

    def body(u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def collect_refinement_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                               scenarios: List[Scenario], dt: float, num_steps: int):
    """Plain-Python-loop history collector for animation/plotting.

    Calls the EXACT SAME `_refine_and_step_pair` helper used by
    `propagate_with_refinement`'s jittable loss (fix #3 -- see module
    docstring), so this can never numerically diverge from the optimized
    loss the way the original animation's independent reimplementation did.

    Returns (steps, pairs):
      steps[k]['t']            : float time at step k+1
      steps[k]['pair_obs'][pi] : (obs_i, obs_j, intersection_or_None)
      pairs                    : list of (i, j) scenario-index tuples
    """
    n = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    x1 = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
    obs1 = [observed_output(xi, s) for xi, s in zip(x1, scenarios)]

    def intersect_or_none(a, b):
        lo = jnp.maximum(a.lower, b.lower)
        hi = jnp.minimum(a.upper, b.upper)
        return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None

    steps = [{'t': dt, 'pair_obs': [
        (obs1[i], obs1[j], intersect_or_none(obs1[i], obs1[j])) for i, j in pairs
    ]}]
    pair_states = [(x1[i], x1[j]) for i, j in pairs]

    for k in range(num_steps - 1):
        new_states, new_pair_obs = [], []
        for (i, j), (xi, xj) in zip(pairs, pair_states):
            x_next_i, x_next_j, obs_next_i, obs_next_j, _, _ = _refine_and_step_pair(
                xi, xj, scenarios[i], scenarios[j], u_seq[k + 1], dt
            )
            new_pair_obs.append((obs_next_i, obs_next_j, intersect_or_none(obs_next_i, obs_next_j)))
            new_states.append((x_next_i, x_next_j))
        steps.append({'t': (k + 2) * dt, 'pair_obs': new_pair_obs})
        pair_states = new_states

    return steps, pairs
