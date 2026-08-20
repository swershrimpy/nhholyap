"""
Faulty Nonholonomic Car – Separating Input, Active Fault Diagnosis with immrax
===============================================================================
Reachable-set-based active fault diagnosis for the faulty nonholonomic car
using immrax for interval arithmetic and natural-embedding propagation.

Scenarios
---------
  Nominal           alpha = 1.0      (full steering authority, perfect sensors)
  Actuator Fault    alpha ∈ [0, 0.5] (reduced steering gain)
  Sensor Fault      alpha = 1.0      (same dynamics as nominal, but px/py sensors
                                      report a fixed +0.2 m offset)

System (Separated-Input form)
-----------------------------
  State   x = [px, py, φ]         (2-D position + heading)
  Control u = [v, ω]               (forward velocity + steering rate)

  All scenarios share the same dynamics (CarNomActSystem):
      ṗx = v · cos φ
      ṗy = v · sin φ
      φ̇  = alpha · ω             (alpha = 1 → nominal/sensor-fault;
                                   alpha ∈ [0,0.5] → actuator fault)

Output for Separation
---------------------
  Nominal / Actuator Fault:  y = [px,       py      ]
  Sensor Fault:              y = [px + 0.2, py + 0.2]

Follows the pattern of go2/go2_separating_input_immrax.py.
"""

import sys
import os
from pathlib import Path

# File is at examples/faulty_car/<name>.py  →  add this directory to sys.path
# so that interval_functions can be imported directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np
from functools import partial
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from interval_functions import overlap_size_lax

# Control input box constraints: v ∈ [-1, 1] m/s, ω ∈ [-1, 1] rad/s.
_U_LO = jnp.array([-1.0, -1.0])
_U_HI = jnp.array([ 1.0,  1.0])


def _pair_indices(n: int):
    """Static (i, j) index arrays for all C(n, 2) unordered pairs, i < j.
    See admire/nonlinear_chain/integrator_chain/car_fault_diagnosis/
    quadrotor_fault_diagnosis's identical helper for the full rationale."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return jnp.array([i for i, j in pairs]), jnp.array([j for i, j in pairs])


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    """Project control input (or batch) onto the feasible box."""
    return jnp.clip(u, _U_LO, _U_HI)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  System Definitions
# ══════════════════════════════════════════════════════════════════════════════

class CarNomActSystem(irx.System):
    """Separated-input nonholonomic car for nominal and actuator-fault scenarios.

    State   x = [px, py, φ]   (2-D position + heading, m / rad)
    Control u = [v, ω]         (forward velocity + steering rate)
    Params  p = [alpha]         (steering effectiveness; alpha=1 → nominal)

    Dynamics:
        ṗx = v · cos φ
        ṗy = v · sin φ
        φ̇  = alpha · ω
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        v, omega = u[0], u[1]
        alpha    = p[0]
        phi      = x[2]
        return jnp.array([
            v * jnp.cos(phi),
            v * jnp.sin(phi),
            alpha * omega,
        ])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Fault Scenarios
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One fault scenario: a named (embedding, parameter-interval) pair.

    obs_offset : 2-D array [δpx, δpy].  The observed output for this scenario
                 is y = [px + δpx, py + δpy].  For nominal / actuator-fault
                 scenarios δ = 0; for the sensor-fault scenario δ = [0.2, 0.2].
    """
    name: str
    emb_system: object        # irx.natemb(…) result
    p_interval: irx.Interval  # parameter interval for this scenario
    obs_offset: jnp.ndarray = field(default_factory=lambda: jnp.zeros(2))
    obs_scale: jnp.ndarray = field(default_factory=lambda: jnp.ones(1))


# Module-level singletons (created once, reused across calls)
_NOM_ACT_SYS = CarNomActSystem()
_NOM_ACT_EMB = irx.natemb(_NOM_ACT_SYS)


def create_scenarios(
    actuator_alpha_lo: float = 0.0,
    actuator_alpha_hi: float = 0.5,
) -> List[Scenario]:
    """Return the three fault scenarios used throughout this module.

    Parameters
    ----------
    actuator_alpha_lo / hi:
        Range of the steering-fault effectiveness parameter alpha
        (1 = nominal, 0 = complete steering loss).

    All three scenarios share CarNomActSystem dynamics.  The sensor fault
    is distinguished purely by a +0.2 m output offset on px and py.
    """
    return [
        Scenario(
            name="Nominal",
            emb_system=_NOM_ACT_EMB,
            p_interval=irx.icentpert(jnp.array([1.0]), jnp.zeros(1)),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=_NOM_ACT_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo]),
                upper=jnp.array([actuator_alpha_hi]),
            ),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=_NOM_ACT_EMB,
            p_interval=irx.icentpert(jnp.array([1.0]), jnp.zeros(1)),
            obs_offset=jnp.array([0.2, 0.2]),
            obs_scale=jnp.array([0.95])
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
    _t    = jnp.zeros(())   # t = 0 as a JAX scalar
    x_ut  = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    """Propagate the initial interval *num_steps* Euler steps under a constant u.

    Returns the full 3-D state interval [px, py, φ] at the end of the horizon.
    JAX-compatible (uses lax.fori_loop, JIT-able and differentiable w.r.t. u).
    """
    emb = scenario.emb_system
    p   = scenario.p_interval

    def body(i, x_carry):
        return euler_step(emb, x_carry, u, p, dt)

    return jax.lax.fori_loop(0, num_steps, body, x0_ivl)


def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate and record the state interval at the end of every segment.

    Returns an irx.Interval whose lower/upper have shape
    (num_segments, state_dim) — one slice per segment end.
    Designed to be vmapped over p_ivl to parallelise across scenarios that
    share the same emb_sys.
    """
    def segment(x_ivl, u_k):
        def euler_body(_, x): return euler_step(emb_sys, x, u_k, p_ivl, dt)
        x_end = jax.lax.fori_loop(0, steps_per_segment, euler_body, x_ivl)
        return x_end, x_end   # carry, stacked output

    _, x_hist = jax.lax.scan(segment, x0_ivl, u_seq)
    return x_hist   # Interval: lower/upper shape (num_segments, state_dim)


def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                 scenario: Scenario,
                                 dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate x0_ivl through a sequence of control inputs.

    u_seq has shape (num_segments, 2).  u_seq[k] is held constant for
    *steps_per_segment* Euler steps, then u_seq[k+1] takes over, etc.
    Total horizon = num_segments × steps_per_segment × dt seconds.
    Returns the final state interval only.
    """
    x_hist = _propagate_history(
        x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment
    )
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

    Minimising this loss maximises the separation of all reachable sets in
    the (px, py) output space, making fault diagnosis easier.

    Parameters
    ----------
    u         : constant control input [v, ω]
    x0_ivl    : initial state interval
    scenarios : list of Scenario objects
    dt        : Euler step size (s)
    num_steps : number of Euler steps

    Returns
    -------
    Scalar overlap area (m²); lower is better.
    """
    pos_ivls = [
        position_interval(propagate_scenario(x0_ivl, u, s, dt, num_steps))
        for s in scenarios
    ]
    n = len(pos_ivls)
    lo_stack = jnp.stack([iv.lower for iv in pos_ivls])
    hi_stack = jnp.stack([iv.upper for iv in pos_ivls])
    pair_i, pair_j = _pair_indices(n)
    ivl_i = irx.Interval(lower=lo_stack[pair_i], upper=hi_stack[pair_i])
    ivl_j = irx.Interval(lower=lo_stack[pair_j], upper=hi_stack[pair_j])
    return jnp.sum(jax.vmap(overlap_size_lax)(ivl_i, ivl_j))


def separation_loss_multistep(u_seq: jnp.ndarray,
                              x0_ivl: irx.Interval,
                              scenarios: List[Scenario],
                              dt: float,
                              steps_per_segment: int) -> jnp.ndarray:
    """Min over segments of the pairwise position-interval overlap sum.

    For each segment end k, computes the sum of pairwise position-interval
    overlaps across all scenarios.  Returns the minimum over all K segment
    ends — the loss is zero if the scenarios are fully separated at ANY
    point in the trajectory.

    Scenarios that share the same emb_system are propagated in parallel via
    jax.vmap over their stacked p_intervals.

    Parameters
    ----------
    u_seq            : (num_segments, 2) sequence of control inputs
    x0_ivl           : initial state interval
    scenarios        : list of Scenario objects
    dt               : Euler step size (s)
    steps_per_segment: Euler steps each control input is held for

    Returns
    -------
    Scalar (m²); lower is better.
    """
    num_segments = u_seq.shape[0]
    n = len(scenarios)

    # ── Group scenarios by emb_system identity ────────────────────────────
    emb_groups: Dict[int, tuple] = {}
    for idx, s in enumerate(scenarios):
        eid = id(s.emb_system)
        if eid not in emb_groups:
            emb_groups[eid] = (s.emb_system, [], [])
        emb_groups[eid][1].append(idx)
        emb_groups[eid][2].append(s.p_interval)

    # ── Propagate each group in parallel (vmap over p_ivl) ───────────────
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

        # x_hist_batch: Interval with lower/upper shape (B, num_segments, 3)
        x_hist_batch = jax.vmap(prop_one)(p_batch)

        for local_i, global_i in enumerate(indices):
            x_hist_all[global_i] = irx.Interval(
                lower=x_hist_batch.lower[local_i],
                upper=x_hist_batch.upper[local_i],
            )

    # ── Overlap sum at each segment end, then take the minimum ───────────
    pair_i, pair_j = _pair_indices(n)

    def overlap_at_k(k):
        pos_ivls_k = [
            irx.Interval(
                lower=x_hist_all[i].lower[k, :2],
                upper=x_hist_all[i].upper[k, :2],
            )
            for i in range(n)
        ]
        lo_stack = jnp.stack([iv.lower for iv in pos_ivls_k])
        hi_stack = jnp.stack([iv.upper for iv in pos_ivls_k])
        ivl_i = irx.Interval(lower=lo_stack[pair_i], upper=hi_stack[pair_i])
        ivl_j = irx.Interval(lower=lo_stack[pair_j], upper=hi_stack[pair_j])
        return jnp.sum(jax.vmap(overlap_size_lax)(ivl_i, ivl_j))

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
                 learning_rate: float = 0.05,
                 num_iters: int = 200,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        """Plain gradient descent.  Returns (u_optimal, final_loss)."""
        if u_init is None:
            u_init = jnp.array([0.5, 0.3])
        u = u_init

        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)

            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss  = float(self.loss_fn(u))
                gnorm = float(jnp.linalg.norm(g))
                print(f"  Iter {i:4d}  loss={loss:.6f}  "
                      f"u=[{float(u[0]):+.3f},{float(u[1]):+.3f}]"
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
                overlaps[key] = float(overlap_size_lax(pos_ivls[i], pos_ivls[j]))

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
        self.loss_fn = jax.jit(_loss)            # (S,2) -> scalar
        self.grad_fn = jax.jit(jax.grad(_loss))  # (S,2) -> (S,2)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Multi-start Optimization
# ══════════════════════════════════════════════════════════════════════════════

def optimize_multistart(opt: SeparatingInputOptimizer,
                        num_restarts: int = 5,
                        learning_rate: float = 0.05,
                        num_iters: int = 200,
                        verbose: bool = False,
                        seed: int = 42,
                        ) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent using a pre-built optimizer.

    Returns (best_u, best_loss, best_stats).
    best_stats includes 'restart_times_s': list of wall-clock seconds per restart.
    """
    best_u, best_loss, best_stats = None, float('inf'), None
    key  = jax.random.PRNGKey(seed)
    restart_times: List[float] = []

    for r in range(num_restarts):
        key, subkey = jax.random.split(key)
        u_init = (jax.random.normal(subkey, (2,)) * 0.3
                  + jnp.array([0.5, 0.3]))

        if verbose:
            print(f"\n{'─'*60}\nRestart {r+1}/{num_restarts}\n{'─'*60}")

        _t0 = time.perf_counter()
        u_opt, loss = opt.optimize(u_init, learning_rate, num_iters, verbose)
        restart_times.append(time.perf_counter() - _t0)

        stats = opt.evaluate(u_opt)

        if loss < best_loss:
            best_loss, best_u, best_stats = loss, u_opt, stats
            if verbose:
                print(f"  → New best: {loss:.6f}")
        if loss == 0:
            break

    best_stats['restart_times_s'] = restart_times
    return best_u, best_loss, best_stats


def optimize_parallel(
    opt: SeparatingInputOptimizer,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    max_workers: Optional[int] = None,
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[jnp.ndarray, float, Dict]:
    """Parallel multi-start gradient descent using ThreadPoolExecutor.

    Returns (best_u, best_loss, best_stats).
    """
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    subkeys = jax.random.split(jax.random.PRNGKey(seed), num_restarts)

    # Trigger JIT compilation on caller thread to avoid compile races.
    _u_warm = jnp.array([0.5, 0.3])
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
        u_init = (jax.random.normal(key, (2,)) * 0.3
                  + jnp.array([0.5, 0.3]))
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


def optimize_parallel_gpu(
    opt: SeparatingInputOptimizer,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
):
    """GPU-parallel multi-start optimization via vmap + fori_loop.

    Returns: best_u (2,), best_loss, all_u_final (R,2), all_losses (R,)
    """
    key = jax.random.PRNGKey(seed)
    u0  = (jax.random.normal(key, (num_restarts, 2)) * 0.3
           + jnp.array([0.5, 0.3]))

    batched_loss = jax.vmap(opt.loss_fn)   # (R,2) -> (R,)
    batched_grad = jax.vmap(opt.grad_fn)   # (R,2) -> (R,2)

    def body(_, u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses  = batched_loss(u_final)

    best_idx  = jnp.argmin(losses)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_parallel_gpu_rejit(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
):
    """Convenience wrapper: build optimizer then call optimize_parallel_gpu."""
    key = jax.random.PRNGKey(seed)
    u0  = (jax.random.normal(key, (num_restarts, 2)) * 0.3
           + jnp.array([0.5, 0.3]))
    def loss_fn_ss(u):
        return separation_loss(
            u=u,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            num_steps=num_steps
        )
    grad_fn_ss = jax.grad(loss_fn_ss)
    batched_loss = jax.vmap(loss_fn_ss)   # (R,2) -> (R,)
    batched_grad = jax.vmap(grad_fn_ss)   # (R,2) -> (R,2)

    def body(_, u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses  = batched_loss(u_final)

    best_idx  = jnp.argmin(losses)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_multistep_gpu(
    opt: MultistepSequenceOptimizer,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
):
    """GPU-parallel multi-start for control sequences.

    Returns: best_u_seq (S,2), best_loss, all_u_seq_final (R,S,2), all_losses (R,)
    """
    key = jax.random.PRNGKey(seed)
    u0  = (jax.random.normal(key, (num_restarts, opt.num_segments, 2)) * 0.1
           + jnp.array([0.5, 0.3]))

    batched_loss = jax.vmap(opt.loss_fn)   # (R,S,2) -> (R,)
    batched_grad = jax.vmap(opt.grad_fn)   # (R,S,2) -> (R,S,2)

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses  = batched_loss(u_final)

    best_idx  = jnp.argmin(losses)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_multistep(
    scenarios: List[Scenario],
    x0_ivl: irx.Interval,
    dt: float,
    steps_per_segment: int,
    num_segments: int,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    num_restarts: int = 50,
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent over a sequence of control inputs.

    Optimises u_seq of shape (num_segments, 2), where u_seq[k] is applied
    to all scenarios for *steps_per_segment* Euler steps before switching to
    u_seq[k+1].  Total horizon = num_segments × steps_per_segment × dt s.

    Loss = minimum over segment ends of pairwise position-interval overlap sum.

    Returns
    -------
    (u_seq_opt, loss_opt, stats)
    u_seq_opt : (num_segments, 2) optimal control sequence
    loss_opt  : final overlap loss (m²)
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

    # Build stats dict.
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
            overlaps[key_ij] = float(overlap_size_lax(pos_ivls[i], pos_ivls[j]))
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


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Propagation with Observation-Based State Refinement
# ══════════════════════════════════════════════════════════════════════════════

def _obs_interval(x_ivl: irx.Interval, scenario: 'Scenario') -> irx.Interval:
    """Observed output interval: y = [px + δpx, py + δpy] for scenario."""
    return irx.Interval(
        lower=scenario.obs_scale * x_ivl.lower[:2] + scenario.obs_offset,
        upper=scenario.obs_scale * x_ivl.upper[:2] + scenario.obs_offset,
    )


def propagate_with_refinement(
    x_ivl: irx.Interval,
    u_seq: jnp.ndarray,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int = 2,
) -> jnp.ndarray:
    """Multi-step propagation with mid-horizon observation-based state refinement.

    Step 1 (computed once)
    ----------------------
    Propagate every scenario one Euler step from x_ivl using u_seq[0].
    The observed output for scenario i is y_i = [px + obs_offset_i[0],
    py + obs_offset_i[1]].
    First cost = Σ_{i<j} overlap(y_i, y_j)  in observed output space.

    Steps 2 … num_steps (repeated num_steps − 1 times, per pair)
    -------------------------------------------------------------
    Step k uses control u_seq[k].  For each ordered pair (i, j):
      1. Compute y(tk) = intersection of current observed output intervals.
      2. Reconstruct x_refined for each scenario by inverting its obs model:
             x_refined_k[0:2] = y(tk) − obs_offset_k
             x_refined_k[2]   = phi interval from the current propagated state
      3. Propagate x_refined one Euler step with u_seq[k].
      4. Accumulate overlap of the propagated observed output intervals.
      5. Carry the propagated (not refined) intervals as the new current state
         for this pair on the next iteration.

    After the first refinement the two scenarios in a pair have different state
    intervals (their px,py were tightened differently).  All subsequent steps
    therefore track state per-pair rather than per-scenario.

    All overlap computations are in observed output space throughout.

    Returns
    -------
    min over steps of per-step pairwise overlap sum (scalar, m²)

    Parameters
    ----------
    x_ivl     : initial state interval  [px, py, phi]
    u_seq     : control sequence  shape (num_steps, 2); u_seq[k] is applied
                at step k+1 (u_seq[0] drives step 1, u_seq[1] drives step 2, …)
    scenarios : list of Scenario objects; each must have .obs_offset (shape (2,))
    dt        : Euler step size (s)
    num_steps : total number of propagation steps (≥ 1).
                Step 1 is always computed; each additional step applies one
                refinement then one propagation.
    """
    n = len(scenarios)
    pair_i, pair_j = _pair_indices(n)
    xlen = x_ivl.lower.shape[0]   # 3

    # helpers: flatten/unflatten Interval <-> (2*xlen,) array
    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

    # ── Step 1: propagate all scenarios one Euler step with u_seq[0] ──────
    x1_ivls  = [euler_step(s.emb_system, x_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
    obs1_ivls = [_obs_interval(x1, s) for x1, s in zip(x1_ivls, scenarios)]

    obs1_lo = jnp.stack([iv.lower for iv in obs1_ivls])
    obs1_hi = jnp.stack([iv.upper for iv in obs1_ivls])
    step1_cost = jnp.sum(jax.vmap(overlap_size_lax)(
        irx.Interval(lower=obs1_lo[pair_i], upper=obs1_hi[pair_i]),
        irx.Interval(lower=obs1_lo[pair_j], upper=obs1_hi[pair_j]),
    ))

    x1_lo = jnp.stack([iv.lower for iv in x1_ivls])
    x1_hi = jnp.stack([iv.upper for iv in x1_ivls])
    pxi_arr = jnp.concatenate([x1_lo[pair_i], x1_hi[pair_i]], axis=-1)  # (n_pairs, 2*xlen)
    pxj_arr = jnp.concatenate([x1_lo[pair_j], x1_hi[pair_j]], axis=-1)

    obs_offset = jnp.stack([s.obs_offset for s in scenarios])   # (n, 2)
    obs_scale = jnp.stack([s.obs_scale for s in scenarios])     # (n,) or (n, 1)
    p_lo = jnp.stack([s.p_interval.lower for s in scenarios])
    p_hi = jnp.stack([s.p_interval.upper for s in scenarios])

    off_pi, off_pj = obs_offset[pair_i], obs_offset[pair_j]
    scale_pi, scale_pj = obs_scale[pair_i], obs_scale[pair_j]
    p_pi = irx.Interval(lower=p_lo[pair_i], upper=p_hi[pair_i])
    p_pj = irx.Interval(lower=p_lo[pair_j], upper=p_hi[pair_j])

    emb_sys = scenarios[0].emb_system   # shared across all scenarios (see module usage elsewhere)

    # ── Steps 2 … num_steps via fori_loop, vmapped over all n_pairs ──────
    # Was a Python "for idx, (i, j) in enumerate(pairs):" loop -- see
    # _pair_indices' docstring. NOTE: preserves the original's exact
    # (asymmetric) scale usage -- BOTH x_ref_i and x_ref_j divide by pair
    # j's obs_scale (scale_j1 below), matching the pre-existing code
    # verbatim (not "fixed" here; this refactor only changes batching, see
    # commit message).
    def step_one_pair(pxi_arr_1, pxj_arr_1, off_i1, scale_i1, p_i1,
                      off_j1, scale_j1, p_j1, u_k):
        x_curr_i = arr_to_ivl(pxi_arr_1)
        x_curr_j = arr_to_ivl(pxj_arr_1)

        obs_i = irx.Interval(lower=scale_i1 * x_curr_i.lower[:2] + off_i1,
                             upper=scale_i1 * x_curr_i.upper[:2] + off_i1)
        obs_j = irx.Interval(lower=scale_j1 * x_curr_j.lower[:2] + off_j1,
                             upper=scale_j1 * x_curr_j.upper[:2] + off_j1)

        y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
        y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
        has_overlap = jnp.all(y_hi >= y_lo)

        fallback = (x_curr_i.lower[:2] + x_curr_i.upper[:2]) / 2
        y_lo_safe = jnp.where(has_overlap, y_lo, fallback)
        y_hi_safe = jnp.where(has_overlap, y_hi, fallback)

        x_ref_i = irx.Interval(
            lower=jnp.array([
                (y_lo_safe[0] - off_i1[0]) / scale_j1[0],
                (y_lo_safe[1] - off_i1[1]) / scale_j1[0],
                x_curr_i.lower[2],
            ]),
            upper=jnp.array([
                (y_hi_safe[0] - off_i1[0]) / scale_j1[0],
                (y_hi_safe[1] - off_i1[1]) / scale_j1[0],
                x_curr_i.upper[2],
            ]),
        )
        x_ref_j = irx.Interval(
            lower=jnp.array([
                (y_lo_safe[0] - off_j1[0]) / scale_j1[0],
                (y_lo_safe[1] - off_j1[1]) / scale_j1[0],
                x_curr_j.lower[2],
            ]),
            upper=jnp.array([
                (y_hi_safe[0] - off_j1[0]) / scale_j1[0],
                (y_hi_safe[1] - off_j1[1]) / scale_j1[0],
                x_curr_j.upper[2],
            ]),
        )

        x_next_i = euler_step(emb_sys, x_ref_i, u_k, p_i1, dt)
        x_next_j = euler_step(emb_sys, x_ref_j, u_k, p_j1, dt)

        obs_next_i = irx.Interval(lower=scale_i1 * x_next_i.lower[:2] + off_i1,
                                  upper=scale_i1 * x_next_i.upper[:2] + off_i1)
        obs_next_j = irx.Interval(lower=scale_j1 * x_next_j.lower[:2] + off_j1,
                                  upper=scale_j1 * x_next_j.upper[:2] + off_j1)
        raw_cost = overlap_size_lax(obs_next_i, obs_next_j)
        pair_cost = jnp.where(has_overlap, raw_cost, jnp.array(0.0))
        return ivl_to_arr(x_next_i), ivl_to_arr(x_next_j), pair_cost

    step_all_pairs = jax.vmap(step_one_pair, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None))

    def step_body(k, carry):
        pxi_arr, pxj_arr, min_cost = carry
        u_k = u_seq[k + 1]   # u_seq[0] was used at step 1
        new_pxi, new_pxj, pair_costs = step_all_pairs(
            pxi_arr, pxj_arr, off_pi, scale_pi, p_pi, off_pj, scale_pj, p_pj, u_k
        )
        step_cost = jnp.sum(pair_costs)
        return (new_pxi, new_pxj, jnp.minimum(min_cost, step_cost))

    init_carry = (pxi_arr, pxj_arr, step1_cost)
    _, _, min_cost_final = jax.lax.fori_loop(0, num_steps - 1, step_body, init_carry)
    return min_cost_final


def refined_overlap_loss(
    u_seq: jnp.ndarray,
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int = 2,
) -> jnp.ndarray:
    """Separation loss using multi-step propagation with state refinement.

    Incorporates observation-based state refinement between steps.  Minimising
    this loss finds a control sequence that keeps fault scenarios separated in
    observed output space over the full num_steps horizon, accounting for the
    information gained after each step.

    Parameters
    ----------
    u_seq     : control sequence  shape (num_steps, 2); u_seq[k] is applied
                at step k+1
    x0_ivl    : initial state interval
    scenarios : list of Scenario objects (must have .obs_offset)
    dt        : Euler step size (s)
    num_steps : total propagation steps (step 1 + num_steps−1 refinement steps)

    Returns
    -------
    Scalar overlap area (m²); lower is better.
    """
    return propagate_with_refinement(x0_ivl, u_seq, scenarios, dt, num_steps)


def optimize_refined_gpu(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int = 2,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent minimising refined_overlap_loss.

    Follows the same vmap + lax.fori_loop pattern as optimize_parallel_gpu_rejit
    but uses the multi-step refined loss instead of the single-step loss.

    Parameters
    ----------
    x0_ivl        : initial state interval
    scenarios     : list of Scenario objects (must have .obs_offset)
    dt            : Euler step size (s)
    num_steps     : propagation steps passed to refined_overlap_loss
    num_restarts  : number of parallel gradient-descent restarts
    learning_rate : gradient step size
    num_iters     : number of gradient steps per restart
    seed          : PRNG seed

    Returns
    -------
    (best_u_seq, best_loss, all_u_seq_final, all_losses)
    best_u_seq      : (num_steps, 2) optimal control sequence
    best_loss       : scalar — refined overlap loss at best_u_seq
    all_u_seq_final : (num_restarts, num_steps, 2) final sequences for every restart
    all_losses      : (num_restarts,) final loss for every restart
    """
    key = jax.random.PRNGKey(seed)
    # Initialise each restart as a (num_steps, 2) sequence of random controls.
    u0 = (
        jax.random.normal(key, (num_restarts, num_steps, 2)) * 0.3
        + jnp.array([0.5, 0.3])
    )

    def loss_fn(u_seq):
        return refined_overlap_loss(
            u_seq, x0_ivl=x0_ivl, scenarios=scenarios, dt=dt, num_steps=num_steps
        )

    grad_fn      = jax.grad(loss_fn)
    batched_loss = jax.vmap(loss_fn)   # (R, N, 2) → (R,)
    batched_grad = jax.vmap(grad_fn)   # (R, N, 2) → (R, N, 2)

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses  = batched_loss(u_final)

    best_idx   = jnp.argmin(losses)
    best_u_seq = u_final[best_idx]
    best_loss  = losses[best_idx]
    return best_u_seq, best_loss, u_final, losses


def refined_overlap_cbf_loss(
    u_seq: jnp.ndarray,
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,   # (N, 3) — each row [cx, cy, r_obs]
    cbf_weight: float = 1.0,
) -> jnp.ndarray:
    """Separation + CBF loss for an open-loop control sequence.

    Combines the refined separation loss (from propagate_with_refinement)
    with a soft CBF penalty summed over all scenarios and all steps:

        loss = min_k overlap_k  +  cbf_weight · Σ_s Σ_k cbf_penalty(x_s_k)

    The CBF barrier function is
        h(x) = (px − cx)² + (py − cy)² − r_obs²  ≥  0

    and its penalty over a position-interval box is computed at the
    closest point to the obstacle centre (convex minimisation):
        penalty = max(0, −h_min)

    CBF propagation uses unrefined per-scenario intervals (conservative)
    and is implemented with lax.fori_loop to keep the graph O(1) in num_steps.

    Parameters
    ----------
    u_seq      : (num_steps, 2) open-loop control sequence
    x0_ivl     : initial state interval
    scenarios  : list of Scenario objects
    dt         : Euler step size (s)
    num_steps  : total propagation steps
    obstacles  : (N, 3) — one row [cx, cy, r_obs] per obstacle
    cbf_weight : weight on the CBF penalty term
    """
    # ── separation loss (handles refinement internally via fori_loop) ─────
    sep_loss = propagate_with_refinement(x0_ivl, u_seq, scenarios, dt, num_steps)

    # ── CBF penalty: unrefined per-scenario propagation ───────────────────
    xlen = x0_ivl.lower.shape[0]

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

    def _cbf_penalty_ivl(x_ivl: irx.Interval) -> jnp.ndarray:
        """Sum of max(0, -h_min) over all obstacles for one state interval."""
        def single_obs(obs):
            cx, cy, r_obs = obs[0], obs[1], obs[2]
            px_star = jnp.clip(cx, x_ivl.lower[0], x_ivl.upper[0])
            py_star = jnp.clip(cy, x_ivl.lower[1], x_ivl.upper[1])
            h_min = (px_star - cx) ** 2 + (py_star - cy) ** 2 - r_obs ** 2
            return jnp.maximum(0.0, -h_min)
        return jnp.sum(jax.vmap(single_obs)(obstacles))

    cbf_pen = jnp.array(0.0)
    # Python loop over scenarios is fine (fixed small count, not scaled by num_steps)
    for s in scenarios:
        def body(k, carry, _s=s):
            x_arr, pen = carry
            x      = arr_to_ivl(x_arr)
            x_next = euler_step(_s.emb_system, x, u_seq[k], _s.p_interval, dt)
            return ivl_to_arr(x_next), pen + _cbf_penalty_ivl(x_next)

        _, s_pen = jax.lax.fori_loop(
            0, num_steps, body, (ivl_to_arr(x0_ivl), jnp.array(0.0))
        )
        cbf_pen = cbf_pen + s_pen

    return sep_loss + cbf_weight * cbf_pen


def optimize_refined_cbf_gpu(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    obstacles: jnp.ndarray,          # (N, 3) — [cx, cy, r_obs] per row
    cbf_weight: float = 1.0,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent minimising refined_overlap_cbf_loss.

    Identical structure to optimize_refined_gpu but optimises the combined
    separation + CBF loss so the returned control sequence simultaneously
    separates the fault scenarios and avoids the obstacle set.

    Returns
    -------
    (best_u_seq, best_loss, all_u_seq_final, all_losses)
    """
    key = jax.random.PRNGKey(seed)
    u0  = (
        jax.random.normal(key, (num_restarts, num_steps, 2)) * 0.3
        + jnp.array([0.5, 0.3])
    )

    def loss_fn(u_seq):
        return refined_overlap_cbf_loss(
            u_seq,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            num_steps=num_steps,
            obstacles=obstacles,
            cbf_weight=cbf_weight,
        )

    batched_loss = jax.vmap(loss_fn)
    batched_grad = jax.vmap(jax.grad(loss_fn))

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final    = jax.lax.fori_loop(0, num_iters, body, u0)
    losses     = batched_loss(u_final)
    best_idx   = jnp.argmin(losses)
    best_u_seq = u_final[best_idx]
    best_loss  = losses[best_idx]
    return best_u_seq, best_loss, u_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("FAULTY CAR SEPARATING INPUT — immrax")
    print("=" * 70)

    # ── Scenarios ──────────────────────────────────────────────────────────
    scenarios = create_scenarios()
    print(f"\n{len(scenarios)} fault scenarios:")
    for s in scenarios:
        print(f"  • {s.name}")
        print(f"    p ∈ [{np.array(s.p_interval.lower)}, {np.array(s.p_interval.upper)}]")

    # ── Initial state interval ─────────────────────────────────────────────
    x0_ivl = irx.icentpert(
        jnp.array([0.1, 0.1, 0.0]),
        jnp.array([0.1, 0.1, 0.1]),
    )
    print(f"\nInitial state interval:")
    print(f"  px ∈ [{float(x0_ivl.lower[0]):.3f}, {float(x0_ivl.upper[0]):.3f}] m")
    print(f"  py ∈ [{float(x0_ivl.lower[1]):.3f}, {float(x0_ivl.upper[1]):.3f}] m")
    print(f"  φ  ∈ [{float(x0_ivl.lower[2]):.3f}, {float(x0_ivl.upper[2]):.3f}] rad")

    # ── Optimization parameters ────────────────────────────────────────────
    dt, num_steps = 0.1, 50
    print(f"\nPropagation: {num_steps} × {dt} s = {num_steps * dt} s total")
    print("Separation objective: position space (px, py)")

    # ── Run optimization ───────────────────────────────────────────────────
    print("\nRunning multi-start optimization …\n")
    opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt, num_steps)
    u_opt, loss_opt, stats = optimize_multistart(
        opt,
        num_restarts=5,
        learning_rate=0.05,
        num_iters=200,
        verbose=True,
    )

    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOptimal separating input:")
    print(f"  v     = {float(u_opt[0]):+.4f} m/s")
    print(f"  omega = {float(u_opt[1]):+.4f} rad/s")
    print(f"\nTotal overlap: {loss_opt:.6f} m²")

    print(f"\nPairwise overlaps:")
    for k, v in stats['pairwise_overlaps'].items():
        print(f"  {k}: {v:.6f} m²")

    print(f"\nPosition interval volumes:")
    for k, v in stats['volumes'].items():
        print(f"  {k}: {v:.6f} m²")

    print(f"\nFinal position intervals (what an observer measures):")
    for s, iv in zip(scenarios, stats['position_intervals']):
        wx = float(iv.upper[0] - iv.lower[0])
        wy = float(iv.upper[1] - iv.lower[1])
        print(f"  {s.name}:")
        print(f"    px ∈ [{float(iv.lower[0]):+.4f}, {float(iv.upper[0]):+.4f}] m  (width {wx:.4f})")
        print(f"    py ∈ [{float(iv.lower[1]):+.4f}, {float(iv.upper[1]):+.4f}] m  (width {wy:.4f})")

    print("\n" + "=" * 70)
    print("Scenario descriptions")
    print("=" * 70)
    print("""
  Nominal
    alpha = 1 (full steering authority) and perfect omega sensor.

  Actuator Fault
    alpha ∈ [0, 0.5]: the steering rate is reduced by 50–100%.
    The car turns less than commanded → different trajectory.

  Sensor Fault
    Same dynamics as nominal (alpha = 1), but the position sensors report
    px + 0.2 m and py + 0.2 m instead of the true position.  The state
    evolves identically to nominal; faults are detectable only in output space.
    """)
    print("✓ Done")
