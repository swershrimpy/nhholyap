"""
GO2 Separating Input – Active Fault Diagnosis with immrax
==========================================================
Reachable-set-based active fault diagnosis for the Unitree Go2 using
immrax for interval arithmetic and natural-embedding propagation.

Scenarios
---------
  Nominal               α = 1,          perfect sensors
  Actuator Fault        α ∈ [0.6, 0.8]  reduced yaw-rate authority
  Sensor Fault          vy_meas = 0 + noise ∈ [−ε, +ε]
                        (faulty lateral-velocity sensor; robot's dead-reckoned
                         position drifts because it integrates vy_meas ≈ 0
                         instead of the true commanded vy)

Follows the immrax usage pattern from admire/admire_fault_demo.ipynb and the
Go2 scenario layout from go2_separating_input_demo.ipynb.
"""

import sys
import os
from pathlib import Path

# File is at examples/go2/<name>.py  →  parents[1] = examples/
_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np
from functools import partial
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from faulty_car.interval_functions import overlap_size_lax

# Control input box constraints: vx, vy ∈ [-1, 1]; ω unconstrained.
_U_LO = jnp.array([-.6, -.6, -.6])
_U_HI = jnp.array([ .6,  .6,  .6])


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    """Project a control input (or batch) onto the feasible box.

    Works for any leading batch dimensions — clips only the last axis.
    vx ∈ [-1, 1], vy ∈ [-1, 1], ω unconstrained.
    """
    return jnp.clip(u, _U_LO, _U_HI)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  System Definitions
# ══════════════════════════════════════════════════════════════════════════════

class Go2NomActSystem(irx.System):
    """Unicycle dynamics for the nominal and actuator-fault scenarios.

    State   x = [px, py, θ]         (position + heading, m / rad)
    Control u = [vx, vy, ω]         (body-frame velocities + yaw rate)
    Params  p = [α, beta]                  (actuator effectiveness; α = 1 → nominal)

    Dynamics:
        ṗx = vx·cos θ − α·vy·sin θ
        ṗy = vx·sin θ + α·vy·cos θ
        θ̇  = beta * ω
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

    When the lateral-velocity sensor fails it reports vy_meas ≈ 0 + noise
    instead of the true commanded vy.  The robot's onboard odometry integrates
    this corrupted measurement, so the *estimated* position diverges from the
    true position.

    State   x = [p̂x, p̂y, θ̂]        (robot's dead-reckoned pose estimate)
    Control u = [vx, vy_cmd, ω]     (commanded; vy_cmd is *ignored* here)
    Params  p = [vy_noise]           (vy reading from faulty sensor,
                                      vy_noise ∈ [−ε, +ε])
    vy_corrupted = (1 - omega ** 2) * alpha * vy + omega ** 2 * vy_noise
    Dynamics (what the odometry integrates):
        ṗ̂x = vx·cos θ̂ − vy_corrupted·sin θ̂
        ṗ̂y = vx·sin θ̂ + vy_corrupted·cos θ̂
        θ̂̇  = beta * ω
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


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Fault Scenarios
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One fault scenario: a named (embedding, parameter-interval) pair."""
    name: str
    emb_system: object        # irx.natemb(…) result
    p_interval: irx.Interval  # parameter interval for this scenario


# Module-level singletons (created once, reused across calls)
_NOM_ACT_SYS = Go2NomActSystem()
_NOM_ACT_EMB = irx.natemb(_NOM_ACT_SYS)

_SF_SYS = Go2SensorFaultSystem()
_SF_EMB = irx.natemb(_SF_SYS)


def create_scenarios(
    actuator_alpha_lo: float = 0.60,
    actuator_alpha_hi: float = 0.80,
    actuator_beta_low: float = 0.60,
    actuator_beta_high: float = .80,
    sensor_noise_bound: float = 0.25,
) -> List[Scenario]:
    """Return the three fault scenarios used throughout this module.

    Parameters
    ----------
    actuator_alpha_lo / hi:
        Range of the actuator effectiveness parameter α (nominal = 1).
    sensor_noise_bound:
        Half-width of the vy-sensor noise interval; vy_noise ∈ [-ε, +ε].
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
            name="Actuator and Sensor Fault",
            emb_system=_SF_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([-sensor_noise_bound, actuator_alpha_lo, actuator_beta_low]),
                upper=jnp.array([ sensor_noise_bound, actuator_alpha_hi, actuator_beta_high]),
            )
        )
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Interval Propagation
# ══════════════════════════════════════════════════════════════════════════════

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    """One forward-Euler interval step via the natural embedding.

    Follows the faulty-car style:
        x_new_ut = f_emb(t=0, x_ut, u, p) * dt  +  x_ut
        x_new    = ut2i(x_new_ut)

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

    Returns the full 3-D state interval [px, py, θ] at the end of the horizon.
    JAX-compatible (uses lax.fori_loop so the function is JIT-able and
    differentiable w.r.t. *u*).
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

    u_seq has shape (num_segments, 3).  u_seq[k] is held constant for
    *steps_per_segment* Euler steps, then u_seq[k+1] takes over, etc.
    Total horizon = num_segments × steps_per_segment × dt seconds.
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

    Minimising this loss maximises the separation of the three reachable sets
    in the (px, py) output space, making fault diagnosis easier.

    Parameters
    ----------
    u         : constant control input [vx, vy, ω]
    x0_ivl    : initial state interval
    scenarios : list of Scenario objects
    dt        : Euler step size (s)
    num_steps : number of Euler steps

    Returns
    -------
    Scalar overlap volume (m²); lower is better.
    """
    pos_ivls = [
        position_interval(propagate_scenario(x0_ivl, u, s, dt, num_steps))
        for s in scenarios
    ]
    n = len(pos_ivls)
    total = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            total = total + overlap_size_lax(pos_ivls[i], pos_ivls[j])
    return total


def separation_loss_multistep(u_seq: jnp.ndarray,
                              x0_ivl: irx.Interval,
                              scenarios: List[Scenario],
                              dt: float,
                              steps_per_segment: int) -> jnp.ndarray:
    """Min over segments of the pairwise position-interval overlap sum.

    For each segment end k, computes the sum of pairwise position-interval
    overlaps across all scenarios.  Returns the minimum over all K segment
    ends — i.e., the loss is zero if the scenarios are fully separated at
    ANY point in the trajectory.

    Scenarios that share the same emb_system are propagated in parallel via
    jax.vmap over their stacked p_intervals.

    Parameters
    ----------
    u_seq            : (num_segments, 3) sequence of control inputs
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
    # emb_id -> (emb_sys, [global_indices], [p_ivl])
    emb_groups: Dict[int, tuple] = {}
    for idx, s in enumerate(scenarios):
        eid = id(s.emb_system)
        if eid not in emb_groups:
            emb_groups[eid] = (s.emb_system, [], [])
        emb_groups[eid][1].append(idx)
        emb_groups[eid][2].append(s.p_interval)

    # ── Propagate each group in parallel (vmap over p_ivl) ───────────────
    # x_hist_all[i]: Interval with lower/upper shape (num_segments, state_dim)
    x_hist_all: List[irx.Interval] = [None] * n
    for emb_sys, indices, p_ivls in emb_groups.values():
        # Stack p_intervals: Interval with lower/upper shape (B, param_dim)
        p_batch = irx.Interval(
            lower=jnp.stack([p.lower for p in p_ivls]),
            upper=jnp.stack([p.upper for p in p_ivls]),
        )

        def prop_one(p_ivl_single):
            return _propagate_history(
                x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment
            )

        # x_hist_batch: Interval with lower/upper shape (B, num_segments, state_dim)
        x_hist_batch = jax.vmap(prop_one)(p_batch)

        for local_i, global_i in enumerate(indices):
            x_hist_all[global_i] = irx.Interval(
                lower=x_hist_batch.lower[local_i],
                upper=x_hist_batch.upper[local_i],
            )

    # ── Overlap sum at each segment end, then take the minimum ───────────
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
                total = total + overlap_size_lax(pos_ivls_k[i], pos_ivls_k[j])
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
        """Plain gradient descent.  Returns (u_optimal, final_loss)."""
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
    JIT-ted loss and grad functions) and passes it here.  Re-using the same
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

    def body(_, u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    # One compiled loop on device.
    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
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
    """
    Revised version of GPU-parallel multistart optimization for a single controller.
    """
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

    Mirrors optimize_multistart_gpu but for u_seq with shape (num_segments, 3).
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

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    # One compiled loop on device.
    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
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
    return optimize_multistep_gpu(
        MultistepSequenceOptimizer(
            scenarios=scenarios,
            x0_ivl=x0_ivl,
            dt=dt,
            steps_per_segment=steps_per_segment,
            num_segments=num_segments,
        ),
        num_restarts=num_restarts,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=seed
    )


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
    u_seq[k+1].  Total horizon = num_segments × steps_per_segment × dt s.

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
# 6.  CBF Obstacle Avoidance — Lagrangian-Penalised Separation
# ══════════════════════════════════════════════════════════════════════════════

# Default obstacle: 0.5 m radius circle centred at (−1, 0) in the world frame.
_OBS_CENTER = jnp.array([-1.0, 0.0])
_OBS_RADIUS = 0.5


def obstacle_cbf_value(
    pos_ivl: irx.Interval,
    obs_center: jnp.ndarray = _OBS_CENTER,
    obs_radius: float = _OBS_RADIUS,
) -> jnp.ndarray:
    """Minimum of the CBF  h(p) = ‖p − c‖² − r²  over a position interval.

    h(p) ≥ 0 iff p lies outside the circular obstacle (safe region).  For an
    axis-aligned rectangular position interval the minimum of h is attained at
    the point of the rectangle closest to the obstacle centre:

        closest = clip(obs_center, pos_ivl.lower, pos_ivl.upper)
        h_min   = ‖closest − obs_center‖² − obs_radius²

    Parameters
    ----------
    pos_ivl    : 2-D position interval  [px_lo, py_lo] … [px_hi, py_hi]
    obs_center : (2,) obstacle centre   [cx, cy]
    obs_radius : obstacle radius  r  (m)

    Returns
    -------
    Scalar h_min.  Negative → interval intersects the obstacle (unsafe).
    Zero or positive → interval is entirely outside the obstacle (safe).
    """
    closest = jnp.clip(obs_center, pos_ivl.lower, pos_ivl.upper)
    dist_sq = jnp.sum((closest - obs_center) ** 2)
    return dist_sq - obs_radius ** 2


def separation_loss_cbf_multistep(
    u_seq: jnp.ndarray,
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    steps_per_segment: int,
    obs_center: jnp.ndarray = _OBS_CENTER,
    obs_radius: float = _OBS_RADIUS,
    cbf_lambda: float = 10.0,
) -> jnp.ndarray:
    """Lagrangian-penalised separation objective with CBF obstacle avoidance.

    Combines the fault-separating objective of separation_loss_multistep with a
    Control Barrier Function (CBF) obstacle-avoidance constraint via Lagrangian
    dualization:

        L(u; λ) = min_k Σ_{i<j} overlap(P_i^k, P_j^k)
                + λ · Σ_{k,i} max(0, −h_min(P_i^k))

    where P_i^k is the 2-D position interval of scenario i at segment end k,
    and h_min(P) = min_{p ∈ P} h(p) with CBF h(p) = ‖p − c‖² − r².

    The first term drives the reachable sets apart (fault diagnosis).  The
    second term penalises any scenario interval that enters the obstacle; the
    Lagrange multiplier λ = cbf_lambda trades off the two objectives.  A
    single propagation pass is shared between both terms.

    Parameters
    ----------
    u_seq            : (num_segments, 3) control sequence  [vx, vy, ω]
    x0_ivl           : initial state interval
    scenarios        : list of Scenario objects
    dt               : Euler step size (s)
    steps_per_segment: Euler steps held per control segment
    obs_center       : (2,) obstacle centre  (default: (−1, 0))
    obs_radius       : obstacle radius in metres  (default: 0.5)
    cbf_lambda       : Lagrange multiplier λ; larger → stricter obstacle safety

    Returns
    -------
    Scalar Lagrangian value.  Minimising drives separation and safety together.
    """
    num_segments = u_seq.shape[0]
    n = len(scenarios)

    # ── Group scenarios by emb_system (mirrors separation_loss_multistep) ──
    emb_groups: Dict[int, tuple] = {}
    for idx, s in enumerate(scenarios):
        eid = id(s.emb_system)
        if eid not in emb_groups:
            emb_groups[eid] = (s.emb_system, [], [])
        emb_groups[eid][1].append(idx)
        emb_groups[eid][2].append(s.p_interval)

    # ── Single propagation pass (vmap over p_ivl within each group) ──────
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

    # ── Separation loss  (min over segments of pairwise overlap sum) ─────
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
                total = total + overlap_size_lax(pos_ivls_k[i], pos_ivls_k[j])
        return total

    segment_overlaps = jnp.stack([overlap_at_k(k) for k in range(num_segments)])
    overlap_loss = jnp.min(segment_overlaps)

    # ── CBF penalty  (sum over all segments and scenarios) ───────────────
    # For each position interval P_i^k we compute h_min = min_{p ∈ P_i^k} h(p)
    # (worst-case proximity to the obstacle), then penalise any negative value
    # with relu(−h_min).  Summing over all (k, i) pairs ensures the trajectory
    # of every scenario stays clear of the obstacle at every segment boundary.
    cbf_total = jnp.array(0.0)
    for k in range(num_segments):
        for i in range(n):
            pos_ivl_ki = irx.Interval(
                lower=x_hist_all[i].lower[k, :2],
                upper=x_hist_all[i].upper[k, :2],
            )
            h_min = obstacle_cbf_value(pos_ivl_ki, obs_center, obs_radius)
            cbf_total = cbf_total + jnp.maximum(jnp.array(0.0), -h_min)

    return overlap_loss + cbf_lambda * cbf_total


class CBFMultistepOptimizer:
    """Multistep sequence optimizer with CBF obstacle-avoidance constraints.

    Wraps separation_loss_cbf_multistep (the Lagrangian-penalised objective)
    in JIT-compiled loss and gradient callables.  Follows the same interface
    as MultistepSequenceOptimizer so it can be passed to optimize_multistep_gpu.
    """

    def __init__(
        self,
        scenarios: List[Scenario],
        x0_ivl: irx.Interval,
        dt: float,
        steps_per_segment: int,
        num_segments: int,
        obs_center: jnp.ndarray = _OBS_CENTER,
        obs_radius: float = _OBS_RADIUS,
        cbf_lambda: float = 10.0,
    ):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments
        self.obs_center = obs_center
        self.obs_radius = obs_radius
        self.cbf_lambda = cbf_lambda

        _loss = partial(
            separation_loss_cbf_multistep,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            steps_per_segment=steps_per_segment,
            obs_center=obs_center,
            obs_radius=obs_radius,
            cbf_lambda=cbf_lambda,
        )
        self.loss_fn = jax.jit(_loss)            # (S, 3) → scalar
        self.grad_fn = jax.jit(jax.grad(_loss))  # (S, 3) → (S, 3)


def optimize_multistep_cbf(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    steps_per_segment: int,
    num_segments: int,
    obs_center: jnp.ndarray = _OBS_CENTER,
    obs_radius: float = _OBS_RADIUS,
    cbf_lambda: float = 10.0,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    seed: int = 42,
):
    """GPU-parallel multi-start optimisation with CBF obstacle avoidance.

    Solves the Lagrangian-penalised problem:

        min_{u_seq}  separation_loss(u_seq) + λ · cbf_penalty(u_seq)

    where the CBF encodes the circular obstacle as a safe-set constraint:

        h(p) = ‖p − obs_center‖² − obs_radius² ≥ 0   (safe iff outside circle)

    The penalty term  λ · Σ_{k,i} max(0, −h_min(P_i^k))  is a Lagrangian
    relaxation of the hard CBF constraint; each P_i^k is the position interval
    of scenario i at segment end k, and h_min(P) is its worst-case (minimum)
    CBF value.  Increasing cbf_lambda enforces safety more strictly at the
    possible cost of higher residual overlap.

    Mirrors optimize_multistep_gpu_rejit but uses CBFMultistepOptimizer in
    place of MultistepSequenceOptimizer.

    Parameters
    ----------
    x0_ivl           : initial state interval
    scenarios        : list of Scenario objects
    dt               : Euler step size (s)
    steps_per_segment: Euler steps held per control segment
    num_segments     : number of control segments in the sequence
    obs_center       : (2,) obstacle centre in world frame  (default: (−1, 0) m)
    obs_radius       : obstacle radius in metres  (default: 0.5)
    cbf_lambda       : Lagrange multiplier λ; larger → stricter obstacle avoidance
    num_restarts     : number of parallel random initialisations
    learning_rate    : gradient descent step size
    num_iters        : number of gradient steps per restart
    seed             : PRNG seed for random initialisation

    Returns
    -------
    (best_u_seq, best_loss, all_u_seq_final, all_losses)
    best_u_seq      : (num_segments, 3) optimal control sequence
    best_loss       : Lagrangian value at best_u_seq
    all_u_seq_final : (num_restarts, num_segments, 3) final iterates for all restarts
    all_losses      : (num_restarts,) final Lagrangian values
    """
    """GPU-parallel multi-start optimization for control sequences.

    Mirrors optimize_multistart_gpu but for u_seq with shape (num_segments, 3).
    Returns:
      best_u_seq, best_loss, all_u_seq_final, all_final_losses
    """
    key = jax.random.PRNGKey(seed)
    u0 = (
        jax.random.normal(key, (num_restarts, num_segments, 3)) * 0.1
        + jnp.array([0.5, 0.0, 0.3])
    )
    def loss_fn_cbf(u):
        return separation_loss_cbf_multistep(
            u_seq=u,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            steps_per_segment=steps_per_segment,
            obs_center=obs_center,
            obs_radius=obs_radius,
            cbf_lambda=cbf_lambda,
        ) 
    
    grad_fn_cbf = jax.grad(loss_fn_cbf)

    # Vectorize loss/grad across restart axis.
    batched_loss = jax.vmap(loss_fn_cbf)   # (R,S,3) -> (R,)
    batched_grad = jax.vmap(grad_fn_cbf)   # (R,S,3) -> (R,S,3)

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    # One compiled loop on device.
    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses = batched_loss(u_final)

    best_idx = jnp.argmin(losses)
    best_u = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("GO2 SEPARATING INPUT — immrax")
    print("=" * 70)

    # ── Scenarios ──────────────────────────────────────────────────────────
    scenarios = create_scenarios()
    print(f"\n{len(scenarios)} fault scenarios:")
    for s in scenarios:
        print(f"  • {s.name}")
        print(f"    p ∈ [{np.array(s.p_interval.lower)}, {np.array(s.p_interval.upper)}]")

    # ── Initial state interval ─────────────────────────────────────────────
    x0_ivl = irx.Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([ 0.05,  0.05,  0.02]),
    )
    print(f"\nInitial state interval:")
    print(f"  px ∈ [{float(x0_ivl.lower[0]):.3f}, {float(x0_ivl.upper[0]):.3f}] m")
    print(f"  py ∈ [{float(x0_ivl.lower[1]):.3f}, {float(x0_ivl.upper[1]):.3f}] m")
    print(f"  θ  ∈ [{float(x0_ivl.lower[2]):.3f}, {float(x0_ivl.upper[2]):.3f}] rad")

    # ── Optimization parameters ────────────────────────────────────────────
    dt, num_steps = 0.5, 10
    print(f"\nPropagation: {num_steps} × {dt} s = {num_steps * dt} s total")
    print("Separation objective: position space (px, py)")

    # ── Run optimization ───────────────────────────────────────────────────
    print("\nRunning multi-start optimization …\n")
    opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt, num_steps)
    u_opt, loss_opt, stats = optimize_multistart(
        opt,
        num_restarts=3,
        learning_rate=0.01,
        num_iters=150,
        verbose=True,
    )

    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOptimal separating input:")
    print(f"  vx    = {float(u_opt[0]):+.4f} m/s")
    print(f"  vy    = {float(u_opt[1]):+.4f} m/s")
    print(f"  omega = {float(u_opt[2]):+.4f} rad/s")
    print(f"\nTotal overlap: {loss_opt:.6f} m²")

    print(f"\nPairwise overlaps:")
    for k, v in stats['pairwise_overlaps'].items():
        print(f"  {k}: {v:.6f} m²")

    print(f"\nPosition interval volumes:")
    for k, v in stats['volumes'].items():
        print(f"  {k}: {v:.6f} m²")

    print(f"\nFinal output intervals (what an observer measures):")
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
    α = 1 (full yaw authority) and correct sensors.
    Output = true dead-reckoned position.

  Actuator Fault
    α ∈ [0.6, 0.8]: the turn rate is reduced by 20–40%.
    The robot steers less than commanded → different trajectory.

  Sensor Fault (vy = 0 + noise)
    The lateral-velocity sensor returns ≈ 0 instead of the true vy.
    The robot's odometry integrates vy_meas ≈ 0, so the estimated py
    differs from the truth whenever vy ≠ 0 is commanded.
    The larger |vy| in the optimal input, the stronger this signature.
    """)
    print("✓ Done")
