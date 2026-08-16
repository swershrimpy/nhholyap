"""
Crazyflie controller discrimination via legitimate waypoint control
====================================================================
This module achieves the SAME discrimination outcome as
`crazyflie_chain_controllers.py`'s GPS-spoof pipeline — identifying which of
4 candidate outer-loop controllers is flying a Crazyflie — but through
**direct, legitimate flight control** (a designed waypoint sequence) with no
sensor deception at all.

Key structural insight (PLAN.md §9): in `ChainControllerSystem.f`, the
tracking error is `e_p = (pos + b) - rp`. That is algebraically identical to
`e_p = pos - (rp - b)` — corrupting what the controller perceives and moving
the commanded target enter the closed-loop ODE identically. So the existing
discrimination machinery works unchanged; what changes is:
  (1) The reference is now genuinely TIME-VARYING (a real flown mission)
  (2) Box constraints are arena-scale, not stealth-scale
  (3) Deployment goes through QPS's `set_poses()` waypoint API

This module imports and reuses the System-agnostic helpers from
`crazyflie_chain_controllers.py` (euler_step, observed_output,
separation_loss_multistep, propagate_with_refinement, discriminate_controller,
_overlap_volume, _output_overlap_volume) rather than duplicating them.

System
------
  State   x = [p(3), v(3), a(3), j(3), integ_e_p(3), cmd(3)]   (18 states)
  Control u = ref15(15) — instantaneous reference [rp,rv,ra,rj,rs] at each
              step, sampled from `build_mission_reference`
  Params  p = [k_p, k_v, k_a, k_i, k_j, base_flag]   (6,), same as
              ChainControllerSystem

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/crazyflie_waypoint_controllers.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import immrax as irx

from adaptive_spoofing.crazyflie_chain_controllers import (
    _CANDIDATE_THETA, QPS_DT, Scenario,
    observed_output, _overlap_volume, _output_overlap_volume,
    _DEFAULT_W_BAR,
)
from adaptive_spoofing.crazyflie_waypoint_trajectory import (
    traj_coeffs_from_waypoint, sample_reference, build_mission_reference,
)
from adaptive_spoofing.crazyflie_12d import ARENA_X_HALF, ARENA_Y_HALF, ARENA_Z_RANGE

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# RQ3 cruise mission waypoints (from experiments/RQ3/rq3_mission.py)
P0 = jnp.array([-1.2, 0.0, 1.0])   # start
P1 = jnp.array([1.2, 0.0, 1.0])    # end
ALT = 1.0

# QPS default hop duration (quadrotarium.py:152)
DEFAULT_T_HOP = 4.0

# Default number of intermediate waypoints
DEFAULT_K = 5

# Per-waypoint offset bounds (meters): how far from the nominal line each
# waypoint can be displaced for discrimination. Arena-scale, not stealth-scale.
DEFAULT_OFFSET_LIM_XY = 0.5
DEFAULT_OFFSET_LIM_Z = 0.3


# ══════════════════════════════════════════════════════════════════════════════
# 1. WaypointTrackingSystem
# ══════════════════════════════════════════════════════════════════════════════

class WaypointTrackingSystem(irx.System):
    """18-state chain-of-integrator outer loop with a masked-theta controller
    law — identical to ChainControllerSystem but with TIME-VARYING reference.

    State   x = [p(3), v(3), a(3), j(3), integ_e_p(3), cmd(3)]
    Control u = ref15(15)  — instantaneous [rp, rv, ra, rj, rs] reference
    Params  p = [k_p, k_v, k_a, k_i, k_j, base_flag]   (6,)

    Key difference from ChainControllerSystem: ref15 is passed as `u`
    (control input) each step, not baked into the instance. This makes the
    reference time-varying — each step receives a different slice of the
    mission reference — and enables differentiating through the waypoint →
    reference → dynamics chain.

    No spoof bias: `e_p = pos - rp` (true position, no deception).
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 18

    def f(self, t, x, u, p):
        # u is ref15: [rp(3), rv(3), ra(3), rj(3), rs(3)]
        rp, rv, ra, rj, rs = u[0:3], u[3:6], u[6:9], u[9:12], u[12:15]
        k_p, k_v, k_a, k_i, k_j, base_flag = p[0], p[1], p[2], p[3], p[4], p[5]

        pos, vel, acc, jerk = x[0:3], x[3:6], x[6:9], x[9:12]
        integ, cmd = x[12:15], x[15:18]

        # No bias: true tracking error
        e_p = pos - rp
        e_v = vel - rv
        e_a = acc - ra
        e_j = jerk - rj

        target = rs - k_p * e_p - k_v * e_v - k_a * e_a - k_i * integ - k_j * e_j
        j_dot = (1.0 - base_flag) * target + base_flag * cmd

        p_dot = vel
        v_dot = acc
        a_dot = jerk
        integ_dot = e_p
        cmd_dot = -k_j * e_j

        return jnp.concatenate([p_dot, v_dot, a_dot, j_dot, integ_dot, cmd_dot])


# Module-level cache for the WaypointTrackingSystem embedding
_WPT_EMB_CACHE: Dict[str, Tuple[WaypointTrackingSystem, object]] = {}


def get_system_and_embedding() -> Tuple[WaypointTrackingSystem, object]:
    """Return (system, natural_embedding) for WaypointTrackingSystem, cached."""
    key = "waypoint_tracking"
    if key not in _WPT_EMB_CACHE:
        sys_ = WaypointTrackingSystem()
        emb = irx.natemb(sys_)
        _WPT_EMB_CACHE[key] = (sys_, emb)
    return _WPT_EMB_CACHE[key]


def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    """One forward-Euler interval step via the natural embedding."""
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scenario construction
# ══════════════════════════════════════════════════════════════════════════════

def create_scenarios(names: Optional[List[str]] = None) -> List[Scenario]:
    """Return the four controller scenarios sharing one WaypointTrackingSystem
    embedding — same structure as crazyflie_chain_controllers.create_scenarios
    but with WaypointTrackingSystem."""
    _, emb = get_system_and_embedding()
    want = list(_CANDIDATE_THETA.keys()) if names is None else names
    scenarios = []
    for name in want:
        theta = jnp.array(_CANDIDATE_THETA[name])
        scenarios.append(Scenario(name, emb, irx.Interval(lower=theta, upper=theta)))
    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# 3. Mission waypoint generation
# ══════════════════════════════════════════════════════════════════════════════

def nominal_waypoints(K: int = DEFAULT_K) -> jnp.ndarray:
    """K waypoints evenly spaced from P0 to P1 along RQ3's cruise line.

    Returns
    -------
    waypoints : (K, 3) nominal waypoint positions
    """
    # Linearly interpolate between P0 and P1 (exclusive of P0, inclusive of P1)
    alphas = jnp.linspace(1.0 / K, 1.0, K)
    return P0[None, :] + alphas[:, None] * (P1 - P0)[None, :]


def _offset_bounds(K: int,
                   offset_lim_xy: float = DEFAULT_OFFSET_LIM_XY,
                   offset_lim_z: float = DEFAULT_OFFSET_LIM_Z) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute per-waypoint offset bounds: box constraint intersected with arena.

    Returns
    -------
    lo, hi : (K, 3) lower/upper bounds for per-waypoint offsets
    """
    nom = nominal_waypoints(K)

    # Per-waypoint box: ±offset_lim around the nominal point
    raw_lo = jnp.full((K, 3), jnp.array([-offset_lim_xy, -offset_lim_xy, -offset_lim_z]))
    raw_hi = jnp.full((K, 3), jnp.array([offset_lim_xy, offset_lim_xy, offset_lim_z]))

    # Arena bounds: absolute position must stay inside
    arena_lo = jnp.array([-ARENA_X_HALF, -ARENA_Y_HALF, ARENA_Z_RANGE[0]])
    arena_hi = jnp.array([ARENA_X_HALF, ARENA_Y_HALF, ARENA_Z_RANGE[1]])

    # Offset bounds = intersection of raw box with (arena - nominal)
    offset_lo = jnp.maximum(raw_lo, arena_lo[None, :] - nom)
    offset_hi = jnp.minimum(raw_hi, arena_hi[None, :] - nom)

    return offset_lo, offset_hi


def _project_offsets(offsets: jnp.ndarray, K: int,
                     offset_lim_xy: float = DEFAULT_OFFSET_LIM_XY,
                     offset_lim_z: float = DEFAULT_OFFSET_LIM_Z) -> jnp.ndarray:
    """Project waypoint offsets to feasible box (arena ∩ offset limits)."""
    lo, hi = _offset_bounds(K, offset_lim_xy, offset_lim_z)
    return jnp.clip(offsets, lo, hi)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Waypoint-space separation optimizer
# ══════════════════════════════════════════════════════════════════════════════

def _waypoints_to_u_seq(offsets: jnp.ndarray, x0_state4x3: jnp.ndarray,
                        T_hop: float, dt: float) -> jnp.ndarray:
    """Convert waypoint offsets to the reference sequence (u_seq).

    Parameters
    ----------
    offsets : (K, 3) per-waypoint offset from nominal
    x0_state4x3 : (4, 3) initial state
    T_hop : hop duration
    dt : sampling interval

    Returns
    -------
    u_seq : (T_total, 15) per-step reference [rp, rv, ra, rj, rs]
    """
    K = offsets.shape[0]
    waypoints = nominal_waypoints(K) + offsets
    return build_mission_reference(waypoints, x0_state4x3, T_hop, dt)


def _propagate_all_scenarios(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                             scenarios: List[Scenario], dt: float) -> jnp.ndarray:
    """Propagate all scenarios forward under the full reference sequence.
    Returns per-scenario state history as stacked lower/upper arrays.

    Uses the WaypointTrackingSystem embedding (15-dim u = ref15 per step).
    """
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    num_steps = u_seq.shape[0]

    def prop_one(p_ivl):
        def step_fn(x_ivl, u_k):
            x_next = euler_step(emb_sys, x_ivl, u_k, p_ivl, dt)
            return x_next, x_next
        _, x_hist = jax.lax.scan(step_fn, x0_ivl, u_seq)
        return x_hist  # Interval with lower/upper shape (num_steps, 18)

    x_hist_batch = jax.vmap(prop_one)(p_batch)
    return x_hist_batch  # lower/upper shape (n_scenarios, num_steps, 18)


def waypoint_separation_loss(offsets: jnp.ndarray, x0_ivl: irx.Interval,
                             scenarios: List[Scenario],
                             x0_state4x3: jnp.ndarray,
                             T_hop: float, dt: float) -> jnp.ndarray:
    """Separation loss for waypoint-based discrimination.

    Forward path: offsets → waypoints → build_mission_reference → u_seq →
    propagate all scenarios → pairwise observed-output overlap.

    Returns the minimum over steps of the total pairwise observed-output
    overlap (same semantics as the spoof pipeline's propagate_with_refinement).
    """
    u_seq = _waypoints_to_u_seq(offsets, x0_state4x3, T_hop, dt)
    n = len(scenarios)
    num_steps = u_seq.shape[0]

    x_hist_batch = _propagate_all_scenarios(x0_ivl, u_seq, scenarios, dt)

    # Compute pairwise overlap at each step (on observable 12 dims only)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    i_idx = jnp.array([i for i, j in pairs])
    j_idx = jnp.array([j for i, j in pairs])

    def overlap_at_k(k):
        lower_i = x_hist_batch.lower[i_idx, k, :12]
        upper_i = x_hist_batch.upper[i_idx, k, :12]
        lower_j = x_hist_batch.lower[j_idx, k, :12]
        upper_j = x_hist_batch.upper[j_idx, k, :12]
        # Per-axis overlap width (clamped to 0). Use sum-of-widths instead
        # of prod-of-widths for gradient stability: with 12 dims, the product
        # goes to zero (and kills gradients) as soon as ANY axis separates,
        # which happens almost immediately on a multi-hop mission. The sum
        # provides gradient signal as long as any axis still overlaps.
        widths = jnp.maximum(jnp.minimum(upper_i, upper_j) - jnp.maximum(lower_i, lower_j), 0.0)
        return jnp.sum(widths)

    segment_overlaps = jax.vmap(overlap_at_k)(jnp.arange(num_steps))
    # Use nanmin to handle steps where intervals may have overflowed to NaN
    # (unstable controllers can cause inf tube widths)
    return jnp.where(jnp.all(jnp.isnan(segment_overlaps)),
                     jnp.array(0.0), jnp.nanmin(segment_overlaps))


def optimize_waypoints_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario],
                           x0_state4x3: jnp.ndarray,
                           K: int = DEFAULT_K, T_hop: float = DEFAULT_T_HOP,
                           dt: float = QPS_DT,
                           num_restarts: int = 48, learning_rate: float = 0.01,
                           num_iters: int = 200, seed: int = 42,
                           offset_lim_xy: float = DEFAULT_OFFSET_LIM_XY,
                           offset_lim_z: float = DEFAULT_OFFSET_LIM_Z,
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Multi-start projected gradient descent over waypoint offsets.

    Returns
    -------
    best_offsets : (K, 3) best waypoint offsets
    best_loss : scalar loss value
    all_offsets : (num_restarts, K, 3) final offsets for all restarts
    all_losses : (num_restarts,) final losses
    """
    key = jax.random.PRNGKey(seed)
    noise_scale = offset_lim_xy * 0.3
    offsets_init = jax.random.normal(key, (num_restarts, K, 3)) * noise_scale

    # Project initial offsets to feasible set
    project = partial(_project_offsets, K=K, offset_lim_xy=offset_lim_xy,
                      offset_lim_z=offset_lim_z)
    offsets_init = jax.vmap(project)(offsets_init)

    def loss_fn(offsets):
        return waypoint_separation_loss(offsets, x0_ivl, scenarios,
                                        x0_state4x3, T_hop, dt)

    batched_loss = jax.vmap(loss_fn)
    batched_grad = jax.vmap(jax.grad(loss_fn))

    def body(offsets_batch, _i):
        g = batched_grad(offsets_batch)
        updated = offsets_batch - learning_rate * g
        return jax.vmap(project)(updated)

    def scan_body(carry, i):
        return body(carry, i), None

    offsets_final, _ = jax.lax.scan(scan_body, offsets_init, jnp.arange(num_iters))
    losses = batched_loss(offsets_final)
    best_idx = jnp.argmin(losses)

    return offsets_final[best_idx], losses[best_idx], offsets_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 5. Simulation and discrimination (waypoint-specific)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_true_trajectory(x0_point: jnp.ndarray, u_seq: jnp.ndarray,
                             theta6: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Point (non-interval) rollout of ONE controller under a time-varying
    reference, for generating ground-truth observed trajectories.

    Parameters
    ----------
    x0_point : (18,) initial state
    u_seq : (T, 15) per-step reference [rp, rv, ra, rj, rs]
    theta6 : (6,) controller parameters
    dt : time step

    Returns
    -------
    traj : (T, 12) observed output trajectory [pos, vel, acc, jerk]
    """
    sys_, _ = get_system_and_embedding()
    theta6 = jnp.asarray(theta6)

    def step(x, u_k):
        x_next = x + dt * sys_.f(jnp.zeros(()), x, u_k, theta6)
        return x_next, x_next[:12]

    _, traj = jax.lax.scan(step, jnp.asarray(x0_point), u_seq)
    return traj  # (T, 12)


def discriminate_controller_waypoint(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                     observed_traj: jnp.ndarray,
                                     scenarios: List[Scenario], dt: float,
                                     w_bar: float = _DEFAULT_W_BAR) -> Dict:
    """Run each scenario forward under the time-varying reference, checking
    at every step whether the true observed output stays inside the predicted
    interval.

    Same logic as crazyflie_chain_controllers.discriminate_controller but
    using the WaypointTrackingSystem embedding (15-dim u, not 3-dim bias).
    """
    T = u_seq.shape[0]
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def run_one(p_ivl):
        def step(carry, k):
            x_ivl, falsified, fail_step = carry
            x_next = euler_step(emb_sys, x_ivl, u_seq[k], p_ivl, dt)
            y_pred = observed_output(x_next)
            y_true = observed_traj[k]
            contained = jnp.all((y_true >= y_pred.lower - w_bar) &
                                (y_true <= y_pred.upper + w_bar))
            falsified_new = falsified | jnp.logical_not(contained)
            fail_step_new = jnp.where(
                jnp.logical_not(contained) & jnp.logical_not(falsified), k, fail_step)

            # Measurement update
            refine_now = contained
            obs_lo = jnp.where(refine_now, y_true - w_bar, x_next.lower[:12])
            obs_hi = jnp.where(refine_now, y_true + w_bar, x_next.upper[:12])
            x_ref = irx.Interval(
                lower=jnp.concatenate([obs_lo, x_next.lower[12:18]]),
                upper=jnp.concatenate([obs_hi, x_next.upper[12:18]]),
            )
            return (x_ref, falsified_new, fail_step_new), contained

        init = (x0_ivl, jnp.array(False), jnp.array(-1))
        (_, falsified, fail_step), contained_hist = jax.lax.scan(step, init, jnp.arange(T))
        return falsified, fail_step, contained_hist

    falsified, fail_step, contained_hist = jax.vmap(run_one)(p_batch)

    return {
        'falsified': {s.name: bool(falsified[i]) for i, s in enumerate(scenarios)},
        'fail_step': {s.name: int(fail_step[i]) for i, s in enumerate(scenarios)},
        'contained_history': {s.name: np.array(contained_hist[i]) for i, s in enumerate(scenarios)},
        'survivors': [s.name for i, s in enumerate(scenarios) if not bool(falsified[i])],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Module-level demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Waypoint-based controller discrimination (no GPS spoofing)")
    print("=" * 70)

    scenarios = create_scenarios()
    x0_state4x3 = jnp.zeros((4, 3)).at[0].set(P0)  # start at P0 at rest
    x0_ivl = irx.icentpert(jnp.zeros(18).at[0:3].set(P0), jnp.full(18, 1e-3))

    K = 3  # fewer waypoints for the quick demo
    print(f"\nNominal waypoints (K={K}):")
    nom = nominal_waypoints(K)
    print(f"  {np.array(nom)}")

    # Quick optimization with few restarts/iterations for demo
    print(f"\nOptimizing waypoint offsets ({K} waypoints, T_hop={DEFAULT_T_HOP}s)...")
    best_offsets, best_loss, _, all_losses = optimize_waypoints_gpu(
        x0_ivl, scenarios, x0_state4x3,
        K=K, T_hop=DEFAULT_T_HOP, dt=QPS_DT,
        num_restarts=16, learning_rate=0.005, num_iters=50, seed=0,
    )
    print(f"Best loss: {float(best_loss):.6f}")
    print(f"Best offsets:\n  {np.array(best_offsets)}")

    # Sanity check: discriminate each candidate
    waypoints = nom + best_offsets
    u_seq = build_mission_reference(waypoints, x0_state4x3, DEFAULT_T_HOP, QPS_DT)
    print(f"\nReference trajectory shape: {u_seq.shape}")

    print("\nDiscrimination check (each candidate as ground truth):")
    all_ok = True
    for true_name in _CANDIDATE_THETA:
        true_theta = jnp.array(_CANDIDATE_THETA[true_name])
        x0_point = jnp.zeros(18).at[0:3].set(P0)
        observed = simulate_true_trajectory(x0_point, u_seq, true_theta, dt=QPS_DT)
        result = discriminate_controller_waypoint(x0_ivl, u_seq, observed, scenarios, dt=QPS_DT)
        survivors = result["survivors"]
        ok = survivors == [true_name]
        all_ok &= ok
        print(f"  true={true_name:16s}  survivors={survivors}  {'OK' if ok else 'AMBIGUOUS'}")

    print(f"\nAll uniquely discriminated: {all_ok}")
