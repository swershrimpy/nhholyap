"""
JAX-differentiable waypoint trajectory generation for the Crazyflie chain layer
================================================================================
This module is a faithful JAX transcription of QPS's `nominal_traj.py`
(`qps/utilities/nominal_traj.py`), specifically the two functions:
  - `traj_coeffs_from_state_and_waypoint` — solves for 7th-order polynomial
    coefficients per axis given start state [pos,vel,acc,jerk] and an end
    position (with zero end vel/acc/jerk), over a hop duration T.
  - `trajectory_generator` — evaluates the polynomial and its derivatives
    (up to snap) at a given time t within a hop.

The JAX transcription is differentiable w.r.t. the waypoint end-position,
enabling gradient-based optimization of waypoint offsets for controller
discrimination. Validated bit-for-bit against an independent NumPy oracle
(same pattern as `crazyflie_12d.py`'s `test_matches_qps_forward_model`).

The polynomial solve: 8 boundary conditions (start pos/vel/acc/jerk, end
pos with zero vel/acc/jerk) → 8 coefficients per axis → `jnp.linalg.solve`
on a fixed 8×8 matrix that depends only on T (not on the waypoint position),
so it's cheap and fully differentiable.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/crazyflie_waypoint_trajectory.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


# ══════════════════════════════════════════════════════════════════════════════
# Core trajectory generation
# ══════════════════════════════════════════════════════════════════════════════

def _build_boundary_matrix(T: float) -> jnp.ndarray:
    """Build the 8×8 boundary condition matrix for a 7th-order polynomial.

    The polynomial is p(t) = c0 + c1*t + c2*t^2 + ... + c7*t^7.
    Boundary conditions (per axis):
      t=0: p(0)=x0, p'(0)=v0, p''(0)=a0, p'''(0)=j0
      t=T: p(T)=xf, p'(T)=0, p''(T)=0, p'''(T)=0

    This is the same matrix `nominal_traj.py:traj_coeffs_from_state_and_waypoint`
    constructs, just expressed in JAX.
    """
    T = jnp.asarray(T, dtype=jnp.float32)
    # Row 0: p(0) = c0 = x0
    # Row 1: p'(0) = c1 = v0
    # Row 2: p''(0) = 2*c2 = a0
    # Row 3: p'''(0) = 6*c3 = j0
    # Row 4: p(T) = sum(ci * T^i) = xf
    # Row 5: p'(T) = sum(i*ci * T^(i-1)) = 0
    # Row 6: p''(T) = sum(i*(i-1)*ci * T^(i-2)) = 0
    # Row 7: p'''(T) = sum(i*(i-1)*(i-2)*ci * T^(i-3)) = 0

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    T6 = T5 * T
    T7 = T6 * T

    M = jnp.array([
        # c0, c1, c2, c3, c4, c5, c6, c7
        [1., 0., 0., 0., 0., 0., 0., 0.],           # p(0) = c0
        [0., 1., 0., 0., 0., 0., 0., 0.],           # p'(0) = c1
        [0., 0., 2., 0., 0., 0., 0., 0.],           # p''(0) = 2*c2
        [0., 0., 0., 6., 0., 0., 0., 0.],           # p'''(0) = 6*c3
        [1., T, T2, T3, T4, T5, T6, T7],            # p(T)
        [0., 1., 2.*T, 3.*T2, 4.*T3, 5.*T4, 6.*T5, 7.*T6],  # p'(T)
        [0., 0., 2., 6.*T, 12.*T2, 20.*T3, 30.*T4, 42.*T5],  # p''(T)
        [0., 0., 0., 6., 24.*T, 60.*T2, 120.*T3, 210.*T4],   # p'''(T)
    ], dtype=jnp.float32)
    return M


def traj_coeffs_from_waypoint(end_pos: jnp.ndarray, start_state4x3: jnp.ndarray,
                              T: float) -> jnp.ndarray:
    """Compute per-axis 7th-order polynomial coefficients.

    Parameters
    ----------
    end_pos : (3,) target position at end of hop
    start_state4x3 : (4, 3) start state [pos, vel, acc, jerk] per axis
    T : hop duration (seconds)

    Returns
    -------
    coeffs : (3, 8) polynomial coefficients per axis, c0..c7
    """
    M = _build_boundary_matrix(T)

    # Defensive float32 cast: without it, this silently follows whatever
    # jax_enable_x64 happens to be at call time (a GLOBAL, mutable JAX
    # setting) rather than this module's own explicit float32 convention.
    # Concretely: importing crazyflie_firmware_controllers.py pulls in
    # rq3_crazyflie_surrogates.py, which calls
    # jax.config.update("jax_enable_x64", True) unconditionally on import
    # (not something this project can or should change -- that file lives
    # outside this repo). If BOTH modules get imported into the same
    # process (e.g. pytest collecting the whole tests/ directory), plain
    # jnp.zeros(...)/jnp.array(...) calls made by CALLERS of this function
    # after that point silently become float64, while build_mission_reference
    # below still builds its own next_state as float32 -- a carry dtype
    # mismatch inside jax.lax.scan. Verified: all 59 tests in this module
    # pass in isolation; the failure only appeared when collected alongside
    # test_crazyflie_firmware_controllers.py in the same pytest run. Casting
    # here (and in build_mission_reference below) makes this module's
    # dtype behavior independent of that ambient global state.
    start_state4x3 = jnp.asarray(start_state4x3, dtype=jnp.float32)
    end_pos = jnp.asarray(end_pos, dtype=jnp.float32)

    # RHS: [x0, v0, a0, j0, xf, 0, 0, 0] per axis
    rhs = jnp.zeros((8, 3), dtype=jnp.float32)
    rhs = rhs.at[0, :].set(start_state4x3[0])  # pos
    rhs = rhs.at[1, :].set(start_state4x3[1])  # vel
    rhs = rhs.at[2, :].set(start_state4x3[2])  # acc
    rhs = rhs.at[3, :].set(start_state4x3[3])  # jerk
    rhs = rhs.at[4, :].set(end_pos)             # end pos
    # rows 5,6,7 are 0 (zero end vel/acc/jerk)

    # Solve M @ coeffs = rhs for each axis (solve batched over columns)
    coeffs = jnp.linalg.solve(M, rhs)  # (8, 3)
    return coeffs.T  # (3, 8)


def sample_reference(coeffs: jnp.ndarray, t: float) -> jnp.ndarray:
    """Evaluate polynomial reference [pos, vel, acc, jerk, snap] at time t.

    Parameters
    ----------
    coeffs : (3, 8) polynomial coefficients per axis
    t : time within the hop [0, T]

    Returns
    -------
    ref15 : (15,) [pos(3), vel(3), acc(3), jerk(3), snap(3)]
    """
    t = jnp.asarray(t, dtype=jnp.float32)

    # Powers of t
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    t7 = t6 * t

    # Position: p(t) = c0 + c1*t + c2*t^2 + ... + c7*t^7
    t_powers = jnp.array([1., t, t2, t3, t4, t5, t6, t7])
    pos = coeffs @ t_powers  # (3,)

    # Velocity: p'(t) = c1 + 2*c2*t + 3*c3*t^2 + ... + 7*c7*t^6
    dt_powers = jnp.array([0., 1., 2.*t, 3.*t2, 4.*t3, 5.*t4, 6.*t5, 7.*t6])
    vel = coeffs @ dt_powers  # (3,)

    # Acceleration: p''(t) = 2*c2 + 6*c3*t + 12*c4*t^2 + ... + 42*c7*t^5
    d2t_powers = jnp.array([0., 0., 2., 6.*t, 12.*t2, 20.*t3, 30.*t4, 42.*t5])
    acc = coeffs @ d2t_powers  # (3,)

    # Jerk: p'''(t) = 6*c3 + 24*c4*t + 60*c5*t^2 + 120*c6*t^3 + 210*c7*t^4
    d3t_powers = jnp.array([0., 0., 0., 6., 24.*t, 60.*t2, 120.*t3, 210.*t4])
    jerk = coeffs @ d3t_powers  # (3,)

    # Snap: p''''(t) = 24*c4 + 120*c5*t + 360*c6*t^2 + 840*c7*t^3
    d4t_powers = jnp.array([0., 0., 0., 0., 24., 120.*t, 360.*t2, 840.*t3])
    snap = coeffs @ d4t_powers  # (3,)

    return jnp.concatenate([pos, vel, acc, jerk, snap])


def build_mission_reference(waypoints: jnp.ndarray, x0_state4x3: jnp.ndarray,
                            T_hop: float, dt: float) -> jnp.ndarray:
    """Build the full mission reference trajectory by chaining K hops.

    Each hop is a 7th-order polynomial from the planned rest state at the
    previous waypoint to the current waypoint. Hop 0 starts at `x0_state4x3`;
    hop k>0 starts at `[waypoints[k-1], 0, 0, 0]` (the boundary conditions
    guarantee every hop ends at rest, so this is deterministic and doesn't
    depend on propagated/uncertain state).

    Parameters
    ----------
    waypoints : (K, 3) sequence of waypoint positions
    x0_state4x3 : (4, 3) initial state [pos, vel, acc, jerk]
    T_hop : duration of each hop (seconds)
    dt : sampling interval (seconds)

    Returns
    -------
    ref_traj : (T_total, 15) reference trajectory sampled at dt intervals,
               where T_total = K * steps_per_hop and each row is
               [pos, vel, acc, jerk, snap].
    """
    # Defensive float32 cast -- see the same note in traj_coeffs_from_waypoint.
    # x0_state4x3 in particular is the INITIAL CARRY jax.lax.scan is given
    # below; compute_one_hop's returned next_state is hardcoded float32, so
    # if the caller's x0_state4x3 came from a plain jnp.zeros(...)/jnp.array(...)
    # made while jax_enable_x64 happened to be globally True, scan raises
    # "carry input and carry output must have equal types" -- fixed here so
    # this function's behavior doesn't depend on ambient global JAX config.
    waypoints = jnp.asarray(waypoints, dtype=jnp.float32)
    x0_state4x3 = jnp.asarray(x0_state4x3, dtype=jnp.float32)

    K = waypoints.shape[0]
    # steps_per_hop must be a concrete Python int (not traced) since it's
    # used as the length argument to jnp.arange inside jax.lax.scan.
    # T_hop and dt are always plain Python floats (never traced), so this
    # is safe even when waypoints are traced.
    steps_per_hop = int(round(float(T_hop) / float(dt)))

    # Pre-compute the sample times (concrete shape, independent of traced values)
    times = (jnp.arange(steps_per_hop, dtype=jnp.float32) + 1) * dt  # (steps_per_hop,)

    def compute_one_hop(carry, waypoint_idx):
        """Compute one hop's coefficients and sample it."""
        start_state = carry  # (4, 3)
        end_pos = waypoints[waypoint_idx]

        coeffs = traj_coeffs_from_waypoint(end_pos, start_state, T_hop)

        # Sample at dt intervals within the hop
        samples = jax.vmap(partial(sample_reference, coeffs))(times)  # (steps_per_hop, 15)

        # Next hop starts at planned rest: [end_pos, 0, 0, 0]
        next_state = jnp.zeros((4, 3), dtype=jnp.float32).at[0].set(end_pos)

        return next_state, samples

    _, all_samples = jax.lax.scan(compute_one_hop, x0_state4x3, jnp.arange(K))
    # all_samples shape: (K, steps_per_hop, 15)
    # Reshape to (K * steps_per_hop, 15)
    return all_samples.reshape(-1, 15)


if __name__ == "__main__":
    # Smoke test: a single hop from rest at origin to [1, 0, 1]
    x0 = jnp.zeros((4, 3))  # at rest at origin
    end = jnp.array([1.0, 0.0, 1.0])
    T = 4.0

    coeffs = traj_coeffs_from_waypoint(end, x0, T)
    print(f"Coefficients shape: {coeffs.shape}")

    # Check boundary conditions
    ref_start = sample_reference(coeffs, 0.0)
    ref_end = sample_reference(coeffs, T)
    print(f"At t=0: pos={ref_start[:3]}, vel={ref_start[3:6]}, acc={ref_start[6:9]}, jerk={ref_start[9:12]}")
    print(f"At t=T: pos={ref_end[:3]}, vel={ref_end[3:6]}, acc={ref_end[6:9]}, jerk={ref_end[9:12]}")

    # Mission: 3 waypoints
    waypoints = jnp.array([[0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [1.2, 0.0, 1.0]])
    ref_traj = build_mission_reference(waypoints, x0, T_hop=4.0, dt=0.02)
    print(f"\nMission reference shape: {ref_traj.shape} (expected {3 * int(4.0/0.02)} steps)")
    print(f"Final position: {ref_traj[-1, :3]}")
    print(f"Final velocity: {ref_traj[-1, 3:6]}")
