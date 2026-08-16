"""Tests for crazyflie_waypoint_trajectory.py — JAX trajectory generation.

Validates that the JAX transcription of QPS's `nominal_traj.py` reproduces
the same polynomial coefficients and reference samples as an independent
NumPy oracle, across random waypoints/start-states/T values.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python -m pytest examples/adaptive_spoofing/tests/test_crazyflie_waypoint_trajectory.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

_EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_waypoint_trajectory import (
    _build_boundary_matrix,
    traj_coeffs_from_waypoint,
    sample_reference,
    build_mission_reference,
)


# ──────────────────────────────────────────────────────────────────────────
# Independent NumPy transcription of nominal_traj.py (oracle)
# ──────────────────────────────────────────────────────────────────────────

def _numpy_boundary_matrix(T: float) -> np.ndarray:
    """NumPy oracle for the 8×8 boundary matrix."""
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    T6 = T5 * T
    T7 = T6 * T
    return np.array([
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 2., 0., 0., 0., 0., 0.],
        [0., 0., 0., 6., 0., 0., 0., 0.],
        [1., T, T2, T3, T4, T5, T6, T7],
        [0., 1., 2.*T, 3.*T2, 4.*T3, 5.*T4, 6.*T5, 7.*T6],
        [0., 0., 2., 6.*T, 12.*T2, 20.*T3, 30.*T4, 42.*T5],
        [0., 0., 0., 6., 24.*T, 60.*T2, 120.*T3, 210.*T4],
    ], dtype=np.float64)


def _numpy_traj_coeffs(end_pos: np.ndarray, start_state4x3: np.ndarray,
                       T: float) -> np.ndarray:
    """NumPy oracle: solve for polynomial coefficients."""
    M = _numpy_boundary_matrix(T)
    rhs = np.zeros((8, 3))
    rhs[0] = start_state4x3[0]  # pos
    rhs[1] = start_state4x3[1]  # vel
    rhs[2] = start_state4x3[2]  # acc
    rhs[3] = start_state4x3[3]  # jerk
    rhs[4] = end_pos
    coeffs = np.linalg.solve(M, rhs)  # (8, 3)
    return coeffs.T  # (3, 8)


def _numpy_sample_reference(coeffs: np.ndarray, t: float) -> np.ndarray:
    """NumPy oracle: evaluate polynomial and derivatives at t."""
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    t7 = t6 * t

    t_powers = np.array([1., t, t2, t3, t4, t5, t6, t7])
    pos = coeffs @ t_powers

    dt_powers = np.array([0., 1., 2.*t, 3.*t2, 4.*t3, 5.*t4, 6.*t5, 7.*t6])
    vel = coeffs @ dt_powers

    d2t_powers = np.array([0., 0., 2., 6.*t, 12.*t2, 20.*t3, 30.*t4, 42.*t5])
    acc = coeffs @ d2t_powers

    d3t_powers = np.array([0., 0., 0., 6., 24.*t, 60.*t2, 120.*t3, 210.*t4])
    jerk = coeffs @ d3t_powers

    d4t_powers = np.array([0., 0., 0., 0., 24., 120.*t, 360.*t2, 840.*t3])
    snap = coeffs @ d4t_powers

    return np.concatenate([pos, vel, acc, jerk, snap])


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBoundaryMatrix:
    def test_shape(self):
        M = _build_boundary_matrix(4.0)
        assert M.shape == (8, 8)

    def test_matches_numpy_oracle(self):
        for T in [1.0, 2.5, 4.0, 0.5]:
            M_jax = np.array(_build_boundary_matrix(T))
            M_np = _numpy_boundary_matrix(T)
            np.testing.assert_allclose(M_jax, M_np, atol=1e-4, rtol=1e-4)

    def test_invertible(self):
        M = _build_boundary_matrix(4.0)
        det = jnp.linalg.det(M)
        assert abs(float(det)) > 1e-6


class TestTrajCoeffs:
    @pytest.mark.parametrize("seed", range(8))
    def test_matches_numpy_oracle(self, seed):
        rng = np.random.default_rng(seed)
        end_pos = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        start_state = rng.uniform(-0.5, 0.5, (4, 3)).astype(np.float32)
        T = rng.uniform(1.0, 5.0)

        coeffs_jax = np.array(traj_coeffs_from_waypoint(
            jnp.array(end_pos), jnp.array(start_state), T))
        coeffs_np = _numpy_traj_coeffs(end_pos.astype(np.float64),
                                       start_state.astype(np.float64), T)

        np.testing.assert_allclose(coeffs_jax, coeffs_np, atol=1e-3, rtol=1e-3)

    def test_boundary_conditions_at_start(self):
        """p(0) = x0, p'(0) = v0, p''(0) = a0, p'''(0) = j0."""
        start_state = jnp.array([[0.5, 0.3, 1.0],
                                  [0.1, -0.2, 0.0],
                                  [0.0, 0.1, -0.1],
                                  [0.05, 0.0, 0.0]])
        end_pos = jnp.array([1.0, 0.0, 1.5])
        T = 4.0

        coeffs = traj_coeffs_from_waypoint(end_pos, start_state, T)
        ref = sample_reference(coeffs, 0.0)

        np.testing.assert_allclose(np.array(ref[:3]), np.array(start_state[0]), atol=1e-5)
        np.testing.assert_allclose(np.array(ref[3:6]), np.array(start_state[1]), atol=1e-5)
        np.testing.assert_allclose(np.array(ref[6:9]), np.array(start_state[2]), atol=1e-5)
        np.testing.assert_allclose(np.array(ref[9:12]), np.array(start_state[3]), atol=1e-5)

    def test_boundary_conditions_at_end(self):
        """p(T) = end_pos, p'(T) = 0, p''(T) = 0, p'''(T) = 0."""
        start_state = jnp.zeros((4, 3))
        end_pos = jnp.array([1.2, 0.5, 1.0])
        T = 4.0

        coeffs = traj_coeffs_from_waypoint(end_pos, start_state, T)
        ref = sample_reference(coeffs, T)

        np.testing.assert_allclose(np.array(ref[:3]), np.array(end_pos), atol=1e-4)
        np.testing.assert_allclose(np.array(ref[3:6]), 0.0, atol=1e-4)
        np.testing.assert_allclose(np.array(ref[6:9]), 0.0, atol=1e-4)
        np.testing.assert_allclose(np.array(ref[9:12]), 0.0, atol=1e-4)


class TestSampleReference:
    @pytest.mark.parametrize("seed", range(8))
    def test_matches_numpy_oracle(self, seed):
        rng = np.random.default_rng(seed)
        end_pos = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        start_state = rng.uniform(-0.3, 0.3, (4, 3)).astype(np.float32)
        T = rng.uniform(2.0, 5.0)
        t_sample = rng.uniform(0.0, T)

        coeffs = traj_coeffs_from_waypoint(jnp.array(end_pos), jnp.array(start_state), T)
        ref_jax = np.array(sample_reference(coeffs, t_sample))

        coeffs_np = _numpy_traj_coeffs(end_pos.astype(np.float64),
                                       start_state.astype(np.float64), T)
        ref_np = _numpy_sample_reference(coeffs_np, t_sample)

        np.testing.assert_allclose(ref_jax, ref_np, atol=1e-3, rtol=1e-3)


class TestBuildMissionReference:
    def test_shape(self):
        waypoints = jnp.array([[0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [1.2, 0.0, 1.0]])
        x0 = jnp.zeros((4, 3))
        T_hop = 4.0
        dt = 0.02

        ref = build_mission_reference(waypoints, x0, T_hop, dt)
        expected_steps = 3 * int(T_hop / dt)  # K * steps_per_hop
        assert ref.shape == (expected_steps, 15)

    def test_each_hop_reaches_target(self):
        """Each hop's final sample should be at (or very near) the waypoint
        with zero vel/acc/jerk — the boundary conditions guarantee this."""
        waypoints = jnp.array([[0.4, 0.0, 1.0], [0.8, 0.0, 1.0], [1.2, 0.0, 1.0]])
        x0 = jnp.zeros((4, 3))
        T_hop = 4.0
        dt = 0.02
        steps_per_hop = int(T_hop / dt)

        ref = build_mission_reference(waypoints, x0, T_hop, dt)

        for k in range(3):
            end_idx = (k + 1) * steps_per_hop - 1
            ref_end = ref[end_idx]
            np.testing.assert_allclose(np.array(ref_end[:3]),
                                       np.array(waypoints[k]), atol=1e-3)
            np.testing.assert_allclose(np.array(ref_end[3:6]), 0.0, atol=1e-3)
            np.testing.assert_allclose(np.array(ref_end[6:9]), 0.0, atol=1e-3)
            np.testing.assert_allclose(np.array(ref_end[9:12]), 0.0, atol=1e-3)

    def test_continuity_at_hop_boundaries(self):
        """Position at the start of hop k+1 should be near the end of hop k."""
        waypoints = jnp.array([[0.5, 0.1, 1.0], [1.0, -0.1, 1.0]])
        x0 = jnp.zeros((4, 3))
        T_hop = 4.0
        dt = 0.02
        steps_per_hop = int(T_hop / dt)

        ref = build_mission_reference(waypoints, x0, T_hop, dt)

        # End of hop 0
        end_hop0 = ref[steps_per_hop - 1]
        # Start of hop 1
        start_hop1 = ref[steps_per_hop]

        # Position at start of hop 1 should be near waypoints[0] (the
        # planned rest state). Note: start_hop1 is sampled at t=dt (not t=0)
        # so there's a small offset, but position should be very close.
        np.testing.assert_allclose(np.array(start_hop1[:3]),
                                   np.array(waypoints[0]), atol=0.01)

    def test_differentiable_wrt_waypoints(self):
        """build_mission_reference should be differentiable w.r.t. waypoints."""
        x0 = jnp.zeros((4, 3))

        def loss(waypoints):
            ref = build_mission_reference(waypoints, x0, 4.0, 0.02)
            return jnp.sum(ref[:, :3] ** 2)

        waypoints = jnp.array([[0.5, 0.0, 1.0], [1.0, 0.0, 1.0]])
        grad = jax.grad(loss)(waypoints)
        assert grad.shape == (2, 3)
        assert not jnp.all(grad == 0)  # non-trivial gradient


# Need jax for the differentiability test
import jax
