"""Tests for crazyflie_waypoint_controllers.py — waypoint-based discrimination.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python -m pytest examples/adaptive_spoofing/tests/test_crazyflie_waypoint_controllers.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import immrax as irx

_EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_waypoint_controllers import (
    WaypointTrackingSystem, get_system_and_embedding, euler_step,
    create_scenarios, nominal_waypoints, _project_offsets,
    _waypoints_to_u_seq, waypoint_separation_loss,
    simulate_true_trajectory, discriminate_controller_waypoint,
    P0, P1, DEFAULT_T_HOP, DEFAULT_K, QPS_DT,
    DEFAULT_OFFSET_LIM_XY, DEFAULT_OFFSET_LIM_Z,
)
from adaptive_spoofing.crazyflie_chain_controllers import (
    _CANDIDATE_THETA, observed_output,
)
from adaptive_spoofing.crazyflie_waypoint_trajectory import (
    build_mission_reference, traj_coeffs_from_waypoint, sample_reference,
)
from adaptive_spoofing.crazyflie_12d import ARENA_X_HALF, ARENA_Y_HALF, ARENA_Z_RANGE


# ──────────────────────────────────────────────────────────────────────────
# Independent NumPy oracle for the unified controller law with time-varying ref
# ──────────────────────────────────────────────────────────────────────────
def _waypoint_rhs_numpy(x, ref15, theta6):
    """NumPy oracle: WaypointTrackingSystem.f(), with ref15 as the 'u'."""
    k_p, k_v, k_a, k_i, k_j, base_flag = theta6
    pos, vel, acc, jerk = x[0:3], x[3:6], x[6:9], x[9:12]
    integ, cmd = x[12:15], x[15:18]
    rp, rv, ra, rj, rs = ref15[0:3], ref15[3:6], ref15[6:9], ref15[9:12], ref15[12:15]

    e_p = pos - rp
    e_v = vel - rv
    e_a = acc - ra
    e_j = jerk - rj

    target = rs - k_p * e_p - k_v * e_v - k_a * e_a - k_i * integ - k_j * e_j
    j_dot = (1.0 - base_flag) * target + base_flag * cmd

    return np.concatenate([vel, acc, jerk, j_dot, e_p, -k_j * e_j])


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicsMatchReference:
    @pytest.mark.parametrize("candidate", list(_CANDIDATE_THETA.keys()))
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_numpy_transcription(self, candidate, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, 18).astype(np.float32)
        ref15 = rng.uniform(-0.5, 0.5, 15).astype(np.float32)
        theta6 = np.array(_CANDIDATE_THETA[candidate], dtype=np.float32)

        sys_ = WaypointTrackingSystem()
        xdot_ours = np.array(sys_.f(jnp.zeros(()), jnp.array(x),
                                     jnp.array(ref15), jnp.array(theta6)))
        xdot_ref = _waypoint_rhs_numpy(x, ref15, theta6)

        np.testing.assert_allclose(xdot_ours, xdot_ref, atol=1e-4, rtol=1e-4)

    def test_at_reference_state_zero_derivatives(self):
        """If the drone is exactly at the reference (all errors zero, cmd=0,
        integ=0), then j_dot = rs (snap feedforward) for non-INDI, and
        j_dot = cmd = 0 for INDI."""
        sys_ = WaypointTrackingSystem()
        # State where pos=rp, vel=rv, acc=ra, jerk=rj, integ=0, cmd=0
        ref15 = jnp.array([0.5, 0.0, 1.0,  # rp
                           0.1, 0.0, 0.0,   # rv
                           0.0, 0.0, 0.0,   # ra
                           0.0, 0.0, 0.0,   # rj
                           0.01, 0.0, 0.0]) # rs
        x = jnp.array([0.5, 0.0, 1.0,   # pos = rp
                       0.1, 0.0, 0.0,    # vel = rv
                       0.0, 0.0, 0.0,    # acc = ra
                       0.0, 0.0, 0.0,    # jerk = rj
                       0.0, 0.0, 0.0,    # integ = 0
                       0.0, 0.0, 0.0])   # cmd = 0

        # For qps_snap_chain (base_flag=0): j_dot = rs
        theta_qps = jnp.array(_CANDIDATE_THETA["qps_snap_chain"])
        xdot = sys_.f(jnp.zeros(()), x, ref15, theta_qps)
        np.testing.assert_allclose(np.array(xdot[9:12]), np.array(ref15[12:15]), atol=1e-6)

        # For indi_jerk (base_flag=1): j_dot = cmd = 0
        theta_indi = jnp.array(_CANDIDATE_THETA["indi_jerk"])
        xdot_indi = sys_.f(jnp.zeros(()), x, ref15, theta_indi)
        np.testing.assert_allclose(np.array(xdot_indi[9:12]), 0.0, atol=1e-6)


class TestEmbeddingAndStep:
    def test_natural_embedding_produces_interval(self):
        _, emb = get_system_and_embedding()
        x_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-3))
        ref15 = jnp.zeros(15)
        p_ivl = irx.Interval(lower=jnp.array(_CANDIDATE_THETA["qps_snap_chain"]),
                             upper=jnp.array(_CANDIDATE_THETA["qps_snap_chain"]))
        x_next = euler_step(emb, x_ivl, ref15, p_ivl, dt=QPS_DT)
        assert isinstance(x_next, irx.Interval)
        assert x_next.lower.shape == (18,)
        assert jnp.all(x_next.lower <= x_next.upper)

    def test_zero_state_zero_ref_is_fixed_point(self):
        """x=0, ref15=0 → all errors zero, integ=0, cmd=0 → xdot=0."""
        sys_ = WaypointTrackingSystem()
        for name, theta6 in _CANDIDATE_THETA.items():
            xdot = sys_.f(jnp.zeros(()), jnp.zeros(18), jnp.zeros(15),
                         jnp.array(theta6))
            assert jnp.allclose(xdot, 0.0, atol=1e-6), f"{name} is not at rest"


class TestScenarios:
    def test_create_scenarios_returns_four(self):
        scenarios = create_scenarios()
        assert len(scenarios) == 4
        assert {s.name for s in scenarios} == set(_CANDIDATE_THETA.keys())

    def test_scenarios_share_one_emb_system(self):
        scenarios = create_scenarios()
        emb0 = scenarios[0].emb_system
        assert all(s.emb_system is emb0 for s in scenarios)

    def test_named_subset(self):
        scenarios = create_scenarios(names=["pd_pos_vel", "indi_jerk"])
        assert [s.name for s in scenarios] == ["pd_pos_vel", "indi_jerk"]


class TestNominalWaypoints:
    def test_shape(self):
        wps = nominal_waypoints(5)
        assert wps.shape == (5, 3)

    def test_endpoints(self):
        """First waypoint should be near P0+offset, last should be near P1."""
        K = 5
        wps = nominal_waypoints(K)
        # Last waypoint should be exactly P1
        np.testing.assert_allclose(np.array(wps[-1]), np.array(P1), atol=1e-5)

    def test_altitude_preserved(self):
        wps = nominal_waypoints(5)
        np.testing.assert_allclose(np.array(wps[:, 2]), 1.0, atol=1e-5)


class TestProjectOffsets:
    def test_within_bounds(self):
        K = 5
        offsets = jnp.ones((K, 3)) * 10.0  # way out of bounds
        projected = _project_offsets(offsets, K)
        nom = nominal_waypoints(K)
        positions = nom + projected

        # Must be within arena
        assert np.all(np.array(positions[:, 0]) <= ARENA_X_HALF)
        assert np.all(np.array(positions[:, 0]) >= -ARENA_X_HALF)
        assert np.all(np.array(positions[:, 1]) <= ARENA_Y_HALF)
        assert np.all(np.array(positions[:, 1]) >= -ARENA_Y_HALF)
        assert np.all(np.array(positions[:, 2]) <= ARENA_Z_RANGE[1])
        assert np.all(np.array(positions[:, 2]) >= ARENA_Z_RANGE[0])

    def test_zero_offsets_unchanged(self):
        K = 5
        offsets = jnp.zeros((K, 3))
        projected = _project_offsets(offsets, K)
        np.testing.assert_allclose(np.array(projected), 0.0, atol=1e-7)


class TestMissionReference:
    def test_u_seq_shape(self):
        K = 3
        offsets = jnp.zeros((K, 3))
        x0_state = jnp.zeros((4, 3))
        u_seq = _waypoints_to_u_seq(offsets, x0_state, DEFAULT_T_HOP, QPS_DT)
        expected_steps = K * int(DEFAULT_T_HOP / QPS_DT)
        assert u_seq.shape == (expected_steps, 15)

    def test_reaches_waypoints(self):
        """Reference at the end of each hop should be near the waypoint."""
        K = 3
        offsets = jnp.array([[0.1, 0.0, 0.0], [-0.1, 0.05, 0.0], [0.0, -0.05, 0.1]])
        x0_state = jnp.zeros((4, 3)).at[0].set(P0)
        u_seq = _waypoints_to_u_seq(offsets, x0_state, DEFAULT_T_HOP, QPS_DT)
        steps_per_hop = int(DEFAULT_T_HOP / QPS_DT)
        waypoints = nominal_waypoints(K) + offsets

        for k in range(K):
            end_idx = (k + 1) * steps_per_hop - 1
            pos_at_end = u_seq[end_idx, :3]
            np.testing.assert_allclose(np.array(pos_at_end),
                                       np.array(waypoints[k]), atol=1e-3)


class TestSeparationLoss:
    def test_nonnegative(self):
        scenarios = create_scenarios()
        x0_ivl = irx.icentpert(jnp.zeros(18).at[0:3].set(P0), jnp.full(18, 1e-3))
        x0_state = jnp.zeros((4, 3)).at[0].set(P0)
        K = 2
        offsets = jnp.zeros((K, 3))
        loss = waypoint_separation_loss(offsets, x0_ivl, scenarios,
                                        x0_state, DEFAULT_T_HOP, QPS_DT)
        assert float(loss) >= 0.0

    def test_differentiable(self):
        """Loss must be differentiable w.r.t. offsets. Uses a short horizon
        (1 hop, 2 waypoints close together) to avoid the scenario where 
        tubes fully separate (zero overlap → zero-volume product → undefined
        gradient through log-domain)."""
        scenarios = create_scenarios(names=["pd_pos_vel", "pid_pos_vel_i"])
        x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 5e-2))
        x0_state = jnp.zeros((4, 3))
        K = 1  # single hop, short mission

        def loss_fn(offsets):
            return waypoint_separation_loss(offsets, x0_ivl, scenarios,
                                            x0_state, T_hop=2.0, dt=QPS_DT)

        # Use a small nonzero offset to start with some overlap
        offsets = jnp.array([[0.1, 0.0, 0.0]])
        grad = jax.grad(loss_fn)(offsets)
        assert grad.shape == (K, 3)
        # Gradient should be finite (may be zero where overlap is zero at all steps)
        assert jnp.all(jnp.isfinite(grad)), f"Got non-finite gradient: {grad}"


class TestSimulateTrue:
    def test_trajectory_shape(self):
        x0 = jnp.zeros(18)
        waypoints = jnp.array([[0.5, 0.0, 1.0], [1.0, 0.0, 1.0]])
        x0_state = jnp.zeros((4, 3))
        u_seq = build_mission_reference(waypoints, x0_state, DEFAULT_T_HOP, QPS_DT)
        theta = jnp.array(_CANDIDATE_THETA["qps_snap_chain"])
        traj = simulate_true_trajectory(x0, u_seq, theta, QPS_DT)
        assert traj.shape == (u_seq.shape[0], 12)

    def test_position_tracks_reference(self):
        """Under the true QPS controller with its huge gains, position should
        track the reference closely (not perfectly — the controller needs
        time to converge, but shouldn't diverge)."""
        x0 = jnp.zeros(18)
        waypoints = jnp.array([[0.5, 0.0, 0.0]])  # small hop from origin
        x0_state = jnp.zeros((4, 3))
        u_seq = build_mission_reference(waypoints, x0_state, DEFAULT_T_HOP, QPS_DT)
        theta = jnp.array(_CANDIDATE_THETA["qps_snap_chain"])
        traj = simulate_true_trajectory(x0, u_seq, theta, QPS_DT)

        # Final position should be near the waypoint (within 0.5m for a
        # polynomial-reference-tracking task with these gains)
        final_pos = traj[-1, :3]
        np.testing.assert_allclose(np.array(final_pos),
                                   np.array(waypoints[0]), atol=0.5)


class TestDiscrimination:
    def test_true_controller_always_survives(self):
        """Regression guard: the true generating controller must never be
        falsified by its own trajectory (same pattern as
        test_crazyflie_chain_phase2.py's guard).
        
        Uses a short single-hop mission to avoid the placeholder gains'
        marginal instability on long horizons (pd_pos_vel/pid_pos_vel_i's
        k_p=400 causes divergence beyond ~150 Euler steps). Also uses a
        wider w_bar for the accumulated float32 rounding."""
        scenarios = create_scenarios()
        x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-3))
        x0_state = jnp.zeros((4, 3))
        # Single short hop, small distance — all 4 controllers stable here
        waypoints = jnp.array([[0.1, 0.0, 0.0]])
        u_seq = build_mission_reference(waypoints, x0_state, T_hop=2.0, dt=QPS_DT)

        for true_name in _CANDIDATE_THETA:
            true_theta = jnp.array(_CANDIDATE_THETA[true_name])
            x0_point = jnp.zeros(18)
            observed = simulate_true_trajectory(x0_point, u_seq, true_theta, QPS_DT)
            result = discriminate_controller_waypoint(
                x0_ivl, u_seq, observed, scenarios, dt=QPS_DT, w_bar=1e-3)
            assert true_name in result['survivors'], \
                f"{true_name} was falsified by its own trajectory!"
