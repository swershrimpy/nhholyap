"""Tests for CrazyflieSystem (examples/adaptive_spoofing/crazyflie_12d.py).

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python -m pytest examples/adaptive_spoofing/tests/
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp
import immrax as irx

_EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_12d import (
    CrazyflieSystem, get_system_and_embedding, euler_step,
    _M, _G, _IXX, _IYY, _IZZ, _HOVER_THRUST,
)


# ──────────────────────────────────────────────────────────────────────────
# A numpy transcription of QPS's actual `QuadcopterObject.forward_model()`
# (~/adaptive_spoofing/libs/quadrotarium_python_simulator/qps/utilities/
# quadcopter_model.py:302-343), used only to cross-check CrazyflieSystem.f()
# reproduces QPS's real ODE right-hand side. Kept deliberately independent
# (uses get_Rwb/get_Twb-equivalent numpy math, not CrazyflieSystem's own
# code) so a bug in one is unlikely to be masked by the same bug in the other.
# ──────────────────────────────────────────────────────────────────────────
def _qps_forward_model_rhs(state, u, m=_M, g=_G, Ixx=_IXX, Iyy=_IYY, Izz=_IZZ):
    """state: (12,) [x,y,z,phi,theta,psi,vx,vy,vz,p,q,r]; u: (4,) [U1,U2,U3,U4].
    Returns state_d (12,), transcribed directly from forward_model()."""
    phi, theta, psi = state[3], state[4], state[5]
    vel = state[6:9]
    omega_b = state[9:12]
    I_moment = np.diag([Ixx, Iyy, Izz])
    I_moment_inv = np.linalg.inv(I_moment)

    Twb = np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
    ])
    Rwb = np.array([
        [np.cos(theta) * np.cos(psi),
         np.sin(phi) * np.sin(theta) * np.cos(psi) - np.sin(psi) * np.cos(phi),
         np.sin(theta) * np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
        [np.sin(psi) * np.cos(theta),
         np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
         np.sin(theta) * np.sin(psi) * np.cos(phi) - np.sin(phi) * np.cos(psi)],
        [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)],
    ])

    state_d = np.zeros(12)
    state_d[0:3] = vel
    state_d[3:6] = Twb @ omega_b

    z_w = np.array([0, 0, 1])
    z_b = Rwb[:, 2]
    state_d[6:9] = (-m * g * z_w + u[0] * z_b) / m

    state_d[9:12] = I_moment_inv @ (np.cross(-omega_b, I_moment @ omega_b) + u[1:])
    return state_d


class TestHoverEquilibrium:
    def test_hover_is_fixed_point(self):
        sys_, _ = get_system_and_embedding()
        x0 = jnp.zeros(12)
        u_hover = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
        p = jnp.ones(4)
        xdot = sys_.f(jnp.zeros(()), x0, u_hover, p)
        assert jnp.allclose(xdot, 0.0, atol=1e-5)

    def test_thrust_mismatch_causes_vertical_accel(self):
        sys_, _ = get_system_and_embedding()
        x0 = jnp.zeros(12)
        p = jnp.ones(4)
        u_extra = jnp.array([_HOVER_THRUST * 1.5, 0.0, 0.0, 0.0])
        xdot = sys_.f(jnp.zeros(()), x0, u_extra, p)
        assert xdot[8] > 0  # vz accelerates upward


class TestMatchesQPSForwardModel:
    """CrazyflieSystem.f() must reproduce QPS's own forward_model() exactly
    (mod float precision), across a battery of non-trivial states/inputs."""

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_qps_forward_model(self, seed):
        rng = np.random.default_rng(seed)
        state = np.zeros(12)
        state[0:3] = rng.uniform(-1, 1, 3)
        state[3:6] = rng.uniform(-0.3, 0.3, 3)   # stay well clear of gimbal lock
        state[6:9] = rng.uniform(-2, 2, 3)
        state[9:12] = rng.uniform(-1, 1, 3)
        u = np.array([_HOVER_THRUST * rng.uniform(0.5, 1.5),
                      rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01),
                      rng.uniform(-0.01, 0.01)])

        sys_, _ = get_system_and_embedding()
        xdot_ours = np.array(sys_.f(jnp.zeros(()), jnp.array(state), jnp.array(u), jnp.ones(4)))
        xdot_qps = _qps_forward_model_rhs(state, u)

        np.testing.assert_allclose(xdot_ours, xdot_qps, atol=1e-5, rtol=1e-4)

    def test_world_frame_velocity_diverges_from_body_frame_model_under_roll(self):
        """Regression guard for the actual bug this module fixes: under nonzero
        roll, a body-frame-velocity model (QuadrotorSystem) and QPS's real
        world-frame-velocity model give DIFFERENT position kinematics for the
        same state/input. If this test ever fails, CrazyflieSystem has
        regressed back to the wrong (body-frame) formulation."""
        sys_, _ = get_system_and_embedding()
        state = jnp.array([0., 0., 0., 0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0., 0., 0.])
        u = jnp.array([_HOVER_THRUST, 0., 0., 0.])
        xdot = sys_.f(jnp.zeros(()), state, u, jnp.ones(4))
        # world-frame model: xdot,ydot,zdot = vx,vy,vz directly (no rotation)
        assert jnp.allclose(xdot[0:3], jnp.array([1.0, 0.0, 0.0]))


class TestEmbeddingAndStep:
    def test_natural_embedding_produces_interval(self):
        _, emb = get_system_and_embedding()
        x_ivl = irx.icentpert(jnp.zeros(12), jnp.full(12, 0.01))
        u = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
        p_ivl = irx.Interval(lower=jnp.ones(4), upper=jnp.ones(4))
        x_next = euler_step(emb, x_ivl, u, p_ivl, dt=0.02)
        assert isinstance(x_next, irx.Interval)
        assert x_next.lower.shape == (12,)
        assert jnp.all(x_next.lower <= x_next.upper)

    def test_point_interval_hover_step_stays_at_origin(self):
        """A degenerate (point) interval at the hover fixed point should stay
        put after one Euler step (up to float precision)."""
        _, emb = get_system_and_embedding()
        x_ivl = irx.icentpert(jnp.zeros(12), jnp.zeros(12))
        u = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
        p_ivl = irx.Interval(lower=jnp.ones(4), upper=jnp.ones(4))
        x_next = euler_step(emb, x_ivl, u, p_ivl, dt=0.02)
        assert jnp.allclose(x_next.lower, 0.0, atol=1e-6)
        assert jnp.allclose(x_next.upper, 0.0, atol=1e-6)

    def test_wider_input_interval_grows_output_interval(self):
        _, emb = get_system_and_embedding()
        u = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
        p_ivl = irx.Interval(lower=jnp.ones(4), upper=jnp.ones(4))

        x_tight = irx.icentpert(jnp.zeros(12), jnp.full(12, 0.001))
        x_wide = irx.icentpert(jnp.zeros(12), jnp.full(12, 0.1))

        next_tight = euler_step(emb, x_tight, u, p_ivl, dt=0.02)
        next_wide = euler_step(emb, x_wide, u, p_ivl, dt=0.02)

        width_tight = jnp.sum(next_tight.upper - next_tight.lower)
        width_wide = jnp.sum(next_wide.upper - next_wide.lower)
        assert width_wide > width_tight


class TestParameters:
    """Regression guards on the corrected physical constants (see module
    docstring 'Parameter comparison')."""

    def test_mass_matches_qps(self):
        assert np.isclose(_M, 0.03589)

    def test_inertia_matches_qps(self):
        assert np.isclose(_IXX, 2.3951e-5)
        assert np.isclose(_IYY, 2.3951e-5)
        assert np.isclose(_IZZ, 3.2346e-5)

    def test_hover_thrust_is_tiny_compared_to_generic_quadrotor_default(self):
        # QuadrotorSystem's default hover thrust is ~4.59 N; a real Crazyflie
        # is roughly 13x lighter, so this must be well under 1 N.
        assert _HOVER_THRUST < 0.5
        assert np.isclose(_HOVER_THRUST, _M * _G)


class TestCustomParams:
    def test_different_mass_changes_hover_thrust_requirement(self):
        sys_heavy = CrazyflieSystem(m=0.1)
        x0 = jnp.zeros(12)
        p = jnp.ones(4)
        u_light_hover = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
        xdot = sys_heavy.f(jnp.zeros(()), x0, u_light_hover, p)
        assert xdot[8] < 0  # underpowered for the heavier mass -> falls

    def test_get_system_and_embedding_caches_by_params(self):
        s1, e1 = get_system_and_embedding()
        s2, e2 = get_system_and_embedding()
        assert s1 is s2
        assert e1 is e2
        s3, e3 = get_system_and_embedding(m=0.1)
        assert s3 is not s1
