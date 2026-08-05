import os
import numpy as np
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax.numpy as jnp
from pathlib import Path
import sys

# Import module under test from examples/go2
GO2_DIR = Path(__file__).resolve().parents[1]
if str(GO2_DIR) not in sys.path:
    sys.path.insert(0, str(GO2_DIR))

from go2_separating_feedback_controller import (  # noqa: E402
    Interval,
    FaultScenario,
    interval_dynamics_step,
    observation_interval,
    propagate_feedback_scenario,
)


def test_observation_interval_linear_bounds_match_manual():
    x = Interval(
        lower=jnp.array([-1.0, 2.0, -0.5]),
        upper=jnp.array([3.0, 4.0, 1.5]),
    )
    C = jnp.array([
        [2.0, -1.0, 0.5],
        [-3.0, 0.0, -2.0],
    ])

    y = observation_interval(x, C)

    expected_lower = np.array([
        2.0 * (-1.0) + (-1.0) * 4.0 + 0.5 * (-0.5),
        (-3.0) * 3.0 + 0.0 * 2.0 + (-2.0) * 1.5,
    ])
    expected_upper = np.array([
        2.0 * 3.0 + (-1.0) * 2.0 + 0.5 * 1.5,
        (-3.0) * (-1.0) + 0.0 * 4.0 + (-2.0) * (-0.5),
    ])

    np.testing.assert_allclose(np.asarray(y.lower), expected_lower, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(y.upper), expected_upper, rtol=1e-6, atol=1e-6)
    assert np.all(np.asarray(y.lower) <= np.asarray(y.upper))


def test_interval_dynamics_step_handles_sin_minimum_crossing():
    # Theta interval includes -pi/2, so sin lower bound should reach -1.
    x = Interval(
        lower=jnp.array([0.0, 0.0, -2.0]),
        upper=jnp.array([0.0, 0.0, -1.0]),
    )
    u = jnp.array([0.0, 1.0, 0.0])
    alpha = Interval(lower=jnp.array([1.0]), upper=jnp.array([1.0]))

    x_next = interval_dynamics_step(x, u, alpha, dt=1.0)

    # dpx/dt = -sin(theta), theta in [-2,-1] => dpx in [sin(1), 1]
    assert float(x_next.lower[0]) <= np.sin(1.0) + 1e-6
    assert float(x_next.upper[0]) >= 1.0 - 1e-6


def test_sensor_fault_changes_feedback_closed_loop_behavior():
    N = 4
    x_nom = jnp.zeros((N, 3))
    u_nom = jnp.zeros((N, 3))
    y_nom = jnp.zeros((N, 3))
    C = jnp.eye(3)

    x0 = Interval(
        lower=jnp.array([0.0, 0.0, 0.1]),
        upper=jnp.array([0.0, 0.0, 0.1]),
    )

    # Feedback reacts to measured theta only, driving omega command.
    K = jnp.zeros((N, 3, 3))
    K = K.at[:, 2, 2].set(1.0)

    nominal = FaultScenario(
        name="Nominal",
        alpha_range=(1.0, 1.0),
        scale_range=(1.0, 1.0),
        bias_range=(0.0, 0.0),
    )
    sensor_fault = FaultScenario(
        name="Sensor Fault",
        alpha_range=(1.0, 1.0),
        scale_range=(1.2, 1.2),
        bias_range=(0.3, 0.3),
    )

    x_final_nom, _ = propagate_feedback_scenario(
        x0_int=x0,
        x_nom_traj=x_nom,
        u_nom_traj=u_nom,
        y_nom_traj=y_nom,
        K_traj=K,
        C=C,
        scenario=nominal,
        dt=0.1,
        obs_uncertainty=0.0,
    )
    x_final_fault, _ = propagate_feedback_scenario(
        x0_int=x0,
        x_nom_traj=x_nom,
        u_nom_traj=u_nom,
        y_nom_traj=y_nom,
        K_traj=K,
        C=C,
        scenario=sensor_fault,
        dt=0.1,
        obs_uncertainty=0.0,
    )

    # Sensor fault should alter observation and therefore the feedback path,
    # producing a different final heading.
    assert not np.isclose(float(x_final_nom.lower[2]), float(x_final_fault.lower[2]))
