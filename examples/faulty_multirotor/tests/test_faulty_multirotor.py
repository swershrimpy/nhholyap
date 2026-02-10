"""
Unit tests for the FaultyPlanarMultirotor system,
interval propagation, and fault models.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax.numpy as jnp
import immrax as irx
from faulty_planar_multirotor import FaultyPlanarMultirotor
from interval_functions import overlap_size_lax, propagate_interval_euler


class TestFaultyPlanarMultirotorInit(unittest.TestCase):
    """Test system initialization and dimensions."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()

    def test_state_dimension(self):
        self.assertEqual(self.sys.xlen, 5)

    def test_input_dimension(self):
        self.assertEqual(self.sys.ulen, 2)

    def test_disturbance_dimension(self):
        self.assertEqual(self.sys.wlen, 1)

    def test_output_noise_dimension(self):
        self.assertEqual(self.sys.vlen, 3)

    def test_parameter_dimension(self):
        self.assertEqual(self.sys.plen, 2)

    def test_evolution_type(self):
        self.assertEqual(self.sys.evolution, 'continuous')

    def test_gravity_constant(self):
        self.assertAlmostEqual(self.sys.g, 9.81)


class TestNominalDynamics(unittest.TestCase):
    """Test dynamics under nominal (no fault) conditions."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()
        self.p_nominal = jnp.array([1.0, 1.0])
        self.w_zero = jnp.array([0.0])

    def test_hover_equilibrium(self):
        """At hover (theta=0, u1=g, u2=0), x_dot should be ~zero."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        u = jnp.array([9.81, 0.0])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, self.p_nominal)
        np.testing.assert_allclose(x_dot, jnp.zeros(5), atol=1e-10)

    def test_freefall(self):
        """With zero thrust, vertical acceleration should be -g."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        u = jnp.array([0.0, 0.0])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, self.p_nominal)
        self.assertAlmostEqual(float(x_dot[0]), 0.0)  # px_dot = vx = 0
        self.assertAlmostEqual(float(x_dot[1]), 0.0)  # py_dot = vy = 0
        self.assertAlmostEqual(float(x_dot[2]), 0.0)  # vx_dot = 0
        self.assertAlmostEqual(float(x_dot[3]), -9.81, places=5)  # vy_dot = -g
        self.assertAlmostEqual(float(x_dot[4]), 0.0)  # theta_dot = 0

    def test_position_rate_equals_velocity(self):
        """px_dot = vx and py_dot = vy regardless of other states."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 0.5])
        u = jnp.array([9.81, 0.1])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, self.p_nominal)
        self.assertAlmostEqual(float(x_dot[0]), float(x[2]))
        self.assertAlmostEqual(float(x_dot[1]), float(x[3]))

    def test_angular_acceleration(self):
        """theta_dot = u2 under nominal conditions."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        u2_val = 1.5
        u = jnp.array([9.81, u2_val])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, self.p_nominal)
        self.assertAlmostEqual(float(x_dot[4]), u2_val)

    def test_tilted_thrust(self):
        """With theta != 0, thrust decomposes into x and y components."""
        theta = jnp.pi / 6  # 30 degrees
        x = jnp.array([0.0, 0.0, 0.0, 0.0, theta])
        u1 = 12.0
        u = jnp.array([u1, 0.0])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, self.p_nominal)

        expected_vx_dot = -u1 * jnp.sin(theta)
        expected_vy_dot = u1 * jnp.cos(theta) - 9.81
        self.assertAlmostEqual(float(x_dot[2]), float(expected_vx_dot), places=5)
        self.assertAlmostEqual(float(x_dot[3]), float(expected_vy_dot), places=5)

    def test_disturbance_effect(self):
        """Disturbance w1 should only affect vx_dot."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        u = jnp.array([9.81, 0.0])
        w = jnp.array([2.5])
        x_dot = self.sys.f(0.0, x, u, w, self.p_nominal)
        self.assertAlmostEqual(float(x_dot[2]), 2.5, places=5)
        # Other derivatives should be same as hover
        self.assertAlmostEqual(float(x_dot[0]), 0.0)
        self.assertAlmostEqual(float(x_dot[1]), 0.0)
        self.assertAlmostEqual(float(x_dot[3]), 0.0, places=5)
        self.assertAlmostEqual(float(x_dot[4]), 0.0)


class TestActuatorFault(unittest.TestCase):
    """Test actuator fault model (broken rotor)."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()
        self.w_zero = jnp.array([0.0])

    def test_half_thrust(self):
        """With p[0]=0.5, effective thrust should be halved."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        u = jnp.array([9.81, 0.0])
        p_fault = jnp.array([0.5, 1.0])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, p_fault)
        # vy_dot = 0.5*9.81*cos(0) - 9.81 = -4.905
        expected_vy_dot = 0.5 * 9.81 - 9.81
        self.assertAlmostEqual(float(x_dot[3]), expected_vy_dot, places=5)

    def test_zero_thrust_fault(self):
        """With p[0]=0, thrust is zero -> freefall."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        u = jnp.array([9.81, 0.0])
        p_fault = jnp.array([0.0, 1.0])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, p_fault)
        self.assertAlmostEqual(float(x_dot[3]), -9.81, places=5)

    def test_angular_fault(self):
        """With p[1]=0.5, angular acceleration is halved."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        u = jnp.array([9.81, 2.0])
        p_fault = jnp.array([1.0, 0.5])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, p_fault)
        self.assertAlmostEqual(float(x_dot[4]), 1.0)  # 0.5 * 2.0

    def test_full_fault_both(self):
        """Both actuator parameters at 0 -> freefall, no angular control."""
        x = jnp.array([0.0, 5.0, 1.0, -1.0, 0.3])
        u = jnp.array([15.0, 3.0])
        p_fault = jnp.array([0.0, 0.0])
        x_dot = self.sys.f(0.0, x, u, self.w_zero, p_fault)
        self.assertAlmostEqual(float(x_dot[0]), float(x[2]))  # vx
        self.assertAlmostEqual(float(x_dot[1]), float(x[3]))  # vy
        self.assertAlmostEqual(float(x_dot[2]), 0.0)  # no thrust
        self.assertAlmostEqual(float(x_dot[3]), -9.81, places=5)  # freefall
        self.assertAlmostEqual(float(x_dot[4]), 0.0)  # no angular control

    def test_nominal_is_identity(self):
        """p=[1,1] should give same result as base PlanarMultirotor dynamics."""
        x = jnp.array([1.0, 2.0, 0.5, -0.3, 0.2])
        u = jnp.array([10.0, 0.5])
        w = jnp.array([0.1])
        p_nominal = jnp.array([1.0, 1.0])

        x_dot = self.sys.f(0.0, x, u, w, p_nominal)
        # Manual computation
        expected = jnp.array([
            x[2],
            x[3],
            -u[0] * jnp.sin(x[4]) + w[0],
            u[0] * jnp.cos(x[4]) - 9.81,
            u[1],
        ])
        np.testing.assert_allclose(x_dot, expected, atol=1e-6)


class TestSensorFaultObserver(unittest.TestCase):
    """Test the observer (sensor) model with IMU drift."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()

    def test_nominal_observation(self):
        """With zero noise, observer returns [px, py, theta]."""
        x = jnp.array([1.0, 2.0, 0.5, -0.3, 0.4])
        v = jnp.array([0.0, 0.0, 0.0])
        y = self.sys.h(0.0, x, v)
        np.testing.assert_allclose(y, jnp.array([1.0, 2.0, 0.4]), atol=1e-10)

    def test_imu_drift_bias(self):
        """IMU drift adds bias to theta reading."""
        x = jnp.array([1.0, 2.0, 0.5, -0.3, 0.4])
        drift = 0.15
        v = jnp.array([0.0, 0.0, drift])
        y = self.sys.h(0.0, x, v)
        self.assertAlmostEqual(float(y[0]), 1.0)
        self.assertAlmostEqual(float(y[1]), 2.0)
        self.assertAlmostEqual(float(y[2]), 0.4 + drift)

    def test_position_noise(self):
        """Position noise affects px and py readings."""
        x = jnp.array([3.0, 4.0, 0.0, 0.0, 0.0])
        v = jnp.array([0.1, -0.2, 0.0])
        y = self.sys.h(0.0, x, v)
        self.assertAlmostEqual(float(y[0]), 3.1, places=5)
        self.assertAlmostEqual(float(y[1]), 3.8, places=5)
        self.assertAlmostEqual(float(y[2]), 0.0, places=5)

    def test_all_noise(self):
        """All noise components active."""
        x = jnp.array([1.0, 2.0, 0.5, -0.3, 0.4])
        v = jnp.array([0.05, -0.05, 0.1])
        y = self.sys.h(0.0, x, v)
        np.testing.assert_allclose(y, jnp.array([1.05, 1.95, 0.5]), atol=1e-10)

    def test_observer_output_dimension(self):
        """Observer should return 3 values."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        v = jnp.array([0.0, 0.0, 0.0])
        y = self.sys.h(0.0, x, v)
        self.assertEqual(y.shape, (3,))


class TestNumpyDynamics(unittest.TestCase):
    """Test that f_np matches f for consistency."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()

    def test_np_matches_jax(self):
        """f_np and f should give the same results."""
        x = np.array([1.0, 2.0, 0.5, -0.3, 0.2])
        u = np.array([10.0, 0.5])
        w = np.array([0.1])
        p = np.array([0.7, 0.9])

        x_dot_np = self.sys.f_np(0.0, x, u, w, p)
        x_dot_jax = np.array(self.sys.f(0.0, jnp.array(x), jnp.array(u),
                                         jnp.array(w), jnp.array(p)))
        np.testing.assert_allclose(x_dot_np, x_dot_jax, atol=1e-6)


class TestIntervalPropagation(unittest.TestCase):
    """Test interval propagation under faults."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()

    def test_euler_propagation_nominal(self):
        """Interval propagation at hover should remain small."""
        emb = irx.natemb(self.sys)
        x0 = irx.icentpert(
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        )
        u = irx.icentpert(jnp.array([9.81, 0.0]), jnp.zeros(2))
        w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))
        p = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))

        x1 = propagate_interval_euler(emb, x0, u, w, p, 0.01)
        # After one small step at hover, intervals should still be small
        widths = x1.upper - x1.lower
        for i in range(5):
            self.assertGreater(float(widths[i]), 0.0, f"Width of state {i} should be positive")
            self.assertLess(float(widths[i]), 1.0, f"Width of state {i} should be bounded")

    def test_fault_widens_intervals(self):
        """Uncertain fault parameter should produce wider intervals."""
        emb = irx.natemb(self.sys)
        x0 = irx.icentpert(
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        )
        u = irx.icentpert(jnp.array([9.81, 0.0]), jnp.zeros(2))
        w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))

        # Nominal: exact parameter
        p_nom = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
        x1_nom = propagate_interval_euler(emb, x0, u, w, p_nom, 0.01)

        # Faulty: uncertain parameter
        p_fault = irx.icentpert(jnp.array([0.5, 0.8]), jnp.array([0.25, 0.1]))
        x1_fault = propagate_interval_euler(emb, x0, u, w, p_fault, 0.01)

        # Faulty intervals should be at least as wide
        widths_nom = x1_nom.upper - x1_nom.lower
        widths_fault = x1_fault.upper - x1_fault.lower

        # At least vy (index 3) should be wider because thrust uncertainty
        self.assertGreaterEqual(float(widths_fault[3]), float(widths_nom[3]) - 1e-6)

    def test_interval_contains_point(self):
        """Interval propagation should contain the point propagation."""
        emb = irx.natemb(self.sys)
        x0_center = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        x0_pert = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])
        x0_ivl = irx.icentpert(x0_center, x0_pert)
        u_val = jnp.array([9.81, 0.0])
        u = irx.icentpert(u_val, jnp.zeros(2))
        w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))
        p = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))

        dt = 0.01
        x1_ivl = propagate_interval_euler(emb, x0_ivl, u, w, p, dt)

        # Propagate center point
        x_dot = self.sys.f(0.0, x0_center, u_val, jnp.zeros(1), jnp.array([1.0, 1.0]))
        x1_center = x0_center + dt * x_dot

        # Center should be contained in the interval
        for i in range(5):
            self.assertGreaterEqual(float(x1_ivl.upper[i]), float(x1_center[i]) - 1e-6)
            self.assertLessEqual(float(x1_ivl.lower[i]), float(x1_center[i]) + 1e-6)


class TestOverlapFunction(unittest.TestCase):
    """Test interval overlap computation."""

    def test_no_overlap(self):
        i1 = irx.interval(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
        i2 = irx.interval(jnp.array([2.0, 2.0]), jnp.array([3.0, 3.0]))
        result = overlap_size_lax(i1, i2)
        self.assertAlmostEqual(float(result), 0.0)

    def test_full_overlap(self):
        i1 = irx.interval(jnp.array([0.0, 0.0]), jnp.array([4.0, 4.0]))
        i2 = irx.interval(jnp.array([1.0, 1.0]), jnp.array([3.0, 3.0]))
        result = overlap_size_lax(i1, i2)
        self.assertAlmostEqual(float(result), 4.0)

    def test_partial_overlap(self):
        i1 = irx.interval(jnp.array([0.0, 0.0]), jnp.array([2.0, 2.0]))
        i2 = irx.interval(jnp.array([1.0, 1.0]), jnp.array([3.0, 3.0]))
        result = overlap_size_lax(i1, i2)
        self.assertAlmostEqual(float(result), 1.0)


class TestSimulationConsistency(unittest.TestCase):
    """Integration-level tests for simulation behavior."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()

    def _euler_sim(self, x0, u_func, w_func, p, dt, steps):
        """Simple Euler simulation helper."""
        x = np.array(x0, dtype=float)
        trajectory = [x.copy()]
        for i in range(steps):
            t = i * dt
            u = np.array(u_func(t, x), dtype=float)
            w = np.array(w_func(t, x), dtype=float)
            x = x + dt * self.sys.f_np(t, x, u, w, p)
            trajectory.append(x.copy())
        return np.array(trajectory)

    def test_hover_stability(self):
        """Starting at hover equilibrium should stay there."""
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        p = np.array([1.0, 1.0])
        states = self._euler_sim(
            x0,
            lambda t, x: np.array([9.81, 0.0]),
            lambda t, x: np.array([0.0]),
            p, 0.001, 1000,
        )
        # Should remain near zero
        np.testing.assert_allclose(states[-1], x0, atol=1e-3)

    def test_freefall_trajectory(self):
        """Zero thrust -> parabolic freefall in y."""
        x0 = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
        p = np.array([1.0, 1.0])
        dt = 0.001
        steps = 1000
        states = self._euler_sim(
            x0,
            lambda t, x: np.array([0.0, 0.0]),
            lambda t, x: np.array([0.0]),
            p, dt, steps,
        )
        t_final = steps * dt
        # y should decrease: y = y0 - 0.5*g*t^2
        expected_y = 10.0 - 0.5 * 9.81 * t_final**2
        self.assertAlmostEqual(states[-1, 1], expected_y, places=1)

    def test_actuator_fault_causes_descent(self):
        """With 50% thrust at hover input, the quadrotor should descend."""
        x0 = np.array([0.0, 5.0, 0.0, 0.0, 0.0])
        p_fault = np.array([0.5, 1.0])
        dt = 0.01
        steps = 100
        states = self._euler_sim(
            x0,
            lambda t, x: np.array([9.81, 0.0]),  # hover input, but only 50% effective
            lambda t, x: np.array([0.0]),
            p_fault, dt, steps,
        )
        # Should have descended from y=5
        self.assertLess(states[-1, 1], 5.0)
        # vy should be negative (falling)
        self.assertLess(states[-1, 3], 0.0)

    def test_sensor_fault_divergence(self):
        """IMU drift should cause the controller to make wrong corrections."""
        import scipy.linalg

        g = 9.81
        A = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, -g],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=float)
        B = np.array([
            [0, 0], [0, 0], [0, 0], [1, 0], [0, 1],
        ], dtype=float)
        Q = np.diag([10.0, 10.0, 1.0, 1.0, 5.0])
        R = np.diag([1.0, 1.0])
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)

        x_eq = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
        u_eq = np.array([g, 0.0])
        drift_rate = 0.1

        def controller_nominal(t, x):
            u = u_eq - K @ (x - x_eq)
            u[0] = max(u[0], 0.0)
            return u

        def controller_drifted(t, x):
            x_s = x.copy()
            x_s[4] = x[4] + drift_rate * t
            u = u_eq - K @ (x_s - x_eq)
            u[0] = max(u[0], 0.0)
            return u

        x0 = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
        p = np.array([1.0, 1.0])
        dt = 0.01
        steps = 300

        states_nom = self._euler_sim(x0, controller_nominal, lambda t, x: np.array([0.0]), p, dt, steps)
        states_drift = self._euler_sim(x0, controller_drifted, lambda t, x: np.array([0.0]), p, dt, steps)

        # After some time, the drifted trajectory should diverge from nominal
        dist = np.linalg.norm(states_nom[-1, :2] - states_drift[-1, :2])
        self.assertGreater(dist, 0.1, "Sensor fault should cause trajectory divergence")


if __name__ == "__main__":
    unittest.main()
