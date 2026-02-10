import jax
import jax.numpy as jnp
import numpy as np
import immrax as irx


class FaultyPlanarMultirotor(irx.system.System):
    """
    Planar multirotor with actuator and sensor fault models.

    States: [px, py, vx, vy, theta]
        px, py  - 2D position
        vx, vy  - 2D velocity
        theta   - pitch angle

    Inputs: [u1, u2]
        u1 - total thrust magnitude
        u2 - angular (pitch) acceleration

    Disturbances: [w1]
        w1 - additive disturbance on vx dynamics

    Parameters: [p1, p2]
        p1 - thrust effectiveness (1.0 = nominal, <1.0 = rotor fault)
        p2 - angular accel effectiveness (1.0 = nominal)

    Actuator Fault Model:
        A broken rotor reduces thrust effectiveness. Modeled as
        p1 * u1 in the dynamics, where p1 in [0, 1].
        p1 = 1.0 means nominal; p1 = 0.5 means 50% thrust loss.
        A broken rotor also affects angular acceleration input.

    Sensor Fault Model (IMU drift):
        The observer h() reports theta with an additive bias,
        modeling IMU yaw drift. The bias is part of the measurement
        noise/offset vector v.

    Observer:
        y = [px, py, theta_measured]
        where theta_measured = theta + v[2] (v[2] is the IMU drift bias)
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 5   # px, py, vx, vy, theta
        self.ulen = 2   # u1 (thrust), u2 (angular accel)
        self.wlen = 1   # disturbance on vx
        self.vlen = 3   # noise on px, py, theta observations
        self.plen = 2   # p1 (thrust effectiveness), p2 (angular effectiveness)
        self.g = 9.81

    def f(self, t, x, u, w, p):
        """Continuous-time dynamics with fault parameters."""
        vx = x[2]
        vy = x[3]
        theta = x[4]

        u1 = u[0]
        u2 = u[1]
        w1 = w[0]

        # p[0]: thrust effectiveness (actuator fault on rotor)
        # p[1]: angular accel effectiveness
        thrust = p[0] * u1
        ang_accel = p[1] * u2

        return jnp.array([
            vx,
            vy,
            -thrust * jnp.sin(theta) + w1,
            thrust * jnp.cos(theta) - self.g,
            ang_accel,
        ])

    def h(self, t, x, v):
        """
        Observer: measures px, py, and theta with noise.
        v[2] models IMU drift (sensor fault on yaw reading).
        """
        return jnp.array([
            x[0] + v[0],   # px + noise
            x[1] + v[1],   # py + noise
            x[4] + v[2],   # theta + IMU drift bias
        ])

    def f_np(self, t, x, u, w, p):
        """NumPy version for simulation outside JAX (e.g., animations)."""
        vx = x[2]
        vy = x[3]
        theta = x[4]

        u1 = u[0]
        u2 = u[1]
        w1 = w[0]

        thrust = p[0] * u1
        ang_accel = p[1] * u2

        return np.array([
            vx,
            vy,
            -thrust * np.sin(theta) + w1,
            thrust * np.cos(theta) - self.g,
            ang_accel,
        ])
