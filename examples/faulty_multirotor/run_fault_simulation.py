"""
Simulation script for the Faulty Planar Multirotor.

Demonstrates:
1. Actuator fault: one rotor breaks, reducing thrust effectiveness
2. Sensor fault: IMU drift causes incorrect yaw (theta) readings,
   leading the controller to apply incorrect corrections

Produces animations saved as GIF files.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scipy.linalg
from faulty_planar_multirotor import FaultyPlanarMultirotor
from visualization_functions import (
    simulate_multirotor,
    create_multirotor_animation,
    create_sensor_fault_animation,
)


# --------------------------------------------------------------------------- #
# System and parameters
# --------------------------------------------------------------------------- #
multirotor = FaultyPlanarMultirotor()
g = 9.81

# Equilibrium: hover at origin
x_eq = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
u_eq = np.array([g, 0.0])  # thrust = g to counteract gravity, zero angular accel

# Nominal parameters (no fault)
p_nominal = np.array([1.0, 1.0])

# Actuator fault: one rotor broken -> 50% thrust, 70% angular effectiveness
p_actuator_fault = np.array([0.5, 0.7])

# --------------------------------------------------------------------------- #
# Linearize around hover for LQR design
# --------------------------------------------------------------------------- #
# Jacobians of f w.r.t. x and u at equilibrium (analytical):
#   dx/dt = f(x, u, w=0, p=[1,1])
#
# At hover: theta=0, u1=g, u2=0
#   A = df/dx:
#     row 0 (px_dot):  [0, 0, 1, 0, 0]
#     row 1 (py_dot):  [0, 0, 0, 1, 0]
#     row 2 (vx_dot):  [0, 0, 0, 0, -u1*cos(theta)] = [0,0,0,0,-g]
#     row 3 (vy_dot):  [0, 0, 0, 0, -u1*sin(theta)] = [0,0,0,0, 0]
#     row 4 (theta_dot): [0, 0, 0, 0, 0]
#
#   B = df/du:
#     row 0: [0, 0]
#     row 1: [0, 0]
#     row 2: [-sin(theta), 0] = [0, 0]
#     row 3: [cos(theta), 0]  = [1, 0]
#     row 4: [0, 1]

A = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, -g],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=float)

B = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
], dtype=float)

# LQR weights
Q_lqr = np.diag([10.0, 10.0, 1.0, 1.0, 5.0])
R_lqr = np.diag([1.0, 1.0])

# Solve continuous-time algebraic Riccati equation
P_lqr = scipy.linalg.solve_continuous_are(A, B, Q_lqr, R_lqr)
K_lqr = np.linalg.solve(R_lqr, B.T @ P_lqr)  # K such that u = -K(x - x_eq) + u_eq


def lqr_controller(t, x, K=K_lqr, x_ref=x_eq, u_ref=u_eq):
    """LQR feedback controller around hover equilibrium."""
    dx = x - x_ref
    u = u_ref - K @ dx
    # Clamp thrust to be non-negative
    u[0] = max(u[0], 0.0)
    return u


def zero_disturbance(t, x):
    return np.array([0.0])


# --------------------------------------------------------------------------- #
# Simulation parameters
# --------------------------------------------------------------------------- #
dt = 0.01
T_sim = 5.0
num_steps = int(T_sim / dt)
x0 = np.array([-2.0, 4.0, 0.5, -0.5, 0.1])  # off-equilibrium initial condition


# --------------------------------------------------------------------------- #
# 1. Nominal simulation (no fault)
# --------------------------------------------------------------------------- #
print("Running nominal simulation...")
times_nom, states_nom, controls_nom = simulate_multirotor(
    multirotor, x0, lqr_controller, zero_disturbance, p_nominal, dt, num_steps,
)

# --------------------------------------------------------------------------- #
# 2. Actuator fault simulation (broken rotor)
# --------------------------------------------------------------------------- #
print("Running actuator fault simulation...")
times_act, states_act, controls_act = simulate_multirotor(
    multirotor, x0, lqr_controller, zero_disturbance, p_actuator_fault, dt, num_steps,
)

# --------------------------------------------------------------------------- #
# 3. Sensor fault simulation (IMU drift on theta)
# --------------------------------------------------------------------------- #
# The controller uses a biased theta reading. We simulate this by wrapping
# the controller so it receives theta + drift_rate * t instead of true theta.
imu_drift_rate = 0.05  # rad/s drift

print("Running sensor fault simulation...")


def controller_with_imu_drift(t, x):
    """Controller that sees a drifting theta measurement."""
    x_sensed = x.copy()
    x_sensed[4] = x[4] + imu_drift_rate * t  # biased theta reading
    return lqr_controller(t, x_sensed)


times_sens, states_sens, controls_sens = simulate_multirotor(
    multirotor, x0, controller_with_imu_drift, zero_disturbance, p_nominal, dt, num_steps,
)

# Build the "perceived" trajectory (what the controller thinks the state is)
states_perceived = states_sens.copy()
states_perceived[:, 4] = states_sens[:, 4] + imu_drift_rate * times_sens

# --------------------------------------------------------------------------- #
# 4. Create animations
# --------------------------------------------------------------------------- #
output_dir = os.path.dirname(__file__)

print("Creating actuator fault comparison animation...")
create_multirotor_animation(
    trajectories=[
        (times_nom, states_nom),
        (times_act, states_act),
    ],
    labels=["Nominal", "Actuator Fault (50% thrust)"],
    colors=["green", "red"],
    dt=dt,
    filename=os.path.join(output_dir, "actuator_fault_animation.gif"),
    title="Actuator Fault: Broken Rotor",
    speed_factor=4,
)

print("Creating sensor fault animation...")
create_sensor_fault_animation(
    traj_nominal=(times_nom, states_nom),
    traj_sensed=(times_sens, states_perceived),
    dt=dt,
    filename=os.path.join(output_dir, "sensor_fault_animation.gif"),
    title="Sensor Fault: IMU Yaw Drift",
    speed_factor=4,
)

# Also create a combined 3-trajectory animation
print("Creating combined fault comparison animation...")
create_multirotor_animation(
    trajectories=[
        (times_nom, states_nom),
        (times_act, states_act),
        (times_sens, states_sens),
    ],
    labels=["Nominal", "Actuator Fault", "Sensor Fault (IMU drift)"],
    colors=["green", "red", "orange"],
    dt=dt,
    filename=os.path.join(output_dir, "combined_fault_animation.gif"),
    title="Fault Comparison: Nominal vs Actuator vs Sensor",
    speed_factor=4,
)

print("All simulations and animations complete.")
