import jax
import jax.numpy as jnp
import immrax as irx
from typing import Tuple
from inverted_pendulum import SimpleInvertedPendulum
import numpy as np
import sys
import os
import copy

# Add the parent directory of 'nhholyap' to the Python path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "nhholyap"))

jax.devices()


if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from duality_clf import find_lyapunov_and_gain, find_lyapunov_and_gain_polytope

X_PERTURBATION = jnp.array([.1, 1])
T_MAX_PLOTTING = 7
def get_polytope_list_from_corners(A_corners, B_corners):
    A_list = []
    B_list = []
    for A_corner in A_corners:
        for B_corner in B_corners:
            A_list.append(copy.deepcopy(A_corner))
            B_list.append(copy.deepcopy(B_corner))
    return A_list, B_list

def duality_linear_controller_inverted_pendulum(x0: jnp.array, mu):
    """
    Compute a linear feedback control law on inverted pendulum using duality CLF constraint with input norm constraint.
    Input:
    x0: initial state of pendulum
    Output:
    u: feedback control effort.
    K: feedback gain matrix.
    """
    sys = SimpleInvertedPendulum()
    sys_mjacM = irx.mjacM(sys.f)
    Ms = sys_mjacM(
        jnp.array([0.0]), # bound for t. Any trivial bound works for TI system.
        irx.icentpert(x0, X_PERTURBATION), #perturbation on x
        irx.icentpert(jnp.zeros(1), jnp.array(mu)), #perturbation on u.
        centers=(
            (
                jnp.array([0.]), # center for t. Does not matter
                x0, # center of x
                jnp.array([0.]) # center of u
            ),))
    ldi_A = Ms[0][1] #LDI Set of A matrices.
    ldi_B = Ms[0][2] #LDI Set of B matrices.
    A_corners = irx.get_sparse_corners(ldi_A)
    B_corners = irx.get_sparse_corners(ldi_B)
    # perform permutation to get true list of A and B from corner matrices.
    A_list, B_list = get_polytope_list_from_corners(A_corners=A_corners, B_corners=B_corners)
    solved_structure = find_lyapunov_and_gain_polytope(A_list=A_list, B_list=B_list, mu=mu, x0=x0.reshape((2, 1)))
    feedback_gain = solved_structure["K"]
    control_effort = feedback_gain @ x0
    return control_effort, feedback_gain

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
x_init = np.array([np.pi, 0.0])  # Example initial state (e.g., [angle, angular velocity])
x_des = np.array([0.0, 0.0])  # Desired state (e.g., upright position, no velocity)
mu = 3.0  # Bound on the norm of control input
dt = 0.01  # Time step for integration
pendulum_length = 0.5  # Length of the pendulum
stop_threshold = 0.01  # Threshold for stopping condition (distance to desired state)

# Initialize the pendulum system
pendulum = SimpleInvertedPendulum()

# Current state
x_current = x_init

# Create a figure for visualization
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Configure trajectory plots
axs[0, 0].set_xlim(left=0, right=T_MAX_PLOTTING, auto=False)
axs[0, 0].set_ylim(-np.pi, np.pi)
axs[0, 0].set_title("Theta (Angle) Trajectory")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Theta (rad)")
line_theta, = axs[0, 0].plot([], [], color="blue", label="Theta")

axs[0, 1].set_xlim(left=0, right=T_MAX_PLOTTING, auto=False)
axs[0, 1].set_ylim(-10, 10)
axs[0, 1].set_title("Theta Dot (Angular Velocity) Trajectory")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Theta Dot (rad/s)")
line_theta_dot, = axs[0, 1].plot([], [], color="green", label="Theta Dot")

# Configure pendulum animation plot
axs[1, 0].set_xlim(-pendulum_length - 0.1, pendulum_length + 0.1)
axs[1, 0].set_ylim(-pendulum_length - 0.1, pendulum_length + 0.1)
axs[1, 0].set_aspect("equal", adjustable="datalim")
axs[1, 0].set_title("Inverted Pendulum Animation")
pendulum_line, = axs[1, 0].plot([], [], color="black", linewidth=2, label="Pendulum")
pendulum_ball, = axs[1, 0].plot([], [], 'o', color="red", markersize=10, label="Ball")

# Configure control input plot (new panel)
axs[1, 1].set_xlim(left=0, right=T_MAX_PLOTTING, auto=False)
axs[1, 1].set_ylim(-mu, mu)
axs[1, 1].set_title("Control Input (u)")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Control Input (u)")
line_u, = axs[1, 1].plot([], [], color="purple", label="u")

# Initialize data storage for trajectories
trajectory_time = []
trajectory_theta = []
trajectory_theta_dot = []
trajectory_u = []

# Update function for the animation
def update(frame):
    global x_current
    # fig.clear()
    # Compute control input using the controller
    u, K = duality_linear_controller_inverted_pendulum(x_current, mu)

    # Simulate the pendulum's dynamics for one time step
    x_next = x_current + pendulum.f(0, x_current, u) * dt

    if any(np.abs(pendulum.f(0, x_current, u) * dt) > X_PERTURBATION):
        print(f"WARNING: immrax perturbation bound exceeded. Actual Delta X: {pendulum.f(0, x_current, u) * dt}s")

    # Update the current state
    x_current = x_next

    # Get the pendulum angle (theta) and angular velocity (theta_dot)
    theta = x_current[0]
    theta_dot = x_current[1]

    # Update trajectory data
    trajectory_time.append(frame * dt)
    trajectory_theta.append(theta)
    trajectory_theta_dot.append(theta_dot)
    trajectory_u.append(u)

    # Update trajectory plots
    line_theta.set_data(trajectory_time, trajectory_theta)
    line_theta_dot.set_data(trajectory_time, trajectory_theta_dot)
    line_u.set_data(trajectory_time, trajectory_u)

    # # Update axis limits for trajectory plots
    # axs[0, 0].set_xlim(0, trajectory_time[-1] + dt)
    # axs[0, 1].set_xlim(0, trajectory_time[-1] + dt)
    # axs[1, 1].set_xlim(0, trajectory_time[-1] + dt)

    # Compute pendulum end coordinates
    x_end = pendulum_length * np.sin(theta)
    y_end = pendulum_length * np.cos(theta)

    # Update pendulum line and ball position
    pendulum_line.set_data([0, x_end], [0, y_end])  # Line from origin to pendulum end
    pendulum_ball.set_data([x_end], [y_end])  # Ball at pendulum end
    # plt.draw()
    # Check stopping condition: distance from desired state
    if np.linalg.norm(x_current - x_des) < stop_threshold:
        ani.event_source.stop()  # Stop the animation
        print(f"GOAL REACHED. ANIMATION TERMINATED. Total time: {frame * dt}")

    return line_theta, line_theta_dot, pendulum_line, pendulum_ball, line_u


# Set animation speed to 10x by adjusting the interval
animation_speed = 1  
interval = dt * 1000 / animation_speed  # Convert dt to milliseconds and scale by speed

# Create the animation with the adjusted interval
ani = FuncAnimation(fig, update, frames=int(T_MAX_PLOTTING/dt), interval=interval, blit=True)

# Show the animation
plt.tight_layout()
plt.show()