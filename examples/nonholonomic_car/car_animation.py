
import pickle
from typing import Tuple
import numpy as np
import jax.numpy as jnp
import sys
import os
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
# Add the parent directory of 'nhholyap' to the Python path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from duality_clf import find_lyapunov_and_gain_polytope
import copy
import pandas as pd
from nhholyap.examples.nonholonomic_car.nonholonomic_car import NonHolonomicCar, get_polytope_list_from_corners
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import immrax as irx
from functools import reduce

# Parameters
x_init = np.array([0, 0, 0, 1])  # Initial state: [x, y, phi, v]
x_des = np.array([5, 0.0, np.pi/4, 1.0])  # Desired state: [x_des, y_des, phi_des, v_des]
mu = 1e8  # Bound on the norm of control input
dt = 0.02  # Time step for integration
stop_threshold = 0.1  # Threshold for stopping condition (distance to desired state)
car_length = 1.0  # Length of the car

# Load A_list and B_list
with open("polytope_data.pkl", "rb") as f:
    data = pickle.load(f)
    A_list = data["A_list"]
    B_list = data["B_list"]
    A_ldi_list = data["A_ldi_list"]
    B_ldi_list = data["B_ldi_list"]

A_ldi_union = irx.Interval(lower=reduce(jnp.minimum, [A.lower for A in A_ldi_list[41:50]]), upper=reduce(jnp.maximum, [A.upper for A in A_ldi_list[41:50]]))
B_ldi_union = irx.Interval(lower=reduce(jnp.minimum, [B.lower for B in B_ldi_list[41:50]]), upper=reduce(jnp.maximum, [B.upper for B in B_ldi_list[41:50]]))


A_union_corners = irx.get_sparse_corners(A_ldi_union)  # List of (a, b) np arrays
B_union_corners = irx.get_sparse_corners(B_ldi_union)  # List of (a, b) np arrays

A_list, B_list = get_polytope_list_from_corners(A_corners=A_union_corners, B_corners=B_union_corners)

# print(A_ldi_union)

filename = "../nonholonomic_car_trajectory.csv"
data = pd.read_csv(filename)
X_PERTURBATION = np.array([.1, .1, 1, 1])
mu = 1
# Extract states (X, Y, Phi, V) and control inputs (Omega, A)
x_ref = data[["X", "Y", "Phi", "V"]].to_numpy()  # Nx4 array
u_ref = data[["Omega", "A"]].fillna(0).to_numpy()  # Nx2 array, fill NaN with 0 for control inputs
print(x_ref[-1])
control_gain = find_lyapunov_and_gain_polytope(A_list=A_list, B_list=B_list, mu=mu, x0=x_init.reshape((4,1)), solver_verbose=False, print_constraint=False)["K"]
# print(control_gain)
# Update function for the animation



# Initialize the nonholonomic car system
car = NonHolonomicCar()

# Current state
x_current = x_init

# Create a figure with 7 subplots
fig, axs = plt.subplots(4, 2, figsize=(12, 12))
axs = axs.flatten()


# Configure trajectory plots
state_labels = ['X Position', 'Y Position', 'Steering Angle (phi)', 'Linear Velocity (v)']
state_lines = []
state_data = [[], [], [], []]

for i in range(4):
    axs[i].set_xlim(0, 500 * dt)
    axs[i].set_ylim(-5, 10)  # Adjust limits as needed
    axs[i].set_title(state_labels[i])
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel(state_labels[i])
    line, = axs[i].plot([], [], label=state_labels[i])
    state_lines.append(line)

# Configure control effort plots
control_labels = ['Angular Velocity (omega)', 'Linear Acceleration (a)']
control_lines = []
control_data = [[], []]

for i in range(2):
    axs[i + 4].set_xlim(0, 500 * dt)
    axs[i + 4].set_ylim(-mu, mu)
    axs[i + 4].set_title(control_labels[i])
    axs[i + 4].set_xlabel("Time (s)")
    axs[i + 4].set_ylabel(control_labels[i])
    line, = axs[i + 4].plot([], [], label=control_labels[i])
    control_lines.append(line)

# Configure car animation plot
axs[6].set_xlim(-2, 7)
axs[6].set_ylim(-2, 7)
axs[6].set_title("Nonholonomic Car Animation")
car_line, = axs[6].plot([], [],'s', color="black", markersize=5, label="Car")
# car_front, = axs[6].plot([], [], 'o', color="red", markersize=5, label="Front")

# Initialize time and trajectories
trajectory_time = []
NUM_FRAMES = 500

def update_car(frame):
    global x_current

    # Compute control input using the controller
    distances = np.linalg.norm(x_ref - x_current, axis=1)  # Compute distances to all reference states
    closest_index = np.argmin(distances)  # Get the index of the closest reference state

    u_star = u_ref[closest_index]  # Use corresponding reference control
    x_star = x_ref[closest_index]
    u_fb = control_gain @ (x_current - x_star)
    u = u_fb + u_star
    # print(f"Feedback control: {u_fb}  reference: {u_star}")

    omega, a = u  # Control input components: [angular velocity, linear acceleration]

    # Simulate the car's dynamics for one time step
    x_next = x_current + car.f(0, x_current, u, jnp.zeros(1)) * dt

    # Update the current state
    x_current = x_next

    # Extract state variables
    x, y, phi, v = x_current

    # Update time and state data
    trajectory_time.append(frame * dt)
    state_data[0].append(x)    # X Position
    state_data[1].append(y)    # Y Position
    state_data[2].append(phi)  # Steering angle
    state_data[3].append(v)    # Linear velocity
    control_data[0].append(omega)  # Angular velocity
    control_data[1].append(a)      # Linear acceleration

    # Update state trajectory plots
    for i, line in enumerate(state_lines):
        line.set_data(trajectory_time, state_data[i])
        axs[i].set_xlim(0, trajectory_time[-1] + dt)

    # Update control effort plots
    for i, line in enumerate(control_lines):
        line.set_data(trajectory_time, control_data[i])
        axs[i + 4].set_xlim(0, trajectory_time[-1] + dt)


    # Update car line and front marker
    car_line.set_data([x], [y])  # Line from car center to front

    # Check stopping condition: distance from desired state
    if np.linalg.norm(x_current - x_des) < stop_threshold or frame > NUM_FRAMES:
        ani_car.event_source.stop()  # Stop the animation

    return state_lines + control_lines + [car_line]

# Set animation speed to 10x by adjusting the interval
animation_speed = 10  # 10x faster animation
interval = dt * 1000 / animation_speed  # Convert dt to milliseconds and scale by speed

# Create the animation with the adjusted interval
ani_car = FuncAnimation(fig, update_car, frames=NUM_FRAMES, interval=interval, blit=True)

# Show the animation
plt.tight_layout()
plt.show()