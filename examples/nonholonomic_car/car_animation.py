
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
mu = 1  # Bound on the norm of control input
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

# Function to compute a control gain for a given A and B list
def compute_control_gain(A_list, B_list, mu, x_init):
    try:
        # print(A_list)
        return find_lyapunov_and_gain_polytope(
            A_list=A_list, B_list=B_list, mu=mu,
            x0=x_init.reshape((4,1)), solver_verbose=False, print_constraint=False
        )["K"]
    except Exception as e:
        print(f"Control gain computation failed: {e}")  # Print error message
        return None  # Indicate failure

# Recursive function for binary search
def find_gains_recursively(A_ldi_list, B_ldi_list, mu, x_init, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(A_ldi_list)

    # Compute union of LDIs over the given segment
    A_ldi_union = irx.Interval(
        lower=reduce(jnp.minimum, [A.lower for A in A_ldi_list[start_idx:end_idx]]), 
        upper=reduce(jnp.maximum, [A.upper for A in A_ldi_list[start_idx:end_idx]])
    )
    B_ldi_union = irx.Interval(
        lower=reduce(jnp.minimum, [B.lower for B in B_ldi_list[start_idx:end_idx]]), 
        upper=reduce(jnp.maximum, [B.upper for B in B_ldi_list[start_idx:end_idx]])
    )

    # Convert to polytope form
    A_union_corners = irx.get_sparse_corners(A_ldi_union)
    B_union_corners = irx.get_sparse_corners(B_ldi_union)
    A_list, B_list = get_polytope_list_from_corners(A_corners=A_union_corners, B_corners=B_union_corners)

    # Try computing the control gain for this segment
    control_gain = compute_control_gain(A_list, B_list, mu, x_init)

    if control_gain is not None:
        print(f"Synthesis succeeded! {start_idx} to {end_idx}")
        return [(start_idx, end_idx, control_gain)]  # Return the successful gain for this range
    else:
        print(f"Failed to synthesize control gain from {start_idx} to {end_idx}.")
        # print("A_LDI:")
        # print(A_ldi_union)
        pass

    # If no gain found and only one element left, return failure
    if end_idx - start_idx <= 1:
        return []  # No valid gain

    # Bisect the segment and recursively find gains for each half
    mid_idx = (start_idx + end_idx) // 2
    left_gains = find_gains_recursively(A_ldi_list, B_ldi_list, mu, x_init, start_idx, mid_idx)
    right_gains = find_gains_recursively(A_ldi_list, B_ldi_list, mu, x_init, mid_idx, end_idx)

    return left_gains + right_gains  # Combine results

# Load saved data
with open("polytope_data.pkl", "rb") as f:
    data = pickle.load(f)

A_ldi_list = data["A_ldi_list"]
B_ldi_list = data["B_ldi_list"]

# Perform binary search for control gains
gains = find_gains_recursively(A_ldi_list, B_ldi_list, mu, x_init, start_idx=0, end_idx=None)

# Save gains with their corresponding trajectory segments
with open("control_gains.pkl", "wb") as f:
    pickle.dump({"gains": gains}, f)

# Print results
for start_idx, end_idx, gain in gains:
    print(f"Gain found for indices {start_idx} to {end_idx}: \n{gain}")

# print(A_ldi_union
# Load trajectory reference
filename = "/home/user/output_feedback/nhholyap/nonholonomic_car_trajectory.csv"
data = pd.read_csv(filename)
X_PERTURBATION = np.array([.1, .1, 1, 1])
mu = 1

# Extract reference states and control inputs
x_ref = data[["X", "Y", "Phi", "V"]].to_numpy()  # Nx4 array
u_ref = data[["Omega", "A"]].fillna(0).to_numpy()  # Nx2 array
print(x_ref[-1])



# Initialize the nonholonomic car system
car = NonHolonomicCar()

# Current state
x_current = x_init

# Create a figure with 6 subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.flatten()

# Configure trajectory plots (only phi and v remain)
state_labels = ['Steering Angle (phi)', 'Linear Velocity (v)']
state_lines = []
state_data = [[], []]  # Only phi and v now

for i in range(2):
    axs[i].set_xlim(0, 500 * dt)
    axs[i].set_ylim(-5, 10)
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
    axs[i + 2].set_xlim(0, 500 * dt)
    axs[i + 2].set_ylim(-mu, mu)
    axs[i + 2].set_title(control_labels[i])
    axs[i + 2].set_xlabel("Time (s)")
    axs[i + 2].set_ylabel(control_labels[i])
    line, = axs[i + 2].plot([], [], label=control_labels[i])
    control_lines.append(line)

# Configure feedback control plot
axs[4].set_xlim(0, 500 * dt)
axs[4].set_ylim(-mu, mu)
axs[4].set_title("Feedback Control (u_fb)")
axs[4].set_xlabel("Time (s)")
axs[4].set_ylabel("Feedback Control (u_fb)")
feedback_line, = axs[4].plot([], [], label="Feedback Control (u_fb)")
feedback_data = []

# Configure car animation plot (with trajectory history)
axs[5].set_xlim(-2, 7)
axs[5].set_ylim(-2, 7)
axs[5].set_title("Nonholonomic Car Animation")
car_line, = axs[5].plot([], [], 's', color="black", markersize=5, label="Car")
car_traj, = axs[5].plot([], [], '-', color="blue", label="Trajectory")

# Initialize time and trajectories
trajectory_time = []
x_history, y_history = [], []
NUM_FRAMES = 500

def update_car(frame):
    global x_current

    # Compute control input based on closest reference point
    distances = np.linalg.norm(x_ref - x_current, axis=1)
    closest_index = np.argmin(distances)
    
    # Find the corresponding gain K for this segment
    for (start_idx, end_idx, K) in gains:
        if start_idx <= closest_index < end_idx:
            control_gain = K
            break
    else:
        control_gain = np.zeros((2, len(x_current)))  # Default zero gain if not found

    # Compute control input
    u_star = u_ref[closest_index]
    x_star = x_ref[closest_index]
    u_fb = control_gain @ (x_current - x_star)
    u = u_star + u_fb

    omega, a = u

    # Simulate car dynamics
    x_next = x_current + car.f(0, x_current, u, jnp.zeros(1)) * dt
    x_current = x_next

    # Extract state variables
    x, y, phi, v = x_current

    # Update time and state data
    trajectory_time.append(frame * dt)
    state_data[0].append(phi)
    state_data[1].append(v)
    control_data[0].append(omega)
    control_data[1].append(a)
    feedback_data.append(np.linalg.norm(u_fb))

    x_history.append(x)
    y_history.append(y)

    # Update state trajectory plots
    for i, line in enumerate(state_lines):
        line.set_data(trajectory_time, state_data[i])
        axs[i].set_xlim(0, trajectory_time[-1] + dt)

    # Update control effort plots
    for i, line in enumerate(control_lines):
        line.set_data(trajectory_time, control_data[i])
        axs[i + 2].set_xlim(0, trajectory_time[-1] + dt)

    # Update feedback control plot
    feedback_line.set_data(trajectory_time, feedback_data)
    axs[4].set_xlim(0, trajectory_time[-1] + dt)

    # Update car animation (position and trajectory)
    car_line.set_data([x], [y])
    car_traj.set_data(x_history, y_history)

    # Stop condition
    if np.linalg.norm(x_current - x_des) < stop_threshold or frame > NUM_FRAMES:
        ani_car.event_source.stop()

    return state_lines + control_lines + [feedback_line, car_line, car_traj]


# Set animation speed
animation_speed = 10
interval = dt * 1000 / animation_speed

# Create animation
ani_car = FuncAnimation(fig, update_car, frames=NUM_FRAMES, interval=interval, blit=True)

# Show animation
plt.tight_layout()
plt.show()
