import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import csv

# Dynamics of the nonholonomic car
def car_dynamics(t, state, u):
    x, y, phi, v = state
    omega, a = u
    dx = v * np.cos(phi)
    dy = v * np.sin(phi)
    dphi = omega
    dv = a
    return [dx, dy, dphi, dv]

# Cost function for optimization
def cost_function(u, stateA, stateB, t_span, dt, car_length):
    u = u.reshape(-1, 2)
    state = np.array(stateA)
    num_steps = len(u)
    cost = 0

    for i in range(num_steps):
        t_eval = [0, dt]
        result = solve_ivp(car_dynamics, t_eval, state, args=(u[i],), t_eval=[dt])
        state = result.y[:, -1]
        cost += np.linalg.norm(state - stateB)**2

    return cost

# Generate optimal trajectory with time tracking
def generate_trajectory_with_time(stateA, stateB, t_span, dt, car_length):
    num_steps = int(t_span / dt)

    # Initial guess for controls [omega, a]
    u_init = np.zeros((num_steps, 2)).flatten()  # Start with zero inputs

    # Optimization bounds
    omega_max = 1.0  # Maximum angular velocity
    a_max = 1.0      # Maximum linear acceleration
    bounds = [(-omega_max, omega_max), (-a_max, a_max)] * num_steps

    # Optimize controls
    result = minimize(
        cost_function,
        u_init,
        args=(stateA, stateB, t_span, dt, car_length),
        bounds=bounds,
        method='L-BFGS-B',
        options={'disp': True}
    )

    # Extract optimized controls
    u_opt = result.x.reshape(-1, 2)

    # Simulate the car dynamics with optimized controls
    state = np.array(stateA)
    trajectory = [state]
    control_inputs = []  # Store control inputs (u)
    time_states = [0.0]  # Start at t = 0 for states
    time_controls = []   # Will store t for control inputs

    for i in range(len(u_opt)):
        # Integrate dynamics for one step
        t_eval = [0, dt]
        result = solve_ivp(car_dynamics, t_eval, state, args=(u_opt[i],), t_eval=[dt])
        state = result.y[:, -1]  # Update state to the last time step
        trajectory.append(state)

        # Save the control input and corresponding time
        control_inputs.append(u_opt[i])
        time_controls.append(time_states[-1])  # Controls are applied at the beginning of each step

        # Save the next state's time
        time_states.append(time_states[-1] + dt)

    trajectory = np.array(trajectory)
    control_inputs = np.array(control_inputs)
    return trajectory, control_inputs, time_states, time_controls

# Save trajectory and controls to a CSV file
def save_trajectory_to_csv(filename, trajectory, u_opt, time_states, time_controls):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time_State", "X", "Y", "Phi", "V", "Time_Control", "Omega", "A"])
        
        for i in range(len(time_states)):
            time_state = time_states[i]
            x, y, phi, v = trajectory[i]
            
            if i < len(time_controls):
                time_control = time_controls[i]
                omega, a = u_opt[i]
            else:
                time_control = ""
                omega, a = "", ""
            
            writer.writerow([time_state, x, y, phi, v, time_control, omega, a])

# Visualize the trajectory
def plot_trajectory(trajectory, time_states, u_opt, time_controls):
    x_traj = trajectory[:, 0]
    y_traj = trajectory[:, 1]
    phi_traj = trajectory[:, 2]
    v_traj = trajectory[:, 3]

    omega_traj = u_opt[:, 0]
    a_traj = u_opt[:, 1]

    # Plot states over time
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_states, x_traj, label="x (Position)")
    plt.plot(time_states, y_traj, label="y (Position)")
    plt.title("Positions vs. Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time_states, phi_traj, label="phi (Orientation)")
    plt.plot(time_states, v_traj, label="v (Velocity)")
    plt.title("Orientation and Velocity vs. Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad) / Velocity (m/s)")
    plt.legend()
    plt.grid()

    # Plot controls over time
    plt.subplot(3, 1, 3)
    plt.plot(time_controls, omega_traj, label="omega (Angular Velocity)")
    plt.plot(time_controls, a_traj, label="a (Linear Acceleration)")
    plt.title("Controls vs. Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    stateA = [0.0, 0.0, 0.0, 1.0]  # Initial state: [x, y, phi, v]
    stateB = [5.0, 0.0, np.pi / 4, 1.0]  # Target state: [x, y, phi, v]
    t_span = 10.0  # Total time
    dt = 0.1  # Time step
    car_length = 1.0  # Length of the car

    # Generate trajectory
    trajectory, u_opt, time_states, time_controls = generate_trajectory_with_time(stateA, stateB, t_span, dt, car_length)

    # Save to CSV
    filename = "nonholonomic_car_trajectory.csv"
    save_trajectory_to_csv(filename, trajectory, u_opt, time_states, time_controls)
    print(f"Trajectory saved to {filename}")

    # Plot trajectory and controls
    plot_trajectory(trajectory, time_states, u_opt, time_controls)
