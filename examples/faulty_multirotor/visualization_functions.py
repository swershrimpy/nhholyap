import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches


def draw_quadrotor(ax, px, py, theta, arm_length=0.5, color='black', alpha=1.0):
    """Draw a simple planar quadrotor at position (px, py) with angle theta."""
    # Rotor positions in body frame
    left_rotor_body = np.array([-arm_length, 0.0])
    right_rotor_body = np.array([arm_length, 0.0])

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    # Transform to world frame
    center = np.array([px, py])
    left_rotor = center + R @ left_rotor_body
    right_rotor = center + R @ right_rotor_body

    # Draw arm
    arm_line, = ax.plot(
        [left_rotor[0], right_rotor[0]],
        [left_rotor[1], right_rotor[1]],
        color=color, linewidth=2.5, alpha=alpha,
    )

    # Draw rotors as circles
    rotor_radius = 0.12
    left_circle = plt.Circle(left_rotor, rotor_radius, color=color, fill=True, alpha=alpha)
    right_circle = plt.Circle(right_rotor, rotor_radius, color=color, fill=True, alpha=alpha)
    ax.add_patch(left_circle)
    ax.add_patch(right_circle)

    # Draw center body
    body_circle = plt.Circle(center, 0.08, color=color, fill=True, alpha=alpha)
    ax.add_patch(body_circle)

    return arm_line, left_circle, right_circle, body_circle


def simulate_multirotor(sys, x0, u_func, w_func, p, dt, num_steps):
    """
    Simulate the faulty planar multirotor using forward Euler.

    Args:
        sys: FaultyPlanarMultirotor instance
        x0: initial state (5,)
        u_func: callable(t, x) -> u (2,)
        w_func: callable(t, x) -> w (1,)
        p: parameter array (2,)
        dt: time step
        num_steps: number of steps

    Returns:
        times: (num_steps+1,)
        states: (num_steps+1, 5)
        controls: (num_steps, 2)
    """
    states = [np.array(x0, dtype=float)]
    controls = []
    times = [0.0]
    x = np.array(x0, dtype=float)

    for i in range(num_steps):
        t = i * dt
        u = np.array(u_func(t, x), dtype=float)
        w = np.array(w_func(t, x), dtype=float)
        x_dot = sys.f_np(t, x, u, w, p)
        x = x + dt * x_dot
        states.append(x.copy())
        controls.append(u.copy())
        times.append(t + dt)

    return np.array(times), np.array(states), np.array(controls)


def create_multirotor_animation(
    trajectories,
    labels,
    colors,
    dt,
    arm_length=0.5,
    filename="multirotor_fault_animation.gif",
    title="Planar Multirotor Fault Simulation",
    speed_factor=2,
):
    """
    Create an animation showing multiple multirotor trajectories.

    Args:
        trajectories: list of (times, states) tuples
        labels: list of labels for each trajectory
        colors: list of colors for each trajectory
        dt: simulation time step
        arm_length: quadrotor arm length for drawing
        filename: output filename (gif)
        title: plot title
        speed_factor: animation speed multiplier
    """
    num_frames = min(len(t[1]) for t in trajectories)

    # Compute axis limits from all trajectories
    all_px = np.concatenate([s[:, 0] for _, s in trajectories])
    all_py = np.concatenate([s[:, 1] for _, s in trajectories])
    margin = 2.0
    xlim = (all_px.min() - margin, all_px.max() + margin)
    ylim = (all_py.min() - margin, all_py.max() + margin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_anim = axes[0]
    ax_states = axes[1]

    ax_anim.set_xlim(xlim)
    ax_anim.set_ylim(ylim)
    ax_anim.set_aspect('equal')
    ax_anim.set_xlabel('X Position [m]')
    ax_anim.set_ylabel('Y Position [m]')
    ax_anim.set_title(title)
    ax_anim.grid(True, alpha=0.3)

    # Trail lines
    trail_lines = []
    for label, color in zip(labels, colors):
        line, = ax_anim.plot([], [], '-', color=color, alpha=0.4, linewidth=1.5, label=label)
        trail_lines.append(line)

    ax_anim.legend(loc='upper right', fontsize=9)

    # State subplot: theta over time
    ax_states.set_xlabel('Time [s]')
    ax_states.set_ylabel('Pitch Angle theta [rad]')
    ax_states.set_title('Pitch Angle Over Time')
    ax_states.grid(True, alpha=0.3)
    theta_lines = []
    for label, color in zip(labels, colors):
        line, = ax_states.plot([], [], '-', color=color, linewidth=1.5, label=label)
        theta_lines.append(line)
    ax_states.legend(loc='upper right', fontsize=9)

    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=11)

    def init():
        for line in trail_lines + theta_lines:
            line.set_data([], [])
        time_text.set_text('')
        return trail_lines + theta_lines + [time_text]

    def update(frame):
        # Clear quadrotor patches from previous frame
        for patch in list(ax_anim.patches):
            patch.remove()
        # Remove arm lines from previous frame (keep trail lines)
        lines_to_remove = [l for l in ax_anim.lines if l not in trail_lines]
        for l in lines_to_remove:
            l.remove()

        for i, ((times, states), color) in enumerate(zip(trajectories, colors)):
            px = states[frame, 0]
            py = states[frame, 1]
            theta = states[frame, 4]

            draw_quadrotor(ax_anim, px, py, theta, arm_length=arm_length, color=color)

            # Update trail
            trail_lines[i].set_data(states[:frame + 1, 0], states[:frame + 1, 1])

            # Update theta plot
            theta_lines[i].set_data(times[:frame + 1], states[:frame + 1, 4])

        t = trajectories[0][0][frame]
        time_text.set_text(f't = {t:.2f} s')

        # Auto-scale theta plot
        if frame > 0:
            all_theta = np.concatenate([s[:frame + 1, 4] for _, s in trajectories])
            ax_states.set_xlim(0, trajectories[0][0][frame] + 0.1)
            theta_margin = 0.1
            ax_states.set_ylim(all_theta.min() - theta_margin, all_theta.max() + theta_margin)

        return trail_lines + theta_lines + [time_text]

    interval_ms = max(1, int(dt * 1000 / speed_factor))
    ani = FuncAnimation(
        fig, update, frames=num_frames, init_func=init,
        interval=interval_ms, blit=False,
    )

    if filename:
        ani.save(filename, writer='pillow', fps=max(1, int(speed_factor / dt)))
        print(f"Animation saved to {filename}")

    plt.close(fig)
    return ani


def create_sensor_fault_animation(
    traj_nominal,
    traj_sensed,
    dt,
    arm_length=0.5,
    filename="sensor_fault_animation.gif",
    title="IMU Drift Sensor Fault",
    speed_factor=2,
):
    """
    Animate a quadrotor with sensor fault showing true vs. perceived state.

    Args:
        traj_nominal: (times, states) - true trajectory
        traj_sensed: (times, states_perceived) - trajectory as perceived with IMU drift
        dt: time step
        arm_length: drawing arm length
        filename: output file
        title: plot title
        speed_factor: speed multiplier
    """
    times_n, states_n = traj_nominal
    times_s, states_s = traj_sensed
    num_frames = min(len(states_n), len(states_s))

    all_px = np.concatenate([states_n[:, 0], states_s[:, 0]])
    all_py = np.concatenate([states_n[:, 1], states_s[:, 1]])
    margin = 2.0
    xlim = (all_px.min() - margin, all_px.max() + margin)
    ylim = (all_py.min() - margin, all_py.max() + margin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_anim = axes[0]
    ax_drift = axes[1]

    ax_anim.set_xlim(xlim)
    ax_anim.set_ylim(ylim)
    ax_anim.set_aspect('equal')
    ax_anim.set_xlabel('X Position [m]')
    ax_anim.set_ylabel('Y Position [m]')
    ax_anim.set_title(title)
    ax_anim.grid(True, alpha=0.3)

    trail_true, = ax_anim.plot([], [], '-', color='green', alpha=0.5, linewidth=1.5, label='True')
    trail_sensed, = ax_anim.plot([], [], '--', color='red', alpha=0.5, linewidth=1.5, label='Perceived (IMU drift)')
    ax_anim.legend(loc='upper right', fontsize=9)

    ax_drift.set_xlabel('Time [s]')
    ax_drift.set_ylabel('Theta [rad]')
    ax_drift.set_title('True vs. Perceived Pitch Angle')
    ax_drift.grid(True, alpha=0.3)
    theta_true_line, = ax_drift.plot([], [], '-', color='green', linewidth=1.5, label='True theta')
    theta_sensed_line, = ax_drift.plot([], [], '--', color='red', linewidth=1.5, label='Perceived theta')
    ax_drift.legend(loc='upper right', fontsize=9)

    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=11)

    def update(frame):
        for patch in list(ax_anim.patches):
            patch.remove()
        lines_to_remove = [l for l in ax_anim.lines if l not in [trail_true, trail_sensed]]
        for l in lines_to_remove:
            l.remove()

        # True state
        draw_quadrotor(ax_anim, states_n[frame, 0], states_n[frame, 1],
                       states_n[frame, 4], arm_length=arm_length, color='green', alpha=0.8)
        trail_true.set_data(states_n[:frame + 1, 0], states_n[:frame + 1, 1])

        # Perceived state (with IMU drift)
        draw_quadrotor(ax_anim, states_s[frame, 0], states_s[frame, 1],
                       states_s[frame, 4], arm_length=arm_length, color='red', alpha=0.5)
        trail_sensed.set_data(states_s[:frame + 1, 0], states_s[:frame + 1, 1])

        # Theta comparison
        theta_true_line.set_data(times_n[:frame + 1], states_n[:frame + 1, 4])
        theta_sensed_line.set_data(times_s[:frame + 1], states_s[:frame + 1, 4])

        t = times_n[frame]
        time_text.set_text(f't = {t:.2f} s')

        if frame > 0:
            ax_drift.set_xlim(0, times_n[frame] + 0.1)
            all_theta = np.concatenate([states_n[:frame + 1, 4], states_s[:frame + 1, 4]])
            m = 0.1
            ax_drift.set_ylim(all_theta.min() - m, all_theta.max() + m)

        return [trail_true, trail_sensed, theta_true_line, theta_sensed_line, time_text]

    interval_ms = max(1, int(dt * 1000 / speed_factor))
    ani = FuncAnimation(
        fig, update, frames=num_frames,
        interval=interval_ms, blit=False,
    )

    if filename:
        ani.save(filename, writer='pillow', fps=max(1, int(speed_factor / dt)))
        print(f"Animation saved to {filename}")

    plt.close(fig)
    return ani
