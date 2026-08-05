import argparse
import os

import numpy as np


def _load_columns(csv_path: str):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"No data rows found in CSV: {csv_path}")

    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    t = data["sample_ts_ns"] / 1e9
    t = t - t[0]
    x = data["sport_x"]
    y = data["sport_y"]
    heading = data["sport_heading"]
    vx = data["sport_vx"]
    vy = data["sport_vy"]
    vyaw = data["sport_vyaw"]
    return t, x, y, heading, vx, vy, vyaw


def generate_plots_from_csv(csv_path: str, plots_dir: str):
    import matplotlib.pyplot as plt

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    os.makedirs(plots_dir, exist_ok=True)
    t, x, y, heading, vx, vy, vyaw = _load_columns(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(t, x, label="x")
    plt.plot(t, y, label="y")
    plt.title("Robot Position vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    pos_path = os.path.join(plots_dir, "position_vs_time.png")
    plt.tight_layout()
    plt.savefig(pos_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, vx, label="vx")
    plt.plot(t, vy, label="vy")
    plt.plot(t, vyaw, label="vyaw")
    plt.title("Robot Velocities vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    vel_path = os.path.join(plots_dir, "velocities_vs_time.png")
    plt.tight_layout()
    plt.savefig(vel_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, heading, label="heading")
    plt.title("Robot Heading (Yaw) vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (rad)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    heading_path = os.path.join(plots_dir, "heading_vs_time.png")
    plt.tight_layout()
    plt.savefig(heading_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(x, y, label="trajectory")
    plt.scatter(x[0], y[0], marker="o", label="start")
    plt.scatter(x[-1], y[-1], marker="x", label="end")
    plt.title("Robot 2D Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    traj_path = os.path.join(plots_dir, "trajectory_2d.png")
    plt.tight_layout()
    plt.savefig(traj_path, dpi=150)
    plt.close()

    paths = [pos_path, vel_path, heading_path, traj_path]
    print("Saved plots:")
    for path in paths:
        print(f"  {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Visualize GO2 rigid body logs from CSV.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "data_rigid_bodies.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory where trajectory plots are saved as PNG files.",
    )
    args = parser.parse_args()
    generate_plots_from_csv(args.input_csv, args.plots_dir)


if __name__ == "__main__":
    main()
