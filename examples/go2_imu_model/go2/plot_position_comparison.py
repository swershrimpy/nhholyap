import argparse
import csv
import os

import numpy as np


def compute_position_comparison(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"No data rows in CSV: {csv_path}")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    t = data["sample_ts_ns"] / 1e9
    t = t - t[0]

    sport_x = data["sport_x"]
    sport_y = data["sport_y"]

    # OptiTrack is Y-up. Ground plane is X-Z.
    opti_x = -data["rigid_y"]
    opti_y = data["rigid_x"]

    # Zero OptiTrack trajectory at first sample.
    opti_x = opti_x - opti_x[0]
    opti_y = opti_y - opti_y[0]

    err_x = opti_x - sport_x
    err_y = opti_y - sport_y
    return t, sport_x, sport_y, opti_x, opti_y, err_x, err_y


def save_comparison_csv(
    output_csv: str,
    t: np.ndarray,
    sport_x: np.ndarray,
    sport_y: np.ndarray,
    opti_x: np.ndarray,
    opti_y: np.ndarray,
    err_x: np.ndarray,
    err_y: np.ndarray,
):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t_sec",
                "sport_x_m",
                "sport_y_m",
                "optitrack_x_zeroed_m",
                "optitrack_y_zeroed_m",
                "error_x_m",
                "error_y_m",
            ]
        )
        for i in range(len(t)):
            writer.writerow(
                [
                    float(t[i]),
                    float(sport_x[i]),
                    float(sport_y[i]),
                    float(opti_x[i]),
                    float(opti_y[i]),
                    float(err_x[i]),
                    float(err_y[i]),
                ]
            )
    print(f"Saved position comparison CSV: {output_csv}")


def save_position_plots(
    output_time_plot: str,
    output_traj_plot: str,
    t: np.ndarray,
    sport_x: np.ndarray,
    sport_y: np.ndarray,
    opti_x: np.ndarray,
    opti_y: np.ndarray,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, sport_x, label="sport_x", linewidth=1.4)
    axes[0].plot(t, opti_x, label="optitrack_x_zeroed", linewidth=1.2)
    axes[0].set_ylabel("x (m)")
    axes[0].set_title("Position Comparison vs Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, sport_y, label="sport_y", linewidth=1.4)
    axes[1].plot(t, opti_y, label="optitrack_y_zeroed", linewidth=1.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("y (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    os.makedirs(os.path.dirname(output_time_plot) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_time_plot, dpi=150)
    plt.close(fig)
    print(f"Saved time-series plot: {output_time_plot}")

    plt.figure(figsize=(7, 7))
    plt.plot(sport_x, sport_y, label="robot (sport)", linewidth=1.4)
    plt.plot(opti_x, opti_y, label="optitrack zeroed (Y-up -> XZ)", linewidth=1.2)
    plt.scatter(sport_x[0], sport_y[0], marker="o", s=35, label="sport start")
    plt.scatter(opti_x[0], opti_y[0], marker="x", s=35, label="optitrack start")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("2D Trajectory Comparison")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(output_traj_plot) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_traj_plot, dpi=150)
    plt.close()
    print(f"Saved 2D trajectory plot: {output_traj_plot}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare OptiTrack position (Y-up ground plane) against sport_x/sport_y."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "data_rigid_bodies.csv"),
        help="Input CSV path from log_data_rigid_bodies.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "position_comparison.csv"),
        help="Output CSV with sport and OptiTrack position comparison.",
    )
    parser.add_argument(
        "--output-time-plot",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "position_comparison_vs_time.png"),
        help="Output PNG path for x/y vs time comparison.",
    )
    parser.add_argument(
        "--output-traj-plot",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "position_comparison_2d.png"),
        help="Output PNG path for 2D trajectory comparison.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only compute/save CSV and skip plot generation.",
    )
    args = parser.parse_args()

    t, sport_x, sport_y, opti_x, opti_y, err_x, err_y = compute_position_comparison(
        args.input_csv
    )
    save_comparison_csv(args.output_csv, t, sport_x, sport_y, opti_x, opti_y, err_x, err_y)

    if not args.no_plot:
        save_position_plots(
            args.output_time_plot,
            args.output_traj_plot,
            t,
            sport_x,
            sport_y,
            opti_x,
            opti_y,
        )


if __name__ == "__main__":
    main()
