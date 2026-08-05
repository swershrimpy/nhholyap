import argparse
import csv
import os

import numpy as np


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _quat_to_rotmat_xyzw(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray):
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    norm = np.where(norm == 0.0, 1.0, norm)
    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    return r00, r01, r02, r10, r11, r12, r20, r21, r22


def _optitrack_yaw_y_up_from_quat(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray, forward_axis: str
) -> np.ndarray:
    # OptiTrack world is Y-up, so yaw is rotation about +Y in the XZ ground plane.
    # r00, _, r02, r10, _, r12, r20, _, r22 = _quat_to_rotmat_xyzw(qx, qy, qz, qw)

    # if forward_axis == "x":
    #     fx = r00
    #     fz = r20
    # elif forward_axis == "z":
    #     fx = r02
    #     fz = r22
    # else:
    #     raise ValueError(f"Unsupported forward axis: {forward_axis}")

    # return np.arctan2(fz, fx)
    roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
    pitch = np.arcsin(2 * (qw * qy - qz * qx))
    yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))
    return yaw


def plot_optitrack_vs_imu_yaw(csv_path: str, output_path: str, forward_axis: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"No data rows in CSV: {csv_path}")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    t = data["sample_ts_ns"] / 1e9
    t = t - t[0]

    imu_yaw = _wrap_pi(data["sport_heading"])
    opti_yaw = _wrap_pi(
        _optitrack_yaw_y_up_from_quat(
            data["rigid_qx"], data["rigid_qy"], data["rigid_qz"], data["rigid_qw"], forward_axis
        )
    )

    # Align initial phase so both curves start from the same heading reference.
    offset = imu_yaw[0] - opti_yaw[0]
    opti_yaw_aligned = _wrap_pi(opti_yaw + offset)
    yaw_error = _wrap_pi(opti_yaw_aligned - imu_yaw)
    return t, imu_yaw, opti_yaw_aligned, yaw_error


def save_yaw_comparison_csv(
    output_csv: str, t: np.ndarray, imu_yaw: np.ndarray, opti_yaw: np.ndarray, yaw_error: np.ndarray
):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "imu_yaw_rad", "optitrack_yaw_rad", "yaw_error_rad"])
        for i in range(len(t)):
            writer.writerow([float(t[i]), float(imu_yaw[i]), float(opti_yaw[i]), float(yaw_error[i])])
    print(f"Saved yaw comparison CSV: {output_csv}")


def save_yaw_plot(output_path: str, t: np.ndarray, imu_yaw: np.ndarray, opti_yaw: np.ndarray, yaw_error: np.ndarray):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, imu_yaw, label="IMU sport_heading", linewidth=1.5)
    axes[0].plot(t, opti_yaw, label="OptiTrack yaw (Y-up)", linewidth=1.2)
    axes[0].set_ylabel("Yaw (rad)")
    axes[0].set_title("Yaw Comparison: OptiTrack Ground Truth vs IMU")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        t, yaw_error, label="Yaw error (OptiTrack - IMU)", color="tab:red", linewidth=1.2
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved yaw comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute OptiTrack yaw (Y-up) from quaternion and compare to IMU sport_heading."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "data_rigid_bodies.csv"),
        help="Input CSV path from log_data_rigid_bodies.py.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "yaw_comparison.png"),
        help="Output PNG path for yaw comparison figure.",
    )
    parser.add_argument(
        "--forward-axis",
        type=str,
        choices=["x", "z"],
        default="x",
        help="Rigid body forward axis in local frame for yaw extraction.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "yaw_comparison.csv"),
        help="Output CSV with computed IMU yaw, OptiTrack yaw, and yaw error.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only compute and save yaw comparison CSV (skip PNG plot generation).",
    )
    args = parser.parse_args()
    t, imu_yaw, opti_yaw, yaw_error = plot_optitrack_vs_imu_yaw(
        args.input_csv, args.output_plot, args.forward_axis
    )
    save_yaw_comparison_csv(args.output_csv, t, imu_yaw, opti_yaw, yaw_error)
    if not args.no_plot:
        save_yaw_plot(args.output_plot, t, imu_yaw, opti_yaw, yaw_error)


if __name__ == "__main__":
    main()
