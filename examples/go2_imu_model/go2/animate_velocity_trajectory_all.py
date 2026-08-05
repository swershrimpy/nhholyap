import argparse
from pathlib import Path

import numpy as np

REQUIRED_COLUMNS = {
    "sample_ts_ns",
    "sport_heading",
    "sport_vx",
    "sport_vy",
    "rigid_x",
    "rigid_y",
}

COLOR_IMU = "tab:blue"
COLOR_OPTI = "tab:orange"


def _load_csv(csv_path: Path) -> np.ndarray:
    data = np.genfromtxt(
        str(csv_path),
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
    )
    if data.size == 0:
        raise ValueError(f"No data rows found in CSV: {csv_path}")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    names = set(data.dtype.names or ())
    if not REQUIRED_COLUMNS.issubset(names):
        missing = sorted(REQUIRED_COLUMNS - names)
        raise ValueError(f"Missing required columns: {missing}")
    return data


def _keep_strictly_increasing_time(data: np.ndarray) -> np.ndarray:
    t = data["sample_ts_ns"] / 1e9
    order = np.argsort(t)
    data = data[order]
    t = t[order]
    keep = np.r_[True, np.diff(t) > 1e-9]
    return data[keep]


def _world_to_body(vx_world: np.ndarray, vy_world: np.ndarray, heading: np.ndarray):
    c = np.cos(heading)
    s = np.sin(heading)
    vx_body = c * vx_world + s * vy_world
    vy_body = -s * vx_world + c * vy_world
    return vx_body, vy_body


def _integrate_body_velocity(vx: np.ndarray, vy: np.ndarray, heading: np.ndarray, t: np.ndarray):
    if len(t) < 2:
        return np.zeros_like(t), np.zeros_like(t)

    dt = np.diff(t)
    c = np.cos(heading[:-1])
    s = np.sin(heading[:-1])

    dx_world = (c * vx[:-1] - s * vy[:-1]) * dt
    dy_world = (s * vx[:-1] + c * vy[:-1]) * dt

    x = np.r_[0.0, np.cumsum(dx_world)]
    y = np.r_[0.0, np.cumsum(dy_world)]
    return x, y


def _build_series(data: np.ndarray):
    data = _keep_strictly_increasing_time(data)

    t = data["sample_ts_ns"] / 1e9
    t = t - t[0]

    heading = data["sport_heading"]
    imu_vx = data["sport_vx"]
    imu_vy = data["sport_vy"]

    # OptiTrack world-frame position in the project convention.
    opti_x_world = -data["rigid_y"]
    opti_y_world = data["rigid_x"]

    # True world-frame velocity from finite differences, then rotate to robot frame.
    true_vx_world = np.gradient(opti_x_world, t)
    true_vy_world = np.gradient(opti_y_world, t)
    true_vx, true_vy = _world_to_body(true_vx_world, true_vy_world, heading)

    # Trajectory reconstruction from body-frame velocities.
    imu_x, imu_y = _integrate_body_velocity(imu_vx, imu_vy, heading, t)
    true_x, true_y = _integrate_body_velocity(true_vx, true_vy, heading, t)

    return t, imu_vx, imu_vy, true_vx, true_vy, imu_x, imu_y, true_x, true_y


def _render_animation(
    csv_path: Path,
    output_path: Path,
    t: np.ndarray,
    imu_vx: np.ndarray,
    imu_vy: np.ndarray,
    true_vx: np.ndarray,
    true_vy: np.ndarray,
    imu_x: np.ndarray,
    imu_y: np.ndarray,
    true_x: np.ndarray,
    true_y: np.ndarray,
    fps: int,
    sample_step: int,
    slowdown_factor: int,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    fig = plt.figure(figsize=(14, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])

    ax_vx = fig.add_subplot(gs[0, 0])
    ax_vy = fig.add_subplot(gs[1, 0], sharex=ax_vx)
    ax_xy = fig.add_subplot(gs[:, 1])

    fig.suptitle(f"Velocity + Trajectory Animation: {csv_path.name}", fontsize=12)

    # Full velocity curves.
    ax_vx.plot(t, imu_vx, label="vx measured (IMU)", linewidth=1.3, alpha=0.9, color=COLOR_IMU)
    ax_vx.plot(t, true_vx, label="vx true (OptiTrack)", linewidth=1.5, color=COLOR_OPTI)
    ax_vy.plot(t, imu_vy, label="vy measured (IMU)", linewidth=1.3, alpha=0.9, color=COLOR_IMU)
    ax_vy.plot(t, true_vy, label="vy true (OptiTrack)", linewidth=1.5, color=COLOR_OPTI)

    # Moving current-time markers on velocity plots.
    vx_marker_imu, = ax_vx.plot(
        [t[0]], [imu_vx[0]], marker="o", linestyle="None", markersize=5, color=COLOR_IMU
    )
    vx_marker_true, = ax_vx.plot(
        [t[0]], [true_vx[0]], marker="o", linestyle="None", markersize=5, color=COLOR_OPTI
    )
    vy_marker_imu, = ax_vy.plot(
        [t[0]], [imu_vy[0]], marker="o", linestyle="None", markersize=5, color=COLOR_IMU
    )
    vy_marker_true, = ax_vy.plot(
        [t[0]], [true_vy[0]], marker="o", linestyle="None", markersize=5, color=COLOR_OPTI
    )

    vx_cursor = ax_vx.axvline(t[0], color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    vy_cursor = ax_vy.axvline(t[0], color="k", linestyle="--", linewidth=1.0, alpha=0.8)

    vx_label = ax_vx.text(t[0], 0.0, "", fontsize=9, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    vy_label = ax_vy.text(t[0], 0.0, "", fontsize=9, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})

    ax_vx.set_ylabel("vx (m/s)")
    ax_vy.set_xlabel("Time (s)")
    ax_vy.set_ylabel("vy (m/s)")
    ax_vx.grid(True, alpha=0.3)
    ax_vy.grid(True, alpha=0.3)
    ax_vx.legend(loc="upper right")
    ax_vy.legend(loc="upper right")
    fig.text(
        0.012,
        0.988,
        f"Playback: 1/{slowdown_factor}x speed ({slowdown_factor}x slower)",
        fontsize=10,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )

    # Trajectory subplot.
    traj_true_line, = ax_xy.plot(
        [], [], label="true trajectory (OptiTrack)", linewidth=1.8, color=COLOR_OPTI
    )
    traj_imu_line, = ax_xy.plot(
        [], [], label="measured trajectory (IMU odom)", linewidth=1.5, color=COLOR_IMU
    )
    traj_true_dot, = ax_xy.plot(
        [], [], marker="o", linestyle="None", markersize=6, color=COLOR_OPTI
    )
    traj_imu_dot, = ax_xy.plot(
        [], [], marker="o", linestyle="None", markersize=6, color=COLOR_IMU
    )

    ax_xy.scatter([true_x[0]], [true_y[0]], marker="x", s=45, label="start")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("2D Trajectory")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    # Keep a stable square viewport with margin so trajectories are clear and not clipped.
    x_all = np.r_[true_x, imu_x]
    y_all = np.r_[true_y, imu_y]
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 0.5)
    half = 0.5 * span * 1.12
    ax_xy.set_xlim(xmid - half, xmid + half)
    ax_xy.set_ylim(ymid - half, ymid + half)

    vx_ymin, vx_ymax = ax_vx.get_ylim()
    vy_ymin, vy_ymax = ax_vy.get_ylim()
    vx_label_y = vx_ymax - 0.08 * (vx_ymax - vx_ymin)
    vy_label_y = vy_ymax - 0.08 * (vy_ymax - vy_ymin)

    frame_indices = np.arange(0, len(t), max(1, sample_step), dtype=int)
    if frame_indices[-1] != len(t) - 1:
        frame_indices = np.r_[frame_indices, len(t) - 1]
    if slowdown_factor > 1:
        frame_indices = np.repeat(frame_indices, slowdown_factor)

    def _init():
        traj_true_line.set_data([], [])
        traj_imu_line.set_data([], [])
        traj_true_dot.set_data([], [])
        traj_imu_dot.set_data([], [])
        return (
            traj_true_line,
            traj_imu_line,
            traj_true_dot,
            traj_imu_dot,
            vx_marker_imu,
            vx_marker_true,
            vy_marker_imu,
            vy_marker_true,
            vx_cursor,
            vy_cursor,
            vx_label,
            vy_label,
        )

    def _update(frame_k: int):
        i = frame_indices[frame_k]
        ti = t[i]

        traj_true_line.set_data(true_x[: i + 1], true_y[: i + 1])
        traj_imu_line.set_data(imu_x[: i + 1], imu_y[: i + 1])
        traj_true_dot.set_data([true_x[i]], [true_y[i]])
        traj_imu_dot.set_data([imu_x[i]], [imu_y[i]])

        vx_marker_imu.set_data([ti], [imu_vx[i]])
        vx_marker_true.set_data([ti], [true_vx[i]])
        vy_marker_imu.set_data([ti], [imu_vy[i]])
        vy_marker_true.set_data([ti], [true_vy[i]])

        vx_cursor.set_xdata([ti, ti])
        vy_cursor.set_xdata([ti, ti])

        vx_label.set_position((ti, vx_label_y))
        vy_label.set_position((ti, vy_label_y))
        vx_label.set_text(f"t={ti:.2f}s\\nidx={i}")
        vy_label.set_text(f"t={ti:.2f}s\\nidx={i}")

        return (
            traj_true_line,
            traj_imu_line,
            traj_true_dot,
            traj_imu_dot,
            vx_marker_imu,
            vx_marker_true,
            vy_marker_imu,
            vy_marker_true,
            vx_cursor,
            vy_cursor,
            vx_label,
            vy_label,
        )

    anim = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(frame_indices),
        interval=1000.0 / max(1, fps),
        blit=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2200)
    anim.save(str(output_path), writer=writer, dpi=140)
    plt.close(fig)


def process_csv(
    csv_path: Path,
    output_suffix: str,
    fps: int,
    sample_step: int,
    slowdown_factor: int,
):
    try:
        data = _load_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: {exc}")
        return None

    try:
        series = _build_series(data)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: failed to process series ({exc})")
        return None

    t = series[0]
    if len(t) < 3:
        print(f"[skip] {csv_path.name}: not enough valid samples")
        return None

    out_path = csv_path.with_name(f"{csv_path.stem}{output_suffix}")
    _render_animation(
        csv_path,
        out_path,
        *series,
        fps=fps,
        sample_step=sample_step,
        slowdown_factor=max(1, slowdown_factor),
    )
    print(f"[ok] {csv_path.name} -> {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-CSV animation videos with vx/vy time traces and 2D "
            "trajectory comparison (IMU measured vs OptiTrack true)."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_velocity_trajectory.mp4",
        help="Output video filename suffix.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Video frames per second.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=2,
        help="Use every N-th sample as an animation frame (reduces file size/runtime).",
    )
    parser.add_argument(
        "--slowdown-factor",
        type=int,
        default=5,
        help="Playback slowdown multiplier (e.g. 5 means 5x slower than current frame progression).",
    )
    args = parser.parse_args()

    try:
        from matplotlib.animation import writers

        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Matplotlib ffmpeg writer is not available. Install ffmpeg to create MP4 videos."
            )
    except Exception as exc:
        raise RuntimeError(f"Cannot create MP4 animations: {exc}") from exc

    csv_files = sorted(args.dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.dataset_dir}")

    outputs = []
    for csv_path in csv_files:
        out = process_csv(
            csv_path,
            args.output_suffix,
            args.fps,
            args.sample_step,
            args.slowdown_factor,
        )
        if out is not None:
            outputs.append(out)

    print(f"Generated {len(outputs)} video(s) in {args.dataset_dir}")


if __name__ == "__main__":
    main()
