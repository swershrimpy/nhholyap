import argparse
from pathlib import Path

import numpy as np

REQUIRED_COLUMNS = {
    "sample_ts_ns",
    "sport_heading",
    "sport_vy",
    "sport_vyaw",
    "rigid_x",
    "rigid_y",
    "rigid_qx",
    "rigid_qy",
    "rigid_qz",
    "rigid_qw",
}


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


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _world_to_body(vx_world: np.ndarray, vy_world: np.ndarray, heading: np.ndarray):
    c = np.cos(heading)
    s = np.sin(heading)
    vx_body = c * vx_world + s * vy_world
    vy_body = -s * vx_world + c * vy_world
    return vx_body, vy_body


def _optitrack_yaw_from_quat(qx, qy, qz, qw) -> np.ndarray:
    # Same yaw extraction form used in existing yaw comparison script.
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))


def _prepare_sorted_unique(data: np.ndarray) -> np.ndarray:
    t = data["sample_ts_ns"] / 1e9
    order = np.argsort(t)
    data = data[order]
    t = t[order]
    keep = np.r_[True, np.diff(t) > 1e-9]
    return data[keep]


def _compute_series(data: np.ndarray):
    data = _prepare_sorted_unique(data)

    t = data["sample_ts_ns"] / 1e9
    t = t - t[0]
    if len(t) < 2:
        raise ValueError("Not enough valid samples")

    heading = data["sport_heading"]
    vy_measured = data["sport_vy"]
    vyaw_measured = data["sport_vyaw"]

    # OptiTrack planar velocity in world frame -> robot frame.
    opti_x_world = -data["rigid_y"]
    opti_y_world = data["rigid_x"]
    vx_world = np.gradient(opti_x_world, t)
    vy_world = np.gradient(opti_y_world, t)
    _, vy_optitrack = _world_to_body(vx_world, vy_world, heading)

    # OptiTrack yaw-rate from quaternion yaw derivative.
    yaw = _optitrack_yaw_from_quat(
        data["rigid_qx"],
        data["rigid_qy"],
        data["rigid_qz"],
        data["rigid_qw"],
    )
    yaw = np.unwrap(_wrap_pi(yaw))
    vyaw_optitrack = np.gradient(yaw, t)

    return t, vy_measured, vy_optitrack, vyaw_measured, vyaw_optitrack


def _save_plot(
    csv_path: Path,
    out_path: Path,
    t: np.ndarray,
    vy_measured: np.ndarray,
    vy_optitrack: np.ndarray,
    vyaw_measured: np.ndarray,
    vyaw_optitrack: np.ndarray,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, vy_measured, label="vy measured", color="tab:blue", linewidth=1.35)
    axes[0].plot(t, vy_optitrack, label="vy optitrack", color="tab:orange", linewidth=1.35)
    axes[0].set_ylabel("vy (m/s)")
    axes[0].set_title(f"vy and vyaw Comparison vs Time: {csv_path.name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, vyaw_measured, label="vyaw measured", color="tab:blue", linewidth=1.35)
    axes[1].plot(t, vyaw_optitrack, label="vyaw optitrack", color="tab:orange", linewidth=1.35)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("vyaw (rad/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_csv(csv_path: Path, output_suffix: str):
    try:
        data = _load_csv(csv_path)
        series = _compute_series(data)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: {exc}")
        return None

    out_path = csv_path.with_name(f"{csv_path.stem}{output_suffix}")
    _save_plot(csv_path, out_path, *series)
    print(f"[ok] {csv_path.name} -> {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create one plot per CSV with vy measured vs optitrack and vyaw measured "
            "vs optitrack on a shared timescale."
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
        default="_vy_vyaw_comparison.png",
        help="Output filename suffix per input CSV.",
    )
    args = parser.parse_args()

    csv_files = sorted(args.dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.dataset_dir}")

    n_ok = 0
    for csv_path in csv_files:
        out = process_csv(csv_path, args.output_suffix)
        if out is not None:
            n_ok += 1

    print(f"Generated {n_ok} plot(s) in {args.dataset_dir}")


if __name__ == "__main__":
    main()
