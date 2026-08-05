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


def _load_structured_csv(csv_path: Path) -> np.ndarray:
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

    return data


def _compute_optitrack_velocity(data: np.ndarray):
    t = data["sample_ts_ns"] / 1e9
    t = t - t[0]

    # Match existing GO2 convention: OptiTrack Y-up world projected to X-Z ground plane.
    opti_x = -data["rigid_y"]
    opti_y = data["rigid_x"]

    opti_vx_world = np.gradient(opti_x, t)
    opti_vy_world = np.gradient(opti_y, t)

    # Rotate world-frame velocity into robot frame using heading.
    heading = data["sport_heading"]
    c = np.cos(heading)
    s = np.sin(heading)
    opti_vx_robot = c * opti_vx_world + s * opti_vy_world
    opti_vy_robot = -s * opti_vx_world + c * opti_vy_world
    return t, opti_vx_robot, opti_vy_robot


def _save_plot(csv_path: Path, output_path: Path, t, sport_vx, sport_vy, opti_vx, opti_vy):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, sport_vx, label="vx (measured)", linewidth=1.4)
    axes[0].plot(t, opti_vx, label="vx (optitrack)", linewidth=1.2)
    axes[0].set_ylabel("vx (m/s)")
    axes[0].set_title(f"Velocity Comparison: {csv_path.name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, sport_vy, label="vy (measured)", linewidth=1.4)
    axes[1].plot(t, opti_vy, label="vy (optitrack)", linewidth=1.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("vy (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _has_required_columns(data: np.ndarray) -> bool:
    names = set(data.dtype.names or ())
    return REQUIRED_COLUMNS.issubset(names)


def process_csv(csv_path: Path, output_suffix: str) -> Path | None:
    try:
        data = _load_structured_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: {exc}")
        return None

    if not _has_required_columns(data):
        print(f"[skip] {csv_path.name}: missing required columns {sorted(REQUIRED_COLUMNS)}")
        return None

    t, opti_vx, opti_vy = _compute_optitrack_velocity(data)
    sport_vx = data["sport_vx"]
    sport_vy = data["sport_vy"]

    output_path = csv_path.with_name(f"{csv_path.stem}{output_suffix}")
    _save_plot(csv_path, output_path, t, sport_vx, sport_vy, opti_vx, opti_vy)
    print(f"[ok] {csv_path.name} -> {output_path.name}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create vx/vy measured vs optitrack time-series plots for all CSV files in a dataset folder."
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
        default="_velocity_comparison.png",
        help="Suffix appended to each input CSV stem for output PNG name.",
    )
    args = parser.parse_args()

    csv_files = sorted(args.dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {args.dataset_dir}")

    generated = []
    for csv_path in csv_files:
        out = process_csv(csv_path, args.output_suffix)
        if out is not None:
            generated.append(out)

    print(f"Generated {len(generated)} plot(s) in {args.dataset_dir}")


if __name__ == "__main__":
    main()
