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


def _world_to_body(vx_world: np.ndarray, vy_world: np.ndarray, heading: np.ndarray):
    c = np.cos(heading)
    s = np.sin(heading)
    vx_body = c * vx_world + s * vy_world
    vy_body = -s * vx_world + c * vy_world
    return vx_body, vy_body


def _compute_errors(data: np.ndarray):
    t = data["sample_ts_ns"] / 1e9
    order = np.argsort(t)
    t = t[order]
    data = data[order]

    keep = np.r_[True, np.diff(t) > 1e-9]
    t = t[keep]
    data = data[keep]
    t = t - t[0]

    heading = data["sport_heading"]
    imu_vx = data["sport_vx"]
    imu_vy = data["sport_vy"]

    opti_x_world = -data["rigid_y"]
    opti_y_world = data["rigid_x"]
    true_vx_world = np.gradient(opti_x_world, t)
    true_vy_world = np.gradient(opti_y_world, t)
    true_vx, true_vy = _world_to_body(true_vx_world, true_vy_world, heading)

    err_vx = true_vx - imu_vx
    err_vy = true_vy - imu_vy
    return t, err_vx, err_vy


def _save_error_plot(
    output_path: Path,
    t: np.ndarray,
    err: np.ndarray,
    ylabel: str,
    title: str,
    color: str,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, err, linewidth=1.3, color=color)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def process_csv(csv_path: Path, vx_suffix: str, vy_suffix: str):
    try:
        data = _load_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: {exc}")
        return None, None

    try:
        t, err_vx, err_vy = _compute_errors(data)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: failed to compute errors ({exc})")
        return None, None

    if len(t) < 2:
        print(f"[skip] {csv_path.name}: not enough samples")
        return None, None

    out_vx = csv_path.with_name(f"{csv_path.stem}{vx_suffix}")
    out_vy = csv_path.with_name(f"{csv_path.stem}{vy_suffix}")

    _save_error_plot(
        out_vx,
        t,
        err_vx,
        ylabel="vx error (m/s)",
        title=f"vx Error vs Time: {csv_path.name}",
        color="tab:blue",
    )
    _save_error_plot(
        out_vy,
        t,
        err_vy,
        ylabel="vy error (m/s)",
        title=f"vy Error vs Time: {csv_path.name}",
        color="tab:orange",
    )

    print(f"[ok] {csv_path.name} -> {out_vx.name}, {out_vy.name}")
    return out_vx, out_vy


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create per-CSV velocity error plots (vx error vs time and vy error vs time)."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--vx-suffix",
        type=str,
        default="_vx_error_vs_time.png",
        help="Suffix for vx error plot filenames.",
    )
    parser.add_argument(
        "--vy-suffix",
        type=str,
        default="_vy_error_vs_time.png",
        help="Suffix for vy error plot filenames.",
    )
    args = parser.parse_args()

    csv_files = sorted(args.dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.dataset_dir}")

    n = 0
    for csv_path in csv_files:
        out_vx, out_vy = process_csv(csv_path, args.vx_suffix, args.vy_suffix)
        if out_vx is not None and out_vy is not None:
            n += 1

    print(f"Generated error-plot pairs for {n} CSV file(s) in {args.dataset_dir}")


if __name__ == "__main__":
    main()
