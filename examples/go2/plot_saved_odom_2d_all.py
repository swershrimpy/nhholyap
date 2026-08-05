import argparse
import csv
from pathlib import Path


def _read_xy(csv_path: Path):
    x_vals = []
    y_vals = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError("CSV missing required columns: x, y")

        for row in reader:
            try:
                x_vals.append(float(row["x"]))
                y_vals.append(float(row["y"]))
            except (TypeError, ValueError):
                continue

    if not x_vals:
        raise ValueError("No valid x/y rows found")
    return x_vals, y_vals


def _save_plot(csv_path: Path, out_path: Path, x_vals, y_vals):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 7))
    plt.plot(x_vals, y_vals, color="tab:blue", linewidth=1.6, label="odom trajectory")
    plt.scatter([x_vals[0]], [y_vals[0]], color="green", marker="o", s=45, label="start")
    plt.scatter([x_vals[-1]], [y_vals[-1]], color="red", marker="x", s=55, label="end")
    plt.title(f"2D Odom Trajectory: {csv_path.name}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_file(csv_path: Path, output_suffix: str):
    try:
        x_vals, y_vals = _read_xy(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: {exc}")
        return None

    out_path = csv_path.with_name(f"{csv_path.stem}{output_suffix}")
    _save_plot(csv_path, out_path, x_vals, y_vals)
    print(f"[ok] {csv_path.name} -> {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Read saved odom CSV files and create one 2D x-y trajectory plot per CSV."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "separating_controller_logs",
        help="Directory containing odom CSV files.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_trajectory_2d.png",
        help="Suffix appended to each CSV stem for output plot filename.",
    )
    args = parser.parse_args()

    csv_files = sorted(args.logs_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.logs_dir}")

    n = 0
    for csv_path in csv_files:
        out = process_file(csv_path, args.output_suffix)
        if out is not None:
            n += 1

    print(f"Generated {n} 2D trajectory plot(s) in {args.logs_dir}")


if __name__ == "__main__":
    main()
