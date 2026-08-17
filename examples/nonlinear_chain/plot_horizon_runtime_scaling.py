"""
Companion plotting script for unrefined_horizon_scaling.py /
refined_horizon_scaling.py's PLANNING-HORIZON runtime/RAM/VRAM sweeps.

Reads unrefined_horizon_scaling.csv and refined_horizon_scaling.csv (one row
per horizon length, N=10 and num_scenarios=7 both fixed) and produces SIX
single-panel figures -- one metric (run time or host RAM) crossed with one
scope (unrefined only, refined only, both overlaid):
  - horizon_runtime_scaling_unrefined_runtime.pdf
  - horizon_runtime_scaling_unrefined_ram.pdf
  - horizon_runtime_scaling_refined_runtime.pdf
  - horizon_runtime_scaling_refined_ram.pdf
  - horizon_runtime_scaling_combined_runtime.pdf
  - horizon_runtime_scaling_combined_ram.pdf
Run time and host RAM are always separate figures, never two panels or a
twin axis on one figure -- their units (ms vs. MB) aren't comparable on a
shared scale. GPU VRAM and compile time are not plotted here: VRAM stays
in the single-digit MB range across the whole horizon sweep (see this
file's git history for the earlier four-panel compile/run/RAM/VRAM
version, if that's wanted again) and isn't a significant contributor next
to host RAM's multi-GB range.

Host RAM here is peak_rss_mb -- the worker subprocess's own peak RSS,
externally polled from /proc by the driver (see
unrefined_horizon_scaling.py's docstring) -- not GPU VRAM.

Each row has a `status` column ('ok'/'timeout'/'oom'/'error') -- see
runtime_scaling_common.run_worker_with_limits. Only 'ok' rows have numeric
timing/memory data and are plotted as lines/markers; a dashed vertical line
per series marks the first horizon that did NOT complete, and the console
output lists exactly which horizons failed and why.

Does not run any JAX code itself -- run both horizon scripts first to
(re)generate the CSVs, then run this script to plot them.
"""

import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from unrefined_horizon_scaling import CSV_PATH as UNREFINED_CSV_PATH
from refined_horizon_scaling import CSV_PATH as REFINED_CSV_PATH

_OUT_STEM = _HERE / "horizon_runtime_scaling"

# Categorical slots 1/2 from the project's validated palette (light mode):
# same blue-for-unrefined / orange-for-refined assignment as
# plot_scenario_runtime_scaling.py and plot_refinement_runtime_scaling.py.
_COLOR_UNREFINED = "#2a78d6"   # slot 1 (blue)
_COLOR_REFINED = "#eb6834"     # slot 2 (orange)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif'],
    'text.usetex': False,
})


def load_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run unrefined_horizon_scaling.py and "
            f"refined_horizon_scaling.py first to generate it."
        )
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["horizon_s"] = float(row["horizon_s"])
        row["N"] = int(float(row["N"]))
        row["num_scenarios"] = int(float(row["num_scenarios"]))
        if row["status"] == "ok":
            for key in ("num_actuator_faults", "num_sensor_faults",
                        "run_time_repeats", "num_restarts", "num_iters"):
                row[key] = int(float(row[key]))
            for key in ("compile_time_ms", "run_time_single_ms", "run_time_avg_ms",
                        "memory_kb", "loss", "peak_rss_mb"):
                row[key] = float(row[key])
    rows.sort(key=lambda r: r["horizon_s"])
    return rows


def _report_failures(rows, label):
    failures = [r for r in rows if r["status"] != "ok"]
    if not failures:
        print(f"  {label}: all {len(rows)} horizons completed.")
        return
    print(f"  {label}: {len(failures)} of {len(rows)} horizons did not complete:")
    for r in failures:
        print(f"    horizon={r['horizon_s']:.2f}s  status={r['status']:8s}  "
              f"elapsed={r['elapsed_s']}s  peak_rss={r['peak_rss_mb']}MB")


_METRICS = {
    "runtime": ("run_time_avg_ms", "Run time (ms)", "Post-compilation run time"),
    "ram": ("peak_rss_mb", "Host RAM (MB)", "Peak host RAM"),
}


def make_plot(series, key: str, ylabel: str, title: str, out_path: Path):
    """series: list of (rows, color, label) tuples -- one entry for a
    single-optimizer figure, two (unrefined + refined) for the overlaid
    comparison. Always a single panel, one metric (given by `key`) vs.
    horizon -- run time and host RAM are never on the same axes (different
    units, ms vs. MB)."""
    fig, ax = plt.subplots(figsize=(6, 4.4))

    for rows, color, label in series:
        ok_rows = [r for r in rows if r["status"] == "ok"]
        xs = [r["horizon_s"] for r in ok_rows]
        ys = [r[key] for r in ok_rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker='o', markersize=5, label=label)

        # Mark the first horizon that failed to complete (if any) with a
        # dashed vertical line in this series' color, so a truncated curve
        # reads as "hit a resource limit here", not a missing point.
        failed = sorted(r["horizon_s"] for r in rows if r["status"] != "ok")
        if failed:
            ax.axvline(failed[0], color=color, linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel("Planning horizon (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.85", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if len(series) > 1:
        ax.legend(frameon=False)
    ax.set_title(f"nonlinear_chain (N=10, 7 fault scenarios)\n{title}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    unrefined_rows = load_rows(UNREFINED_CSV_PATH)
    refined_rows = load_rows(REFINED_CSV_PATH)
    print(f"Loaded {len(unrefined_rows)} unrefined rows from {UNREFINED_CSV_PATH}")
    print(f"Loaded {len(refined_rows)} refined rows from {REFINED_CSV_PATH}")
    print("\nCompletion summary:")
    _report_failures(unrefined_rows, "Unrefined")
    _report_failures(refined_rows, "Refined")

    unrefined_series = (unrefined_rows, _COLOR_UNREFINED, "Unrefined multistep")
    refined_series = (refined_rows, _COLOR_REFINED, "Refined multistep")

    scopes = (
        ("unrefined", [unrefined_series]),
        ("refined", [refined_series]),
        ("combined", [unrefined_series, refined_series]),
    )

    for scope_name, series in scopes:
        for metric_name, (key, ylabel, title) in _METRICS.items():
            out_path = Path(f"{_OUT_STEM}_{scope_name}_{metric_name}.pdf")
            make_plot(series, key, ylabel, title, out_path)
            print(f"Wrote plot: {out_path}")
