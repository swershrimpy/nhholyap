"""
Companion plotting script for unrefined_horizon_scaling.py /
refined_horizon_scaling.py's PLANNING-HORIZON runtime/RAM/VRAM sweeps.

Reads unrefined_horizon_scaling.csv and refined_horizon_scaling.csv (one row
per horizon length, N=10 and num_scenarios=7 both fixed) and produces THREE
single-panel, twin-axis figures:
  - horizon_runtime_scaling_unrefined.pdf -- unrefined multistep only
  - horizon_runtime_scaling_refined.pdf   -- refined multistep only
  - horizon_runtime_scaling_combined.pdf  -- both optimizers overlaid
Each panel plots run time (left y-axis) and host RAM (right y-axis) as two
lines on a shared x-axis of planning steps (num_segments / num_steps --
Euler steps at the fixed dt=0.02s, not horizon in seconds). No titles, by
request -- the legend is the only in-figure text identifying which line is
which, so legend labels always name both the optimizer and the metric even
in the single-optimizer figures.

This is a deliberate twin y-axis (independent left/right scales), unlike
this project's usual default of separate panels for differently-scaled
metrics (see integrator_chain's plot_post_compile_runtime_filtered.py for
the same tradeoff written out) -- curve crossings between the run-time and
RAM lines carry no meaning here, only each line's own shape against its own
axis is readable. Color still means optimizer identity (blue=unrefined,
orange=refined, matching every other plot in this project); linestyle
(solid=run time, dashed=RAM) and marker (circle=run time, square=RAM)
distinguish metric within a color, which is what the 4-line combined figure
needs since color alone doesn't carry a second dimension. GPU VRAM and
compile time are not plotted here -- VRAM stays in the single-digit MB
range across the whole sweep and isn't a significant contributor next to
host RAM's multi-GB range.

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

OUT_PDF_UNREFINED = _HERE / "horizon_runtime_scaling_unrefined.pdf"
OUT_PDF_REFINED = _HERE / "horizon_runtime_scaling_refined.pdf"
OUT_PDF_COMBINED = _HERE / "horizon_runtime_scaling_combined.pdf"

# Categorical slots 1/2 from the project's validated palette (light mode):
# same blue-for-unrefined / orange-for-refined assignment as
# plot_scenario_runtime_scaling.py and plot_refinement_runtime_scaling.py.
_COLOR_UNREFINED = "#2a78d6"   # slot 1 (blue)
_COLOR_REFINED = "#eb6834"     # slot 2 (orange)
_INK = "0.15"

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
    # unrefined_horizon_scaling.csv keys this "num_segments", refined_horizon_scaling.csv
    # keys it "num_steps" -- both are the same thing (Euler steps at dt=0.02s), so
    # normalize to one name here rather than have every caller know which CSV it is.
    n_key = "num_segments" if rows and "num_segments" in rows[0] else "num_steps"
    for row in rows:
        row["n_steps"] = int(float(row[n_key]))
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
    rows.sort(key=lambda r: r["n_steps"])
    return rows


def _report_failures(rows, label):
    failures = [r for r in rows if r["status"] != "ok"]
    if not failures:
        print(f"  {label}: all {len(rows)} horizons completed.")
        return
    print(f"  {label}: {len(failures)} of {len(rows)} horizons did not complete:")
    for r in failures:
        print(f"    n_steps={r['n_steps']:3d}  status={r['status']:8s}  "
              f"elapsed={r['elapsed_s']}s  peak_rss={r['peak_rss_mb']}MB")


def make_plot(series, out_path: Path):
    """series: list of (rows, color, label) tuples -- one entry for a
    single-optimizer figure, two (unrefined + refined) for the overlaid
    comparison. Single panel, twin y-axis: run time (left) and host RAM
    (right) vs. number of planning steps."""
    fig, ax_run = plt.subplots(figsize=(6.5, 4.4))
    ax_ram = ax_run.twinx()

    handles = []
    for rows, color, label in series:
        ok_rows = [r for r in rows if r["status"] == "ok"]
        xs = [r["n_steps"] for r in ok_rows]

        l_run, = ax_run.plot(xs, [r["run_time_avg_ms"] for r in ok_rows],
                             color=color, linewidth=2, linestyle='-',
                             marker='o', markersize=5, label=f"{label} -- run time", zorder=3)
        l_ram, = ax_ram.plot(xs, [r["peak_rss_mb"] for r in ok_rows],
                             color=color, linewidth=2, linestyle='--',
                             marker='s', markersize=4.5, markerfacecolor='white',
                             markeredgewidth=1.0, label=f"{label} -- host RAM", zorder=2)
        handles += [l_run, l_ram]

        # Mark the first horizon that failed to complete (if any) with a
        # dotted vertical line in this series' color, so a truncated curve
        # reads as "hit a resource limit here", not a missing point.
        failed = sorted(r["n_steps"] for r in rows if r["status"] != "ok")
        if failed:
            ax_run.axvline(failed[0], color=color, linestyle=':', linewidth=1, alpha=0.6)

    ax_run.set_xlabel("Number of planning steps")
    ax_run.set_ylabel("Run time (ms)")
    ax_ram.set_ylabel("Host RAM (MB)")
    ax_run.grid(True, color="0.85", linewidth=0.8)
    ax_ram.grid(False)
    ax_run.spines["top"].set_visible(False)
    ax_ram.spines["top"].set_visible(False)
    ax_run.tick_params(axis="y", labelcolor=_INK)
    ax_ram.tick_params(axis="y", labelcolor=_INK)

    ax_run.legend(handles=handles, frameon=False, loc="upper left",
                  fontsize=9, labelcolor=_INK, handlelength=2.4)

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

    unrefined_series = (unrefined_rows, _COLOR_UNREFINED, "Unrefined")
    refined_series = (refined_rows, _COLOR_REFINED, "Refined")

    out1 = make_plot([unrefined_series], OUT_PDF_UNREFINED)
    print(f"\nWrote plot: {out1}")

    out2 = make_plot([refined_series], OUT_PDF_REFINED)
    print(f"Wrote plot: {out2}")

    out3 = make_plot([unrefined_series, refined_series], OUT_PDF_COMBINED)
    print(f"Wrote plot: {out3}")
