"""
Unrefined-vs-refined dimensionality-scaling plot for integrator_chain's
N=2..100 sweep, combining integrator_unrefined_runtime.csv
(unrefined_demo_stable.py, job 11944255) and integrator_refinement_runtime.csv
(refinement_demo_stable.py, job 11915477) -- the two CSVs
plot_unrefined_runtime_scaling.py / plot_refinement_runtime_scaling.py
already produce from their respective logs.

Produces THREE single-panel, twin-axis figures:
  - integrator_unrefined_vs_refined_scaling_unrefined.pdf -- unrefined only
  - integrator_unrefined_vs_refined_scaling_refined.pdf   -- refined only
  - integrator_unrefined_vs_refined_scaling_combined.pdf  -- both overlaid (4 curves)
Each plots run time (left y-axis) and host RAM (right y-axis) as two lines
vs. order N. Same design as nonlinear_chain's plot_horizon_runtime_scaling.py:
a deliberate twin y-axis (see that script's docstring for the tradeoff),
no titles (legend is the only in-figure text, so entries always name both
optimizer and metric), color = optimizer identity (blue=unrefined,
orange=refined, consistent with every other plot in this project),
linestyle+marker = metric identity (solid+circle=run time,
dashed+square=host RAM) since color alone can't carry both dimensions in
the 4-curve combined figure.

Host RAM here is peak_host_rss_bytes -- the per-order resettable peak from
integrator_separating_input._HostPeakRSS (see that module's docstring).
GPU VRAM (peak_gpu_mem_bytes) is not plotted: it's a much smaller number
(single/double-digit MB across the whole sweep, see
plot_refinement_runtime_scaling.py's printed summary) and not a
significant contributor next to host RAM's multi-GB range.

Usage: python plot_unrefined_vs_refined_scaling.py
(reads the two CSVs alongside this script -- run
plot_unrefined_runtime_scaling.py and plot_refinement_runtime_scaling.py
first if they don't exist yet)
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

UNREFINED_CSV_PATH = _HERE / "integrator_unrefined_runtime.csv"
REFINED_CSV_PATH = _HERE / "integrator_refinement_runtime.csv"

OUT_PDF_UNREFINED = _HERE / "integrator_unrefined_vs_refined_scaling_unrefined.pdf"
OUT_PDF_REFINED = _HERE / "integrator_unrefined_vs_refined_scaling_refined.pdf"
OUT_PDF_COMBINED = _HERE / "integrator_unrefined_vs_refined_scaling_combined.pdf"

# Categorical slots 1/2 from the project's validated palette (light mode):
# same blue-for-unrefined / orange-for-refined assignment as every other
# unrefined/refined comparison in this project.
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
            f"{csv_path} not found -- run plot_unrefined_runtime_scaling.py / "
            f"plot_refinement_runtime_scaling.py on the matching Slurm log first."
        )
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["N"] = int(float(row["N"]))
        row["compile_time_ms"] = float(row["compile_time_ms"])
        row["run_time_ms"] = float(row["run_time_ms"])
        row["peak_host_rss_mb"] = float(row["peak_host_rss_bytes"]) / 1e6
    rows.sort(key=lambda r: r["N"])
    return rows


def make_plot(series, out_path: Path):
    """series: list of (rows, color, label) tuples -- one entry for a
    single-optimizer figure, two (unrefined + refined) for the overlaid
    comparison. Single panel, twin y-axis: run time (left) and host RAM
    (right) vs. order N."""
    fig, ax_run = plt.subplots(figsize=(6.5, 4.4))
    ax_ram = ax_run.twinx()

    handles = []
    for rows, color, label in series:
        Ns = [r["N"] for r in rows]

        # No markers -- N=2..100 is 99 points, dense enough on a ~6.5in axis
        # that markers merge into a solid band (see
        # plot_refinement_runtime_scaling.py's own plain-line style for this
        # same N range); linestyle alone (solid vs. dashed) carries metric
        # identity here instead.
        l_run, = ax_run.plot(Ns, [r["run_time_ms"] for r in rows],
                             color=color, linewidth=2, linestyle='-',
                             label=f"{label} -- run time", zorder=3)
        l_ram, = ax_ram.plot(Ns, [r["peak_host_rss_mb"] for r in rows],
                             color=color, linewidth=2, linestyle='--',
                             label=f"{label} -- host RAM", zorder=2)
        handles += [l_run, l_ram]

    ax_run.set_xlabel("Integrator-chain order N")
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

    unrefined_series = (unrefined_rows, _COLOR_UNREFINED, "Unrefined")
    refined_series = (refined_rows, _COLOR_REFINED, "Refined")

    out1 = make_plot([unrefined_series], OUT_PDF_UNREFINED)
    print(f"Wrote plot: {out1}")

    out2 = make_plot([refined_series], OUT_PDF_REFINED)
    print(f"Wrote plot: {out2}")

    out3 = make_plot([unrefined_series, refined_series], OUT_PDF_COMBINED)
    print(f"Wrote plot: {out3}")
