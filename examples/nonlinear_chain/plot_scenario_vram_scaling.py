"""
GPU VRAM (device peak_bytes_in_use) vs. fault-scenario count, at the
n256i8 reference config (256 restarts x 8 iters -- same config
plot_scenario_runtime_scaling_best.py reads host RAM from), for both
optimizers.

Companion to plot_scenario_runtime_scaling_best.py's finding that host RAM
is flat across the scenario-count sweep (both paths vmap their pairwise term
over _pair_indices(n) rather than unrolling a Python loop per pair, so the
compiled program's op count -- and therefore host/compiler memory -- doesn't
grow with scenario count; see that discussion). VRAM is the other half of
the picture: it DOES grow with scenario count, because vmap still means more
actual tensor data moving through the batched dimension -- more pairs is
more real bytes on the device, just not a bigger compiled program.

Single panel, single y-axis (VRAM only, MB) vs. number of fault scenarios,
one line per optimizer -- unlike the twin-axis runtime+RAM plots elsewhere
in this directory, there's only one metric here so no second axis is needed.
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

_COLOR_UNREFINED = "#2a78d6"   # slot 1 (blue)
_COLOR_REFINED = "#eb6834"     # slot 2 (orange)
_INK = "0.15"

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif'],
    'text.usetex': False,
})

OUT_PDF = _HERE / "scenario_vram_scaling.pdf"

UNREFINED_CSV = _HERE / "unrefined_scenario_scaling_v2.csv"   # n256i8
REFINED_CSV = _HERE / "refined_scenario_scaling_v2.csv"       # n256i8


def load_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["num_scenarios"] = int(float(row["num_scenarios"]))
        if row["status"] == "ok":
            row["memory_kb"] = float(row["memory_kb"])
    rows.sort(key=lambda r: r["num_scenarios"])
    return rows


if __name__ == "__main__":
    unrefined_rows = [r for r in load_rows(UNREFINED_CSV) if r["status"] == "ok"]
    refined_rows = [r for r in load_rows(REFINED_CSV) if r["status"] == "ok"]

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    for rows, color, label in (
        (unrefined_rows, _COLOR_UNREFINED, "Unrefined"),
        (refined_rows, _COLOR_REFINED, "Refined"),
    ):
        xs = [r["num_scenarios"] for r in rows]
        ys = [r["memory_kb"] / 1024.0 for r in rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker='o', markersize=5, label=label)

    ax.set_xlabel("Number of fault scenarios")
    ax.set_ylabel("GPU VRAM, peak device memory (MB)")
    ax.grid(True, color="0.85", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=_INK)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote plot: {OUT_PDF}")
