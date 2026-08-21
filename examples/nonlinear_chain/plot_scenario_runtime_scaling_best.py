"""
"Best measured runtime" companion to plot_scenario_runtime_scaling.py, on the
SCENARIO-COUNT axis -- the direct analog of plot_horizon_runtime_scaling_best.py
(see that module's docstring for the full design rationale, repeated briefly
here).

Pools every gradient-descent (num_iters > 0) scenario-count sweep CSV on disk
for each optimizer and, at every scenario count, takes the FASTEST
run_time_avg_ms measured across all of them. Zeroth-order (num_iters=0) CSVs
are excluded -- no gradient descent to trade against restart width, so mixing
it into a "best GD runtime" comparison is apples-to-oranges.

Host RAM is NOT read off whichever config won on run time at a given step --
see plot_horizon_runtime_scaling_best.py's docstring for why that produces a
spurious jump when the winning config switches restart-count baseline
mid-plot. Instead RAM is read from ONE FIXED reference config
(REFERENCE_CONFIG, "n256i8" -- the config that wins run time at nearly every
point below and has full 3-21 scenario-count coverage for both optimizers
with no failures) at every scenario count, regardless of which config won on
speed.

Pooled per optimizer (all N=10, dt=0.02s -- see each CSV's producing script
for its exact num_restarts/num_iters and fixed horizon):
  unrefined: unrefined_scenario_scaling.csv        (20 restarts x 15 iters)
             unrefined_scenario_scaling_v2.csv      (256 x 8)
             unrefined_scenario_scaling_bw_gd5.csv  (4096 x 5)
             unrefined_scenario_scaling_ms4096.csv  (4096 x 3)
             unrefined_scenario_scaling_wide_gd8.csv (1.2M x 8)
  refined:   refined_scenario_scaling.csv, _v2.csv, _bw_gd5.csv, _ms4096.csv,
             _wide_gd8.csv (200k x 8 for the last one)

Produces the same style of single-panel, twin-axis PDF as
plot_horizon_runtime_scaling_best.py:
  scenario_runtime_scaling_best_unrefined.pdf
  scenario_runtime_scaling_best_refined.pdf
  scenario_runtime_scaling_best_combined.pdf

Does not run any JAX code -- run the relevant *_scenario_scaling*.sbatch jobs
first to (re)generate the CSVs it pools.
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

OUT_PDF_UNREFINED = _HERE / "scenario_runtime_scaling_best_unrefined.pdf"
OUT_PDF_REFINED = _HERE / "scenario_runtime_scaling_best_refined.pdf"
OUT_PDF_COMBINED = _HERE / "scenario_runtime_scaling_best_combined.pdf"

UNREFINED_CONFIG_CSVS = [
    ("n20i15", _HERE / "unrefined_scenario_scaling.csv"),
    ("n256i8", _HERE / "unrefined_scenario_scaling_v2.csv"),
    ("n4096i5", _HERE / "unrefined_scenario_scaling_bw_gd5.csv"),
    ("n4096i3", _HERE / "unrefined_scenario_scaling_ms4096.csv"),
    ("n1200000i8", _HERE / "unrefined_scenario_scaling_wide_gd8.csv"),
]
REFINED_CONFIG_CSVS = [
    ("n20i15", _HERE / "refined_scenario_scaling.csv"),
    ("n256i8", _HERE / "refined_scenario_scaling_v2.csv"),
    ("n4096i5", _HERE / "refined_scenario_scaling_bw_gd5.csv"),
    ("n4096i3", _HERE / "refined_scenario_scaling_ms4096.csv"),
    ("n200000i8", _HERE / "refined_scenario_scaling_wide_gd8.csv"),
]

# Fixed config RAM is read from, regardless of which config wins on run time
# at a given scenario count -- see module docstring.
REFERENCE_CONFIG = "n256i8"


def load_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["num_scenarios"] = int(float(row["num_scenarios"]))
        row["N"] = int(float(row["N"]))
        if row["status"] == "ok":
            for key in ("run_time_repeats", "num_restarts", "num_iters"):
                row[key] = int(float(row[key]))
            for key in ("compile_time_ms", "run_time_avg_ms", "memory_kb",
                        "loss", "peak_rss_mb"):
                row[key] = float(row[key])
    rows.sort(key=lambda r: r["num_scenarios"])
    return rows


def best_rows(config_csvs, label):
    """Pool every (config_name, csv_path) sweep and return, per scenario
    count, the row with the lowest run_time_avg_ms among all 'ok' rows there
    -- with peak_rss_mb REPLACED by REFERENCE_CONFIG's reading at the same
    scenario count (see module docstring). A scenario count is dropped if
    REFERENCE_CONFIG has no 'ok' row there. Skips CSVs that don't exist."""
    by_n = {}
    ref_ram_by_n = {}
    for config_name, csv_path in config_csvs:
        if not csv_path.exists():
            print(f"  [{label}] skipping {csv_path.name} (not found)")
            continue
        for row in load_rows(csv_path):
            if row["status"] != "ok":
                continue
            n = row["num_scenarios"]
            if config_name == REFERENCE_CONFIG:
                ref_ram_by_n[n] = row["peak_rss_mb"]
            row = dict(row)
            row["config"] = config_name
            if n not in by_n or row["run_time_avg_ms"] < by_n[n]["run_time_avg_ms"]:
                by_n[n] = row

    out = []
    for n in sorted(by_n):
        if n not in ref_ram_by_n:
            print(f"  [{label}] dropping num_scenarios={n}: {REFERENCE_CONFIG} has no 'ok' row there")
            continue
        row = dict(by_n[n])
        row["peak_rss_mb"] = ref_ram_by_n[n]
        out.append(row)
    return out


def _report(rows, label):
    print(f"  {label}: {len(rows)} scenario counts (run time = best config, "
          f"RAM = {REFERENCE_CONFIG} reference):")
    for r in rows:
        print(f"    num_scenarios={r['num_scenarios']:3d}  run_time_config={r['config']:11s}  "
              f"run_time_avg={r['run_time_avg_ms']:9.3f}ms  peak_rss={r['peak_rss_mb']:7.1f}MB")


def make_plot(series, out_path: Path):
    """series: list of (rows, color, label) tuples. Same visual design as
    plot_horizon_runtime_scaling.make_plot -- run time left/solid/circle,
    host RAM right/dashed/square, color = optimizer identity, no titles."""
    fig, ax_run = plt.subplots(figsize=(6.5, 4.4))
    ax_ram = ax_run.twinx()

    handles = []
    for rows, color, label in series:
        xs = [r["num_scenarios"] for r in rows]

        l_run, = ax_run.plot(xs, [r["run_time_avg_ms"] for r in rows],
                             color=color, linewidth=2, linestyle='-',
                             marker='o', markersize=5, label=f"{label} -- run time", zorder=3)
        l_ram, = ax_ram.plot(xs, [r["peak_rss_mb"] for r in rows],
                             color=color, linewidth=2, linestyle='--',
                             marker='s', markersize=4.5, markerfacecolor='white',
                             markeredgewidth=1.0, label=f"{label} -- host RAM", zorder=2)
        handles += [l_run, l_ram]

    ax_run.set_xlabel("Number of fault scenarios")
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
    print("Pooling configs (excluding zeroth-order):")
    unrefined_rows = best_rows(UNREFINED_CONFIG_CSVS, "Unrefined")
    refined_rows = best_rows(REFINED_CONFIG_CSVS, "Refined")
    print()
    _report(unrefined_rows, "Unrefined")
    _report(refined_rows, "Refined")

    unrefined_series = (unrefined_rows, _COLOR_UNREFINED, "Unrefined")
    refined_series = (refined_rows, _COLOR_REFINED, "Refined")

    out1 = make_plot([unrefined_series], OUT_PDF_UNREFINED)
    print(f"\nWrote plot: {out1}")

    out2 = make_plot([refined_series], OUT_PDF_REFINED)
    print(f"Wrote plot: {out2}")

    out3 = make_plot([unrefined_series, refined_series], OUT_PDF_COMBINED)
    print(f"Wrote plot: {out3}")
