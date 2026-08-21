"""
"Best measured runtime" companion to plot_horizon_runtime_scaling.py.

That script plots ONE config's horizon sweep (whatever unrefined_/
refined_horizon_scaling.py's CSV_PATH currently points at -- the *_v2.csv
files by default). This script instead pools every gradient-descent
(num_iters > 0) horizon sweep CSV on disk for each optimizer and, at every
horizon step, takes the FASTEST run_time_avg_ms measured across all of them
-- an empirical lower envelope over restart-width/iteration-count/GPU-type
configs actually run, rather than any single config's curve. Zeroth-order
(num_iters=0) CSVs are deliberately excluded: that mode has no gradient
descent to trade against restart width, so mixing it into a "best GD
runtime" comparison would be apples-to-oranges (see
unrefined_horizon_scaling_bw_zeroth.sbatch's docstring for why it's a
separate baseline, not a config variant of the same search).

Pooled per optimizer (all N=10, num_scenarios=7, dt=0.02s -- see each CSV's
producing script for its exact num_restarts/num_iters):
  unrefined: unrefined_horizon_scaling.csv       (20 restarts x 15 iters)
             unrefined_horizon_scaling_v2.csv     (256 x 8)
             unrefined_horizon_scaling_bw_gd5.csv (4096 x 5)
             unrefined_horizon_scaling_ms4096.csv (4096 x 3)
  refined:   refined_horizon_scaling.csv, _v2.csv, _bw_gd5.csv, _ms4096.csv
             (same four restart/iter configs)

Host RAM is NOT read off whichever config happened to win on run time at
each step -- an earlier version of this script did that and produced a
spurious-looking spike at n_steps=1-2, where the fastest config switches
from n4096i5 (~3020MB, roughly flat across horizon) to n256i8 (~1140MB,
also roughly flat) and the RAM line jumped ~1.9GB for a reason that had
nothing to do with horizon length. Host RAM here is dominated by restart
COUNT, not horizon length -- each individual config's peak_rss_mb is nearly
flat across the whole horizon sweep (see the CSVs directly: n4096i5 sits at
~3018-3025MB from n_steps=1 to 50; n256i8 sits at ~1035-1140MB likewise) --
so stitching RAM from whichever config's run time happened to win mixes two
different restart-count baselines into one curve and reads as a horizon
effect that isn't real.

Instead, RAM is read from ONE FIXED reference config (REFERENCE_CONFIG
below, n256i8 -- the config that already wins run time at 15 of 17 horizon
steps, so it's the natural choice and has full step coverage for both
optimizers) at every horizon step, regardless of which config won on run
time. Run time is still the genuine per-step minimum across all pooled
configs.

Produces the same three single-panel, twin-axis PDFs as
plot_horizon_runtime_scaling.py, under a `_best` suffix so neither script
overwrites the other's output:
  horizon_runtime_scaling_best_unrefined.pdf
  horizon_runtime_scaling_best_refined.pdf
  horizon_runtime_scaling_best_combined.pdf

Does not run any JAX code -- run the relevant *_horizon_scaling*.sbatch jobs
first to (re)generate the CSVs it pools.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_horizon_runtime_scaling import load_rows, _COLOR_UNREFINED, _COLOR_REFINED, _INK

OUT_PDF_UNREFINED = _HERE / "horizon_runtime_scaling_best_unrefined.pdf"
OUT_PDF_REFINED = _HERE / "horizon_runtime_scaling_best_refined.pdf"
OUT_PDF_COMBINED = _HERE / "horizon_runtime_scaling_best_combined.pdf"

UNREFINED_CONFIG_CSVS = [
    ("n20i15", _HERE / "unrefined_horizon_scaling.csv"),
    ("n256i8", _HERE / "unrefined_horizon_scaling_v2.csv"),
    ("n4096i5", _HERE / "unrefined_horizon_scaling_bw_gd5.csv"),
    ("n4096i3", _HERE / "unrefined_horizon_scaling_ms4096.csv"),
]
REFINED_CONFIG_CSVS = [
    ("n20i15", _HERE / "refined_horizon_scaling.csv"),
    ("n256i8", _HERE / "refined_horizon_scaling_v2.csv"),
    ("n4096i5", _HERE / "refined_horizon_scaling_bw_gd5.csv"),
    ("n4096i3", _HERE / "refined_horizon_scaling_ms4096.csv"),
]

# Fixed config RAM is read from, regardless of which config wins on run
# time at a given step -- see module docstring for why. Must match a
# config_name used in the two lists above.
REFERENCE_CONFIG = "n256i8"


def best_rows(config_csvs, label):
    """Pool every (config_name, csv_path) sweep and return, per horizon
    step, the single row with the lowest run_time_avg_ms among all 'ok'
    rows at that step -- plus which config it came from, for the console
    report. peak_rss_mb on the returned row is REPLACED with the reading
    from REFERENCE_CONFIG at the same step (see module docstring) rather
    than left as the winning-run-time row's own RAM, so run time and RAM
    don't silently jump between two different restart-count baselines at
    the same plotted point. A step is dropped if REFERENCE_CONFIG has no
    'ok' row there. Skips CSVs that don't exist (e.g. a config not yet run)."""
    by_step = {}
    ref_ram_by_step = {}
    for config_name, csv_path in config_csvs:
        if not csv_path.exists():
            print(f"  [{label}] skipping {csv_path.name} (not found)")
            continue
        for row in load_rows(csv_path):
            if row["status"] != "ok":
                continue
            n = row["n_steps"]
            if config_name == REFERENCE_CONFIG:
                ref_ram_by_step[n] = row["peak_rss_mb"]
            row = dict(row)
            row["config"] = config_name
            if n not in by_step or row["run_time_avg_ms"] < by_step[n]["run_time_avg_ms"]:
                by_step[n] = row

    out = []
    for n in sorted(by_step):
        if n not in ref_ram_by_step:
            print(f"  [{label}] dropping n_steps={n}: {REFERENCE_CONFIG} has no 'ok' row there")
            continue
        row = dict(by_step[n])
        row["peak_rss_mb"] = ref_ram_by_step[n]
        out.append(row)
    return out


def _report(rows, label):
    print(f"  {label}: {len(rows)} horizon steps (run time = best config, "
          f"RAM = {REFERENCE_CONFIG} reference):")
    for r in rows:
        print(f"    n_steps={r['n_steps']:3d}  run_time_config={r['config']:8s}  "
              f"run_time_avg={r['run_time_avg_ms']:8.3f}ms  peak_rss={r['peak_rss_mb']:7.1f}MB")


def make_plot(series, out_path: Path):
    """series: list of (rows, color, label) tuples. Same visual design as
    plot_horizon_runtime_scaling.make_plot -- see there for the twin-axis
    rationale (run time left/solid/circle, host RAM right/dashed/square,
    color = optimizer identity, no titles)."""
    fig, ax_run = plt.subplots(figsize=(6.5, 4.4))
    ax_ram = ax_run.twinx()

    handles = []
    for rows, color, label in series:
        xs = [r["n_steps"] for r in rows]

        l_run, = ax_run.plot(xs, [r["run_time_avg_ms"] for r in rows],
                             color=color, linewidth=2, linestyle='-',
                             marker='o', markersize=5, label=f"{label} -- run time", zorder=3)
        l_ram, = ax_ram.plot(xs, [r["peak_rss_mb"] for r in rows],
                             color=color, linewidth=2, linestyle='--',
                             marker='s', markersize=4.5, markerfacecolor='white',
                             markeredgewidth=1.0, label=f"{label} -- host RAM", zorder=2)
        handles += [l_run, l_ram]

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
