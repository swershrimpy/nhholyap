"""
Companion plotting script for unrefined_scenario_scaling.py /
refined_scenario_scaling.py's fault-scenario-count runtime sweeps.

Reads unrefined_scenario_scaling.csv and refined_scenario_scaling.csv (one
row per total-scenario-count, N=10 fixed) and produces
scenario_runtime_scaling.pdf: a three-panel figure comparing unrefined vs.
refined multistep optimization -- compile time, steady-state run time
(timeit-style average), and memory usage -- all vs. number of fault
scenarios.

Each row has a `status` column ('ok'/'timeout'/'oom'/'error') -- see
runtime_scaling_common.run_worker_with_limits. Only 'ok' rows have numeric
timing/memory data and are plotted as lines/markers; a dashed vertical line
per series marks the first scenario count that did NOT complete, and the
console output lists exactly which totals failed and why, so a truncated
line is legible as "ran out of budget here" rather than a silent gap.

Does not run any JAX code itself -- run both scaling scripts first to
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

from unrefined_scenario_scaling import CSV_PATH as UNREFINED_CSV_PATH
from refined_scenario_scaling import CSV_PATH as REFINED_CSV_PATH

OUT_PDF = _HERE / "scenario_runtime_scaling.pdf"

# Categorical slots 1/2 from the project's validated palette (light mode):
# blue for unrefined, orange for refined -- consistent with
# plot_refinement_runtime_scaling.py's run-time-panel color assignment.
_COLOR_UNREFINED = "#2a78d6"   # slot 1 (blue)
_COLOR_REFINED = "#eb6834"     # slot 2 (orange)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': False,
})


def load_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run unrefined_scenario_scaling.py and "
            f"refined_scenario_scaling.py first to generate it."
        )
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["num_scenarios"] = int(float(row["num_scenarios"]))
        row["N"] = int(float(row["N"]))
        if row["status"] == "ok":
            for key in ("num_actuator_faults", "num_sensor_faults",
                        "run_time_repeats", "num_restarts", "num_iters"):
                row[key] = int(float(row[key]))
            for key in ("compile_time_ms", "run_time_single_ms", "run_time_avg_ms",
                        "memory_kb", "loss"):
                row[key] = float(row[key])
    rows.sort(key=lambda r: r["num_scenarios"])
    return rows


def _report_failures(rows, label):
    failures = [r for r in rows if r["status"] != "ok"]
    if not failures:
        print(f"  {label}: all {len(rows)} scenario counts completed.")
        return
    print(f"  {label}: {len(failures)} of {len(rows)} scenario counts did not complete:")
    for r in failures:
        print(f"    num_scenarios={r['num_scenarios']:3d}  status={r['status']:8s}  "
              f"elapsed={r['elapsed_s']}s  peak_rss={r['peak_rss_mb']}MB")


def make_plot(unrefined_rows, refined_rows, out_path: Path = OUT_PDF):
    fig, (ax_compile, ax_run, ax_mem) = plt.subplots(1, 3, figsize=(16, 4.4))

    series = (
        (unrefined_rows, _COLOR_UNREFINED, "Unrefined multistep"),
        (refined_rows, _COLOR_REFINED, "Refined multistep"),
    )

    for ax, key, ylabel, title in (
        (ax_compile, "compile_time_ms", "Compile time (s)", "JIT compile time"),
        (ax_run, "run_time_avg_ms", "Run time (ms)", "Post-compilation run time"),
        (ax_mem, "memory_kb", "Memory (MB)", "Memory usage"),
    ):
        for rows, color, label in series:
            ok_rows = [r for r in rows if r["status"] == "ok"]
            xs = [r["num_scenarios"] for r in ok_rows]
            if key == "compile_time_ms":
                ys = [r[key] / 1e3 for r in ok_rows]
            elif key == "memory_kb":
                ys = [r[key] / 1024.0 for r in ok_rows]
            else:
                ys = [r[key] for r in ok_rows]
            ax.plot(xs, ys, color=color, linewidth=2, marker='o', markersize=5, label=label)

            # Mark the first scenario count that failed to complete (if any)
            # with a dashed vertical line in this series' color, so a
            # truncated curve reads as "hit a resource limit here", not a
            # missing/buggy data point.
            failed = sorted(r["num_scenarios"] for r in rows if r["status"] != "ok")
            if failed:
                ax.axvline(failed[0], color=color, linestyle='--', linewidth=1, alpha=0.6)

        ax.set_xlabel("Number of fault scenarios")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, color="0.85", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax_run.legend(frameon=False)
    fig.suptitle("nonlinear_chain (N=10) -- fault-scenario-count runtime/memory scaling\n"
                 "(dashed line = first scenario count that timed out / hit the memory cap)")
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

    pdf_path = make_plot(unrefined_rows, refined_rows)
    print(f"\nWrote plot: {pdf_path}")
