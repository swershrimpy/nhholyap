"""
Dimensionality-scaling runtime plot for refinement_demo_stable.py's N=2..100
sweep (see that file's module docstring for the nan bugfix this is the
corrected/rerun version of).

Unlike nonlinear_chain's *_scenario_scaling.py or admire_success_rate_analysis.py,
refinement_demo_stable.py writes NO csv/npz at all -- it only prints
compile/run time (via time_jit on the refined multistart optimizer) to
stdout, once per order N, in its Slurm log. So this script:
  1. Parses that log (LOG_PATH below) for each order N's "refined
     multistart: compile ... ms   run ... ms" line and the "Memory
     snapshot: {...}" line that follows it, pairing both with the
     preceding "order N=<N>" header.
  2. Writes the extracted (N, compile_time_ms, run_time_ms,
     peak_gpu_mem_bytes) rows to integrator_refinement_runtime.csv -- a
     durable, structured artifact, so this doesn't need to re-scrape the
     raw log (or keep the huge log file around) to replot later.
  3. Plots compile time and run time vs. N from that CSV ->
     integrator_refinement_runtime_scaling.pdf.

peak_gpu_mem_bytes is `peak_bytes_in_use` from the XLA allocator snapshot
that time_jit takes after each order's timed calls. That counter is a
process-lifetime running maximum -- it is never reset between orders -- so
in general it is only an upper bound on any single order's own peak. Here
it happens to be exact: every one of the 99 orders sets a NEW record
(the series is strictly increasing, 0.148 MB at N=2 -> 148.6 MB at N=100,
checked in __main__ below), so the value recorded at order N is by
construction the peak reached *while running order N*. If a future rerun
ever produces a flat stretch, the flat points are censored (upper bounds,
not measurements) and the check in __main__ will say so.

Only the REFINED multistart optimizer's timing is available this way --
refinement_demo_stable.py's run_for_order doesn't run the unrefined
optimizer through time_jit, so there's only one series here (no
unrefined-vs-refined comparison, unlike nonlinear_chain's runtime plot).

Usage: python plot_refinement_runtime_scaling.py [log_path]
(defaults to LOG_PATH below if omitted)
"""

import csv
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_PATH = Path("/storage/scratch1/9/xni32/nhholyap/logs/integrator_refinement_scaling_stable_11774929.out")
# Write the durable CSV where the rest of this package keeps its data (and
# where plot_post_compile_runtime_filtered.py reads it from); fall back to
# alongside this script for the original repo layout, where code and data
# share a directory.
_DATA_DIR = _HERE.parents[1] / "data" / "integrator_chain"
OUT_CSV = (_DATA_DIR if _DATA_DIR.is_dir() else _HERE) / "integrator_refinement_runtime.csv"
_PLOT_DIR = _HERE.parents[1] / "plots"
OUT_PDF = (_PLOT_DIR if _PLOT_DIR.is_dir() else _HERE) / "integrator_refinement_runtime_scaling.pdf"

# Categorical slot 1 from the project's validated palette (light mode) --
# only one series here (refined multistart), so a legend box isn't needed
# (see nonlinear_chain/plot_scenario_runtime_scaling.py for the two-series
# unrefined/refined comparison, which uses slots 1/2).
_COLOR_REFINED = "#2a78d6"   # slot 1 (blue)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif'],
    'text.usetex': False,
})

_ORDER_RE = re.compile(r"order N=(\d+)")
_TIMING_RE = re.compile(r"refined multistart: compile\s+([\d.]+) ms\s+run\s+([\d.]+) ms")
_PEAK_MEM_RE = re.compile(r"'peak_bytes_in_use': ([\d.eE+-]+)")


def parse_log(log_path: Path):
    """Returns list of (N, compile_time_ms, run_time_ms, peak_gpu_mem_bytes),
    sorted by N.

    The memory snapshot is printed a few lines AFTER the timing line for the
    same order, so `current_n` is only cleared once both have been seen.
    peak_gpu_mem_bytes is None for an order whose snapshot is missing or
    reports CPU RSS instead of device stats (time_jit's fallback when no GPU
    is present -- that log has no 'peak_bytes_in_use' key at all).
    """
    rows = []
    current_n = None
    timing = None
    with open(log_path) as f:
        for line in f:
            m_order = _ORDER_RE.search(line)
            if m_order:
                current_n = int(m_order.group(1))
                timing = None
                continue
            if current_n is None:
                continue
            m_timing = _TIMING_RE.search(line)
            if m_timing:
                timing = (float(m_timing.group(1)), float(m_timing.group(2)))
                continue
            if "Memory snapshot" in line and timing is not None:
                m_mem = _PEAK_MEM_RE.search(line)
                peak = float(m_mem.group(1)) if m_mem else None
                rows.append((current_n, timing[0], timing[1], peak))
                current_n = None   # each N contributes exactly one such pair
                timing = None
    rows.sort(key=lambda r: r[0])
    return rows


def write_csv(rows, out_csv: Path):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "compile_time_ms", "run_time_ms", "peak_gpu_mem_bytes"])
        writer.writerows(
            (n, c, r, "" if m is None else f"{m:.0f}") for n, c, r, m in rows
        )


def make_plot(rows, out_path: Path = OUT_PDF):
    Ns = [r[0] for r in rows]
    compile_s = [r[1] / 1e3 for r in rows]
    run_ms = [r[2] for r in rows]

    fig, (ax_compile, ax_run) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax_compile.plot(Ns, compile_s, color=_COLOR_REFINED, linewidth=2)
    ax_run.plot(Ns, run_ms, color=_COLOR_REFINED, linewidth=2)

    for ax, ylabel, title in (
        (ax_compile, "Compile time (s)", "JIT compile time"),
        (ax_run, "Run time (ms)", "Post-compilation run time"),
    ):
        ax.set_xlabel("Integrator-chain order N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, color="0.85", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("integrator_chain refined multistart optimizer -- "
                 "compile/run time vs. order N (N=2..100)\n"
                 "(30 restarts x 20 iters per point; see refinement_demo_stable.py)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_PATH
    rows = parse_log(log_path)
    if not rows:
        print(f"No 'refined multistart' timing lines found in {log_path}", file=sys.stderr)
        sys.exit(1)

    write_csv(rows, OUT_CSV)
    print(f"Parsed {len(rows)} (N, compile_time_ms, run_time_ms, peak_gpu_mem_bytes) "
          f"rows from {log_path}")
    print(f"Wrote CSV -> {OUT_CSV}")

    Ns_found = [r[0] for r in rows]
    expected = set(range(2, 101))
    missing = sorted(expected - set(Ns_found))
    if missing:
        print(f"Missing N values (no timing line found): {missing}", file=sys.stderr)

    # peak_bytes_in_use is a process-lifetime running max (see module
    # docstring): it equals order N's own peak only at orders that set a new
    # record. Report any flat/censored points rather than silently treating
    # an upper bound as a measurement.
    mem_rows = [(n, m) for n, _, _, m in rows if m is not None]
    if not mem_rows:
        print("No peak_bytes_in_use in this log (CPU-only run?) -- no memory column.")
    else:
        best = -1.0
        censored = []
        for n, m in mem_rows:
            if m <= best:
                censored.append(n)
            best = max(best, m)
        print(f"Peak GPU memory: {mem_rows[0][1] / 1e6:.3f} MB at N={mem_rows[0][0]} "
              f"-> {mem_rows[-1][1] / 1e6:.3f} MB at N={mem_rows[-1][0]}")
        if censored:
            print(f"WARNING: {len(censored)} order(s) did not set a new peak, so their "
                  f"memory value is an upper bound, not a measurement: {censored}",
                  file=sys.stderr)
        else:
            print("Every order set a new peak -> each memory value is that order's own peak.")

    pdf_path = make_plot(rows)
    print(f"Wrote plot: {pdf_path}")
