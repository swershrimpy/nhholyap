"""
Dimensionality-scaling runtime plot for unrefined_demo_stable.py's N=2..100
sweep -- the unrefined-only counterpart to plot_refinement_runtime_scaling.py
(see unrefined_demo_stable.py's module docstring for why this is a separate
script/sweep rather than a second series added to the refined one: the
refined sweep's run_for_order never times the unrefined optimizer at all).

Same log-scraping approach as plot_refinement_runtime_scaling.py, just
matching "unrefined multistart: compile ... ms   run ... ms" lines instead
of "refined multistart: ...":
  1. Parses the log for each order N's timing line and the "Memory
     snapshot: {...}" line that follows it, pairing both with the
     preceding "order N=<N>" header.
  2. Writes (N, compile_time_ms, run_time_ms, peak_gpu_mem_bytes,
     peak_host_rss_bytes) rows to integrator_unrefined_runtime.csv.
  3. Plots compile time and run time vs. N ->
     integrator_unrefined_runtime_scaling.pdf.

peak_gpu_mem_bytes is peak_bytes_in_use, a process-lifetime running max (see
plot_refinement_runtime_scaling.py's docstring for the full caveat) --
__main__ below checks whether every order here also set a new record, same
as the refined sweep did, rather than assuming it. peak_host_rss_bytes is
reset per order (integrator_separating_input._HostPeakRSS), so it needs no
equivalent running-max caveat.

Usage: python plot_unrefined_runtime_scaling.py <log_path>
"""

import csv
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_DATA_DIR = _HERE.parents[1] / "data" / "integrator_chain"
OUT_CSV = (_DATA_DIR if _DATA_DIR.is_dir() else _HERE) / "integrator_unrefined_runtime.csv"
_PLOT_DIR = _HERE.parents[1] / "plots"
OUT_PDF = (_PLOT_DIR if _PLOT_DIR.is_dir() else _HERE) / "integrator_unrefined_runtime_scaling.pdf"

# Categorical slot 1 from the project's validated palette (light mode) --
# same blue used for "unrefined" everywhere else in this project.
_COLOR_UNREFINED = "#2a78d6"   # slot 1 (blue)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif'],
    'text.usetex': False,
})

_ORDER_RE = re.compile(r"order N=(\d+)")
_TIMING_RE = re.compile(r"unrefined multistart: compile\s+([\d.]+) ms\s+run\s+([\d.]+) ms")
_PEAK_MEM_RE = re.compile(r"'peak_bytes_in_use': ([\d.eE+-]+)")
_HOST_RSS_RE = re.compile(r"'host_peak_rss_bytes': ([\d.eE+-]+)")


def parse_log(log_path: Path):
    """Returns list of (N, compile_time_ms, run_time_ms, peak_gpu_mem_bytes,
    peak_host_rss_bytes), sorted by N. See plot_refinement_runtime_scaling.py's
    parse_log for the pairing/None-handling logic this mirrors exactly."""
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
                m_host = _HOST_RSS_RE.search(line)
                host_peak = float(m_host.group(1)) if m_host else None
                rows.append((current_n, timing[0], timing[1], peak, host_peak))
                current_n = None   # each N contributes exactly one such pair
                timing = None
    rows.sort(key=lambda r: r[0])
    return rows


def write_csv(rows, out_csv: Path):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "compile_time_ms", "run_time_ms", "peak_gpu_mem_bytes",
                          "peak_host_rss_bytes"])
        writer.writerows(
            (n, c, r, "" if m is None else f"{m:.0f}", "" if h is None else f"{h:.0f}")
            for n, c, r, m, h in rows
        )


def make_plot(rows, out_path: Path = OUT_PDF):
    Ns = [r[0] for r in rows]
    compile_s = [r[1] / 1e3 for r in rows]
    run_ms = [r[2] for r in rows]

    fig, (ax_compile, ax_run) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax_compile.plot(Ns, compile_s, color=_COLOR_UNREFINED, linewidth=2)
    ax_run.plot(Ns, run_ms, color=_COLOR_UNREFINED, linewidth=2)

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

    fig.suptitle("integrator_chain unrefined multistart optimizer -- "
                 "compile/run time vs. order N (N=2..100)\n"
                 "(30 restarts x 20 iters per point; see unrefined_demo_stable.py)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python plot_unrefined_runtime_scaling.py <log_path>", file=sys.stderr)
        sys.exit(2)
    log_path = Path(sys.argv[1])
    rows = parse_log(log_path)
    if not rows:
        print(f"No 'unrefined multistart' timing lines found in {log_path}", file=sys.stderr)
        sys.exit(1)

    write_csv(rows, OUT_CSV)
    print(f"Parsed {len(rows)} (N, compile_time_ms, run_time_ms, peak_gpu_mem_bytes, "
          f"peak_host_rss_bytes) rows from {log_path}")
    print(f"Wrote CSV -> {OUT_CSV}")

    Ns_found = [r[0] for r in rows]
    expected = set(range(2, 101))
    missing = sorted(expected - set(Ns_found))
    if missing:
        print(f"Missing N values (no timing line found): {missing}", file=sys.stderr)

    mem_rows = [(n, m) for n, _, _, m, _ in rows if m is not None]
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
            print(f"NOTE: {len(censored)} order(s) did not set a new peak, so their "
                  f"memory value is an upper bound, not a measurement: {censored}",
                  file=sys.stderr)
        else:
            print("Every order set a new peak -> each memory value is that order's own peak.")

    host_rows = [(n, h) for n, _, _, _, h in rows if h is not None]
    if not host_rows:
        print("No host_peak_rss_bytes in this log (predates host-RAM instrumentation) "
              "-- no host RAM column.")
    else:
        print(f"Peak host RSS: {host_rows[0][1] / 1e6:.3f} MB at N={host_rows[0][0]} "
              f"-> {host_rows[-1][1] / 1e6:.3f} MB at N={host_rows[-1][0]} "
              f"(min {min(h for _, h in host_rows) / 1e6:.3f} MB, "
              f"max {max(h for _, h in host_rows) / 1e6:.3f} MB)")

    pdf_path = make_plot(rows)
    print(f"Wrote plot: {pdf_path}")
