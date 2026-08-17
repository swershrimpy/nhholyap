"""
Post-compilation runtime + memory usage, twin-axis, for the nonlinear_chain
(N=10 fixed) fault-scenario-count sweep -- a companion to
plot_scenario_runtime_scaling.py's 3-panel compile/run/memory figure, not a
replacement: this reuses that script's `load_rows` PARSER directly (same
reasoning as elsewhere in this project -- one parser, not two that can
drift), but renders only run_time_avg_ms ("post-compilation run time" --
the timeit-style steady-state average, not the noisier single dispatched
call) and peak_rss_mb, each as ONE curve, no compile-time panel.

Data source: the PACE sweep (examples/pace_runtime_scaling_package/
pace_pack/data/nonlinear_chain/{unrefined,refined}_scenario_scaling.csv),
NOT this directory's own {unrefined,refined}_scenario_scaling.csv. The two
are different runs, not duplicates: this directory's local CSVs came from a
resource-capped subprocess sweep (WORKER_TIMEOUT_S=150s,
WORKER_MEM_LIMIT_MB=4096 -- see unrefined_scenario_scaling.py) that only
reaches num_scenarios=8 before status flips to 'oom'/'timeout'; the PACE
sweep (run_{unrefined,refined}_scaling.sbatch) sets both caps to `math.inf`
and completes the full num_scenarios=3..21 range, status='ok' throughout.
That is the range this script now plots.

Memory column: peak_rss_mb, NOT memory_kb, and this is a checked choice,
not a default -- memory_kb is CROSS-PLATFORM-INCONSISTENT between the two
sweeps, not just a units difference. Both scripts get their per-call memory
figure from nonlinear_chain_separating_input.py's `_memory_snapshot()`,
which returns GPU device stats (`jax.devices()[0].memory_stats()`) when a
GPU is present and falls back to host `ru_maxrss_kb` only when it is not.
The local sweep hardcodes `JAX_PLATFORMS=cpu` (no GPU -> memory_kb is host
RSS, tracking peak_rss_mb*1024 closely); the PACE sbatch script sets
`JAX_PLATFORMS=cuda` on a V100 node -> memory_kb there is GPU VRAM, a
completely different and far smaller quantity (kilobytes to tens of MB,
vs. peak_rss_mb's 1-14GB host figure) that would silently mean something
different from every "Memory usage" plotted so far in this project if used
here unlabeled. peak_rss_mb comes from run_worker_with_limits' own
subprocess VmRSS-polling watchdog in runtime_scaling_common.py, which is
byte-for-byte identical code in both the local and PACE copies (diffed,
not assumed) -- it is the one memory figure genuinely comparable across
both sweeps and across this project's other host-RAM plots.

Produces exactly 3 PDFs:
  nonlinear_chain_runtime_memory_unrefined.pdf -- unrefined only, 2 curves
    (runtime, memory) on a twin axis.
  nonlinear_chain_runtime_memory_refined.pdf   -- same, refined only.
  nonlinear_chain_runtime_memory_combined.pdf  -- both methods together,
    4 curves.

Every row in both PACE CSVs is status='ok' (num_scenarios 3-21 for both
methods); no truncation marker is needed or drawn.

Color convention, deliberately different between the single-method and
combined figures -- both are read top-to-bottom by axis label color, not by
memorizing a hue across figures, so this is not an inconsistency a reader
has to resolve:
  - Single-method PDFs: color encodes METRIC (runtime vs. memory), since
    method identity is fixed by which PDF you're looking at. Slot 1 (blue)
    = runtime, slot 2 (orange) = memory, matching integrator_chain's
    plot_post_compile_runtime_filtered.py's twin-axis convention exactly.
  - Combined PDF: color encodes METHOD (blue = unrefined, orange = refined,
    matching plot_scenario_runtime_scaling.py's own already-established
    mapping for these two exact methods), and metric becomes a line-style
    split (solid = runtime, dashed = memory) since color is spent on method.

No titles (per request); x-axis forced to integer ticks (num_scenarios is
a count, never fractional); figures sized flat/wide rather than square --
none of these needed the 1:1 aspect a square axes region implies.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from plot_scenario_runtime_scaling import load_rows

_PACE_DATA_DIR = (_HERE.parents[0] / "pace_runtime_scaling_package" / "pace_pack"
                 / "data" / "nonlinear_chain")
UNREFINED_CSV_PATH = _PACE_DATA_DIR / "unrefined_scenario_scaling.csv"
REFINED_CSV_PATH = _PACE_DATA_DIR / "refined_scenario_scaling.csv"

_OUT_UNREFINED = _HERE / "nonlinear_chain_runtime_memory_unrefined.pdf"
_OUT_REFINED = _HERE / "nonlinear_chain_runtime_memory_refined.pdf"
_OUT_COMBINED = _HERE / "nonlinear_chain_runtime_memory_combined.pdf"

# Categorical slots 1/2 of the project's validated palette (light mode);
# validated earlier this session as a 2-slot set: CVD/normal-vision/
# lightness/chroma all PASS.
_COLOR_RUNTIME = "#2a78d6"     # slot 1 (blue)  -- metric role, single-method PDFs
_COLOR_MEMORY = "#eb6834"      # slot 2 (orange) -- metric role, single-method PDFs
_COLOR_UNREFINED = "#2a78d6"   # slot 1 (blue)  -- method role, combined PDF
_COLOR_REFINED = "#eb6834"     # slot 2 (orange) -- method role, combined PDF
_INK = "0.15"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def _ok(rows):
    return [r for r in rows if r["status"] == "ok"]


def _style_axes(ax, xlabel="Number of fault scenarios"):
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color="0.85", linewidth=0.8)
    ax.spines["top"].set_visible(False)


def make_single_method_plot(rows, out_path: Path, method_label: str):
    """One PDF: run time (left, blue) + memory (right, orange) for ONE method."""
    ok_rows = _ok(rows)
    xs = [r["num_scenarios"] for r in ok_rows]
    run_ms = [r["run_time_avg_ms"] for r in ok_rows]
    mem_mb = [float(r["peak_rss_mb"]) for r in ok_rows]

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax_mem = ax.twinx()

    l_mem, = ax_mem.plot(xs, mem_mb, color=_COLOR_MEMORY, linewidth=1.8, linestyle="--",
                         marker="s", markersize=4.5, zorder=2)
    l_run, = ax.plot(xs, run_ms, color=_COLOR_RUNTIME, linewidth=1.8, linestyle="-",
                     marker="o", markersize=4.5, zorder=3)
    ax.set_zorder(ax_mem.get_zorder() + 1)
    ax.patch.set_visible(False)

    _style_axes(ax)
    ax.set_ylabel("Run time (ms)", color=_COLOR_RUNTIME)
    ax_mem.set_ylabel("Memory (MB)", color=_COLOR_MEMORY)
    ax_mem.grid(False)
    ax_mem.spines["top"].set_visible(False)
    ax.spines["left"].set_color(_COLOR_RUNTIME)
    ax.tick_params(axis="y", color=_COLOR_RUNTIME, labelcolor=_INK)
    ax.spines["right"].set_color(_COLOR_MEMORY)
    ax_mem.spines["right"].set_color(_COLOR_MEMORY)
    ax_mem.tick_params(axis="y", color=_COLOR_MEMORY, labelcolor=_INK)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[{method_label}] {len(xs)} points, num_scenarios={xs}")
    print(f"Wrote {out_path}")


def make_combined_plot(unrefined_rows, refined_rows, out_path: Path = _OUT_COMBINED):
    """One PDF: both methods, runtime (solid) + memory (dashed), method=color."""
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax_mem = ax.twinx()

    for rows, color, label in ((unrefined_rows, _COLOR_UNREFINED, "Unrefined"),
                               (refined_rows, _COLOR_REFINED, "Refined")):
        ok_rows = _ok(rows)
        xs = [r["num_scenarios"] for r in ok_rows]
        run_ms = [r["run_time_avg_ms"] for r in ok_rows]
        mem_mb = [float(r["peak_rss_mb"]) for r in ok_rows]
        ax.plot(xs, run_ms, color=color, linewidth=1.8, linestyle="-",
               marker="o", markersize=4.5, zorder=3)
        ax_mem.plot(xs, mem_mb, color=color, linewidth=1.8, linestyle="--",
                   marker="s", markersize=4.5, zorder=2)

    ax.set_zorder(ax_mem.get_zorder() + 1)
    ax.patch.set_visible(False)
    _style_axes(ax)
    ax.set_ylabel("Run time (ms)")
    ax_mem.set_ylabel("Memory (MB)")
    ax_mem.grid(False)
    ax_mem.spines["top"].set_visible(False)

    # Method = color (2 handles), metric = linestyle (2 handles) -- a 2x2
    # legend rather than 4 same-weight entries, since those are the two
    # independent things being encoded.
    handles = [
        Line2D([0], [0], color=_COLOR_UNREFINED, lw=2, label="Unrefined"),
        Line2D([0], [0], color=_COLOR_REFINED, lw=2, label="Refined"),
        Line2D([0], [0], color=_INK, lw=1.8, linestyle="-", marker="o", markersize=4.5,
              label="Run time"),
        Line2D([0], [0], color=_INK, lw=1.8, linestyle="--", marker="s", markersize=4.5,
              label="Memory"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2,
             fontsize=8, labelcolor=_INK, handlelength=2.2, columnspacing=1.2)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    unrefined_rows = load_rows(UNREFINED_CSV_PATH)
    refined_rows = load_rows(REFINED_CSV_PATH)
    print(f"Loaded {len(unrefined_rows)} unrefined rows, {len(refined_rows)} refined rows")

    make_single_method_plot(unrefined_rows, _OUT_UNREFINED, "unrefined")
    make_single_method_plot(refined_rows, _OUT_REFINED, "refined")
    make_combined_plot(unrefined_rows, refined_rows, _OUT_COMBINED)
