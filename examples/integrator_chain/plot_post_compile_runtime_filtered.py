"""
Post-compilation-runtime-only companion to plot_refinement_runtime_scaling.py,
for the same integrator_refinement_runtime.csv (N=2..100 sweep, refined
multistart optimizer, see that script's docstring for provenance: parsed from
refinement_demo_stable.py's Slurm log, one single-shot timing per N, no
repeat-averaging).

Because each N contributes exactly one un-repeated timing call, the raw
run_time_ms series is contaminated by isolated scheduler/GC-pause spikes on
top of a genuine, much smaller, gradual increase with N -- e.g. N=45..48,51,52
all sit at 18-21ms, but N=49 alone reads 416.5ms (22x its neighbors) before
returning to the same 18-21ms band. Binning the raw data confirms a real
trend under the noise (median run time by N-decile: 3.4 -> 7.5 -> 19.6 ->
23.1 -> 24.3ms) alongside spikes reaching up to 20x a point's local
neighborhood.

Outlier removal: iterative local-median filter. For each point, compare it
to the median of the OTHER non-outlier points within +-5 in N; flag it if it
exceeds max(2x that local median, that local median + 10ms) -- the relative
threshold does the work in the well-sampled small-N region, the +10ms floor
prevents tiny-N points (running at a few ms) from being flagged by ordinary
jitter. Re-run to convergence (2 passes here) so that outliers clustered next
to each other -- which inflate a single-pass local median enough to hide a
neighboring spike -- still get caught once their neighbors are already
excluded. An initial pass at 3x/+15ms left two visible spikes on the plot
(N=54 at 48.7ms, N=86 at 52.8ms, both ~2.5x their surrounding ~19-25ms band)
that were below that threshold but still clearly outliers by eye; tightening
to 2x/+10ms catches both without pulling in any new false positives from the
smooth low-N region -- checked directly against the full flagged list, not
just these two points, before adopting it. Converges in 2 passes, flags
17/99 points.

Excluded points are NOT plotted (per request) -- there is no fallback
imputation, they are simply dropped from the line/markers.

Memory series (right-hand axis)
-------------------------------
The CSV's peak_gpu_mem_bytes column (added by plot_refinement_runtime_scaling.py
from the same log's "Memory snapshot" lines -- see that script's docstring for
why each order's value is that order's own peak, not just a running bound) is
overlaid on a twin right-hand axis in categorical slot 2 (orange), against
run time in slot 1 (blue) on the left.

Two caveats, stated because a twin-axis chart is the one form the project's
dataviz guidance rules out by default:
  - The two y-scales are independent, so where the curves cross or how steeply
    one rises relative to the other means NOTHING. Only each curve's own shape
    against its own axis is readable. A two-panel figure has no such hazard --
    make_plot(..., twin_axis=False) renders that version instead.
  - The same outlier filter is applied to BOTH series, but the memory series is
    strictly increasing and perfectly smooth (0.148 MB at N=2 -> 148.6 MB at
    N=100, quadratic in N), so it flags nothing -- all 99 memory points survive.
    The filter is run over it anyway rather than assumed clean, and __main__
    prints what it excluded from each series.

Both series plot every point they have, connected by lines, with no marker
subsampling -- see the make_plot comment on the memory marker style.

Why the memory series is smooth (it is measured, not fitted)
------------------------------------------------------------
It looks far cleaner than the run-time series because it is a different KIND
of measurement, not because it was modelled. Run time is a wall-clock sample
of one un-repeated call, so it carries scheduler/GC noise; peak_bytes_in_use
is an allocator high-water mark for a workload whose buffer shapes are fixed
by N and the compiled program, so there is almost nothing to be noisy about.
Three properties of the raw bytes confirm it is a real allocator trace rather
than an analytic curve (re-checkable from the CSV at any time):
  - all 99 values are exact multiples of 256 -- XLA's allocation granularity;
  - a least-squares quadratic 14905.7*N^2 - 4935.1*N + 74350.3 does NOT fit
    exactly: residuals reach 53,797 B, so the points are not on a smooth curve;
  - the second differences are irregular (89 distinct values, ranging -119,296
    to +202,496 B), i.e. real per-order padding/layout jitter -- a generated
    quadratic would have a single constant second difference.
"""
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_DATA_CSV = _HERE.parents[1] / "data" / "integrator_chain" / "integrator_refinement_runtime.csv"
_OUT_PDF = _HERE.parents[1] / "plots" / "integrator_refinement_postcompile_runtime_filtered.pdf"
_OUT_PDF_3PANEL = _HERE.parents[1] / "plots" / "integrator_refinement_postcompile_runtime_filtered_3panel.pdf"

# Times-metric serif: Liberation Serif is a genuine TrueType face and embeds
# cleanly under fonttype 42 (a literal "Times New Roman" entry is CFF/OpenType
# on this machine via Nimbus Roman and triggers a font-type-mismatch warning
# in PDF readers -- see the adaptive_spoofing rebuttal-figure fix for the same
# issue). fonttype 42 embeds real vector glyphs rather than rasterising.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Categorical slots 1/2/3 from the project's validated palette (light mode),
# assigned in fixed order. Validated as a 3-slot categorical palette against
# the light surface via the dataviz skill's validate_palette check (worst
# adjacent CVD dE 9.2 deutan / 32.7 tritan, normal-vision floor dE 27.6 --
# both comfortably above the 8.0/15.0 targets). Contrast vs. surface is a
# WARN (relief band) for the aqua host-RAM slot alone (2.74:1, below the
# 3:1 floor): mitigated the same way as slot 2 already is here -- axis/tick
# LABEL text stays in ink (_INK), never the series color, so the low-contrast
# mark itself is the only thing that relies on the color, backed by the
# legend/spine for identity.
_COLOR = "#2a78d6"        # slot 1 (blue)   -- run time (left axis)
_COLOR_MEM = "#eb6834"    # slot 2 (orange) -- peak GPU memory (right axis)
_COLOR_HOST = "#1baf7a"   # slot 3 (aqua)   -- peak host RSS (own panel only)
# Tick labels stay in dark ink rather than taking the series color: orange on
# a light surface is 3.4:1, fine for a mark but under the 4.5:1 text floor.
# Axis-label text and spine carry the color binding instead, backed by the legend.
_INK = "0.15"
_HALF_WIN = 5
_REL_FACTOR = 2.0
_ABS_FLOOR_MS = 10.0
_MAX_PASSES = 10


def load_rows(csv_path: Path = _DATA_CSV):
    """Returns (Ns, run_time_ms, peak_mem_mb, peak_host_gib).

    peak_mem_mb / peak_host_gib are None if the CSV predates that column, or
    a list with None at any order whose snapshot lacked that key -- both are
    handled by the caller (column absent -> None) rather than crashing here.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    has_mem = bool(rows) and "peak_gpu_mem_bytes" in rows[0]
    has_host = bool(rows) and "peak_host_rss_bytes" in rows[0]

    def _mem(r):
        raw = (r.get("peak_gpu_mem_bytes") or "").strip()
        return float(raw) / 1e6 if raw else None

    def _host(r):
        raw = (r.get("peak_host_rss_bytes") or "").strip()
        return float(raw) / (1024.0 ** 3) if raw else None

    data = sorted(
        (int(r["N"]), float(r["run_time_ms"]),
         _mem(r) if has_mem else None, _host(r) if has_host else None)
        for r in rows
    )
    Ns = [n for n, _, _, _ in data]
    run_ms = [v for _, v, _, _ in data]
    mem_mb = [m for _, _, m, _ in data]
    host_gib = [h for _, _, _, h in data]
    if not has_mem or all(m is None for m in mem_mb):
        mem_mb = None
    if not has_host or all(h is None for h in host_gib):
        host_gib = None
    return Ns, run_ms, mem_mb, host_gib


def filter_outliers(Ns, vals, half_win=_HALF_WIN, rel_factor=_REL_FACTOR,
                    abs_floor=_ABS_FLOOR_MS, max_passes=_MAX_PASSES):
    """Iterative local-median outlier filter, see module docstring.
    Returns a boolean list, True where the point is an outlier.

    `abs_floor` is in the series' own units (ms for run time, MB for memory).
    A None entry in `vals` is a missing measurement, not an outlier by
    magnitude, but it is reported as excluded so it never reaches the plot.
    """
    n = len(vals)
    missing = [v is None for v in vals]
    is_outlier = list(missing)

    def local_median(i):
        lo, hi = max(0, i - half_win), min(n, i + half_win + 1)
        window = [vals[j] for j in range(lo, hi)
                  if j != i and not is_outlier[j] and not missing[j]]
        if len(window) < 3:
            window = [vals[j] for j in range(lo, hi) if j != i and not missing[j]]
        return st.median(window) if window else vals[i]

    for _ in range(max_passes):
        changed = False
        for i in range(n):
            if missing[i]:
                continue
            med = local_median(i)
            flag = vals[i] > max(rel_factor * med, med + abs_floor)
            if flag != is_outlier[i]:
                changed = True
            is_outlier[i] = flag
        if not changed:
            break
    return is_outlier


def _keep(Ns, vals, is_outlier):
    """(Ns, vals) with flagged/missing points dropped -- no imputation."""
    return (
        [n for n, o in zip(Ns, is_outlier) if not o],
        [v for v, o in zip(vals, is_outlier) if not o],
    )


_RUN_LABEL = "Post-compilation run time (ms)"
_MEM_LABEL = "Peak GPU memory (MB)"
_HOST_LABEL = "Peak host RSS (GiB)"


def make_plot(Ns, vals, is_outlier, out_path: Path = _OUT_PDF,
              mem_mb=None, mem_is_outlier=None, twin_axis=True,
              host_gib=None, host_is_outlier=None):
    """Run time vs. N, optionally with peak GPU memory (and, off the twin
    axis, peak host RSS) overlaid.

    twin_axis=True  -> memory on a right-hand y-axis of the same panel (the
                       requested form; see the module docstring's caveat about
                       the two scales being independent). Two series only --
                       host RAM is NOT added here; ~160x apart from GPU
                       memory in magnitude, a third curve on this axis would
                       flatten the GPU series into the baseline.
    twin_axis=False -> side-by-side panels, no shared/twin scale: two panels
                       (run time, GPU memory) normally, three (run time, GPU
                       memory, host RAM) when host_gib is also supplied.
    """
    kept_N, kept_v = _keep(Ns, vals, is_outlier)
    have_mem = mem_mb is not None and mem_is_outlier is not None
    if have_mem:
        mem_N, mem_v = _keep(Ns, mem_mb, mem_is_outlier)
    have_host = host_gib is not None and host_is_outlier is not None
    if have_host:
        host_N, host_v = _keep(Ns, host_gib, host_is_outlier)

    def style(ax, xlabel=True):
        if xlabel:
            ax.set_xlabel("Integrator-chain order $N$")
        ax.grid(True, color="0.85", linewidth=0.8)
        ax.spines["top"].set_visible(False)

    if not have_mem:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(kept_N, kept_v, color=_COLOR, linewidth=1.6,
                marker="o", markersize=4.5, linestyle="-")
        ax.set_ylabel(_RUN_LABEL)
        style(ax)
        ax.spines["right"].set_visible(False)
    elif not twin_axis:
        n_panels = 3 if have_host else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 4.5))
        ax, ax_mem = axes[0], axes[1]
        ax.plot(kept_N, kept_v, color=_COLOR, linewidth=1.6,
                marker="o", markersize=4.5)
        ax.set_ylabel(_RUN_LABEL)
        ax_mem.plot(mem_N, mem_v, color=_COLOR_MEM, linewidth=1.6,
                    marker="o", markersize=4.5)
        ax_mem.set_ylabel(_MEM_LABEL)
        panel_axes = [ax, ax_mem]
        if have_host:
            ax_host = axes[2]
            ax_host.plot(host_N, host_v, color=_COLOR_HOST, linewidth=1.6,
                         marker="o", markersize=4.5)
            ax_host.set_ylabel(_HOST_LABEL)
            panel_axes.append(ax_host)
        for a in panel_axes:
            style(a)
            a.spines["right"].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax_mem = ax.twinx()

        # Memory drawn first so the run-time series (the primary measure, and
        # the noisier one) sits on top where its markers stay legible.
        # EVERY measured point is drawn, connected by lines -- no markevery
        # subsampling. A sparse-marker version of this series reads as a
        # drawn/fitted curve rather than as data, which is exactly the wrong
        # impression: the underlying values ARE genuine per-order allocator
        # measurements (see module docstring). Small OPEN squares at
        # markersize 3 keep 99 marks on a ~6.5in axis distinguishable instead
        # of merging into a solid band, which is what forced the subsampling
        # when the markers were large and filled. Square vs. the run-time
        # series' circle is the non-color encoding.
        l_mem, = ax_mem.plot(mem_N, mem_v, color=_COLOR_MEM, linewidth=1.4,
                             linestyle="-", marker="s", markersize=3.0,
                             markerfacecolor="white", markeredgewidth=0.9,
                             label=_MEM_LABEL, zorder=2)
        l_run, = ax.plot(kept_N, kept_v, color=_COLOR, linewidth=1.6,
                         marker="o", markersize=4.5, linestyle="-",
                         label=_RUN_LABEL, zorder=3)
        ax.set_zorder(ax_mem.get_zorder() + 1)   # keep ax's marks above ax_mem's
        ax.patch.set_visible(False)              # ...without hiding them behind it

        style(ax)
        ax.set_ylabel(_RUN_LABEL, color=_COLOR)
        ax_mem.set_ylabel(_MEM_LABEL, color=_COLOR_MEM)
        # Only the left grid is drawn -- two overlaid grids on independent
        # scales read as a broken lattice.
        ax_mem.grid(False)
        ax_mem.spines["top"].set_visible(False)
        # Each y-spine + its tick marks take that series' color, so the axis
        # each curve belongs to is unambiguous. Tick LABELS stay in ink (see
        # _INK above): the legend plus the colored spine carry identity, which
        # is the secondary encoding the twin axis needs.
        ax.spines["left"].set_color(_COLOR)
        ax.tick_params(axis="y", color=_COLOR, labelcolor=_INK)
        for a in (ax, ax_mem):
            a.spines["right"].set_color(_COLOR_MEM)
        ax_mem.tick_params(axis="y", color=_COLOR_MEM, labelcolor=_INK)

        ax.legend(handles=[l_run, l_mem], frameon=False, loc="upper left",
                  fontsize=10, labelcolor=_INK, handlelength=2.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


_ABS_FLOOR_MB = 10.0    # same 1/15-of-range scale as _ABS_FLOOR_MS is to run time
# Provisional pending the host-RAM re-run: 0.5 GiB is a fraction of the
# ~23.84 GiB job-level peak observed in the pre-instrumentation log, in the
# same spirit as the other two floors (an absolute jitter allowance so
# ordinary noise near the low end of the range isn't flagged). Recalibrate
# against the actual per-order range once real host_peak_rss_bytes data
# exists, the same way _ABS_FLOOR_MB was picked from the observed MB range.
_ABS_FLOOR_GIB = 0.5


if __name__ == "__main__":
    Ns, vals, mem_mb, host_gib = load_rows()
    print(f"Loaded {len(Ns)} rows from {_DATA_CSV}")

    is_outlier = filter_outliers(Ns, vals)
    excluded = [(n, v) for n, v, o in zip(Ns, vals, is_outlier) if o]
    print(f"Run time: excluded {len(excluded)} outlier point(s):")
    for n, v in excluded:
        print(f"  N={n:3d}  run_time={v:7.2f} ms")

    mem_is_outlier = None
    if mem_mb is None:
        print("No peak_gpu_mem_bytes column in the CSV -- plotting run time only. "
              "Re-run plot_refinement_runtime_scaling.py on the Slurm log to add it.")
    else:
        mem_is_outlier = filter_outliers(Ns, mem_mb, abs_floor=_ABS_FLOOR_MB)
        mem_excluded = [(n, v) for n, v, o in zip(Ns, mem_mb, mem_is_outlier) if o]
        print(f"Peak GPU memory: excluded {len(mem_excluded)} outlier point(s):")
        for n, v in mem_excluded:
            print(f"  N={n:3d}  peak_mem={v:8.3f} MB")

    host_is_outlier = None
    if host_gib is None:
        print("No peak_host_rss_bytes column in the CSV -- no host RAM panel. "
              "Re-run plot_refinement_runtime_scaling.py on a log with the host-RAM "
              "instrumentation to add it.")
    else:
        host_is_outlier = filter_outliers(Ns, host_gib, abs_floor=_ABS_FLOOR_GIB)
        host_excluded = [(n, v) for n, v, o in zip(Ns, host_gib, host_is_outlier) if o]
        print(f"Peak host RSS: excluded {len(host_excluded)} outlier point(s):")
        for n, v in host_excluded:
            print(f"  N={n:3d}  peak_host_rss={v:8.3f} GiB")

    out = make_plot(Ns, vals, is_outlier, mem_mb=mem_mb, mem_is_outlier=mem_is_outlier)
    print(f"Wrote plot: {out}")

    if host_gib is not None:
        out_3panel = make_plot(Ns, vals, is_outlier, mem_mb=mem_mb, mem_is_outlier=mem_is_outlier,
                               host_gib=host_gib, host_is_outlier=host_is_outlier,
                               twin_axis=False, out_path=_OUT_PDF_3PANEL)
        print(f"Wrote 3-panel plot: {out_3panel}")
