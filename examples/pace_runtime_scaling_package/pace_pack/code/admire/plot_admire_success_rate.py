"""
Success-rate figure for the ADMIRE repeated-restart sweep, built for the
reviewer response on statistical evaluation (Response 14 in 26-2796-AR.tex).

Reads the 9 npz files written by admire_success_rate_analysis.py --
3 methods x 3 horizons, each holding 9 configs (3 input_limit x 3 x0_width)
x 20 random GD restarts -- and renders two panels:

  (a) restart-level success rate vs. horizon, one bar group per method.
      Success = final separation loss below SUCCESS_THRESHOLD (1e-6), i.e.
      the optimizer found an input that drives all 11 ADMIRE fault models to
      pairwise-disjoint output intervals.
  (b) restart-level success rate vs. initial-set half-width x0_width for the
      refinement method, one bar group per horizon. This is the panel that
      shows WHAT governs success: x0_width dominates, and input_limit
      (averaged over here) barely matters.

NaN handling -- read before quoting panel (a)'s 2.0s numbers
------------------------------------------------------------
At the 2.0s horizon only, some restarts finish with a NaN loss (the
jnp.where-unselected-branch autodiff gotcha that refinement_demo_stable.py
fixes for integrator_chain; it is NOT fixed in admire_success_rate_analysis.py).
It is very unevenly distributed across methods:
    single_step          103/180 restarts NaN (57.2%)
    multistep_unrefined   41/180 (22.8%)
    multistep_refined      2/180 ( 1.1%)
A NaN restart counts as a failure, so this depresses the BASELINES far more
than the proposed method, i.e. it flatters our own result. The 0.5s and 1.0s
horizons have zero NaN in every method and are the clean comparison. This
script marks any (method, horizon) cell with >5% NaN restarts by hatching the
bar and annotating it, so the figure cannot be read without seeing the caveat.

Usage: python plot_admire_success_rate.py [out_pdf]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parents[1] / "data" / "admire"
_OUT = _HERE.parents[1] / "plots" / "admire_success_rate.pdf"

METHODS = [
    ("single_step", "Single-step"),
    ("multistep_unrefined", "Multistep, no refinement"),
    ("multistep_refined", "Multistep + refinement (ours)"),
]
HORIZONS = ["0.5", "1.0", "2.0"]
# Categorical slots 1/2/3 of the project's validated palette (light mode).
# Validated as a 3-slot set: CVD/normal-vision/lightness/chroma all PASS;
# slot 3 (aqua) trips the 3:1 contrast WARN, whose stated relief is visible
# labels or an accompanying table -- both are present (bar value labels here,
# and the same numbers as a table in the response text).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
_INK = "0.15"
_NAN_FLAG = 0.05   # hatch+annotate any cell with more than this NaN fraction

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def load(method, horizon):
    d = np.load(_DATA / f"admire_success_rate_data_{method}_{horizon}s.npz",
                allow_pickle=True)
    return {k: d[k] for k in d.files}


def _style(ax):
    ax.grid(True, axis="y", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, 110)   # headroom so the value label on a 100% bar clears the title
    ax.set_ylabel("Restart-level success rate (%)")


def make_figure(out_pdf: Path = _OUT):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    width = 0.26

    # ---- panel (a): success vs horizon, by method -------------------------
    x = np.arange(len(HORIZONS))
    for mi, (mkey, mlabel) in enumerate(METHODS):
        vals, nan_frac = [], []
        for h in HORIZONS:
            d = load(mkey, h)
            vals.append(d["restart_success_rate"].mean() * 100)
            nan_frac.append(float(np.isnan(d["losses_final_all_restarts"]).mean()))
        pos = x + (mi - 1) * width
        bars = ax_a.bar(pos, vals, width * 0.92, color=COLORS[mi], label=mlabel,
                        edgecolor="white", linewidth=0.6)
        for b, v, nf in zip(bars, vals, nan_frac):
            if nf > _NAN_FLAG:
                b.set_hatch("///")
                b.set_edgecolor("white")
            ax_a.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}",
                      ha="center", va="bottom", fontsize=8, color=_INK)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{h} s\n({int(float(h)/0.1)} steps)" for h in HORIZONS])
    ax_a.set_xlabel("Planning horizon")
    ax_a.set_title("(a) Success rate vs. horizon")
    _style(ax_a)
    ax_a.legend(frameon=False, loc="upper left", labelcolor=_INK)
    ax_a.text(0.99, 0.97, "hatched: >5% of restarts NaN",
              transform=ax_a.transAxes, ha="right", va="top",
              fontsize=7.5, color="0.4")

    # ---- panel (b): success vs x0 width, refinement method ----------------
    widths = sorted({float(w) for w in load("multistep_refined", "1.0")["x0_width"]})
    xb = np.arange(len(widths))
    for hi, h in enumerate(HORIZONS):
        d = load("multistep_refined", h)
        vals = [d["restart_success_rate"][np.isclose(d["x0_width"], w)].mean() * 100
                for w in widths]
        pos = xb + (hi - 1) * width
        bars = ax_b.bar(pos, vals, width * 0.92, color=COLORS[hi],
                        label=f"horizon {h} s", edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax_b.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}",
                      ha="center", va="bottom", fontsize=8, color=_INK)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([f"{w:g}" for w in widths])
    ax_b.set_xlabel("Initial-set half-width of $x_0$")
    ax_b.set_title("(b) Ours: what governs success")
    _style(ax_b)
    ax_b.legend(frameon=False, loc="center right", labelcolor=_INK)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _OUT
    print(f"{'method':24s}{'horiz':>7}{'cfg':>7}{'restart%':>10}{'NaN%':>8}")
    for mkey, mlabel in METHODS:
        for h in HORIZONS:
            d = load(mkey, h)
            cfg = f"{int(d['config_success'].sum())}/9"
            restart_pct = d['restart_success_rate'].mean() * 100
            nan_pct = np.isnan(d['losses_final_all_restarts']).mean() * 100
            print(f"{mkey:24s}{h:>7}{cfg:>7}{restart_pct:>10.1f}{nan_pct:>8.1f}")
    print(f"\nWrote {make_figure(out)}")
