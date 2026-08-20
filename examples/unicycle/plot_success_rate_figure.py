"""
Success-rate figure for the unicycle (unicycle) repeated-restart
study, for the reviewer response on statistical evaluation (Response 14 in
26-2796-AR.tex).

Reads the two npz files written by success_rate_analysis_early_stop.py for
the multistep methods -- unrefined and refined, each holding 84 configs
(3 x0 centers x 14 initial-set half-widths 0.02..0.15 x 2 actuator
alpha-ranges) x 100 random GD restarts with gd_early_stop
(car_separating_input.py) -- and plots scenario-level success rate against
initial-set half-width, one line per method.

The single-step controller's data (success_rate_data_early_stop_single_step.npz)
is NOT read here and does not appear in the figure -- Response 14 in
26-2796-AR.tex compares only the proposed method against the unrefined
multistep ablation.

Only the gd_early_stop data (success_rate_data_early_stop_*.npz) is used;
the fixed-iteration jax.lax.scan sweep (success_rate_analysis.py's
success_rate_data_*.npz, no "early_stop" infix) is NOT read here. The
optimizer used for every number in this figure and in Response 14 is
gd_early_stop -- the response reports one optimizer's results, not a
comparison between two.

Scenario-level means a configuration counts as solved when at least one of
its 100 restarts drives the objective to zero -- multistart is part of the
method, so the per-restart hit rate is an internal detail, not a reported
capability. The restart-level fraction is deliberately NOT plotted.

Success = final separation loss below SUCCESS_THRESHOLD (1e-6), i.e. all
three model pairs reach disjoint observed-output intervals, at DT=0.5 over
10 steps (a 5s horizon -- the same horizon as the manuscript's unicycle
experiment). There are no NaN restarts anywhere in this study.

Usage: python plot_success_rate_figure.py [out_pdf]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "unicycle_success_rate.pdf"
_PREFIX = "early_stop_"   # the only variant this figure reports

METHODS = [
    ("multistep_unrefined", "Multistep, no refinement", "s", "--"),
    ("multistep_refined", "Multistep + refinement (ours)", "D", "-"),
]
# Categorical slots 1/2 of the project's validated palette (light mode).
# Marker shape and dash pattern are the non-color encoding.
COLORS = ["#eb6834", "#1baf7a"]
_INK = "0.15"

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


def load(method):
    d = np.load(_HERE / f"success_rate_data_{_PREFIX}{method}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def by_width(d):
    """Scenario-level success rate (%) per initial-set half-width.

    A scenario (one configuration) counts as solved if ANY of its 100
    restarts drove the objective below the threshold -- i.e. config_success,
    not the fraction of restarts that succeeded. Six scenarios share each
    half-width (3 initial-state centers x 2 actuator alpha-ranges), so each
    plotted point is out of 6.
    """
    w = np.round(d["x0_width"], 3)
    widths = np.array(sorted(set(w)))
    return widths, np.array([d["config_success"][w == x].mean() * 100
                             for x in widths])


def make_figure(out_pdf: Path = _OUT):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for (mkey, mlabel, marker, ls), color in zip(METHODS, COLORS):
        d = load(mkey)
        widths, vals = by_width(d)
        ax.plot(widths, vals, color=color, linewidth=1.7, linestyle=ls,
                marker=marker, markersize=4.5, label=mlabel)
    ax.set_xlabel("Initial-set half-width")
    ax.set_ylabel("Scenario-level success rate (%)")
    ax.grid(True, color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-3, 103)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", labelcolor=_INK)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _OUT
    print(f"{'method':30s}{'scenarios':>14}{'widest ok':>10}{'NaN':>6}")
    for mkey, mlabel, _, _ in METHODS:
        d = load(mkey)
        cs = d["config_success"]
        nan = int(np.isnan(d["losses_final_all_restarts"]).sum())
        widths, vals = by_width(d)
        last = widths[np.nonzero(vals)[0][-1]] if vals.any() else float("nan")
        print(f"{mlabel:30s}"
              f"{f'{int(cs.sum())}/{cs.size} ({cs.mean()*100:.1f}%)':>14}"
              f"{last:>10.2f}{nan:>6d}")
    print(f"\nWrote {make_figure(out)}")
