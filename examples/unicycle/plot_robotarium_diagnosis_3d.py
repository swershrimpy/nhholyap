"""
3D (px, time, py) illustration of the Robotarium missed-diagnosis study
(robotarium_diagnosis_mc.py), for Response 14 in 26-2796-AR.tex.

Shows the TRUE candidate model's fine-resolution predicted output-interval
"tube" (translucent boxes at each of the 10 segment-boundary times, same
Poly3DCollection convention as animate_refinement_3d.py: axis order
(px, time, py), flat rectangle per time slice) for one config, with a
handful of individual Monte Carlo trial trajectories overlaid:
  - in-bound trials (true params drawn from the assumed intervals) in
    status "good" green -- these should thread through the tube at every
    sample time, and did in 99.7% of the full sweep's in-bound trials.
  - adversarial trials (params pushed just outside the assumed box) in
    status "critical" red -- most of these visibly exit the tube, matching
    the sweep's 92.1% adversarial missed rate.

Only the FINE prediction resolution is used (matches the Robotarium ground
truth's own integration step -- see robotarium_diagnosis_mc.py's docstring
for why the coarse, steps_per_segment=1 resolution is a discretization
artifact, not reported here or in Response 14's summary).

Trajectories are plotted at full Robotarium substep resolution (250 points
over the 5s horizon) for a smooth curve, not just the 10 sample points used
for the containment check -- the containment check itself only evaluates at
the 10 segment boundaries (small circular markers), matching what the
sweep actually tests.

Usage: python plot_robotarium_diagnosis_3d.py [out_pdf]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from robotarium_diagnosis_mc import (
    predicted_y_history, sample_x0, sample_alpha, sample_sensor,
    DT, NUM_STEPS, N_SUB, DT_SUB, MODEL_NAMES,
)
from rps.robotarium import Robotarium

_OUT = _HERE / "robotarium_diagnosis_3d.pdf"

# Config/mode/method illustrated -- config 8 of multistep_refined's
# successful set: x0_center=(0.1,0.1,0), x0_width=0.06 (moderate -- large
# enough for a visible tube, well inside the >=success half-widths),
# alpha in [0,0.5] (wide/easy actuator-fault band). True mode = Actuator
# Fault: the only mode with a genuine interval-valued uncertain parameter
# (Nominal/Sensor Fault both fix alpha=1 as a point value), so its tube
# reflects both x0- and alpha-uncertainty, and it had the best adversarial
# robustness of the three modes in the full sweep (84.1% missed vs. 96%).
METHOD = "multistep_refined"
CONFIG_IDX = 8
TRUE_MODE = "Actuator Fault"
N_CORRECT = 5
N_ADV = 5
SEED = 7

_COLOR_TUBE = "#2a78d6"     # categorical slot 1 (blue) -- the prediction itself
_COLOR_GOOD = "#0ca30c"     # status: good -- in-bound trial
_COLOR_CRIT = "#d03b3b"     # status: critical -- adversarial trial
_INK = "0.15"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def run_trial_full(u_seq, x0_true, alpha_true):
    """Single-trial (B=1) Robotarium run, recording EVERY substep's [px,py]
    pose for a smooth trajectory -- unlike robotarium_diagnosis_mc.py's
    run_robotarium_batch, which only records the 10 segment-boundary poses
    the containment check actually needs."""
    r = Robotarium(number_of_robots=1, show_figure=False, sim_in_real_time=False,
                   initial_conditions=x0_true.T.copy())
    r.max_wheel_velocity = 1e6
    r.time_step = DT_SUB
    times, pos = [0.0], [r.poses[:2, 0].copy()]
    for seg in range(NUM_STEPS):
        v_cmd = np.array([float(u_seq[seg, 0])])
        omega_cmd = np.array([float(u_seq[seg, 1])]) * alpha_true
        for sub in range(N_SUB):
            r.get_poses()
            r.set_velocities(np.array([0]), np.vstack([v_cmd, omega_cmd]))
            r.step()
            times.append(seg * DT + (sub + 1) * DT_SUB)
            pos.append(r.poses[:2, 0].copy())
    plt.close(r.figure)
    return np.array(times), np.array(pos)   # (T,), (T,2)


def make_plot(out_pdf: Path = _OUT):
    d = np.load(_HERE / f"success_rate_data_early_stop_{METHOD}.npz", allow_pickle=True)
    assert bool(d["config_success"][CONFIG_IDX]), "chosen config must have succeeded"
    u_seq = d["u_opt"][CONFIG_IDX]
    x0_center = d["x0_center"][CONFIG_IDX]
    x0_width = float(d["x0_width"][CONFIG_IDX])
    alpha_lo = float(d["alpha_lo"][CONFIG_IDX])
    alpha_hi = float(d["alpha_hi"][CONFIG_IDX])
    obs_offset = d["sensor_offset"][CONFIG_IDX]
    obs_scale = float(d["sensor_scale"][CONFIG_IDX])

    fine, _coarse = predicted_y_history(u_seq, x0_center, x0_width,
                                        alpha_lo, alpha_hi, obs_offset, obs_scale)
    lo, hi = fine[TRUE_MODE]   # each (NUM_STEPS, 2) -- [px,py] bounds per segment boundary
    seg_times = np.arange(1, NUM_STEPS + 1) * DT

    rng = np.random.default_rng(SEED)
    trials = []   # (times, pos, label, contained_at_end)
    for adversarial, n, label in ((False, N_CORRECT, "in-bound"), (True, N_ADV, "adversarial")):
        for _ in range(n):
            x0_true = sample_x0(rng, x0_center, x0_width, 1, adversarial)
            alpha_true = sample_alpha(rng, TRUE_MODE, alpha_lo, alpha_hi, 1, adversarial)
            times, pos = run_trial_full(u_seq, x0_true, alpha_true)
            seg_pos = pos[N_SUB::N_SUB]   # positions at the 10 segment boundaries
            contained = np.all((seg_pos >= lo) & (seg_pos <= hi), axis=-1)
            first_exit = int(np.argmin(contained)) if not contained.all() else None
            trials.append((times, pos, label, first_exit))

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # ── Predicted tube: translucent box per segment boundary, axis order
    #    (px, time, py) -- matches animate_refinement_3d.py's convention.
    for k in range(NUM_STEPS):
        t_k = seg_times[k]
        xl, yl = lo[k]; xh, yh = hi[k]
        verts = [[(xl, t_k, yl), (xh, t_k, yl), (xh, t_k, yh), (xl, t_k, yh)]]
        poly = Poly3DCollection(verts, alpha=0.22)
        poly.set_facecolor(_COLOR_TUBE)
        poly.set_edgecolor((*mcolors.to_rgb(_COLOR_TUBE), 0.7))
        ax.add_collection3d(poly)
        ax.plot([xl, xh, xh, xl, xl], [t_k] * 5, [yl, yl, yh, yh, yl],
               color=_COLOR_TUBE, linewidth=1.0, alpha=0.7, zorder=3)

    # ── Trial trajectories ------------------------------------------------
    # Color encodes which CONDITION the trial was drawn from (in-bound vs.
    # adversarial), not the outcome -- the outcome (contained vs. excluded)
    # is shown separately by truncation + the 'x' marker below, so a rare
    # in-bound trial that happens to be excluded (0.3% of the full sweep)
    # is still drawn green (correctly labelled by its CONDITION) with its
    # own exit marked, rather than mislabelled by conflating the two.
    for times, pos, label, first_exit in trials:
        color = _COLOR_GOOD if label == "in-bound" else _COLOR_CRIT
        # Truncate a trajectory shortly after it first leaves the tube, so
        # the exit reads clearly instead of a long post-exclusion tail
        # cluttering the plot.
        if first_exit is not None:
            cutoff_t = seg_times[min(first_exit + 2, NUM_STEPS - 1)]
            keep = times <= cutoff_t
        else:
            keep = np.ones_like(times, dtype=bool)
        ax.plot(pos[keep, 0], times[keep], pos[keep, 1], color=color,
               linewidth=1.6, alpha=0.9, zorder=5)
        seg_idx = np.arange(NUM_STEPS)
        seg_keep = seg_times[seg_idx] <= (times[keep][-1] if keep.any() else 0)
        ax.scatter(pos[N_SUB::N_SUB][seg_keep, 0], seg_times[seg_keep],
                  pos[N_SUB::N_SUB][seg_keep, 1], color=color, s=14, zorder=6)
        if first_exit is not None:
            ax.scatter([pos[N_SUB::N_SUB][first_exit, 0]], [seg_times[first_exit]],
                      [pos[N_SUB::N_SUB][first_exit, 1]], color=_COLOR_CRIT,
                      s=55, marker="x", linewidths=2.2, zorder=7)

    ax.set_xlabel("$p_x$ (m)")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("$p_y$ (m)")
    ax.set_title(f"Predicted reachable output tube vs. Robotarium ground truth "
                f"(true mode: {TRUE_MODE}, {METHOD.replace('_', ' ')})")
    ax.view_init(elev=18, azim=-60)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=_COLOR_TUBE, lw=6, alpha=0.35, label="Predicted output interval (tube)"),
        Line2D([0], [0], color=_COLOR_GOOD, lw=2, label="In-bound trial (stays inside)"),
        Line2D([0], [0], color=_COLOR_CRIT, lw=2, label="Adversarial trial"),
        Line2D([0], [0], color=_COLOR_CRIT, lw=0, marker="x", markersize=8,
              markeredgewidth=2.2, label="First step outside the tube"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, labelcolor=_INK,
             bbox_to_anchor=(0.02, 0.98))

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _OUT
    p = make_plot(out)
    print(f"Wrote {p}")
