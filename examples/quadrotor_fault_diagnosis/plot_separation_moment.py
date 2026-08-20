"""
Plots a single "moment of separation" figure for the 12-state quadrotor
fault-diagnosis module: one PDF, two 3D panels, each showing ONLY the
single segment where all 5 scenarios' RAW (independent per-scenario)
boxes are simultaneously pairwise disjoint with the largest worst-pair
margin -- i.e. the segment where separation is most visually obvious, not
a trajectory smeared across several segments like `solve_and_plot.py`'s
plots.

Per RESULTS.md ("Round 2: fixing the loss..."), segments 0-4 are all
genuinely disjoint for the saved `u_seq`, with segment 2 having the best
worst-pairwise margin (+0.0251). This script recomputes that margin
directly from the saved u_seq rather than hardcoding "segment 2", so it
stays correct if `results/quadrotor_separating_input.npz` is regenerated.

Panel choice -- VELOCITY (vx,vy,vz) and BODY RATES (p,q,r), NOT position
and attitude. An earlier version of this script plotted position/attitude
and looked wrong: ActuatorFault_2/3/4's boxes appeared to still overlap.
They do -- in position AND in attitude. Checking the raw per-dimension
`_signed_separation_margin` at the chosen segment shows every one of the
10 scenario pairs' disjointness is carried by exactly one of {vz, p, q,
r}: the thrust fault (ActuatorFault_1) separates via vz (body velocity,
margin +0.286 -- an order of magnitude clearer than position z's own
margin of +0.004, which is also positive but not the dominant dimension),
and the three moment faults separate from each other and from
Nominal/ActuatorFault_1 via p/q/r (body rates) -- NOT phi/theta/psi
(attitude angles: EVERY pairwise margin among those three is negative,
i.e. genuinely overlapping). This makes physical sense: thrust directly
drives vertical acceleration (velocity), and roll/pitch/yaw moments
directly drive angular acceleration (rates) -- position and attitude are
integrals of these and lag behind. `main()` verifies this assumption
still holds (asserts every pair's argmax-margin dimension is one of
vx,vy,vz,p,q,r) before plotting, so a stale panel choice fails loudly
instead of silently mis-plotting again if the solve changes.

Reuses the saved solve from `solve_and_plot.py` (results/quadrotor_
separating_input.npz) instead of re-running the expensive GD solve.

Usage:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/quadrotor_fault_diagnosis/plot_separation_moment.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax
import jax.numpy as jnp
import immrax as irx

from quadrotor_separating_input import (
    create_scenarios, _propagate_history, _signed_separation_margin, QuadrotorSystem,
)
from solve_and_plot import (
    _draw_box3d, SCENARIO_COLORS, RESULTS_DIR, ALPHA_TRUE_FAULT,
)

NPZ_PATH = RESULTS_DIR / "quadrotor_separating_input.npz"
STATE_NAMES = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']


def euler_point_rollout(x0, u_seq, alpha, dt):
    """Point rollout using the SAME single-big-Euler-step integrator the
    reachable-set box is built from (`euler_step` in quadrotor_separating_
    input.py, one step of size dt per segment) -- NOT the finer RK4
    rollout `solve_and_plot.py` uses for its smoother multi-segment
    trajectory plots. That mismatch (fine RK4 point vs. coarse
    single-Euler-step box) is why an earlier version of this plot showed
    dots landing outside their own box: the box only soundly bounds the
    Euler-discretized map's reachable set, not the true continuous flow, so
    a more accurately integrated point isn't guaranteed to land inside it.
    Using the box's own integrator here instead means the point is exactly
    the box's center-point image, guaranteed inside by construction
    (soundness of the natural embedding), and the plot honestly shows what
    was actually optimized/would be deployed rather than a more-accurate
    but box-inconsistent trajectory."""
    sys_ = QuadrotorSystem()

    def f(x, u):
        return sys_.f(jnp.zeros(()), x, u, alpha)

    xs = [x0]
    x = x0
    for k in range(u_seq.shape[0]):
        x = x + dt * f(x, u_seq[k])
        xs.append(x)
    return jnp.stack(xs)


def main():
    d = np.load(NPZ_PATH)
    u_seq = jnp.array(d["u_seq"])
    dt = float(d["dt"])
    x0_width = float(d["x0_width"])
    alpha_lo, alpha_hi = float(d["alpha_lo"]), float(d["alpha_hi"])
    disjoint_segments = [int(k) for k in d["disjoint_segments"]]
    if not disjoint_segments:
        raise RuntimeError(
            f"{NPZ_PATH} has no genuinely-disjoint segments (disjoint_segments "
            "is empty) -- nothing to plot. Re-run solve_and_plot.py first.")

    scenarios = create_scenarios(alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    x0_ivl = irx.icentpert(jnp.zeros(12), jnp.full(12, x0_width))

    # All scenarios share one emb_system (only p_interval differs) -- vmap
    # over stacked p_intervals instead of a Python loop that separately
    # traces/compiles _propagate_history once per scenario (same fix as
    # run_quadrotor_diagnosis_qps.py's predicted_histories; ~2.7x faster,
    # numerically identical).
    _emb_sys = scenarios[0].emb_system
    _p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    _hist_batch = jax.vmap(lambda p: _propagate_history(x0_ivl, u_seq, _emb_sys, p, dt, 1))(_p_batch)
    hist_by_name = {s.name: irx.Interval(lower=_hist_batch.lower[i], upper=_hist_batch.upper[i])
                    for i, s in enumerate(scenarios)}

    # Pick, among the certified-disjoint segments, the one with the largest
    # worst-pairwise margin -- the most visually obvious moment of separation.
    # Also record, per pair, WHICH dimension carries that pair's margin, so
    # we can pick panels that actually show it (see module docstring).
    best_k, best_margin = None, -float("inf")
    per_pair_dim_at_best = {}
    for k in disjoint_segments:
        worst_margin_k = float("inf")
        pair_dims_k = {}
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                ivl_i = irx.Interval(lower=hist_by_name[scenarios[i].name].lower[k],
                                     upper=hist_by_name[scenarios[i].name].upper[k])
                ivl_j = irx.Interval(lower=hist_by_name[scenarios[j].name].lower[k],
                                     upper=hist_by_name[scenarios[j].name].upper[k])
                margins = np.array(_signed_separation_margin(ivl_i, ivl_j))
                m = float(np.max(margins))
                pair_dims_k[(scenarios[i].name, scenarios[j].name)] = STATE_NAMES[int(np.argmax(margins))]
                worst_margin_k = min(worst_margin_k, m)
        if worst_margin_k > best_margin:
            best_margin = worst_margin_k
            best_k = k
            per_pair_dim_at_best = pair_dims_k
    print(f"Best disjoint segment: {best_k} (t={best_k*dt:.2f}s), "
         f"worst-pairwise margin={best_margin:+.5f}")
    dims_used = sorted(set(per_pair_dim_at_best.values()))
    print(f"Dimensions actually carrying a pairwise separation: {dims_used}")
    for pair, dim in per_pair_dim_at_best.items():
        print(f"  {pair[0]:16s} vs {pair[1]:16s}  -> {dim}")

    # Panel choice: velocity (vx,vy,vz) + body rates (p,q,r) -- see module
    # docstring for why, and the empirical check right here (fails loudly
    # instead of silently mis-plotting if a future re-solve moves the
    # separating dimensions somewhere these two panels don't cover).
    panel_dims = {"vx", "vy", "vz", "p", "q", "r"}
    uncovered = set(dims_used) - panel_dims
    if uncovered:
        raise RuntimeError(
            f"Separating dimension(s) {uncovered} at segment {best_k} are not covered by "
            f"the velocity/body-rate panels ({panel_dims}) -- update this script's panel "
            "choice before trusting the figure.")

    # Point trajectories, one Euler rollout per scenario up to and including
    # segment best_k, for a small trajectory tail giving context to the box.
    # Uses the box's own single-big-Euler-step integrator (euler_point_
    # rollout, NOT solve_and_plot.py's finer RK4 rollout) -- see that
    # function's docstring for why the dot must match the box's
    # integrator to be guaranteed contained in it.
    box_lo_vel, box_hi_vel = {}, {}
    box_lo_rate, box_hi_rate = {}, {}
    point_traj_vel, point_traj_rate = {}, {}
    for s in scenarios:
        hist = hist_by_name[s.name]
        box_lo_vel[s.name] = np.array(hist.lower[best_k])[6:9]
        box_hi_vel[s.name] = np.array(hist.upper[best_k])[6:9]
        box_lo_rate[s.name] = np.array(hist.lower[best_k])[9:12]
        box_hi_rate[s.name] = np.array(hist.upper[best_k])[9:12]
        alpha_point = jnp.ones(4) if s.name == "Nominal" else (
            jnp.ones(4).at[int(s.name[-1]) - 1].set(ALPHA_TRUE_FAULT))
        traj = euler_point_rollout(jnp.zeros(12), u_seq[:best_k + 1], alpha_point, dt)
        point_traj_vel[s.name] = np.array(traj)[:, 6:9]
        point_traj_rate[s.name] = np.array(traj)[:, 9:12]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 11,
    })

    fig = plt.figure(figsize=(12.5, 6.0))
    ax_vel = fig.add_subplot(121, projection="3d")
    ax_rate = fig.add_subplot(122, projection="3d")

    # Several scenarios' boxes nearly coincide in one projection (e.g. the
    # 4 non-thrust scenarios nearly coincide in velocity; Nominal and the
    # thrust fault nearly coincide in body rates) -- draw largest-extent
    # box first so smaller, nested boxes are layered on top and stay
    # visible instead of being hidden inside a bigger wireframe.
    order_vel = sorted(scenarios, key=lambda s: -float(np.sum(box_hi_vel[s.name] - box_lo_vel[s.name])))
    order_rate = sorted(scenarios, key=lambda s: -float(np.sum(box_hi_rate[s.name] - box_lo_rate[s.name])))

    for s in order_vel:
        color = SCENARIO_COLORS[s.name]
        vel = point_traj_vel[s.name]
        ax_vel.plot(vel[:, 0], vel[:, 1], vel[:, 2], color=color, linewidth=1.4,
                   alpha=0.6, zorder=4)
        ax_vel.scatter(*vel[-1], color=color, s=14, zorder=6)
        _draw_box3d(ax_vel, box_lo_vel[s.name], box_hi_vel[s.name], color, alpha=0.9, linewidth=1.8)

    for s in order_rate:
        color = SCENARIO_COLORS[s.name]
        rate = point_traj_rate[s.name]
        ax_rate.plot(rate[:, 0], rate[:, 1], rate[:, 2], color=color, linewidth=1.4,
                    alpha=0.6, zorder=4)
        ax_rate.scatter(*rate[-1], color=color, s=14, zorder=6)
        _draw_box3d(ax_rate, box_lo_rate[s.name], box_hi_rate[s.name], color, alpha=0.9, linewidth=1.8)

    # Shared legend (scenario order, not draw order) via proxy handles so
    # it's independent of either panel's z-order-driven draw sequence.
    legend_handles = [Line2D([0], [0], color=SCENARIO_COLORS[s.name], lw=2, label=s.name)
                      for s in scenarios]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.90, 0.85),
              frameon=False, fontsize=8)

    ax_vel.set_xlabel("$v_x$ (m/s)", labelpad=8)
    ax_vel.set_ylabel("$v_y$ (m/s)", labelpad=8)
    ax_vel.set_zlabel("$v_z$ (m/s)", labelpad=10)
    ax_vel.set_title("Body velocity", fontsize=11)

    ax_rate.set_xlabel("$p$ (rad/s)", labelpad=8)
    ax_rate.set_ylabel("$q$ (rad/s)", labelpad=8)
    ax_rate.set_zlabel("$r$ (rad/s)", labelpad=10)
    ax_rate.set_title("Body rates", fontsize=11)

    fig.suptitle(f"Reachable-set separation at segment {best_k} "
                f"(t={best_k*dt:.2f}s, worst-pairwise margin={best_margin:+.4f})",
                fontsize=11, y=0.98)
    fig.text(0.5, 0.925,
             "Velocity and body rates are plotted (not position/attitude) because every "
             "pairwise separation at this segment is carried by vz, p, q, or r -- see "
             "this script's module docstring for the per-dimension margin check.",
             ha="center", fontsize=8, style="italic")
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.05, top=0.86, wspace=0.25)

    out_pdf = RESULTS_DIR / "quadrotor_separation_moment.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
