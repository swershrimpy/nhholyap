"""
Solves ONE refined separating input for the 12-state quadrotor's default
5-scenario set (Nominal + 4 ActuatorFault_i), now parameterized with REAL
Crazyflie physical constants (quadrotor_separating_input.py's _M/_IXX/_IYY/
_IZZ, updated from generic-literature defaults -- see that module's
comment for the QPS-sourced values and derivation), and plots the result.

Optimization layer: the MULTISTEP-UNREFINED layer (`separation_loss_multistep`
/ `optimize_multistep_gpu_rejit`), NOT the refined layer this script
originally used. This matters, not just a naming change: the refined
layer's loss operates on PAIRWISE-INTERSECTED boxes (each scenario pair
gets its own tightened, mutually-refined box whenever they overlap), which
is a genuinely different object from the RAW, independent per-scenario
boxes that both this script's own plots and run_quadrotor_diagnosis_qps.py
actually use for containment checks. An earlier attempt at this fix
switched only the refined layer's cost function and got loss=0.0 at
num_steps=3 -- but checking the RAW boxes the plots/diagnosis actually use
showed NO step was genuinely disjoint (margin stayed ~-0.02 throughout):
the refined optimizer's "disjoint at some step" was true only of the
mutually-refined boxes, which nothing downstream of this script consumes.
Switching to the multistep-unrefined layer makes the optimized quantity
and the deployed/plotted quantity the SAME object, so a loss of 0 here
means what it says.

Horizon: num_segments=20, dt=0.1s -> 2.0s. Unlike the refined layer (whose
per-step refinement work must be Python-unrolled -- ~74s/3.8GB compile at
just num_steps=3), separation_loss_multistep's per-segment work is fully
`jax.lax.scan`ned, so compile cost is roughly FLAT in num_segments
(measured: ~28-34s at num_segments in {10,20,50}) -- this is what actually
makes "just increase the horizon" cheap to try here.

Loss: margin_tau=0.01 (quadrotor_separating_input.py's
`separation_loss_multistep`), not the module's default (None ->
_overlap_volume-style product). Replaces the original version of this
script's _overlap_volume-based loss~1e-19 read as "effectively exact
separation" -- checking that by hand (RESULTS.md "Important caveat") showed
it was misleading: a product of 12 per-dimension overlap widths reads as
~0 purely from this module's boxes being small in absolute terms, even
with every dimension still overlapping 60-90%. The margin-based loss
targets max-over-dimensions signed separation directly instead.

Second bug found and fixed while building this (see
`_soft_separation_loss`'s docstring "Bias correction" in
quadrotor_separating_input.py): the FIRST version of the margin loss used
plain `tau*logsumexp(margins/tau)`, which is an UPPER bound on the true
max margin by up to `tau*log(12)~=0.025` -- comparable to this module's
actual ~0.02 margins, so it ALSO falsely reported loss=0.0 while every
dimension was still genuinely overlapping. Subtracting `tau*log(12)`
turns it into a safe LOWER bound instead, so loss=0.0 now is a real
certificate.

Optimizer note: plain (unnormalized) gradient descent gets stuck here --
gradient norm through 20 unrolled Euler steps reaches O(1e3) (a standard
RNN-like compounding effect) while the moment control box is only O(1e-4)
wide, so any fixed learning_rate either overshoots the box every step
(clipped back to the same boundary point, permanently stuck) or crawls.
`optimize_multistep_gpu(..., normalize_grad=True)` fixes this -- see that
function's docstring.

The script verifies genuine (not proxy-loss) separation directly: for
every segment, checks whether max-over-dimensions signed margin is
non-negative for EVERY scenario pair simultaneously.

Produces:
  results/quadrotor_separating_input.npz  -- u_seq, loss, x0_ivl, scenario
                                             fault ranges (consumed by
                                             run_quadrotor_diagnosis_qps.py)
  results/quadrotor_position_3d.pdf       -- 3D (x,y,z) reachable-position
                                             boxes + each scenario's point
                                             trajectory under u_seq, same
                                             wireframe-cuboid convention as
                                             adaptive_spoofing's
                                             plot_legacy_position_3d.py
  results/quadrotor_attitude_3d.pdf       -- same convention, (phi,theta,psi)
                                             instead of (x,y,z). ActuatorFault_2/3/4
                                             (roll/pitch/yaw moment faults)
                                             separate almost entirely through
                                             attitude, not position -- this
                                             plot is where that separation is
                                             actually visible.

Usage:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/quadrotor_fault_diagnosis/solve_and_plot.py
"""
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import jax
import jax.numpy as jnp
import immrax as irx

from quadrotor_separating_input import (
    create_scenarios, optimize_multistep_gpu_rejit, time_jit,
    QuadrotorSystem, _overlap_volume, _signed_separation_margin,
)

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DT = 0.1                 # s per Euler step
NUM_SEGMENTS = 20        # 2.0s horizon -- see module docstring "Horizon"
MARGIN_TAU = 0.01
NUM_RESTARTS = 40
GD_LEARNING_RATE = 5e-6  # step size in the NORMALIZED gradient direction --
                         # see optimize_multistep_gpu's normalize_grad docstring
GD_NUM_ITERS = 400
X0_WIDTH = 0.01           # initial-state half-width, uniform over all 12 dims
ALPHA_LO, ALPHA_HI = 0.5, 0.9
ALPHA_TRUE_FAULT = 0.7    # concrete point value for the point-simulated
                          # trajectories (midpoint of [ALPHA_LO,ALPHA_HI]),
                          # matching simulate_and_render.py's convention

SCENARIO_COLORS = {
    "Nominal": "tab:blue", "ActuatorFault_1": "tab:orange",
    "ActuatorFault_2": "tab:green", "ActuatorFault_3": "tab:red",
    "ActuatorFault_4": "tab:purple",
}

_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def _draw_box3d(ax, lo3, hi3, color, alpha=0.55, linewidth=1.1):
    corners = [(x, y, z) for x in (lo3[0], hi3[0])
              for y in (lo3[1], hi3[1]) for z in (lo3[2], hi3[2])]
    segments = [(corners[i], corners[j]) for i, j in _EDGES]
    ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=linewidth,
                                         alpha=alpha))


def rk4_rollout(x0, u_seq, alpha, dt, sub_steps=10):
    """Point (non-interval) RK4 rollout of QuadrotorSystem.f, `sub_steps`
    per Euler-step-sized segment of u_seq, for a smoother trajectory than
    one RK4 step per segment would give. Returns (num_steps*sub_steps+1, 12)."""
    sys_ = QuadrotorSystem()
    h = dt / sub_steps

    def f(x, u):
        return sys_.f(jnp.zeros(()), x, u, alpha)

    def rk4_step(x, u):
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    xs = [x0]
    x = x0
    for k in range(u_seq.shape[0]):
        u = u_seq[k]
        for _ in range(sub_steps):
            x = rk4_step(x, u)
            xs.append(x)
    return jnp.stack(xs)


def main():
    print(f"Solving multistep-unrefined separating input: {NUM_SEGMENTS} segments x "
         f"dt={DT}s = {NUM_SEGMENTS*DT:.2f}s horizon, x0_width={X0_WIDTH}, "
         f"alpha in [{ALPHA_LO},{ALPHA_HI}], margin_tau={MARGIN_TAU}")
    scenarios = create_scenarios(alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI)
    x0_ivl = irx.icentpert(jnp.zeros(12), jnp.full(12, X0_WIDTH))

    def full_multistart(seed):
        return optimize_multistep_gpu_rejit(
            x0_ivl, scenarios, dt=DT, steps_per_segment=1, num_segments=NUM_SEGMENTS,
            num_restarts=NUM_RESTARTS, learning_rate=GD_LEARNING_RATE, num_iters=GD_NUM_ITERS,
            seed=seed, margin_tau=MARGIN_TAU, normalize_grad=True)

    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 0)
    print(f"compile {compile_t*1e3:8.2f} ms   run {run_t*1e3:7.3f} ms")

    u_seq, loss, u_all, losses_all = jitted_fn(0)
    n_zero = int(jnp.sum(losses_all == 0.0))
    print(f"Best margin-loss: {float(loss):.6e}  ({n_zero}/{NUM_RESTARTS} restarts hit "
         f"loss=0.0, a certified-disjoint-at-some-segment solution -- see "
         f"_soft_separation_loss's 'Bias correction' docstring for why loss=0.0 is safe here)")

    NUM_STEPS = int(u_seq.shape[0])
    for k in range(NUM_STEPS):
        print(f"  u_seq[{k:2d}] = [U1={float(u_seq[k,0]):.5f} N, U2={float(u_seq[k,1]):.6e}, "
             f"U3={float(u_seq[k,2]):.6e}, U4={float(u_seq[k,3]):.6e}]")

    # Per-segment separation check across the WHOLE horizon (not just the
    # final segment) -- matches what separation_loss_multistep's "min over
    # segments" actually optimizes: the online diagnosis rule only needs
    # ONE segment where every pair is simultaneously disjoint (a model
    # excluded there stays excluded, regardless of what the boxes do
    # afterward). Reports both the volume-product metric (for direct
    # comparison to the original _overlap_volume-based version of this
    # script) and the margin-based genuine-disjointness check.
    from quadrotor_separating_input import _propagate_history
    hist_by_name = {s.name: _propagate_history(x0_ivl, u_seq, s.emb_system, s.p_interval, DT, 1)
                    for s in scenarios}
    print("Per-segment pairwise separation (raw, independent per-scenario boxes):")
    disjoint_segments = []
    for k in range(NUM_STEPS):
        all_disjoint_k = True
        worst_margin_k = float("inf")
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                ivl_i = irx.Interval(lower=hist_by_name[scenarios[i].name].lower[k],
                                     upper=hist_by_name[scenarios[i].name].upper[k])
                ivl_j = irx.Interval(lower=hist_by_name[scenarios[j].name].lower[k],
                                     upper=hist_by_name[scenarios[j].name].upper[k])
                m = float(jnp.max(_signed_separation_margin(ivl_i, ivl_j)))
                all_disjoint_k &= (m >= 0.0)
                worst_margin_k = min(worst_margin_k, m)
        if all_disjoint_k:
            disjoint_segments.append(k)
        print(f"  segment {k:2d}: all pairs disjoint={all_disjoint_k!s:5s}  "
             f"worst_pair_margin={worst_margin_k:+.5f}")
    all_separated = len(disjoint_segments) > 0
    print(f"Segments with ALL pairs GENUINELY disjoint: {disjoint_segments}")
    print(f"At least one genuinely-disjoint segment found: {all_separated}")

    # ── Per-scenario reachable-position-box history + point trajectories ──
    # Position (x,y,z) and attitude (phi,theta,psi) are both kept: a quick
    # final-state check (see solve_and_plot.py's usage notes) showed that at
    # this horizon, ActuatorFault_2/3/4 (roll/pitch/yaw moment faults)
    # separate almost entirely through ATTITUDE, not position -- only
    # ActuatorFault_1 (thrust) moves position (z) meaningfully. A
    # position-only plot is correct but visually undersells the separation
    # for 3 of 4 fault scenarios, so both projections are plotted.
    box_lo, box_hi = {}, {}
    box_lo_att, box_hi_att = {}, {}
    point_traj, point_traj_att = {}, {}
    for s in scenarios:
        hist = hist_by_name[s.name]
        box_lo[s.name] = np.array(hist.lower)[:, 0:3]   # (num_steps, 3) position only
        box_hi[s.name] = np.array(hist.upper)[:, 0:3]
        box_lo_att[s.name] = np.array(hist.lower)[:, 3:6]   # attitude: phi, theta, psi
        box_hi_att[s.name] = np.array(hist.upper)[:, 3:6]
        alpha_point = jnp.ones(4) if s.name == "Nominal" else (
            jnp.ones(4).at[int(s.name[-1]) - 1].set(ALPHA_TRUE_FAULT))
        traj = rk4_rollout(jnp.zeros(12), u_seq, alpha_point, DT)
        point_traj[s.name] = np.array(traj)[:, 0:3]
        point_traj_att[s.name] = np.array(traj)[:, 3:6]

    np.savez(RESULTS_DIR / "quadrotor_separating_input.npz",
            u_seq=np.array(u_seq), loss=float(loss), dt=DT, num_steps=NUM_STEPS,
            x0_width=X0_WIDTH, alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
            scenario_names=np.array([s.name for s in scenarios]),
            all_separated=all_separated,
            disjoint_segments=np.array(disjoint_segments, dtype=np.int32))
    print(f"Saved {RESULTS_DIR / 'quadrotor_separating_input.npz'}")

    # ── 3D position plot ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 11,
    })
    # Plot window: through a couple segments past the last genuinely-disjoint
    # one (not the full NUM_SEGMENTS=20). Reachable-box growth under a
    # single-big-Euler-step natural embedding, unrolled over many segments,
    # is well known to balloon (wrapping-effect over-approximation
    # compounding step to step) well beyond anything physically meaningful
    # -- by segment 19 the boxes span multiple meters/radians despite the
    # true point trajectories moving centimeters/degrees. Since diagnosis
    # only needs ONE disjoint segment (see main()'s per-segment check), and
    # segments 0-4 are ALL genuinely disjoint here, plotting through
    # segment 6 shows the actual separation and a bit of context without
    # the late-segment blowup drowning it out.
    plot_segments = min(NUM_STEPS, max(disjoint_segments, default=0) + 3)
    plot_pts = plot_segments * 10 + 1   # rk4_rollout's sub_steps=10 per segment

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    for s in scenarios:
        color = SCENARIO_COLORS[s.name]
        pos = point_traj[s.name][:plot_pts]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=2.0,
               label=s.name, zorder=5)
        ax.scatter(pos[::10, 0], pos[::10, 1], pos[::10, 2], color=color, s=10, zorder=6)
        for k in range(plot_segments):
            _draw_box3d(ax, box_lo[s.name][k], box_hi[s.name][k], color)
    ax.set_xlabel("$x$ (m)", labelpad=8)
    ax.set_ylabel("$y$ (m)", labelpad=8)
    ax.set_zlabel("$z$ (m)", labelpad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.02, top=1.0)
    out_pdf = RESULTS_DIR / "quadrotor_position_3d.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")

    # ── 3D attitude plot (phi, theta, psi) ──
    # This is the projection where ActuatorFault_2/3/4 are actually
    # visually distinguishable -- see comment above box_lo_att.
    fig2 = plt.figure(figsize=(7.0, 6.0))
    ax2 = fig2.add_subplot(111, projection="3d")
    for s in scenarios:
        color = SCENARIO_COLORS[s.name]
        att = point_traj_att[s.name][:plot_pts]
        ax2.plot(att[:, 0], att[:, 1], att[:, 2], color=color, linewidth=2.0,
                 label=s.name, zorder=5)
        ax2.scatter(att[::10, 0], att[::10, 1], att[::10, 2], color=color, s=10, zorder=6)
        for k in range(plot_segments):
            _draw_box3d(ax2, box_lo_att[s.name][k], box_hi_att[s.name][k], color)
    ax2.set_xlabel(r"$\phi$ (rad)", labelpad=8)
    ax2.set_ylabel(r"$\theta$ (rad)", labelpad=8)
    ax2.set_zlabel(r"$\psi$ (rad)", labelpad=12)
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    fig2.subplots_adjust(left=0.0, right=0.88, bottom=0.02, top=1.0)
    out_pdf2 = RESULTS_DIR / "quadrotor_attitude_3d.pdf"
    fig2.savefig(out_pdf2)
    plt.close(fig2)
    print(f"Wrote {out_pdf2}")


if __name__ == "__main__":
    main()
