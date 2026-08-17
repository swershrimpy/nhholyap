"""
3D (p_x, p_y, p_z) reachable-tube plot for the legacy chain bank -- POSITION
ONLY, dropping the velocity panels that plot_legacy_reachable_tubes.py's
6-panel time-series figure also shows. A companion to that script, not a
replacement: it reuses `_reachable_history`/`create_scenarios`/
`simulate_true_trajectory`/`CANDIDATE_COLORS` from it directly rather than
recomputing anything, so the two can't silently drift apart (same reasoning
as `_simulate_and_plot`'s own docstring).

Each candidate's per-step position-reachable box (3D interval
[px_lo,px_hi]x[py_lo,py_hi]x[pz_lo,pz_hi], from the SAME unrefined
natural-embedding propagation the 2D plot uses) is drawn as a wireframe
cuboid -- 12 edges, no filled faces -- rather than translucent
Poly3DCollection faces: with multiple candidates' boxes chained along a
moving trajectory (unlike animate_refinement_3d.py's fixed-time-slice
tubes, these boxes advance in all 3 plotted dims at once, so filled,
overlapping faces would rapidly muddy into an unreadable blend). The boxes
are subsampled to ~10 steps along the horizon (always including the first
and last) for the same legibility reason -- one box per one of e.g. 30
raw steps would be dense enough to obscure the trajectory itself.

Defaults to the same data this conversation was just looking at: TIGHT
regime (results/synthesized_spoof.npz, width=2e-3), pid_pos_vel_i excluded
-- pass --all-candidates / --npz to change either.

Usage:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/plot_legacy_position_3d.py [out_pdf] [--all-candidates] [--npz NAME]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import jax.numpy as jnp
import immrax as irx

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Times-metric serif, matching every other figure in this project (e.g.
# run_firmware_discrimination.py): Liberation Serif is metrically compatible
# with Times New Roman and a genuine TrueType face, so it embeds cleanly
# under fonttype 42; Nimbus Roman (the URW Times clone) is metric-compatible
# too but OpenType/CFF, which trips a "font type mismatch" warning in some
# PDF readers -- hence Liberation first. fonttype 42 embeds real vector
# glyphs instead of rasterising.
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

from plot_legacy_reachable_tubes import (
    RESULTS_DIR, CANDIDATE_COLORS, CANDIDATE_NAMES, _CANDIDATE_THETA,
    _reachable_history,
)
from adaptive_spoofing.crazyflie_chain_controllers import (
    create_scenarios, simulate_true_trajectory,
)

_OUT = RESULTS_DIR / "legacy_position_3d.pdf"
_MAX_BOXES_PER_CANDIDATE = 10

# 12 edges of a cuboid as pairs into the 8-corner list built by
# itertools.product-style (x,y,z) in {lo,hi}^3, corner index = 4*ix+2*iy+iz.
_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def _box_corners(lo3, hi3):
    corners = []
    for ix in (lo3[0], hi3[0]):
        for iy in (lo3[1], hi3[1]):
            for iz in (lo3[2], hi3[2]):
                corners.append((ix, iy, iz))
    return corners


def _draw_box3d(ax, lo3, hi3, color, alpha=0.55, linewidth=1.1):
    corners = _box_corners(lo3, hi3)
    segments = [(corners[i], corners[j]) for i, j in _EDGES]
    ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=linewidth,
                                         alpha=alpha))


def make_plot(npz_name="synthesized_spoof.npz", candidate_names=None, out_pdf=_OUT):
    candidate_names = list(candidate_names or CANDIDATE_NAMES)
    data = np.load(RESULTS_DIR / npz_name, allow_pickle=True)
    u_seq = jnp.array(data["u_seq"])
    dt = float(data["dt"])
    ref15 = jnp.array(data["ref15"])
    x0_ivl = irx.Interval(lower=jnp.array(data["x0_ivl_lower"]), upper=jnp.array(data["x0_ivl_upper"]))
    num_steps = u_seq.shape[0]
    print(f"Loaded {npz_name}: shape={u_seq.shape}, dt={dt}, candidates={candidate_names}")

    scenarios = create_scenarios(ref15, names=candidate_names)
    lowers, uppers = _reachable_history(scenarios, x0_ivl, u_seq, dt)   # (T+1, n, 18)

    x0_point = jnp.zeros(18)
    trajectories = {}
    for name in candidate_names:
        theta6 = jnp.array(_CANDIDATE_THETA[name])
        traj = simulate_true_trajectory(x0_point, u_seq, theta6, ref15, dt)   # (T, 12)
        traj_full = np.concatenate([np.array(x0_point[:12])[None, :], np.array(traj)], axis=0)
        trajectories[name] = traj_full[:, 0:3]   # position only, (T+1, 3)

    stride = max(1, (num_steps + 1) // _MAX_BOXES_PER_CANDIDATE)
    box_steps = sorted(set(list(range(0, num_steps + 1, stride)) + [num_steps]))
    print(f"Drawing boxes at {len(box_steps)} of {num_steps + 1} steps (stride={stride})")

    # No title (per request) means no reserved top margin; figsize trimmed
    # from the original 8.5x7.5 now that there's no title row to fill, and
    # subplots_adjust below claims back most of what mplot3d otherwise
    # leaves as dead border padding around a 3D axes.
    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    for i, name in enumerate(candidate_names):
        color = CANDIDATE_COLORS[name]
        pos = trajectories[name]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=2.0,
               label=name, zorder=5)
        ax.scatter(pos[box_steps, 0], pos[box_steps, 1], pos[box_steps, 2],
                  color=color, s=14, zorder=6)
        for k in box_steps:
            _draw_box3d(ax, lowers[k, i, 0:3], uppers[k, i, 0:3], color)

    # Extra labelpad on z keeps its (rotated, right-side) label clear of the
    # tick labels next to it -- mplot3d's default pad is tuned for x/y and
    # clips the z-label against the tick numbers otherwise; bbox_inches=
    # "tight" alone does not fix this because mplot3d under-reports the
    # z-label's extent to Matplotlib's own tight-bbox calculation.
    ax.set_xlabel("$p_x$ (m)", labelpad=8)
    ax.set_ylabel("$p_y$ (m)", labelpad=8)
    ax.set_zlabel("$p_z$ (m)", labelpad=12)
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.3)
    # Push the axes to fill the figure, leaving only the margin the z-label
    # actually needs on the right -- the default mplot3d axes position
    # already assumes a title row and generous all-around padding that
    # "tight_layout"/"bbox_inches=tight" don't fully reclaim for 3D axes.
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.02, top=1.0)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    args = sys.argv[1:]
    all_candidates = "--all-candidates" in args
    args = [a for a in args if a != "--all-candidates"]
    npz_name = "synthesized_spoof.npz"
    if "--npz" in args:
        i = args.index("--npz")
        npz_name = args[i + 1]
        args = args[:i] + args[i + 2:]
    out_pdf = Path(args[0]) if args else _OUT
    names = None if all_candidates else [n for n in CANDIDATE_NAMES if n != "pid_pos_vel_i"]
    p = make_plot(npz_name=npz_name, candidate_names=names, out_pdf=out_pdf)
    print(f"Wrote {p}")
