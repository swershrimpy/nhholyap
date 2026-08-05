"""
animate_refinement_3d_sparse.py
================================
Animates the 3-D refinement interval plot with sparse interval boxes.

Key behavior:
- Total frames = (dt * num_steps) / fps = total_time / fps
- Every dt/fps frames, a new group of output interval boxes is drawn
- The true trajectory is rolled out at 30 fps (every 0.033 s)
- Interval boxes appear only at ctrl_dt steps (e.g., every 0.1 s)

Usage
-----
  python animate_refinement_3d_sparse.py --fault-mode actuator --x0 0.1 0.1 0.0
  python animate_refinement_3d_sparse.py --fault-mode nominal --sim-dt 0.033
"""

import argparse
import shutil
import sys
from pathlib import Path

import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from faulty_car_separating_input import (
    CarNomActSystem,
    create_scenarios,
    euler_step,
    _obs_interval,
    optimize_refined_gpu,
)

# ── Constants ─────────────────────────────────────────────────────────────────
FADE_FRAMES    = 3      # frames over which an invalidated pair fades to invisible
DEFAULT_AZIM   = 45.0   # default fixed camera azimuth (degrees)
DEFAULT_ELEV   = 20.0   # default fixed camera elevation (degrees)
LIMIT_PAD      = 0.02   # fractional padding around the full-data bounding box (tighter)

# Controller defaults — define the time axis and interval propagation grid
CTRL_DT        = 1    # controller Euler step size (s)
CTRL_NUM_STEPS = 5     # controller steps  →  0.1 × 50 = 5 s total

# Animation default — controls playback speed only (30 fps)
DEFAULT_SIM_DT = 0.033  # animation frame interval (s)  →  ~30 fps


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Animated 3-D refinement interval plot with sparse intervals."
    )
    p.add_argument(
        "--fault-mode",
        choices=["nominal", "actuator", "sensor"],
        default="nominal",
        help="Fault mode the trajectory is simulated under (default: nominal).",
    )
    p.add_argument(
        "--x0",
        nargs=3, type=float, metavar=("PX", "PY", "PHI"),
        default=[0.1, 0.1, 0.0],
        help="Initial state [px py phi] (default: 0.1 0.1 0.0).",
    )
    p.add_argument(
        "--ctrl-dt", type=float, default=CTRL_DT,
        help=(
            f"Controller Euler step size in seconds (default: {CTRL_DT}). "
            "Used for interval propagation."
        ),
    )
    p.add_argument(
        "--ctrl-num-steps", type=int, default=CTRL_NUM_STEPS,
        help=(
            f"Number of controller steps (default: {CTRL_NUM_STEPS}). "
            f"Total time axis = ctrl_dt × ctrl_num_steps "
            f"(default: {CTRL_DT} × {CTRL_NUM_STEPS} = {CTRL_DT * CTRL_NUM_STEPS:.1f} s)."
        ),
    )
    p.add_argument(
        "--plot-dt", type=float, default=None,
        help=(
            "How often to reveal a new group of interval boxes (seconds). "
            "Must be a multiple of ctrl_dt. Defaults to ctrl_dt (reveal every step). "
            "Example: --ctrl-dt 0.1 --plot-dt 1.0 propagates at 0.1s but only "
            "shows new boxes every 1.0s."
        ),
    )
    p.add_argument(
        "--sim-dt", type=float, default=DEFAULT_SIM_DT,
        help=(
            f"Animation frame interval in seconds (default: {DEFAULT_SIM_DT} ≈ 30 fps). "
            "Controls playback speed only. Video FPS = round(1 / sim_dt). "
            "The trajectory is rolled out at this rate."
        ),
    )
    p.add_argument(
        "--azim", type=float, default=DEFAULT_AZIM,
        help=f"Fixed camera azimuth in degrees (default: {DEFAULT_AZIM}).",
    )
    p.add_argument(
        "--elev", type=float, default=DEFAULT_ELEV,
        help=f"Fixed camera elevation in degrees (default: {DEFAULT_ELEV}).",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="Output video DPI (default: 150).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output .mp4 path (default: auto-named in script directory).",
    )
    return p.parse_args()


# ── Trajectory simulation ─────────────────────────────────────────────────────

def simulate_trajectory(x0: jnp.ndarray, u_seq: jnp.ndarray,
                        alpha: float, dt: float) -> np.ndarray:
    """Euler-integrate CarNomActSystem from x0 using u_seq at step size dt.

    Returns traj of shape (num_steps+1, 3): [px, py, phi] at each step.
    """
    sys_obj = CarNomActSystem()
    p    = jnp.array([alpha])
    x    = x0
    traj = [np.array(x)]
    for k in range(u_seq.shape[0]):
        dx = sys_obj.f(jnp.zeros(()), x, u_seq[k], p)
        x  = x + dx * dt
        traj.append(np.array(x))
    return np.stack(traj)


# ── Refinement history ────────────────────────────────────────────────────────

def _intersect_obs(a, b):
    lo = jnp.maximum(a.lower, b.lower)
    hi = jnp.minimum(a.upper, b.upper)
    return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None


def collect_refinement_history(x0_ivl, u_seq, scenarios, dt, num_steps):
    """Returns (steps, pairs).

    steps[k]['t']            : float time at step k+1
    steps[k]['pair_obs'][pi] : (obs_i, obs_j, intersection_or_None)
    """
    n     = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    x1   = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt)
            for s in scenarios]
    obs1 = [_obs_interval(xi, s) for xi, s in zip(x1, scenarios)]
    steps = [{'t': dt, 'pair_obs': [
        (obs1[i], obs1[j], _intersect_obs(obs1[i], obs1[j])) for i, j in pairs
    ]}]
    pair_states = [(x1[i], x1[j]) for i, j in pairs]

    for k in range(num_steps):
        new_states, new_pair_obs = [], []
        for (i, j), (xi, xj) in zip(pairs, pair_states):
            oi = _obs_interval(xi, scenarios[i])
            oj = _obs_interval(xj, scenarios[j])
            y_lo = jnp.maximum(oi.lower, oj.lower)
            y_hi = jnp.minimum(oi.upper, oj.upper)
            hov   = bool(jnp.all(y_hi >= y_lo))
            fb    = (xi.lower[:2] + xi.upper[:2]) / 2
            ys_lo = jnp.where(hov, y_lo, fb)
            ys_hi = jnp.where(hov, y_hi, fb)

            xi_ref = irx.Interval(
                lower=jnp.array([ys_lo[0] - scenarios[i].obs_offset[0],
                                  ys_lo[1] - scenarios[i].obs_offset[1], xi.lower[2]]),
                upper=jnp.array([ys_hi[0] - scenarios[i].obs_offset[0],
                                  ys_hi[1] - scenarios[i].obs_offset[1], xi.upper[2]]),
            )
            xj_ref = irx.Interval(
                lower=jnp.array([ys_lo[0] - scenarios[j].obs_offset[0],
                                  ys_lo[1] - scenarios[j].obs_offset[1], xj.lower[2]]),
                upper=jnp.array([ys_hi[0] - scenarios[j].obs_offset[0],
                                  ys_hi[1] - scenarios[j].obs_offset[1], xj.upper[2]]),
            )
            xn_i = euler_step(scenarios[i].emb_system, xi_ref, u_seq[k + 1],
                              scenarios[i].p_interval, dt)
            xn_j = euler_step(scenarios[j].emb_system, xj_ref, u_seq[k + 1],
                              scenarios[j].p_interval, dt)
            on_i  = _obs_interval(xn_i, scenarios[i])
            on_j  = _obs_interval(xn_j, scenarios[j])
            new_pair_obs.append((on_i, on_j, _intersect_obs(on_i, on_j)))
            new_states.append((xn_i, xn_j))

        steps.append({'t': (k + 2) * dt, 'pair_obs': new_pair_obs})
        pair_states = new_states

    return steps, pairs


# ── Validity computation ──────────────────────────────────────────────────────

def compute_validity(ref_steps, ref_pairs, traj_obs_px, traj_obs_py):
    """Compute valid[pair_idx, ctrl_step_idx] boolean matrix.

    A pair is valid at controller step k if the trajectory's observed output
    at that step falls inside both scenarios' refined output intervals.
    Once invalid, stays invalid (no re-entry).

    ref_steps indices 0..N-1 correspond to traj indices 1..N.
    """
    n_pairs = len(ref_pairs)
    n_steps = len(ref_steps)
    valid   = np.ones((n_pairs, n_steps), dtype=bool)

    for pi in range(n_pairs):
        still_valid = True
        for k, step in enumerate(ref_steps):
            if not still_valid:
                valid[pi, k] = False
                continue
            obs_i, obs_j, _ = step['pair_obs'][pi]
            yx = traj_obs_px[k + 1]
            yy = traj_obs_py[k + 1]
            inside_i = (float(obs_i.lower[0]) <= yx <= float(obs_i.upper[0]) and
                        float(obs_i.lower[1]) <= yy <= float(obs_i.upper[1]))
            inside_j = (float(obs_j.lower[0]) <= yx <= float(obs_j.upper[0]) and
                        float(obs_j.lower[1]) <= yy <= float(obs_j.upper[1]))
            if not (inside_i and inside_j):
                still_valid = False
                valid[pi, k] = False

    return valid


# ── Dynamic axis limits (precomputed with linear interpolation between ctrl_dt steps) ─

def _obs_bounds(obs):
    return (float(obs.lower[0]), float(obs.upper[0]),
            float(obs.lower[1]), float(obs.upper[1]))


def _invalidation_step(valid, pi):
    """Return the first controller step index where pair pi becomes invalid, or None."""
    for k in range(valid.shape[1]):
        if not valid[pi, k]:
            return k
    return None


def compute_limits_at_plot_step(plot_k, steps_per_plot, ref_steps, ref_pairs, valid,
                                traj_obs_px, traj_obs_py, traj_times, sim_dt, ctrl_dt):
    """Compute px and py axis limits from data visible up to plot_k.

    'Visible' means: trajectory up to t = plot_k * plot_dt, and interval
    boxes for ctrl_dt steps 0 .. plot_k*steps_per_plot - 1.
    Returns (px_lo, px_hi, py_lo, py_hi) with LIMIT_PAD applied.
    """
    # Trajectory points visible up to this plot reveal time
    t_plot = plot_k * steps_per_plot * ctrl_dt
    traj_end = min(int(t_plot / sim_dt) + 1, len(traj_obs_px))
    all_x = list(traj_obs_px[:traj_end])
    all_y = list(traj_obs_py[:traj_end])

    # Interval boxes revealed so far (ctrl steps 0 .. plot_k*steps_per_plot - 1)
    visible_ctrl_steps = plot_k * steps_per_plot
    for s_idx in range(min(visible_ctrl_steps, len(ref_steps))):
        for pi in range(len(ref_pairs)):
            obs_i, obs_j, _ = ref_steps[s_idx]['pair_obs'][pi]
            for obs in (obs_i, obs_j):
                xl, xh, yl, yh = _obs_bounds(obs)
                all_x += [xl, xh]
                all_y += [yl, yh]

    if not all_x:
        return -0.5, 0.5, -0.5, 0.5

    xl, xh = min(all_x), max(all_x)
    yl, yh = min(all_y), max(all_y)

    if xh - xl < 1e-4:
        xh += 0.05; xl -= 0.05
    if yh - yl < 1e-4:
        yh += 0.05; yl -= 0.05

    pad_x = LIMIT_PAD * (xh - xl)
    pad_y = LIMIT_PAD * (yh - yl)
    return xl - pad_x, xh + pad_x, yl - pad_y, yh + pad_y


def precompute_interpolated_limits(ctrl_dt, sim_dt, ctrl_num_steps,
                                    ref_steps, ref_pairs, valid,
                                    traj_obs_px, traj_obs_py,
                                    steps_per_plot=1):
    """Precompute axis limits at each plot_dt step, then interpolate for all frames.

    Limits only change when a new group of interval boxes is revealed (every
    steps_per_plot ctrl_dt steps = plot_dt seconds).

    Returns array of shape (num_frames, 4): [px_lo, px_hi, py_lo, py_hi] per frame.
    """
    plot_dt       = steps_per_plot * ctrl_dt
    num_plot_steps = int(round(ctrl_num_steps / steps_per_plot))
    traj_times    = np.arange(len(traj_obs_px)) * sim_dt

    # Compute limits at each plot_dt keyframe
    plot_limits = []
    for plot_k in range(num_plot_steps + 1):
        plot_limits.append(compute_limits_at_plot_step(
            plot_k, steps_per_plot, ref_steps, ref_pairs, valid,
            traj_obs_px, traj_obs_py, traj_times, sim_dt, ctrl_dt))
    plot_limits = np.array(plot_limits)  # shape (num_plot_steps+1, 4)

    # Interpolate for every animation frame
    num_frames  = len(traj_obs_px)
    frame_times = np.arange(num_frames) * sim_dt
    plot_times  = np.arange(num_plot_steps + 1) * plot_dt

    px_lo = np.interp(frame_times, plot_times, plot_limits[:, 0])
    px_hi = np.interp(frame_times, plot_times, plot_limits[:, 1])
    py_lo = np.interp(frame_times, plot_times, plot_limits[:, 2])
    py_hi = np.interp(frame_times, plot_times, plot_limits[:, 3])

    return np.column_stack([px_lo, px_hi, py_lo, py_hi])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Check ffmpeg
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install it (e.g. `apt install ffmpeg` "
              "or `conda install ffmpeg`) and re-run.", file=sys.stderr)
        sys.exit(1)

    fault_mode     = args.fault_mode
    x0_state       = jnp.array(args.x0)
    ctrl_dt        = args.ctrl_dt
    ctrl_num_steps = args.ctrl_num_steps
    sim_dt         = args.sim_dt
    azim           = args.azim
    elev           = args.elev
    fps            = max(1, round(1.0 / sim_dt))

    # plot_dt: how often to reveal new interval boxes (must be multiple of ctrl_dt)
    plot_dt = args.plot_dt if args.plot_dt is not None else ctrl_dt
    plot_dt = max(ctrl_dt, round(plot_dt / ctrl_dt) * ctrl_dt)  # snap to nearest ctrl_dt multiple
    # steps_per_plot: how many ctrl_dt steps between each box reveal
    steps_per_plot = max(1, round(plot_dt / ctrl_dt))
    plot_dt = steps_per_plot * ctrl_dt  # exact value after snapping

    total_time = ctrl_dt * ctrl_num_steps

    # Total frames = total_time / fps
    num_frames = max(1, round(total_time / sim_dt))
    print(f"Controller : ctrl_dt={ctrl_dt} s  ×  ctrl_num_steps={ctrl_num_steps}"
          f"  →  {total_time:.2f} s time axis")
    print(f"Plot every : plot_dt={plot_dt:.3f} s  ({steps_per_plot} ctrl steps per reveal)")
    print(f"Animation  : {num_frames} frames  "
          f"(covers {total_time:.2f} s at {fps} fps)")
    print(f"Camera     : azim={azim}°  elev={elev}°  (fixed)")

    # ── Problem setup ─────────────────────────────────────────────────────
    scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
    x0_ivl    = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.1, 0.1, 0.1]))

    _mode_map = {
        "nominal":  (0, 1.0),
        "actuator": (1, 0.25),   # midpoint of [0.0, 0.5]
        "sensor":   (2, 1.0),
    }
    scenario_idx, alpha = _mode_map[fault_mode]
    traj_scenario = scenarios[scenario_idx]
    print(f"Fault mode : {traj_scenario.name}  (alpha={alpha})")
    print(f"x0         : {args.x0}")

    # ── Optimise (uses ctrl_dt and ctrl_num_steps) ─────────────────────────
    print("Optimising separating input …")
    u_opt, loss_opt, _, _ = optimize_refined_gpu(
        x0_ivl=x0_ivl, scenarios=scenarios, dt=ctrl_dt,
        num_restarts=500, learning_rate=1, num_iters=50, seed=42,
        num_steps=ctrl_num_steps,
    )
    print(f"  loss={loss_opt:.6f}  u_opt shape={u_opt.shape}")

    # ── Simulate trajectory at 30 fps (uses sim_dt) ───────────────────────
    print("Simulating trajectory at 30 fps …")
    # Number of trajectory steps = total_time / sim_dt + 1
    num_traj_steps = int(total_time / sim_dt) + 1
    traj_30fps   = simulate_trajectory(x0_state, u_opt, alpha, sim_dt)
    # Ensure we have enough trajectory points for all frames
    if traj_30fps.shape[0] < num_traj_steps:
        sys_obj = CarNomActSystem()
        last_u = u_opt[-1]
        x = traj_30fps[-1]
        for _ in range(num_traj_steps - traj_30fps.shape[0]):
            dx = sys_obj.f(jnp.zeros(()), x, last_u, jnp.array([alpha]))
            x  = x + dx * sim_dt
            traj_30fps = np.vstack([traj_30fps, x])
    traj_times_30fps = np.array([k * sim_dt for k in range(traj_30fps.shape[0])])

    obs_off     = np.array(traj_scenario.obs_offset)
    traj_obs_px_30fps = traj_30fps[:, 0] + obs_off[0]
    traj_obs_py_30fps = traj_30fps[:, 1] + obs_off[1]

    # ── Refinement history (uses ctrl_dt) ─────────────────────────────────
    print("Collecting refinement history …")
    ref_steps, ref_pairs = collect_refinement_history(
        x0_ivl, u_opt, scenarios, ctrl_dt, ctrl_num_steps)
    print(f"  {len(ref_steps)} steps, {len(ref_pairs)} pairs")

    # ── Validity matrix ────────────────────────────────────────────────────
    print("Computing validity …")
    valid = compute_validity(ref_steps, ref_pairs, traj_obs_px_30fps, traj_obs_py_30fps)
    for pi, (i, j) in enumerate(ref_pairs):
        inv = _invalidation_step(valid, pi)
        label = f"t={inv * ctrl_dt:.3f}s (step {inv})" if inv is not None else "never"
        print(f"  Pair ({scenarios[i].name}, {scenarios[j].name}): "
              f"invalidated at {label}")

    # ── Fixed t axis span ─────────────────────────────────────────────────
    print(f"Time axis  : [0, {total_time:.2f}] s (fixed)")

    # ── Colour setup ──────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'text.usetex': False,
    })
    cmap       = plt.colormaps['tab10'].resampled(len(scenarios))
    traj_color = cmap(scenario_idx)

    pair_labels = [
        f"{scenarios[i].name} – {scenarios[j].name}"
        for i, j in ref_pairs
    ]

    # ── Precompute interpolated axis limits for all frames ─────────────────
    print("Precomputing interpolated axis limits …")
    frame_limits = precompute_interpolated_limits(
        ctrl_dt, sim_dt, ctrl_num_steps,
        ref_steps, ref_pairs, valid,
        traj_obs_px_30fps, traj_obs_py_30fps,
        steps_per_plot=steps_per_plot)

    # ── Figure setup ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # ── Frame update function ─────────────────────────────────────────────

    def update(frame_idx):
        ax.cla()

        t_now = traj_times_30fps[frame_idx]
        # ── Draw interval boxes only at plot_dt steps up to current time ──
        # ctrl_k: how many ctrl_dt steps have elapsed (for validity/fading)
        ctrl_k  = int(t_now / ctrl_dt)
        # plot_k: how many plot_dt groups have been revealed
        plot_k  = int(t_now / plot_dt)

        for s_idx in range(min(ctrl_k, len(ref_steps))):
            # Only reveal this step's box if it falls within a revealed plot group
            if s_idx // steps_per_plot >= plot_k:
                continue
            t_s = ref_steps[s_idx]['t']
            for pi, (i, j) in enumerate(ref_pairs):
                inv_step = _invalidation_step(valid, pi)

                if inv_step is None:
                    box_alpha = 0.35
                elif s_idx < inv_step:
                    box_alpha = 0.35
                else:
                    frames_since_inv = ctrl_k - inv_step
                    if frames_since_inv >= FADE_FRAMES:
                        continue
                    box_alpha = 0.35 * (1.0 - frames_since_inv / FADE_FRAMES)

                obs_i, obs_j, inter = ref_steps[s_idx]['pair_obs'][pi]

                for obs, color in [(obs_i, cmap(i)), (obs_j, cmap(j))]:
                    xl = float(obs.lower[0]); xh = float(obs.upper[0])
                    yl = float(obs.lower[1]); yh = float(obs.upper[1])
                    verts = [[(xl, t_s, yl), (xh, t_s, yl),
                               (xh, t_s, yh), (xl, t_s, yh)]]
                    poly  = Poly3DCollection(verts, alpha=box_alpha)
                    poly.set_facecolor(color)
                    poly.set_edgecolor((*matplotlib.colors.to_rgb(color), box_alpha))
                    ax.add_collection3d(poly)

                if inter is not None and box_alpha > 0:
                    xl = float(inter.lower[0]); xh = float(inter.upper[0])
                    yl = float(inter.lower[1]); yh = float(inter.upper[1])
                    ax.plot([xl, xh, xh, xl, xl],
                            [t_s, t_s, t_s, t_s, t_s],
                            [yl, yl, yh, yh, yl],
                            color='black', linewidth=1.5,
                            alpha=min(box_alpha / 0.35, 1.0), zorder=5)

        # ── Trajectory line up to current frame (30 fps) ──────────────────
        if frame_idx > 0:
            ax.plot(traj_obs_px_30fps[:frame_idx + 1],
                    traj_times_30fps[:frame_idx + 1],
                    traj_obs_py_30fps[:frame_idx + 1],
                    color=traj_color, linewidth=2.5, linestyle='-',
                    marker='o', markersize=3, zorder=10,
                    label=f"Trajectory ({traj_scenario.name})")

        # ── Current measurement dot ───────────────────────────────────────
        ax.scatter([traj_obs_px_30fps[frame_idx]], [t_now], [traj_obs_py_30fps[frame_idx]],
                   color='red', s=80, zorder=15, depthshade=False,
                   label="Current measurement")

        # ── Dynamic axis limits with linear interpolation between ctrl_dt steps ─
        px_lo, px_hi, py_lo, py_hi = frame_limits[frame_idx]
        ax.set_xlim(px_lo, px_hi)
        ax.set_ylim(0.0, total_time)
        ax.set_zlim(py_lo, py_hi)

        # ── Camera (fixed) ────────────────────────────────────────────────
        ax.view_init(elev=elev, azim=azim)

        # ── Labels (no axis labels for px, time, py) ─────────────────────
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # ── Title: show valid pairs based on what's been revealed ────────
        if plot_k == 0:
            valid_labels = pair_labels[:]
        else:
            # Last revealed ctrl step index
            s_idx_now = min(plot_k * steps_per_plot - 1, len(ref_steps) - 1)
            valid_labels = [
                pair_labels[pi]
                for pi in range(len(ref_pairs))
                if valid[pi, s_idx_now]
            ]
        valid_str = ", ".join(valid_labels) if valid_labels else "None (fault isolated!)"
        ax.set_title(
            f"t = {t_now:.2f} s  |  Valid pairs: {valid_str}",
            fontsize=10, pad=10,
        )

        # ── Legend ────────────────────────────────────────────────────────
        legend_handles = []
        for k_sc, s in enumerate(scenarios):
            legend_handles.append(
                plt.Line2D([0], [0], color=cmap(k_sc), linewidth=3, label=s.name)
            )
        legend_handles.append(
            plt.Line2D([0], [0], color='black', linewidth=1.5, label='Intersection')
        )
        legend_handles.append(
            plt.Line2D([0], [0], color=traj_color, linewidth=2.5,
                       marker='o', markersize=4,
                       label=f"Trajectory ({traj_scenario.name})")
        )
        legend_handles.append(
            plt.Line2D([0], [0], color='red', marker='o', markersize=6,
                       linestyle='None', label='Current measurement')
        )
        ax.legend(handles=legend_handles, loc='upper left', fontsize=8,
                  bbox_to_anchor=(0.0, 1.0))

        return []

    # ── Build and save animation ──────────────────────────────────────────
    frame_interval_ms = round(sim_dt * 1000)
    print(f"Rendering {num_frames} frames  "
          f"(interval={frame_interval_ms} ms, {fps} fps) …")
    ani = FuncAnimation(fig, update, frames=num_frames,
                        interval=frame_interval_ms, blit=False)

    out_path = args.output or str(
        HERE / f"refinement_anim_sparse_{fault_mode}_ctrl{ctrl_dt}x{ctrl_num_steps}_sim{sim_dt}.mp4"
    )
    ani.save(out_path, writer='ffmpeg', fps=fps, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
