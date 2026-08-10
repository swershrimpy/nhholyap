"""
animate_refinement_3d.py
========================
Animates the 3-D refinement interval plot as the controller executes on the
system, overlaying a simulated point trajectory and dynamically showing only
still-valid fault pairs.

Reproduces examples/faulty_car/animate_refinement_3d.py's output, but calls
this folder's `car_separating_input.collect_refinement_history` for the
per-pair refinement math instead of re-deriving it locally -- the original
script's local reimplementation had silently drifted from the optimizer's
own refinement logic (a different, second bug beyond the wrong-obs_scale
one -- see PLAN.md bug #3). Using the shared helper means the animation and
the optimized loss can never diverge again.

Timing model (two separate dt values)
--------------------------------------
  ctrl_dt / ctrl_num_steps
      The controller's Euler step size and number of steps. These define:
        - the interval propagation grid  (ref_steps[0..ctrl_num_steps-1])
        - the trajectory integration     (traj[0..ctrl_num_steps])
        - the total time axis length     (ctrl_dt x ctrl_num_steps = 5 s default)
      Defaults: ctrl_dt=0.1 s, ctrl_num_steps=50  ->  5 s total.

  sim_dt  (--sim-dt, default 0.033 s ~= 30 fps)
      The animation frame interval. Controls only how fast the video plays.
      Video FPS = round(1 / sim_dt).
      Each animation frame maps to the nearest controller step index, so the
      time axis always spans exactly ctrl_dt x ctrl_num_steps seconds.

Features
--------
- Forward-reveal: interval boxes and trajectory accumulate as time progresses.
- t axis: always shows the full [0, total_time] span from frame 1.
- px/py axes: fixed limits computed once from all data, padded by LIMIT_PAD.
- Camera: fixed azimuth and elevation (configurable via CLI).
- Validity: a pair (i,j) is "still valid" at step k if the trajectory's
  observed output falls inside both scenarios' refined output intervals.
  Invalid pairs fade out over FADE_FRAMES frames then disappear.
- Output: .mp4 via matplotlib FuncAnimation + ffmpeg.

Usage
-----
  python animate_refinement_3d.py --fault-mode actuator --x0 0.1 0.1 0.0
  python animate_refinement_3d.py --fault-mode nominal --sim-dt 0.05
  python animate_refinement_3d.py --fault-mode sensor --azim 60 --elev 30
  python animate_refinement_3d.py --fault-mode actuator --output my_anim.mp4
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

from car_separating_input import (
    CarNomActSystem,
    create_scenarios,
    optimize_refined_gpu,
    collect_refinement_history,
)

# ── Constants ─────────────────────────────────────────────────────────────────
FADE_FRAMES = 3      # frames over which an invalidated pair fades to invisible
DEFAULT_AZIM = 45.0   # default fixed camera azimuth (degrees)
DEFAULT_ELEV = 20.0   # default fixed camera elevation (degrees)
LIMIT_PAD = 0.08   # fractional padding around the full-data bounding box

# Controller defaults -- define the time axis and interval propagation grid
CTRL_DT = 0.1    # controller Euler step size (s)
CTRL_NUM_STEPS = 50     # controller steps  ->  0.1 x 50 = 5 s total

# Animation default -- controls playback speed only
DEFAULT_SIM_DT = 0.033  # animation frame interval (s)  ->  ~30 fps


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Animated 3-D refinement interval plot with trajectory overlay."
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
            "Used for interval propagation and trajectory integration. "
            "Total time axis = ctrl_dt x ctrl_num_steps."
        ),
    )
    p.add_argument(
        "--ctrl-num-steps", type=int, default=CTRL_NUM_STEPS,
        help=(
            f"Number of controller steps (default: {CTRL_NUM_STEPS}). "
            f"Total time axis = ctrl_dt x ctrl_num_steps "
            f"(default: {CTRL_DT} x {CTRL_NUM_STEPS} = {CTRL_DT * CTRL_NUM_STEPS:.1f} s)."
        ),
    )
    p.add_argument(
        "--sim-dt", type=float, default=DEFAULT_SIM_DT,
        help=(
            f"Animation frame interval in seconds (default: {DEFAULT_SIM_DT} ~= 30 fps). "
            "Controls playback speed only. Video FPS = round(1 / sim_dt). "
            "Does NOT affect the time axis length or interval propagation."
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

    Returns traj of shape (num_steps+1, 3): [px, py, phi] at each step. Not
    shared with car_separating_input.py: this is a plain point rollout, a
    different concern from interval propagation (that part of the original
    animation script was not duplicated/buggy, so it's kept local here too).
    """
    sys_obj = CarNomActSystem()
    p = jnp.array([alpha])
    x = x0
    traj = [np.array(x)]
    for k in range(u_seq.shape[0]):
        dx = sys_obj.f(jnp.zeros(()), x, u_seq[k], p)
        x = x + dx * dt
        traj.append(np.array(x))
    return np.stack(traj)


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
    valid = np.ones((n_pairs, n_steps), dtype=bool)

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


def _invalidation_step(valid, pi):
    """Return the first controller step index where pair pi becomes invalid, or None."""
    for k in range(valid.shape[1]):
        if not valid[pi, k]:
            return k
    return None


# ── Fixed axis limits (computed once from all data) ───────────────────────────

def _obs_bounds(obs):
    return (float(obs.lower[0]), float(obs.upper[0]),
            float(obs.lower[1]), float(obs.upper[1]))


def compute_fixed_limits(ref_steps, ref_pairs, traj_obs_px, traj_obs_py):
    """Compute px and py axis limits from the full dataset (all steps, all pairs).

    Returns (px_lo, px_hi, py_lo, py_hi) with LIMIT_PAD applied.
    These limits are used unchanged for every animation frame.
    """
    all_x, all_y = list(traj_obs_px), list(traj_obs_py)

    for step in ref_steps:
        for pi in range(len(ref_pairs)):
            obs_i, obs_j, _ = step['pair_obs'][pi]
            for obs in (obs_i, obs_j):
                xl, xh, yl, yh = _obs_bounds(obs)
                all_x += [xl, xh]
                all_y += [yl, yh]

    xl, xh = min(all_x), max(all_x)
    yl, yh = min(all_y), max(all_y)

    if xh - xl < 1e-4:
        xh += 0.05; xl -= 0.05
    if yh - yl < 1e-4:
        yh += 0.05; yl -= 0.05

    pad_x = LIMIT_PAD * (xh - xl)
    pad_y = LIMIT_PAD * (yh - yl)
    return xl - pad_x, xh + pad_x, yl - pad_y, yh + pad_y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install it (e.g. `apt install ffmpeg` "
              "or `conda install ffmpeg`) and re-run.", file=sys.stderr)
        sys.exit(1)

    fault_mode = args.fault_mode
    x0_state = jnp.array(args.x0)
    ctrl_dt = args.ctrl_dt
    ctrl_num_steps = args.ctrl_num_steps
    sim_dt = args.sim_dt
    azim = args.azim
    elev = args.elev
    fps = max(1, round(1.0 / sim_dt))

    total_time = ctrl_dt * ctrl_num_steps

    print(f"Controller : ctrl_dt={ctrl_dt} s  x  ctrl_num_steps={ctrl_num_steps}"
          f"  ->  {total_time:.2f} s time axis")
    print(f"Animation  : sim_dt={sim_dt:.4f} s  ->  {fps} fps")
    print(f"Camera     : azim={azim} deg  elev={elev} deg  (fixed)")

    # ── Problem setup ─────────────────────────────────────────────────────
    scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
    x0_ivl = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.1, 0.1, 0.1]))

    _mode_map = {
        "nominal": (0, 1.0),
        "actuator": (1, 0.25),   # midpoint of [0.0, 0.5]
        "sensor": (2, 1.0),
    }
    scenario_idx, alpha = _mode_map[fault_mode]
    traj_scenario = scenarios[scenario_idx]
    print(f"Fault mode : {traj_scenario.name}  (alpha={alpha})")
    print(f"x0         : {args.x0}")

    # ── Optimise (uses ctrl_dt and ctrl_num_steps) ─────────────────────────
    print("Optimising separating input ...")
    u_opt, loss_opt, _, _ = optimize_refined_gpu(
        x0_ivl=x0_ivl, scenarios=scenarios, dt=ctrl_dt,
        num_restarts=500, learning_rate=1, num_iters=50, seed=42,
        num_steps=ctrl_num_steps,
    )
    print(f"  loss={loss_opt:.6f}  u_opt shape={u_opt.shape}")

    # ── Simulate trajectory (uses ctrl_dt) ────────────────────────────────
    print("Simulating trajectory ...")
    traj = simulate_trajectory(x0_state, u_opt, alpha, ctrl_dt)
    ctrl_times = np.array([k * ctrl_dt for k in range(ctrl_num_steps + 1)])

    obs_off = np.array(traj_scenario.obs_offset)
    obs_scale = float(np.array(traj_scenario.obs_scale)[0])
    traj_obs_px = obs_scale * traj[:, 0] + obs_off[0]
    traj_obs_py = obs_scale * traj[:, 1] + obs_off[1]

    # ── Refinement history (uses ctrl_dt) -- shared helper, see module docstring ──
    print("Collecting refinement history ...")
    ref_steps, ref_pairs = collect_refinement_history(
        x0_ivl, u_opt, scenarios, ctrl_dt, ctrl_num_steps)
    print(f"  {len(ref_steps)} steps, {len(ref_pairs)} pairs")

    # ── Validity matrix ────────────────────────────────────────────────────
    print("Computing validity ...")
    valid = compute_validity(ref_steps, ref_pairs, traj_obs_px, traj_obs_py)
    for pi, (i, j) in enumerate(ref_pairs):
        inv = _invalidation_step(valid, pi)
        label = f"t={inv * ctrl_dt:.3f}s (step {inv})" if inv is not None else "never"
        print(f"  Pair ({scenarios[i].name}, {scenarios[j].name}): "
              f"invalidated at {label}")

    # ── Fixed px/py axis limits (computed once from all data) ─────────────
    print("Computing fixed axis limits ...")
    px_lo, px_hi, py_lo, py_hi = compute_fixed_limits(
        ref_steps, ref_pairs, traj_obs_px, traj_obs_py)
    print(f"  px in [{px_lo:.3f}, {px_hi:.3f}]  py in [{py_lo:.3f}, {py_hi:.3f}]")

    # ── Animation frame -> controller step mapping ─────────────────────────
    num_anim_frames = max(1, round(total_time / sim_dt)) + 1
    anim_times = np.linspace(0.0, total_time, num_anim_frames)
    ctrl_step_for_frame = np.round(
        anim_times / ctrl_dt
    ).astype(int).clip(0, ctrl_num_steps)

    print(f"Animation  : {num_anim_frames} frames  "
          f"(covers {total_time:.2f} s at {fps} fps)")

    # ── Colour setup ──────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.usetex': False,
    })
    cmap = plt.colormaps['tab10'].resampled(len(scenarios))
    traj_color = cmap(scenario_idx)

    pair_labels = [
        f"{scenarios[i].name} - {scenarios[j].name}"
        for i, j in ref_pairs
    ]

    # ── Figure setup ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # ── Frame update function ─────────────────────────────────────────────

    def update(frame_idx):
        ax.cla()

        ctrl_k = int(ctrl_step_for_frame[frame_idx])
        t_now = ctrl_times[ctrl_k]

        # ── Draw interval boxes for controller steps 0..ctrl_k-1 ─────────
        for s_idx in range(min(ctrl_k, len(ref_steps))):
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
                    poly = Poly3DCollection(verts, alpha=box_alpha)
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

        # ── Trajectory line up to ctrl_k ──────────────────────────────────
        if ctrl_k > 0:
            ax.plot(traj_obs_px[:ctrl_k + 1],
                    ctrl_times[:ctrl_k + 1],
                    traj_obs_py[:ctrl_k + 1],
                    color=traj_color, linewidth=2.5, linestyle='-',
                    marker='o', markersize=3, zorder=10,
                    label=f"Trajectory ({traj_scenario.name})")

        # ── Current measurement dot ───────────────────────────────────────
        ax.scatter([traj_obs_px[ctrl_k]], [t_now], [traj_obs_py[ctrl_k]],
                   color='red', s=80, zorder=15, depthshade=False,
                   label="Current measurement")

        # ── Axis limits: fixed px/py, full t span always visible ─────────
        ax.set_xlim(px_lo, px_hi)
        ax.set_ylim(0.0, total_time)
        ax.set_zlim(py_lo, py_hi)

        # ── Camera (fixed) ────────────────────────────────────────────────
        ax.view_init(elev=elev, azim=azim)

        # ── Labels ────────────────────────────────────────────────────────
        ax.set_xlabel('$p_x$ (m)', fontsize=10, labelpad=8)
        ax.set_ylabel('Time (s)', fontsize=11, labelpad=8)
        ax.set_zlabel('$p_y$ (m)', fontsize=10, labelpad=8)

        # ── Title: show valid pairs ───────────────────────────────────────
        if ctrl_k == 0:
            valid_labels = pair_labels[:]
        else:
            s_idx_now = min(ctrl_k - 1, len(ref_steps) - 1)
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
    print(f"Rendering {num_anim_frames} frames  "
          f"(interval={frame_interval_ms} ms, {fps} fps) ...")
    ani = FuncAnimation(fig, update, frames=num_anim_frames,
                        interval=frame_interval_ms, blit=False)

    out_path = args.output or str(
        HERE / f"refinement_anim_{fault_mode}_ctrl{ctrl_dt}x{ctrl_num_steps}.mp4"
    )
    ani.save(out_path, writer='ffmpeg', fps=fps, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
