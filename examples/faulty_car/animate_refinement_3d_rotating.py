"""
animate_refinement_3d_rotating.py
===================================
Like animate_refinement_3d_sparse.py but with a freeze-and-rotate pause
at every ctrl_dt step so the viewer can inspect the newly-added interval
boxes from multiple angles.

Frame structure (per ctrl_dt step k):
  1. TRAVEL segment  : trajectory advances from t_{k-1} to t_k at sim_dt fps
  2. FREEZE+ROTATE   : camera sweeps +D° then back to base azim over rot_dur seconds
     - new interval box for step k is revealed at the start of this segment
     - invalidation text shown if any pair was just invalidated
     - trajectory dot stays frozen at t_k

Axis limits interpolate linearly between ctrl_dt-step limits (same as sparse script).
All interval boxes from previous steps remain visible.

Usage
-----
  python animate_refinement_3d_rotating.py --fault-mode actuator --x0 0.1 0.1 0.0
  python animate_refinement_3d_rotating.py --fault-mode actuator \\
      --ctrl-dt 0.1 --ctrl-num-steps 50 --rot-deg 90 --rot-dur 1.5
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
DEFAULT_AZIM   = 45.0
DEFAULT_ELEV   = 20.0
LIMIT_PAD      = 0.02

CTRL_DT        = 1
CTRL_NUM_STEPS = 5
DEFAULT_SIM_DT = 0.033   # ~30 fps

DEFAULT_ROT_DEG = 90.0   # degrees to sweep out (and back) during freeze
DEFAULT_ROT_DUR = 1.5    # seconds for the full sweep (out + back)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="3-D refinement animation with freeze-and-rotate at each ctrl_dt step."
    )
    p.add_argument("--fault-mode", choices=["nominal", "actuator", "sensor"],
                   default="nominal")
    p.add_argument("--x0", nargs=3, type=float, metavar=("PX", "PY", "PHI"),
                   default=[0.1, 0.1, 0.0])
    p.add_argument("--ctrl-dt",        type=float, default=CTRL_DT)
    p.add_argument("--ctrl-num-steps", type=int,   default=CTRL_NUM_STEPS)
    p.add_argument("--sim-dt",         type=float, default=DEFAULT_SIM_DT,
                   help="Animation frame interval (s). Video FPS = round(1/sim_dt).")
    p.add_argument("--azim",  type=float, default=DEFAULT_AZIM)
    p.add_argument("--elev",  type=float, default=DEFAULT_ELEV)
    p.add_argument("--rot-deg", type=float, default=DEFAULT_ROT_DEG,
                   help="Degrees to sweep during freeze pause (default 90).")
    p.add_argument("--rot-dur", type=float, default=DEFAULT_ROT_DUR,
                   help="Duration of freeze+rotate pause in seconds (default 1.5).")
    p.add_argument("--dpi",    type=int,   default=150)
    p.add_argument("--output", type=str,   default=None)
    return p.parse_args()


# ── Physics helpers (identical to sparse script) ──────────────────────────────

def simulate_trajectory(x0, u_seq, alpha, dt):
    sys_obj = CarNomActSystem()
    p = jnp.array([alpha])
    x = x0
    traj = [np.array(x)]
    for k in range(u_seq.shape[0]):
        dx = sys_obj.f(jnp.zeros(()), x, u_seq[k], p)
        x  = x + dx * dt
        traj.append(np.array(x))
    return np.stack(traj)


def _intersect_obs(a, b):
    lo = jnp.maximum(a.lower, b.lower)
    hi = jnp.minimum(a.upper, b.upper)
    return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None


def collect_refinement_history(x0_ivl, u_seq, scenarios, dt, num_steps):
    n     = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    x1    = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt)
             for s in scenarios]
    obs1  = [_obs_interval(xi, s) for xi, s in zip(x1, scenarios)]
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
            hov  = bool(jnp.all(y_hi >= y_lo))
            fb   = (xi.lower[:2] + xi.upper[:2]) / 2
            ys_lo = jnp.where(hov, y_lo, fb)
            ys_hi = jnp.where(hov, y_hi, fb)
            xi_ref = irx.Interval(
                lower=jnp.array([ys_lo[0]-scenarios[i].obs_offset[0],
                                  ys_lo[1]-scenarios[i].obs_offset[1], xi.lower[2]]),
                upper=jnp.array([ys_hi[0]-scenarios[i].obs_offset[0],
                                  ys_hi[1]-scenarios[i].obs_offset[1], xi.upper[2]]),
            )
            xj_ref = irx.Interval(
                lower=jnp.array([ys_lo[0]-scenarios[j].obs_offset[0],
                                  ys_lo[1]-scenarios[j].obs_offset[1], xj.lower[2]]),
                upper=jnp.array([ys_hi[0]-scenarios[j].obs_offset[0],
                                  ys_hi[1]-scenarios[j].obs_offset[1], xj.upper[2]]),
            )
            xn_i = euler_step(scenarios[i].emb_system, xi_ref, u_seq[k+1],
                              scenarios[i].p_interval, dt)
            xn_j = euler_step(scenarios[j].emb_system, xj_ref, u_seq[k+1],
                              scenarios[j].p_interval, dt)
            on_i = _obs_interval(xn_i, scenarios[i])
            on_j = _obs_interval(xn_j, scenarios[j])
            new_pair_obs.append((on_i, on_j, _intersect_obs(on_i, on_j)))
            new_states.append((xn_i, xn_j))
        steps.append({'t': (k+2)*dt, 'pair_obs': new_pair_obs})
        pair_states = new_states
    return steps, pairs


def compute_validity(ref_steps, ref_pairs, traj_obs_px, traj_obs_py):
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
            yx = traj_obs_px[k+1]; yy = traj_obs_py[k+1]
            inside_i = (float(obs_i.lower[0]) <= yx <= float(obs_i.upper[0]) and
                        float(obs_i.lower[1]) <= yy <= float(obs_i.upper[1]))
            inside_j = (float(obs_j.lower[0]) <= yx <= float(obs_j.upper[0]) and
                        float(obs_j.lower[1]) <= yy <= float(obs_j.upper[1]))
            if not (inside_i and inside_j):
                still_valid = False
                valid[pi, k] = False
    return valid


def _obs_bounds(obs):
    return (float(obs.lower[0]), float(obs.upper[0]),
            float(obs.lower[1]), float(obs.upper[1]))


def _invalidation_step(valid, pi):
    for k in range(valid.shape[1]):
        if not valid[pi, k]:
            return k
    return None


# ── Axis-limit helpers ────────────────────────────────────────────────────────

def compute_limits_at_ctrl_step(ctrl_k, ref_steps, ref_pairs, valid,
                                traj_obs_px, traj_obs_py):
    all_x = list(traj_obs_px[:ctrl_k+1])
    all_y = list(traj_obs_py[:ctrl_k+1])
    for s_idx in range(min(ctrl_k, len(ref_steps))):
        for pi in range(len(ref_pairs)):
            obs_i, obs_j, _ = ref_steps[s_idx]['pair_obs'][pi]
            for obs in (obs_i, obs_j):
                xl, xh, yl, yh = _obs_bounds(obs)
                all_x += [xl, xh]; all_y += [yl, yh]
    if not all_x:
        return -0.5, 0.5, -0.5, 0.5
    xl, xh = min(all_x), max(all_x)
    yl, yh = min(all_y), max(all_y)
    if xh - xl < 1e-4: xh += 0.05; xl -= 0.05
    if yh - yl < 1e-4: yh += 0.05; yl -= 0.05
    pad_x = LIMIT_PAD * (xh - xl)
    pad_y = LIMIT_PAD * (yh - yl)
    return xl - pad_x, xh + pad_x, yl - pad_y, yh + pad_y


# ── Frame schedule builder ────────────────────────────────────────────────────

def build_frame_schedule(ctrl_dt, ctrl_num_steps, sim_dt, rot_dur, rot_deg, fps):
    """Return a list of frame descriptors.

    Each entry is a dict:
      phase      : 'travel' | 'rotate'
      ctrl_k     : which ctrl_dt step we are AT (intervals drawn up to here)
      traj_idx   : index into traj arrays (frozen during rotate)
      azim_offset: degrees added to base azim  (0 during travel)
      show_text  : bool – show invalidation text this frame
      limit_alpha: float in [0,1] for interpolating limits between ctrl_k-1 and ctrl_k
    """
    frames = []
    traj_frames_per_ctrl = max(1, round(ctrl_dt / sim_dt))  # travel frames per step
    rot_frames_total     = max(2, round(rot_dur * fps))      # total rotate frames
    half = rot_frames_total // 2

    running_traj = 0

    for ctrl_k in range(ctrl_num_steps):
        # ── TRAVEL: advance trajectory from ctrl_k to ctrl_k+1 ──────────
        for f in range(traj_frames_per_ctrl):
            lim_alpha = (f + 1) / traj_frames_per_ctrl   # 0→1 toward ctrl_k+1 limits
            frames.append(dict(
                phase='travel',
                ctrl_k=ctrl_k + 1,
                traj_idx=running_traj,
                azim_offset=0.0,
                show_text=False,
                limit_alpha=lim_alpha,
            ))
            running_traj += 1

        # ── ROTATE: freeze at ctrl_k+1, sweep camera ────────────────────
        frozen_traj = running_traj - 1
        for f in range(rot_frames_total):
            if f < half:
                offset = rot_deg * (f / half)
            else:
                offset = rot_deg * (1.0 - (f - half) / max(1, rot_frames_total - half))
            frames.append(dict(
                phase='rotate',
                ctrl_k=ctrl_k + 1,
                traj_idx=frozen_traj,
                azim_offset=offset,
                show_text=True,
                limit_alpha=1.0,
            ))

    return frames


# ── Draw scene (shared between travel and rotate) ─────────────────────────────

def draw_scene(ax, ctrl_k, traj_idx, azim_offset,
               show_text, invalidation_msgs,
               ref_steps, ref_pairs, valid, scenarios, cmap,
               traj_obs_px, traj_obs_py, traj_times,
               traj_color, traj_scenario,
               px_lo, px_hi, py_lo, py_hi,
               total_time, base_azim, elev, pair_labels):

    ax.cla()

    # ── Interval boxes: all steps up to ctrl_k ───────────────────────────
    for s_idx in range(min(ctrl_k, len(ref_steps))):
        t_s = ref_steps[s_idx]['t']
        for pi, (i, j) in enumerate(ref_pairs):
            inv_step = _invalidation_step(valid, pi)
            if inv_step is not None and s_idx >= inv_step:
                box_alpha = 0.15   # dim but keep visible
            else:
                box_alpha = 0.35

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

            if inter is not None:
                xl = float(inter.lower[0]); xh = float(inter.upper[0])
                yl = float(inter.lower[1]); yh = float(inter.upper[1])
                ax.plot([xl, xh, xh, xl, xl],
                        [t_s]*5, [yl, yl, yh, yh, yl],
                        color='black', linewidth=1.5, alpha=box_alpha/0.35, zorder=5)

    # ── Trajectory up to traj_idx ─────────────────────────────────────────
    end = traj_idx + 1
    if end > 1:
        ax.plot(traj_obs_px[:end], traj_times[:end], traj_obs_py[:end],
                color=traj_color, linewidth=2.5, linestyle='-',
                marker='o', markersize=3, zorder=10)

    # ── Current dot ───────────────────────────────────────────────────────
    ax.scatter([traj_obs_px[traj_idx]], [traj_times[traj_idx]], [traj_obs_py[traj_idx]],
               color='red', s=80, zorder=15, depthshade=False)

    # ── Axis limits ───────────────────────────────────────────────────────
    ax.set_xlim(px_lo, px_hi)
    ax.set_ylim(0.0, total_time)
    ax.set_zlim(py_lo, py_hi)

    # ── Camera ────────────────────────────────────────────────────────────
    ax.view_init(elev=elev, azim=base_azim + azim_offset)

    # ── No axis labels ────────────────────────────────────────────────────
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')

    # ── Title ─────────────────────────────────────────────────────────────
    t_now = traj_times[traj_idx]
    if ctrl_k == 0:
        valid_labels = pair_labels[:]
    else:
        s_idx_now = min(ctrl_k - 1, len(ref_steps) - 1)
        valid_labels = [pair_labels[pi] for pi in range(len(ref_pairs))
                        if valid[pi, s_idx_now]]
    valid_str = ", ".join(valid_labels) if valid_labels else "None (fault isolated!)"
    ax.set_title(f"t = {t_now:.2f} s  |  Valid: {valid_str}", fontsize=10, pad=10)

    # ── Invalidation text (only during rotate phase) ──────────────────────
    if show_text and invalidation_msgs:
        msg = "\n".join(invalidation_msgs)
        ax.text2D(0.5, 0.02, msg,
                  transform=ax.transAxes,
                  ha='center', va='bottom', fontsize=9,
                  color='crimson',
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    # ── Legend ────────────────────────────────────────────────────────────
    handles = [plt.Line2D([0],[0], color=cmap(k), linewidth=3, label=s.name)
               for k, s in enumerate(scenarios)]
    handles += [
        plt.Line2D([0],[0], color='black', linewidth=1.5, label='Intersection'),
        plt.Line2D([0],[0], color=traj_color, linewidth=2.5,
                   marker='o', markersize=4, label=f"Traj ({traj_scenario.name})"),
        plt.Line2D([0],[0], color='red', marker='o', markersize=6,
                   linestyle='None', label='Current'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=8,
              bbox_to_anchor=(0.0, 1.0))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found.", file=sys.stderr); sys.exit(1)

    fault_mode     = args.fault_mode
    x0_state       = jnp.array(args.x0)
    ctrl_dt        = args.ctrl_dt
    ctrl_num_steps = args.ctrl_num_steps
    sim_dt         = args.sim_dt
    base_azim      = args.azim
    elev           = args.elev
    rot_deg        = args.rot_deg
    rot_dur        = args.rot_dur
    fps            = max(1, round(1.0 / sim_dt))
    total_time     = ctrl_dt * ctrl_num_steps

    print(f"Controller : ctrl_dt={ctrl_dt} s × ctrl_num_steps={ctrl_num_steps} → {total_time:.2f} s")
    print(f"FPS        : {fps}  sim_dt={sim_dt} s")
    print(f"Rotate     : ±{rot_deg}° over {rot_dur} s at each ctrl_dt step")

    # ── Setup ─────────────────────────────────────────────────────────────
    scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
    x0_ivl    = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.1, 0.1, 0.1]))

    _mode_map = {"nominal": (0, 1.0), "actuator": (1, 0.25), "sensor": (2, 1.0)}
    scenario_idx, alpha = _mode_map[fault_mode]
    traj_scenario = scenarios[scenario_idx]
    print(f"Fault mode : {traj_scenario.name}  (alpha={alpha})")

    # ── Optimise ──────────────────────────────────────────────────────────
    print("Optimising separating input …")
    u_opt, loss_opt, _, _ = optimize_refined_gpu(
        x0_ivl=x0_ivl, scenarios=scenarios, dt=ctrl_dt,
        num_restarts=500, learning_rate=1, num_iters=50, seed=42,
        num_steps=ctrl_num_steps,
    )
    print(f"  loss={loss_opt:.6f}  u_opt shape={u_opt.shape}")

    # ── Simulate trajectory ────────────────────────────────────────────────
    print("Simulating trajectory …")
    num_traj_steps = int(total_time / sim_dt) + 1
    traj = simulate_trajectory(x0_state, u_opt, alpha, sim_dt)
    if traj.shape[0] < num_traj_steps:
        sys_obj = CarNomActSystem()
        x = traj[-1]
        for _ in range(num_traj_steps - traj.shape[0]):
            dx = sys_obj.f(jnp.zeros(()), x, u_opt[-1], jnp.array([alpha]))
            x  = x + dx * sim_dt
            traj = np.vstack([traj, x])
    traj_times = np.array([k * sim_dt for k in range(traj.shape[0])])
    obs_off = np.array(traj_scenario.obs_offset)
    traj_obs_px = traj[:, 0] + obs_off[0]
    traj_obs_py = traj[:, 1] + obs_off[1]

    # ── Refinement history ─────────────────────────────────────────────────
    print("Collecting refinement history …")
    ref_steps, ref_pairs = collect_refinement_history(
        x0_ivl, u_opt, scenarios, ctrl_dt, ctrl_num_steps)
    print(f"  {len(ref_steps)} steps, {len(ref_pairs)} pairs")

    # ── Validity ───────────────────────────────────────────────────────────
    print("Computing validity …")
    valid = compute_validity(ref_steps, ref_pairs, traj_obs_px, traj_obs_py)
    pair_labels = [f"{scenarios[i].name} – {scenarios[j].name}" for i, j in ref_pairs]
    for pi, (i, j) in enumerate(ref_pairs):
        inv = _invalidation_step(valid, pi)
        label = f"t={inv*ctrl_dt:.3f}s (step {inv})" if inv is not None else "never"
        print(f"  Pair ({scenarios[i].name}, {scenarios[j].name}): invalidated at {label}")

    # ── Precompute axis limits at each ctrl_dt step ────────────────────────
    print("Precomputing axis limits …")
    ctrl_limits = np.array([
        compute_limits_at_ctrl_step(k, ref_steps, ref_pairs, valid,
                                    traj_obs_px, traj_obs_py)
        for k in range(ctrl_num_steps + 1)
    ])  # shape (ctrl_num_steps+1, 4)

    # ── Build frame schedule ───────────────────────────────────────────────
    schedule = build_frame_schedule(ctrl_dt, ctrl_num_steps, sim_dt, rot_dur, rot_deg, fps)
    num_frames = len(schedule)
    print(f"Total frames: {num_frames}  "
          f"(travel={ctrl_num_steps * max(1,round(ctrl_dt/sim_dt))}, "
          f"rotate={ctrl_num_steps * max(2,round(rot_dur*fps))})")

    # ── Precompute per-frame invalidation messages ─────────────────────────
    # For each ctrl_k, collect pairs that were JUST invalidated at that step
    newly_invalidated = {}  # ctrl_k → list of messages
    for pi, (i, j) in enumerate(ref_pairs):
        inv = _invalidation_step(valid, pi)
        if inv is not None:
            k = inv + 1   # ctrl_k when this becomes visible (step inv → drawn at ctrl_k=inv+1)
            newly_invalidated.setdefault(k, []).append(
                f"✗ {scenarios[i].name} – {scenarios[j].name} invalidated at t={inv*ctrl_dt:.2f}s"
            )

    # ── Colour setup ──────────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'sans-serif', 'text.usetex': False})
    cmap       = plt.colormaps['tab10'].resampled(len(scenarios))
    traj_color = cmap(scenario_idx)

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        fd = schedule[frame_idx]
        ctrl_k     = fd['ctrl_k']
        traj_idx   = min(fd['traj_idx'], traj.shape[0] - 1)
        azim_off   = fd['azim_offset']
        show_text  = fd['show_text']
        lim_alpha  = fd['limit_alpha']

        # Interpolate axis limits between ctrl_k-1 and ctrl_k
        k_lo = max(ctrl_k - 1, 0)
        k_hi = min(ctrl_k, ctrl_num_steps)
        lims = (1.0 - lim_alpha) * ctrl_limits[k_lo] + lim_alpha * ctrl_limits[k_hi]
        px_lo, px_hi, py_lo, py_hi = lims

        inv_msgs = newly_invalidated.get(ctrl_k, []) if show_text else []

        draw_scene(
            ax, ctrl_k, traj_idx, azim_off,
            show_text, inv_msgs,
            ref_steps, ref_pairs, valid, scenarios, cmap,
            traj_obs_px, traj_obs_py, traj_times,
            traj_color, traj_scenario,
            px_lo, px_hi, py_lo, py_hi,
            total_time, base_azim, elev, pair_labels,
        )
        return []

    frame_interval_ms = round(sim_dt * 1000)
    print(f"Rendering {num_frames} frames at {fps} fps …")
    ani = FuncAnimation(fig, update, frames=num_frames,
                        interval=frame_interval_ms, blit=False)

    out_path = args.output or str(
        HERE / f"refinement_anim_rotating_{fault_mode}_ctrl{ctrl_dt}x{ctrl_num_steps}.mp4"
    )
    ani.save(out_path, writer='ffmpeg', fps=fps, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
