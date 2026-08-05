"""
animate_refinement_3d_rotating_twopanel.py
===========================================
Two-panel version of animate_refinement_3d_rotating.py.

Left panel  : Unrefined output-reachable intervals (Nominal + Actuator Fault)
Right panel : Refined intervals for the Nominal vs Actuator Fault pair

Both panels:
- Share the same trajectory, time axis, and freeze-and-rotate schedule
- Draw trajectory on both panels
- Use the same fixed camera (azim/elev) and same rotation sweep
- Use static axis limits computed once from all data (--dynamic-limits flag reserved for future)

Usage
-----
  python animate_refinement_3d_rotating_twopanel.py --fault-mode actuator --x0 0.1 0.1 0.0
  python animate_refinement_3d_rotating_twopanel.py --fault-mode actuator \\
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
DEFAULT_AZIM    = 45.0
DEFAULT_ELEV    = 20.0
LIMIT_PAD       = 0.05

CTRL_DT         = 1
CTRL_NUM_STEPS  = 5
DEFAULT_SIM_DT  = 0.033

DEFAULT_ROT_DEG = 90.0
DEFAULT_ROT_DUR = 1.5

# Nominal=0, Actuator Fault=1 — the pair we focus on
PAIR_I, PAIR_J  = 0, 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-panel 3-D refinement animation: unrefined vs refined (Nominal/Actuator)."
    )
    p.add_argument("--fault-mode", choices=["nominal", "actuator", "sensor"],
                   default="actuator")
    p.add_argument("--x0", nargs=3, type=float, metavar=("PX", "PY", "PHI"),
                   default=[0.1, 0.1, 0.0])
    p.add_argument("--ctrl-dt",        type=float, default=CTRL_DT)
    p.add_argument("--ctrl-num-steps", type=int,   default=CTRL_NUM_STEPS)
    p.add_argument("--sim-dt",         type=float, default=DEFAULT_SIM_DT)
    p.add_argument("--azim",           type=float, default=DEFAULT_AZIM)
    p.add_argument("--elev",           type=float, default=DEFAULT_ELEV)
    p.add_argument("--rot-deg",        type=float, default=DEFAULT_ROT_DEG)
    p.add_argument("--rot-dur",        type=float, default=DEFAULT_ROT_DUR)
    p.add_argument("--dpi",            type=int,   default=150)
    p.add_argument("--output",         type=str,   default=None)
    # Reserved for future dynamic axis limits
    p.add_argument("--dynamic-limits", action="store_true", default=False,
                   help="(Future) Enable dynamic per-frame axis limits. Currently ignored.")
    return p.parse_args()


# ── Physics helpers ───────────────────────────────────────────────────────────

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


def collect_unrefined_history(x0_ivl, u_seq, scenarios, dt, num_steps):
    """Propagate each scenario independently (no refinement). Returns list of
    dicts: steps[k]['t'], steps[k]['obs'][sc_idx] = Interval."""
    n = len(scenarios)
    xs = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt)
          for s in scenarios]
    steps = [{'t': dt, 'obs': [_obs_interval(xs[k], scenarios[k]) for k in range(n)]}]

    for step_k in range(num_steps):
        xs = [euler_step(s.emb_system, xs[k], u_seq[step_k + 1], s.p_interval, dt)
              for k, s in enumerate(scenarios)]
        steps.append({'t': (step_k + 2) * dt,
                      'obs': [_obs_interval(xs[k], scenarios[k]) for k in range(n)]})
    return steps


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


def _invalidation_step(valid, pi):
    for k in range(valid.shape[1]):
        if not valid[pi, k]:
            return k
    return None


# ── Static axis limits ────────────────────────────────────────────────────────

def compute_static_limits(unref_steps, ref_steps, ref_pairs, pair_idx,
                           traj_obs_px, traj_obs_py):
    """Compute fixed px/py limits from ALL data across both panels."""
    all_x, all_y = list(traj_obs_px), list(traj_obs_py)

    # Unrefined: Nominal (PAIR_I) and Actuator Fault (PAIR_J) only
    for step in unref_steps:
        for sc_idx in (PAIR_I, PAIR_J):
            obs = step['obs'][sc_idx]
            all_x += [float(obs.lower[0]), float(obs.upper[0])]
            all_y += [float(obs.lower[1]), float(obs.upper[1])]

    # Refined: the selected pair
    for step in ref_steps:
        obs_i, obs_j, _ = step['pair_obs'][pair_idx]
        for obs in (obs_i, obs_j):
            all_x += [float(obs.lower[0]), float(obs.upper[0])]
            all_y += [float(obs.lower[1]), float(obs.upper[1])]

    xl, xh = min(all_x), max(all_x)
    yl, yh = min(all_y), max(all_y)
    if xh - xl < 1e-4: xh += 0.05; xl -= 0.05
    if yh - yl < 1e-4: yh += 0.05; yl -= 0.05
    pad_x = LIMIT_PAD * (xh - xl)
    pad_y = LIMIT_PAD * (yh - yl)
    return xl - pad_x, xh + pad_x, yl - pad_y, yh + pad_y


# ── Frame schedule ────────────────────────────────────────────────────────────

def build_frame_schedule(ctrl_dt, ctrl_num_steps, sim_dt, rot_dur, rot_deg, fps):
    frames = []
    traj_frames_per_ctrl = max(1, round(ctrl_dt / sim_dt))
    rot_frames_total     = max(2, round(rot_dur * fps))
    half = rot_frames_total // 2
    running_traj = 0

    for ctrl_k in range(ctrl_num_steps):
        for f in range(traj_frames_per_ctrl):
            lim_alpha = (f + 1) / traj_frames_per_ctrl
            frames.append(dict(phase='travel', ctrl_k=ctrl_k+1,
                               traj_idx=running_traj, azim_offset=0.0,
                               show_text=False, limit_alpha=lim_alpha))
            running_traj += 1
        frozen_traj = running_traj - 1
        for f in range(rot_frames_total):
            if f < half:
                offset = rot_deg * (f / half)
            else:
                offset = rot_deg * (1.0 - (f - half) / max(1, rot_frames_total - half))
            frames.append(dict(phase='rotate', ctrl_k=ctrl_k+1,
                               traj_idx=frozen_traj, azim_offset=offset,
                               show_text=True, limit_alpha=1.0))
    return frames


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_unrefined_panel(ax, ctrl_k, traj_idx, azim_offset, show_text,
                          unref_steps, scenarios, cmap,
                          traj_obs_px, traj_obs_py, traj_times,
                          traj_color, traj_scenario,
                          px_lo, px_hi, py_lo, py_hi,
                          total_time, base_azim, elev):
    ax.cla()

    # Draw unrefined boxes for Nominal and Actuator Fault up to ctrl_k
    for s_idx in range(min(ctrl_k, len(unref_steps))):
        t_s = unref_steps[s_idx]['t']
        for sc_idx in (PAIR_I, PAIR_J):
            obs   = unref_steps[s_idx]['obs'][sc_idx]
            color = cmap(sc_idx)
            xl = float(obs.lower[0]); xh = float(obs.upper[0])
            yl = float(obs.lower[1]); yh = float(obs.upper[1])
            verts = [[(xl, t_s, yl), (xh, t_s, yl), (xh, t_s, yh), (xl, t_s, yh)]]
            poly  = Poly3DCollection(verts, alpha=0.35)
            poly.set_facecolor(color)
            poly.set_edgecolor((*matplotlib.colors.to_rgb(color), 0.35))
            ax.add_collection3d(poly)

        # Intersection outline between the two
        obs_i = unref_steps[s_idx]['obs'][PAIR_I]
        obs_j = unref_steps[s_idx]['obs'][PAIR_J]
        xl = float(max(obs_i.lower[0], obs_j.lower[0]))
        xh = float(min(obs_i.upper[0], obs_j.upper[0]))
        yl = float(max(obs_i.lower[1], obs_j.lower[1]))
        yh = float(min(obs_i.upper[1], obs_j.upper[1]))
        if xh >= xl and yh >= yl:
            ax.plot([xl, xh, xh, xl, xl], [t_s]*5, [yl, yl, yh, yh, yl],
                    color='black', linewidth=1.5, zorder=5)

    # Trajectory
    end = traj_idx + 1
    if end > 1:
        ax.plot(traj_obs_px[:end], traj_times[:end], traj_obs_py[:end],
                color=traj_color, linewidth=2.5, linestyle='-',
                marker='o', markersize=3, zorder=10)
    ax.scatter([traj_obs_px[traj_idx]], [traj_times[traj_idx]], [traj_obs_py[traj_idx]],
               color='red', s=80, zorder=15, depthshade=False)

    ax.set_xlim(px_lo, px_hi); ax.set_ylim(0.0, total_time); ax.set_zlim(py_lo, py_hi)
    ax.view_init(elev=elev, azim=base_azim + azim_offset)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.set_title('Unrefined', fontsize=11, pad=8)


def draw_refined_panel(ax, ctrl_k, traj_idx, azim_offset, show_text,
                        invalidation_msgs,
                        ref_steps, pair_idx, valid, scenarios, cmap,
                        traj_obs_px, traj_obs_py, traj_times,
                        traj_color, traj_scenario,
                        px_lo, px_hi, py_lo, py_hi,
                        total_time, base_azim, elev):
    ax.cla()

    for s_idx in range(min(ctrl_k, len(ref_steps))):
        t_s = ref_steps[s_idx]['t']
        inv_step = _invalidation_step(valid, pair_idx)
        if inv_step is not None and s_idx >= inv_step:
            box_alpha = 0.15
        else:
            box_alpha = 0.35

        obs_i, obs_j, inter = ref_steps[s_idx]['pair_obs'][pair_idx]
        for obs, color in [(obs_i, cmap(PAIR_I)), (obs_j, cmap(PAIR_J))]:
            xl = float(obs.lower[0]); xh = float(obs.upper[0])
            yl = float(obs.lower[1]); yh = float(obs.upper[1])
            verts = [[(xl, t_s, yl), (xh, t_s, yl), (xh, t_s, yh), (xl, t_s, yh)]]
            poly  = Poly3DCollection(verts, alpha=box_alpha)
            poly.set_facecolor(color)
            poly.set_edgecolor((*matplotlib.colors.to_rgb(color), box_alpha))
            ax.add_collection3d(poly)
        if inter is not None:
            xl = float(inter.lower[0]); xh = float(inter.upper[0])
            yl = float(inter.lower[1]); yh = float(inter.upper[1])
            ax.plot([xl, xh, xh, xl, xl], [t_s]*5, [yl, yl, yh, yh, yl],
                    color='black', linewidth=1.5, alpha=box_alpha/0.35, zorder=5)

    end = traj_idx + 1
    if end > 1:
        ax.plot(traj_obs_px[:end], traj_times[:end], traj_obs_py[:end],
                color=traj_color, linewidth=2.5, linestyle='-',
                marker='o', markersize=3, zorder=10)
    ax.scatter([traj_obs_px[traj_idx]], [traj_times[traj_idx]], [traj_obs_py[traj_idx]],
               color='red', s=80, zorder=15, depthshade=False)

    ax.set_xlim(px_lo, px_hi); ax.set_ylim(0.0, total_time); ax.set_zlim(py_lo, py_hi)
    ax.view_init(elev=elev, azim=base_azim + azim_offset)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')

    t_now = traj_times[traj_idx]
    inv   = _invalidation_step(valid, pair_idx)
    if inv is not None and ctrl_k > inv:
        status = f'Invalidated at t={inv * (t_now / max(ctrl_k, 1)):.2f}s'
    else:
        status = 'Valid'
    ax.set_title(f'Refined  |  t={t_now:.2f}s  |  {status}', fontsize=11, pad=8)

    if show_text and invalidation_msgs:
        msg = "\n".join(invalidation_msgs)
        ax.text2D(0.5, 0.02, msg, transform=ax.transAxes,
                  ha='center', va='bottom', fontsize=9, color='crimson',
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))


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

    print(f"Controller : ctrl_dt={ctrl_dt}s × {ctrl_num_steps} steps → {total_time:.2f}s")
    print(f"Rotate     : ±{rot_deg}° over {rot_dur}s  |  FPS={fps}")

    scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
    x0_ivl    = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.1, 0.1, 0.1]))

    _mode_map = {"nominal": (0, 1.0), "actuator": (1, 0.25), "sensor": (2, 1.0)}
    scenario_idx, alpha = _mode_map[fault_mode]
    traj_scenario = scenarios[scenario_idx]
    print(f"Fault mode : {traj_scenario.name}  (alpha={alpha})")

    print("Optimising …")
    u_opt, loss_opt, _, _ = optimize_refined_gpu(
        x0_ivl=x0_ivl, scenarios=scenarios, dt=ctrl_dt,
        num_restarts=500, learning_rate=1, num_iters=50, seed=42,
        num_steps=ctrl_num_steps,
    )
    print(f"  loss={loss_opt:.6f}")

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
    obs_off     = np.array(traj_scenario.obs_offset)
    traj_obs_px = traj[:, 0] + obs_off[0]
    traj_obs_py = traj[:, 1] + obs_off[1]

    print("Collecting histories …")
    unref_steps = collect_unrefined_history(x0_ivl, u_opt, scenarios, ctrl_dt, ctrl_num_steps)
    ref_steps, ref_pairs = collect_refinement_history(x0_ivl, u_opt, scenarios, ctrl_dt, ctrl_num_steps)
    pair_idx = ref_pairs.index((PAIR_I, PAIR_J))
    print(f"  {len(ref_steps)} steps  |  pair index={pair_idx} "
          f"({scenarios[PAIR_I].name} vs {scenarios[PAIR_J].name})")

    print("Computing validity …")
    valid = compute_validity(ref_steps, ref_pairs, traj_obs_px, traj_obs_py)
    inv   = _invalidation_step(valid, pair_idx)
    print(f"  Pair invalidated at: "
          f"{'t='+str(inv*ctrl_dt)+'s' if inv is not None else 'never'}")

    print("Computing static axis limits …")
    px_lo, px_hi, py_lo, py_hi = compute_static_limits(
        unref_steps, ref_steps, ref_pairs, pair_idx, traj_obs_px, traj_obs_py)
    print(f"  px=[{px_lo:.3f}, {px_hi:.3f}]  py=[{py_lo:.3f}, {py_hi:.3f}]")

    newly_invalidated = {}
    if inv is not None:
        k = inv + 1
        newly_invalidated[k] = [
            f"✗ {scenarios[PAIR_I].name} – {scenarios[PAIR_J].name} "
            f"invalidated at t={inv*ctrl_dt:.2f}s"
        ]

    schedule   = build_frame_schedule(ctrl_dt, ctrl_num_steps, sim_dt, rot_dur, rot_deg, fps)
    num_frames = len(schedule)
    print(f"Total frames: {num_frames}")

    plt.rcParams.update({'font.family': 'sans-serif', 'text.usetex': False})
    cmap       = plt.colormaps['tab10'].resampled(len(scenarios))
    traj_color = cmap(scenario_idx)

    fig = plt.figure(figsize=(16, 7))
    ax_left  = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')

    # Shared legend at top
    legend_handles = [
        plt.Line2D([0],[0], color=cmap(PAIR_I), linewidth=3,
                   label=scenarios[PAIR_I].name),
        plt.Line2D([0],[0], color=cmap(PAIR_J), linewidth=3,
                   label=scenarios[PAIR_J].name),
        plt.Line2D([0],[0], color='black', linewidth=1.5, label='Intersection'),
        plt.Line2D([0],[0], color=traj_color, linewidth=2.5,
                   marker='o', markersize=4, label=f'Trajectory ({traj_scenario.name})'),
        plt.Line2D([0],[0], color='red', marker='o', markersize=6,
                   linestyle='None', label='Current'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, 0.98))

    def update(frame_idx):
        fd         = schedule[frame_idx]
        ctrl_k     = fd['ctrl_k']
        traj_idx   = min(fd['traj_idx'], traj.shape[0] - 1)
        azim_off   = fd['azim_offset']
        show_text  = fd['show_text']
        inv_msgs   = newly_invalidated.get(ctrl_k, []) if show_text else []

        draw_unrefined_panel(
            ax_left, ctrl_k, traj_idx, azim_off, show_text,
            unref_steps, scenarios, cmap,
            traj_obs_px, traj_obs_py, traj_times,
            traj_color, traj_scenario,
            px_lo, px_hi, py_lo, py_hi,
            total_time, base_azim, elev,
        )
        draw_refined_panel(
            ax_right, ctrl_k, traj_idx, azim_off, show_text, inv_msgs,
            ref_steps, pair_idx, valid, scenarios, cmap,
            traj_obs_px, traj_obs_py, traj_times,
            traj_color, traj_scenario,
            px_lo, px_hi, py_lo, py_hi,
            total_time, base_azim, elev,
        )
        return []

    frame_interval_ms = round(sim_dt * 1000)
    print(f"Rendering {num_frames} frames at {fps} fps …")
    ani = FuncAnimation(fig, update, frames=num_frames,
                        interval=frame_interval_ms, blit=False)

    out_path = args.output or str(
        HERE / f"refinement_anim_twopanel_{fault_mode}_ctrl{ctrl_dt}x{ctrl_num_steps}.mp4"
    )
    ani.save(out_path, writer='ffmpeg', fps=fps, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
