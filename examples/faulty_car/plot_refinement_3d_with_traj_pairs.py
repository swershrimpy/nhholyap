"""
plot_refinement_3d_with_traj_pairs.py
======================================
Like plot_refinement_3d_with_traj.py but allows selecting which fault pairs to plot.

The user specifies:
  --fault-mode   {nominal, actuator, sensor}   fault mode the trajectory runs under
  --x0           px py phi                     initial state (default: 0.1 0.1 0.0)
  --pairs        PAIR1 PAIR2 ...               which pairs to plot (default: all)
                                               format: "Nominal-Actuator" or "0-1"

When --pairs is not given, all 3 pairs are plotted (4 panels: unrefined + 3 refined).
When --pairs is given, only those pairs are plotted (1 + len(pairs) panels).

Usage
-----
  python plot_refinement_3d_with_traj_pairs.py --fault-mode actuator --x0 0.1 0.1 0.0
  python plot_refinement_3d_with_traj_pairs.py --fault-mode actuator --pairs "Nominal-Actuator"
  python plot_refinement_3d_with_traj_pairs.py --fault-mode actuator --pairs "0-1" "1-2"
  python plot_refinement_3d_with_traj_pairs.py --fault-mode nominal --pairs "Nominal-Sensor"
"""

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="3-D refinement interval plot with trajectory overlay, selectable pairs."
    )
    p.add_argument(
        "--fault-mode",
        choices=["nominal", "actuator", "sensor"],
        default="nominal",
        help="Fault mode the trajectory is simulated under (default: nominal).",
    )
    p.add_argument(
        "--x0",
        nargs=3,
        type=float,
        metavar=("PX", "PY", "PHI"),
        default=[0.1, 0.1, 0.0],
        help="Initial state [px py phi] for the trajectory (default: 0.1 0.1 0.0).",
    )
    p.add_argument(
        "--pairs",
        nargs='*',
        default=None,
        help=(
            "Which pairs to plot. Format: 'Nominal-Actuator' or '0-1'. "
            "If not given, all pairs are plotted. "
            "Example: --pairs 'Nominal-Actuator' 'Actuator-Sensor'"
        ),
    )
    return p.parse_args()


# ── Trajectory simulation ─────────────────────────────────────────────────────

def simulate_trajectory(x0: jnp.ndarray, u_seq: jnp.ndarray,
                         alpha: float, dt: float) -> np.ndarray:
    """Euler-integrate CarNomActSystem from x0 using u_seq."""
    sys = CarNomActSystem()
    p   = jnp.array([alpha])
    x   = x0
    traj = [np.array(x)]
    for k in range(u_seq.shape[0]):
        dx = sys.f(jnp.zeros(()), x, u_seq[k], p)
        x  = x + dx * dt
        traj.append(np.array(x))
    return np.stack(traj)


# ── Refinement history ────────────────────────────────────────────────────────

def _intersect_obs(a, b):
    lo = jnp.maximum(a.lower, b.lower)
    hi = jnp.minimum(a.upper, b.upper)
    return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None


def collect_refinement_history(x0_ivl, u_seq, scenarios, dt, num_steps):
    n     = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    x1    = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
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
            xn_i = euler_step(scenarios[i].emb_system, xi_ref, u_seq[k + 1], scenarios[i].p_interval, dt)
            xn_j = euler_step(scenarios[j].emb_system, xj_ref, u_seq[k + 1], scenarios[j].p_interval, dt)
            on_i  = _obs_interval(xn_i, scenarios[i])
            on_j  = _obs_interval(xn_j, scenarios[j])
            new_pair_obs.append((on_i, on_j, _intersect_obs(on_i, on_j)))
            new_states.append((xn_i, xn_j))

        steps.append({'t': (k + 2) * dt, 'pair_obs': new_pair_obs})
        pair_states = new_states

    return steps, pairs


def collect_output_history(x0_ivl, u_seq, scenario, dt, num_steps):
    history = []
    x = x0_ivl
    for k in range(num_steps + 1):
        history.append(_obs_interval(x, scenario))
        if k < num_steps:
            x = euler_step(scenario.emb_system, x, u_seq[k], scenario.p_interval, dt)
    return history


# ── Pair parsing ──────────────────────────────────────────────────────────────

def parse_pair_spec(pair_spec, scenarios):
    """Parse a pair specification like 'Nominal-Actuator' or '0-1' into (i, j) indices.
    
    Returns (i, j) or raises ValueError.
    """
    scenario_names = [s.name for s in scenarios]
    
    # Try numeric format: "0-1"
    if '-' in pair_spec:
        parts = pair_spec.split('-')
        if len(parts) == 2:
            try:
                i, j = int(parts[0]), int(parts[1])
                if 0 <= i < len(scenarios) and 0 <= j < len(scenarios) and i != j:
                    return (min(i, j), max(i, j))
            except ValueError:
                pass
            
            # Try name format: "Nominal-Actuator"
            name_i, name_j = parts[0].strip(), parts[1].strip()
            try:
                i = scenario_names.index(name_i)
                j = scenario_names.index(name_j)
                if i != j:
                    return (min(i, j), max(i, j))
            except ValueError:
                pass
    
    raise ValueError(
        f"Invalid pair spec '{pair_spec}'. Use format '0-1' or 'Nominal-Actuator'. "
        f"Available scenarios: {', '.join(scenario_names)}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    fault_mode = args.fault_mode
    x0_state   = jnp.array(args.x0)

    # ── Problem setup ─────────────────────────────────────────────────────
    scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
    x0_ivl    = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.1, 0.1, 0.1]))
    dt        = 0.5
    num_steps = 10

    # Map CLI name → scenario index and point-mass alpha
    _mode_map = {
        "nominal":  (0, 1.0),
        "actuator": (1, 0.25),
        "sensor":   (2, 1.0),
    }
    scenario_idx, alpha = _mode_map[fault_mode]
    traj_scenario = scenarios[scenario_idx]
    print(f"Fault mode : {traj_scenario.name}  (alpha = {alpha})")
    print(f"Initial state: px={float(x0_state[0]):.3f}  py={float(x0_state[1]):.3f}  phi={float(x0_state[2]):.3f}")

    # ── Parse pair selection ──────────────────────────────────────────────
    all_pairs = [(i, j) for i in range(len(scenarios)) for j in range(i + 1, len(scenarios))]
    
    if args.pairs is None or len(args.pairs) == 0:
        # Plot all pairs
        selected_pairs = all_pairs
        print(f"Pairs: all {len(all_pairs)} pairs")
    else:
        # Parse user-specified pairs
        selected_pairs = []
        for pair_spec in args.pairs:
            try:
                pair = parse_pair_spec(pair_spec, scenarios)
                if pair not in selected_pairs:
                    selected_pairs.append(pair)
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
        print(f"Pairs: {len(selected_pairs)} selected")
        for i, j in selected_pairs:
            print(f"  - {scenarios[i].name} vs {scenarios[j].name}")

    # ── Optimise ──────────────────────────────────────────────────────────
    print("Optimising separating input …")
    u_opt, loss_opt, _, _ = optimize_refined_gpu(
        x0_ivl=x0_ivl, scenarios=scenarios, dt=dt,
        num_restarts=500, learning_rate=1, num_iters=50, seed=42, num_steps=num_steps,
    )
    print(f"  loss = {loss_opt:.6f},  u_opt shape = {u_opt.shape}")

    # ── Simulate trajectory ───────────────────────────────────────────────
    print("Simulating trajectory …")
    traj = simulate_trajectory(x0_state, u_opt, alpha, dt)
    traj_times = np.array([k * dt for k in range(num_steps + 1)])
    print(f"  trajectory shape = {traj.shape}")

    # ── Refinement history ────────────────────────────────────────────────
    print("Collecting refinement history …")
    ref_steps, ref_pairs = collect_refinement_history(x0_ivl, u_opt, scenarios, dt, num_steps)
    print(f"  {len(ref_steps)} steps, {len(ref_pairs)} pairs total")

    # ── Unrefined output histories ────────────────────────────────────────
    output_histories = [
        collect_output_history(x0_ivl, u_opt, s, dt, num_steps)
        for s in scenarios
    ]
    step_indices = np.arange(0, num_steps + 1, 1)
    times        = [k * dt for k in step_indices]

    # ── Font config ───────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{mathptmx}  % Times New Roman for LaTeX math/text
        """,
    })

    cmap = plt.colormaps['tab10'].resampled(len(scenarios))
    traj_color = cmap(scenario_idx)

    # ── Build figure ──────────────────────────────────────────────────────
    num_panels = 1 + len(selected_pairs)  # 1 unrefined + N refined
    fig, axs = plt.subplots(
        1, num_panels, figsize=(5 * num_panels, 5),
        subplot_kw={'projection': '3d'},
        gridspec_kw={'wspace': 0.02, 'hspace': 0},
    )
    
    # Handle single panel case (axs is not a list)
    if num_panels == 1:
        axs = [axs]

    all_px, all_py = [], []

    # ── Panel 0: unrefined intervals for all scenarios ────────────────────
    ax0 = axs[0]
    for idx, t in zip(step_indices, times):
        for k_sc, (hist, s) in enumerate(zip(output_histories, scenarios)):
            color = cmap(k_sc)
            iv    = hist[idx]
            xl, xh = float(iv.lower[0]), float(iv.upper[0])
            yl, yh = float(iv.lower[1]), float(iv.upper[1])
            all_px += [xl, xh]
            all_py += [yl, yh]
            verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
            poly  = Poly3DCollection(verts, alpha=0.35)
            poly.set_facecolor(color)
            poly.set_edgecolor(color)
            ax0.add_collection3d(poly)
        # pairwise overlap outlines
        intervals = [hist[idx] for hist in output_histories]
        for (p, q) in [(0, 1), (0, 2), (1, 2)]:
            iv1, iv2 = intervals[p], intervals[q]
            xl = float(max(iv1.lower[0], iv2.lower[0]))
            xh = float(min(iv1.upper[0], iv2.upper[0]))
            yl = float(max(iv1.lower[1], iv2.lower[1]))
            yh = float(min(iv1.upper[1], iv2.upper[1]))
            if xh >= xl and yh >= yl:
                ax0.plot([xl, xh, xh, xl, xl],
                         [t,  t,  t,  t,  t],
                         [yl, yl, yh, yh, yl],
                         color='black', linewidth=2.0, zorder=5)

    # ── Panels 1+: refined per selected pair ──────────────────────────────
    for panel_idx, (i, j) in enumerate(selected_pairs):
        ax_ref = axs[panel_idx + 1]
        
        # Find this pair's index in ref_pairs
        pair_idx = ref_pairs.index((i, j))
        
        for step in ref_steps:
            t = step['t']
            obs_i, obs_j, inter = step['pair_obs'][pair_idx]
            for obs, color in [(obs_i, cmap(i)), (obs_j, cmap(j))]:
                xl, xh = float(obs.lower[0]), float(obs.upper[0])
                yl, yh = float(obs.lower[1]), float(obs.upper[1])
                all_px += [xl, xh]
                all_py += [yl, yh]
                verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
                poly  = Poly3DCollection(verts, alpha=0.35)
                poly.set_facecolor(color)
                poly.set_edgecolor(color)
                ax_ref.add_collection3d(poly)
            if inter is not None:
                xl, xh = float(inter.lower[0]), float(inter.upper[0])
                yl, yh = float(inter.lower[1]), float(inter.upper[1])
                ax_ref.plot([xl, xh, xh, xl, xl],
                            [t,  t,  t,  t,  t],
                            [yl, yl, yh, yh, yl],
                            color='black', linewidth=2.0, zorder=5)

    # ── Overlay trajectory on all panels ─────────────────────────────────
    obs_off = np.array(traj_scenario.obs_offset)
    traj_px = traj[:, 0] + obs_off[0]
    traj_py = traj[:, 1] + obs_off[1]

    all_px += list(traj_px)
    all_py += list(traj_py)

    for ax in axs:
        ax.plot(
            traj_px, traj_times, traj_py,
            color=traj_color, linewidth=2.5, linestyle='-',
            marker='o', markersize=4, zorder=10,
            label=f"Trajectory ({traj_scenario.name})",
        )

    # ── Axis limits and labels ────────────────────────────────────────────
    pad    = 0.05
    t_vals = [s['t'] for s in ref_steps]
    for ax in axs:
        ax.set_xlim(min(all_px) - pad, max(all_px) + pad)
        ax.set_ylim(t_vals[0], t_vals[-1])
        ax.set_zlim(min(all_py) - pad, max(all_py) + pad)
        ax.set_xlabel('$p_x$ (m)', fontsize=10, labelpad=8)
        ax.set_ylabel('Time (s)', fontsize=11, labelpad=8)
        ax.set_zlabel('$p_y$ (m)', fontsize=10, labelpad=8)
        ax.view_init(elev=15, azim=20, roll=0)

    # ── Legend ────────────────────────────────────────────────────────────
    lines = [
        axs[0].plot([], [], [], color=cmap(0), linewidth=3, label=scenarios[0].name)[0],
        axs[0].plot([], [], [], color=cmap(1), linewidth=3, label=scenarios[1].name)[0],
        axs[0].plot([], [], [], color=cmap(2), linewidth=3, label=scenarios[2].name)[0],
        axs[0].plot([], [], [], color='black', linewidth=2, label='Intersection')[0],
        axs[0].plot([], [], [], color=traj_color, linewidth=2.5, linestyle='-',
                    marker='o', markersize=4,
                    label=rf"Trajectory ({traj_scenario.name}, $\alpha$={alpha})")[0],
    ]
    fig.legend(handles=lines, bbox_to_anchor=(0.50, 0.80), loc='center',
               borderaxespad=0., ncol=5, fontsize=9)

    # ── Panel labels ──────────────────────────────────────────────────────
    labels = ['(A) Original output-reachable intervals']
    for idx, (i, j) in enumerate(selected_pairs):
        label = f"({chr(66 + idx)}) {scenarios[i].name} vs {scenarios[j].name}\nAfter Refinement"
        labels.append(label)
    
    for idx, label in enumerate(labels):
        x = (idx + 0.6) / (num_panels + 1) + 0.05
        fig.text(x, 0.72, label, ha='center', fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ── Output filename ───────────────────────────────────────────────────
    pair_str = "_".join([f"{i}{j}" for i, j in selected_pairs]) if selected_pairs else "all"
    fname = HERE / f'refinement_3d_traj_pairs_{fault_mode}_{pair_str}_dt{dt}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {fname}")


if __name__ == "__main__":
    main()
