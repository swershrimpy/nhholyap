"""
Plot figures showing the waypoint-based controller separation:
1. Per-pair observed-output overlap volume over the mission horizon.
2. Per-scenario observed-output reachable tubes (position) over time.
3. The waypoint/reference signal itself.

Requires results/synthesized_waypoints.npz (run synthesize_waypoints.py first).

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/plot_waypoint_separation.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import immrax as irx

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_waypoint_controllers import (
    create_scenarios, euler_step, nominal_waypoints,
    P0, QPS_DT,
)
from adaptive_spoofing.crazyflie_chain_controllers import (
    observed_output, _output_overlap_volume, _CANDIDATE_THETA,
)
from adaptive_spoofing.crazyflie_waypoint_trajectory import build_mission_reference

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CANDIDATE_COLORS = {
    "qps_snap_chain": "tab:blue",
    "pd_pos_vel": "tab:orange",
    "pid_pos_vel_i": "tab:green",
    "indi_jerk": "tab:red",
}


def _history(scenarios, x0_ivl, u_seq, dt):
    """Per-step, per-scenario state-interval history."""
    n = len(scenarios)
    num_steps = u_seq.shape[0]
    lowers = np.zeros((num_steps + 1, n, 18))
    uppers = np.zeros((num_steps + 1, n, 18))
    x_ivls = [x0_ivl] * n
    lowers[0] = np.stack([np.array(x0_ivl.lower)] * n)
    uppers[0] = np.stack([np.array(x0_ivl.upper)] * n)
    for k in range(num_steps):
        new_ivls = []
        for i, s in enumerate(scenarios):
            x_next = euler_step(s.emb_system, x_ivls[i], u_seq[k], s.p_interval, dt)
            new_ivls.append(x_next)
            lowers[k + 1, i] = np.array(x_next.lower)
            uppers[k + 1, i] = np.array(x_next.upper)
        x_ivls = new_ivls
    return lowers, uppers


def _pairwise_overlap_over_time(lowers, uppers, scenarios):
    n = len(scenarios)
    T = lowers.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    overlap = {f"{scenarios[i].name} vs {scenarios[j].name}": np.zeros(T) for i, j in pairs}
    for k in range(T):
        for i, j in pairs:
            ivl_i = irx.Interval(lower=jnp.array(lowers[k, i]), upper=jnp.array(uppers[k, i]))
            ivl_j = irx.Interval(lower=jnp.array(lowers[k, j]), upper=jnp.array(uppers[k, j]))
            overlap[f"{scenarios[i].name} vs {scenarios[j].name}"][k] = float(
                _output_overlap_volume(ivl_i, ivl_j))
    return overlap


def main():
    data = np.load(RESULTS_DIR / "synthesized_waypoints.npz", allow_pickle=True)
    waypoints = jnp.array(data["waypoints"])
    T_hop = float(data["T_hop"])
    dt = float(data["dt"])
    K = int(data["K"])
    x0_state4x3 = jnp.array(data["x0_state4x3"])
    x0_ivl = irx.Interval(lower=jnp.array(data["x0_ivl_lower"]),
                           upper=jnp.array(data["x0_ivl_upper"]))

    scenarios = create_scenarios()
    u_seq = build_mission_reference(waypoints, x0_state4x3, T_hop, dt)
    num_steps = u_seq.shape[0]
    t = np.arange(num_steps + 1) * dt

    print("Propagating reachable-set history under the optimized waypoints...")
    lowers, uppers = _history(scenarios, x0_ivl, u_seq, dt)

    overlap = _pairwise_overlap_over_time(lowers, uppers, scenarios)

    # ── Figure 1: pairwise overlap volume over time ───────────────────────
    pair_names = list(overlap.keys())
    n_pairs = len(pair_names)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flat
    for ax, name in zip(axes, pair_names):
        ax.plot(t, overlap[name], color="crimson")
        ax.set_title(name, fontsize=10)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.grid(alpha=0.3)
    for ax in list(axes)[-ncols:]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Waypoint-based controller discrimination: pairwise overlap", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "waypoint_pairwise_overlap.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'waypoint_pairwise_overlap.png'}")

    # ── Figure 2: per-scenario position reachable tubes ───────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    dim_labels = ["x position (m)", "y position (m)", "z position (m)"]
    for ax, label, dim in zip(axes, dim_labels, [0, 1, 2]):
        for i, s in enumerate(scenarios):
            color = CANDIDATE_COLORS.get(s.name, None)
            ax.fill_between(t, lowers[:, i, dim], uppers[:, i, dim],
                            alpha=0.35, color=color, label=s.name)
        # Overlay the reference position
        ref_pos = np.array(u_seq[:, dim])
        ax.plot(np.arange(1, num_steps + 1) * dt, ref_pos, 'k--', linewidth=0.8,
                label="reference" if dim == 0 else None)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Per-controller reachable position tubes under the waypoint mission", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "waypoint_reachable_tubes.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'waypoint_reachable_tubes.png'}")

    # ── Figure 3: waypoint signal (3D positions + nominal line) ───────────
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    nom = np.array(nominal_waypoints(K))
    wps = np.array(waypoints)

    # Plot nominal line
    ax.plot(nom[:, 0], nom[:, 1], nom[:, 2], 'k--', linewidth=1, label="nominal")
    ax.scatter(nom[:, 0], nom[:, 1], nom[:, 2], c='gray', s=30, zorder=5)

    # Plot optimized waypoints
    ax.plot(wps[:, 0], wps[:, 1], wps[:, 2], 'r-', linewidth=2, label="optimized")
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c='red', s=50, zorder=5)

    # Plot P0
    ax.scatter([float(P0[0])], [float(P0[1])], [float(P0[2])],
               c='green', s=80, marker='^', label="P0 (start)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Waypoint mission: nominal vs. optimized for discrimination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "waypoint_mission_3d.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'waypoint_mission_3d.png'}")

    plt.close("all")


if __name__ == "__main__":
    main()
