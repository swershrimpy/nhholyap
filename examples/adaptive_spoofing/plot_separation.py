"""
Plot figures showing the synthesized spoof bias's controller separation:
1. Per-pair observed-output overlap volume over the attack horizon, optimized
   bias vs. zero bias (no attack) baseline.
2. Per-scenario observed-output reachable tubes (position, velocity) over
   time, showing the four controllers' predicted trajectories diverging.

Requires results/synthesized_spoof.npz (run synthesize_spoof.py first).

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/plot_separation.py
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

from adaptive_spoofing.crazyflie_chain_controllers import (
    create_scenarios, hover_reference, QPS_DT, _CANDIDATE_THETA,
    euler_step, observed_output, _output_overlap_volume,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CANDIDATE_COLORS = {
    "qps_snap_chain": "tab:blue",
    "pd_pos_vel": "tab:orange",
    "pid_pos_vel_i": "tab:green",
    "indi_jerk": "tab:red",
}


def _history(scenarios, x0_ivl, u_seq, dt):
    """Per-step, per-scenario state-interval history (num_steps+1, n_scenarios, 18)."""
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
            overlap[f"{scenarios[i].name} vs {scenarios[j].name}"][k] = float(_output_overlap_volume(ivl_i, ivl_j))
    return overlap


def main():
    data = np.load(RESULTS_DIR / "synthesized_spoof.npz", allow_pickle=True)
    u_seq = jnp.array(data["u_seq"])
    dt = float(data["dt"])
    ref15 = jnp.array(data["ref15"])
    x0_ivl = irx.Interval(lower=jnp.array(data["x0_ivl_lower"]), upper=jnp.array(data["x0_ivl_upper"]))

    scenarios = create_scenarios(ref15)
    num_steps = u_seq.shape[0]
    t = np.arange(num_steps + 1) * dt

    print("Propagating reachable-set history under the optimized bias...")
    lowers_opt, uppers_opt = _history(scenarios, x0_ivl, u_seq, dt)
    print("Propagating reachable-set history under zero bias (no attack)...")
    lowers_zero, uppers_zero = _history(scenarios, x0_ivl, jnp.zeros_like(u_seq), dt)

    overlap_opt = _pairwise_overlap_over_time(lowers_opt, uppers_opt, scenarios)
    overlap_zero = _pairwise_overlap_over_time(lowers_zero, uppers_zero, scenarios)

    # ── Figure 1: pairwise overlap volume over time, optimized vs zero bias ──
    pair_names = list(overlap_opt.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, name in zip(axes.flat, pair_names):
        ax.plot(t, overlap_zero[name], color="gray", linestyle="--", label="zero bias (no attack)")
        ax.plot(t, overlap_opt[name], color="crimson", label="optimized spoof bias")
        ax.set_title(name, fontsize=10)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed-output\npairwise overlap volume")
    fig.suptitle("Controller-pair reachable-output overlap: optimized spoof vs. no attack", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pairwise_overlap.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'pairwise_overlap.png'}")

    # ── Figure 2: per-scenario position/velocity reachable tubes over time ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    dim_labels = [("x position (m)", 0), ("y position (m)", 1),
                 ("x velocity (m/s)", 3), ("y velocity (m/s)", 4)]
    for ax, (label, dim) in zip(axes.flat, dim_labels):
        for i, s in enumerate(scenarios):
            color = CANDIDATE_COLORS.get(s.name, None)
            ax.fill_between(t, lowers_opt[:, i, dim], uppers_opt[:, i, dim],
                            alpha=0.35, color=color, label=s.name)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Per-controller reachable-output tubes under the optimized spoof bias", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "reachable_tubes.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'reachable_tubes.png'}")

    # ── Figure 3: the spoof bias signal itself ──
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate(["bias_x", "bias_y", "bias_z"]):
        ax.plot(t[:-1], np.array(u_seq)[:, i], label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spoof position bias (m)")
    ax.set_title("Synthesized separating spoof-bias signal")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "spoof_bias_signal.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'spoof_bias_signal.png'}")


if __name__ == "__main__":
    main()
