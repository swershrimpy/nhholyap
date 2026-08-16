"""
Verify the synthesized firmware-controller spoof bias by SIMULATION.

run_firmware_discrimination.py already checks discrimination using
`discriminate_controller` against a point-simulated "true" trajectory
(`simulate_true_trajectory`) -- but that check is folded into a larger
script and only prints pass/fail. This script isolates that verification
step and makes it visible: for each of the 4 REAL firmware controllers in
turn, simulate what the DRONE WOULD ACTUALLY DO (a plain forward rollout of
that controller's exact closed-loop equations, using its real
firmware-derived gains -- no intervals, no reachability abstraction) under
the SAME optimized spoof-bias sequence, plot all 4 resulting trajectories
together, and confirm they're visibly distinct AND correctly discriminated.

Note on fidelity: this is NOT a CrazySim (Gazebo/ROS2 firmware-SITL)
simulation -- that would exercise the actual compiled firmware binary, but
needs a from-scratch Docker build (ROS2 Humble + Gazebo Garden + firmware
compile) that didn't fit this machine's current disk headroom (26GB free,
94% used). This script instead simulates the SAME closed-loop equations
`crazyflie_firmware_controllers.py` already uses for reachability (imported
directly from rq3_crazyflie_surrogates.py, the project's own source of
truth for these gains) via a plain point rollout -- the "plant + real
control law" fidelity is identical; what's missing is the firmware
binary/physics-engine layer underneath it.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/verify_firmware_simulation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_firmware_controllers import (
    create_scenarios, CANDIDATE_NAMES, QPS_DT,
    simulate_true_trajectory, discriminate_controller,
)
import immrax as irx

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CANDIDATE_COLORS = {
    "cf_pid": "tab:blue", "cf_mellinger": "tab:orange",
    "cf_indi": "tab:green", "cf_brescianini": "tab:red",
}

OBSERVABLE_WIDTH = jnp.concatenate([
    jnp.full(3, 0.05), jnp.full(3, 0.05), jnp.full(3, 0.017), jnp.full(3, 0.05), jnp.full(9, 0.01),
])
W_BAR = 1e-3


def main():
    with open(RESULTS_DIR / "firmware_discrimination_result.json") as f:
        result = json.load(f)
    u_seq = jnp.array(result["wide_regime"]["u_star"])
    num_steps = u_seq.shape[0]
    print(f"Loaded synthesized spoof bias: shape={u_seq.shape}, "
         f"max|bias|={float(jnp.max(jnp.abs(u_seq))):.4f} m")

    scenarios = create_scenarios()
    x0_point = jnp.zeros(21)
    x0_ivl = irx.icentpert(jnp.zeros(21), OBSERVABLE_WIDTH)

    # ── Simulate all 4 controllers' ACTUAL closed-loop response ──
    trajectories = {}
    for name in CANDIDATE_NAMES:
        scen = [s for s in scenarios if s.name == name][0]
        traj = simulate_true_trajectory(x0_point, u_seq, scen)   # (T, 12)
        trajectories[name] = np.array(traj)
        print(f"{name:16s} final position: {traj[-1, 0:3]}  final attitude: {traj[-1, 6:9]}")

    # ── Confirm the 4 simulated trajectories are actually distinguishable ──
    t = np.arange(1, num_steps + 1) * QPS_DT
    final_positions = np.stack([trajectories[n][-1, 0:3] for n in CANDIDATE_NAMES])
    pairwise_final_dist = {}
    for i, ni in enumerate(CANDIDATE_NAMES):
        for j, nj in enumerate(CANDIDATE_NAMES):
            if j > i:
                d = float(np.linalg.norm(final_positions[i] - final_positions[j]))
                pairwise_final_dist[f"{ni} vs {nj}"] = d
    print("\nFinal-position separation between simulated trajectories:")
    for k, v in pairwise_final_dist.items():
        print(f"  {k}: {v*1000:.3f} mm")
    min_sep = min(pairwise_final_dist.values())

    # ── Plot: position + attitude trajectories, all 4 controllers overlaid ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    labels = [("x position (m)", 0), ("y position (m)", 1), ("z position (m)", 2),
             ("roll (rad)", 6), ("pitch (rad)", 7), ("yaw (rad)", 8)]
    for ax, (label, dim) in zip(axes.flat, labels):
        for name in CANDIDATE_NAMES:
            ax.plot(t, trajectories[name][:, dim], color=CANDIDATE_COLORS[name],
                   label=name, linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Simulated closed-loop response of all 4 real firmware controllers\n"
                "under the SAME synthesized spoof-bias sequence", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "firmware_simulated_trajectories.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'firmware_simulated_trajectories.png'}")

    # ── Re-confirm discrimination against each simulated trajectory ──
    print("\nDiscrimination verdict, reacting to each simulated trajectory in turn:")
    all_ok = True
    per_controller = {}
    for true_name in CANDIDATE_NAMES:
        observed = jnp.array(trajectories[true_name])
        verdict = discriminate_controller(x0_ivl, u_seq, observed, scenarios, w_bar=W_BAR)
        ok = verdict["survivors"] == [true_name]
        all_ok &= ok
        per_controller[true_name] = {
            "survivors": verdict["survivors"], "fail_step": verdict["fail_step"], "correct": ok,
        }
        print(f"  true={true_name:16s}  survivors={verdict['survivors']}  "
             f"fail_step={verdict['fail_step']}  {'OK' if ok else 'WRONG'}")

    print(f"\nMinimum final-position separation across all pairs: {min_sep*1000:.3f} mm")
    print("VERDICT:", "PASS -- trajectory is suitable for discrimination" if all_ok and min_sep > 1e-6
         else "CHECK -- see per-controller verdicts above")

    with open(RESULTS_DIR / "firmware_simulation_verification.json", "w") as f:
        json.dump({
            "u_seq": np.array(u_seq).tolist(),
            "final_positions": {n: trajectories[n][-1, 0:3].tolist() for n in CANDIDATE_NAMES},
            "pairwise_final_separation_m": pairwise_final_dist,
            "min_separation_m": min_sep,
            "discrimination_verdicts": per_controller,
            "all_correct": all_ok,
            "note": "Point-simulation verification (real controller equations, no CrazySim/Gazebo/firmware-binary layer -- see module docstring).",
        }, f, indent=2)
    print(f"Saved {RESULTS_DIR / 'firmware_simulation_verification.json'}")


if __name__ == "__main__":
    main()
