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
    propagate_history,
)
import immrax as irx

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CANDIDATE_COLORS = {
    "cf_pid": "tab:blue", "cf_mellinger": "tab:orange",
    "cf_indi": "tab:green", "cf_brescianini": "tab:red",
}

# WIDE regime: realistic per-axis-scaled uncertainty (position/velocity ~5cm(/s),
# attitude ~1deg, rates ~0.05rad/s, memory generously loose) -- see
# run_firmware_discrimination.py's WIDE_WIDTH, same values.
WIDE_WIDTH = jnp.concatenate([
    jnp.full(3, 0.05), jnp.full(3, 0.05), jnp.full(3, 0.017), jnp.full(3, 0.05), jnp.full(9, 0.01),
])
# TIGHT regime: a well-converged state estimate -- see run_firmware_discrimination.py's
# TIGHT_WIDTH, same value. Under this uncertainty the 4 controllers separate
# instantly (loss_at_zero_bias=0.0 in results/firmware_discrimination_result.json's
# tight_regime), so no synthesized bias is needed -- plotted with zero bias.
TIGHT_WIDTH = 1e-3
# Must match run_firmware_discrimination.py's NUM_STEPS -- both regimes there
# use the same horizon.
NUM_STEPS = 3
W_BAR = 1e-3


def _simulate_and_plot(regime_name, x0_ivl, u_seq, scenarios, out_stem, title_suffix, json_stem=None):
    """Simulate all 4 controllers' closed-loop response under `u_seq` from
    prior `x0_ivl`, overlay each controller's own output-reachable tube,
    save the plot, re-confirm discrimination, and save the verification
    JSON. Shared by both the WIDE (optimized-bias) and TIGHT (zero-bias)
    regimes so they can't independently drift out of sync."""
    num_steps = u_seq.shape[0]
    x0_point = jnp.zeros(21)
    print(f"\n{'=' * 70}\n{regime_name}: max|bias|={float(jnp.max(jnp.abs(u_seq))):.4f} m, "
         f"num_steps={num_steps}")

    # ── Simulate all 4 controllers' ACTUAL closed-loop response, and (for the
    # SAME scenario) the output-reachable tube each controller model predicts
    # from the shared x0_ivl prior under the same bias sequence -- both
    # prepended with the shared t=0 state/box so the line and its tube start
    # at the same point. ──
    trajectories = {}        # (T, 12) -- used as-is by discriminate_controller below
    trajectories_full = {}   # (T+1, 12), t=0 state prepended -- used for plotting only
    tube_lower = {}
    tube_upper = {}
    for name in CANDIDATE_NAMES:
        scen = [s for s in scenarios if s.name == name][0]
        traj = simulate_true_trajectory(x0_point, u_seq, scen)   # (T, 12)
        trajectories[name] = np.array(traj)
        trajectories_full[name] = np.concatenate(
            [np.array(x0_point[:12])[None, :], np.array(traj)], axis=0)  # (T+1, 12)
        print(f"{name:16s} final position: {traj[-1, 0:3]}  final attitude: {traj[-1, 6:9]}")

        # NOTE: observed_output() assumes a single (21,) state interval, not a
        # (T+1, 21) batch -- slice the observable dims directly here instead.
        hist = propagate_history(x0_ivl, u_seq, scen)             # (T+1, 21) interval
        tube_lower[name] = np.array(hist.lower)[:, :12]
        tube_upper[name] = np.array(hist.upper)[:, :12]

    # ── Confirm the 4 simulated trajectories are actually distinguishable ──
    t = np.arange(0, num_steps + 1) * QPS_DT   # includes t=0 (shared prior), matches traj_full/tube length T+1
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

    # ── Plot: position + attitude trajectories, all 4 controllers overlaid,
    # each with its own output-reachable tube (the interval of observed
    # states consistent with that controller model from the shared x0_ivl
    # prior, propagated under the same bias sequence via propagate_history)
    # shaded behind it in matching color. Every candidate's simulated line
    # stays inside its own tube by construction -- what the plot shows is how
    # much the four tubes stop overlapping each other as the spoof bias
    # drives them apart. ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    labels = [("x position (m)", 0), ("y position (m)", 1), ("z position (m)", 2),
             ("roll (rad)", 6), ("pitch (rad)", 7), ("yaw (rad)", 8)]
    for ax, (label, dim) in zip(axes.flat, labels):
        for name in CANDIDATE_NAMES:
            color = CANDIDATE_COLORS[name]
            ax.fill_between(t, tube_lower[name][:, dim], tube_upper[name][:, dim],
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(t, trajectories_full[name][:, dim], color=color,
                   label=name, linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Simulated closed-loop response of all 4 real firmware controllers\n{title_suffix}",
                fontsize=13)
    fig.tight_layout()
    out_png = RESULTS_DIR / f"{out_stem}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_png}")

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

    out_json = RESULTS_DIR / f"{json_stem or out_stem}_verification.json"
    with open(out_json, "w") as f:
        json.dump({
            "regime": regime_name,
            "u_seq": np.array(u_seq).tolist(),
            "final_positions": {n: trajectories[n][-1, 0:3].tolist() for n in CANDIDATE_NAMES},
            "pairwise_final_separation_m": pairwise_final_dist,
            "min_separation_m": min_sep,
            "discrimination_verdicts": per_controller,
            "all_correct": all_ok,
            "note": "Point-simulation verification (real controller equations, no CrazySim/Gazebo/firmware-binary layer -- see module docstring).",
        }, f, indent=2)
    print(f"Saved {out_json}")


def main():
    scenarios = create_scenarios()

    # ── WIDE regime: realistic per-axis uncertainty, optimized spoof bias
    # (loaded from run_firmware_discrimination.py's synthesis output) ──
    with open(RESULTS_DIR / "firmware_discrimination_result.json") as f:
        result = json.load(f)
    u_star = jnp.array(result["wide_regime"]["u_star"])
    x0_wide = irx.icentpert(jnp.zeros(21), WIDE_WIDTH)
    _simulate_and_plot(
        "WIDE regime (optimized spoof bias)", x0_wide, u_star, scenarios,
        out_stem="firmware_simulated_trajectories",
        json_stem="firmware_simulation",   # preserves the pre-existing results/firmware_simulation_verification.json name (referenced by PLAN.md)
        title_suffix="under the SAME synthesized spoof-bias sequence (WIDE, realistic per-axis "
                     "uncertainty), with each controller's output-reachable tube overlaid",
    )

    # ── TIGHT regime: well-converged state estimate, ZERO bias -- separation
    # is already loss=0.0 at zero bias here (results/firmware_discrimination_result.json's
    # tight_regime), so no synthesized input is needed; plotted to show that
    # visually. ──
    x0_tight = irx.icentpert(jnp.zeros(21), jnp.full(21, TIGHT_WIDTH))
    u_zero = jnp.zeros((NUM_STEPS, 3))
    _simulate_and_plot(
        "TIGHT regime (zero bias)", x0_tight, u_zero, scenarios,
        out_stem="firmware_simulated_trajectories_tight",
        title_suffix="under ZERO spoof bias (TIGHT, well-converged-estimate uncertainty), "
                     "with each controller's output-reachable tube overlaid",
    )


if __name__ == "__main__":
    main()
