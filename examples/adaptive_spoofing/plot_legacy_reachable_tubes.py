"""
Simulated closed-loop response of the LEGACY (fictional, Sections 1-10)
4-controller bank -- qps_snap_chain/pd_pos_vel/pid_pos_vel_i/indi_jerk from
crazyflie_chain_controllers.py -- under a separating spoof-bias sequence,
with each controller's own output-reachable tube (unrefined natural-embedding
propagation from the shared x0_ivl prior) overlaid in matching color. Mirrors
verify_firmware_simulation.py's plot for the REAL firmware bank; this is the
same figure for the superseded toy bank that PLAN.md Sec 11 kept around as a
"discrimination machinery" demo.

Two regimes, each its own PDF:
- TIGHT (results/synthesized_spoof.npz, from synthesize_spoof.py): uniform
  width=2e-3 initial uncertainty (updated from an earlier 1e-3 pass), 30-step
  horizon.
- WIDE (results/synthesized_spoof_wide.npz, from synthesize_spoof_wide.py):
  realistic per-axis uncertainty, 10-step horizon (shorter -- see that
  script's docstring for why: this bank's gains amplify a forward-Euler
  step's interval width far more aggressively than the firmware bank's, so
  wider x0 + a long horizon blows reachable-set widths up past any
  physically-meaningful scale).

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/plot_legacy_reachable_tubes.py
"""
import json
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
    create_scenarios, _CANDIDATE_THETA, euler_step,
    simulate_true_trajectory, discriminate_controller,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CANDIDATE_COLORS = {
    "qps_snap_chain": "tab:blue",
    "pd_pos_vel": "tab:orange",
    "pid_pos_vel_i": "tab:green",
    "indi_jerk": "tab:red",
}
CANDIDATE_NAMES = tuple(_CANDIDATE_THETA.keys())


def _reachable_history(scenarios, x0_ivl, u_seq, dt):
    """Per-step, per-scenario state-interval history (num_steps+1,
    n_scenarios, 18) -- plain unrefined natural-embedding propagation (no
    cross-candidate refinement), matching plot_separation.py's `_history`
    helper for the legacy bank."""
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


def _simulate_and_plot(regime_name, npz_name, out_pdf_stem, title_suffix, json_stem=None,
                       candidate_names=None):
    """candidate_names: subset of CANDIDATE_NAMES to include (default: all 4).
    The bias sequence loaded from `npz_name` was optimized for the FULL bank
    (see synthesize_spoof.py) -- excluding a candidate here only changes what
    gets simulated/plotted/discriminated-against, it does not re-synthesize
    the bias for the smaller bank. discriminate_controller below only sees
    the reduced `scenarios` list, so an excluded candidate cannot appear as a
    surviving false positive; it is simply absent from the comparison."""
    candidate_names = candidate_names or CANDIDATE_NAMES
    data = np.load(RESULTS_DIR / npz_name, allow_pickle=True)
    u_seq = jnp.array(data["u_seq"])
    dt = float(data["dt"])
    ref15 = jnp.array(data["ref15"])
    x0_ivl = irx.Interval(lower=jnp.array(data["x0_ivl_lower"]), upper=jnp.array(data["x0_ivl_upper"]))
    num_steps = u_seq.shape[0]
    print(f"\n{'=' * 70}\n{regime_name}: loaded {npz_name}, shape={u_seq.shape}, dt={dt}, "
         f"max|bias|={float(jnp.max(jnp.abs(u_seq))):.4f} m, candidates={candidate_names}")

    scenarios = create_scenarios(ref15, names=list(candidate_names))
    x0_point = jnp.zeros(18)

    # ── Simulate each controller's ACTUAL closed-loop response (point
    # rollout, no interval abstraction), and its output-reachable tube from
    # the shared x0_ivl prior under the SAME bias sequence. ──
    print("Simulating closed-loop trajectories and reachable tubes...")
    lowers, uppers = _reachable_history(scenarios, x0_ivl, u_seq, dt)   # (T+1, n, 18)

    trajectories = {}         # (T, 12) -- fed to discriminate_controller as-is
    trajectories_full = {}    # (T+1, 12), t=0 state prepended -- for plotting
    for name in candidate_names:
        theta6 = jnp.array(_CANDIDATE_THETA[name])
        traj = simulate_true_trajectory(x0_point, u_seq, theta6, ref15, dt)   # (T, 12)
        trajectories[name] = np.array(traj)
        trajectories_full[name] = np.concatenate(
            [np.array(x0_point[:12])[None, :], np.array(traj)], axis=0)     # (T+1, 12)
        print(f"{name:16s} final position: {traj[-1, 0:3]}  final velocity: {traj[-1, 3:6]}")

    t = np.arange(0, num_steps + 1) * dt

    final_positions = np.stack([trajectories[n][-1, 0:3] for n in candidate_names])
    pairwise_final_dist = {}
    for i, ni in enumerate(candidate_names):
        for j, nj in enumerate(candidate_names):
            if j > i:
                d = float(np.linalg.norm(final_positions[i] - final_positions[j]))
                pairwise_final_dist[f"{ni} vs {nj}"] = d
    print("\nFinal-position separation between simulated trajectories:")
    for k, v in pairwise_final_dist.items():
        print(f"  {k}: {v*1000:.3f} mm")
    min_sep = min(pairwise_final_dist.values())

    # ── Plot: position + velocity, all 4 legacy controllers overlaid, each
    # with its own output-reachable tube shaded behind it in matching color ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    labels = [("x position (m)", 0), ("y position (m)", 1), ("z position (m)", 2),
             ("x velocity (m/s)", 3), ("y velocity (m/s)", 4), ("z velocity (m/s)", 5)]
    for ax, (label, dim) in zip(axes.flat, labels):
        for i, name in enumerate(candidate_names):
            color = CANDIDATE_COLORS[name]
            ax.fill_between(t, lowers[:, i, dim], uppers[:, i, dim],
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(t, trajectories_full[name][:, dim], color=color,
                   label=name, linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Legacy bank ({'/'.join(candidate_names)}): simulated "
                f"closed-loop response\n{title_suffix}", fontsize=13)
    fig.tight_layout()
    out_pdf = RESULTS_DIR / f"{out_pdf_stem}.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\nSaved {out_pdf}")

    # ── Re-confirm discrimination against each simulated trajectory ──
    print("\nDiscrimination verdict, reacting to each simulated trajectory in turn:")
    all_ok = True
    per_controller = {}
    for true_name in candidate_names:
        observed = jnp.array(trajectories[true_name])
        verdict = discriminate_controller(x0_ivl, u_seq, observed, scenarios, dt=dt)
        ok = verdict["survivors"] == [true_name]
        all_ok &= ok
        per_controller[true_name] = {
            "survivors": verdict["survivors"], "fail_step": verdict["fail_step"], "correct": ok,
        }
        print(f"  true={true_name:16s}  survivors={verdict['survivors']}  {'OK' if ok else 'WRONG'}")

    print(f"\nMinimum final-position separation across all pairs: {min_sep*1000:.3f} mm")
    print("VERDICT:", "PASS -- trajectory is suitable for discrimination" if all_ok and min_sep > 1e-6
         else "CHECK -- see per-controller verdicts above")

    out_json = RESULTS_DIR / f"{json_stem or out_pdf_stem}_verification.json"
    with open(out_json, "w") as f:
        json.dump({
            "regime": regime_name,
            "u_seq": np.array(u_seq).tolist(),
            "dt": dt,
            "num_steps": num_steps,
            "final_positions": {n: trajectories[n][-1, 0:3].tolist() for n in candidate_names},
            "pairwise_final_separation_m": pairwise_final_dist,
            "min_separation_m": min_sep,
            "discrimination_verdicts": per_controller,
            "all_correct": all_ok,
        }, f, indent=2)
    print(f"Saved {out_json}")


def main():
    _simulate_and_plot(
        "TIGHT regime", "synthesized_spoof.npz",
        out_pdf_stem="legacy_reachable_tubes",
        json_stem="legacy_reachable_tubes",
        title_suffix="under the synthesized separating spoof bias (TIGHT, width=2e-3 uncertainty), "
                     "with each controller's output-reachable tube overlaid",
    )
    # Same TIGHT bias sequence, minus pid_pos_vel_i -- see _simulate_and_plot's
    # docstring: the bias was optimized for the full 4-candidate bank, this
    # just drops one candidate from what's plotted/discriminated-against, it
    # does not re-synthesize for the smaller bank.
    _simulate_and_plot(
        "TIGHT regime, no pid_pos_vel_i", "synthesized_spoof.npz",
        out_pdf_stem="legacy_reachable_tubes_no_pid",
        json_stem="legacy_reachable_tubes_no_pid",
        title_suffix="under the synthesized separating spoof bias (TIGHT, width=2e-3 uncertainty), "
                     "with each controller's output-reachable tube overlaid -- pid_pos_vel_i excluded",
        candidate_names=[n for n in CANDIDATE_NAMES if n != "pid_pos_vel_i"],
    )
    _simulate_and_plot(
        "WIDE regime", "synthesized_spoof_wide.npz",
        out_pdf_stem="legacy_reachable_tubes_wide",
        title_suffix="under the synthesized separating spoof bias (WIDE, realistic per-axis "
                     "uncertainty), with each controller's output-reachable tube overlaid",
    )


if __name__ == "__main__":
    main()
