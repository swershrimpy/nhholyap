"""
Model discrimination over the 4 REAL Crazyflie firmware controllers
(cf_pid, cf_mellinger, cf_indi, cf_brescianini) -- replaces the fictional
bank from crazyflie_chain_controllers.py per PLAN.md / user direction.

Two regimes, both reported (see module docstring in
crazyflie_firmware_controllers.py for the system definitions):

1. Tight initial state uncertainty (a well-converged state estimate): the
   four real controllers' reachable outputs separate INSTANTLY (within one
   20ms tick) at ZERO spoof bias -- their real firmware gains differ enough
   in scale/structure that no attack is needed. Not a bug: verified
   separately that all four correctly survive as their own ground truth.
2. Wider, more realistic initial uncertainty: genuine overlap exists at zero
   bias, and a synthesized spoof-bias sequence measurably reduces it. This
   is the "run model discrimination" case with actual work for the
   optimizer to do.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/run_firmware_discrimination.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import immrax as irx

# Paper-figure style: a Times-metric serif so the figures match the
# manuscript's body text. Liberation Serif is metrically compatible with
# Times New Roman and is a genuine TrueType face, so it embeds cleanly under
# fonttype 42; Nimbus Roman (the URW Times clone) is also metric-compatible
# but is OpenType/CFF, which makes readers warn "mismatch between font type
# and embedded font file" -- hence Liberation first.
# fonttype 42 embeds real glyphs rather than rasterising, so the PDF stays
# vector and the text stays selectable/searchable.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_firmware_controllers import (
    create_scenarios, CANDIDATE_NAMES, QPS_DT,
    SeparatingInputOptimizer, optimize_parallel_gpu,
    simulate_true_trajectory, discriminate_controller,
    observed_output, _BIAS_LIM, ATTITUDE_LIMIT_DEG,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CANDIDATE_COLORS = {
    "cf_pid": "tab:blue", "cf_mellinger": "tab:orange",
    "cf_indi": "tab:green", "cf_brescianini": "tab:red",
}

TIGHT_WIDTH = 1e-3
# Realistic, physically-scaled per-dimension uncertainty for a converged
# state estimate: position/velocity a few cm(/s), attitude ~1deg, rates
# small, memory (hidden, unobservable) left generously loose. NOT a scalar
# width -- see PLAN.md: this system's natural-embedding interval propagation
# has a strong "wrapping effect" (~2x width growth PER STEP, verified
# empirically across all 4 candidates, independent of starting scale) once
# attitude/rate uncertainty is more than a couple of degrees, so realistic
# per-axis scaling (not a single blanket number) matters here far more than
# in this project's other, more diagonal/decoupled systems.
WIDE_WIDTH = jnp.concatenate([
    jnp.full(3, 0.05), jnp.full(3, 0.05), jnp.full(3, 0.017), jnp.full(3, 0.05), jnp.full(9, 0.01),
])
# Short horizon: because of the ~2x/step wrapping effect above, reachable
# sets here are only meaningful for a few steps regardless of starting
# width -- unlike crazyflie_chain_controllers.py's much more diagonal chain
# dynamics, which tolerated dozens of steps. Discrimination among these 4
# REAL controllers happens fast anyway (see the tight-regime zero-bias
# result below), so this isn't a meaningful restriction in practice.
NUM_STEPS = 3


def check_ground_truth_survives(scenarios, x0_ivl, u_seq, w_bar=None):
    kwargs = {} if w_bar is None else {"w_bar": w_bar}
    out = {}
    for name in CANDIDATE_NAMES:
        true_scen = [s for s in scenarios if s.name == name][0]
        x0_point = jnp.concatenate([x0_ivl.lower[:12] * 0.0, jnp.zeros(9)])  # center = 0
        observed = simulate_true_trajectory(x0_point, u_seq, true_scen)
        result = discriminate_controller(x0_ivl, u_seq, observed, scenarios, **kwargs)
        out[name] = result
    return out


def main():
    scenarios = create_scenarios()

    print("=" * 70)
    print("Regime 1: TIGHT initial uncertainty (width=%.0e) -- zero bias" % TIGHT_WIDTH)
    x0_tight = irx.icentpert(jnp.zeros(21), jnp.full(21, TIGHT_WIDTH))
    u_zero = jnp.zeros((NUM_STEPS, 3))
    opt_tight = SeparatingInputOptimizer(scenarios, x0_tight, num_steps=NUM_STEPS)
    loss_tight = float(opt_tight.loss_fn(u_zero))
    print(f"separation_loss at zero bias: {loss_tight:.3e}")
    results_tight = check_ground_truth_survives(scenarios, x0_tight, u_zero)
    for name, r in results_tight.items():
        print(f"  true={name:16s}  survivors={r['survivors']}  {'OK' if r['survivors']==[name] else 'WRONG'}")

    print("\n" + "=" * 70)
    print("Regime 2: WIDE (realistic, per-axis-scaled) initial uncertainty -- synthesizing a spoof bias")
    x0_wide = irx.icentpert(jnp.zeros(21), WIDE_WIDTH)
    opt_wide = SeparatingInputOptimizer(scenarios, x0_wide, num_steps=NUM_STEPS)
    loss_zero_wide = float(opt_wide.loss_fn(u_zero))
    print(f"separation_loss at zero bias: {loss_zero_wide:.3e}")

    t0 = time.perf_counter()
    u_star, loss_star, u_all, losses_all = optimize_parallel_gpu(
        opt_wide, num_restarts=32, learning_rate=0.03, num_iters=200, seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"Optimized in {elapsed:.1f}s. Best loss={float(loss_star):.3e} "
         f"(median over restarts={float(jnp.median(losses_all)):.3e})")
    print(f"max|bias|={float(jnp.max(jnp.abs(u_star))):.4f} m")

    result_eval = opt_wide.evaluate(u_star)
    print("\nPairwise overlaps at optimized bias:")
    for k, v in result_eval['pairwise_overlaps'].items():
        print(f"  {k}: {v:.3e}")

    # w_bar for the WIDE regime: the reaction step's own point rollout and
    # the interval propagation are still two independently-run computations
    # of the same equations, so some float-precision slack is needed (same
    # reasoning as crazyflie_chain_controllers.py's w_bar discussion) --
    # swept empirically rather than assumed.
    print("\nReacting to the optimized bias -- does each candidate survive its own trajectory?")
    w_bar_used = 1e-3
    results_wide = check_ground_truth_survives(scenarios, x0_wide, u_star, w_bar=w_bar_used)
    all_ok = True
    for name, r in results_wide.items():
        ok = r['survivors'] == [name]
        all_ok &= ok
        print(f"  true={name:16s}  survivors={r['survivors']}  fail_step={r['fail_step']}  {'OK' if ok else 'WRONG'}")

    # ── Plot: pairwise overlap over time, zero vs optimized bias ──
    def history_overlaps(u_seq):
        histories = [None] * len(scenarios)
        from adaptive_spoofing.crazyflie_firmware_controllers import propagate_history, _output_overlap_volume
        for i, s in enumerate(scenarios):
            histories[i] = propagate_history(x0_wide, u_seq, s)
        T1 = histories[0].lower.shape[0]
        n = len(scenarios)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        out = {f"{scenarios[i].name} vs {scenarios[j].name}": np.zeros(T1) for i, j in pairs}
        for k in range(T1):
            for i, j in pairs:
                ivl_i = irx.Interval(lower=histories[i].lower[k], upper=histories[i].upper[k])
                ivl_j = irx.Interval(lower=histories[j].lower[k], upper=histories[j].upper[k])
                out[f"{scenarios[i].name} vs {scenarios[j].name}"][k] = float(_output_overlap_volume(ivl_i, ivl_j))
        return out

    overlap_zero = history_overlaps(u_zero)
    overlap_opt = history_overlaps(u_star)
    t = np.arange(NUM_STEPS + 1) * QPS_DT

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, name in zip(axes.flat, overlap_zero.keys()):
        ax.plot(t, overlap_zero[name], color="gray", linestyle="--", label="zero bias")
        ax.plot(t, overlap_opt[name], color="crimson", label="optimized spoof bias")
        ax.set_title(name)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed-output\npairwise overlap volume")
    fig.suptitle("Real firmware controllers (cf_pid/cf_mellinger/cf_indi/cf_brescianini): "
                f"overlap under realistic per-axis-scaled initial uncertainty\n"
                f"(spoof bias projected onto the ${{\\pm}}{_BIAS_LIM:g}$ m box that keeps attitude "
                f"within the firmware's {ATTITUDE_LIMIT_DEG:g}$^\\circ$ envelope)", fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS_DIR / f"firmware_pairwise_overlap.{ext}",
                    dpi=150, bbox_inches="tight")
        print(f"\nSaved {RESULTS_DIR / f'firmware_pairwise_overlap.{ext}'}")

    # ── Plot: the synthesized bias itself, against the projection box ──
    figb, axb = plt.subplots(figsize=(7, 3.4))
    # Each bias is held over [k*dt, (k+1)*dt), so the staircase needs one extra
    # sample to close the final hold -- otherwise the last step is invisible and
    # the trace appears to stop a tick early. Markers go on the real samples only.
    tb = np.arange(NUM_STEPS + 1) * QPS_DT
    ub = np.vstack([np.array(u_star), np.array(u_star)[-1:]])
    for i, lab in enumerate(("bias $x$", "bias $y$", "bias $z$")):
        line, = axb.step(tb, ub[:, i], where="post", label=lab)
        axb.plot(tb[:NUM_STEPS], ub[:NUM_STEPS, i], "o", ms=3.5, color=line.get_color())
    axb.axhline(_BIAS_LIM, color="k", ls="--", lw=1)
    axb.axhline(-_BIAS_LIM, color="k", ls="--", lw=1,
                label=fr"projection box $\pm{_BIAS_LIM:g}$ m")
    axb.set_xlabel("time (s)")
    axb.set_ylabel("spoof position bias (m)")
    axb.set_title("Synthesized separating spoof bias under the attitude-derived input limit")
    axb.grid(alpha=0.3)
    axb.legend(ncol=2)
    figb.tight_layout()
    for ext in ("pdf", "png"):
        figb.savefig(RESULTS_DIR / f"firmware_spoof_bias.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved {RESULTS_DIR / f'firmware_spoof_bias.{ext}'}")

    # ── Save results ──
    out = {
        "candidate_names": list(CANDIDATE_NAMES),
        "tight_regime": {
            "width": TIGHT_WIDTH, "loss_at_zero_bias": loss_tight,
            "ground_truth_survives": {n: r["survivors"] == [n] for n, r in results_tight.items()},
        },
        "wide_regime": {
            "width": np.array(WIDE_WIDTH).tolist(), "loss_at_zero_bias": loss_zero_wide,
            "loss_at_optimized_bias": float(loss_star),
            "u_star": np.array(u_star).tolist(),
            "pairwise_overlaps_optimized": result_eval['pairwise_overlaps'],
            "w_bar_used": w_bar_used,
            "ground_truth_survives": {n: r["survivors"] == [n] for n, r in results_wide.items()},
            "fail_steps": {n: r["fail_step"] for n, r in results_wide.items()},
        },
        "all_ground_truth_checks_passed": all_ok and all(
            r["survivors"] == [n] for n, r in results_tight.items()),
    }
    with open(RESULTS_DIR / "firmware_discrimination_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {RESULTS_DIR / 'firmware_discrimination_result.json'}")


if __name__ == "__main__":
    main()
