"""
Single-stage output-feedback separating controller, swept over integration dt.
==============================================================================
"Single stage" in this folder's sense (see single_step_demo.py): ONE constant
control decision held over the whole horizon -- here one constant gain and
feedforward

    u = clip(K @ y + r,  U_LO, U_HI),      y = [pb, qb, rb]

i.e. `admire_output_feedback.optimize_output_feedback_gpu(shared_theta=True)`,
the time-invariant parameterisation the car solver uses as its primary form.
This is the whole controller: 40 numbers, no per-step schedule.

Why sweep dt
------------
The blow-up that sank the per-step output-feedback runs (40.9x box growth at
dt=0.15, separating nothing) is a wrapping effect, and wrapping is a function
of the integration step: each Euler step bounds the flow over a whole dt with
one interval evaluation, so a coarse dt over-approximates more, and under
feedback that extra width re-enters through K@y on the NEXT step. The car
solver attributed its own blow-up to exactly this ("the coarse single-Euler-
step-per-segment integration (no sub-stepping)"). So the open question is
whether finer integration buys enough tightness to make output feedback
useful, or whether the conservatism is structural.

Two sweeps, because dt confounds two things at once:

  A. FIXED STEP COUNT (num_steps=10), dt varied -- horizon = 10*dt moves with
     dt. Shows the combined dt/horizon effect.
  B. FIXED HORIZON (~1.5s, the setting at which the open-loop unrefined demo
     achieves full separation), dt and num_steps varied together. Isolates
     integration resolution from horizon length -- the actual question.
  C. num_steps=1, dt varied -- the degenerate single-Euler-step case the
     coarse end of the sweep is heading towards. Note refinement is VACUOUS
     here: intersection refinement first acts between step 1 and step 2, so a
     one-step horizon exercises none of it.

Baseline arm
------------
Every configuration is solved TWICE: once with the gains free, and once with
`gain_mask=0`, which collapses the law to the constant open-loop input
u = clip(r, U_LO, U_HI). That is the honest comparison -- the same optimizer,
budget, seeds and horizon, differing only in whether feedback is allowed --
and answers "does output feedback beat open loop HERE" per dt rather than
against a differently-tuned reference. The mask is a traced scalar, so both
arms share one compile.

Cost
----
`dt` and `gain_mask` are traced, so one compile serves every dt and both arms
at a given num_steps; only num_steps (a scan length) forces a recompile. The
sweep below touches 6 distinct step counts, so ~6 compiles at ~2 min each
dominates the runtime -- the solves themselves are ~2.5s apiece. Left
unchunked deliberately: --restart-chunk 8 is ~1.9x faster per solve but adds
~73s of compile, which does not pay back over so few solves per compile.

Usage:
  /home/user/immrax-venv/bin/python single_stage_dt_sweep.py
"""
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

import admire_output_feedback as ofb

NUM_RESTARTS = 32
NUM_ITERS = 60
LEARNING_RATE = 0.05
SEED = 42
X0_HALFWIDTH = 5e-3          # same as both multistep demos, so results compare

# (label, dt, num_steps). Grouped by num_steps: every group is ONE compile.
CONFIGS = [
    # C. single Euler step (refinement vacuous)
    ('1-step', 1.00, 1),
    ('1-step', 0.50, 1),
    ('1-step', 0.25, 1),
    ('1-step', 0.20, 1),
    ('1-step', 0.10, 1),
    # A. fixed step count, horizon rides along with dt
    ('fixed-N', 1.00, 10),
    ('fixed-N', 0.50, 10),
    ('fixed-N', 0.25, 10),
    ('fixed-N', 0.20, 10),
    ('fixed-N', 0.15, 10),
    ('fixed-N', 0.10, 10),
    ('fixed-N', 0.05, 10),
    # B. fixed ~1.5s horizon, integration resolution varied
    ('fixed-T', 0.50, 3),
    ('fixed-T', 0.25, 6),
    ('fixed-T', 0.10, 15),
    ('fixed-T', 0.05, 30),
]


def main():
    print(f"Devices: {jax.devices()}")
    cl_scenarios = ofb.create_cl_scenarios(fault_effectiveness=0.0)
    n = len(cl_scenarios)
    n_pairs = n * (n - 1) // 2
    print(f"{n} closed-loop scenarios, {n_pairs} pairs; single-stage "
          f"(one constant K, r = {ofb.THETA_LEN} numbers)")

    x0_nom = jnp.zeros(ofb.NUM_STATES).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(ofb.NUM_STATES) * X0_HALFWIDTH)

    solvers = {}          # num_steps -> jitted solver (dt, gain_mask are traced)

    def solver_for(num_steps):
        if num_steps not in solvers:
            solvers[num_steps] = jax.jit(lambda dt, gain_mask, N=num_steps:
                ofb.optimize_output_feedback_gpu(
                    x0_ivl=x0_ivl, cl_scenarios=cl_scenarios, dt=dt,
                    num_steps=N, num_restarts=NUM_RESTARTS,
                    learning_rate=LEARNING_RATE, num_iters=NUM_ITERS,
                    seed=SEED, shared_theta=True, optimizer='adam',
                    gain_mask=gain_mask))
        return solvers[num_steps]

    rows = []
    hdr = (f"{'sweep':8s} {'dt':>5s} {'N':>3s} {'horizon':>8s} {'arm':>9s} "
           f"{'loss':>11s} {'|K|max':>7s} {'growth':>8s} {'disjoint steps':>16s}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for label, dt, num_steps in CONFIGS:
        solve = solver_for(num_steps)
        for arm, mask in (('feedback', 1.0), ('open-loop', 0.0)):
            t0 = time.perf_counter()
            best_theta, best_loss, _, losses = jax.block_until_ready(
                solve(jnp.float32(dt), jnp.float32(mask)))
            elapsed = time.perf_counter() - t0

            theta_seq = ofb.expand_theta(best_theta, num_steps)
            K, r = ofb.theta_to_K_r(theta_seq[0])
            hist = ofb.raw_output_histories(x0_ivl, theta_seq, cl_scenarios, dt)
            disjoint = ofb.raw_disjoint_segments(hist)
            growth = ofb.output_box_growth(x0_ivl, hist)
            n_nan = int(jnp.sum(jnp.isnan(losses)))

            print(f"{label:8s} {dt:5.2f} {num_steps:3d} {num_steps*dt:7.2f}s "
                  f"{arm:>9s} {float(best_loss):11.4e} "
                  f"{float(jnp.max(jnp.abs(K))):7.4f} {growth:7.1f}x "
                  f"{str(disjoint):>16s}"
                  + (f"  [{n_nan} NaN]" if n_nan else "")
                  + (f"  ({elapsed:.0f}s incl. compile)" if elapsed > 10 else ""),
                  flush=True)

            rows.append(dict(sweep=label, dt=dt, num_steps=num_steps, arm=arm,
                             loss=float(best_loss), k_max=float(jnp.max(jnp.abs(K))),
                             r_max=float(jnp.max(jnp.abs(r))), growth=growth,
                             n_disjoint=len(disjoint), n_nan=n_nan,
                             full_separation=bool(disjoint),
                             theta=np.asarray(best_theta)))

    out = _HERE / 'single_stage_dt_sweep.npz'
    np.savez(out,
             **{k: np.array([row[k] for row in rows])
                for k in ('sweep', 'dt', 'num_steps', 'arm', 'loss', 'k_max',
                          'r_max', 'growth', 'n_disjoint', 'n_nan',
                          'full_separation')},
             theta=np.stack([row['theta'] for row in rows]),
             x0_halfwidth=X0_HALFWIDTH, num_restarts=NUM_RESTARTS,
             num_iters=NUM_ITERS, learning_rate=LEARNING_RATE, seed=SEED)
    print(f"\nSaved {out}")

    # ── Verdict: did feedback beat its own open-loop arm anywhere? ────────
    print("\nPer-config comparison (feedback vs its own open-loop arm):")
    wins = 0
    for i in range(0, len(rows), 2):
        fb, ol = rows[i], rows[i + 1]
        better = fb['loss'] < ol['loss']
        wins += better
        print(f"  {fb['sweep']:8s} dt={fb['dt']:<5.2f} N={fb['num_steps']:<3d} "
              f"feedback {fb['loss']:.3e} vs open-loop {ol['loss']:.3e}  "
              f"-> {'FEEDBACK BETTER' if better else 'no gain from feedback'}")
    print(f"\nFeedback beat open loop in {wins}/{len(rows)//2} configurations.")
    sep = [r for r in rows if r['full_separation']]
    if sep:
        print("Configurations reaching FULL raw separation (all "
              f"{n_pairs} pairs disjoint at some step):")
        for r in sep:
            print(f"  {r['sweep']:8s} dt={r['dt']:<5.2f} N={r['num_steps']:<3d} "
                  f"{r['arm']:>9s}  loss={r['loss']:.3e}  growth={r['growth']:.1f}x")
    else:
        print(f"No configuration reached full raw separation of all {n_pairs} pairs.")


if __name__ == "__main__":
    main()
