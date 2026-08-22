"""
Single-stage output feedback: N=10 x dt=0.1  vs  N=5 x dt=0.2.
==============================================================
Both cover the SAME 1.0s horizon, so the only difference is integration
resolution: ten fine Euler steps against five coarse ones. This is the
cleanest form of the question the dt sweep raised -- does finer integration
help or hurt a single-stage output-feedback separating controller? -- with
horizon held fixed so it cannot confound the answer.

Two effects pull in opposite directions:
  finer dt  -> each Euler step over-approximates the flow less (less
               wrapping per step), but there are more steps for feedback
               width to compound through, and each step advances the state
               less, so the scenarios diverge more slowly.
  coarser dt -> fewer compounding steps and more divergence per step, but
               each step's interval enclosure is looser and the forward-Euler
               discretisation is a worse stand-in for the ODE.

Every configuration is run twice, gains free and `gain_mask=0` (the constant
open-loop input u = clip(r)), so each has its own like-for-like baseline.

Reported metrics, in increasing order of how much they can be trusted:
  loss            the REFINED objective. Saturates at exactly 0.0 for most
                  configurations at this horizon, so it mostly does not
                  discriminate -- and a refined zero is not a raw
                  certificate (see multistep_refined_demo.py's caveat).
  max pairs sep   most pairs (of 55) simultaneously separated at any one
                  step, on RAW independently-propagated boxes. The partial-
                  credit metric; 54/55 and 3/55 are both "not separated" but
                  are not the same result.
  full-sep steps  steps where all 55 are separated at once -- the actual
                  diagnosability certificate.

Usage:
  /home/user/immrax-venv/bin/python compare_n10dt01_n5dt02.py
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
X0_HALFWIDTH = 5e-3

CONFIGS = [(10, 0.10), (5, 0.20)]


def main():
    print(f"Devices: {jax.devices()}")
    cl_scenarios = ofb.create_cl_scenarios(fault_effectiveness=0.0)
    n = len(cl_scenarios)
    n_pairs = n * (n - 1) // 2
    x0_nom = jnp.zeros(ofb.NUM_STATES).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(ofb.NUM_STATES) * X0_HALFWIDTH)
    print(f"{n} scenarios / {n_pairs} pairs, single-stage (one constant K, r), "
          f"1.0s horizon both ways\n")

    rows = []
    for num_steps, dt in CONFIGS:
        solve = jax.jit(lambda dt_, mask, N=num_steps: ofb.optimize_output_feedback_gpu(
            x0_ivl=x0_ivl, cl_scenarios=cl_scenarios, dt=dt_, num_steps=N,
            num_restarts=NUM_RESTARTS, learning_rate=LEARNING_RATE,
            num_iters=NUM_ITERS, seed=SEED, shared_theta=True,
            optimizer='adam', gain_mask=mask))

        for arm, mask in (('feedback', 1.0), ('open-loop', 0.0)):
            t0 = time.perf_counter()
            best_theta, best_loss, _, losses = jax.block_until_ready(
                solve(jnp.float32(dt), jnp.float32(mask)))
            elapsed = time.perf_counter() - t0

            theta_seq = ofb.expand_theta(best_theta, num_steps)
            K, r = ofb.theta_to_K_r(theta_seq[0])
            hist = ofb.raw_output_histories(x0_ivl, theta_seq, cl_scenarios, dt)
            counts = ofb.raw_disjoint_pair_counts(hist)
            full = ofb.raw_disjoint_segments(hist)
            growth = ofb.output_box_growth(x0_ivl, hist)
            n_nan = int(jnp.sum(jnp.isnan(losses)))

            print(f"N={num_steps:<3d} dt={dt:<5.2f} {arm:>9s}  "
                  f"loss={float(best_loss):.4e}  "
                  f"max pairs sep={int(counts.max()):2d}/{n_pairs}  "
                  f"full-sep steps={full}  growth={growth:.1f}x  "
                  f"|K|max={float(jnp.max(jnp.abs(K))):.4f}  "
                  f"{n_nan} NaN  ({elapsed:.0f}s)")
            print(f"{'':>22s}pairs separated per step (t=dt..N*dt): "
                  f"{list(map(int, counts))}", flush=True)

            rows.append(dict(num_steps=num_steps, dt=dt, arm=arm,
                             loss=float(best_loss), counts=np.asarray(counts),
                             max_sep=int(counts.max()), n_full=len(full),
                             growth=growth, k_max=float(jnp.max(jnp.abs(K))),
                             r_max=float(jnp.max(jnp.abs(r))), n_nan=n_nan,
                             theta=np.asarray(best_theta)))
        print()

    np.savez(_HERE / 'compare_n10dt01_n5dt02.npz',
             **{k: np.array([row[k] for row in rows])
                for k in ('num_steps', 'dt', 'arm', 'loss', 'max_sep', 'n_full',
                          'growth', 'k_max', 'r_max', 'n_nan')},
             counts=np.stack([np.pad(r['counts'], (0, 10 - len(r['counts'])),
                                     constant_values=-1) for r in rows]),
             theta=np.stack([r['theta'] for r in rows]))

    fb = [r for r in rows if r['arm'] == 'feedback']
    print("Verdict (raw separation, not the saturated loss):")
    for r in fb:
        print(f"  N={r['num_steps']:<3d} dt={r['dt']:<5.2f}  "
              f"{r['max_sep']}/{n_pairs} pairs at best step, "
              f"{r['n_full']} fully-separated step(s), {r['growth']:.1f}x growth")
    # Rank on the headline metric, but say so explicitly when it TIES --
    # max() would otherwise silently return the first entry and read as a win.
    key = lambda r: (r['n_full'], r['max_sep'])
    best = max(fb, key=key)
    tied = [r for r in fb if key(r) == key(best)]
    if len(tied) > 1:
        print("  -> TIE on (full-sep steps, max pairs) = "
              f"{key(best)}; separating on the tie-breakers instead:")
        for r in tied:
            print(f"     N={r['num_steps']:<3d} dt={r['dt']:<5.2f}  "
                  f"growth={r['growth']:.1f}x  |K|max={r['k_max']:.4f}  "
                  f"pairs separated per step {list(map(int, r['counts']))}")
        print("     (compare per-step counts at MATCHED TIMES -- step k of a "
              "dt=0.1 run is t=0.1(k+1), not the same instant as step k of a "
              "dt=0.2 run)")
    else:
        print(f"  -> N={best['num_steps']}, dt={best['dt']} separates more")
    print(f"\nSaved {_HERE / 'compare_n10dt01_n5dt02.npz'}")


if __name__ == "__main__":
    main()
