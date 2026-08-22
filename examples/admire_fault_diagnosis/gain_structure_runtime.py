"""
Does a sparser gain matrix make the output-feedback solver faster?
==================================================================
`admire_output_feedback.py` uses a DENSE gain, K in R^{10x3}: every one of
the 10 surface deflections is driven by all 3 measured angular rates. That
is 30 gain parameters and a 30-product interval matrix-vector product inside
every evaluation of the closed-loop `f`. This script asks whether shrinking
that helps runtime, by solving the identical problem with three gain
structures:

  dense      u = clip(K @ y + r)                 K in R^{10x3}, 30 gains
             the committed parameterisation. 30 interval products per f.
  diagonal   u = clip([k * y, 0...0] + r)        k in R^3,       3 gains
             literally diagonal for a 10x3 matrix: only entries (0,0),
             (1,1), (2,2) are nonzero, so only the first three surfaces get
             any feedback at all and the other seven are pure feedforward.
             3 interval products per f.
  per-input  u = clip(g * y[i mod 3] + r)        g in R^10,     10 gains
             a fairer sparse comparison: still one gain per surface, so all
             10 keep feedback, but each reads a single rate instead of all
             three. 10 interval products per f.

Everything else is held identical -- single-stage (one constant gain and
feedforward), N=10, dt=0.1, Adam, 32 restarts x 60 iterations, seed 42, x0
half-width 5e-3 -- so the only difference is the gain structure. The dense
row should reproduce the committed default's measured 2707ms.

Prediction being tested
-----------------------
That this will NOT speed things up much, because the matvec is the smallest
of the three things feedback adds to `f`. The two larger ones survive any
nonzero gain structure:

  1. `u` becomes INTERVAL-valued. ADMIRE's actuator term is
     B_bar @ (diag(p) @ u). Open loop multiplies an interval by a POINT (2
     products and a sign select per channel); any feedback makes it a true
     interval x interval product (4 products and a 4-way min/max per
     channel). Diagonal or dense, u is still an interval.
  2. `d(u)/dx` becomes nonzero. Open loop's actuator term is constant in x,
     so it contributes nothing to the Jacobian. Under feedback it depends on
     x[3:6], and reverse mode gains a path from all 10 control channels back
     into 3 state components at every step. A diagonal K narrows that path
     but does not remove it.

Only cost 3 -- the 30 interval products of the matvec itself, and the 30 vs
3 vs 10 parameters -- is what this script varies. If the timings come out
close, cost 1 and 2 dominate and gain sparsity is not a runtime lever. If
diagonal is markedly faster, the matvec mattered more than the reasoning
above suggests and that reasoning needs revisiting.

Quality is reported alongside runtime, because a faster controller that
diagnoses less is not a win: a diagonal gain gives up authority (seven
surfaces lose feedback entirely), so it may well separate fewer pairs.

Usage:
  /home/user/immrax-venv/bin/python gain_structure_runtime.py
"""
import statistics
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
from admire_separating_input import (
    _U_LO, _U_HI, _scan_loop, get_system_and_embedding, propagate_with_refinement,
)

_OL_SYS, _ = get_system_and_embedding()

NUM_STEPS = 10
DT = 0.1
NUM_RESTARTS = 32
NUM_ITERS = 60
LEARNING_RATE = 0.05
SEED = 42
X0_HALFWIDTH = 5e-3
INIT_STD_GAIN = 0.3
INIT_SCALE_FF = 0.05
REPS = 10


# ── Gain structures ───────────────────────────────────────────────────────────
# Each returns the 10-vector pre-clip control from the gain block and y.
# `.at[].set()` is deliberately avoided in favour of concatenate: these run
# INSIDE the natural embedding's traced f, and concatenate is the safer
# primitive to push interval-valued operands through.

def _ctrl_dense(gain, y):
    return gain.reshape(10, 3) @ y


def _ctrl_diagonal(gain, y):
    # K = diag(k) for a 10x3 matrix: rows 3..9 are identically zero.
    return jnp.concatenate([gain * y, jnp.zeros(7)])


def _ctrl_per_input(gain, y):
    # Surface i reads rate (i mod 3): y0,y1,y2,y0,y1,y2,y0,y1,y2,y0
    return gain * jnp.concatenate([y, y, y, y[:1]])


VARIANTS = {
    'dense':     (_ctrl_dense, 30),
    'diagonal':  (_ctrl_diagonal, 3),
    'per-input': (_ctrl_per_input, 10),
}


def _make_embedding(ctrl_fn, n_gain):
    class CL(irx.System):
        def __init__(self):
            self.evolution = 'continuous'
            self.xlen = 9

        def f(self, t, x, u, p):
            gain, r = u[:n_gain], u[n_gain:]
            u_ctrl = jnp.clip(ctrl_fn(gain, x[3:6]) + r, _U_LO, _U_HI)
            return _OL_SYS.f(t, x, u_ctrl, p)

    return irx.natemb(CL())


def _solve(emb, n_gain, cl_scenarios, x0_ivl):
    """Single-stage multi-restart Adam, mirroring
    `admire_output_feedback.optimize_output_feedback_gpu`.

    Written out locally rather than imported because that function is bound
    to the dense 40-wide layout (THETA_LEN, _THETA_LO/_HI, _project_theta);
    the whole point here is to vary the width. The update rule, the
    per-coordinate step scale, `_scan_loop`, and the NaN-safe selection are
    kept identical so the timings compare like for like.
    """
    theta_len = n_gain + 10
    lo = jnp.concatenate([jnp.full(n_gain, -ofb._K_MAX), _U_LO])
    hi = jnp.concatenate([jnp.full(n_gain, ofb._K_MAX), _U_HI])
    step_scale = (hi - lo) / 2

    k_gain, k_ff = jax.random.split(jax.random.PRNGKey(SEED))
    theta0 = jnp.clip(jnp.concatenate([
        jax.random.normal(k_gain, (NUM_RESTARTS, n_gain)) * INIT_STD_GAIN,
        jax.random.uniform(k_ff, (NUM_RESTARTS, 10),
                           minval=-INIT_SCALE_FF, maxval=INIT_SCALE_FF),
    ], axis=-1), lo, hi)

    def loss_fn(theta):
        theta_seq = jnp.broadcast_to(theta, (NUM_STEPS, theta_len))
        return propagate_with_refinement(theta_seq, x0_ivl, cl_scenarios, DT, NUM_STEPS)

    batched = jax.vmap(jax.value_and_grad(loss_fn))
    b1, b2, eps = 0.9, 0.999, 1e-8

    def body(carry):
        theta, m, v, t = carry
        _, g = batched(theta)
        t = t + 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        return (jnp.clip(theta - LEARNING_RATE * step_scale * step, lo, hi), m, v, t)

    zeros = jnp.zeros_like(theta0)
    theta_final, _, _, _ = _scan_loop(
        body, (theta0, zeros, zeros, jnp.zeros((), jnp.int32)), NUM_ITERS)
    losses, _ = batched(theta_final)
    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best = jnp.argmin(losses_valid)
    return theta_final[best], losses[best], losses


def main():
    print(f"Devices: {jax.devices()}")
    x0_nom = jnp.zeros(9).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(9) * X0_HALFWIDTH)
    n_pairs = 55
    print(f"Single-stage, N={NUM_STEPS}, dt={DT}, {NUM_RESTARTS} restarts x "
          f"{NUM_ITERS} iters, Adam, seed {SEED}\n")

    rows = []
    for name, (ctrl_fn, n_gain) in VARIANTS.items():
        emb = _make_embedding(ctrl_fn, n_gain)
        scen = [ofb.dataclasses.replace(s, emb_system=emb)
                for s in ofb.create_scenarios(fault_effectiveness=0.0)]

        solve = jax.jit(lambda: _solve(emb, n_gain, scen, x0_ivl))
        t0 = time.perf_counter()
        jax.block_until_ready(solve())
        compile_s = time.perf_counter() - t0

        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            out = jax.block_until_ready(solve())
            ts.append((time.perf_counter() - t0) * 1e3)
        best_theta, best_loss, losses = out

        theta_seq = jnp.broadcast_to(best_theta, (NUM_STEPS, n_gain + 10))
        hist = ofb.raw_output_histories(x0_ivl, theta_seq, scen, DT)
        counts = ofb.raw_disjoint_pair_counts(hist)
        full = ofb.raw_disjoint_segments(hist)
        growth = ofb.output_box_growth(x0_ivl, hist)

        median = statistics.median(ts)
        print(f"{name:10s} gains={n_gain:2d} params={n_gain+10:2d}  "
              f"median={median:8.2f}ms  min={min(ts):8.2f}ms  "
              f"per-iter={median/NUM_ITERS:5.2f}ms  compile={compile_s:5.1f}s")
        print(f"{'':10s} loss={float(best_loss):.4e}  "
              f"max pairs sep={int(counts.max())}/{n_pairs}  "
              f"full-sep={full}  growth={growth:.1f}x  "
              f"{int(jnp.sum(jnp.isnan(losses)))} NaN", flush=True)

        rows.append(dict(name=name, n_gain=n_gain, median_ms=median,
                         min_ms=min(ts), compile_s=compile_s,
                         loss=float(best_loss), max_sep=int(counts.max()),
                         n_full=len(full), growth=growth,
                         theta=np.asarray(best_theta)))

    np.savez(_HERE / 'gain_structure_runtime.npz',
             **{k: np.array([r[k] for r in rows])
                for k in ('name', 'n_gain', 'median_ms', 'min_ms', 'compile_s',
                          'loss', 'max_sep', 'n_full', 'growth')})

    base = rows[0]['median_ms']
    print("\nSpeedup vs dense (30 gains):")
    for r in rows:
        print(f"  {r['name']:10s} {r['n_gain']:2d} gains -> "
              f"{base / r['median_ms']:.2f}x   "
              f"({r['median_ms']:.0f}ms vs {base:.0f}ms), "
              f"separation {r['max_sep']}/{n_pairs}")
    print("\nIf these are within ~10-20% of each other, the interval matvec was "
          "not the cost -- interval-valued u and the du/dx Jacobian path are, "
          "and gain sparsity is not a runtime lever.")
    print(f"Saved {_HERE / 'gain_structure_runtime.npz'}")


if __name__ == "__main__":
    main()
