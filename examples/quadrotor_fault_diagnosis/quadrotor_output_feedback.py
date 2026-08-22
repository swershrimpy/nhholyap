"""
Quadrotor Output-Feedback Separating Controller (intersection refinement)
=========================================================================
The quadrotor counterpart of
examples/admire_fault_diagnosis/admire_output_feedback.py, following the
same layout: encode the controller INSIDE the System's `f` and pass the
controller parameters as the natural embedding's `u`, so immrax bounds the
closed-loop vector field and this folder's existing interval machinery --
`euler_step`, `propagate_with_refinement`, `_propagate_history`,
`_run_unrolled_or_loop_nocheckpoint` -- applies unchanged with `u_step`
reinterpreted as `theta_step`. Nothing about the refinement is restated
here; it is imported from quadrotor_separating_input.py.

Controller
----------
    u_k = clip(K_k @ y_k + r_k,  U_LO, U_HI)

    u : [U1, U2, U3, U4]  thrust (N) and roll/pitch/yaw moments (N*m)
    y : the MEASURED subset of the 12 states, `OUTPUT_IDX`, default the
        IMU-like set [phi, theta, psi, p, q, r] (attitude + body rates)

    theta_k = [K_k.flatten() (4*len(OUTPUT_IDX)), r_k (4)]
    theta_seq : (num_steps, THETA_LEN), or a single vector for the
                single-stage (time-invariant) controller, which is the
                default -- see "Gain structure" below for why small,
                sparse controllers are preferred here.

The open-loop sequence is the K=0 face of this parameterisation, so any
open-loop solution from this folder is a feasible warm start, and
`gain_mask=0` collapses the law to constant open-loop u = clip(r) for a
like-for-like baseline sharing one compile.

Three differences from the ADMIRE module, all forced by this system
------------------------------------------------------------------
1. PER-CHANNEL GAIN BOUNDS. ADMIRE's 10 surface deflections all share one
   +-0.05 rad box, so a single scalar _K_MAX works. Here the control
   channels differ in scale by ~1800x: thrust has a half-range of 0.176 N
   (0.5..1.5 x hover) while the moment channels have ~1e-4 N*m. One scalar
   bound would be simultaneously far too loose for the moments and far too
   tight for thrust. `_K_MAX_VEC` is therefore per-INPUT-channel, expressed
   as `_K_REL` times that channel's own control half-range -- the same
   convention `_U_NOISE_SCALE` already uses in quadrotor_separating_input.py.
   Adam's per-coordinate `_STEP_SCALE` handles the same disparity in the
   update rule.

2. FEEDFORWARD CENTRED ON HOVER. r must start near [hover_thrust, 0, 0, 0],
   not at zero -- a zero-thrust restart is not a perturbation of hover, it
   is freefall. `_init_theta` centres r there, matching
   `optimize_refined_gpu`'s `u_init`.

3. WHAT "OUTPUT" MEANS. This folder has NO sensor-fault model: its
   observation map is the identity, so `propagate_with_refinement` refines
   on the full 12-state interval (see its docstring and PLAN.md Sec 3).
   Diagnosis therefore uses the whole state, while the CONTROLLER reads
   only `OUTPUT_IDX`. That asymmetry is deliberate and standard -- a
   set-valued observer estimating the full state, driving a controller that
   consumes measured outputs -- but it does mean `OUTPUT_IDX` selects the
   controller's inputs ONLY, and changing it does not change what the
   diagnosis is allowed to see. Set OUTPUT_IDX to all 12 indices to recover
   full state feedback.

Gain structure
--------------
`--gain-structure` picks how many entries of K may be nonzero: `dense`
(every input reads every measured output), `diagonal` (only K[i,i]), or
`rate` (the physically natural quadrotor pairing -- roll moment reads p,
pitch reads q, yaw reads r, thrust gets no feedback).

This is not a runtime knob. Measured on the ADMIRE counterpart, going from
30 gains to 3 changed runtime by 1.6% (2532ms -> 2490ms) -- the interval
matvec is not what feedback costs. What it changed was CONSERVATISM: the
dense controller reached 54/55 pairs with 2.7x box growth and never fully
separated, while the 3-gain diagonal one reached 55/55 with 1.4x growth.
Each nonzero K_ij injects |K_ij| * width(y_j) of extra interval width per
step, so fewer feedback paths mean tighter boxes, and tighter boxes
separate more easily. Expect the same trade here, more sharply: this
system's `_overlap_volume` is a product over TWELVE dimensions, so it is
even more sensitive to box growth than ADMIRE's three.

Objective
---------
`pair_cost_fn` defaults to `_soft_separation_loss`, NOT the
`_overlap_volume` that quadrotor_separating_input.py keeps for backward
compatibility. That module's own docstrings and RESULTS.md record why: a
product of 12 per-dimension overlap widths reads as ~0, with a vanishing
gradient, while every dimension still substantially overlaps, because the
boxes are small in absolute terms at these near-hover horizons. Feedback
makes that worse by widening the boxes unevenly. Pass
`--pair-cost overlap-volume` to reproduce the old behaviour.

Defaults
--------
dt=0.1, 5 steps (a 0.5s horizon) and x0 half-width 0.01, matching
solve_and_plot.py so results compare directly against the open-loop study
that certifies separation on segments 1-4 there. Single-stage, Adam, 32
restarts x 60 iterations.

Usage:
  /home/user/immrax-venv/bin/python quadrotor_output_feedback.py --help
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

from quadrotor_separating_input import (
    Scenario,
    _HOVER_THRUST,
    _U_HI,
    _U_LO,
    _memory_snapshot,
    _overlap_volume,
    _propagate_history,
    _run_unrolled_or_loop_nocheckpoint,
    _soft_separation_loss,
    create_scenarios,
    get_system_and_embedding,
    propagate_with_refinement,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Controller parameterisation
# ══════════════════════════════════════════════════════════════════════════════

NUM_STATES = 12
NUM_INPUTS = 4

# Measured outputs: attitude + body rates, an IMU-like set. Indices into
# x = [x, y, z, phi, theta, psi, u, v, w, p, q, r].
OUTPUT_IDX = (3, 4, 5, 9, 10, 11)
NUM_OUTPUTS = len(OUTPUT_IDX)
NUM_GAINS = NUM_INPUTS * NUM_OUTPUTS
THETA_LEN = NUM_GAINS + NUM_INPUTS

_U_HALFRANGE = (_U_HI - _U_LO) / 2                    # (4,)

# Gain bound, per INPUT channel (see docstring point 1). _K_REL = 10 means a
# unit measured output may command at most 10x that channel's control
# half-range before the clip -- deliberately generous, since y is O(0.01)
# rad and O(0.1) rad/s near hover, so a tighter bound would leave the
# feedback unable to do anything. Tighten it if box growth is the binding
# problem, which the ADMIRE results suggest it will be.
_K_REL = 10.0
_K_MAX_VEC = _K_REL * _U_HALFRANGE                    # (4,)

# theta = [K.flatten() row-major over (NUM_INPUTS, NUM_OUTPUTS), r (4)]
_THETA_LO = jnp.concatenate([
    jnp.repeat(-_K_MAX_VEC, NUM_OUTPUTS), _U_LO])
_THETA_HI = jnp.concatenate([
    jnp.repeat(_K_MAX_VEC, NUM_OUTPUTS), _U_HI])
_STEP_SCALE = (_THETA_HI - _THETA_LO) / 2


def _project_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """Clip controller parameters (or any batch) onto the feasible box."""
    return jnp.clip(theta, _THETA_LO, _THETA_HI)


def theta_to_K_r(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return theta[:NUM_GAINS].reshape(NUM_INPUTS, NUM_OUTPUTS), theta[NUM_GAINS:]


def K_r_to_theta(K: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([K.flatten(), r])


# ── Gain structures ───────────────────────────────────────────────────────────
# Masks over the (NUM_INPUTS, NUM_OUTPUTS) gain matrix. Applied as a constant
# multiply so one code path serves every structure; XLA folds the zeros away.
def _gain_mask_matrix(structure: str) -> jnp.ndarray:
    if structure == 'dense':
        return jnp.ones((NUM_INPUTS, NUM_OUTPUTS))
    if structure == 'diagonal':
        return jnp.eye(NUM_INPUTS, NUM_OUTPUTS)
    if structure == 'rate':
        # roll moment <- p, pitch <- q, yaw <- r; thrust gets no feedback.
        # OUTPUT_IDX order is (phi, theta, psi, p, q, r), so p/q/r are 3,4,5.
        m = np.zeros((NUM_INPUTS, NUM_OUTPUTS))
        for u_i, y_i in ((1, 3), (2, 4), (3, 5)):
            m[u_i, y_i] = 1.0
        return jnp.asarray(m)
    raise ValueError(f"unknown gain structure {structure!r}")


GAIN_STRUCTURES = ('dense', 'diagonal', 'rate')


def _select_outputs(x: jnp.ndarray) -> jnp.ndarray:
    """Gather the measured outputs out of the state, via STATIC slices only.

    Deliberately not `x[jnp.asarray(OUTPUT_IDX)]`. This runs inside the
    natural embedding's traced `f`, and fancy indexing lowers to a `gather`
    whose index clamping emits `lt` -- which immrax's inclusion registry has
    no rule for, so the embedding fails to build with
    `NotImplementedError: lt not in inclusion_registry`. A concatenation of
    single-element static slices lowers to slice + concatenate, both of
    which have inclusion rules, and generalises to any OUTPUT_IDX.
    """
    return jnp.concatenate([x[i:i + 1] for i in OUTPUT_IDX])


def closed_loop_input(theta_step: jnp.ndarray, y: jnp.ndarray,
                      mask: jnp.ndarray) -> jnp.ndarray:
    """The control actually applied given a measured output y.

    The deployable half of a solved controller; everything else in this
    module predicts what this law does to the reachable sets.
    """
    K, r = theta_to_K_r(theta_step)
    return jnp.clip((K * mask) @ y + r, _U_LO, _U_HI)


def expand_theta(theta: jnp.ndarray, num_steps: int) -> jnp.ndarray:
    """Broadcast a decision variable to the (num_steps, THETA_LEN) sequence
    the loss consumes. A 1-D input is the single-stage controller."""
    if theta.ndim == 1:
        return jnp.broadcast_to(theta, (num_steps, THETA_LEN))
    return theta


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Closed-loop system & scenarios
# ══════════════════════════════════════════════════════════════════════════════

def make_cl_embedding(structure: str = 'dense', **phys):
    """Natural embedding of the quadrotor under u = clip(K @ y + r).

    `f` delegates to the SAME QuadrotorSystem instance the open-loop path
    uses (quadrotor_separating_input's cached singleton) rather than
    re-transcribing the Newton-Euler dynamics, so the closed-loop and
    open-loop studies cannot drift apart on a physical constant.
    """
    ol_sys, _ = get_system_and_embedding(**phys)
    mask = _gain_mask_matrix(structure)

    class QuadrotorOutputFeedbackCLSystem(irx.System):
        """u = theta = [K.flatten(), r] (concrete); p = alpha (interval)."""

        def __init__(self):
            self.evolution = 'continuous'
            self.xlen = NUM_STATES

        def f(self, t, x, u, p):
            return ol_sys.f(t, x, closed_loop_input(u, _select_outputs(x), mask), p)

    return irx.natemb(QuadrotorOutputFeedbackCLSystem())


def create_cl_scenarios(structure: str = 'dense', alpha_lo: float = 0.5,
                        alpha_hi: float = 0.9, **phys) -> List[Scenario]:
    """The 5 open-loop fault scenarios, rebound to the closed-loop embedding.

    Derived from `create_scenarios` rather than restating the alpha
    intervals, so the fault set stays identical to the open-loop study by
    construction. All 5 share one embedding, which is what lets
    `propagate_with_refinement` vmap over scenarios and pairs.
    """
    emb = make_cl_embedding(structure, **phys)
    return [dataclasses.replace(s, emb_system=emb)
            for s in create_scenarios(alpha_lo=alpha_lo, alpha_hi=alpha_hi, **phys)]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Loss  (intersection refinement)
# ══════════════════════════════════════════════════════════════════════════════

def refined_separation_loss(theta_seq: jnp.ndarray, x0_ivl: irx.Interval,
                            cl_scenarios: List[Scenario], dt: float,
                            num_steps: int, pair_cost_fn=_soft_separation_loss):
    """Min-over-steps pairwise separation cost under output feedback.

    Thin alias for `quadrotor_separating_input.propagate_with_refinement`:
    that function only ever forwards its control argument to `euler_step` ->
    `emb_sys.f`, so with the closed-loop embedding it consumes theta rows
    exactly as it consumed control rows. Same vmap over the 10 scenario
    pairs, same refinement, same cost function. Note the argument ORDER
    differs from the ADMIRE module's (x0 first here).
    """
    return propagate_with_refinement(x0_ivl, theta_seq, cl_scenarios, dt,
                                     num_steps, pair_cost_fn)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def theta0_from_open_loop(u_seq: jnp.ndarray) -> jnp.ndarray:
    """Turn an open-loop control sequence into the equivalent theta_seq (K=0)."""
    return jnp.concatenate(
        [jnp.zeros((u_seq.shape[0], NUM_GAINS)), u_seq], axis=-1)


def _init_theta(key, num_restarts: int, num_steps: int, shared_theta: bool,
                init_std_gain: float, theta0_mean: Optional[jnp.ndarray]):
    """Sample restarts: gains scaled per input channel, r centred on HOVER.

    Centring r on hover (not zero) matters -- see docstring point 2. Gains
    are drawn relative to each channel's own bound so the thrust and moment
    rows get comparable *relative* excitation despite the ~1800x scale gap.
    """
    shape = (num_restarts,) if shared_theta else (num_restarts, num_steps)
    k_gain, k_ff = jax.random.split(key)
    gains = (jax.random.normal(k_gain, (*shape, NUM_GAINS)) * init_std_gain
             * jnp.repeat(_K_MAX_VEC, NUM_OUTPUTS))
    r_hover = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    ff = r_hover + jax.random.normal(k_ff, (*shape, NUM_INPUTS)) * 0.1 * _U_HALFRANGE
    theta0 = jnp.concatenate([gains, ff], axis=-1)
    if theta0_mean is not None:
        # Restart 0 is theta0_mean EXACTLY, not a perturbation of it --
        # otherwise a warm start never evaluates the point it starts from
        # and can return worse than its own seed.
        theta0 = theta0.at[0].set(jnp.zeros_like(theta0[0])) + theta0_mean[None, ...]
    return _project_theta(theta0)


def optimize_output_feedback_gpu(x0_ivl: irx.Interval, cl_scenarios: List[Scenario],
                                 dt: float, num_steps: int,
                                 num_restarts: int = 32, learning_rate: float = 0.05,
                                 num_iters: int = 60, seed: int = 42,
                                 init_std_gain: float = 0.3,
                                 shared_theta: bool = True,
                                 theta0_mean: Optional[jnp.ndarray] = None,
                                 restart_chunk: Optional[int] = None,
                                 optimizer: str = 'adam',
                                 gain_mask=1.0,
                                 pair_cost_fn=_soft_separation_loss):
    """Multi-start projected Adam on the refined output-feedback loss.

    Mirrors the ADMIRE module's optimizer -- fused value_and_grad (one
    forward trace, not two), a scanned outer loop, NaN-safe restart
    selection, the traced `gain_mask` open-loop arm and the `restart_chunk`
    memory knob -- with this folder's `_run_unrolled_or_loop_nocheckpoint`
    as the GD driver instead of ADMIRE's `_scan_loop`.

    `optimizer`: 'adam' (default) or 'gd'. Plain GD is kept only to
    reproduce older results: measured on the ADMIRE counterpart, 60 GD
    iterations at lr=0.05 moved the winning restart's gains by 1.2e-4
    against a norm of 4.93, i.e. the reported optimum was just the best
    random draw. The separation costs here are small in absolute terms for
    exactly the reason `_soft_separation_loss` exists, so the same failure
    applies; Adam divides that scale out.

    `dt` and `gain_mask` may be TRACED scalars -- neither changes the graph,
    so one compile can sweep step sizes and run both arms. `num_steps` is a
    scan length and must stay static.

    Returns (best_theta, best_loss, theta_final, losses) in DECISION-variable
    shape, gain mask applied.
    """
    theta0 = _init_theta(jax.random.PRNGKey(seed), num_restarts, num_steps,
                         shared_theta, init_std_gain, theta0_mean)

    def _mask(theta):
        return theta.at[..., :NUM_GAINS].multiply(gain_mask)

    def loss_fn(theta):
        return refined_separation_loss(expand_theta(_mask(theta), num_steps),
                                       x0_ivl, cl_scenarios, dt, num_steps,
                                       pair_cost_fn)

    value_and_grad = jax.value_and_grad(loss_fn)
    if restart_chunk is None or restart_chunk >= num_restarts:
        batched = jax.vmap(value_and_grad)
    elif num_restarts % restart_chunk:
        raise ValueError(
            f"restart_chunk ({restart_chunk}) must divide num_restarts "
            f"({num_restarts}) exactly -- a remainder makes lax.map compile a "
            "second vmapped body for the leftovers.")
    else:
        batched = lambda tb: jax.lax.map(value_and_grad, tb, batch_size=restart_chunk)

    if optimizer == 'gd':
        def body(carry, _i):
            theta, m, v, t = carry
            _, g = batched(theta)
            return (_project_theta(theta - learning_rate * g), m, v, t)
    elif optimizer == 'adam':
        b1, b2, eps = 0.9, 0.999, 1e-8

        def body(carry, _i):
            theta, m, v, t = carry
            _, g = batched(theta)
            t = t + 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            step = (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
            return (_project_theta(theta - learning_rate * _STEP_SCALE * step),
                    m, v, t)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'gd', got {optimizer!r}")

    zeros = jnp.zeros_like(theta0)
    theta_final, _, _, _ = _run_unrolled_or_loop_nocheckpoint(
        body, (theta0, zeros, zeros, jnp.zeros((), jnp.int32)), num_iters)
    losses, _ = batched(theta_final)
    theta_final = _mask(theta_final)

    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx = jnp.argmin(losses_valid)
    return theta_final[best_idx], losses[best_idx], theta_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Evaluation  (raw, un-refined)
# ══════════════════════════════════════════════════════════════════════════════

def raw_state_histories(x0_ivl: irx.Interval, theta_seq: jnp.ndarray,
                        cl_scenarios: List[Scenario], dt: float) -> irx.Interval:
    """Per-scenario closed-loop state history with NO cross-scenario
    refinement. Shapes (n_scenarios, num_steps, 12)."""
    emb_sys = cl_scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in cl_scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in cl_scenarios]),
    )
    return jax.vmap(
        lambda p: _propagate_history(x0_ivl, theta_seq, emb_sys, p, dt, 1)
    )(p_batch)


def raw_pair_separated(hist: irx.Interval) -> np.ndarray:
    """(n_pairs, num_steps) bool: is this pair's raw STATE box disjoint here?

    On the full 12-state, because this folder's observation map is the
    identity -- diagnosis sees the whole state (docstring point 3), so
    that is the space separation must be checked in, not `OUTPUT_IDX`.
    """
    lo, hi = np.asarray(hist.lower), np.asarray(hist.upper)
    ii, jj = np.triu_indices(lo.shape[0], k=1)
    return np.any((lo[ii] > hi[jj]) | (lo[jj] > hi[ii]), axis=-1)


def raw_disjoint_segments(hist: irx.Interval) -> List[int]:
    """Steps at which ALL scenario pairs' raw state boxes are disjoint."""
    sep = raw_pair_separated(hist)
    return [k for k in range(sep.shape[1]) if bool(np.all(sep[:, k]))]


def raw_disjoint_pair_counts(hist: irx.Interval) -> np.ndarray:
    """(num_steps,) count of separated pairs, out of n*(n-1)/2.

    The partial-credit metric: once several configurations drive the refined
    loss to ~0 it stops discriminating, while all-or-nothing disjointness is
    too coarse to show which is closer.
    """
    return raw_pair_separated(hist).sum(axis=0)


def state_box_growth(x0_ivl: irx.Interval, hist: irx.Interval) -> float:
    """Largest state-box width over the horizon, relative to the initial one.

    The blow-up detector. Interval conservatism under feedback shows up here
    long before it shows up in the loss, and this system's 12-dimensional
    overlap product is especially sensitive to it.
    """
    w0 = float(jnp.max(x0_ivl.upper - x0_ivl.lower))
    w = float(np.max(np.asarray(hist.upper) - np.asarray(hist.lower)))
    return w / w0 if w0 > 0 else float('inf')


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Demo
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Solve a quadrotor output-feedback separating controller "
                    "(intersection refinement).")
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--steps', type=int, default=5, help="horizon in steps")
    p.add_argument('--restarts', type=int, default=32)
    p.add_argument('--iters', type=int, default=60)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--x0-halfwidth', type=float, default=0.01)
    p.add_argument('--init-std-gain', type=float, default=0.3)
    p.add_argument('--gain-structure', choices=GAIN_STRUCTURES, default='dense')
    p.add_argument('--pair-cost', choices=['soft', 'overlap-volume'], default='soft',
                   help="'soft' is the margin proxy; 'overlap-volume' is the "
                        "12-dimension width product kept for compatibility")
    p.add_argument('--optimizer', choices=['adam', 'gd'], default='adam')
    p.add_argument('--restart-chunk', type=int, default=None)
    p.add_argument('--shared-theta', action=argparse.BooleanOptionalAction,
                   default=True,
                   help="single-stage: ONE time-invariant (K, r) for the whole "
                        "horizon (default)")
    p.add_argument('--baseline', action='store_true',
                   help="also solve the gain_mask=0 open-loop arm for a "
                        "like-for-like comparison (shares one compile)")
    p.add_argument('--out', type=str, default='results/quadrotor_output_feedback.npz')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    print(f"Devices: {jax.devices()}")

    pair_cost_fn = _soft_separation_loss if args.pair_cost == 'soft' else _overlap_volume
    cl_scenarios = create_cl_scenarios(structure=args.gain_structure)
    n = len(cl_scenarios)
    n_pairs = n * (n - 1) // 2
    n_gains = int(_gain_mask_matrix(args.gain_structure).sum())
    print(f"{n} closed-loop scenarios / {n_pairs} pairs; "
          f"gain structure '{args.gain_structure}' ({n_gains} free gains), "
          f"outputs {OUTPUT_IDX}")

    # Hover equilibrium, perturbed uniformly -- matches this folder's demos.
    x0_ivl = irx.icentpert(jnp.zeros(NUM_STATES),
                           jnp.ones(NUM_STATES) * args.x0_halfwidth)
    print(f"Horizon: {args.steps} x {args.dt}s = {args.steps*args.dt:.3f}s   "
          f"restarts={args.restarts}  iters={args.iters}  lr={args.lr}  "
          f"pair-cost={args.pair_cost}")

    def run(seed, gain_mask):
        return optimize_output_feedback_gpu(
            x0_ivl=x0_ivl, cl_scenarios=cl_scenarios, dt=args.dt,
            num_steps=args.steps, num_restarts=args.restarts,
            learning_rate=args.lr, num_iters=args.iters, seed=seed,
            init_std_gain=args.init_std_gain, shared_theta=args.shared_theta,
            restart_chunk=args.restart_chunk, optimizer=args.optimizer,
            gain_mask=gain_mask, pair_cost_fn=pair_cost_fn)

    jitted = jax.jit(run)
    print("\nCompiling (trig-heavy embedding: tan(theta), 1/cos(theta)) ...")
    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(args.seed, jnp.float32(1.0)))
    compile_t = time.perf_counter() - t0
    print(f"Compile time: {compile_t:.1f}s")

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(args.seed, jnp.float32(1.0)))
    run_t = time.perf_counter() - t0
    print(f"Post-compile run time: {run_t*1e3:.2f}ms")
    print(f"Memory: {_memory_snapshot()}")

    arms = [('feedback', out)]
    if args.baseline:
        arms.append(('open-loop', jax.block_until_ready(
            jitted(args.seed, jnp.float32(0.0)))))

    rows = []
    for arm, (best_theta, best_loss, _, losses) in arms:
        theta_seq = expand_theta(best_theta, args.steps)
        K, r = theta_to_K_r(theta_seq[0])
        hist = raw_state_histories(x0_ivl, theta_seq, cl_scenarios, args.dt)
        counts = raw_disjoint_pair_counts(hist)
        full = raw_disjoint_segments(hist)
        growth = state_box_growth(x0_ivl, hist)
        n_nan = int(jnp.sum(jnp.isnan(losses)))
        print(f"\n{arm}:  loss={float(best_loss):.6e}  ({n_nan}/{args.restarts} NaN)")
        print(f"  max pairs separated = {int(counts.max())}/{n_pairs}   "
              f"per step {list(map(int, counts))}")
        print(f"  fully-separated steps = {full}   state-box growth = {growth:.1f}x")
        print(f"  |K|max per channel = "
              f"{[float(v) for v in jnp.max(jnp.abs(K), axis=1)]}")
        print(f"  r = {[float(v) for v in r]}  (hover thrust {_HOVER_THRUST:.4f})")
        rows.append(dict(arm=arm, loss=float(best_loss), growth=growth,
                         max_sep=int(counts.max()), n_full=len(full),
                         first_full=(full[0] if full else -1),
                         theta=np.asarray(best_theta)))

    # Box growth is only interpretable RELATIVE to the open-loop arm. An
    # absolute threshold is meaningless across systems: measured here, the
    # gain_mask=0 baseline itself grows ~86x at a 0.5s horizon over 12 states,
    # so an ADMIRE-calibrated ">10x means feedback is blowing up" rule fires on
    # a controller with K=0 and diagnoses nothing. What matters is the excess
    # growth feedback adds over its own baseline.
    if len(rows) == 2:
        fb, ol = rows[0], rows[1]
        excess = fb['growth'] / ol['growth'] if ol['growth'] > 0 else float('inf')
        print(f"\nFeedback vs open-loop: box growth {fb['growth']:.1f}x vs "
              f"{ol['growth']:.1f}x ({excess:.2f}x excess), "
              f"first fully-separated step {fb['first_full']} vs {ol['first_full']}")
        if excess > 2.0:
            print("  WARNING: feedback more than doubles the baseline's box "
                  "growth -- its interval conservatism is dominating; try "
                  "--gain-structure rate/diagonal or a smaller _K_REL.")
    elif rows[0]['growth'] > 10.0:
        print(f"\nNote: box growth {rows[0]['growth']:.1f}x. Re-run with "
              "--baseline to see whether feedback caused it -- this system's "
              "open-loop arm grows comparably on its own.")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _HERE / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             **{k: np.array([r[k] for r in rows])
                for k in ('arm', 'loss', 'growth', 'max_sep', 'n_full',
                          'first_full')},
             theta=np.stack([r['theta'] for r in rows]),
             dt=args.dt, num_steps=args.steps,
             gain_structure=args.gain_structure, output_idx=np.array(OUTPUT_IDX),
             shared_theta=args.shared_theta, k_rel=_K_REL,
             x0_halfwidth=args.x0_halfwidth,
             compile_t=compile_t, run_t=run_t)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
