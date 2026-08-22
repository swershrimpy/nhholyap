"""
ADMIRE Output-Feedback Separating Controller (intersection refinement)
======================================================================
Solves for an OUTPUT-FEEDBACK controller that drives the 11 ADMIRE fault
scenarios' reachable angular-rate sets apart, instead of the open-loop
control SEQUENCE solved for by `admire_separating_input.py`.

Controller
----------
At step k the aircraft measures its own angular rates and applies

    u_k = clip(K_k @ y_k + r_k,  U_LO, U_HI),      y_k = x_k[3:6] = [pb,qb,rb]

with K_k a 10x3 gain and r_k a 10-vector feedforward, so the decision
variable per step is

    theta_k = [K_k.flatten() (30), r_k (10)]  in R^40
    theta_seq : (num_steps, 40)               (or a single (40,) with
                                               shared_theta=True -- the
                                               time-invariant (K, r)
                                               parameterisation used by
                                               faulty_car_output_feedback_cbf.py's
                                               primary refined solver)

The open-loop sequence is the K=0 face of this parameterisation, so any
open-loop solution is a feasible warm start (see `theta0_from_open_loop`).

Ported from examples/faulty_car/faulty_car_output_feedback_cbf.py (the
unicycle/car output-feedback solver), keeping its central idea: encode the
controller INSIDE the System's `f`, and pass theta as the `u` argument of
the natural embedding. That way immrax bounds the closed-loop vector field
directly and the whole existing interval machinery -- Euler step, pairwise
intersection refinement, batched GD -- applies unchanged with `u_step`
reinterpreted as `theta_step`. Concretely, this module adds a system, a
scenario constructor, a parameterisation and an optimizer, and reuses
`admire_separating_input`'s `_step_one_pair` / `propagate_with_refinement`
/ `_propagate_history` / `_scan_loop` verbatim rather than restating the
refinement math (the recurring "independent reimplementations drift apart"
failure mode recorded in that module's docstring and unicycle's PLAN.md).

Two simplifications vs. the car version, both deliberate:

1. No tracking form. The car has a `u = clip(K(y - y_hat) + u_ff)` variant
   with a reference sequence y_hat. Here y_hat would be pure redundancy:
   K(y - y_hat) + u_ff = K y + (u_ff - K y_hat), and since y_hat is a
   constant w.r.t. the interval arithmetic, both the value and the interval
   WIDTH of the pre-clip control are identical to the K y + r form (the
   width of K@y depends only on |K| and width(y)). Only the centre moves,
   and r already parameterises the centre.
2. No observation model in `p`. The car needed obs_scale/obs_offset in p to
   share one embedding across a SENSOR-fault scenario. Every ADMIRE
   scenario here is an actuator fault observed through the same identity
   map y = x[3:6], so all 11 already share one embedding (`_CL_EMB`) and
   one vmap, and `_step_one_pair`'s refinement -- which writes the pair's
   intersected output straight back into x[3:6] -- needs no inverse
   observation map.

Why output feedback is not free (interval conservatism)
------------------------------------------------------
The natural inclusion function evaluates clip(K y + r) over the box for y
and then the plant over the box for x INDEPENDENTLY, losing the fact that
both y's are the same variable. So the closed-loop boxes are wider than the
true closed-loop reachable sets, by roughly |K| * width(y) per step, and
that extra width feeds back into the next step. Large gains therefore blow
the boxes up rather than separating them -- exactly the effect logged at
length in faulty_car_output_feedback_cbf.py's `_K_MAX` history (K_MAX >= 3
there produced loss=0 "solutions" whose boxes had grown 10-16x and were
physically meaningless). `_K_MAX` below is this module's counterpart knob
and is deliberately conservative; `main()` reports box growth alongside the
loss so a blown-up "solution" is visible instead of silently reported as a
separation certificate.

ADMIRE additionally divides by Vt, cos(beta) and cos(theta) and takes
tan(theta) (see admire.py), so a box that grows until it contains
theta = +-pi/2 produces Inf/NaN rather than a merely loose bound. Restart
selection is NaN-safe, and `main()` prints the NaN count.

What this solver actually found (and an open defect)
----------------------------------------------------
Measured at dt=0.15 x 10 steps -- the horizon at which the OPEN-LOOP
unrefined demo achieves full separation (multistep_unrefined_result.npz:
loss 0.0, all 55 pairs raw-disjoint at segments 8 and 9):

  seed                     best loss   raw-disjoint   box growth  |K|max
  warm (open-loop u_seq)     0.0e+00     [8, 9]          1.8x      0.0000
  cold (random restarts)     1.2e-05     []             40.9x      0.7444

The warm run's winner is restart 0, i.e. the K=0 open-loop seed ITSELF --
all 31 perturbed restarts scored worse. The cold run's genuine feedback
(|K|max 0.74) grew the boxes 40.9x and separated nothing, against 1.1x
growth for the same open-loop u_seq at dt=0.1. So on the evidence so far
output feedback does not help this problem: the feedback term's interval
conservatism (see above) costs more than the gains buy.

That conclusion is NOT yet safe, because the optimizer is barely searching.
Over 60 iterations at learning_rate=0.05 the cold run's winning restart
moved its gains by ||K_final - K_init|| = 1.2e-4 against ||K_init|| = 4.93
-- a relative movement of 2.4e-5 -- so the reported "optimum" is just the
best of 32 random draws, not a descended one. (Tell-tale: |K|max comes out
as the same 0.7444 at dt=0.1 and dt=0.15, because it is the same draw.)
This is exactly the pathology logged in
faulty_car_output_feedback_cbf.py's `optimize_output_feedback_cbf_vmapped_gpu`
docstring, and the mechanism is scale: the loss is a min-over-steps of
overlap VOLUMES, ~1e-5 here, with non-overlapping pairs masked to zero, so
only one step and a handful of pairs carry gradient and the raw magnitudes
are ~1e-4. A plain GD step of lr * g is then ~1e-5 per iteration. FIXED by
`optimizer='adam'` (now the default), which divides that scale out: 5 Adam
iterations move theta by 2.194 where 60 GD iterations moved it by 1.2e-4.

With Adam AND the single-stage parameterisation the conclusion above
REVERSES -- output feedback does help, and the per-step schedule was the
wrong shape for the problem. Measured (single-stage, 32 restarts x 60
iters, x0 half-width 5e-3), reporting time to full separation of all 55
pairs, feedback vs its own gain_mask=0 open-loop arm:

    N   dt     horizon   feedback      open-loop
    1   0.20    0.20s     0.20s         never
    1   0.25    0.25s     0.25s         never
    1   0.50    0.50s     0.50s         never
    3   0.50    1.50s     0.50s         1.50s
    6   0.25    1.50s     1.00s         1.25s
   15   0.10    1.50s     never         1.20s
   30   0.05    1.50s     0.95s         1.05s

Feedback separates EARLIER wherever it separates at all, and is the only
thing that separates in a single Euler step -- but it costs 3-8x box growth
against open loop's 1.1-1.5x, it fails at dt=0.1, and `_K_MAX` binds in 13
of 16 feedback rows, so those rows are bound-limited rather than converged.

Which criterion counts
----------------------
Two different questions, and this module reports both:

  REFINED separation (loss == 0) is the diagnosability certificate. The
  recursion tracks the AMBIGUOUS set -- the states of each model still
  consistent with a measurement history explainable by BOTH -- so an empty
  intersection means no measurement exists that both models can explain,
  and whichever is true the other is refuted. It is worst-case over
  measurements: `_step_one_pair` declares separation only when the
  intersection is empty for ALL possible y, never counting on the actual
  measurement falling outside the other model's prediction. Because the
  cost of an already-separated pair is held at 0, `loss == 0` at step k
  means every pair separated at or before k -- the correct
  accumulate-refutations-over-time reading, not "all separated at one
  instant".

  RAW disjointness (`raw_disjoint_segments`) is STRICTLY STRONGER: are the
  models distinguishable from a SINGLE snapshot, with no measurement
  history? Deployable without a set-valued observer, but not what active
  fault diagnosis requires. Earlier notes in this repo treat it as the
  honest check and the refined loss as a mere proxy; that understates the
  refined result. Both are reported because they answer different
  questions.

  Worked example (N=10, dt=0.1, single-stage feedback): at t=1.0s all 55
  pairs separate on refined boxes -- loss exactly 0, i.e. fully diagnosable
  within 1.0s -- while 54/55 separate on raw boxes. The holdout is Nominal
  vs Flap, this project's recorded hardest pair, never raw-separated at any
  step of a 1.0s horizon (the open-loop demo needs 1.5s to close it).

Two caveats survive both criteria. Soundness is w.r.t. the forward-Euler
MAP, not the ODE -- which bites hardest on the coarse single-step results
above, whose whole advantage rests on a dt a validated integrator might not
support. And there is no measurement noise: y = x[3:6] exactly, so real
sensor noise would need Y bloated by the noise bound before intersecting,
shrinking margins that are already ~1e-9 at the final step.

Where the vmaps and scans go (and why)
--------------------------------------
Per the resource goals -- fast runtime, fast compile, low host RAM, low
VRAM -- the axes are mapped as follows. Compile cost for ADMIRE is
dominated by reverse-mode AD through the division-heavy dynamics and is
proportional to the number of DISTINCT traces of `f` in the jaxpr, not to
horizon/restarts/iterations:

  axis                     construct           why
  -----------------------  ------------------  --------------------------
  11 scenarios (step 1)    jax.vmap            one traced `f`, not 11;
                                               all scenarios share _CL_EMB
  55 scenario pairs        jax.vmap            one traced `_step_one_pair`,
                                               not 55
  num_steps horizon        lax.scan            graph size O(1) in horizon
                           + jax.checkpoint    residuals O(carry) per step,
                                               not O(step intermediates)
  num_iters GD loop        lax.scan            graph size O(1) in iters
  num_restarts             jax.vmap            parallel; the ONLY axis that
                           (or lax.map with    multiplies live VRAM, hence
                            batch_size=chunk)  the chunking knob
  loss + gradient          value_and_grad      one forward trace, not two

The jaxpr therefore holds exactly three traces of the closed-loop `f` (the
step-1 vmap, and the two sides of `_step_one_pair`) regardless of horizon,
iteration count or restart count.

`restart_chunk` caps the restart axis: setting it to c evaluates restarts c
at a time under one scan (lax.map's batch_size) instead of all at once. It
must divide num_restarts exactly -- a remainder makes lax.map compile a
SECOND vmapped body for the leftovers.

It was added as a VRAM knob, but on this hardware it is ALSO the faster
setting, which is the opposite of the runtime-for-memory trade it looks
like. Measured (RTX 4070 Laptop, 11 scenarios / 55 pairs, 10 steps,
32 restarts x 60 iters, dt=0.15):

    restarts  iters  chunk   compile    run      per-iter
      32       60     -       111.5s   2513.9ms   41.9ms
      32      120     -       124.5s   5058.7ms   42.2ms
       8       60     -       120.2s    365.5ms    6.1ms
      32       60     8       184.5s   1301.1ms   21.7ms   (same result)

Two things to read off. Run time is exactly linear in num_iters (2.02x for
2x the iterations, ~zero intercept), so wall = num_iters * per-iter. But
per-iter is SUPERLINEAR in restarts: 4x the restarts (8 -> 32) costs 6.9x
per iteration. The likely cause is cache, not occupancy -- peak device
memory is 76MB at 32 restarts vs ~19MB at 8, against this card's 32MB L2,
and the work is thousands of tiny 18-element kernels where bandwidth and
launch overhead dominate and FLOPs are irrelevant. Chunking keeps the
working set in L2, so four sequential chunks of 8 beat one batch of 32
(1301ms vs 2514ms, 1.93x) at a quarter the live memory, for +73s of
compile (lax.map's scan-over-chunks is more to compile than a plain vmap).

Full measured resource profile at the first row above: compile 111.5s,
run 2.5s, peak DEVICE memory 76.1MB (nvidia-smi shows ~6.2GB, but that is
JAX's preallocated pool -- `pool_bytes` -- not consumption), peak HOST RSS
4.83GB (compile-bound, as for every other loss in this folder).

Defaults
--------
The out-of-the-box configuration is the one benchmarked below: SINGLE-STAGE
(one constant K, r), num_steps=10, dt=0.1 (a 1.0s horizon), Adam, 32 restarts
x 60 iterations, x0 half-width 5e-3. Measured post-compile runtime 2707ms
(median of 20; min 2688, max 2733; 45.1ms per GD iteration) after a 110-154s
compile. N=5 x dt=0.2 over the same 1.0s horizon runs in 1131ms (18.9ms per
iteration) -- 2.39x faster for half the steps, i.e. slightly superlinear in
N with negligible fixed overhead. `--no-shared-theta` switches to the per-step schedule, and
`--optimizer gd` to the non-descending plain-GD step; both are kept only to
reproduce earlier results.

Usage:
  /home/user/immrax-venv/bin/python admire_output_feedback.py --help
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

from admire_separating_input import (
    Scenario,
    _U_LO,
    _U_HI,
    _memory_snapshot,
    _propagate_history,
    _scan_loop,
    create_scenarios,
    get_system_and_embedding,
    propagate_with_refinement,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Controller parameterisation
# ══════════════════════════════════════════════════════════════════════════════

NUM_STATES = 9
NUM_INPUTS = 10          # surface deflections
NUM_OUTPUTS = 3          # [pb, qb, rb] = x[3:6]
THETA_LEN = NUM_INPUTS * NUM_OUTPUTS + NUM_INPUTS      # 30 gain + 10 feedforward

# Gain bound. u is limited to +-0.05 rad and y starts at +-5e-3 rad/s, so a
# gain of 1.0 already contributes width(K@y) ~ 3 * width(y) to the control
# interval -- comparable to the full 0.1 rad control span once the boxes
# have grown a little. Larger bounds buy authority the clip mostly throws
# away while widening every downstream box (see the module docstring's
# conservatism note and the car solver's K_MAX history). Raise it only
# together with the box-growth check `main()` prints.
_K_MAX = 1.0

_THETA_LO = jnp.concatenate([jnp.full(NUM_INPUTS * NUM_OUTPUTS, -_K_MAX), _U_LO])
_THETA_HI = jnp.concatenate([jnp.full(NUM_INPUTS * NUM_OUTPUTS, _K_MAX), _U_HI])


def _project_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """Project controller parameters (or any batch of them) onto the box.

    Broadcasts over every leading axis, so it covers the (40,),
    (num_steps, 40) and (num_restarts, num_steps, 40) shapes alike.
    """
    return jnp.clip(theta, _THETA_LO, _THETA_HI)


# Per-coordinate half-range, used to make the Adam step scale-free: a step
# of `learning_rate` moves each coordinate that fraction of its OWN range.
# Without this the gain block (range 2.0) and the feedforward block (range
# 0.1) would get the same absolute step, i.e. 20x more of r's range than of
# K's per iteration.
_STEP_SCALE = (_THETA_HI - _THETA_LO) / 2


def _apply_gain_mask(theta: jnp.ndarray, gain_mask) -> jnp.ndarray:
    """Scale the gain block by `gain_mask` (a TRACED scalar).

    gain_mask=0 collapses the controller to the constant-feedforward
    (open-loop) law u = clip(r, U_LO, U_HI), which is the honest baseline to
    compare an output-feedback solution against. Keeping it traced rather
    than a Python bool means one compiled graph serves both arms of that
    A/B, instead of paying ADMIRE's ~2-minute compile twice. The gradient
    w.r.t. the masked gains is exactly zero, so those coordinates simply
    never move.
    """
    return theta.at[..., :NUM_INPUTS * NUM_OUTPUTS].multiply(gain_mask)


def theta_to_K_r(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split one 40-vector into its (10, 3) gain and (10,) feedforward."""
    return theta[:NUM_INPUTS * NUM_OUTPUTS].reshape(NUM_INPUTS, NUM_OUTPUTS), theta[NUM_INPUTS * NUM_OUTPUTS:]


def K_r_to_theta(K: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([K.flatten(), r])


def closed_loop_input(theta_step: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """The control actually applied given a measured output y = [pb,qb,rb].

    The deployable half of a solved controller: everything else in this
    module predicts what this law does to the reachable sets.
    """
    K, r = theta_to_K_r(theta_step)
    return jnp.clip(K @ y + r, _U_LO, _U_HI)


def expand_theta(theta: jnp.ndarray, num_steps: int) -> jnp.ndarray:
    """Broadcast a decision variable to the (num_steps, 40) sequence the loss
    consumes. A (40,) input is the time-invariant controller (shared_theta),
    a (num_steps, 40) input is passed through."""
    if theta.ndim == 1:
        return jnp.broadcast_to(theta, (num_steps, THETA_LEN))
    return theta


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Closed-loop system & scenarios
# ══════════════════════════════════════════════════════════════════════════════

_OL_SYS, _ = get_system_and_embedding()


class AdmireOutputFeedbackCLSystem(irx.System):
    """ADMIRE under u = clip(K @ y + r), with theta passed as `u`.

    immrax convention (same as the car CL systems):
        u = theta = [K.flatten() (30), r (10)]   concrete, the optimisation
                                                 variable
        p = actuator effectiveness (10)          abstract/interval-valued,
                                                 what distinguishes scenarios

    `f` delegates to the SAME `AdmireNineDoFLinAct` instance the open-loop
    path uses (`admire_separating_input`'s module singleton) rather than
    re-transcribing the dynamics, so the closed-loop and open-loop studies
    cannot drift apart on an aerodynamic coefficient.
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = NUM_STATES

    def f(self, t, x, u, p):
        y = x[3:6]                                  # measured angular rates
        return _OL_SYS.f(t, x, closed_loop_input(u, y), p)


_CL_SYS = AdmireOutputFeedbackCLSystem()
_CL_EMB = irx.natemb(_CL_SYS)


def get_cl_system_and_embedding():
    return _CL_SYS, _CL_EMB


def create_cl_scenarios(fault_effectiveness: float = 0.0) -> List[Scenario]:
    """The 11 open-loop fault scenarios, rebound to the closed-loop embedding.

    Derived from `create_scenarios` instead of restating the p intervals, so
    the fault set stays identical to the open-loop study by construction.
    All 11 share the single `_CL_EMB`, which is what lets every propagation
    site below be one vmap rather than a Python loop over scenarios.
    """
    return [dataclasses.replace(s, emb_system=_CL_EMB)
            for s in create_scenarios(fault_effectiveness=fault_effectiveness)]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Loss  (intersection refinement)
# ══════════════════════════════════════════════════════════════════════════════

def refined_separation_loss(theta_seq: jnp.ndarray, x0_ivl: irx.Interval,
                            cl_scenarios: List[Scenario], dt: float,
                            num_steps: int) -> jnp.ndarray:
    """Min-over-steps pairwise output-overlap volume under output feedback.

    Thin alias for `admire_separating_input.propagate_with_refinement`: that
    function only ever forwards its control argument to `emb_sys.f`, so with
    the closed-loop embedding it consumes theta rows exactly as it consumed
    control rows -- same vmap-over-55-pairs, same lax.scan over the horizon,
    same jax.checkpoint, same `_step_one_pair` refinement (including its
    fixed no-overlap fallback). Nothing about the refinement is
    reimplemented here.

    With output feedback the refinement does strictly more work than in the
    open-loop case: intersecting a pair's predicted outputs tightens x[3:6],
    which is also the controller's input, so the refined control interval
    tightens too.
    """
    return propagate_with_refinement(theta_seq, x0_ivl, cl_scenarios, dt, num_steps)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def theta0_from_open_loop(u_seq: jnp.ndarray) -> jnp.ndarray:
    """Turn an open-loop control sequence into the equivalent theta_seq (K=0).

    Lets a solved open-loop sequence (e.g. multistep_refined_result.npz's
    `u_seq`) warm-start the output-feedback search from a point whose loss
    is already the open-loop loss, so GD can only improve on it.
    """
    num_steps = u_seq.shape[0]
    return jnp.concatenate(
        [jnp.zeros((num_steps, NUM_INPUTS * NUM_OUTPUTS)), u_seq], axis=-1)


def _init_theta(key, num_restarts: int, num_steps: int, shared_theta: bool,
                init_std_gain: float, init_scale_ff: float,
                theta0_mean: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Sample restarts: Gaussian gains, uniform feedforward over the u box.

    The gain spread matters more than it looks: the car solver found
    (init_std=0.1) that too tight a spread leaves every restart with K ~ 0,
    i.e. an open-loop controller that GD then barely moves, so the reported
    optimum is just the best random draw. Default 0.3 spreads restarts over
    a third of the [-K_MAX, K_MAX] range while staying inside it.
    """
    shape = (num_restarts, THETA_LEN) if shared_theta else (num_restarts, num_steps, THETA_LEN)
    k_gain, k_ff = jax.random.split(key)
    gains = jax.random.normal(k_gain, (*shape[:-1], NUM_INPUTS * NUM_OUTPUTS)) * init_std_gain
    ff = jax.random.uniform(k_ff, (*shape[:-1], NUM_INPUTS),
                            minval=-init_scale_ff, maxval=init_scale_ff)
    theta0 = jnp.concatenate([gains, ff], axis=-1)
    if theta0_mean is not None:
        # Restart 0 is theta0_mean EXACTLY, not a perturbation of it. Without
        # this the warm start is strictly worse than the point it starts
        # from: every restart carries an N(0, init_std_gain) gain and a
        # full-range feedforward perturbation, so the seed's own loss is
        # never actually evaluated and GD can only be compared against
        # perturbed neighbours. Measured cost of getting this wrong: seeding
        # from the open-loop optimum (loss 2.9e-7) returned 1.0e-5.
        theta0 = theta0.at[0].set(jnp.zeros_like(theta0[0]))
        theta0 = theta0 + theta0_mean[None, ...]
    return _project_theta(theta0)


def _batched_value_and_grad(loss_fn, num_restarts: int, restart_chunk: Optional[int]):
    """vmap the fused value-and-grad over restarts, optionally in chunks.

    Restarts are the only fully-parallel axis, so they are also the only
    axis whose peak device memory is linear -- `restart_chunk` scans over
    chunks of that size instead (via lax.map's batch_size), capping live
    VRAM at one chunk's working set. The traced body is the same either
    way, so compile time and results are unchanged.
    """
    value_and_grad = jax.value_and_grad(loss_fn)
    if restart_chunk is None or restart_chunk >= num_restarts:
        return jax.vmap(value_and_grad)
    if num_restarts % restart_chunk:
        raise ValueError(
            f"restart_chunk ({restart_chunk}) must divide num_restarts "
            f"({num_restarts}) exactly -- a remainder makes lax.map compile a "
            "second vmapped body for the leftover restarts, roughly doubling "
            "the (dominant) compile cost.")
    return lambda theta_batch: jax.lax.map(value_and_grad, theta_batch,
                                           batch_size=restart_chunk)


def optimize_output_feedback_gpu(x0_ivl: irx.Interval, cl_scenarios: List[Scenario],
                                 dt: float, num_steps: int,
                                 num_restarts: int = 32, learning_rate: float = 0.05,
                                 num_iters: int = 60, seed: int = 42,
                                 init_std_gain: float = 0.3, init_scale_ff: float = 0.05,
                                 shared_theta: bool = False,
                                 theta0_mean: Optional[jnp.ndarray] = None,
                                 restart_chunk: Optional[int] = None,
                                 optimizer: str = 'adam',
                                 gain_mask=1.0):
    """Multi-start projected GD on the refined output-feedback loss.

    Mirrors `admire_separating_input.optimize_refined_gpu` -- fused
    value_and_grad (one forward trace, not two), `_scan_loop` for the outer
    GD loop (graph size O(1) in num_iters), NaN-safe restart selection --
    with the control sequence replaced by controller parameters and
    `_project_u` by `_project_theta`, plus the `restart_chunk` VRAM knob.

    `optimizer`: 'adam' (default) or 'gd'. Plain GD does not work on this
    loss and 'gd' is kept only to reproduce earlier results -- the loss is a
    min-over-steps of overlap VOLUMES (~1e-5 here) with non-overlapping
    pairs masked to zero, so raw gradients are ~1e-4 and a `lr * g` step
    moves theta by ~1e-5 per iteration. Measured: 60 GD iterations at
    lr=0.05 moved the winning restart's gains by 1.2e-4 against a norm of
    4.93, i.e. the "optimum" was just the best of the random draws. Adam
    divides out that scale; combined with `_STEP_SCALE` a step moves each
    coordinate `learning_rate` of its own range.

    `dt` and `gain_mask` may be TRACED scalars -- neither changes the graph,
    so one compile can sweep integration step sizes and can run both the
    output-feedback and the gain-masked (open-loop) arm. `num_steps` is a
    scan length and must stay static.

    Returns (best_theta, best_loss, theta_final, losses) in DECISION-variable
    shape: (40,) per restart when shared_theta, else (num_steps, 40), with
    the gain mask already applied. Pass the result through `expand_theta`
    before handing it to the loss or to any propagation helper.
    """
    theta0 = _init_theta(jax.random.PRNGKey(seed), num_restarts, num_steps,
                         shared_theta, init_std_gain, init_scale_ff, theta0_mean)

    def loss_fn(theta):
        return refined_separation_loss(
            expand_theta(_apply_gain_mask(theta, gain_mask), num_steps),
            x0_ivl, cl_scenarios, dt, num_steps)

    batched = _batched_value_and_grad(loss_fn, num_restarts, restart_chunk)

    if optimizer == 'gd':
        def body(carry):
            theta_batch, _m, _v, _t = carry
            _, g = batched(theta_batch)
            return (_project_theta(theta_batch - learning_rate * g), _m, _v, _t)
    elif optimizer == 'adam':
        b1, b2, eps = 0.9, 0.999, 1e-8

        def body(carry):
            theta_batch, m, v, t = carry
            _, g = batched(theta_batch)
            t = t + 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            step = (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
            return (_project_theta(theta_batch - learning_rate * _STEP_SCALE * step),
                    m, v, t)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'gd', got {optimizer!r}")

    zeros = jnp.zeros_like(theta0)
    theta_final, _, _, _ = _scan_loop(
        body, (theta0, zeros, zeros, jnp.zeros((), jnp.int32)), num_iters)
    losses, _ = batched(theta_final)
    theta_final = _apply_gain_mask(theta_final, gain_mask)

    # NaN-safe: ADMIRE's divisions (by Vt, cos beta, cos theta) turn a
    # blown-up box into NaN rather than a loose bound, and plain argmin does
    # not reliably skip NaN entries.
    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx = jnp.argmin(losses_valid)
    return theta_final[best_idx], losses[best_idx], theta_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Evaluation  (raw, un-refined -- the honest diagnosability check)
# ══════════════════════════════════════════════════════════════════════════════

def raw_output_histories(x0_ivl: irx.Interval, theta_seq: jnp.ndarray,
                         cl_scenarios: List[Scenario], dt: float) -> irx.Interval:
    """Per-scenario closed-loop state history with NO cross-scenario refinement.

    Shapes (n_scenarios, num_steps, 9). A low REFINED loss is a certificate
    under the refinement's own semantics -- boxes built by intersecting
    scenarios' PREDICTED outputs -- which is not the same claim as the
    independently-propagated boxes being disjoint. Every demo below reports
    this raw check too, per multistep_refined_demo.py's caveat.
    """
    emb_sys = cl_scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in cl_scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in cl_scenarios]),
    )
    return jax.vmap(
        lambda p: _propagate_history(x0_ivl, theta_seq, emb_sys, p, dt, 1)
    )(p_batch)


def raw_pair_separated(hist: irx.Interval) -> np.ndarray:
    """(n_pairs, num_steps) bool: is this pair's raw output box disjoint here?

    A pair is separated iff its boxes miss on at least one output axis. The
    one place this predicate is computed -- `raw_disjoint_segments` and
    `raw_disjoint_pair_counts` both read it, so an all-or-nothing verdict and
    a partial-progress count can never disagree about what "separated" means.
    """
    lo = np.asarray(hist.lower)[:, :, 3:6]      # (n, num_steps, 3)
    hi = np.asarray(hist.upper)[:, :, 3:6]
    ii, jj = np.triu_indices(lo.shape[0], k=1)
    return np.any((lo[ii] > hi[jj]) | (lo[jj] > hi[ii]), axis=-1)


def raw_disjoint_segments(hist: irx.Interval) -> List[int]:
    """Steps at which ALL scenario pairs' raw output boxes are disjoint."""
    sep = raw_pair_separated(hist)
    return [k for k in range(sep.shape[1]) if bool(np.all(sep[:, k]))]


def raw_disjoint_pair_counts(hist: irx.Interval) -> np.ndarray:
    """(num_steps,) count of separated pairs, out of n*(n-1)/2.

    The partial-credit metric. Once several configurations drive the REFINED
    loss to exactly 0.0 it stops discriminating between them (and a refined
    zero is not a raw certificate anyway), while all-or-nothing disjointness
    is too coarse to show which configuration is closer. This sits between
    the two: 54/55 and 3/55 are both "not separated" but are not remotely
    the same result.
    """
    return raw_pair_separated(hist).sum(axis=0)


def output_box_growth(x0_ivl: irx.Interval, hist: irx.Interval) -> float:
    """Largest output-box width over the horizon, relative to the initial one.

    The blow-up detector: interval conservatism under feedback shows up here
    long before it shows up in the loss (a wide-but-disjoint box still
    scores 0). See the module docstring.
    """
    w0 = float(jnp.max(x0_ivl.upper[3:6] - x0_ivl.lower[3:6]))
    w = float(np.max(np.asarray(hist.upper)[:, :, 3:6] - np.asarray(hist.lower)[:, :, 3:6]))
    return w / w0 if w0 > 0 else float('inf')


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Demo
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Solve an ADMIRE output-feedback separating controller "
                    "(intersection refinement).")
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--steps', type=int, default=10, help="horizon in steps")
    p.add_argument('--restarts', type=int, default=32)
    p.add_argument('--iters', type=int, default=60)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--x0-halfwidth', type=float, default=5e-3)
    p.add_argument('--init-std-gain', type=float, default=0.3)
    p.add_argument('--restart-chunk', type=int, default=None,
                   help="evaluate restarts this many at a time (must divide "
                        "--restarts); caps VRAM at one chunk's working set")
    p.add_argument('--optimizer', choices=['adam', 'gd'], default='adam',
                   help="'gd' only reproduces earlier results; it does not "
                        "descend this loss (see the optimizer docstring)")
    p.add_argument('--shared-theta', action=argparse.BooleanOptionalAction,
                   default=True,
                   help="single-stage: ONE time-invariant (K, r) for the whole "
                        "horizon (default). --no-shared-theta solves a per-step "
                        "schedule instead, which measured strictly worse here "
                        "(see the docstring's results table)")
    p.add_argument('--warm-start', type=str, default=None,
                   help="npz with a `u_seq` (num_steps, 10) open-loop "
                        "solution to centre the restarts on (K=0)")
    p.add_argument('--out', type=str, default='output_feedback_result.npz')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    print(f"Devices: {jax.devices()}")

    cl_scenarios = create_cl_scenarios(fault_effectiveness=0.0)
    n = len(cl_scenarios)
    print(f"{n} closed-loop fault scenarios (Nominal + 10 total-actuator-loss), "
          f"{n*(n-1)//2} pairs")

    x0_nom = jnp.zeros(NUM_STATES).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(NUM_STATES) * args.x0_halfwidth)

    theta0_mean = None
    if args.warm_start:
        u_seq = jnp.asarray(np.load(args.warm_start)['u_seq'])
        if u_seq.shape != (args.steps, NUM_INPUTS):
            raise ValueError(f"{args.warm_start} holds u_seq {u_seq.shape}, "
                             f"expected ({args.steps}, {NUM_INPUTS})")
        theta0_mean = theta0_from_open_loop(u_seq)
        if args.shared_theta:
            raise ValueError("--warm-start is a per-step (num_steps, 10) "
                             "sequence and cannot centre a single-stage "
                             "(time-invariant) search, which is now the "
                             "default -- pass --no-shared-theta alongside it")
        print(f"Warm-starting from {args.warm_start} (K=0, r=u_seq)")

    n_theta = THETA_LEN if args.shared_theta else args.steps * THETA_LEN
    print(f"Horizon: {args.steps} x {args.dt}s = {args.steps*args.dt:.2f}s   "
          f"restarts={args.restarts}  iters={args.iters}  lr={args.lr}  "
          f"chunk={args.restart_chunk}")
    print(f"Decision variables: {n_theta} per restart "
          f"({'time-invariant' if args.shared_theta else 'per-step'} K, r)")

    def run(seed):
        return optimize_output_feedback_gpu(
            x0_ivl=x0_ivl, cl_scenarios=cl_scenarios, dt=args.dt,
            num_steps=args.steps, num_restarts=args.restarts,
            learning_rate=args.lr, num_iters=args.iters, seed=seed,
            init_std_gain=args.init_std_gain, shared_theta=args.shared_theta,
            theta0_mean=theta0_mean, restart_chunk=args.restart_chunk,
            optimizer=args.optimizer,
        )

    jitted = jax.jit(run)
    print("\nCompiling (reverse-mode AD through ADMIRE's division-heavy "
          "dynamics dominates; expect minutes) ...")
    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(args.seed))
    compile_t = time.perf_counter() - t0
    print(f"Compile time: {compile_t:.1f}s")

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(args.seed))
    run_t = time.perf_counter() - t0
    print(f"Post-compile run time: {run_t*1e3:.2f}ms")
    print(f"Memory: {_memory_snapshot()}")

    best_theta, best_loss, theta_final, losses = out
    n_nan = int(jnp.sum(jnp.isnan(losses)))
    print(f"\nBest refined loss: {float(best_loss):.6e}  "
          f"({n_nan}/{args.restarts} restarts NaN)")
    if n_nan < args.restarts:
        print(f"Loss distribution: min={float(jnp.nanmin(losses)):.6e}  "
              f"mean={float(jnp.nanmean(losses)):.6e}  "
              f"max={float(jnp.nanmax(losses)):.6e}")

    theta_seq = expand_theta(best_theta, args.steps)
    K0, r0 = theta_to_K_r(theta_seq[0])
    print(f"\nStep-0 gain |K|max={float(jnp.max(jnp.abs(K0))):.4f} "
          f"(bound {_K_MAX})   feedforward |r|max={float(jnp.max(jnp.abs(r0))):.4f}")
    n_sat = int(jnp.sum(jnp.abs(theta_seq[:, :NUM_INPUTS*NUM_OUTPUTS]) >= _K_MAX - 1e-6))
    print(f"Gain entries saturated at the bound: {n_sat}/"
          f"{theta_seq.shape[0]*NUM_INPUTS*NUM_OUTPUTS}")

    hist = raw_output_histories(x0_ivl, theta_seq, cl_scenarios, args.dt)
    disjoint = raw_disjoint_segments(hist)
    growth = output_box_growth(x0_ivl, hist)
    print(f"\nRAW (independent-box) check -- steps where all {n*(n-1)//2} pairs "
          f"are disjoint: {disjoint}")
    print(f"Output-box growth over the horizon: {growth:.1f}x")
    if float(best_loss) < 1e-6 and not disjoint:
        print("\nWARNING: refined loss ~0 but NO step shows raw independent-box "
              "disjointness -- do not report this as a separation certificate.")
    if growth > 10.0:
        print("WARNING: boxes grew >10x -- the interval conservatism of the "
              "feedback term is dominating (see module docstring); consider a "
              "smaller _K_MAX or a shorter horizon.")

    out_path = _HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    np.savez(out_path,
             theta=np.asarray(best_theta), theta_seq=np.asarray(theta_seq),
             loss=float(best_loss), losses=np.asarray(losses),
             dt=args.dt, num_steps=args.steps, shared_theta=args.shared_theta,
             k_max=_K_MAX, x0_ivl_lower=np.asarray(x0_ivl.lower),
             x0_ivl_upper=np.asarray(x0_ivl.upper),
             compile_t=compile_t, run_t=run_t,
             disjoint_segments=np.array(disjoint, dtype=np.int32),
             box_growth=growth)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
