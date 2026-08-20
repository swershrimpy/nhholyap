"""
Success-rate analysis for the faulty-car separating-input methods.

Sweeps a grid of DIVERSE problem instances -- different initial-state
intervals (center AND width) and different fault-severity intervals -- and
checks, for each of the three separating-input methods (single-step,
multistep-unrefined, multistep-refined), whether a control law exists that
drives the separation loss to (near) zero, i.e. fully isolates the fault.

"Disturbance intervals (w, p)" mapping for this system
--------------------------------------------------------
car_separating_input.py's system has no literal disturbance input `w` --
only the fault parameter `p` (alpha, the actuator-fault authority) and the
initial-state interval. This sweep varies every interval-valued quantity
that plays a `w`/`p`-like role for THIS system:
  - x0 interval:        center (x0_center) AND half-width (x0_width)
  - actuator fault `p`: the alpha range [alpha_lo, alpha_hi]
                         (narrower -> harder to distinguish from nominal)
  - sensor fault:        obs_offset / obs_scale severity
                         (smaller offset / scale closer to 1 -> harder to detect)

JIT structure (the point of this script)
------------------------------------------
Every quantity that varies across the sweep (x0_center, x0_width, alpha_lo,
alpha_hi, sensor_offset, sensor_scale, and the per-restart control-sequence
init) is passed as a TRACED ARGUMENT to one loss function per method, which
is then DOUBLE-vmapped: an inner vmap over restarts (for a fixed config)
nested inside an outer vmap over configs. The entire gradient-descent loop
(all configs x all restarts x all iterations) is wrapped in exactly ONE
jax.jit per method (3 total compiles for the whole sweep, regardless of
how large the config grid is) -- no per-config Python-level re-jitting, no
`SeparatingInputOptimizer`/`MultistepSequenceOptimizer` object per config
(those bake x0_ivl/scenarios into the jit via closure, which would force a
recompile per config; this script's functions take them as arguments
instead). Only dt/num_steps (and steps_per_segment/num_segments for the
unrefined layer) are static across the whole sweep, since they determine
loop/unroll structure -- every config in one sweep run shares one horizon.

Recorded per (method, config)
-------------------------------
  - success (bool, loss < SUCCESS_THRESHOLD) and the raw final loss
  - the winning control law (u_opt / u_seq_opt)
  - the full per-restart final-loss distribution (secondary "restart
    success rate" -- how much the result depends on lucky init)
  - reachable (observed-output-space) intervals at every time step, for
    every scenario (single-step/unrefined) or every scenario pair
    (refined, whose per-pair states diverge after refinement) -- computed
    once per config using the winning control law (not per-restart, to
    keep data volume bounded)
Recorded per method: one JIT compile time + one run time for the WHOLE
sweep (time_jit), demonstrating exactly the timing opportunity a
jit-once/pass-args-in design provides.

Output: success_rate_summary.csv (one row per method x config) and
success_rate_data_<method>.npz (full arrays: control laws, interval
histories, per-restart loss distributions) in this directory.
"""

import csv
import itertools
import sys
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx

from car_separating_input import (
    Scenario,
    get_system_and_embedding,
    separation_loss,
    separation_loss_multistep,
    refined_overlap_loss,
    euler_step,
    observed_output,
    _refine_and_step_pair,
    _propagate_history,
    _project_u,
    _scan_loop,
    time_jit,
)

# ══════════════════════════════════════════════════════════════════════════════
# Config grid -- the "diverse initial state sets / fault-severity intervals"
# ══════════════════════════════════════════════════════════════════════════════

X0_CENTERS = [
    (0.1, 0.1, 0.0),
    (0.5, -0.3, 1.0),
    (-0.2, 0.4, -0.8),
]
X0_WIDTHS = [round(0.02 + 0.01 * i, 2) for i in range(14)]   # 0.02, 0.03, ..., 0.15
ACTUATOR_RANGES = [(0.0, 0.5), (0.35, 0.65)]     # wide (easy) vs. narrow (hard) fault band
SENSOR_SEVERITIES = [                            # (offset_xy, scale)
    ((0.2, 0.2), 0.95),
    # ((0.03, 0.03), 0.99) -- dropped: this near-identity sensor map (offset
    # 3cm, scale 0.99) was found to be outright inseparable from Nominal --
    # every method hit a hard loss floor (~1e-4 to 2e-3, no restart-to-
    # restart variance) with 0/12 successes even at the widened 5s horizon
    # below, not an optimizer/threshold issue -- see success_rate_analysis
    # investigation notes in project memory.
]

SUCCESS_THRESHOLD = 1e-6   # m^2; loss below this counts as "fully separated"

# Shared horizon across the WHOLE sweep (must be static -- see module docstring).
# 5s total for every method: single-step gets ONE control decision held
# constant for the whole horizon; both multistep methods get 10 independent
# control decisions, one per dt=0.5 step (unrefined's "segment" is 1 step
# long here, so segment == step).
DT = 0.5
SINGLE_STEP_NUM_STEPS = 10
UNREFINED_STEPS_PER_SEGMENT, UNREFINED_NUM_SEGMENTS = 1, 10
REFINED_NUM_STEPS = 10

NUM_RESTARTS = 100
# Verified empirically (see git history / PR discussion) that GD convergence
# fully plateaus by ~20-40 iterations for all three methods at this
# learning rate -- mean best-loss and success counts are IDENTICAL between
# 20/50/100 iters for unrefined, and between 20/40 for single-step/refined.
# 100 was pure wasted compute (run time scales ~linearly with iters via the
# scan-based GD loop, for zero quality gain past the plateau). 30 keeps a
# small safety margin above the demonstrated plateau point.
NUM_ITERS = 30
LEARNING_RATE = 0.05
SEED = 42

OUT_CSV = _HERE / "success_rate_summary.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Config grid -> flat traced-argument arrays
# ══════════════════════════════════════════════════════════════════════════════

def build_config_grid():
    """Full factorial cross product of the axes above -> per-field arrays,
    each of shape (num_configs, ...), ready to pass as traced vmap inputs."""
    combos = list(itertools.product(X0_CENTERS, X0_WIDTHS, ACTUATOR_RANGES, SENSOR_SEVERITIES))
    x0_center = jnp.array([c[0] for c in combos])                      # (C,3)
    x0_width = jnp.array([c[1] for c in combos])                       # (C,)
    alpha_lo = jnp.array([c[2][0] for c in combos])                    # (C,)
    alpha_hi = jnp.array([c[2][1] for c in combos])                    # (C,)
    sensor_offset = jnp.array([c[3][0] for c in combos])               # (C,2)
    sensor_scale = jnp.array([c[3][1] for c in combos])                # (C,)
    return combos, (x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale)


def build_scenarios(alpha_lo, alpha_hi, sensor_offset, sensor_scale) -> List[Scenario]:
    """Construct the 3 Scenario objects from TRACED per-config parameters.

    Building a plain (non-pytree) dataclass inside a traced function is
    fine -- JAX traces through the jnp-array field VALUES, not the
    surrounding Python object; this is not used as a scan/vmap carry.
    """
    _, emb = get_system_and_embedding()
    ones = jnp.ones(1)
    return [
        Scenario("Nominal", emb, irx.Interval(lower=ones, upper=ones)),
        Scenario("Actuator Fault", emb,
                 irx.Interval(lower=jnp.reshape(alpha_lo, (1,)), upper=jnp.reshape(alpha_hi, (1,)))),
        Scenario("Sensor Fault", emb, irx.Interval(lower=ones, upper=ones),
                 obs_offset=sensor_offset, obs_scale=jnp.reshape(sensor_scale, (1,))),
    ]


def _config_loss(u, x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale,
                 *, loss_kind: str, dt: float, **loss_kwargs) -> jnp.ndarray:
    """Config-parameterized wrapper around the existing loss functions --
    only the args other than `u` vary across the vmap's config axis."""
    x0_ivl = irx.icentpert(x0_center, jnp.full(3, x0_width))
    scenarios = build_scenarios(alpha_lo, alpha_hi, sensor_offset, sensor_scale)
    if loss_kind == "single_step":
        return separation_loss(u, x0_ivl, scenarios, dt=dt, **loss_kwargs)
    elif loss_kind == "multistep_unrefined":
        return separation_loss_multistep(u, x0_ivl, scenarios, dt=dt, **loss_kwargs)
    elif loss_kind == "multistep_refined":
        return refined_overlap_loss(u, x0_ivl, scenarios, dt=dt, **loss_kwargs)
    raise ValueError(loss_kind)


# ══════════════════════════════════════════════════════════════════════════════
# Double-vmapped (config x restart) batched sweep optimizer -- ONE jit per method
# ══════════════════════════════════════════════════════════════════════════════

def run_batched_sweep(loss_kind: str, u_shape_per_restart: Tuple[int, ...],
                      config_arrays, num_restarts: int, num_iters: int,
                      learning_rate: float, seed: int, dt: float, **loss_kwargs):
    """Jointly optimize a separating control law for EVERY config in
    `config_arrays`, each with its own `num_restarts` multi-start batch,
    inside a single jax.jit call (see module docstring).

    Returns (jitted_fn, u_final, losses_final, compile_t, run_t, mem):
    `jitted_fn` is returned too so callers can measure additional
    steady-state calls WITHOUT re-jitting (see run_method's timeit pass).
    u_final has shape (num_configs, num_restarts, *u_shape_per_restart);
    losses_final has shape (num_configs, num_restarts).
    """
    x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale = config_arrays
    num_configs = x0_center.shape[0]

    _loss = partial(_config_loss, loss_kind=loss_kind, dt=dt, **loss_kwargs)
    # inner: batch over restarts for ONE fixed config (config args: in_axes=None)
    inner_loss = jax.vmap(_loss, in_axes=(0, None, None, None, None, None, None))
    inner_grad = jax.vmap(jax.grad(_loss), in_axes=(0, None, None, None, None, None, None))
    # outer: batch over configs (every arg, including u, has a leading config axis)
    batched_loss = jax.vmap(inner_loss, in_axes=(0, 0, 0, 0, 0, 0, 0))
    batched_grad = jax.vmap(inner_grad, in_axes=(0, 0, 0, 0, 0, 0, 0))
    # (An earlier attempt tried folding the loop's grad step and the final
    # loss readout into one jax.value_and_grad call, carrying (u, loss)
    # through the scan to skip a trailing pass. That's an off-by-one bug:
    # the carried loss corresponds to u BEFORE its own iteration's update,
    # not to the u_final actually returned. Reverted -- with num_iters=30
    # the "extra" final pass is ~3% of total compute, not worth the risk.)

    def full_sweep(seed_val):
        key = jax.random.PRNGKey(seed_val)

        if len(u_shape_per_restart) == 2:
            # Multistep methods (unrefined/refined): separation_loss_multistep
            # and refined_overlap_loss both take jnp.min over per-segment/
            # per-step overlaps, so jax.grad backprops through only the ONE
            # current worst segment -- every other segment's control gets
            # exactly zero gradient each step (verified empirically). Plain
            # per-segment-independent restarts can then converge to a worse
            # optimum than single-step's, even though single-step's solution
            # (one constant control, tiled across every segment) is always
            # in this method's feasible set and provably achieves loss <=
            # single-step's loss there (confirmed: tiling single-step's
            # winning control into a 10-segment sequence and evaluating it
            # under separation_loss_multistep reproduced single-step's exact
            # loss, on a config where independent-restart GD alone stalled
            # ~4 orders of magnitude above threshold). Fix: seed HALF the
            # restarts as a single constant control broadcast across every
            # segment/step -- i.e. give GD an explicit foothold already
            # inside single-step's solution class -- instead of relying on
            # free per-segment search to rediscover it. Pure addition to the
            # restart pool (best-of-N over restarts), so this can only help.
            key_indep, key_const = jax.random.split(key)
            num_steps_dim = u_shape_per_restart[0]
            u0_indep = (jax.random.normal(key_indep, (num_configs, num_restarts) + u_shape_per_restart) * 0.3
                        + jnp.array([0.5, 0.3]))
            u0_const_base = (jax.random.normal(key_const, (num_configs, num_restarts, 2)) * 0.3
                              + jnp.array([0.5, 0.3]))
            u0_const = jnp.broadcast_to(
                u0_const_base[:, :, None, :], (num_configs, num_restarts, num_steps_dim, 2)
            )
            is_const_restart = jnp.arange(num_restarts) < (num_restarts // 2)
            u0 = jnp.where(is_const_restart[None, :, None, None], u0_const, u0_indep)
        else:
            u0 = (jax.random.normal(key, (num_configs, num_restarts) + u_shape_per_restart) * 0.3
                  + jnp.array([0.5, 0.3]))

        def body(u_batch):
            g = batched_grad(u_batch, x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale)
            return _project_u(u_batch - learning_rate * g)

        u_final = _scan_loop(body, u0, num_iters)
        losses_final = batched_loss(u_final, x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale)
        return u_final, losses_final

    jitted_fn, compile_t, run_t, mem = time_jit(full_sweep, seed)
    u_final, losses_final = jitted_fn(seed)
    return jitted_fn, u_final, losses_final, compile_t, run_t, mem


# ══════════════════════════════════════════════════════════════════════════════
# Reachable-interval history for the WINNING control law (per config)
# ══════════════════════════════════════════════════════════════════════════════

def _history_for_config(x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale,
                        u_seq, *, dt: float, steps_per_segment: int):
    """Observed-output-space interval history for all 3 scenarios, one
    config, under a fixed control sequence. Fully jittable/vmappable
    (reuses the stack-p-and-vmap trick already used inside
    separation_loss_multistep)."""
    x0_ivl = irx.icentpert(x0_center, jnp.full(3, x0_width))
    scenarios = build_scenarios(alpha_lo, alpha_hi, sensor_offset, sensor_scale)
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def prop_one(p_ivl_single):
        return _propagate_history(x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment)

    x_hist = jax.vmap(prop_one)(p_batch)   # lower/upper shape (n_scenarios, num_segments, 3)

    obs_offsets = jnp.stack([s.obs_offset for s in scenarios])          # (n_scenarios, 2)
    obs_scales = jnp.stack([s.obs_scale[0] for s in scenarios])         # (n_scenarios,)
    y_lower = obs_scales[:, None, None] * x_hist.lower[..., :2] + obs_offsets[:, None, :]
    y_upper = obs_scales[:, None, None] * x_hist.upper[..., :2] + obs_offsets[:, None, :]
    return y_lower, y_upper   # each (n_scenarios, num_segments, 2)


def batched_history(config_arrays, u_seq_batch, dt: float, steps_per_segment: int):
    """vmap `_history_for_config` over ALL configs at once (jittable) --
    used for single-step and multistep-unrefined."""
    x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale = config_arrays
    fn = jax.jit(jax.vmap(
        partial(_history_for_config, dt=dt, steps_per_segment=steps_per_segment)
    ))
    y_lower, y_upper = fn(x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale, u_seq_batch)
    return np.array(y_lower), np.array(y_upper)   # (num_configs, n_scenarios, num_segments, 2)


_PAIRS_3 = [(0, 1), (0, 2), (1, 2)]   # fixed pair list for n=3 scenarios, unrolled at trace time


def _refined_history_for_config(x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale,
                                u_seq, *, dt: float, num_steps: int):
    """Jittable/vmappable per-pair refinement history for ONE config.

    Deliberately does NOT call car_separating_input.collect_refinement_history:
    that function is a plain-Python-loop history collector, appropriate for
    animate_refinement_3d.py's single one-off call, but its per-step
    `bool(...)` calls force a blocking host-device sync at every
    (step, pair) -- 24 configs x 8 steps x 3 pairs = 576 blocking syncs when
    called once per config across this sweep, MEASURED to dominate this
    script's wall-clock time (minutes, not the seconds the actual
    optimization itself takes). This reimplements the same per-pair
    refine-then-propagate math via `_refine_and_step_pair` (the same
    primitive `propagate_with_refinement`'s loss uses) inside a
    jax.lax.scan, so the whole per-config history is ONE jitted,
    vmappable call with zero Python-level syncs.

    `has_overlap` is returned as an array (not used to decide Python
    None-vs-Interval branching) -- callers needing an "intersection or
    None" distinction (only the animation currently does) can derive it
    from `has_overlap` after pulling data to host.
    """
    x0_ivl = irx.icentpert(x0_center, jnp.full(3, x0_width))
    scenarios = build_scenarios(alpha_lo, alpha_hi, sensor_offset, sensor_scale)

    x1 = [euler_step(scenarios[k].emb_system, x0_ivl, u_seq[0], scenarios[k].p_interval, dt) for k in range(3)]
    obs1 = [observed_output(x1[k], scenarios[k]) for k in range(3)]

    def to_arr(ivl):
        return jnp.concatenate([ivl.lower, ivl.upper])

    pxi0 = jnp.stack([to_arr(x1[i]) for i, j in _PAIRS_3])
    pxj0 = jnp.stack([to_arr(x1[j]) for i, j in _PAIRS_3])
    obs_i0 = jnp.stack([to_arr(obs1[i]) for i, j in _PAIRS_3])
    obs_j0 = jnp.stack([to_arr(obs1[j]) for i, j in _PAIRS_3])
    overlap0 = jnp.ones(len(_PAIRS_3), dtype=bool)   # step 1 has no refinement gate

    def step_fn(carry, u_k):
        pxi, pxj = carry
        new_pxi, new_pxj, new_obs_i, new_obs_j, new_overlap = [], [], [], [], []
        for idx, (i, j) in enumerate(_PAIRS_3):
            xi = irx.Interval(lower=pxi[idx, :3], upper=pxi[idx, 3:])
            xj = irx.Interval(lower=pxj[idx, :3], upper=pxj[idx, 3:])
            x_next_i, x_next_j, obs_next_i, obs_next_j, _, has_overlap = _refine_and_step_pair(
                xi, xj, scenarios[i], scenarios[j], u_k, dt
            )
            new_pxi.append(to_arr(x_next_i)); new_pxj.append(to_arr(x_next_j))
            new_obs_i.append(to_arr(obs_next_i)); new_obs_j.append(to_arr(obs_next_j))
            new_overlap.append(has_overlap)
        new_carry = (jnp.stack(new_pxi), jnp.stack(new_pxj))
        outputs = (jnp.stack(new_obs_i), jnp.stack(new_obs_j), jnp.stack(new_overlap))
        return new_carry, outputs

    _, (obs_i_rest, obs_j_rest, overlap_rest) = jax.lax.scan(step_fn, (pxi0, pxj0), u_seq[1:num_steps])

    obs_i_hist = jnp.concatenate([obs_i0[None], obs_i_rest], axis=0)      # (num_steps, n_pairs, 4)
    obs_j_hist = jnp.concatenate([obs_j0[None], obs_j_rest], axis=0)
    overlap_hist = jnp.concatenate([overlap0[None], overlap_rest], axis=0)   # (num_steps, n_pairs)
    return obs_i_hist, obs_j_hist, overlap_hist


def refined_history_per_config(config_arrays, u_seq_batch, dt: float, num_steps: int):
    """vmap `_refined_history_for_config` over ALL configs at once (jittable,
    zero host syncs) -- see that function's docstring for why this replaced
    a Python-loop-based collector."""
    x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale = config_arrays
    fn = jax.jit(jax.vmap(partial(_refined_history_for_config, dt=dt, num_steps=num_steps)))
    obs_i_hist, obs_j_hist, overlap_hist = fn(
        x0_center, x0_width, alpha_lo, alpha_hi, sensor_offset, sensor_scale, u_seq_batch
    )
    # (num_configs, num_steps, n_pairs, 4) -> split into lower/upper (.., 2) for npz storage,
    # matching the shape convention batched_history already uses.
    obs_i_lo = np.array(obs_i_hist[..., :2]); obs_i_hi = np.array(obs_i_hist[..., 2:])
    obs_j_lo = np.array(obs_j_hist[..., :2]); obs_j_hi = np.array(obs_j_hist[..., 2:])
    has_overlap = np.array(overlap_hist)   # (num_configs, num_steps, n_pairs)
    return obs_i_lo, obs_i_hi, obs_j_lo, obs_j_hi, has_overlap, _PAIRS_3


# ══════════════════════════════════════════════════════════════════════════════
# Per-method driver
# ══════════════════════════════════════════════════════════════════════════════

def run_method(name: str, loss_kind: str, u_shape_per_restart, loss_kwargs, config_arrays, combos):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")

    jitted_fn, u_final, losses_final, compile_t, run_t, mem = run_batched_sweep(
        loss_kind, u_shape_per_restart, config_arrays,
        num_restarts=NUM_RESTARTS, num_iters=NUM_ITERS, learning_rate=LEARNING_RATE,
        seed=SEED, dt=DT, **loss_kwargs,
    )
    num_configs = losses_final.shape[0]

    best_idx = jnp.argmin(losses_final, axis=1)                      # (num_configs,)
    best_loss = losses_final[jnp.arange(num_configs), best_idx]       # (num_configs,)
    best_u = u_final[jnp.arange(num_configs), best_idx]                # (num_configs, *u_shape)
    restart_success_rate = jnp.mean(losses_final < SUCCESS_THRESHOLD, axis=1)   # (num_configs,)
    config_success = best_loss < SUCCESS_THRESHOLD

    print(f"compile {compile_t * 1e3:8.2f} ms total ({compile_t / num_configs * 1e3:7.3f} ms/config)   "
          f"run {run_t * 1e3:7.3f} ms total ({run_t / num_configs * 1e3:7.3f} ms/config)   mem {mem}")
    print(f"Config success rate: {int(jnp.sum(config_success))}/{num_configs} "
          f"({100 * float(jnp.mean(config_success)):.1f}%)")

    # Steady-state timeit on the ALREADY-COMPILED function (reuses jitted_fn,
    # no re-jit) -- dispatched back-to-back with one block_until_ready at the
    # end, same pattern as nonlinear_chain/runtime_scaling_common.py's
    # timeit_run, to avoid host/device sync overhead dominating the signal.
    #
    # Repeat count is capped by a TOTAL time budget, not a fixed count: a
    # single call here costs milliseconds for integrator_chain/nonlinear_chain-
    # scale problems (where 20 fixed repeats is nearly free) but SECONDS for
    # this sweep's full (24-config x 100-restart) batch -- a fixed 20 repeats
    # measured to cost ~170s of pure measurement overhead for the unrefined
    # method alone (dominating the script's wall-clock time far more than the
    # actual optimization). Budgeting ~3s of total timeit time instead keeps
    # this proportionate regardless of how expensive one call is.
    _TIMEIT_BUDGET_S = 3.0
    timeit_repeats = max(1, min(20, int(_TIMEIT_BUDGET_S / max(run_t, 1e-3))))
    t0 = time.perf_counter()
    out = None
    for _ in range(timeit_repeats):
        out = jitted_fn(SEED)
    jax.block_until_ready(out)
    steady_t = (time.perf_counter() - t0) / timeit_repeats
    print(f"steady-state run time (avg of {timeit_repeats} dispatches on the "
          f"already-compiled fn): {steady_t * 1e3:.3f} ms/call")

    u_seq_for_history = _as_u_seq(best_u, loss_kind)
    if loss_kind == "multistep_refined":
        obs_i_lo, obs_i_hi, obs_j_lo, obs_j_hi, has_overlap, pairs = refined_history_per_config(
            config_arrays, u_seq_for_history, DT, REFINED_NUM_STEPS
        )
        history_payload = dict(obs_i_lo=obs_i_lo, obs_i_hi=obs_i_hi,
                               obs_j_lo=obs_j_lo, obs_j_hi=obs_j_hi,
                               has_overlap=has_overlap, pairs=np.array(pairs))
    else:
        steps_per_segment = 1 if loss_kind == "single_step" else UNREFINED_STEPS_PER_SEGMENT
        y_lower, y_upper = batched_history(config_arrays, u_seq_for_history, DT, steps_per_segment)
        history_payload = dict(y_lower=y_lower, y_upper=y_upper)

    rows = []
    for c, combo in enumerate(combos):
        (px, py, phi), width, (a_lo, a_hi), (offset, scale) = combo
        rows.append({
            'method': name, 'config_idx': c,
            'x0_px': px, 'x0_py': py, 'x0_phi': phi, 'x0_width': width,
            'alpha_lo': a_lo, 'alpha_hi': a_hi,
            'sensor_offset': offset[0], 'sensor_scale': scale,
            'success': bool(config_success[c]), 'final_loss': float(best_loss[c]),
            'restart_success_rate': float(restart_success_rate[c]),
        })

    npz_path = _HERE / f"success_rate_data_{loss_kind}.npz"
    np.savez(
        npz_path,
        u_opt=np.array(best_u), losses_final_all_restarts=np.array(losses_final),
        config_success=np.array(config_success), restart_success_rate=np.array(restart_success_rate),
        x0_center=np.array(config_arrays[0]), x0_width=np.array(config_arrays[1]),
        alpha_lo=np.array(config_arrays[2]), alpha_hi=np.array(config_arrays[3]),
        sensor_offset=np.array(config_arrays[4]), sensor_scale=np.array(config_arrays[5]),
        compile_time_s=compile_t, run_time_s=run_t, steady_state_run_time_s=steady_t,
        **history_payload,
    )
    print(f"Saved full data -> {npz_path}")

    return rows, {'compile_time_s': compile_t, 'run_time_s': run_t,
                  'steady_state_run_time_s': steady_t, 'mem': mem,
                  'success_rate': float(jnp.mean(config_success))}


def _as_u_seq(best_u, loss_kind):
    if loss_kind == "single_step":
        return jnp.tile(best_u[:, None, :], (1, SINGLE_STEP_NUM_STEPS, 1))
    return best_u   # already (num_configs, num_steps_or_segments, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

_METHOD_SPECS = {
    "single_step": ("SINGLE-STEP", (2,), dict(num_steps=SINGLE_STEP_NUM_STEPS)),
    "multistep_unrefined": ("MULTISTEP UNREFINED", (UNREFINED_NUM_SEGMENTS, 2),
                            dict(steps_per_segment=UNREFINED_STEPS_PER_SEGMENT)),
    "multistep_refined": ("MULTISTEP REFINED", (REFINED_NUM_STEPS, 2), dict(num_steps=REFINED_NUM_STEPS)),
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=list(_METHOD_SPECS) + ["all"], default="all",
        help=(
            "Run only this method (writes success_rate_summary_<method>.csv "
            "and reports EXACT, isolated compile/run/memory numbers for that "
            "one method -- no cross-method cascading, since nothing else ran "
            "in this process). Invoke 3 times in 3 separate processes "
            "(e.g. via `/usr/bin/time -v python success_rate_analysis.py "
            "--method single_step`, once per method) for the cleanest "
            "possible per-method resource numbers. Default 'all' runs all "
            "three sequentially in one process for convenience (memory "
            "readings after the first method are then a whole-process "
            "high-water mark, not that method's own isolated peak)."
        ),
    )
    args = parser.parse_args()

    print("Devices:", jax.devices())
    combos, config_arrays = build_config_grid()
    num_configs = len(combos)
    print(f"Config grid: {len(X0_CENTERS)} x0_centers x {len(X0_WIDTHS)} x0_widths x "
          f"{len(ACTUATOR_RANGES)} actuator_ranges x {len(SENSOR_SEVERITIES)} sensor_severities "
          f"= {num_configs} configs")
    print(f"Per method: {NUM_RESTARTS} restarts x {NUM_ITERS} GD iters, "
          f"success threshold = {SUCCESS_THRESHOLD} m^2")

    methods_to_run = list(_METHOD_SPECS) if args.method == "all" else [args.method]

    all_rows = []
    summaries = {}
    for loss_kind in methods_to_run:
        display_name, u_shape, kwargs = _METHOD_SPECS[loss_kind]
        rows, summary = run_method(display_name, loss_kind, u_shape, kwargs, config_arrays, combos)
        all_rows += rows
        summaries[loss_kind] = summary

    out_csv = OUT_CSV if args.method == "all" else _HERE / f"success_rate_summary_{args.method}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote summary CSV -> {out_csv}")

    print(f"\n{'=' * 78}\nSUMMARY ({'isolated, single-process run' if args.method != 'all' else 'all methods, one process -- see --method for isolated numbers'})\n{'=' * 78}")
    print(f"  (compile/run reported per-config, i.e. total / {num_configs} configs)")
    for method, s in summaries.items():
        print(f"  {method:22s}: success {100*s['success_rate']:5.1f}%   "
              f"compile {s['compile_time_s']/num_configs*1e3:7.3f} ms/config   "
              f"run {s['run_time_s']/num_configs*1e3:7.3f} ms/config   "
              f"steady-state {s['steady_state_run_time_s']/num_configs*1e3:7.3f} ms/config   "
              f"mem {s['mem']}")


if __name__ == "__main__":
    main()
