"""
Nth-Order Integrator Chain – Active Fault Diagnosis with immrax
=================================================================
Reachable-set-based active fault diagnosis for a canonical order-N integrator
chain, using immrax for interval arithmetic and natural-embedding
propagation.

System
------
  State   x = [x1, ..., xN]     (any order N >= 1)
  Control u = scalar, kept as shape (1,) to match the (num_segments, 1)
              control-sequence convention used elsewhere in this repo.
  Params  p = [alpha]            (actuator effectiveness; alpha=1 -> nominal)

  Dynamics:
      x1˙ = x2
      x2˙ = x3
      ...
      x_{N-1}˙ = xN
      xN˙ = alpha * u

  Written as a single slice+concatenate expression (no Python loop over N),
  so the traced jaxpr size is O(N) with no unrolled-per-state Python
  overhead, and the *same* two lines of code work for every N.

Sensor model
------------
  All N states are directly observed:  y = beta ⊙ x + xi
  beta, xi are per-state fault-interval parameters (beta assumed strictly
  positive in every scenario, so the affine map is invertible -- see
  `_invert_observation`).  Unlike go2/faulty_car, the sensor fault does NOT
  change the dynamics -- every scenario shares one `emb_system`; only the
  post-propagation output map differs.  The separation loss is computed on
  this observed output `y`, not on the raw state `x` -- this is the fix for
  a gap in faulty_car_separating_input.py, whose `obs_offset`/`obs_scale`
  fields are defined on `Scenario` but never actually applied inside
  `separation_loss`, so its Sensor-Fault scenario (identical dynamics to
  Nominal) was never actually separable in that file's optimized loss.

Scenarios
---------
  Nominal            alpha=1,             beta=1, xi=0
  Actuator Fault      alpha in [lo,hi],    beta=1, xi=0
  Sensor Fault        alpha=1,             beta in [lo,hi] (per-state), xi in [-eps,eps] (per-state)
  Simultaneous Fault  alpha in [lo,hi],    beta in [lo,hi], xi in [-eps,eps]

Three algorithmic layers (mirroring go2_separating_input_immrax.py /
faulty_car_separating_input.py):
  1. Single-step separating input (Section 3)
  2. Multistep "unrefined"/"uninformed" separating input sequence (Section 4)
  3. Multistep intersection-refinement separating input sequence (Section 5),
     generalizing faulty_car's propagate_with_refinement_pseudocode.md to a
     fully-observed N-vector state and an arbitrary scenario count.

CBF obstacle avoidance (present in go2) is intentionally NOT ported here: a
circular 2-D obstacle has no natural analogue for this system's single
spatial-like state x1.

Memory / performance notes (see project plan for the full estimate):
  - float32 throughout (JAX default); never enable x64.
  - jax.checkpoint on every propagation loop body, INCLUDING the refinement
    loop -- faulty_car's propagate_with_refinement lacks this, which is a
    real (if usually small) memory regression at higher N / num_steps for no
    runtime benefit.
  - flat-array (ivl_to_arr/arr_to_ivl) pytree carries, not nested Interval
    structs, to minimize pytree-dispatch overhead.
  - _overlap_volume replaces overlap_size_lax's lax.cond with a branchless
    clipped-product formula (same result, no per-call branch).
  - Scenario propagation is vmapped across scenarios wherever they share one
    emb_system (always true in this module), instead of looping in Python.
  - Small propagation loops (Euler steps per call, per segment) are fully
    Python-unrolled under a single jax.checkpoint via _run_unrolled_or_loop
    when their static length is <= _UNROLL_THRESHOLD -- this removes
    lax.fori_loop/scan dispatch overhead for the tiny horizons used here,
    falling back to a checkpointed jax.lax.scan for longer horizons to keep
    compile time bounded.
  - The OUTER gradient-descent iteration loop (_run_unrolled_or_loop_nocheckpoint)
    deliberately does NOT unroll, even though it's small: empirically,
    jax.lax.scan alone (vs. lax.fori_loop) already captures nearly all of
    the achievable run-time win for that loop, while unrolling it on top
    multiplies compile time by the iteration count for negligible extra
    run-time gain (measured: a 20-iteration multistep GD loop went from
    ~4s to ~130s compile for ~5% extra speed) -- unrolling the *inner*
    per-step physics is worth it, unrolling the *outer* optimization loop
    is not.
"""

import sys
import time
import resource
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Optional, Dict

# File is at examples/integrator_chain/<name>.py  ->  parents[1] = examples/
_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

# Control input box constraint: u in [-1, 1] (scalar, kept as shape (1,)).
_U_LO = jnp.array([-1.0])
_U_HI = jnp.array([1.0])

# Loops (Euler steps, GD iterations) with a static length <= this are fully
# unrolled into straight-line code instead of lax.fori_loop/scan -- see
# `_run_unrolled_or_loop` below.  Below this size, unrolling removes
# while-loop dispatch/sync overhead (the dominant run-time cost for the
# small horizons used in this module's demos); above it, we fall back to
# the loop-based form to keep compile time / trace size bounded for
# long-horizon callers.
_UNROLL_THRESHOLD = 64


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    """Project a control input (or batch) onto the feasible box.

    Works for any leading batch dimensions -- clips only the last axis.
    """
    return jnp.clip(u, _U_LO, _U_HI)


def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    """Branchless pairwise axis-aligned-box overlap volume.

    Equivalent to faulty_car.interval_functions.overlap_size_lax, but
    without its lax.cond: clipping each dimension's intersection width to
    >= 0 before taking the product already sends the result to exactly 0
    whenever any dimension fails to overlap (mirroring cond's "no_overlap"
    branch), with no branch needed. This matters because this function is
    called O(scenario_pairs) times per loss evaluation, including from
    inside jax.checkpoint-wrapped loop/scan bodies where removing a
    lax.cond noticeably shrinks the per-step compiled program.
    """
    widths = jnp.maximum(
        jnp.minimum(ivl1.upper, ivl2.upper) - jnp.maximum(ivl1.lower, ivl2.lower),
        0.0,
    )
    return jnp.prod(widths)


def _run_unrolled_or_loop(step_fn, init, n: int, unroll_threshold: int = _UNROLL_THRESHOLD):
    """Apply step_fn (carry, i) -> carry exactly n times, starting from init.

    For n <= unroll_threshold: fully unrolls into a Python loop wrapped in a
    single jax.checkpoint (one recompute-the-whole-block boundary on the
    backward pass -- O(1) extra memory, same asymptotic recompute cost as
    per-step checkpointing, but no XLA while-loop control overhead on the
    forward pass).
    For n > unroll_threshold: falls back to jax.lax.fori_loop with a
    per-iteration jax.checkpoint (bounds compiled program size / compile
    time for long horizons).
    """
    if n <= unroll_threshold:
        def body(carry):
            for i in range(n):
                carry = step_fn(carry, i)
            return carry
        return jax.checkpoint(body)(init)
    else:
        @jax.checkpoint
        def scan_body(carry, i):
            return step_fn(carry, i), None
        carry, _ = jax.lax.scan(scan_body, init, xs=jnp.arange(n))
        return carry


def _run_unrolled_or_loop_nocheckpoint(step_fn, init, n: int):
    """Advance `init` through step_fn exactly n times via jax.lax.scan.

    Used for the outer gradient-descent iteration loop, which is never
    reverse-mode-differentiated through as a whole (only each iteration's
    own gradient is), so no jax.checkpoint is needed here.

    Measured on this hardware: jax.lax.scan has substantially lower
    per-iteration dispatch/sync overhead than jax.lax.fori_loop, and
    switching to it (with NO additional unrolling) already captures nearly
    all of the available run-time win on its own -- unrolling *on top* of
    scan barely moves run time further while multiplying compile time by
    the iteration count (measured: fully unrolling a 20-iteration multistep
    GD loop pushed compile time from ~4s to ~130s for single-digit-percent
    extra run-time gain). So this deliberately always uses plain (unrolled=1)
    scan rather than a Python-unrolled loop, regardless of n.
    """
    def body(carry, i):
        return step_fn(carry, i), None
    carry, _ = jax.lax.scan(body, init, xs=jnp.arange(n))
    return carry


def _plain_unroll(step_fn, init, n: int):
    """Bare Python-unrolled loop, no checkpoint, no scan.

    For small inner loops that are already nested inside a caller-provided
    checkpoint boundary (e.g. _propagate_history's per-segment Euler loop,
    where the whole segment is jax.checkpoint-wrapped one level up) -- a
    second checkpoint or scan here would be pure overhead for no benefit.
    """
    carry = init
    for i in range(n):
        carry = step_fn(carry, i)
    return carry


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Timing / Memory Helper
# ══════════════════════════════════════════════════════════════════════════════

def _memory_snapshot() -> Dict[str, float]:
    """Best-effort memory snapshot: GPU device stats if available, else CPU RSS."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            return {k: float(v) for k, v in stats.items() if 'bytes' in k}
    except Exception:
        pass
    return {'ru_maxrss_kb': float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def time_jit(fn, *args, **kwargs):
    """JIT-compile `fn` and separately measure compile time vs. run time.

    Returns (jitted_fn, compile_time_s, run_time_s, memory_info).
    compile_time_s includes the first call (tracing + XLA lowering + the
    first execution); run_time_s is a second call on the now-compiled
    function -- this should be much smaller than compile_time_s.
    """
    jitted_fn = jax.jit(fn)

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted_fn(*args, **kwargs))
    compile_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = jitted_fn(*args, **kwargs)
    run_time_s = time.perf_counter() - t0

    del out
    return jitted_fn, compile_time_s, run_time_s, _memory_snapshot()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  System Definition
# ══════════════════════════════════════════════════════════════════════════════

class IntegratorSystem(irx.System):
    """Order-N integrator chain with actuator-fault authority alpha.

    State   x = [x1, ..., xN]
    Control u = shape (1,)
    Params  p = [alpha]     (alpha=1 -> full actuator authority)

    Dynamics:
        x_dot = concatenate([x[1:], alpha * u])
    """

    def __init__(self, N: int):
        self.evolution = 'continuous'
        self.xlen = N
        self.N = N

    def f(self, t, x, u, p):
        alpha = p[0]
        return jnp.concatenate([x[1:], alpha * u])


# Module-level cache: one irx.System + one irx.natemb(...) per distinct N,
# so repeated calls for the same N reuse the compiled embedding instead of
# rebuilding/retracing it every time.
_EMB_CACHE: Dict[int, Tuple[IntegratorSystem, object]] = {}


def get_system_and_embedding(N: int) -> Tuple[IntegratorSystem, object]:
    """Return (system, natural_embedding) for order N, cached by N."""
    if N not in _EMB_CACHE:
        sys_ = IntegratorSystem(N)
        emb = irx.natemb(sys_)
        _EMB_CACHE[N] = (sys_, emb)
    return _EMB_CACHE[N]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Scenarios & Output Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One fault scenario: shared dynamics (via alpha) + an output fault model.

    beta, xi : shape (N,) interval-valued fault parameters for the affine
               sensor map y = beta ⊙ x + xi.  beta must be strictly positive
               in every scenario (enforced in create_scenarios) so that the
               map is invertible -- required by the intersection-refinement
               algorithm in Section 5.
    """
    name: str
    emb_system: object
    p_interval: irx.Interval   # alpha interval, shape (1,)
    beta: irx.Interval         # shape (N,)
    xi: irx.Interval           # shape (N,)


def observed_output(x_ivl: irx.Interval, scenario: Scenario) -> irx.Interval:
    """y = beta ⊙ x + xi, via interval arithmetic.

    beta is assumed strictly positive everywhere (see Scenario docstring),
    so the affine map is monotone increasing in x and this reduces to the
    elementwise multiply-then-add on each bound.
    """
    beta, xi = scenario.beta, scenario.xi
    return irx.Interval(
        lower=beta.lower * x_ivl.lower + xi.lower,
        upper=beta.upper * x_ivl.upper + xi.upper,
    )


def _invert_observation(y_ivl: irx.Interval, scenario: Scenario) -> irx.Interval:
    """Outer-enclose x from y = beta*x + xi, given uncertain beta (>0) and xi.

    v = y - xi        (interval subtraction)
    x = v / beta       (interval division by a strictly-positive interval)

    Division is computed via the 4-corner min/max, which is correct
    regardless of the sign of v (unlike a naive elementwise reuse of the
    multiplication formula, which is only correct when v's sign is known
    ahead of time).
    """
    beta, xi = scenario.beta, scenario.xi
    v_lower = y_ivl.lower - xi.upper
    v_upper = y_ivl.upper - xi.lower
    corners = jnp.stack([
        v_lower / beta.lower, v_lower / beta.upper,
        v_upper / beta.lower, v_upper / beta.upper,
    ])
    return irx.Interval(lower=jnp.min(corners, axis=0), upper=jnp.max(corners, axis=0))


def create_scenarios(
    N: int,
    actuator_alpha_lo: float = 0.5,
    actuator_alpha_hi: float = 0.9,
    sensor_beta_lo: float = 0.8,
    sensor_beta_hi: float = 1.2,
    sensor_xi_bound: float = 0.1,
) -> List[Scenario]:
    """Return the four fault scenarios used throughout this module.

    Parameters
    ----------
    N : integrator order (state dimension)
    actuator_alpha_lo / hi : range of the actuator-fault authority alpha
    sensor_beta_lo / hi    : range of the per-state sensor gain fault beta
                             (must stay strictly positive)
    sensor_xi_bound        : half-width of the per-state sensor bias/noise
                             fault interval xi in [-sensor_xi_bound, +sensor_xi_bound]
    """
    _, emb = get_system_and_embedding(N)

    ones_N, zeros_N = jnp.ones(N), jnp.zeros(N)

    def point_ivl(v):
        return irx.Interval(lower=v, upper=v)

    alpha_nominal = irx.icentpert(jnp.array([1.0]), jnp.zeros(1))
    alpha_fault = irx.Interval(
        lower=jnp.array([actuator_alpha_lo]), upper=jnp.array([actuator_alpha_hi])
    )

    beta_nominal, xi_nominal = point_ivl(ones_N), point_ivl(zeros_N)
    beta_fault = irx.Interval(
        lower=jnp.full(N, sensor_beta_lo), upper=jnp.full(N, sensor_beta_hi)
    )
    xi_fault = irx.Interval(
        lower=jnp.full(N, -sensor_xi_bound), upper=jnp.full(N, sensor_xi_bound)
    )

    scenarios = [
        Scenario("Nominal", emb, alpha_nominal, beta_nominal, xi_nominal),
        Scenario("Actuator Fault", emb, alpha_fault, beta_nominal, xi_nominal),
        Scenario("Sensor Fault", emb, alpha_nominal, beta_fault, xi_fault),
        Scenario("Simultaneous Fault", emb, alpha_fault, beta_fault, xi_fault),
    ]

    for s in scenarios:
        if not bool(jnp.all(s.beta.lower > 0)):
            raise ValueError(
                f"Scenario {s.name!r}: beta interval must be strictly positive "
                f"(got lower={s.beta.lower}) -- refinement inversion is unsound otherwise."
            )

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Single-Step Path
# ══════════════════════════════════════════════════════════════════════════════

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    """One forward-Euler interval step via the natural embedding.

    Note: t must be a JAX array (not a Python scalar) so that
    eqx.filter_make_jaxpr traces it as an abstract input and the jaxpr
    invar count matches the natif_jaxpr arg count.
    """
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


def _propagate_by_params(x0_ivl: irx.Interval, u: jnp.ndarray, emb_sys,
                         p_ivl: irx.Interval, dt: float, num_steps: int) -> irx.Interval:
    """Propagate *num_steps* Euler steps given raw (emb_sys, p_ivl) rather than
    a Scenario -- the vmappable primitive behind propagate_scenario and the
    scenario-batched loss functions (only p_ivl varies across a vmap axis;
    emb_sys is shared/static)."""
    def step(x_carry, _i):
        return euler_step(emb_sys, x_carry, u, p_ivl, dt)
    return _run_unrolled_or_loop(step, x0_ivl, num_steps)


def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    """Propagate the initial interval *num_steps* Euler steps under constant u."""
    return _propagate_by_params(x0_ivl, u, scenario.emb_system, scenario.p_interval, dt, num_steps)


def _propagate_all_scenarios(x0_ivl: irx.Interval, u: jnp.ndarray,
                             scenarios: List[Scenario], dt: float, num_steps: int) -> List[irx.Interval]:
    """Propagate every scenario from the same x0_ivl/u and return their
    observed-output intervals.

    All scenarios share one emb_system (sensor faults only change the output
    map, never the dynamics -- see module docstring), so this vmaps over the
    stacked p_intervals in a single batched propagation instead of looping
    over scenarios in Python (which would trace one independent
    fori_loop/unrolled-block per scenario)."""
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x_final_batch = jax.vmap(lambda p: _propagate_by_params(x0_ivl, u, emb_sys, p, dt, num_steps))(p_batch)
    return [
        observed_output(
            irx.Interval(lower=x_final_batch.lower[i], upper=x_final_batch.upper[i]),
            scenarios[i],
        )
        for i in range(len(scenarios))
    ]


def separation_loss(u: jnp.ndarray,
                    x0_ivl: irx.Interval,
                    scenarios: List[Scenario],
                    dt: float,
                    num_steps: int) -> jnp.ndarray:
    """Sum of pairwise observed-output-interval overlaps.

    Minimising this loss maximises the separation of the scenarios'
    reachable sets in *observed output* space y = beta*x + xi.
    """
    y_ivls = _propagate_all_scenarios(x0_ivl, u, scenarios, dt, num_steps)
    n = len(y_ivls)
    total = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            total = total + _overlap_volume(y_ivls[i], y_ivls[j])
    return total


class SeparatingInputOptimizer:
    """Gradient-descent optimizer for a fault-separating constant control input."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.num_steps = num_steps

        _loss = partial(separation_loss, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def optimize(self, u_init: Optional[jnp.ndarray] = None,
                 learning_rate: float = 0.01, num_iters: int = 150,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        if u_init is None:
            u_init = jnp.array([0.5])
        u = u_init
        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)
            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = float(self.loss_fn(u))
                print(f"  Iter {i:4d}  loss={loss:.6f}  u={float(u[0]):+.4f}  |g|={float(jnp.linalg.norm(g)):.4f}")
        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        y_ivls = _propagate_all_scenarios(self.x0_ivl, u, self.scenarios, self.dt, self.num_steps)
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_overlap_volume(y_ivls[i], y_ivls[j]))
        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, y_ivls)
        }
        return {'output_intervals': y_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes}


def optimize_parallel_gpu(opt: 'SeparatingInputOptimizer', num_restarts: int = 100,
                          learning_rate: float = 0.01, num_iters: int = 150,
                          seed: int = 42):
    """GPU-parallel multi-start gradient descent for a constant separating input."""
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, 1)) * 0.3 + jnp.array([0.5])

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u, _i):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def optimize_parallel_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                dt: float, num_steps: int, num_restarts: int = 100,
                                learning_rate: float = 0.01, num_iters: int = 150,
                                seed: int = 42):
    return optimize_parallel_gpu(
        SeparatingInputOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt, num_steps=num_steps),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Multistep ("Unrefined" / "Uninformed") Path
# ══════════════════════════════════════════════════════════════════════════════

def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate and record the state interval at the end of every segment.

    Returns an irx.Interval whose lower/upper have shape
    (num_segments, N) -- one slice per segment end.

    Two levels of gradient checkpointing (mirrors go2's pattern): the
    per-Euler-step body and the per-segment scan body are both
    jax.checkpoint-wrapped, so peak backward-pass memory scales as
    O(num_segments x N) rather than O(num_segments x embedding-op-count).
    """
    def segment(x_ivl, u_k):
        # segment as a whole is already jax.checkpoint-wrapped below (one
        # boundary per segment); the inner per-Euler-step loop doesn't need
        # its own nested checkpoint on top of that.
        def step(x, _i): return euler_step(emb_sys, x, u_k, p_ivl, dt)
        x_end = _plain_unroll(step, x_ivl, steps_per_segment)
        return x_end, x_end

    _, x_hist = jax.lax.scan(jax.checkpoint(segment), x0_ivl, u_seq)
    return x_hist


def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                 scenario: Scenario, dt: float,
                                 steps_per_segment: int) -> irx.Interval:
    """Propagate x0_ivl through a sequence of control inputs; return final interval."""
    x_hist = _propagate_history(
        x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment
    )
    return irx.Interval(lower=x_hist.lower[-1], upper=x_hist.upper[-1])


def separation_loss_multistep(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                              scenarios: List[Scenario], dt: float,
                              steps_per_segment: int) -> jnp.ndarray:
    """Min over segments of the pairwise observed-output-interval overlap sum.

    All scenarios in this module share one emb_system (sensor faults only
    change the output map, never the dynamics), so -- unlike go2 -- every
    scenario can be propagated in a single vmap over stacked p_intervals,
    with no need to bucket by emb_system identity first.
    """
    num_segments = u_seq.shape[0]
    n = len(scenarios)
    emb_sys = scenarios[0].emb_system

    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def prop_one(p_ivl_single):
        return _propagate_history(x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment)

    # x_hist_batch: Interval with lower/upper shape (n, num_segments, N)
    x_hist_batch = jax.vmap(prop_one)(p_batch)

    def overlap_at_k(k):
        y_ivls_k = [
            observed_output(
                irx.Interval(lower=x_hist_batch.lower[i, k], upper=x_hist_batch.upper[i, k]),
                scenarios[i],
            )
            for i in range(n)
        ]
        total = jnp.array(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                total = total + _overlap_volume(y_ivls_k[i], y_ivls_k[j])
        return total

    segment_overlaps = jnp.stack([overlap_at_k(k) for k in range(num_segments)])
    return jnp.min(segment_overlaps)


class MultistepSequenceOptimizer:
    """Container for multistep loss/grad callables and sequence shape."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, steps_per_segment: int, num_segments: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments

        _loss = partial(separation_loss_multistep, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, steps_per_segment=steps_per_segment)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))


def optimize_multistep_gpu(opt: 'MultistepSequenceOptimizer', num_restarts: int = 100,
                           learning_rate: float = 0.01, num_iters: int = 150, seed: int = 42):
    """GPU-parallel multi-start optimization for control sequences."""
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, opt.num_segments, 1)) * 0.1 + jnp.array([0.5])

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u_batch, _i):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def optimize_multistep_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                 dt: float, steps_per_segment: int, num_segments: int,
                                 num_restarts: int = 100, learning_rate: float = 0.01,
                                 num_iters: int = 150, seed: int = 42):
    return optimize_multistep_gpu(
        MultistepSequenceOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
                                   steps_per_segment=steps_per_segment, num_segments=num_segments),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
    )


def optimize_multistep(scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float,
                       steps_per_segment: int, num_segments: int,
                       learning_rate: float = 0.01, num_iters: int = 300,
                       num_restarts: int = 100, verbose: bool = False,
                       seed: int = 42) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent over a sequence of control inputs (unrefined loss)."""
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    ms_opt = MultistepSequenceOptimizer(
        scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
        steps_per_segment=steps_per_segment, num_segments=num_segments,
    )

    _t0 = time.perf_counter()
    u_seq, loss_opt_jax, _, final_losses = optimize_multistep_gpu(
        opt=ms_opt, num_restarts=num_restarts, learning_rate=learning_rate,
        num_iters=num_iters, seed=seed,
    )
    elapsed = time.perf_counter() - _t0

    best_idx = int(jnp.argmin(final_losses))
    loss_opt = float(loss_opt_jax)

    if verbose:
        mean_loss = float(jnp.mean(final_losses))
        print(f"Multistep GPU multistart complete: best_loss={loss_opt:.6f}  "
              f"mean_final_loss={mean_loss:.6f}  best_restart={best_idx+1}/{num_restarts}")

    y_ivls = [
        observed_output(propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment), s)
        for s in scenarios
    ]
    n = len(scenarios)
    overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            overlaps[f"{scenarios[i].name} vs {scenarios[j].name}"] = float(_overlap_volume(y_ivls[i], y_ivls[j]))
    volumes = {s.name: float(jnp.prod(iv.upper - iv.lower)) for s, iv in zip(scenarios, y_ivls)}
    stats = {
        'output_intervals': y_ivls,
        'pairwise_overlaps': overlaps,
        'volumes': volumes,
        'optimization_time_s': elapsed,
        'best_restart_idx': best_idx,
        'all_restart_losses': np.array(final_losses),
        'num_restarts': num_restarts,
    }
    return u_seq, loss_opt, stats


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Intersection-Refinement Path
# ══════════════════════════════════════════════════════════════════════════════

def propagate_with_refinement(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                              scenarios: List[Scenario], dt: float,
                              num_steps: int = 2) -> jnp.ndarray:
    """Multi-step propagation with per-pair observation-based state refinement.

    Generalizes faulty_car's propagate_with_refinement (see
    propagate_with_refinement_pseudocode.md) to a fully-observed N-vector
    state and an arbitrary scenario count.  Since every state is observed
    here (unlike faulty_car's unobserved theta), refinement tightens the
    *entire* state vector every step -- no unobserved slice to special-case.

    Step 1 (computed once)
    -----------------------
    Propagate every scenario one Euler step from x0_ivl using u_seq[0].
    First cost = sum of pairwise overlaps of the observed outputs y_i.

    Steps 2 .. num_steps (one fori_loop iteration per pair, per step)
    --------------------------------------------------------------------
    For each ordered pair (i, j):
      1. y_int = intersection of the pair's current observed-output intervals.
      2. x_ref_i, x_ref_j = _invert_observation(y_int, scenario) for each of
         i, j (full N-vector refinement).  If the pair's outputs don't
         currently overlap, skip refinement for this pair this step (carry
         the un-refined state forward instead) and contribute 0 to the cost.
      3. Propagate x_ref_i, x_ref_j one Euler step with u_seq[k].
      4. Accumulate overlap of the newly propagated observed outputs.
    Each pair tracks its own state independently after the first refinement
    (their per-pair refined states diverge), so state is carried as
    (n_pairs, 2N) flattened-interval arrays through a jax.lax.fori_loop.

    The step body is jax.checkpoint-wrapped (a gap in the ported-from
    faulty_car version, fixed here): without it, reverse-mode AD through the
    fori_loop retains O(num_steps) worth of per-step residuals; with it,
    only O(1) (the loop's own carry) is retained, and the cheap O(N) forward
    ops are recomputed on the backward pass instead.

    Returns
    -------
    min over steps of per-step pairwise overlap sum (scalar)
    """
    n = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    N = x0_ivl.lower.shape[0]

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:N], upper=arr[N:])

    # ── Step 1: propagate all scenarios one Euler step with u_seq[0] ──────
    # (single vmap over the shared emb_system's stacked p_intervals, instead
    # of n separate Python-level euler_step calls.)
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x1_batch = jax.vmap(lambda p: euler_step(emb_sys, x0_ivl, u_seq[0], p, dt))(p_batch)
    x1_ivls = [irx.Interval(lower=x1_batch.lower[i], upper=x1_batch.upper[i]) for i in range(n)]
    y1_ivls = [observed_output(x1, s) for x1, s in zip(x1_ivls, scenarios)]
    step1_cost = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            step1_cost = step1_cost + _overlap_volume(y1_ivls[i], y1_ivls[j])

    pxi_arr = jnp.stack([ivl_to_arr(x1_ivls[i]) for i, j in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x1_ivls[j]) for i, j in pairs])

    # NOTE: checkpointing is applied once, around the whole step loop, by
    # _run_unrolled_or_loop below -- step_body itself is intentionally not
    # jax.checkpoint-wrapped to avoid a redundant nested boundary.
    def step_body(carry, k):
        pxi_arr, pxj_arr, min_cost = carry
        u_k = u_seq[k + 1]
        step_cost = jnp.array(0.0)
        new_pxi, new_pxj = [], []

        for idx, (i, j) in enumerate(pairs):
            x_curr_i = arr_to_ivl(pxi_arr[idx])
            x_curr_j = arr_to_ivl(pxj_arr[idx])

            y_i = observed_output(x_curr_i, scenarios[i])
            y_j = observed_output(x_curr_j, scenarios[j])

            y_lo = jnp.maximum(y_i.lower, y_j.lower)
            y_hi = jnp.minimum(y_i.upper, y_j.upper)
            has_overlap = jnp.all(y_hi >= y_lo)
            y_int = irx.Interval(lower=y_lo, upper=y_hi)

            x_ref_i_overlap = _invert_observation(y_int, scenarios[i])
            x_ref_j_overlap = _invert_observation(y_int, scenarios[j])

            x_ref_i = irx.Interval(
                lower=jnp.where(has_overlap, x_ref_i_overlap.lower, x_curr_i.lower),
                upper=jnp.where(has_overlap, x_ref_i_overlap.upper, x_curr_i.upper),
            )
            x_ref_j = irx.Interval(
                lower=jnp.where(has_overlap, x_ref_j_overlap.lower, x_curr_j.lower),
                upper=jnp.where(has_overlap, x_ref_j_overlap.upper, x_curr_j.upper),
            )

            x_next_i = euler_step(scenarios[i].emb_system, x_ref_i, u_k, scenarios[i].p_interval, dt)
            x_next_j = euler_step(scenarios[j].emb_system, x_ref_j, u_k, scenarios[j].p_interval, dt)

            raw_cost = _overlap_volume(
                observed_output(x_next_i, scenarios[i]),
                observed_output(x_next_j, scenarios[j]),
            )
            step_cost = step_cost + jnp.where(has_overlap, raw_cost, jnp.array(0.0))
            new_pxi.append(ivl_to_arr(x_next_i))
            new_pxj.append(ivl_to_arr(x_next_j))

        return (jnp.stack(new_pxi), jnp.stack(new_pxj), jnp.minimum(min_cost, step_cost))

    init_carry = (pxi_arr, pxj_arr, step1_cost)
    _, _, min_cost_final = _run_unrolled_or_loop(step_body, init_carry, num_steps - 1)
    return min_cost_final


def refined_overlap_loss(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                         scenarios: List[Scenario], dt: float,
                         num_steps: int = 2) -> jnp.ndarray:
    """Separation loss using multi-step propagation with state refinement."""
    return propagate_with_refinement(x0_ivl, u_seq, scenarios, dt, num_steps)


def optimize_refined_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                         num_steps: int = 2, num_restarts: int = 50,
                         learning_rate: float = 0.05, num_iters: int = 200,
                         seed: int = 42) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent minimising refined_overlap_loss."""
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, num_steps, 1)) * 0.3 + jnp.array([0.5])

    def loss_fn_refined(u_seq):
        return refined_overlap_loss(u_seq, x0_ivl, scenarios, dt, num_steps)

    batched_loss = jax.vmap(loss_fn_refined)
    batched_grad = jax.vmap(jax.grad(loss_fn_refined))

    def body(u_batch, _i):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses
