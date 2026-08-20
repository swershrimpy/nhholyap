"""
Decoupled Cubic-Drift Chain – Active Fault Diagnosis with immrax
=================================================================
Reachable-set-based active fault diagnosis for N dynamically-decoupled
channels with a nonlinear (cubic/Duffing-type) drift and per-channel
heterogeneous physical constants, using immrax for interval arithmetic and
natural-embedding propagation. See PLAN.md / TASKS.md in this directory for
the full design writeup this module implements.

System
------
  State   x = [x1, ..., xN]
  Control u = [u1, ..., uN]      (one control input per channel, shape (N,) --
                                   unlike integrator_chain's shared scalar u)
  Params  p = alpha               (per-channel actuator-fault authority,
                                    shape (N,); alpha_i=1 -> nominal)

  Dynamics (vectorized, no Python loop over N):
      x_dot = a * x - b * x**3 + alpha * u

  a, b (shape (N,)) are FIXED per-channel physical constants -- drawn once at
  system-construction time (see `default_channel_params`), not part of `p`;
  they make channel i physically different from channel j but are never
  disputed by a fault scenario. Channels are dynamically decoupled (x_i
  depends only on x_i, u_i); coupling only happens through the shared
  fault-separation objective, exactly as in go2/admire/faulty_car.

Sensor model
------------
  All N states are directly observed:  y = beta ⊙ x + xi
  Same functional form and invertibility requirement (beta > 0) as
  integrator_chain; ported near-verbatim since both `observed_output` and
  `_invert_observation` already operate elementwise over an N-vector.

Scenarios
---------
  Nominal              alpha=1 (all channels),         beta=1, xi=0
  ActuatorFault_i       alpha_i in [lo,hi], else 1,      beta=1, xi=0
  SensorFault_i         alpha=1,             beta_i in [lo,hi], xi_i in [-eps,eps], else beta=1,xi=0

  Total scenario count is `1 + Ka + Ks`, where `Ka`/`Ks` (number of
  actuator-/sensor-fault scenarios, one per faulted channel) each default to
  `N` -- giving the `2N+1` baseline (Nominal + N ActuatorFault_i +
  N SensorFault_i) -- but are independently user-configurable via
  `create_scenarios(..., num_actuator_faults=Ka, num_sensor_faults=Ks)` or
  explicit `actuator_fault_indices`/`sensor_fault_indices` lists (see
  `create_scenarios` docstring).

Three algorithmic layers (mirroring integrator_separating_input.py /
go2_separating_input_immrax.py / faulty_car_separating_input.py):
  1. Single-step separating input (Section 3)
  2. Multistep "unrefined"/"uninformed" separating input sequence (Section 4)
  3. Multistep intersection-refinement separating input sequence (Section 5)

Memory / performance notes (see PLAN.md §6 for the full discussion):
  - float32 throughout (JAX default); never enable x64.
  - jax.checkpoint on propagation loop bodies that are emitted as a
    jax.lax.scan; NOT on the Python-unrolled ones, where it wrapped the
    whole chain rather than one step and cost ~1/3 of the runtime to save
    kilobytes -- see _run_unrolled_or_loop's docstring.
  - flat-array (ivl_to_arr/arr_to_ivl) pytree carries in the refinement loop,
    not nested Interval structs, to minimize pytree-dispatch overhead.
  - _overlap_volume replaces overlap_size_lax's lax.cond with a branchless
    clipped-product formula.
  - Scenario propagation is vmapped across scenarios (all scenarios share one
    emb_system; only p=alpha and the output map (beta,xi) differ).
  - O(N^2) pairwise-overlap cost: `1 + K + N` scenarios means
    `C(1+K+N, 2)` pairwise-overlap terms per loss evaluation, growing
    quadratically with N/K -- keep demo N small (3-10) or restrict
    `separation_loss` to a curated pair subset for larger N.
"""

import sys
import time
import resource
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Optional, Dict

# File is at examples/nonlinear_chain/<name>.py  ->  parents[1] = examples/
_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

# Control input box constraint: u_i in [-1, 1] for every channel. Kept as
# plain Python scalars (not a shape-(N,) array) since N varies per system
# instance -- jnp.clip broadcasts scalars against any (..., N) array.
_U_LO = -1.0
_U_HI = 1.0

# Loops (Euler steps, GD iterations) with a static length <= this are fully
# unrolled into straight-line code instead of lax.fori_loop/scan -- see
# `_run_unrolled_or_loop` below. Ported unchanged from integrator_chain.
_UNROLL_THRESHOLD = 64


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    """Project a control input (or batch) onto the feasible box [-1, 1]^N."""
    return jnp.clip(u, _U_LO, _U_HI)


def _pair_indices(n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Static (i, j) index arrays for all C(n, 2) unordered pairs, i < j.

    Turns a Python-level "for i: for j:" pairwise sum into a single
    jax.vmap call: gather scenario i's and j's stacked data at these
    indices, then vmap the per-pair function over the resulting (P, ...)
    arrays. n is always a static Python int (len(scenarios)), so this list
    comprehension runs once at trace time -- the returned arrays are baked
    into the trace as constants. Mirrors admire_separating_input.py's
    identical helper (same fix, same shape of problem).
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return jnp.array([i for i, j in pairs]), jnp.array([j for i, j in pairs])


def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    """Branchless pairwise axis-aligned-box overlap volume. Ported verbatim
    from integrator_separating_input.py (already generic over N)."""
    widths = jnp.maximum(
        jnp.minimum(ivl1.upper, ivl2.upper) - jnp.maximum(ivl1.lower, ivl2.lower),
        0.0,
    )
    return jnp.prod(widths)


def _run_unrolled_or_loop(step_fn, init, n: int, unroll_threshold: int = _UNROLL_THRESHOLD):
    """Apply step_fn (carry, i) -> carry exactly n times, starting from init.

    The unrolled branch deliberately does NOT wrap the chain in
    jax.checkpoint (it used to, ported from integrator_separating_input.py).
    That call wrapped the WHOLE n-step chain rather than one step, so under
    jax.grad the program executed 3n step bodies -- n forward, n
    rematerialised during the backward pass, n vjp -- where a plain unroll
    executes 2n. What it bought was not storing the n intermediate carries,
    and those are tiny: the refinement loop's carry is a few
    (n_pairs, 2, 2N) float32 arrays, under a megabyte per step even at the
    largest measured sweep point, against a measured *device* memory peak of
    20-25MB on a 32GB V100 (the memory_kb column of refined_*_scaling.csv;
    the multi-gigabyte peak_rss_mb column is host-side compiler memory,
    which remat does not help). So the checkpoint was trading roughly a
    third of the runtime for a memory saving four orders of magnitude away
    from mattering.

    The scan branch keeps its checkpoint: there the decorator sits on
    scan_body, i.e. it is a genuine per-step remat, and n > unroll_threshold
    is exactly the regime where storing every carry starts to add up.
    """
    if n <= unroll_threshold:
        return _plain_unroll(step_fn, init, n)
    else:
        @jax.checkpoint
        def scan_body(carry, i):
            return step_fn(carry, i), None
        carry, _ = jax.lax.scan(scan_body, init, xs=jnp.arange(n))
        return carry


def _run_unrolled_or_loop_nocheckpoint(step_fn, init, n: int):
    """Advance `init` through step_fn exactly n times via jax.lax.scan.
    Ported verbatim from integrator_separating_input.py (see its docstring
    for why the outer GD loop deliberately doesn't unroll)."""
    def body(carry, i):
        return step_fn(carry, i), None
    carry, _ = jax.lax.scan(body, init, xs=jnp.arange(n))
    return carry


def _plain_unroll(step_fn, init, n: int):
    """Bare Python-unrolled loop, no checkpoint, no scan. Ported verbatim."""
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
    """JIT-compile `fn` and separately measure compile time vs. run time."""
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

class CubicChainSystem(irx.System):
    """Decoupled per-channel cubic-drift chain with actuator-fault authority alpha.

    State   x = [x_1, ..., x_N]
    Control u = [u_1, ..., u_N]           (shape (N,), NOT scalar)
    Params  p = alpha                      (shape (N,), alpha_i=1 -> nominal)

    Dynamics (vectorized, no Python loop over N):
        x_dot = a * x - b * x**3 + alpha * u
    """

    def __init__(self, N: int, a: jnp.ndarray, b: jnp.ndarray):
        self.evolution = 'continuous'
        self.xlen = N
        self.N = N
        self.a = a   # shape (N,), fixed per-channel linear-drift coefficient
        self.b = b   # shape (N,), fixed per-channel cubic-damping coefficient

    def f(self, t, x, u, p):
        alpha = p
        return self.a * x - self.b * x**3 + alpha * u


def default_channel_params(N: int, a_lo: float = 0.5, a_hi: float = 1.5,
                           b_lo: float = 0.2, b_hi: float = 0.8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic default (a, b) per channel via linspace over the given
    ranges -- see PLAN.md §6 ("a, b provenance") for why linspace was chosen
    over a seeded RNG: reproducible with zero extra API surface."""
    a = jnp.linspace(a_lo, a_hi, N)
    b = jnp.linspace(b_lo, b_hi, N)
    return a, b


# Module-level cache: one CubicChainSystem + one irx.natemb(...) per distinct
# (N, a, b) triple -- a/b are part of the traced system (unlike
# integrator_chain, where the cache key was just N), so they're folded into
# the key as hashable tuples.
_EMB_CACHE: Dict[Tuple[int, Tuple[float, ...], Tuple[float, ...]], Tuple[CubicChainSystem, object]] = {}


def get_system_and_embedding(N: int, a: jnp.ndarray, b: jnp.ndarray) -> Tuple[CubicChainSystem, object]:
    """Return (system, natural_embedding) for the given (N, a, b), cached."""
    key = (N, tuple(np.asarray(a).tolist()), tuple(np.asarray(b).tolist()))
    if key not in _EMB_CACHE:
        sys_ = CubicChainSystem(N, a, b)
        emb = irx.natemb(sys_)
        _EMB_CACHE[key] = (sys_, emb)
    return _EMB_CACHE[key]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Scenarios & Output Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One fault scenario: shared dynamics (via alpha) + an output fault model.

    beta, xi : shape (N,) interval-valued fault parameters for the affine
               sensor map y = beta ⊙ x + xi. beta must be strictly positive
               in every scenario (enforced in create_scenarios) so that the
               map is invertible -- required by the intersection-refinement
               algorithm in Section 5.
    """
    name: str
    emb_system: object
    p_interval: irx.Interval   # alpha interval, shape (N,)
    beta: irx.Interval         # shape (N,)
    xi: irx.Interval           # shape (N,)


def observed_output(x_ivl: irx.Interval, scenario: Scenario) -> irx.Interval:
    """y = beta ⊙ x + xi, via interval arithmetic. Ported verbatim from
    integrator_separating_input.py (already elementwise over N)."""
    beta, xi = scenario.beta, scenario.xi
    return irx.Interval(
        lower=beta.lower * x_ivl.lower + xi.lower,
        upper=beta.upper * x_ivl.upper + xi.upper,
    )


def _invert_observation(y_ivl: irx.Interval, scenario: Scenario) -> irx.Interval:
    """Outer-enclose x from y = beta*x + xi, given uncertain beta (>0) and xi.
    Ported verbatim from integrator_separating_input.py."""
    beta, xi = scenario.beta, scenario.xi
    v_lower = y_ivl.lower - xi.upper
    v_upper = y_ivl.upper - xi.lower
    corners = jnp.stack([
        v_lower / beta.lower, v_lower / beta.upper,
        v_upper / beta.lower, v_upper / beta.upper,
    ])
    return irx.Interval(lower=jnp.min(corners, axis=0), upper=jnp.max(corners, axis=0))


def _resolve_fault_indices(
    N: int,
    num_faults: Optional[int],
    fault_indices: Optional[List[int]],
    count_param_name: str,
    indices_param_name: str,
) -> List[int]:
    """Resolve which channels get a fault scenario of one type (actuator or
    sensor -- the two callers share this logic, only the param names differ
    for error messages).

    - If `fault_indices` is given, it takes precedence: use exactly those
      channels (order preserved), after validating every index is in range
      `[0, N)` and there are no duplicates -- out-of-range indices are
      rejected (raise ValueError) rather than silently dropped or clamped.
    - Otherwise, `num_faults` (default N, i.e. every channel -- the `2N+1`
      baseline) selects the FIRST K channels `0, 1, ..., K-1`.
    """
    if fault_indices is not None:
        indices = list(fault_indices)
        out_of_range = [i for i in indices if i < 0 or i >= N]
        if out_of_range:
            raise ValueError(
                f"{indices_param_name} out of range for N={N}: {out_of_range}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"{indices_param_name} contains duplicates: {indices}")
        return indices

    K = N if num_faults is None else num_faults
    if not (1 <= K <= N):
        raise ValueError(f"{count_param_name} must be in [1, N={N}], got {K}")
    return list(range(K))


def create_scenarios(
    N: int,
    a: jnp.ndarray,
    b: jnp.ndarray,
    actuator_alpha_lo: float = 0.5,
    actuator_alpha_hi: float = 0.9,
    sensor_beta_lo: float = 0.8,
    sensor_beta_hi: float = 1.2,
    sensor_xi_bound: float = 0.1,
    num_actuator_faults: Optional[int] = None,
    actuator_fault_indices: Optional[List[int]] = None,
    num_sensor_faults: Optional[int] = None,
    sensor_fault_indices: Optional[List[int]] = None,
) -> List[Scenario]:
    """Return `1 + Ka + Ks` fault scenarios: Nominal, Ka single-channel
    ActuatorFault_i, and Ks single-channel SensorFault_i.

    Parameters
    ----------
    N : number of channels (state dimension)
    a, b : shape-(N,) fixed per-channel drift/damping constants (see
           `default_channel_params`)
    actuator_alpha_lo / hi : range of the per-channel actuator-fault
                             authority alpha
    sensor_beta_lo / hi    : range of the per-channel sensor gain fault beta
                             (must stay strictly positive)
    sensor_xi_bound        : half-width of the per-channel sensor bias/noise
                             fault interval xi in [-sensor_xi_bound, +sensor_xi_bound]
    num_actuator_faults    : Ka, the number of ActuatorFault_i scenarios to
                             generate (one per faulted channel). Defaults to
                             N (every channel faulted once -> the `2N+1`
                             baseline when num_sensor_faults is also
                             defaulted). Ignored if `actuator_fault_indices`
                             is given. Must satisfy `1 <= Ka <= N`.
    actuator_fault_indices : explicit list of channel indices to generate
                             ActuatorFault_i scenarios for, overriding
                             `num_actuator_faults`. Every index must be in
                             `[0, N)` with no duplicates, or this raises
                             ValueError.
    num_sensor_faults      : Ks, the number of SensorFault_i scenarios to
                             generate (one per faulted channel). Defaults to
                             N, mirroring `num_actuator_faults`. Ignored if
                             `sensor_fault_indices` is given. Must satisfy
                             `1 <= Ks <= N`.
    sensor_fault_indices    : explicit list of channel indices to generate
                             SensorFault_i scenarios for, overriding
                             `num_sensor_faults`. Same validation as
                             `actuator_fault_indices`.
    """
    _, emb = get_system_and_embedding(N, a, b)

    ones_N, zeros_N = jnp.ones(N), jnp.zeros(N)

    def point_ivl(v):
        return irx.Interval(lower=v, upper=v)

    beta_nominal, xi_nominal = point_ivl(ones_N), point_ivl(zeros_N)

    scenarios = [
        Scenario("Nominal", emb, point_ivl(ones_N), beta_nominal, xi_nominal),
    ]

    actuator_indices = _resolve_fault_indices(
        N, num_actuator_faults, actuator_fault_indices,
        "num_actuator_faults", "actuator_fault_indices",
    )
    for i in actuator_indices:
        alpha_lo = ones_N.at[i].set(actuator_alpha_lo)
        alpha_hi = ones_N.at[i].set(actuator_alpha_hi)
        scenarios.append(Scenario(
            f"ActuatorFault_{i}", emb,
            irx.Interval(lower=alpha_lo, upper=alpha_hi),
            beta_nominal, xi_nominal,
        ))

    sensor_indices = _resolve_fault_indices(
        N, num_sensor_faults, sensor_fault_indices,
        "num_sensor_faults", "sensor_fault_indices",
    )
    for i in sensor_indices:
        beta_lo = ones_N.at[i].set(sensor_beta_lo)
        beta_hi = ones_N.at[i].set(sensor_beta_hi)
        xi_lo = zeros_N.at[i].set(-sensor_xi_bound)
        xi_hi = zeros_N.at[i].set(sensor_xi_bound)
        scenarios.append(Scenario(
            f"SensorFault_{i}", emb,
            point_ivl(ones_N),
            irx.Interval(lower=beta_lo, upper=beta_hi),
            irx.Interval(lower=xi_lo, upper=xi_hi),
        ))

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
    scenario-batched loss functions."""
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
    observed-output intervals -- a single vmap over stacked p_intervals
    since all scenarios share one emb_system."""
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
    """Sum of pairwise observed-output-interval overlaps."""
    y_ivls = _propagate_all_scenarios(x0_ivl, u, scenarios, dt, num_steps)
    n = len(y_ivls)
    # vmap over all C(n,2) pairs at once instead of a Python "for i: for j:"
    # double loop -- see _pair_indices' docstring.
    lo_stack = jnp.stack([iv.lower for iv in y_ivls])
    hi_stack = jnp.stack([iv.upper for iv in y_ivls])
    pair_i, pair_j = _pair_indices(n)
    ivl_i = irx.Interval(lower=lo_stack[pair_i], upper=hi_stack[pair_i])
    ivl_j = irx.Interval(lower=lo_stack[pair_j], upper=hi_stack[pair_j])
    return jnp.sum(jax.vmap(_overlap_volume)(ivl_i, ivl_j))


class SeparatingInputOptimizer:
    """Gradient-descent optimizer for a fault-separating constant control input."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.num_steps = num_steps
        self.N = x0_ivl.lower.shape[0]

        _loss = partial(separation_loss, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def optimize(self, u_init: Optional[jnp.ndarray] = None,
                 learning_rate: float = 0.01, num_iters: int = 150,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        if u_init is None:
            u_init = jnp.full(self.N, 0.5)
        u = u_init
        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)
            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = float(self.loss_fn(u))
                print(f"  Iter {i:4d}  loss={loss:.6f}  |u|={float(jnp.linalg.norm(u)):.4f}  |g|={float(jnp.linalg.norm(g)):.4f}")
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
    u0 = jax.random.normal(key, (num_restarts, opt.N)) * 0.3 + 0.5

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
    Returns an irx.Interval whose lower/upper have shape (num_segments, N)."""
    def segment(x_ivl, u_k):
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
    """Min over segments of the pairwise observed-output-interval overlap sum."""
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

    # observed_output per (scenario, segment) -- still O(n) via vmap over
    # scenarios (unchanged, already cheap); beta/xi differ per scenario so
    # this vmap is over the SAME scenario-stacked axis x_hist_batch already has.
    beta_lo = jnp.stack([s.beta.lower for s in scenarios])
    beta_hi = jnp.stack([s.beta.upper for s in scenarios])
    xi_lo = jnp.stack([s.xi.lower for s in scenarios])
    xi_hi = jnp.stack([s.xi.upper for s in scenarios])

    def obs_one(x_lo, x_hi, b_lo, b_hi, xi_l, xi_h):
        return b_lo * x_lo + xi_l, b_hi * x_hi + xi_h

    # x_hist_batch.lower/upper: (n, num_segments, N) -> vmap over scenario axis
    y_lo, y_hi = jax.vmap(obs_one)(x_hist_batch.lower, x_hist_batch.upper,
                                   beta_lo, beta_hi, xi_lo, xi_hi)  # (n, num_segments, N)

    # vmap over all C(n,2) pairs AND all segments at once instead of a
    # Python "for i: for j:" double loop repeated once per segment
    # (n_pairs x num_segments unrolled calls previously) -- see
    # _pair_indices' docstring.
    pair_i, pair_j = _pair_indices(n)
    ivl_i = irx.Interval(lower=y_lo[pair_i], upper=y_hi[pair_i])  # (P, num_segments, N)
    ivl_j = irx.Interval(lower=y_lo[pair_j], upper=y_hi[pair_j])
    pairwise_at_segment = jax.vmap(jax.vmap(_overlap_volume))(ivl_i, ivl_j)  # (P, num_segments)
    segment_overlaps = jnp.sum(pairwise_at_segment, axis=0)  # (num_segments,)
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
        self.N = x0_ivl.lower.shape[0]

        _loss = partial(separation_loss_multistep, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, steps_per_segment=steps_per_segment)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))


def optimize_multistep_gpu(opt: 'MultistepSequenceOptimizer', num_restarts: int = 100,
                           learning_rate: float = 0.01, num_iters: int = 150, seed: int = 42):
    """GPU-parallel multi-start optimization for control sequences."""
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, opt.num_segments, opt.N)) * 0.1 + 0.5

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
    Ported from integrator_separating_input.py -- see that module's docstring
    for the full algorithm description; unchanged here beyond N/scenario-count.

    The per-pair step body is vmapped over all C(n,2) pairs (see
    _pair_indices' docstring) rather than a Python "for idx, (i, j) in
    enumerate(pairs):" loop -- previously unrolled into n_pairs separate
    unfused ops PER propagation step (n_pairs x (num_steps-1) calls total),
    the same anti-pattern separation_loss_multistep had. All scenarios
    share one emb_system (already used unbatched in Step 1 below), so only
    the per-scenario beta/xi/p_interval need to be pair-gathered.
    """
    n = len(scenarios)
    pair_i, pair_j = _pair_indices(n)
    N = x0_ivl.lower.shape[0]

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:N], upper=arr[N:])

    # ── Step 1: propagate all scenarios one Euler step with u_seq[0] ──────
    emb_sys = scenarios[0].emb_system
    p_lo = jnp.stack([s.p_interval.lower for s in scenarios])
    p_hi = jnp.stack([s.p_interval.upper for s in scenarios])
    p_batch = irx.Interval(lower=p_lo, upper=p_hi)
    x1_batch = jax.vmap(lambda p: euler_step(emb_sys, x0_ivl, u_seq[0], p, dt))(p_batch)
    x1_ivls = [irx.Interval(lower=x1_batch.lower[i], upper=x1_batch.upper[i]) for i in range(n)]
    y1_ivls = [observed_output(x1, s) for x1, s in zip(x1_ivls, scenarios)]

    y1_lo = jnp.stack([iv.lower for iv in y1_ivls])
    y1_hi = jnp.stack([iv.upper for iv in y1_ivls])
    step1_cost = jnp.sum(jax.vmap(_overlap_volume)(
        irx.Interval(lower=y1_lo[pair_i], upper=y1_hi[pair_i]),
        irx.Interval(lower=y1_lo[pair_j], upper=y1_hi[pair_j]),
    ))

    # (P, 2N) carries, gathered by pair index instead of a Python per-pair
    # ivl_to_arr + stack loop.
    pxi_arr = jnp.concatenate([x1_batch.lower[pair_i], x1_batch.upper[pair_i]], axis=-1)
    pxj_arr = jnp.concatenate([x1_batch.lower[pair_j], x1_batch.upper[pair_j]], axis=-1)

    # Per-pair beta/xi (for observed_output/_invert_observation) and
    # p_interval (for euler_step), pre-gathered once outside the scan --
    # emb_sys itself is shared, so it stays an unbatched closure variable.
    beta_lo = jnp.stack([s.beta.lower for s in scenarios])
    beta_hi = jnp.stack([s.beta.upper for s in scenarios])
    xi_lo = jnp.stack([s.xi.lower for s in scenarios])
    xi_hi = jnp.stack([s.xi.upper for s in scenarios])

    def _gather(lo, hi, idx):
        return irx.Interval(lower=lo[idx], upper=hi[idx])

    beta_pi, beta_pj = _gather(beta_lo, beta_hi, pair_i), _gather(beta_lo, beta_hi, pair_j)
    xi_pi, xi_pj = _gather(xi_lo, xi_hi, pair_i), _gather(xi_lo, xi_hi, pair_j)
    p_pi, p_pj = _gather(p_lo, p_hi, pair_i), _gather(p_lo, p_hi, pair_j)

    class _ScenLike:
        """Minimal duck-typed stand-in for a Scenario, exposing only
        .beta/.xi -- observed_output/_invert_observation only ever read
        those two fields, so a per-pair-vmapped (beta, xi) slice can pass
        through the SAME, already-correct functions unchanged instead of
        re-deriving their math inline for the batched case."""
        __slots__ = ("beta", "xi")

        def __init__(self, beta, xi):
            self.beta = beta
            self.xi = xi

    def step_one_pair(pxi_arr_1, pxj_arr_1, beta_i1, xi_i1, p_i1,
                      beta_j1, xi_j1, p_j1, u_k):
        x_curr_i = arr_to_ivl(pxi_arr_1)
        x_curr_j = arr_to_ivl(pxj_arr_1)
        scen_i = _ScenLike(beta_i1, xi_i1)
        scen_j = _ScenLike(beta_j1, xi_j1)

        y_i = observed_output(x_curr_i, scen_i)
        y_j = observed_output(x_curr_j, scen_j)

        y_lo = jnp.maximum(y_i.lower, y_j.lower)
        y_hi = jnp.minimum(y_i.upper, y_j.upper)
        has_overlap = jnp.all(y_hi >= y_lo)
        y_int = irx.Interval(lower=y_lo, upper=y_hi)

        x_ref_i_overlap = _invert_observation(y_int, scen_i)
        x_ref_j_overlap = _invert_observation(y_int, scen_j)

        x_ref_i = irx.Interval(
            lower=jnp.where(has_overlap, x_ref_i_overlap.lower, x_curr_i.lower),
            upper=jnp.where(has_overlap, x_ref_i_overlap.upper, x_curr_i.upper),
        )
        x_ref_j = irx.Interval(
            lower=jnp.where(has_overlap, x_ref_j_overlap.lower, x_curr_j.lower),
            upper=jnp.where(has_overlap, x_ref_j_overlap.upper, x_curr_j.upper),
        )

        x_next_i = euler_step(emb_sys, x_ref_i, u_k, p_i1, dt)
        x_next_j = euler_step(emb_sys, x_ref_j, u_k, p_j1, dt)

        raw_cost = _overlap_volume(
            observed_output(x_next_i, scen_i),
            observed_output(x_next_j, scen_j),
        )
        pair_cost = jnp.where(has_overlap, raw_cost, jnp.array(0.0))
        return ivl_to_arr(x_next_i), ivl_to_arr(x_next_j), pair_cost

    step_all_pairs = jax.vmap(step_one_pair, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None))

    def step_body(carry, k):
        pxi_arr, pxj_arr, min_cost = carry
        u_k = u_seq[k + 1]
        new_pxi, new_pxj, pair_costs = step_all_pairs(
            pxi_arr, pxj_arr, beta_pi, xi_pi, p_pi, beta_pj, xi_pj, p_pj, u_k
        )
        step_cost = jnp.sum(pair_costs)
        return (new_pxi, new_pxj, jnp.minimum(min_cost, step_cost))

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
    N = x0_ivl.lower.shape[0]
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, num_steps, N)) * 0.3 + 0.5

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
