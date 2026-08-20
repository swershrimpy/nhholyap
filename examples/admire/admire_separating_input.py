"""
ADMIRE Separating Input – Active Fault Diagnosis with immrax
============================================================
Reachable-set-based active fault diagnosis for the ADMIRE aircraft using
immrax for interval arithmetic and natural-embedding propagation.

System
------
AdmireNineDoFLinAct: 9-DOF nonlinear aircraft with linearised actuator map
    State   x = [Vt, α, β, pb, qb, rb, ψ, θ, φ]   (9 states)
    Control u = [rc, lc, roe, rie, lie, loe, rudder, flap, yaw_tv, pitch_tv]
                                                      (10 inputs, rad)
    Params  p = [p0, …, p9]  actuator effectiveness ∈ [0, 1] per surface

Fault Scenarios (11 total)
--------------------------
  Nominal          p = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   (all surfaces healthy)
  Right Canard     p[0] = 0  (complete loss)
  Left Canard      p[1] = 0
  Right Outer Elev p[2] = 0
  Right Inner Elev p[3] = 0
  Left Inner Elev  p[4] = 0
  Left Outer Elev  p[5] = 0
  Rudder           p[6] = 0
  Flap             p[7] = 0
  Yaw TV           p[8] = 0
  Pitch TV         p[9] = 0

Output Space for Separation
----------------------------
Angular rates [pb, qb, rb] (state indices 3:6) are most sensitive to
actuator faults and serve as the observable output for fault diagnosis.

Follows the immrax usage pattern from go2/go2_separating_input_immrax.py.
"""

import sys
import os
from pathlib import Path

# File is at examples/admire/<name>.py  →  parents[1] = examples/
_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np
from functools import partial
from typing import List, Tuple, Optional, Dict, Any

# plotting helper dependency (used in type hints only)
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass

from faulty_car.interval_functions import overlap_size_log
from admire import AdmireNineDoFLinAct

# Control input box constraints: all 10 surface deflections ∈ [−0.05, 0.05] rad.
_U_LO = jnp.ones(10) * -0.05
_U_HI = jnp.ones(10) *  0.05


def _pair_indices(n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Static (i, j) index arrays for all C(n, 2) unordered pairs, i < j.

    Used to turn a Python-level "for i: for j:" pairwise sum into a single
    jax.vmap call: gather scenario i=0 and j's stacked data at these
    indices, then vmap the per-pair function over the resulting (P, ...)
    arrays. n is always a static Python int (len(scenarios)), so this list
    comprehension runs once at trace time, not per call -- the returned
    arrays are baked into the trace as constants, same as
    admire_refined_sequence_optimizer.py's identical `pairs`/`pair_i`/
    `pair_j` construction, which this mirrors so both the unrefined and
    refined loss paths build their pair indices the same way.
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return jnp.array([i for i, j in pairs]), jnp.array([j for i, j in pairs])


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    """Project a control input (or batch) onto the feasible box.

    Works for any leading batch dimensions — clips only the last axis.
    All 10 surface deflections ∈ [−0.5, 0.5] rad.
    """
    return jnp.clip(u, _U_LO, _U_HI)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Fault Scenarios
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One fault scenario: a named (embedding, parameter-interval) pair."""
    name: str
    emb_system: object        # irx.natemb(…) result
    p_interval: irx.Interval  # parameter interval for this scenario


# Module-level singleton (created once, reused across calls)
_ADMIRE_SYS = AdmireNineDoFLinAct()
_ADMIRE_EMB = irx.natemb(_ADMIRE_SYS)

# Human-readable names for the 10 control surfaces
_SURFACE_NAMES = [
    "Right Canard",       # idx 0
    "Left Canard",        # idx 1
    "Right Outer Elev",   # idx 2
    "Right Inner Elev",   # idx 3
    "Left Inner Elev",    # idx 4
    "Left Outer Elev",    # idx 5
    "Rudder",             # idx 6
    "Flap",               # idx 7
    "Yaw TV",             # idx 8
    "Pitch TV",           # idx 9
]


def create_scenarios(fault_effectiveness: float = 0.0) -> List[Scenario]:
    """Return 11 fault scenarios: nominal + one complete loss per surface.

    Parameters
    ----------
    fault_effectiveness:
        Actuator gain for the failed surface (0 = complete loss, default).
        Values in (0, 1) model partial faults.

    Returns
    -------
    List of 11 Scenario objects.  All share the same embedding system
    (_ADMIRE_EMB), so vmap over p_intervals is maximally efficient.
    """
    p_nominal = jnp.ones(10)

    scenarios = [
        Scenario(
            name="Nominal",
            emb_system=_ADMIRE_EMB,
            p_interval=irx.icentpert(p_nominal, jnp.zeros(10)),
        )
    ]

    for i, name in enumerate(_SURFACE_NAMES):
        p_fault = p_nominal.at[i].set(fault_effectiveness)
        scenarios.append(
            Scenario(
                name=name,
                emb_system=_ADMIRE_EMB,
                p_interval=irx.icentpert(p_fault, jnp.zeros(10)),
            )
        )

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Interval Propagation
# ══════════════════════════════════════════════════════════════════════════════

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    """One forward-Euler interval step via the natural embedding.

    Follows the admire_fault_demo style:
        x_new_ut = f_emb(t=0, x_ut, u, p) * dt  +  x_ut
        x_new    = ut2i(x_new_ut)

    Note: t must be a JAX array (not a Python scalar) so that
    eqx.filter_make_jaxpr traces it as an abstract input and the
    jaxpr invar count matches the natif_jaxpr arg count.
    """
    _t    = jnp.zeros(())   # t = 0 as a JAX scalar (so it is a jaxpr invar)
    x_ut  = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    """Propagate the initial interval *num_steps* Euler steps under a constant u.

    Returns the full 9-D state interval at the end of the horizon.
    JAX-compatible (uses lax.fori_loop so the function is JIT-able and
    differentiable w.r.t. *u*).
    """
    emb = scenario.emb_system
    p   = scenario.p_interval

    def body(i, x_carry):
        return euler_step(emb, x_carry, u, p, dt)

    return jax.lax.fori_loop(0, num_steps, body, x0_ivl)


def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate and record the state interval at the end of every segment.

    Returns an irx.Interval whose lower/upper have shape
    (num_segments, state_dim) — one slice per segment end.
    Designed to be vmapped over p_ivl to parallelise across scenarios.
    """
    def segment(x_ivl, u_k):
        def euler_body(_, x): return euler_step(emb_sys, x, u_k, p_ivl, dt)
        x_end = jax.lax.fori_loop(0, steps_per_segment, euler_body, x_ivl)
        return x_end, x_end   # carry, stacked output

    _, x_hist = jax.lax.scan(segment, x0_ivl, u_seq)
    return x_hist   # Interval: lower/upper shape (num_segments, state_dim)


def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                 scenario: Scenario,
                                 dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate x0_ivl through a sequence of control inputs.

    u_seq has shape (num_segments, 10).  u_seq[k] is held constant for
    *steps_per_segment* Euler steps, then u_seq[k+1] takes over, etc.
    Total horizon = num_segments × steps_per_segment × dt seconds.
    Returns the final state interval only.
    """
    x_hist = _propagate_history(
        x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment
    )
    return irx.Interval(lower=x_hist.lower[-1], upper=x_hist.upper[-1])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Loss Function & Gradient
# ══════════════════════════════════════════════════════════════════════════════

def output_interval(x_ivl: irx.Interval) -> irx.Interval:
    """Extract the angular-rate sub-interval [pb, qb, rb] (indices 3:6)."""
    return irx.Interval(lower=x_ivl.lower[3:6], upper=x_ivl.upper[3:6])


def separation_loss(u: jnp.ndarray,
                    x0_ivl: irx.Interval,
                    scenarios: List[Scenario],
                    dt: float,
                    num_steps: int) -> jnp.ndarray:
    """Sum of pairwise angular-rate-interval overlaps.

    Minimising this loss maximises the separation of all reachable sets in
    the [pb, qb, rb] output space, making actuator-fault diagnosis easier.

    Parameters
    ----------
    u         : constant control input (10,) in rad
    x0_ivl    : initial 9-D state interval
    scenarios : list of Scenario objects
    dt        : Euler step size (s)
    num_steps : number of Euler steps

    Returns
    -------
    Scalar overlap volume (rad³); lower is better.
    """
    out_ivls = [
        output_interval(propagate_scenario(x0_ivl, u, s, dt, num_steps))
        for s in scenarios
    ]
    n = len(out_ivls)
    # vmap over all C(n,2) pairs at once instead of a Python "for i: for j:"
    # double loop -- see _pair_indices' docstring. Same math (sum of
    # overlap_size_log over every unordered pair), verified equal
    # (value, JIT, and gradient) against the loop version before this
    # change shipped.
    lo_stack = jnp.stack([iv.lower for iv in out_ivls])  # (n, 3)
    hi_stack = jnp.stack([iv.upper for iv in out_ivls])  # (n, 3)
    pair_i, pair_j = _pair_indices(n)
    ivl_i = irx.Interval(lower=lo_stack[pair_i], upper=hi_stack[pair_i])  # (P, 3)
    ivl_j = irx.Interval(lower=lo_stack[pair_j], upper=hi_stack[pair_j])  # (P, 3)
    return jnp.sum(jax.vmap(overlap_size_log)(ivl_i, ivl_j))


def separation_loss_multistep(u_seq: jnp.ndarray,
                              x0_ivl: irx.Interval,
                              scenarios: List[Scenario],
                              dt: float,
                              steps_per_segment: int) -> jnp.ndarray:
    """Min over segments of the pairwise angular-rate-interval overlap sum.

    For each segment end k, computes the sum of pairwise output-interval
    overlaps across all scenarios.  Returns the minimum over all K segment
    ends — i.e., the loss is zero if the scenarios are fully separated at
    ANY point in the trajectory.

    All ADMIRE scenarios share the same embedding system, so vmap over
    their stacked p_intervals gives maximum parallelism.

    Parameters
    ----------
    u_seq            : (num_segments, 10) sequence of control inputs
    x0_ivl           : initial 9-D state interval
    scenarios        : list of Scenario objects
    dt               : Euler step size (s)
    steps_per_segment: Euler steps each control input is held for

    Returns
    -------
    Scalar (rad³); lower is better.
    """
    num_segments = u_seq.shape[0]
    n = len(scenarios)

    # ── Group scenarios by emb_system identity ────────────────────────────
    # For ADMIRE, all scenarios share _ADMIRE_EMB → one group, maximum vmap.
    emb_groups: Dict[int, tuple] = {}
    for idx, s in enumerate(scenarios):
        eid = id(s.emb_system)
        if eid not in emb_groups:
            emb_groups[eid] = (s.emb_system, [], [])
        emb_groups[eid][1].append(idx)
        emb_groups[eid][2].append(s.p_interval)

    # ── Propagate each group in parallel (vmap over p_ivl) ───────────────
    x_hist_all: List[irx.Interval] = [None] * n
    for emb_sys, indices, p_ivls in emb_groups.values():
        p_batch = irx.Interval(
            lower=jnp.stack([p.lower for p in p_ivls]),
            upper=jnp.stack([p.upper for p in p_ivls]),
        )

        def prop_one(p_ivl_single):
            return _propagate_history(
                x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment
            )

        # x_hist_batch: Interval with lower/upper shape (B, num_segments, 9)
        x_hist_batch = jax.vmap(prop_one)(p_batch)

        for local_i, global_i in enumerate(indices):
            x_hist_all[global_i] = irx.Interval(
                lower=x_hist_batch.lower[local_i],
                upper=x_hist_batch.upper[local_i],
            )

    # ── Overlap sum at each segment end, then take the minimum ───────────
    # Double vmap over (pairs, segments) instead of a Python "for i: for j:"
    # loop repeated once per segment (n_pairs x num_segments unrolled calls
    # previously) -- see _pair_indices' docstring. Same math, verified equal
    # (value, JIT, and gradient) against the loop version before this
    # change shipped. This was the dominant cost that made this "unrefined"
    # path slower in practice than admire_refined_sequence_optimizer.py's
    # propagate_with_refinement_admire, despite doing conceptually less
    # work per step (no refinement) -- that module already vmaps its own
    # pairwise computation for exactly this reason (see its docstring).
    lo_stack = jnp.stack([x_hist_all[i].lower[:, 3:6] for i in range(n)])  # (n, num_segments, 3)
    hi_stack = jnp.stack([x_hist_all[i].upper[:, 3:6] for i in range(n)])  # (n, num_segments, 3)
    pair_i, pair_j = _pair_indices(n)
    ivl_i = irx.Interval(lower=lo_stack[pair_i], upper=hi_stack[pair_i])  # (P, num_segments, 3)
    ivl_j = irx.Interval(lower=lo_stack[pair_j], upper=hi_stack[pair_j])  # (P, num_segments, 3)
    pairwise_at_segment = jax.vmap(jax.vmap(overlap_size_log))(ivl_i, ivl_j)  # (P, num_segments)
    segment_overlaps = jnp.sum(pairwise_at_segment, axis=0)  # (num_segments,)
    return jnp.min(segment_overlaps)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Optimizer
# ══════════════════════════════════════════════════════════════════════════════

class SeparatingInputOptimizer:
    """Gradient-descent optimizer for fault-separating control inputs."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl    = x0_ivl
        self.dt        = dt
        self.num_steps = num_steps

        _loss = partial(separation_loss,
                        x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def optimize(self,
                 u_init: Optional[jnp.ndarray] = None,
                 learning_rate: float = 0.05,
                 num_iters: int = 200,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        """Plain gradient descent.  Returns (u_optimal, final_loss)."""
        if u_init is None:
            u_init = jnp.zeros(10)
        u = u_init

        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)

            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss  = float(self.loss_fn(u))
                gnorm = float(jnp.linalg.norm(g))
                print(f"  Iter {i:4d}  loss={loss:.6f}  |g|={gnorm:.4f}")

        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        """Return output intervals, pairwise overlaps, and volumes for *u*."""
        out_ivls = [
            output_interval(
                propagate_scenario(self.x0_ivl, u, s, self.dt, self.num_steps)
            )
            for s in self.scenarios
        ]
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(overlap_size_log(out_ivls[i], out_ivls[j]))

        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, out_ivls)
        }
        return {
            'output_intervals': out_ivls,
            'pairwise_overlaps': overlaps,
            'volumes': volumes,
        }


class MultistepSequenceOptimizer:
    """Container for multistep loss/grad callables and sequence shape."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, steps_per_segment: int, num_segments: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments

        _loss = partial(
            separation_loss_multistep,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            steps_per_segment=steps_per_segment,
        )
        self.loss_fn = jax.jit(_loss)            # (S,10) -> scalar
        self.grad_fn = jax.jit(jax.grad(_loss))  # (S,10) -> (S,10)

    def evaluate(self, u: jnp.ndarray) -> Dict:
        """Return output intervals, pairwise overlaps, and volumes for *u_seq*."""
        if u.ndim != 2 or u.shape != (self.num_segments, 10):
            raise ValueError(
                f"u must have shape ({self.num_segments}, 10), got {tuple(u.shape)}"
            )

        out_ivls = [
            output_interval(
                propagate_scenario_multistep(
                    self.x0_ivl, u, s, self.dt, self.steps_per_segment
                )
            )
            for s in self.scenarios
        ]
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(overlap_size_log(out_ivls[i], out_ivls[j]))

        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, out_ivls)
        }
        return {
            'output_intervals': out_ivls,
            'pairwise_overlaps': overlaps,
            'volumes': volumes,
        }
# ══════════════════════════════════════════════════════════════════════════════
# 5.  Multi-start Optimization
# ══════════════════════════════════════════════════════════════════════════════

def optimize_multistart(opt: SeparatingInputOptimizer,
                        num_restarts: int = 5,
                        learning_rate: float = 0.05,
                        num_iters: int = 200,
                        verbose: bool = False,
                        seed: int = 42,
                        ) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent using a pre-built optimizer.

    Returns (best_u, best_loss, best_stats).
    best_stats includes 'restart_times_s': list of wall-clock seconds per restart.
    """
    best_u, best_loss, best_stats = None, float('inf'), None
    key = jax.random.PRNGKey(seed)
    restart_times: List[float] = []

    for r in range(num_restarts):
        key, subkey = jax.random.split(key)
        u_init = jax.random.uniform(subkey, (10,), minval=-0.1, maxval=0.1)

        if verbose:
            print(f"\n{'─'*60}\nRestart {r+1}/{num_restarts}\n{'─'*60}")

        _t0 = time.perf_counter()
        u_opt, loss = opt.optimize(u_init, learning_rate, num_iters, verbose)
        restart_times.append(time.perf_counter() - _t0)

        stats = opt.evaluate(u_opt)

        if loss < best_loss:
            best_loss, best_u, best_stats = loss, u_opt, stats
            if verbose:
                print(f"  → New best: {loss:.6f}")
        if loss == 0:
            break

    best_stats['restart_times_s'] = restart_times
    return best_u, best_loss, best_stats


def optimize_parallel(
    opt: SeparatingInputOptimizer,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    max_workers: Optional[int] = None,
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[jnp.ndarray, float, Dict]:
    """Parallel multi-start gradient descent using a pre-built optimizer.

    Uses ThreadPoolExecutor for I/O-friendly parallelism.
    Returns (best_u, best_loss, best_stats).
    """
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    subkeys = jax.random.split(jax.random.PRNGKey(seed), num_restarts)

    # Trigger JIT compilation once on the caller thread to avoid compile races.
    _u_warm = jnp.zeros(10)
    _ = opt.loss_fn(_u_warm)
    _ = opt.grad_fn(_u_warm)

    if max_workers is None:
        max_workers = min(num_restarts, max(1, (os.cpu_count() or 1)))
    else:
        max_workers = max(1, min(max_workers, num_restarts))

    restart_times: List[float] = [0.0] * num_restarts
    best_u, best_loss, best_stats = None, float("inf"), None

    def _run_one(restart_idx: int):
        key = subkeys[restart_idx]
        u_init = jax.random.uniform(key, (10,), minval=-0.1, maxval=0.1)
        t0 = time.perf_counter()
        u_opt, loss = opt.optimize(u_init, learning_rate, num_iters, verbose=False)
        elapsed = time.perf_counter() - t0
        stats = opt.evaluate(u_opt)
        return restart_idx, u_opt, float(loss), stats, elapsed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, r): r
            for r in range(num_restarts)
        }
        for fut in as_completed(futures):
            r, u_opt, loss, stats, elapsed = fut.result()
            restart_times[r] = elapsed

            if verbose:
                print(f"Restart {r+1}/{num_restarts}: loss={loss:.6f} time={elapsed:.3f}s")

            if loss < best_loss:
                best_u, best_loss, best_stats = u_opt, loss, stats
                if verbose:
                    print(f"  -> New best from restart {r+1}: {best_loss:.6f}")

    best_stats['restart_times_s'] = restart_times
    return best_u, best_loss, best_stats


def optimize_parallel_gpu(
    opt: SeparatingInputOptimizer,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
):
    """GPU-parallel multi-start optimization for a single constant input.

    Batches all restarts via vmap and runs a single fori_loop on device.
    Returns: best_u, best_loss, all_u_final (R,10), all_losses (R,)
    """
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.uniform(key, (num_restarts, 10), minval=-0.1, maxval=0.1)

    batched_loss = jax.vmap(opt.loss_fn)   # (R,10) -> (R,)
    batched_grad = jax.vmap(opt.grad_fn)   # (R,10) -> (R,10)

    def body(_, u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses  = batched_loss(u_final)

    best_idx  = jnp.argmin(losses)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_parallel_gpu_rejit(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
):
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.uniform(key, (num_restarts, 10), minval=-0.1, maxval=0.1)

    def loss_fn_single(u):
        return separation_loss(
            u=u,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            num_steps=num_steps,
        )

    grad_fn_single = jax.grad(loss_fn_single)

    batched_loss = jax.vmap(loss_fn_single)   # (R,10) -> (R,)
    batched_grad = jax.vmap(grad_fn_single)   # (R,10) -> (R,10)

    def body(_, u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses  = batched_loss(u_final)

    best_idx  = jnp.argmin(losses)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses
    # """Convenience wrapper: build optimizer then call optimize_parallel_gpu."""
    # return optimize_parallel_gpu(
    #     SeparatingInputOptimizer(
    #         scenarios=scenarios,
    #         x0_ivl=x0_ivl,
    #         dt=dt,
    #         num_steps=num_steps,
    #     ),
    #     num_restarts=num_restarts,
    #     learning_rate=learning_rate,
    #     num_iters=num_iters,
    #     seed=seed,
    # )

def optimize_multistep_gpu_rejit(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    steps_per_segment: int,
    num_segments: int,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    num_restarts: int = 50,
    verbose: bool = False,
    seed: int = 42,
):
    """GPU-parallel multi-start optimization for control sequences.

    Returns: best_u_seq (S,10), best_loss, all_u_seq_final (R,S,10), all_losses (R,)
    """
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.uniform(
        key, (num_restarts, num_segments, 10), minval=-0.05, maxval=0.05
    )

    def loss_fn_multistep(u):
        return separation_loss_multistep(
            u_seq=u,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            steps_per_segment=steps_per_segment,
        )

    grad_fn_multistep = jax.grad(loss_fn_multistep)

    batched_loss = jax.vmap(loss_fn_multistep)   # (R,S,10) -> (R,)
    batched_grad = jax.vmap(grad_fn_multistep)   # (R,S,10) -> (R,S,10)

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses  = batched_loss(u_final)

    best_idx  = jnp.argmin(losses)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_multistep_gpu_fused(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    steps_per_segment: int,
    num_segments: int,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    num_restarts: int = 50,
    seed: int = 42,
    init_scale: float = 0.05,
    u0_mean: jnp.ndarray = None,
):
    """Same as optimize_multistep_gpu_rejit but fuses batched_loss/batched_grad
    into one jax.vmap(jax.value_and_grad(...)) call instead of separate
    tracings -- jax.grad already reruns the forward pass internally, so
    building batched_loss=vmap(loss_fn) and batched_grad=vmap(grad(loss_fn))
    separately duplicates that forward-graph compile (same fix applied to
    faulty_car_output_feedback_cbf.py's CBF optimizer in that module's
    history). separation_loss_multistep itself was already vmapped over
    scenarios (see its docstring), so this only fixes the outer optimizer's
    redundant tracing, not a Python-loop-over-scenarios issue.

    Caller should wrap this in an outer jax.jit before timing/deploying it
    (matches this project's established "always jit-wrap the whole
    optimize_*_gpu call" finding -- calling this class of optimizer eagerly
    risks a severe host-memory blowup).

    `init_scale`: half-width of the uniform u0 sampling range (was hardcoded
    to 0.05 in optimize_multistep_gpu_rejit). _project_u's actual clip bound
    is +-0.5 rad -- 10x wider than the original 0.05 init range. Found (this
    session, mirroring faulty_car_output_feedback_cbf.py's init_std finding)
    that this narrow default leaves GD unable to move the loss meaningfully
    (0.0534 at 40 iters -> 0.0502 at 200 iters, ~6% improvement for 5x more
    iterations) -- widen this to let restarts actually explore the feasible
    control range instead of relying on GD to discover it from a narrow
    near-zero start.

    `u0_mean`: optional (num_segments, 10) array. If given, restarts are
    sampled as u0_mean + U(-init_scale, init_scale) instead of centered at
    zero -- lets a caller warm-start from an informed/"smart" guess (see
    admire_staged_opt_minimal.ipynb, which hand-designs an initial guess
    that specifically excites the historically-hardest-to-separate channel
    and converges in ~10 GD steps from a single trajectory, no random
    restarts at all -- num_restarts/num_iters/init_scale can all be cut
    drastically once the search starts from a good point instead of blind
    random exploration).

    Returns: best_u_seq (S,10), best_loss, all_u_seq_final (R,S,10), all_losses (R,)
    """
    key = jax.random.PRNGKey(seed)
    noise = jax.random.uniform(
        key, (num_restarts, num_segments, 10), minval=-init_scale, maxval=init_scale
    )
    u0 = noise if u0_mean is None else noise + u0_mean[None, :, :]

    def loss_fn(u):
        return separation_loss_multistep(
            u_seq=u, x0_ivl=x0_ivl, scenarios=scenarios,
            dt=dt, steps_per_segment=steps_per_segment,
        )

    batched_value_and_grad = jax.vmap(jax.value_and_grad(loss_fn))

    def body(_, carry):
        u_batch, _prev_losses = carry
        losses, g = batched_value_and_grad(u_batch)
        return (_project_u(u_batch - learning_rate * g), losses)

    init_losses = jnp.zeros(num_restarts)
    u_final, losses = jax.lax.fori_loop(0, num_iters, body, (u0, init_losses))
    # Re-evaluate at u_final so `losses` isn't one-iteration stale (the
    # fori_loop's carried `losses` reflect the PRE-update u of the final
    # iteration) -- matches optimize_multistep_gpu_rejit's semantics of
    # returning batched_loss(u_final).
    losses, _ = batched_value_and_grad(u_final)

    # NaN-safe selection (a restart can go NaN, e.g. from a too-large dt --
    # see multistep_unrefined_demo.py's history -- and plain jnp.argmin does
    # NOT reliably skip NaN entries, so an un-guarded argmin can silently
    # return a NaN "best" even when better non-NaN restarts exist). Matches
    # optimize_refined_sequence_gpu's existing NaN-guard convention.
    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx  = jnp.argmin(losses_valid)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_multistep_gpu(
    opt: MultistepSequenceOptimizer,
    num_restarts: int = 50,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    seed: int = 42,
):
    """GPU-parallel multi-start optimization for control sequences.

    Returns: best_u_seq (S,10), best_loss, all_u_seq_final (R,S,10), all_losses (R,)
    """
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.uniform(
        key, (num_restarts, opt.num_segments, 10), minval=-0.05, maxval=0.05
    )

    batched_loss = jax.vmap(opt.loss_fn)   # (R,S,10) -> (R,)
    batched_grad = jax.vmap(opt.grad_fn)   # (R,S,10) -> (R,S,10)

    def body(_, u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = jax.lax.fori_loop(0, num_iters, body, u0)
    losses  = batched_loss(u_final)

    best_idx  = jnp.argmin(losses)
    best_u    = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses


def optimize_multistep(
    scenarios: List[Scenario],
    x0_ivl: irx.Interval,
    dt: float,
    steps_per_segment: int,
    num_segments: int,
    learning_rate: float = 0.05,
    num_iters: int = 200,
    num_restarts: int = 50,
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent over a sequence of control inputs.

    Optimises u_seq of shape (num_segments, 10), where u_seq[k] is applied
    to all scenarios for *steps_per_segment* Euler steps before switching to
    u_seq[k+1].  Total horizon = num_segments × steps_per_segment × dt s.

    Uses separation_loss_multistep as the objective (minimum overlap over all
    segment ends).

    Returns
    -------
    (u_seq_opt, loss_opt, stats)
    u_seq_opt : (num_segments, 10) optimal control sequence
    loss_opt  : final overlap loss (rad³)
    stats     : dict with 'output_intervals', 'pairwise_overlaps',
                'volumes', 'optimization_time_s', 'best_restart_idx',
                'all_restart_losses', 'num_restarts'
    """
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    ms_opt = MultistepSequenceOptimizer(
        scenarios=scenarios,
        x0_ivl=x0_ivl,
        dt=dt,
        steps_per_segment=steps_per_segment,
        num_segments=num_segments,
    )

    _t0 = time.perf_counter()
    u_seq, loss_opt_jax, _, final_losses = optimize_multistep_gpu(
        opt=ms_opt,
        num_restarts=num_restarts,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=seed,
    )
    elapsed = time.perf_counter() - _t0

    best_idx = int(jnp.argmin(final_losses))
    loss_opt = float(loss_opt_jax)

    if verbose:
        mean_loss = float(jnp.mean(final_losses))
        print(
            f"Multistep GPU multistart complete: best_loss={loss_opt:.6f}  "
            f"mean_final_loss={mean_loss:.6f}  best_restart={best_idx+1}/{num_restarts}"
        )

    # Build stats dict (same schema as SeparatingInputOptimizer.evaluate).
    out_ivls = [
        output_interval(
            propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment)
        )
        for s in scenarios
    ]
    n = len(scenarios)
    overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            key_ij = f"{scenarios[i].name} vs {scenarios[j].name}"
            overlaps[key_ij] = float(overlap_size_log(out_ivls[i], out_ivls[j]))
    volumes = {
        s.name: float(jnp.prod(iv.upper - iv.lower))
        for s, iv in zip(scenarios, out_ivls)
    }
    stats = {
        'output_intervals':    out_ivls,
        'pairwise_overlaps':   overlaps,
        'volumes':             volumes,
        'optimization_time_s': elapsed,
        'best_restart_idx':    best_idx,
        'all_restart_losses':  np.array(final_losses),
        'num_restarts':        num_restarts,
    }
    return u_seq, loss_opt, stats


# ══════════════════════════════════════════════════════════════════════════════
# Utility plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_interval_history(
    history: Dict[str, Any],
    dim_names: List[str],
    initial_interval: Optional[irx.Interval] = None,
    color_map: Optional[Dict[str, str]] = None,
    save_to_pdf: bool = False,
    pdf_pages: Optional[PdfPages] = None
) -> None:
    """
    Generates a 3D plot of state interval history, starting from a given initial interval.
    The history is rendered as a smooth, faint tube, and the final state
    is shown as a more solid, concrete box.

    Args:
        history: The output dictionary from the simulation function.
        dim_names: A list of names for each state dimension for axis labeling.
        initial_interval: (Optional) A specific interval to use as the starting
                          point for all history tubes.
        color_map: (Optional) A dictionary mapping scenario names to colors.
        save_to_pdf: If True, saves the plot to a PDF file.
        pdf_pages: (Optional) An existing PdfPages object to add the plot to.
    """
    # local imports to avoid adding global dependencies
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Rectangle

    # --- 1. Setup and Color Map Generation (Unchanged) ---
    if color_map is None:
        scenarios = list(history.keys())
        fault_names = [name for name in scenarios if name != "Nominal"]
        cmap = plt.cm.get_cmap('viridis', len(fault_names))
        generated_colors = [cmap(i) for i in range(len(fault_names))]
        color_map = {name: color for name, color in zip(fault_names, generated_colors)}
        if "Nominal" in scenarios: color_map["Nominal"] = "blue"

    print("Generating smooth 3D history plot for dimensions 4, 5, and 6...")
    scenarios = list(history.keys())

    # --- 2. Axis Limit Calculation (MODIFIED to include initial_interval) ---
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    # Include the initial interval in the bounds calculation if it exists
    if initial_interval:
        min_x = min(min_x, initial_interval.lower[3]); max_x = max(max_x, initial_interval.upper[3])
        min_y = min(min_y, initial_interval.lower[4]); max_y = max(max_y, initial_interval.upper[4])
        min_z = min(min_z, initial_interval.lower[5]); max_z = max(max_z, initial_interval.upper[5])
    
    # --- helper to unify state sequence ---
    def _state_sequence(states):
        # states may be a list of irx.Interval or a single Interval with
        # leading segment dimension.  Return a list of Interval objects.
        if isinstance(states, irx.Interval):
            n = states.lower.shape[0]
            return [
                irx.Interval(lower=states.lower[i], upper=states.upper[i])
                for i in range(n)
            ]
        else:
            return states

    # Include all history points in the bounds calculation
    for name in scenarios:
        for ivl in _state_sequence(history[name]["states"]):
            min_x = min(min_x, ivl.lower[3]); max_x = max(max_x, ivl.upper[3])
            min_y = min(min_y, ivl.lower[4]); max_y = max(max_y, ivl.upper[4])
            min_z = min(min_z, ivl.lower[5]); max_z = max(max_z, ivl.upper[5])

    def get_padded_limits(min_val, max_val, padding_factor=0.05):
        range_val = max_val - min_val
        if range_val == 0: range_val = abs(max_val) * 0.1 or 0.1
        padding = range_val * padding_factor
        return [min_val - padding, max_val + padding]

    x_lims = get_padded_limits(min_x, max_x)
    y_lims = get_padded_limits(min_y, max_y)
    z_lims = get_padded_limits(min_z, max_z)
    
    # --- 3. Plotting Setup and Helper Functions (Unchanged) ---
    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    def get_cuboid_vertices(ivl):
        x0, x1 = ivl.lower[3], ivl.upper[3]
        y0, y1 = ivl.lower[4], ivl.upper[4]
        z0, z1 = ivl.lower[5], ivl.upper[5]
        return [(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0),
                (x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1)]

    def create_cuboid_faces(vertices):
        return [[vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]], [vertices[1], vertices[2], vertices[6], vertices[5]]]

    def create_tube_segment_faces(v1, v2):
        return [[v1[0],v1[1],v2[1],v2[0]], [v1[4],v1[5],v2[5],v2[4]], [v1[0],v1[4],v2[4],v2[0]],
                [v1[3],v1[7],v2[7],v2[3]], [v1[1],v1[5],v2[5],v2[1]], [v1[2],v1[6],v2[6],v2[2]]]

    ####################################################################
    ### SECTION 4: MODIFIED - PLOT TUBES FROM INITIAL INTERVAL       ###
    ####################################################################
    for name in scenarios:
        all_states = _state_sequence(history[name]["states"])
        
        # Define the full path for the tube, starting with the initial interval if provided
        tube_path = ([initial_interval] + all_states) if initial_interval else all_states
        if len(tube_path) < 2:
            # If there's no path to draw a tube, just plot the final state if it exists
            if all_states:
                final_ivl = all_states[-1]
                final_verts = get_cuboid_vertices(final_ivl)
                final_box = Poly3DCollection(create_cuboid_faces(final_verts), facecolors=color_map[name],
                                             linewidths=1, edgecolors='k', alpha=0.5)
                ax_3d.add_collection3d(final_box)
            continue

        # --- Part A: Plot the history as a faint, smooth tube ---
        tube_faces = []
        # Add a "start cap" for the tube using the first interval in our path
        start_cap_verts = get_cuboid_vertices(tube_path[0])
        tube_faces.extend(create_cuboid_faces(start_cap_verts))

        # Create the smooth tube segments connecting consecutive intervals along the path
        for i in range(len(tube_path) - 1):
            v1 = get_cuboid_vertices(tube_path[i])
            v2 = get_cuboid_vertices(tube_path[i+1])
            tube_faces.extend(create_tube_segment_faces(v1, v2))

        tube_collection = Poly3DCollection(
            tube_faces, facecolors=color_map[name], linewidths=0, alpha=0.15
        )
        ax_3d.add_collection3d(tube_collection)

        # --- Part B: Plot the actual final interval as a more solid box ---
        # This is always the last state from the history, not from the tube_path
        if all_states:
            final_ivl = all_states[-1]
            final_verts = get_cuboid_vertices(final_ivl)
            final_box_collection = Poly3DCollection(
                create_cuboid_faces(final_verts), facecolors=color_map[name],
                linewidths=1, edgecolors='k', alpha=0.5
            )
            ax_3d.add_collection3d(final_box_collection)
        
    # --- 5. Formatting, Limits, and Labels (MODIFIED z-label pad) ---
    TITLE_FONTSIZE = 32; LABEL_FONTSIZE = 16; TICK_FONTSIZE = 16
    ax_3d.set_xlim(x_lims); ax_3d.set_ylim(y_lims); ax_3d.set_zlim(z_lims)
    ax_3d.set_xlabel(dim_names[3], fontsize=LABEL_FONTSIZE, labelpad=15)
    ax_3d.set_ylabel(dim_names[4], fontsize=LABEL_FONTSIZE, labelpad=15)
    # Increase labelpad for the z-axis to prevent it from being clipped
    ax_3d.set_zlabel(dim_names[5], fontsize=LABEL_FONTSIZE, labelpad=50)
    ax_3d.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax_3d.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax_3d.tick_params(axis='z', labelsize=TICK_FONTSIZE)
    legend_patches = [Rectangle((0, 0), 1, 1, fc=color_map[name], alpha=0.6, label=name) for name in scenarios]
    ax_3d.legend(handles=legend_patches, fontsize=TICK_FONTSIZE, loc='upper left', bbox_to_anchor=(-0.1, 0.95))
    ax_3d.grid(True)
    plt.tight_layout()

    # --- 6. Save/Display Logic (Unchanged) ---
    if save_to_pdf:
        create_own_pdf = pdf_pages is None
        if create_own_pdf:
            default_filename = "3d_smooth_history_plot.pdf"
            print(f"No PdfPages object provided. Saving to new file: {default_filename}")
            pdf_pages = PdfPages(default_filename)
        pdf_pages.savefig(fig_3d, bbox_inches='tight', pad_inches=0.)
        if create_own_pdf:
            pdf_pages.close()
            print("PDF generation complete.")
    else:
        plt.show()
    plt.close(fig_3d)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ADMIRE SEPARATING INPUT — immrax")
    print("=" * 70)

    # ── Scenarios ──────────────────────────────────────────────────────────
    scenarios = create_scenarios()
    print(f"\n{len(scenarios)} fault scenarios:")
    for s in scenarios:
        print(f"  • {s.name}")

    # ── Initial state interval ─────────────────────────────────────────────
    x0_nom = jnp.zeros(9)
    x0_nom = x0_nom.at[0].set(343.0 * 0.3)   # Vt ≈ 102.9 m/s
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(9) * 0.01)

    print(f"\nInitial state interval (centre ± 0.01):")
    print(f"  Vt   ∈ [{float(x0_ivl.lower[0]):.3f}, {float(x0_ivl.upper[0]):.3f}] m/s")
    print(f"  α    ∈ [{float(x0_ivl.lower[1]):.4f}, {float(x0_ivl.upper[1]):.4f}] rad")
    print(f"  β    ∈ [{float(x0_ivl.lower[2]):.4f}, {float(x0_ivl.upper[2]):.4f}] rad")

    # ── Optimization parameters ────────────────────────────────────────────
    dt, num_steps = 0.1, 20
    print(f"\nPropagation: {num_steps} × {dt} s = {num_steps * dt} s total")
    print("Separation objective: angular rates (pb, qb, rb)")

    # ── Run GPU-parallel optimization ──────────────────────────────────────
    print("\nRunning GPU-parallel multi-start optimization …\n")
    opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt, num_steps)

    # Warm up JIT before timing
    _ = opt.loss_fn(jnp.zeros(10))

    _t0 = time.perf_counter()
    best_u, best_loss, _, all_losses = optimize_parallel_gpu(
        opt,
        num_restarts=20,
        learning_rate=0.05,
        num_iters=200,
    )
    elapsed = time.perf_counter() - _t0
    stats = opt.evaluate(best_u)

    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOptimization time: {elapsed:.2f} s ({20} restarts × {200} iters)")
    print(f"\nOptimal separating input (rad):")
    surface_labels = ["rc", "lc", "roe", "rie", "lie", "loe", "rud", "flap", "yaw_tv", "pitch_tv"]
    for label, val in zip(surface_labels, best_u):
        print(f"  {label:8s} = {float(val):+.4f}")
    print(f"\nTotal overlap: {best_loss:.6f} rad³")

    total_pairs = sum(1 for v in stats['pairwise_overlaps'].values() if v > 0)
    print(f"Non-zero overlapping pairs: {total_pairs}")

    print(f"\nAngular-rate interval volumes (rad³):")
    for k, v in stats['volumes'].items():
        print(f"  {k}: {v:.2e}")

    print("\n✓ Done")
