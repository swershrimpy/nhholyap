"""
Chain-of-Integrator Crazyflie Outer Loop -- Controller Discrimination (Phase 1)
=================================================================================
Phase 1 of the adaptive_spoofing controller-discrimination pipeline (see
PLAN.md Sec 5-6 for the full design and the decisions this implements).

This module answers "implement four controlled crazyflies under the four
different controllers" and "design a spoofing signal that can discriminate
which controller is being used" at the FLAT-OUTPUT CHAIN layer QPS's own
`experiments/RQ3/rq3_model_bank.py` / `qps_surrogate.py` operate on -- not
the rigid-body plant (`crazyflie_12d.py`'s `CrazyflieSystem`), per PLAN.md
Sec 6 Q1. `CrazyflieSystem` is untouched by this module; it remains the
downstream validation/rendering target for Phase 3.

System
------
  State   x = [p(3), v(3), a(3), j(3), integ_e_p(3), cmd(3)]   (18 states)
            position / velocity / acceleration / jerk (QPS's own
            get_estimated_states() layout, PLUS two auxiliary states this
            module adds -- see "Unified controller law" below), all in the
            same frame as QPS's chain (`AA,bb = gen_chain_of_integrators()`).
  Control u = b   (3,)   attacker's position-spoof bias, added to the
            position the CONTROLLER reads (not the true state) -- this is
            exactly QPS's own `spoof_bias` hook (`quadrotarium.py` Sec 4
            "RQ3 spoof hook") and RQ3's attacker model (`qps_surrogate.py`).
            This is the design variable the separating-input optimizer below
            searches over -- the "spoofing signal".
  Params  p = [k_p, k_v, k_a, k_i, k_j, base_flag]   (6,)   the MASKED
            controller-identity gains (PLAN.md Sec 6 Q2) -- "which
            controller" is which fixed (k_p,...,k_j,base_flag) point
            interval a Scenario carries, reusing the SAME emb_system/traced
            f across all four the way every other fault-diagnosis module in
            this repo reuses one emb_system across scenarios.

Unified controller law (see PLAN.md Sec 2-4 R2/R3 for the derivation)
-----------------------------------------------------------------------
`rq3_model_bank.py`'s four candidates are near-unifiable: three
(`qps_snap_chain`, `pd_pos_vel`, `pid_pos_vel_i`) are all
"snap = feedforward - theta . tracking_error" state feedback with a
progressively smaller theta mask; the fourth (`indi_jerk`) is structurally
different -- an INCREMENTAL correction to the PREVIOUS commanded snap,
`snap_k = snap_{k-1} - k_j*e_j`, not a function of the reference at all.

To keep all four inside ONE traced continuous-time system (so the existing
vmap-over-stacked-params machinery applies unchanged), this module:
  - keeps `integ_e_p` as a literal integral state, `d(integ)/dt = e_p`
    (the continuous analog of `pid_pos_vel_i`'s discrete `cumsum(e_p)*dt`);
  - adds `cmd`, a "commanded-snap" auxiliary state used ONLY by `indi_jerk`,
    with `d(cmd)/dt = -k_j * e_j` -- the continuous relaxation of "correct
    the previous commanded snap by the jerk error every step" (still
    evolves for every scenario, but is only READ into j_dot when
    base_flag=1, so it's inert for the other three);
  - blends the two mechanisms with `base_flag in {0,1}` (a plain multiply,
    NOT a jnp.where/comparison -- comparisons on Interval-valued params are
    not meaningful under interval arithmetic, see PLAN.md R2):

      target = s_ref - k_p*e_p - k_v*e_v - k_a*e_a - k_i*integ - k_j*e_j
      j_dot  = (1 - base_flag) * target + base_flag * cmd

  This is a genuine modeling choice for `indi_jerk` (approximating its
  discrete incremental law as a continuous ODE) rather than a literal
  transcription -- flagged here explicitly since `rq3_model_bank.py`'s
  version is exactly discrete. The other three candidates are NOT
  approximated: `target` above is their exact continuous-time law (this is
  literally `nominal_snap_input_u`'s equation 12 for `qps_snap_chain`).

Why continuous evolution (not discrete)
-----------------------------------------
Every other System in this repo uses `evolution='continuous'` +
`irx.natemb` + a `euler_step` wrapper (`dx*dt + x`). The chain dynamics here
are pure linear/polynomial -- no trig, no division -- so there is no
compile-cost reason to deviate (the compile-cost issue documented for
`quadrotor_fault_diagnosis`/`crazyflie_12d.py` is specifically from
`tan`/`1/cos` terms in the RIGID-BODY Euler kinematics, absent here
entirely). Matching the established convention also means every existing
helper pattern (`euler_step`, vmap-over-scenarios, vmap-over-pairs
refinement) transfers with no changes beyond swapping the System in.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import immrax as irx

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
QPS_DT = 0.02          # QPS's own simulation step (quadrotarium.py)

# Default hover reference: [p_ref(3), v_ref(3), a_ref(3), j_ref(3), s_ref(3)],
# all derivatives zero except a hover position setpoint -- matches
# qps_surrogate.py's hover_reference() / QPS's HOVERING mission state.
DEFAULT_HOVER_POS = jnp.array([0.0, 0.0, 1.0])


def hover_reference(pos_sp: jnp.ndarray = DEFAULT_HOVER_POS) -> jnp.ndarray:
    """A static 15-vector reference: [pos_sp, 0, 0, 0, 0]."""
    return jnp.concatenate([jnp.asarray(pos_sp, dtype=jnp.float32), jnp.zeros(12)])


# Default spoof-bias box (attacker's design-variable bounds). No CBF/killswitch
# is modeled here (PLAN.md R6) -- this box is the stand-in safety envelope, sized
# well under QPS's `rq3_killswitch.py` default `bias_lim` conventions.
_BIAS_LIM = 0.3   # meters, per axis
_U_LO = jnp.full(3, -_BIAS_LIM)
_U_HI = jnp.full(3, _BIAS_LIM)

_UNROLL_THRESHOLD = 64


def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(u, _U_LO, _U_HI)


# ══════════════════════════════════════════════════════════════════════════════
# 1. System definition
# ══════════════════════════════════════════════════════════════════════════════

class ChainControllerSystem(irx.System):
    """18-state chain-of-integrator outer loop with a masked-theta controller
    law selecting which of the four `rq3_model_bank.py` candidates is active.

    State   x = [p(3), v(3), a(3), j(3), integ_e_p(3), cmd(3)]
    Control u = b(3)      attacker's position-spoof bias
    Params  p = [k_p, k_v, k_a, k_i, k_j, base_flag]   (6,), see module
              docstring "Unified controller law".

    The reference trajectory (the victim's mission setpoint) is baked into
    the instance, not passed through f -- it is exogenous "environment"
    data, not something the attacker or the controller-identity parameter
    controls, matching how CrazyflieSystem bakes in physical constants.
    """

    def __init__(self, ref15: jnp.ndarray = None):
        self.evolution = 'continuous'
        self.xlen = 18
        self.ref15 = hover_reference() if ref15 is None else jnp.asarray(ref15)

    def f(self, t, x, u, p):
        b = u
        k_p, k_v, k_a, k_i, k_j, base_flag = p[0], p[1], p[2], p[3], p[4], p[5]

        pos, vel, acc, jerk = x[0:3], x[3:6], x[6:9], x[9:12]
        integ, cmd = x[12:15], x[15:18]

        rp, rv, ra, rj, rs = (self.ref15[0:3], self.ref15[3:6], self.ref15[6:9],
                              self.ref15[9:12], self.ref15[12:15])

        e_p = (pos + b) - rp
        e_v = vel - rv
        e_a = acc - ra
        e_j = jerk - rj

        target = rs - k_p * e_p - k_v * e_v - k_a * e_a - k_i * integ - k_j * e_j
        j_dot = (1.0 - base_flag) * target + base_flag * cmd

        p_dot = vel
        v_dot = acc
        a_dot = jerk
        integ_dot = e_p
        cmd_dot = -k_j * e_j

        return jnp.concatenate([p_dot, v_dot, a_dot, j_dot, integ_dot, cmd_dot])


# Module-level cache: one ChainControllerSystem + one irx.natemb(...) per
# distinct reference (compared by value, like the other modules' physical
# params) -- mirrors get_system_and_embedding elsewhere in this repo.
_EMB_CACHE: Dict[Tuple[float, ...], Tuple[ChainControllerSystem, object]] = {}


def get_system_and_embedding(ref15: jnp.ndarray = None) -> Tuple[ChainControllerSystem, object]:
    ref15 = hover_reference() if ref15 is None else jnp.asarray(ref15)
    key = tuple(np.asarray(ref15).tolist())
    if key not in _EMB_CACHE:
        sys_ = ChainControllerSystem(ref15)
        emb = irx.natemb(sys_)
        _EMB_CACHE[key] = (sys_, emb)
    return _EMB_CACHE[key]


def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    """One forward-Euler interval step via the natural embedding -- identical
    pattern to every other module in this repo (`t` must be a JAX array)."""
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)


# ══════════════════════════════════════════════════════════════════════════════
# Observability: indices 0:12 ([p,v,a,j], QPS's get_estimated_states() layout)
# are what an outside observer (or attacker) can actually see -- true position/
# velocity/acceleration/jerk, reconstructed from tracking a real drone.
# Indices 12:18 (integ_e_p, cmd) are CONTROLLER-INTERNAL memory -- a real
# observer has no access to the victim's own integrator/INDI state. Every
# discrimination signal below (separation loss, refinement, online
# monitoring) is computed on this OBSERVED projection, not the full state --
# "output-anticipating", not "full-state-anticipating". Hidden dims still
# evolve and get vmapped/propagated (needed for the dynamics to be correct),
# they just never enter a cost, an overlap check, or a refinement.
# ══════════════════════════════════════════════════════════════════════════════

def observed_output(x_ivl: irx.Interval) -> irx.Interval:
    """Project a full 18-dim state interval down to the 12 observable dims."""
    return irx.Interval(lower=x_ivl.lower[:12], upper=x_ivl.upper[:12])


def _output_overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    return _overlap_volume(observed_output(ivl1), observed_output(ivl2))


# ══════════════════════════════════════════════════════════════════════════════
# 2. The four controller scenarios (masked theta -- see rq3_model_bank.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """One controller hypothesis: shared dynamics structure (ChainControllerSystem),
    differing only in the masked-theta parameter interval."""
    name: str
    emb_system: object
    p_interval: irx.Interval   # [k_p,k_v,k_a,k_i,k_j,base_flag], shape (6,)


# theta = [k_p, k_v, k_a, k_i, k_j], base_flag separate.
# qps_snap_chain's gains are QPS's REAL, exact gains (rq3_model_bank.py
# theta_true=(1680,1066,251,26), poles -5..-8). The other three candidates
# have no numeric gains published anywhere in rq3_model_bank.py (only their
# STRUCTURE is defined there) -- these are illustrative placeholders and are
# flagged as an open item in PLAN.md; tune against real identified data
# before using this for anything beyond a Phase-1 mechanism demo.
_CANDIDATE_THETA = {
    "qps_snap_chain": (1680.0, 1066.0, 251.0, 0.0, 26.0, 0.0),
    "pd_pos_vel":     (400.0, 200.0, 0.0, 0.0, 0.0, 0.0),
    "pid_pos_vel_i":  (400.0, 200.0, 0.0, 50.0, 0.0, 0.0),
    "indi_jerk":      (0.0, 0.0, 0.0, 0.0, 26.0, 1.0),
}


def create_scenarios(ref15: jnp.ndarray = None,
                     names: Optional[List[str]] = None) -> List[Scenario]:
    """Return the four controller scenarios (or a named subset), sharing one
    traced emb_system -- so `_propagate_all_scenarios`'s vmap-over-stacked-p
    pattern applies unchanged."""
    _, emb = get_system_and_embedding(ref15)
    want = list(_CANDIDATE_THETA.keys()) if names is None else names
    scenarios = []
    for name in want:
        theta = jnp.array(_CANDIDATE_THETA[name])
        scenarios.append(Scenario(name, emb, irx.Interval(lower=theta, upper=theta)))
    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# 3. Single-step separating spoof-bias optimizer
# ══════════════════════════════════════════════════════════════════════════════

def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    """Branchless pairwise axis-aligned-box overlap volume (ported verbatim
    from quadrotor_fault_diagnosis/quadrotor_separating_input.py)."""
    widths = jnp.maximum(
        jnp.minimum(ivl1.upper, ivl2.upper) - jnp.maximum(ivl1.lower, ivl2.lower),
        0.0,
    )
    return jnp.prod(widths)


def _run_unrolled_or_loop(step_fn, init, n: int, unroll_threshold: int = _UNROLL_THRESHOLD):
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
    def body(carry, i):
        return step_fn(carry, i), None
    carry, _ = jax.lax.scan(body, init, xs=jnp.arange(n))
    return carry


def _propagate_by_params(x0_ivl: irx.Interval, u: jnp.ndarray, emb_sys,
                         p_ivl: irx.Interval, dt: float, num_steps: int) -> irx.Interval:
    def step(x_carry, _i):
        return euler_step(emb_sys, x_carry, u, p_ivl, dt)
    return _run_unrolled_or_loop(step, x0_ivl, num_steps)


def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray, scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    return _propagate_by_params(x0_ivl, u, scenario.emb_system, scenario.p_interval, dt, num_steps)


def _propagate_all_scenarios(x0_ivl: irx.Interval, u: jnp.ndarray,
                             scenarios: List[Scenario], dt: float, num_steps: int) -> List[irx.Interval]:
    """vmap over stacked p_intervals -- all four scenarios share one emb_system
    (PLAN.md Sec 6 Q2), so this is the same trick every other module uses."""
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x_final_batch = jax.vmap(lambda p: _propagate_by_params(x0_ivl, u, emb_sys, p, dt, num_steps))(p_batch)
    return [
        irx.Interval(lower=x_final_batch.lower[i], upper=x_final_batch.upper[i])
        for i in range(len(scenarios))
    ]


def separation_loss(u: jnp.ndarray, x0_ivl: irx.Interval, scenarios: List[Scenario],
                    dt: float, num_steps: int) -> jnp.ndarray:
    """Sum of pairwise OBSERVED-OUTPUT interval overlaps (C(4,2)=6 pairs).
    Minimising this over the spoof bias `u` maximises reachable-OUTPUT
    separation across the four controller hypotheses -- the discrimination
    signal. Deliberately projects to `observed_output` rather than using the
    full 18-dim state: the hidden controller-memory dims (integ, cmd) can
    diverge freely without that divergence being detectable by anything that
    only watches the drone fly, so optimizing full-state separation would
    overstate what's actually discriminable."""
    x_ivls = _propagate_all_scenarios(x0_ivl, u, scenarios, dt, num_steps)
    n = len(x_ivls)
    total = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            total = total + _output_overlap_volume(x_ivls[i], x_ivls[j])
    return total


class SeparatingInputOptimizer:
    """Gradient-descent optimizer for a controller-discriminating spoof bias.
    Structurally identical to the fault-diagnosis modules' optimizer, with
    the control input `u` reinterpreted as the attacker's position bias."""

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.num_steps = num_steps

        _loss = partial(separation_loss, x0_ivl=x0_ivl, scenarios=scenarios, dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def optimize(self, u_init: Optional[jnp.ndarray] = None, learning_rate: float = 0.01,
                num_iters: int = 150, verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        u = jnp.zeros(3) if u_init is None else u_init
        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)
            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = float(self.loss_fn(u))
                print(f"  Iter {i:4d}  loss={loss:.6f}  u={np.array(u).round(4)}  |g|={float(jnp.linalg.norm(g)):.4f}")
        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        x_ivls = _propagate_all_scenarios(self.x0_ivl, u, self.scenarios, self.dt, self.num_steps)
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_output_overlap_volume(x_ivls[i], x_ivls[j]))
        volumes = {s.name: float(jnp.prod(observed_output(iv).upper - observed_output(iv).lower))
                  for s, iv in zip(self.scenarios, x_ivls)}
        return {'state_intervals': x_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes}


def optimize_parallel_gpu(opt: 'SeparatingInputOptimizer', num_restarts: int = 100,
                          learning_rate: float = 0.01, num_iters: int = 150, seed: int = 42):
    """GPU-parallel multi-start gradient descent for a constant spoof bias."""
    key = jax.random.PRNGKey(seed)
    noise_scale = _BIAS_LIM * 0.5
    u0 = jax.random.normal(key, (num_restarts, 3)) * noise_scale

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u, _i):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


# ══════════════════════════════════════════════════════════════════════════════
# 4. Multistep ("unrefined") spoof-bias sequence
# ══════════════════════════════════════════════════════════════════════════════
# A sequence of biases (one per segment) instead of one constant bias -- lets
# the optimizer exploit the finding from PLAN.md Sec 7: pd_pos_vel and
# pid_pos_vel_i are IDENTICAL for one step and only separate once the
# integral term has had several steps to accumulate, so a single-step spoof
# provably cannot discriminate that pair. Mirrors
# quadrotor_fault_diagnosis/quadrotor_separating_input.py Section 4, with
# the pairwise cost computed on `observed_output`, not full state (see
# Section 3's `separation_loss` docstring for why).

def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray, emb_sys,
                       p_ivl: irx.Interval, dt: float, steps_per_segment: int) -> irx.Interval:
    """Propagate and record the state interval at the end of every segment.
    Both the within-segment loop and the segment loop use jax.lax.scan (this
    system's dynamics are cheap/polynomial, so this is a conservative choice
    for compile-cost headroom, not a necessity the way it was for
    quadrotor_fault_diagnosis's trig-heavy embedding)."""
    def segment(x_ivl, u_k):
        def step(x_carry, _):
            return euler_step(emb_sys, x_carry, u_k, p_ivl, dt), None
        x_end, _ = jax.lax.scan(step, x_ivl, xs=None, length=steps_per_segment)
        return x_end, x_end

    _, x_hist = jax.lax.scan(jax.checkpoint(segment), x0_ivl, u_seq)
    return x_hist


def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray, scenario: Scenario,
                                 dt: float, steps_per_segment: int) -> irx.Interval:
    x_hist = _propagate_history(x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment)
    return irx.Interval(lower=x_hist.lower[-1], upper=x_hist.upper[-1])


def separation_loss_multistep(u_seq: jnp.ndarray, x0_ivl: irx.Interval, scenarios: List[Scenario],
                              dt: float, steps_per_segment: int) -> jnp.ndarray:
    """Min over segments of the pairwise OBSERVED-OUTPUT overlap sum --
    fully vectorized (gather over pair indices + vmap over the segment axis),
    matching quadrotor_fault_diagnosis's compile-cost-driven vectorization."""
    num_segments = u_seq.shape[0]
    n = len(scenarios)
    emb_sys = scenarios[0].emb_system

    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def prop_one(p_ivl_single):
        return _propagate_history(x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment)

    x_hist_batch = jax.vmap(prop_one)(p_batch)   # lower/upper shape (n, num_segments, 18)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    i_idx = jnp.array([i for i, j in pairs])
    j_idx = jnp.array([j for i, j in pairs])

    def overlap_at_k(k):
        lower_i, upper_i = x_hist_batch.lower[i_idx, k, :12], x_hist_batch.upper[i_idx, k, :12]
        lower_j, upper_j = x_hist_batch.lower[j_idx, k, :12], x_hist_batch.upper[j_idx, k, :12]
        widths = jnp.maximum(jnp.minimum(upper_i, upper_j) - jnp.maximum(lower_i, lower_j), 0.0)
        return jnp.sum(jnp.prod(widths, axis=-1))

    segment_overlaps = jax.vmap(overlap_at_k)(jnp.arange(num_segments))
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
    key = jax.random.PRNGKey(seed)
    noise_scale = _BIAS_LIM * 0.5
    u0 = jax.random.normal(key, (num_restarts, opt.num_segments, 3)) * noise_scale

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u_batch, _i):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses


def optimize_multistep(scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float,
                       steps_per_segment: int, num_segments: int,
                       learning_rate: float = 0.01, num_iters: int = 300,
                       num_restarts: int = 100, seed: int = 42) -> Tuple[jnp.ndarray, float, Dict]:
    """Multi-start gradient descent over a sequence of spoof biases (unrefined loss)."""
    ms_opt = MultistepSequenceOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
                                        steps_per_segment=steps_per_segment, num_segments=num_segments)
    u_seq, loss_opt_jax, final_u, final_losses = optimize_multistep_gpu(
        opt=ms_opt, num_restarts=num_restarts, learning_rate=learning_rate,
        num_iters=num_iters, seed=seed,
    )
    x_ivls = [propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment) for s in scenarios]
    n = len(scenarios)
    overlaps = {f"{scenarios[i].name} vs {scenarios[j].name}": float(_output_overlap_volume(x_ivls[i], x_ivls[j]))
               for i in range(n) for j in range(i + 1, n)}
    stats = {'state_intervals': x_ivls, 'pairwise_overlaps': overlaps,
            'all_restart_losses': np.array(final_losses)}
    return u_seq, float(loss_opt_jax), stats


# ══════════════════════════════════════════════════════════════════════════════
# 5. Output-anticipating intersection-refinement spoof-bias sequence
# ══════════════════════════════════════════════════════════════════════════════
# The "react ... using existing code for model discrimination" primitive,
# used here at DESIGN time: at each step, if a pair's OBSERVED-output
# intervals currently overlap, refine that observed sub-block to their
# intersection before propagating the next step -- anticipating that a real
# online monitor (Section 6) would do exactly this once it starts getting
# real observations, so the spoof is optimized against the TIGHTER,
# refinement-anticipating reachable tube rather than the looser unrefined
# one from Section 4. Mirrors quadrotor_fault_diagnosis's
# `propagate_with_refinement` vmap-over-pairs pattern; the one structural
# difference is which sub-block gets intersected:
#
#   quadrotor_fault_diagnosis (no sensor fault): observation = identity on
#     the FULL state, so refining "the observed intersection" tightens all
#     12 state dims for both scenarios.
#   here: observation is a PARTIAL identity (only indices 0:12). Refining
#     the observed intersection can only ever correct the observable 12
#     dims -- the hidden controller-memory dims (integ, cmd; indices 12:18)
#     are carried forward UNCHANGED per scenario. This is not an
#     approximation of quadrotor_fault_diagnosis's approach, it's the
#     correct generalization: you cannot infer a hidden state from an
#     output-space intersection without an explicit (here, nonexistent)
#     inverse map from output back to the hidden block.

def propagate_with_refinement(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                              scenarios: List[Scenario], dt: float,
                              num_steps: int = 2) -> jnp.ndarray:
    """Multi-step propagation with per-pair OBSERVED-OUTPUT refinement.

    Returns
    -------
    min over steps of per-step pairwise OBSERVED-OUTPUT overlap sum (scalar)
    """
    n = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    N = x0_ivl.lower.shape[0]   # 18
    emb_sys = scenarios[0].emb_system

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:N], upper=arr[N:])

    p_i_batch = irx.Interval(
        lower=jnp.stack([scenarios[i].p_interval.lower for i, j in pairs]),
        upper=jnp.stack([scenarios[i].p_interval.upper for i, j in pairs]),
    )
    p_j_batch = irx.Interval(
        lower=jnp.stack([scenarios[j].p_interval.lower for i, j in pairs]),
        upper=jnp.stack([scenarios[j].p_interval.upper for i, j in pairs]),
    )

    # ── Step 1: propagate all scenarios one Euler step with u_seq[0] ──────
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x1_batch = jax.vmap(lambda p: euler_step(emb_sys, x0_ivl, u_seq[0], p, dt))(p_batch)
    x1_ivls = [irx.Interval(lower=x1_batch.lower[i], upper=x1_batch.upper[i]) for i in range(n)]
    step1_cost = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            step1_cost = step1_cost + _output_overlap_volume(x1_ivls[i], x1_ivls[j])

    pxi_arr = jnp.stack([ivl_to_arr(x1_ivls[i]) for i, j in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x1_ivls[j]) for i, j in pairs])

    def refine_one_pair(xi_arr, xj_arr, p_i, p_j, u_k):
        """One pair's refine-then-propagate step. Traced ONCE, vmapped over
        the pair axis by step_body below."""
        x_curr_i = arr_to_ivl(xi_arr)
        x_curr_j = arr_to_ivl(xj_arr)

        # Intersect ONLY the observable 12 dims.
        obs_lo_i, obs_hi_i = x_curr_i.lower[:12], x_curr_i.upper[:12]
        obs_lo_j, obs_hi_j = x_curr_j.lower[:12], x_curr_j.upper[:12]
        x_lo = jnp.maximum(obs_lo_i, obs_lo_j)
        x_hi = jnp.minimum(obs_hi_i, obs_hi_j)
        has_overlap = jnp.all(x_hi >= x_lo)

        refined_obs_lo = jnp.where(has_overlap, x_lo, obs_lo_i)
        refined_obs_hi = jnp.where(has_overlap, x_hi, obs_hi_i)
        x_ref_i = irx.Interval(
            lower=jnp.concatenate([refined_obs_lo, x_curr_i.lower[12:18]]),
            upper=jnp.concatenate([refined_obs_hi, x_curr_i.upper[12:18]]),
        )
        refined_obs_lo_j = jnp.where(has_overlap, x_lo, obs_lo_j)
        refined_obs_hi_j = jnp.where(has_overlap, x_hi, obs_hi_j)
        x_ref_j = irx.Interval(
            lower=jnp.concatenate([refined_obs_lo_j, x_curr_j.lower[12:18]]),
            upper=jnp.concatenate([refined_obs_hi_j, x_curr_j.upper[12:18]]),
        )

        x_next_i = euler_step(emb_sys, x_ref_i, u_k, p_i, dt)
        x_next_j = euler_step(emb_sys, x_ref_j, u_k, p_j, dt)

        raw_cost = _output_overlap_volume(x_next_i, x_next_j)
        pair_cost = jnp.where(has_overlap, raw_cost, jnp.array(0.0))
        return ivl_to_arr(x_next_i), ivl_to_arr(x_next_j), pair_cost

    def step_body(carry, k):
        pxi_arr, pxj_arr, min_cost = carry
        u_k = u_seq[k + 1]
        new_pxi, new_pxj, pair_costs = jax.vmap(
            refine_one_pair, in_axes=(0, 0, 0, 0, None)
        )(pxi_arr, pxj_arr, p_i_batch, p_j_batch, u_k)
        step_cost = jnp.sum(pair_costs)
        return (new_pxi, new_pxj, jnp.minimum(min_cost, step_cost))

    init_carry = (pxi_arr, pxj_arr, step1_cost)
    _, _, min_cost_final = _run_unrolled_or_loop(step_body, init_carry, num_steps - 1)
    return min_cost_final


def refined_overlap_loss(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                         scenarios: List[Scenario], dt: float, num_steps: int = 2) -> jnp.ndarray:
    return propagate_with_refinement(x0_ivl, u_seq, scenarios, dt, num_steps)


def optimize_refined_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                         num_steps: int = 2, num_restarts: int = 50,
                         learning_rate: float = 0.05, num_iters: int = 200,
                         seed: int = 42) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent minimising refined_overlap_loss."""
    key = jax.random.PRNGKey(seed)
    noise_scale = _BIAS_LIM * 0.5
    u0 = jax.random.normal(key, (num_restarts, num_steps, 3)) * noise_scale

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


# ══════════════════════════════════════════════════════════════════════════════
# 6. React: online discrimination from real observations
# ══════════════════════════════════════════════════════════════════════════════
# "React to the spoofing signal using existing code for model discrimination"
# (the original task) -- given the bias sequence actually injected and the
# REAL observed output trajectory it produced, decide which of the four
# controllers is active. This reuses the same primitives as Section 5
# (euler_step, observed_output) but the refinement source is different:
# Section 5 refines a candidate PAIR against EACH OTHER (design time, no
# ground truth yet); this refines each candidate against the single REAL
# observation (reaction time). A candidate is FALSIFIED the first step its
# predicted observed-output interval fails to contain the true observation
# -- the reachable-set analog of rq3_sme.py's "Theta is empty -> rejected"
# LP-feasibility falsification. Surviving candidates get their observable
# state refined to (a tight box around) the true observation each step,
# keeping their tube tight rather than growing conservatively over the whole
# horizon -- a real measurement update, not just cross-candidate intersection.
#
# w_bar: refining to an EXACT zero-width point (lower=upper=y_true) is
# fragile -- the true trajectory (simulate_true_trajectory) and the
# candidate's predicted tube (euler_step through natemb's interval
# arithmetic) are two independently-coded Euler integrations, so even
# though they compute the same math they don't reproduce each other
# bit-for-bit; verified empirically, a point-refined candidate gets
# spuriously falsified 1-2 steps later purely from ~1e-7 float32 rounding,
# not a real inconsistency. w_bar is exactly rq3_sme.py's own `w_bar`
# (bounded, not stochastic, measurement/model-mismatch budget) -- refine to
# `[y_true-w_bar, y_true+w_bar]`, not a point. Default is sized for float32
# rounding at this problem's scale, not a physical noise estimate; widen it
# to model real sensor noise.
#
# No attack synthesis on the identified controller here (PLAN.md Sec 6 Q4 /
# left explicitly to Phase 3) -- this section stops at "which controller is
# it", it does not do anything with that answer.

_DEFAULT_W_BAR = 1e-5

def simulate_true_trajectory(x0_point: jnp.ndarray, u_seq: jnp.ndarray, theta6: jnp.ndarray,
                             ref15: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Point (non-interval) rollout of ONE controller (the 'true' one), for
    generating a ground-truth observed trajectory to feed the discriminator
    in tests/demos. Not part of the reachability machinery -- a plain
    forward-Euler simulation of the raw (unembedded) system."""
    sys_, _ = get_system_and_embedding(ref15)
    theta6 = jnp.asarray(theta6)

    def step(x, u_k):
        x_next = x + dt * sys_.f(jnp.zeros(()), x, u_k, theta6)
        return x_next, x_next[:12]

    _, traj = jax.lax.scan(step, jnp.asarray(x0_point), u_seq)
    return traj   # (len(u_seq), 12)


def discriminate_controller(x0_ivl: irx.Interval, u_seq: jnp.ndarray, observed_traj: jnp.ndarray,
                            scenarios: List[Scenario], dt: float,
                            w_bar: float = _DEFAULT_W_BAR) -> Dict:
    """Run each scenario forward under the REAL injected bias sequence,
    checking at every step whether the true observed output stays inside
    that scenario's predicted observed-output interval (refining the
    observable dims to a `w_bar`-wide box around the true observation when
    it does -- see module comment above for why not an exact point).

    Parameters
    ----------
    x0_ivl : prior state uncertainty shared by all candidates before reacting
    u_seq : (T,3) the spoof bias sequence actually injected (e.g. from
            optimize_refined_gpu)
    observed_traj : (T,12) the REAL observed [p,v,a,j] at each step
    scenarios : the candidate bank (any subset/order of create_scenarios())
    w_bar : bounded measurement/model-mismatch budget (rq3_sme.py's own
            `w_bar`), applied both as containment slack and as the refined
            box half-width -- NOT an exact-point refinement.

    Returns
    -------
    dict with 'falsified' (bool per scenario), 'fail_step' (int, -1 if never
    falsified), 'survivors' (list of scenario names still consistent).
    """
    T = u_seq.shape[0]
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def run_one(p_ivl):
        def step(carry, k):
            x_ivl, falsified, fail_step = carry
            x_next = euler_step(emb_sys, x_ivl, u_seq[k], p_ivl, dt)
            y_pred = observed_output(x_next)
            y_true = observed_traj[k]
            contained = jnp.all((y_true >= y_pred.lower - w_bar) & (y_true <= y_pred.upper + w_bar))
            falsified_new = falsified | jnp.logical_not(contained)
            fail_step_new = jnp.where(jnp.logical_not(contained) & jnp.logical_not(falsified), k, fail_step)

            # Measurement update: narrow the observable dims to a w_bar-wide
            # box around the true observation while still consistent, so the
            # tube stays tight going forward instead of growing over the
            # whole horizon (but never narrower than the noise budget).
            refine_now = contained
            obs_lo = jnp.where(refine_now, y_true - w_bar, x_next.lower[:12])
            obs_hi = jnp.where(refine_now, y_true + w_bar, x_next.upper[:12])
            x_ref = irx.Interval(
                lower=jnp.concatenate([obs_lo, x_next.lower[12:18]]),
                upper=jnp.concatenate([obs_hi, x_next.upper[12:18]]),
            )
            return (x_ref, falsified_new, fail_step_new), contained

        init = (x0_ivl, jnp.array(False), jnp.array(-1))
        (_, falsified, fail_step), contained_hist = jax.lax.scan(step, init, jnp.arange(T))
        return falsified, fail_step, contained_hist

    falsified, fail_step, contained_hist = jax.vmap(run_one)(p_batch)

    return {
        'falsified': {s.name: bool(falsified[i]) for i, s in enumerate(scenarios)},
        'fail_step': {s.name: int(fail_step[i]) for i, s in enumerate(scenarios)},
        'contained_history': {s.name: np.array(contained_hist[i]) for i, s in enumerate(scenarios)},
        'survivors': [s.name for i, s in enumerate(scenarios) if not bool(falsified[i])],
    }


if __name__ == "__main__":
    scenarios = create_scenarios()
    x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-3))

    opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt=QPS_DT, num_steps=5)
    print("Loss at zero bias (no attack):", opt.loss_fn(jnp.zeros(3)))

    u_star, loss_star, u_all, losses_all = optimize_parallel_gpu(
        opt, num_restarts=64, learning_rate=0.02, num_iters=200,
    )
    print(f"\nBest discriminating spoof bias: {np.array(u_star).round(4)} m  (loss={float(loss_star):.6f})")
    result = opt.evaluate(u_star)
    print("\nPairwise overlaps at optimized bias:")
    for k, v in result['pairwise_overlaps'].items():
        print(f"  {k}: {v:.6f}")

    print("\n" + "=" * 70)
    print("Section 5: output-anticipating refinement, hard pair (pd_pos_vel vs pid_pos_vel_i)")
    hard_scenarios = create_scenarios(names=["pd_pos_vel", "pid_pos_vel_i"])
    x0_hard = irx.icentpert(jnp.zeros(18), jnp.full(18, 5e-2))
    u_seq_star, ref_loss_star, _, _ = optimize_refined_gpu(
        x0_hard, hard_scenarios, dt=QPS_DT, num_steps=5, num_restarts=32, num_iters=150,
    )
    print(f"Refined loss at optimized bias sequence: {float(ref_loss_star):.6f}")

    print("\n" + "=" * 70)
    print("Section 6: reacting to the spoof -- is the victim really qps_snap_chain?")
    all_scenarios = create_scenarios()
    u_seq_demo, _, _, _ = optimize_refined_gpu(
        x0_ivl, all_scenarios, dt=QPS_DT, num_steps=5, num_restarts=32, num_iters=150,
    )
    true_theta = jnp.array(_CANDIDATE_THETA["qps_snap_chain"])
    observed = simulate_true_trajectory(jnp.zeros(18), u_seq_demo, true_theta, hover_reference(), dt=QPS_DT)
    result = discriminate_controller(x0_ivl, u_seq_demo, observed, all_scenarios, dt=QPS_DT)
    print("Survivors after reacting to the real drone's response:", result['survivors'])
    print("Falsified at step:", result['fail_step'])
