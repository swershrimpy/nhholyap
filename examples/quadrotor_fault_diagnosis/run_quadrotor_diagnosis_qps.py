"""
QPS-ground-truth statistical diagnosis success-rate experiment for the
12-state quadrotor fault-diagnosis module.

What this measures
-------------------
Takes the ONE saved separating input `u_seq` from solve_and_plot.py
(results/quadrotor_separating_input.npz -- 5 scenarios: Nominal +
ActuatorFault_1..4, num_steps=2, dt=0.1s, x0_width=0.01, alpha in [0.5,0.9])
and checks whether reachability-based diagnosis stays correct when the true
plant is QPS's own verified rigid-body Crazyflie model
(adaptive_spoofing/crazyflie_12d.py's `CrazyflieSystem`), not the model the
controller was SYNTHESIZED against
(quadrotor_separating_input.py's `QuadrotorSystem`).

This is a strictly harder soundness test than unicycle's
robotarium_diagnosis_mc.py, which only tested a DISCRETIZATION mismatch
(one coarse Euler step vs. a finer stepped simulator of the *same* ODE).
Here there is a second, structural mismatch on top of discretization: per
crazyflie_12d.py's own docstring, `QuadrotorSystem` state[6:9] is BODY-frame
velocity (position kinematics rotate it, translational dynamics carry
Coriolis coupling), while QPS's `CrazyflieSystem` state[6:9] is WORLD-frame
velocity (no rotation, no Coriolis terms) -- a genuinely different ODE, not
just a coarser integration of the same one. `QuadrotorSystem` is kept as the
synthesis/reachability model (matching PLAN.md's design), and
`CrazyflieSystem` is used here ONLY as the ground-truth plant, exactly as
"crazysim" was clarified to mean QPS's verified plant equations, not the
literal CrazySim Gazebo/ROS2 SITL stack (not installed on this machine).

Two predicted-interval resolutions are checked, both computed on the
SYNTHESIS model (QuadrotorSystem's embedding, body-frame) via
quadrotor_separating_input._propagate_history -- reused unchanged, not
reimplemented, so nothing about the predicted set can silently drift from
what the optimizer's own loss saw:
  coarse -- steps_per_segment=1 (dt=0.1s per Euler step): EXACTLY what
            optimize_refined_gpu computed and what solve_and_plot.py's
            reachable-box plot draws.
  fine   -- steps_per_segment=FINE_SUBSTEPS (dt=0.1/FINE_SUBSTEPS): matches
            the ground-truth rollout's own substep size, isolating whether
            failures are a discretization artifact of the single coarse
            Euler step vs. a genuine unsoundness of the body-frame model
            against the world-frame plant.
Both are reported separately (never conflated), same rationale as the car
precedent: coarse is what the paper's method literally deploys online; fine
tests the underlying reachability claim on a finer time grid of the SAME
(body-frame) model. Neither directly measures the frame mismatch in
isolation -- that mismatch is present in BOTH columns, since the ground
truth is always CrazyflieSystem (world-frame). Isolating discretization from
frame mismatch would require a third column (CrazyflieSystem-as-synthesis-
model reachability, not built here); out of scope for this experiment.

Ground truth: CrazyflieSystem.f, RK4-integrated at GT_SUBSTEPS substeps per
0.1s segment (dt_sub=0.005s), same rk4_rollout convention as
solve_and_plot.py but on the WORLD-frame model and vmapped across trials
(pure point dynamics, no interval arithmetic -- cheap; no subprocess/
resource-limiting needed, unlike the compile-heavy embedding functions
elsewhere in this project).

Same x0 sample is used for BOTH the synthesis-side interval containment
check (which assumes BODY-frame velocity in indices 6:9) and the
ground-truth rollout (which assumes WORLD-frame velocity in the same
indices) -- not an oversight. X0_WIDTH=0.01 keeps every velocity component
within +-0.01 of zero (near-hover), where body-frame and world-frame
velocity trivially coincide (R(phi=0,theta=0,psi=0)=I at the interval
center, and the perturbation itself is small), so a single x0 sample is a
physically consistent initial condition for both systems at this operating
point. This would NOT be valid at a large-attitude or high-speed operating
point -- flagged here, not silently assumed away.

Per-trial classification (mirrors unicycle's
robotarium_diagnosis_mc.py / Def. "Pairwise pruning" in 26-2796-AR.tex
Response 8): a model is excluded the first time the true state fails to be
contained in its predicted state-interval box at a segment boundary, and
stays excluded.
    correct      -- true model survives, and it's the only survivor
    inconclusive -- true model survives, but so does >=1 false model
    missed       -- true model gets excluded at some point
    wrong        -- missed AND exactly one false model survives

Adversarial trials (stress test, not a fairness claim): x0 pinned on a
random one of the 12 dims strictly outside its assumed half-width (extra
margin 1.0-1.5x X0_WIDTH, random sign); for Actuator-Fault true modes, the
faulted channel's alpha is pulled from a band 0.1 beyond [ALPHA_LO,ALPHA_HI]
on a random side. Reported separately -- missed/wrong diagnoses here are an
expected consequence of leaving the assumed operating envelope, not
evidence against the in-bound soundness claim.

Usage: JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python \
    examples/quadrotor_fault_diagnosis/run_quadrotor_diagnosis_qps.py
"""
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLES_DIR = _HERE.parent
for _p in (_HERE, _EXAMPLES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx

from quadrotor_separating_input import create_scenarios, _propagate_history
from adaptive_spoofing.crazyflie_12d import CrazyflieSystem

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DT = 0.1
# NUM_STEPS is NOT a fixed constant here -- it's read from the saved u_seq's
# own shape in main() (solve_and_plot.py's horizon is now NUM_SEGMENTS=20,
# not the fixed 2 this script originally assumed). Threaded through
# run_cell explicitly rather than reintroduced as a module-level global, so
# there's no stale constant to fall out of sync with the actual npz again.
X0_WIDTH = 0.01
ALPHA_LO, ALPHA_HI = 0.5, 0.9
FINE_SUBSTEPS = 20     # both the "fine" prediction resolution and the
GT_SUBSTEPS = 20       # ground-truth rollout's RK4 substep count -- equal,
                       # so the fine-vs-ground-truth comparison is apples to
                       # apples on time resolution (see module docstring)

N_INBOUND = 2000
N_ADV = 500
SEED = 0
_ADV_X0_MARGIN_LO, _ADV_X0_MARGIN_HI = 1.0, 1.5   # multiples of X0_WIDTH
_ADV_ALPHA_MARGIN = 0.1

MODEL_NAMES = ["Nominal", "ActuatorFault_1", "ActuatorFault_2",
              "ActuatorFault_3", "ActuatorFault_4"]

_gt_sys = CrazyflieSystem()


def _gt_f(x, u, alpha):
    return _gt_sys.f(jnp.zeros(()), x, u, alpha)


# ══════════════════════════════════════════════════════════════════════════
# Ground-truth plant: CrazyflieSystem (world-frame), RK4, vmapped over trials
# ══════════════════════════════════════════════════════════════════════════
def rk4_rollout_batch(x0_batch: jnp.ndarray, u_seq: jnp.ndarray,
                      alpha_batch: jnp.ndarray, dt: float, sub_steps: int) -> jnp.ndarray:
    """x0_batch: (B,12), alpha_batch: (B,4). Returns (B, num_steps, 12) state
    at every segment boundary."""
    h = dt / sub_steps

    def rk4_step(x, u, alpha):
        k1 = _gt_f(x, u, alpha)
        k2 = _gt_f(x + 0.5 * h * k1, u, alpha)
        k3 = _gt_f(x + 0.5 * h * k2, u, alpha)
        k4 = _gt_f(x + h * k3, u, alpha)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout_one(x0, alpha):
        def seg_scan(x, u_k):
            def body(x, _):
                return rk4_step(x, u_k, alpha), None
            x_end, _ = jax.lax.scan(body, x, xs=None, length=sub_steps)
            return x_end, x_end
        _, hist = jax.lax.scan(seg_scan, x0, u_seq)
        return hist   # (num_steps, 12)

    return jax.vmap(rollout_one)(x0_batch, alpha_batch)


_rk4_rollout_batch_jit = jax.jit(rk4_rollout_batch, static_argnums=(4,))


# ══════════════════════════════════════════════════════════════════════════
# Predicted state-interval history (reuses quadrotor_separating_input's own
# propagation code -- the exact function the optimizer's loss uses)
# ══════════════════════════════════════════════════════════════════════════
def predicted_histories(scenarios, x0_ivl, u_seq, steps_per_segment):
    """{model_name: (lo, hi)}, each shape (num_steps, 12).

    All 5 scenarios share one emb_system (only p_interval differs), so this
    vmaps _propagate_history over stacked p_intervals instead of looping in
    Python and separately tracing/compiling once per scenario -- same
    pattern quadrotor_separating_input.py's separation_loss_multistep /
    propagate_with_refinement already use, for the same reason: this
    system's trig-heavy embedding (PLAN.md "Compile-cost finding") makes
    per-scenario Python-loop tracing an avoidable multiplier. Measured
    ~2.7x faster and numerically identical to the old per-scenario loop.
    """
    per_step_dt = DT / steps_per_segment
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def prop_one(p_ivl_single):
        return _propagate_history(x0_ivl, u_seq, emb_sys, p_ivl_single,
                                  per_step_dt, steps_per_segment)

    hist_batch = jax.vmap(prop_one)(p_batch)   # lower/upper: (n_scenarios, num_steps, 12)
    return {s.name: (np.array(hist_batch.lower[i]), np.array(hist_batch.upper[i]))
            for i, s in enumerate(scenarios)}


# ══════════════════════════════════════════════════════════════════════════
# True-parameter sampling
# ══════════════════════════════════════════════════════════════════════════
def sample_x0(rng, batch, adversarial):
    x0 = np.zeros((batch, 12))
    if not adversarial:
        x0[:] = rng.uniform(-X0_WIDTH, X0_WIDTH, size=(batch, 12))
        return x0
    for b in range(batch):
        dim = rng.integers(0, 12)
        sign = rng.choice([-1.0, 1.0])
        extra = rng.uniform(_ADV_X0_MARGIN_LO, _ADV_X0_MARGIN_HI) * X0_WIDTH
        row = rng.uniform(-X0_WIDTH, X0_WIDTH, size=12)
        row[dim] = sign * (X0_WIDTH + extra)
        x0[b] = row
    return x0


def sample_alpha(rng, true_mode, batch, adversarial):
    alpha = np.ones((batch, 4))
    if true_mode == "Nominal":
        return alpha
    ch = int(true_mode[-1]) - 1
    if not adversarial:
        alpha[:, ch] = rng.uniform(ALPHA_LO, ALPHA_HI, size=batch)
        return alpha
    lo_lo, lo_hi = max(ALPHA_LO - _ADV_ALPHA_MARGIN, 0.0), ALPHA_LO
    hi_lo, hi_hi = ALPHA_HI, ALPHA_HI + _ADV_ALPHA_MARGIN
    for b in range(batch):
        if lo_hi > lo_lo and (rng.random() < 0.5 or hi_hi <= hi_lo):
            alpha[b, ch] = rng.uniform(lo_lo, lo_hi)
        else:
            alpha[b, ch] = rng.uniform(hi_lo, hi_hi)
    return alpha


# ══════════════════════════════════════════════════════════════════════════
# Classification
# ══════════════════════════════════════════════════════════════════════════
def _classify(alive, true_idx, outcomes, suffix=""):
    num_alive = alive.sum(axis=0)
    true_alive = alive[true_idx]
    for i in range(alive.shape[1]):
        if true_alive[i] and num_alive[i] == 1:
            outcomes["correct" + suffix] += 1
        elif true_alive[i]:
            outcomes["inconclusive" + suffix] += 1
        else:
            outcomes["missed" + suffix] += 1
            if num_alive[i] == 1:
                outcomes["wrong" + suffix] += 1


def run_cell(rng, u_seq, predicted_fine, predicted_coarse, true_mode, n_trials, adversarial,
            num_steps):
    outcomes = {k + suffix: 0 for k in ("correct", "inconclusive", "missed", "wrong")
               for suffix in ("", "_coarse")}
    true_idx = MODEL_NAMES.index(true_mode)

    x0_true = sample_x0(rng, n_trials, adversarial)
    alpha_true = sample_alpha(rng, true_mode, n_trials, adversarial)

    state_hist = np.array(_rk4_rollout_batch_jit(
        jnp.asarray(x0_true, dtype=jnp.float32), jnp.asarray(u_seq, dtype=jnp.float32),
        jnp.asarray(alpha_true, dtype=jnp.float32), DT, GT_SUBSTEPS))   # (B, num_steps, 12)

    for predicted, prefix in ((predicted_fine, ""), (predicted_coarse, "_coarse")):
        alive = np.ones((len(MODEL_NAMES), n_trials), dtype=bool)
        for k in range(num_steps):
            x_k = state_hist[:, k, :]   # (B, 12)
            for m, name in enumerate(MODEL_NAMES):
                lo, hi = predicted[name]
                contained = np.all((x_k >= lo[k]) & (x_k <= hi[k]), axis=-1)
                alive[m] &= contained
        _classify(alive, true_idx, outcomes, prefix)
    return outcomes


# ══════════════════════════════════════════════════════════════════════════
# Full sweep
# ══════════════════════════════════════════════════════════════════════════
def main(out_path: Path):
    d = np.load(RESULTS_DIR / "quadrotor_separating_input.npz", allow_pickle=True)
    u_seq = jnp.array(d["u_seq"])
    num_steps = int(u_seq.shape[0])
    disjoint_segments = list(d["disjoint_segments"])
    assert list(d["scenario_names"]) == MODEL_NAMES, \
        f"scenario order mismatch: {list(d['scenario_names'])} vs {MODEL_NAMES}"

    # Diagnosis cutoff SWEEP: check containment through segment 1, 2, ...,
    # up to max(disjoint_segments)+1 -- not just the full horizon. Found
    # necessary empirically: at the full 20-segment/2.0s horizon, EVERY
    # scenario (including Nominal at its own exact box-center point, zero
    # sampling noise) missed 100% of the time -- the body-frame
    # (QuadrotorSystem, synthesis) vs. world-frame (CrazyflieSystem, QPS
    # ground truth) velocity mismatch compounds enough by segment ~3 (0.4s)
    # that even the TRUE scenario's own box stops containing the true
    # trajectory (dims vx/vy/vz specifically -- see RESULTS.md "Second
    # caveat" for the full investigation). A real online diagnoser has no
    # reason to keep checking past the point separation is already
    # certified (the excluded-stays-excluded rule needs only ONE clean
    # window), so sweeping the cutoff answers the practically relevant
    # question directly: how much of the horizon can actually be used
    # before the synthesis/ground-truth model mismatch dominates? Swept
    # rather than fixed at one value to avoid cherry-picking a favorable
    # cutoff -- the full sweep is reported in RESULTS.md.
    max_cutoff = max(disjoint_segments, default=num_steps - 1) + 1
    print(f"Loaded u_seq: {num_steps} segments x dt={DT}s = {num_steps*DT:.2f}s horizon "
         f"(disjoint_segments={disjoint_segments}); sweeping diagnosis cutoff 1..{max_cutoff}")

    scenarios = create_scenarios(alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI)
    x0_ivl = irx.icentpert(jnp.zeros(12), jnp.full(12, X0_WIDTH))

    print("Computing predicted state-interval histories (coarse + fine)...")
    predicted_coarse = predicted_histories(scenarios, x0_ivl, u_seq, 1)
    predicted_fine = predicted_histories(scenarios, x0_ivl, u_seq, FINE_SUBSTEPS)

    rng = np.random.default_rng(SEED)
    rows = []
    t_start = time.perf_counter()
    for cutoff in range(1, max_cutoff + 1):
        print(f"=== diagnosis cutoff = {cutoff} (through segment {cutoff - 1}) ===")
        for true_mode in MODEL_NAMES:
            for adversarial, n_trials in ((False, N_INBOUND), (True, N_ADV)):
                out = run_cell(rng, u_seq, predicted_fine, predicted_coarse, true_mode,
                               n_trials, adversarial, cutoff)
                rows.append(dict(cutoff=cutoff, true_mode=true_mode, adversarial=adversarial,
                                 n_trials=n_trials, **out))
                tag = "adversarial" if adversarial else "in-bound"
                print(f"  {true_mode:16s} [{tag:11s}] n={n_trials}: {out}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall time: {total_elapsed:.1f}s over "
         f"{sum(r['n_trials'] for r in rows)} trials")

    np.savez(out_path, rows=np.array(rows, dtype=object),
            dt=DT, num_steps=num_steps, max_cutoff=max_cutoff,
            disjoint_segments=np.array(disjoint_segments, dtype=np.int32), x0_width=X0_WIDTH,
            alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
            fine_substeps=FINE_SUBSTEPS, gt_substeps=GT_SUBSTEPS,
            n_inbound=N_INBOUND, n_adv=N_ADV, seed=SEED,
            total_elapsed_s=total_elapsed)
    print(f"Wrote {out_path}")
    return rows


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "quadrotor_diagnosis_qps.npz"
    main(out)
