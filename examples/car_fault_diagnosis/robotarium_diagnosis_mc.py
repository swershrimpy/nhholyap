"""
Monte Carlo missed/wrong-diagnosis study, driven through the actual
Robotarium simulator (rps.robotarium.Robotarium), for the "missed/wrong
diagnoses are never measured" half of Comment 14 (26-2796-AR.tex) --
Response 14 itself only covers synthesis SUCCESS rate (whether a
separating input exists), not whether the online consistency check stays
correct once you run it against a plant that isn't the exact coarse-Euler
abstraction used to synthesize and bound it.

What this measures
-------------------
Takes the saved open-loop controllers (u_opt) from the early-stopping
success-rate sweep (success_rate_data_early_stop_{multistep_refined,
multistep_unrefined}.npz), restricted to configs that actually achieved
separation (config_success == True) -- an unsuccessful config has no
controller worth diagnosing with. For each such config, under each of the
3 true fault modes, Monte Carlo samples a TRUE parameter instantiation
(initial state x0, actuator authority alpha, sensor offset/scale) from the
assumed intervals, executes u_opt through Robotarium's own stepped
unicycle dynamics (px_dot=v*cos(phi), py_dot=v*sin(phi), phi_dot=omega --
identical functional form to CarNomActSystem, see car_separating_input.py)
at a MUCH finer timestep (0.02s) than the DT=0.5s single-shot Euler step
used both to synthesize u_opt and to compute the predicted reachable
output intervals it's checked against. Robotarium's actuator saturation
(built for a real 0.2 m/s differential-drive robot) is disabled --
overriding max_wheel_velocity post-construction, verified against a
hand-computed Euler reference to match to float64 precision -- since the
paper's own model has no saturation beyond the u in [-1,1]^2 box already
enforced by _project_u; re-imposing the physical robot's tighter limits
would be testing an assumption the paper never makes, not a soundness
property of the paper's own model.

The predicted output-interval history is computed via car_separating_input's
OWN `_propagate_history` + `observed_output` (the exact functions the
optimizer and its loss use) -- but NOT at the literal single-Euler-step-per-
segment resolution the sweep used for speed. That distinction matters and is
not cosmetic: an early debugging pass ran `predicted_y_history` at
steps_per_segment=1 (DT=0.5 in one Euler step, exactly what
success_rate_analysis_early_stop.py itself computes) and got ~100% missed
diagnoses even for Nominal, in-bound, no adversarial perturbation at all.
Tracing one trial by hand showed why: the true (Robotarium-converged)
py-position at the first segment boundary was 0.1362, while the coarse
one-step prediction's upper bound was 0.1229 -- excluded. Refining ONLY the
prediction's internal Euler resolution for the SAME already-synthesized u_opt
(steps_per_segment=5, dt=0.1) already grows the bound to 0.1390 and contains
it; steps_per_segment=25 (dt=0.02, matching the Robotarium ground truth's
own substep size) gives comfortable margin. This is a discretization
artifact of using one big 0.5s Euler step for the reachability bound, NOT
evidence that the interval arithmetic itself is unsound -- immrax's natif is
a sound enclosure of the discrete EULER MAP's own reachable set, not of the
continuous-time ODE, and a single 0.5s step is simply too coarse a stand-in
for the continuous flow when phi changes by ~15-20 degrees within it (typical
for this sweep's optimized omega). So: PRED_SUBSTEPS below matches N_SUB
(both integrate the same synthesized, UNCHANGED control -- no
re-optimization -- at the same fine resolution), making this an apples-to-
apples test of whether the reachability-based diagnosis is sound against a
faithfully-integrated continuous plant. The coarse (steps_per_segment=1)
prediction -- i.e. literally what success_rate_analysis_early_stop.py
computes and checks online -- is ALSO recorded per cell (the `coarse_*`
outcome columns) and must be reported separately, clearly labeled as testing
the as-deployed discretization rather than the underlying method's
soundness; conflating the two would misrepresent a discretization-resolution
choice as an algorithmic defect.

Per-trial classification (mirrors Def. "Pairwise pruning" in 26-2796-AR.tex
Response 8: a model is excluded the first time its predicted interval
fails to contain the measurement, and stays excluded):
    correct      -- true model survives, and it's the only survivor
    inconclusive -- true model survives, but so does >=1 false model
    missed       -- true model gets excluded at some point (should have
                    probability ~0 in-bound if the coarse abstraction
                    soundly bounds the finer "true" trajectory; is not
                    expected to hold under the adversarial condition)
    wrong        -- missed AND exactly one false model survives (a
                    reachability-based diagnoser that has to commit to a
                    single answer would commit to the WRONG one)

Adversarial trials (not a fairness claim -- a stress test): push x0 to a
random dimension pinned strictly outside its assumed half-width (extra
margin 1.0-1.5x the half-width, random sign), and for Actuator Fault, pull
alpha from a band 0.1 beyond [alpha_lo, alpha_hi] on a random side; for
Sensor Fault, perturb the true obs_offset/obs_scale slightly off their
assumed point values. Reported SEPARATELY from in-bound trials -- missed/
wrong diagnoses here are an expected, not alarming, consequence of leaving
the assumed operating envelope, and are not evidence against the paper's
soundness claim (which is conditioned on staying inside it).

Resource footprint (measured on this machine before committing to the full
sweep -- see the module's git history / the conversation that produced
this file for the benchmark): the library's own step() does one O(B^2)
pairwise-collision check per call (irrelevant here -- Monte Carlo trials
batched as Robotarium "robots" are logically independent, not physically
co-located), so cost is NOT minimized by the largest batch size. Benchmarked
B in {1,5,10,25,50} at 250 steps/trial: B=5 was fastest (~4.7ms/trial vs.
~28.7ms/trial at B=50). BATCH_N below is set from that result. Also closes
each Robotarium instance's leaked matplotlib Figure (created unconditionally
in __init__ regardless of show_figure -- confirmed by RuntimeWarning during
benchmarking) via plt.close(), which the upstream class never does itself.

Usage: python robotarium_diagnosis_mc.py [out_npz]
"""
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
import immrax as irx

from car_separating_input import create_scenarios, _propagate_history

_ROBOTARIUM_ROOT = "/home/user/output_feedback/robotarium_python_simulator"
if _ROBOTARIUM_ROOT not in sys.path:
    sys.path.insert(0, _ROBOTARIUM_ROOT)
from rps.robotarium import Robotarium

# ══════════════════════════════════════════════════════════════════════════
# Constants -- must match success_rate_analysis.py / success_rate_analysis_
# early_stop.py exactly, since we're diagnosing THEIR saved controllers.
# ══════════════════════════════════════════════════════════════════════════
DT = 0.5                 # segment length (s) the controllers were optimized at
NUM_STEPS = 10            # matches u_opt.shape[1]
N_SUB = 25                 # Robotarium substeps per segment -> dt_sub=0.02s
DT_SUB = DT / N_SUB
PRED_SUBSTEPS = N_SUB       # fine-prediction resolution -- matches N_SUB so the
                            # "fine" containment check is apples-to-apples against
                            # the ground truth (see module docstring)
BATCH_N = 5                # MC trials per Robotarium instance (see benchmark above)
N_INBOUND = 200
N_ADV = 50
SEED = 0
_ADV_ALPHA_MARGIN = 0.1
_ADV_SENSOR_OFFSET_MARGIN = 0.05
_ADV_SENSOR_SCALE_MARGIN = 0.03

MODEL_NAMES = ["Nominal", "Actuator Fault", "Sensor Fault"]
METHODS = ["multistep_refined", "multistep_unrefined"]


# ══════════════════════════════════════════════════════════════════════════
# Predicted output-interval history (reuses car_separating_input.py's own
# propagation code -- NOT a reimplementation, so the only thing that can
# differ from what the optimizer/loss saw is the plant we check it against)
# ══════════════════════════════════════════════════════════════════════════
def _predicted_at_resolution(scenarios, x0_ivl, u_seq, steps_per_segment):
    """{model_name: (lo, hi)}, each shape (NUM_STEPS, 2) -- predicted [px,py]
    observed-output interval at each of the 10 segment boundaries, integrating
    each segment's UNCHANGED control with `steps_per_segment` Euler substeps
    (steps_per_segment=1 reproduces exactly what the sweep script computes)."""
    dt = DT / steps_per_segment
    u_fine = jnp.repeat(jnp.asarray(u_seq, dtype=jnp.float32), steps_per_segment, axis=0)
    idx = np.arange(steps_per_segment - 1, NUM_STEPS * steps_per_segment, steps_per_segment)
    out = {}
    for s in scenarios:
        x_hist = _propagate_history(x0_ivl, u_fine, s.emb_system, s.p_interval, dt, 1)
        scale = float(s.obs_scale[0])
        offset = np.array(s.obs_offset)
        lo = scale * np.array(x_hist.lower[idx, :2]) + offset
        hi = scale * np.array(x_hist.upper[idx, :2]) + offset
        out[s.name] = (lo, hi)
    return out


def predicted_y_history(u_seq, x0_center, x0_width, alpha_lo, alpha_hi,
                        obs_offset, obs_scale):
    """Returns (fine, coarse): both {model_name: (lo,hi)} predicted-interval
    histories for the SAME synthesized u_seq (no re-optimization) --
    `fine` at PRED_SUBSTEPS resolution (matches the Robotarium ground truth's
    own substep size, for the apples-to-apples soundness test), `coarse` at
    steps_per_segment=1 (exactly what the sweep script itself computes and
    checks online -- see module docstring for why the two must be reported
    separately, not conflated)."""
    scenarios = create_scenarios(
        float(alpha_lo), float(alpha_hi),
        (float(obs_offset[0]), float(obs_offset[1])), float(obs_scale),
    )
    x0_ivl = irx.icentpert(jnp.array(x0_center, dtype=jnp.float32),
                           jnp.full(3, x0_width, dtype=jnp.float32))
    fine = _predicted_at_resolution(scenarios, x0_ivl, u_seq, PRED_SUBSTEPS)
    coarse = _predicted_at_resolution(scenarios, x0_ivl, u_seq, 1)
    return fine, coarse


# ══════════════════════════════════════════════════════════════════════════
# True-parameter sampling
# ══════════════════════════════════════════════════════════════════════════
def sample_x0(rng, x0_center, x0_width, batch, adversarial):
    x0 = np.empty((batch, 3))
    if not adversarial:
        x0[:] = rng.uniform(x0_center - x0_width, x0_center + x0_width, size=(batch, 3))
        return x0
    for b in range(batch):
        dim = rng.integers(0, 3)
        sign = rng.choice([-1.0, 1.0])
        extra = rng.uniform(1.0, 1.5) * x0_width
        for d in range(3):
            if d == dim:
                x0[b, d] = x0_center[d] + sign * (x0_width + extra)
            else:
                x0[b, d] = rng.uniform(x0_center[d] - x0_width, x0_center[d] + x0_width)
    return x0


def sample_alpha(rng, true_mode, alpha_lo, alpha_hi, batch, adversarial):
    if true_mode != "Actuator Fault":
        return np.ones(batch)
    if not adversarial:
        return rng.uniform(alpha_lo, alpha_hi, size=batch)
    alpha = np.empty(batch)
    lo_lo, lo_hi = max(alpha_lo - _ADV_ALPHA_MARGIN, 0.0), alpha_lo
    hi_lo, hi_hi = alpha_hi, alpha_hi + _ADV_ALPHA_MARGIN
    for b in range(batch):
        if lo_hi > lo_lo and (rng.random() < 0.5 or hi_hi <= hi_lo):
            alpha[b] = rng.uniform(lo_lo, lo_hi)
        else:
            alpha[b] = rng.uniform(hi_lo, hi_hi)
    return alpha


def sample_sensor(rng, true_mode, obs_offset, obs_scale, batch, adversarial):
    if true_mode != "Sensor Fault":
        return np.zeros((batch, 2)), np.ones(batch)
    if not adversarial:
        return np.tile(obs_offset, (batch, 1)), np.full(batch, obs_scale)
    offset = obs_offset + rng.uniform(-_ADV_SENSOR_OFFSET_MARGIN, _ADV_SENSOR_OFFSET_MARGIN,
                                      size=(batch, 2))
    scale = obs_scale + rng.uniform(-_ADV_SENSOR_SCALE_MARGIN, _ADV_SENSOR_SCALE_MARGIN,
                                    size=batch)
    return offset, scale


# ══════════════════════════════════════════════════════════════════════════
# Ground-truth plant: actual Robotarium stepped dynamics
# ══════════════════════════════════════════════════════════════════════════
def run_robotarium_batch(u_seq, x0_true, alpha_true):
    """x0_true: (B,3), alpha_true: (B,). Returns [px,py] history (NUM_STEPS,B,2)."""
    batch = x0_true.shape[0]
    r = Robotarium(number_of_robots=batch, show_figure=False, sim_in_real_time=False,
                   initial_conditions=x0_true.T.copy())
    r.max_wheel_velocity = 1e6   # disable the real robot's actuator saturation --
                                 # the paper's model has none beyond the u box itself
    r.time_step = DT_SUB
    pos_hist = np.zeros((NUM_STEPS, batch, 2))
    ids = np.arange(batch)
    for seg in range(NUM_STEPS):
        v_cmd = np.full(batch, float(u_seq[seg, 0]))
        omega_cmd = np.full(batch, float(u_seq[seg, 1])) * alpha_true
        for _ in range(N_SUB):
            r.get_poses()
            r.set_velocities(ids, np.vstack([v_cmd, omega_cmd]))
            r.step()
        pos_hist[seg] = r.poses[:2, :].T.copy()   # raw attribute -- avoids the
                                                    # get_poses/step call-order guard
    plt.close(r.figure)
    return pos_hist


# ══════════════════════════════════════════════════════════════════════════
# One (config, true_mode, adversarial) cell
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


def run_cell(rng, u_seq, predicted_fine, predicted_coarse, true_mode, x0_center, x0_width,
            alpha_lo, alpha_hi, obs_offset, obs_scale, n_trials, adversarial):
    outcomes = {k + suffix: 0 for k in ("correct", "inconclusive", "missed", "wrong")
               for suffix in ("", "_coarse")}
    true_idx = MODEL_NAMES.index(true_mode)
    n_done = 0
    while n_done < n_trials:
        b = min(BATCH_N, n_trials - n_done)
        x0_true = sample_x0(rng, x0_center, x0_width, b, adversarial)
        alpha_true = sample_alpha(rng, true_mode, alpha_lo, alpha_hi, b, adversarial)
        sensor_offset_true, sensor_scale_true = sample_sensor(
            rng, true_mode, obs_offset, obs_scale, b, adversarial)

        pos_hist = run_robotarium_batch(u_seq, x0_true, alpha_true)
        y_true = (sensor_scale_true[None, :, None] * pos_hist
                  + sensor_offset_true[None, :, :])   # (NUM_STEPS,b,2)

        for predicted, prefix in ((predicted_fine, ""), (predicted_coarse, "_coarse")):
            alive = np.ones((3, b), dtype=bool)
            for k in range(NUM_STEPS):
                for m, name in enumerate(MODEL_NAMES):
                    lo, hi = predicted[name]
                    contained = np.all((y_true[k] >= lo[k]) & (y_true[k] <= hi[k]), axis=-1)
                    alive[m] &= contained
            _classify(alive, true_idx, outcomes, prefix)
        n_done += b
    return outcomes


# ══════════════════════════════════════════════════════════════════════════
# Full sweep
# ══════════════════════════════════════════════════════════════════════════
def main(out_path: Path):
    rng = np.random.default_rng(SEED)
    rows = []
    t_start = time.perf_counter()
    n_cells_done = 0

    for method in METHODS:
        npz_path = _HERE / f"success_rate_data_early_stop_{method}.npz"
        d = np.load(npz_path, allow_pickle=True)
        succ = np.nonzero(d["config_success"])[0]
        n_cells_total = len(succ) * 3
        print(f"=== {method}: {len(succ)} successful configs -> "
              f"{n_cells_total} (config, true_mode) cells ===")

        for ci, c in enumerate(succ):
            u_seq = d["u_opt"][c]
            x0_center = d["x0_center"][c]
            x0_width = float(d["x0_width"][c])
            alpha_lo = float(d["alpha_lo"][c])
            alpha_hi = float(d["alpha_hi"][c])
            obs_offset = d["sensor_offset"][c]
            obs_scale = float(d["sensor_scale"][c])

            predicted_fine, predicted_coarse = predicted_y_history(
                u_seq, x0_center, x0_width, alpha_lo, alpha_hi, obs_offset, obs_scale)

            for true_mode in MODEL_NAMES:
                for adversarial, n_trials in ((False, N_INBOUND), (True, N_ADV)):
                    out = run_cell(rng, u_seq, predicted_fine, predicted_coarse, true_mode,
                                   x0_center, x0_width, alpha_lo, alpha_hi, obs_offset, obs_scale,
                                   n_trials, adversarial)
                    rows.append(dict(method=method, config_idx=int(c), true_mode=true_mode,
                                     adversarial=adversarial, n_trials=n_trials, **out))
                n_cells_done += 1
            elapsed = time.perf_counter() - t_start
            if (ci + 1) % 10 == 0 or ci == len(succ) - 1:
                rate = n_cells_done / elapsed
                remaining_cells = (len(succ) - ci - 1) * 3 + (n_cells_total - n_cells_done)
                print(f"  [{method}] config {ci+1}/{len(succ)}  "
                      f"elapsed={elapsed:6.1f}s  ~{rate:.2f} cells/s")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall time: {total_elapsed:.1f}s over {n_cells_done} cells "
          f"({sum(r['n_trials'] for r in rows)} trials)")

    np.savez(out_path, rows=np.array(rows, dtype=object),
            dt=DT, num_steps=NUM_STEPS, n_sub=N_SUB, dt_sub=DT_SUB,
            batch_n=BATCH_N, n_inbound=N_INBOUND, n_adv=N_ADV, seed=SEED,
            total_elapsed_s=total_elapsed)
    print(f"Wrote {out_path}")
    return rows


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "robotarium_diagnosis_mc.npz"
    main(out)
