"""
Re-optimizes the 12 configs whose refined-method separation loss hit exactly
0 (config_success=True) but whose FINE-resolution predicted output intervals
still overlap for some model pair -- the mechanism behind the 0.3% inconclusive
rate in robotarium_diagnosis_mc.py's in-bound results, per the diagnosis in
that conversation. Requested fix: bloat both intervals by epsilon before
computing their overlap volume in the LOSS the optimizer minimizes (exactly
the remedy already proposed in Response 5 of 26-2796-AR.tex for Prop. 2's
boundary-touching gap -- returning Vol([lower-eps,upper+eps] \\cap
[lower-eps,upper+eps]) instead of Vol(ivl1 \\cap ivl2)), so a "successful"
(loss==0) solution must separate the worst-case reachable sets by a genuine
margin of at least ~2*epsilon, not just touch at a boundary.

Implementation: monkeypatches car_separating_input._overlap_volume to the
bloated version for the duration of the re-optimization call only (restored
in a `finally` block), then reuses success_rate_analysis_early_stop.py's
EXISTING run_batched_sweep_early_stop unchanged -- same restart scheme (half
constant-control, half independent-per-step, seed=42), same gd_early_stop
core, same NUM_RESTARTS/LEARNING_RATE/MAX_ITERS -- restricted to just the 12
ambiguous configs' arrays. propagate_with_refinement / _refine_and_step_pair
call `_overlap_volume` via ordinary Python global lookup (not a bound
reference captured at import time), so the monkeypatch is picked up by
whatever JAX traces during THIS call, with no risk of leaking into any other
code that imports car_separating_input afterward.

Epsilon: the user's prior guess was "millimeters." We check that guess
empirically rather than assume it. A quick pre-check
(see the conversation this script was written for) found the ACTUAL
worst-case 2D overlap between the involved pairs' fine-resolution boxes
ranges from ~25mm (Actuator Fault vs. Sensor Fault, easiest configs) up to
~840mm (Nominal vs. Sensor Fault, hardest configs) -- i.e. up to
~80cm, two orders of magnitude past "millimeters" for the worst offenders.
This script sweeps a geometric ladder of epsilon values (5mm up through
200mm) rather than committing to one value blindly, and reports, per
config, the smallest epsilon in the ladder that both (a) re-converges to
loss==0 under the bloated criterion and (b) actually eliminates the
FINE-resolution overlap against the UNBLOATED check (the two are not the
same thing -- see (b) below).

Two-stage validation per (config, epsilon) -- do not trust the bloated
loss alone:
  (a) Does gd_early_stop still find a solution with bloated-loss == 0
      within MAX_ITERS? If not, this epsilon is infeasible for this config
      (no control law achieves that much worst-case margin at all).
  (b) Recompute the FINE-resolution predicted intervals (matching
      robotarium_diagnosis_mc.py's ground-truth resolution, NOT the coarse
      resolution the optimizer's own loss uses) for the new u_opt, using
      the ORIGINAL (unbloated) _overlap_volume, and check whether the
      previously-overlapping pair is now disjoint at every one of the 10
      steps. Passing (a) does not guarantee (b): the bloated-loss margin is
      enforced on the COARSE (as-optimized) boxes, and fine-resolution
      widening (see robotarium_diagnosis_mc.py's docstring) can still eat
      into a margin that looked adequate at coarse resolution. Only
      configs passing BOTH are reported as resolved.

Usage: python reoptimize_epsilon_bloat.py [out_npz]
"""
import sys
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import car_separating_input as csi
import success_rate_analysis as sra
from success_rate_analysis_early_stop import run_batched_sweep_early_stop, MAX_ITERS
from robotarium_diagnosis_mc import predicted_y_history, MODEL_NAMES

AMBIGUOUS_CONFIGS = [13, 44, 48, 68, 71, 73, 74, 75, 76, 77, 78, 80]
EPSILON_LADDER_M = [0.005, 0.010, 0.025, 0.050, 0.100, 0.200]   # 5mm .. 200mm
NUM_RESTARTS = sra.NUM_RESTARTS
LEARNING_RATE = sra.LEARNING_RATE
SEED = sra.SEED
DT = sra.DT
REFINED_NUM_STEPS = sra.REFINED_NUM_STEPS


def _bloat(ivl: irx.Interval, eps: float) -> irx.Interval:
    return irx.Interval(lower=ivl.lower - eps, upper=ivl.upper + eps)


def _make_bloated_overlap_volume(eps: float):
    def bloated(ivl1, ivl2):
        return csi._overlap_volume_original(_bloat(ivl1, eps), _bloat(ivl2, eps))
    return bloated


@contextmanager
def bloated_overlap(eps: float):
    """Monkeypatches car_separating_input._overlap_volume for the duration
    of the `with` block only; always restores it, even on exception."""
    if not hasattr(csi, "_overlap_volume_original"):
        csi._overlap_volume_original = csi._overlap_volume
    original = csi._overlap_volume
    csi._overlap_volume = _make_bloated_overlap_volume(eps)
    try:
        yield
    finally:
        csi._overlap_volume = original


def config_arrays_for(indices, npz):
    return (
        jnp.array(npz["x0_center"][indices]),
        jnp.array(npz["x0_width"][indices]),
        jnp.array(npz["alpha_lo"][indices]),
        jnp.array(npz["alpha_hi"][indices]),
        jnp.array(npz["sensor_offset"][indices]),
        jnp.array(npz["sensor_scale"][indices]),
    )


def check_fine_resolution_disjoint(u_seq, x0_center, x0_width, alpha_lo, alpha_hi,
                                   obs_offset, obs_scale):
    """True iff every pairwise 2D box overlap is exactly 0 at every one of
    the 10 fine-resolution segment boundaries -- the actual apples-to-apples
    soundness condition, computed with the UNBLOATED overlap (imported
    fresh, unaffected by any monkeypatch active elsewhere)."""
    fine, _coarse = predicted_y_history(u_seq, x0_center, x0_width, alpha_lo, alpha_hi,
                                        obs_offset, obs_scale)
    worst_overlap_mm = 0.0
    worst_pair = None
    for i in range(3):
        for j in range(i + 1, 3):
            lo1, hi1 = fine[MODEL_NAMES[i]]
            lo2, hi2 = fine[MODEL_NAMES[j]]
            ox = np.minimum(hi1[:, 0], hi2[:, 0]) - np.maximum(lo1[:, 0], lo2[:, 0])
            oy = np.minimum(hi1[:, 1], hi2[:, 1]) - np.maximum(lo1[:, 1], lo2[:, 1])
            overlap2d = np.where((ox > 0) & (oy > 0), np.minimum(ox, oy), 0.0)
            m = float(overlap2d.max()) * 1000
            if m > worst_overlap_mm:
                worst_overlap_mm = m
                worst_pair = (MODEL_NAMES[i], MODEL_NAMES[j])
    return worst_overlap_mm == 0.0, worst_overlap_mm, worst_pair


def main(out_path: Path):
    npz = np.load(_HERE / "success_rate_data_early_stop_multistep_refined.npz", allow_pickle=True)
    idx = np.array(AMBIGUOUS_CONFIGS)
    config_arrays = config_arrays_for(idx, npz)

    results = {c: {"resolved_at_eps": None, "eps_tried": []} for c in AMBIGUOUS_CONFIGS}
    u_opt_resolved = {}

    for eps in EPSILON_LADDER_M:
        print(f"\n{'='*70}\nEpsilon = {eps*1000:.0f} mm\n{'='*70}")
        t0 = time.perf_counter()
        with bloated_overlap(eps):
            (jitted_fn, u_final, losses_final, iters_final,
             compile_t, run_t, mem) = run_batched_sweep_early_stop(
                "multistep_refined", (REFINED_NUM_STEPS, 2), config_arrays,
                NUM_RESTARTS, MAX_ITERS, LEARNING_RATE, SEED, DT,
                num_steps=REFINED_NUM_STEPS,
            )
        elapsed = time.perf_counter() - t0
        print(f"  optimized {len(idx)} configs in {elapsed:.1f}s "
              f"(compile {compile_t*1e3:.0f}ms, run {run_t*1e3:.0f}ms)")

        best_idx = jnp.argmin(losses_final, axis=1)
        for ci, c in enumerate(AMBIGUOUS_CONFIGS):
            if results[c]["resolved_at_eps"] is not None:
                continue   # already resolved at a smaller epsilon
            u_opt = u_final[ci, best_idx[ci]]
            bloated_loss = float(losses_final[ci, best_idx[ci]])

            x0_center = np.array(npz["x0_center"][c])
            x0_width = float(npz["x0_width"][c])
            alpha_lo = float(npz["alpha_lo"][c])
            alpha_hi = float(npz["alpha_hi"][c])
            obs_offset = np.array(npz["sensor_offset"][c])
            obs_scale = float(npz["sensor_scale"][c])

            disjoint, worst_mm, worst_pair = check_fine_resolution_disjoint(
                np.array(u_opt), x0_center, x0_width, alpha_lo, alpha_hi, obs_offset, obs_scale)

            converged = bloated_loss < 1e-6
            results[c]["eps_tried"].append(
                dict(eps_mm=eps * 1000, converged=converged, bloated_loss=bloated_loss,
                    fine_disjoint=disjoint, worst_overlap_mm=worst_mm, worst_pair=worst_pair))
            status = ("RESOLVED" if (converged and disjoint) else
                     "converged but STILL overlaps at fine res." if converged else
                     "did not converge (no margin achievable)")
            print(f"  config {c:3d}: bloated-loss={bloated_loss:.2e}  "
                  f"fine worst-overlap={worst_mm:7.2f}mm ({worst_pair})  -> {status}")
            if converged and disjoint:
                results[c]["resolved_at_eps"] = eps * 1000
                u_opt_resolved[c] = np.array(u_opt)

        if all(results[c]["resolved_at_eps"] is not None for c in AMBIGUOUS_CONFIGS):
            print("\nAll 12 configs resolved -- stopping the epsilon ladder early.")
            break

    print(f"\n{'='*70}\nSummary\n{'='*70}")
    for c in AMBIGUOUS_CONFIGS:
        r = results[c]
        if r["resolved_at_eps"] is not None:
            print(f"  config {c:3d}: resolved at epsilon={r['resolved_at_eps']:.0f}mm")
        else:
            last = r["eps_tried"][-1] if r["eps_tried"] else None
            print(f"  config {c:3d}: NOT resolved up to epsilon={EPSILON_LADDER_M[-1]*1000:.0f}mm"
                  + (f"  (last: worst-overlap={last['worst_overlap_mm']:.1f}mm, "
                     f"converged={last['converged']})" if last else ""))

    np.savez(out_path,
            ambiguous_configs=np.array(AMBIGUOUS_CONFIGS),
            epsilon_ladder_mm=np.array(EPSILON_LADDER_M) * 1000,
            results=np.array(results, dtype=object),
            u_opt_resolved=np.array(u_opt_resolved, dtype=object))
    print(f"\nWrote {out_path}")
    return results, u_opt_resolved


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "reoptimize_epsilon_bloat.npz"
    main(out)
