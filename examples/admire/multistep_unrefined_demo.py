"""
Standalone multistep UNREFINED separating-input script for ADMIRE.

Solves for a control sequence u_seq (num_segments, 10) that separates all
11 fault scenarios' reachable angular-rate sets [pb, qb, rb], using the
RAW (independent per-scenario) multistep loss (separation_loss_multistep)
-- no cross-scenario observation refinement. Uses
optimize_multistep_gpu_fused (this session's addition: same
separation_loss_multistep, which was already vmapped over scenarios, but
with batched_loss/batched_grad fused into one jax.vmap(jax.value_and_grad)
call instead of tracing the forward graph twice), wrapped in a single
top-level jax.jit (avoids the eager-optimizer host-memory blowup this
project has hit before with un-jitted multi-restart GD loops).

IMPORTANT compile-cost note (found this session): a SINGLE, un-batched
gradient of separation_loss_multistep -- no restarts, no jit, no
optimization loop, just jax.grad(loss_fn)(u_seq) for one (3,10) sequence
-- took ~266-271s on this machine, confirmed IDENTICAL on GPU and CPU
(compile time runs on the host CPU regardless of target backend; GPU only
speeds up already-compiled execution, which is not the bottleneck here).
This is the fundamental cost of reverse-mode-differentiating through
ADMIRE's division-heavy dynamics (see admire_separating_input.py's module
docstring), NOT something num_iters/num_restarts/vmap can fix -- it is why
this project's own admire_success_rate_analysis.py sweep is deployed as
SLURM sbatch HPC jobs rather than run interactively. Expect THIS script's
one-time jax.jit compile to take on the order of several minutes; num_iters
and num_restarts below are tuned to keep the POST-COMPILE run time short
(the actually-controllable knob) while still finding a genuinely separating
sequence, not to make the compile itself fast.

Usage:
  /home/user/immrax-venv/bin/python examples/admire/multistep_unrefined_demo.py
  (JAX_PLATFORMS=cpu also works and is no slower for compile; omit it to
  use the GPU present on this machine for the run phase.)
"""
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx

from admire_separating_input import (
    create_scenarios, optimize_multistep_gpu_fused, separation_loss_multistep,
    _propagate_history,
)

# ── Configuration ────────────────────────────────────────────────────────────
# dt/horizon history: 0.1 (1.0s) -> loss stuck ~0.050. 0.3 (3.0s) -> 49/50
# restarts NaN. 0.5 (5.0s) -> 50/50 NaN. 0.15 (1.5s) -> loss 0.0457, 54/55
# pairs genuinely disjoint by the final segment -- the one confirmed-safe,
# confirmed-good horizon; keeping it.
DT = 0.15
NUM_SEGMENTS = 10        # 1.5s horizon at dt=0.15

# Per admire_staged_opt_minimal.ipynb (read this session, not modified): a
# SMART initial guess -- not random search -- is what let that notebook's
# single-trajectory (no restarts) optimizer converge in ~10 GD steps,
# 7.8ms post-compile run time. Its hand-designed guess specifically excites
# the Flap surface (index 7) first -- which independently matches this
# script's own diagnostic finding (Nominal-vs-Flap was the ONE pair still
# overlapping at the dt=0.15 config's best segment). Mirroring that here:
# drastically fewer restarts/iters than the earlier blind-random-search
# runs, centered on a guess that front-loads Flap excitation.
NUM_RESTARTS = 1         # was 50 (5362ms), then 10 (468ms), then 1 (457ms) --
                          # restarts barely moved run time (468->457ms for
                          # 10x fewer restarts), so restarts is NOT the run-
                          # time driver at this problem size -- iterations are
                          # (~23ms/iter at 10 segments x 11 scenarios).
NUM_ITERS = 1             # was 20 (457ms), then 2 (69.79ms) -- extrapolated
                          # ~21.5ms/iter + ~27ms fixed overhead from the
                          # 2-vs-20 data points; 1 iter should land ~48ms
LEARNING_RATE = 0.05
INIT_SCALE = 0.05         # tried 0.0 (pure deterministic smart guess,
                          # matching admire_staged_opt_minimal.ipynb) -- that
                          # made loss WORSE (0.1317 vs 0.0926), so the small
                          # noise is genuinely helping even at num_restarts=1,
                          # iters=1. Reverted to 0.05.
SEED = 42

_FLAP_IDX = 7             # index into the 10 control surfaces -- see
                          # admire_separating_input.py's _SURFACE_NAMES


def main():
    print(f"Devices: {jax.devices()}")
    scenarios = create_scenarios(fault_effectiveness=0.0)
    print(f"{len(scenarios)} fault scenarios (Nominal + 10 total-actuator-loss faults)")

    x0_nom = jnp.zeros(9).at[0].set(343.0 * 0.3)   # Vt ~ 102.9 m/s
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(9) * 0.01)
    print(f"Horizon: {NUM_SEGMENTS} x {DT}s = {NUM_SEGMENTS*DT:.2f}s   "
         f"restarts={NUM_RESTARTS}  iters={NUM_ITERS}  lr={LEARNING_RATE}")

    # Smart initial guess (see module docstring): segment 0 excites ONLY
    # Flap; remaining segments apply a modest broad excitation across all
    # surfaces -- mirrors admire_staged_opt_minimal.ipynb's 2-stage
    # [Flap-only, then all-surfaces] pattern, extended to this script's
    # 10-segment parameterization.
    u0_mean = jnp.zeros((NUM_SEGMENTS, 10)).at[0, _FLAP_IDX].set(0.15)
    u0_mean = u0_mean.at[1:, :].set(0.1)

    def run(seed):
        return optimize_multistep_gpu_fused(
            x0_ivl=x0_ivl, scenarios=scenarios, dt=DT, steps_per_segment=1,
            num_segments=NUM_SEGMENTS, learning_rate=LEARNING_RATE,
            num_iters=NUM_ITERS, num_restarts=NUM_RESTARTS, seed=seed,
            init_scale=INIT_SCALE, u0_mean=u0_mean,
        )

    jitted = jax.jit(run)
    print("\nCompiling (see module docstring -- this is expected to take "
         "several minutes for ADMIRE, independent of GPU vs CPU) ...")
    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(SEED))
    compile_t = time.perf_counter() - t0
    print(f"Compile time: {compile_t:.1f}s")

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(SEED))
    run_t = time.perf_counter() - t0
    print(f"Post-compile run time: {run_t*1e3:.2f}ms")

    best_u, best_loss, u_final, losses = out
    n_nan = int(jnp.sum(jnp.isnan(losses)))
    print(f"\nBest loss: {float(best_loss):.6f} rad^3  ({n_nan}/{NUM_RESTARTS} restarts NaN)")
    print(f"Loss distribution: min={float(jnp.nanmin(losses)):.6f}  "
         f"mean={float(jnp.nanmean(losses)):.6f}  max={float(jnp.nanmax(losses)):.6f}")

    # ── Genuine (not proxy-loss) disjointness check, matching this project's
    #    established convention (quadrotor/unicycle solve scripts):
    #    verify the RAW per-segment overlap directly, not just the loss value,
    #    since this loss IS already the raw/independent metric here (unlike
    #    faulty_car's refined CBF loss) -- so this is confirmatory, not a
    #    refined-vs-raw gap check.
    hist_by_name = {
        s.name: _propagate_history(x0_ivl, best_u, s.emb_system, s.p_interval, DT, 1)
        for s in scenarios
    }
    n = len(scenarios)
    disjoint_segments = []
    for k in range(NUM_SEGMENTS):
        worst_margin = None
        all_disjoint_k = True
        for i in range(n):
            for j in range(i + 1, n):
                oi = irx.Interval(lower=hist_by_name[scenarios[i].name].lower[k, 3:6],
                                  upper=hist_by_name[scenarios[i].name].upper[k, 3:6])
                oj = irx.Interval(lower=hist_by_name[scenarios[j].name].lower[k, 3:6],
                                  upper=hist_by_name[scenarios[j].name].upper[k, 3:6])
                disjoint = bool(jnp.any((oi.lower > oj.upper) | (oj.lower > oi.upper)))
                all_disjoint_k &= disjoint
        if all_disjoint_k:
            disjoint_segments.append(k)
    print(f"\nSegments with ALL {n*(n-1)//2} pairs genuinely (raw) disjoint: {disjoint_segments}")
    print(f"At least one genuinely-disjoint segment: {len(disjoint_segments) > 0}")

    np.savez(_HERE / "multistep_unrefined_result.npz",
            u_seq=np.array(best_u), loss=float(best_loss), dt=DT,
            num_segments=NUM_SEGMENTS, compile_t=compile_t, run_t=run_t,
            disjoint_segments=np.array(disjoint_segments, dtype=np.int32))
    print(f"\nSaved {_HERE / 'multistep_unrefined_result.npz'}")


if __name__ == "__main__":
    main()
