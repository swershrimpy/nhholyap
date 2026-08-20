"""
Multistep INTERSECTION-REFINEMENT separating-input demo for the ADMIRE
aircraft.

Solves for a control sequence u_seq (num_steps, 10) that separates all 11
fault scenarios via propagate_with_refinement (at each step, every
overlapping scenario pair's shared observed output [pb,qb,rb] is
intersected, tightening both scenarios' state before propagating forward
-- see admire_separating_input.py's `_step_one_pair` docstring for the
memory-optimized vmap-over-pairs implementation, and its module docstring
for the no-overlap-fallback bug fix applied there).

Compile-cost note: see multistep_unrefined_demo.py's module docstring --
a single unbatched gradient of the (simpler) unrefined loss already takes
minutes on this machine, compile-bound (not GPU-acceleratable). The
refined loss additionally vmaps over C(11,2)=55 scenario pairs per step,
so expect this script's compile to take AT LEAST as long, plausibly
several minutes more.

IMPORTANT correctness caveat (mirrors examples/admire/multistep_refined_demo.py):
a low/zero REFINED loss is a genuine diagnosability certificate under the
refinement mechanism's own semantics, but is NOT automatically the same as
the RAW (independent per-scenario) reachable boxes being disjoint --
refined boxes are built by intersecting scenarios' PREDICTED outputs, a
proxy for what an online observer would see, not the same object as
independent propagation. This script checks the raw per-segment
disjointness directly (not just trusting the loss value) before reporting
a result as genuine, and prints both.

Usage:
  /home/user/immrax-venv/bin/python examples/admire_fault_diagnosis/multistep_refined_demo.py
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

from admire_separating_input import create_scenarios, _propagate_history, optimize_refined_gpu

DT = 0.1
NUM_STEPS = 10             # 1.0s horizon
NUM_RESTARTS = 6
NUM_ITERS = 15              # tuned short -- see module docstring: run time
                            # is the controllable knob, not compile time
LEARNING_RATE = 0.05
SEED = 42


def main():
    print(f"Devices: {jax.devices()}")
    scenarios = create_scenarios(fault_effectiveness=0.0)
    print(f"{len(scenarios)} fault scenarios (Nominal + 10 total-actuator-loss faults)")

    x0_nom = jnp.zeros(9).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(9) * 0.01)
    print(f"Horizon: {NUM_STEPS} x {DT}s = {NUM_STEPS*DT:.2f}s   "
          f"restarts={NUM_RESTARTS}  iters={NUM_ITERS}  lr={LEARNING_RATE}")

    def run(seed):
        return optimize_refined_gpu(
            x0_ivl=x0_ivl, scenarios=scenarios, dt=DT, num_steps=NUM_STEPS,
            num_restarts=NUM_RESTARTS, learning_rate=LEARNING_RATE,
            num_iters=NUM_ITERS, seed=seed,
        )

    jitted = jax.jit(run)
    print("\nCompiling (see module docstring -- expect several minutes, "
          "likely more than the unrefined script) ...")
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
    print(f"\nBest (refined) loss: {float(best_loss):.6f}  ({n_nan}/{NUM_RESTARTS} restarts NaN)")
    print(f"Loss distribution: min={float(jnp.nanmin(losses)):.6f}  "
          f"mean={float(jnp.nanmean(losses)):.6f}  max={float(jnp.nanmax(losses)):.6f}")

    # Raw (independent, NOT refined) per-segment disjointness check -- the
    # real diagnosability question; see module docstring caveat.
    hist_by_name = {
        s.name: _propagate_history(x0_ivl, best_u, s.emb_system, s.p_interval, DT, 1)
        for s in scenarios
    }
    n = len(scenarios)
    disjoint_segments = []
    for k in range(NUM_STEPS):
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
    print(f"\nSegments with ALL {n*(n-1)//2} pairs genuinely (RAW, independent-box) "
          f"disjoint: {disjoint_segments}")
    print(f"At least one genuinely-disjoint segment: {len(disjoint_segments) > 0}")
    if best_loss < 1e-3 and len(disjoint_segments) == 0:
        print("\nWARNING: refined loss is near-zero but NO segment shows raw "
              "independent-box disjointness -- this result should NOT be trusted "
              "as a genuine separation certificate (see module docstring).")

    np.savez(_HERE / "multistep_refined_result.npz",
             u_seq=np.array(best_u), loss=float(best_loss), dt=DT,
             num_steps=NUM_STEPS, compile_t=compile_t, run_t=run_t,
             disjoint_segments=np.array(disjoint_segments, dtype=np.int32))
    print(f"\nSaved {_HERE / 'multistep_refined_result.npz'}")


if __name__ == "__main__":
    main()
