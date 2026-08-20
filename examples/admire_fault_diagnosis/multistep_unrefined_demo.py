"""
Multistep UNREFINED separating-input demo for the ADMIRE aircraft.

Solves for a control sequence u_seq (num_segments, 10) that separates all
11 fault scenarios' reachable angular-rate sets [pb, qb, rb], propagating
every scenario independently -- no cross-scenario observation information
is used mid-horizon (see multistep_refined_demo.py for the version that
does use it).

IMPORTANT compile-cost note (see admire_separating_input.py's module
docstring): a SINGLE, un-batched gradient of the multistep loss for this
system took ~266-271s on this machine at 10 segments x 11 scenarios,
identical on GPU and CPU (compile time runs on the host CPU regardless of
target backend -- GPU only speeds up already-compiled execution). This is
the fundamental cost of reverse-mode-differentiating through ADMIRE's
division-heavy dynamics, not something num_iters/num_restarts/vmap can
fix. num_iters/num_restarts below are tuned to keep the POST-COMPILE run
time short, not to make the compile itself fast; expect the JIT compile
below to take several minutes.

Usage:
  /home/user/immrax-venv/bin/python examples/admire_fault_diagnosis/multistep_unrefined_demo.py
  (JAX_PLATFORMS=cpu also works and is no slower for compile.)
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

from admire_separating_input import create_scenarios, optimize_multistep_gpu, _propagate_history

# Per admire/multistep_unrefined_demo.py's tuning history: dt=0.15 over 10
# segments (1.5s horizon) was the confirmed-safe, confirmed-good horizon --
# larger dt/horizon combinations drove most or all restarts to NaN.
DT = 0.15
NUM_SEGMENTS = 10

# A smart initial guess (excites the Flap surface, index 7, first) matches
# admire_staged_opt_minimal.ipynb's hand-designed guess and this project's
# own diagnostic finding that Nominal-vs-Flap is the hardest pair to
# separate -- lets a small restart/iteration budget still find a genuinely
# separating sequence instead of relying on many random restarts.
NUM_RESTARTS = 4
NUM_ITERS = 10
LEARNING_RATE = 0.05
INIT_SCALE = 0.05
SEED = 42
_FLAP_IDX = 7


def main():
    print(f"Devices: {jax.devices()}")
    scenarios = create_scenarios(fault_effectiveness=0.0)
    print(f"{len(scenarios)} fault scenarios (Nominal + 10 total-actuator-loss faults)")

    x0_nom = jnp.zeros(9).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(9) * 0.01)
    print(f"Horizon: {NUM_SEGMENTS} x {DT}s = {NUM_SEGMENTS*DT:.2f}s   "
          f"restarts={NUM_RESTARTS}  iters={NUM_ITERS}  lr={LEARNING_RATE}")

    u0_mean = jnp.zeros((NUM_SEGMENTS, 10)).at[0, _FLAP_IDX].set(0.15)
    u0_mean = u0_mean.at[1:, :].set(0.1)

    def run(seed):
        return optimize_multistep_gpu(
            x0_ivl=x0_ivl, scenarios=scenarios, dt=DT, steps_per_segment=1,
            num_segments=NUM_SEGMENTS, learning_rate=LEARNING_RATE,
            num_iters=NUM_ITERS, num_restarts=NUM_RESTARTS, seed=seed,
            init_scale=INIT_SCALE, u0_mean=u0_mean,
        )

    jitted = jax.jit(run)
    print("\nCompiling (see module docstring -- expect several minutes) ...")
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
    print(f"\nBest loss: {float(best_loss):.6f}  ({n_nan}/{NUM_RESTARTS} restarts NaN)")
    print(f"Loss distribution: min={float(jnp.nanmin(losses)):.6f}  "
          f"mean={float(jnp.nanmean(losses)):.6f}  max={float(jnp.nanmax(losses)):.6f}")

    # Genuine (not proxy-loss) disjointness check, matching this project's
    # established convention (unicycle/quadrotor solve scripts): verify the
    # RAW per-segment overlap directly, not just the loss value.
    hist_by_name = {
        s.name: _propagate_history(x0_ivl, best_u, s.emb_system, s.p_interval, DT, 1)
        for s in scenarios
    }
    n = len(scenarios)
    disjoint_segments = []
    for k in range(NUM_SEGMENTS):
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
