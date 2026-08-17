"""
Synthesize a separating (discriminating) spoof-bias sequence for the full
4-controller bank, using the output-anticipating refinement layer (Phase 2,
Section 5). Saves the result to results/synthesized_spoof.npz for downstream
use by plot_separation.py and run_qps_validation.py.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/synthesize_spoof.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import immrax as irx

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_chain_controllers import (
    create_scenarios, hover_reference, QPS_DT, _CANDIDATE_THETA,
    optimize_refined_gpu, propagate_with_refinement, discriminate_controller,
    simulate_true_trajectory,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Horizon: 30 steps * dt=0.02s = 0.6s. Long enough for the hard pair
# (pd_pos_vel vs pid_pos_vel_i) to separate once the integral term has had
# multiple steps to accumulate (PLAN.md Sec 7 finding), short enough to stay
# a believable single "attack window" against a hovering Crazyflie.
NUM_STEPS = 30
HOVER_POS = jnp.array([0.0, 0.0, 1.0])   # matches the QPS validation run's settle point


def main():
    ref15 = hover_reference(HOVER_POS)
    scenarios = create_scenarios(ref15)
    # TIGHT regime half-width: 2mm (updated from the original 1mm). Checked
    # first, same as run_firmware_discrimination.py does for its own tight
    # regime: refined_overlap_loss at ZERO bias is already 0.0 at this width
    # too (as it was at 1mm), so a synthesized bias is not strictly necessary
    # for separation here either -- the optimizer below still runs and (per
    # the unchanged 1mm result, max|bias|=0.3 despite zero bias also working)
    # is expected to land on some other zero-loss point in that flat region,
    # not on zero itself. This mirrors the ALREADY-existing behavior at 1mm,
    # not a new phenomenon introduced by widening to 2mm.
    x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 2e-3))

    print(f"Synthesizing a {NUM_STEPS}-step separating spoof bias over all 4 controllers...")
    t0 = time.perf_counter()
    u_seq, loss_star, u_all, losses_all = optimize_refined_gpu(
        x0_ivl, scenarios, dt=QPS_DT, num_steps=NUM_STEPS,
        num_restarts=48, learning_rate=0.03, num_iters=250, seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s. Best refined loss = {float(loss_star):.6f} "
         f"(median over restarts = {float(jnp.median(losses_all)):.6f})")

    # Sanity check: does this sequence actually let us discriminate all four
    # controllers online, one at a time as the "true" one?
    print("\nSanity check -- discriminate_controller against each candidate as ground truth:")
    all_ok = True
    for true_name in _CANDIDATE_THETA:
        true_theta = jnp.array(_CANDIDATE_THETA[true_name])
        observed = simulate_true_trajectory(jnp.zeros(18), u_seq, true_theta, ref15, dt=QPS_DT)
        result = discriminate_controller(x0_ivl, u_seq, observed, scenarios, dt=QPS_DT)
        survivors = result["survivors"]
        ok = survivors == [true_name]
        all_ok &= ok
        print(f"  true={true_name:16s}  survivors={survivors}  {'OK' if ok else 'AMBIGUOUS/WRONG'}")

    np.savez(
        RESULTS_DIR / "synthesized_spoof.npz",
        u_seq=np.array(u_seq),
        loss_star=float(loss_star),
        dt=QPS_DT,
        num_steps=NUM_STEPS,
        hover_pos=np.array(HOVER_POS),
        ref15=np.array(ref15),
        x0_ivl_lower=np.array(x0_ivl.lower),
        x0_ivl_upper=np.array(x0_ivl.upper),
        candidate_names=np.array(list(_CANDIDATE_THETA.keys())),
        candidate_thetas=np.array([_CANDIDATE_THETA[n] for n in _CANDIDATE_THETA]),
        all_uniquely_discriminated=all_ok,
    )
    print(f"\nSaved synthesized spoof bias to {RESULTS_DIR / 'synthesized_spoof.npz'}")
    print(f"u_seq shape: {u_seq.shape}, max |bias|: {float(jnp.max(jnp.abs(u_seq))):.4f} m")


if __name__ == "__main__":
    main()
