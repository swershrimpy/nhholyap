"""
Synthesize a discriminating waypoint sequence for the full 4-controller bank,
using the waypoint-space optimizer with output-anticipating refinement.
Saves the result to results/synthesized_waypoints.npz for downstream use by
run_qps_waypoint_validation.py and run_waypoint_discrimination.py.

No GPS spoofing — the waypoint sequence is a legitimate flight plan that
produces different observable responses under different controllers, enabling
identification.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/synthesize_waypoints.py
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

from adaptive_spoofing.crazyflie_waypoint_controllers import (
    create_scenarios, nominal_waypoints, optimize_waypoints_gpu,
    simulate_true_trajectory, discriminate_controller_waypoint,
    _waypoints_to_u_seq,
    P0, P1, DEFAULT_T_HOP, DEFAULT_K, QPS_DT,
)
from adaptive_spoofing.crazyflie_chain_controllers import _CANDIDATE_THETA
from adaptive_spoofing.crazyflie_waypoint_trajectory import build_mission_reference

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Mission parameters
K = DEFAULT_K           # 5 waypoints
T_HOP = DEFAULT_T_HOP  # 4.0s per hop


def main():
    scenarios = create_scenarios()
    x0_state4x3 = jnp.zeros((4, 3)).at[0].set(P0)  # start at P0, at rest
    x0_ivl = irx.icentpert(jnp.zeros(18).at[0:3].set(P0), jnp.full(18, 1e-3))

    print(f"Synthesizing a {K}-waypoint discriminating flight plan...")
    print(f"  Nominal line: P0={np.array(P0)} → P1={np.array(P1)}")
    print(f"  T_hop={T_HOP}s, dt={QPS_DT}s, total mission time={K * T_HOP}s")
    print(f"  Nominal waypoints:\n    {np.array(nominal_waypoints(K))}")

    t0 = time.perf_counter()
    best_offsets, best_loss, all_offsets, all_losses = optimize_waypoints_gpu(
        x0_ivl, scenarios, x0_state4x3,
        K=K, T_hop=T_HOP, dt=QPS_DT,
        num_restarts=48, learning_rate=0.005, num_iters=200, seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Best loss = {float(best_loss):.6f} "
          f"(median over restarts = {float(jnp.median(all_losses)):.6f})")

    waypoints = nominal_waypoints(K) + best_offsets
    print(f"\nOptimized waypoints:\n  {np.array(waypoints)}")
    print(f"Offsets from nominal:\n  {np.array(best_offsets)}")

    # Build the full reference trajectory
    u_seq = build_mission_reference(waypoints, x0_state4x3, T_HOP, QPS_DT)
    print(f"Reference trajectory shape: {u_seq.shape}")

    # Sanity check: discriminate each candidate
    print("\nSanity check — discriminate_controller against each candidate as ground truth:")
    all_ok = True
    for true_name in _CANDIDATE_THETA:
        true_theta = jnp.array(_CANDIDATE_THETA[true_name])
        x0_point = jnp.zeros(18).at[0:3].set(P0)
        observed = simulate_true_trajectory(x0_point, u_seq, true_theta, dt=QPS_DT)
        result = discriminate_controller_waypoint(x0_ivl, u_seq, observed, scenarios, dt=QPS_DT)
        survivors = result["survivors"]
        ok = survivors == [true_name]
        all_ok &= ok
        print(f"  true={true_name:16s}  survivors={survivors}  {'OK' if ok else 'AMBIGUOUS/WRONG'}")

    np.savez(
        RESULTS_DIR / "synthesized_waypoints.npz",
        waypoints=np.array(waypoints),
        offsets=np.array(best_offsets),
        nominal_waypoints=np.array(nominal_waypoints(K)),
        loss_star=float(best_loss),
        T_hop=T_HOP,
        dt=QPS_DT,
        K=K,
        P0=np.array(P0),
        P1=np.array(P1),
        x0_state4x3=np.array(x0_state4x3),
        x0_ivl_lower=np.array(x0_ivl.lower),
        x0_ivl_upper=np.array(x0_ivl.upper),
        candidate_names=np.array(list(_CANDIDATE_THETA.keys())),
        candidate_thetas=np.array([_CANDIDATE_THETA[n] for n in _CANDIDATE_THETA]),
        all_uniquely_discriminated=all_ok,
    )
    print(f"\nSaved to {RESULTS_DIR / 'synthesized_waypoints.npz'}")


if __name__ == "__main__":
    main()
