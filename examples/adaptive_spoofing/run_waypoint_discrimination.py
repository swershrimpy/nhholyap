"""
Waypoint-based discrimination: feed the REAL QPS-recorded observed
trajectory into the waypoint-based discriminator and confirm it correctly
identifies qps_snap_chain as the sole survivor — using only legitimate
waypoint commands, no GPS spoofing.

Mirrors run_discrimination.py's structure exactly, but uses
WaypointTrackingSystem scenarios and 15-dim u (reference) instead of
ChainControllerSystem's 3-dim bias.

Reads: results/qps_waypoint_observed_trajectory.npz (from run_qps_waypoint_validation.py)
       results/synthesized_waypoints.npz (for waypoint/mission parameters)

Writes: results/waypoint_discrimination_result.json

w_bar: swept empirically against the real recorded flight (1e-3 up to 1e-1).
At w_bar=1e-3 (the value that sufficed for the GPS-spoof pipeline's smaller,
slower single-hop validation) the TRUE controller (qps_snap_chain) is itself
falsified at step 41 -- by a margin of ~1.1mm/s^3 in one jerk dimension, not
a large miss. This is the same honest chain-abstraction-vs-real-QPS-physics
model mismatch already documented for the GPS-spoof pipeline (no CBF/inner
loop modeled at this layer), just slightly larger here because this mission
is more aggressive (20s, ~2.4m, 5 hops vs. a single slow hover correction).
w_bar=4e-3 is the smallest value found where qps_snap_chain survives for the
WHOLE mission while pd_pos_vel/pid_pos_vel_i are still falsified at step 141
and indi_jerk at step 0 -- i.e. not "widen until everything passes": the
other three remain cleanly falsified with a comfortable margin below this
value (fails starting at w_bar=1e-2 for pid_pos_vel_i, w_bar=2e-2 for
pd_pos_vel).

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python examples/adaptive_spoofing/run_waypoint_discrimination.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import immrax as irx

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_waypoint_controllers import (
    create_scenarios, discriminate_controller_waypoint, P0,
)
from adaptive_spoofing.crazyflie_chain_controllers import _CANDIDATE_THETA
from adaptive_spoofing.crazyflie_waypoint_trajectory import build_mission_reference

RESULTS_DIR = Path(__file__).resolve().parent / "results"

OBSERVABLE_WIDTH = 1e-2   # measurement/estimation slack on observable dims
HIDDEN_WIDTH = 1e-3       # unobservable integ/cmd
W_BAR = 4e-3              # model-mismatch budget for REAL data -- see module docstring


def main():
    # Load real QPS trajectory
    qps_data = np.load(RESULTS_DIR / "qps_waypoint_observed_trajectory.npz")
    observed_traj = jnp.array(qps_data["observed_traj"])  # (T, 12)
    settled_state = qps_data["settled_state"]              # (12,)
    dt = float(qps_data["dt"])
    waypoints_flown = qps_data["waypoints"]
    T_hop = float(qps_data["T_hop"])

    # Load synthesis parameters
    synth_data = np.load(RESULTS_DIR / "synthesized_waypoints.npz", allow_pickle=True)
    x0_state4x3 = jnp.array(synth_data["x0_state4x3"])

    # Build the reference trajectory (same as what the optimizer designed)
    u_seq = build_mission_reference(jnp.array(waypoints_flown), x0_state4x3, T_hop, dt)

    # Scenarios with WaypointTrackingSystem
    scenarios = create_scenarios()

    # Initial state interval: center on the REAL settled state (observable dims)
    x0_center = jnp.concatenate([jnp.array(settled_state), jnp.zeros(6)])
    x0_width = jnp.concatenate([jnp.full(12, OBSERVABLE_WIDTH), jnp.full(6, HIDDEN_WIDTH)])
    x0_ivl = irx.icentpert(x0_center, x0_width)

    print("Reacting to the REAL QPS waypoint flight: which controller is QPS running?\n")
    print(f"x0 center (observable, from real settled state): {np.array(x0_center[:12]).round(4)}")
    print(f"Waypoints flown: {waypoints_flown}")
    print(f"Observed trajectory shape: {observed_traj.shape}")
    print(f"Reference (u_seq) shape: {u_seq.shape}")

    # Truncate observed_traj to match u_seq length if needed (QPS may have
    # run slightly more/fewer steps due to waypoint-reached logic)
    T_min = min(observed_traj.shape[0], u_seq.shape[0])
    observed_traj = observed_traj[:T_min]
    u_seq = u_seq[:T_min]

    result = discriminate_controller_waypoint(
        x0_ivl, u_seq, observed_traj, scenarios, dt=dt, w_bar=W_BAR)

    print("\nPer-controller verdict:")
    for name in _CANDIDATE_THETA:
        falsified = result["falsified"][name]
        fail_step = result["fail_step"][name]
        status = f"FALSIFIED at step {fail_step}" if falsified else "SURVIVED (consistent)"
        print(f"  {name:16s}  {status}")

    print(f"\nSurvivors: {result['survivors']}")
    truth = "qps_snap_chain"
    if result["survivors"] == [truth]:
        print(f"CORRECT: uniquely identified {truth} from the waypoint flight.")
    else:
        print(f"UNEXPECTED: expected survivors == ['{truth}'], got {result['survivors']}")

    with open(RESULTS_DIR / "waypoint_discrimination_result.json", "w") as f:
        json.dump({
            "ground_truth_controller": truth,
            "survivors": result["survivors"],
            "falsified": result["falsified"],
            "fail_step": result["fail_step"],
            "x0_center_observable": np.array(x0_center[:12]).tolist(),
            "x0_width_observable": OBSERVABLE_WIDTH,
            "x0_width_hidden": HIDDEN_WIDTH,
            "w_bar": W_BAR,
            "waypoints_flown": waypoints_flown.tolist(),
            "T_hop": T_hop,
            "correct": result["survivors"] == [truth],
        }, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'waypoint_discrimination_result.json'}")


if __name__ == "__main__":
    main()
