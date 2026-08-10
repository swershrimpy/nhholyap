"""
Phase: waypoint-based validation through the REAL QPS simulator.

Replays the synthesized discriminating waypoint sequence through QPS's actual
Crazyflie plant + controller (qps_snap_chain), recording the observed
trajectory and a video — using ONLY `q.set_poses()`, no `spoof_bias`, no
GPS deception of any kind.

This script has NO immrax/jax dependency — it only needs the QPS package
(cvxopt, control, matplotlib<3.10, numpy), so it runs under ~/qps-venv.

Reads: results/synthesized_waypoints.npz (produced by synthesize_waypoints.py
       under immrax-venv)
Writes: results/qps_waypoint_observed_trajectory.npz
        results/qps_waypoint_flight.mp4

Run with:
  MPLBACKEND=Agg /home/user/qps-venv/bin/python examples/adaptive_spoofing/run_qps_waypoint_validation.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.lines import Line2D

import qps.quadrotarium as quadrotarium

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FRAMES_DIR = RESULTS_DIR / "_qps_waypoint_frames"

MAX_SETTLE_STEPS = 300     # 6s cap
SETTLE_EXTRA_STEPS = 40    # extra hover steps after "reached"
FRAME_STRIDE = 1           # save every step (dt=0.02s -> 50fps)


def main():
    data = np.load(RESULTS_DIR / "synthesized_waypoints.npz", allow_pickle=True)
    waypoints = data["waypoints"]         # (K, 3)
    T_hop = float(data["T_hop"])
    dt = float(data["dt"])
    K = int(data["K"])
    P0 = data["P0"]

    print(f"Loaded synthesized waypoint sequence: K={K}, T_hop={T_hop}s")
    print(f"  Waypoints:\n  {waypoints}")
    print(f"  Starting from P0={P0}")

    # ── Initialize QPS simulator ──────────────────────────────────────────
    # Start slightly OFF P0 (>0.05m, QPS's own min_distance_between_waypoints
    # tolerance) rather than exactly at it. If the drone starts exactly at
    # the target, set_poses(P0) is never registered as a genuine "new
    # waypoint" (quadrotarium.py's new_waypoint check requires >=0.05m from
    # the current position), so the sim stays in HOVERING with
    # hover_positions still None -- nominal_snap_input_u's hover branch then
    # silently produces NaN. Same fix run_qps_validation.py already uses.
    start_pose = P0 + np.array([0.1, 0.05, -0.05])
    q = quadrotarium.Quadrotarium(number_of_quads=1, initial_conditions=start_pose.reshape(3, 1))
    q.waypoint_time = T_hop   # Force QPS's polynomial duration to match our design

    # ── Settle to initial hover at P0 ─────────────────────────────────────
    target = P0.reshape(3, 1)
    print(f"\nSettling from start_pose={start_pose} to hover at P0={P0}...")
    for step_i in range(1, MAX_SETTLE_STEPS + 1):
        q.set_poses(target)
        q.step()
        if q.quad_mission_state == "HOVERING":
            break
    else:
        raise RuntimeError(f"Did not reach HOVERING within {MAX_SETTLE_STEPS} steps")
    # Extra hover steps to damp residual dynamics
    for _ in range(SETTLE_EXTRA_STEPS):
        q.set_poses(target)
        q.step()
    settled_state = q.get_estimated_states()[:, 0].copy()  # (12,) [p,v,a,j]
    print(f"  Settled after {step_i + SETTLE_EXTRA_STEPS} steps")
    print(f"  Settled state (pos): {settled_state[:3]}")
    print(f"  Settled state (vel): {settled_state[3:6]}")
    if not np.all(np.isfinite(settled_state)):
        raise RuntimeError(f"Settled state is not finite: {settled_state}")

    # ── Fly the waypoint sequence ─────────────────────────────────────────
    print(f"\nFlying the {K}-waypoint mission...")
    all_observations = []
    frame_idx = 0

    # Clear/create frames directory for video
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)

    steps_per_hop = int(T_hop / dt)

    for k in range(K):
        wp = waypoints[k]
        print(f"  Waypoint {k}: {wp}")
        q.set_poses(wp.reshape(3, 1))

        for step_in_hop in range(steps_per_hop):
            q.set_poses(wp.reshape(3, 1))
            q.step()
            obs = q.get_estimated_states()[:, 0].copy()  # (12,) [p,v,a,j]
            all_observations.append(obs)

            # Save frame for video
            if frame_idx % FRAME_STRIDE == 0:
                q.figure.savefig(FRAMES_DIR / f"frame_{frame_idx:05d}.png", dpi=100)
            frame_idx += 1

    observed_traj = np.array(all_observations)  # (K*steps_per_hop, 12)
    print(f"\nRecorded trajectory shape: {observed_traj.shape}")
    print(f"  Final position: {observed_traj[-1, :3]}")

    # ── Save results ──────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "qps_waypoint_observed_trajectory.npz",
        observed_traj=observed_traj,
        waypoints=waypoints,
        T_hop=T_hop,
        dt=dt,
        K=K,
        settled_state=settled_state,
        P0=P0,
    )
    print(f"Saved observed trajectory to {RESULTS_DIR / 'qps_waypoint_observed_trajectory.npz'}")

    # ── Assemble video ────────────────────────────────────────────────────
    video_path = RESULTS_DIR / "qps_waypoint_flight.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(int(1.0 / dt)),
        "-i", str(FRAMES_DIR / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", str(video_path),
    ]
    print(f"\nAssembling video...")
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"  Saved {video_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ffmpeg failed or not found: {e}. Video not produced.")

    # Cleanup frames
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)

    # ── Save reference log ────────────────────────────────────────────────
    ref_log = {
        "P0": P0.tolist(),
        "waypoints": waypoints.tolist(),
        "T_hop": T_hop,
        "dt": dt,
        "K": K,
        "settled_pos": settled_state[:3].tolist(),
        "settled_vel": settled_state[3:6].tolist(),
        "final_pos": observed_traj[-1, :3].tolist(),
    }
    with open(RESULTS_DIR / "qps_waypoint_reference_log.json", "w") as f:
        json.dump(ref_log, f, indent=2)
    print(f"Saved reference log to {RESULTS_DIR / 'qps_waypoint_reference_log.json'}")


if __name__ == "__main__":
    main()
