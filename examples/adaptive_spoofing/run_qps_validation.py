"""
Phase 3 validation: replay the synthesized separating spoof-bias sequence
through the REAL QPS simulator (~/adaptive_spoofing's actual Crazyflie
plant + actual controller, `nominal_snap_input_u` = qps_snap_chain), record
a video of the flight, and log the real observed trajectory + the reference
position/velocity keypoints used -- for run_discrimination.py (a separate
script, run under immrax-venv) to confirm the chain-layer discrimination
pipeline's predictions hold up against the real simulator, not just our own
point-simulation ground truth.

During the attack window, also overlays each candidate controller's
predicted reachable POSITION box (precomputed by compute_reachable_boxes.py,
immrax-venv, from the PREVIOUS run's settled state -- QPS's simulation is
deterministic given the same start pose/target, so a fresh run's settled
state should match closely enough; a warning is printed if it doesn't) as a
colored 3D wireframe, and zooms the camera in on the drone once the attack
starts so the (cm-to-dm scale) boxes are legible against the ~0.2-0.3m drone
body -- see quadcopter_plot.py's radius0/radius1.

This script has NO immrax/jax dependency -- it only needs the QPS package
itself (cvxopt, control, matplotlib<3.10, numpy), so it runs under the
dedicated ~/qps-venv, not immrax-venv. It reads results/synthesized_spoof.npz
and results/reachable_boxes.npz (both plain numpy, produced under
immrax-venv) and writes results/qps_observed_trajectory.npz +
results/qps_reference_log.json + results/qps_spoof_flight.mp4.

Run with:
  MPLBACKEND=Agg /home/user/qps-venv/bin/python examples/adaptive_spoofing/run_qps_validation.py
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import qps.quadrotarium as quadrotarium

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FRAMES_DIR = RESULTS_DIR / "_qps_frames"

MAX_SETTLE_STEPS = 300     # 6s cap -- generous vs. QPS's own 4s waypoint_time
SETTLE_EXTRA_STEPS = 40    # extra hover steps after "reached" to damp residual v/a/j
RECOVERY_STEPS = 40        # extra hover steps after the attack, bias dropped to 0
FRAME_STRIDE = 1           # save every step (dt=0.02s -> 50fps real-time video)

# Same 4 colors used in plot_separation.py, duplicated here since this
# script runs in a separate venv with no import path back to that module.
CANDIDATE_COLORS = {
    "qps_snap_chain": "tab:blue",
    "pd_pos_vel": "tab:orange",
    "pid_pos_vel_i": "tab:green",
    "indi_jerk": "tab:red",
}

_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def box_segments(lower, upper):
    """12 edge segments of an axis-aligned box, for Line3DCollection."""
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    return [[corners[i], corners[j]] for i, j in _BOX_EDGES]


def main():
    data = np.load(RESULTS_DIR / "synthesized_spoof.npz", allow_pickle=True)
    u_seq = data["u_seq"]                       # (num_steps, 3)
    dt = float(data["dt"])
    hover_pos = data["hover_pos"].astype(float)  # (3,) e.g. [0,0,1]
    num_steps = int(data["num_steps"])
    assert u_seq.shape[0] == num_steps

    boxes_path = RESULTS_DIR / "reachable_boxes.npz"
    have_boxes = boxes_path.exists()
    if have_boxes:
        box_data = np.load(boxes_path, allow_pickle=True)
        pos_lower, pos_upper = box_data["pos_lower"], box_data["pos_upper"]   # (num_steps+1, n, 3)
        box_names = [str(n) for n in box_data["scenario_names"]]
        assert pos_lower.shape[0] == num_steps + 1, "reachable_boxes.npz doesn't match this u_seq -- rerun compute_reachable_boxes.py"
        # Fixed camera window for the whole attack: centered on the hover
        # target, sized to comfortably contain every box over the whole
        # horizon plus the drone body itself (radius0=0.1m, see
        # quadcopter_plot.py) with margin.
        max_extent = float(max(np.max(np.abs(pos_upper - hover_pos)), np.max(np.abs(pos_lower - hover_pos))))
        zoom_half = max_extent + 0.2
        print(f"Loaded reachable boxes for {box_names}; zoom half-width={zoom_half:.3f}m")
    else:
        print("No results/reachable_boxes.npz found -- run compute_reachable_boxes.py first "
             "for the box overlay/zoom. Proceeding WITHOUT overlay.")

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)

    # Start away from the hover target (>0.05m, QPS's own reached-tolerance)
    # so it goes through a real EXECUTING trajectory and explicitly sets
    # hover_positions on arrival -- starting exactly AT the target leaves
    # hover_positions=None, which crashes nominal_snap_input_u's hover branch.
    start_pose = hover_pos + np.array([0.3, 0.2, -0.1])
    q = quadrotarium.Quadrotarium(
        number_of_quads=1, show_figure=True, show_overhead_view=False,
        initial_conditions=start_pose.reshape(3, 1),
    )
    target = hover_pos.reshape(3, 1)

    frame_idx = 0

    def save_frame():
        nonlocal frame_idx
        if frame_idx % FRAME_STRIDE == 0:
            q.figure.savefig(FRAMES_DIR / f"frame_{frame_idx:05d}.png", dpi=100)
        frame_idx += 1

    print(f"Flying from {start_pose} to hover at {hover_pos} ...")
    # quad_mission_state defaults to "HOVERING" before any set_poses/step()
    # call, so checking it BEFORE stepping would exit immediately without
    # ever actually commanding the quad anywhere -- always take at least one
    # step first, then check the (now real, waypoint-logic-derived) state.
    settle_steps = None
    for settle_steps in range(1, MAX_SETTLE_STEPS + 1):
        q.set_poses(target)
        q.step()
        save_frame()
        if q.quad_mission_state == "HOVERING":
            break
    else:
        raise RuntimeError(f"Did not reach HOVERING within {MAX_SETTLE_STEPS} steps -- check waypoint_time/tolerance")
    current_pos = q.get_poses()[:, 0]
    print(f"Reached HOVERING at step {settle_steps} ({settle_steps * dt:.2f}s), position={current_pos} "
         f"(target={hover_pos}, |error|={np.linalg.norm(current_pos - hover_pos):.4f}m). "
         f"Settling {SETTLE_EXTRA_STEPS} more steps...")
    for _ in range(SETTLE_EXTRA_STEPS):
        q.set_poses(target)
        q.step()
        save_frame()

    settled_state = q.get_estimated_states()[:, 0].copy()   # (12,) [p,v,a,j]
    print(f"Settled state (pos,vel,acc,jerk): {settled_state}")

    if have_boxes:
        # Boxes were precomputed (immrax-venv) from a PRIOR run's settled
        # state -- QPS has no randomness, so this run's should match closely.
        old_settled_pos = (pos_lower[0, 0] + pos_upper[0, 0]) / 2.0
        pos_diff = np.linalg.norm(settled_state[0:3] - old_settled_pos)
        if pos_diff > 0.01:
            print(f"WARNING: this run's settled position differs from the one used to "
                 f"compute reachable_boxes.npz by {pos_diff:.4f}m -- boxes may be "
                 f"slightly offset from the real drone. Rerun compute_reachable_boxes.py "
                 f"after this script if you want them realigned.")

    print(f"\nInjecting the synthesized {num_steps}-step spoof bias "
         f"(max |bias|={np.max(np.abs(u_seq)):.4f} m)...")

    if have_boxes:
        # Zoom in once the attack begins -- the boxes are cm-to-dm scale,
        # illegible against the full arena view used during the settle phase.
        q.axes.set_xlim3d(hover_pos[0] - zoom_half, hover_pos[0] + zoom_half)
        q.axes.set_ylim3d(hover_pos[1] - zoom_half, hover_pos[1] + zoom_half)
        q.axes.set_zlim3d(hover_pos[2] - zoom_half, hover_pos[2] + zoom_half)
        legend_handles = [Line2D([0], [0], color=CANDIDATE_COLORS.get(n, "gray"), lw=2, label=n)
                          for n in box_names]
        q.axes.legend(handles=legend_handles, loc="upper left", fontsize=7,
                     title="reachable position\nper hypothesized controller")
        box_collections = []

    observed_traj = np.zeros((num_steps, 12))
    for k in range(num_steps):
        q.spoof_bias = u_seq[k].reshape(3, 1)
        q.set_poses(target)
        q.step()

        if have_boxes:
            for coll in box_collections:
                coll.remove()
            box_collections = []
            for i, name in enumerate(box_names):
                segs = box_segments(pos_lower[k + 1, i], pos_upper[k + 1, i])
                coll = Line3DCollection(segs, colors=CANDIDATE_COLORS.get(name, "gray"),
                                        linewidths=1.5, alpha=0.9)
                q.axes.add_collection3d(coll)
                box_collections.append(coll)

        save_frame()
        observed_traj[k] = q.get_estimated_states()[:, 0]

    print(f"Attack done. Recovering ({RECOVERY_STEPS} steps, bias dropped to 0)...")
    if have_boxes:
        for coll in box_collections:
            coll.remove()
    q.spoof_bias = np.zeros((3, 1))
    for _ in range(RECOVERY_STEPS):
        q.set_poses(target)
        q.step()
        save_frame()

    q.call_at_scripts_end()

    np.savez(
        RESULTS_DIR / "qps_observed_trajectory.npz",
        observed_traj=observed_traj,     # (num_steps, 12) -- what the discriminator sees
        u_seq=u_seq,
        dt=dt,
        settled_state=settled_state,
        hover_pos=hover_pos,
        start_pose=start_pose,
    )
    print(f"Saved real QPS observed trajectory to {RESULTS_DIR / 'qps_observed_trajectory.npz'}")

    # "Record reference position keypoints and velocities somewhere": the
    # reference this whole pipeline is designed against is a static hover
    # setpoint (position keypoint) with zero velocity feed-forward -- log it
    # explicitly, plus the actual start/settle/target positions and the
    # per-step TRUE velocity trace QPS produced, for reproducibility.
    reference_log = {
        "reference_type": "static_hover",
        "position_keypoints_m": {
            "start_pose": start_pose.tolist(),
            "hover_setpoint (position_sp)": hover_pos.tolist(),
            "settled_position_at_attack_start": settled_state[0:3].tolist(),
        },
        "velocity_reference_m_s": {
            "v_ref (feed-forward, static hover)": [0.0, 0.0, 0.0],
            "settled_velocity_at_attack_start": settled_state[3:6].tolist(),
        },
        "dt_s": dt,
        "settle_steps": settle_steps + SETTLE_EXTRA_STEPS,
        "attack_steps": num_steps,
        "recovery_steps": RECOVERY_STEPS,
        "observed_velocity_trace_m_s": observed_traj[:, 3:6].tolist(),
        "observed_position_trace_m": observed_traj[:, 0:3].tolist(),
    }
    with open(RESULTS_DIR / "qps_reference_log.json", "w") as f:
        json.dump(reference_log, f, indent=2)
    print(f"Saved reference position/velocity log to {RESULTS_DIR / 'qps_reference_log.json'}")

    print(f"\nEncoding video from {frame_idx} frames...")
    out_mp4 = RESULTS_DIR / "qps_spoof_flight.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(round(1.0 / dt)),
        "-i", str(FRAMES_DIR / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4),
    ], check=True)
    print(f"Saved video to {out_mp4}")
    shutil.rmtree(FRAMES_DIR)


if __name__ == "__main__":
    main()
