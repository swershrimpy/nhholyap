# Requirements: Crazyflie Waypoint Discrimination

## Overview

This feature adds legitimate waypoint-based controller discrimination to the Crazyflie adaptive spoofing pipeline. Instead of GPS spoofing, the system designs a commanded waypoint sequence that produces different observable responses under different controllers, enabling identification of which outer-loop controller is flying.

## User Stories

### 1. As a researcher, I want a JAX-differentiable polynomial trajectory generator so that I can optimize waypoint positions via gradient descent.

**Acceptance Criteria:**
- 1.1 The module provides `_build_boundary_matrix(T)` returning the 8×8 matrix for 7th-order polynomial boundary conditions.
- 1.2 `traj_coeffs_from_waypoint(end_pos, start_state4x3, T)` solves for polynomial coefficients per axis, matching QPS's `nominal_traj.py` numerically.
- 1.3 `sample_reference(coeffs, t)` evaluates the polynomial and its derivatives (pos through snap) at time t.
- 1.4 `build_mission_reference(waypoints, x0_state4x3, T_hop, dt)` chains K hops into a full (K×steps_per_hop, 15) reference trajectory.
- 1.5 The trajectory satisfies boundary conditions: start state matched at t=0, end position reached at t=T with zero vel/acc/jerk.
- 1.6 The full mission reference is continuous at hop boundaries (position at hop k+1 start ≈ waypoint k).
- 1.7 All functions are differentiable w.r.t. waypoint positions via JAX autodiff.

### 2. As a researcher, I want a WaypointTrackingSystem that drives the 18-state chain via time-varying reference so that discrimination uses legitimate waypoint commands instead of spoof bias.

**Acceptance Criteria:**
- 2.1 `WaypointTrackingSystem` has 18-state dynamics with control input `u = ref15` (15-dim reference: rp, rv, ra, rj, rs).
- 2.2 The system's `f(t, x, u, p)` implements the same chain dynamics as `ChainControllerSystem` but with the reference as a control input rather than a constant setpoint.
- 2.3 `get_system_and_embedding()` returns the system and its natural embedding for interval arithmetic.
- 2.4 `create_scenarios()` constructs the 4 candidate controller scenarios reusing `_CANDIDATE_THETA`.
- 2.5 `euler_step` performs one Euler integration step on an interval state with the given reference input.

### 3. As a researcher, I want a waypoint-sequence optimizer that maximizes controller separation so that I can find discriminating flight plans.

**Acceptance Criteria:**
- 3.1 `nominal_waypoints(K)` returns a K-waypoint straight-line path from P0 to P1.
- 3.2 `_project_offsets` enforces arena-scale box constraints on waypoint offsets.
- 3.3 `_waypoints_to_u_seq` converts offsets + nominal waypoints into a full reference trajectory via `build_mission_reference`.
- 3.4 `waypoint_separation_loss` computes sum-of-widths (negative separation) over all scenarios — nonnegative and differentiable.
- 3.5 `optimize_waypoints_gpu` performs multi-start projected gradient descent with configurable restarts, learning rate, and iterations.
- 3.6 The optimizer returns best offsets, best loss, and all restart results.

### 4. As a researcher, I want to simulate trajectories and discriminate controllers from observed data so that the full pipeline works end-to-end.

**Acceptance Criteria:**
- 4.1 `simulate_true_trajectory` rolls out a point state under a given controller theta and reference sequence.
- 4.2 `discriminate_controller_waypoint` propagates interval states, checks containment of observed data with w_bar slack, and identifies surviving controllers.
- 4.3 The true generating controller always survives discrimination (regression guard).
- 4.4 Non-matching controllers are falsified (their predicted reachable sets do not contain the observed trajectory).

### 5. As a researcher, I want scripts to synthesize, validate, and visualize waypoint-based discrimination so that I can run the full experimental workflow.

**Acceptance Criteria:**
- 5.1 `synthesize_waypoints.py` runs the optimizer and saves `results/synthesized_waypoints.npz` with waypoints, offsets, loss, and metadata.
- 5.2 `run_qps_waypoint_validation.py` deploys the waypoint sequence through the real QPS simulator via `q.set_poses()`, recording observed trajectory and video.
- 5.3 `run_waypoint_discrimination.py` feeds the real QPS trajectory into `discriminate_controller_waypoint` and saves the result JSON.
- 5.4 `plot_waypoint_separation.py` produces pairwise overlap, reachable tube, and 3D waypoint visualizations.

### 6. As a researcher, I want comprehensive tests validating correctness of the trajectory generator and discrimination system.

**Acceptance Criteria:**
- 6.1 Trajectory coefficients match an independent NumPy oracle across random inputs (parametrized over seeds).
- 6.2 Reference samples match the NumPy oracle at random time points.
- 6.3 Boundary conditions are verified at both t=0 and t=T.
- 6.4 Mission reference shape, hop-boundary continuity, and differentiability are tested.
- 6.5 WaypointTrackingSystem dynamics match a NumPy transcription oracle.
- 6.6 Separation loss is nonnegative and differentiable.
- 6.7 The true controller always survives discrimination (parametrized regression guard).
