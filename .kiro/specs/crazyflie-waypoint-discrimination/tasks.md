# Implementation Plan: Crazyflie Waypoint Discrimination

## Overview

Implement legitimate waypoint-based controller discrimination for the Crazyflie adaptive spoofing pipeline. The approach exploits the algebraic equivalence between spoof bias and waypoint offset in the chain dynamics, building a JAX-differentiable trajectory generator, a WaypointTrackingSystem, a multi-start gradient optimizer, and deployment/validation scripts.

## Tasks

- [x] 1. Implement JAX-differentiable polynomial trajectory generator
  - [x] 1.1 Create `crazyflie_waypoint_trajectory.py` with `_build_boundary_matrix(T)` — 8×8 matrix for 7th-order polynomial boundary conditions
    - Implement power-of-T entries for position/vel/acc/jerk constraints at t=0 and t=T
    - _Requirements: 1.1_

  - [x] 1.2 Implement `traj_coeffs_from_waypoint(end_pos, start_state4x3, T)` — solve for (3, 8) polynomial coefficients per axis
    - Construct RHS from start state and end position (zero end vel/acc/jerk)
    - Use `jnp.linalg.solve` on the boundary matrix
    - _Requirements: 1.2, 1.5_

  - [x] 1.3 Implement `sample_reference(coeffs, t)` — evaluate polynomial and derivatives up to snap at time t
    - Compute power vectors for pos, vel, acc, jerk, snap
    - Return (15,) array [pos, vel, acc, jerk, snap]
    - _Requirements: 1.3_

  - [x] 1.4 Implement `build_mission_reference(waypoints, x0_state4x3, T_hop, dt)` — chain K hops into full reference trajectory
    - Iterate over waypoints, solve coefficients per hop, sample at each dt step
    - Propagate end state of each hop as start state of the next
    - Return (K × steps_per_hop, 15) array
    - _Requirements: 1.4, 1.6, 1.7_

  - [x]* 1.5 Write tests for trajectory generator (`test_crazyflie_waypoint_trajectory.py`)
    - NumPy oracle for boundary matrix, coefficients, and samples
    - Parametrized oracle comparison over 8 random seeds
    - Boundary condition verification at t=0 and t=T
    - Shape, continuity, and differentiability tests for `build_mission_reference`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 2. Checkpoint — Verify trajectory generator tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement WaypointTrackingSystem and scenario construction
  - [x] 3.1 Create `crazyflie_waypoint_controllers.py` with `WaypointTrackingSystem` class
    - 18-state system (pos, vel, acc, jerk, integ_e_p, cmd) with `u = ref15` as control input
    - Implement `f(t, x, u, p)` with same chain dynamics but time-varying reference from u
    - _Requirements: 2.1, 2.2_

  - [x] 3.2 Implement `get_system_and_embedding()` and `euler_step`
    - Return system + natural embedding for interval arithmetic
    - Euler step on interval state with reference input and parameter interval
    - _Requirements: 2.3, 2.5_

  - [x] 3.3 Implement `create_scenarios()` reusing `_CANDIDATE_THETA` from `crazyflie_chain_controllers.py`
    - Construct 4 Scenario objects with shared embedding system and per-controller parameter intervals
    - _Requirements: 2.4_

  - [x] 3.4 Implement `nominal_waypoints(K)` — straight-line path from P0 to P1
    - _Requirements: 3.1_

  - [x]* 3.5 Write tests for WaypointTrackingSystem dynamics and scenarios (`test_crazyflie_waypoint_controllers.py`)
    - NumPy oracle comparison for dynamics (parametrized over candidates and seeds)
    - Embedding and step sanity checks
    - Scenario creation and naming tests
    - Nominal waypoint shape, endpoints, and altitude tests
    - _Requirements: 6.5_

- [x] 4. Implement waypoint-sequence optimizer
  - [x] 4.1 Implement `_offset_bounds(K)` and `_project_offsets(offsets, K)` — arena-scale box constraints
    - Define per-axis bounds for waypoint offsets
    - Clip offsets to feasible region
    - _Requirements: 3.2_

  - [x] 4.2 Implement `_waypoints_to_u_seq(offsets, x0_state4x3, ...)` — convert offsets to full reference trajectory
    - Add offsets to nominal waypoints, call `build_mission_reference`
    - _Requirements: 3.3_

  - [x] 4.3 Implement `_propagate_all_scenarios(x0_ivl, u_seq, scenarios, dt)` — propagate interval states for all candidates
    - Return per-scenario interval histories for loss computation
    - _Requirements: 3.4_

  - [x] 4.4 Implement `waypoint_separation_loss(offsets, x0_ivl, scenarios, ...)` — sum-of-widths loss
    - Propagate all scenarios, compute aggregate width of reachable output sets
    - Ensure nonnegative, differentiable w.r.t. offsets
    - _Requirements: 3.4_

  - [x] 4.5 Implement `optimize_waypoints_gpu(x0_ivl, scenarios, ...)` — multi-start projected gradient descent
    - Random initialization of offsets, projected onto feasible set each iteration
    - Return best offsets, best loss, and all restart results
    - _Requirements: 3.5, 3.6_

  - [x]* 4.6 Write tests for optimizer components
    - Projection bounds verification
    - Separation loss nonnegative and differentiable
    - Mission reference shape and waypoint-reaching tests
    - _Requirements: 6.6_

- [x] 5. Checkpoint — Verify optimizer and system tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement simulation and discrimination pipeline
  - [x] 6.1 Implement `simulate_true_trajectory(x0_point, u_seq, theta, dt)` — point-state rollout under a given controller
    - Euler integration of WaypointTrackingSystem dynamics at a single point
    - Return (T, obs_dim) observed trajectory
    - _Requirements: 4.1_

  - [x] 6.2 Implement `discriminate_controller_waypoint(x0_ivl, u_seq, observed, scenarios, dt, w_bar)` — interval-based discrimination
    - Propagate interval states per scenario, check containment of observed data with w_bar slack
    - Return survivors, falsified flags, and fail steps
    - _Requirements: 4.2, 4.3, 4.4_

  - [x]* 6.3 Write regression guard test `test_true_controller_always_survives`
    - Parametrized over all 4 candidates as ground truth
    - Verify the generating controller is never falsified by its own trajectory
    - _Requirements: 6.7_

- [x] 7. Implement end-to-end scripts
  - [x] 7.1 Create `synthesize_waypoints.py` — run optimizer, save results/synthesized_waypoints.npz
    - Load scenarios, run `optimize_waypoints_gpu`, sanity-check discrimination for all 4 candidates
    - _Requirements: 5.1_

  - [x] 7.2 Create `run_qps_waypoint_validation.py` — deploy through real QPS simulator via `q.set_poses()`
    - Settle to P0, fly K waypoints recording observed state at each dt step
    - Save observed trajectory NPZ and flight video
    - _Requirements: 5.2_

  - [x] 7.3 Create `run_waypoint_discrimination.py` — feed real QPS trajectory into discriminator
    - Load observed trajectory and synthesized waypoints, run `discriminate_controller_waypoint`
    - Save discrimination result JSON
    - _Requirements: 5.3_

  - [x] 7.4 Create `plot_waypoint_separation.py` — visualization script
    - Pairwise overlap volume over time, per-controller reachable position tubes, 3D waypoint plot
    - _Requirements: 5.4_

- [x] 8. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- The implementation uses Python with JAX for differentiable computation and immrax for interval arithmetic
- All 129 tests across `test_crazyflie_waypoint_trajectory.py` and `test_crazyflie_waypoint_controllers.py` pass
