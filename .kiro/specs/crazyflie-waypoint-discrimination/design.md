# Design Document: Crazyflie Waypoint Discrimination

## Overview

This feature adds a legitimate waypoint-based controller discrimination capability to the `examples/adaptive_spoofing/` pipeline. Instead of injecting a stealthy position-sensor spoof bias to discriminate which of 4 candidate outer-loop controllers is flying a Crazyflie, the system designs a sequence of commanded waypoints that produces the same algebraic discrimination signal through direct, legitimate flight control.

The key insight is that in `ChainControllerSystem.f`, the tracking error `e_p = (pos + b) - rp` is algebraically identical to `e_p = pos - (rp - b)` — corrupting what the controller perceives (spoof) and moving the commanded target (waypoint) enter the closed-loop ODE identically. This means the existing `separation_loss`/multistep/refinement/`discriminate_controller` machinery already computes the right dynamics for a legitimate-waypoint framing, with the reference now genuinely time-varying.

The implementation provides: (1) a JAX-differentiable transcription of QPS's polynomial trajectory generator (`nominal_traj.py`), (2) a `WaypointTrackingSystem` that drives the chain via time-varying reference instead of spoof bias, (3) a waypoint-sequence optimizer that maximizes controller separation, and (4) real QPS deployment/validation via `q.set_poses()`.

## Architecture

```mermaid
graph TD
    subgraph "JAX / immrax-venv"
        TG[crazyflie_waypoint_trajectory.py<br/>Polynomial trajectory generator]
        WC[crazyflie_waypoint_controllers.py<br/>WaypointTrackingSystem + Optimizer]
        CC[crazyflie_chain_controllers.py<br/>Existing: separation_loss, discriminate_controller]
    end

    subgraph "Scripts"
        SW[synthesize_waypoints.py<br/>Optimize waypoint offsets]
        RD[run_waypoint_discrimination.py<br/>Offline discrimination]
        PL[plot_waypoint_separation.py<br/>Visualization]
    end

    subgraph "QPS / qps-venv"
        QV[run_qps_waypoint_validation.py<br/>Real deployment via q.set_poses]
        QPS_SIM[QPS Simulator<br/>nominal_traj.py + snap controller]
    end

    TG --> WC
    CC --> WC
    WC --> SW
    SW --> QV
    SW --> RD
    QV --> RD
    RD --> PL
    QV --> QPS_SIM
