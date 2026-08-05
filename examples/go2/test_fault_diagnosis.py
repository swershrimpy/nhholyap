#!/usr/bin/env python3
"""
Test script for fault diagnosis system

This script demonstrates the fault diagnosis capabilities without requiring
actual robot hardware by using simulated data.
"""

import numpy as np
import matplotlib.pyplot as plt
from fault_diagnosis import FaultDiagnosisGo2, FaultParameters, OnlineFaultDiagnosis
import os

def generate_test_trajectory(N=20, dt=0.5, trajectory_type='circular'):
    """
    Generate test trajectory

    Args:
        N: Number of steps
        dt: Time step
        trajectory_type: 'circular', 'straight', or 'figure8'

    Returns:
        nominal_traj, nominal_input
    """
    nominal_traj = np.zeros((3, N+1))
    nominal_input = np.zeros((3, N))

    if trajectory_type == 'circular':
        # Circular trajectory
        radius = 2.0
        angular_vel = 0.3
        for t in range(N+1):
            angle = angular_vel * t * dt
            nominal_traj[0, t] = radius * np.cos(angle)
            nominal_traj[1, t] = radius * np.sin(angle)
            nominal_traj[2, t] = angle + np.pi/2

            if t < N:
                nominal_input[0, t] = radius * angular_vel  # Forward velocity
                nominal_input[1, t] = 0.0
                nominal_input[2, t] = angular_vel

    elif trajectory_type == 'straight':
        # Straight line with turns
        for t in range(N+1):
            if t < N // 3:
                nominal_traj[0, t] = 0.5 * t * dt
                nominal_traj[1, t] = 0.0
                nominal_traj[2, t] = 0.0
                if t < N:
                    nominal_input[0, t] = 0.5
                    nominal_input[1, t] = 0.0
                    nominal_input[2, t] = 0.0
            elif t < 2 * N // 3:
                t_rel = t - N // 3
                angle = 0.4 * t_rel * dt
                nominal_traj[0, t] = nominal_traj[0, N//3]
                nominal_traj[1, t] = nominal_traj[1, N//3]
                nominal_traj[2, t] = angle
                if t < N:
                    nominal_input[0, t] = 0.0
                    nominal_input[1, t] = 0.0
                    nominal_input[2, t] = 0.4
            else:
                t_rel = t - 2 * N // 3
                nominal_traj[0, t] = nominal_traj[0, 2*N//3] + 0.5 * t_rel * dt * np.cos(nominal_traj[2, 2*N//3])
                nominal_traj[1, t] = nominal_traj[1, 2*N//3] + 0.5 * t_rel * dt * np.sin(nominal_traj[2, 2*N//3])
                nominal_traj[2, t] = nominal_traj[2, 2*N//3]
                if t < N:
                    nominal_input[0, t] = 0.5
                    nominal_input[1, t] = 0.0
                    nominal_input[2, t] = 0.0

    elif trajectory_type == 'figure8':
        # Figure-8 trajectory
        for t in range(N+1):
            angle = 0.2 * t * dt
            nominal_traj[0, t] = 2.0 * np.sin(angle)
            nominal_traj[1, t] = 1.0 * np.sin(2 * angle)
            nominal_traj[2, t] = np.arctan2(
                2.0 * np.cos(2 * angle),
                2.0 * np.cos(angle)
            )

            if t < N:
                # Approximate velocities
                nominal_input[0, t] = 0.6
                nominal_input[1, t] = 0.0
                nominal_input[2, t] = 0.4 * np.cos(angle)

    return nominal_traj, nominal_input


def test_actuator_fault():
    """Test actuator fault diagnosis"""
    print("\n" + "="*70)
    print("TEST 1: ACTUATOR FAULT (Reduced Turn Rate)")
    print("="*70)

    # Generate trajectory
    N = 30
    dt = 0.5
    nominal_traj, nominal_input = generate_test_trajectory(N, dt, 'circular')

    # Observation matrix
    C = np.array([[-0.0329,  0.9805, -0.1938],
                  [-0.8052, -0.5551, -0.2087],
                  [-0.8518, -0.4816,  0.2061]])
    state_offset = np.array([-0.46, 0.34, 0.0])

    # Create fault diagnosis instance
    fd = FaultDiagnosisGo2(nominal_traj, nominal_input, C, state_offset, dt)

    # Inject actuator fault
    true_faults = FaultParameters(
        actuator_effectiveness=0.6,  # 40% reduction in turn rate
        sensor_scale=1.0,
        sensor_bias=0.0
    )

    print("\nInjected Fault:")
    print(f"  Actuator effectiveness: {true_faults.actuator_effectiveness:.3f} (40% reduction)")

    # Generate faulty observations
    faulty_traj = fd.simulate_faulty_trajectory(true_faults)
    measured_traj = fd.apply_sensor_fault(faulty_traj, true_faults)
    measured_obs = fd.compute_observations(measured_traj)

    # Add realistic noise
    measured_obs += 0.02 * np.random.randn(*measured_obs.shape)

    # Diagnose
    errors, estimated_faults = fd.diagnose(measured_obs, verbose=True)

    # Compute true errors
    true_errors = fd.compute_fault_induced_errors(true_faults)

    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    print(f"True actuator effectiveness:      {true_faults.actuator_effectiveness:.4f}")
    print(f"Estimated actuator effectiveness: {estimated_faults.actuator_effectiveness:.4f}")
    print(f"\nTrue position error:     ({true_errors[0]:+.4f}, {true_errors[1]:+.4f}) m")
    print(f"Estimated position error: ({errors[0]:+.4f}, {errors[1]:+.4f}) m")
    print(f"\nTrue heading error:      {true_errors[2]:+.4f} rad ({np.degrees(true_errors[2]):+.2f}°)")
    print(f"Estimated heading error: {errors[2]:+.4f} rad ({np.degrees(errors[2]):+.2f}°)")

    # Plot trajectories
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(nominal_traj[0, :], nominal_traj[1, :], 'b-', linewidth=2, label='Nominal')
    plt.plot(faulty_traj[0, :], faulty_traj[1, :], 'r--', linewidth=2, label='Faulty')
    plt.scatter(nominal_traj[0, 0], nominal_traj[1, 0], s=100, c='g', marker='o', label='Start')
    plt.scatter(nominal_traj[0, -1], nominal_traj[1, -1], s=100, c='k', marker='x', label='Goal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Comparison (Actuator Fault)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    t_array = np.arange(N+1) * dt
    plt.plot(t_array, np.degrees(nominal_traj[2, :]), 'b-', linewidth=2, label='Nominal')
    plt.plot(t_array, np.degrees(faulty_traj[2, :]), 'r--', linewidth=2, label='Faulty')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (deg)')
    plt.title('Heading Evolution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('fault_diagnosis_actuator.pdf')
    print(f"\nPlot saved to: fault_diagnosis_actuator.pdf")


def test_sensor_fault():
    """Test sensor fault diagnosis"""
    print("\n" + "="*70)
    print("TEST 2: SENSOR FAULT (IMU Drift)")
    print("="*70)

    # Generate trajectory
    N = 30
    dt = 0.5
    nominal_traj, nominal_input = generate_test_trajectory(N, dt, 'circular')

    # Observation matrix
    C = np.array([[-0.0329,  0.9805, -0.1938],
                  [-0.8052, -0.5551, -0.2087],
                  [-0.8518, -0.4816,  0.2061]])
    state_offset = np.array([-0.46, 0.34, 0.0])

    # Create fault diagnosis instance
    fd = FaultDiagnosisGo2(nominal_traj, nominal_input, C, state_offset, dt)

    # Inject sensor fault
    true_faults = FaultParameters(
        actuator_effectiveness=1.0,
        sensor_scale=1.08,   # 8% scale error
        sensor_bias=0.15     # ~8.6 degree bias
    )

    print("\nInjected Fault:")
    print(f"  Sensor scale: {true_faults.sensor_scale:.3f} (8% error)")
    print(f"  Sensor bias:  {true_faults.sensor_bias:.4f} rad ({np.degrees(true_faults.sensor_bias):.2f}°)")

    # Generate faulty observations
    faulty_traj = fd.simulate_faulty_trajectory(true_faults)
    measured_traj = fd.apply_sensor_fault(faulty_traj, true_faults)
    measured_obs = fd.compute_observations(measured_traj)

    # Add noise
    measured_obs += 0.02 * np.random.randn(*measured_obs.shape)

    # Diagnose
    errors, estimated_faults = fd.diagnose(measured_obs, verbose=True)

    # Compute true errors
    true_errors = fd.compute_fault_induced_errors(true_faults)

    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    print(f"True sensor scale:      {true_faults.sensor_scale:.4f}")
    print(f"Estimated sensor scale: {estimated_faults.sensor_scale:.4f}")
    print(f"\nTrue sensor bias:      {true_faults.sensor_bias:+.4f} rad ({np.degrees(true_faults.sensor_bias):+.2f}°)")
    print(f"Estimated sensor bias: {estimated_faults.sensor_bias:+.4f} rad ({np.degrees(estimated_faults.sensor_bias):+.2f}°)")

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    t_array = np.arange(N+1) * dt
    plt.plot(t_array, np.degrees(faulty_traj[2, :]), 'b-', linewidth=2, label='True Heading')
    plt.plot(t_array, np.degrees(measured_traj[2, :]), 'r--', linewidth=2, label='Measured Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (deg)')
    plt.title('Sensor Fault: Measured vs True Heading')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    heading_error = measured_traj[2, :] - faulty_traj[2, :]
    plt.plot(t_array, np.degrees(heading_error), 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Measurement Error (deg)')
    plt.title('Sensor Measurement Error Over Time')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('fault_diagnosis_sensor.pdf')
    print(f"\nPlot saved to: fault_diagnosis_sensor.pdf")


def test_combined_faults():
    """Test diagnosis with both actuator and sensor faults"""
    print("\n" + "="*70)
    print("TEST 3: COMBINED FAULTS (Actuator + Sensor)")
    print("="*70)

    # Generate trajectory
    N = 30
    dt = 0.5
    nominal_traj, nominal_input = generate_test_trajectory(N, dt, 'figure8')

    # Observation matrix
    C = np.array([[-0.0329,  0.9805, -0.1938],
                  [-0.8052, -0.5551, -0.2087],
                  [-0.8518, -0.4816,  0.2061]])
    state_offset = np.array([-0.46, 0.34, 0.0])

    # Create fault diagnosis instance
    fd = FaultDiagnosisGo2(nominal_traj, nominal_input, C, state_offset, dt)

    # Inject both faults
    true_faults = FaultParameters(
        actuator_effectiveness=0.75,  # 25% reduction
        sensor_scale=1.05,            # 5% scale error
        sensor_bias=0.08              # ~4.6 degree bias
    )

    print("\nInjected Faults:")
    print(f"  Actuator effectiveness: {true_faults.actuator_effectiveness:.3f} (25% reduction)")
    print(f"  Sensor scale: {true_faults.sensor_scale:.3f} (5% error)")
    print(f"  Sensor bias:  {true_faults.sensor_bias:.4f} rad ({np.degrees(true_faults.sensor_bias):.2f}°)")

    # Generate faulty observations
    faulty_traj = fd.simulate_faulty_trajectory(true_faults)
    measured_traj = fd.apply_sensor_fault(faulty_traj, true_faults)
    measured_obs = fd.compute_observations(measured_traj)

    # Add noise
    measured_obs += 0.02 * np.random.randn(*measured_obs.shape)

    # Diagnose
    errors, estimated_faults = fd.diagnose(measured_obs, verbose=True)

    # Compute true errors
    true_errors = fd.compute_fault_induced_errors(true_faults)

    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    print(f"True actuator effectiveness:      {true_faults.actuator_effectiveness:.4f}")
    print(f"Estimated actuator effectiveness: {estimated_faults.actuator_effectiveness:.4f}")
    print(f"\nTrue sensor scale:      {true_faults.sensor_scale:.4f}")
    print(f"Estimated sensor scale: {estimated_faults.sensor_scale:.4f}")
    print(f"\nTrue sensor bias:      {true_faults.sensor_bias:+.4f} rad ({np.degrees(true_faults.sensor_bias):+.2f}°)")
    print(f"Estimated sensor bias: {estimated_faults.sensor_bias:+.4f} rad ({np.degrees(estimated_faults.sensor_bias):+.2f}°)")
    print(f"\nTrue position error:     ({true_errors[0]:+.4f}, {true_errors[1]:+.4f}) m")
    print(f"Estimated position error: ({errors[0]:+.4f}, {errors[1]:+.4f}) m")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(nominal_traj[0, :], nominal_traj[1, :], 'b-', linewidth=2, label='Nominal')
    plt.plot(faulty_traj[0, :], faulty_traj[1, :], 'r--', linewidth=2, label='Faulty')
    plt.scatter(nominal_traj[0, 0], nominal_traj[1, 0], s=100, c='g', marker='o', label='Start')
    plt.scatter(nominal_traj[0, -1], nominal_traj[1, -1], s=100, c='k', marker='x', label='Goal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Combined Faults: Trajectory Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('fault_diagnosis_combined.pdf')
    print(f"\nPlot saved to: fault_diagnosis_combined.pdf")


def test_online_diagnosis():
    """Test online fault diagnosis"""
    print("\n" + "="*70)
    print("TEST 4: ONLINE FAULT DIAGNOSIS")
    print("="*70)

    # Generate trajectory
    N = 40
    dt = 0.5
    nominal_traj, nominal_input = generate_test_trajectory(N, dt, 'circular')

    # Observation matrix
    C = np.array([[-0.0329,  0.9805, -0.1938],
                  [-0.8052, -0.5551, -0.2087],
                  [-0.8518, -0.4816,  0.2061]])
    state_offset = np.array([-0.46, 0.34, 0.0])

    # Create fault diagnosis instance
    fd = FaultDiagnosisGo2(nominal_traj, nominal_input, C, state_offset, dt)

    # Simulate fault that develops over time
    fault_evolution = []
    online_fd = OnlineFaultDiagnosis(nominal_traj, nominal_input, C, state_offset,
                                     window_size=15, dt=dt)

    print("\nSimulating fault that develops over time...")

    for t in range(N+1):
        # Fault develops gradually
        if t < N // 2:
            alpha = 1.0  # No fault initially
        else:
            # Gradual degradation
            alpha = 1.0 - 0.5 * (t - N//2) / (N - N//2)

        fault = FaultParameters(actuator_effectiveness=alpha, sensor_scale=1.0, sensor_bias=0.0)
        fault_evolution.append(fault)

        # Generate observation
        faulty_traj_t = fd.simulate_faulty_trajectory(fault)
        measured_traj_t = fd.apply_sensor_fault(faulty_traj_t, fault)
        obs = C @ (measured_traj_t[:, t] + state_offset)
        obs += 0.01 * np.random.randn(3)

        # Update online diagnosis
        errors, est_fault = online_fd.add_observation(obs, t)

        if errors is not None and t % 5 == 0:
            print(f"\nTime step {t}/{N}:")
            print(f"  True actuator effectiveness: {alpha:.3f}")
            print(f"  Estimated effectiveness:     {est_fault.actuator_effectiveness:.3f}")
            print(f"  Estimated errors: px={errors[0]:+.3f}, py={errors[1]:+.3f}, yaw={errors[2]:+.3f}")

    # Plot evolution
    t_array = np.arange(N+1) * dt
    true_alpha = [f.actuator_effectiveness for f in fault_evolution]

    estimated_alpha = [1.0] * (N+1)
    for i, fault in enumerate(online_fd.fault_history):
        idx = online_fd.time_history[i] if i < len(online_fd.time_history) else -1
        if idx >= 0 and idx < N+1:
            estimated_alpha[idx] = fault.actuator_effectiveness

    plt.figure(figsize=(10, 5))
    plt.plot(t_array, true_alpha, 'b-', linewidth=2, label='True Actuator Effectiveness')
    plt.plot(t_array, estimated_alpha, 'r--', linewidth=2, label='Estimated (Online)')
    plt.xlabel('Time (s)')
    plt.ylabel('Actuator Effectiveness')
    plt.title('Online Fault Diagnosis: Tracking Evolving Fault')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fault_diagnosis_online.pdf')
    print(f"\nPlot saved to: fault_diagnosis_online.pdf")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    print("\n" + "="*70)
    print("UNITREE GO2 FAULT DIAGNOSIS TEST SUITE")
    print("="*70)

    # Run all tests
    test_actuator_fault()
    test_sensor_fault()
    test_combined_faults()
    test_online_diagnosis()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
    print("\nGenerated plots:")
    print("  - fault_diagnosis_actuator.pdf")
    print("  - fault_diagnosis_sensor.pdf")
    print("  - fault_diagnosis_combined.pdf")
    print("  - fault_diagnosis_online.pdf")
