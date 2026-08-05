#!/usr/bin/env python3
"""
Simple example demonstrating fault diagnosis API

This example shows how to use the fault diagnosis system to detect
actuator and sensor faults from observations.
"""

import numpy as np
from fault_diagnosis import FaultDiagnosisGo2, FaultParameters


def main():
    print("="*70)
    print("SIMPLE FAULT DIAGNOSIS EXAMPLE")
    print("="*70)

    # Step 1: Create a simple nominal trajectory
    print("\nStep 1: Creating nominal trajectory...")

    N = 25  # Number of time steps
    dt = 0.5  # Time step in seconds

    # Circular trajectory
    nominal_traj = np.zeros((3, N+1))
    nominal_input = np.zeros((3, N))

    radius = 1.5
    omega = 0.4

    for t in range(N+1):
        angle = omega * t * dt
        nominal_traj[0, t] = radius * np.cos(angle)  # x
        nominal_traj[1, t] = radius * np.sin(angle)  # y
        nominal_traj[2, t] = angle + np.pi/2         # heading

        if t < N:
            nominal_input[0, t] = radius * omega  # Forward velocity
            nominal_input[1, t] = 0.0              # Lateral velocity
            nominal_input[2, t] = omega            # Yaw rate

    print(f"  Trajectory: {N} steps, circular path with radius {radius} m")

    # Step 2: Define observation model
    print("\nStep 2: Setting up observation model...")

    C = np.array([[-0.0329,  0.9805, -0.1938],
                  [-0.8052, -0.5551, -0.2087],
                  [-0.8518, -0.4816,  0.2061]])

    state_offset = np.array([-0.46, 0.34, 0.0])

    print(f"  Observation dimension: {C.shape[0]}")
    print(f"  State dimension: {C.shape[1]}")

    # Step 3: Create fault diagnosis instance
    print("\nStep 3: Initializing fault diagnosis...")

    fd = FaultDiagnosisGo2(
        nominal_traj=nominal_traj,
        nominal_input=nominal_input,
        C_matrix=C,
        state_offset=state_offset,
        dt=dt
    )

    print("  Fault diagnosis system ready")

    # Step 4: Simulate a fault scenario
    print("\nStep 4: Simulating robot with faults...")

    # Define the faults we want to simulate
    true_fault = FaultParameters(
        actuator_effectiveness=0.65,  # 35% reduction in turn rate
        sensor_scale=1.06,             # 6% scale error in IMU
        sensor_bias=0.12               # ~6.9 degree bias in IMU
    )

    print(f"  Injecting ACTUATOR fault: {(1-true_fault.actuator_effectiveness)*100:.0f}% turn rate reduction")
    print(f"  Injecting SENSOR fault: {(true_fault.sensor_scale-1)*100:.0f}% scale error, {np.degrees(true_fault.sensor_bias):.1f}° bias")

    # Simulate what the robot would actually do with these faults
    faulty_trajectory = fd.simulate_faulty_trajectory(true_fault)
    measured_trajectory = fd.apply_sensor_fault(faulty_trajectory, true_fault)
    measured_observations = fd.compute_observations(measured_trajectory)

    # Add some realistic sensor noise
    np.random.seed(42)
    measured_observations += 0.02 * np.random.randn(*measured_observations.shape)

    print(f"  Generated {measured_observations.shape[1]} observations")

    # Step 5: Diagnose the faults
    print("\nStep 5: Running fault diagnosis algorithm...")
    print("-" * 70)

    errors, estimated_fault = fd.diagnose(measured_observations, verbose=True)

    # Step 6: Compare with ground truth
    print("\nStep 6: Comparing with ground truth...")
    print("="*70)

    true_errors = fd.compute_fault_induced_errors(true_fault)

    print("\nACTUATOR FAULT:")
    print(f"  True:      {true_fault.actuator_effectiveness:.4f}")
    print(f"  Estimated: {estimated_fault.actuator_effectiveness:.4f}")
    print(f"  Error:     {abs(true_fault.actuator_effectiveness - estimated_fault.actuator_effectiveness):.4f}")

    print("\nSENSOR SCALE:")
    print(f"  True:      {true_fault.sensor_scale:.4f}")
    print(f"  Estimated: {estimated_fault.sensor_scale:.4f}")
    print(f"  Error:     {abs(true_fault.sensor_scale - estimated_fault.sensor_scale):.4f}")

    print("\nSENSOR BIAS:")
    print(f"  True:      {true_fault.sensor_bias:.4f} rad ({np.degrees(true_fault.sensor_bias):.2f}°)")
    print(f"  Estimated: {estimated_fault.sensor_bias:.4f} rad ({np.degrees(estimated_fault.sensor_bias):.2f}°)")
    print(f"  Error:     {abs(true_fault.sensor_bias - estimated_fault.sensor_bias):.4f} rad")

    print("\nFAULT-INDUCED POSITION ERROR:")
    print(f"  True:      ({true_errors[0]:+.4f}, {true_errors[1]:+.4f}) m")
    print(f"  Estimated: ({errors[0]:+.4f}, {errors[1]:+.4f}) m")
    print(f"  Magnitude: {np.linalg.norm(errors[:2]):.4f} m")

    print("\nFAULT-INDUCED HEADING ERROR:")
    print(f"  True:      {true_errors[2]:+.4f} rad ({np.degrees(true_errors[2]):+.2f}°)")
    print(f"  Estimated: {errors[2]:+.4f} rad ({np.degrees(errors[2]):+.2f}°)")

    print("\n" + "="*70)

    # Step 7: Output the main result
    print("\nFINAL OUTPUT (as requested):")
    print("-" * 70)
    print(f"Fault diagnosis output [px, py, yaw]:")
    print(f"  px   = {errors[0]:+.4f} m")
    print(f"  py   = {errors[1]:+.4f} m")
    print(f"  yaw  = {errors[2]:+.4f} rad ({np.degrees(errors[2]):+.2f}°)")
    print("="*70)

    # Step 8: Interpretation
    print("\nINTERPRETATION:")
    print("-" * 70)

    if estimated_fault.actuator_effectiveness < 0.95:
        reduction = (1.0 - estimated_fault.actuator_effectiveness) * 100
        print(f"✗ ACTUATOR FAULT: Turn rate reduced by {reduction:.1f}%")
        if reduction > 30:
            print(f"  → CRITICAL: Severe actuator degradation")
        elif reduction > 15:
            print(f"  → WARNING: Significant actuator degradation")
        else:
            print(f"  → NOTICE: Minor actuator degradation")
    else:
        print(f"✓ No actuator fault detected")

    if abs(estimated_fault.sensor_scale - 1.0) > 0.05 or abs(estimated_fault.sensor_bias) > 0.05:
        print(f"✗ SENSOR FAULT: IMU drift detected")
        if abs(estimated_fault.sensor_scale - 1.0) > 0.05:
            scale_error = (estimated_fault.sensor_scale - 1.0) * 100
            print(f"  → Scale error: {scale_error:+.2f}%")
        if abs(estimated_fault.sensor_bias) > 0.05:
            print(f"  → Bias: {np.degrees(estimated_fault.sensor_bias):+.2f}°")
            if abs(estimated_fault.sensor_bias) > 0.15:
                print(f"  → CRITICAL: Large IMU bias, recalibration required")
    else:
        print(f"✓ No sensor fault detected")

    error_magnitude = np.linalg.norm(errors[:2])
    if error_magnitude > 1.0:
        print(f"\n⚠ Position error ({error_magnitude:.2f} m) exceeds safe threshold")
        print(f"  → Recommend stopping and performing maintenance")
    elif error_magnitude > 0.5:
        print(f"\n⚠ Position error ({error_magnitude:.2f} m) is significant")
        print(f"  → Recommend trajectory correction or replanning")

    print("="*70)

    return errors, estimated_fault


if __name__ == "__main__":
    errors, faults = main()

    print("\n✓ Example completed successfully!")
    print(f"\nReturned values:")
    print(f"  errors = {errors}")
    print(f"  faults.actuator_effectiveness = {faults.actuator_effectiveness:.4f}")
    print(f"  faults.sensor_scale = {faults.sensor_scale:.4f}")
    print(f"  faults.sensor_bias = {faults.sensor_bias:.4f}")
