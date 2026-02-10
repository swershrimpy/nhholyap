"""
Demo script for separating input optimization.

This script demonstrates how to use the separating_input_optimizer module
to find control inputs that maximize separation between different fault scenarios.
"""

import jax.numpy as jnp
import immrax as irx
from faulty_planar_multirotor import FaultyPlanarMultirotor
from separating_input_optimizer import (
    SeparatingInputOptimizer,
    optimize_separating_input_multistart,
)
import time


def demo_two_scenarios():
    """Demo with two fault scenarios: nominal vs. 50% thrust loss."""
    print("=" * 70)
    print("DEMO 1: Two Fault Scenarios (Nominal vs. 50% Thrust Loss)")
    print("=" * 70)

    # System setup
    sys = FaultyPlanarMultirotor()

    # Initial state interval (larger uncertainty to create overlap)
    x0 = irx.icentpert(
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # Center: at origin, zero velocity/angle
        jnp.array([0.3, 0.3, 0.2, 0.2, 0.1]),  # Larger perturbations
    )

    # No disturbance
    w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))

    # Fault scenarios
    p_nominal = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
    p_fault = irx.icentpert(jnp.array([0.5, 1.0]), jnp.zeros(2))

    print("\nFault Scenarios:")
    print(f"  - Nominal: thrust=100%, angular=100%")
    print(f"  - Fault:   thrust=50%, angular=100%")

    # Create optimizer
    optimizer = SeparatingInputOptimizer(
        system=sys,
        fault_parameters=[p_nominal, p_fault],
        x0_interval=x0,
        w_interval=w,
        dt=0.02,
        num_steps=10,  # Fewer steps to create more overlap
        state_slice=slice(0, 2),  # Only consider position (px, py)
    )

    # Try hover input (baseline)
    print("\n--- Baseline: Hover Input u = [9.81, 0.0] ---")
    u_hover = jnp.array([9.81, 0.0])
    loss_hover, intervals_hover = optimizer.evaluate(u_hover)
    print(f"Overlap size: {float(loss_hover):.6f}")

    # Optimize
    print("\n--- Optimizing Separating Input ---")
    t0 = time.time()
    u_opt, loss_opt = optimizer.optimize(
        u_initial=u_hover,
        learning_rate=1e-1,
        num_iterations=100,
        verbose=False,
    )
    t1 = time.time()

    print(f"Optimal control: u = [{float(u_opt[0]):.3f}, {float(u_opt[1]):.3f}]")
    print(f"Optimal overlap: {float(loss_opt):.6f}")
    if float(loss_hover) > 1e-8:
        print(f"Improvement: {(1 - float(loss_opt)/float(loss_hover))*100:.1f}% reduction in overlap")
    else:
        print(f"Note: Baseline already had zero overlap (perfect separation)")
    print(f"Time: {t1-t0:.3f}s")

    # Evaluate final intervals
    _, intervals_opt = optimizer.evaluate(u_opt)
    print("\nFinal position intervals (optimized):")
    for i, interval in enumerate(intervals_opt):
        scenario = "Nominal" if i == 0 else "Fault"
        print(f"  {scenario}: px=[{float(interval.lower[0]):.3f}, {float(interval.upper[0]):.3f}], "
              f"py=[{float(interval.lower[1]):.3f}, {float(interval.upper[1]):.3f}]")


def demo_three_scenarios():
    """Demo with three fault scenarios."""
    print("\n" + "=" * 70)
    print("DEMO 2: Three Fault Scenarios (100%, 70%, 40% Thrust)")
    print("=" * 70)

    sys = FaultyPlanarMultirotor()

    x0 = irx.icentpert(
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        jnp.array([0.3, 0.3, 0.2, 0.2, 0.1]),  # Larger uncertainty
    )

    w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))

    # Three scenarios with different thrust levels
    p_100 = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
    p_70 = irx.icentpert(jnp.array([0.7, 1.0]), jnp.zeros(2))
    p_40 = irx.icentpert(jnp.array([0.4, 1.0]), jnp.zeros(2))

    print("\nFault Scenarios:")
    print(f"  - Scenario 1: thrust=100%")
    print(f"  - Scenario 2: thrust=70%")
    print(f"  - Scenario 3: thrust=40%")

    # Use multi-start optimization for better results
    print("\n--- Multi-Start Optimization (3 restarts) ---")
    t0 = time.time()
    u_opt, loss_opt = optimize_separating_input_multistart(
        system=sys,
        fault_parameters=[p_100, p_70, p_40],
        x0_interval=x0,
        w_interval=w,
        dt=0.02,
        num_steps=15,
        num_restarts=3,
        learning_rate=1e-1,
        num_iterations=80,
        state_slice=slice(0, 2),
        random_key=42,
    )
    t1 = time.time()

    print(f"Optimal control: u = [{float(u_opt[0]):.3f}, {float(u_opt[1]):.3f}]")
    print(f"Total pairwise overlap: {float(loss_opt):.6f}")
    print(f"Time: {t1-t0:.3f}s")

    # Evaluate to get final intervals
    optimizer = SeparatingInputOptimizer(
        system=sys,
        fault_parameters=[p_100, p_70, p_40],
        x0_interval=x0,
        w_interval=w,
        dt=0.02,
        num_steps=15,
        state_slice=slice(0, 2),
    )
    _, intervals = optimizer.evaluate(u_opt)

    print("\nFinal position intervals:")
    labels = ["100% thrust", "70% thrust", "40% thrust"]
    for i, (interval, label) in enumerate(zip(intervals, labels)):
        print(f"  {label}: px=[{float(interval.lower[0]):.3f}, {float(interval.upper[0]):.3f}], "
              f"py=[{float(interval.lower[1]):.3f}, {float(interval.upper[1]):.3f}]")


def demo_uncertain_fault():
    """Demo with uncertain fault parameters."""
    print("\n" + "=" * 70)
    print("DEMO 3: Uncertain Fault Parameters")
    print("=" * 70)

    sys = FaultyPlanarMultirotor()

    x0 = irx.icentpert(
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        jnp.array([0.3, 0.3, 0.2, 0.2, 0.1]),  # Larger uncertainty
    )

    w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))

    # Nominal and uncertain fault
    p_nominal = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
    # Fault with uncertainty: thrust could be anywhere in [0.3, 0.7]
    p_fault_uncertain = irx.icentpert(jnp.array([0.5, 1.0]), jnp.array([0.2, 0.0]))

    print("\nFault Scenarios:")
    print(f"  - Nominal: thrust=100% (certain)")
    print(f"  - Fault:   thrust=50% ± 20% (uncertain)")

    optimizer = SeparatingInputOptimizer(
        system=sys,
        fault_parameters=[p_nominal, p_fault_uncertain],
        x0_interval=x0,
        w_interval=w,
        dt=0.02,
        num_steps=10,  # Fewer steps for more overlap
        state_slice=slice(0, 2),
    )

    print("\n--- Optimization ---")
    u_opt, loss_opt = optimizer.optimize(
        u_initial=jnp.array([9.81, 0.0]),
        learning_rate=1e-1,
        num_iterations=100,
    )

    print(f"Optimal control: u = [{float(u_opt[0]):.3f}, {float(u_opt[1]):.3f}]")
    print(f"Overlap: {float(loss_opt):.6f}")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "   Separating Input Optimization Demos".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    demo_two_scenarios()
    demo_three_scenarios()
    demo_uncertain_fault()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
