"""
Separating Input Optimizer for Unitree Go2 Fault Diagnosis

This module implements active fault diagnosis by computing optimal control inputs
that maximize separation between reachable sets of different fault scenarios:
1. Nominal (no fault)
2. Sensor fault (IMU drift)
3. Actuator fault (reduced turn rate)

Follows the pattern from faulty_multirotor/separating_input_optimizer.py
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import List, Tuple, Optional, Dict
import immrax as irx
from functools import partial
from dataclasses import dataclass
import numpy as np


@dataclass
class Go2FaultScenario:
    """Container for a fault scenario with parameters"""
    name: str
    actuator_effectiveness: irx.Interval  # alpha in [0,1]
    sensor_scale: irx.Interval           # a near 1.0
    sensor_bias: irx.Interval            # b in radians

    def to_parameter_interval(self) -> irx.Interval:
        """Stack all parameters into a single interval vector"""
        return irx.Interval(
            lower=jnp.array([
                self.actuator_effectiveness.lower[0],
                self.sensor_scale.lower[0],
                self.sensor_bias.lower[0]
            ]),
            upper=jnp.array([
                self.actuator_effectiveness.upper[0],
                self.sensor_scale.upper[0],
                self.sensor_bias.upper[0]
            ])
        )


def create_go2_fault_scenarios() -> List[Go2FaultScenario]:
    """
    Create three fault scenarios for Go2:
    1. Nominal (no fault)
    2. Sensor fault (IMU drift: 5-10% scale error, 5-15 degree bias)
    3. Actuator fault (20-40% turn rate reduction)

    Returns:
        List of Go2FaultScenario objects
    """
    scenarios = []

    # Scenario 1: Nominal (no fault)
    scenarios.append(Go2FaultScenario(
        name="Nominal",
        actuator_effectiveness=irx.Interval(
            lower=jnp.array([1.0]),
            upper=jnp.array([1.0])
        ),
        sensor_scale=irx.Interval(
            lower=jnp.array([1.0]),
            upper=jnp.array([1.0])
        ),
        sensor_bias=irx.Interval(
            lower=jnp.array([0.0]),
            upper=jnp.array([0.0])
        )
    ))

    # Scenario 2: Sensor fault (IMU drift)
    scenarios.append(Go2FaultScenario(
        name="Sensor Fault",
        actuator_effectiveness=irx.Interval(
            lower=jnp.array([1.0]),
            upper=jnp.array([1.0])
        ),
        sensor_scale=irx.Interval(
            lower=jnp.array([1.05]),
            upper=jnp.array([1.10])
        ),
        sensor_bias=irx.Interval(
            lower=jnp.array([0.087]),  # 5 degrees
            upper=jnp.array([0.262])   # 15 degrees
        )
    ))

    # Scenario 3: Actuator fault (reduced turn rate)
    scenarios.append(Go2FaultScenario(
        name="Actuator Fault",
        actuator_effectiveness=irx.Interval(
            lower=jnp.array([0.60]),
            upper=jnp.array([0.80])
        ),
        sensor_scale=irx.Interval(
            lower=jnp.array([1.0]),
            upper=jnp.array([1.0])
        ),
        sensor_bias=irx.Interval(
            lower=jnp.array([0.0]),
            upper=jnp.array([0.0])
        )
    ))

    return scenarios


class Go2System(irx.System):
    """
    Unicycle dynamics for Go2 robot with fault parameters.

    State: x = [px, py, theta]  (position and heading)
    Control: u = [vx, vy, omega]  (velocities and yaw rate)
    Fault parameters: p = [alpha, a, b]
        - alpha: actuator effectiveness (0 to 1)
        - a: sensor scale (around 1.0)
        - b: sensor bias (radians)

    Dynamics with faults:
        dx/dt = vx * cos(theta_true) - vy * sin(theta_true)
        dy/dt = vx * sin(theta_true) + vy * cos(theta_true)
        dtheta/dt = alpha * omega  (actuator fault affects this)

        theta_measured = a * theta_true + b  (sensor fault)
    """

    def __init__(self):
        super().__init__(xlen=3, ulen=3, wlen=0, plen=3)

    def f(self, t, x, u, w, p):
        """
        System dynamics with faults.

        Args:
            t: time (unused)
            x: state [px, py, theta]
            u: control [vx, vy, omega]
            w: disturbance (unused)
            p: parameters [alpha, a, b]

        Returns:
            State derivative
        """
        px, py, theta = x[0], x[1], x[2]
        vx, vy, omega = u[0], u[1], u[2]
        alpha = p[0]  # actuator effectiveness

        # Unicycle dynamics with actuator fault
        dpx_dt = vx * jnp.cos(theta) - vy * jnp.sin(theta)
        dpy_dt = vx * jnp.sin(theta) + vy * jnp.cos(theta)
        dtheta_dt = alpha * omega  # Actuator fault: reduced yaw rate

        return jnp.array([dpx_dt, dpy_dt, dtheta_dt])


def pairwise_overlap_sum(intervals: List[irx.Interval],
                         overlap_fn) -> float:
    """
    Compute sum of overlaps between all pairs of intervals.

    Args:
        intervals: List of Interval objects
        overlap_fn: Function to compute overlap size between two intervals

    Returns:
        Total overlap score (scalar, lower is better)
    """
    n = len(intervals)
    if n < 2:
        return jnp.array(0.0)

    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap_val = overlap_fn(intervals[i], intervals[j])
            overlaps.append(overlap_val)

    return jnp.sum(jnp.array(overlaps))


def overlap_size_lax(interval1: irx.Interval, interval2: irx.Interval) -> float:
    """
    Compute the volume of the intersection of two intervals (JAX-compatible).

    Args:
        interval1: First interval
        interval2: Second interval

    Returns:
        Volume of intersection (0 if no overlap)
    """
    intersection_lower = jnp.maximum(interval1.lower, interval2.lower)
    intersection_upper = jnp.minimum(interval1.upper, interval2.upper)

    def has_overlap(operands):
        lower, upper = operands
        return jnp.prod(upper - lower)

    def no_overlap(operands):
        _ = operands
        return jnp.array(0.0)

    has_no_overlap = jnp.any(intersection_upper < intersection_lower)

    return lax.cond(
        jnp.logical_not(has_no_overlap),
        has_overlap,
        no_overlap,
        (intersection_lower, intersection_upper),
    )


def propagate_interval_euler(
    embedding_system: irx.System,
    x0_interval: irx.Interval,
    u_interval,
    w_interval,
    p_interval,
    dt: float,
) -> irx.Interval:
    """Forward Euler interval propagation for one time step."""
    xt_ut = (
        embedding_system.f(0.0, irx.i2ut(x0_interval), u_interval, w_interval, p_interval)
        * dt
        + irx.i2ut(x0_interval)
    )
    return irx.ut2i(xt_ut)


class Go2SeparatingInputOptimizer:
    """
    Optimizer for finding control inputs that maximize separation between
    different fault scenario reachable sets for the Go2 robot.
    """

    def __init__(
        self,
        scenarios: List[Go2FaultScenario],
        x0_interval: irx.Interval,
        dt: float,
        num_steps: int,
        state_slice: Optional[slice] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            scenarios: List of fault scenarios (nominal, sensor, actuator)
            x0_interval: Initial state interval [px, py, theta]
            dt: Time step for propagation
            num_steps: Number of propagation steps
            state_slice: Optional slice to select which states for overlap
                        (e.g., slice(0, 2) for position only)
        """
        self.scenarios = scenarios
        self.system = Go2System()
        self.embedding_system = irx.natemb(self.system)
        self.x0_interval = x0_interval
        self.w_interval = irx.Interval(
            lower=jnp.array([]),
            upper=jnp.array([])
        )  # No disturbance
        self.dt = dt
        self.num_steps = num_steps
        self.state_slice = state_slice if state_slice is not None else slice(None)

        # Create parameter intervals for each scenario
        self.fault_parameters = [s.to_parameter_interval() for s in scenarios]

        # JIT-compiled functions
        self._loss_fn_jitted = jax.jit(self._loss_fn)
        self._loss_grad_jitted = jax.jit(jax.grad(self._loss_fn))

    def _propagate_one_scenario(
        self,
        u: jnp.ndarray,
        p_interval: irx.Interval,
    ) -> irx.Interval:
        """
        Propagate initial state interval forward under control u and fault parameters p.

        Args:
            u: Control input (constant over propagation)
            p_interval: Fault parameter interval

        Returns:
            Final state interval after num_steps
        """
        def step_fn(i, x_propagated):
            return propagate_interval_euler(
                self.embedding_system,
                x_propagated,
                u,
                self.w_interval,
                p_interval,
                self.dt
            )

        return jax.lax.fori_loop(0, self.num_steps, step_fn, self.x0_interval)

    def _loss_fn(self, u: jnp.ndarray) -> float:
        """
        Loss function: sum of pairwise overlaps between all fault scenarios.

        We want to MINIMIZE this (less overlap = better separation).

        Args:
            u: Control input to evaluate

        Returns:
            Total overlap (lower is better for separation)
        """
        # Propagate all scenarios
        final_intervals = []
        for p_interval in self.fault_parameters:
            x_final = self._propagate_one_scenario(u, p_interval)
            # Extract only the states we care about
            x_final_sliced = irx.Interval(
                lower=x_final.lower[self.state_slice],
                upper=x_final.upper[self.state_slice]
            )
            final_intervals.append(x_final_sliced)

        # Compute pairwise overlap sum
        return pairwise_overlap_sum(final_intervals, overlap_size_lax)

    def optimize(
        self,
        u_initial: Optional[jnp.ndarray] = None,
        learning_rate: float = 1e-2,
        num_iterations: int = 200,
        verbose: bool = False,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Run gradient descent to find optimal separating input.

        Args:
            u_initial: Initial guess for control input
            learning_rate: Step size for gradient descent
            num_iterations: Number of optimization iterations
            verbose: Whether to print progress

        Returns:
            Tuple of (optimal_u, final_loss)
        """
        # Initialize control input
        if u_initial is None:
            # Default: moderate forward velocity, no lateral, some yaw rate
            u_initial = jnp.array([0.5, 0.0, 0.3])

        u_current = u_initial

        # Gradient descent loop
        for i in range(num_iterations):
            grad = self._loss_grad_jitted(u_current)
            u_current = u_current - learning_rate * grad

            if verbose and (i % 20 == 0 or i == num_iterations - 1):
                loss = self._loss_fn_jitted(u_current)
                grad_norm = jnp.linalg.norm(grad)
                print(f"Iteration {i:3d}: loss = {loss:.6f}, grad_norm = {grad_norm:.6f}, "
                      f"u = [{u_current[0]:.3f}, {u_current[1]:.3f}, {u_current[2]:.3f}]")

        final_loss = self._loss_fn_jitted(u_current)
        return u_current, final_loss

    def evaluate(self, u: jnp.ndarray) -> Tuple[float, List[irx.Interval], Dict]:
        """
        Evaluate a control input and return loss, intervals, and statistics.

        Args:
            u: Control input to evaluate

        Returns:
            Tuple of (overlap_loss, list_of_final_intervals, statistics)
        """
        final_intervals = []
        for p_interval in self.fault_parameters:
            x_final = self._propagate_one_scenario(u, p_interval)
            final_intervals.append(x_final)

        # Compute loss on sliced intervals
        sliced_intervals = [
            irx.Interval(lower=x.lower[self.state_slice], upper=x.upper[self.state_slice])
            for x in final_intervals
        ]
        loss = pairwise_overlap_sum(sliced_intervals, overlap_size_lax)

        # Compute statistics
        stats = {
            'scenario_names': [s.name for s in self.scenarios],
            'final_intervals': final_intervals,
            'sliced_intervals': sliced_intervals,
            'pairwise_overlaps': self._compute_pairwise_overlaps(sliced_intervals),
            'interval_volumes': [self._interval_volume(iv) for iv in sliced_intervals],
        }

        return loss, final_intervals, stats

    def _compute_pairwise_overlaps(self, intervals: List[irx.Interval]) -> Dict:
        """Compute all pairwise overlaps with labels"""
        overlaps = {}
        n = len(intervals)
        for i in range(n):
            for j in range(i + 1, n):
                overlap_val = float(overlap_size_lax(intervals[i], intervals[j]))
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = overlap_val
        return overlaps

    def _interval_volume(self, interval: irx.Interval) -> float:
        """Compute volume of an interval"""
        return float(jnp.prod(interval.upper - interval.lower))


def optimize_separating_input_multistart(
    scenarios: List[Go2FaultScenario],
    x0_interval: irx.Interval,
    dt: float,
    num_steps: int,
    num_restarts: int = 5,
    learning_rate: float = 1e-2,
    num_iterations: int = 200,
    state_slice: Optional[slice] = None,
    random_key: int = 42,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, float, Dict]:
    """
    Multi-start optimization to find global optimum for separating input.

    Runs optimization from multiple random starting points and returns the best result.

    Args:
        scenarios: List of fault scenarios
        x0_interval: Initial state interval
        dt: Time step
        num_steps: Number of propagation steps
        num_restarts: Number of random restarts
        learning_rate: Gradient descent learning rate
        num_iterations: Number of GD iterations per restart
        state_slice: Optional slice for which states to consider
        random_key: Random seed
        verbose: Print progress

    Returns:
        Tuple of (best_u, best_loss, best_stats)
    """
    optimizer = Go2SeparatingInputOptimizer(
        scenarios=scenarios,
        x0_interval=x0_interval,
        dt=dt,
        num_steps=num_steps,
        state_slice=state_slice,
    )

    # Generate random starting points
    key = jax.random.PRNGKey(random_key)
    keys = jax.random.split(key, num_restarts)

    best_u = None
    best_loss = float('inf')
    best_stats = None

    for i, subkey in enumerate(keys):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Restart {i+1}/{num_restarts}")
            print('='*60)

        # Random initialization
        u_init = jax.random.normal(subkey, (3,)) * 0.5 + jnp.array([0.5, 0.0, 0.3])

        # Optimize from this starting point
        u_opt, loss = optimizer.optimize(
            u_initial=u_init,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            verbose=verbose,
        )

        # Evaluate to get full statistics
        _, _, stats = optimizer.evaluate(u_opt)

        if verbose:
            print(f"\nRestart {i+1} result: loss = {loss:.6f}, u = {u_opt}")

        # Track best result
        if loss < best_loss:
            best_loss = loss
            best_u = u_opt
            best_stats = stats
            if verbose:
                print(f"  → New best!")

    return best_u, best_loss, best_stats


if __name__ == "__main__":
    print("="*70)
    print("GO2 SEPARATING INPUT OPTIMIZER - Test")
    print("="*70)

    # Create fault scenarios
    scenarios = create_go2_fault_scenarios()
    print(f"\nCreated {len(scenarios)} fault scenarios:")
    for s in scenarios:
        print(f"  - {s.name}")

    # Initial state interval (small uncertainty around origin)
    x0_interval = irx.Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([0.05, 0.05, 0.02])
    )

    # Optimization parameters
    dt = 0.5
    num_steps = 10

    print(f"\nPropagation: {num_steps} steps @ {dt}s = {num_steps*dt}s total")
    print(f"State slice: position only (first 2 dimensions)")

    # Run optimization
    print("\nRunning multi-start optimization...")
    u_opt, loss_opt, stats = optimize_separating_input_multistart(
        scenarios=scenarios,
        x0_interval=x0_interval,
        dt=dt,
        num_steps=num_steps,
        num_restarts=3,
        learning_rate=1e-2,
        num_iterations=100,
        state_slice=slice(0, 2),  # Only consider position for separation
        verbose=True
    )

    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nOptimal separating input:")
    print(f"  vx    = {u_opt[0]:.4f} m/s")
    print(f"  vy    = {u_opt[1]:.4f} m/s")
    print(f"  omega = {u_opt[2]:.4f} rad/s")
    print(f"\nFinal overlap loss: {loss_opt:.6f}")

    print(f"\nPairwise overlaps:")
    for key, val in stats['pairwise_overlaps'].items():
        print(f"  {key}: {val:.6f}")

    print(f"\nInterval volumes (position space):")
    for name, vol in zip(stats['scenario_names'], stats['interval_volumes']):
        print(f"  {name}: {vol:.6f} m²")

    print("\n" + "="*70)
    print("✓ Test completed successfully")
    print("="*70)
