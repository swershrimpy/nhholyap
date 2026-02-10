"""
Separating Input Optimizer

This module computes optimal "separating inputs" using gradient descent.
A separating input is a control signal that minimizes the overlap between
interval reachable sets for different fault scenarios, making faults easier
to distinguish.

The optimizer uses JAX for automatic differentiation and JIT compilation.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import List, Tuple, Callable, Optional
import immrax as irx
from functools import partial


def pairwise_overlap_sum(intervals: List[irx.Interval],
                         overlap_fn: Callable,
                         scaling_weights: Optional[jnp.ndarray] = None) -> float:
    """
    Compute sum of overlaps between all pairs of intervals.

    Args:
        intervals: List of Interval objects
        overlap_fn: Function to compute overlap size between two intervals
        scaling_weights: Optional weights for each pairwise overlap (shape: n*(n-1)/2)

    Returns:
        Total overlap score (scalar)
    """
    n = len(intervals)
    if n < 2:
        return jnp.array(0.0)

    # Compute all pairwise overlaps
    overlaps = []
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            overlap_val = overlap_fn(intervals[i], intervals[j])
            if scaling_weights is not None:
                overlap_val = overlap_val * scaling_weights[pair_idx]
                pair_idx += 1
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
    t_end: float,
) -> irx.Interval:
    """Forward Euler interval propagation for one time step."""
    xt_ut = (
        embedding_system.f(0.0, irx.i2ut(x0_interval), u_interval, w_interval, p_interval)
        * t_end
        + irx.i2ut(x0_interval)
    )
    return irx.ut2i(xt_ut)


class SeparatingInputOptimizer:
    """
    Optimizer for finding control inputs that minimize overlap between
    different fault scenario reachable sets.
    """

    def __init__(
        self,
        system: irx.System,
        fault_parameters: List[irx.Interval],
        x0_interval: irx.Interval,
        w_interval,
        dt: float,
        num_steps: int,
        state_slice: Optional[slice] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            system: The dynamical system (e.g., FaultyPlanarMultirotor)
            fault_parameters: List of parameter intervals for each fault scenario
            x0_interval: Initial state interval
            w_interval: Disturbance interval
            dt: Time step for propagation
            num_steps: Number of propagation steps
            state_slice: Optional slice to select which states to consider for overlap
                        (e.g., slice(0, 2) for position only)
        """
        self.system = system
        self.embedding_system = irx.natemb(system)
        self.fault_parameters = fault_parameters
        self.x0_interval = x0_interval
        self.w_interval = w_interval
        self.dt = dt
        self.num_steps = num_steps
        self.state_slice = state_slice if state_slice is not None else slice(None)

        # Create JIT-compiled versions of key functions
        self._loss_fn_jitted = jax.jit(self._loss_fn)
        self._loss_grad_jitted = jax.jit(jax.grad(self._loss_fn))

    def _propagate_one_scenario(
        self,
        u: jnp.ndarray,
        p_interval: irx.Interval,
    ) -> irx.Interval:
        """
        Propagate initial state interval forward under control u and fault parameter p.

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
        learning_rate: float = 1e-1,
        num_iterations: int = 100,
        verbose: bool = False,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Run gradient descent to find optimal separating input.

        Args:
            u_initial: Initial guess for control input (random if None)
            learning_rate: Step size for gradient descent
            num_iterations: Number of optimization iterations
            verbose: Whether to print progress

        Returns:
            Tuple of (optimal_u, final_loss)
        """
        # Initialize control input
        if u_initial is None:
            u_initial = jnp.ones(self.system.ulen)

        # Gradient descent loop
        def gradient_step(i, u_current):
            grad = self._loss_grad_jitted(u_current)
            u_next = u_current - learning_rate * grad

            if verbose:
                loss = self._loss_fn_jitted(u_current)
                jax.debug.print("Iteration {i}: loss = {loss}, grad_norm = {gnorm}",
                              i=i, loss=loss, gnorm=jnp.linalg.norm(grad))

            return u_next

        u_optimal = jax.lax.fori_loop(0, num_iterations, gradient_step, u_initial)
        final_loss = self._loss_fn_jitted(u_optimal)

        return u_optimal, final_loss

    def evaluate(self, u: jnp.ndarray) -> Tuple[float, List[irx.Interval]]:
        """
        Evaluate a control input and return both loss and final intervals.

        Args:
            u: Control input to evaluate

        Returns:
            Tuple of (overlap_loss, list_of_final_intervals)
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

        return loss, final_intervals


def optimize_separating_input_multistart(
    system: irx.System,
    fault_parameters: List[irx.Interval],
    x0_interval: irx.Interval,
    w_interval,
    dt: float,
    num_steps: int,
    num_restarts: int = 5,
    learning_rate: float = 1e-1,
    num_iterations: int = 100,
    state_slice: Optional[slice] = None,
    random_key: int = 42,
) -> Tuple[jnp.ndarray, float]:
    """
    Multi-start optimization to find global optimum.

    Runs optimization from multiple random starting points and returns the best result.

    Args:
        system: The dynamical system
        fault_parameters: List of parameter intervals for each fault scenario
        x0_interval: Initial state interval
        w_interval: Disturbance interval
        dt: Time step
        num_steps: Number of propagation steps
        num_restarts: Number of random restarts
        learning_rate: Gradient descent learning rate
        num_iterations: Number of GD iterations per restart
        state_slice: Optional slice for which states to consider
        random_key: Random seed

    Returns:
        Tuple of (best_u, best_loss)
    """
    optimizer = SeparatingInputOptimizer(
        system=system,
        fault_parameters=fault_parameters,
        x0_interval=x0_interval,
        w_interval=w_interval,
        dt=dt,
        num_steps=num_steps,
        state_slice=state_slice,
    )

    # Generate random starting points
    key = jax.random.PRNGKey(random_key)
    keys = jax.random.split(key, num_restarts)

    best_u = None
    best_loss = float('inf')

    for i, subkey in enumerate(keys):
        # Random initialization
        u_init = jax.random.normal(subkey, (system.ulen,))

        # Optimize from this starting point
        u_opt, loss = optimizer.optimize(
            u_initial=u_init,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            verbose=False,
        )

        # Track best result
        if loss < best_loss:
            best_loss = loss
            best_u = u_opt

    return best_u, best_loss
