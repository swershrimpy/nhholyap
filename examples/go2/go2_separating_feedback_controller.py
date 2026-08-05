"""
Separating Output Feedback Controller for Unitree Go2

Design an output feedback control law that:
1. Tracks the nominal trajectory plan
2. Maximizes separation between fault scenarios for diagnosis

Controller form:
    u[t] = u_nom[t] + K[t] @ (y[t] - y_nom[t])

where:
    - u_nom[t]: nominal feedforward control
    - K[t]: time-varying feedback gain matrix (optimized)
    - y[t]: observation from DINOv2 vision
    - y_nom[t]: nominal observation
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class Interval:
    """Interval representation for reachable sets"""
    lower: jnp.ndarray
    upper: jnp.ndarray

    @property
    def mid(self):
        return (self.lower + self.upper) / 2

    @property
    def rad(self):
        return (self.upper - self.lower) / 2


@dataclass
class FaultScenario:
    """Fault scenario with parameter ranges"""
    name: str
    alpha_range: Tuple[float, float]  # actuator effectiveness
    scale_range: Tuple[float, float]  # sensor scale
    bias_range: Tuple[float, float]   # sensor bias (rad)


def create_scenarios() -> List[FaultScenario]:
    """Create nominal, sensor fault, and actuator fault scenarios"""
    return [
        FaultScenario(
            name="Nominal",
            alpha_range=(1.0, 1.0),
            scale_range=(1.0, 1.0),
            bias_range=(0.0, 0.0)
        ),
        FaultScenario(
            name="Sensor Fault",
            alpha_range=(1.0, 1.0),
            scale_range=(1.25, 1.50),  # 5-10% scale error
            bias_range=(0.087, 0.262)  # 5-15 degrees
        ),
        FaultScenario(
            name="Actuator Fault",
            alpha_range=(0.60, 0.80),  # 20-40% reduction
            scale_range=(1.0, 1.0),
            bias_range=(0.0, 0.0)
        )
    ]


def _contains_shifted_multiple(lower: jnp.ndarray, upper: jnp.ndarray, shift: float) -> jnp.ndarray:
    """Return True iff [lower, upper] contains shift + 2*pi*k for some integer k."""
    two_pi = 2.0 * jnp.pi
    k_min = jnp.ceil((lower - shift) / two_pi)
    k_max = jnp.floor((upper - shift) / two_pi)
    return k_min <= k_max


def _cos_bounds(theta_lower: jnp.ndarray, theta_upper: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    c1 = jnp.cos(theta_lower)
    c2 = jnp.cos(theta_upper)
    c_min = jnp.minimum(c1, c2)
    c_max = jnp.maximum(c1, c2)

    has_cos_max = _contains_shifted_multiple(theta_lower, theta_upper, shift=0.0)
    has_cos_min = _contains_shifted_multiple(theta_lower, theta_upper, shift=jnp.pi)

    c_max = jnp.where(has_cos_max, 1.0, c_max)
    c_min = jnp.where(has_cos_min, -1.0, c_min)
    return c_min, c_max


def _sin_bounds(theta_lower: jnp.ndarray, theta_upper: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    s1 = jnp.sin(theta_lower)
    s2 = jnp.sin(theta_upper)
    s_min = jnp.minimum(s1, s2)
    s_max = jnp.maximum(s1, s2)

    has_sin_max = _contains_shifted_multiple(theta_lower, theta_upper, shift=jnp.pi / 2.0)
    has_sin_min = _contains_shifted_multiple(theta_lower, theta_upper, shift=-jnp.pi / 2.0)

    s_max = jnp.where(has_sin_max, 1.0, s_max)
    s_min = jnp.where(has_sin_min, -1.0, s_min)
    return s_min, s_max


def interval_dynamics_step(x_int: Interval, u: jnp.ndarray,
                           alpha_int: Interval, dt: float) -> Interval:
    """
    Single step of interval dynamics propagation.

    Args:
        x_int: State interval [px, py, theta]
        u: Control input [vx, vy, omega]
        alpha_int: Actuator effectiveness interval
        dt: Time step

    Returns:
        Next state interval
    """
    px_l, py_l, th_l = x_int.lower[0], x_int.lower[1], x_int.lower[2]
    px_u, py_u, th_u = x_int.upper[0], x_int.upper[1], x_int.upper[2]

    vx, vy, omega = u[0], u[1], u[2]
    alpha_l, alpha_u = alpha_int.lower[0], alpha_int.upper[0]

    cos_min, cos_max = _cos_bounds(th_l, th_u)
    sin_min, sin_max = _sin_bounds(th_l, th_u)

    # dpx/dt = vx * cos(theta) - vy * sin(theta)
    term1_vals = vx * jnp.array([cos_min, cos_max])
    term2_vals = -vy * jnp.array([sin_min, sin_max])
    dpx_min = jnp.min(term1_vals) + jnp.min(term2_vals)
    dpx_max = jnp.max(term1_vals) + jnp.max(term2_vals)

    # dpy/dt = vx * sin(theta) + vy * cos(theta)
    term1_vals = vx * jnp.array([sin_min, sin_max])
    term2_vals = vy * jnp.array([cos_min, cos_max])
    dpy_min = jnp.min(term1_vals) + jnp.min(term2_vals)
    dpy_max = jnp.max(term1_vals) + jnp.max(term2_vals)

    # dtheta/dt = alpha * omega
    dth_vals = jnp.array([alpha_l * omega, alpha_u * omega])
    dth_min, dth_max = jnp.min(dth_vals), jnp.max(dth_vals)

    # Euler step
    px_next_l = px_l + dt * dpx_min
    px_next_u = px_u + dt * dpx_max
    py_next_l = py_l + dt * dpy_min
    py_next_u = py_u + dt * dpy_max
    th_next_l = th_l + dt * dth_min
    th_next_u = th_u + dt * dth_max

    return Interval(
        lower=jnp.array([px_next_l, py_next_l, th_next_l]),
        upper=jnp.array([px_next_u, py_next_u, th_next_u])
    )


def observation_interval(x_int: Interval, C: jnp.ndarray) -> Interval:
    """
    Compute observation interval y = C @ x

    For each observation dimension, find min/max over the state interval.
    """
    n_obs = C.shape[0]
    y_lower = jnp.zeros(n_obs)
    y_upper = jnp.zeros(n_obs)

    for i in range(n_obs):
        # For c_j * x_j, use lower bound when c_j >= 0 else upper bound for min,
        # and the opposite for max.
        row = C[i]
        y_i_lower = jnp.sum(jnp.where(row >= 0, row * x_int.lower, row * x_int.upper))
        y_i_upper = jnp.sum(jnp.where(row >= 0, row * x_int.upper, row * x_int.lower))

        y_lower = y_lower.at[i].set(y_i_lower)
        y_upper = y_upper.at[i].set(y_i_upper)

    return Interval(lower=y_lower, upper=y_upper)


def _apply_sensor_fault_to_state_interval(x_int: Interval, scenario: FaultScenario) -> Interval:
    """Map true state interval to measured state interval under sensor fault model."""
    a_l, a_u = scenario.scale_range
    b_l, b_u = scenario.bias_range

    theta_l = x_int.lower[2]
    theta_u = x_int.upper[2]

    theta_candidates = jnp.array([
        a_l * theta_l + b_l,
        a_l * theta_l + b_u,
        a_l * theta_u + b_l,
        a_l * theta_u + b_u,
    ])

    theta_meas_l = jnp.min(theta_candidates)
    theta_meas_u = jnp.max(theta_candidates)

    measured_lower = x_int.lower.at[2].set(theta_meas_l)
    measured_upper = x_int.upper.at[2].set(theta_meas_u)
    return Interval(lower=measured_lower, upper=measured_upper)


def propagate_feedback_scenario(
    x0_int: Interval,
    x_nom_traj: jnp.ndarray,
    u_nom_traj: jnp.ndarray,
    y_nom_traj: jnp.ndarray,
    K_traj: jnp.ndarray,
    C: jnp.ndarray,
    scenario: FaultScenario,
    dt: float,
    obs_uncertainty: float = 0.05
) -> Tuple[Interval, List[Interval]]:
    """
    Propagate interval through dynamics with output feedback control.

    Args:
        x0_int: Initial state interval
        x_nom_traj: Nominal state trajectory [N, 3]
        u_nom_traj: Nominal control trajectory [N, 3]
        y_nom_traj: Nominal observation trajectory [N, obs_dim]
        K_traj: Feedback gain matrices [N, 3, obs_dim]
        C: Observation matrix [obs_dim, 3]
        scenario: Fault scenario
        dt: Time step
        obs_uncertainty: Observation noise/uncertainty

    Returns:
        (final_interval, interval_history)
    """
    alpha_int = Interval(
        lower=jnp.array([scenario.alpha_range[0]]),
        upper=jnp.array([scenario.alpha_range[1]])
    )

    N = len(x_nom_traj)
    x_current = x0_int
    interval_history = [x_current]

    for t in range(N - 1):
        # Observation is based on measured state, which includes sensor faults.
        x_meas_int = _apply_sensor_fault_to_state_interval(x_current, scenario)
        y_int = observation_interval(x_meas_int, C)

        # Add observation uncertainty
        y_int = Interval(
            lower=y_int.lower - obs_uncertainty,
            upper=y_int.upper + obs_uncertainty
        )

        # Compute observation error interval
        y_err_int = Interval(
            lower=y_int.lower - y_nom_traj[t],
            upper=y_int.upper - y_nom_traj[t]
        )

        # Compute feedback control interval: K @ y_err
        # For each control dimension: u_fb[i] = sum_j K[i,j] * y_err[j]
        K_t = K_traj[t]  # [3, obs_dim]
        u_fb_lower = jnp.zeros(3)
        u_fb_upper = jnp.zeros(3)

        for i in range(3):
            # For each control dimension
            contrib_lower = jnp.sum(
                jnp.where(K_t[i] > 0,
                         K_t[i] * y_err_int.lower,
                         K_t[i] * y_err_int.upper)
            )
            contrib_upper = jnp.sum(
                jnp.where(K_t[i] > 0,
                         K_t[i] * y_err_int.upper,
                         K_t[i] * y_err_int.lower)
            )
            u_fb_lower = u_fb_lower.at[i].set(contrib_lower)
            u_fb_upper = u_fb_upper.at[i].set(contrib_upper)

        # Total control = feedforward + feedback
        # We need to propagate both bounds
        u_total_lower = u_nom_traj[t] + u_fb_lower
        u_total_upper = u_nom_traj[t] + u_fb_upper

        # Conservative: use control that expands interval most
        # This is complex, so we'll use a simplified approach:
        # Propagate with both extremes and take union
        x_next_lower = interval_dynamics_step(x_current, u_total_lower, alpha_int, dt)
        x_next_upper = interval_dynamics_step(x_current, u_total_upper, alpha_int, dt)

        x_current = Interval(
            lower=jnp.minimum(x_next_lower.lower, x_next_upper.lower),
            upper=jnp.maximum(x_next_lower.upper, x_next_upper.upper)
        )

        interval_history.append(x_current)

    return x_current, interval_history


def interval_overlap_volume(int1: Interval, int2: Interval):
    """Compute volume of intersection between two intervals"""
    intersection_lower = jnp.maximum(int1.lower, int2.lower)
    intersection_upper = jnp.minimum(int1.upper, int2.upper)

    # Check if there's overlap
    has_overlap = jnp.all(intersection_upper >= intersection_lower)

    volume = jnp.where(
        has_overlap,
        jnp.prod(intersection_upper - intersection_lower),
        0.0
    )

    return volume


class SeparatingFeedbackOptimizer:
    """
    Optimize output feedback gains to achieve:
    1. Trajectory tracking (stay close to nominal)
    2. Fault separation (maximize distinguishability)
    """

    def __init__(
        self,
        x_nom_traj: np.ndarray,
        u_nom_traj: np.ndarray,
        C: np.ndarray,
        scenarios: List[FaultScenario],
        x0_int: Interval,
        dt: float,
        lambda_sep: float = 1.0,
        lambda_track: float = 0.1,
        obs_uncertainty: float = 0.05
    ):
        """
        Args:
            x_nom_traj: Nominal state trajectory [N, 3]
            u_nom_traj: Nominal control trajectory [N, 3]
            C: Observation matrix [obs_dim, 3]
            scenarios: List of fault scenarios
            x0_int: Initial state interval
            dt: Time step
            lambda_sep: Weight on separation loss
            lambda_track: Weight on tracking loss
            obs_uncertainty: Observation noise level
        """
        self.x_nom_traj = jnp.array(x_nom_traj)
        self.u_nom_traj = jnp.array(u_nom_traj)
        self.C = jnp.array(C)
        self.scenarios = scenarios
        self.x0_int = x0_int
        self.dt = dt
        self.lambda_sep = lambda_sep
        self.lambda_track = lambda_track
        self.obs_uncertainty = obs_uncertainty

        self.N = len(x_nom_traj)
        self.obs_dim = C.shape[0]

        # Compute nominal observations
        self.y_nom_traj = jnp.array([C @ x for x in x_nom_traj])

        print(f"Initialized SeparatingFeedbackOptimizer:")
        print(f"  Trajectory length: {self.N}")
        print(f"  Observation dimension: {self.obs_dim}")
        print(f"  λ_separation: {lambda_sep}")
        print(f"  λ_tracking: {lambda_track}")

    def loss_fn(self, K_flat: jnp.ndarray) -> float:
        """
        Combined loss: separation + tracking

        Args:
            K_flat: Flattened feedback gains [N * 3 * obs_dim]

        Returns:
            Total loss
        """
        # Reshape to [N, 3, obs_dim]
        K_traj = K_flat.reshape(self.N, 3, self.obs_dim)

        # 1. Separation loss: minimize overlap between scenarios
        final_intervals = []
        for scenario in self.scenarios:
            x_final, _ = propagate_feedback_scenario(
                self.x0_int,
                self.x_nom_traj,
                self.u_nom_traj,
                self.y_nom_traj,
                K_traj,
                self.C,
                scenario,
                self.dt,
                self.obs_uncertainty
            )
            # Use position only for separation
            pos_int = Interval(lower=x_final.lower[:2], upper=x_final.upper[:2])
            final_intervals.append(pos_int)

        # Pairwise overlaps
        separation_loss = 0.0
        for i in range(len(final_intervals)):
            for j in range(i + 1, len(final_intervals)):
                overlap = interval_overlap_volume(final_intervals[i], final_intervals[j])
                separation_loss += overlap

        # 2. Tracking loss: stay close to nominal
        # Penalize large feedback gains (keep close to feedforward)
        tracking_loss = jnp.mean(jnp.square(K_flat))

        # Combined loss
        total_loss = (self.lambda_sep * separation_loss +
                     self.lambda_track * tracking_loss)

        return total_loss

    def optimize(
        self,
        K_init: Optional[np.ndarray] = None,
        learning_rate: float = 0.01,
        num_iters: int = 100,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize feedback gains using gradient descent.

        Args:
            K_init: Initial gains [N, 3, obs_dim], if None use zeros
            learning_rate: Learning rate
            num_iters: Number of iterations
            verbose: Print progress

        Returns:
            (K_opt, final_loss)
        """
        if K_init is None:
            K_init = np.zeros((self.N, 3, self.obs_dim))

        K_flat = jnp.array(K_init.flatten())

        # JIT compile loss and gradient
        loss_jit = jax.jit(self.loss_fn)
        grad_jit = jax.jit(jax.grad(self.loss_fn))

        if verbose:
            print(f"\nStarting optimization...")
            print(f"  Initial K shape: {K_init.shape}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Iterations: {num_iters}")

        for i in range(num_iters):
            grad = grad_jit(K_flat)
            K_flat = K_flat - learning_rate * grad

            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = loss_jit(K_flat)
                grad_norm = jnp.linalg.norm(grad)
                print(f"Iter {i:3d}: loss={float(loss):.6f}, "
                      f"|grad|={float(grad_norm):.6f}")

        final_loss = float(loss_jit(K_flat))
        K_opt = np.array(K_flat.reshape(self.N, 3, self.obs_dim))

        return K_opt, final_loss

    def evaluate(self, K: np.ndarray) -> Dict:
        """
        Evaluate feedback controller and return detailed statistics.

        Args:
            K: Feedback gains [N, 3, obs_dim]

        Returns:
            Dictionary with evaluation results
        """
        K_traj = jnp.array(K)

        # Propagate all scenarios
        results = {}
        for scenario in self.scenarios:
            x_final, history = propagate_feedback_scenario(
                self.x0_int,
                self.x_nom_traj,
                self.u_nom_traj,
                self.y_nom_traj,
                K_traj,
                self.C,
                scenario,
                self.dt,
                self.obs_uncertainty
            )
            results[scenario.name] = {
                'final_interval': x_final,
                'history': history
            }

        # Compute overlaps
        position_intervals = {}
        for name, data in results.items():
            x_final = data['final_interval']
            position_intervals[name] = Interval(
                lower=x_final.lower[:2],
                upper=x_final.upper[:2]
            )

        overlaps = {}
        scenario_names = list(position_intervals.keys())
        for i in range(len(scenario_names)):
            for j in range(i + 1, len(scenario_names)):
                key = f"{scenario_names[i]} vs {scenario_names[j]}"
                overlap = interval_overlap_volume(
                    position_intervals[scenario_names[i]],
                    position_intervals[scenario_names[j]]
                )
                overlaps[key] = float(overlap)

        # Compute tracking error (deviation from nominal)
        tracking_error = np.linalg.norm(K)

        return {
            'results': results,
            'position_intervals': position_intervals,
            'overlaps': overlaps,
            'total_overlap': sum(overlaps.values()),
            'tracking_error': tracking_error
        }


if __name__ == "__main__":
    print("="*70)
    print("SEPARATING OUTPUT FEEDBACK CONTROLLER OPTIMIZATION")
    print("="*70)

    # Use synthetic nominal trajectory for demonstration
    # (The real plan file has 31-dim state for full quadruped,
    #  but we demonstrate with simple 3D unicycle model)

    print(f"\nCreating synthetic nominal trajectory...")
    print("(Using simple unicycle model for demonstration)")

    # Create synthetic nominal trajectory
    N = 20
    dt = 0.5
    t = np.linspace(0, (N-1)*dt, N)

    # Simple circular trajectory
    radius = 2.0
    omega_nom = 0.3
    x_nom = radius * np.cos(omega_nom * t)
    y_nom = radius * np.sin(omega_nom * t)
    theta_nom = omega_nom * t + np.pi/2

    x_nom_traj = np.column_stack([x_nom, y_nom, theta_nom])

    # Compute velocities
    vx_nom = -radius * omega_nom * np.sin(omega_nom * t)
    vy_nom = radius * omega_nom * np.cos(omega_nom * t)
    omega_cmd = np.ones(N) * omega_nom

    u_nom_traj = np.column_stack([vx_nom, vy_nom, omega_cmd])

    # Simple observation matrix (identity for demonstration)
    C = np.eye(3)

    print(f"\nTrajectory details:")
    print(f"  Length: {N} steps")
    print(f"  Time step: {dt}s")
    print(f"  Total time: {N*dt}s")
    print(f"  State dimension: {x_nom_traj.shape[1]}")
    print(f"  Control dimension: {u_nom_traj.shape[1]}")
    print(f"  Observation dimension: {C.shape[0]}")

    # Create fault scenarios
    scenarios = create_scenarios()
    print(f"\nFault scenarios:")
    for s in scenarios:
        print(f"  {s.name}:")
        print(f"    α ∈ [{s.alpha_range[0]:.2f}, {s.alpha_range[1]:.2f}]")

    # Initial state interval
    x0_int = Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([0.05, 0.05, 0.02])
    )

    print(f"\nInitial state uncertainty:")
    print(f"  px ∈ [{x0_int.lower[0]:.3f}, {x0_int.upper[0]:.3f}] m")
    print(f"  py ∈ [{x0_int.lower[1]:.3f}, {x0_int.upper[1]:.3f}] m")
    print(f"  θ  ∈ [{x0_int.lower[2]:.3f}, {x0_int.upper[2]:.3f}] rad")

    # Create optimizer
    optimizer = SeparatingFeedbackOptimizer(
        x_nom_traj=x_nom_traj,
        u_nom_traj=u_nom_traj,
        C=C,
        scenarios=scenarios,
        x0_int=x0_int,
        dt=dt,
        lambda_sep=1.0,
        lambda_track=0.01,  # Small weight to allow some deviation
        obs_uncertainty=0.05
    )
