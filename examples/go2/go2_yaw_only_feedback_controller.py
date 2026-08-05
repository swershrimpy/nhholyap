"""
Yaw-Only Output Feedback Controller for Fault Diagnosis

Design output feedback control law with ONLY yaw observations:
    u[t] = u_nom[t] + K[t] * (yaw[t] - yaw_nom[t])

Key features:
- Only observes yaw angle (no position feedback)
- Hidden angular velocity state
- Optimizes K[t] to maximize fault separation

State: x = [px, py, yaw, omega]
Observation: y = yaw (scalar!)
Control: u = [vx, vy, u_omega]
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
    gyro_scale_range: Tuple[float, float]  # gyroscope scale (a)
    gyro_bias_range: Tuple[float, float]   # gyroscope bias (b) in rad/s


def create_scenarios() -> List[FaultScenario]:
    """Create fault scenarios for extended system"""
    return [
        FaultScenario(
            name="Nominal",
            alpha_range=(1.0, 1.0),
            gyro_scale_range=(1.0, 1.0),
            gyro_bias_range=(0.0, 0.0)
        ),
        FaultScenario(
            name="Sensor Fault",
            alpha_range=(1.0, 1.0),
            gyro_scale_range=(1.25, 1.50),  # 25-50% scale error
            gyro_bias_range=(0.05, 0.15)    # 0.05-0.15 rad/s bias
        ),
        FaultScenario(
            name="Actuator Fault",
            alpha_range=(0.60, 0.80),  # 20-40% reduction
            gyro_scale_range=(1.0, 1.0),
            gyro_bias_range=(0.0, 0.0)
        )
    ]


def _contains_shifted_multiple(lower: jnp.ndarray, upper: jnp.ndarray,
                               shift: float) -> jnp.ndarray:
    """Check if interval contains shift + 2*pi*k for some integer k"""
    two_pi = 2.0 * jnp.pi
    k_min = jnp.ceil((lower - shift) / two_pi)
    k_max = jnp.floor((upper - shift) / two_pi)
    return k_min <= k_max


def _cos_bounds(theta_lower: jnp.ndarray, theta_upper: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute tight bounds on cos(theta) over interval"""
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
    """Compute tight bounds on sin(theta) over interval"""
    s1 = jnp.sin(theta_lower)
    s2 = jnp.sin(theta_upper)
    s_min = jnp.minimum(s1, s2)
    s_max = jnp.maximum(s1, s2)

    has_sin_max = _contains_shifted_multiple(theta_lower, theta_upper, shift=jnp.pi / 2.0)
    has_sin_min = _contains_shifted_multiple(theta_lower, theta_upper, shift=-jnp.pi / 2.0)

    s_max = jnp.where(has_sin_max, 1.0, s_max)
    s_min = jnp.where(has_sin_min, -1.0, s_min)
    return s_min, s_max


def interval_dynamics_step_extended(
    x_int: Interval,
    u: jnp.ndarray,
    alpha_int: Interval,
    dt: float
) -> Interval:
    """
    Propagate extended state interval through dynamics.

    State: x = [px, py, yaw, omega]
    Control: u = [vx, vy, u_omega]
    """
    px_l, py_l, yaw_l, omega_l = x_int.lower
    px_u, py_u, yaw_u, omega_u = x_int.upper

    vx, vy, u_omega = u[0], u[1], u[2]
    alpha_l, alpha_u = alpha_int.lower[0], alpha_int.upper[0]

    cos_min, cos_max = _cos_bounds(yaw_l, yaw_u)
    sin_min, sin_max = _sin_bounds(yaw_l, yaw_u)

    # dpx/dt = vx * cos(yaw) - vy * sin(yaw)
    term1_vals = vx * jnp.array([cos_min, cos_max])
    term2_vals = -vy * jnp.array([sin_min, sin_max])
    dpx_min = jnp.min(term1_vals) + jnp.min(term2_vals)
    dpx_max = jnp.max(term1_vals) + jnp.max(term2_vals)

    # dpy/dt = vx * sin(yaw) + vy * cos(yaw)
    term1_vals = vx * jnp.array([sin_min, sin_max])
    term2_vals = vy * jnp.array([cos_min, cos_max])
    dpy_min = jnp.min(term1_vals) + jnp.min(term2_vals)
    dpy_max = jnp.max(term1_vals) + jnp.max(term2_vals)

    # dyaw/dt = omega
    dyaw_min = omega_l
    dyaw_max = omega_u

    # domega/dt = alpha * u_omega
    domega_vals = jnp.array([alpha_l * u_omega, alpha_u * u_omega])
    domega_min = jnp.min(domega_vals)
    domega_max = jnp.max(domega_vals)

    # Euler integration
    return Interval(
        lower=jnp.array([
            px_l + dt * dpx_min,
            py_l + dt * dpy_min,
            yaw_l + dt * dyaw_min,
            omega_l + dt * domega_min
        ]),
        upper=jnp.array([
            px_u + dt * dpx_max,
            py_u + dt * dpy_max,
            yaw_u + dt * dyaw_max,
            omega_u + dt * domega_max
        ])
    )


def apply_gyro_fault_to_yaw_interval(
    x_int: Interval,
    scenario: FaultScenario,
    dt: float,
    accumulated_yaw_error_int: Interval
) -> Tuple[Interval, Interval]:
    """
    Apply gyroscope fault and return observed yaw interval.

    Returns:
        (yaw_observed_interval, updated_yaw_error_interval)
    """
    a_l, a_u = scenario.gyro_scale_range
    b_l, b_u = scenario.gyro_bias_range

    omega_l = x_int.lower[3]
    omega_u = x_int.upper[3]

    # Compute increment to yaw error: dt * ((a-1) * omega + b)
    scale_error_l = a_l - 1.0
    scale_error_u = a_u - 1.0

    scale_omega_vals = jnp.array([
        scale_error_l * omega_l,
        scale_error_l * omega_u,
        scale_error_u * omega_l,
        scale_error_u * omega_u
    ])
    scale_omega_min = jnp.min(scale_omega_vals)
    scale_omega_max = jnp.max(scale_omega_vals)

    rate_error_min = scale_omega_min + b_l
    rate_error_max = scale_omega_max + b_u

    dyaw_error_min = dt * rate_error_min
    dyaw_error_max = dt * rate_error_max

    new_yaw_error_l = accumulated_yaw_error_int.lower[0] + dyaw_error_min
    new_yaw_error_u = accumulated_yaw_error_int.upper[0] + dyaw_error_max

    new_yaw_error_int = Interval(
        lower=jnp.array([new_yaw_error_l]),
        upper=jnp.array([new_yaw_error_u])
    )

    # Observed yaw = true yaw + yaw error
    yaw_obs_int = Interval(
        lower=jnp.array([x_int.lower[2] + new_yaw_error_l]),
        upper=jnp.array([x_int.upper[2] + new_yaw_error_u])
    )

    return yaw_obs_int, new_yaw_error_int


def propagate_yaw_feedback_scenario(
    x0_int: Interval,
    x_nom_traj: jnp.ndarray,
    u_nom_traj: jnp.ndarray,
    yaw_nom_traj: jnp.ndarray,
    K_traj: jnp.ndarray,
    scenario: FaultScenario,
    dt: float,
    obs_uncertainty: float = 0.02
) -> Tuple[Interval, List[Interval]]:
    """
    Propagate interval with YAW-ONLY feedback control.

    Controller: u[t] = u_nom[t] + K[t] * (yaw_obs[t] - yaw_nom[t])

    Args:
        x0_int: Initial state interval [px, py, yaw, omega]
        x_nom_traj: Nominal state trajectory [N, 4]
        u_nom_traj: Nominal control trajectory [N, 3]
        yaw_nom_traj: Nominal yaw trajectory [N]
        K_traj: Feedback gain matrices [N, 3] - maps scalar yaw error to 3D control
        scenario: Fault scenario
        dt: Time step
        obs_uncertainty: Yaw observation noise

    Returns:
        (final_interval, interval_history)
    """
    alpha_int = Interval(
        lower=jnp.array([scenario.alpha_range[0]]),
        upper=jnp.array([scenario.alpha_range[1]])
    )

    N = len(x_nom_traj)
    x_current = x0_int
    yaw_error_int = Interval(lower=jnp.array([0.0]), upper=jnp.array([0.0]))
    interval_history = [x_current]

    for t in range(N - 1):
        # Observe yaw (with sensor fault)
        yaw_obs_int, yaw_error_int = apply_gyro_fault_to_yaw_interval(
            x_current, scenario, dt, yaw_error_int
        )

        # Add observation uncertainty
        yaw_obs_int = Interval(
            lower=yaw_obs_int.lower - obs_uncertainty,
            upper=yaw_obs_int.upper + obs_uncertainty
        )

        # Compute yaw error: yaw_obs - yaw_nom
        yaw_err_int = Interval(
            lower=yaw_obs_int.lower - yaw_nom_traj[t],
            upper=yaw_obs_int.upper - yaw_nom_traj[t]
        )

        # Feedback control: u_fb = K[t] * yaw_error
        # K[t] is [3] vector (one gain per control dimension)
        K_t = K_traj[t]  # [3]

        # u_fb bounds: K[i] * yaw_error for each control i
        u_fb_lower = jnp.zeros(3)
        u_fb_upper = jnp.zeros(3)

        for i in range(3):
            K_i = K_t[i]
            # If K_i > 0: min when yaw_err is min, max when yaw_err is max
            # If K_i < 0: opposite
            contrib_lower = jnp.where(
                K_i >= 0,
                K_i * yaw_err_int.lower[0],
                K_i * yaw_err_int.upper[0]
            )
            contrib_upper = jnp.where(
                K_i >= 0,
                K_i * yaw_err_int.upper[0],
                K_i * yaw_err_int.lower[0]
            )
            u_fb_lower = u_fb_lower.at[i].set(contrib_lower)
            u_fb_upper = u_fb_upper.at[i].set(contrib_upper)

        # Total control = feedforward + feedback
        u_total_lower = u_nom_traj[t] + u_fb_lower
        u_total_upper = u_nom_traj[t] + u_fb_upper

        # Propagate with both extremes and take union
        x_next_lower = interval_dynamics_step_extended(x_current, u_total_lower, alpha_int, dt)
        x_next_upper = interval_dynamics_step_extended(x_current, u_total_upper, alpha_int, dt)

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

    has_overlap = jnp.all(intersection_upper >= intersection_lower)

    volume = jnp.where(
        has_overlap,
        jnp.prod(intersection_upper - intersection_lower),
        0.0
    )

    return volume


class YawOnlyFeedbackOptimizer:
    """
    Optimize YAW-ONLY feedback gains to maximize fault separation.

    Controller: u[t] = u_nom[t] + K[t] * (yaw[t] - yaw_nom[t])

    Objectives:
    1. Maximize separation between fault scenarios
    2. Maintain tracking of nominal trajectory
    """

    def __init__(
        self,
        x_nom_traj: np.ndarray,
        u_nom_traj: np.ndarray,
        scenarios: List[FaultScenario],
        x0_int: Interval,
        dt: float,
        lambda_sep: float = 1.0,
        lambda_track: float = 0.01,
        obs_uncertainty: float = 0.02
    ):
        """
        Args:
            x_nom_traj: Nominal state trajectory [N, 4]
            u_nom_traj: Nominal control trajectory [N, 3]
            scenarios: List of fault scenarios
            x0_int: Initial state interval
            dt: Time step
            lambda_sep: Weight on separation loss
            lambda_track: Weight on tracking loss
            obs_uncertainty: Yaw observation noise level
        """
        self.x_nom_traj = jnp.array(x_nom_traj)
        self.u_nom_traj = jnp.array(u_nom_traj)
        self.scenarios = scenarios
        self.x0_int = x0_int
        self.dt = dt
        self.lambda_sep = lambda_sep
        self.lambda_track = lambda_track
        self.obs_uncertainty = obs_uncertainty

        self.N = len(x_nom_traj)

        # Extract nominal yaw trajectory
        self.yaw_nom_traj = jnp.array(x_nom_traj[:, 2])

        print(f"Initialized YawOnlyFeedbackOptimizer:")
        print(f"  Trajectory length: {self.N}")
        print(f"  Observation: YAW ONLY (scalar)")
        print(f"  Feedback gains shape: [{self.N}, 3] (maps yaw error to u)")
        print(f"  λ_separation: {lambda_sep}")
        print(f"  λ_tracking: {lambda_track}")

    def loss_fn(self, K_flat: jnp.ndarray) -> float:
        """
        Combined loss: separation + tracking

        Args:
            K_flat: Flattened feedback gains [N * 3]

        Returns:
            Total loss
        """
        # Reshape to [N, 3]
        K_traj = K_flat.reshape(self.N, 3)

        # 1. Separation loss: minimize overlap between scenarios
        final_intervals = []
        for scenario in self.scenarios:
            x_final, _ = propagate_yaw_feedback_scenario(
                self.x0_int,
                self.x_nom_traj,
                self.u_nom_traj,
                self.yaw_nom_traj,
                K_traj,
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

        # 2. Tracking loss: penalize large feedback gains
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
            K_init: Initial gains [N, 3], if None use zeros
            learning_rate: Learning rate
            num_iters: Number of iterations
            verbose: Print progress

        Returns:
            (K_opt, final_loss)
        """
        if K_init is None:
            K_init = np.zeros((self.N, 3))

        K_flat = jnp.array(K_init.flatten())

        # JIT compile loss and gradient
        loss_jit = jax.jit(self.loss_fn)
        grad_jit = jax.jit(jax.grad(self.loss_fn))

        if verbose:
            print(f"\nStarting optimization...")
            print(f"  K shape: [{self.N}, 3] → {self.N * 3} parameters")
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
        K_opt = np.array(K_flat.reshape(self.N, 3))

        return K_opt, final_loss

    def evaluate(self, K: np.ndarray) -> Dict:
        """
        Evaluate feedback controller and return detailed statistics.

        Args:
            K: Feedback gains [N, 3]

        Returns:
            Dictionary with evaluation results
        """
        K_traj = jnp.array(K)

        # Propagate all scenarios
        results = {}
        for scenario in self.scenarios:
            x_final, history = propagate_yaw_feedback_scenario(
                self.x0_int,
                self.x_nom_traj,
                self.u_nom_traj,
                self.yaw_nom_traj,
                K_traj,
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

        # Compute tracking error
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
    print("YAW-ONLY OUTPUT FEEDBACK CONTROLLER OPTIMIZATION")
    print("="*70)

    # Create synthetic nominal trajectory (circular)
    N = 20
    dt = 0.25
    t = np.linspace(0, (N-1)*dt, N)

    radius = 1.5
    omega_nom = 0.5

    px_nom = radius * np.cos(omega_nom * t) - radius
    py_nom = radius * np.sin(omega_nom * t)
    yaw_nom = omega_nom * t + np.pi/2
    omega_state_nom = np.ones(N) * omega_nom

    x_nom_traj = np.column_stack([px_nom, py_nom, yaw_nom, omega_state_nom])

    vx_nom = -radius * omega_nom * np.sin(omega_nom * t)
    vy_nom = radius * omega_nom * np.cos(omega_nom * t)
    u_omega_nom = np.zeros(N)

    u_nom_traj = np.column_stack([vx_nom, vy_nom, u_omega_nom])

    print(f"\nNominal trajectory:")
    print(f"  State: [px, py, yaw, omega] - 4D")
    print(f"  Control: [vx, vy, u_omega] - 3D")
    print(f"  Observation: yaw - SCALAR (position hidden!)")
    print(f"  Length: {N} steps, {N*dt}s")

    # Create scenarios
    scenarios = create_scenarios()
    print(f"\nFault scenarios:")
    for s in scenarios:
        print(f"  {s.name}:")
        print(f"    α ∈ {s.alpha_range}")
        print(f"    gyro scale ∈ {s.gyro_scale_range}")
        print(f"    gyro bias ∈ {s.gyro_bias_range}")

    # Initial state interval
    x0_int = Interval(
        lower=jnp.array([-0.03, -0.03, -0.02, -0.01]),
        upper=jnp.array([0.03, 0.03, 0.02, 0.01])
    )

    print(f"\nInitial state uncertainty:")
    print(f"  px ∈ [{x0_int.lower[0]:.3f}, {x0_int.upper[0]:.3f}] m")
    print(f"  py ∈ [{x0_int.lower[1]:.3f}, {x0_int.upper[1]:.3f}] m")
    print(f"  yaw ∈ [{x0_int.lower[2]:.3f}, {x0_int.upper[2]:.3f}] rad")
    print(f"  omega ∈ [{x0_int.lower[3]:.3f}, {x0_int.upper[3]:.3f}] rad/s")

    # Create optimizer
    optimizer = YawOnlyFeedbackOptimizer(
        x_nom_traj=x_nom_traj,
        u_nom_traj=u_nom_traj,
        scenarios=scenarios,
        x0_int=x0_int,
        dt=dt,
        lambda_sep=1.0,
        lambda_track=0.02,
        obs_uncertainty=0.02
    )

    # Optimize
    print("\n" + "="*70)
    print("OPTIMIZING YAW-ONLY FEEDBACK GAINS")
    print("="*70)

    K_opt, loss_opt = optimizer.optimize(
        learning_rate=0.02,
        num_iters=100,
        verbose=True
    )

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)

    # Evaluate
    stats = optimizer.evaluate(K_opt)

    print(f"\nResults:")
    print(f"  Total overlap: {stats['total_overlap']:.6f} m²")
    print(f"  Tracking error: {stats['tracking_error']:.6f}")

    print(f"\nPairwise overlaps:")
    for key, val in stats['overlaps'].items():
        print(f"  {key}: {val:.6f} m²")

    print(f"\nFinal position intervals:")
    for name, pos_int in stats['position_intervals'].items():
        print(f"  {name}:")
        print(f"    px ∈ [{pos_int.lower[0]:+.4f}, {pos_int.upper[0]:+.4f}] m")
        print(f"    py ∈ [{pos_int.lower[1]:+.4f}, {pos_int.upper[1]:+.4f}] m")

    print(f"\nOptimal yaw-only feedback gain statistics:")
    print(f"  Shape: {K_opt.shape} → {K_opt.size} total parameters")
    print(f"  Each timestep: K[t] ∈ ℝ³ maps scalar yaw error to [vx, vy, u_omega]")
    print(f"  Mean: {np.mean(K_opt):.6f}")
    print(f"  Std:  {np.std(K_opt):.6f}")
    print(f"  Max:  {np.max(np.abs(K_opt)):.6f}")

    # Show example gains
    print(f"\nExample gains at t=0:")
    print(f"  K[0] = [{K_opt[0, 0]:.4f}, {K_opt[0, 1]:.4f}, {K_opt[0, 2]:.4f}]")
    print(f"  Interpretation:")
    print(f"    If yaw error = +1 rad:")
    print(f"      Δvx    = {K_opt[0, 0]:+.4f} m/s")
    print(f"      Δvy    = {K_opt[0, 1]:+.4f} m/s")
    print(f"      Δu_ω   = {K_opt[0, 2]:+.4f} rad/s²")

    print("\n" + "="*70)
    print("✓ Yaw-only feedback controller optimization complete")
    print("="*70)
