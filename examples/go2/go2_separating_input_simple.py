"""
Separating Input Optimizer for Unitree Go2 - Self-Contained Version

Active fault diagnosis: compute inputs that maximize separation between
nominal, sensor fault, and actuator fault reachable sets.

This version uses JAX and implements interval arithmetic directly.
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class Interval:
    """Simple interval representation"""
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
    scale_range: Tuple[float, float]  # sensor scale (legacy display field)
    bias_range: Tuple[float, float]   # sensor bias (rad, legacy display field)
    # Velocity sensor fault fields
    v_negated: bool = False                    # velocity measurement is negated
    v_bias_range: Tuple[float, float] = (0.0, 0.0)    # additive bias on v_meas [m/s]
    yaw_bias_range: Tuple[float, float] = (0.0, 0.0)  # additive bias on ω_meas [rad/s]


def create_scenarios() -> List[FaultScenario]:
    """Create nominal, sensor fault, actuator fault, and velocity sensor fault scenarios"""
    return [
        FaultScenario(
            name="Nominal",
            alpha_range=(1.0, 1.0),
            scale_range=(1.0, 1.0),
            bias_range=(0.0, 0.0),
        ),
        # FaultScenario(
        #     name="Sensor Fault",
        #     alpha_range=(1.0, 1.0),
        #     scale_range=(1.05, 1.10),  # 5-10% scale error
        #     bias_range=(0.087, 0.262)  # 5-15 degrees
        # ),
        FaultScenario(
            name="Actuator Fault",
            alpha_range=(0.60, 0.80),  # 20-40% reduction
            scale_range=(1.0, 1.0),
            bias_range=(0.0, 0.0),
        ),
        FaultScenario(
            name="Velocity Sensor Fault",
            alpha_range=(1.0, 1.0),   # dynamics unchanged
            scale_range=(1.0, 1.0),
            bias_range=(0.0, 0.0),
            v_negated=True,
            # v_meas = -v_true + bias,  bias ∈ [0.05, 0.15] m/s
            v_bias_range=(0.05, 0.15),
            # ω_meas = ω_true + bias_ω,  bias_ω ∈ [0.02, 0.06] rad/s (~1–3.4 °/s)
            yaw_bias_range=(0.02, 0.06),
        ),
    ]


def interval_dynamics(x_int: Interval, u: jnp.ndarray,
                     alpha_int: Interval, dt: float) -> Interval:
    """
    Propagate state interval through unicycle dynamics with actuator fault.

    dx/dt = vx * cos(theta) - vy * sin(theta)
    dy/dt = vx * sin(theta) + vy * cos(theta)
    dtheta/dt = alpha * omega

    Args:
        x_int: State interval [px, py, theta]
        u: Control [vx, vy, omega]
        alpha_int: Actuator effectiveness interval
        dt: Time step

    Returns:
        Next state interval
    """
    px_l, py_l, th_l = x_int.lower[0], x_int.lower[1], x_int.lower[2]
    px_u, py_u, th_u = x_int.upper[0], x_int.upper[1], x_int.upper[2]

    vx, vy, omega = u[0], u[1], u[2]
    alpha_l, alpha_u = alpha_int.lower[0], alpha_int.upper[0]

    # For each state derivative, compute interval bounds
    # Using interval arithmetic with cos/sin ranges

    # cos(theta) range
    cos_vals = jnp.array([jnp.cos(th_l), jnp.cos(th_u)])
    cos_min, cos_max = jnp.min(cos_vals), jnp.max(cos_vals)
    # Handle wrapping - use where instead of if for JAX compatibility
    cos_min = jnp.where(th_u - th_l > jnp.pi, -1.0, cos_min)
    cos_max = jnp.where((th_l < jnp.pi) & (jnp.pi < th_u), 1.0, cos_max)
    cos_max = jnp.where((th_l < -jnp.pi) & (-jnp.pi < th_u), 1.0, cos_max)

    # sin(theta) range
    sin_vals = jnp.array([jnp.sin(th_l), jnp.sin(th_u)])
    sin_min, sin_max = jnp.min(sin_vals), jnp.max(sin_vals)
    sin_min = jnp.where(th_u - th_l > jnp.pi, -1.0, sin_min)
    sin_max = jnp.where((th_l < jnp.pi/2) & (jnp.pi/2 < th_u), 1.0, sin_max)
    sin_max = jnp.where((th_l < -3*jnp.pi/2) & (-3*jnp.pi/2 < th_u), 1.0, sin_max)

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


def interval_output_step(x_int: Interval,
                         vx_eff_int: Interval,
                         vy_eff: float,
                         omega_eff_int: Interval,
                         dt: float) -> Interval:
    """
    One Euler step for the *output* (measured) state with interval-valued velocities.

    Used for the velocity sensor fault, where the robot's odometry integrates
    a corrupted velocity:
        v_meas  = -v_true + bias_v    (negated + additive bias interval)
        ω_meas  = ω_true  + bias_ω   (correct direction + small bias interval)

    Dynamics:
        dpx_out/dt = vx_eff * cos(yaw_out) - vy_eff * sin(yaw_out)
        dpy_out/dt = vx_eff * sin(yaw_out) + vy_eff * cos(yaw_out)
        dyaw_out/dt = omega_eff

    Because vx_eff is an interval [vx_l, vx_u] (not a scalar), the products
    vx_eff * cos and vx_eff * sin require full interval multiplication
    (all four corner products).

    Args:
        x_int:        Output state interval [px_out, py_out, yaw_out]
        vx_eff_int:   Effective forward velocity interval, e.g. [-vx+b_l, -vx+b_u]
        vy_eff:       Effective lateral velocity (scalar, negated for sensor fault)
        omega_eff_int: Effective yaw-rate interval, e.g. [ω+b_l, ω+b_u]
        dt:           Time step

    Returns:
        Next output state interval
    """
    px_l, py_l, th_l = x_int.lower[0], x_int.lower[1], x_int.lower[2]
    px_u, py_u, th_u = x_int.upper[0], x_int.upper[1], x_int.upper[2]

    vx_l, vx_u     = vx_eff_int.lower[0], vx_eff_int.upper[0]
    omega_l, omega_u = omega_eff_int.lower[0], omega_eff_int.upper[0]

    # --- cos(yaw_out) range over [th_l, th_u] ---
    cos_vals = jnp.array([jnp.cos(th_l), jnp.cos(th_u)])
    cos_min, cos_max = jnp.min(cos_vals), jnp.max(cos_vals)
    cos_min = jnp.where(th_u - th_l > jnp.pi, -1.0, cos_min)
    cos_max = jnp.where((th_l < jnp.pi) & (jnp.pi < th_u), 1.0, cos_max)
    cos_max = jnp.where((th_l < -jnp.pi) & (-jnp.pi < th_u), 1.0, cos_max)

    # --- sin(yaw_out) range ---
    sin_vals = jnp.array([jnp.sin(th_l), jnp.sin(th_u)])
    sin_min, sin_max = jnp.min(sin_vals), jnp.max(sin_vals)
    sin_min = jnp.where(th_u - th_l > jnp.pi, -1.0, sin_min)
    sin_max = jnp.where((th_l < jnp.pi/2) & (jnp.pi/2 < th_u), 1.0, sin_max)
    sin_max = jnp.where((th_l < -3*jnp.pi/2) & (-3*jnp.pi/2 < th_u), 1.0, sin_max)

    # --- dpx_out/dt = vx_eff * cos(yaw) - vy_eff * sin(yaw) ---
    # vx_eff ∈ [vx_l, vx_u] is an interval: full 4-corner multiplication
    vx_cos = jnp.array([vx_l * cos_min, vx_l * cos_max,
                         vx_u * cos_min, vx_u * cos_max])
    vy_sin = -vy_eff * jnp.array([sin_min, sin_max])   # vy_eff is scalar
    dpx_min = jnp.min(vx_cos) + jnp.min(vy_sin)
    dpx_max = jnp.max(vx_cos) + jnp.max(vy_sin)

    # --- dpy_out/dt = vx_eff * sin(yaw) + vy_eff * cos(yaw) ---
    vx_sin = jnp.array([vx_l * sin_min, vx_l * sin_max,
                         vx_u * sin_min, vx_u * sin_max])
    vy_cos = vy_eff * jnp.array([cos_min, cos_max])
    dpy_min = jnp.min(vx_sin) + jnp.min(vy_cos)
    dpy_max = jnp.max(vx_sin) + jnp.max(vy_cos)

    # --- dyaw_out/dt = omega_eff ∈ [omega_l, omega_u] ---
    return Interval(
        lower=jnp.array([px_l + dt * dpx_min,
                          py_l + dt * dpy_min,
                          th_l + dt * omega_l]),
        upper=jnp.array([px_u + dt * dpx_max,
                          py_u + dt * dpy_max,
                          th_u + dt * omega_u]),
    )


def propagate_scenario(x0_int: Interval, u: jnp.ndarray,
                      scenario: FaultScenario, dt: float,
                      num_steps: int) -> Interval:
    """
    Propagate initial interval and return the *output* (measurement) interval.

    For all scenarios the output is what an observer would measure:

    * **Nominal / Actuator Fault** – no sensor corruption, so the output equals
      the true state.  The actuator effectiveness α ∈ alpha_range scales ω in
      the true dynamics, creating different trajectories.

    * **Velocity Sensor Fault** (v_negated=True) – dynamics are unchanged
      (α = 1), but the odometry integrates a corrupted velocity measurement:
          v_meas  = -v_true + bias_v,   bias_v  ∈ v_bias_range   [m/s]
          ω_meas  =  ω_true + bias_ω,   bias_ω  ∈ yaw_bias_range [rad/s]
      The output state (px_meas, py_meas, yaw_meas) is propagated separately
      using interval_output_step, which handles the interval-valued vx_eff.

    Args:
        x0_int:    Initial interval (same starting point for all scenarios)
        u:         Constant control input [vx, vy, omega]
        scenario:  Fault scenario parameters
        dt:        Time step
        num_steps: Number of Euler steps

    Returns:
        Final *output* interval [px_out, py_out, yaw_out]
    """
    if not scenario.v_negated:
        # ── Standard path: output = true state ──────────────────────────────
        alpha_int = Interval(
            lower=jnp.array([scenario.alpha_range[0]]),
            upper=jnp.array([scenario.alpha_range[1]])
        )
        x_current = x0_int
        for _ in range(num_steps):
            x_current = interval_dynamics(x_current, u, alpha_int, dt)
        return x_current

    else:
        # ── Velocity sensor fault: output integrates corrupted measurement ──
        vx, vy, omega = u[0], u[1], u[2]

        # Effective measured velocity: v_meas = -v_true + bias
        #   Forward: vx_eff ∈ [-vx + b_l, -vx + b_u]
        #   Lateral: vy_eff = -vy  (scalar, same negation, no separate bias)
        vx_eff_int = Interval(
            lower=jnp.array([-vx + scenario.v_bias_range[0]]),
            upper=jnp.array([-vx + scenario.v_bias_range[1]]),
        )
        vy_eff = -vy

        # Effective measured yaw rate: ω_meas = ω_true + bias_ω
        omega_eff_int = Interval(
            lower=jnp.array([omega + scenario.yaw_bias_range[0]]),
            upper=jnp.array([omega + scenario.yaw_bias_range[1]]),
        )

        x_out = x0_int
        for _ in range(num_steps):
            x_out = interval_output_step(x_out, vx_eff_int, vy_eff, omega_eff_int, dt)
        return x_out


def interval_overlap_volume(int1: Interval, int2: Interval):
    """
    Compute volume of intersection between two intervals.

    Returns 0 if no overlap.
    """
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


def compute_separation_loss(u: jnp.ndarray, x0_int: Interval,
                           scenarios: List[FaultScenario],
                           dt: float, num_steps: int,
                           position_only: bool = True) -> float:
    """
    Compute total overlap between all scenario pairs.

    Lower is better (more separation).

    Args:
        u: Control input
        x0_int: Initial state interval
        scenarios: List of fault scenarios
        dt: Time step
        num_steps: Propagation steps
        position_only: If True, only consider px, py for overlap

    Returns:
        Total pairwise overlap volume
    """
    # Propagate each scenario
    final_intervals = []
    for scenario in scenarios:
        x_final = propagate_scenario(x0_int, u, scenario, dt, num_steps)

        if position_only:
            # Extract only position (px, py)
            x_final = Interval(
                lower=x_final.lower[:2],
                upper=x_final.upper[:2]
            )

        final_intervals.append(x_final)

    # Compute pairwise overlaps
    total_overlap = 0.0
    for i in range(len(final_intervals)):
        for j in range(i + 1, len(final_intervals)):
            overlap = interval_overlap_volume(final_intervals[i], final_intervals[j])
            total_overlap += overlap

    return total_overlap


class SeparatingInputOptimizer:
    """Optimizer for finding separating inputs using gradient descent"""

    def __init__(self, scenarios: List[FaultScenario], x0_int: Interval,
                 dt: float, num_steps: int, position_only: bool = True):
        self.scenarios = scenarios
        self.x0_int = x0_int
        self.dt = dt
        self.num_steps = num_steps
        self.position_only = position_only

        # Create loss function and gradient
        def loss_fn(u):
            return compute_separation_loss(
                u, x0_int, scenarios, dt, num_steps, position_only
            )

        self.loss_fn = jax.jit(loss_fn)
        self.grad_fn = jax.jit(jax.grad(loss_fn))

    def optimize(self, u_init: Optional[jnp.ndarray] = None,
                learning_rate: float = 0.01, num_iters: int = 200,
                verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        """
        Run gradient descent.

        Returns:
            (optimal_u, final_loss)
        """
        if u_init is None:
            u_init = jnp.array([0.5, 0.0, 0.3])  # Default: forward + turn

        u = u_init

        for i in range(num_iters):
            grad = self.grad_fn(u)
            u = u - learning_rate * grad

            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = self.loss_fn(u)
                print(f"Iter {i:3d}: loss={loss:.6f}, "
                      f"u=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}], "
                      f"|grad|={jnp.linalg.norm(grad):.6f}")

        final_loss = self.loss_fn(u)
        return u, final_loss

    def evaluate(self, u: jnp.ndarray) -> Dict:
        """
        Evaluate a control input and return detailed statistics.
        """
        # Propagate all scenarios
        final_intervals = []
        for scenario in self.scenarios:
            x_final = propagate_scenario(self.x0_int, u, scenario,
                                        self.dt, self.num_steps)
            final_intervals.append(x_final)

        # Position-only intervals for overlap
        pos_intervals = [
            Interval(lower=x.lower[:2], upper=x.upper[:2])
            for x in final_intervals
        ]

        # Compute pairwise overlaps
        overlaps = {}
        n = len(self.scenarios)
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlap = interval_overlap_volume(pos_intervals[i], pos_intervals[j])
                overlaps[key] = overlap

        # Compute volumes
        volumes = {}
        for i, scenario in enumerate(self.scenarios):
            vol = jnp.prod(pos_intervals[i].upper - pos_intervals[i].lower)
            volumes[scenario.name] = float(vol)  # Convert at the end for display only

        return {
            'final_intervals': final_intervals,
            'position_intervals': pos_intervals,
            'pairwise_overlaps': overlaps,
            'volumes': volumes,
            'total_overlap': sum(overlaps.values())
        }


def optimize_multistart(scenarios: List[FaultScenario], x0_int: Interval,
                       dt: float, num_steps: int, num_restarts: int = 5,
                       learning_rate: float = 0.01, num_iters: int = 200,
                       verbose: bool = False) -> Tuple[jnp.ndarray, float, Dict]:
    """
    Multi-start optimization to find global optimum.

    Returns:
        (best_u, best_loss, best_stats)
    """
    optimizer = SeparatingInputOptimizer(scenarios, x0_int, dt, num_steps)

    best_u = None
    best_loss = float('inf')
    best_stats = None

    key = jax.random.PRNGKey(42)

    for restart in range(num_restarts):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Restart {restart+1}/{num_restarts}")
            print('='*60)

        # Random initialization
        key, subkey = jax.random.split(key)
        u_init = jax.random.normal(subkey, (3,)) * 0.3 + jnp.array([0.5, 0.0, 0.3])

        # Optimize
        u_opt, loss = optimizer.optimize(
            u_init=u_init,
            learning_rate=learning_rate,
            num_iters=num_iters,
            verbose=verbose
        )

        # Evaluate
        stats = optimizer.evaluate(u_opt)

        if verbose:
            print(f"\nResult: loss={loss:.6f}")

        if loss < best_loss:
            best_loss = loss
            best_u = u_opt
            best_stats = stats
            if verbose:
                print("  → New best!")

    return best_u, best_loss, best_stats


if __name__ == "__main__":
    print("="*70)
    print("GO2 SEPARATING INPUT OPTIMIZATION")
    print("="*70)

    # Create scenarios
    scenarios = create_scenarios()
    print(f"\nFault scenarios:")
    for s in scenarios:
        print(f"  {s.name}:")
        print(f"    α ∈ [{s.alpha_range[0]:.2f}, {s.alpha_range[1]:.2f}]")
        print(f"    a ∈ [{s.scale_range[0]:.2f}, {s.scale_range[1]:.2f}]")
        print(f"    b ∈ [{np.degrees(s.bias_range[0]):.1f}°, "
              f"{np.degrees(s.bias_range[1]):.1f}°]")

    # Initial state interval (small uncertainty)
    x0_int = Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([0.05, 0.05, 0.02])
    )

    print(f"\nInitial state interval:")
    print(f"  px ∈ [{x0_int.lower[0]:.3f}, {x0_int.upper[0]:.3f}] m")
    print(f"  py ∈ [{x0_int.lower[1]:.3f}, {x0_int.upper[1]:.3f}] m")
    print(f"  θ  ∈ [{x0_int.lower[2]:.3f}, {x0_int.upper[2]:.3f}] rad")

    # Optimization settings
    dt = 0.5
    num_steps = 10
    total_time = dt * num_steps

    print(f"\nPropagation: {num_steps} steps × {dt}s = {total_time}s")
    print(f"Objective: Maximize separation in position space (px, py)")

    # Run optimization
    print("\nRunning multi-start optimization...")
    u_opt, loss_opt, stats = optimize_multistart(
        scenarios=scenarios,
        x0_int=x0_int,
        dt=dt,
        num_steps=num_steps,
        num_restarts=3,
        learning_rate=0.01,
        num_iters=150,
        verbose=True
    )

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nOptimal separating input:")
    print(f"  vx    = {u_opt[0]:+.4f} m/s")
    print(f"  vy    = {u_opt[1]:+.4f} m/s")
    print(f"  omega = {u_opt[2]:+.4f} rad/s")

    print(f"\nTotal overlap (position): {loss_opt:.6f} m²")

    print(f"\nPairwise overlaps:")
    for key, val in stats['pairwise_overlaps'].items():
        print(f"  {key}: {val:.6f} m²")

    print(f"\nReachable set volumes:")
    for key, val in stats['volumes'].items():
        print(f"  {key}: {val:.6f} m²")

    print(f"\nFinal position intervals:")
    for i, scenario in enumerate(scenarios):
        pos_int = stats['position_intervals'][i]
        print(f"  {scenario.name}:")
        print(f"    px ∈ [{pos_int.lower[0]:+.4f}, {pos_int.upper[0]:+.4f}] m")
        print(f"    py ∈ [{pos_int.lower[1]:+.4f}, {pos_int.upper[1]:+.4f}] m")

    print("\n" + "="*70)
    print("✓ Optimization complete")
    print("="*70)
