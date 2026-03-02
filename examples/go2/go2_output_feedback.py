"""
GO2 Linear Output-Feedback Controller for Active Fault Diagnosis
================================================================
Solves for a linear output-feedback controller K ∈ R^{3×3}, r ∈ R^3 that
maximally separates the reachable sets of the three fault scenarios in the
(px, py) position space:

    u_k = clip(K @ y_k + r,  u_lo, u_hi)

where y_k is the robot's *observed* state at step k:

  Nominal / Actuator-fault scenario  : y = x  (true state)
  Sensor-fault scenario              : y = x̂  (dead-reckoned estimated state)

The key insight is that each immrax system's state IS the observed quantity
that feeds the controller, so passing theta = [K.flatten(), r] as the "u"
argument to the natural embedding lets the closed-loop vector field

    f_cl(x, theta, p) = f(x,  K @ x + r,  p)

be bounded by the natural embedding with x abstract (interval) and theta
concrete (the optimization variable we differentiate through).

Decision variable
-----------------
    theta ∈ R^12 = [K.flatten(), r]     K : (3,3) gain,  r : (3,) feedforward

Optimization
------------
    min_{theta}  Σ_{i<j}  overlap( pos_interval_i(theta),  pos_interval_j(theta) )

solved with batched gradient descent (same GPU-parallel structure as
optimize_multistep_gpu_rejit in go2_separating_input_immrax.py).
"""

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np
from functools import partial
from typing import List, Tuple, Optional, Dict

from go2.go2_separating_input_immrax import (
    Scenario,
    euler_step,
    propagate_scenario,
    position_interval,
    optimize_parallel_gpu,   # reused for the GPU loop
    _U_LO, _U_HI,
)
from faulty_car.interval_functions import overlap_size_lax


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Controller parameterisation
# ══════════════════════════════════════════════════════════════════════════════

# K entries are clipped to [-K_MAX, K_MAX]; r is clipped to the control box.
_K_MAX = 2.0
_THETA_LO = jnp.concatenate([jnp.full(9, -_K_MAX), _U_LO])   # (12,)
_THETA_HI = jnp.concatenate([jnp.full(9,  _K_MAX), _U_HI])   # (12,)


def _project_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """Project controller parameters onto the feasible box.

    Works for any leading batch dimensions.
    K entries ∈ [-K_MAX, K_MAX],  r ∈ [u_lo, u_hi].
    """
    return jnp.clip(theta, _THETA_LO, _THETA_HI)


def theta_to_K_r(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Unpack flat theta into (K, r)."""
    return theta[:9].reshape(3, 3), theta[9:12]


def K_r_to_theta(K: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """Pack (K, r) into flat theta."""
    return jnp.concatenate([K.flatten(), r])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Closed-loop system classes
# ══════════════════════════════════════════════════════════════════════════════

class Go2NomActCLSystem(irx.System):
    """Nominal / actuator-fault unicycle under linear output feedback.

    In the immrax convention the function signature is f(t, x, u, p).
    For the closed-loop system **u carries the controller parameters**:

        u = theta = [K.flatten(), r]  ∈ R^12     (concrete, optimisation variable)
        p = [alpha, beta]             ∈ R^2       (abstract, fault parameters)

    The natural embedding traces this function with x and p as abstract
    (interval) inputs and theta concrete, so the closed-loop vector field

        f_cl(x, theta, p) = unicycle( x,  clip(K @ x + r, u_lo, u_hi),  p )

    is correctly bounded over x ∈ X and p ∈ P for any fixed theta.

    Dynamics (same as Go2NomActSystem but with state-dependent control):
        ṗx = vx·cos θ − α·vy·sin θ
        ṗy = vx·sin θ + α·vy·cos θ
        θ̇  = β·ω
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        K = u[:9].reshape(3, 3)
        r = u[9:12]
        ctrl  = jnp.clip(K @ x + r, _U_LO, _U_HI)
        vx, vy, omega = ctrl[0], ctrl[1], ctrl[2]
        alpha, beta = p[0], p[1]
        th = x[2]
        return jnp.array([
            vx * jnp.cos(th) - alpha * vy * jnp.sin(th),
            vx * jnp.sin(th) + alpha * vy * jnp.cos(th),
            beta * omega,
        ])


class Go2SensorFaultCLSystem(irx.System):
    """Sensor-fault dead-reckoned pose under linear output feedback.

    The robot's odometry tracks the *estimated* position x̂ = [p̂x, p̂y, θ̂].
    The controller receives this estimated state:

        u = clip(K @ x̂ + r, u_lo, u_hi)

    The lateral-velocity sensor is faulty, so the odometry integrates

        vy_corrupted = (1 − ω²)·α·vy + ω²·vy_noise

    instead of the true commanded vy.

    Signature (immrax convention):
        u = theta = [K.flatten(), r] ∈ R^12      (concrete, controller params)
        p = [vy_noise, alpha, beta]  ∈ R^3        (abstract, fault params)
            vy_noise ∈ [−ε, +ε],  alpha = beta = 1 for pure sensor fault

    Dynamics (what the onboard odometry integrates):
        ṗ̂x = vx·cos θ̂ − vy_corrupted·sin θ̂
        ṗ̂y = vx·sin θ̂ + vy_corrupted·cos θ̂
        θ̂̇  = β·ω
    """

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        K = u[:9].reshape(3, 3)
        r = u[9:12]
        ctrl  = jnp.clip(K @ x + r, _U_LO, _U_HI)
        vx, vy, omega = ctrl[0], ctrl[1], ctrl[2]
        vy_noise, alpha, beta = p[0], p[1], p[2]
        th = x[2]
        vy_corrupted = (1 - omega ** 2) * alpha * vy + omega ** 2 * vy_noise
        return jnp.array([
            vx * jnp.cos(th) - vy_corrupted * jnp.sin(th),
            vx * jnp.sin(th) + vy_corrupted * jnp.cos(th),
            beta * omega,
        ])


# Module-level singletons — created once, reused across calls.
_NOM_ACT_CL_SYS = Go2NomActCLSystem()
_NOM_ACT_CL_EMB = irx.natemb(_NOM_ACT_CL_SYS)

_SF_CL_SYS = Go2SensorFaultCLSystem()
_SF_CL_EMB = irx.natemb(_SF_CL_SYS)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Fault scenarios (closed-loop)
# ══════════════════════════════════════════════════════════════════════════════

def create_cl_scenarios(
    actuator_alpha_lo: float = 0.60,
    actuator_alpha_hi: float = 0.80,
    actuator_beta_lo:  float = 0.60,
    actuator_beta_hi:  float = 0.80,
    sensor_noise_bound: float = 0.25,
) -> List[Scenario]:
    """Three closed-loop fault scenarios for output-feedback optimisation.

    Mirrors create_scenarios() from go2_separating_input_immrax but uses the
    closed-loop system embeddings (_NOM_ACT_CL_EMB, _SF_CL_EMB).  The p_interval
    encodes only the *fault* parameters; the controller parameters are passed
    separately as the 'u' argument during propagation.

    Returns
    -------
    [Nominal, Actuator-Fault, Sensor-Fault]  —  list of Scenario objects.
    """
    return [
        Scenario(
            name="Nominal",
            emb_system=_NOM_ACT_CL_EMB,
            p_interval=irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2)),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=_NOM_ACT_CL_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo, actuator_beta_lo]),
                upper=jnp.array([actuator_alpha_hi, actuator_beta_hi]),
            ),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=_SF_CL_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([-sensor_noise_bound, 1.0, 1.0]),
                upper=jnp.array([ sensor_noise_bound, 1.0, 1.0]),
            ),
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Separation loss under output feedback
# ══════════════════════════════════════════════════════════════════════════════

def separation_loss_cl(
    theta: jnp.ndarray,
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    num_steps: int,
) -> jnp.ndarray:
    """Sum of pairwise position-interval overlaps under linear output feedback.

    For each closed-loop scenario, propagates the initial state interval
    num_steps Euler steps under the controller u = clip(K @ x + r, ...).
    Returns the sum of pairwise 2-D position-interval overlaps — minimising
    this drives the fault scenarios apart in (px, py) space.

    Because the closed-loop system classes encode K and r *inside* their f(),
    we reuse propagate_scenario() verbatim by passing theta as the 'u' slot:

        propagate_scenario(x0_ivl, theta, scenario, dt, num_steps)
             ↕
        euler_step(emb, x_ivl, theta, p_ivl, dt)
             ↕
        emb.f(t, x_ut, theta, p_ivl)          # CL system interprets theta as K, r

    Parameters
    ----------
    theta       : (12,) controller params  [K.flatten(), r]
    x0_ivl      : initial state interval
    cl_scenarios: list from create_cl_scenarios()
    dt          : Euler step size (s)
    num_steps   : number of Euler steps in the horizon

    Returns
    -------
    Scalar overlap volume (m²).  Zero = scenarios fully separated.
    """
    pos_ivls = [
        position_interval(propagate_scenario(x0_ivl, theta, s, dt, num_steps))
        for s in cl_scenarios
    ]
    n = len(pos_ivls)
    total = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            total = total + overlap_size_lax(pos_ivls[i], pos_ivls[j])
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Optimizer
# ══════════════════════════════════════════════════════════════════════════════

class OutputFeedbackOptimizer:
    """Gradient-descent optimizer for the linear output-feedback controller.

    Wraps separation_loss_cl and its gradient in JIT-compiled callables.
    The interface matches SeparatingInputOptimizer so it can slot into any
    existing GPU-parallel driver that calls .loss_fn and .grad_fn.

    Decision variable : theta ∈ R^12 = [K.flatten(), r]
    Controller        : u_k = clip(K @ y_k + r,  u_lo, u_hi)
    Objective         : min_theta  Σ_{i<j} overlap(pos_ivl_i, pos_ivl_j)
    """

    def __init__(
        self,
        cl_scenarios: List[Scenario],
        x0_ivl: irx.Interval,
        dt: float,
        num_steps: int,
    ):
        self.cl_scenarios = cl_scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.num_steps = num_steps

        _loss = partial(
            separation_loss_cl,
            x0_ivl=x0_ivl,
            cl_scenarios=cl_scenarios,
            dt=dt,
            num_steps=num_steps,
        )
        self.loss_fn = jax.jit(_loss)            # (12,) → scalar
        self.grad_fn = jax.jit(jax.grad(_loss))  # (12,) → (12,)

    def evaluate(self, theta: jnp.ndarray) -> Dict:
        """Return position intervals, pairwise overlaps, and volumes for theta."""
        K, r = theta_to_K_r(theta)
        pos_ivls = [
            position_interval(
                propagate_scenario(self.x0_ivl, theta, s, self.dt, self.num_steps)
            )
            for s in self.cl_scenarios
        ]
        n = len(self.cl_scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.cl_scenarios[i].name} vs {self.cl_scenarios[j].name}"
                overlaps[key] = float(overlap_size_lax(pos_ivls[i], pos_ivls[j]))
        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.cl_scenarios, pos_ivls)
        }
        return {
            'K': np.array(K),
            'r': np.array(r),
            'position_intervals': pos_ivls,
            'pairwise_overlaps':  overlaps,
            'volumes':            volumes,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  GPU-parallel optimisation
# ══════════════════════════════════════════════════════════════════════════════

def optimize_output_feedback_gpu(
    cl_scenarios: List[Scenario],
    x0_ivl: irx.Interval,
    dt: float,
    num_steps: int,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 200,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GPU-parallel multi-start gradient descent over theta ∈ R^12.

    Initialises num_restarts random controller parameters, runs gradient
    descent on all in parallel via jax.vmap + jax.lax.fori_loop, and returns
    the best result.  Mirrors optimize_parallel_gpu but acts on theta (12,)
    rather than a constant control u (3,).

    Initialisation:  K ≈ 0  (small Gaussian),  r ≈ [0.5, 0.0, 0.3].

    Returns
    -------
    (best_theta, best_loss, all_theta_final, all_losses)
    """
    key = jax.random.PRNGKey(seed)
    theta_init_mean = jnp.concatenate([jnp.zeros(9), jnp.array([0.5, 0.0, 0.3])])
    theta0 = (
        jax.random.normal(key, (num_restarts, 12)) * 0.1 + theta_init_mean
    )

    def feedback_loss(u):
        return separation_loss_cl(
            theta=u,
            x0_ivl=x0_ivl,
            cl_scenarios=cl_scenarios,
            
        )

    batched_loss = jax.vmap(opt.loss_fn)   # (R, 12) → (R,)
    batched_grad = jax.vmap(opt.grad_fn)   # (R, 12) → (R, 12)

    def body(_, theta_batch):
        g = batched_grad(theta_batch)
        return _project_theta(theta_batch - learning_rate * g)

    theta_final = jax.lax.fori_loop(0, num_iters, body, theta0)
    losses = batched_loss(theta_final)

    best_idx   = jnp.argmin(losses)
    best_theta = theta_final[best_idx]
    best_loss  = losses[best_idx]
    return best_theta, best_loss, theta_final, losses


def optimize_output_feedback(
    x0_ivl: irx.Interval,
    cl_scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 200,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Solve for the linear output-feedback controller that maximises fault-scenario separation.

    Finds K ∈ R^{3×3} and r ∈ R^3 such that the closed-loop trajectories

        u_k = clip(K @ y_k + r,  u_lo, u_hi)

    produce maximally separated reachable sets in (px, py) across the three
    fault scenarios.  The observed output y_k is scenario-dependent:

      Nominal / Actuator fault : y_k = x_k  (true unicycle state)
      Sensor fault             : y_k = x̂_k  (dead-reckoned estimated state)

    Uses GPU-parallel gradient descent over num_restarts random initialisations
    (same pattern as optimize_multistep_gpu_rejit).

    Parameters
    ----------
    x0_ivl      : initial state interval
    cl_scenarios: closed-loop fault scenarios from create_cl_scenarios()
    dt          : Euler step size (s)
    num_steps   : number of Euler steps in the evaluation horizon
    num_restarts: number of parallel random initialisations
    learning_rate: gradient descent step size
    num_iters   : number of gradient steps per restart
    seed        : PRNG seed for random initialisation

    Returns
    -------
    (K, r, loss)
    K    : (3, 3) optimal output-feedback gain matrix
    r    : (3,)   optimal feedforward term
    loss : final overlap (m²); zero = full separation across all fault modes
    """
    opt = OutputFeedbackOptimizer(cl_scenarios, x0_ivl, dt, num_steps)
    best_theta, best_loss, _, _ = optimize_output_feedback_gpu(
        opt,
        num_restarts=num_restarts,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=seed,
    )
    K, r = theta_to_K_r(best_theta)
    return K, r, float(best_loss)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("GO2 OUTPUT-FEEDBACK CONTROLLER — fault-separation optimisation")
    print("=" * 70)

    cl_scenarios = create_cl_scenarios()
    print(f"\n{len(cl_scenarios)} closed-loop fault scenarios:")
    for s in cl_scenarios:
        print(f"  • {s.name}")
        print(f"    p ∈ [{np.array(s.p_interval.lower)}, {np.array(s.p_interval.upper)}]")

    x0_ivl = irx.Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([ 0.05,  0.05,  0.02]),
    )
    dt, num_steps = 0.2, 10
    print(f"\nInitial state interval:")
    print(f"  px ∈ [{float(x0_ivl.lower[0]):.3f}, {float(x0_ivl.upper[0]):.3f}] m")
    print(f"  py ∈ [{float(x0_ivl.lower[1]):.3f}, {float(x0_ivl.upper[1]):.3f}] m")
    print(f"  θ  ∈ [{float(x0_ivl.lower[2]):.3f}, {float(x0_ivl.upper[2]):.3f}] rad")
    print(f"\nHorizon: {num_steps} × {dt} s = {num_steps * dt:.1f} s")

    print("\nRunning output-feedback optimisation …\n")
    K, r, loss = optimize_output_feedback(
        x0_ivl, cl_scenarios, dt, num_steps,
        num_restarts=50, learning_rate=0.01, num_iters=200, seed=42,
    )

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOptimal gain matrix K:")
    for row in K:
        print(f"  [{row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f}]")
    print(f"\nFeedforward r: [{r[0]:+.4f},  {r[1]:+.4f},  {r[2]:+.4f}]")
    print(f"\nOverlap loss : {loss:.6f} m²")

    # Diagnostics
    opt = OutputFeedbackOptimizer(cl_scenarios, x0_ivl, dt, num_steps)
    theta = K_r_to_theta(K, r)
    stats = opt.evaluate(theta)
    print(f"\nPairwise overlaps:")
    for k, v in stats['pairwise_overlaps'].items():
        print(f"  {k}: {v:.6f} m²")
    print(f"\nPosition interval volumes:")
    for k, v in stats['volumes'].items():
        print(f"  {k}: {v:.6f} m²")
    print(f"\nFinal position intervals:")
    for s, iv in zip(cl_scenarios, stats['position_intervals']):
        wx = float(iv.upper[0] - iv.lower[0])
        wy = float(iv.upper[1] - iv.lower[1])
        print(f"  {s.name}:")
        print(f"    px ∈ [{float(iv.lower[0]):+.4f}, {float(iv.upper[0]):+.4f}] m  (width {wx:.4f})")
        print(f"    py ∈ [{float(iv.lower[1]):+.4f}, {float(iv.upper[1]):+.4f}] m  (width {wy:.4f})")

    print("\n✓ Done")
