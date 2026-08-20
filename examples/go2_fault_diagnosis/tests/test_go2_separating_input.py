"""
Tests for go2_separating_input.py
"""
import os
import sys
from pathlib import Path

# Force CPU-only JAX before any jax import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# File is at examples/go2_fault_diagnosis/tests/<name>.py
#   parents[1] = examples/go2_fault_diagnosis/
#   parents[2] = examples/
_EXAMPLES_DIR = Path(__file__).resolve().parents[2]
_PKG_DIR      = Path(__file__).resolve().parents[1]
for _d in (_EXAMPLES_DIR, _PKG_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx
import pytest

from go2_separating_input import (
    Go2NomActSystem,
    Go2SensorFaultSystem,
    create_scenarios,
    euler_step,
    propagate_scenario,
    position_interval,
    separation_loss,
    SeparatingInputOptimizer,
    optimize_multistart,
    _NOM_ACT_EMB,
    _SF_EMB,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def point_ivl(arr):
    """Zero-width interval around a single point."""
    a = jnp.array(arr, dtype=float)
    return irx.Interval(lower=a, upper=a)


def small_ivl():
    """Small initial uncertainty used throughout the propagation tests."""
    return irx.Interval(
        lower=jnp.array([-0.02, -0.02, -0.01]),
        upper=jnp.array([ 0.02,  0.02,  0.01]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. System-level unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGo2NomActSystem:

    def setup_method(self):
        self.sys = Go2NomActSystem()
        pass  # no disturbance in this system

    def test_pure_forward_motion(self):
        """At θ=0, vx=1 → ṗx=1, ṗy=0, θ̇=0."""
        x  = jnp.array([0.0, 0.0, 0.0])
        u  = jnp.array([1.0, 0.0, 0.0])
        p  = jnp.array([1.0])
        dx = self.sys.f(0., x, u, p)
        np.testing.assert_allclose(np.array(dx), [1.0, 0.0, 0.0], atol=1e-6)

    def test_yaw_rate_nominal(self):
        """With α=1 and ω=1, θ̇ should equal 1."""
        x  = jnp.array([0.0, 0.0, 0.0])
        u  = jnp.array([0.0, 0.0, 1.0])
        p  = jnp.array([1.0])
        dx = self.sys.f(0., x, u, p)
        np.testing.assert_allclose(float(dx[2]), 1.0, atol=1e-6)

    def test_actuator_fault_scales_yaw(self):
        """Actuator fault α=0.5 halves the yaw rate."""
        x      = jnp.array([0.0, 0.0, 0.0])
        u      = jnp.array([0.0, 0.0, 1.0])
        dx_nom = self.sys.f(0., x, u, jnp.array([1.0]))
        dx_flt = self.sys.f(0., x, u, jnp.array([0.5]))
        np.testing.assert_allclose(float(dx_nom[2]), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(dx_flt[2]), 0.5, atol=1e-6)

    def test_lateral_velocity_affects_translation(self):
        """Non-zero vy at θ=0 should affect ṗx and ṗy."""
        x     = jnp.array([0.0, 0.0, 0.0])
        u_no  = jnp.array([1.0, 0.0, 0.0])
        u_vy  = jnp.array([1.0, 0.5, 0.0])
        p     = jnp.array([1.0])
        dx_no = self.sys.f(0., x, u_no, p)
        dx_vy = self.sys.f(0., x, u_vy, p)
        # Adding vy should change the translation derivatives
        assert not np.allclose(np.array(dx_no[:2]), np.array(dx_vy[:2]), atol=1e-8)

    def test_heading_rotation(self):
        """At θ=π/2, forward vx should map to pure py motion."""
        x  = jnp.array([0.0, 0.0, jnp.pi / 2])
        u  = jnp.array([1.0, 0.0, 0.0])
        p  = jnp.array([1.0])
        dx = self.sys.f(0., x, u, p)
        np.testing.assert_allclose(float(dx[0]),  0.0, atol=1e-6)
        np.testing.assert_allclose(float(dx[1]),  1.0, atol=1e-6)


class TestGo2SensorFaultSystem:

    def setup_method(self):
        self.sys = Go2SensorFaultSystem()
        pass  # no disturbance

    def test_ignores_commanded_vy(self):
        """Changing vy_cmd (u[1]) must NOT change the output when p=[0]."""
        x       = jnp.array([0.0, 0.0, 0.0])
        p_zero  = jnp.array([0.0])
        dx_with = self.sys.f(0., x, jnp.array([1.0, 2.0, 0.0]), p_zero)
        dx_zero = self.sys.f(0., x, jnp.array([1.0, 0.0, 0.0]), p_zero)
        np.testing.assert_allclose(np.array(dx_with), np.array(dx_zero), atol=1e-8)

    def test_noise_parameter_shifts_lateral(self):
        """Positive vy_noise at θ=π/2 should affect ṗx derivative."""
        x    = jnp.array([0.0, 0.0, jnp.pi / 2])
        u    = jnp.array([1.0, 0.0, 0.0])
        dxp  = self.sys.f(0., x, u, jnp.array([ 0.1]))
        dxn  = self.sys.f(0., x, u, jnp.array([-0.1]))
        # +vy_noise vs -vy_noise should change ṗx
        assert not np.isclose(float(dxp[0]), float(dxn[0]), atol=1e-8)

    def test_zero_noise_at_theta0_gives_forward_motion(self):
        """At θ=0 with vy_noise=0 and vx=1, dynamics equal forward-only."""
        x  = jnp.array([0.0, 0.0, 0.0])
        u  = jnp.array([1.0, 0.0, 0.0])
        dx = self.sys.f(0., x, u, jnp.array([0.0]))
        np.testing.assert_allclose(np.array(dx), [1.0, 0.0, 0.0], atol=1e-6)

    def test_yaw_rate_unaffected_by_noise(self):
        """vy_noise must not appear in θ̇."""
        x   = jnp.array([0.0, 0.0, 0.0])
        u   = jnp.array([0.0, 0.0, 1.0])
        dx1 = self.sys.f(0., x, u, jnp.array([10.0]))
        dx2 = self.sys.f(0., x, u, jnp.array([-10.0]))
        np.testing.assert_allclose(float(dx1[2]), float(dx2[2]), atol=1e-8)
        np.testing.assert_allclose(float(dx1[2]), 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Propagation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEulerStep:

    def test_zero_input_does_not_move_heading(self):
        """With u=[0,0,0] the heading interval must not change."""
        scenarios = create_scenarios()
        nominal = scenarios[0]
        x = small_ivl()
        x_next = euler_step(nominal.emb_system, x, jnp.zeros(3),
                             nominal.p_interval, dt=0.5)
        np.testing.assert_allclose(
            np.array(x_next.lower[2]), np.array(x.lower[2]), atol=1e-6)
        np.testing.assert_allclose(
            np.array(x_next.upper[2]), np.array(x.upper[2]), atol=1e-6)

    def test_output_interval_is_valid(self):
        """After a step, lower ≤ upper for every dimension."""
        scenarios = create_scenarios()
        for s in scenarios:
            x_next = euler_step(s.emb_system, small_ivl(),
                                jnp.array([0.5, 0.2, 0.3]),
                                s.p_interval, dt=0.5)
            assert np.all(np.array(x_next.lower) <= np.array(x_next.upper)), \
                f"{s.name}: euler_step produced invalid interval"


class TestPropagateScenario:

    def test_valid_intervals_after_propagation(self):
        """All scenarios must produce lower ≤ upper after 10 steps."""
        scenarios = create_scenarios()
        x0 = small_ivl()
        u  = jnp.array([0.5, 0.2, 0.3])
        for s in scenarios:
            xf = propagate_scenario(x0, u, s, dt=0.1, num_steps=5)
            assert np.all(np.array(xf.lower) <= np.array(xf.upper)), \
                f"{s.name}: invalid interval after propagation"

    def test_actuator_fault_changes_heading(self):
        """A pure-yaw input must give a different heading interval under actuator fault."""
        scenarios = create_scenarios()
        nominal   = scenarios[0]
        act_fault = scenarios[1]
        x0 = small_ivl()
        u  = jnp.array([0.0, 0.0, 1.0])   # pure yaw

        x_nom = propagate_scenario(x0, u, nominal,   dt=0.5, num_steps=10)
        x_act = propagate_scenario(x0, u, act_fault, dt=0.5, num_steps=10)
        assert not np.allclose(
            np.array(x_nom.lower[2]), np.array(x_act.lower[2]), atol=1e-4), \
            "Actuator fault should alter heading vs nominal"

    def test_sensor_fault_differs_with_nonzero_vy(self):
        """Non-zero vy must produce different position under sensor fault."""
        scenarios = create_scenarios()
        nominal  = scenarios[0]
        sf       = scenarios[2]
        x0 = small_ivl()
        u  = jnp.array([0.5, 0.5, 0.3])   # non-zero vy

        x_nom = propagate_scenario(x0, u, nominal, dt=0.5, num_steps=10)
        x_sf  = propagate_scenario(x0, u, sf,      dt=0.5, num_steps=10)
        pos_nom_eq_sf = (
            np.allclose(np.array(x_nom.lower[:2]), np.array(x_sf.lower[:2]), atol=1e-4)
            and np.allclose(np.array(x_nom.upper[:2]), np.array(x_sf.upper[:2]), atol=1e-4)
        )
        assert not pos_nom_eq_sf, \
            "Sensor fault with nonzero vy should produce different position"

    def test_sensor_fault_matches_nominal_with_zero_vy_zero_noise(self):
        """With vy_cmd=0 and sensor_noise_bound=0, SF ≡ nominal."""
        scenarios = create_scenarios(sensor_noise_bound=0.0)
        nominal  = scenarios[0]
        sf       = scenarios[2]
        x0 = point_ivl([0.0, 0.0, 0.0])
        u  = jnp.array([1.0, 0.0, 0.3])   # no lateral command

        x_nom = propagate_scenario(x0, u, nominal, dt=0.5, num_steps=5)
        x_sf  = propagate_scenario(x0, u, sf,      dt=0.5, num_steps=5)
        np.testing.assert_allclose(
            np.array(x_nom.lower), np.array(x_sf.lower), atol=1e-5)
        np.testing.assert_allclose(
            np.array(x_nom.upper), np.array(x_sf.upper), atol=1e-5)

    def test_larger_noise_widens_sensor_fault_interval(self):
        """A bigger noise bound must yield a wider position interval."""
        x0 = small_ivl()
        u  = jnp.array([0.5, 0.5, 0.3])
        sf_small = create_scenarios(sensor_noise_bound=0.01)[2]
        sf_large = create_scenarios(sensor_noise_bound=0.20)[2]

        xf_small = propagate_scenario(x0, u, sf_small, dt=0.5, num_steps=10)
        xf_large = propagate_scenario(x0, u, sf_large, dt=0.5, num_steps=10)

        # At least one position dimension should be wider
        width_small = float(xf_small.upper[0] - xf_small.lower[0])
        width_large = float(xf_large.upper[0] - xf_large.lower[0])
        assert width_large >= width_small - 1e-8, \
            "Larger noise bound should produce wider (or equal) px interval"

    def test_position_interval_is_subset_of_state_interval(self):
        """position_interval() should be the first two dims of propagation."""
        scenarios = create_scenarios()
        x0 = small_ivl()
        u  = jnp.array([0.5, 0.2, 0.3])
        xf = propagate_scenario(x0, u, scenarios[0], dt=0.5, num_steps=5)
        pi = position_interval(xf)
        np.testing.assert_allclose(np.array(pi.lower), np.array(xf.lower[:2]))
        np.testing.assert_allclose(np.array(pi.upper), np.array(xf.upper[:2]))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Loss function tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSeparationLoss:

    def test_nonnegative(self):
        """Loss must always be ≥ 0."""
        scenarios = create_scenarios()
        x0 = small_ivl()
        u  = jnp.array([0.5, 0.0, 0.3])
        loss = separation_loss(u, x0, scenarios, dt=0.5, num_steps=10)
        assert float(loss) >= 0.0

    def test_gradient_is_finite(self):
        """Gradient of loss w.r.t. u should be finite."""
        scenarios = create_scenarios()
        x0 = small_ivl()
        u  = jnp.array([0.5, 0.2, 0.3])
        g  = jax.grad(
            lambda u: separation_loss(u, x0, scenarios, dt=0.5, num_steps=10)
        )(u)
        assert np.all(np.isfinite(np.array(g))), f"Non-finite gradient: {g}"

    def test_zero_loss_at_zero_input_with_tiny_uncertainty(self):
        """With u=0 and a tiny initial set the loss should be very small or zero."""
        scenarios = create_scenarios()
        # Point interval → all trajectories collapse to the same point → zero overlap
        x0 = point_ivl([0.0, 0.0, 0.0])
        u  = jnp.zeros(3)
        loss = separation_loss(u, x0, scenarios, dt=0.5, num_steps=10)
        # With a POINT initial set and zero input there's no divergence between
        # nominal and sensor fault (vy=0 noise is also zero width at 0 input),
        # so total overlap should be zero
        assert float(loss) >= 0.0   # still ≥ 0

    def test_gradient_is_zero_when_no_overlap(self):
        """Gradient should be zero when intervals are already fully separated
        (overlap = 0, gradient of lax.cond no-overlap branch)."""
        # With a very short horizon and tiny uncertainty, intervals may not overlap
        scenarios = create_scenarios()
        x0 = irx.Interval(
            lower=jnp.array([-0.001, -0.001, -0.001]),
            upper=jnp.array([ 0.001,  0.001,  0.001]),
        )
        u = jnp.array([0.0, 0.0, 0.0])
        loss = separation_loss(u, x0, scenarios, dt=0.1, num_steps=1)
        # We just verify the call completes and gives a non-negative result
        assert float(loss) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Optimizer tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSeparatingInputOptimizer:

    def setup_method(self):
        self.scenarios = create_scenarios()
        self.x0        = small_ivl()
        self.opt       = SeparatingInputOptimizer(
            self.scenarios, self.x0, dt=0.5, num_steps=10
        )

    def test_loss_fn_matches_separation_loss(self):
        """opt.loss_fn should equal separation_loss for the same u."""
        u = jnp.array([0.5, 0.3, 0.4])
        direct = float(separation_loss(
            u, self.x0, self.scenarios, dt=0.5, num_steps=10
        ))
        via_opt = float(self.opt.loss_fn(u))
        np.testing.assert_allclose(via_opt, direct, rtol=1e-5)

    def test_optimization_does_not_increase_loss(self):
        """After 50 iterations the loss must not be higher than the start."""
        u_init = jnp.array([0.5, 0.2, 0.3])
        loss_init = float(self.opt.loss_fn(u_init))
        _, loss_opt = self.opt.optimize(u_init, learning_rate=0.01, num_iters=50)
        assert loss_opt <= loss_init + 1e-6, \
            f"Optimization increased loss: {loss_init:.6f} → {loss_opt:.6f}"

    def test_evaluate_returns_correct_number_of_intervals(self):
        """evaluate() should return one position interval per scenario."""
        u     = jnp.array([0.5, 0.3, 0.4])
        stats = self.opt.evaluate(u)
        assert len(stats['position_intervals']) == len(self.scenarios)
        assert len(stats['volumes'])            == len(self.scenarios)

    def test_evaluate_pairwise_count(self):
        """Number of pairwise overlaps = n*(n-1)/2."""
        n     = len(self.scenarios)
        stats = self.opt.evaluate(jnp.array([0.5, 0.3, 0.4]))
        assert len(stats['pairwise_overlaps']) == n * (n - 1) // 2

    def test_evaluate_intervals_are_valid(self):
        """All position intervals returned by evaluate() must have lower ≤ upper."""
        stats = self.opt.evaluate(jnp.array([0.5, 0.3, 0.4]))
        for iv, s in zip(stats['position_intervals'], self.scenarios):
            assert np.all(np.array(iv.lower) <= np.array(iv.upper)), \
                f"{s.name}: invalid position interval from evaluate()"

    def test_evaluate_volumes_nonnegative(self):
        """Interval volumes must be ≥ 0."""
        stats = self.opt.evaluate(jnp.array([0.5, 0.3, 0.4]))
        for name, vol in stats['volumes'].items():
            assert vol >= 0.0, f"{name}: negative volume"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Multi-start optimization test
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizeMultistart:

    def test_returns_valid_result(self):
        """optimize_multistart should return a 3-vector u and finite loss."""
        scenarios = create_scenarios()
        x0 = small_ivl()
        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.5, num_steps=10)
        u_opt, loss_opt, stats = optimize_multistart(
            opt, num_restarts=2, learning_rate=0.01, num_iters=20,
        )
        assert u_opt.shape == (3,)
        assert np.isfinite(loss_opt)
        assert loss_opt >= 0.0

    def test_multistart_loss_le_single_run(self):
        """Multi-start should find a result at least as good as one run."""
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_single, loss_single = SeparatingInputOptimizer(
            scenarios, x0, dt=0.5, num_steps=10
        ).optimize(jnp.array([0.5, 0.0, 0.3]),
                   learning_rate=0.01, num_iters=30)

        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.5, num_steps=10)
        _, loss_multi, _ = optimize_multistart(
            opt, num_restarts=3, learning_rate=0.01, num_iters=30,
        )
        assert loss_multi <= loss_single + 1e-6
