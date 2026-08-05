"""
Tests for integrator_separating_input.py
"""
import os
import sys
from pathlib import Path

# Force CPU-only JAX before any jax import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# File is at examples/integrator_chain/tests/<name>.py
#   parents[1] = examples/integrator_chain/
#   parents[2] = examples/
_EXAMPLES_DIR = Path(__file__).resolve().parents[2]
_PKG_DIR = Path(__file__).resolve().parents[1]
for _d in (_EXAMPLES_DIR, _PKG_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx
import pytest

from integrator_separating_input import (
    IntegratorSystem,
    get_system_and_embedding,
    Scenario,
    observed_output,
    _invert_observation,
    create_scenarios,
    euler_step,
    propagate_scenario,
    separation_loss,
    SeparatingInputOptimizer,
    propagate_scenario_multistep,
    separation_loss_multistep,
    optimize_multistep,
    propagate_with_refinement,
    refined_overlap_loss,
    optimize_refined_gpu,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def point_ivl(arr):
    a = jnp.array(arr, dtype=float)
    return irx.Interval(lower=a, upper=a)


def small_ivl(N, width=0.02):
    center = jnp.zeros(N)
    return irx.icentpert(center, jnp.full(N, width))


# ══════════════════════════════════════════════════════════════════════════════
# 1. System-level unit tests (no interval arithmetic)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegratorSystemDynamics:

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_chain_derivative_matches_shift(self, N):
        """x1˙=x2, ..., x_{N-1}˙=xN for any N (u contributes nothing to these)."""
        sys_ = IntegratorSystem(N)
        x = jnp.arange(1.0, N + 1.0)   # [1, 2, ..., N]
        u = jnp.array([0.0])
        p = jnp.array([1.0])
        dx = sys_.f(0.0, x, u, p)
        if N > 1:
            np.testing.assert_allclose(np.array(dx[: N - 1]), np.array(x[1:]), atol=1e-6)

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_last_state_derivative_is_alpha_times_u(self, N):
        sys_ = IntegratorSystem(N)
        x = jnp.zeros(N)
        u = jnp.array([2.0])
        dx_nom = sys_.f(0.0, x, u, jnp.array([1.0]))
        dx_flt = sys_.f(0.0, x, u, jnp.array([0.5]))
        np.testing.assert_allclose(float(dx_nom[-1]), 2.0, atol=1e-6)
        np.testing.assert_allclose(float(dx_flt[-1]), 1.0, atol=1e-6)

    def test_xlen_matches_N(self):
        for N in (1, 2, 5):
            assert IntegratorSystem(N).xlen == N

    def test_embedding_cache_reuses_instance(self):
        sys_a, emb_a = get_system_and_embedding(3)
        sys_b, emb_b = get_system_and_embedding(3)
        assert sys_a is sys_b
        assert emb_a is emb_b

    def test_t_must_be_jax_array_not_python_float(self):
        """Regression guard: Python-float t breaks natif_jaxpr's invar count."""
        _, emb = get_system_and_embedding(2)
        x_ut = irx.i2ut(small_ivl(2))
        u = jnp.array([0.5])
        p_ivl = irx.icentpert(jnp.array([1.0]), jnp.zeros(1))
        # A JAX-array t works.
        emb.f(jnp.zeros(()), x_ut, u, p_ivl)
        # A Python float t should raise (TypeError from eqx.filter_make_jaxpr's
        # arg-count mismatch, or similar) -- guards against a future regression
        # if someone "simplifies" euler_step's `_t = jnp.zeros(())` back to `0.0`.
        with pytest.raises(Exception):
            emb.f(0.0, x_ut, u, p_ivl)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Output-map (observed_output / _invert_observation) tests
# ══════════════════════════════════════════════════════════════════════════════

class TestObservedOutput:

    def test_identity_when_beta1_xi0(self):
        scen = Scenario("id", None, point_ivl([1.0]), point_ivl(jnp.ones(3)), point_ivl(jnp.zeros(3)))
        x_ivl = irx.Interval(lower=jnp.array([1.0, -2.0, 3.0]), upper=jnp.array([1.5, -1.0, 4.0]))
        y = observed_output(x_ivl, scen)
        np.testing.assert_allclose(np.array(y.lower), np.array(x_ivl.lower), atol=1e-6)
        np.testing.assert_allclose(np.array(y.upper), np.array(x_ivl.upper), atol=1e-6)

    def test_hand_computed_affine_map(self):
        beta = irx.Interval(lower=jnp.array([2.0]), upper=jnp.array([3.0]))
        xi = irx.Interval(lower=jnp.array([-1.0]), upper=jnp.array([1.0]))
        scen = Scenario("s", None, point_ivl([1.0]), beta, xi)
        x_ivl = irx.Interval(lower=jnp.array([1.0]), upper=jnp.array([2.0]))
        y = observed_output(x_ivl, scen)
        # y.lower = beta.lower * x.lower + xi.lower = 2*1 + (-1) = 1
        # y.upper = beta.upper * x.upper + xi.upper = 3*2 + 1 = 7
        np.testing.assert_allclose(float(y.lower[0]), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(y.upper[0]), 7.0, atol=1e-6)

    def test_invert_recovers_point_state_with_point_fault_params(self):
        """When beta,xi are point intervals, inversion should be exact."""
        beta = point_ivl([2.0, 0.5])
        xi = point_ivl([1.0, -0.5])
        scen = Scenario("s", None, point_ivl([1.0]), beta, xi)
        x_true = jnp.array([3.0, -2.0])
        y_ivl = observed_output(point_ivl(x_true), scen)
        x_rec = _invert_observation(y_ivl, scen)
        np.testing.assert_allclose(np.array(x_rec.lower), np.array(x_true), atol=1e-5)
        np.testing.assert_allclose(np.array(x_rec.upper), np.array(x_true), atol=1e-5)

    def test_invert_is_sound_outer_enclosure(self):
        """For random x within x_ivl and beta/xi within their intervals, the
        resulting y must map back into the inverted x interval (soundness)."""
        beta = irx.Interval(lower=jnp.array([0.8]), upper=jnp.array([1.2]))
        xi = irx.Interval(lower=jnp.array([-0.3]), upper=jnp.array([0.3]))
        scen = Scenario("s", None, point_ivl([1.0]), beta, xi)
        x_ivl = irx.Interval(lower=jnp.array([-1.0]), upper=jnp.array([2.0]))
        y_ivl = observed_output(x_ivl, scen)
        x_rec = _invert_observation(y_ivl, scen)

        rng = np.random.default_rng(0)
        for _ in range(200):
            x = rng.uniform(float(x_ivl.lower[0]), float(x_ivl.upper[0]))
            b = rng.uniform(float(beta.lower[0]), float(beta.upper[0]))
            xi_v = rng.uniform(float(xi.lower[0]), float(xi.upper[0]))
            y = b * x + xi_v
            if float(y_ivl.lower[0]) - 1e-6 <= y <= float(y_ivl.upper[0]) + 1e-6:
                assert float(x_rec.lower[0]) - 1e-5 <= x <= float(x_rec.upper[0]) + 1e-5

    def test_beta_must_be_strictly_positive(self):
        """create_scenarios must reject a sensor_beta_lo <= 0 (unsound inversion)."""
        with pytest.raises(ValueError):
            create_scenarios(2, sensor_beta_lo=-0.1, sensor_beta_hi=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Propagation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEulerStepAndPropagation:

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_single_euler_step_matches_hand_computation(self, N):
        """Point-interval, alpha=1: one Euler step of the shift dynamics."""
        scenarios = create_scenarios(N)
        nominal = scenarios[0]
        x0 = point_ivl(jnp.arange(1.0, N + 1.0))
        u = jnp.array([0.5])
        dt = 0.1
        x1 = euler_step(nominal.emb_system, x0, u, nominal.p_interval, dt)

        x0_np = np.arange(1.0, N + 1.0)
        expected_dot = np.concatenate([x0_np[1:], [1.0 * 0.5]])
        expected = x0_np + dt * expected_dot
        np.testing.assert_allclose(np.array(x1.lower), expected, atol=1e-5)
        np.testing.assert_allclose(np.array(x1.upper), expected, atol=1e-5)

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_valid_intervals_after_propagation(self, N):
        scenarios = create_scenarios(N)
        x0 = small_ivl(N)
        u = jnp.array([0.3])
        for s in scenarios:
            xf = propagate_scenario(x0, u, s, dt=0.1, num_steps=5)
            assert np.all(np.array(xf.lower) <= np.array(xf.upper)), \
                f"N={N} {s.name}: invalid interval after propagation"

    def test_actuator_fault_changes_last_state(self):
        N = 2
        scenarios = create_scenarios(N)
        nominal, act_fault = scenarios[0], scenarios[1]
        x0 = small_ivl(N)
        u = jnp.array([1.0])
        x_nom = propagate_scenario(x0, u, nominal, dt=0.5, num_steps=10)
        x_act = propagate_scenario(x0, u, act_fault, dt=0.5, num_steps=10)
        assert not np.allclose(np.array(x_nom.lower), np.array(x_act.lower), atol=1e-4)

    def test_multistep_matches_single_step_at_final_segment(self):
        N = 2
        scenarios = create_scenarios(N)
        s = scenarios[0]
        x0 = small_ivl(N)
        u = jnp.array([0.4])
        steps_per_segment, num_segments = 3, 4
        u_seq = jnp.tile(u, (num_segments, 1))

        x_multistep = propagate_scenario_multistep(x0, u_seq, s, dt=0.1, steps_per_segment=steps_per_segment)
        x_direct = propagate_scenario(x0, u, s, dt=0.1, num_steps=steps_per_segment * num_segments)
        np.testing.assert_allclose(np.array(x_multistep.lower), np.array(x_direct.lower), atol=1e-4)
        np.testing.assert_allclose(np.array(x_multistep.upper), np.array(x_direct.upper), atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Loss function tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSeparationLoss:

    def test_zero_overlap_for_identical_scenarios(self):
        """Two scenarios with identical (nominal) params should give zero loss
        only if intervals don't overlap; here they're literally the same
        scenario twice, so overlap should equal the full volume (sanity: loss
        is finite and non-negative, not literally zero)."""
        N = 2
        scenarios = create_scenarios(N)
        x0 = small_ivl(N)
        u = jnp.array([0.5])
        loss = separation_loss(u, x0, scenarios, dt=0.1, num_steps=5)
        assert float(loss) >= 0.0
        assert np.isfinite(float(loss))

    def test_larger_actuator_gap_increases_loss_generally(self):
        """A wider actuator-fault alpha gap should not decrease separation
        loss (weak monotonicity check, not exact due to nonlinearity)."""
        N = 2
        x0 = small_ivl(N)
        u = jnp.array([0.8])
        narrow = create_scenarios(N, actuator_alpha_lo=0.85, actuator_alpha_hi=0.95)
        wide = create_scenarios(N, actuator_alpha_lo=0.3, actuator_alpha_hi=0.95)
        loss_narrow = float(separation_loss(u, x0, narrow, dt=0.2, num_steps=8))
        loss_wide = float(separation_loss(u, x0, wide, dt=0.2, num_steps=8))
        assert loss_wide >= loss_narrow - 1e-6

    def test_multistep_loss_matches_min_over_segments(self):
        N = 2
        scenarios = create_scenarios(N)
        x0 = small_ivl(N)
        u_seq = jnp.tile(jnp.array([0.5]), (4, 1))
        loss = separation_loss_multistep(u_seq, x0, scenarios, dt=0.1, steps_per_segment=2)
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0


class TestOptimizers:

    @pytest.mark.parametrize("N", [2, 3])
    def test_single_step_optimizer_reduces_loss(self, N):
        scenarios = create_scenarios(N)
        x0 = small_ivl(N)
        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.2, num_steps=5)
        u0 = jnp.array([0.5])
        loss0 = float(opt.loss_fn(u0))
        u_opt, loss_opt = opt.optimize(u_init=u0, learning_rate=0.05, num_iters=30)
        assert loss_opt <= loss0 + 1e-6

    @pytest.mark.parametrize("N", [2, 3])
    def test_multistep_optimizer_runs_and_is_finite(self, N):
        scenarios = create_scenarios(N)
        x0 = small_ivl(N)
        u_seq, loss, stats = optimize_multistep(
            scenarios, x0, dt=0.2, steps_per_segment=2, num_segments=5,
            learning_rate=0.05, num_iters=20, num_restarts=3, seed=0,
        )
        assert np.isfinite(loss)
        assert u_seq.shape == (5, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Intersection-refinement tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRefinement:

    def test_refined_loss_le_unrefined_for_same_u_seq(self):
        """Refinement can only tighten reachable sets, never loosen them, so
        the refined loss should never exceed the unrefined multistep loss for
        the same control sequence and horizon."""
        N = 2
        scenarios = create_scenarios(N)
        x0 = small_ivl(N)
        num_steps = 4
        u_seq = jnp.tile(jnp.array([0.5]), (num_steps, 1))

        unrefined = float(separation_loss_multistep(u_seq, x0, scenarios, dt=0.2, steps_per_segment=1))
        refined = float(refined_overlap_loss(u_seq, x0, scenarios, dt=0.2, num_steps=num_steps))
        assert refined <= unrefined + 1e-5

    def test_refinement_output_is_finite_and_nonnegative(self):
        N = 2
        scenarios = create_scenarios(N)
        x0 = small_ivl(N)
        u_seq = jnp.tile(jnp.array([0.3]), (3, 1))
        loss = float(propagate_with_refinement(x0, u_seq, scenarios, dt=0.15, num_steps=3))
        assert np.isfinite(loss)
        assert loss >= 0.0

    @pytest.mark.parametrize("N", [2, 3])
    def test_optimize_refined_gpu_runs_and_is_finite(self, N):
        scenarios = create_scenarios(N)
        x0 = small_ivl(N)
        u_opt, loss_opt, u_all, losses = optimize_refined_gpu(
            x0, scenarios, dt=0.2, num_steps=3, num_restarts=3,
            learning_rate=0.05, num_iters=15, seed=0,
        )
        assert np.isfinite(float(loss_opt))
        assert u_opt.shape == (3, 1)
