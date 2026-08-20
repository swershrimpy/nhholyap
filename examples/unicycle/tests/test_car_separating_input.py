"""
Tests for car_separating_input.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

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

from car_separating_input import (
    CarNomActSystem,
    get_system_and_embedding,
    Scenario,
    create_scenarios,
    observed_output,
    _invert_observation,
    euler_step,
    propagate_scenario,
    separation_loss,
    SeparatingInputOptimizer,
    optimize_parallel_gpu,
    propagate_scenario_multistep,
    separation_loss_multistep,
    optimize_multistep,
    propagate_with_refinement,
    refined_overlap_loss,
    optimize_refined_gpu,
    collect_refinement_history,
    _refine_and_step_pair,
    _overlap_volume,
)


def point_ivl(arr):
    a = jnp.array(arr, dtype=float)
    return irx.Interval(lower=a, upper=a)


def small_ivl(center=(0.1, 0.1, 0.0), width=0.05):
    c = jnp.array(center)
    return irx.icentpert(c, jnp.full(3, width))


# ══════════════════════════════════════════════════════════════════════════════
# 1. System-level unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCarSystemDynamics:

    def test_xlen_is_3(self):
        assert CarNomActSystem().xlen == 3

    def test_zero_control_is_fixed_point(self):
        sys_ = CarNomActSystem()
        dx = sys_.f(jnp.zeros(()), jnp.array([1.0, -2.0, 0.5]), jnp.zeros(2), jnp.array([1.0]))
        np.testing.assert_allclose(np.array(dx), np.zeros(3), atol=1e-10)

    def test_derivative_matches_hand_computation(self):
        sys_ = CarNomActSystem()
        x = jnp.array([0.0, 0.0, jnp.pi / 4])
        u = jnp.array([2.0, 0.5])
        dx = sys_.f(jnp.zeros(()), x, u, jnp.array([0.5]))
        expected = np.array([2.0 * np.cos(np.pi / 4), 2.0 * np.sin(np.pi / 4), 0.5 * 0.5])
        np.testing.assert_allclose(np.array(dx), expected, atol=1e-6)

    def test_embedding_cache_reuses_instance(self):
        sys_a, emb_a = get_system_and_embedding()
        sys_b, emb_b = get_system_and_embedding()
        assert sys_a is sys_b
        assert emb_a is emb_b

    def test_t_must_be_jax_array_not_python_float(self):
        """Regression guard: Python-float t breaks natif_jaxpr's invar count."""
        _, emb = get_system_and_embedding()
        x_ut = irx.i2ut(small_ivl())
        u = jnp.array([0.5, 0.3])
        p_ivl = irx.icentpert(jnp.array([1.0]), jnp.zeros(1))
        emb.f(jnp.zeros(()), x_ut, u, p_ivl)
        with pytest.raises(Exception):
            emb.f(0.0, x_ut, u, p_ivl)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scenario / observation-model tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScenariosAndObservation:

    def test_three_scenarios_created(self):
        scenarios = create_scenarios()
        assert [s.name for s in scenarios] == ["Nominal", "Actuator Fault", "Sensor Fault"]

    def test_nominal_and_actuator_fault_have_no_output_offset(self):
        scenarios = create_scenarios()
        for s in scenarios[:2]:
            np.testing.assert_allclose(np.array(s.obs_offset), np.zeros(2))
            np.testing.assert_allclose(np.array(s.obs_scale), np.ones(1))

    def test_sensor_fault_alpha_matches_nominal(self):
        """Sensor Fault's dynamics (alpha) are identical to Nominal's -- it is
        distinguished purely by the observation model, not the dynamics."""
        scenarios = create_scenarios()
        nominal, sensor = scenarios[0], scenarios[2]
        np.testing.assert_allclose(np.array(nominal.p_interval.lower), np.array(sensor.p_interval.lower))
        np.testing.assert_allclose(np.array(nominal.p_interval.upper), np.array(sensor.p_interval.upper))

    def test_actuator_fault_alpha_isolated(self):
        scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
        act = scenarios[1]
        assert float(act.p_interval.lower[0]) == 0.0
        assert float(act.p_interval.upper[0]) == 0.5

    def test_observed_output_identity_when_offset0_scale1(self):
        scen = Scenario("id", None, point_ivl([1.0]))
        x_ivl = irx.Interval(lower=jnp.array([1.0, -2.0, 0.0]), upper=jnp.array([1.5, -1.0, 0.1]))
        y = observed_output(x_ivl, scen)
        np.testing.assert_allclose(np.array(y.lower), np.array(x_ivl.lower[:2]))
        np.testing.assert_allclose(np.array(y.upper), np.array(x_ivl.upper[:2]))

    def test_observed_output_applies_offset_and_scale(self):
        scen = Scenario("s", None, point_ivl([1.0]), obs_offset=jnp.array([0.2, 0.2]), obs_scale=jnp.array([0.95]))
        x_ivl = point_ivl([1.0, 2.0, 0.0])
        y = observed_output(x_ivl, scen)
        np.testing.assert_allclose(float(y.lower[0]), 0.95 * 1.0 + 0.2, atol=1e-6)
        np.testing.assert_allclose(float(y.lower[1]), 0.95 * 2.0 + 0.2, atol=1e-6)

    def test_invert_observation_round_trip(self):
        scen = Scenario("s", None, point_ivl([1.0]), obs_offset=jnp.array([0.2, -0.1]), obs_scale=jnp.array([0.8]))
        x_true = irx.Interval(lower=jnp.array([1.0, -2.0, 0.3]), upper=jnp.array([1.4, -1.6, 0.5]))
        y = observed_output(x_true, scen)
        x_rec = _invert_observation(y, scen, x_true)
        np.testing.assert_allclose(np.array(x_rec.lower[:2]), np.array(x_true.lower[:2]), atol=1e-5)
        np.testing.assert_allclose(np.array(x_rec.upper[:2]), np.array(x_true.upper[:2]), atol=1e-5)
        # phi passed through unchanged from phi_source
        np.testing.assert_allclose(np.array(x_rec.lower[2]), np.array(x_true.lower[2]), atol=1e-6)
        np.testing.assert_allclose(np.array(x_rec.upper[2]), np.array(x_true.upper[2]), atol=1e-6)

    def test_bug1_regression_sensor_fault_observed_output_differs_from_nominal(self):
        """Regression guard for the fixed bug: even though Sensor Fault's
        STATE is identical to Nominal's (same alpha, same dynamics), its
        OBSERVED output must differ (via obs_offset/obs_scale) -- and that
        difference must be nonzero even at u=0 (no motion), proving the
        overlap is not an irreducible constant the way it was in the
        original faulty_car_separating_input.py (see PLAN.md bug #1)."""
        scenarios = create_scenarios()
        nominal, sensor = scenarios[0], scenarios[2]
        x0 = small_ivl()
        u_zero = jnp.zeros(2)
        x_nom = propagate_scenario(x0, u_zero, nominal, dt=0.1, num_steps=3)
        x_sensor = propagate_scenario(x0, u_zero, sensor, dt=0.1, num_steps=3)
        # states ARE identical (alpha, dynamics identical, u=0 -> no motion)
        np.testing.assert_allclose(np.array(x_nom.lower), np.array(x_sensor.lower), atol=1e-6)
        # but observed outputs must NOT be identical
        y_nom = observed_output(x_nom, nominal)
        y_sensor = observed_output(x_sensor, sensor)
        assert not np.allclose(np.array(y_nom.lower), np.array(y_sensor.lower), atol=1e-6)
        # and therefore they must not overlap at all
        overlap = float(_overlap_volume(y_nom, y_sensor))
        assert overlap == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Propagation / loss tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPropagationAndLoss:

    def test_single_euler_step_matches_hand_computation(self):
        scenarios = create_scenarios()
        nominal = scenarios[0]
        x0 = point_ivl([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.5])
        dt = 0.1
        x1 = euler_step(nominal.emb_system, x0, u, nominal.p_interval, dt)
        expected = np.array([1.0 * np.cos(0.0), 1.0 * np.sin(0.0), 1.0 * 0.5]) * dt
        np.testing.assert_allclose(np.array(x1.lower), expected, atol=1e-5)

    def test_valid_intervals_after_propagation(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        u = jnp.array([0.5, 0.3])
        for s in scenarios:
            xf = propagate_scenario(x0, u, s, dt=0.1, num_steps=5)
            assert np.all(np.array(xf.lower) <= np.array(xf.upper))

    def test_separation_loss_finite_and_nonnegative(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        loss = separation_loss(jnp.array([0.5, 0.3]), x0, scenarios, dt=0.1, num_steps=5)
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0

    def test_multistep_matches_single_step_at_final_segment(self):
        scenarios = create_scenarios()
        s = scenarios[0]
        x0 = small_ivl()
        u = jnp.array([0.4, 0.2])
        steps_per_segment, num_segments = 3, 4
        u_seq = jnp.tile(u, (num_segments, 1))
        x_multi = propagate_scenario_multistep(x0, u_seq, s, dt=0.1, steps_per_segment=steps_per_segment)
        x_direct = propagate_scenario(x0, u, s, dt=0.1, num_steps=steps_per_segment * num_segments)
        np.testing.assert_allclose(np.array(x_multi.lower), np.array(x_direct.lower), atol=1e-4)

    def test_multistep_loss_finite(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_seq = jnp.tile(jnp.array([0.5, 0.3]), (4, 1))
        loss = separation_loss_multistep(u_seq, x0, scenarios, dt=0.1, steps_per_segment=2)
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Optimizer tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizers:

    def test_single_step_optimizer_reduces_loss(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.1, num_steps=5)
        u0 = jnp.array([0.5, 0.3])
        loss0 = float(opt.loss_fn(u0))
        u_opt, loss_opt = opt.optimize(u_init=u0, learning_rate=0.05, num_iters=30)
        assert loss_opt <= loss0 + 1e-6

    def test_optimize_parallel_gpu_runs_and_is_finite(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.1, num_steps=5)
        u_opt, loss_opt, u_all, losses = optimize_parallel_gpu(
            opt, num_restarts=5, learning_rate=0.05, num_iters=20, seed=0
        )
        assert np.isfinite(float(loss_opt))
        assert u_opt.shape == (2,)

    def test_multistep_optimizer_runs_and_is_finite(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_seq, loss, stats = optimize_multistep(
            scenarios, x0, dt=0.1, steps_per_segment=2, num_segments=5,
            learning_rate=0.05, num_iters=20, num_restarts=3, seed=0,
        )
        assert np.isfinite(loss)
        assert u_seq.shape == (5, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Intersection-refinement tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRefinement:

    def test_refine_and_step_pair_no_overlap_falls_back_to_unchanged_state(self):
        """Regression guard for the fixed bug (#4): on no-overlap, the pair's
        state must fall back to the UNCHANGED input interval, not some
        fallback derived from one scenario's center alone."""
        scenarios = create_scenarios()
        nominal, act = scenarios[0], scenarios[1]
        # Two disjoint, far-apart state intervals -> guaranteed no observed overlap
        x_i = irx.Interval(lower=jnp.array([0.0, 0.0, 0.0]), upper=jnp.array([0.01, 0.01, 0.01]))
        x_j = irx.Interval(lower=jnp.array([50.0, 50.0, 0.0]), upper=jnp.array([50.01, 50.01, 0.01]))
        u_k = jnp.array([0.0, 0.0])  # zero control: euler_step is identity on state
        x_next_i, x_next_j, _, _, pair_cost, has_overlap = _refine_and_step_pair(
            x_i, x_j, nominal, act, u_k, dt=0.1
        )
        assert bool(has_overlap) is False
        assert float(pair_cost) == 0.0
        # with u=0, euler_step leaves state unchanged -> x_next should equal x_i/x_j exactly
        np.testing.assert_allclose(np.array(x_next_i.lower), np.array(x_i.lower), atol=1e-6)
        np.testing.assert_allclose(np.array(x_next_j.lower), np.array(x_j.lower), atol=1e-6)

    def test_refined_loss_le_unrefined_for_same_u_seq(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        num_steps = 5
        u_seq = jnp.tile(jnp.array([0.4, 0.2]), (num_steps, 1))
        unrefined = float(separation_loss_multistep(u_seq, x0, scenarios, dt=0.1, steps_per_segment=1))
        refined = float(refined_overlap_loss(u_seq, x0, scenarios, dt=0.1, num_steps=num_steps))
        assert refined <= unrefined + 1e-5

    def test_refinement_output_finite_and_nonnegative(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_seq = jnp.tile(jnp.array([0.3, 0.1]), (4, 1))
        loss = float(propagate_with_refinement(x0, u_seq, scenarios, dt=0.1, num_steps=4))
        assert np.isfinite(loss)
        assert loss >= 0.0

    def test_optimize_refined_gpu_runs_and_is_finite(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_opt, loss_opt, u_all, losses = optimize_refined_gpu(
            x0, scenarios, dt=0.1, num_steps=4, num_restarts=5,
            learning_rate=0.05, num_iters=20, seed=0,
        )
        assert np.isfinite(float(loss_opt))
        assert u_opt.shape == (4, 2)

    def test_collect_refinement_history_matches_propagate_with_refinement(self):
        """The history collector uses the exact same per-pair helper as the
        jittable loss (fix #3) -- their pairwise overlap at the final step
        must match the loss's own per-step overlaps (up to the min-over-steps
        the jittable version takes)."""
        scenarios = create_scenarios()
        x0 = small_ivl()
        num_steps = 4
        u_seq = jnp.tile(jnp.array([0.3, 0.15]), (num_steps, 1))

        steps, pairs = collect_refinement_history(x0, u_seq, scenarios, dt=0.1, num_steps=num_steps)
        assert len(steps) == num_steps
        assert pairs == [(0, 1), (0, 2), (1, 2)]

        # Reconstruct the same per-step pairwise-overlap-sum sequence the
        # jittable loss takes a min over, from the history's own intervals,
        # and confirm the loss's returned value equals that minimum.
        step_sums = []
        for step in steps:
            total = 0.0
            for obs_i, obs_j, _ in step['pair_obs']:
                total += float(_overlap_volume(obs_i, obs_j))
            step_sums.append(total)
        expected_min = min(step_sums)
        actual = float(refined_overlap_loss(u_seq, x0, scenarios, dt=0.1, num_steps=num_steps))
        np.testing.assert_allclose(actual, expected_min, atol=1e-5)
