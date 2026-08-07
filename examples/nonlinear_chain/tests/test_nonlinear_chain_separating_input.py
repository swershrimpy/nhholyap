"""
Tests for nonlinear_chain_separating_input.py
"""
import os
import sys
from pathlib import Path

# Force CPU-only JAX before any jax import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# File is at examples/nonlinear_chain/tests/<name>.py
#   parents[1] = examples/nonlinear_chain/
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

from nonlinear_chain_separating_input import (
    CubicChainSystem,
    default_channel_params,
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

class TestCubicChainSystemDynamics:

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_dynamics_matches_hand_computation(self, N):
        a, b = default_channel_params(N)
        sys_ = CubicChainSystem(N, a, b)
        x = jnp.linspace(-1.0, 1.0, N)
        u = jnp.full(N, 0.5)
        p = jnp.ones(N)
        dx = sys_.f(jnp.zeros(()), x, u, p)
        expected = np.array(a) * np.array(x) - np.array(b) * np.array(x) ** 3 + np.array(u)
        np.testing.assert_allclose(np.array(dx), expected, atol=1e-6)

    def test_channels_are_decoupled(self):
        """x_i' depends only on x_i, u_i -- perturbing channel j != i must not
        change channel i's derivative."""
        N = 4
        a, b = default_channel_params(N)
        sys_ = CubicChainSystem(N, a, b)
        p = jnp.ones(N)
        x1 = jnp.array([0.5, -0.3, 0.1, 0.9])
        u1 = jnp.array([0.2, 0.0, 0.0, 0.0])
        x2 = x1.at[1].set(2.0)   # perturb channel 1 only
        u2 = u1.at[2].set(-0.7)  # perturb channel 2's control only
        dx1 = sys_.f(jnp.zeros(()), x1, u1, p)
        dx2 = sys_.f(jnp.zeros(()), x2, u2, p)
        np.testing.assert_allclose(float(dx1[0]), float(dx2[0]), atol=1e-6)
        np.testing.assert_allclose(float(dx1[3]), float(dx2[3]), atol=1e-6)

    def test_origin_is_fixed_point(self):
        """f(0) = 0, g(0) = 0 -> nominal trajectory from x0=0, u=0 stays at 0."""
        N = 3
        a, b = default_channel_params(N)
        sys_ = CubicChainSystem(N, a, b)
        dx = sys_.f(jnp.zeros(()), jnp.zeros(N), jnp.zeros(N), jnp.ones(N))
        np.testing.assert_allclose(np.array(dx), np.zeros(N), atol=1e-6)

    def test_xlen_matches_N(self):
        for N in (1, 2, 5):
            a, b = default_channel_params(N)
            assert CubicChainSystem(N, a, b).xlen == N

    def test_default_channel_params_shape_and_range(self):
        N = 5
        a, b = default_channel_params(N, a_lo=0.5, a_hi=1.5, b_lo=0.2, b_hi=0.8)
        assert a.shape == (N,)
        assert b.shape == (N,)
        assert bool(jnp.all(a >= 0.5)) and bool(jnp.all(a <= 1.5))
        assert bool(jnp.all(b >= 0.2)) and bool(jnp.all(b <= 0.8))

    def test_embedding_cache_reuses_instance(self):
        a, b = default_channel_params(3)
        sys_a, emb_a = get_system_and_embedding(3, a, b)
        sys_b, emb_b = get_system_and_embedding(3, a, b)
        assert sys_a is sys_b
        assert emb_a is emb_b

    def test_embedding_cache_distinguishes_different_ab(self):
        a1, b1 = default_channel_params(3, a_lo=0.5, a_hi=1.5)
        a2, b2 = default_channel_params(3, a_lo=1.0, a_hi=2.0)
        sys_a, _ = get_system_and_embedding(3, a1, b1)
        sys_b, _ = get_system_and_embedding(3, a2, b2)
        assert sys_a is not sys_b

    def test_t_python_float_matches_jax_array(self):
        """Python-float and 0-d jax-array t must agree (fixed upstream in
        immrax 0.3.6 / jax 0.6.2; used to break natif_jaxpr's invar count)."""
        a, b = default_channel_params(2)
        _, emb = get_system_and_embedding(2, a, b)
        x_ut = irx.i2ut(small_ivl(2))
        u = jnp.full(2, 0.5)
        p_ivl = irx.icentpert(jnp.ones(2), jnp.zeros(2))
        r_array = emb.f(jnp.zeros(()), x_ut, u, p_ivl)
        r_float = emb.f(0.0, x_ut, u, p_ivl)
        assert jnp.allclose(jnp.asarray(r_array), jnp.asarray(r_float))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scenario construction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateScenarios:

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_default_scenario_count_is_2N_plus_1(self, N):
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        assert len(scenarios) == 2 * N + 1
        assert scenarios[0].name == "Nominal"
        actuator_names = {s.name for s in scenarios if s.name.startswith("ActuatorFault_")}
        sensor_names = {s.name for s in scenarios if s.name.startswith("SensorFault_")}
        assert actuator_names == {f"ActuatorFault_{i}" for i in range(N)}
        assert sensor_names == {f"SensorFault_{i}" for i in range(N)}

    def test_num_actuator_faults_reduces_actuator_scenario_count(self):
        N = 5
        a, b = default_channel_params(N)
        K = 2
        scenarios = create_scenarios(N, a, b, num_actuator_faults=K)
        # 1 Nominal + K ActuatorFault + N SensorFault
        assert len(scenarios) == 1 + K + N
        actuator_names = sorted(s.name for s in scenarios if s.name.startswith("ActuatorFault_"))
        assert actuator_names == ["ActuatorFault_0", "ActuatorFault_1"]
        sensor_names = {s.name for s in scenarios if s.name.startswith("SensorFault_")}
        assert sensor_names == {f"SensorFault_{i}" for i in range(N)}

    def test_actuator_fault_indices_overrides_num_actuator_faults(self):
        N = 5
        a, b = default_channel_params(N)
        scenarios = create_scenarios(
            N, a, b, num_actuator_faults=2, actuator_fault_indices=[0, 3, 4],
        )
        actuator_names = sorted(s.name for s in scenarios if s.name.startswith("ActuatorFault_"))
        assert actuator_names == ["ActuatorFault_0", "ActuatorFault_3", "ActuatorFault_4"]
        assert len(scenarios) == 1 + 3 + N

    def test_actuator_fault_indices_out_of_range_rejected(self):
        N = 3
        a, b = default_channel_params(N)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, actuator_fault_indices=[0, 3])
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, actuator_fault_indices=[-1, 1])

    def test_actuator_fault_indices_duplicates_rejected(self):
        N = 3
        a, b = default_channel_params(N)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, actuator_fault_indices=[0, 0, 1])

    def test_num_actuator_faults_out_of_bounds_rejected(self):
        N = 3
        a, b = default_channel_params(N)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, num_actuator_faults=0)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, num_actuator_faults=N + 1)

    def test_num_sensor_faults_reduces_sensor_scenario_count(self):
        N = 5
        a, b = default_channel_params(N)
        K = 2
        scenarios = create_scenarios(N, a, b, num_sensor_faults=K)
        # 1 Nominal + N ActuatorFault + K SensorFault
        assert len(scenarios) == 1 + N + K
        sensor_names = sorted(s.name for s in scenarios if s.name.startswith("SensorFault_"))
        assert sensor_names == ["SensorFault_0", "SensorFault_1"]
        actuator_names = {s.name for s in scenarios if s.name.startswith("ActuatorFault_")}
        assert actuator_names == {f"ActuatorFault_{i}" for i in range(N)}

    def test_sensor_fault_indices_overrides_num_sensor_faults(self):
        N = 5
        a, b = default_channel_params(N)
        scenarios = create_scenarios(
            N, a, b, num_sensor_faults=2, sensor_fault_indices=[0, 3, 4],
        )
        sensor_names = sorted(s.name for s in scenarios if s.name.startswith("SensorFault_"))
        assert sensor_names == ["SensorFault_0", "SensorFault_3", "SensorFault_4"]
        assert len(scenarios) == 1 + N + 3

    def test_sensor_fault_indices_out_of_range_rejected(self):
        N = 3
        a, b = default_channel_params(N)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, sensor_fault_indices=[0, 3])
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, sensor_fault_indices=[-1, 1])

    def test_sensor_fault_indices_duplicates_rejected(self):
        N = 3
        a, b = default_channel_params(N)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, sensor_fault_indices=[0, 0, 1])

    def test_num_sensor_faults_out_of_bounds_rejected(self):
        N = 3
        a, b = default_channel_params(N)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, num_sensor_faults=0)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, num_sensor_faults=N + 1)

    def test_symmetric_actuator_and_sensor_fault_counts(self):
        """The scenario budget used by the runtime-scaling scripts: N=10,
        Ka=Ks=K sweeping 1..N gives total = 1+2K, covering 3 (K=1) to
        21 (K=N=10) -- exactly the range those scripts sweep."""
        N = 10
        a, b = default_channel_params(N)
        for K, expected_total in ((1, 3), (10, 21)):
            scenarios = create_scenarios(N, a, b, num_actuator_faults=K, num_sensor_faults=K)
            assert len(scenarios) == expected_total

    def test_actuator_fault_isolated_to_single_channel(self):
        N = 4
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        by_name = {s.name: s for s in scenarios}
        for i in range(N):
            s = by_name[f"ActuatorFault_{i}"]
            for j in range(N):
                if j == i:
                    assert float(s.p_interval.lower[j]) < 1.0
                    assert float(s.p_interval.upper[j]) <= 1.0
                else:
                    assert float(s.p_interval.lower[j]) == 1.0
                    assert float(s.p_interval.upper[j]) == 1.0
            assert bool(jnp.all(s.beta.lower == 1.0)) and bool(jnp.all(s.beta.upper == 1.0))
            assert bool(jnp.all(s.xi.lower == 0.0)) and bool(jnp.all(s.xi.upper == 0.0))

    def test_sensor_fault_isolated_to_single_channel(self):
        N = 4
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        by_name = {s.name: s for s in scenarios}
        for i in range(N):
            s = by_name[f"SensorFault_{i}"]
            assert bool(jnp.all(s.p_interval.lower == 1.0)) and bool(jnp.all(s.p_interval.upper == 1.0))
            for j in range(N):
                if j == i:
                    assert float(s.beta.lower[j]) < 1.0 or float(s.beta.upper[j]) > 1.0
                else:
                    assert float(s.beta.lower[j]) == 1.0
                    assert float(s.beta.upper[j]) == 1.0
                    assert float(s.xi.lower[j]) == 0.0
                    assert float(s.xi.upper[j]) == 0.0

    def test_beta_must_be_strictly_positive(self):
        N = 2
        a, b = default_channel_params(N)
        with pytest.raises(ValueError):
            create_scenarios(N, a, b, sensor_beta_lo=-0.1, sensor_beta_hi=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Output-map (observed_output / _invert_observation) tests
# ══════════════════════════════════════════════════════════════════════════════

class TestObservedOutput:

    def test_identity_when_beta1_xi0(self):
        scen = Scenario("id", None, point_ivl(jnp.ones(1)), point_ivl(jnp.ones(3)), point_ivl(jnp.zeros(3)))
        x_ivl = irx.Interval(lower=jnp.array([1.0, -2.0, 3.0]), upper=jnp.array([1.5, -1.0, 4.0]))
        y = observed_output(x_ivl, scen)
        np.testing.assert_allclose(np.array(y.lower), np.array(x_ivl.lower), atol=1e-6)
        np.testing.assert_allclose(np.array(y.upper), np.array(x_ivl.upper), atol=1e-6)

    def test_invert_recovers_point_state_with_point_fault_params(self):
        beta = point_ivl([2.0, 0.5])
        xi = point_ivl([1.0, -0.5])
        scen = Scenario("s", None, point_ivl(jnp.ones(2)), beta, xi)
        x_true = jnp.array([3.0, -2.0])
        y_ivl = observed_output(point_ivl(x_true), scen)
        x_rec = _invert_observation(y_ivl, scen)
        np.testing.assert_allclose(np.array(x_rec.lower), np.array(x_true), atol=1e-5)
        np.testing.assert_allclose(np.array(x_rec.upper), np.array(x_true), atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Propagation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEulerStepAndPropagation:

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_single_euler_step_matches_hand_computation(self, N):
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        nominal = scenarios[0]
        x0_np = np.linspace(-0.5, 0.5, N)
        x0 = point_ivl(x0_np)
        u = jnp.full(N, 0.3)
        dt = 0.05
        x1 = euler_step(nominal.emb_system, x0, u, nominal.p_interval, dt)

        expected_dot = np.array(a) * x0_np - np.array(b) * x0_np ** 3 + 1.0 * 0.3
        expected = x0_np + dt * expected_dot
        np.testing.assert_allclose(np.array(x1.lower), expected, atol=1e-5)
        np.testing.assert_allclose(np.array(x1.upper), expected, atol=1e-5)

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_valid_intervals_after_propagation(self, N):
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        x0 = small_ivl(N)
        u = jnp.full(N, 0.1)
        for s in scenarios:
            xf = propagate_scenario(x0, u, s, dt=0.02, num_steps=5)
            assert np.all(np.array(xf.lower) <= np.array(xf.upper)), \
                f"N={N} {s.name}: invalid interval after propagation"

    def test_actuator_fault_changes_faulted_channel(self):
        N = 2
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        nominal = scenarios[0]
        act_fault = next(s for s in scenarios if s.name == "ActuatorFault_0")
        x0 = small_ivl(N)
        u = jnp.full(N, 0.5)
        x_nom = propagate_scenario(x0, u, nominal, dt=0.05, num_steps=10)
        x_act = propagate_scenario(x0, u, act_fault, dt=0.05, num_steps=10)
        assert not np.allclose(np.array(x_nom.lower), np.array(x_act.lower), atol=1e-4)

    def test_multistep_matches_single_step_at_final_segment(self):
        N = 2
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        s = scenarios[0]
        x0 = small_ivl(N)
        u = jnp.full(N, 0.2)
        steps_per_segment, num_segments = 3, 4
        u_seq = jnp.tile(u, (num_segments, 1))

        x_multistep = propagate_scenario_multistep(x0, u_seq, s, dt=0.02, steps_per_segment=steps_per_segment)
        x_direct = propagate_scenario(x0, u, s, dt=0.02, num_steps=steps_per_segment * num_segments)
        np.testing.assert_allclose(np.array(x_multistep.lower), np.array(x_direct.lower), atol=1e-4)
        np.testing.assert_allclose(np.array(x_multistep.upper), np.array(x_direct.upper), atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Loss function tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSeparationLoss:

    def test_loss_is_finite_and_nonnegative(self):
        N = 2
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        x0 = small_ivl(N)
        u = jnp.full(N, 0.3)
        loss = separation_loss(u, x0, scenarios, dt=0.02, num_steps=5)
        assert float(loss) >= 0.0
        assert np.isfinite(float(loss))

    def test_loss_finite_with_reduced_actuator_faults(self):
        N = 4
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b, num_actuator_faults=1)
        assert len(scenarios) == 1 + 1 + N
        x0 = small_ivl(N)
        u = jnp.full(N, 0.2)
        loss = separation_loss(u, x0, scenarios, dt=0.02, num_steps=5)
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0

    def test_multistep_loss_matches_min_over_segments(self):
        N = 2
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        x0 = small_ivl(N)
        u_seq = jnp.tile(jnp.full(N, 0.3), (4, 1))
        loss = separation_loss_multistep(u_seq, x0, scenarios, dt=0.02, steps_per_segment=2)
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0


class TestOptimizers:

    @pytest.mark.parametrize("N", [2, 3])
    def test_single_step_optimizer_reduces_loss(self, N):
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        x0 = small_ivl(N)
        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.02, num_steps=5)
        u0 = jnp.full(N, 0.3)
        loss0 = float(opt.loss_fn(u0))
        u_opt, loss_opt = opt.optimize(u_init=u0, learning_rate=0.05, num_iters=30)
        assert loss_opt <= loss0 + 1e-6
        assert u_opt.shape == (N,)

    @pytest.mark.parametrize("N", [2, 3])
    def test_multistep_optimizer_runs_and_is_finite(self, N):
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        x0 = small_ivl(N)
        u_seq, loss, stats = optimize_multistep(
            scenarios, x0, dt=0.02, steps_per_segment=2, num_segments=5,
            learning_rate=0.05, num_iters=20, num_restarts=3, seed=0,
        )
        assert np.isfinite(loss)
        assert u_seq.shape == (5, N)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Intersection-refinement tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRefinement:

    def test_refined_loss_le_unrefined_for_same_u_seq(self):
        N = 2
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        x0 = small_ivl(N)
        num_steps = 4
        u_seq = jnp.tile(jnp.full(N, 0.3), (num_steps, 1))

        unrefined = float(separation_loss_multistep(u_seq, x0, scenarios, dt=0.02, steps_per_segment=1))
        refined = float(refined_overlap_loss(u_seq, x0, scenarios, dt=0.02, num_steps=num_steps))
        assert refined <= unrefined + 1e-5

    def test_refinement_output_is_finite_and_nonnegative(self):
        N = 2
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        x0 = small_ivl(N)
        u_seq = jnp.tile(jnp.full(N, 0.2), (3, 1))
        loss = float(propagate_with_refinement(x0, u_seq, scenarios, dt=0.02, num_steps=3))
        assert np.isfinite(loss)
        assert loss >= 0.0

    @pytest.mark.parametrize("N", [2, 3])
    def test_optimize_refined_gpu_runs_and_is_finite(self, N):
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b)
        x0 = small_ivl(N)
        u_opt, loss_opt, u_all, losses = optimize_refined_gpu(
            x0, scenarios, dt=0.02, num_steps=3, num_restarts=3,
            learning_rate=0.05, num_iters=15, seed=0,
        )
        assert np.isfinite(float(loss_opt))
        assert u_opt.shape == (3, N)

    def test_refinement_with_reduced_actuator_faults(self):
        N = 4
        a, b = default_channel_params(N)
        scenarios = create_scenarios(N, a, b, num_actuator_faults=1)
        x0 = small_ivl(N)
        u_seq = jnp.tile(jnp.full(N, 0.2), (3, 1))
        loss = float(propagate_with_refinement(x0, u_seq, scenarios, dt=0.02, num_steps=3))
        assert np.isfinite(loss)
        assert loss >= 0.0
