"""
Tests for quadrotor_separating_input.py
"""
import os
import sys
from pathlib import Path

# Force CPU-only JAX before any jax import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# File is at examples/quadrotor_fault_diagnosis/tests/<name>.py
#   parents[1] = examples/quadrotor_fault_diagnosis/
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

from quadrotor_separating_input import (
    QuadrotorSystem,
    get_system_and_embedding,
    Scenario,
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
    _HOVER_THRUST,
    _M, _G, _IXX, _IYY, _IZZ,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def point_ivl(arr):
    a = jnp.array(arr, dtype=float)
    return irx.Interval(lower=a, upper=a)


def small_ivl(width=0.01):
    center = jnp.zeros(12)
    return irx.icentpert(center, jnp.full(12, width))


_HOVER_U = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])


# ══════════════════════════════════════════════════════════════════════════════
# 1. System-level unit tests (no interval arithmetic)
# ══════════════════════════════════════════════════════════════════════════════

class TestQuadrotorSystemDynamics:

    def test_xlen_is_12(self):
        assert QuadrotorSystem().xlen == 12

    def test_hover_equilibrium_is_exact_fixed_point(self):
        """At x=0 (level attitude, zero velocity/rates) with u = [mg,0,0,0]
        and alpha nominal, every one of the 12 derivatives must be exactly
        0 -- thrust exactly cancels gravity and every trig/coupling term
        vanishes. See PLAN.md Sec 1.

        atol=1e-5, not the original 1e-10: verified (via a float64 rerun of
        this exact call, jax_enable_x64=True, giving exactly 0.0 in every
        component) that the ~9.5e-7 residual now observed in float32 is a
        rounding artifact of the real Crazyflie's much smaller mass scale
        (_M=0.03589kg vs. the original generic-quadrotor default
        _M=0.468kg), not a physics or dynamics-implementation bug -- the
        SAME `U1/self.m - g` cancellation that landed exactly at machine
        epsilon for the old, larger mass leaves a small nonzero float32
        residual at this mass instead. 1e-5 is comfortably above the
        observed residual while still catching a genuine mismatch (e.g. a
        sign error) many orders of magnitude larger than rounding noise."""
        sys_ = QuadrotorSystem()
        dx = sys_.f(jnp.zeros(()), jnp.zeros(12), _HOVER_U, jnp.ones(4))
        np.testing.assert_allclose(np.array(dx), np.zeros(12), atol=1e-5)

    def test_derivative_at_origin_matches_hand_computation(self):
        """At x=0, every kinematic/Coriolis term vanishes (all velocities
        and rates are 0), so dx reduces to exactly
        [0,0,0,0,0,0,0,0, U1/m-g, U2/Ixx, U3/Iyy, U4/Izz]."""
        sys_ = QuadrotorSystem()
        u = jnp.array([2.0, 0.3, -0.2, 0.1])
        dx = sys_.f(jnp.zeros(()), jnp.zeros(12), u, jnp.ones(4))
        expected = np.zeros(12)
        expected[8] = 2.0 / _M - _G
        expected[9] = 0.3 / _IXX
        expected[10] = -0.2 / _IYY
        expected[11] = 0.1 / _IZZ
        np.testing.assert_allclose(np.array(dx), expected, atol=1e-8)

    def test_each_actuator_channel_isolated_at_origin(self):
        """At x=0, alpha_i only rescales u_i's own dedicated derivative
        component (index 8 for thrust, 9/10/11 for roll/pitch/yaw) and
        nothing else -- the clean isolation the fault-diagnosis scenarios
        rely on being distinguishable in the first place."""
        sys_ = QuadrotorSystem()
        u = jnp.array([2.0, 0.3, -0.2, 0.1])
        deriv_index = {0: 8, 1: 9, 2: 10, 3: 11}
        for chan, idx in deriv_index.items():
            alpha_nom = jnp.ones(4)
            alpha_fault = jnp.ones(4).at[chan].set(0.5)
            dx_nom = sys_.f(jnp.zeros(()), jnp.zeros(12), u, alpha_nom)
            dx_fault = sys_.f(jnp.zeros(()), jnp.zeros(12), u, alpha_fault)
            for k in range(12):
                if k == idx:
                    assert not np.isclose(float(dx_nom[k]), float(dx_fault[k]), atol=1e-9), \
                        f"channel {chan}: expected derivative {idx} to change"
                else:
                    np.testing.assert_allclose(float(dx_nom[k]), float(dx_fault[k]), atol=1e-9,
                                               err_msg=f"channel {chan} leaked into derivative {k}")

    def test_embedding_cache_reuses_instance(self):
        sys_a, emb_a = get_system_and_embedding()
        sys_b, emb_b = get_system_and_embedding()
        assert sys_a is sys_b
        assert emb_a is emb_b

    def test_t_must_be_jax_array_not_python_float(self):
        """Regression guard: Python-float t breaks natif_jaxpr's invar count."""
        _, emb = get_system_and_embedding()
        x_ut = irx.i2ut(small_ivl())
        p_ivl = irx.icentpert(jnp.ones(4), jnp.zeros(4))
        emb.f(jnp.zeros(()), x_ut, _HOVER_U, p_ivl)
        with pytest.raises(Exception):
            emb.f(0.0, x_ut, _HOVER_U, p_ivl)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scenario construction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateScenarios:

    def test_five_scenarios_created(self):
        scenarios = create_scenarios()
        assert len(scenarios) == 5
        names = [s.name for s in scenarios]
        assert names == ["Nominal", "ActuatorFault_1", "ActuatorFault_2",
                          "ActuatorFault_3", "ActuatorFault_4"]

    def test_nominal_alpha_is_all_ones(self):
        nominal = create_scenarios()[0]
        np.testing.assert_allclose(np.array(nominal.p_interval.lower), np.ones(4))
        np.testing.assert_allclose(np.array(nominal.p_interval.upper), np.ones(4))

    def test_actuator_fault_isolated_to_single_channel(self):
        scenarios = create_scenarios(alpha_lo=0.5, alpha_hi=0.9)
        by_name = {s.name: s for s in scenarios}
        for i, name in enumerate(["ActuatorFault_1", "ActuatorFault_2",
                                   "ActuatorFault_3", "ActuatorFault_4"]):
            s = by_name[name]
            for j in range(4):
                if j == i:
                    assert float(s.p_interval.lower[j]) == pytest.approx(0.5)
                    assert float(s.p_interval.upper[j]) == pytest.approx(0.9)
                else:
                    assert float(s.p_interval.lower[j]) == 1.0
                    assert float(s.p_interval.upper[j]) == 1.0

    def test_scenarios_share_one_embedding(self):
        scenarios = create_scenarios()
        emb0 = scenarios[0].emb_system
        assert all(s.emb_system is emb0 for s in scenarios)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Propagation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEulerStepAndPropagation:

    def test_single_euler_step_matches_hand_computation_at_hover(self):
        scenarios = create_scenarios()
        nominal = scenarios[0]
        x0 = point_ivl(jnp.zeros(12))
        dt = 0.01
        x1 = euler_step(nominal.emb_system, x0, _HOVER_U, nominal.p_interval, dt)
        # at exact hover equilibrium, one Euler step should leave the state at 0
        np.testing.assert_allclose(np.array(x1.lower), np.zeros(12), atol=1e-8)
        np.testing.assert_allclose(np.array(x1.upper), np.zeros(12), atol=1e-8)

    def test_valid_intervals_after_propagation(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        for s in scenarios:
            xf = propagate_scenario(x0, _HOVER_U, s, dt=0.01, num_steps=5)
            assert np.all(np.array(xf.lower) <= np.array(xf.upper)), \
                f"{s.name}: invalid interval after propagation"

    def test_actuator_fault_changes_propagated_interval(self):
        scenarios = create_scenarios()
        nominal, act_fault = scenarios[0], scenarios[1]
        x0 = small_ivl()
        u = jnp.array([_HOVER_THRUST, 0.01, 0.0, 0.0])
        x_nom = propagate_scenario(x0, u, nominal, dt=0.01, num_steps=10)
        x_act = propagate_scenario(x0, u, act_fault, dt=0.01, num_steps=10)
        assert not np.allclose(np.array(x_nom.lower), np.array(x_act.lower), atol=1e-6)

    def test_multistep_matches_single_step_at_final_segment(self):
        scenarios = create_scenarios()
        s = scenarios[0]
        x0 = small_ivl()
        u = jnp.array([_HOVER_THRUST, 0.005, 0.0, 0.0])
        steps_per_segment, num_segments = 3, 4
        u_seq = jnp.tile(u, (num_segments, 1))

        x_multistep = propagate_scenario_multistep(x0, u_seq, s, dt=0.01, steps_per_segment=steps_per_segment)
        x_direct = propagate_scenario(x0, u, s, dt=0.01, num_steps=steps_per_segment * num_segments)
        np.testing.assert_allclose(np.array(x_multistep.lower), np.array(x_direct.lower), atol=1e-4)
        np.testing.assert_allclose(np.array(x_multistep.upper), np.array(x_direct.upper), atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Loss function tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSeparationLoss:

    def test_loss_is_finite_and_nonnegative(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        loss = separation_loss(_HOVER_U, x0, scenarios, dt=0.01, num_steps=5)
        assert float(loss) >= 0.0
        assert np.isfinite(float(loss))

    def test_multistep_loss_matches_min_over_segments(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_seq = jnp.tile(_HOVER_U, (4, 1))
        loss = separation_loss_multistep(u_seq, x0, scenarios, dt=0.01, steps_per_segment=2)
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0


class TestOptimizers:

    def test_single_step_optimizer_reduces_loss(self):
        # num_steps kept small deliberately -- this system's trig-heavy
        # embedding (tan(theta), 1/cos(theta)) makes gradient-compile time
        # grow very steeply with unrolled step count (measured: num_steps=1
        # ~2s, num_steps=5 ~56s/4.3GB for a single restart). See PLAN.md /
        # module docstring "compile-cost note".
        scenarios = create_scenarios()
        x0 = small_ivl()
        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.01, num_steps=2)
        u0 = _HOVER_U
        loss0 = float(opt.loss_fn(u0))
        u_opt, loss_opt = opt.optimize(u_init=u0, learning_rate=0.01, num_iters=20)
        assert loss_opt <= loss0 + 1e-6
        assert u_opt.shape == (4,)

    def test_multistep_optimizer_runs_and_is_finite(self):
        # steps_per_segment=1 (unrolled depth per scan iteration) kept
        # minimal for the same compile-cost reason as above; num_segments
        # is cheap to grow since segments are scanned, not unrolled.
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_seq, loss, stats = optimize_multistep(
            scenarios, x0, dt=0.01, steps_per_segment=1, num_segments=3,
            learning_rate=0.01, num_iters=10, num_restarts=2, seed=0,
        )
        assert np.isfinite(loss)
        assert u_seq.shape == (3, 4)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Intersection-refinement tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRefinement:

    # num_steps kept at the minimum useful value (2 = one initial step + one
    # refined step) throughout this class -- even after vmapping the
    # per-pair loop (see propagate_with_refinement's docstring), refinement
    # compile cost still grows steeply with num_steps for this system
    # (measured post-fix: num_steps=2 ~22s, num_steps=3 ~48s/3.6GB).

    def test_refined_loss_le_unrefined_for_same_u_seq(self):
        """Refinement can only tighten reachable sets, never loosen them, so
        the refined loss should never exceed the unrefined multistep loss
        for the same control sequence and horizon."""
        scenarios = create_scenarios()
        x0 = small_ivl()
        num_steps = 2
        u_seq = jnp.tile(_HOVER_U, (num_steps, 1))

        unrefined = float(separation_loss_multistep(u_seq, x0, scenarios, dt=0.01, steps_per_segment=1))
        refined = float(refined_overlap_loss(u_seq, x0, scenarios, dt=0.01, num_steps=num_steps))
        assert refined <= unrefined + 1e-5

    def test_refinement_output_is_finite_and_nonnegative(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_seq = jnp.tile(jnp.array([_HOVER_THRUST, 0.005, -0.005, 0.0]), (2, 1))
        loss = float(propagate_with_refinement(x0, u_seq, scenarios, dt=0.01, num_steps=2))
        assert np.isfinite(loss)
        assert loss >= 0.0

    def test_optimize_refined_gpu_runs_and_is_finite(self):
        scenarios = create_scenarios()
        x0 = small_ivl()
        u_opt, loss_opt, u_all, losses = optimize_refined_gpu(
            x0, scenarios, dt=0.01, num_steps=2, num_restarts=2,
            learning_rate=0.01, num_iters=8, seed=0,
        )
        assert np.isfinite(float(loss_opt))
        assert u_opt.shape == (2, 4)
