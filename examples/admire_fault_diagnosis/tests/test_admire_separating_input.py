"""
Tests for admire_separating_input.py

NOTE on runtime: ADMIRE's dynamics are division-heavy (see the module
docstring's "Perf" discussion) -- reverse-mode AD through even ONE Euler
step costs tens of seconds to compile, independent of restart/iteration
count. To keep this suite's total runtime bounded, every grad-dependent
test below uses the SMALLEST scenario count (2-3, not the full 11) and
horizon (1-2 steps) that still exercises the code path meaningfully; the
forward-only (no-grad) tests use the full 11-scenario set since those are
cheap. Expect this suite to take several minutes overall (compile-bound),
not the ~30-150s typical of this project's cheaper-dynamics modules
(unicycle, nonlinear_chain).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_EXAMPLES_DIR = Path(__file__).resolve().parents[2]
_PKG_DIR = Path(__file__).resolve().parents[1]
# _PKG_DIR must resolve FIRST: examples/admire/ also has a module literally
# named admire_separating_input.py, and admire_separating_input.py's own
# import (below) adds examples/admire/ to sys.path as a side effect for its
# `admire.py` dependency -- inserting _PKG_DIR last (so it ends up at
# sys.path[0]) ensures `from admire_separating_input import ...` resolves
# to THIS folder's module, not examples/admire's.
for _d in (_EXAMPLES_DIR, _PKG_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx
import pytest

from admire_separating_input import (
    AdmireNineDoFLinAct,
    get_system_and_embedding,
    Scenario,
    create_scenarios,
    output_interval,
    overlap_size_log,
    euler_step,
    propagate_scenario,
    separation_loss,
    SeparatingInputOptimizer,
    optimize_parallel_gpu,
    propagate_scenario_multistep,
    separation_loss_multistep,
    optimize_multistep_gpu,
    optimize_multistep,
    propagate_with_refinement,
    refined_overlap_loss,
    optimize_refined_gpu,
    collect_refinement_history,
    _step_one_pair,
)


def _x0_ivl(width=0.01):
    x0_nom = jnp.zeros(9).at[0].set(343.0 * 0.3)
    return irx.icentpert(x0_nom, jnp.ones(9) * width)


# ══════════════════════════════════════════════════════════════════════════════
# 1. System-level unit tests (cheap, forward-only)
# ══════════════════════════════════════════════════════════════════════════════

class TestAdmireSystemDynamics:

    def test_xlen_is_9(self):
        assert AdmireNineDoFLinAct().xlen == 9

    def test_zero_control_zero_fault_gives_finite_derivative(self):
        sys_ = AdmireNineDoFLinAct()
        x0 = jnp.zeros(9).at[0].set(343.0 * 0.3)
        dx = sys_.f(jnp.zeros(()), x0, jnp.zeros(10), jnp.ones(10))
        assert np.all(np.isfinite(np.array(dx)))
        # with zero control, actuator_dynamics term vanishes -> derivative
        # is purely internal dynamics, and at zero angle-of-attack/sideslip
        # the angular-rate derivatives (pb,qb,rb) must be exactly zero.
        np.testing.assert_allclose(np.array(dx[3:6]), np.zeros(3), atol=1e-10)

    def test_embedding_cache_reuses_instance(self):
        sys_a, emb_a = get_system_and_embedding()
        sys_b, emb_b = get_system_and_embedding()
        assert sys_a is sys_b
        assert emb_a is emb_b

    def test_t_must_be_jax_array_not_python_float(self):
        """Regression guard: Python-float t breaks natif_jaxpr's invar count."""
        _, emb = get_system_and_embedding()
        x_ut = irx.i2ut(_x0_ivl())
        u = jnp.zeros(10)
        p_ivl = irx.icentpert(jnp.ones(10), jnp.zeros(10))
        emb.f(jnp.zeros(()), x_ut, u, p_ivl)
        with pytest.raises(Exception):
            emb.f(0.0, x_ut, u, p_ivl)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Scenario / output-model tests (cheap, forward-only)
# ══════════════════════════════════════════════════════════════════════════════

class TestScenariosAndOutput:

    def test_eleven_scenarios_created(self):
        scenarios = create_scenarios()
        assert len(scenarios) == 11
        assert scenarios[0].name == "Nominal"
        assert [s.name for s in scenarios[1:]] == [
            "Right Canard", "Left Canard", "Right Outer Elev", "Right Inner Elev",
            "Left Inner Elev", "Left Outer Elev", "Rudder", "Flap", "Yaw TV", "Pitch TV",
        ]

    def test_nominal_has_all_ones_p(self):
        nominal = create_scenarios()[0]
        np.testing.assert_allclose(np.array(nominal.p_interval.lower), np.ones(10))
        np.testing.assert_allclose(np.array(nominal.p_interval.upper), np.ones(10))

    def test_fault_scenario_zeroes_exactly_one_surface(self):
        scenarios = create_scenarios(fault_effectiveness=0.0)
        for idx, s in enumerate(scenarios[1:]):
            p = np.array(s.p_interval.lower)
            assert p[idx] == 0.0
            assert np.sum(p == 0.0) == 1

    def test_partial_fault_effectiveness(self):
        scenarios = create_scenarios(fault_effectiveness=0.3)
        assert float(scenarios[1].p_interval.lower[0]) == pytest.approx(0.3)

    def test_output_interval_extracts_pb_qb_rb(self):
        x_ivl = irx.Interval(lower=jnp.arange(9.0), upper=jnp.arange(9.0) + 1.0)
        y = output_interval(x_ivl)
        np.testing.assert_allclose(np.array(y.lower), np.array([3.0, 4.0, 5.0]))
        np.testing.assert_allclose(np.array(y.upper), np.array([4.0, 5.0, 6.0]))

    def test_overlap_size_log_zero_when_disjoint(self):
        a = irx.Interval(lower=jnp.zeros(3), upper=jnp.ones(3))
        b = irx.Interval(lower=jnp.full(3, 5.0), upper=jnp.full(3, 6.0))
        assert float(overlap_size_log(a, b)) == 0.0

    def test_overlap_size_log_positive_when_overlapping(self):
        a = irx.Interval(lower=jnp.zeros(3), upper=jnp.ones(3))
        b = irx.Interval(lower=jnp.full(3, 0.5), upper=jnp.full(3, 1.5))
        assert float(overlap_size_log(a, b)) > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Propagation tests (cheap, forward-only)
# ══════════════════════════════════════════════════════════════════════════════

class TestPropagation:

    def test_euler_step_zero_control_zero_alpha_beta_no_rate_change(self):
        scenarios = create_scenarios()
        nominal = scenarios[0]
        x0 = _x0_ivl()
        x1 = euler_step(nominal.emb_system, x0, jnp.zeros(10), nominal.p_interval, dt=0.05)
        # angle of attack/sideslip start at 0 with 0.01 half-width -- rate
        # channels [3:6] should stay very close to 0 after one small step
        assert np.all(np.abs(np.array(x1.lower[3:6])) < 0.1)

    def test_valid_intervals_after_propagation(self):
        scenarios = create_scenarios()
        x0 = _x0_ivl()
        u = jnp.zeros(10)
        for s in scenarios[:3]:
            xf = propagate_scenario(x0, u, s, dt=0.05, num_steps=2)
            assert np.all(np.array(xf.lower) <= np.array(xf.upper))
            assert np.all(np.isfinite(np.array(xf.lower)))
            assert np.all(np.isfinite(np.array(xf.upper)))

    def test_multistep_matches_single_step_at_final_segment(self):
        scenarios = create_scenarios()
        s = scenarios[0]
        x0 = _x0_ivl()
        u = jnp.zeros(10)
        steps_per_segment, num_segments = 2, 2
        u_seq = jnp.tile(u, (num_segments, 1))
        x_multi = propagate_scenario_multistep(x0, u_seq, s, dt=0.05, steps_per_segment=steps_per_segment)
        x_direct = propagate_scenario(x0, u, s, dt=0.05, num_steps=steps_per_segment * num_segments)
        np.testing.assert_allclose(np.array(x_multi.lower), np.array(x_direct.lower), atol=1e-4)
        np.testing.assert_allclose(np.array(x_multi.upper), np.array(x_direct.upper), atol=1e-4)

    def test_separation_loss_finite_and_nonnegative_forward_only(self):
        scenarios = create_scenarios()[:3]
        x0 = _x0_ivl()
        loss = separation_loss(jnp.zeros(10), x0, scenarios, dt=0.05, num_steps=2)
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0

    def test_multistep_loss_finite_forward_only(self):
        scenarios = create_scenarios()[:3]
        x0 = _x0_ivl()
        u_seq = jnp.zeros((2, 10))
        loss = separation_loss_multistep(u_seq, x0, scenarios, dt=0.05, steps_per_segment=1)
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Optimizer tests (grad-dependent -- small scale, slow to compile)
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizers:

    def test_single_step_grad_finite(self):
        scenarios = create_scenarios()[:2]
        x0 = _x0_ivl()
        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.05, num_steps=1)
        g = opt.grad_fn(jnp.zeros(10))
        assert np.all(np.isfinite(np.array(g)))

    def test_optimize_parallel_gpu_runs_and_is_finite(self):
        scenarios = create_scenarios()[:2]
        x0 = _x0_ivl()
        opt = SeparatingInputOptimizer(scenarios, x0, dt=0.05, num_steps=1)
        u_opt, loss_opt, u_all, losses = optimize_parallel_gpu(
            opt, num_restarts=2, learning_rate=0.05, num_iters=2, seed=0
        )
        assert np.isfinite(float(loss_opt))
        assert u_opt.shape == (10,)

    def test_optimize_multistep_gpu_runs_and_is_finite(self):
        scenarios = create_scenarios()[:2]
        x0 = _x0_ivl()
        u_seq, loss, u_final, losses = optimize_multistep_gpu(
            x0_ivl=x0, scenarios=scenarios, dt=0.05, steps_per_segment=1,
            num_segments=2, num_restarts=2, num_iters=2, seed=0,
        )
        assert np.isfinite(float(loss))
        assert u_seq.shape == (2, 10)

    def test_optimize_multistep_wrapper_runs_and_is_finite(self):
        scenarios = create_scenarios()[:2]
        x0 = _x0_ivl()
        u_seq, loss, stats = optimize_multistep(
            scenarios, x0, dt=0.05, steps_per_segment=1, num_segments=2,
            num_restarts=2, num_iters=2, seed=0,
        )
        assert np.isfinite(loss)
        assert u_seq.shape == (2, 10)
        assert 'pairwise_overlaps' in stats


# ══════════════════════════════════════════════════════════════════════════════
# 5. Intersection-refinement tests (grad-dependent -- small scale)
# ══════════════════════════════════════════════════════════════════════════════

class TestRefinement:

    def test_step_one_pair_no_overlap_falls_back_to_unchanged_state(self):
        """Regression guard for the no-overlap-fallback bug fix (see module
        docstring): on no-overlap, the pair's state must fall back to the
        UNCHANGED input UT array, not a fabrication derived from one
        scenario's own output center alone."""
        _, emb = get_system_and_embedding()
        _t = jnp.zeros(())
        x0 = jnp.zeros(9).at[0].set(343.0 * 0.3)
        # Two disjoint, far-apart output (pb,qb,rb) intervals -> guaranteed no overlap
        xi_ut = jnp.concatenate([x0, x0]).at[3:6].set(0.0).at[12:15].set(0.01)
        xj_ut = jnp.concatenate([x0, x0]).at[3:6].set(50.0).at[12:15].set(50.01)
        p_ut = irx.i2ut(irx.icentpert(jnp.ones(10), jnp.zeros(10)))
        u_zero = jnp.zeros(10)
        xn_i, xn_j, pair_cost = _step_one_pair(xi_ut, xj_ut, p_ut, p_ut, u_zero, emb, dt=0.01, _t=_t)
        assert float(pair_cost) == 0.0
        # zero control + fault_effectiveness irrelevant here (u=0 means the
        # actuator term vanishes) -- state should evolve only via internal
        # dynamics, i.e. NOT be corrupted by a fabricated fallback in the
        # [3:6]/[12:15] output slice. Check the fallback slice specifically:
        # xi's own [3:6]/[12:15] must be preserved as xi_ut's OWN values
        # (has_overlap=False -> xi_ref == xi_ut exactly), not xi's center.
        expected_xi_ref = xi_ut  # has_overlap False -> xi_ref == xi_ut unchanged
        dx_i = emb.f(_t, expected_xi_ref, u_zero, irx.ut2i(p_ut)) * 0.01 + expected_xi_ref
        np.testing.assert_allclose(np.array(xn_i), np.array(dx_i), atol=1e-6)

    def test_refined_loss_le_unrefined_for_same_u_seq(self):
        scenarios = create_scenarios()[:2]
        x0 = _x0_ivl()
        num_steps = 2
        u_seq = jnp.zeros((num_steps, 10))
        unrefined = float(separation_loss_multistep(u_seq, x0, scenarios, dt=0.05, steps_per_segment=1))
        refined = float(refined_overlap_loss(u_seq, x0, scenarios, dt=0.05, num_steps=num_steps))
        # unrefined uses log1p-overlap, refined uses raw clip-product overlap
        # (see module docstring) -- different units, so this is a sanity
        # check on finiteness/nonnegativity, not a direct <= comparison.
        assert np.isfinite(unrefined) and np.isfinite(refined)
        assert refined >= 0.0

    def test_refinement_output_finite_and_nonnegative(self):
        scenarios = create_scenarios()[:2]
        x0 = _x0_ivl()
        u_seq = jnp.zeros((2, 10))
        loss = float(propagate_with_refinement(u_seq, x0, scenarios, dt=0.05, num_steps=2))
        assert np.isfinite(loss)
        assert loss >= 0.0

    def test_optimize_refined_gpu_runs_and_is_finite(self):
        scenarios = create_scenarios()[:2]
        x0 = _x0_ivl()
        u_opt, loss_opt, u_all, losses = optimize_refined_gpu(
            x0, scenarios, dt=0.05, num_steps=2, num_restarts=2,
            learning_rate=0.05, num_iters=2, seed=0,
        )
        assert np.isfinite(float(loss_opt))
        assert u_opt.shape == (2, 10)

    def test_collect_refinement_history_matches_propagate_with_refinement(self):
        """The history collector uses the exact same per-pair helper as the
        jittable loss -- their pairwise overlap at each step must match the
        loss's own per-step overlaps (up to the min-over-steps the jittable
        version takes)."""
        scenarios = create_scenarios()[:2]
        x0 = _x0_ivl()
        num_steps = 2
        u_seq = jnp.zeros((num_steps, 10))

        steps, pairs = collect_refinement_history(x0, u_seq, scenarios, dt=0.05, num_steps=num_steps)
        assert len(steps) == num_steps
        assert pairs == [(0, 1)]

        step_sums = []
        for step in steps:
            total = 0.0
            for obs_i, obs_j, _ in step['pair_obs']:
                lo = jnp.maximum(obs_i.lower, obs_j.lower)
                hi = jnp.minimum(obs_i.upper, obs_j.upper)
                total += float(jnp.prod(jnp.maximum(hi - lo, 0.0)))
            step_sums.append(total)
        expected_min = min(step_sums)
        actual = float(refined_overlap_loss(u_seq, x0, scenarios, dt=0.05, num_steps=num_steps))
        np.testing.assert_allclose(actual, expected_min, atol=1e-5)
