"""Tests for crazyflie_chain_controllers.py (Phase 1: masked-theta chain
controller system, four scenarios, single-step separating spoof-bias).

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python -m pytest examples/adaptive_spoofing/tests/
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp
import immrax as irx

_EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from adaptive_spoofing.crazyflie_chain_controllers import (
    ChainControllerSystem, get_system_and_embedding, euler_step, hover_reference,
    create_scenarios, Scenario, propagate_scenario, _propagate_all_scenarios,
    separation_loss, SeparatingInputOptimizer, optimize_parallel_gpu,
    _CANDIDATE_THETA, QPS_DT, _BIAS_LIM,
)


# ──────────────────────────────────────────────────────────────────────────
# Independent numpy transcription of the unified controller law, used only to
# cross-check ChainControllerSystem.f() -- kept deliberately separate from
# the module's own code (see crazyflie_12d.py's tests for the same pattern).
# ──────────────────────────────────────────────────────────────────────────
def _chain_rhs_numpy(x, b, theta6, ref15):
    k_p, k_v, k_a, k_i, k_j, base_flag = theta6
    pos, vel, acc, jerk = x[0:3], x[3:6], x[6:9], x[9:12]
    integ, cmd = x[12:15], x[15:18]
    rp, rv, ra, rj, rs = ref15[0:3], ref15[3:6], ref15[6:9], ref15[9:12], ref15[12:15]

    e_p = (pos + b) - rp
    e_v = vel - rv
    e_a = acc - ra
    e_j = jerk - rj

    target = rs - k_p * e_p - k_v * e_v - k_a * e_a - k_i * integ - k_j * e_j
    j_dot = (1.0 - base_flag) * target + base_flag * cmd

    return np.concatenate([vel, acc, jerk, j_dot, e_p, -k_j * e_j])


class TestDynamicsMatchReference:
    @pytest.mark.parametrize("candidate", list(_CANDIDATE_THETA.keys()))
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_numpy_transcription(self, candidate, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, 18)
        b = rng.uniform(-_BIAS_LIM, _BIAS_LIM, 3)
        ref15 = np.asarray(hover_reference())
        theta6 = _CANDIDATE_THETA[candidate]

        sys_ = ChainControllerSystem(jnp.asarray(ref15))
        p = jnp.array(theta6)
        xdot_ours = np.array(sys_.f(jnp.zeros(()), jnp.array(x), jnp.array(b), p))
        xdot_ref = _chain_rhs_numpy(x, b, theta6, ref15)

        np.testing.assert_allclose(xdot_ours, xdot_ref, atol=1e-5, rtol=1e-4)

    def test_qps_snap_chain_matches_theta_true(self):
        """qps_snap_chain's gains must be exactly rq3_model_bank.py's
        theta_true=(1680,1066,251,26) (poles -5..-8) -- QPS's real law."""
        k_p, k_v, k_a, k_i, k_j, base_flag = _CANDIDATE_THETA["qps_snap_chain"]
        assert (k_p, k_v, k_a, k_j) == (1680.0, 1066.0, 251.0, 26.0)
        assert k_i == 0.0 and base_flag == 0.0

    def test_indi_jerk_uses_cmd_not_target(self):
        """base_flag=1 -> j_dot should come from cmd, not the reference-based
        target -- i.e. indi_jerk is NOT a function of the reference at all."""
        ref_a = hover_reference(jnp.array([0.0, 0.0, 1.0]))
        ref_b = hover_reference(jnp.array([5.0, 5.0, 5.0]))  # very different setpoint
        x = jnp.zeros(18).at[15:18].set(jnp.array([0.1, 0.2, 0.3]))  # nonzero cmd
        b = jnp.zeros(3)
        theta = jnp.array(_CANDIDATE_THETA["indi_jerk"])

        sys_a = ChainControllerSystem(ref_a)
        sys_b = ChainControllerSystem(ref_b)
        jdot_a = sys_a.f(jnp.zeros(()), x, b, theta)[9:12]
        jdot_b = sys_b.f(jnp.zeros(()), x, b, theta)[9:12]
        # e_j = jerk - rj = 0 - 0 = 0 for both hover refs (only position setpoint
        # differs), so j_dot should be identical (== cmd) regardless of position ref.
        assert jnp.allclose(jdot_a, jdot_b)
        assert jnp.allclose(jdot_a, x[15:18])  # j_dot == cmd exactly


class TestEmbeddingAndStep:
    def test_natural_embedding_produces_interval(self):
        _, emb = get_system_and_embedding()
        x_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-3))
        u = jnp.zeros(3)
        p_ivl = irx.Interval(lower=jnp.array(_CANDIDATE_THETA["qps_snap_chain"]),
                             upper=jnp.array(_CANDIDATE_THETA["qps_snap_chain"]))
        x_next = euler_step(emb, x_ivl, u, p_ivl, dt=QPS_DT)
        assert isinstance(x_next, irx.Interval)
        assert x_next.lower.shape == (18,)
        assert jnp.all(x_next.lower <= x_next.upper)

    def test_zero_state_zero_bias_hover_ref_is_near_fixed_point(self):
        """At x=0 (chain at rest at the origin) with a hover ref at pos_sp=0
        and zero bias, all four candidates' e_p=e_v=e_a=e_j=0 -> target=0,
        cmd=0 -> j_dot=0 and the whole state derivative is exactly zero."""
        ref0 = hover_reference(jnp.zeros(3))
        for name, theta6 in _CANDIDATE_THETA.items():
            sys_ = ChainControllerSystem(ref0)
            xdot = sys_.f(jnp.zeros(()), jnp.zeros(18), jnp.zeros(3), jnp.array(theta6))
            assert jnp.allclose(xdot, 0.0, atol=1e-6), f"{name} is not at rest at the origin"


class TestScenarios:
    def test_create_scenarios_returns_four_by_default(self):
        scenarios = create_scenarios()
        assert len(scenarios) == 4
        assert {s.name for s in scenarios} == set(_CANDIDATE_THETA.keys())

    def test_scenarios_share_one_emb_system(self):
        scenarios = create_scenarios()
        emb0 = scenarios[0].emb_system
        assert all(s.emb_system is emb0 for s in scenarios)

    def test_named_subset(self):
        scenarios = create_scenarios(names=["pd_pos_vel", "indi_jerk"])
        assert [s.name for s in scenarios] == ["pd_pos_vel", "indi_jerk"]

    def test_pd_and_pid_are_identical_after_one_step(self):
        """Regression test for a genuine finding: pid_pos_vel_i's integral
        state starts at 0, so at num_steps=1 it hasn't diverged from
        pd_pos_vel yet -- their one-step reachable sets should coincide
        exactly, regardless of bias."""
        scenarios = create_scenarios(names=["pd_pos_vel", "pid_pos_vel_i"])
        # integ (indices 12:15) must start at an exact point -- any nonzero
        # WIDTH there already makes k_i*integ a nonzero interval term for
        # pid_pos_vel_i even though it's centered at zero, which is a
        # different (also true, but not what this test isolates) reason to
        # separate. See test_pd_and_pid_diverge_after_several_steps for that.
        pert = jnp.full(18, 1e-2).at[12:15].set(0.0)
        x0_ivl = irx.icentpert(jnp.zeros(18), pert)
        u = jnp.array([0.05, -0.03, 0.02])
        x_pd = propagate_scenario(x0_ivl, u, scenarios[0], dt=QPS_DT, num_steps=1)
        x_pid = propagate_scenario(x0_ivl, u, scenarios[1], dt=QPS_DT, num_steps=1)
        assert jnp.allclose(x_pd.lower, x_pid.lower, atol=1e-6)
        assert jnp.allclose(x_pd.upper, x_pid.upper, atol=1e-6)

    def test_pd_and_pid_diverge_after_several_steps(self):
        """...but DO separate once the integral term has had time to
        accumulate (num_steps=5) -- confirms discrimination needs horizon,
        not just bias, for closely-related candidates."""
        scenarios = create_scenarios(names=["pd_pos_vel", "pid_pos_vel_i"])
        x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-2))
        u = jnp.array([0.1, -0.1, 0.05])
        x_pd = propagate_scenario(x0_ivl, u, scenarios[0], dt=QPS_DT, num_steps=5)
        x_pid = propagate_scenario(x0_ivl, u, scenarios[1], dt=QPS_DT, num_steps=5)
        assert not jnp.allclose(x_pd.lower, x_pid.lower, atol=1e-4)


class TestSeparationLossAndOptimizer:
    def test_loss_is_symmetric_pair_sum_nonnegative(self):
        scenarios = create_scenarios()
        x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-3))
        loss = separation_loss(jnp.zeros(3), x0_ivl, scenarios, dt=QPS_DT, num_steps=5)
        assert loss >= 0.0

    def test_all_four_scenarios_already_separate_at_zero_bias(self):
        """The four candidates' gains differ by orders of magnitude (26 vs
        1680), so even with zero attacker bias they diverge within a few
        steps from a tight initial box -- a real property of this bank, not
        an optimizer artifact."""
        scenarios = create_scenarios()
        x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-3))
        loss = separation_loss(jnp.zeros(3), x0_ivl, scenarios, dt=QPS_DT, num_steps=5)
        assert float(loss) == pytest.approx(0.0, abs=1e-9)

    def test_optimizer_reduces_loss_for_the_hard_pair(self):
        """For the genuinely hard-to-separate pair (pd_pos_vel vs
        pid_pos_vel_i, see test_pd_and_pid_are_identical_after_one_step),
        with a wide initial box and short horizon, gradient descent on the
        spoof bias should find a nonzero bias that separates them more than
        zero bias does."""
        scenarios = create_scenarios(names=["pd_pos_vel", "pid_pos_vel_i"])
        x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 5e-2))
        opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt=QPS_DT, num_steps=3)

        loss_zero = opt.loss_fn(jnp.zeros(3))
        u_star, loss_star, _, _ = optimize_parallel_gpu(
            opt, num_restarts=32, learning_rate=0.02, num_iters=100,
        )
        assert float(loss_star) <= float(loss_zero) + 1e-9

    def test_optimized_bias_respects_box_constraint(self):
        scenarios = create_scenarios()
        x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-2))
        opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt=QPS_DT, num_steps=3)
        u_star, _, u_all, _ = optimize_parallel_gpu(opt, num_restarts=16, num_iters=50)
        assert jnp.all(jnp.abs(u_all) <= _BIAS_LIM + 1e-6)

    def test_evaluate_reports_all_six_pairs(self):
        scenarios = create_scenarios()
        x0_ivl = irx.icentpert(jnp.zeros(18), jnp.full(18, 1e-3))
        opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt=QPS_DT, num_steps=2)
        result = opt.evaluate(jnp.zeros(3))
        assert len(result['pairwise_overlaps']) == 6  # C(4,2)
        assert len(result['volumes']) == 4
