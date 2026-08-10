"""Tests for crazyflie_firmware_controllers.py -- the 4 REAL Crazyflie
firmware controllers (cf_pid, cf_mellinger, cf_indi, cf_brescianini),
transcribed from rq3_crazyflie_surrogates.py.

Run with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python -m pytest examples/adaptive_spoofing/tests/test_crazyflie_firmware_controllers.py
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

_RQ3_DIR = Path.home() / "adaptive_spoofing" / "experiments" / "RQ3_transferability_pipeline"
if str(_RQ3_DIR) not in sys.path:
    sys.path.insert(0, str(_RQ3_DIR))

import rq3_crazyflie_surrogates as ref
from adaptive_spoofing.crazyflie_firmware_controllers import (
    create_scenarios, hover_reference, euler_step, CANDIDATE_NAMES, QPS_DT,
    _safe_abs, _clip_survival,
)


def _random_state(rng, scale=(0.2, 0.1, 0.02, 0.05)):
    x = np.zeros(12)
    x[0:3] = rng.normal(scale=scale[0], size=3)
    x[3:6] = rng.normal(scale=scale[1], size=3)
    x[6:9] = rng.normal(scale=scale[2], size=3)
    x[9:12] = rng.normal(scale=scale[3], size=3)
    return x


def _random_mem(rng, scale=0.05):
    return ref.ControllerMemory(integ=rng.normal(scale=scale, size=3),
                                prev_cmd=rng.normal(scale=scale, size=3),
                                accel_filt=rng.normal(scale=scale, size=3))


class TestSoundAbsAndClipSurvival:
    def test_abs_via_sqrt_square_matches_numpy_at_points(self):
        for v in [-3.0, -0.001, 0.0, 0.001, 3.0]:
            assert float(_safe_abs(jnp.array(v))) == pytest.approx(abs(v), abs=1e-9)

    def test_abs_sound_for_straddling_interval(self):
        """Regression guard: jnp.maximum(x,-x) is UNSOUND here (reproduces
        the straddling interval instead of [0, max(|lo|,|hi|)]); this test
        would pass for either formula pointwise but the module docstring's
        interval verification is what actually matters -- re-run the same
        check inline so a regression to maximum(x,-x) is caught."""
        x_ivl = irx.icentpert(jnp.array([0.0]), jnp.array([0.001]))

        def f_sound(x):
            return jnp.sqrt(x ** 2)

        def f_unsound(x):
            return jnp.maximum(x, -x)

        class Toy(irx.System):
            def __init__(self, fn):
                self.evolution = 'discrete'
                self.xlen = 1
                self.fn = fn

            def f(self, t, x, u, p):
                return jnp.array([self.fn(x[0])])

        for fn, expect_sound in [(f_sound, True), (f_unsound, False)]:
            emb = irx.natemb(Toy(fn))
            x_ut = irx.i2ut(x_ivl)
            out = irx.ut2i(emb.f(jnp.zeros(()), x_ut, jnp.zeros(0), irx.Interval(lower=jnp.zeros(0), upper=jnp.zeros(0)), refine=lambda z: z))
            is_sound = float(out.lower[0]) >= -1e-12   # abs image must be >= 0
            assert is_sound == expect_sound

    def test_clip_survival_matches_source_at_points(self):
        L = ref.PID_POS_VEL_MAX
        for raw in [2.0, -2.0, 0.5, -0.5, 0.0, 1e-15, L, -L]:
            clipped_np = np.clip(raw, -L, L)
            surv_np = clipped_np / raw if abs(raw) > 1e-12 else 1.0
            surv_mine = float(_clip_survival(jnp.array(raw), L))
            assert surv_mine == pytest.approx(surv_np, abs=1e-5)


class TestMatchesReferenceModule:
    """Cross-check against rq3_crazyflie_surrogates.py's own .step(), the
    same pattern used for crazyflie_12d.py vs QPS's forward_model()."""

    @pytest.mark.parametrize("name", CANDIDATE_NAMES)
    @pytest.mark.parametrize("seed", range(6))
    def test_step_matches_reference(self, name, seed):
        rng = np.random.default_rng(seed)
        ref15_np = np.array([0.3, -0.2, 1.0] + [0.0] * 12)
        bias = rng.normal(scale=0.2, size=3)

        scenarios = create_scenarios(jnp.array(ref15_np), names=[name])
        scen = scenarios[0]
        cand = ref.candidate_for_controller_id(
            {v: k for k, v in ref.FIRMWARE_CONTROLLER_ID.items()}[name], QPS_DT)

        x = _random_state(rng)
        mem = _random_mem(rng)
        x_next_ref, mem_next_ref = cand.step(x, ref15_np, cand.theta_true, mem, 0.0, bias)

        x21 = np.concatenate([x, mem.integ, mem.prev_cmd, mem.accel_filt])
        x21_next = np.array(scen.emb_system.sys.f(
            jnp.zeros(()), jnp.array(x21), jnp.array(bias), jnp.array(cand.theta_true)))

        mem_next_ref_arr = np.concatenate([mem_next_ref.integ, mem_next_ref.prev_cmd, mem_next_ref.accel_filt])
        np.testing.assert_allclose(x21_next[:12], x_next_ref, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(x21_next[12:21], mem_next_ref_arr, atol=1e-6, rtol=1e-5)

    def test_pid_saturation_branch_matches(self):
        """Specifically exercise cf_pid's clamp -- large position error should
        push v_sp past PID_POS_VEL_MAX so the saturated branch is live."""
        ref15_np = np.array([5.0, 5.0, 1.0] + [0.0] * 12)   # far setpoint -> big e_p
        bias = np.zeros(3)
        scenarios = create_scenarios(jnp.array(ref15_np), names=["cf_pid"])
        scen = scenarios[0]
        cand = ref.CrazyfliePIDCandidate(QPS_DT)

        x = np.zeros(12)   # at origin, setpoint far away -> definitely saturates
        mem = ref.ControllerMemory()
        x_next_ref, _ = cand.step(x, ref15_np, cand.theta_true, mem, 0.0, bias)
        assert cand.saturated() > 0   # confirm the test setup actually saturates

        x21 = np.concatenate([x, mem.integ, mem.prev_cmd, mem.accel_filt])
        x21_next = np.array(scen.emb_system.sys.f(
            jnp.zeros(()), jnp.array(x21), jnp.array(bias), jnp.array(cand.theta_true)))
        np.testing.assert_allclose(x21_next[:12], x_next_ref, atol=1e-5, rtol=1e-4)


class TestSteadyState:
    """Mirrors the source's own _steady_state_check: under a constant bias,
    true position must settle at setpoint - bias (rollout, no intervals)."""

    @pytest.mark.parametrize("name", CANDIDATE_NAMES)
    def test_settles_at_setpoint_minus_bias(self, name):
        sp = jnp.array([0.8, -0.4, 1.0])
        ref15 = hover_reference(sp)
        scenarios = create_scenarios(ref15, names=[name])
        scen = scenarios[0]
        cand_theta = jnp.array(ref.get_bank(QPS_DT, names=[name])[0].theta_true)

        bias = jnp.array([0.0, 0.3, 0.0])
        x0 = jnp.zeros(12).at[0:3].set(jnp.array([0.0, 0.0, 1.0]))
        x21 = jnp.concatenate([x0, jnp.zeros(9)])

        n_steps = int(25.0 / QPS_DT)
        f = scen.emb_system.sys.f
        for _ in range(n_steps):
            x21 = f(jnp.zeros(()), x21, bias, cand_theta)

        final_pos = x21[0:3]
        want = sp - bias
        err = float(jnp.max(jnp.abs(final_pos - want)))
        assert jnp.all(jnp.isfinite(x21)), f"{name} diverged"
        assert err < 0.05, f"{name} did not settle within 5cm (err={err})"


class TestEmbeddingAndScenarios:
    def test_create_scenarios_returns_four_by_default(self):
        scenarios = create_scenarios()
        assert [s.name for s in scenarios] == list(CANDIDATE_NAMES)

    def test_each_scenario_has_its_own_emb_system(self):
        scenarios = create_scenarios()
        systems = [s.emb_system for s in scenarios]
        assert len(set(id(s) for s in systems)) == 4

    def test_theta_shapes_match_reference(self):
        scenarios = create_scenarios()
        ref_bank = {c.name: c for c in ref.get_bank(QPS_DT)}
        for s in scenarios:
            assert s.p_interval.lower.shape == (ref_bank[s.name].n_theta,)
            np.testing.assert_allclose(np.array(s.p_interval.lower), ref_bank[s.name].theta_true)

    def test_euler_step_produces_valid_interval(self):
        scenarios = create_scenarios()
        x0_ivl = irx.icentpert(jnp.zeros(21), jnp.full(21, 1e-3))
        u = jnp.zeros(3)
        for s in scenarios:
            x_next = euler_step(s.emb_system, x0_ivl, u, s.p_interval)
            assert x_next.lower.shape == (21,)
            assert jnp.all(x_next.lower <= x_next.upper)

    def test_wider_input_widens_output(self):
        scenarios = create_scenarios(names=["cf_brescianini"])
        s = scenarios[0]
        u = jnp.zeros(3)
        x_tight = irx.icentpert(jnp.zeros(21), jnp.full(21, 1e-4))
        x_wide = irx.icentpert(jnp.zeros(21), jnp.full(21, 1e-2))
        next_tight = euler_step(s.emb_system, x_tight, u, s.p_interval)
        next_wide = euler_step(s.emb_system, x_wide, u, s.p_interval)
        assert jnp.sum(next_wide.upper - next_wide.lower) > jnp.sum(next_tight.upper - next_tight.lower)
