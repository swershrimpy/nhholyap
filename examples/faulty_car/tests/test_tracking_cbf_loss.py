"""
test_tracking_cbf_loss.py
=========================
Tests for tracking_cbf_loss() in faulty_car_output_feedback_cbf.py.

Each test verifies that the loss is > 0 when the controller does NOT drive the
refined output overlap to zero.  Four complementary checks are used:

  1. Zero-input controller — the car cannot move; all scenarios share the same
     trajectory so their output intervals overlap at every step.

  2. Known-failing controller (theta_track_fail.npz) — explicitly identified as
     failing to separate the scenario output intervals.

  3. Formerly-zero-loss controller (theta_track_opt.npz) — optimised under the
     buggy loss (Bug 3: theta_0 applied twice in step0_overlaps) and reported
     loss = 0.  With the fix the loss must be > 0.

  4. Oracle rollout consistency — roll out theta_track_opt.npz with the correct
     per-scenario embeddings and, at each step k, check whether the one-step
     lookahead from the FULL state interval intersection (propagated with
     theta_{k+1}) yields overlapping output intervals for any pair.  If the
     oracle detects overlap the loss must be > 0.  This is the ground-truth
     check: it mirrors plot_refinement_sec19.py and is independent of the loss
     implementation.

Run from nhholyap/examples/faulty_car/ with:
  JAX_PLATFORMS=cpu /home/user/immrax-venv/bin/python3 -m pytest tests/test_tracking_cbf_loss.py -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import immrax as irx

# ── path setup ────────────────────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).resolve().parent
_FC_DIR    = _TESTS_DIR.parent                        # nhholyap/examples/faulty_car
_ROOT_DIR  = _FC_DIR.parent.parent.parent             # output_feedback/
_FAULT_DIR = _ROOT_DIR / 'robotarium_python_simulator/rps/examples/fault_diagnosis'

for _p in (str(_FC_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from faulty_car_output_feedback_cbf import (
    tracking_cbf_loss,
    create_track_cl_scenarios,
    cl_euler_step,
)
from faulty_car_separating_input import _obs_interval
from interval_functions import overlap_size_lax


# ══════════════════════════════════════════════════════════════════════════════
# Oracle helper
# ══════════════════════════════════════════════════════════════════════════════

def oracle_has_refined_overlap(theta_seq, y_hat_seq, x0_ivl, cl_scenarios, dt):
    """Return True if any (pair, step) has overlapping one-step-ahead REFINED
    output intervals when the full unrefined state intervals are used as input.

    Algorithm (mirrors plot_refinement_sec19.py bottom-row panels):
      For each step k = 0 .. num_steps-2:
        1. Propagate each scenario from x0_ivl using its own embedding system,
           accumulating full state intervals x_ivl[k] at each step.
        2. For each scenario pair (ia, ib):
           a. Compute obs(x_ivl[k][ia]) and obs(x_ivl[k][ib]).
           b. Find their output intersection.
           c. If non-empty: map intersection back to state space for each
              scenario, propagate one step with theta_{k+1}, check whether the
              resulting output intervals overlap.
           d. Return True as soon as any overlap > 0 is found.
    Returns False only when every propagated refined pair is separated at every
    step, i.e. the controller genuinely separates all pairs.
    """
    n         = len(cl_scenarios)
    num_steps = theta_seq.shape[0]
    pairs     = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def pack(k):
        return jnp.concatenate([theta_seq[k, :4], y_hat_seq[k], theta_seq[k, 4:6]])

    # Accumulate full state intervals: x_history[k] = list of n state intervals
    # after propagating x0 through steps 0..k (using theta_0..theta_k).
    x_ivls   = [x0_ivl] * n
    x_history = []
    for k in range(num_steps):
        full_theta_k = pack(k)
        x_ivls = [
            cl_euler_step(
                cl_scenarios[si].emb_system,
                x_ivls[si],
                full_theta_k,
                cl_scenarios[si].p_interval,
                dt,
            )
            for si in range(n)
        ]
        x_history.append(x_ivls)

    # One-step lookahead refinement at each step k → propagate with theta_{k+1}
    for k in range(num_steps - 1):
        full_next = pack(k + 1)
        for ia, ib in pairs:
            xi = x_history[k][ia]
            xj = x_history[k][ib]

            obs_i = _obs_interval(xi, cl_scenarios[ia])
            obs_j = _obs_interval(xj, cl_scenarios[ib])

            y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
            y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
            if not bool(jnp.all(y_hi >= y_lo)):
                continue  # output intervals already separated at this step

            # Map intersection back to state space for each scenario
            si_s = float(cl_scenarios[ia].obs_scale[0])
            sj_s = float(cl_scenarios[ib].obs_scale[0])

            xi_ref = irx.Interval(
                lower=jnp.array([
                    (y_lo[0] - cl_scenarios[ia].obs_offset[0]) / si_s,
                    (y_lo[1] - cl_scenarios[ia].obs_offset[1]) / si_s,
                    xi.lower[2],
                ]),
                upper=jnp.array([
                    (y_hi[0] - cl_scenarios[ia].obs_offset[0]) / si_s,
                    (y_hi[1] - cl_scenarios[ia].obs_offset[1]) / si_s,
                    xi.upper[2],
                ]),
            )
            xj_ref = irx.Interval(
                lower=jnp.array([
                    (y_lo[0] - cl_scenarios[ib].obs_offset[0]) / sj_s,
                    (y_lo[1] - cl_scenarios[ib].obs_offset[1]) / sj_s,
                    xj.lower[2],
                ]),
                upper=jnp.array([
                    (y_hi[0] - cl_scenarios[ib].obs_offset[0]) / sj_s,
                    (y_hi[1] - cl_scenarios[ib].obs_offset[1]) / sj_s,
                    xj.upper[2],
                ]),
            )

            xn_i = cl_euler_step(
                cl_scenarios[ia].emb_system, xi_ref, full_next,
                cl_scenarios[ia].p_interval, dt,
            )
            xn_j = cl_euler_step(
                cl_scenarios[ib].emb_system, xj_ref, full_next,
                cl_scenarios[ib].p_interval, dt,
            )

            prop_overlap = float(overlap_size_lax(
                _obs_interval(xn_i, cl_scenarios[ia]),
                _obs_interval(xn_j, cl_scenarios[ib]),
            ))
            if prop_overlap > 0.0:
                return True  # found a (pair, step) that is not separated

    return False  # all pairs separated at every step — controller is genuinely good


# ══════════════════════════════════════════════════════════════════════════════
# Test class
# ══════════════════════════════════════════════════════════════════════════════

class TestTrackingCbfLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cl_scenarios = create_track_cl_scenarios(
            actuator_alpha_lo=0.0, actuator_alpha_hi=0.5
        )

        fail = np.load(_FAULT_DIR / 'theta_track_fail.npz')
        cls.fail_theta   = jnp.array(fail['theta_seq'])    # (5, 6)
        cls.fail_y_hat   = jnp.array(fail['y_hat_seq'])    # (5, 2)
        cls.fail_x0_ivl  = irx.icentpert(
            jnp.array(fail['x0_center']), jnp.array(fail['x0_pert'])
        )
        cls.fail_obstacles = jnp.array(fail['obstacles'])  # (1, 3)
        cls.dt             = float(fail['dt'])              # 1.0 s

        opt = np.load(_FAULT_DIR / 'theta_track_opt.npz')
        cls.opt_theta   = jnp.array(opt['theta_seq'])
        cls.opt_y_hat   = jnp.array(opt['y_hat_seq'])
        cls.opt_x0_ivl  = irx.icentpert(
            jnp.array(opt['x0_center']), jnp.array(opt['x0_pert'])
        )
        cls.opt_obstacles = jnp.array(opt['obstacles'])

    # ── convenience wrapper ──────────────────────────────────────────────────

    def _loss(self, theta_seq, y_hat_seq, x0_ivl,
              obstacles=None, cbf_weight=0.0, num_substeps=1):
        """Call tracking_cbf_loss and return a Python float."""
        if obstacles is None:
            obstacles = self.fail_obstacles
        return float(tracking_cbf_loss(
            theta_seq,
            x0_ivl       = x0_ivl,
            cl_scenarios = self.cl_scenarios,
            dt           = self.dt,
            y_hat_seq    = y_hat_seq,
            obstacles    = obstacles,
            cbf_weight   = cbf_weight,
            num_substeps = num_substeps,
        ))

    # ── Test 1: zero-input controller ────────────────────────────────────────

    def test_zero_input_loss_is_nonzero(self):
        """A zero-input controller keeps the car stationary.  All scenarios
        share the same (non-trivial) initial interval, so their observed output
        intervals overlap at every step.  The separation loss must be > 0.
        """
        num_steps  = self.fail_theta.shape[0]
        zero_theta = jnp.zeros((num_steps, 6))
        zero_y_hat = jnp.zeros((num_steps, 2))

        loss = self._loss(zero_theta, zero_y_hat, self.fail_x0_ivl)
        self.assertGreater(
            loss, 0.0,
            msg=f"Zero-input controller: expected loss > 0, got {loss:.6f}",
        )

    # ── Test 2: known-failing controller ────────────────────────────────────

    def test_fail_controller_loss_is_nonzero(self):
        """theta_track_fail.npz was flagged as failing to separate the output
        intervals.  The separation loss (cbf_weight=0 to isolate it) must be
        > 0."""
        loss = self._loss(
            self.fail_theta, self.fail_y_hat, self.fail_x0_ivl,
            obstacles=self.fail_obstacles, cbf_weight=0.0,
        )
        self.assertGreater(
            loss, 0.0,
            msg=(f"Fail controller: expected separation loss > 0, got {loss:.6f}. "
                 "tracking_cbf_loss may be returning 0 spuriously."),
        )

    # ── Test 3: formerly-zero-loss controller ───────────────────────────────

    def test_formerly_zero_loss_controller_is_now_nonzero(self):
        """theta_track_opt.npz was produced by the buggy loss (Bug 3: theta_0
        applied twice in step0_overlaps) and achieved loss = 0 by exploiting
        that double application.

        plot_refinement_sec19.py confirms that the one-step-ahead REFINED
        output intervals overlap at every step for every pair, so the fixed
        loss must be > 0.
        """
        loss = self._loss(
            self.opt_theta, self.opt_y_hat, self.opt_x0_ivl,
            obstacles=self.opt_obstacles, cbf_weight=0.0,
        )
        self.assertGreater(
            loss, 0.0,
            msg=(f"Formerly-zero controller: expected loss > 0 with the fix, "
                 f"got {loss:.6f}.  Bug 3 may have been reintroduced "
                 "(step0_overlaps using full_0 instead of the first scan theta)."),
        )

    # ── Test 4: oracle rollout consistency ──────────────────────────────────

    def test_oracle_rollout_consistency(self):
        """Ground-truth check: roll out theta_track_opt.npz independently of
        tracking_cbf_loss and detect refined output overlap directly.

        Step A  (oracle) – At each step k, take the full state intervals of
        both scenarios in a pair, intersect their observed outputs, map back to
        state space, and propagate one step forward with theta_{k+1}.  If the
        propagated output intervals overlap for any (pair, step) the oracle
        returns True.

        Step B  (loss) – Assert that tracking_cbf_loss > 0 whenever the oracle
        says True.  This ensures the function faithfully reflects the physical
        reality computed independently.

        Note: we also assert the oracle itself returns True for this controller,
        since plot_refinement_sec19.py confirms overlap at every step.  That
        assertion protects against a silent regression where theta_track_opt.npz
        is replaced by a genuinely separating controller.
        """
        oracle_overlap = oracle_has_refined_overlap(
            self.opt_theta, self.opt_y_hat, self.opt_x0_ivl,
            self.cl_scenarios, self.dt,
        )

        # Sanity: the oracle must find overlap (controller is not genuinely good)
        self.assertTrue(
            oracle_overlap,
            msg=("Oracle found NO refined output overlap for theta_track_opt.npz. "
                 "This controller was produced by the buggy loss and should NOT "
                 "achieve genuine refined separation.  Check that the correct "
                 "theta_track_opt.npz file is loaded and that oracle_has_refined_overlap "
                 "uses the correct (fixed) per-scenario embedding systems."),
        )

        # Consistency: loss must agree with oracle
        loss = self._loss(
            self.opt_theta, self.opt_y_hat, self.opt_x0_ivl,
            obstacles=self.opt_obstacles, cbf_weight=0.0,
        )
        self.assertGreater(
            loss, 0.0,
            msg=(f"Oracle detected refined output overlap but tracking_cbf_loss "
                 f"returned {loss:.6f}.  The loss function disagrees with the "
                 "independent rollout — Bug 3 or Bug 2 may have been reintroduced."),
        )


if __name__ == '__main__':
    unittest.main()
