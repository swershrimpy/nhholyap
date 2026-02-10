"""
Unit tests for the separating_input_optimizer module.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax.numpy as jnp
import immrax as irx
from faulty_planar_multirotor import FaultyPlanarMultirotor
from separating_input_optimizer import (
    SeparatingInputOptimizer,
    overlap_size_lax,
    pairwise_overlap_sum,
    propagate_interval_euler,
    optimize_separating_input_multistart,
)


class TestOverlapFunctions(unittest.TestCase):
    """Test overlap computation functions."""

    def test_overlap_no_intersection(self):
        """Two non-overlapping intervals should have zero overlap."""
        i1 = irx.interval(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
        i2 = irx.interval(jnp.array([2.0, 2.0]), jnp.array([3.0, 3.0]))
        overlap = overlap_size_lax(i1, i2)
        self.assertAlmostEqual(float(overlap), 0.0)

    def test_overlap_full_containment(self):
        """One interval fully contained in another."""
        i1 = irx.interval(jnp.array([0.0, 0.0]), jnp.array([4.0, 4.0]))
        i2 = irx.interval(jnp.array([1.0, 1.0]), jnp.array([3.0, 3.0]))
        overlap = overlap_size_lax(i1, i2)
        # i2 volume = (3-1) * (3-1) = 4
        self.assertAlmostEqual(float(overlap), 4.0)

    def test_overlap_partial(self):
        """Partial overlap between intervals."""
        i1 = irx.interval(jnp.array([0.0, 0.0]), jnp.array([2.0, 2.0]))
        i2 = irx.interval(jnp.array([1.0, 1.0]), jnp.array([3.0, 3.0]))
        overlap = overlap_size_lax(i1, i2)
        # Intersection: [1,2] x [1,2] = 1 * 1 = 1
        self.assertAlmostEqual(float(overlap), 1.0)

    def test_overlap_identical(self):
        """Identical intervals should have overlap equal to their volume."""
        i1 = irx.interval(jnp.array([1.0, 2.0]), jnp.array([3.0, 5.0]))
        i2 = irx.interval(jnp.array([1.0, 2.0]), jnp.array([3.0, 5.0]))
        overlap = overlap_size_lax(i1, i2)
        # Volume = (3-1) * (5-2) = 6
        self.assertAlmostEqual(float(overlap), 6.0)

    def test_overlap_edge_touching(self):
        """Intervals touching at edge should have zero overlap."""
        i1 = irx.interval(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
        i2 = irx.interval(jnp.array([1.0, 0.0]), jnp.array([2.0, 1.0]))
        overlap = overlap_size_lax(i1, i2)
        self.assertAlmostEqual(float(overlap), 0.0)


class TestPairwiseOverlapSum(unittest.TestCase):
    """Test pairwise overlap sum computation."""

    def test_two_intervals(self):
        """Sum with two intervals."""
        i1 = irx.interval(jnp.array([0.0, 0.0]), jnp.array([2.0, 2.0]))
        i2 = irx.interval(jnp.array([1.0, 1.0]), jnp.array([3.0, 3.0]))
        total = pairwise_overlap_sum([i1, i2], overlap_size_lax)
        # Only one pair, overlap = 1
        self.assertAlmostEqual(float(total), 1.0)

    def test_three_intervals_no_overlap(self):
        """Three intervals with no overlaps."""
        i1 = irx.interval(jnp.array([0.0]), jnp.array([1.0]))
        i2 = irx.interval(jnp.array([2.0]), jnp.array([3.0]))
        i3 = irx.interval(jnp.array([4.0]), jnp.array([5.0]))
        total = pairwise_overlap_sum([i1, i2, i3], overlap_size_lax)
        self.assertAlmostEqual(float(total), 0.0)

    def test_three_intervals_with_overlaps(self):
        """Three intervals with some overlaps."""
        # 1D intervals for simplicity
        i1 = irx.interval(jnp.array([0.0]), jnp.array([3.0]))
        i2 = irx.interval(jnp.array([2.0]), jnp.array([5.0]))
        i3 = irx.interval(jnp.array([4.0]), jnp.array([6.0]))
        # i1-i2: [2,3] = 1
        # i1-i3: no overlap = 0
        # i2-i3: [4,5] = 1
        # Total = 2
        total = pairwise_overlap_sum([i1, i2, i3], overlap_size_lax)
        self.assertAlmostEqual(float(total), 2.0)

    def test_single_interval(self):
        """Single interval should give zero (no pairs)."""
        i1 = irx.interval(jnp.array([0.0]), jnp.array([1.0]))
        total = pairwise_overlap_sum([i1], overlap_size_lax)
        self.assertAlmostEqual(float(total), 0.0)

    def test_empty_list(self):
        """Empty list should give zero."""
        total = pairwise_overlap_sum([], overlap_size_lax)
        self.assertAlmostEqual(float(total), 0.0)


class TestIntervalPropagation(unittest.TestCase):
    """Test interval propagation function."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()
        self.emb = irx.natemb(self.sys)

    def test_propagation_increases_uncertainty(self):
        """Forward propagation should generally increase interval width."""
        x0 = irx.icentpert(
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        )
        u = irx.icentpert(jnp.array([9.81, 0.0]), jnp.zeros(2))
        w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))
        p = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))

        x1 = propagate_interval_euler(self.emb, x0, u, w, p, 0.1)

        # Width should be positive for all states
        widths = x1.upper - x1.lower
        for i in range(5):
            self.assertGreater(float(widths[i]), 0.0)

    def test_propagation_with_uncertain_parameter(self):
        """Uncertain parameter should lead to wider intervals."""
        x0 = irx.icentpert(
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        )
        u = irx.icentpert(jnp.array([9.81, 0.0]), jnp.zeros(2))
        w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))

        # Nominal
        p_nom = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
        x1_nom = propagate_interval_euler(self.emb, x0, u, w, p_nom, 0.1)

        # With uncertainty
        p_unc = irx.icentpert(jnp.array([0.7, 1.0]), jnp.array([0.3, 0.0]))
        x1_unc = propagate_interval_euler(self.emb, x0, u, w, p_unc, 0.1)

        # Uncertain case should have wider intervals
        width_nom = x1_nom.upper[3] - x1_nom.lower[3]  # vy
        width_unc = x1_unc.upper[3] - x1_unc.lower[3]
        self.assertGreater(float(width_unc), float(width_nom) - 1e-6)


class TestSeparatingInputOptimizer(unittest.TestCase):
    """Test the SeparatingInputOptimizer class."""

    def setUp(self):
        """Set up a simple test case with two fault scenarios."""
        self.sys = FaultyPlanarMultirotor()
        self.x0 = irx.icentpert(
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        )
        self.w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))

        # Two fault scenarios: nominal and 50% thrust loss
        self.p_nominal = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
        self.p_fault = irx.icentpert(jnp.array([0.5, 1.0]), jnp.zeros(2))

        self.optimizer = SeparatingInputOptimizer(
            system=self.sys,
            fault_parameters=[self.p_nominal, self.p_fault],
            x0_interval=self.x0,
            w_interval=self.w,
            dt=0.01,
            num_steps=10,
            state_slice=slice(0, 2),  # Only consider position
        )

    def test_optimizer_initialization(self):
        """Test that optimizer initializes correctly."""
        self.assertEqual(self.optimizer.num_steps, 10)
        self.assertEqual(self.optimizer.dt, 0.01)
        self.assertEqual(len(self.optimizer.fault_parameters), 2)

    def test_loss_function_shape(self):
        """Loss function should return a scalar."""
        u = jnp.array([9.81, 0.0])
        loss = self.optimizer._loss_fn(u)
        self.assertEqual(loss.shape, ())  # Scalar

    def test_loss_is_nonnegative(self):
        """Loss (overlap) should always be non-negative."""
        u = jnp.array([9.81, 0.0])
        loss = self.optimizer._loss_fn(u)
        self.assertGreaterEqual(float(loss), 0.0)

    def test_optimization_reduces_loss(self):
        """Optimization should reduce loss."""
        u_init = jnp.array([9.81, 0.0])
        initial_loss = self.optimizer._loss_fn(u_init)

        u_opt, final_loss = self.optimizer.optimize(
            u_initial=u_init,
            learning_rate=1e-1,
            num_iterations=50,
            verbose=False,
        )

        # Final loss should be less than or equal to initial
        self.assertLessEqual(float(final_loss), float(initial_loss) + 1e-6)

    def test_evaluate_returns_correct_types(self):
        """Evaluate should return loss and list of intervals."""
        u = jnp.array([9.81, 0.0])
        loss, intervals = self.optimizer.evaluate(u)

        self.assertIsInstance(loss, jnp.ndarray)
        self.assertEqual(loss.shape, ())
        self.assertIsInstance(intervals, list)
        self.assertEqual(len(intervals), 2)
        self.assertIsInstance(intervals[0], irx.Interval)

    def test_different_inputs_give_different_losses(self):
        """Different control inputs should generally give different losses."""
        u1 = jnp.array([9.81, 0.0])
        u2 = jnp.array([5.0, 0.5])

        loss1 = self.optimizer._loss_fn(u1)
        loss2 = self.optimizer._loss_fn(u2)

        # Losses should be different (with high probability)
        self.assertNotAlmostEqual(float(loss1), float(loss2), places=3)

    def test_gradient_has_correct_shape(self):
        """Gradient should have same shape as input."""
        u = jnp.array([9.81, 0.0])
        grad = self.optimizer._loss_grad_jitted(u)
        self.assertEqual(grad.shape, u.shape)


class TestMultistartOptimization(unittest.TestCase):
    """Test multi-start optimization function."""

    def setUp(self):
        self.sys = FaultyPlanarMultirotor()
        self.x0 = irx.icentpert(
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.05, 0.05, 0.05, 0.05, 0.05]),
        )
        self.w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))
        self.p_nominal = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
        self.p_fault = irx.icentpert(jnp.array([0.5, 1.0]), jnp.array([0.2, 0.0]))

    def test_multistart_returns_correct_types(self):
        """Multi-start should return control and loss."""
        u_opt, loss = optimize_separating_input_multistart(
            system=self.sys,
            fault_parameters=[self.p_nominal, self.p_fault],
            x0_interval=self.x0,
            w_interval=self.w,
            dt=0.01,
            num_steps=5,
            num_restarts=2,
            num_iterations=10,
            state_slice=slice(0, 2),
        )

        self.assertEqual(u_opt.shape, (2,))  # 2D control
        self.assertIsInstance(loss, (float, jnp.ndarray))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_multistart_finds_better_solution(self):
        """Multi-start should find as good or better solution than single start."""
        # Single start
        optimizer = SeparatingInputOptimizer(
            system=self.sys,
            fault_parameters=[self.p_nominal, self.p_fault],
            x0_interval=self.x0,
            w_interval=self.w,
            dt=0.01,
            num_steps=5,
            state_slice=slice(0, 2),
        )
        u_single, loss_single = optimizer.optimize(
            u_initial=jnp.ones(2),
            learning_rate=1e-1,
            num_iterations=20,
        )

        # Multi-start
        u_multi, loss_multi = optimize_separating_input_multistart(
            system=self.sys,
            fault_parameters=[self.p_nominal, self.p_fault],
            x0_interval=self.x0,
            w_interval=self.w,
            dt=0.01,
            num_steps=5,
            num_restarts=3,
            num_iterations=20,
            state_slice=slice(0, 2),
            random_key=42,
        )

        # Multi-start should find at least as good a solution
        self.assertLessEqual(float(loss_multi), float(loss_single) + 1e-3)


class TestThreeFaultScenarios(unittest.TestCase):
    """Test optimizer with three different fault scenarios."""

    def test_three_scenarios_optimization(self):
        """Test optimization with three fault parameter sets."""
        sys = FaultyPlanarMultirotor()
        x0 = irx.icentpert(
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.05, 0.05, 0.05, 0.05, 0.05]),
        )
        w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))

        # Three scenarios: nominal, 50% thrust, 80% thrust
        p_nominal = irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2))
        p_fault_50 = irx.icentpert(jnp.array([0.5, 1.0]), jnp.zeros(2))
        p_fault_80 = irx.icentpert(jnp.array([0.8, 1.0]), jnp.zeros(2))

        optimizer = SeparatingInputOptimizer(
            system=sys,
            fault_parameters=[p_nominal, p_fault_50, p_fault_80],
            x0_interval=x0,
            w_interval=w,
            dt=0.01,
            num_steps=10,
            state_slice=slice(0, 2),
        )

        u_opt, loss = optimizer.optimize(
            u_initial=jnp.array([10.0, 0.0]),
            learning_rate=1e-1,
            num_iterations=30,
        )

        # Should complete successfully
        self.assertEqual(u_opt.shape, (2,))
        self.assertGreaterEqual(float(loss), 0.0)

        # Should have three pairwise overlaps
        _, intervals = optimizer.evaluate(u_opt)
        self.assertEqual(len(intervals), 3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_propagation_steps(self):
        """Test with num_steps=0 (no propagation)."""
        sys = FaultyPlanarMultirotor()
        x0 = irx.icentpert(jnp.zeros(5), jnp.ones(5) * 0.1)
        w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))
        p1 = irx.icentpert(jnp.ones(2), jnp.zeros(2))
        p2 = irx.icentpert(jnp.array([0.5, 1.0]), jnp.zeros(2))

        # With 0 steps, final interval should equal initial
        optimizer = SeparatingInputOptimizer(
            system=sys,
            fault_parameters=[p1, p2],
            x0_interval=x0,
            w_interval=w,
            dt=0.01,
            num_steps=0,
            state_slice=slice(0, 2),
        )

        u = jnp.array([9.81, 0.0])
        loss, intervals = optimizer.evaluate(u)

        # Both scenarios should end at same initial interval
        # So overlap should be maximum (equal to initial interval size)
        self.assertGreater(float(loss), 0.0)

    def test_very_small_timestep(self):
        """Test with very small dt."""
        sys = FaultyPlanarMultirotor()
        x0 = irx.icentpert(jnp.zeros(5), jnp.ones(5) * 0.1)
        w = irx.icentpert(jnp.zeros(1), jnp.zeros(1))
        p1 = irx.icentpert(jnp.ones(2), jnp.zeros(2))
        p2 = irx.icentpert(jnp.array([0.5, 1.0]), jnp.zeros(2))

        optimizer = SeparatingInputOptimizer(
            system=sys,
            fault_parameters=[p1, p2],
            x0_interval=x0,
            w_interval=w,
            dt=1e-6,  # Very small
            num_steps=10,
            state_slice=slice(0, 2),
        )

        u_opt, loss = optimizer.optimize(num_iterations=10)
        self.assertIsNotNone(u_opt)
        self.assertGreaterEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
