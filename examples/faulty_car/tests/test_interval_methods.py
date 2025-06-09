import unittest
import jax.numpy as jnp
import immrax as irx
from interval_functions import overlap_size, overlap_size_lax

class TestOverlapSize(unittest.TestCase):
    def test_no_overlap_1d(self):
        interval1 = irx.interval(jnp.array([0]), jnp.array([1]))
        interval2 = irx.interval(jnp.array([2]), jnp.array([3]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 0)

    def test_partial_overlap_1d(self):
        interval1 = irx.interval(jnp.array([0]), jnp.array([3]))
        interval2 = irx.interval(jnp.array([2]), jnp.array([4]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 1)

    def test_full_overlap_1d(self):
        interval1 = irx.interval(jnp.array([1]), jnp.array([4]))
        interval2 = irx.interval(jnp.array([2]), jnp.array([3]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 1)

    def test_identical_intervals_1d(self):
        interval1 = irx.interval(jnp.array([2]), jnp.array([5]))
        interval2 = irx.interval(jnp.array([2]), jnp.array([5]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 3)

    def test_negative_intervals_1d(self):
        interval1 = irx.interval(jnp.array([-5]), jnp.array([-2]))
        interval2 = irx.interval(jnp.array([-4]), jnp.array([-3]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 1)

    def test_commutivity_1d(self):
        interval1 = irx.interval(jnp.array([0]), jnp.array([3]))
        interval2 = irx.interval(jnp.array([2]), jnp.array([4]))
        result1 = overlap_size(interval1, interval2)
        result2 = overlap_size(interval2, interval1)
        self.assertEqual(result1, result2)

    def test_commutivity_2d(self):
        interval1 = irx.interval(jnp.array([0, 1]), jnp.array([2, 3]))
        interval2 = irx.interval(jnp.array([1, 2]), jnp.array([3, 4]))
        result1 = overlap_size(interval1, interval2)
        result2 = overlap_size(interval2, interval1)
        self.assertEqual(result1, result2)

    def test_no_overlap_2d(self):
        interval1 = irx.interval(jnp.array([0, 1]), jnp.array([2, 3]))
        interval2 = irx.interval(jnp.array([4, 5]), jnp.array([6, 7]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 0)

    def test_partial_overlap_2d(self):
        interval1 = irx.interval(jnp.array([0, 3]), jnp.array([2, 4]))
        interval2 = irx.interval(jnp.array([1, 3]), jnp.array([3, 6]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 1)

    def test_full_overlap_2d(self):
        interval1 = irx.interval(jnp.array([0, 0]), jnp.array([5, 5]))
        interval2 = irx.interval(jnp.array([1, 1]), jnp.array([4, 4]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 9)

    def test_identical_intervals_2d(self):
        interval1 = irx.interval(jnp.array([2, 5]), jnp.array([3, 6]))
        interval2 = irx.interval(jnp.array([2, 5]), jnp.array([3, 6]))
        result = overlap_size(interval1, interval2)
        self.assertEqual(result, 1)

class TestOverlapSizeLax(unittest.TestCase):
    def test_no_overlap_1d_lax(self):
        interval1 = irx.interval(jnp.array([0.]), jnp.array([1.]))
        interval2 = irx.interval(jnp.array([2.]), jnp.array([3.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 0)

    def test_partial_overlap_1d_lax(self):
        interval1 = irx.interval(jnp.array([0.]), jnp.array([3.]))
        interval2 = irx.interval(jnp.array([2.]), jnp.array([4.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 1)

    def test_full_overlap_1d_lax(self):
        interval1 = irx.interval(jnp.array([1.]), jnp.array([4.]))
        interval2 = irx.interval(jnp.array([2.]), jnp.array([3.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 1)

    def test_identical_intervals_1d_lax(self):
        interval1 = irx.interval(jnp.array([2.]), jnp.array([5.]))
        interval2 = irx.interval(jnp.array([2.]), jnp.array([5.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 3)

    def test_negative_intervals_1d_lax(self):
        interval1 = irx.interval(jnp.array([-5.]), jnp.array([-2.]))
        interval2 = irx.interval(jnp.array([-4.]), jnp.array([-3.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 1)

    def test_commutivity_1d_lax(self):
        interval1 = irx.interval(jnp.array([0.]), jnp.array([3.]))
        interval2 = irx.interval(jnp.array([2.]), jnp.array([4.]))
        result1 = overlap_size_lax(interval1, interval2)
        result2 = overlap_size_lax(interval2, interval1)
        self.assertEqual(result1, result2)

    def test_commutivity_2d_lax(self):
        interval1 = irx.interval(jnp.array([0., 1.]), jnp.array([2., 3.]))
        interval2 = irx.interval(jnp.array([1., 2.]), jnp.array([3., 4.]))
        result1 = overlap_size_lax(interval1, interval2)
        result2 = overlap_size_lax(interval2, interval1)
        self.assertEqual(result1, result2)

    def test_no_overlap_2d_lax(self):
        interval1 = irx.interval(jnp.array([0., 1.]), jnp.array([2., 3.]))
        interval2 = irx.interval(jnp.array([4., 5.]), jnp.array([6., 7.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 0)

    def test_partial_overlap_2d_lax(self):
        interval1 = irx.interval(jnp.array([0., 3.]), jnp.array([2., 4.]))
        interval2 = irx.interval(jnp.array([1., 3.]), jnp.array([3., 6.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 1)

    def test_full_overlap_2d_lax(self):
        interval1 = irx.interval(jnp.array([0., 0.]), jnp.array([5., 5.]))
        interval2 = irx.interval(jnp.array([1., 1.]), jnp.array([4., 4.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 9)

    def test_identical_intervals_2d_lax(self):
        interval1 = irx.interval(jnp.array([2., 5.]), jnp.array([3., 6.]))
        interval2 = irx.interval(jnp.array([2., 5.]), jnp.array([3., 6.]))
        result = overlap_size_lax(interval1, interval2)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()