import unittest
import math
import jax
import jax.numpy as jnp
import immrax as irx
from interval_functions import overlap_size, overlap_size_lax, overlap_size_log

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


class TestOverlapSizeLog(unittest.TestCase):
    """Tests for overlap_size_log: returns sum(ln(width_i + 1)) over intersection dims,
    or 0.0 if the intersection is empty."""

    # ── helper ────────────────────────────────────────────────────────────────
    def _ivl(self, lo, hi):
        return irx.interval(jnp.array(lo, dtype=float), jnp.array(hi, dtype=float))

    def assertClose(self, result, expected, places=5):
        self.assertAlmostEqual(float(result), expected, places=places)

    # ── no-overlap cases → 0.0 ────────────────────────────────────────────────
    def test_no_overlap_1d(self):
        result = overlap_size_log(self._ivl([0.], [1.]), self._ivl([2.], [3.]))
        self.assertClose(result, 0.0)

    def test_no_overlap_2d(self):
        result = overlap_size_log(
            self._ivl([0., 1.], [2., 3.]),
            self._ivl([4., 5.], [6., 7.]),
        )
        self.assertClose(result, 0.0)

    def test_no_overlap_partial_dim(self):
        # Overlaps in dim 0 but not dim 1 → empty intersection → 0.0
        result = overlap_size_log(
            self._ivl([0., 0.], [2., 1.]),
            self._ivl([1., 3.], [3., 5.]),
        )
        self.assertClose(result, 0.0)

    # ── 1-D overlap cases ─────────────────────────────────────────────────────
    def test_partial_overlap_1d(self):
        # Intersection [2, 3], width=1 → ln(2)
        result = overlap_size_log(self._ivl([0.], [3.]), self._ivl([2.], [4.]))
        self.assertClose(result, math.log(2.0))

    def test_full_containment_1d(self):
        # Intersection [2, 3], width=1 → ln(2)
        result = overlap_size_log(self._ivl([1.], [4.]), self._ivl([2.], [3.]))
        self.assertClose(result, math.log(2.0))

    def test_identical_intervals_1d(self):
        # Intersection [2, 5], width=3 → ln(4)
        result = overlap_size_log(self._ivl([2.], [5.]), self._ivl([2.], [5.]))
        self.assertClose(result, math.log(4.0))

    def test_negative_intervals_1d(self):
        # Intersection [-4, -3], width=1 → ln(2)
        result = overlap_size_log(self._ivl([-5.], [-2.]), self._ivl([-4.], [-3.]))
        self.assertClose(result, math.log(2.0))

    def test_zero_width_intersection_1d(self):
        # Intervals touch at x=1 → width=0 → ln(0+1)=0
        result = overlap_size_log(self._ivl([0.], [1.]), self._ivl([1.], [2.]))
        self.assertClose(result, 0.0)

    def test_large_width_1d(self):
        # Intersection [0, 9], width=9 → ln(10)
        result = overlap_size_log(self._ivl([0.], [10.]), self._ivl([0.], [9.]))
        self.assertClose(result, math.log(10.0))

    # ── 2-D overlap cases ─────────────────────────────────────────────────────
    def test_partial_overlap_2d(self):
        # Intersection [1,2]×[3,4], widths=[1,1] → 2·ln(2)
        result = overlap_size_log(
            self._ivl([0., 3.], [2., 4.]),
            self._ivl([1., 3.], [3., 6.]),
        )
        self.assertClose(result, 2.0 * math.log(2.0))

    def test_full_containment_2d(self):
        # Intersection [1,4]×[1,4], widths=[3,3] → 2·ln(4)
        result = overlap_size_log(
            self._ivl([0., 0.], [5., 5.]),
            self._ivl([1., 1.], [4., 4.]),
        )
        self.assertClose(result, 2.0 * math.log(4.0))

    def test_identical_intervals_2d(self):
        # Intersection [2,3]×[5,6], widths=[1,1] → 2·ln(2)
        result = overlap_size_log(
            self._ivl([2., 5.], [3., 6.]),
            self._ivl([2., 5.], [3., 6.]),
        )
        self.assertClose(result, 2.0 * math.log(2.0))

    def test_mixed_widths_2d(self):
        # Intersection [1,2]×[0,3], widths=[1,3] → ln(2)+ln(4)
        result = overlap_size_log(
            self._ivl([0., 0.], [2., 3.]),
            self._ivl([1., 0.], [3., 3.]),
        )
        self.assertClose(result, math.log(2.0) + math.log(4.0))

    # ── commutativity ─────────────────────────────────────────────────────────
    def test_commutativity_1d(self):
        a = self._ivl([0.], [3.])
        b = self._ivl([2.], [4.])
        self.assertClose(overlap_size_log(a, b), float(overlap_size_log(b, a)))

    def test_commutativity_2d(self):
        a = self._ivl([0., 1.], [2., 3.])
        b = self._ivl([1., 2.], [3., 4.])
        self.assertClose(overlap_size_log(a, b), float(overlap_size_log(b, a)))

    # ── JAX-traceability ──────────────────────────────────────────────────────
    def test_jit_compilable(self):
        fn = jax.jit(overlap_size_log)
        result = fn(self._ivl([0.], [3.]), self._ivl([2.], [4.]))
        self.assertClose(result, math.log(2.0))

    def test_grad_compilable(self):
        # gradient w.r.t. interval1.upper should be non-zero when there is overlap
        def loss(upper1):
            iv1 = irx.interval(jnp.array([0.]), upper1)
            iv2 = irx.interval(jnp.array([2.]), jnp.array([4.]))
            return overlap_size_log(iv1, iv2)
        g = jax.grad(loss)(jnp.array([3.]))
        self.assertFalse(jnp.any(jnp.isnan(g)))


if __name__ == "__main__":
    unittest.main()