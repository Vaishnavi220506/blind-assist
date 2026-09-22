import unittest
import numpy as np
from query_occupancy_data import geometric_labels, visible_labels, observation_tokens, QUERIES


class LabelsTest(unittest.TestCase):
    def test_full_extent_and_nearest_of_multiple_objects(self):
        # Centre outside the corridor; the object's full left edge enters it.
        a = (np.array([.29, .5, 1.3]), np.array([.8, .7, 1.5]))
        b = (np.array([-.1, .5, 2.3]), np.array([.1, .7, 2.5]))
        c, d = geometric_labels([a, b])
        self.assertAlmostEqual(d[1], 1.3, places=5)
        self.assertEqual(c[1], 2)
        self.assertEqual(c[4], 6)

    def test_boundary_distance_and_missing_visibility_are_distinct(self):
        c, d = geometric_labels([(np.array([-.1, -.1, 3.]), np.array([.1, .1, 3.2]))])
        self.assertEqual(c[4], 5)
        v = visible_labels(np.full((360, 640), np.nan), [])
        self.assertFalse(v['coverage'].any())
        self.assertEqual(sum(v['count']), 0)
        self.assertEqual(c[4], 5)  # Empty visible mask never changes geometric truth.

    def test_unknown_return_is_zero_with_invalid_flag(self):
        boxes = np.tile([0, 0, 192, 256], (64, 1))
        x = observation_tokens(np.full(64, np.nan), boxes)
        self.assertTrue(np.isfinite(x).all())
        self.assertFalse(x[:, :2].any())

    def test_unexplained_visible_surface_invalidates_negative(self):
        depth = np.full((360, 640), 2., np.float32)
        v = visible_labels(depth, [])
        self.assertGreater(v['unexplained'][4], 3)
        self.assertGreater(v['count'][4], 0)


if __name__ == '__main__':
    unittest.main()
