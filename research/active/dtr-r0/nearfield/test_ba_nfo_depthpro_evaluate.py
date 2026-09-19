import unittest

import numpy as np

import evaluate_ba_nfo_depthpro as e


class FrozenBoundaryTest(unittest.TestCase):
    def test_scale_invariance_direction_and_strict_ratio(self):
        depth = np.array([[1., 1.25, 4.], [1., 1.25, 4.]], np.float32)
        known = np.ones_like(depth, bool)
        edges = e.ratio_edges(depth, known)
        self.assertEqual(sum(int(x.sum()) for x in edges), 2)
        counts, _ = e.si_counts(depth*7., depth, known)
        self.assertEqual(e.boundary_metrics(counts)['f1'], 1.)
        reverse = np.array([[4., 4., 1.], [4., 4., 1.]], np.float32)
        counts, _ = e.si_counts(reverse, depth, known)
        self.assertEqual(e.boundary_metrics(counts)['f1'], 0.)

    def test_unknown_never_creates_boundary_and_cutoff_independence(self):
        truth = np.array([[1., np.nan, 4.], [1., np.nan, 4.]], np.float32)
        known = np.isfinite(truth)
        self.assertEqual(sum(int(x.sum()) for x in e.near_edges(truth < 2, known)), 0)
        self.assertEqual(sum(int(x.sum()) for x in e.ratio_edges(truth, known)), 0)
        truth = np.array([[1., 4.], [1., 4.]], np.float32)
        counts, crossing = e.si_counts(truth*10., truth, np.ones_like(truth, bool))
        self.assertEqual(e.boundary_metrics(counts)['f1'], 1.)
        self.assertEqual(e.near_crossing_metrics(crossing)['recall'], 1.)
        near_counts = e.boundary_counts(e.near_edges(truth*10. < 2, np.ones_like(truth, bool)),
                                       e.near_edges(truth < 2, np.ones_like(truth, bool)), 1)
        self.assertEqual(e.boundary_metrics(near_counts)['f1'], 0.)

    def test_one_pixel_tolerance_and_empty_denominators(self):
        known = np.ones((4, 7), bool)
        truth = np.zeros_like(known); truth[:, :2] = True
        shifted = np.zeros_like(known); shifted[:, :3] = True
        actual = e.boundary_counts(e.near_edges(shifted, known), e.near_edges(truth, known), 1)
        self.assertEqual(actual.tolist(), [4, 4, 4, 4])
        self.assertEqual(e.boundary_metrics(actual)['f1'], 1.)
        shifted[:, :4] = True
        actual = e.boundary_counts(e.near_edges(shifted, known), e.near_edges(truth, known), 1)
        self.assertEqual(e.boundary_metrics(actual)['f1'], 0.)
        empty = e.boundary_metrics([0, 0, 0, 0])
        self.assertIsNone(empty['f1'])
        self.assertEqual(e.boundary_metrics([0, 4, 0, 0])['f1'], 0.)


if __name__ == '__main__':
    unittest.main()
