"""Synthetic invariants for spatial range assignment; no cohort truth loaded."""
import unittest

import numpy as np

from ba_nfo_depthpro_echo import predict, regions


class EchoTests(unittest.TestCase):
    def test_background_echo_does_not_move_small_front_and_crosses_zones(self):
        depth = np.full((6, 8), 4., np.float32)
        depth[2:4, 2:6] = 1.5
        boxes = [[0, 0, 6, 4], [0, 4, 6, 8]]
        # Eight near pixels are insufficient to dominate either zone's energy.
        depth[3, 2:6] = 4.
        result, receipt = predict(depth, boxes, [6., 6.])
        near = depth == 1.5
        self.assertEqual(len(np.unique(result['labels'][near])), 1)
        np.testing.assert_array_equal(result['layered_depth'][near], depth[near])
        np.testing.assert_allclose(result['layered_depth'][~near], 6.)
        np.testing.assert_allclose(result['global_depth'][near], 2.25)
        self.assertEqual(receipt['anchored_regions'], 1)

    def test_single_region_arms_equal_and_missing_returns_are_identity(self):
        depth = np.full((4, 8), 3., np.float32)
        boxes = [[0, 0, 4, 4], [0, 4, 4, 8]]
        result, _ = predict(depth, boxes, [1.5, 1.5])
        np.testing.assert_array_equal(result['global_depth'], result['layered_depth'])
        np.testing.assert_allclose(result['layered_depth'], 1.5)
        result, receipt = predict(depth, boxes, [None, float('nan')])
        np.testing.assert_array_equal(result['layered_depth'], depth)
        np.testing.assert_array_equal(result['global_depth'], depth)
        self.assertEqual(receipt['anchored_pixels'], 0)

    def test_no_gradual_chain_or_small_region_pruning(self):
        depth = np.tile(np.array([1., 1.2, 1.4, 1.6, 2., 4.], np.float32), (3, 1))
        depth[1, 5] = .5
        labels = regions(depth)
        for label in np.unique(labels):
            z = depth[labels == label]
            self.assertLessEqual(float(z.max()), 1.25*float(z.min()))
        self.assertEqual(int((labels == labels[1, 5]).sum()), 1)
        np.testing.assert_array_equal(labels, regions(depth))

    def test_histogram_peak_not_integrated_wall_energy(self):
        depth = np.repeat(np.r_[1.5, np.linspace(3.01, 3.69, 9)].astype(np.float32)[:, None], 10, axis=1)
        result, receipt = predict(depth, [[0, 0, 10, 10]], [1.2])
        self.assertEqual(receipt['anchors'][0]['predicted_bin'], 15)
        np.testing.assert_allclose(result['layered_depth'][0], 1.2)
        np.testing.assert_array_equal(result['layered_depth'][1:], depth[1:])
        # An existing return with no plausible predicted in-range support is
        # recorded as unused, without inventing a pixel assignment.
        far = np.full((3, 3), 10., np.float32)
        result, receipt = predict(far, [[0, 0, 3, 3]], [1.2])
        self.assertEqual(receipt['anchors'], [])
        np.testing.assert_array_equal(result['layered_depth'], far)


if __name__ == '__main__':
    unittest.main()
