import unittest
from unittest.mock import patch
import numpy as np
from regional_depth_data import sample_geometry, build_observation, depth_labels
from tof_fov45_core import boxes45
from ba_camera_corridor import sample_indices


class RegionalDepthDataTest(unittest.TestCase):
    def test_lattice_points_stay_in_each_zone(self):
        boxes = boxes45()
        yy, xx, a, b = sample_geometry(boxes)
        source_y, source_x = sample_indices()
        self.assertEqual(yy.shape, (64, 64))
        self.assertEqual(xx.dtype, np.int64)
        for zi, (y0, x0, y1, x1) in enumerate(boxes):
            row, col = divmod(zi, 8)
            sy, sx = slice(row * 8, (row + 1) * 8), slice(col * 8, (col + 1) * 8)
            self.assertTrue(np.isin(yy[sy, sx], source_y[y0:y1]).all())
            self.assertTrue(np.isin(xx[sy, sx], source_x[x0:x1]).all())
        focal = 640 / (2 * np.tan(np.deg2rad(50)))
        np.testing.assert_allclose(a, (xx + .5 - 320) / focal, atol=2e-8)
        np.testing.assert_allclose(b, (yy + .5 - 180) / focal, atol=2e-8)

    def test_exact_rgb_depth_and_channel_alignment(self):
        boxes = boxes45()
        yy, xx, a, b = sample_geometry(boxes)
        full_y, full_x = np.mgrid[:360, :640]
        rgb = np.stack([full_y % 256, full_x % 256, (full_x + full_y) % 256], -1).astype(np.uint8)
        native = (1 + full_x * .001 + full_y * .002).astype(np.float32)
        ranges = np.linspace(.1, 7.9, 64).astype(np.float32)
        observation = build_observation(rgb, ranges, boxes)
        labels = depth_labels(native, boxes)
        np.testing.assert_array_equal(observation[:3], rgb[yy, xx].transpose(2, 0, 1).astype(np.float32) / 255)
        np.testing.assert_array_equal(labels['depth'], native[yy, xx])
        np.testing.assert_array_equal(observation[5], a)
        np.testing.assert_array_equal(observation[6], b)
        self.assertEqual(observation.dtype, np.float32)
        for row, col in [(0, 0), (7, 8), (31, 47), (63, 63)]:
            self.assertEqual(observation[3, row, col], ranges[(row // 8) * 8 + col // 8] / np.float32(8))
        d = labels['depth']
        expected = (d >= .3) & (d <= 3) & (abs(a * d) <= .3) & (b * d >= -.2) & (b * d <= .9)
        np.testing.assert_array_equal(labels['occupancy'], expected)

    def test_missing_far_bins_and_no_truth_in_observation(self):
        boxes = boxes45()
        yy, xx, _, _ = sample_geometry(boxes)
        native = np.full((360, 640), 1.25, np.float32)
        samples = [np.nan, np.inf, -1., 0., .05, 7.99, 8., 99.]
        for col, value in enumerate(samples):
            native[yy[0, col], xx[0, col]] = value
        labels = depth_labels(native, boxes)
        np.testing.assert_array_equal(labels['class_index'][0, :8], [-100, -100, -100, -100, 0, 79, 80, 80])
        self.assertFalse(labels['valid'][0, :4].any())
        self.assertTrue((labels['depth'][0, :4] == 0).all())
        self.assertFalse(labels['occupancy'][0, :8].any())
        ranges = np.full(64, np.nan, np.float32)
        ranges[:4] = [np.inf, 0., 8., .1]
        rgb = np.zeros((360, 640, 3), np.uint8)
        with patch('regional_depth_data.depth_labels', side_effect=AssertionError('Evaluator API called')):
            observation = build_observation(rgb, ranges, boxes)
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue((observation[3:5, :8, :24] == 0).all())
        self.assertTrue((observation[4, :8, 24:32] == 1).all())
        original = observation.copy()
        native[:] = np.nan
        depth_labels(native, boxes)
        np.testing.assert_array_equal(build_observation(rgb, ranges, boxes), original)

    def test_reject_bad_geometry_and_shapes(self):
        with self.assertRaises(ValueError):
            sample_geometry(boxes45().astype(float))
        bad = boxes45().copy()
        bad[1, 1] += 1
        with self.assertRaises(ValueError):
            sample_geometry(bad)
        with self.assertRaises(ValueError):
            depth_labels(np.zeros((192, 256), np.float32), boxes45())
        with self.assertRaises(ValueError):
            build_observation(np.zeros((360, 640, 3), np.float32), np.zeros(64), boxes45())


if __name__ == '__main__':
    unittest.main()
