"""Synthetic geometry/proxy parity only; never opens the real 96-frame cohort."""
import json
import unittest

import numpy as np

from ba_camera_corridor import sample_indices, sample_native
from ba_nfo_data import sensor
from cross_zone_anchor_core import component_anchors, trace_sensor, zone_map
from tof_fov45_core import boxes45, geometry_description, simulate


class Fov45Tests(unittest.TestCase):
    def assert_traces_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            for key in ('pixel_indices', 'weights'):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ('zone_id', 'observed', 'distance_m', 'winner_bin', 'reason'):
                self.assertEqual(a[key], b[key])

    def test_old_boxes_exact_float32_and_float64_parity(self):
        for dtype in (np.float32, np.float64):
            depth = np.random.default_rng(71).uniform(.01, 9, (192, 256)).astype(dtype)
            depth.ravel()[::29], depth.ravel()[::37] = np.nan, np.inf
            depth.ravel()[::41] = -1
            identity = 'synthetic-old-parity'
            _, _, boxes, expected = sensor(depth, identity)
            values, traces = simulate(depth, identity, boxes)
            # Includes identical float32 NaN bit patterns, not approximate ranges.
            np.testing.assert_array_equal(values.view(np.uint32), expected.view(np.uint32))
            self.assert_traces_equal(traces, trace_sensor(depth, identity, boxes, expected))

    def test_candidate_geometry_64_nonoverlap_and_nearest_quantization(self):
        boxes, info = boxes45(), geometry_description()
        self.assertEqual(boxes.shape, (64, 4))
        self.assertEqual(boxes.dtype, np.int64)
        mapped = zone_map(boxes, (192, 256))
        self.assertEqual(set(np.unique(mapped)), set(range(64)) | {-1})
        self.assertEqual(int((mapped >= 0).sum()), info['footprint_sampled_pixels'])
        for axis in ('x', 'y'):
            ideal = np.array(info['ideal_lowres_boundaries'][axis])
            rounded = np.array(info['lowres_boundaries'][axis])
            np.testing.assert_array_equal(rounded, np.floor(ideal + .5).astype(np.int64))
            self.assertTrue((np.abs(rounded - ideal) <= .5).all())
            self.assertTrue((np.diff(rounded) > 0).all())
            total = 256 if axis == 'x' else 192
            np.testing.assert_array_equal(rounded + rounded[::-1], np.full(9, total))
        json.dumps(info, allow_nan=False)

    def test_quantized_fov_matches_geometric_edges_and_actual_ray_centers(self):
        info = geometry_description()
        focal = 640 / (2 * np.tan(np.deg2rad(50)))
        yy, xx = sample_indices()
        for axis, native_size, low_size, native_indices in (('x', 640, 256, xx), ('y', 360, 192, yy)):
            lo, hi = np.array(info['lowres_boundaries'][axis])[[0, -1]]
            edge_angles = np.rad2deg(np.arctan((np.array([lo, hi]) * native_size / low_size - native_size / 2) / focal))
            center_angles = np.rad2deg(np.arctan((native_indices[[lo, hi - 1]] + .5 - native_size / 2) / focal))
            self.assertAlmostEqual(info['quantized_edge_angles'][axis]['total_fov_deg'], float(np.diff(edge_angles)[0]))
            np.testing.assert_array_equal(info['actual_sample_center_angles'][axis]['first_last_deg'], center_angles)

    def test_same_sample_lattice_and_no_depth_mutation(self):
        native = (np.arange(360 * 640).reshape(360, 640) / (360 * 640) + 1).astype(np.float32)
        sampled = sample_native(native)
        before = sampled.copy()
        yy, xx = sample_indices()
        np.testing.assert_array_equal(sampled, native[yy[:, None], xx[None, :]])
        values, traces = simulate(sampled, 'unchanged-samples', boxes45())
        np.testing.assert_array_equal(sampled, before)
        self.assertEqual(values.dtype, np.float32)
        mapped = zone_map(boxes45(), sampled.shape).ravel()
        for record in traces:
            self.assertTrue(np.all(mapped[record['pixel_indices']] == record['zone_id']))

    def test_dropout_is_unobserved_with_no_hidden_contributors(self):
        depth = np.full((192, 256), 2., np.float32)
        values, traces = simulate(depth, 'dropout-check', boxes45())
        dropped = [t for t in traces if t['reason'] == 'SIMULATED_DROPOUT']
        self.assertGreater(len(dropped), 0)
        for record in dropped:
            self.assertFalse(record['observed'])
            self.assertIsNone(record['distance_m'])
            self.assertIsNone(record['winner_bin'])
            self.assertEqual(record['pixel_indices'].size, 0)
            self.assertEqual(record['weights'].size, 0)
            self.assertTrue(np.isnan(values[record['zone_id']]))

    def test_invalid_and_out_of_range_samples_do_not_create_anchors(self):
        for invalid in (np.nan, np.inf, -.1, 0., .099, 8., 9.):
            values, traces = simulate(np.full((192, 256), invalid), 'invalid', boxes45())
            self.assertTrue(np.isnan(values).all())
            self.assertTrue(all(t['reason'] == 'INSUFFICIENT_HITS' and t['pixel_indices'].size == 0 for t in traces))

    def test_mixed_winner_never_gets_pure_ownership(self):
        depth = np.full((192, 256), 2., np.float32)
        _, traces = simulate(depth, 'mixed-ownership', boxes45())
        chosen = next(t for t in traces if t['observed'])
        labels = np.zeros(depth.shape, np.int32)
        indices = chosen['pixel_indices']
        labels.ravel()[indices[:len(indices) // 2]] = 1
        labels.ravel()[indices[len(indices) // 2:]] = 2
        possible, pure = component_anchors(labels, [chosen], 3.)
        self.assertEqual(possible, {1: {chosen['zone_id']}, 2: {chosen['zone_id']}})
        self.assertEqual(pure, {})

    def test_exact_winning_bin_samples_exclude_other_range(self):
        depth = np.full((192, 256), 2., np.float32)
        y0, x0, y1, x1 = boxes45()[0]
        depth[y0, x0] = 6.
        _, traces = simulate(depth, 'contributor-bin', boxes45())
        self.assertTrue(traces[0]['observed'])
        self.assertEqual(traces[0]['winner_bin'], 20)
        self.assertNotIn(y0 * 256 + x0, traces[0]['pixel_indices'])
        self.assertEqual(traces[0]['pixel_indices'].size, (y1 - y0) * (x1 - x0) - 1)

    def test_box_validation(self):
        depth, boxes = np.full((192, 256), 2.), boxes45()
        with self.assertRaisesRegex(ValueError, '64 integer'):
            simulate(depth, 'validation', boxes.astype(float))
        boxes[1] = boxes[0]
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            simulate(depth, 'validation', boxes)


if __name__ == '__main__':
    unittest.main()
