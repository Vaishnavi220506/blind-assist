import copy
import json
import unittest

import numpy as np

from mz144_corridor_margin import _depth_modes, _range_factor, _surface, extract_margin
from test_mz143_corridor_features import inputs, target


class MarginTests(unittest.TestCase):
    def test_physical_frontplane_recovers_from_inverse_square_regional_range(self):
        row, image = inputs(); intr = row['rgb_intrinsics']
        roi = [400., 110., 455., 160.]
        mask = np.ones(image.shape[:2], np.uint8)
        factor = _range_factor(row, roi, mask, 0.)
        yy, xx = np.mgrid[110:161, 400:456]
        rays = np.stack([np.ones(xx.shape), (xx-intr['cx'])/intr['fx'],
                         (intr['cy']-yy)/intr['fy']], -1)
        unit = rays/np.linalg.norm(rays, axis=-1)[..., None]
        depth = 2.5
        ranges = depth/unit[..., 0]
        weights = 1/ranges**2
        measured = float(np.sum(weights*ranges)/np.sum(weights))
        self.assertGreater(abs(measured-depth), .05)
        self.assertAlmostEqual(measured*factor['factor'], depth, places=10)

    def test_slot_alternation_counts_distinct_zones_only(self):
        def obs(zone, slot, d):
            return dict(zone=zone, slot=slot, depth=d, interval=[d-.1, d+.1],
                coarse_depth=d, signal_r2=.3, quadrature=dict(mask_used=True))
        self.assertEqual(_depth_modes([obs(1, 0, 2.), obs(1, 1, 2.)]), [])
        result = _depth_modes([obs(1, 0, 3.8), obs(1, 1, 2.), obs(9, 0, 2.03), obs(9, 1, 3.8)])
        self.assertGreaterEqual(len(result), 2)
        self.assertEqual(result[0]['returns'], [[1, 1], [9, 0]])
        self.assertEqual(result[0]['distinct_zones'], 2)
        self.assertLess(result[0]['depth_m'], 2.1)

    def test_plane_inside_forward_query_has_positive_margin(self):
        row, _ = inputs()
        inside = _surface(row, [[300, 120], [335, 120], [335, 250], [300, 250]], 2., 0.)
        outside = _surface(row, [[440, 120], [450, 120], [450, 250], [440, 250]], 2., 0.)
        self.assertGreater(inside['score'], 0.)
        self.assertLess(outside['score'], 0.)

    def test_missing_and_merged_only_are_unknown_with_raw_slots_retained(self):
        row, image = inputs(); row['tof_zones'][28]['targets'] = [target(status='SIM_MERGED')]
        result = extract_margin(row, image, 0.)
        self.assertTrue(result['unknown']); self.assertEqual(result['score'], -10.)
        self.assertEqual(result['audit']['public_slots'], 1)
        row['tof_packet_received'] = False
        absent = extract_margin(row, image, 0.)
        self.assertTrue(absent['unknown']); self.assertEqual(absent['audit']['public_slots'], 1)

    def test_metadata_invariance_nonmutation_and_second_slot_input(self):
        row, image = inputs()
        row['tof_zones'][20]['targets'] = [target(3.8), target(2.)]
        row['tof_zones'][28]['targets'] = [target(2.)]
        original = copy.deepcopy(row); pixels = image.copy()
        result = extract_margin(row, image, 0.)
        other = copy.deepcopy(row)
        other.update(id='secret_label', truth=True, camera=dict(yaw=90), family='head',
                     native_bounds=[dict(x=999)], pair_id='private', time_s=100)
        changed = extract_margin(other, image, 0.)
        self.assertEqual(result, changed)
        self.assertEqual(row, original); np.testing.assert_array_equal(image, pixels)
        self.assertFalse(result['unknown'])
        self.assertTrue(any([20, 1] in c.get('selected_returns', []) for c in result['candidates']))
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
