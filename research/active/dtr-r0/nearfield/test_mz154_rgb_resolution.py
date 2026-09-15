import copy
import unittest

import cv2
import numpy as np

import mz153_temporal_depth as old
from mz154_rgb_resolution import make_inputs, full_rgb_depth


def observation():
    return dict(id='synthetic', tof_packet_received=True, rgb_intrinsics=dict(
        width=640, height=360, fx=457., fy=457., cx=320., cy=180.),
        tof_zones=[dict(zone_id=r*8+c, theta_bounds_deg=[-22.5+c*5.625, -22.5+(c+1)*5.625],
                        phi_bounds_deg=[22.5-(r+1)*5.625, 22.5-r*5.625], targets=[])
                   for r in range(8) for c in range(8)])


class ResolutionAdapterTests(unittest.TestCase):
    def test_original_depth_slots_and_exact_upsampling(self):
        row = observation()
        row['tof_zones'][10]['targets'] = [
            dict(status='SIM_VALID', distance_m=2., signal_strength_proxy=1.),
            dict(status='SIM_VALID', distance_m=3., signal_strength_proxy=5.)]
        row['tof_zones'][11]['targets'] = [
            dict(status='SIM_MERGED', distance_m=1., signal_strength_proxy=10.)]
        image = np.random.default_rng(154016).integers(0, 256, (360, 640, 3), dtype=np.uint8)
        old_rgb, old_depth, old_audit = old.make_input(row, image)
        direct, enlarged, depth, audit = make_inputs(row, image)
        np.testing.assert_array_equal(depth, old_depth)
        np.testing.assert_array_equal(enlarged, cv2.resize(
            old_rgb.transpose(1, 2, 0), (256, 256), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))
        self.assertEqual(audit['mz153']['used'], old_audit['used'])
        self.assertEqual(audit['mz153']['used'][0]['slot'], 1)
        self.assertEqual(np.count_nonzero(depth), 1)
        self.assertEqual(direct.shape, (3, 256, 256))
        self.assertEqual(direct.dtype, np.float32)
        self.assertGreater(np.max(np.abs(direct-enlarged)), .05)
        changed = copy.deepcopy(row)
        changed.update(id='different', family='unread', native_geometry='unread', label=1)
        for first, second in zip((direct, enlarged, depth), make_inputs(changed, image)[:3]):
            np.testing.assert_array_equal(first, second)

    def test_direct_projection_and_old_shared_mask(self):
        row = observation()
        image = np.full((360, 640, 3), [0, 128, 255], np.uint8)
        direct, _, _, _ = make_inputs(row, image)
        np.testing.assert_array_equal(direct[:, 128, 128], np.array([1., 128/255., 0.], np.float32))
        self.assertTrue(np.all(direct[:, 0] == 0))
        result, shared = full_rgb_depth(row, np.full((256, 256), .5, np.float32))
        old_result, old_shared = old.full_rgb_depth(row, np.full((128, 128), .5, np.float32))
        np.testing.assert_array_equal(shared, old_shared)
        np.testing.assert_array_equal(np.isfinite(result), np.isfinite(old_result))
        self.assertTrue(np.allclose(result[shared], 2.))
        self.assertTrue(np.isnan(result[~shared]).all())

    def test_unknown_positive_weight_contributor_is_not_interpolated_away(self):
        for missing in (0., -1., float('nan'), float('inf')):
            raw = np.full((256, 256), .5, np.float32)
            raw[128, 128] = missing
            result, shared = full_rgb_depth(observation(), raw)
            self.assertTrue(shared[180, 320])
            self.assertTrue(np.isnan(result[180, 320]))


if __name__ == '__main__':
    unittest.main()
