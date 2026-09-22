import unittest

import numpy as np

import ba_camera_corridor as c


class GeometryTests(unittest.TestCase):
    def test_connectivity_is_four_not_diagonal(self):
        self.assertEqual(c.largest(np.eye(4, dtype=bool)), 1)

    def test_depth_ray_cone_is_not_fixed_central_rectangle(self):
        # Same ray is inside at1m and outside at3m at ten degrees.
        a = np.tan(np.deg2rad(10.))
        self.assertGreater(float(c.ray_limit(np.array(a), np.array(0.))), 1.)
        self.assertLess(float(c.ray_limit(np.array(a), np.array(0.))), 3.)
        limit = c.ray_limit(np.array([0., 0.]), np.array([-.2, .2]))
        np.testing.assert_allclose(limit, [1., 3.])

    def test_nfo_far_bin_no_singleton_contact_alert_and_near_is_ambiguous(self):
        p = np.zeros((4, c.LOW_H, c.LOW_W), np.float32)
        result, possible, _ = c.nfo_readout(p)
        self.assertFalse(result['alert']); self.assertFalse(possible.any())
        p[:] = 1.
        result, _, _ = c.nfo_readout(p)
        self.assertTrue(result['alert']); self.assertTrue(result['unknown'])
        self.assertTrue(result['ambiguous'])  # Unresolved0.3m lower limit.

    def test_raw_zone_angular_ambiguity_retained(self):
        result, anchors = c.tof_readout(np.array([[96, 128, 120, 160]]), np.array([2.]))
        self.assertTrue(result['alert']); self.assertTrue(result['ambiguous'])
        self.assertFalse(anchors[0]['definite'])
        result, _ = c.tof_readout(np.array([[96, 128, 120, 160]]), np.array([6.]))
        self.assertFalse(result['alert']); self.assertTrue(result['unknown'])

    def test_depth_positive_support_and_missing_do_not_become_clear(self):
        depth = np.full((c.LOW_H, c.LOW_W), 6., np.float32)
        result, _, _ = c.depth_readout(depth)
        self.assertFalse(result['alert']); self.assertEqual(result['state'], 'NO_SUPPORTED_HIT')
        depth[90:100, 124:132] = 2.
        result, _, _ = c.depth_readout(depth)
        self.assertTrue(result['alert']); self.assertFalse(result['ambiguous'])


if __name__ == '__main__':
    unittest.main()
