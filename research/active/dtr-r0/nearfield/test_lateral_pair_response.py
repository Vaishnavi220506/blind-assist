"""Focused geometry checks for the evaluator-only footprint diagnostic."""
import unittest
import numpy as np
from audit_lateral_pair_response import ray_box_mask


class FootprintTests(unittest.TestCase):
    def test_zero_direction_inside_and_outside_slab(self):
        a = np.zeros((1, 1)); b = a.copy()
        self.assertTrue(ray_box_mask([-.1,-.1,1], [.1,.1,2], a, b)[0, 0])
        self.assertFalse(ray_box_mask([.1,-.1,1], [.2,.1,2], a, b)[0, 0])

    def test_side_face_hit_missed_by_front_face(self):
        a = np.array([[.15]]); b = np.zeros_like(a)
        # At front z=1, x=.15 is outside; at z=4/3 the ray enters the side.
        self.assertTrue(ray_box_mask([.2,-.1,1], [.4,.1,2], a, b)[0, 0])

    def test_behind_camera_and_mirror(self):
        a = np.array([[.2, -.2]]); b = np.zeros_like(a)
        self.assertFalse(ray_box_mask([-.4,-.1,-2], [.4,.1,-1], a, b).any())
        np.testing.assert_array_equal(ray_box_mask([.1,-.1,1], [.3,.1,2], a, b), [[True, False]])
        np.testing.assert_array_equal(ray_box_mask([-.3,-.1,1], [-.1,.1,2], a, b), [[False, True]])


if __name__ == '__main__':
    unittest.main()
