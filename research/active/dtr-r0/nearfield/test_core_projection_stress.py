"""Focused projection and causality checks; no consumed outcomes used."""
import unittest
import numpy as np
from run_core_projection_stress import shifted_boxes, readout, max_silence
from tof_fov45_core import boxes45


class ProjectionStressTests(unittest.TestCase):
    def test_geometry_preserves_zone_size_order_vertical_edges_and_input(self):
        source = boxes45()
        original = source.copy()
        for shift in (-2, 0, 2):
            moved = shifted_boxes(source, shift)
            np.testing.assert_array_equal(moved[:, [0, 2]], source[:, [0, 2]])
            np.testing.assert_array_equal(moved[:, [1, 3]], source[:, [1, 3]] + shift)
            np.testing.assert_array_equal(moved[:, 2:] - moved[:, :2], source[:, 2:] - source[:, :2])
        np.testing.assert_array_equal(source, original)

    def test_invalid_projection_is_not_clipped(self):
        for shift in (-300, 300, .5):
            with self.assertRaises(ValueError):
                shifted_boxes(boxes45(), shift)

    def test_hold_never_renews_and_resets_on_gap(self):
        def scored(value):
            return dict(score=value, baseline=dict(definite_zones=0, alert=value > 0,
                possible_zones=int(value > 0), valid_zones=64, ambiguous=value > 0))
        strong, weak = scored(.8), scored(0.)
        first, unknown, prior = readout(strong, None, 0.)
        second, _, prior = readout(weak, prior, .2)
        third, _, _ = readout(weak, prior, .4)
        self.assertTrue(first['strong'] and second['hold'] and unknown)
        self.assertFalse(second['strong'] or third['hold'])
        self.assertFalse(readout(weak, (0., True), .4)[0]['hold'])
        self.assertFalse(readout(weak, None, 0.)[0]['hold'])

    def test_silence_counts_initial_internal_terminal_but_not_negative_frames(self):
        rows = [dict(truth=bool(t), flags={'a': bool(a)}) for t, a in
                [(0, 0), (1, 0), (1, 0), (1, 1), (1, 0), (0, 0), (0, 0)]]
        self.assertEqual(max_silence(rows, 'a'), 2)


if __name__ == '__main__':
    unittest.main()
