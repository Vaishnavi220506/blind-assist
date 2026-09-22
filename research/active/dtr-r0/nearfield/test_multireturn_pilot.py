"""Synthetic pre-freeze contracts; never consume pilot labels or result files."""
import unittest
from unittest.mock import patch

import numpy as np

import multireturn_pilot_20260920 as pilot
from tof_corridor_calibration import support_score


class MultiReturnPilotTests(unittest.TestCase):
    def setUp(self):
        self.boxes = pilot.boxes45()

    def scene(self, secondary=2.05):
        depth = np.full((192, 256), np.nan, dtype=np.float32)
        for y0, x0, y1, x1 in self.boxes:
            depth[y0:y1, x0:x1] = 1.05
            depth[y0:y0 + 2, x0:x1] = secondary
        return depth

    def test_first_slot_exact_parity_including_dropout_and_lineage(self):
        for depth in (self.scene(), self.scene(1.45),
                      np.full((192, 256), np.nan, np.float32)):
            with self.subTest(secondary=float(depth[40, 90])):
                before = depth.copy()
                expected, original = pilot.simulate(depth, 'multireturn-unit', self.boxes)
                public, lineage = pilot.simulate_two(depth, 'multireturn-unit', self.boxes)
                np.testing.assert_array_equal(public['ranges'][:, 0], expected)
                np.testing.assert_array_equal(depth, before)
                for old, new in zip(original, lineage):
                    self.assertEqual(old['pixel_indices'].tolist(), new['indices'][0])
                    self.assertEqual(old['winner_bin'], new['bins'][0])
                missing = ~np.isfinite(expected)
                self.assertTrue(np.isnan(public['ranges'][missing, 1]).all())
                self.assertTrue((public['status'][missing] == 'SIM_NO_RETURN').all())
                if np.isfinite(depth).any():
                    self.assertTrue(missing.any(), 'Fixture must exercise actual packet dropout')

    def test_secondary_separation_is_applied_to_native_means(self):
        close, _ = pilot.simulate_two(self.scene(1.45), 'separation', self.boxes)
        far, lineage = pilot.simulate_two(self.scene(2.05), 'separation', self.boxes)
        self.assertFalse(np.isfinite(close['ranges'][:, 1]).any())
        self.assertTrue(np.isfinite(far['ranges'][:, 1]).any())
        depth = self.scene(2.05).ravel()
        for z, trace in enumerate(lineage):
            if np.isfinite(far['ranges'][z, 1]):
                first = depth[trace['indices'][0]].mean()
                second = depth[trace['indices'][1]].mean()
                self.assertGreaterEqual(abs(float(first) - float(second)), .6)
                self.assertEqual(far['status'][z, 1], 'SIM_VALID')
                self.assertAlmostEqual(float(far['sigma'][z, 1]), .01 + .02 * second, places=7)

    def test_secondary_needs_four_samples(self):
        depth = np.full((192, 256), np.nan, np.float32)
        for y0, x0, y1, x1 in self.boxes:
            depth[y0:y1, x0:x1] = 1.05
            depth[y0, x0:x0 + 3] = 2.05
        public, _ = pilot.simulate_two(depth, 'count-floor', self.boxes)
        self.assertFalse(np.isfinite(public['ranges'][:, 1]).any())

    def test_secondary_is_repeatable_and_earlier_zone_does_not_shift_rng(self):
        depth = self.scene()
        original, _ = pilot.simulate_two(depth, 'rng-independence', self.boxes)
        repeat, _ = pilot.simulate_two(depth, 'rng-independence', self.boxes)
        for key in original:
            np.testing.assert_array_equal(original[key], repeat[key])
        y0, x0, y1, x1 = self.boxes[0]
        depth[y0:y1, x0:x1] = 1.05  # Remove only this zone's second return.
        changed, _ = pilot.simulate_two(depth, 'rng-independence', self.boxes)
        for key in original:
            np.testing.assert_array_equal(original[key][1:], changed[key][1:])
        np.testing.assert_array_equal(original['ranges'][:, 0], changed['ranges'][:, 0])

    def test_missing_ranges_stay_unknown(self):
        ranges = np.full((64, 2), np.nan, np.float32)
        decisions, _ = pilot.readouts(self.boxes, ranges)
        for mode, decision in decisions.items():
            with self.subTest(mode=mode):
                self.assertFalse(decision['alert'])
                self.assertTrue(decision['unknown'])

    def test_known_inside_outside_and_closest_is_not_always_best(self):
        ranges = np.full((64, 2), np.nan, np.float32)
        ranges[35] = [5., 1.]
        decisions, _ = pilot.readouts(self.boxes, ranges)
        self.assertFalse(decisions['strongest']['alert'])
        self.assertTrue(decisions['closest_exported']['alert'])
        self.assertTrue(decisions['two_returns']['alert'])
        self.assertFalse(decisions['two_returns']['unknown'])
        ranges[35] = [.1, 1.]
        decisions, _ = pilot.readouts(self.boxes, ranges)
        self.assertFalse(decisions['closest_exported']['alert'])
        self.assertTrue(decisions['two_returns']['alert'])

    def test_disjoint_supports_do_not_fill_the_empty_gap(self):
        ranges = np.full((64, 2), np.nan, np.float32)
        ranges[35] = [.1, 5.]
        decisions, scored = pilot.readouts(self.boxes, ranges)
        self.assertEqual([s['score'] for s in scored], [0., 0.])
        self.assertFalse(decisions['two_returns']['alert'])
        self.assertTrue(decisions['two_returns']['unknown'])
        intervals = [s['anchors'][0]['interval_m'] for s in scored]
        self.assertLess(intervals[0][1], .3)
        self.assertGreater(intervals[1][0], 3.)
        # A merged envelope would fabricate positive corridor support.
        envelope = support_score((-.1, 0.), (0., .1), (intervals[0][0], intervals[1][1]))
        self.assertGreater(envelope['joint'], 0.)

    def test_readout_uses_observations_only_without_mutating_them(self):
        ranges = np.full((64, 2), np.nan, np.float32)
        ranges[35] = [5., 1.]
        before = ranges.copy()
        expected, _ = pilot.readouts(self.boxes, ranges)
        with (patch.object(pilot, 'read', side_effect=AssertionError('No file reads')),
              patch.object(pilot, 'simulate_two', side_effect=AssertionError('No native depth'))):
            actual, _ = pilot.readouts(self.boxes, ranges)
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(ranges, before)


if __name__ == '__main__':
    unittest.main()
