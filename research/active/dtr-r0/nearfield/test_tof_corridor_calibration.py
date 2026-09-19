"""Synthetic checks for geometry, threshold retention, and UNKNOWN semantics."""
import unittest

import numpy as np

from tof_corridor_calibration import support_score, score_frame, decide, calibrate
from tof_fov45_core import boxes45


class CalibrationTest(unittest.TestCase):
    def test_supported_and_depth_only(self):
        self.assertAlmostEqual(support_score((-.01, .01), (-.01, .01), (1., 2.))['joint'], 1.)
        self.assertAlmostEqual(support_score((-.01, .01), (-.01, .01), (2., 4.))['joint'], .5)

    def test_outside_and_exact_contact(self):
        self.assertEqual(support_score((1., 2.), (0., .01), (1., 2.))['joint'], 0.)
        self.assertEqual(support_score((0., .01), (0., .01), (3., 4.))['joint'], 0.)

    def test_closed_form_matches_independent_dense_integral(self):
        # Different branch crossings on both axes, plus truncated depth.
        for aa, bb, interval in [((-.2, .4), (-.3, .5), (.2, 3.4)),
                                  ((.08, .19), (-.12, -.03), (1.2, 3.8)),
                                  ((-.4, -.1), (.2, .6), (.4, 2.8))]:
            lo, hi = interval
            z = lo + (np.arange(200000) + .5) * (hi - lo) / 200000
            fx = np.maximum(0, np.minimum(aa[1], .3 / z) - np.maximum(aa[0], -.3 / z)) / (aa[1] - aa[0])
            fy = np.maximum(0, np.minimum(bb[1], .9 / z) - np.maximum(bb[0], -.2 / z)) / (bb[1] - bb[0])
            expected = np.mean(fx * fy * (z >= .3) * (z <= 3))
            self.assertAlmostEqual(support_score(aa, bb, interval)['joint'], expected, places=5)

    def test_missing_never_clear_or_alert(self):
        scored = score_frame(boxes45(), np.full(64, np.nan))
        for cutoff in (0., 1.):
            decision = decide(scored, cutoff)
            self.assertFalse(decision['alert'])
            self.assertTrue(decision['unknown'])

    def test_definite_support_bypass_and_anchors_unchanged(self):
        values = np.full(64, np.nan)
        values[36] = 1.
        scored = score_frame(boxes45(), values)
        anchors = repr(scored['anchors'])
        self.assertGreater(scored['baseline']['definite_zones'], 0)
        self.assertTrue(decide(scored, 1.)['alert'])
        self.assertEqual(repr(scored['anchors']), anchors)

    def test_maximal_retention_threshold_and_ties(self):
        def row(name, score, definite=0):
            return dict(id=name, score=score, baseline=dict(alert=True, definite_zones=definite,
                        possible_zones=1, valid_zones=1, ambiguous=not bool(definite)))
        rows = [row('positive', .2), row('lower_fp', .1), row('tied_fp', .2), row('supported', .1, 1)]
        selected = calibrate(rows, [True, False, False, True])
        self.assertEqual(selected['threshold'], .2)
        self.assertEqual(selected['binding_ids'], ['positive'])
        self.assertEqual([decide(r, .2)['alert'] for r in rows], [True, False, True, True])
        self.assertFalse(decide(rows[0], np.nextafter(.2, 1.))['alert'])
        withheld = decide(rows[1], .2)
        self.assertTrue(withheld['unknown'])
        self.assertTrue(withheld['geometry_ambiguous'])


if __name__ == '__main__':
    unittest.main()
