import unittest

import numpy as np

from tof_lateral_core import (lateral_relation, eligible, local_input, simple_geometry,
                              learned_decision, frame_decision)
from tof_fov45_core import boxes45


class LateralTest(unittest.TestCase):
    def test_contact_and_full_interval(self):
        self.assertEqual(lateral_relation((.1, .2), (3., 3.)), 'CROSSING')
        self.assertEqual(lateral_relation((.1, .2), (2., 3.)), 'CROSSING')
        self.assertEqual(lateral_relation((.2, .3), (2., 3.)), 'OUTSIDE')

    def test_no_return_and_far_return_not_eligible(self):
        anchor = dict(possible=True, definite=False, interval_m=[2.5, 3.5])
        for z in (np.nan, 3., 3.1):
            self.assertFalse(eligible(boxes45()[29], z, anchor))

    def test_uniform_rgb_falls_back(self):
        result = simple_geometry(np.full((360, 640, 3), 128, np.uint8), boxes45()[29], [2.5, 3.])
        self.assertEqual(result['relation'], 'UNKNOWN')
        self.assertFalse(result['suppress'])

    def test_input_geometry_independent_of_rgb(self):
        a = local_input(np.zeros((360, 640, 3), np.uint8), boxes45()[29], 2.5, [2.2, 2.8])
        b = local_input(np.full((360, 640, 3), 255, np.uint8), boxes45()[29], 2.5, [2.2, 2.8])
        self.assertEqual(a.shape, (8, 48, 48))
        np.testing.assert_array_equal(a[3:], b[3:])
        self.assertFalse(np.array_equal(a[:3], b[:3]))

    def test_outside_confidence_and_definite_bypass(self):
        self.assertFalse(learned_decision([.1, .8, .1])['suppress'])
        self.assertTrue(learned_decision([.02, .96, .02])['suppress'])
        base = dict(alert=True, definite_zones=1, unknown=False, state='ALERT_SUPPORTED')
        result = frame_decision(base, [dict(zone=0, possible=True, definite=True)],
                                [dict(zone=0, joint=1.)], .1, {0})
        self.assertTrue(result['alert'])

    def test_residual_vote_and_unknown_silence(self):
        base = dict(alert=True, definite_zones=0, unknown=True, state='ALERT_AMBIGUOUS')
        anchors = [dict(zone=z, possible=True, definite=False) for z in (0, 1)]
        scores = [dict(zone=z, joint=.5) for z in (0, 1)]
        self.assertTrue(frame_decision(base, anchors, scores, .1, {0})['alert'])
        result = frame_decision(base, anchors, scores, .1, {0, 1})
        self.assertFalse(result['alert'])
        self.assertTrue(result['unknown'])


if __name__ == '__main__':
    unittest.main()
