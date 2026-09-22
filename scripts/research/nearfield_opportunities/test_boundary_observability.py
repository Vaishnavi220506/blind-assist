"""Hand fixtures for the observation diagnostic; no saved-cohort scoring."""
from dataclasses import asdict
import math
import unittest
from unittest.mock import patch

import active_view as av
import boundary_observability as probe


class ObservabilityTests(unittest.TestCase):
    def test_view_translation_and_rotation_budgets(self):
        self.assertAlmostEqual(sum(math.dist(a, b) for a, b in zip(probe.PATH, probe.PATH[1:])), .12)
        for name, yaws in probe.schedules().items():
            self.assertEqual((len(yaws), yaws[0]), (13, 0.))
            total = sum(abs(a-b) for a, b in zip(yaws, yaws[1:]))
            self.assertAlmostEqual(total, 0. if name == 'fixed' else math.radians(5.625))

    def test_fixed_coarse_identity_and_yaw_wall_range(self):
        value = av.Scene((av.Box(.2, 1.3, .15),))
        for view, camera in zip(probe.capture(value, 'fixed'), probe.PATH):
            self.assertEqual(view['bins']['coarse'], list(av.observe(value, camera)))
        for view in probe.capture(av.Scene(()), 'sweep_plus'):
            for angle, raw in zip(av.ANGLES, view['raw_ranges']):
                self.assertAlmostEqual(raw, (4.2-view['camera'][1])/math.cos(angle+view['yaw_rad']))

    def test_precision_control_uses_same_raw_returns(self):
        with patch.object(av, 'hit_distance', return_value=1.0494):
            a = probe.capture(av.Scene((av.Box(0., 1., .5),)), 'fixed')
        with patch.object(av, 'hit_distance', return_value=1.0504):
            b = probe.capture(av.Scene((av.Box(0., 1., .5),)), 'fixed')
        self.assertEqual(a[0]['bins']['coarse'], [10]*8)
        self.assertEqual(b[0]['bins']['coarse'], [11]*8)
        self.assertEqual(a[0]['bins']['fine'], [1049]*8)
        self.assertTrue(probe.compare(a, b)['raw_distinct'])

    def test_lateral_gauge_flips_labels_at_identical_ranges(self):
        for face in ('left', 'right'):
            for inside in (True, False):
                sign = 1 if face == 'right' else -1
                b = av.Box(sign*(.35+(-.0004 if inside else .0004)), 1.4, .10)
                row = dict(id='fixture', stratum='boundary_'+face, boxes=[asdict(b)], wall_z=4.2)
                w = probe.lateral_counterexample(row)
                self.assertTrue(w['domain_valid'] and w['opposite_label'])
                for schedule in probe.schedules():
                    result = probe.compare(probe.capture(probe.scene(row), schedule),
                        probe.capture(probe.scene(w['alternative']), schedule, w['bias']['pose_x_m']))
                    self.assertLessEqual(result['max_raw_difference_m'], 1e-12)
                    self.assertEqual(result['changed_bins'], dict(coarse=0, fine=0))
        with self.assertRaises(ValueError):
            probe.lateral_counterexample(dict(stratum='boundary_far'))

    def test_observation_never_reads_truth(self):
        with patch.object(av, 'intersects_query', side_effect=AssertionError('Capture cannot read truth')):
            for name in probe.schedules():
                self.assertEqual(len(probe.capture(av.Scene(()), name)), 13)


if __name__ == '__main__':
    unittest.main()
