"""Focused independent checks of Development metrics and denominator handling."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cnh_segment_joint_analysis as joint


class SegmentAnalysisTests(unittest.TestCase):
    def packed(self):
        # An empty pair; a near event first seen far; a nonmonotonic near event.
        y = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]], bool)
        w = np.array([[np.nan]*3, [1.4, 1., .7], [.7, 1.3, .8]])
        return dict(y=y, w=w, empty=~y.any(1), near=(y & (w <= 1.)).any(1),
                    strata=np.array(['', 'tiny', 'realistic']),
                    frame_strata=np.array([['']*3, ['tiny']*3, ['realistic']*3]),
                    unit=np.array([128, 128, 129]),
                    s={'M0': np.array([[1., 0, 0], [3., 4., 5.], [4., 5., 5.]])})

    def test_first_positive_hit_and_empty_pair_definition(self):
        p = self.packed()
        row = joint.metrics.evaluate(p, 'M0', (2.,))
        self.assertEqual(row['all']['timely'], dict(numerator=1, denominator=2, rate=.5))
        self.assertEqual(row['all']['late']['numerator'], 1)
        self.assertEqual(row['all']['false_alert']['denominator'], 1)
        self.assertNotIn('false_alert', row['tiny'])
        # Higher threshold can make a nonmonotonic event timely; preserve first-hit rule.
        row = joint.metrics.evaluate(p, 'M0', (4.5,))
        self.assertEqual(row['realistic']['timely']['rate'], 1.)

    def test_calibration_includes_disabled_threshold(self):
        p = self.packed()
        p['s']['M0'][0, 0] = 20.
        self.assertTrue(np.isinf(joint.metrics.select(p, 'M0', .2)[0]))

    def test_unknown_and_error_exclude_fallback_and_fake_truth(self):
        p = self.packed()
        candidate = np.ones((3, 3), bool)
        eligible = candidate.copy()
        eligible[1, 0] = False
        distance = np.ones((3, 3))
        truth = np.ones((3, 3))*.8
        truth[2, 0] = np.nan
        p['extra'] = {'M0': dict(candidate=candidate, eligible=eligible, distance=distance, truth_distance=truth)}
        p['s']['M2__IDEAL'] = p['s']['M0']
        p['extra']['M2__IDEAL'] = p['extra']['M0']
        result = joint.diagnostics(p, 'M2__IDEAL', 2.)['all']
        self.assertEqual(result['fallback_query_frames']['numerator'], 1)
        self.assertIsNone(joint.diagnostics(p, 'M0', 2.)['all']['fallback_query_frames'])
        self.assertEqual(result['unconfirmed_candidate_query_frames']['denominator'], 8)
        self.assertEqual(result['unconfirmed_candidate_query_frames']['numerator'], 3)
        self.assertEqual(result['distance_error']['count'], 4)
        self.assertAlmostEqual(result['distance_error']['mae_m'], .2)
        self.assertEqual(result['confirmed_without_truth']['numerator'], 1)
        self.assertEqual(joint.diagnostics(p, 'M0', np.inf)['all']['distance_error']['count'], 0)

    def test_unit_paired_identical_policy_is_zero(self):
        p = self.packed()
        p['s']['M1__IDEAL'] = p['s']['M0'].copy()
        result = joint.paired_delta(p, 'M1__IDEAL', 2., 2.)
        self.assertEqual(result['delta'], 0.)
        self.assertEqual(result['ci95'], [0., 0.])

    def test_object_unknown_includes_dropped_and_distance_uses_winning_query(self):
        score = np.full((3, 6), -np.inf)
        score[0, [0, 2]] = [3., 4.]
        score[2, 0] = 5.  # false mask
        distance = np.ones((3, 6))*1.2
        distance[0, 0] = 8.  # must not select this weaker query
        data = dict(score=score, distance=distance, truth=np.array([1., 1., np.nan]),
                    **{'class': np.array(['tiny', 'tiny', 'false'])})
        out = joint.object_diagnostics(data, (0, 2, 4), 2.)
        self.assertEqual(out['tiny']['unconfirmed'], dict(numerator=1, denominator=2, rate=.5))
        self.assertAlmostEqual(out['tiny']['distance_error']['mae_m'], .2)
        self.assertAlmostEqual(out['tiny']['distance_error']['p90_abs_m'], .2)
        self.assertEqual(out['fake_confirmation']['rate'], 1.)
        self.assertEqual(joint.object_diagnostics(data, (0, 2, 4), np.inf)['tiny']['unconfirmed']['rate'], 1.)

    def test_reject_incomplete_or_forbidden_cohort(self):
        with tempfile.TemporaryDirectory() as d:
            shape = (9, 6)
            data = dict(split=np.array('audit'), frame=np.arange(3, 12), config=np.zeros(9),
                        labels=np.zeros(shape), witness=np.full(shape, np.nan), strata=np.full(shape, ''),
                        M0=np.zeros(shape), M1__IDEAL=np.zeros(shape), M2__IDEAL=np.zeros(shape))
            path = Path(d)/'unit143.npz'
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, 'outside'):
                joint.load(d)
            path.unlink()
            np.savez(Path(d)/'unit128.npz', **data)
            with self.assertRaisesRegex(ValueError, 'all 32 calib'):
                joint.load(d)


if __name__ == '__main__':
    unittest.main()
