"""Focused synthetic evaluation contract tests; no consumed outcomes loaded."""
import unittest
import numpy as np
import cnh_segment_learned_eval as E


class LearnedSegmentEvalTests(unittest.TestCase):
    def test_cohort_v2_exclusion_and_v4_own_id_scheme(self):
        self.assertNotIn(143, E.expected_cohort('v2')['audit'])
        self.assertEqual(E.expected_cohort('v4')['audit'], list(range(32, 96)))
        self.assertEqual(E.expected_cohort('v4')['calib'], list(range(32)))

    def test_ap_floor_does_not_mutate_threshold_scores(self):
        y = np.tile(np.array([[0], [1], [0], [1]]), (1, 6))
        score = np.tile(np.array([[-np.inf], [2.], [0.], [1.]]), (1, 6))
        units = {128: dict(y=y, main=np.ones(4, bool), **{'M1__SHIFT05': score})}
        result = E.unit_ap(units, [128], ('M1__SHIFT05',))
        self.assertEqual(result['HEAD']['128']['M1__SHIFT05'], 1.)
        self.assertTrue(np.isneginf(score[0]).all())

    def test_paired_macro_uses_units_not_frame_counts(self):
        rows = {g: {'128': {'S2': .1, 'NN': .2, 'AUG__SHIFT05': .4},
                    '129': {'S2': .7, 'NN': .6, 'AUG__SHIFT05': .8}} for g in ('HEAD', 'BODY')}
        macro, paired = E.ap_comparisons(rows, ('S2', 'NN', 'AUG__SHIFT05'))
        self.assertAlmostEqual(macro['BODY']['AUG__SHIFT05'], .6)
        result = paired['HEAD']['AUG__SHIFT05-minus-NN']
        self.assertAlmostEqual(result['delta'], .2)
        np.testing.assert_allclose(result['ci95'], [.2, .2])
        self.assertEqual(result['units'], 2)
        self.assertEqual(paired['BODY']['NN-minus-NN']['ci95'], [0., 0.])

    def test_threshold_ties_respect_budget_and_ignore_pre3(self):
        seqs = []
        for peak in (4., 4., 2., -np.inf):
            scores = np.zeros((12, 6))
            scores[:3] = 1000.
            scores[3:] = peak
            seqs.append(dict(unit=1, frame=np.arange(12), y=np.zeros((12, 6), int),
                             w=np.ones((12, 6)), strata=np.full((12, 6), ''), s={'M1__SHIFT05': scores}))
        threshold = E.L.threshold_for_budget(seqs, 'M1__SHIFT05', (0,), .25)
        self.assertGreater(threshold, 4.)
        rows = E.L.pairs(seqs, 'M1__SHIFT05', (1, 1), (0,), threshold)
        self.assertEqual(E.L.summary(rows)['false_alert_rate'], 0.)

    def test_alarm_first_positive_hit_preserves_nonmonotonic_witness(self):
        y = np.zeros((12, 6), int)
        y[3:6, 0] = 1
        score = np.zeros((12, 6))
        score[3:6, 0] = (2., 3., 3.)
        w = np.ones((12, 6))
        w[3:6, 0] = (.8, 1.3, .7)
        seqs = [dict(unit=1, frame=np.arange(12), y=y, w=w, strata=np.full((12, 6), 'tiny'), s={'NN': score})]
        low = E.L.summary(E.L.pairs(seqs, 'NN', (1, 1), (0,), 1.))
        high = E.L.summary(E.L.pairs(seqs, 'NN', (1, 1), (0,), 2.5))
        self.assertEqual(low['timely'], 0.)
        self.assertEqual(high['timely'], 1.)


if __name__ == '__main__':
    unittest.main()
