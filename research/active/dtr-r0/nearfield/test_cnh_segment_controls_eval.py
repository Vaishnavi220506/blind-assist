"""Synthetic controls tests: paired references and frozen-threshold stress."""
import unittest
import numpy as np
import cnh_segment_controls_eval as C


class ControlsEvalTests(unittest.TestCase):
    def test_comparisons_are_same_condition(self):
        pairs = set(C.comparisons())
        self.assertIn(('AUG__FA10', 'MASK__FA10'), pairs)
        self.assertIn(('AUG__FA10', 'ZERO_AUG__FA10'), pairs)
        self.assertIn(('IDEAL__BG', 'ZERO_IDEAL__BG'), pairs)
        self.assertNotIn(('AUG__FA10', 'MASK__IDEAL'), pairs)
        self.assertNotIn(('IDEAL__BG', 'ZERO_AUG__BG'), pairs)

    def test_unit_paired_control_differences(self):
        per = {g: {} for g, _ in C.L.GROUPS}
        for g in per:
            for i in range(3):
                row = {a: .4+i*.1 for a in C.ARMS}
                row['AUG__BG'] += .2
                row['ZERO_AUG__BG'] -= .1
                per[g][str(128+i)] = row
        macro, paired = C.ap_comparisons(per)
        self.assertAlmostEqual(macro['HEAD']['AUG__BG'], .7)
        self.assertAlmostEqual(paired['HEAD']['AUG__BG-minus-MASK__BG']['delta'], .2)
        np.testing.assert_allclose(paired['BODY']['AUG__BG-minus-ZERO_AUG__BG']['ci95'], [.3, .3])

    def test_frozen_ideal_threshold_is_not_stress_recalibration(self):
        arms = ('NN', 'AUG__IDEAL', 'AUG__FA5')
        units = {}
        for u in (96, 128):
            y = np.zeros((24, 6), int)
            y[12:] = 1
            w = np.ones((24, 6))*1.2
            w[22:] = .8
            score = np.ones((24, 6))
            score[12:] = 5.
            units[u] = dict(y=y, w=w, strata=np.full((24, 6), 'tiny'), main=np.ones(24, bool),
                config=np.repeat([0, 1], 12), frame=np.tile(np.arange(12), 2),
                NN=score, AUG__IDEAL=score, AUG__FA5=score+2.)
        results = C.alarm_results(units, {'calib': [96], 'audit': [128]}, arms)
        primary = results['per_condition']['HEAD|AUG__FA5|0.10']
        fixed = results['frozen_ideal']['HEAD|AUG__FA5|0.10']
        self.assertEqual(primary['threshold_calibrated_on'], 'AUG__FA5')
        self.assertEqual(fixed['threshold_calibrated_on'], 'AUG__IDEAL')
        self.assertGreater(primary['threshold'], fixed['threshold'])
        self.assertEqual(primary['all']['false_alert_rate'], 0.)
        self.assertEqual(fixed['all']['false_alert_rate'], 1.)
        self.assertEqual(primary['tiny']['near'], fixed['tiny']['near'])
        self.assertEqual(primary['all']['empty_pairs'], fixed['all']['empty_pairs'])
        self.assertNotIn('false_alert_rate', primary['tiny'])


if __name__ == '__main__':
    unittest.main()
