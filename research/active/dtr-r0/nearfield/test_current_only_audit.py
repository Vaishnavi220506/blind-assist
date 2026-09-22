"""Threshold contract checks, including the pre-exit confound and atomic ties."""
import math
import unittest

from run_current_only_audit import cutoff


class CurrentOnlyCutoffTest(unittest.TestCase):
    def choose(self, scores, truth, baseline):
        rows = [dict(id=str(i), a=a, base_group_id='fixture', layout_relation='BOUNDARY', time_s=i*.2)
                for i, a in enumerate(baseline)]
        predictions = [dict(id=str(i), scores={'original': s}) for i, s in enumerate(scores)]
        labels = [dict(id=str(i), truth=y) for i, y in enumerate(truth)]
        return cutoff(rows, predictions, labels, 'original')

    def test_preexit_positive_does_not_constrain_current_threshold(self):
        selected = self.choose([.75, .06, .42], [True, False, True], [False]*3)
        self.assertEqual(selected['threshold'], math.nextafter(.06, math.inf))
        self.assertEqual(selected['forbidden_ids'], ['1'])
        self.assertGreaterEqual(.75, selected['threshold'])
        self.assertGreaterEqual(.42, selected['threshold'])

    def test_atomic_tie_cannot_rescue_positive_at_negative_maximum(self):
        selected = self.choose([.42, .42], [True, False], [False, False])
        self.assertGreater(selected['threshold'], .42)

    def test_existing_baseline_false_positive_does_not_raise_cutoff(self):
        selected = self.choose([10., .2], [False, False], [True, False])
        self.assertEqual(selected['threshold'], math.nextafter(.2, math.inf))
        self.assertEqual(selected['forbidden_ids'], ['1'])


if __name__ == '__main__':
    unittest.main()
