"""Focused checks for selection and the prediction/evaluator boundary."""
import ast
from pathlib import Path
import unittest

import numpy as np

import data_coverage_learning as learning


def fixture():
    metadata, truth = [], []
    for relation in ('INSIDE', 'BOUNDARY', 'OUTSIDE'):
        for frame in range(24):
            metadata.append(dict(clip_id=relation, time_s=frame * .2, layout_relation=relation))
            truth.append(relation != 'OUTSIDE' and 8 <= frame <= 13)
    return metadata, np.array(truth), np.zeros(72, bool)


class CoverageLearningTest(unittest.TestCase):
    def test_selection_keeps_boundary_rescue_and_rejects_lower_false_alert_score(self):
        metadata, truth, a = fixture()
        logits = np.full(72, -3.)
        logits[24 + 8:24 + 13] = 1.  # Hold supplies the last positive, no false tail.
        logits[48 + 10] = .5
        selected, candidates = learning.choose_threshold(logits, truth, metadata, a)
        self.assertEqual(selected['threshold'], 1.)
        self.assertEqual(selected['Boundary_hold_TP'], 6)
        self.assertFalse(selected['disabled'])
        self.assertFalse(next(r for r in candidates if r['threshold'] == .5)['admissible'])

    def test_current_true_but_hold_false_tail_can_force_disabled_branch(self):
        metadata, truth, a = fixture()
        logits = np.full(72, -3.)
        logits[24 + 13] = 2.
        selected, _ = learning.choose_threshold(logits, truth, metadata, a)
        self.assertTrue(selected['disabled'])
        self.assertEqual(selected['threshold'], 3.)
        self.assertTrue(all(r['pass_cost'] for r in selected['cost'].values()))

    def test_equal_flags_choose_highest_finite_threshold(self):
        metadata, truth, a = fixture()
        a[24 + 8:24 + 13] = True
        logits = np.full(72, -3.)
        logits[24 + 10] = 2.
        selected, _ = learning.choose_threshold(logits, truth, metadata, a)
        self.assertEqual(selected['Boundary_hold_TP'], 6)
        self.assertTrue(selected['disabled'])

    def test_nonfinite_scores_rejected(self):
        metadata, truth, a = fixture()
        logits = np.zeros(72); logits[1] = np.nan
        with self.assertRaises(ValueError):
            learning.choose_threshold(logits, truth, metadata, a)

    def test_evaluation_truth_not_in_prediction_function(self):
        tree = ast.parse(Path(learning.__file__).read_text(encoding='utf-8'))
        for function_name in ('predict', '_public', '_predict_head'):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
            strings = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            self.assertFalse(any('labels.json' in value or 'evaluator/' in value or 'source-admission.json' in value for value in strings))

    def test_event_onset_loss_is_detected(self):
        result = {}
        for region in ('Core', 'Boundary'):
            arms = {}
            for mode in ('current', 'hold'):
                for name, delay in (('A', .2), ('N', .4)):
                    arms[name + '_' + mode] = dict(events=[dict(clip_id='clip', detected=True,
                        first_in_event_alert_delay_s=delay)])
            result[region] = dict(arms=arms)
        self.assertFalse(learning.event_retention(result)['pass_retention'])
        for region in result.values():
            for mode in ('current', 'hold'):
                region['arms']['N_' + mode]['events'][0]['first_in_event_alert_delay_s'] = 0.
        self.assertTrue(learning.event_retention(result)['pass_retention'])


if __name__ == '__main__':
    unittest.main()
