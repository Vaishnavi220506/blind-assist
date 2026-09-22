"""Source-only checks: no model training, old predictions or held outcomes."""
import copy
import unittest

from core_transfer_spec import specification as old_core
from core_workpoint_transfer_spec import specification as old_workpoint
from core_hold_validation_spec import specification as old_hold
from full_event_spec_20260920 import specification as old_full
from spatial_bce_spec import specification, check_spec, bounds, classify
from launch_spatial_bce import validate_cases
from full_event_metrics_20260920 import evaluate, ARMS


class SpatialSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = specification()

    def test_new_groups_complete_events_and_old_disjointness(self):
        report = check_spec(self.spec, [old_core(), old_workpoint(), old_hold(), old_full()])
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['compared_old_cohorts'], 4)
        validate_cases(self.spec)
        self.assertEqual(self.spec, specification())

    def test_detect_split_leakage_and_pair_shape_mutation(self):
        changed = copy.deepcopy(self.spec)
        changed['cases'][0]['split'] = 'test' if changed['cases'][0]['split'] != 'test' else 'train'
        with self.assertRaises(AssertionError):
            check_spec(changed, [])
        changed = copy.deepcopy(self.spec)
        changed['cases'][24]['objects'][0]['size_m'][0] += .001
        with self.assertRaises(AssertionError):
            check_spec(changed, [])

    def test_detect_consumed_source_and_broken_sequence(self):
        with self.assertRaises(AssertionError):
            check_spec(self.spec, [self.spec])
        changed = copy.deepcopy(self.spec)
        changed['cases'][1]['frame_in_clip'] = 0
        with self.assertRaises(AssertionError):
            validate_cases(changed)

    def test_extent_and_complete_event_evaluator_compatibility(self):
        rows = []
        for case in self.spec['cases'][:72]:
            label = classify(*bounds(case))
            rows.append(dict(id=case['name'], clip_id=case['clip_id'], time_s=case['time_s'],
                truth=label['truth'], layout_relation=case['layout_relation'], layer=case['layer'],
                background=case['background'], phase=case['phase'],
                flags={arm: label['truth'] for arm in ARMS}, current_unknown={arm: False for arm in ARMS}))
        result = evaluate(rows)['overall']['arms'][ARMS[0]]
        self.assertEqual(result['event_count'], 2)
        self.assertEqual(result['detected_events'], 2)
        self.assertEqual(result['frames']['FP'], 0)
        self.assertTrue(all(e['exit_observed'] and not e['entry_left_censored'] for e in result['events']))
        # All positive Boundary object centers remain outside the corridor:
        # positive truth must come from extent intersection, never center location.
        for case in self.spec['cases']:
            if case['layout_relation'] == 'BOUNDARY' and classify(*bounds(case))['truth']:
                self.assertGreater(abs(case['objects'][0]['center_m'][1]-case['camera']['y']), .3)


if __name__ == '__main__':
    unittest.main()
