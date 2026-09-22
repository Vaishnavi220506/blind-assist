"""Source-only admission; no captured outcomes, fitting or protected labels."""
from collections import Counter, defaultdict
import copy
import json
import unittest

from core_transfer_spec import specification as old_core
from core_workpoint_transfer_spec import specification as old_workpoint
from core_hold_validation_spec import specification as old_hold
from full_event_spec_20260920 import specification as old_full
from spatial_bce_spec import specification as old_bce
from spatial_complement_transfer_spec import specification as old_transfer
from data_coverage_spec import (
    specification, check_spec, bounds, classify, RANGES, FAMILIES,
    SPLIT_GROUPS, MATERIALS, BACKGROUNDS,
)
from full_event_metrics_20260920 import evaluate, ARMS


class DataCoverageSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = specification()

    def test_complete_source_disjoint_from_six_old_cohorts(self):
        report = check_spec(self.spec, [old_core(), old_workpoint(), old_hold(),
                                       old_full(), old_bce(), old_transfer()])
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['compared_old_cohorts'], 6)
        self.assertEqual(Counter(c['split'] for c in self.spec['cases']),
                         dict(train=1728, dev=576, evaluation=1152))
        self.assertEqual(self.spec, specification())
        self.assertEqual(check_spec(json.loads(json.dumps(self.spec)), [])['status'], 'PASS')

    def test_broader_stratified_metric_shapes_and_balanced_appearance(self):
        groups = defaultdict(list)
        for c in self.spec['cases']:
            if c['layout_relation'] == 'INSIDE' and c['frame_in_clip'] == 0:
                groups[(c['type_id'], c['split'])].append(c)
        for family, _, old_size, _ in FAMILIES:
            for split, count in SPLIT_GROUPS.items():
                rows = groups[(family, split)]
                for axis, (lo, hi) in enumerate(RANGES[family]):
                    values = [(c['objects'][0]['size_m']+[c['objects'][0]['center_m'][2]])[axis]
                              for c in rows]
                    self.assertEqual({int((v-lo)/(hi-lo)*count) for v in values}, set(range(count)))
                    if split == 'train' and axis < 3:
                        self.assertLess(min(values), old_size[axis]*.86)
                        self.assertGreater(max(values), old_size[axis]*1.14)
            train = groups[(family, 'train')]
            self.assertEqual({c['objects'][0]['material'].rsplit('/', 1)[-1] for c in train}, set(MATERIALS))
            self.assertEqual({c['objects'][1]['material'].rsplit('/', 1)[-1] for c in train}, set(BACKGROUNDS))

    def test_outside_and_far_negatives_and_full_extent_positive_events(self):
        for c in self.spec['cases']:
            label = classify(*bounds(c))
            if c['layout_relation'] == 'OUTSIDE' or c['frame_in_clip'] in (0, 23):
                self.assertFalse(label['truth'])
            if c['layout_relation'] != 'OUTSIDE' and 8 <= c['frame_in_clip'] <= 13:
                self.assertTrue(label['truth'])
            if c['layout_relation'] == 'BOUNDARY' and label['truth']:
                self.assertGreater(abs(c['objects'][0]['center_m'][1]-c['camera']['y']), .3)
                self.assertTrue(label['boundary'])
        rows = []
        for c in self.spec['cases'][:72]:
            label = classify(*bounds(c))
            rows.append(dict(id=c['name'], clip_id=c['clip_id'], time_s=c['time_s'],
                truth=label['truth'], layout_relation=c['layout_relation'], layer=c['layer'],
                background=c['background'], phase=c['phase'],
                flags={a: label['truth'] for a in ARMS}, current_unknown={a: False for a in ARMS}))
        metrics = evaluate(rows)['overall']['arms'][ARMS[0]]
        self.assertEqual(metrics['event_count'], 2)
        self.assertEqual(metrics['detected_events'], 2)
        self.assertEqual(metrics['frames']['FP'], 0)
        self.assertTrue(all(e['exit_observed'] and not e['entry_left_censored'] for e in metrics['events']))

    def test_reject_partition_or_old_geometry_reuse(self):
        changed = copy.deepcopy(self.spec)
        changed['cases'][0]['split'] = 'dev' if changed['cases'][0]['split'] != 'dev' else 'train'
        with self.assertRaises(AssertionError):
            check_spec(changed, [])
        with self.assertRaises(AssertionError):
            check_spec(self.spec, [self.spec])

    def test_reject_relation_noise_appearance_and_shape_mutations(self):
        for field in ('noise', 'material', 'shape'):
            changed = copy.deepcopy(self.spec)
            case = changed['cases'][24]
            if field == 'noise':
                case['sensor_noise_key'] += '_different'
            elif field == 'material':
                case['objects'][0]['material'] += '_different'
            else:
                case['objects'][0]['size_m'][0] += .001
            with self.subTest(field=field), self.assertRaises(AssertionError):
                check_spec(changed, [])

    def test_reject_broken_time_pose_and_geometry_range(self):
        for field in ('time', 'pose', 'range'):
            changed = copy.deepcopy(self.spec)
            if field == 'time':
                changed['cases'][0]['time_s'] = .1
            elif field == 'pose':
                for c in changed['cases'][:72]:
                    c['camera']['yaw'] = 1.
            else:
                for c in changed['cases'][:72]:
                    c['objects'][0]['size_m'][0] = 10.
            with self.subTest(field=field), self.assertRaises(AssertionError):
                check_spec(changed, [])


if __name__ == '__main__':
    unittest.main()
