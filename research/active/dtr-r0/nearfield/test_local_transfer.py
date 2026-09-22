"""Synthetic boundary checks only; never opens the new scientific cohort."""
from __future__ import annotations

import unittest

import numpy as np

from local_transfer_inference import public_pair_checks, validate_identities
from local_transfer_metrics import GATES, make_rows, report
from local_transfer_audit import independently_decide, runs_and_events, scalar_counts


def synthetic():
    ids, baseline, classes, raw, local = [], [], [], [], []
    for group in range(8):
        layer = 'BODY' if group < 4 else 'HEAD'
        target = 1 if layer == 'BODY' else 4
        for appearance in ('base', 'changed'):
            for relation in ('INSIDE', 'BOUNDARY', 'OUTSIDE'):
                clip = f'g{group}_{appearance}_{relation}'
                for frame in range(12):
                    positive = relation != 'OUTSIDE' and 2 <= frame <= 9
                    ids.append(dict(index=len(ids), id=f'{clip}_{frame:02}', clip_id=clip,
                        frame_in_clip=frame, time_s=frame*.2, appearance=appearance,
                        appearance_pair_id=f'g{group}_{relation}_{frame:02}', base_group_id=f'g{group}',
                        type_id=f'family{group//2}', layer=layer, layout_relation=relation))
                    baseline.append(dict(alert=positive and frame >= 4, unknown=True))
                    y, r, l = np.full(6, 6), np.full(6, .1), np.full(6, .1)
                    if positive:
                        y[target] = 2
                        l[target] = .9
                        if frame >= 3:
                            r[target] = .9
                    classes.append(y); raw.append(r); local.append(l)
    labels = dict(indices=np.arange(576), classes=np.asarray(classes),
        valid=np.ones((576, 6), bool), distances=np.where(np.asarray(classes) < 6, 1.5, np.inf))
    probabilities = dict(raw=np.asarray(raw), local=np.asarray(local))
    paired = {}
    for i, meta in enumerate(ids):
        paired.setdefault(meta['appearance_pair_id'], {})[meta['appearance']] = i
    public = dict(status='PASS', pairs=[dict(pair_id=key, base_index=value['base'], changed_index=value['changed'],
        tof_exactly_equal=True, rgb_changed=True, normalized_rgb_L1=.1) for key, value in sorted(paired.items())],
        mean_normalized_rgb_L1=.1, issues=[])
    cutoffs = dict(raw=.8, local=.8)
    return ids, baseline, labels, probabilities, public, cutoffs


class FrozenTransferTests(unittest.TestCase):
    def test_complete_paired_metrics_and_independent_gates(self):
        ids, base, labels, probs, public, cutoffs = synthetic()
        self.assertEqual(len(validate_identities(ids, {'old_geometry'})), 8)
        rows = make_rows(ids, base, probs, labels, cutoffs)
        result = report(rows, cutoffs, labels, public)
        self.assertEqual(result['source_admissibility']['status'], 'PASS')
        self.assertEqual(result['gates']['retained_transfer_component'], 'PASS')
        self.assertEqual(result['gates']['strict_system']['status'], 'PASS')
        self.assertEqual(len(result['appearance_pairs']['pairs']), 288)
        self.assertEqual(result['geometry_ordering']['by_appearance']['base']['local']['eligible_pairs'], 64)
        independent = independently_decide(rows, GATES, True, public, labels,
            result['appearance_pairs']['pairs'], result['geometry_ordering']['pairs'])
        self.assertEqual(independent['retained'], 'PASS')
        self.assertEqual(independent['bootstrap95'], result['gates']['component']['bootstrap']['recall_delta_95'])
        for arm in ('A_current', 'raw', 'local', 'raw_standalone', 'local_standalone'):
            self.assertEqual(scalar_counts(rows, arm), result['metrics']['arms'][arm]['frames']['all_known'])
            self.assertEqual(len(runs_and_events(rows, arm)[0]), 32)

    def test_inclusive_frozen_cutoff_and_a_retention(self):
        ids, base, labels, probs, public, cutoffs = synthetic()
        probs['local'][0, 4] = .8
        probs['local'][1, 4] = np.nextafter(.8, -np.inf)
        rows = make_rows(ids, base, probs, labels, cutoffs)
        self.assertTrue(rows[0]['predictions']['local_standalone']['alert'])
        self.assertFalse(rows[1]['predictions']['local_standalone']['alert'])
        self.assertTrue(all(not b['alert'] or r['predictions']['local']['alert'] for b, r in zip(base, rows)))
        self.assertTrue(all(r['predictions']['local']['unknown'] for r in rows))

    def test_invalid_truth_is_not_evaluable_without_dropping_rows(self):
        ids, base, labels, probs, public, cutoffs = synthetic()
        labels['valid'][2, 1] = False
        rows = make_rows(ids, base, probs, labels, cutoffs)
        result = report(rows, cutoffs, labels, public)
        self.assertEqual(len(rows), 576)
        self.assertIsNone(rows[2]['truth'])
        self.assertEqual(result['source_admissibility']['status'], 'NOT_EVALUABLE')
        self.assertEqual(result['gates']['retained_transfer_component'], 'NOT_EVALUABLE')

    def test_ranking_does_not_imply_a_fixed_cutoff_hit(self):
        ids, base, labels, probs, public, cutoffs = synthetic()
        for i, m in enumerate(ids):
            if m['layout_relation'] == 'BOUNDARY':
                probs['local'][i] = .7
        rows = make_rows(ids, base, probs, labels, cutoffs)
        result = report(rows, cutoffs, labels, public)
        pairs = [p for p in result['geometry_ordering']['pairs'] if p['eligible']]
        self.assertTrue(all(p['arms']['local']['frame_ordered'] for p in pairs))
        self.assertTrue(any(not p['arms']['local']['actual_boundary_alert'] for p in pairs))
        self.assertEqual(result['gates']['geometry_sensitivity']['status'], 'PASS')
        self.assertEqual(result['gates']['component']['status'], 'FAIL')

    def test_small_rescue_opportunity_is_not_a_failure(self):
        ids, base, labels, probs, public, cutoffs = synthetic()
        for i in range(576):
            base[i]['alert'] = bool((labels['classes'][i, [1, 4]] < 6).any())
        result = report(make_rows(ids, base, probs, labels, cutoffs), cutoffs, labels, public)
        self.assertEqual(result['gates']['appearance_stability']['base_rescues'], 0)
        self.assertEqual(result['gates']['appearance_stability']['rescue_status'], 'NOT_EVALUABLE')

    def test_component_failure_takes_precedence_over_insufficient_rescues(self):
        ids, base, labels, probs, public, cutoffs = synthetic()
        for i in range(576):
            base[i]['alert'] = bool((labels['classes'][i, [1, 4]] < 6).any())
        rows = make_rows(ids, base, probs, labels, cutoffs)
        result = report(rows, cutoffs, labels, public)
        self.assertEqual(result['gates']['component']['status'], 'FAIL')
        self.assertEqual(result['gates']['appearance_stability']['status'], 'NOT_EVALUABLE')
        self.assertEqual(result['gates']['retained_transfer_component'], 'FAIL')
        independent = independently_decide(rows, GATES, True, public, labels,
            result['appearance_pairs']['pairs'], result['geometry_ordering']['pairs'])
        self.assertEqual(independent['component'], 'FAIL')
        self.assertEqual(independent['appearance'], 'NOT_EVALUABLE')
        self.assertEqual(independent['retained'], 'FAIL')

    def test_background_pair_checks_exact_tof_and_real_rgb_change(self):
        ids = [dict(id='b', appearance_pair_id='p', appearance='base'),
               dict(id='c', appearance_pair_id='p', appearance='changed')]
        rgb = np.array([[[[0]]], [[[1]]]], np.uint8)
        tof = np.zeros((2, 64, 6), np.float32)
        answer = public_pair_checks(ids, rgb, tof)
        self.assertTrue(answer['pairs'][0]['tof_exactly_equal'])
        self.assertTrue(answer['pairs'][0]['rgb_changed'])
        self.assertEqual(answer['pairs'][0]['normalized_rgb_L1'], 1/255)
        self.assertEqual(answer['mean_normalized_rgb_L1'], 1/255)
        tof[1, 0, 0] = 1
        self.assertFalse(public_pair_checks(ids, rgb, tof)['pairs'][0]['tof_exactly_equal'])

    def test_changed_pixels_with_insufficient_amplitude_are_not_evaluable(self):
        pair_ids = [dict(id='b', appearance_pair_id='p', appearance='base'),
                    dict(id='c', appearance_pair_id='p', appearance='changed')]
        rgb = np.array([[[[0, 0]]], [[[1, 0]]]], np.uint8)
        tiny = public_pair_checks(pair_ids, rgb, np.zeros((2, 64, 6), np.float32))
        self.assertTrue(tiny['pairs'][0]['rgb_changed'])
        self.assertEqual(tiny['mean_normalized_rgb_L1'], .5/255)
        self.assertIn('Mean normalized RGB L1 is below 1/255', tiny['issues'])
        ids, base, labels, probs, public, cutoffs = synthetic()
        for pair in public['pairs']:
            pair['normalized_rgb_L1'] = .5/255
        public['mean_normalized_rgb_L1'] = .5/255
        result = report(make_rows(ids, base, probs, labels, cutoffs), cutoffs, labels, public)
        self.assertEqual(result['source_admissibility']['status'], 'NOT_EVALUABLE')
        self.assertEqual(result['gates']['retained_transfer_component'], 'NOT_EVALUABLE')

    def test_appearance_losses_and_two_isolated_false_segments(self):
        ids, base, labels, probs, public, cutoffs = synthetic()
        for i, m in enumerate(ids):
            if m['appearance'] == 'changed' and m['base_group_id'] == 'g0' and m['layout_relation'] == 'OUTSIDE' and m['frame_in_clip'] in (3, 9):
                probs['local'][i, 4] = .99
        rows = make_rows(ids, base, probs, labels, cutoffs)
        result = report(rows, cutoffs, labels, public)
        self.assertEqual(result['gates']['appearance_stability']['FP_delta'], 2)
        self.assertEqual(result['gates']['appearance_stability']['false_segment_delta'], 2)
        self.assertEqual(result['gates']['appearance_stability']['status'], 'FAIL')
        self.assertEqual(len(runs_and_events(rows, 'local')[1]), 2)


if __name__ == '__main__':
    unittest.main()
