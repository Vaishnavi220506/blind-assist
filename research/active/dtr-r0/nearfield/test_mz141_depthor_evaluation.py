"""Synthetic metric/selection checks without capture or model access."""
import copy
import json
import unittest

import numpy as np

from evaluate_mz141_depthor import ARMS, FAMILIES, LARGE, THIN, extra_metrics, summarize


def record(mae, coverage=.6, over=0, pixels=100, ring_error=0, over_near=0):
    agreeing = int(pixels*coverage)
    return dict(
        pixels=200, valid_prediction_pixels=200,
        native_bounds_reference_pixels=200, unknown_reference_pixels=0,
        target=dict(visible_pixels=pixels, comparable_pixels=pixels,
            range_agreeing_pixels=agreeing,
            absolute_error_m=dict(count=pixels, mean=mae, median=mae, p90=mae*2, maximum=mae*3),
            transverse_columns=dict(visible=100, with_at_least_half_pixels_agreeing=int(100*coverage))),
        silhouette_neighborhood=dict(pixels=100, referenced_pixels=100, unknown_reference_pixels=0,
            invalid_prediction_pixels=0, target_depth_band_spurious_pixels=0,
            closer_or_target_like_error_pixels=ring_error),
        target_gt_median=3., target_pred_median=3.+mae, target_over_1m_pixels=over,
        non_target_known_pixels=100, non_target_comparable_pixels=100,
        non_target_over_near_pixels=over_near, non_target_mae=.1,
    )


def cases(pixel_mae=.4, surface_mae=.38):
    result = []
    for partition in ('fit', 'heldout'):
        for family in FAMILIES:
            thin = family in THIN
            result.append(dict(id=partition+'_'+family, family=family, scene_group=family+'_scene3',
                partition=partition, arms=dict(
                    frozen=record(1.2 if thin else .2, coverage=0. if thin else .6, over=80 if thin else 0),
                    pixel=record(pixel_mae if thin else .2, over=10 if thin else 0),
                    surface=record(surface_mae if thin else .2, over=10 if thin else 0))))
    return result


class MZ141GeometryTests(unittest.TestCase):
    def test_unknown_and_invalid_do_not_become_non_target_errors(self):
        gt = np.array([[3., 4., np.nan], [3., 4., 5.]])
        target = np.array([[True, False, False], [True, False, False]])
        prediction = np.array([[1.5, 3., .1], [3.1, np.nan, 5.2]])
        result = extra_metrics(prediction, dict(depth=gt, target_visible=target))
        self.assertEqual(result['target_over_1m_pixels'], 1)
        self.assertEqual(result['non_target_known_pixels'], 3)
        self.assertEqual(result['non_target_comparable_pixels'], 2)
        self.assertEqual(result['non_target_over_near_pixels'], 1)
        self.assertAlmostEqual(result['non_target_mae'], .6)
        self.assertEqual(result['target_gt_median'], 3.)
        json.dumps(result, allow_nan=False)

    def test_pooled_mae_weights_pixels_but_frame_p90_is_labeled_mean(self):
        data = cases()
        first = next(c for c in data if c['partition'] == 'heldout' and c['family'] == THIN[0])
        first['arms']['pixel'] = record(.2, pixels=10)
        second = copy.deepcopy(first)
        second.update(id=first['id']+'_second', scene_group='second_scene')
        second['arms']['pixel'] = record(.8, pixels=90)
        data.append(second)
        stats = summarize(data)['partitions']['heldout']['arms']['pixel']['families'][THIN[0]]
        self.assertAlmostEqual(stats['target_mae_m'], .74)
        self.assertAlmostEqual(stats['target_per_frame_p90_mean_m'], 1.)
        self.assertEqual(stats['target_absolute_error_count'], 100)

    def test_prefers_simple_pixel_when_surface_extra_gain_is_small(self):
        result = summarize(cases())
        self.assertTrue(result['gates']['pixel']['eligible'])
        self.assertTrue(result['gates']['surface']['eligible'])
        self.assertFalse(result['surface_extra_gain']['passed'])
        self.assertEqual(result['selected_arm'], 'pixel')
        json.dumps(result, allow_nan=False)

    def test_selects_surface_for_extra_gain_or_only_admissible_arm(self):
        self.assertEqual(summarize(cases(surface_mae=.35))['selected_arm'], 'surface')
        self.assertEqual(summarize(cases(pixel_mae=.7, surface_mae=.35))['selected_arm'], 'surface')

    def test_rejects_spill_even_when_thin_depth_and_aggregate_precision_improve(self):
        data = cases(surface_mae=.2)
        case = next(c for c in data if c['partition'] == 'heldout' and c['family'] == LARGE[0])
        case['arms']['surface']['non_target_over_near_pixels'] = 3
        case['arms']['pixel']['silhouette_neighborhood']['closer_or_target_like_error_pixels'] = 3
        result = summarize(data)
        self.assertIsNone(result['selected_arm'])
        self.assertFalse(result['gates']['surface']['families'][LARGE[0]]['whole_non_target_overnear_noninferior'])
        self.assertFalse(result['gates']['pixel']['families'][LARGE[0]]['ring_closer_noninferior'])

    def test_fit_gains_do_not_override_holdout_failure_or_invalid_outputs(self):
        data = cases(pixel_mae=.8, surface_mae=.8)
        for c in data:
            if c['partition'] == 'fit':
                c['arms']['pixel'] = record(.01)
                c['arms']['surface'] = record(.01)
        self.assertIsNone(summarize(data)['selected_arm'])
        data = cases()
        for c in data:
            if c['partition'] == 'heldout':
                for arm in ('pixel', 'surface'):
                    c['arms'][arm]['valid_prediction_pixels'] -= 1
        self.assertIsNone(summarize(data)['selected_arm'])

    def test_missing_family_and_duplicate_rows_do_not_admit(self):
        data = [c for c in cases() if c['family'] != THIN[0]]
        self.assertIsNone(summarize(data)['selected_arm'])
        data = cases()
        with self.assertRaises(ValueError):
            summarize(data+[copy.deepcopy(data[0])])


if __name__ == '__main__':
    unittest.main()
