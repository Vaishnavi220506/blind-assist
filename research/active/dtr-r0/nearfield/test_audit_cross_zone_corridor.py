"""Synthetic only: no cohort arrays, source truth, or model import."""
import unittest
import numpy as np

from audit_cross_zone_corridor import frame_lineage, aggregate, event_report, baseline_parity, target_footprint
from evaluate_ba_camera_corridor import summarize


def trace(zid, indices, value=2., observed=True):
    return dict(zone_id=zid, observed=observed, distance_m=value if observed else None,
                winner_bin=20 if observed else None, pixel_indices=np.asarray(indices, int),
                weights=np.ones(len(indices)), reason='synthetic')


def row(index, lineage, truth=True, boundary=False, alert=False):
    pred = dict(alert=alert, unknown=False, ambiguous=False)
    return dict(clip_id='s00', pair_id='pair', frame_in_clip=index, time_s=index * .2,
                truth=truth, boundary=boundary, predictions={a: pred.copy() for a in ('raw_tof', 'nfo', 'depthpro_global')},
                lineage=lineage)


class CrossZoneTests(unittest.TestCase):
    def test_original_two_cm_native_footprint_and_point_sampling(self):
        from ba_camera_corridor import sample_indices, sample_native
        yy, xx = sample_indices()
        y, x = int(yy[96]), int(xx[128])
        native = np.full((360, 640), np.nan)
        native[y, x], native[y, x + 1] = 2.02, 2.020001
        focal = 640 / (2 * np.tan(np.deg2rad(50)))
        center = [2., (x + .5 - 320) / focal * 2, -(y + .5 - 180) / focal * 2]
        source = dict(case=dict(target_name='target', camera=dict(x=0., y=0., z=0., pitch=0., yaw=0., roll=0.)),
            geometry=dict(objects=[dict(name='target', actor_path='actor', render_bounds_center_m=center,
                render_bounds_extent_m=[0., .02, .02], trace=dict(hit_expected_actor=True, hit_actor_path='actor'))]),
            check=dict(native_valid_pixels=2, visible_target_pixels=1, competing_corridor_pixels=1))
        mask, admission = target_footprint(native, source)
        self.assertTrue(mask[y, x])
        self.assertFalse(mask[y, x + 1])
        self.assertEqual(int(sample_native(mask).sum()), 1)
        self.assertEqual(admission, source['check'])

    def test_other_zone_pure_with_local_background(self):
        out = frame_lineage([True, False, True, False], [0, 0, 1, 1], [trace(0, [1]), trace(1, [2])])
        self.assertEqual(out['target_pixels_without_local_target_return'], 1)
        self.assertEqual(out['observed_range_lt_3m']['pure_cross_zone_target_pixels'], 1)
        self.assertEqual(out['local_absence_observed_non_target_pixels'], 1)

    def test_mixed_is_possible_not_pure_and_not_local_absence(self):
        out = frame_lineage([True, False, True, False], [0, 0, 1, 1], [trace(0, [1]), trace(1, [2, 3])])
        self.assertEqual(out['all_ranges']['pure_anchor_zones'], 0)
        self.assertEqual(out['all_ranges']['mixed_cross_zone_target_pixels'], 1)
        self.assertEqual(out['target_pixels_without_local_target_return'], 1)
        self.assertEqual(out['zones'][1]['target_weight_fraction'], .5)

    def test_saved_range_strict_threshold_no_truth_replacement(self):
        for value, near in ((2.999, True), (3., False), (3.1, False)):
            out = frame_lineage([True, False, True], [0, 0, 1], [trace(0, [1]), trace(1, [2], value)])
            self.assertTrue(out['all_ranges']['pure_cross_zone_available'])
            self.assertEqual(out['observed_range_lt_3m']['pure_cross_zone_available'], near)

    def test_unobserved_not_anchor_and_outside_fov_not_cross_zone(self):
        out = frame_lineage([True, True, True], [-1, 0, 1], [trace(0, [], observed=False), trace(1, [2])])
        self.assertEqual(out['target_pixels_outside_sensor_zones'], 1)
        self.assertEqual(out['local_absence_unobserved_pixels'], 1)
        self.assertEqual(out['all_ranges']['pure_cross_zone_target_pixels'], 1)
        with self.assertRaisesRegex(ValueError, 'Unobserved data'):
            frame_lineage([True], [0], [trace(0, [0], observed=False)])

    def test_reject_contributor_from_another_zone(self):
        with self.assertRaisesRegex(ValueError, 'zone mismatch'):
            frame_lineage([True, False], [0, 1], [trace(0, [1]), trace(1, [1])])

    def test_no_cross_frame_anchor_transfer(self):
        anchor_only = frame_lineage([True, False], [0, 1], [trace(0, [0]), trace(1, [1])])
        missing_only = frame_lineage([True, False], [0, 0], [trace(0, [1])])
        summary = aggregate([row(0, anchor_only), row(1, missing_only)])
        self.assertEqual(summary['all_ranges']['frames_with_pure_anchor_available'], 1)
        self.assertEqual(summary['all_ranges']['frames_with_pure_cross_zone_available'], 0)

    def test_boundary_witness_does_not_rescue_interior_event(self):
        witness = frame_lineage([True, False, True], [0, 0, 1], [trace(0, [1]), trace(1, [2])])
        absent = frame_lineage([True, False], [0, 0], [trace(0, [1])])
        rows = [row(0, witness, boundary=True), row(1, absent)]
        events = summarize(rows, .2)['arms']['nfo']['events']
        event = event_report(rows, events)[0]
        self.assertTrue(event['nfo_missed_interior_event'])
        self.assertEqual(event['entry']['all_ranges']['frames_with_pure_cross_zone_available'], 1)
        self.assertEqual(event['interior']['all_ranges']['frames_with_pure_cross_zone_available'], 0)

    def test_old_baseline_parity_and_unknown_event_split(self):
        lineage = frame_lineage([True], [0], [trace(0, [0])])
        rows = [row(0, lineage), row(1, lineage, truth=None), row(2, lineage, alert=True)]
        metrics = summarize(rows, .2)
        old = dict(metrics=metrics, by_clip={'s00': metrics}, by_pair={'pair': metrics})
        self.assertEqual(baseline_parity(rows, old, .2)['arms']['nfo']['event_count'], 2)
        old['metrics']['arms']['nfo']['event_count'] = 1
        with self.assertRaisesRegex(ValueError, 'parity'):
            baseline_parity(rows, old, .2)


if __name__ == '__main__':
    unittest.main()
