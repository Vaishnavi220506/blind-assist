"""Focused guards for split scope and native-containment interpretation."""
import copy
import unittest

from run_mz139_surface_fit import native_audit, replay_payload, select_ids


class EvaluationContractTests(unittest.TestCase):
    def test_dev_cannot_be_reduced_and_train_pilot_covers_four_families(self):
        spec = {'frames': [dict(id=f'{split}-{family}-{i}', split=split, family=family)
            for split, count in [('train', 48), ('dev', 12)]
            for family in ('head', 'body', 'rod', 'boundary') for i in range(count)]}
        self.assertEqual(len(select_ids(spec, 'dev')), 48)
        self.assertEqual(len(select_ids(spec, 'train', pilot=True)), 12)
        for options in ({'pilot': True}, {'limit': 12}):
            with self.assertRaises(ValueError):
                select_ids(spec, 'dev', **options)
        with self.assertRaises(ValueError):
            select_ids(spec, 'test')

    def test_proximity_does_not_count_as_exact_native_geometry_retention(self):
        point = [1., .11, 1.]
        row = {'id': 'opaque'}
        evaluation = dict(body_origin_m=[0., 0., 0.], zonal_tof_native=[dict(zone_id=1,
            returned_lineage=[dict(target_index=0, hit_indices=[0])],
            private_rays=[dict(subray=0, hit_point_m=point, actor_id='actor')])])
        item = dict(zone_id=1, target_slot=0, status='SIM_VALID',
                    localized_xyz=[[.8, 1.2], [-.2, .2], [.8, 1.2]],
                    coarse_xyz=[[.8, 1.2], [-.2, .2], [.8, 1.2]])
        prediction = dict(replacement_keys=[[1, 0]], surface_candidate=False,
            surfaces=[dict(center=[1., 0., 1.], half_extent=[.1, .1, .1])])

        def contains(p, surface, tolerance):
            return all(abs(v-c) <= h+tolerance for v, c, h in zip(p, surface['center'], surface['half_extent']))

        returns, points = native_audit(row, evaluation, prediction,
                                      {'spatial_evidence': [item]}, contains)
        self.assertTrue(returns[0]['native_lineage_available'])
        self.assertTrue(points[0]['corridor'])
        self.assertFalse(points[0]['modeled_contains'])
        self.assertFalse(points[0]['geometry_retained'])
        self.assertTrue(points[0]['modeled_sigma_scale_proximity'])
        self.assertTrue(points[0]['incumbent_alert_support'])
        untouched = copy.deepcopy(prediction)
        untouched['replacement_keys'] = []
        _, unchanged = native_audit(row, evaluation, untouched, {'spatial_evidence': [item]}, contains)
        self.assertTrue(unchanged[0]['original_support_unchanged'])
        self.assertIsNone(unchanged[0]['modeled_contains'])

    def test_explicit_replay_ignores_only_fit_elapsed_time(self):
        before = dict(id='opaque', candidate=True, fit=dict(seconds=1., loss=2.))
        after = copy.deepcopy(before)
        after['fit']['seconds'] = 100.
        self.assertEqual(replay_payload(before), replay_payload(after))
        after['fit']['loss'] = 3.
        self.assertNotEqual(replay_payload(before), replay_payload(after))
        self.assertEqual(before['fit']['seconds'], 1.)


if __name__ == '__main__':
    unittest.main()
