"""Structural generator and hand observation fixtures, no scored cohort solve."""
import math
import unittest
from unittest.mock import patch

import active_view as av
import bent_path_observation as bent
import bias_drift_observation as drift


class DriftTests(unittest.TestCase):
    def test_generation_balance_pairs_domain_and_deterministic_exclusion(self):
        stats = {}
        rows = drift.generate(statistics=stats)
        self.assertEqual(rows, drift.generate())
        self.assertEqual(len({drift.geometry_key(r) for r in rows}), 180)
        self.assertGreaterEqual(stats['general_draws'], 120)
        self.assertEqual(sum(r['truth'] for r in rows), 90)
        self.assertEqual(sum(r['truth'] for r in rows if r['stratum'] == 'general'), 60)
        for r in rows:
            b = r['boxes'][0]
            self.assertTrue(-.9 <= b['x'] <= .9 and .6 <= b['z'] <= 3.4 and .03 <= b['width'] <= .6)
            self.assertEqual(av.intersects_query(drift.scene(r)), r['truth'])
        for face in ('left', 'right', 'far'):
            selected = [r for r in rows if r['stratum'] == 'boundary_'+face]
            self.assertEqual((len(selected), sum(r['truth'] for r in selected)), (20, 10))
            for a, b in zip(selected[::2], selected[1::2]):
                self.assertEqual(a['pair'], b['pair'])
                self.assertEqual(a['boxes'][0]['width'], b['boxes'][0]['width'])
                coord = 'x' if face == 'far' else 'z'
                self.assertEqual(a['boxes'][0][coord], b['boxes'][0][coord])
        forbidden = drift.geometry_key(rows[0])
        self.assertNotIn(forbidden, {drift.geometry_key(r) for r in drift.generate([forbidden])})

    def test_linear_profiles_shared_per_view_bounded_sign_and_reverse(self):
        self.assertEqual(len(drift.CONDITIONS), 13)
        for family in drift.FAMILIES:
            up = [drift.injection(family+'_drift_up', i) for i in range(13)]
            down = [drift.injection(family+'_drift_down', i) for i in range(13)]
            for a, b in zip(up, reversed(down)):
                for key in a:
                    self.assertAlmostEqual(a[key], b[key], places=17)
            self.assertEqual(up[0], drift.injection(family+'_constant_minus', 0))
            self.assertEqual(up[-1], drift.injection(family+'_constant_plus', 12))
            self.assertTrue(all(v == 0 for v in up[6].values()))
            for a in up:
                self.assertLessEqual(abs(a['range_m']), .002)
                self.assertLessEqual(abs(a['pose_x_m']), .001)
                self.assertEqual(a['pose_x_m'], a['pose_z_m'])
        with self.assertRaises(ValueError): drift.injection('range_drift_up', 13)
        with self.assertRaises(ValueError): drift.injection('unknown', 0)

    def test_constant_controls_match_original_observer_exactly(self):
        value = av.Scene((av.Box(.34, 1.4, .08),))
        mapping = dict(nominal='nominal', range_constant_plus='range_plus2mm',
                       range_constant_minus='range_minus2mm', pose_constant_plus='pose_plus1mm',
                       pose_constant_minus='pose_minus1mm')
        for new, old in mapping.items():
            for row, camera in zip(drift.observe(value, new), drift.PATH):
                reference = bent.observe(value, camera, old)
                for key in ('bins', 'actual_camera', 'raw_ranges', 'biased_ranges'):
                    self.assertEqual(row[key], reference[key])

    def test_drift_precedes_quantization_and_public_has_no_metadata(self):
        value = av.Scene((av.Box(0., 1., .5),))
        with patch.object(av, 'hit_distance', return_value=1.049):
            views = drift.observe(value, 'range_drift_up')
        self.assertEqual(views[0]['bins'], [10]*8)
        self.assertEqual(views[-1]['bins'], [11]*8)
        for public in drift.public(views):
            self.assertEqual(set(public), {'camera', 'bins'})
        public = drift.public(views)
        public[0]['bins'][0] = 999
        self.assertEqual(views[0]['bins'][0], 10)

    def test_pose_drift_actual_path_is_recorded_and_not_equal_nominal_travel(self):
        value = av.Scene(())
        for condition in ('pose_drift_up', 'pose_drift_down'):
            views = drift.observe(value, condition)
            actual = sum(math.dist(a['actual_camera'], b['actual_camera']) for a, b in zip(views, views[1:]))
            self.assertNotAlmostEqual(actual, .12, places=6)
            for i, row in enumerate(views):
                bias = drift.injection(condition, i)
                self.assertEqual(row['actual_camera'], [round(row['camera'][0]+bias['pose_x_m'], 12),
                                                        round(row['camera'][1]+bias['pose_z_m'], 12)])

    def test_observation_does_not_read_query_truth(self):
        with patch.object(av, 'intersects_query', side_effect=AssertionError('No truth in capture')):
            for condition in drift.CONDITIONS:
                self.assertEqual(len(drift.observe(av.Scene((av.Box(.1, 2., .1),)), condition)), 13)


if __name__ == '__main__':
    unittest.main()
