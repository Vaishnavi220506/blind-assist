import unittest
import numpy as np
from cnh_v5_evaluate import threshold, guard_ranking, alert_stats


class V5MetricsTests(unittest.TestCase):
    def test_calibration_budget_and_ties(self):
        peaks = np.arange(100, dtype=np.float32)
        self.assertEqual(int((peaks.astype(float) > threshold(peaks, .01, strict=True)).sum()), 1)
        self.assertEqual(int((peaks.astype(float) >= threshold(peaks, .01)).sum()), 1)
        tied = np.ones(100)
        self.assertEqual(int((tied >= threshold(tied, .01)).sum()), 0)

    def test_or_ranking(self):
        values = np.array([-10., 0., 10., 100.])
        r = guard_ranking(values, np.array([True, False, False, False]))
        self.assertTrue(r[0] > r[3] > r[2] > r[1])

    def test_false_episodes_merge_overlapping_queries(self):
        p = dict(y=np.zeros((1, 9, 3)), w=np.full((1,9,3), np.nan), strata=np.full((1,9,3),'none'))
        alarms = np.zeros((1,9,3), bool)
        alarms[0,0:2,0] = True
        alarms[0,1:3,1] = True
        alarms[0,5,2] = True
        r = alert_stats(p, alarms)
        self.assertEqual(r['false_pairs'], 3)
        self.assertEqual(r['false_episodes'], 2)
        self.assertEqual(r['empty_group_frames'], 9)
        self.assertAlmostEqual(r['empty_group_minutes'], .03)
        self.assertAlmostEqual(r['false_episodes_per_simulated_empty_minute'], 2/.03)

    def test_first_positive_hit_defines_timeliness(self):
        p = dict(y=np.zeros((1,9,3)), w=np.full((1,9,3), np.nan), strata=np.full((1,9,3),'none'))
        p['y'][0,0:3,0] = 1
        p['w'][0,0:3,0] = [2., .8, 1.2]
        p['strata'][0,0:3,0] = 'tiny'
        alarms = np.zeros((1,9,3), bool)
        alarms[0,1:3,0] = True
        r = alert_stats(p, alarms)
        self.assertEqual(r['all']['near'], 1)
        self.assertEqual(r['all']['timely_count'], 0)
        self.assertEqual(r['all']['late_count'], 1)
        self.assertEqual(r['empty_group_sequences'], 0)
        self.assertIsNone(r['false_episodes_per_simulated_empty_minute'])


if __name__ == '__main__':
    unittest.main()
