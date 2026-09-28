import unittest
import numpy as np
from cnh_proposal_attribution import point_score, binary_metrics, episode_threshold, episodes, threshold


class AttributionChecks(unittest.TestCase):
    def test_eight_independent_exposures_signal_normalization(self):
        h=np.full((8,8,8,16),10.);a=np.ones((8,8,8));b=np.zeros((8,8,16))
        poses=np.repeat(np.eye(4)[None],8,axis=0)
        score,coverage=point_score(h,a,b,poses,np.eye(4))
        np.testing.assert_allclose(score,np.sqrt(8)*10/4,rtol=1e-12)
        self.assertTrue((coverage>0).all())

    def test_common_world_rotation_does_not_change_motion_scores(self):
        h=np.random.default_rng(42).normal(size=(8,8,8,16));a=np.ones((8,8,8));b=np.zeros((8,8,16))
        poses=np.repeat(np.eye(4)[None],8,axis=0);poses[:,2,3]=np.linspace(-1,0,8)
        turn=np.array([[0,0,1,2],[0,1,0,3],[-1,0,0,4],[0,0,0,1.]])
        x=point_score(h,a,b,poses,np.eye(4))[0]
        y=point_score(h,a,b,turn@poses,np.eye(4))[0]
        np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)

    def test_threshold_ties_cannot_exceed_calibration_budget(self):
        values=np.r_[np.ones(9),2.]
        self.assertLessEqual(int((values>=threshold(values)).sum()),1)

    def test_episode_scan_rejects_always_on_shortcut(self):
        y=np.zeros((10,9),bool);s=np.tile(np.arange(9),(10,1))
        t=episode_threshold(y,s)
        self.assertLessEqual(episodes(y,s,t)['episodes_per_min'],2)
        self.assertGreater(t,s.max())

    def test_error_denominator_closes(self):
        m=binary_metrics([1,1,0,0],[1,0,1,0])
        self.assertEqual(m['errors'],2);self.assertEqual(m['ber'],.5)
        self.assertEqual(m['positive'],m['tp']+m['fn'])
        self.assertEqual(m['negative'],m['tn']+m['fp'])


if __name__=='__main__':unittest.main()
