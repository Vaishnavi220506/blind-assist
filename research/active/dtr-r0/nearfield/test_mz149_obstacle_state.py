import unittest
import numpy as np
from mz149_obstacle_state import fit_parameters,filter_scores


class StateTests(unittest.TestCase):
    def test_transition_counts_respect_episode_and_gap(self):
        rows=[dict(episode_id=e,time_s=t) for e,t in [('a',0),('a',.25),('a',.5),('b',.75),('b',1.5)]]
        p=fit_parameters(rows,[True,True,False,False,True])
        self.assertEqual(p['transition_counts'],[[0,0],[1,1]])
        self.assertEqual(p['episode_start_counts'],[1,2])

    def test_persistence_prefix_causality_and_reset(self):
        p=dict(transition=[[.98,.02],[.02,.98]],initial_positive=.5,training_positive_prior=.5)
        rows=[dict(episode_id='a',time_s=i*.25) for i in range(6)]
        scores=np.array([.95,.93,.04,.01,.01,.01]);whole=filter_scores(rows,scores,p)
        self.assertGreater(whole[2],scores[2]);self.assertLess(whole[-1],.05)
        for i in range(1,7):np.testing.assert_array_equal(filter_scores(rows[:i],scores[:i],p),whole[:i])
        rows[2]=dict(episode_id='b',time_s=.5,truth=True,native_bounds=[99])
        self.assertAlmostEqual(filter_scores(rows,scores,p)[2],scores[2])


if __name__=='__main__':unittest.main()
