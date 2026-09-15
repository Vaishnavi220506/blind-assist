import unittest
import numpy as np
from mz145_causal_confirmation import predict,fit_onset


class CausalTests(unittest.TestCase):
    def rows(self):return [dict(episode_id='a',time_s=.25*i) for i in range(5)]

    def test_strong_immediate_weak_confirmation_and_no_future(self):
        rows=self.rows();scores=[.6,.15,.8,.12,.1]
        flags=predict(rows,scores,.1,.7)
        np.testing.assert_array_equal(flags,[False,True,True,True,True])
        for length in range(1,6):
            np.testing.assert_array_equal(predict(rows[:length],scores[:length],.1,.7),flags[:length])

    def test_episode_and_gap_reset(self):
        rows=self.rows();rows[2]['episode_id']='b';rows[3]['episode_id']='b';rows[3]['time_s']=2.
        np.testing.assert_array_equal(predict(rows,[.2]*5,.1,.7),[False,True,False,False,False])

    def test_train_cutoff_retains_starts_and_later_weak_true_frames(self):
        rows=self.rows();s=[.8,.15,.05,.6,.12];y=[True,True,False,True,True]
        high=fit_onset(rows,s,y,y,.1)
        self.assertEqual(high,.6)
        np.testing.assert_array_equal(predict(rows,s,.1,high),y)


if __name__=='__main__':unittest.main()
