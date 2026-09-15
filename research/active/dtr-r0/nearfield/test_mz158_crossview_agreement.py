import copy
import unittest
import cv2
import numpy as np
from mz148_background_residual import CausalResidual
from mz158_crossview_agreement import RawResidual,decisions


class AgreementTests(unittest.TestCase):
    def test_raw_adapter_parity_prefix_reset_and_metadata(self):
        rng=np.random.default_rng(158016);image=rng.integers(0,256,(72,128,3),dtype=np.uint8)
        rows=[dict(episode_id='a' if i<6 else 'b',time_s=(i if i<4 else i+1)*.25) for i in range(8)]
        raw=RawResidual();old=CausalResidual();snapshots=[]
        for i,row in enumerate(rows):
            frame=np.roll(image,i,axis=1);before=copy.deepcopy(row);pixels=frame.copy()
            a=raw.update(row,frame);b=old.update(row,frame)['unregistered']
            np.testing.assert_array_equal(a,b);self.assertEqual(row,before)
            np.testing.assert_array_equal(frame,pixels);snapshots.append(a)
        again=RawResidual()
        for i,row in enumerate(rows[:5]):
            contaminated=dict(row,truth=True,family='secret',native_bounds=['hidden'])
            np.testing.assert_array_equal(again.update(contaminated,np.roll(image,i,axis=1)),snapshots[i])

    def test_agreement_after_each_causal_rule(self):
        class Model:
            def __init__(self,p):self.p=np.array(p)
            def predict_proba(self,x):return np.c_[1-self.p,self.p]
        rows=[dict(episode_id='a',time_s=i*.25) for i in range(4)]
        models={'static':Model([.8,.3,.8,.3]),'raw_change':Model([.3,.3,.8,.1])}
        cuts={key:dict(low=.2,high=.7) for key in models}
        scores,flags=decisions(rows,np.zeros((4,2)),np.zeros((4,3)),models,cuts)
        np.testing.assert_array_equal(flags['agreement'],[False,True,True,False])
        self.assertTrue(np.all(~flags['agreement']|flags['static']))
        self.assertTrue(np.all(~flags['agreement']|flags['raw_change']))


if __name__=='__main__':unittest.main()
