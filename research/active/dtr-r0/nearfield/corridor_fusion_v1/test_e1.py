import unittest
import numpy as np
from depth_features import normalize, region
from run_e1 import select_threshold


class FeatureContract(unittest.TestCase):
    def test_relative_affine_invariance(self):
        d=np.arange(120,dtype=np.float32).reshape(10,12)
        self.assertEqual(normalize(d).dtype,np.float32)
        np.testing.assert_allclose(normalize(d),normalize(d*3+18),atol=1e-6)

    def test_outside_window_has_missing_mask(self):
        d=np.ones((10,12),np.float32)
        self.assertFalse(region(d,d,d,[30,30,50,50]).any())
        self.assertEqual(region(d,d,d,[0,0,12,10])[0],1)

    def test_high_recall_threshold_differs_from_f1(self):
        y=np.array([1,1,1,0,0,0,0],bool)
        scores=np.array([.9,.8,.1,.7,.6,.5,.4])
        f1,_=select_threshold(y,scores)
        hi,_=select_threshold(y,scores,True)
        self.assertEqual(f1['threshold'],.8)
        self.assertEqual(hi['threshold'],.1)
        self.assertEqual(hi['recall'],1.)

if __name__=='__main__':unittest.main()
