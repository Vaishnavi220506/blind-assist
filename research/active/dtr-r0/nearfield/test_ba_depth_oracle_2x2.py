import unittest
import numpy as np
from ba_depth_oracle_2x2 import arms_1d,bands


class OracleTests(unittest.TestCase):
    def test_exact_ceilings(self):
        for pred in (np.array([1.,1.,1.,3.]),np.array([3.,3.,1.,3.])):
            gt=np.array([1.,3.,1.,3.]);a=arms_1d(pred,gt)
            k=int((pred<2).sum());g=int((gt<2).sum())
            self.assertEqual(a['mass_only'].sum(),g)
            self.assertEqual(a['placement_only'].sum(),k)
            self.assertEqual((a['placement_only']&(gt<2)).sum(),min(k,g))
            self.assertEqual((a['placement_only']^(gt<2)).sum(),abs(k-g))

    def test_decimal_band_endpoints(self):
        gt=np.array([1.95,1.9,1.8,1.79],np.float32)
        pred=np.array([2.05,2.1,2.2,2.21],np.float32)
        self.assertTrue(np.array_equal(bands(gt,pred,np.ones(4,bool)),np.eye(4,dtype=int)))


if __name__=='__main__':unittest.main()
