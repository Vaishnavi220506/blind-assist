import unittest
import numpy as np
from ba_rpcc_o0 import select

class RpccTests(unittest.TestCase):
    def test_count_clips_and_stable_order(self):
        d=np.array([2.,1.,1.,3.])
        self.assertEqual(select(d,0).sum(),0);self.assertEqual(select(d,9).sum(),4)
        self.assertTrue(select(d,2)[1:3].all())

if __name__=='__main__':unittest.main()
