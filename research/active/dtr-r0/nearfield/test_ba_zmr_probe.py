import unittest
import numpy as np
import cv2
from ba_zmr_probe import rank_mask,connected_mask


class ZmrTests(unittest.TestCase):
    def test_exact_mass_and_stable_ties(self):
        depth=np.ones((5,6));rgb=np.zeros((5,6,3),np.uint8)
        for k in (0,1,11,30):
            a=rank_mask(depth,k);b=connected_mask(depth,rgb,k)
            self.assertEqual(a.sum(),k);self.assertEqual(b.sum(),k)
            self.assertTrue(a.ravel()[:k].all())
            if k:self.assertEqual(cv2.connectedComponents(b.astype(np.uint8),connectivity=4)[0],2)

    def test_ordering_and_disjoint_components(self):
        d=np.array([[1.,8.,1.],[8.,8.,8.]])
        r=rank_mask(d,2)
        self.assertTrue(r[0,0] and r[0,2])
        c=connected_mask(d,np.zeros((2,3,3),np.uint8),2)
        self.assertFalse(c[0,0] and c[0,2])


if __name__=='__main__':unittest.main()
