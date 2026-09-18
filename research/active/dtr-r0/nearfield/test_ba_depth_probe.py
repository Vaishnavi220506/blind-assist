import unittest
import numpy as np
from ba_depth_probe import expand_zones, mixed_mask, boundary_mask, counts, rates


class ProbeTests(unittest.TestCase):
    def test_missing_and_clipped_rectangle(self):
        p=expand_zones(np.array([[1.,0.],[2.,0.]]),np.array([[-2,-2,2,2],[2,2,4,4]]),[True,False],(4,4))
        self.assertEqual(np.isfinite(p).sum(),4)
        c=rates(counts(p,np.ones((4,4)),np.ones((4,4),bool),2))
        self.assertEqual(c['unknown_positive'],12)
        self.assertEqual(c['fn'],0)
        self.assertEqual(c['positive_detection_fraction'],.25)

    def test_mixed_union_and_invalid_boundary(self):
        gt=np.ones((10,10));gt[:,5:]=4
        valid=np.ones_like(gt,bool)
        m,n=mixed_mask(gt,[[0,0,10,10],[0,0,10,10]],valid)
        self.assertEqual(n,2)
        self.assertEqual(m.sum(),100)
        valid[:,5:]=False
        self.assertEqual(boundary_mask(gt,valid,5).sum(),0)

    def test_reference_identity(self):
        gt=np.array([[1.,3.]])
        c=rates(counts(gt,gt,np.ones_like(gt,bool),2))
        self.assertEqual(c['iou'],1)
        self.assertEqual(c['tp'],1)
        self.assertEqual(c['tn'],1)


if __name__=='__main__':unittest.main()
