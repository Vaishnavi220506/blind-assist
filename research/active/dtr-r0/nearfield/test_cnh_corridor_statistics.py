import unittest
import numpy as np
from cnh_corridor_statistics import paired_cluster_ber


class BootstrapTests(unittest.TestCase):
    def test_negative_only_cluster_retained(self):
        r=paired_cluster_ber([1,0,0,0],[1,0,0,0],[1,0,1,1],[10,10,20,20],draws=1000)
        self.assertEqual(r['units'],2)
        self.assertAlmostEqual(r['point'],1/3)
        self.assertGreater(r['ci'][1],0)
        self.assertGreater(r['excluded_draws'],0)
        self.assertEqual(r['valid_draws']+r['excluded_draws'],1000)

    def test_point_matches_pooled_and_pairing_permutation(self):
        y=np.array([1,0,1,1,0,0,0]);b=np.array([0,0,1,1,1,0,0]);m=np.array([1,1,0,1,0,0,1]);u=np.array([1,1,2,2,3,3,3])
        r=paired_cluster_ber(y,b,m,u,draws=1000)
        err=lambda p:.5*((y&~p.astype(bool)).sum()/y.sum()+((1-y)&p).sum()/(1-y).sum())
        self.assertAlmostEqual(r['point'],err(m)-err(b))
        ix=np.array([6,2,4,0,5,1,3])
        self.assertEqual(r,paired_cluster_ber(y[ix],b[ix],m[ix],u[ix],draws=1000))

    def test_identical_predictions_zero_interval(self):
        r=paired_cluster_ber([1,0,1,0],[1,0,0,1],[1,0,0,1],[0,0,1,1])
        self.assertEqual(r['point'],0)
        self.assertEqual(r['ci'],[0.,0.])

    def test_missing_class_empty_and_invalid(self):
        for y,p,u in [([0,0],[1,0],[0,1]),([],[],[])]:
            r=paired_cluster_ber(y,p,p,u)
            self.assertIsNone(r['point']);self.assertIsNone(r['ci'])
        with self.assertRaises(ValueError):paired_cluster_ber([2],[0],[0],[0])
        with self.assertRaises(ValueError):paired_cluster_ber([0],[0],[0],[0],draws=0)


if __name__=='__main__':unittest.main()
