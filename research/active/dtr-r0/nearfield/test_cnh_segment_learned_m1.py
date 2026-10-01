import unittest
import numpy as np
import torch
from cnh_segment_learned_m1 import (linear_statistics, weights_for_coverage,
                                     reduce_queries, cohort, WINDOWS_R)


class M1Test(unittest.TestCase):
    def test_exact_transport_covariance_not_diagonal(self):
        rng=np.random.default_rng(1823)
        weights=rng.normal(size=(7,5))
        variance=rng.uniform(.2,2,size=5)
        matrices=[rng.uniform(0,1,size=(5,5)) for _ in range(3)]
        old=[rng.uniform(.2,3,size=5) for _ in matrices]
        residual=rng.normal(size=5)
        past_residual=[rng.normal(size=5) for _ in matrices]
        total=residual+sum(a@r for a,r in zip(matrices,past_residual))
        cov=np.diag(variance)+sum((a*v)@a.T for a,v in zip(matrices,old))
        tt=lambda x:torch.tensor(x,dtype=torch.float64)
        m,v=linear_statistics(tt(weights),tt(total),tt(variance),[(tt(a),tt(v)) for a,v in zip(matrices,old)])
        np.testing.assert_allclose(m.numpy(),weights@total,rtol=1e-12)
        exact=np.einsum('bi,ij,bj->b',weights,cov,weights)
        np.testing.assert_allclose(v.numpy(),exact,rtol=1e-12)
        self.assertGreater(np.max(np.abs(exact-(weights*weights)@np.diag(cov))),1.)

    def test_range_weights_query_and_unknown(self):
        coverage=np.zeros((8,8)); coverage[3,4]=.25; coverage[4,4]=.5
        w=weights_for_coverage(coverage).reshape(-1,8,8,16)
        for k,(b,width) in enumerate(WINDOWS_R):
            self.assertEqual(w[k,:,:,0].sum(),0)
            self.assertEqual(w[k].sum(),.75*width)
            np.testing.assert_allclose(w[k,:,:,b],coverage)
        membership=np.zeros((len(WINDOWS_R),6),bool)
        membership[2,0]=True; membership[4,0]=True; membership[2,1]=True
        z=np.arange(len(WINDOWS_R),dtype=float)-20
        score=reduce_queries(z,membership)
        self.assertEqual(score[0],-16)
        self.assertEqual(score[1],-18)
        self.assertTrue(np.isneginf(score[2:]).all())

    def test_correct_dataset_cohorts_and_validation(self):
        self.assertEqual(len(cohort('v2')),95)
        self.assertNotIn(143,cohort('v2'))
        self.assertEqual(cohort('v4'),list(range(96)))
        with self.assertRaises(ValueError): weights_for_coverage(np.ones((8,8))*1.01)
        with self.assertRaises(ValueError): weights_for_coverage(np.zeros((8,7)))

if __name__=='__main__': unittest.main()
