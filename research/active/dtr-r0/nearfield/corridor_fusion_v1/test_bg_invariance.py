import unittest
import numpy as np
import torch
from bg_invariance import RawResidual, loss
from ccrl import local_pairs


class ObjectiveTests(unittest.TestCase):
    def test_zero_start_preserves_baseline(self):
        m=RawResidual(np.zeros(2224),np.ones(2224))
        b=torch.tensor([-.2,.7]);x=torch.randn(2,2224)
        torch.testing.assert_close(m(x,b),b,rtol=0,atol=0)

    def test_invariance_gradient_closes_gap(self):
        a=torch.tensor([2.,-1.],requires_grad=True);y=torch.ones(2)
        empty=torch.empty((0,2),dtype=torch.long);bg=torch.tensor([[0,1]])
        delta=loss(a,y,empty,bg,'B2')-loss(a,y,empty,bg,'B1')
        delta.backward()
        self.assertGreater(float(a.grad[0]),0)
        self.assertLess(float(a.grad[1]),0)

    def test_rank_gradient_opens_gap(self):
        a=torch.tensor([0.,0.],requires_grad=True);y=torch.tensor([1.,0.])
        empty=torch.empty((0,2),dtype=torch.long);pair=torch.tensor([[0,1]])
        (loss(a,y,pair,empty,'B1')-loss(a,y,pair,empty,'B0')).backward()
        self.assertLess(float(a.grad[0]),0)
        self.assertGreater(float(a.grad[1]),0)

    def test_cross_split_pairs_removed(self):
        p=np.array([[0,1],[2,3],[1,3]])
        np.testing.assert_array_equal(local_pairs(p,np.array([2,3])),[[0,1]])


if __name__=='__main__':unittest.main()
