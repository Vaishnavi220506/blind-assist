"""Task-loss invariants and exact paired schedule, without expensive fitting."""
import unittest
import numpy as np
import torch
from mz136_corridor_pair import CorridorNet, corridor_rank
from mz120_occupancy import OccupancyNet
from run_mz136_corridor_pair import make_schedule


class CorridorPairTests(unittest.TestCase):
    def test_correct_order_absolute_classification_distinct(self):
        y=torch.tensor([1.,0.,0.,1.])
        good=torch.tensor([2.,-2.,-2.,2.],requires_grad=True)
        self.assertEqual(float(corridor_rank(good,y)),0.)
        wrong=-good
        self.assertEqual(float(corridor_rank(wrong,y)),5.)
        # Ranking alone cannot repair two positive outputs.
        both_positive=torch.tensor([5.,3.,3.,5.])
        self.assertEqual(float(corridor_rank(both_positive,y)),0.)
        self.assertGreater(float(torch.nn.functional.binary_cross_entropy_with_logits(both_positive,y)),1.)

    def test_unchanged_pair_finite_zero_gradient(self):
        s=torch.tensor([2.,-3.],requires_grad=True)
        v=corridor_rank(s,torch.tensor([1.,1.])); v.backward()
        self.assertEqual(float(v),0.)
        self.assertTrue(torch.equal(s.grad,torch.zeros_like(s)))

    def test_same_capacity_and_schedule(self):
        a=OccupancyNet(); b=CorridorNet()
        self.assertEqual(sum(p.numel() for p in a.parameters()),11305)
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        pairs=[dict(a=0,b=1),dict(a=2,b=3)]
        s,aug=make_schedule(pairs,20); t,other=make_schedule(pairs,20)
        np.testing.assert_array_equal(s,t);np.testing.assert_array_equal(aug,other)
        np.testing.assert_array_equal(s[:,1::2],s[:,::2]+1)
        np.testing.assert_array_equal(aug[:,:,::2],aug[:,:,1::2])


if __name__=='__main__':unittest.main()
