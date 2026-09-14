"""No fold preprocessing leakage; midpoint corrects shifts rather than order."""
import unittest
import torch
from fit_mz136_grouped_readout import fit_linear, midpoint_loss, grouped_folds


class GroupedReadoutTests(unittest.TestCase):
    def test_midpoint_penalizes_shared_offset_at_identical_pair_margin(self):
        pair=torch.tensor([[0,1]])
        centered=torch.tensor([1.,-1.])
        shifted=torch.tensor([10.,8.],requires_grad=True)
        self.assertEqual(float(midpoint_loss(centered,pair)),0.)
        value=midpoint_loss(shifted,pair)
        self.assertEqual(float(value.detach()),81.)
        value.backward()
        torch.testing.assert_close(shifted.grad,torch.tensor([9.,9.]))

    def test_held_features_do_not_change_fit_or_normalization(self):
        torch.set_num_threads(2);torch.manual_seed(136014)
        features=torch.randn(6,4)
        changed=features.clone();changed[4:]+=10000
        base=torch.zeros(6);target=torch.tensor([1.,0.,1.,0.,1.,0.])
        pairs=[dict(a=0,b=1),dict(a=2,b=3)]
        a=fit_linear(features,base,target,[0,1,2,3],pairs,.01,.1,'cpu')
        b=fit_linear(changed,base,target,[0,1,2,3],pairs,.01,.1,'cpu')
        for key in ('mean','scale'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.testing.assert_close(a['head'].weight,b['head'].weight,rtol=0,atol=0)
        torch.testing.assert_close(a['head'].bias,b['head'].bias,rtol=0,atol=0)

    def test_complete_scene_and_pair_groups_remain_together(self):
        pairs=[]
        for family in range(4):
            for scene in range(4):
                for t in range(6):
                    i=len(pairs)*2
                    pairs.append(dict(a=i,b=i+1,scene_group=f'family{family}_scene{scene}'))
        folds=grouped_folds(pairs)
        for train,held,_ in folds:
            self.assertEqual(len(train),144);self.assertEqual(len(held),48)
            for p in pairs:self.assertEqual(p['a'] in held,p['b'] in held)


if __name__=='__main__':unittest.main()
