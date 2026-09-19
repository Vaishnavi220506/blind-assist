import unittest
import numpy as np
import torch
import ba_nfo_matched as m
from ba_nfo_targetfit import target_loss,target_masks


class TestTargetLoss(unittest.TestCase):
    def test_full_support_same_loss_and_gradient(self):
        torch.manual_seed(7)
        a=torch.randn(2,4,3,5,requires_grad=True);b=a.detach().clone().requires_grad_()
        depth=torch.full((2,3,5),3.);depth[:,:1]=1.2;depth[0,0,0]=float('nan')
        old=m.loss_fn(a,depth,'nfo');new=target_loss(b,depth,torch.ones_like(depth,dtype=torch.bool))
        old.backward();new.backward()
        self.assertTrue(torch.equal(old,new));self.assertTrue(torch.equal(a.grad,b.grad))

    def test_outside_and_unknown_zero_output_gradient(self):
        torch.manual_seed(8)
        out=torch.randn(2,4,4,5,requires_grad=True)
        depth=torch.full((2,4,5),3.);depth[0,1,1]=1.;depth[1,2,2]=float('nan')
        mask=torch.from_numpy(target_masks([dict(box=[1,1,3,3]),dict(box=[1,1,3,3])],(4,5)))
        loss=target_loss(out,depth,mask);loss.backward()
        valid=mask&torch.isfinite(depth)
        self.assertTrue(torch.all(out.grad.permute(0,2,3,1)[~valid]==0))
        self.assertTrue(torch.all(out.grad.permute(0,2,3,1)[valid].abs().sum(-1)>0))
        altered=depth.clone();altered[~mask]=.3
        self.assertTrue(torch.equal(loss,target_loss(out.detach(),altered,mask)))
        self.assertEqual(int(mask.sum()),8)


if __name__=='__main__':unittest.main()
