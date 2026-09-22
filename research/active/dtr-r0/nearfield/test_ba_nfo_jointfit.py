import unittest
import torch
import ba_nfo_matched as m
from ba_nfo_targetfit import target_loss
from ba_nfo_jointfit import joint_loss


class TestJointLoss(unittest.TestCase):
    def test_half_gradients_and_unknown(self):
        torch.manual_seed(19)
        logits=torch.randn(2,4,3,5,requires_grad=True)
        depth=torch.full((2,3,5),3.);depth[0,1,1]=1.;depth[1,0,0]=float('nan')
        mask=torch.zeros_like(depth,dtype=torch.bool);mask[:,1:,:3]=True
        loss,full,local=joint_loss(logits,depth,mask)
        gf=torch.autograd.grad(full,logits,retain_graph=True)[0]
        gl=torch.autograd.grad(local,logits,retain_graph=True)[0]
        gj=torch.autograd.grad(loss,logits)[0]
        torch.testing.assert_close(gj,.5*gf+.5*gl)
        outside=~mask&torch.isfinite(depth)
        torch.testing.assert_close(gj.permute(0,2,3,1)[outside],.5*gf.permute(0,2,3,1)[outside])
        self.assertTrue(torch.all(gj.permute(0,2,3,1)[~torch.isfinite(depth)]==0))
        self.assertTrue(torch.equal(loss,.5*full+.5*local))

    def test_full_mask_recovers_original_loss(self):
        torch.manual_seed(20)
        logits=torch.randn(2,4,3,5,requires_grad=True)
        depth=torch.rand(2,3,5)*3+.1
        loss,_,_=joint_loss(logits,depth,torch.ones_like(depth,dtype=torch.bool))
        self.assertTrue(torch.equal(loss,m.loss_fn(logits,depth,'nfo')))


if __name__=='__main__':unittest.main()
