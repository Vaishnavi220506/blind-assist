import unittest

import numpy as np
import torch
from torch.nn import functional as F

import ba_nfo_fullsupport as s


class FullSupportTest(unittest.TestCase):
    def test_whole_zone_one_pixel_boundary_missing_and_unknown(self):
        depth=np.full((2,15),3.,np.float32)
        depth[0,0]=1.999; depth[0,5]=1.2; depth[1,14]=np.nan
        a=dict(depth=depth,boxes=np.array([[0,0,2,5],[0,5,2,10],[0,10,2,15]]),values=np.array([3.,np.nan,3.]))
        truth,dm=s.domains(a)
        self.assertEqual(int(dm['small_foreground'].sum()),20)
        self.assertEqual(int((dm['small_foreground']&truth).sum()),2)
        self.assertEqual(int(dm['pure_far'].sum()),9)
        self.assertEqual(int(dm['far_small'].sum()),10)
        self.assertFalse(dm['full'][1,14])
        self.assertFalse((dm['small_foreground']&dm['pure_far']).any())

    def test_loss_and_gradient_match_explicit_weighted_pixels(self):
        torch.manual_seed(5)
        output=torch.randn(2,4,2,3,requires_grad=True)
        depth=torch.tensor([[[1.99,3.,float('nan')],[1.,4.,2.]],[[3.,3.,3.],[3.,3.,3.]]])
        small=torch.zeros_like(depth,dtype=torch.bool); small[0,0,:]=True
        pure=torch.zeros_like(small); pure[1]=True
        actual=s.support_loss(output,depth,small,pure)
        valid=torch.isfinite(depth)&(depth>0)
        target=(torch.nan_to_num(depth,nan=1.)[:,None]<torch.tensor([1.,1.5,2.,3.])[None,:,None,None]).float()
        pixel=F.binary_cross_entropy_with_logits(output,target,reduction='none').mean(1)
        pixel=pixel+.2*F.relu(output.sigmoid()[:,:-1]-output.sigmoid()[:,1:]).mean(1)
        weights=.5*valid/valid.sum()+.25*(small&valid)/(small&valid).sum()+.25*(pure&valid)/(pure&valid).sum()
        expected=(pixel*weights).sum()
        torch.testing.assert_close(actual,expected)
        g1=torch.autograd.grad(actual,output,retain_graph=True)[0]
        g2=torch.autograd.grad(expected,output)[0]
        torch.testing.assert_close(g1,g2)
        self.assertTrue(torch.equal(g1[0,:,0,2],torch.zeros(4)))
        empty=torch.zeros_like(small)
        torch.testing.assert_close(s.support_loss(output,depth,empty,empty),s.m.loss_fn(output,depth,'nfo'))


if __name__=='__main__': unittest.main()
