import unittest
import numpy as np
import torch
import ba_nfo_matched as m
from ba_nfo_zone_readout import ZoneReadout,public_boxes
from ba_nfo_local_diagnostic import auc
from ba_nfo_conditional_diagnostic import curve


class ZoneReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        self.net=ZoneReadout(m.Net('nfo'))
        values=np.full(64,3.,np.float32);values[1]=np.nan;values[2]=1.
        self.z=torch.from_numpy(m.public_zones(values)[None])
        self.raw=torch.randn(1,4,m.H,m.W)

    def test_zero_identity_and_frozen_backbone(self):
        out,_=self.net.adjust(self.raw,self.z)
        self.assertTrue(torch.equal(out,self.raw))
        self.net.train()
        self.assertFalse(self.net.base.training)
        self.assertTrue(all(not p.requires_grad for p in self.net.base.parameters()))
        self.assertEqual(sum(p.numel() for p in self.net.parameters() if p.requires_grad),8)

    def test_gate_and_shared_offset_preserve_zone_order(self):
        with torch.no_grad():self.net.readout.bias.fill_(.4)
        out,offset=self.net.adjust(self.raw,self.z)
        self.assertEqual(offset[0,1],0);self.assertEqual(offset[0,2],0)
        self.assertTrue(torch.equal(out[:,:,0,:],self.raw[:,:,0,:]))
        y0,x0,y1,x1=public_boxes()[0]
        before=torch.cummax(self.raw,dim=1).values[0,2,y0:y1,x0:x1].flatten()
        after=torch.cummax(out,dim=1).values[0,2,y0:y1,x0:x1].flatten()
        torch.testing.assert_close(after-before,torch.full_like(before,float(offset[0,0].detach())))
        self.assertTrue(torch.equal(before.argsort(),after.argsort()))
        scores=m.probabilities(out,'nfo')
        self.assertTrue(torch.all(scores[:,:-1]<=scores[:,1:]))

    def test_gradient_only_readout(self):
        out,_=self.net.adjust(self.raw,self.z)
        d=torch.full((1,m.H,m.W),4.)
        d[:,50:80,50:80]=1.;d[:,:10]=float('nan')
        m.loss_fn(out,d,'nfo').backward()
        self.assertGreater(float(self.net.readout.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in self.net.base.parameters()))

    def test_auc_ties(self):
        self.assertAlmostEqual(auc(curve(np.array([.5,.5]),np.array([True,False]))),.5)
        self.assertAlmostEqual(auc(curve(np.array([.9,.1]),np.array([True,False]))),1.)


if __name__=='__main__':unittest.main()
