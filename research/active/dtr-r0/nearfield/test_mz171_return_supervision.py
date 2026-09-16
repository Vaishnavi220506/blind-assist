import unittest
import numpy as np
import torch
from mz171_return_supervision import slot_weights,slot_report,loss


class ReturnSupervisionTests(unittest.TestCase):
    def test_known_class_balance_and_unknown(self):
        t=torch.zeros(3,132);k=torch.zeros_like(t,dtype=torch.bool)
        t[0,0]=1;k[0,:5]=True;k[1,:3]=True
        w=slot_weights(t,k)
        torch.testing.assert_close(w.sum(1),torch.tensor([1.,1.,0.]))
        self.assertEqual(float(w[0,0]),.5)
        self.assertEqual(float(w[0,1:5].sum()),.5)
        self.assertEqual(float(w[~k].sum()),0.)
        k[0,128]=True
        with self.assertRaises(ValueError):slot_weights(t,k)

    def test_losing_return_branch_gets_only_direct_gradients(self):
        tok=torch.full((1,132),-1.,requires_grad=True)
        pix=torch.full((1,2,2),2.,requires_grad=True)
        out=dict(token=tok,pixel=pix,frame=torch.maximum(pix.flatten(1).amax(1),tok.amax(1)))
        target=torch.zeros(1,132);target[0,0]=1
        known=torch.zeros(1,132,dtype=torch.bool);known[0,:2]=True
        args=(out,torch.ones(1),torch.ones(1,2,2),torch.full((1,2,2),.25),target,known)
        a,_=loss(*args,'control');a.backward(retain_graph=True)
        self.assertEqual(float(tok.grad.abs().sum()),0.)
        tok.grad.zero_();b,_=loss(*args,'witness');b.backward()
        self.assertLess(float(tok.grad[0,0]),0.)
        self.assertGreater(float(tok.grad[0,1]),0.)
        self.assertEqual(float(tok.grad[0,2:].abs().sum()),0.)

    def test_slot_report_does_not_label_radar_or_missing_as_negative(self):
        z=np.full((1,132),-1.);t=np.zeros((1,132));k=np.zeros((1,132),bool)
        t[0,0]=1;k[0,:2]=True;z[0,[0,3,128]]=1
        r=slot_report(z,t,k,[0])
        self.assertEqual(r['counts'],dict(TP=1,FP=0,FN=0,TN=1))
        self.assertEqual(r['unknown_positive_tof_slots'],1)
        self.assertEqual(r['unknown_positive_radar_slots'],1)
        self.assertEqual(r['balanced_accuracy'],1.)


if __name__=='__main__':unittest.main()
