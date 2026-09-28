import inspect
import unittest
import numpy as np
import torch
import cnh_subzone_aux as A

class SubzoneTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_no_surface_and_simple_plane(self):
        poses=np.eye(4)[None]
        empty=A.targets_for_geometry([],poses)
        np.testing.assert_array_equal(empty[...,16],1)
        np.testing.assert_array_equal(empty[...,:16],0)
        boxes=[dict(lo=[-8,-8,1.],hi=[8,8,1.1],rho=.3)]
        target=A.targets_for_geometry(boxes,poses)
        np.testing.assert_allclose(target[...,3],1,atol=1e-6)
        np.testing.assert_allclose(target.sum(-1),1,atol=1e-6)
    def test_target_independent_of_danger_and_reflectance(self):
        self.assertEqual(list(inspect.signature(A.targets_for_geometry).parameters),['boxes','poses'])
        a=[dict(lo=[-.4,-.2,1.2],hi=[.4,.2,1.4],rho=.2)]
        b=[dict(a[0],rho=.9)]
        np.testing.assert_array_equal(A.targets_for_geometry(a,np.eye(4)[None]),A.targets_for_geometry(b,np.eye(4)[None]))
    def test_same_initial_readout_and_export(self):
        torch.manual_seed(2);plain=A.L.Readout().eval()
        torch.manual_seed(2);aux=A.AuxiliaryReadout().eval()
        for k,v in plain.state_dict().items():torch.testing.assert_close(v,aux.readout.state_dict()[k],rtol=0,atol=0)
        x=torch.randn(2,2,8,8,16);sup=torch.ones(2,6,8,8,16,dtype=torch.bool)
        with torch.no_grad():
            expected=plain(x,sup);actual,logits=aux(x,sup)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        self.assertEqual(logits.shape,(2,4,8,8,17))
        self.assertEqual(set(aux.readout.state_dict()),set(plain.state_dict()))
    def test_auxiliary_gradient_reaches_shared_features(self):
        aux=A.AuxiliaryReadout()
        x=torch.randn(2,2,8,8,16);sup=torch.ones(2,6,8,8,16,dtype=torch.bool)
        score,logits=aux(x,sup)
        loss=-torch.nn.functional.log_softmax(logits,-1)[...,3].mean()
        loss.backward()
        self.assertGreater(float(aux.readout.body[0].weight.grad.abs().sum()),0.)
        self.assertIsNone(aux.readout.head[0].weight.grad)

if __name__=='__main__':unittest.main()
