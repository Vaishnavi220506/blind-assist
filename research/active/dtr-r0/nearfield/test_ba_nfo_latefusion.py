import unittest
import torch
import ba_nfo_matched as m
from ba_nfo_latefusion import LateFusion


class TestLateFusion(unittest.TestCase):
    def test_identity_and_branch_gradient(self):
        torch.set_num_threads(4);torch.manual_seed(19)
        base=m.Net('nfo');net=LateFusion(base)
        rgb=torch.randint(0,256,(1,3,192,256));zones=torch.randn(1,6,8,8)
        expected=base(rgb,zones);actual=net(rgb,zones)
        self.assertTrue(torch.equal(expected,actual))
        actual.square().mean().backward()
        self.assertGreater(net.correction.weight.grad.abs().sum().item(),0)
        self.assertEqual(net.detail[0].weight.grad.abs().sum().item(),0)
        with torch.no_grad():net.correction.weight.add_(.01)
        self.assertFalse(torch.equal(net(rgb,zones),expected))

    def test_exact_broadcast_and_outside_coverage(self):
        net=LateFusion(m.Net('nfo'))
        z=torch.arange(64.).reshape(1,1,8,8).expand(1,16,8,8).clone().requires_grad_()
        a=net.broadcast(z)
        self.assertEqual(a.shape,(1,16,192,256))
        self.assertTrue(torch.equal(a[0,0][net.index>=0],net.index[net.index>=0].float()))
        self.assertTrue(torch.all(a[0,:,net.index<0]==0))
        # First x boundary is floor(26+204/8)=51, not nearest-resize boundary52.
        self.assertEqual(net.index[19,50].item(),0)
        self.assertEqual(net.index[19,51].item(),1)
        a.sum().backward()
        self.assertTrue(torch.all(z.grad>0))


if __name__=='__main__':unittest.main()
