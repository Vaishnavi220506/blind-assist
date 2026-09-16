import unittest
import torch
from mz165_pretrained_task import ContextTaskNet
from mz161_dense_task import loss

class Tests(unittest.TestCase):
    def batch(self):
        return dict(image=torch.randn(2,22,48,64),context=torch.randn(2,384,4,5),
            tokens=torch.randn(2,132,21),valid=torch.ones(2,132,dtype=torch.bool))
    def test_context_receives_dense_gradients_and_preserves_native_size(self):
        torch.manual_seed(165);m=ContextTaskNet();b=self.batch();o=m(b)
        self.assertEqual(tuple(o['pixel'].shape),(2,48,64))
        target=torch.ones_like(o['pixel']);weight=target/(48*64)
        total,_=loss(o,torch.ones(2),target,weight,'dense');total.backward()
        self.assertGreater(m.context.weight.grad.abs().sum().item(),0)
        self.assertGreater(m.stem.weight.grad.abs().sum().item(),0)
    def test_rgb_context_cannot_gate_or_modify_independent_return_branch(self):
        torch.manual_seed(166);m=ContextTaskNet().eval();b=self.batch()
        with torch.no_grad():
            a=m(b);c=m({**b,'context':b['context']*9,'image':b['image']*0})
        torch.testing.assert_close(a['token'],c['token'],rtol=0,atol=0)
        torch.testing.assert_close(c['frame'],torch.maximum(c['local'],c['independent']))
        self.assertTrue(torch.all(c['local']==-30))
        empty=m({**b,'valid':torch.zeros_like(b['valid'])})
        self.assertTrue(torch.all(empty['independent']==-30))
    def test_identical_initialization_and_equal_feature_response(self):
        torch.manual_seed(165016);a=ContextTaskNet()
        torch.manual_seed(165016);b=ContextTaskNet();data=self.batch()
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        torch.testing.assert_close(a(data)['pixel'],b(data)['pixel'],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
