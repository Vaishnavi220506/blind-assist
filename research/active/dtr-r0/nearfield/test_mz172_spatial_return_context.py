import unittest
import torch
from mz172_spatial_return_context import SpatialReturnTaskNet


class ContextTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(172016)

    def batch(self):
        image=torch.randn(1,22,16,16);image[:,15]=1
        return dict(image=image,context=torch.randn(1,384,2,2),tokens=torch.randn(1,132,21),
            valid=torch.ones(1,132,dtype=torch.bool),zone_grid=torch.zeros(1,64,64,2),
            zone_visible=torch.ones(1,64,64,dtype=torch.bool))

    def test_raw_fallback_and_radar(self):
        m=SpatialReturnTaskNet('registered');b=self.batch();b['zone_visible'].zero_()
        with torch.no_grad():m.returns[-1].bias.fill_(-40)
        out=m(b)
        self.assertTrue(torch.equal(out['token'],out['token_raw']))
        self.assertTrue(torch.all(out['token_context']==-30))
        self.assertTrue(torch.all(out['attention']==0))
        self.assertTrue(torch.isfinite(out['frame']).all())
        b['zone_visible'].fill_(True);out=m(b)
        self.assertTrue(torch.equal(out['token'][:,128:],out['token_raw'][:,128:]))
        self.assertTrue(torch.all(out['token_context'][:,128:]==-30))

    def test_invalid_slots_and_partial_visual_context(self):
        m=SpatialReturnTaskNet('registered');b=self.batch();b['valid'][:,3]=False
        b['zone_visible'][:,:,17:]=False
        with torch.no_grad():m.contextual_return[-1].bias.fill_(5)
        out=m(b)
        self.assertEqual(float(out['token'][0,3]),-30.)
        self.assertTrue(torch.all(out['attention'][:,:,17:]==0))
        self.assertTrue(torch.allclose(out['attention'].sum(-1),b['valid'][:,:128].float()))
        self.assertTrue(out['context_valid'][0,2]);self.assertFalse(out['context_valid'][0,3])

    def test_spatial_correspondence_and_pooled_invariance(self):
        a=SpatialReturnTaskNet('registered');p=SpatialReturnTaskNet('pooled');p.load_state_dict(a.state_dict())
        for m in (a,p):
            with torch.no_grad():
                m.return_query.weight.zero_();m.return_query.bias.zero_();m.return_query.weight[0,0]=1
                m.spatial_key.weight.zero_();m.spatial_key.bias.zero_();m.spatial_key.weight[0,-2]=8
                m.visual_value.weight.zero_();m.visual_value.bias.zero_();m.visual_value.weight[0,0]=1
        t=torch.ones(1,132,21);v=torch.ones(1,64,64,dtype=torch.bool)
        s=torch.zeros(1,64,64,46);s[:,:,0,0]=1
        perm=s.reshape(1,64,8,8,46).flip(3).reshape(1,64,64,46)
        c1,w=p.pool_samples(t,s,v);c2,_=p.pool_samples(t,perm,v)
        self.assertTrue(torch.equal(c1,c2));self.assertTrue(torch.all(w==1/64))
        r1,_=a.pool_samples(t,s,v);r2,_=a.pool_samples(t,perm,v)
        self.assertGreater(float((r1-r2).abs().max()),.01)


if __name__=='__main__':unittest.main()
