import unittest
import copy
import numpy as np
import torch
from mz161_dense_task import DenseTaskNet,loss,ray_query,encode,CHANNELS,TOKEN_CHANNELS
from test_mz143_corridor_features import inputs,target

class Tests(unittest.TestCase):
    def test_independent_sensor_and_dense_loss(self):
        torch.manual_seed(161)
        model=DenseTaskNet()
        batch=dict(image=torch.randn(2,CHANNELS,48,64),tokens=torch.zeros(2,132,TOKEN_CHANNELS),valid=torch.ones(2,132,dtype=torch.bool))
        output=model(batch)
        target=torch.ones(2,48,64);weight=torch.ones_like(target)/(48*64)
        a,_=loss(output,torch.ones(2),target,weight,'frame')
        b,_=loss(output,torch.ones(2),target,weight,'dense')
        self.assertGreater(float(b.detach()),float(a.detach()));b.backward()
        self.assertGreater(float(model.stem.weight.grad.abs().sum()),0)
        self.assertTrue(torch.allclose(output['frame'],torch.maximum(output['local'],output['independent'])))
        self.assertEqual(output['token'].shape,(2,132))
    def test_shared_processing_translation(self):
        torch.manual_seed(162);model=DenseTaskNet().eval()
        x=torch.randn(1,CHANNELS,96,128)
        base=dict(image=x,tokens=torch.zeros(1,132,TOKEN_CHANNELS),valid=torch.zeros(1,132,dtype=torch.bool))
        shifted={**base,'image':torch.roll(x,3,dims=-1)}
        with torch.no_grad(): a=model(base)['pixel'];b=model(shifted)['pixel']
        self.assertTrue(torch.allclose(a[:,34:-34,34:-37],b[:,34:-34,37:-34],atol=1e-6,rtol=1e-5))
    def test_ray_query(self):
        row=dict(rgb_intrinsics=dict(width=5,height=5,fx=4.,fy=4.,cx=2.,cy=2.),camera_pitch_deg=0.,camera_in_body_m=[0,0,1.7])
        dirs,q=ray_query(row,0.)
        self.assertTrue(np.allclose(dirs[2,2],[1,0,0]))
        self.assertTrue(np.allclose(q[:,2,2],[.05,.9,1]))
        self.assertTrue(np.isfinite(q).all())
    def test_public_encoding_and_half_open_regions(self):
        row,image=inputs();row['rgb_intrinsics'].update(cx=320.,cy=180.)
        for z in row['tof_zones']:z['targets']=[target(1.+z['zone_id']/100)]
        before=copy.deepcopy(row);a=encode(row,image,0.)
        self.assertEqual(a['tokens'].shape,(132,TOKEN_CHANNELS))
        self.assertEqual(a['image'].shape,(CHANNELS,360,640))
        noisy=copy.deepcopy(row);noisy.update(id='secret',truth=True,native_bounds=[123],family='head')
        b=encode(noisy,image,0.)
        np.testing.assert_array_equal(a['image'],b['image']);np.testing.assert_array_equal(a['tokens'],b['tokens'])
        self.assertEqual(row,before)
        from mz115_spatial_allocation import zone_box
        occupied=np.zeros((360,640),int)
        for z in row['tof_zones']:
            box=zone_box(z,row['rgb_intrinsics'])
            l,t,r,b=[int(np.ceil(x)) for x in box]
            occupied[max(0,t):min(360,b),max(0,l):min(640,r)]+=1
        self.assertLessEqual(int(occupied.max()),1)

if __name__=='__main__':unittest.main()
