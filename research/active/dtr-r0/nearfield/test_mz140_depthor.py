import unittest
import numpy as np
import torch
from mz140_depthor import local_conv,make_input

class AdapterTests(unittest.TestCase):
    def test_bpnet_scalar_order_and_padding(self):
        rng=np.random.default_rng(140)
        for k in [3,5,7]:
            x=rng.normal(size=(2,2,4,5));weight=rng.normal(size=(2,2*k*k,4,5));expected=np.zeros_like(x)
            for b in range(2):
                for c in range(2):
                    for y in range(4):
                        for z in range(5):
                            for a in range(k):
                                for d in range(k):
                                    yy,xx=y+a-k//2,z+d-k//2
                                    if 0<=yy<4 and 0<=xx<5:expected[b,c,y,z]+=x[b,c,yy,xx]*weight[b,c*k*k+a*k+d,y,z]
            got=local_conv(torch.tensor(x),torch.tensor(weight)).numpy()
            np.testing.assert_allclose(got,expected,atol=1e-12)

    def test_sparse_is_axis_depth_and_preserves_missing(self):
        row=dict(tof_packet_received=True,rgb_intrinsics=dict(width=640,height=360,fx=450,fy=450,cx=320,cy=180),
          tof_zones=[dict(zone_id=0,theta_bounds_deg=[5,10],phi_bounds_deg=[-2,2],targets=[dict(status='SIM_VALID',distance_m=2.)]),
                     dict(zone_id=1,theta_bounds_deg=[-2,2],phi_bounds_deg=[-2,2],targets=[dict(status='SIM_MERGED',distance_m=2.)]),
                     dict(zone_id=2,theta_bounds_deg=[60,65],phi_bounds_deg=[-2,2],targets=[dict(status='SIM_VALID',distance_m=2.)])])
        before=repr(row);data,audit=make_input(row,np.zeros((360,640,3),np.uint8),'cpu')
        self.assertEqual(data['image'].shape,(1,3,480,640));self.assertEqual((data['sparse_depth']>0).sum(),1)
        self.assertLess(audit['used'][0]['axis_depth_m'],2.);self.assertEqual(len(audit['excluded']),2)
        self.assertEqual(before,repr(row))

if __name__=='__main__':unittest.main()
