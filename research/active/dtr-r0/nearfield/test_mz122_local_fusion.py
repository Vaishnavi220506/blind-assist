import copy
import unittest
import numpy as np
import torch
from mz120_occupancy import OccupancyNet
from mz122_local_fusion import LocalFusionNet,point_geometry


class LocalFusionTests(unittest.TestCase):
    def test_common_weights_are_exactly_transferable(self):
        torch.manual_seed(120013);early=OccupancyNet();late=LocalFusionNet()
        result=late.load_state_dict(early.state_dict(),strict=False)
        self.assertEqual(set(result.missing_keys),{'position.weight','position.bias','local_fusion.0.weight','local_fusion.0.bias'})
        self.assertFalse(result.unexpected_keys)
        for k,v in early.state_dict().items():self.assertTrue(torch.equal(v,late.state_dict()[k]))

    def test_pooling_preserves_tuples_but_not_feature_position_swaps(self):
        torch.manual_seed(12);model=LocalFusionNet()
        samples=torch.randn(1,45,49,24)
        b=dict(imu=torch.zeros(1,5),grid=torch.rand(1,45,49,2)*2-1,
               tof=torch.randn(1,129,8),point_local=torch.ones(1,45,49,129,dtype=torch.bool))
        with torch.no_grad():
            ref=model.fuse_samples(samples,b)[0]
            perm=torch.arange(48,-1,-1);paired=dict(b,grid=b['grid'][:,:,perm],point_local=b['point_local'][:,:,perm])
            together=model.fuse_samples(samples[:,:,perm],paired)[0]
            swapped=model.fuse_samples(samples[:,:,perm],b)[0]
        torch.testing.assert_close(ref,together,atol=1e-6,rtol=1e-5)
        self.assertGreater((ref-swapped).abs().max().item(),1e-4)

    def test_point_zone_mask_preserves_explicit_missingness_tokens(self):
        row=dict(rgb_intrinsics=dict(width=3,height=3,cx=1,cy=1,fx=100,fy=100),
                 tof_zones=[dict(theta_bounds_deg=[-.1,.1],phi_bounds_deg=[-.1,.1])])
        encoded=dict(grid=np.array([[[0,0],[40,40]]],np.float32),local=np.ones((1,3),bool),
                     tof=np.array([[1,0,1,1,0,0,0,1],[0]*8,[0]*8],np.float32))
        mask=point_geometry(row,encoded)
        np.testing.assert_array_equal(mask,[[[True,True,True],[False,False,True]]])
        changed=copy.deepcopy(row);changed.update(native_bounds='poison',family='danger')
        np.testing.assert_array_equal(mask,point_geometry(changed,encoded))


if __name__=='__main__':unittest.main()
