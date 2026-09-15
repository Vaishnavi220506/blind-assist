import unittest
import torch
from torch.nn import functional as F
from mz153_dvsr_runtime import modulated_deform_adapter

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required for the intended backend')
class DeformMappingTests(unittest.TestCase):
    def test_zero_offsets_unit_mask_match_grouped_convolution(self):
        torch.manual_seed(153016)
        for groups,deform_groups in [(1,1),(2,1),(1,2),(2,2)]:
            x=torch.randn(2,8,11,13,device='cuda');w=torch.randn(6,8//groups,3,3,device='cuda');b=torch.randn(6,device='cuda')
            offset=torch.zeros(2,18*deform_groups,11,13,device='cuda');mask=torch.ones(2,9*deform_groups,11,13,device='cuda')
            actual=modulated_deform_adapter(x,offset,mask,w,b,padding=1,groups=groups,deform_groups=deform_groups)
            # The CUDA deform operator accumulates FP32; disable cuDNN TF32
            # only for this reference comparison, restoring flags afterward.
            with torch.backends.cudnn.flags(allow_tf32=False):
                expected=F.conv2d(x,w,b,padding=1,groups=groups)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)

    def test_interleaved_offsets_are_dy_then_dx(self):
        x=torch.arange(30,device='cuda',dtype=torch.float32).reshape(1,1,5,6)
        w=torch.ones(1,1,1,1,device='cuda');mask=torch.ones(1,1,5,6,device='cuda')
        offset=torch.zeros(1,2,5,6,device='cuda');offset[:,0]=1
        down=modulated_deform_adapter(x,offset,mask,w)
        torch.testing.assert_close(down[:,:,:-1,:],x[:,:,1:,:]);self.assertTrue(torch.equal(down[:,:,-1,:],torch.zeros_like(down[:,:,-1,:])))
        offset.zero_();offset[:,1]=1
        right=modulated_deform_adapter(x,offset,mask,w)
        torch.testing.assert_close(right[:,:,:,:-1],x[:,:,:,1:]);self.assertTrue(torch.equal(right[:,:,:,-1],torch.zeros_like(right[:,:,:,-1])))

if __name__=='__main__':unittest.main()
