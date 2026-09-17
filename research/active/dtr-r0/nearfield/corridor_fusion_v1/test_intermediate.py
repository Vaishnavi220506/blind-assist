import unittest
import numpy as np
import torch
from intermediate_features import sample_maps,transform


class IntermediateContract(unittest.TestCase):
    def test_spatial_sample_and_missing(self):
        maps=torch.tensor([[[[1.,2.],[3.,4.]]]])
        uv=np.array([[.25,.25],[.75,.75],[2.,2.]],np.float32)
        got=sample_maps(maps,uv,np.array([True,True,False])).numpy()
        np.testing.assert_array_equal(got[0,:,0],[1,4,0])

    def test_mask_survives_nonzero_pca_mean(self):
        class Stub:
            def transform(self,x):return np.ones((len(x),8))*5
        tokens=np.zeros((2,2,48,384),np.float32)
        mask=np.ones((2,48),bool);mask[0,0]=False
        out=transform(tokens,mask,[Stub(),Stub()])
        self.assertEqual(out.shape,(2,816))
        self.assertFalse(out[0,:8].any());self.assertFalse(out[0,384:392].any())
        self.assertEqual(out[0,768],0);self.assertEqual(out[1,768],1)

if __name__=='__main__':unittest.main()
