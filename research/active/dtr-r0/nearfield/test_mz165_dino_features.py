"""Narrow generated-input feature layout and preprocessing tests."""
import unittest
import numpy as np
import torch

from mz165_dino_features import extract,prepare_rgb


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.eval()
    def forward_features(self,image):
        self.input=image
        t=torch.arange(37*66*384,device=image.device,dtype=torch.float32).reshape(1,37*66,384)
        return {'x_norm_patchtokens':t,'x_norm_clstoken':torch.full((1,384),-99.)}


class FeatureTests(unittest.TestCase):
    def test_rgb_range_channel_order_normalization_and_shape(self):
        rgb=np.zeros((360,640,3),np.uint8);rgb[:,:,0]=255
        x=prepare_rgb(rgb)
        self.assertEqual(tuple(x.shape),(1,3,518,924))
        expected=torch.tensor([(1-.485)/.229,-.456/.224,-.406/.225])
        torch.testing.assert_close(x[0,:,200,400],expected)
        self.assertEqual(rgb[0,0].tolist(),[255,0,0])
    def test_patch_row_column_channels_and_no_cls(self):
        model=FakeBackbone()
        f,a=extract(model,np.zeros((360,640,3),np.uint8))
        self.assertEqual(f.shape,(384,37,66));self.assertEqual(f.dtype,np.float32)
        self.assertEqual(float(f[12,23,45]),float((23*66+45)*384+12))
        self.assertEqual(a['finite_elements'],384*37*66)
        self.assertFalse(model.input.requires_grad)
    def test_invalid_input_and_unfrozen_rejected(self):
        with self.assertRaises(ValueError):prepare_rgb(np.zeros((360,640,3),np.float32))
        with self.assertRaises(ValueError):prepare_rgb(np.zeros((518,924,3),np.uint8))
        model=FakeBackbone().train()
        with self.assertRaises(ValueError):extract(model,np.zeros((360,640,3),np.uint8))


if __name__=='__main__':unittest.main()
