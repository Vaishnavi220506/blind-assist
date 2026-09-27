from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from cnh_segment_controls_train import Readout,load_unit,tensor_input,predict


class MaskControlTest(unittest.TestCase):
    def test_mask_broadcast_without_histogram_inputs(self):
        mask=np.arange(128,dtype=np.float32).reshape(2,8,8)/128
        tensor=tensor_input(mask,'cpu')
        self.assertEqual(tuple(tensor.shape),(2,1,8,8,16))
        for b in range(16): np.testing.assert_array_equal(tensor[:,0,:,:,b].numpy(),mask)
        with self.assertRaises(ValueError): tensor_input(np.zeros((2,8,8,16)),'cpu')

    def test_metadata_loader_needs_no_z_keys_and_rejects_misalignment(self):
        root=Path(__file__).resolve().parents[4]/'artifacts.local/work'
        with tempfile.TemporaryDirectory(dir=root,prefix='maskcontrol-test-') as tmp:
            f,m=Path(tmp)/'features.npz',Path(tmp)/'masks.npz'
            support=np.ones((2,6,8,8,16),bool)
            np.savez(f,labels=np.zeros((2,6)),main=np.ones(2,bool),config=[0,0],frame=[0,1],
                     split='train',sup=np.packbits(support,axis=-1))
            np.savez(m,IDEAL=np.ones((2,8,8))*.25,config=[0,0],frame=[0,1],split='train')
            d=load_unit(f,m,'train')
            self.assertNotIn('x',d)
            np.testing.assert_array_equal(d['sup'],support)
            self.assertEqual(d['mask'].shape,(2,8,8))
            np.savez(m,IDEAL=np.ones((2,8,8))*.25,config=[0,0],frame=[1,0],split='train')
            with self.assertRaises(ValueError):load_unit(f,m,'train')

    def test_one_channel_model_state_and_batched_prediction(self):
        torch.set_num_threads(1);torch.manual_seed(1)
        net=Readout().cpu()
        self.assertEqual(net.body[0].in_channels,1)
        self.assertEqual(net.body[0].out_channels,16)
        self.assertEqual(net.emb.num_embeddings,6)
        mask=np.random.default_rng(3).uniform(size=(3,8,8)).astype(np.float32)
        support=np.ones((3,6,8,8,16),bool)
        a=predict(net,mask,support,batch=2)
        copy=Readout().cpu();copy.load_state_dict(net.state_dict())
        b=predict(copy,mask,support,batch=3)
        self.assertEqual(a.shape,(3,6));self.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)

if __name__=='__main__':unittest.main()
