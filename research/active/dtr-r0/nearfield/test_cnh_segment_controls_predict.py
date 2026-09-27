import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import cnh_segment_controls_predict as P
import cnh_segment_learned_train as T


class Capture(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.inputs = []

    def forward(self, x, sup):
        assert x.shape[1] == self.channels
        self.inputs.append(x.detach().cpu().numpy())
        return x.mean((1,2,3,4))[:,None].expand(-1,6)


class ControlsPredictTests(unittest.TestCase):
    def test_zero_tof_keeps_raw_mask_and_support(self):
        net = Capture(3)
        mask = np.full((3,8,8),.375,dtype=np.float16)
        x = np.zeros((3,2,8,8,16),dtype=np.float32)
        sup = np.ones((3,6,8,8,16),dtype=bool)
        with patch.object(T.L,'DEV','cpu'):
            out = T.predict(net,x,sup,mask,batch=2)
        captured = np.concatenate(net.inputs)
        self.assertTrue(np.all(captured[:,:2] == 0))
        self.assertTrue(np.all(captured[:,2] == .375))
        np.testing.assert_allclose(out,.125)

    def test_mask_only_broadcast_and_batch_independence(self):
        mask = np.arange(3*64,dtype=np.float32).reshape(3,8,8)/(3*64)
        sup = np.ones((3,6,8,8,16),dtype=bool)
        with patch.object(P.L,'DEV','cpu'):
            net=Capture(1)
            a=P.mask_predict(net,sup,mask,batch=2)
            b=P.mask_predict(Capture(1),sup,mask,batch=3)
        np.testing.assert_allclose(a,b)
        x=np.concatenate(net.inputs)
        np.testing.assert_array_equal(x[:,0,:,:,0],mask)
        np.testing.assert_array_equal(x[:,:,:,:,0],x[:,:,:,:,-1])

    def test_mask_model_structure(self):
        net=P.MaskReadout()
        self.assertEqual(net.body[0].in_channels,1)
        self.assertEqual(sum(p.numel() for p in net.parameters()),44401)

    def test_mask_freeze_rejects_incomplete_training(self):
        artifact_root=Path(__file__).resolve().parents[4]/'artifacts.local/work'
        with tempfile.TemporaryDirectory(dir=artifact_root,prefix='control-freeze-test-') as temp:
            root=Path(temp); models={}
            for seed in range(3):
                p=root/f'MASK_seed{seed}.pt';p.write_bytes(b'fixture')
                models[str(p)]=P.sha(p)
                p.with_suffix('.json').write_text(json.dumps(dict(status='complete',completed_epochs=20,model_sha256=P.sha(p))))
            (root/'FROZEN.json').write_text(json.dumps(dict(status='frozen',models=models,source_sha256={})))
            P.validate_mask_frozen(root)
            receipt=root/'MASK_seed1.json'
            d=json.loads(receipt.read_text());d['completed_epochs']=19
            receipt.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                P.validate_mask_frozen(root)


if __name__=='__main__':
    unittest.main()
