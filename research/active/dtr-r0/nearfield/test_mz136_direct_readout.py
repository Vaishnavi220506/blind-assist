"""Checkpoint equivalence, inference independence, and successful-fit reporting."""
import json
import unittest
import numpy as np
import torch
from mz136_corridor_pair import CorridorNet
from mz136_direct_readout import DirectReadoutNet
from repair_mz136_train_fit import describe


class DirectReadoutTests(unittest.TestCase):
    def test_zero_residual_and_no_batch_dependent_statistics(self):
        torch.manual_seed(136014)
        torch.set_num_threads(2)
        base, fixed = CorridorNet().eval(), DirectReadoutNet().eval()
        fixed.load_state_dict(base.state_dict(), strict=False)
        b = dict(rgb=torch.rand(2,3,32,48),grid=torch.rand(2,45,49,2)*2-1,
                 tof=torch.rand(2,129,8),radar=torch.rand(2,9,7),
                 local=torch.ones(2,45,129,dtype=torch.bool),imu=torch.rand(2,5))
        with torch.no_grad():
            torch.testing.assert_close(base(b), fixed(b), atol=0, rtol=0)
            fixed.readout.weight.normal_(0,.1)
            fixed.feature_scale.fill_(.1)
            whole = fixed(b)
            separate = torch.cat([fixed({k:v[i:i+1] for k,v in b.items()}) for i in range(2)])
            torch.testing.assert_close(whole, separate, atol=1e-5, rtol=1e-5)
        self.assertEqual(sum(p.numel() for p in fixed.readout.parameters()), 385)

    def test_pass_and_fail_reports_are_json_serializable(self):
        pairs = [dict(a=0,b=1,family='boundary')]
        gt = np.array([True,False])
        for scores, expected in [(np.array([1.,-1.]),True),(np.array([-1.,-1.]),False)]:
            value = describe(gt,scores,pairs)
            self.assertIs(value['fit_pass'],expected)
            json.dumps(value,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
