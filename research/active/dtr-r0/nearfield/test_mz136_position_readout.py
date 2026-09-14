"""Position preservation and checkpoint compatibility, without fitting."""
import unittest
import torch
from mz136_corridor_pair import CorridorNet
from mz136_position_readout import PositionCorridorNet


class PositionReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(136014)
        torch.set_num_threads(2)
        self.base = CorridorNet().eval()
        self.fixed = PositionCorridorNet().eval()
        missing, extra = self.fixed.load_state_dict(self.base.state_dict(), strict=False)
        self.assertEqual(missing, ['horizontal.weight'])
        self.assertEqual(extra, [])
        self.batch = dict(rgb=torch.rand(2, 3, 32, 48), grid=torch.rand(2, 45, 49, 2)*2-1,
                          tof=torch.rand(2, 129, 8), radar=torch.rand(2, 9, 7),
                          local=torch.ones(2, 45, 129, dtype=torch.bool), imu=torch.rand(2, 5))

    def test_zero_residual_preserves_checkpoint_output(self):
        with torch.no_grad():
            torch.testing.assert_close(self.fixed(self.batch), self.base(self.batch), rtol=0, atol=0)

    def test_horizontal_order_can_change_score_after_learning(self):
        swapped = dict(self.batch)
        swapped['grid'] = self.batch['grid'].unflatten(2, (7, 7)).flip(3).flatten(2, 3)
        with torch.no_grad():
            # Same sample set; old mean/max readout is invariant to column order.
            torch.testing.assert_close(self.base(self.batch), self.base(swapped), atol=1e-6, rtol=0)
            self.fixed.horizontal.weight.normal_(0, .1)
            difference = (self.fixed(self.batch)-self.fixed(swapped)).abs().max().item()
        self.assertGreater(difference, 1e-6)


if __name__ == '__main__':
    unittest.main()
