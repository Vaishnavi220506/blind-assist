import unittest
import numpy as np
import torch
from mz140_depthor import local_conv
from mz141_depthor_training import supervision, loss_contribution


class TrainingTests(unittest.TestCase):
    def test_local_conv_backward_both_arguments(self):
        torch.manual_seed(141)
        for kernel in (3, 5, 7):
            x = torch.randn(1, 2, 2, 3, dtype=torch.double, requires_grad=True)
            weight = torch.randn(1, 2*kernel*kernel, 2, 3, dtype=torch.double, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(local_conv, (x, weight), fast_mode=True))

    def test_unknown_mask_surface_balance_and_ring_penalty(self):
        depth = np.full((24, 40), 5., np.float32)
        owner = np.ones((24, 40), np.int32)
        owner[2:22, 18:20] = 0
        target = owner == 0
        depth[target] = 2.
        depth[:, :2] = np.nan
        owner[:, :2] = -1
        labels = supervision(dict(depth=depth, owner=owner, target_visible=target, names=['shape0', 'env0']))
        self.assertAlmostEqual(float(labels['balanced_weight'][target].sum()), .25, places=5)
        p = torch.tensor(labels['depth'], requires_grad=True)
        with torch.no_grad():
            p[:, :2] = 999.
        self.assertEqual(float(loss_contribution(p, labels, 'surface', 1, 1)), 0.)
        with torch.no_grad():
            p[5:15, 20:22] = 2.
        loss = loss_contribution(p, labels, 'surface', 1, 1)
        self.assertGreater(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(p.grad[:, :2].abs().sum()), 0.)
        self.assertGreater(float(p.grad[5:15, 20:22].abs().sum()), 0.)

    def test_pixel_batch_denominator_and_empty_ring(self):
        labels = supervision(dict(depth=np.array([[2., 3.]], np.float32),
            owner=np.array([[0, 0]]), target_visible=np.ones((1, 2), bool), names=['shape0']))
        p = torch.tensor([[3., 5.]])
        self.assertEqual(float(loss_contribution(p, labels, 'pixel', 4, 2)), .75)
        self.assertEqual(float(loss_contribution(p, labels, 'surface', 4, 2)), .75)


if __name__ == '__main__':
    unittest.main()
