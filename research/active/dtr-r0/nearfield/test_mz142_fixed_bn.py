"""CPU-only synthetic BatchNorm behavior; no checkpoint or capture access."""
import unittest

import torch
from torch import nn

from mz142_fixed_bn import bn_digest, training_mode


class SyntheticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_anything = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Dropout(.5))
        self.depth_anything.requires_grad_(False)
        self.bn = nn.BatchNorm2d(3, momentum=.23)
        self.dropout = nn.Dropout(.25)
        with torch.no_grad():
            self.depth_anything[0].weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
            self.depth_anything[0].bias.zero_()
            self.bn.running_mean.copy_(torch.tensor([.3, -.2, .5]))
            self.bn.running_var.copy_(torch.tensor([1.5, .8, 2.]))
            self.bn.num_batches_tracked.fill_(17)
            self.bn.weight.copy_(torch.tensor([.7, 1.1, 1.3]))
            self.bn.bias.copy_(torch.tensor([.2, -.1, .4]))

    def forward(self, image):
        return self.dropout(self.bn(self.depth_anything(image)))


class FixedBatchNormTests(unittest.TestCase):
    def test_modes_restore_repeatedly_without_changing_gradient_selection(self):
        model = SyntheticModel()
        selected = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        before = bn_digest(model)
        for _ in range(3):
            model.eval()
            self.assertIs(training_mode(model), model)
            self.assertTrue(model.training)
            self.assertTrue(model.dropout.training)
            self.assertFalse(model.depth_anything.training)
            self.assertFalse(model.depth_anything[1].training)
            self.assertFalse(model.bn.training)
            self.assertTrue(model.bn.track_running_stats)
            self.assertEqual(model.bn.momentum, .23)
            self.assertEqual(selected, {name: parameter.requires_grad for name, parameter in model.named_parameters()})
            self.assertEqual(before, bn_digest(model))

    def test_optimizer_updates_affine_parameters_but_never_running_buffers(self):
        torch.manual_seed(142)
        model = training_mode(SyntheticModel())
        before = bn_digest(model)
        initial_weight = model.bn.weight.detach().clone()
        initial_bias = model.bn.bias.detach().clone()
        optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.05)
        image = torch.linspace(.2, 2., 48).reshape(2, 3, 2, 4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            prediction.square().mean().backward()
            self.assertGreater(float(model.bn.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(model.bn.bias.grad.abs().sum()), 0.)
            self.assertTrue(all(parameter.grad is None for parameter in model.depth_anything.parameters()))
            optimizer.step()
            self.assertEqual(before, bn_digest(model))
        self.assertFalse(torch.equal(initial_weight, model.bn.weight))
        self.assertFalse(torch.equal(initial_bias, model.bn.bias))
        # eval BN applies the saved statistics, not current-image statistics.
        with torch.no_grad():
            expected = (image-model.bn.running_mean[None, :, None, None])
            expected = expected/torch.sqrt(model.bn.running_var[None, :, None, None]+model.bn.eps)
            expected = expected*model.bn.weight[None, :, None, None]+model.bn.bias[None, :, None, None]
            torch.testing.assert_close(model.bn(image), expected)

    def test_digest_ignores_affine_but_detects_each_running_buffer(self):
        model = SyntheticModel()
        before = bn_digest(model)
        with torch.no_grad():
            model.bn.weight.add_(1.)
            model.bn.bias.sub_(1.)
        self.assertEqual(before, bn_digest(model))
        for key in ('running_mean', 'running_var', 'num_batches_tracked'):
            buffer = getattr(model.bn, key)
            saved = buffer.clone()
            buffer.add_(1)
            self.assertNotEqual(before, bn_digest(model), key)
            buffer.copy_(saved)
            self.assertEqual(before, bn_digest(model))

    def test_rejects_untracked_or_missing_buffers_instead_of_current_frame_stats(self):
        model = SyntheticModel()
        model.bn.track_running_stats = False
        with self.assertRaisesRegex(ValueError, 'track running'):
            training_mode(model)
        model.bn.track_running_stats = True
        model.bn.running_mean = None
        with self.assertRaisesRegex(ValueError, 'running_mean'):
            training_mode(model)
        with self.assertRaisesRegex(ValueError, 'running_mean'):
            bn_digest(model)


if __name__ == '__main__':
    unittest.main()
