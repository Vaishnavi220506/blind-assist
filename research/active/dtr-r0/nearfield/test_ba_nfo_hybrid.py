"""Focused checks for matched initialization and training-only depth supervision."""
import unittest

import torch

import ba_nfo_matched as m
import ba_nfo_hybrid as h


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def inputs(self):
        generator = torch.Generator().manual_seed(91)
        rgb = torch.randint(0, 256, (1, 3, 32, 40), generator=generator,
                            dtype=torch.uint8)
        zones = torch.randn(1, 6, 8, 8, generator=generator)
        return rgb, zones

    def test_main_initialization_matches_original_nfo(self):
        m.seed()
        baseline = m.Net('nfo')
        m.seed()
        hybrid = h.Hybrid()
        baseline_state = baseline.state_dict()
        hybrid_state = hybrid.state_dict()
        self.assertEqual(set(hybrid_state) - set(baseline_state),
                         {'auxiliary_output.weight', 'auxiliary_output.bias'})
        for key, value in baseline_state.items():
            self.assertTrue(torch.equal(value, hybrid_state[key]), key)

    def test_pruned_deployment_has_identical_logits(self):
        m.seed()
        hybrid = h.Hybrid().eval()
        deployed = m.Net('nfo').eval()
        deployed.load_state_dict({key: value for key, value in hybrid.state_dict().items()
                                 if not key.startswith('auxiliary_output.')}, strict=True)
        rgb, zones = self.inputs()
        with torch.no_grad():
            train_logits, depth = hybrid.forward_train(rgb, zones)
            logits = hybrid(rgb, zones)
            deployed_logits = deployed(rgb, zones)
        self.assertEqual(tuple(logits.shape), (1, 4, 32, 40))
        self.assertEqual(tuple(depth.shape), (1, 1, 32, 40))
        torch.testing.assert_close(logits, train_logits, rtol=0, atol=0)
        torch.testing.assert_close(logits, deployed_logits, rtol=0, atol=0)

    def test_unknown_target_has_zero_gradient_in_both_losses(self):
        depth = torch.tensor([[[float('nan'), 1.2, 3.5],
                               [float('inf'), 0., -1.]]])
        known = torch.isfinite(depth) & (depth > 0)
        for arm, channels in [('nfo', 4), ('depth', 1)]:
            with self.subTest(arm=arm):
                output = torch.zeros(1, channels, 2, 3, requires_grad=True)
                loss = m.loss_fn(output, depth, arm)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                gradient = output.grad.permute(0, 2, 3, 1)
                self.assertEqual(int(torch.count_nonzero(gradient[~known])), 0)
                self.assertGreater(float(gradient[known].abs().sum()), 0.)

    def test_auxiliary_loss_reaches_shared_decoder(self):
        m.seed()
        hybrid = h.Hybrid()
        rgb, zones = self.inputs()
        _, prediction = hybrid.forward_train(rgb, zones)
        truth = torch.full((1, 32, 40), 2.5)
        truth[:, 0] = float('nan')
        m.loss_fn(prediction, truth, 'depth').backward()
        for label, parameter in [('auxiliary head', hybrid.auxiliary_output.weight),
                                 ('shared decoder', hybrid.up0[0].weight),
                                 ('shared fusion', hybrid.fuse[0].weight)]:
            self.assertIsNotNone(parameter.grad, label)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), label)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., label)
        self.assertIsNone(hybrid.output.weight.grad)


if __name__ == '__main__':
    unittest.main()
