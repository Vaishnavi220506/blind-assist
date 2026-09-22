"""Synthetic engineering checks, not task-quality or recipe-selection evidence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spatial_bce_model import (Head, RECIPE, _boxes, _load_encoder, build_inputs,
                               load_head, normalized_boxes45, predict, sample_zones)


class SpatialBceTests(unittest.TestCase):
    def test_zone_sampling_retains_coordinate_order(self):
        # A linear x/y feature plane makes grid_sample's geometry independently observable.
        h, w = 144, 256
        y, x = torch.meshgrid((torch.arange(h) + .5) / h, (torch.arange(w) + .5) / w, indexing="ij")
        feature = torch.stack([x, y] * 12)[None]
        sampled = sample_zones(feature).reshape(1, 24, 9, 64)
        b = normalized_boxes45()
        for subrow in range(3):
            for subcol in range(3):
                k = subrow * 3 + subcol
                expected_x = b[:, 1] + (b[:, 3] - b[:, 1]) * (subcol + .5) / 3
                expected_y = b[:, 0] + (b[:, 2] - b[:, 0]) * (subrow + .5) / 3
                np.testing.assert_allclose(sampled[0, 0, k], expected_x, atol=1e-6)
                np.testing.assert_allclose(sampled[0, 1, k], expected_y, atol=1e-6)
        with self.assertRaises(ValueError):
            _boxes(b[::-1])

    def test_missing_range_not_invented_and_angles_are_spatial(self):
        rgb = np.zeros((2, 216, 8, 8), np.float32)
        ranges = np.ones((2, 64), np.float32)
        ranges[0, :5] = [np.nan, np.inf, -1, 0, 8]
        inp = build_inputs(rgb, ranges)
        self.assertEqual(inp.shape, (2, 220, 8, 8))
        self.assertTrue(np.isfinite(inp).all())
        np.testing.assert_array_equal(inp[0, 216:218].reshape(2, 64)[:, :5], 0)
        self.assertEqual(inp[1, 216, 0, 0], .125)
        self.assertEqual(inp[1, 217, 0, 0], 1)
        self.assertLess(inp[0, 218, 0, 0], inp[0, 218, 0, 7])
        self.assertLess(inp[0, 219, 0, 0], inp[0, 219, 7, 0])

    def test_head_gradients_subset_inference_and_checkpoint_roundtrip(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        head = Head()
        x = torch.randn(3, 220, 8, 8)
        y = torch.tensor([0., 1., 0.])
        torch.nn.functional.binary_cross_entropy_with_logits(head(x), y).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        all_logits = predict(head, x.numpy())
        np.testing.assert_allclose(predict(head, x.numpy(), [2, 0]), all_logits[[2, 0]], atol=1e-6)
        artifact_work = Path(__file__).resolve().parents[4] / 'artifacts.local/work'
        with tempfile.TemporaryDirectory(prefix='spatial-bce-test-', dir=artifact_work) as directory:
            checkpoint = Path(directory) / "head.pt"
            torch.save(dict(recipe=RECIPE, state_dict=head.state_dict()), checkpoint)
            restored = load_head(checkpoint)
            np.testing.assert_array_equal(predict(restored, x.numpy()), all_logits)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 81921)

    def test_local_pretrained_encoder_is_frozen_eval(self):
        root = Path(__file__).resolve().parents[4]
        checkpoint = root / "artifacts.local/work/ba-nfo-20260919/torch-cache/checkpoints/mobilenet_v3_small-047dcff4.pth"
        if not checkpoint.exists():
            self.skipTest("Local pretrained checkpoint unavailable; never download for engineering test")
        encoder = _load_encoder(checkpoint)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        with torch.inference_mode():
            features = encoder(torch.zeros(1, 3, 144, 256))
        self.assertEqual(tuple(features.shape), (1, 24, 18, 32))
        self.assertEqual(tuple(sample_zones(features).shape), (1, 216, 8, 8))


if __name__ == "__main__":
    unittest.main()
