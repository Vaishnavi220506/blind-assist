"""Synthetic model-contract tests; no fit or task-quality claims."""
import copy
import unittest
from pathlib import Path

import torch

from query_occupancy_model import (QueryOccupancyNet, canonicalize_tof,
                                   distance_cdf, sample_region_features)


def inputs(batch=2, device="cpu"):
    rows, cols = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    boxes = torch.stack((rows, cols, rows + 1, cols + 1), dim=-1).reshape(64, 4).float() / 8
    tof = torch.cat((torch.full((64, 1), .25), torch.ones(64, 1), boxes), dim=-1)[None].repeat(batch, 1, 1)
    queries = torch.tensor([[x - .3, x + .3, y0, y1]
                            for x in (-.3, 0., .3) for y0, y1 in ((-.2, .42), (.42, .9))])
    return (torch.zeros(batch, 3, 180, 320, device=device), tof.to(device), queries.to(device))


class QueryOccupancyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"QueryOccupancyContractTests network_device={cls.device}; synthetic contracts only", flush=True)
        torch.manual_seed(211)
        cls.model = QueryOccupancyNet(None).to(cls.device).eval()

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = cls.previous_tf32

    def test_cdf_monotone_and_no_surface_excluded(self):
        logits = torch.tensor([[[2., -2., 0., 4., 1., -1., 3.], [-100., -100., -100., -100., -100., -100., 100.]]])
        cdf = distance_cdf(logits)
        self.assertEqual(tuple(cdf.shape), (1, 2, 6))
        self.assertTrue(bool((cdf[..., 1:] >= cdf[..., :-1]).all()))
        self.assertTrue(bool(((cdf >= 0) & (cdf <= 1)).all()))
        torch.testing.assert_close(cdf[..., -1], 1 - logits.softmax(-1)[..., -1])
        self.assertEqual(float(cdf[0, 1, -1]), 0.)
        torch.testing.assert_close(distance_cdf(torch.zeros(7)), torch.arange(1, 7).float() / 7)
        self.assertEqual(distance_cdf(logits.half()).dtype, torch.float32)
        for bad in (torch.tensor([float("nan"), 0.]), torch.zeros(1), torch.ones(7, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                distance_cdf(bad)

    def test_invalid_tof_is_missing_not_a_far_measurement(self):
        _, tof, _ = inputs(1)
        tof[0, :7, 0] = torch.tensor([float("nan"), float("inf"), -1., 0., 1., 8., .9])
        tof[0, 6, 1] = 0
        original = tof.clone()
        clean = canonicalize_tof(tof)
        torch.testing.assert_close(clean[0, :7, :2], torch.zeros(7, 2))
        torch.testing.assert_close(clean[..., 2:], tof[..., 2:])
        torch.testing.assert_close(clean[0, 7, :2], torch.tensor([.25, 1.]))
        torch.testing.assert_close(tof, original, equal_nan=True)
        missing = tof.clone()
        missing[0, :7, 0] = 123456.
        missing[0, :7, 1] = 0
        torch.testing.assert_close(canonicalize_tof(missing), clean)

    def test_region_pooling_respects_boxes_and_coordinates(self):
        y, x = torch.meshgrid((torch.arange(40) + .5) / 40, (torch.arange(80) + .5) / 80, indexing="ij")
        features = torch.stack((x, y))[None]
        boxes = torch.tensor([[[.1, .2, .5, .8], [.5, .1, .9, .3]]])
        sampled = sample_region_features(features, boxes)
        torch.testing.assert_close(sampled, torch.tensor([[[.5, .3], [.2, .7]]]), atol=1e-6, rtol=1e-6)

    def test_shapes_and_same_weights_shared_features_across_modes(self):
        rgb, tof, queries = inputs(device=self.device)
        classifier = copy.deepcopy(self.model)
        classifier.mode = "classifier"
        self.assertEqual(list(classifier.state_dict()), list(self.model.state_dict()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.encoder.parameters()))
        with torch.inference_mode():
            left = self.model.forward_features(rgb, tof, queries)
            right = classifier.forward_features(rgb, tof, queries)
            occupied = self.model(rgb, tof, queries)
            classified = classifier(rgb, tof, queries)
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        self.assertEqual(tuple(left["spatial_features"].shape), (2, 6, 32, 45, 80))
        self.assertEqual(tuple(occupied["distance_logits"].shape), (2, 6, 7))
        self.assertEqual(tuple(occupied["mask_logits"].shape), (2, 6, 45, 80))
        self.assertEqual(tuple(classified["query_logits"].shape), (2, 6))
        self.assertEqual(set(occupied), {"distance_logits", "mask_logits"})
        self.assertEqual(set(classified), {"query_logits"})

    def test_missing_payload_and_regional_permutation_do_not_change_predictions(self):
        rgb, tof, queries = inputs(1, self.device)
        tof[:, :8, 1] = 0
        alternate = tof.clone()
        alternate[:, :8, 0] = float("nan")
        permutation = torch.arange(63, -1, -1, device=self.device)
        with torch.inference_mode():
            base = self.model(rgb, tof, queries)
            missing = self.model(rgb, alternate, queries)
            permuted = self.model(rgb, alternate[:, permutation], queries)
            validity = self.model.forward_features(rgb, alternate, queries)["tof_valid"]
        self.assertEqual(int(validity.sum()), 56)
        for name in base:
            self.assertTrue(bool(torch.isfinite(base[name]).all()))
            torch.testing.assert_close(base[name], missing[name], rtol=0, atol=0)
            torch.testing.assert_close(base[name], permuted[name], atol=2e-6, rtol=1e-5)

    def test_output_gradients_reach_shallow_and_semantic_encoder(self):
        model = copy.deepcopy(self.model)
        rgb, tof, queries = inputs(1, self.device)
        rgb.normal_()
        output = model(rgb, tof, queries[:2])
        loss = output["distance_logits"].square().mean() + output["mask_logits"].square().mean()
        loss.backward()
        for parameter in (model.encoder[0][0].weight, model.high_project.weight, model.low_project.weight,
                          model.region_encoder[0].weight, model.distance_head.weight, model.mask_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(model.classifier_head.weight.grad)

    def test_input_validation(self):
        rgb, tof, queries = inputs(1, self.device)
        cases = [(rgb[:, :, :-1], tof, queries), (rgb, tof[:, :-1], queries),
                 (rgb, tof.repeat(2, 1, 1), queries), (rgb, tof, queries[:0]),
                 (rgb, tof, queries[:, [1, 0, 2, 3]])]
        bad_flags, bad_boxes, bad_rgb = tof.clone(), tof.clone(), rgb.clone()
        bad_flags[0, 0, 1] = .5
        bad_boxes[0, 0, 2] = -.1
        bad_rgb[0, 0, 0, 0] = float("nan")
        cases.extend([(rgb, bad_flags, queries), (rgb, bad_boxes, queries), (bad_rgb, tof, queries)])
        for bad in cases:
            with self.assertRaises(ValueError):
                self.model(*bad)
        for kwargs in ({"mode": "depth"}, {"n_bins": 0}, {"n_bins": True}):
            with self.assertRaises(ValueError):
                QueryOccupancyNet(None, **kwargs)

    def test_local_checkpoint_loading_and_trainable_encoder(self):
        root = Path(__file__).resolve().parents[4]
        checkpoint = root / "artifacts.local/work/ba-nfo-20260919/torch-cache/checkpoints/mobilenet_v3_small-047dcff4.pth"
        if not checkpoint.is_file():
            self.skipTest("Local ImageNet checkpoint unavailable; tests never download")
        model = QueryOccupancyNet(checkpoint, mode="classifier")
        expected = torch.load(checkpoint, map_location="cpu", weights_only=True)
        torch.testing.assert_close(model.encoder[0][0].weight, expected["features.0.0.weight"], rtol=0, atol=0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
