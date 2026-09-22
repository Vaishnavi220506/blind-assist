"""Synthetic spatial-query mechanism checks; no data loading or optimizer steps."""
import copy
import math
import unittest

import torch

from query_occupancy_model import QueryOccupancyNet
from query_spatial_fit_model import DEPTH_HYPOTHESES_M, QuerySpatialFitNet


def inputs(device, batch=1):
    rows, cols = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    boxes = torch.stack((rows, cols, rows + 1, cols + 1), dim=-1).reshape(64, 4).float() / 8
    tof = torch.cat((torch.full((64, 1), .25), torch.ones(64, 1), boxes), dim=-1)[None].repeat(batch, 1, 1)
    queries = torch.tensor([[x - .3, x + .3, lo, hi]
                            for lo, hi in ((.42, .9), (-.2, .42)) for x in (-.3, 0., .3)])
    rgb = torch.randn(batch, 3, 180, 320, device=device)
    return rgb, tof.to(device), queries.to(device)


class SpatialQueryFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        cls.previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SpatialQueryFitTests device={cls.device}; synthetic contracts, no optimizer", flush=True)
        torch.manual_seed(202609224)
        cls.model = QuerySpatialFitNet(None, carrier="spatial").to(cls.device).eval()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = cls.previous_tf32

    def test_zero_residual_matches_original_and_paired_control_exactly(self):
        base = QueryOccupancyNet(None, mode="occupancy").to(self.device).eval()
        base.load_state_dict({name: self.model.state_dict()[name] for name in base.state_dict()}, strict=True)
        control = copy.deepcopy(self.model)
        control.carrier = "global"
        self.assertEqual(list(control.state_dict()), list(self.model.state_dict()))
        self.assertEqual(sum(p.numel() for p in control.parameters()), sum(p.numel() for p in self.model.parameters()))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()) - sum(p.numel() for p in base.parameters()), 768)
        self.assertEqual(int(torch.count_nonzero(self.model.spatial_query_projection.weight)), 0)
        rgb, tof, queries = inputs(self.device, batch=2)
        with torch.inference_mode():
            outputs = [model(rgb, tof, queries) for model in (base, control, self.model)]
        self.assertEqual(set(outputs[0]), {"distance_logits", "mask_logits"})
        for name in outputs[0]:
            torch.testing.assert_close(outputs[0][name], outputs[1][name], rtol=0, atol=0)
            torch.testing.assert_close(outputs[0][name], outputs[2][name], rtol=0, atol=0)

    def test_seeded_construction_has_identical_complete_state_for_both_carriers(self):
        torch.manual_seed(33)
        control = QuerySpatialFitNet(None, "global")
        torch.manual_seed(33)
        spatial = QuerySpatialFitNet(None, "spatial")
        self.assertEqual(list(control.state_dict()), list(spatial.state_dict()))
        for key, value in control.state_dict().items():
            torch.testing.assert_close(value, spatial.state_dict()[key], rtol=0, atol=0)

    def test_fixed_ray_hypotheses_and_signed_distance_geometry(self):
        queries = torch.tensor([[-.3, .3, -.2, .42]], device=self.device)
        geometry = self.model.query_geometry(queries)
        self.assertEqual(tuple(geometry.shape), (1, 24, 45, 80))
        self.assertEqual(tuple(float(v) for v in self.model.query_depth_hypotheses_m.cpu())[1:], DEPTH_HYPOTHESES_M[1:])
        self.assertFalse(self.model.query_depth_hypotheses_m.requires_grad)
        self.assertTrue(bool(((geometry >= -1) & (geometry <= 1)).all()))
        # Native pixel-centre pinhole formula, independent of the model buffer.
        row, col = 22, 40
        for d, depth in enumerate(DEPTH_HYPOTHESES_M):
            x = (((col + .5) / 80) - .5) * 2 * math.tan(math.radians(50)) * depth
            y = (((row + .5) / 45) - .5) * 2 * math.tan(math.radians(50)) * (180 / 320) * depth
            expected = torch.tensor([(x + .3) / .6, (.3 - x) / .6,
                                     (y + .2) / .62, (.42 - y) / .62], device=self.device).clamp(-1, 1)
            torch.testing.assert_close(geometry[0, d * 4:d * 4 + 4, row, col], expected, atol=2e-6, rtol=1e-6)
        self.assertLess(float(geometry[0, 0, 22, 0]), 0.)
        self.assertEqual(float(geometry[0, 20, 22, 0]), -1.)
        self.assertEqual(float(geometry[0, 21, 22, 0]), 1.)

    def test_nonzero_projection_is_spatial_and_responds_to_query_coordinates(self):
        model = copy.deepcopy(self.model)
        with torch.no_grad():
            model.spatial_query_projection.weight[0, 0, 0, 0] = 1.
        queries = torch.tensor([[-.3, .3, -.2, .42], [0., .6, -.2, .42]], device=self.device)
        with torch.inference_mode():
            residual = model.spatial_query_projection(model.query_geometry(queries))
        self.assertGreater(float(residual[0, 0].std()), .1)
        self.assertGreater(float((residual[0] - residual[1]).abs().max()), .1)
        rgb, tof, _ = inputs(self.device)
        control = copy.deepcopy(model)
        control.carrier = "global"
        with torch.inference_mode():
            actual = model(rgb, tof, queries)
            bypass = control(rgb, tof, queries)
            control.spatial_query_projection.weight.zero_()
            control_zero = control(rgb, tof, queries)
        self.assertGreater(float((actual["mask_logits"] - bypass["mask_logits"]).abs().max()), 1e-5)
        for name in bypass:
            torch.testing.assert_close(bypass[name], control_zero[name], rtol=0, atol=0)

    def test_query_permutation_and_arbitrary_query_count_are_equivariant(self):
        model = copy.deepcopy(self.model)
        with torch.no_grad():
            model.spatial_query_projection.weight.normal_(std=.08)
        rgb, tof, queries = inputs(self.device)
        # Unequal, nonstandard rectangles ensure query coordinates drive geometry.
        queries = torch.cat((queries[:2], queries.new_tensor([[-.17, .23, -.4, .31]])), dim=0)
        order = torch.tensor([2, 0, 1], device=self.device)
        with torch.inference_mode():
            original = model(rgb, tof, queries)
            permuted = model(rgb, tof, queries[order])
        self.assertEqual(tuple(original["distance_logits"].shape), (1, 3, 7))
        self.assertEqual(tuple(original["mask_logits"].shape), (1, 3, 45, 80))
        for name in original:
            torch.testing.assert_close(permuted[name], original[name][:, order], atol=2e-6, rtol=1e-5)

    def test_zero_initialized_branch_gets_gradients_and_control_bypasses_it(self):
        rgb, tof, queries = inputs(self.device)
        for carrier in ("global", "spatial"):
            model = copy.deepcopy(self.model)
            model.carrier = carrier
            output = model(rgb, tof, queries[:2])
            loss = output["distance_logits"].square().mean() + output["mask_logits"].square().mean()
            loss.backward()
            gradient = model.spatial_query_projection.weight.grad
            if carrier == "global":
                self.assertIsNone(gradient)
            else:
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_original_validation_and_missing_sensor_semantics_are_preserved(self):
        rgb, tof, queries = inputs(self.device)
        missing = tof.clone()
        missing[:, :8, 0] = float("nan")
        missing[:, :8, 1] = 0
        alternate = missing.clone()
        alternate[:, :8, 0] = 12345.
        with torch.inference_mode():
            first = self.model.forward_features(rgb, missing, queries[:2])
            second = self.model.forward_features(rgb, alternate, queries[:2])
        self.assertEqual(int(first["tof_valid"].sum()), 56)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            QuerySpatialFitNet(None, carrier="unsupported")
        with self.assertRaises(ValueError):
            self.model(rgb[:, :, :-1], tof, queries)
        for bad in (queries[:0], queries[:, [1, 0, 2, 3]], queries * float("nan")):
            with self.assertRaises(ValueError):
                self.model.query_geometry(bad)
        invalid = tof.clone()
        invalid[0, 0, 1] = .5
        with self.assertRaises(ValueError):
            self.model(rgb, invalid, queries)


if __name__ == "__main__":
    unittest.main()
