"""Synthetic adapter tests; no dataset, checkpoint, or evaluator inputs."""
import unittest

import numpy as np
import torch

from mz164_metric_prior import MODEL_SHAPE, camera_matrix, native_intrinsics, predict, prepare_rgb


INTR = dict(width=640, height=360, fx=457.007, fy=457.007, cx=320., cy=180.)


class FakeModel:
    training = False
    resolution_level = 0
    device = torch.device('cpu')

    def __init__(self):
        self.calls = []

    def infer(self, rgb, camera=None, normalize=True):
        self.calls.append((rgb.clone(), None if camera is None else camera.clone(), normalize))
        k = torch.tensor(camera_matrix(INTR)).unsqueeze(0) if camera is None else camera
        h,w = MODEL_SHAPE
        y,x = torch.meshgrid(torch.arange(h)+.5, torch.arange(w)+.5, indexing='ij')
        rays = torch.stack(((x-k[0,0,2])/k[0,0,0],(y-k[0,1,2])/k[0,1,1],torch.ones_like(x)),0)[None]
        rays = torch.nn.functional.normalize(rays, dim=1)
        return dict(depth=torch.full((1,1,h,w), 2.5), intrinsics=k, rays=rays)


class AdapterTests(unittest.TestCase):
    def test_camera_centres_and_independent_scale(self):
        k = camera_matrix(INTR)
        expected = np.array([[INTR['fx'],0,INTR['cx']],[0,INTR['fy'],INTR['cy']],[0,0,1.]])
        np.testing.assert_allclose(native_intrinsics(k), expected, atol=3e-5)
        for u,v in [(0,0),(320,180),(639,359),(27.3,298.1)]:
            sx,sy = 644/640,364/360
            resized_half = np.array([(u+.5)*sx,(v+.5)*sy,1.])
            np.testing.assert_allclose(np.linalg.inv(k)@resized_half,
                [(u-INTR['cx'])/INTR['fx'],(v-INTR['cy'])/INTR['fy'],1.], atol=2e-7)

    def test_both_arms_same_rgb_known_camera_and_axis_depth(self):
        rgb = np.arange(360*640*3, dtype=np.uint32).reshape(360,640,3).astype(np.uint8)
        model = FakeModel()
        first,a = predict(model,rgb,INTR,'known_camera')
        second,b = predict(model,rgb,INTR,'estimated_camera')
        self.assertTrue(torch.equal(model.calls[0][0],model.calls[1][0]))
        self.assertIsNotNone(model.calls[0][1]); self.assertIsNone(model.calls[1][1])
        self.assertEqual(first.shape,(360,640)); self.assertEqual(first.dtype,np.float32)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_allclose(first,2.5,atol=3e-7)
        self.assertLess(a['supplied_ray_max_abs_error'],2e-5)
        self.assertIsNone(b['supplied_intrinsics_resized_half_centres'])

    def test_input_validation_and_metadata_has_no_influence(self):
        with self.assertRaises(ValueError): prepare_rgb(np.zeros((360,640,3),dtype=np.float32))
        with self.assertRaises(ValueError): camera_matrix(dict(INTR, fx=0))
        with self.assertRaises(ValueError): predict(FakeModel(),np.zeros((360,640,3),np.uint8),INTR,'oracle')
        np.testing.assert_array_equal(camera_matrix(INTR),camera_matrix(dict(INTR,id='unused',truth=True)))


if __name__ == '__main__':
    unittest.main()
