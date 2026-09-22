"""Observation-only geometry checks: synthetic planes and public camera matrices.

No source depth, labels, saved model predictions or protected test rows are read.
"""
import json
import unittest

import cv2
import numpy as np

import ba_nfo_depthpro as d


def ray_xy(matrix, height, width):
    yy, xx = np.mgrid[:height, :width]
    uv = np.stack([(xx+.5)/width*2-1, 1-(yy+.5)/height*2,
                   np.ones_like(xx)], -1)
    rays = uv@matrix.T
    return rays[..., 0]/-rays[..., 2], rays[..., 1]/-rays[..., 2]


def public_matrices():
    rows = json.loads((d.OUT/'observations.json').read_text())
    matrices = {tuple(np.asarray(r['camera_matrix']).ravel()) for r in rows}
    return [np.array(x).reshape(3, 3) for x in sorted(matrices)]


class DepthProGeometryTest(unittest.TestCase):
    def check_plane(self, matrix, height, width):
        rgb = np.zeros((height, width, 3), np.uint8)
        _, focal, bx, by = d.rectify(rgb, matrix)
        self.assertGreaterEqual(float(bx.min()), 1.)
        self.assertLessEqual(float(bx.max()), width-2.)
        self.assertGreaterEqual(float(by.min()), 1.)
        self.assertLessEqual(float(by.max()), height-2.)
        rx, ry = ray_xy(matrix, height, width)
        np.testing.assert_allclose((bx+.5-width/2)/focal, rx, atol=1e-6)
        np.testing.assert_allclose((height/2-by-.5)/focal, ry, atol=1e-6)
        # A synthetic oblique plane has affine inverse optical-z on target rays.
        yy, xx = np.mgrid[:height, :width]
        inverse_z = (.5+.03*(xx+.5-width/2)/focal
                     -.02*(height/2-yy-.5)/focal).astype(np.float32)
        predicted_z = 1/inverse_z
        restored = 1/cv2.remap(1/predicted_z, bx, by, cv2.INTER_LINEAR)
        expected = 1/(.5+.03*rx-.02*ry)
        # OpenCV bilinear maps round fractions to1/32pixel. Bound that error,
        # plus float32 arithmetic, rather than demand exact floating equality.
        inverse_error = .05/(64*focal)+2e-7
        self.assertLessEqual(float(np.abs(1/restored-1/expected).max()), inverse_error)
        self.assertTrue(np.isfinite(restored).all())
        self.assertTrue((restored > 0).all())

    def test_all_public_low_matrices_and_worst_native_plane(self):
        matrices = public_matrices()
        self.assertGreater(len(matrices), 1)
        for matrix in matrices:
            self.check_plane(matrix, 192, 256)
        worst = max(matrices, key=lambda x: np.max(np.abs(x[2, :2])))
        self.check_plane(worst, 768, 1024)

    def test_forward_rgb_map_matches_inverse_homography(self):
        matrix = max(public_matrices(), key=lambda x: np.max(np.abs(x[2, :2])))
        height, width = 192, 256
        rx, ry = ray_xy(matrix, height, width)
        rgb = np.stack([rx, ry, np.ones_like(rx)], -1).astype(np.float32)
        rectified, focal, _, _ = d.rectify(rgb, matrix)
        yy, xx = np.mgrid[:height, :width]
        target = np.stack([(xx+.5-width/2)/focal,
                           (height/2-yy-.5)/focal, -np.ones_like(xx)], -1)
        projected = target@np.linalg.inv(matrix).T
        uv = projected[..., :2]/projected[..., 2:]
        sx = (uv[..., 0]+1)*width/2-.5
        sy = (1-uv[..., 1])*height/2-.5
        interior = (sx >= 1) & (sx <= width-2) & (sy >= 1) & (sy <= height-2)
        self.assertTrue(interior.any())
        # RGB interpolation is bilinear in original pixels, whose rays are
        # projective under oblique cameras; agreement has interpolation error.
        self.assertLess(float(np.max(np.abs(rectified[..., :2][interior]-target[..., :2][interior]))), .0005)
        np.testing.assert_allclose(rectified[..., 2][interior], 1.)

    def test_low_and_native_nearest_grid_have_inherited_offset(self):
        # Prepared RGB uses4x4 area bins, while existing depth uses nearest4j.
        # This metadata-only check prevents claiming identical pixel rays.
        matrix = max(public_matrices(), key=lambda x: np.max(np.abs(x[2, :2])))
        low_x, low_y = ray_xy(matrix, 192, 256)
        native_x, native_y = ray_xy(matrix, 768, 1024)
        self.assertGreater(float(np.max(np.abs(low_x-native_x[::4, ::4]))), 1e-4)
        self.assertGreater(float(np.max(np.abs(low_y-native_y[::4, ::4]))), 1e-4)
        # Original pixel-array coordinates corresponding to a low-pixel centre.
        low_center_in_native = (np.arange(256)+.5)*4-.5
        np.testing.assert_array_equal(low_center_in_native-np.arange(0, 1024, 4), 1.5)


if __name__ == '__main__':
    unittest.main()
