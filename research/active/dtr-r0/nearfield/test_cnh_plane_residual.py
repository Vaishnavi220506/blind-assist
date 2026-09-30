"""Focused checks for range recovery and destructive decomposition mistakes."""
from dataclasses import replace
import unittest

import numpy as np

import cnh_proposal_attribution_scenes as S  # selects the retained sensor source
from cnh_route_sensor import synthesize_response
from cnh_plane_residual import PlaneFitter


class PlaneResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, _ = S.nominal_parameters()
        cls.fitter = PlaneFitter(cls.params)

    def test_recover_plane_from_independent_hard_bin_forward(self):
        rays, weights = S.ray_grid()
        distance = 1.5/rays[..., 2]
        obs = synthesize_response(distance, .55, rays[..., 2], weights,
            params=replace(self.params, noise_scale=0), seed=9)
        h = obs['histogram'].reshape(8, 8, 16, 8).sum(-1)
        fit = self.fitter.fit(h, obs['ambient'])
        self.assertLess(abs(fit['offset_m']-1.5), .08)
        self.assertGreater(fit['normal'][2], .99)
        relative_error = np.linalg.norm(fit['residual'])/np.linalg.norm(h-self.fitter.electronics)
        self.assertLess(relative_error, .10)

    def test_signed_input_roundtrip_including_negative_bins(self):
        rng = np.random.default_rng(37)
        h = rng.normal(0, 2, (8, 8, 16))
        fit = self.fitter.fit(h, np.ones((8, 8))*4)
        np.testing.assert_allclose(fit['residual']+fit['plane']+fit['electronics'], h, atol=1e-12)
        self.assertTrue((fit['residual'] < 0).any())

    def test_reject_nonfinite_observation(self):
        h = np.zeros((8, 8, 16)); h[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.fitter.fit(h, np.ones((8, 8)))


if __name__ == '__main__':
    unittest.main()
