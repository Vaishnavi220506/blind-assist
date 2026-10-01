"""Small CPU checks of GLRT, angular mapping, observable gate and covariance."""
import unittest
import numpy as np
import cnh_segment_joint_readout as r


class JointReadoutTests(unittest.TestCase):
    def test_glrt_matches_independent_dense_constrained_search(self):
        # Include interior and both b boundaries, plus a=0 and disappearance.
        for zf, zo, rho, cap in [(2., 1., .3, 2.), (1., -2., .6, 1.),
                                  (2., 4., -.4, 1.), (.3, 3., .8, 2.),
                                  (-1., 3., .2, 2.)]:
            aa = np.linspace(0, 8, 1601)[:, None]
            bb = np.linspace(0, cap, 401)[None, :]
            inv = np.linalg.inv(np.array([[1., rho], [rho, 1.]]))
            gain = (2*(aa*(inv[0, 0]*zf+inv[0, 1]*zo)
                       +bb*(inv[1, 0]*zf+inv[1, 1]*zo))
                    -(inv[0, 0]*aa**2+2*inv[0, 1]*aa*bb+inv[1, 1]*bb**2))
            index = np.unravel_index(gain.argmax(), gain.shape)
            observed = float(r.glrt(zf, zo, rho, cap))
            if index[0] == 0 or zf <= 0:
                self.assertEqual(observed, -np.inf)
            else:
                self.assertAlmostEqual(observed, np.sqrt(max(0, gain[index])), places=4)
        self.assertEqual(float(r.glrt(0., 10., 0., 20.)), -np.inf)
        self.assertAlmostEqual(float(r.glrt(2., 0., 0., 0.)), 2.)

    def test_query_membership_uses_hypothetical_range_and_public_transform(self):
        mask = np.zeros((128, 128), bool)
        mask[55:73, 55:73] = True
        membership, ranges = r.query_membership(mask, np.eye(4))
        self.assertEqual(membership.shape, (len(r.WINDOWS_R), 6))
        self.assertTrue(membership.any())
        self.assertTrue(np.any(membership != membership[0]))
        transform = np.eye(4); transform[0, 3] = 100.
        far, same_ranges = r.query_membership(mask, transform)
        self.assertFalse(far.any())
        np.testing.assert_array_equal(ranges, same_ranges)
        empty, _ = r.query_membership(np.zeros_like(mask), np.eye(4))
        self.assertFalse(empty.any())

    def test_scope_requires_observed_background_and_excludes_other_candidates(self):
        f = np.zeros((8, 8), np.float32); f[3, 3] = .2
        candidate = dict(coverage=f)
        member = np.ones((len(r.WINDOWS_R), 6), bool)
        absolute = np.zeros((8, 8, 16)); absolute[:, :, 8] = 100.
        sc = r.observable_scope([candidate], absolute, [member])[0]
        self.assertTrue(sc['eligible'].all())
        self.assertEqual(sc['ring'].sum(), 8)
        self.assertFalse(sc['ring'][3, 3])
        self.assertTrue(sc['hypotheses'].any())
        self.assertFalse(sc['hypotheses'].all())
        dark = r.observable_scope([candidate], np.zeros_like(absolute), [member])[0]
        self.assertFalse(dark['eligible'].any())
        other = f.copy(); other[:] = 0; other[3, 4] = .2
        rows = r.observable_scope([candidate, dict(coverage=other)], absolute, [member, member])
        self.assertFalse(rows[0]['ring'][3, 4])
        self.assertFalse(rows[0]['eligible'].any())
        wide = f.copy(); wide[3, 3] = .6
        self.assertFalse(r.observable_scope([dict(coverage=wide)], absolute, [member])[0]['eligible'].any())

    def test_linear_moments_matches_dense_splat_covariance(self):
        import torch
        rng = np.random.default_rng(101)
        tensor = lambda x: torch.tensor(x, dtype=torch.float64, device='cpu')
        weights = tensor(rng.normal(size=(4, 7)))
        total = tensor(rng.normal(size=7))
        variance = tensor(rng.uniform(.2, 2., 7))
        past = [(tensor(rng.uniform(0, 1, (7, 7))), tensor(rng.uniform(.2, 2., 7))) for _ in range(3)]
        mean, var, moved = r.linear_moments(weights, total, variance, past, torch)
        dense = torch.diag(variance)
        for A, v in past:
            dense += A@torch.diag(v)@A.T
        expected = weights@dense@weights.T
        torch.testing.assert_close(mean, weights@total)
        torch.testing.assert_close(var, expected.diag())
        cov = (weights[0]*weights[1])@variance
        for a, v in moved:
            cov += (a[0]*a[1])@v
        torch.testing.assert_close(cov, expected[0, 1])
        diagonal_approx = (weights*weights)@dense.diag()
        self.assertFalse(torch.allclose(var, diagonal_approx))


if __name__ == '__main__':
    unittest.main()
