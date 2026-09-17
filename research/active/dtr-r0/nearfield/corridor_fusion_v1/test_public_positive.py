import unittest
import numpy as np
import torch
from public_positive import PositiveHead, spatial_features, add_positive, select_threshold


class PositiveTests(unittest.TestCase):
    def test_geometric_features(self):
        tokens = np.zeros((1, 132, 21), np.float32)
        tokens[0, 0, 5:8] = [1/4, 0, 1/3]
        tokens[0, 0, 8:14] = [.25, .25, 0, 0, 1/3, 1/3]
        x = spatial_features(tokens)
        np.testing.assert_allclose(x[0, 0, 21:], [.8, .3, .6]*3, atol=1e-6)
        self.assertEqual(sum(p.numel() for p in PositiveHead().parameters()), 1537)

    def test_unknown_and_or(self):
        a = np.array([True, False, False])
        logits = np.full((3, 128), 20.)
        valid = np.zeros((3, 128), bool); valid[2, 0] = True
        p, _, branch = add_positive(a, logits, valid, 0.)
        self.assertEqual(p.tolist(), [True, False, True])
        self.assertEqual(branch.tolist(), [False, False, True])

    def test_invalid_nan_masked(self):
        x = torch.full((2, 128, 30), float('nan'))
        valid = torch.zeros((2, 128), dtype=torch.bool)
        self.assertTrue(torch.all(PositiveHead()(x, valid) == -30.))

    def test_final_or_calibration_includes_disable(self):
        a = np.array([True, False]); y = np.array([True, False])
        logits = np.ones((2, 128), np.float32)
        valid = np.ones((2, 128), bool)
        best, _ = select_threshold(a, logits, valid, y, np.ones(2, bool))
        self.assertGreater(best['threshold'], 1.)
        p, _, _ = add_positive(a, logits, valid, best['threshold'])
        np.testing.assert_array_equal(p, a)


if __name__ == '__main__':
    unittest.main()
