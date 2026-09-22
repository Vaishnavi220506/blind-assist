import unittest
import numpy as np
from cnh_fingerprint import cosine, pairwise_auc, evaluate_contrast


class FingerprintChecks(unittest.TestCase):
    def test_signed_shape_and_amplitude_invariance(self):
        np.testing.assert_allclose(cosine([[1, -2], [3, -6], [-1, 2]], [1, -2]), [1, 1, -1])
        self.assertTrue(np.isnan(cosine([[0, 0]], [1, 2])[0]))

    def test_pairwise_ties_and_order(self):
        self.assertEqual(pairwise_auc([1, 1], [1, 1]), .5)
        self.assertEqual(pairwise_auc([1, 2], [-1, 0]), 1)
        self.assertEqual(pairwise_auc([-1, 0], [1, 2]), 0)

    def test_block_gate_cannot_hide_one_failing_block(self):
        p = {"minimum_frames_per_block": 3, "minimum_block_separation": .1, "minimum_pairwise_auc": .9}
        blocks = np.repeat(np.arange(4), 5)
        good = np.ones(20)
        changed = np.full(20, .5)
        self.assertEqual(evaluate_contrast(good, changed, blocks, blocks, p)["verdict"], "SUPPORTED_FOR_RETROSPECTIVE_FINGERPRINT")
        good[-5:] = .55
        self.assertEqual(evaluate_contrast(good, changed, blocks, blocks, p)["verdict"], "NOT_SUPPORTED")
        self.assertEqual(evaluate_contrast(good[:10], changed, blocks[:10], blocks, p)["verdict"], "NOT_EVALUABLE")


if __name__ == '__main__':
    unittest.main()
