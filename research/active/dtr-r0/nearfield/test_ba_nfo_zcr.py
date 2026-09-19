import unittest
import numpy as np
import torch
from ba_nfo_zcr import residual, prediction, select_cutoff


class TestZCR(unittest.TestCase):
    def test_complete_zone_even_median_and_offset_invariance(self):
        x = torch.tensor([[[.1,.2,.3,.8],[.4,.5,.6,.7]]])
        boxes = [[0,0,2,2],[0,2,2,4]]
        r = residual(x, boxes)
        np.testing.assert_allclose(r[0,:,:2], x[0,:,:2]-.3, atol=1e-7)
        np.testing.assert_allclose(r[0,:,2:], x[0,:,2:]-.65, atol=1e-7)
        shifted = x.clone(); shifted[:,:,:2] += .1
        np.testing.assert_allclose(residual(shifted,boxes),r,atol=1e-7)
        # Relative score is not a probability; negatives must not be clipped.
        self.assertLess(r.min(), 0)

    def test_outside_unchanged_and_near_uniform_zone(self):
        x = torch.tensor([[[.9,.9,.07],[.9,.9,.09]]])
        r = residual(x, [[0,0,2,2]])[0].numpy(); s = x[0].numpy()
        np.testing.assert_array_equal(r[:,:2],0)
        self.assertTrue(np.isnan(r[:,2]).all())
        np.testing.assert_array_equal(prediction(s,r,.01)[:,2], [False,True])
        self.assertFalse(prediction(s,r,.01)[:,:2].any())
        self.assertTrue(prediction(s,r,0)[:,:2].all())

    def test_calibration_exact_ties_and_bruteforce(self):
        scores = np.array([-.2,-.1,0,0,.1,.2,.3],np.float32)
        truth = np.array([1,1,0,1,0,1,1],bool)
        chosen, _ = select_cutoff(scores,truth)
        feasible = []
        for cut in np.unique(scores):
            pred = scores>=cut; tp=int((pred&truth).sum()); fp=int((pred&~truth).sum())
            if tp/truth.sum()>=.95: feasible.append((tp/(truth.sum()+fp),-float(cut)))
        expected=max(feasible)
        self.assertEqual(chosen['cutoff'],-expected[1])
        self.assertEqual(chosen['iou'],expected[0])


if __name__ == '__main__':
    unittest.main()
