"""Focused calibration checks; no experimental outcome access."""
import unittest
import numpy as np
from run_mz143_corridor_evidence import operating_threshold


class CalibrationTests(unittest.TestCase):
    def test_largest_threshold_retains_each_incumbent_true_frame(self):
        scores=np.array([.8,.35,.2,.7])
        target=np.array([True,True,False,False])
        baseline=np.array([True,True,True,False])
        cutoff=operating_threshold(scores,target,baseline)
        self.assertEqual(cutoff,.35)
        self.assertTrue(np.all((scores>=cutoff)[target&baseline]))
        self.assertFalse(np.all((scores>cutoff)[target&baseline]))

    def test_zero_positive_score_exposes_all_alert(self):
        cutoff=operating_threshold([0.,.9],[True,False],[True,False])
        self.assertEqual(cutoff,0.)
        self.assertTrue(np.all(np.array([0.,.9])>=cutoff))

    def test_rejects_missing_or_invalid_support(self):
        with self.assertRaises(ValueError):operating_threshold([.5],[False],[True])
        with self.assertRaises(ValueError):operating_threshold([float('nan')],[True],[True])
        with self.assertRaises(ValueError):operating_threshold([.2,.3],[True],[True])


if __name__=='__main__':unittest.main()
