"""Exact threshold coverage, including extreme logits and tied scores."""
import unittest
import numpy as np
from check_mz136_readout_transfer import exact_thresholds


class TransferThresholdTests(unittest.TestCase):
    def test_extreme_logits_and_ties_keep_every_distinct_decision(self):
        scores=np.array([-1000.,-1000.,-15.6,-15.5,0.,1000.])
        flags={tuple(scores>=t) for t in exact_thresholds(scores)}
        self.assertEqual(len(flags),len(np.unique(scores))+1)
        self.assertIn((True,)*len(scores),flags)
        self.assertIn((False,)*len(scores),flags)
        self.assertIn((False,False,False,True,True,True),flags)
        self.assertTrue(all(f[0]==f[1] for f in flags))


if __name__=='__main__':unittest.main()
