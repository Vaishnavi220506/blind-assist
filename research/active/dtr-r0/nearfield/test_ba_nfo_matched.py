"""Decision-critical observation, UNKNOWN and operating-point checks."""
import unittest
import numpy as np
import torch
import ba_nfo_matched as m


class MatchedTests(unittest.TestCase):
    def test_public_fields_keep_missing_distinct(self):
        values=np.full(64,1.4,np.float32);values[0]=np.nan
        zones=m.public_zones(values)
        self.assertEqual(zones.shape,(6,8,8))
        self.assertEqual(zones[0,0,0],0)
        self.assertEqual(zones[1,0,0],0)
        self.assertEqual(zones[2,0,0],1)
        self.assertEqual(zones[3,0,0],1)
        self.assertAlmostEqual(float(zones[0,1,1]),1.4/8,places=6)

    def test_one_pixel_foreground_is_included_and_unknown_excluded(self):
        depth=np.full((10,10),4.,np.float32)
        depth[0,0]=1.4;depth[0,1]=np.nan
        known,truth,mixed,thin=m.masks(depth,np.array([[0,0,10,10]]),2.)
        self.assertEqual(int(truth.sum()),1)
        self.assertEqual(int(mixed.sum()),99)
        self.assertEqual(int(thin.sum()),99)
        self.assertFalse(known[0,1])
        self.assertEqual(m.metrics(m.counts(np.ones_like(known),truth,known))['fn'],0)

    def test_calibration_is_validation_recall_constrained(self):
        hist=np.zeros((2,1001),np.int64)
        hist[0,800]=95;hist[0,200]=5;hist[1,100]=100
        c=m.calibration(hist)
        self.assertTrue(c['feasible'])
        self.assertEqual(c['recall'],1.)
        self.assertEqual(c['iou'],1.)
        hist[:]=0;hist[0,0]=100
        self.assertFalse(m.calibration(hist)['feasible'])

    def test_depth_zero_logit_means_one_meter_and_nesting(self):
        out=torch.zeros(1,1,2,2)
        p=m.probabilities(out,'depth')
        self.assertTrue(torch.allclose(p[:,0],torch.full_like(p[:,0],.5)))
        nfo=m.probabilities(torch.tensor([4.,-4.,2.,-2.]).view(1,4,1,1),'nfo')
        self.assertTrue(bool((nfo[:,:-1]<=nfo[:,1:]).all()))

    def test_unknown_targets_do_not_change_loss(self):
        d=torch.full((1,2,2),2.5);d[0,0,0]=float('nan')
        for arm,channels in [('depth',1),('nfo',4)]:
            a=torch.zeros(1,channels,2,2)
            b=a.clone();b[:,:,0,0]=100
            torch.testing.assert_close(m.loss_fn(a,d,arm),m.loss_fn(b,d,arm))


if __name__=='__main__':
    unittest.main()
