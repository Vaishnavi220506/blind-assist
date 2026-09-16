import unittest
import numpy as np
from mz160_dense_parallax import calibration,depth_bounds,dense_pair

class DenseParallaxTests(unittest.TestCase):
    def records(self,a=20.,b=.5):
        return [dict(zone=i,inverse_depth=float(x),disparity=float(a*x+b)) for i,x in enumerate(np.linspace(.3,1.,16))]

    def test_signed_scale_exclusion_and_depth(self):
        for a in (20.,-20.):
            model=calibration(self.records(a),3)
            self.assertNotIn(3,model['anchor_zones'])
            bounds=depth_bounds(a/2+.5,model)
            self.assertLessEqual(bounds[0],2.);self.assertGreaterEqual(bounds[1],2.)
        self.assertIsNone(calibration(self.records(0.)))
        bad=self.records()
        for r in bad:r['inverse_depth']=.3
        self.assertIsNone(calibration(bad))

    def test_missing_or_zero_crossing_is_unknown(self):
        self.assertIsNone(calibration(self.records()[:7]))
        self.assertIsNone(depth_bounds(.5,calibration(self.records())))
        self.assertIsNone(depth_bounds(2.,None))

    def test_dense_displacement_and_border_mask(self):
        rng=np.random.default_rng(160016)
        left=rng.integers(0,256,(96,256),dtype=np.uint8)
        previous=np.zeros_like(left);previous[:,:-6]=left[:,6:]
        row=dict(rgb_intrinsics=dict(width=256,height=96,fx=200.,fy=200.,cx=128.,cy=48.),camera_pitch_deg=0.)
        d,valid,audit=dense_pair(row,row,left,previous,0.,0.)
        interior=valid[10:-10,65:-65];values=d[10:-10,65:-65][interior]
        self.assertGreater(len(values),3000)
        self.assertLess(abs(float(np.median(values))-6.),.1)
        self.assertFalse(valid[:2].any());self.assertFalse(valid[-2:].any())
        self.assertGreater(audit['valid_disparity_pixels'],3000)

if __name__=='__main__':unittest.main()
