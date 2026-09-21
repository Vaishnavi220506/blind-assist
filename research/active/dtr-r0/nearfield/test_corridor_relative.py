"""Focused boundaries for interval bands, missing evidence and aggregation."""
import unittest
import numpy as np
import torch
from corridor_relative_model import bands, Head
from run_corridor_relative import costs


class RelativeTest(unittest.TestCase):
    def test_partition(self):
        for interval in ([1.,2.],[.1,.4],[2.8,3.4],[3.4,4.]):
            yy,xx,masks,geometry=bands([0,0,192,256],interval)
            self.assertEqual((len(yy),len(xx)),(360,640))
            self.assertTrue(np.all(sum(m.astype(int) for m in masks)==1))
            # Independently project every ray at both clipped depth endpoints.
            f=640/(2*np.tan(np.deg2rad(50)))
            a=(xx[None,:]+.5-320)/f; b=(yy[:,None]+.5-180)/f
            near,far=max(interval[0],.3),min(interval[1],3.)
            if near<far:
                at_far=(abs(a*far)<=.3)&(b*far>=-.2)&(b*far<=.9)
                at_near=(abs(a*near)<=.3)&(b*near>=-.2)&(b*near<=.9)
                np.testing.assert_array_equal(masks[0],at_far)
                np.testing.assert_array_equal(masks[0]|masks[1],at_near)
            else:
                self.assertFalse(masks[0].any() or masks[1].any())
            self.assertTrue(np.isfinite(geometry).all())

    def test_permutation_and_missing(self):
        torch.manual_seed(17); head=Head().eval(); x=torch.rand(2,64,60); x[:,:,-1]=1
        with torch.no_grad():
            torch.testing.assert_close(head(x),head(x[:,torch.randperm(64)]),rtol=1e-6,atol=1e-7)
            x[:,:,-1]=0
            torch.testing.assert_close(head(x),torch.full((2,),-20.))

    def test_invalid_cannot_outvote_low_valid_score(self):
        head=Head().eval(); x=torch.zeros(1,64,60); x[:,0,-1]=1
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head.net[-1].bias.fill_(-30.)
            self.assertEqual(head(x).item(),-30.)

    def test_cost_includes_held_exit(self):
        # A new final-positive trigger carries a false alert to the next negative.
        meta=[dict(clip_id='x',time_s=i*.2,layout_relation='BOUNDARY') for i in range(4)]
        result=costs(np.array([-2,2,-2,-2]),1,[False,True,False,False],meta,[False]*4)
        self.assertEqual(result['Boundary_current']['added_FP'],0)
        self.assertEqual(result['Boundary_hold']['added_FP'],1)
        self.assertFalse(result['Boundary_hold']['pass_cost'])

    def test_disabled_float64_cutoff(self):
        meta=[dict(clip_id='x',time_s=0.,layout_relation='OUTSIDE')]
        result=costs(np.array([1.],np.float32),np.nextafter(1.,np.inf),[False],meta,[False])
        self.assertEqual(result['Core_current']['added_FP'],0)


if __name__=='__main__':
    unittest.main()
