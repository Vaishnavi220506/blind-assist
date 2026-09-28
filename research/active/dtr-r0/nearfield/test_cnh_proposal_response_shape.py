import unittest,sys
from dataclasses import replace
import numpy as np
import cnh_proposal_response_shape as D
sys.path.insert(0,str(D.DATA/'source'))
from cnh_route_sensor import SensorParameters,synthesize_response,RAW_BIN_M

class ResponseShapeTests(unittest.TestCase):
    def setUp(self):
        self.distance=np.full((8,8,1),2.);self.p=SensorParameters(noise_scale=0,crosstalk_fraction=0,neighbour_leak=0,tail_mass=0)
    def synth(self,p,d=None):
        return synthesize_response(self.distance if d is None else d,.5,1.,1.,params=p,seed=3)['histogram']
    def test_nominal_deterministic(self):
        p=replace(self.p,noise_scale=1); np.testing.assert_array_equal(self.synth(p),self.synth(p))
    def test_broadening_spreads_energy(self):
        a=self.synth(self.p)[0,0];b=self.synth(replace(self.p,pulse_sigma_bins=4))[0,0]
        self.assertLess(b.max(),a.max());self.assertAlmostEqual(a.sum(),b.sum(),places=6)
    def test_zero_shift_moves_native_peak(self):
        a=self.synth(self.p)[0,0];b=self.synth(replace(self.p,range_zero_m=4*RAW_BIN_M))[0,0]
        self.assertEqual(int(a.argmax())-int(b.argmax()),4)
    def test_crosstalk_shape_present_without_scene(self):
        d=np.full((8,8,1),np.inf);p=replace(self.p,crosstalk_fraction=.02)
        a=self.synth(p,d)[0,0];b=self.synth(replace(p,crosstalk_range_m=.3),d)[0,0]
        self.assertNotEqual(int(a.argmax()),int(b.argmax()));self.assertGreater(a.sum(),0)
    def test_no_wrap_at_origin(self):
        d=np.full((8,8,1),.02);a=self.synth(replace(self.p,range_zero_m=.15),d)
        self.assertEqual(float(a.sum()),0.)

if __name__=='__main__':unittest.main()
