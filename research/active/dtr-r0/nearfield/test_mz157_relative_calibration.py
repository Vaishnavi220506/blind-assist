import copy
import unittest
import numpy as np
from mz157_relative_calibration import predicted_ranges,fitted,assignments,calibrate


class CalibrationTests(unittest.TestCase):
    def test_oblique_planar_arithmetic_slant_forward_model(self):
        n=np.linspace(1.,1.2,32);d=np.full(32,.5);mode=np.stack([d,n],-1)[None]
        value=predicted_ranges([.6,.2],mode)[0]
        self.assertAlmostEqual(value,float((n/.5).mean()))
        self.assertGreater(abs(value-1/np.mean(.5/n)),.001)

    def test_fit_two_parameters_and_permuted_two_return_slots(self):
        groups=[];expected=[.6,.2]
        for i in range(8):
            modes=np.array([np.stack([np.full(32,.15+i*.025),np.linspace(1,1.05,32)],-1),
                np.stack([np.full(32,.7+i*.025),np.linspace(1,1.05,32)],-1)])
            ranges=predicted_ranges(expected,modes)
            if i%2:ranges=ranges[::-1]
            groups.append(dict(zone=i,modes=modes,ranges=ranges,slots=[0,1]))
        fit=fitted(groups,latent=True)
        self.assertTrue(fit['valid']);np.testing.assert_allclose(fit['parameters'],expected,atol=1e-4)
        self.assertEqual(assignments(groups,expected)[0],[0,1]);self.assertEqual(assignments(groups,expected)[1],[1,0])

    def test_rank_deficiency_is_unknown(self):
        groups=[dict(zone=i,modes=np.array([np.tile([.5,1.],(32,1))]),ranges=np.array([2.]),slots=[0]) for i in range(8)]
        self.assertFalse(fitted(groups,latent=False)['valid'])

    def test_missing_and_merged_only_cannot_calibrate(self):
        row=dict(rgb_intrinsics=dict(width=40,height=30,fx=40.,fy=40.,cx=20.,cy=15.),
            tof_packet_received=False,tof_zones=[dict(zone_id=1,theta_bounds_deg=[-3,3],phi_bounds_deg=[-3,3],
                targets=[dict(status='SIM_MERGED',distance_m=2.)])])
        old=copy.deepcopy(row);relative=np.tile(np.linspace(0,1,40),(30,1))
        for arm in ('median','regional_modes'):
            depth,audit=calibrate(row,relative,arm)
            self.assertTrue(np.isnan(depth).all());self.assertFalse(audit['fit']['valid'])
        self.assertEqual(row,old)


if __name__=='__main__':unittest.main()
