"""Focused model/interface falsifiers for the new angular contrast."""
import math
import random
import unittest
import numpy as np

from mz133_angular_resolution import geometry, trace, reduce_zone, predict
from mz115_zonal_tof import measure_zone, zone_geometry


class AngularResolutionTests(unittest.TestCase):
    def test_legacy_model_equivalence_including_mixed_and_missing(self):
        for ranges in ([1.]*9,[1.,1.1,1.2,2.,2.1,2.2,math.inf,math.inf,3.], [math.inf]*9):
            for packet in (True,False):
                for rho in (1.,.001):
                    hits=[dict(range_m=None if math.isinf(d) else d,reflectance_proxy=rho) for d in ranges]
                    old,_=measure_zone(hits,random.Random(113),packet)
                    new,_,_=reduce_zone(ranges,[rho]*9,random.Random(113),packet,1.,.04)
                    self.assertEqual(new,old)

    def test_dense_lattice_and_physical_active_domain(self):
        z8,r8=geometry(8,12); z32,r32=geometry(32,3)
        a=r8.reshape(-1,3); b=r32.reshape(-1,3)
        np.testing.assert_array_equal(a[np.lexsort(a.T[::-1])],b[np.lexsort(b.T[::-1])])
        for zones, expected in ((z8,32),(z32,512)):
            active=[z for z in zones if z['theta_bounds_deg'][0]>=-11.25 and z['theta_bounds_deg'][1]<=11.25]
            self.assertEqual(len(active),expected)
            self.assertEqual(min(z['theta_bounds_deg'][0] for z in active),-11.25)
            self.assertEqual(max(z['theta_bounds_deg'][1] for z in active),11.25)
        old,_=zone_geometry(17); self.assertEqual(z8[17],old)

    def test_first_hit_occludes_background_and_parallel_miss(self):
        frame=dict(camera=dict(x=0,y=0,z=1,pitch=0,yaw=0,roll=0),objects=[
            dict(center_m=[2,0,1],size_m=[.2,.2,.2],tof_reflectance_proxy=.3)])
        spec=dict(background=dict(center_m=[3,0,1],size_m=[.2,2,2]),floor=None)
        rays=np.array([[[1.,0.,0.],[1.,.2,0.],[1.,2.,0.]]])
        ranges,rho,owners,_,_=trace(frame,spec,rays)
        self.assertAlmostEqual(ranges[0,0],1.9)
        self.assertEqual(owners[0,0],0)
        self.assertAlmostEqual(rho[0,0],.3)
        self.assertEqual(owners[0,1],1)
        self.assertTrue(math.isinf(ranges[0,2]))

    def test_signal_budget_removes_detection_not_claiming_clearance(self):
        high,_,_=reduce_zone([3.]*9,[.5]*9,random.Random(4),True,1.,.04)
        weak,_,_=reduce_zone([3.]*9,[.5]*9,random.Random(4),True,1/16,.16)
        self.assertTrue(high); self.assertFalse(weak)
        zones,_=geometry(8,3)
        row=dict(id='test',time_s=0,camera_pitch_deg=-3,camera_in_body_m=[0,0,1.7],
                 rgb_intrinsics=dict(cx=320,cy=180,fx=457,fy=457),
                 tof_zones=[dict(z,targets=[]) for z in zones])
        p=predict(row,0,False)
        self.assertFalse(p['tof']); self.assertFalse(p['full'])
        self.assertEqual(p['candidate_state'],'UNKNOWN')
        self.assertTrue(predict(row,0,True)['full'])


if __name__=='__main__': unittest.main()
