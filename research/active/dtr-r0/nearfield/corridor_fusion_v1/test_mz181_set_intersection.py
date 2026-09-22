import math
import unittest
from mz181_set_intersection import refine, ray_witness, direction, CORRIDOR


def row(origin=(0,0,1),pitch=0):
    return dict(camera_in_body_m=origin,camera_pitch_deg=pitch,
                rgb_intrinsics=dict(cx=320,cy=180,fx=320,fy=320))


class GeometryTest(unittest.TestCase):
    def test_off_center_thin_support(self):
        r=row();z=dict(theta_bounds_deg=[10,30],phi_bounds_deg=[-2,2])
        t=dict(distance_m=1,range_noise_sigma_m=0)
        self.assertIsNone(ray_witness(20,0,(1,1),r,0))
        self.assertIsNotNone(ray_witness(10,0,(1,1),r,0))
        self.assertTrue(all(refine(r,z,t,0)['possible_by_depth']))

    def test_tangent_parallel_slab(self):
        self.assertIsNotNone(ray_witness(0,0,(.2,.2),row((0,.3,.4)),0))
        self.assertIsNone(ray_witness(0,0,(.2,.2),row((0,.30001,.4)),0))

    def test_real_points_survive_rotation_and_translation(self):
        tested=0
        for yaw in (-45,0,45):
            r=row((.02,-.03,1.6),-8)
            for theta in (-30,-10,10,30):
                for phi in (-20,0,20):
                    d=direction(theta,phi,-8,yaw)
                    point=[o+v for o,v in zip(r['camera_in_body_m'],d)]
                    if all(a<x<b for x,(a,b) in zip(point,CORRIDOR)):
                        z=dict(theta_bounds_deg=[theta-2,theta+3],phi_bounds_deg=[phi-3,phi+2])
                        self.assertTrue(all(refine(r,z,dict(distance_m=1,range_noise_sigma_m=.01),yaw)['possible_by_depth']))
                        tested+=1
        self.assertGreater(tested,3)

    def test_unresolved_range_dependency_is_not_discarded(self):
        theta=math.degrees(math.atan(.2/math.sqrt(.92)))
        z=dict(theta_bounds_deg=[theta,theta],phi_bounds_deg=[theta,theta])
        r=row((0,0,0))
        self.assertIsNone(ray_witness(theta,theta,(1,3),r,0))
        v=refine(r,z,dict(distance_m=2,range_noise_sigma_m=1/3),0)
        self.assertEqual(v['state'],'UNRESOLVED_POSSIBLE')
        self.assertTrue(all(v['possible_by_depth']))

    def test_far_disjoint(self):
        v=refine(row(),dict(theta_bounds_deg=[-1,1],phi_bounds_deg=[-1,1]),dict(distance_m=6,range_noise_sigma_m=.01),0)
        self.assertFalse(any(v['possible_by_depth']))


if __name__=='__main__': unittest.main()
