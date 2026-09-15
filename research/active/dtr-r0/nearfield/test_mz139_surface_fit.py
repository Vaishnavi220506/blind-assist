import math
import unittest
import numpy as np
import torch
from mz139_surface_fit import public_rays,render_population,decode,point_in_surface,surface_intersects,NoNoise
from mz115_zonal_tof import measure_zone


class SurfaceFitTests(unittest.TestCase):
    def row(self):
        return dict(camera_pitch_deg=0.,camera_in_body_m=[0.,0.,0.],
            rgb_intrinsics=dict(width=640,height=360,fx=457.,fy=457.,cx=320.,cy=180.),
            tof_zones=[dict(zone_id=0,theta_bounds_deg=[-4.,4.],phi_bounds_deg=[-4.,4.])])

    def render(self,row,param):
        return render_population(torch.tensor([param],dtype=torch.float64),
            torch.tensor(public_rays(row,[0],0.),dtype=torch.float64),
            torch.tensor(row['camera_in_body_m'],dtype=torch.float64),torch.eye(3,dtype=torch.float64),row['rgb_intrinsics'])

    def test_public_quadrature_and_regional_mean_are_not_center_hit(self):
        row=self.row();rays=public_rays(row,[0],0.)
        np.testing.assert_allclose(np.linalg.norm(rays,axis=-1),1.)
        param=[2.1,0.,0.,math.log(.1),math.log(.5),math.log(.5),0.,math.log(.6)]
        pred=self.render(row,param);rr=pred['ranges'][0,0].numpy()
        self.assertTrue(pred['valid'].all())
        expected=np.sum(1/rr)/np.sum(1/rr**2)
        self.assertAlmostEqual(float(pred['mean'][0,0]),expected)
        self.assertGreater(expected,2.)
        targets,diagnostic=measure_zone([dict(range_m=float(r),reflectance_proxy=.6) for r in rr],NoNoise())
        self.assertEqual(len(targets),1)
        self.assertAlmostEqual(diagnostic['components'][0]['pre_noise_range_m'],expected)

    def test_finite_slab_misses_outside_its_boundary(self):
        row=self.row();param=[2.1,1.,0.,math.log(.1),math.log(.05),math.log(.5),0.,math.log(.6)]
        pred=self.render(row,param)
        self.assertFalse(pred['valid'].any());self.assertEqual(float(pred['strength'][0,0]),0.)

    def test_complete_surface_can_intersect_without_sample_inside(self):
        s=dict(center=[2.,.35,1.],half_extent=[.1,.1,.3],yaw_rad=0.)
        self.assertTrue(surface_intersects(s))
        self.assertTrue(point_in_surface([1.9,.26,1.],s,tolerance=1e-8))
        self.assertFalse(point_in_surface([1.9,.1,1.],s,tolerance=1e-8))

    def test_rotated_surface_sat_and_camera_origin(self):
        s=dict(center=[2.,.7,1.],half_extent=[.1,.1,.3],yaw_rad=math.pi/4)
        self.assertFalse(surface_intersects(s))
        row=self.row();row['camera_in_body_m']=[.5,0.,0.]
        p=[2.1,0.,0.,math.log(.1),math.log(.5),math.log(.5),0.,math.log(.6)]
        pred=self.render(row,p)
        self.assertAlmostEqual(float(pred['ranges'][0,0,4]),1.5)


if __name__=='__main__':unittest.main()
