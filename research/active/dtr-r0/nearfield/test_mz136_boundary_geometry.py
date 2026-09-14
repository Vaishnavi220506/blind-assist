"""Public geometry invariants; missing returns never turn into free-space proof."""
import copy
import math
import unittest
import numpy as np
from mz136_boundary_geometry import camera_to_body, near_cohorts, proposals, ray


class BoundaryGeometryTests(unittest.TestCase):
    def row(self):
        return dict(camera_pitch_deg=-3.,camera_in_body_m=[0.,0.,1.7],
            rgb_intrinsics=dict(width=640,height=360,fx=457.,fy=457.,cx=320.,cy=180.),
            tof_packet_received=True,tof_zones=[dict(zone_id=i,theta_bounds_deg=[0.,5.625],
                phi_bounds_deg=[-5.625,0.],targets=[dict(status='SIM_VALID',distance_m=2.+i*.02,
                range_noise_sigma_m=.04)]) for i in (0,1)])

    def test_camera_rotation_and_pixel_roundtrip(self):
        row=self.row();rotation=camera_to_body(row,7.)
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
        direction=rotation.T@ray(row,350.,220.,7.)
        self.assertAlmostEqual(320.+457.*direction[1]/direction[0],350.)
        self.assertAlmostEqual(180.-457.*direction[2]/direction[0],220.)

    def test_spherical_range_is_not_forward_depth(self):
        row=self.row();groups=near_cohorts(row)
        a,b=map(math.radians,(2.8125,-2.8125))
        self.assertAlmostEqual(groups[0][0]['depth'],2./math.sqrt(1+math.tan(a)**2+math.tan(b)**2))

    def test_missing_and_merged_input_is_preserved(self):
        row=self.row();row['tof_packet_received']=False
        row['tof_zones'][0]['targets'][0]['status']='SIM_MERGED'
        before=copy.deepcopy(row)
        found,_=proposals(row,np.zeros((360,640,3),np.uint8),0.)
        self.assertEqual(found,[])
        self.assertEqual(row,before)
        row['tof_packet_received']=True
        self.assertEqual(len(near_cohorts(row)[0]),1)


if __name__=='__main__':unittest.main()
