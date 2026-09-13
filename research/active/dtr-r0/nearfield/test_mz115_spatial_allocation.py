import itertools
import unittest
import numpy as np
import mz115_spatial_allocation as m
from mz107_rgb_association import pixel_ray


class AllocationTests(unittest.TestCase):
    def test_slant_enclosure_contains_dense_rays(self):
        intr=dict(cx=320.,cy=180.,fx=457.,fy=457.)
        box=[270.,95.,370.,270.];rs=(1.3,1.54);ps=(-3.5,-2.5);ys=(-8.,9.)
        env=m.slant_envelope(box,rs,intr,ps,ys,1.65)
        for u,v,r,p,y in itertools.product(np.linspace(box[0],box[2],5),np.linspace(box[1],box[3],5),rs,np.linspace(*ps,3),np.linspace(*ys,7)):
            point=pixel_ray(u,v,intr,p,y)*r+[0,0,1.65]
            self.assertTrue(all(lo-1e-10<=x<=hi+1e-10 for x,(lo,hi) in zip(point,env)))

    def test_single_zone_and_competition_fall_back(self):
        def evidence(z,b):return dict(zone_id=z,target_slot=0,status='SIM_VALID',forward_depth=2.,signal=.1,range_m=2.,zone_box=b,proposal=None)
        one=[evidence(0,[0,0,10,10])];m.assign_groups(one,[[0,0,5,10]])
        self.assertIsNone(one[0]['proposal'])
        both=[evidence(0,[0,0,10,10]),evidence(1,[0,10,10,20])]
        m.assign_groups(both,[[0,0,5,20],[5,0,10,20]])
        self.assertTrue(all(e['proposal'] is None for e in both))

    def test_roi_stays_inside_measured_zone(self):
        es=[dict(zone_id=k,target_slot=0,status='SIM_VALID',forward_depth=2.,signal=.1,range_m=2.,zone_box=[0,k*10,10,(k+1)*10],proposal=None) for k in range(2)]
        m.assign_groups(es,[[2,-20,5,50]])
        self.assertTrue(all(e['proposal']==0 for e in es))
        self.assertEqual(es[0]['roi'],[0,0,7,10])
        self.assertEqual(es[1]['roi'],[0,10,7,20])

    def test_missing_rgb_and_merged_range_stay_unresolved(self):
        row=dict(tof_packet_received=True,tof_zones=[dict(zone_id=0,theta_bounds_deg=[-2,2],phi_bounds_deg=[-2,2],targets=[dict(distance_m=2.,range_noise_sigma_m=.04,signal_strength_proxy=.1,status='SIM_MERGED')])],
            rgb_intrinsics=dict(cx=320.,cy=180.,fx=457.,fy=457.),time_s=0.,camera_pitch_deg=-3.,camera_in_body_m=[0,0,1.65])
        e=m.allocate(row,dict(proposals=[],integrated_yaw_deg=0.))[0]
        self.assertIsNone(e['proposal']);self.assertEqual(e['range_bounds'],(.02,4.))
        self.assertEqual(e['coarse_xyz'],e['localized_xyz'])
        row['tof_packet_received']=False
        self.assertEqual(m.allocate(row,dict(proposals=[],integrated_yaw_deg=0.)),[])


if __name__=='__main__':unittest.main()
