import copy
import unittest
from mz137_edge_corridor import arm_readout, plane_support, association


class EdgeCorridorTests(unittest.TestCase):
    def inputs(self):
        row=dict(rgb_intrinsics=dict(fx=100.,fy=100.,cx=100.,cy=100.),camera_in_body_m=[.1,.02,1.7],
            tof_zones=[dict(zone_id=i,theta_bounds_deg=[i,i+1]) for i in range(8)])
        item=dict(zone_id=3,target_slot=0,localized_xyz=[[1.,2.],[-.4,.4],[1.,1.8]],
            coarse_xyz=[[1.,2.],[-.4,.4],[1.,1.8]])
        cached=dict(spatial_evidence=[item],common_radar=True,guard_events=[])
        assoc=dict(body_forward_depth_m=2.,rectified_vertical_px=[100.,150.],replaced_returns=[[3,0]])
        return row,cached,dict(candidate=True),dict(candidate=False),assoc

    def test_old_false_alarm_can_be_removed_and_fine_can_flip_it(self):
        r,c,b,rad,a=self.inputs()
        coarse=arm_readout(r,c,b,rad,a,[115.,120.])
        fine=arm_readout(r,c,b,rad,a,[112.,120.])
        self.assertFalse(coarse['candidate'])
        self.assertTrue(fine['candidate'])
        self.assertEqual(plane_support(r,a,[115.,120.])[0],[2.1,2.1])

    def test_unmatched_same_zone_and_radar_are_independent(self):
        r,c,b,rad,a=self.inputs(); other=copy.deepcopy(c['spatial_evidence'][0]);other['target_slot']=1
        c['spatial_evidence'].append(other)
        result=arm_readout(r,c,b,rad,a,[115.,120.])
        self.assertTrue(result['candidate']);self.assertTrue(result['residual_alert'])
        c['spatial_evidence'].pop();rad['candidate']=True
        self.assertTrue(arm_readout(r,c,b,rad,a,[115.,120.])['candidate'])

    def test_fallback_preserves_incumbent_and_inputs(self):
        r,c,b,rad,a=self.inputs();before=copy.deepcopy((r,c,b,rad,a))
        self.assertTrue(arm_readout(r,c,b,rad,None,None)['candidate'])
        arm_readout(r,c,b,rad,a,[115.,120.])
        self.assertEqual((r,c,b,rad,a),before)

    def test_replaced_old_certain_vote_is_not_ored_back(self):
        r,c,b,rad,a=self.inputs()
        c['spatial_evidence'][0]['coarse_xyz']=[[1.,2.],[-.1,.1],[1.,1.8]]
        self.assertFalse(arm_readout(r,c,b,rad,a,[115.,120.])['candidate'])

    def test_public_association_preserves_out_of_view_and_rejects_ambiguity(self):
        row=dict(imu_valid=True,camera_pitch_deg=0.,camera_in_body_m=[0.,0.,1.7],
            rgb_intrinsics=dict(fx=457.,fy=457.,cx=320.,cy=180.,width=640,height=360),
            tof_packet_received=True,tof_zones=[dict(zone_id=i,theta_bounds_deg=[0.,5.625],
                phi_bounds_deg=[-5.625,0.] if i==0 else [-24.,-23.],targets=[dict(
                    status='SIM_VALID',distance_m=2.,range_noise_sigma_m=.04)]) for i in (0,1)])
        edge=dict(seed=dict(zones=[0,1],box=[320,170,350,250]),
                  homography=[[1,0,0],[0,1,0],[0,0,1]],candidate=None)
        got=association(row,edge,0.)
        self.assertEqual(got['replaced_returns'],[[0,0]])
        edge['candidate']={'rectified_edges_px':[999,1000]}
        self.assertEqual(got,association(row,edge,0.))
        row['tof_zones'][0]['targets'].append(copy.deepcopy(row['tof_zones'][0]['targets'][0]))
        self.assertIsNone(association(row,edge,0.))


if __name__=='__main__':unittest.main()
