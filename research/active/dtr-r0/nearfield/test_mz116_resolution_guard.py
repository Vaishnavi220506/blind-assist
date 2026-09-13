import unittest
from mz116_radar_resolution_guard import protect,unresolved_extent
from mz116_merged_collision import witness


class GuardTests(unittest.TestCase):
    def test_erosion_boundary(self):
        self.assertTrue(unresolved_extent([0,0,20,4]))
        self.assertFalse(unresolved_extent([0,0,20,4.01]))

    def test_only_independent_current_radar_can_restore(self):
        row=dict(radar_packet_received=True,radar_valid=[True],radar_range_m=[2.],radar_angle=[0.])
        spatial=dict(spatial_evidence=[dict(slot=0,proposal=0,box=[10,10,30,11])])
        p=dict(candidate=False)
        out=protect(row,p,spatial,0.);self.assertTrue(out['candidate']);self.assertFalse(p['candidate'])
        self.assertEqual(out['guard_events'][0]['height_state'],'HEIGHT_UNKNOWN')
        self.assertFalse(protect(row,p,spatial,0.,distance=1.)['candidate'])
        row['radar_packet_received']=False
        self.assertFalse(protect(row,p,spatial,0.)['candidate'])

    def test_resolved_box_and_unassociated_return_unchanged(self):
        row=dict(radar_packet_received=True,radar_valid=[True],radar_range_m=[2.],radar_angle=[0.])
        for proposal,box in [(0,[0,0,20,20]),(None,[0,0,20,1])]:
            self.assertFalse(protect(row,dict(candidate=False),dict(spatial_evidence=[dict(slot=0,proposal=proposal,box=box)]),0.)['candidate'])

    def test_merged_public_collision(self):
        self.assertEqual(witness()['status'],'TOF_TUPLE_COLLISION_CONFIRMED')


if __name__=='__main__':unittest.main()
