"""Synthetic checks for metric localization, alerts and returned-evidence strata."""
import unittest
import numpy as np
from run_mz164_metric_prior import alert, correct_target_patch, target_stratum


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(rgb_intrinsics=dict(width=3,height=3,fx=100.,fy=100.,cx=1.,cy=1.),
                        camera_pitch_deg=0.,camera_in_body_m=[0.,0.,1.4],imu_valid=True)

    def test_nine_connected_metric_points_and_missing_pose(self):
        depth = np.full((3,3),2.,np.float32)
        self.assertTrue(alert(self.row,depth,0.)['candidate'])
        self.assertFalse(alert(self.row,depth,90.)['candidate'])
        self.assertFalse(alert(dict(self.row,imu_valid=False),depth,0.)['candidate'])
        depth[0,0]=np.nan
        self.assertFalse(alert(self.row,depth,0.)['candidate'])

    def test_same_corridor_wrong_depth_not_localization(self):
        mask=np.ones((3,3),bool)
        wrong=correct_target_patch(mask,mask,np.full((3,3),.5),np.full((3,3),2.7))
        self.assertEqual(wrong['largest_correct_target_patch_pixels'],0)
        right=correct_target_patch(mask,mask,np.full((3,3),2.71),np.full((3,3),2.7))
        self.assertEqual(right['largest_correct_target_patch_pixels'],9)

    def test_unreturned_target_is_not_a_return_and_merged_stays_distinct(self):
        row=dict(tof_zones=[dict(zone_id=0,targets=[dict(status='SIM_MERGED')])])
        native=dict(zonal_tof_native=[dict(zone_id=0,private_rays=[dict(subray=0,actor_id='ep/shape0')],returned_lineage=[])])
        self.assertEqual(target_stratum(row,native),'no_target_return')
        native['zonal_tof_native'][0]['returned_lineage']=[dict(target_index=0,hit_indices=[0])]
        self.assertEqual(target_stratum(row,native),'merged_only')
        row['tof_zones'][0]['targets'][0]['status']='SIM_VALID'
        self.assertEqual(target_stratum(row,native),'valid_supported')


if __name__=='__main__': unittest.main()
