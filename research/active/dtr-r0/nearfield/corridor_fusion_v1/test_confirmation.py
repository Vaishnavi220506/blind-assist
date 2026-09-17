import unittest
import numpy as np
from run_confirmation import compact,qbin,QROOT,physical_values
from specialist_gate import features,policy


class ConfirmationContract(unittest.TestCase):
    def test_bin_edges_and_missing_are_explicit(self):
        self.assertEqual(qbin(None),'missing');self.assertEqual(qbin(np.nan),'missing')
        for v,label in [(3.3,'<=3.3'),(3.6,'(3.3,3.6]'),(3.8,'(3.6,3.8]'),
                (QROOT,'(3.8,tree_threshold]'),(np.nextafter(QROOT,np.inf),'(tree_threshold,4.2]'),
                (4.5,'(4.2,4.5]'),(4.6,'>4.5')]:self.assertEqual(qbin(v),label)

    def test_called_metrics_expose_harm_without_netting(self):
        r=compact([1,1,0,0],[1,1,1,0],[1,0,0,0],[1,1,1,0])
        self.assertEqual((r['called_TP'],r['called_TP_retained'],r['FP_removed'],r['harmful_veto']),(2,1,1,1))
        with self.assertRaises(AssertionError):compact([1],[0],[1],[0])

    def test_source_strata_do_not_change_invocation_features(self):
        row=dict(tof_zones=[],tof_packet_received=False,radar_packet_received=True,imu_valid=True)
        a=features(row,[2],.8,[0]);row.update(wall_distance_m=4.5,background_style='x',target_reflectance=.2,
            texture_grid=[1,2],family='rod',truth=True,C_score=.1)
        np.testing.assert_array_equal(a,features(row,[2],.8,[0]))

    def test_missing_specialist_preserves_a_and_c_cannot_add(self):
        final,called,veto=policy([0,1,1],[1,1,.1],.9,[1,np.nan,0],.7)
        np.testing.assert_array_equal(final,[0,1,1]);np.testing.assert_array_equal(called,[0,1,0])

    def test_physical_strata_read_group_metadata_after_prediction(self):
        spec=dict(frames=[dict(id='x',scene_group='g',wall_distance_m=3.3)],
                  scene_groups=[dict(scene_group='g',target_reflectance=.16,texture_grid=[3,7])])
        self.assertEqual(physical_values(spec,[dict(id='x')],[{}],'target_reflectance'),[.16])
        self.assertEqual(physical_values(spec,[dict(id='x')],[{}],'wall_distance_m'),[3.3])


if __name__=='__main__':unittest.main()
