import copy
import unittest
import numpy as np
from test_mz143_corridor_features import inputs,target,feature
from mz143_corridor_features import extract as original
from mz150_virtual_corridors import extract,training_target


class VirtualQueryTests(unittest.TestCase):
    def test_zero_query_identity_and_translated_sensor_coordinates(self):
        row,image=inputs();row['tof_zones'][28]['targets']=[target()]
        row['radar_range_m'][0]=2.;row['radar_angle'][0]=0.;row['radar_valid'][0]=True
        before=copy.deepcopy(row);a=original(row,image,0.);b=extract(row,image,0.,0.)
        np.testing.assert_array_equal(a['sensor'],b['sensor']);np.testing.assert_array_equal(a['geometry'],b['geometry'])
        c=extract(row,image,0.,.3)
        self.assertAlmostEqual(feature(c,'zone28.slot0.center_y'),feature(a,'zone28.slot0.center_y')-.3,places=6)
        self.assertAlmostEqual(feature(c,'radar0.body_y'),-.3,places=6)
        self.assertEqual(feature(c,'radar1.body_y'),0.)
        self.assertEqual(row,before)

    def test_training_query_changes_truth_without_moving_scene(self):
        e=dict(body_origin_m=[0.,0.,0.],native_bounds=[dict(center_m=[2.,.5,1.],extent_m=[.1,.02,.5])])
        before=copy.deepcopy(e)
        self.assertFalse(training_target(e,0.));self.assertTrue(training_target(e,.3))
        self.assertFalse(training_target(e,-.3));self.assertEqual(e,before)


if __name__=='__main__':unittest.main()
