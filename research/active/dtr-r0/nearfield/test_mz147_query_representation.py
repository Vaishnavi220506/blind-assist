import copy
import unittest
import numpy as np
from test_mz143_corridor_features import inputs,target
from mz147_query_representation import extract,query_masks


class QueryTests(unittest.TestCase):
    def test_body_query_regions_are_geometric_not_image_center(self):
        row,_=inputs();row['camera_in_body_m']=[0.,.2,1.7]
        box,regions=query_masks(row,0.,2.,[0,0,640,360])
        # At body z=1.2, distinct y=-.5,0,+.5 project to these pixels.
        for y,j in [(-.5,0),(0.,1),(.5,2)]:
            u=round(319.5+457*(y-.2)/2);v=round(179.5+457*.5/2)
            self.assertTrue(regions[j][v,u])
            self.assertEqual(sum(int(region[v,u]) for region in regions),1)

    def test_metadata_invariant_and_merged_is_not_exact_pixel_depth(self):
        row,image=inputs();row['tof_zones'][28]['targets']=[target(2.),target(3.,'SIM_MERGED')]
        before=copy.deepcopy(row);result=extract(row,image,0.)
        poison=copy.deepcopy(row);poison.update(truth=True,native_bounds=[{'secret':5}],id='positive')
        poison['tof_zones'][28]['targets'][0]['hit_point_m']=[99]*3
        again=extract(poison,image,0.)
        np.testing.assert_array_equal(result['values'],again['values'])
        self.assertEqual(row,before)
        self.assertEqual(result['audit']['target_plane_hypotheses'],1)
        self.assertEqual(len(result['audit']['slots']),2)
        self.assertEqual(result['values'][result['names'].index('zone28.slot1.valid_plane_hypothesis')],0)
        self.assertTrue(result['audit']['raw_sensor_vector_required'])


if __name__=='__main__':unittest.main()
