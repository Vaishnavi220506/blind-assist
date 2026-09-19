import unittest
import numpy as np

from ba_nfo_data import sensor
from cross_zone_anchor_core import component_anchors,trace_sensor,zone_map
from audit_cross_zone_nfo import domains,opportunity


class AnchorTests(unittest.TestCase):
    def test_original_sensor_parity_across_noise_dropout_and_invalid(self):
        rng=np.random.default_rng(9)
        depth=rng.uniform(.02,12,(192,256)).astype(np.float32)
        depth[::5,::7]=np.nan
        for identity in ('anchor-synthetic-a','anchor-synthetic-b'):
            _,_,boxes,values=sensor(depth,identity)
            traced=trace_sensor(depth,identity,boxes,values)
            self.assertEqual(sum(t['observed'] for t in traced),int(np.isfinite(values).sum()))
            for t in traced:
                if t['observed']:
                    selected=depth.ravel()[t['pixel_indices']]
                    np.testing.assert_array_equal((selected/.1).astype(int),t['winner_bin'])

    def test_changed_saved_range_rejected(self):
        depth=np.full((192,256),1.5,np.float32)
        _,_,boxes,values=sensor(depth,'anchor-test')
        first=np.flatnonzero(np.isfinite(values))[0];values[first]+=.01
        with self.assertRaisesRegex(ValueError,'replay mismatch'):
            trace_sensor(depth,'anchor-test',boxes,values)

    def test_mixed_winning_bin_never_becomes_pure(self):
        labels=np.array([[1,1,0,2]],np.int32)
        traces=[dict(observed=True,distance_m=1.5,zone_id=2,pixel_indices=np.array([0,2])),
                dict(observed=True,distance_m=1.4,zone_id=3,pixel_indices=np.array([0,1])),
                dict(observed=False,distance_m=None,zone_id=4,pixel_indices=np.array([],int)),
                dict(observed=True,distance_m=2.,zone_id=5,pixel_indices=np.array([3]))]
        possible,pure=component_anchors(labels,traces,2.)
        self.assertEqual(possible,{1:{2,3}})
        self.assertEqual(pure,{1:{3}})

    def test_zone_map_does_not_fill_unobserved_image(self):
        result=zone_map(np.array([[1,1,3,3],[1,3,3,5]]),(4,6))
        self.assertEqual(result[0,0],-1)
        self.assertEqual(result[1,2],0)
        self.assertEqual(result[1,3],1)
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            zone_map(np.array([[1,1,3,3],[2,2,4,4]]),(4,6))

    def test_cross_zone_requires_missing_local_anchor_and_preserves_outside(self):
        labels=np.array([[1,1,1,2,0]],np.int32)
        zmap=np.array([[0,1,-1,2,2]],np.int32)
        masks=opportunity(labels,zmap,{1:{0},2:{2}},{1:{0}},{1:{0},2:{2}})
        self.assertEqual(masks['other_pure'].tolist(),[[False,True,True,False,False]])
        self.assertEqual(masks['local_possible'].tolist(),[[True,False,False,True,False]])

    def test_local_owned_far_return_is_not_cross_zone_only(self):
        labels=np.array([[1,1,1]],np.int32);zones=np.array([[0,1,2]])
        masks=opportunity(labels,zones,{1:{0}},{1:{0}},{1:{0,1}})
        self.assertEqual(masks['other_pure'].tolist(),[[False,False,True]])
        self.assertEqual(masks['local_other_range'].tolist(),[[False,True,False]])

    def test_original_small_near_domain_keeps_unknown_out(self):
        depth=np.array([[1.,4.,4.,4.,4.,np.nan]],np.float32)
        truth,dm=domains(depth,np.array([[0,0,1,6]]),np.array([4.]))
        self.assertEqual(int(dm['far_small'].sum()),5)
        self.assertTrue(truth[0,0]);self.assertFalse(dm['full'][0,5])
        self.assertEqual(int(dm['pure_far'].sum()),0)


if __name__=='__main__':
    unittest.main()
