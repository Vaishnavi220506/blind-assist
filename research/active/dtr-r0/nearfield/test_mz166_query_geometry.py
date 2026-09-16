"""Synthetic public-coordinate and evaluator-reference tests; no data I/O."""
from copy import deepcopy
import json
import unittest

import numpy as np

from mz115_zonal_tof import zone_geometry
from mz161_dense_task import ray_query
from mz161_dense_labels import make_labels
from mz166_query_geometry import OFFSETS,public_query,make_query_references,label_for_query


def public_fixture():
    row=dict(rgb_intrinsics=dict(width=7,height=5,fx=4.,fy=4.,cx=3.,cy=2.),
        camera_in_body_m=[0.,0.,1.],camera_pitch_deg=0.,tof_packet_received=True,imu_valid=True,
        radar_packet_received=True,radar_range_m=[2.,None,2.4,None],radar_angle=[10.,None,-20.,None],
        radar_velocity=[0.,None,None,None],radar_valid=[True,False,True,False],
        tof_zones=[dict(zone_geometry(z)[0],targets=[]) for z in range(64)])
    row['tof_zones'][0]['targets']=[dict(distance_m=2.,range_noise_sigma_m=.04,signal_strength_proxy=.1,status=s)
                                      for s in ('SIM_VALID','SIM_MERGED')]
    row['tof_zones'][63]['targets']=deepcopy(row['tof_zones'][0]['targets'])
    image=np.arange(22*5*7,dtype=np.float32).reshape(22,5,7)/800.
    direction,query=ray_query(row,0.);image[13:16]=query;image[16:19]=direction.transpose(2,0,1)
    image=image.astype(np.float16).astype(np.float32)
    tokens=np.zeros((132,21),np.float32);valid=np.zeros(132,bool)
    for i in (0,1,126,127,128,130):
        valid[i]=True;tokens[i,0]=1;tokens[i,6]=.4;tokens[i,10]=.2;tokens[i,11]=.6
    tokens[[1,127],1]=1;tokens[129,12]=-1.;tokens[129,13]=1.;tokens[129,17]=1.
    return row,image,tokens,valid


def native_fixture():
    row=public_fixture()[0];row['id']='synthetic'
    evaluation=dict(id='synthetic',body_origin_m=[0.,0.,0.],
        camera=dict(x=0.,y=0.,z=1.,yaw=0.,pitch=0.,roll=0.),
        native_bounds=[dict(name='shape0',center_m=[2.,.475,1.],extent_m=[.1,.03,.08]),
                       dict(name='context',center_m=[4.,0.,1.],extent_m=[.1,10.,10.])])
    return row,evaluation


class QueryTests(unittest.TestCase):
    def test_zero_exact_independent_copies_and_metadata_invariance(self):
        row,image,tokens,valid=public_fixture();before=deepcopy(row)
        result=public_query(dict(row,id='private',truth=True,native_bounds=[1]),0.,image,tokens,valid,0.)
        for key,source in [('image',image),('tokens',tokens),('valid',valid)]:
            np.testing.assert_array_equal(result[key],source);self.assertFalse(np.shares_memory(result[key],source))
        self.assertEqual(row,before)

    def test_query_sign_directions_slot_preservation_and_width(self):
        row,image,tokens,valid=public_fixture();before=deepcopy((row,image,tokens,valid))
        shifted=public_query(row,0.,image,tokens,valid,.45)
        keep=[i for i in range(22) if i not in (13,14,15)]
        np.testing.assert_array_equal(shifted['image'][keep],image[keep])
        np.testing.assert_array_equal(shifted['valid'],valid)
        np.testing.assert_array_equal(shifted['tokens'][~valid],tokens[~valid])
        cols=[i for i in range(21) if i not in (6,10,11)]
        np.testing.assert_array_equal(shifted['tokens'][:,cols],tokens[:,cols])
        np.testing.assert_array_equal(shifted['tokens'][np.ix_(valid,[6,10,11])],tokens[np.ix_(valid,[6,10,11])]-np.float32(.225))
        self.assertEqual(shifted['image'][15,2,3],0.)
        expected=np.array([.15/.25*np.sqrt(1.0625)/4,.75/.25*np.sqrt(1.0625)/4,1],np.float32).astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(shifted['image'][13:16,2,4],expected)
        for offset in OFFSETS:
            result=public_query(row,0.,image,tokens,valid,offset)
            self.assertAlmostEqual(np.diff(result['audit']['corridor_lateral_bounds_m'])[0],.6)
        self.assertEqual(row,before[0]);np.testing.assert_array_equal(tokens,before[2])

    def test_fixed_raster_central_equivalence_and_shifted_truth(self):
        row,evaluation=native_fixture();old=deepcopy((row,evaluation))
        reference=make_query_references(row,evaluation,17.)
        central=make_labels(row,evaluation,17.)
        zero=label_for_query(reference,0.);positive=label_for_query(reference,.45);negative=label_for_query(reference,-.45)
        for key in ('target','known','weights','target_visible','target_corridor'):
            np.testing.assert_array_equal(zero[key],central[key])
        self.assertEqual(zero['frame'],central['audit']['native_aabb_corridor_truth'])
        self.assertFalse(zero['frame']);self.assertTrue(positive['frame']);self.assertFalse(negative['frame'])
        self.assertTrue(positive['target'][2,4]);self.assertFalse(zero['target'][2,4])
        for offset in OFFSETS:
            result=label_for_query(reference,offset)
            np.testing.assert_array_equal(result['known'],reference['known'])
            np.testing.assert_array_equal(result['target_visible'],reference['target_visible'])
            self.assertEqual(result['audit']['reference_yaw_deg'],0.)
            self.assertEqual(result['audit']['pose_authority'],'SOURCE_COMMANDED_REFERENCE')
            json.dumps(result['audit'],allow_nan=False)
        self.assertEqual((row,evaluation),old)

    def test_unknown_classmass_and_all_actor_occlusion(self):
        row,evaluation=native_fixture();evaluation['native_bounds']=evaluation['native_bounds'][:1]
        ref=make_query_references(row,evaluation,0.)
        labels=label_for_query(ref,.45)
        self.assertEqual(float(labels['weights'].sum()),1.)
        self.assertTrue((labels['weights'][~labels['known']]==0).all())
        evaluation['native_bounds'].append(dict(name='occluder',center_m=[.125,0.,1.],extent_m=[.025,2.,2.]))
        labels=label_for_query(make_query_references(row,evaluation,0.),.45)
        self.assertFalse(labels['target'].any());self.assertFalse(labels['target_visible'].any())
        self.assertTrue(labels['frame']);self.assertTrue(labels['audit']['native_volume_truth_vs_visible_mismatch'])
        evaluation['native_bounds']=[]
        labels=label_for_query(make_query_references(row,evaluation,0.),.45)
        self.assertFalse(labels['known'].any());self.assertEqual(float(labels['weights'].sum()),0.)
        self.assertFalse(labels['frame'])


if __name__=='__main__':unittest.main()
