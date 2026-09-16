"""Synthetic return-face geometry tests; no real data or inference."""
import copy
import json
import unittest
import numpy as np
from mz162_return_labels import prepare_frame,slot_labels


def scene():
    row=dict(id='synthetic',episode_id='ep',tof_packet_received=True,camera_in_body_m=[0.,0.,1.],camera_pitch_deg=0.,
        rgb_intrinsics=dict(width=7,height=5,fx=4.,fy=4.,cx=3.,cy=2.),
        tof_zones=[dict(zone_id=z,targets=[]) for z in range(64)])
    row['tof_zones'][0]['targets']=[dict(distance_m=2.,range_noise_sigma_m=.04,signal_strength_proxy=.1,status='SIM_VALID')]
    e=dict(id='synthetic',body_origin_m=[0.,0.,0.],camera=dict(x=0.,y=0.,z=1.,yaw=0.,pitch=0.,roll=0.),
        native_bounds=[dict(name='anything',center_m=[2.1,0.,1.],extent_m=[.1,2.,2.]),
                       dict(name='other',center_m=[4.1,0.,1.],extent_m=[.1,10.,10.])],
        zonal_tof_native=[dict(zone_id=0,returned_lineage=[dict(target_index=0,hit_indices=[0])],
            private_rays=[dict(subray=0,actor_id='ep/anything',hit_point_m=[2.,0.,1.])])])
    return row,e


class ReturnLabelTests(unittest.TestCase):
    def test_full_face_unsampled_extent_and_relative_slant(self):
        row,e=scene();before=copy.deepcopy((row,e));f=prepare_frame(row,e,17.);label=slot_labels(f,0)
        self.assertEqual(len(f['slots']),128);self.assertTrue(label['member'].all())
        self.assertEqual(f['slots'][0]['annotation_status'],'SINGLE_FACE')
        self.assertAlmostEqual(float(label['relative_slant_residual'][2,3]),0.)
        self.assertAlmostEqual(float(label['relative_slant_residual'][2,4]),np.sqrt(1.0625)-1.,places=6)
        self.assertGreater(int(label['member'].sum()),f['slots'][0]['native_contributors'])
        self.assertFalse(f['reference']['target_visible'].any())
        self.assertEqual((row,e),before);json.dumps(f['audit'],allow_nan=False)

    def test_mixed_union_is_not_single_identity_and_occludes_back_face(self):
        row,e=scene();e['native_bounds'][0]['extent_m']=[.1,.4,.4]
        row['tof_zones'][0]['targets'][0]['status']='SIM_MERGED'
        native=e['zonal_tof_native'][0]
        native['private_rays'].append(dict(subray=1,actor_id='ep/other',hit_point_m=[4.,1.,1.]))
        native['returned_lineage'][0]['hit_indices'].append(1)
        f=prepare_frame(row,e,0.);label=slot_labels(f,0)
        self.assertEqual(f['slots'][0]['annotation_status'],'MIXED_OWNER_SET')
        self.assertTrue(f['slots'][0]['identity_unresolved']);self.assertTrue(label['member'].all())
        self.assertEqual(int(f['reference']['owner'][2,3]),0)
        self.assertAlmostEqual(float(label['relative_slant_residual'][2,3]),0.)
        self.assertGreater(float(label['relative_slant_residual'][2,4]),1.)
        # Public VALID is not a uniqueness proof; same native mixture stays mixed.
        row['tof_zones'][0]['targets'][0]['status']='SIM_VALID'
        self.assertEqual(prepare_frame(row,e,0.)['slots'][0]['annotation_status'],'MIXED_OWNER_SET')

    def test_unknown_owner_and_ambiguous_face_keep_slot_denominator(self):
        row,e=scene();e['zonal_tof_native'][0]['private_rays'][0]['actor_id']=None
        f=prepare_frame(row,e,0.);label=slot_labels(f,0)
        self.assertTrue(f['slots'][0]['public_usable']);self.assertFalse(label['known'].any())
        self.assertIn('UNKNOWN_NATIVE_OWNER',f['slots'][0]['unknown_reasons'])
        e['zonal_tof_native'][0]['private_rays'][0].update(actor_id='ep/anything',hit_point_m=[2.,2.,1.])
        f=prepare_frame(row,e,0.)
        self.assertIn('UNKNOWN_AMBIGUOUS_FACE',f['slots'][0]['unknown_reasons'])
        self.assertFalse(slot_labels(f,0)['residual_known'].any())
        self.assertEqual(f['slots'][1]['annotation_status'],'ABSENT')


if __name__=='__main__':unittest.main()
