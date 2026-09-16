import copy
import unittest
import numpy as np
from mz171_return_labels import make_witness_labels


def fixture():
    target=dict(status='SIM_VALID',distance_m=2.,range_noise_sigma_m=.04,signal_strength_proxy=.1)
    row=dict(id='synthetic',tof_packet_received=True,tof_zones=[dict(zone_id=i,targets=[]) for i in range(64)])
    row['tof_zones'][0]['targets']=[target]
    native=dict(zone_id=0,packet_received=True,returned_lineage=[dict(target_index=0,hit_indices=[0])],
        private_rays=[dict(subray=0,hit_point_m=[12.,20.,31.],actor_id=None)])
    return row,dict(id='synthetic',body_origin_m=[10.,20.,30.],zonal_tof_native=[native])


class WitnessLabelsTest(unittest.TestCase):
    def test_closed_corridor_origin_and_ownerless_point(self):
        row,e=fixture();snapshot=copy.deepcopy((row,e))
        value=make_witness_labels(row,e)
        self.assertEqual(value['target'].dtype,np.float32);self.assertEqual(value['known'].dtype,np.bool_)
        self.assertEqual(value['target'].shape,(132,));self.assertTrue(value['known'][0]);self.assertEqual(value['target'][0],1.)
        self.assertFalse(value['known'][128:].any());self.assertEqual(value['audit']['slots'][0]['ownerless_corridor_contributors'],1)
        self.assertEqual((row,e),snapshot)
        for point in ([.2,-.3,.4],[3.6,.3,2.05]):
            # Zero origin avoids decimal-add/subtract roundoff at exact bounds.
            e['body_origin_m']=[0.,0.,0.];e['zonal_tof_native'][0]['private_rays'][0]['hit_point_m']=point
            self.assertEqual(make_witness_labels(row,e)['target'][0],1.)
        e['zonal_tof_native'][0]['private_rays'][0]['hit_point_m']=[3.60001,0.,1.]
        value=make_witness_labels(row,e);self.assertTrue(value['known'][0]);self.assertEqual(value['target'][0],0.)

    def test_second_merged_outside_rgb_and_metadata_invariance(self):
        row,e=fixture();zone=row['tof_zones'][0];zone['theta_bounds_deg']=[70.,75.]
        zone['targets'].append({**zone['targets'][0],'status':'SIM_MERGED'})
        n=e['zonal_tof_native'][0];n['returned_lineage'].append(dict(target_index=1,hit_indices=[1,2]))
        n['private_rays']+=[dict(subray=1,hit_point_m=[12.,20.,31.]),dict(subray=2,hit_point_m=[12.,22.,31.])]
        a=make_witness_labels(row,e);self.assertTrue(a['known'][1]);self.assertEqual(a['target'][1],1.)
        row['family']='irrelevant';row['truth']=False;e['source_truth']=False;e['native_bounds']=[]
        b=make_witness_labels(row,e);np.testing.assert_array_equal(a['target'],b['target']);np.testing.assert_array_equal(a['known'],b['known'])

    def test_missing_packet_status_and_absence_unknown(self):
        row,e=fixture();row['tof_packet_received']=False
        self.assertFalse(make_witness_labels(row,e)['known'].any())
        row['tof_packet_received']=True;row['tof_zones'][0]['targets'][0]['status']='SIM_INVALID'
        self.assertFalse(make_witness_labels(row,e)['known'].any())
        row['tof_zones'][0]['targets']=[]
        self.assertFalse(make_witness_labels(row,e)['known'].any())

    def test_incomplete_positive_vs_incomplete_outside(self):
        row,e=fixture();n=e['zonal_tof_native'][0];n['returned_lineage'][0]['hit_indices']=[0,1]
        v=make_witness_labels(row,e);self.assertTrue(v['known'][0]);self.assertEqual(v['target'][0],1.)
        n['private_rays'][0]['hit_point_m']=[12.,22.,31.]
        self.assertFalse(make_witness_labels(row,e)['known'][0])
        n['private_rays'].append(dict(subray=1,hit_point_m=[float('nan'),0.,1.]))
        self.assertFalse(make_witness_labels(row,e)['known'][0])
        n['private_rays'][1]['hit_point_m']=[12.,22.,31.]
        v=make_witness_labels(row,e);self.assertTrue(v['known'][0]);self.assertEqual(v['target'][0],0.)

    def test_duplicate_lineage_zone_and_hit_unknown(self):
        row,e=fixture();n=e['zonal_tof_native'][0]
        n['returned_lineage'].append(copy.deepcopy(n['returned_lineage'][0]))
        self.assertFalse(make_witness_labels(row,e)['known'][0])
        n['returned_lineage'].pop();e['zonal_tof_native'].append(copy.deepcopy(n))
        self.assertFalse(make_witness_labels(row,e)['known'][0])
        e['zonal_tof_native'].pop();n['private_rays'].append(copy.deepcopy(n['private_rays'][0]))
        self.assertFalse(make_witness_labels(row,e)['known'][0])
        n['private_rays'].pop();n['private_rays'][0]['hit_point_m']=[12.,22.,31.];n['returned_lineage'][0]['hit_indices']=[0,0]
        self.assertFalse(make_witness_labels(row,e)['known'][0])


if __name__=='__main__':unittest.main()
