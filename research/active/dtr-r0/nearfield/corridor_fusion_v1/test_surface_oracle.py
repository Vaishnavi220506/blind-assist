import unittest
import numpy as np
from surface_oracle import SUPPORT_NAMES,intervene,intersects

def fixture():
    names=['zone00.slot0.usable']+['zone00.slot0.'+k for k in SUPPORT_NAMES]+['radar0.body_y','nearest.side_overlap_m.q50']
    base=np.array([1,1,3,-1,1,.5,1.5,.42,-.12],np.float32)
    hit=dict(subray=0,actor_id='ep/object',hit_point_m=[1.9,.42,1.],theta_deg=10.,phi_deg=0.)
    ev=dict(body_origin_m=[0,0,0],native_bounds=[dict(name='object',center_m=[2,.35,1],extent_m=[.1,.1,.3])],
        zonal_tof_native=[dict(zone_id=0,private_rays=[hit],returned_lineage=[dict(target_index=0,hit_indices=[0])])])
    row=dict(id='f',tof_zones=[dict(zone_id=0,targets=[dict(status='SIM_VALID')])])
    return row,ev,base,names

class OracleContract(unittest.TestCase):
    def test_complete_unsampled_face_preserves_fixed_features(self):
        r,e,b,n=fixture();s,f,a=intervene(r,e,b,n)
        self.assertFalse(a['sampled_point_reachable']);self.assertTrue(a['full_face_reachable'])
        np.testing.assert_array_equal(s[-2:],b[-2:]);np.testing.assert_array_equal(f[-2:],b[-2:])
        self.assertEqual(a['contributors_excluded'],0);self.assertEqual(f[1],f[2])
    def test_unreturned_obstacle_cannot_be_added(self):
        r,e,b,n=fixture();e['zonal_tof_native'][0]['returned_lineage']=[]
        s,f,a=intervene(r,e,b,n);np.testing.assert_array_equal(f,b)
        self.assertFalse(a['full_face_reachable']);self.assertEqual(a['fallback_slots'],1)
    def test_unusable_slot_untouched(self):
        r,e,b,n=fixture();b[0]=0;s,f,a=intervene(r,e,b,n)
        np.testing.assert_array_equal(s,b);np.testing.assert_array_equal(f,b);self.assertEqual(a['usable_slots'],0)
    def test_ambiguous_face_falls_back(self):
        r,e,b,n=fixture();e['zonal_tof_native'][0]['private_rays'][0]['hit_point_m']=[1.9,.25,1.]
        s,f,a=intervene(r,e,b,n);np.testing.assert_array_equal(f,b);self.assertEqual(a['fallback_slots'],1)
    def test_disjoint_faces_do_not_become_exact_union(self):
        self.assertFalse(intersects([[2,2],[.4,.6],[.5,1]]))
        self.assertFalse(intersects([[2,2],[-.6,-.4],[.5,1]]))
        self.assertTrue(intersects([[2,2],[-.6,.6],[.5,1]]))

if __name__=='__main__':unittest.main()
