import unittest
from mz138_support_ceiling import aggregate, linked_hits, support, surface_key


class SupportCeilingTests(unittest.TestCase):
    def fixture(self):
        row=dict(time_s=0.,camera_pitch_deg=0.,camera_in_body_m=[0.,0.,0.],
            rgb_intrinsics=dict(fx=100.,fy=100.,cx=100.,cy=100.),
            tof_zones=[dict(zone_id=i,theta_bounds_deg=[i,i+1]) for i in range(8)])
        item=dict(range_bounds=[.02,4.],localized_xyz=[[1.,2.],[-1.,1.],[1.,1.]],coarse_xyz=[[1.,2.],[-1.,1.],[1.,1.]])
        hits=[dict(actor_id='e/'+name,hit_point_m=[2.,y,1.],theta_deg=y*20.,phi_deg=20.) for name,y in [('a',-.5),('b',.5)]]
        evaluation=dict(body_origin_m=[0.,0.,0.],native_bounds=[])
        return row,item,hits,evaluation

    def test_grouping_removes_false_bridge_without_changing_points(self):
        row,item,hits,e=self.fixture();arms,points,keys=support(row,item,hits,e,0.)
        self.assertTrue(arms['pooled_native']['possible'])
        self.assertFalse(arms['ownership_extent']['possible'])
        self.assertFalse(arms['full_native']['possible'])
        self.assertTrue(all(not arms[a]['certain'] for a in arms))

    def test_one_inside_point_does_not_create_split_certainty_vote(self):
        row,item,hits,e=self.fixture();hits[0]['hit_point_m']=[2.,0.,1.]
        arms,_,_=support(row,item,hits,e,0.)
        self.assertTrue(arms['full_native']['possible']);self.assertFalse(arms['full_native']['certain'])
        self.assertEqual(arms['ownership_extent']['certain'],arms['pooled_native']['certain'])

    def test_zone_dedup_and_frozen_radar_guard(self):
        row,_,_,_=self.fixture();bits=[dict(zone=3,possible=True,certain=False)]*3
        quiet=dict(candidate=False,common_radar=False,guard_events=[])
        self.assertEqual(aggregate(row,bits,quiet)['tof_score'],1.)
        radar=dict(candidate=True,common_radar=False,guard_events=[{'kind':'guard'}])
        self.assertTrue(aggregate(row,[],radar)['candidate'])
        self.assertFalse(aggregate(row,[],radar)['tof_candidate'])

    def test_missing_lineage_falls_back_and_unreturned_ray_is_not_added(self):
        row,item,hits,e=self.fixture();arms,points,_=support(row,item,[],e,0.)
        self.assertTrue(all(a['fallback'] and a['possible'] for a in arms.values()));self.assertEqual(points,[])
        item.update(zone_id=3,target_slot=0)
        rays=[dict(h,subray=i) for i,h in enumerate(hits)]
        native={3:dict(returned_lineage=[dict(target_index=0,hit_indices=[1])],private_rays=rays)}
        self.assertEqual(linked_hits(item,native),[rays[1]])

    def test_face_ambiguity_is_explicit(self):
        bounds={'a':dict(center_m=[1.,1.,1.],extent_m=[1.,1.,1.])}
        self.assertEqual(surface_key(dict(actor_id='e/a',hit_point_m=[0.,1.,1.]),bounds),('e/a','0:MIN'))
        self.assertTrue(surface_key(dict(actor_id='e/a',hit_point_m=[0.,0.,1.]),bounds)[1].startswith('AMBIGUOUS'))

    def test_complete_face_restores_unsampled_extent_without_filling_volume(self):
        from run_mz138_surface_extent import native_face_pieces
        e=dict(body_origin_m=[0.,0.,0.],native_bounds=[dict(name='a',center_m=[2.,.35,1.],extent_m=[.1,.1,.3])])
        pieces=native_face_pieces(dict(surface_keys=[['e/a','0:MIN']]),e)
        self.assertEqual(len(pieces),1)
        self.assertEqual(pieces[0][0],[1.9,1.9])
        self.assertAlmostEqual(pieces[0][1][0],.25)
        self.assertAlmostEqual(pieces[0][1][1],.45)
        from mz138_support_ceiling import possible
        self.assertTrue(possible(pieces[0]))
        self.assertIsNone(native_face_pieces(dict(surface_keys=[['e/a','AMBIGUOUS:0:MIN,1:MIN']]),e))


if __name__=='__main__':unittest.main()
