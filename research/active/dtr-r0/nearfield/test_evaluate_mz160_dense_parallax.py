import copy
import unittest

from evaluate_mz160_dense_parallax import evaluate_records


def fixture(episodes=1,mode='lateral',objects=True,points_per_frame=1):
    rows=[];es=[];geometry=[];baseline=[];candidate=[];frames=[];audits=[]
    context=dict(background=dict(center_m=[12.,0.,2.],size_m=[.1,20.,20.]),
                 floor=dict(center_m=[4.,0.,-.05],size_m=[24.,20.,.1]))
    for episode in range(episodes):
        ep=f'synthetic_{episode}'
        audits.append(dict(episode=ep,camera_motion_mode=mode,objects=[dict(name='shape0',velocity_mps=[0.,0.,0.])]))
        for step in range(4):
            index=len(rows);ident=f'{ep}_{step}';camera=dict(x=0.,y=step*.01 if mode=='lateral' else 0.,z=1.7,yaw=0.,pitch=0.,roll=0.)
            origin=[camera['x'],camera['y'],0.]
            row=dict(id=ident,episode_id=ep,time_s=step*.25,rgb_intrinsics=dict(width=640,height=360,cx=320.,cy=180.,fx=320.,fy=320.),
                     tof_zones=[dict(zone_id=0,targets=[])])
            obj=dict(name='shape0',center_m=[2.,0.,1.7],size_m=[.2,.2,.2],source_role='near_rod')
            frames.append(dict(id=ident,episode=ep,camera=camera,body_origin_m=origin,objects=[obj] if objects else []))
            es.append(dict(id=ident,episode_id=ep,family='rod',camera=camera,body_origin_m=origin,
                           native_bounds=[dict(name='shape0',center_m=obj['center_m'],extent_m=[.1,.1,.1])] if objects else [],zonal_tof_native=[]))
            points=[dict(pixel=[320.,180.],range_bounds_m=[1.8,2.],reference_indices=[index-2,index-3],excluded_zone=-1,alert_support=True) for _ in range(points_per_frame)] if step==3 else []
            geometry.append(dict(id=ident,points=points,pairs=[],state='POINTS' if points else 'UNKNOWN'))
            baseline.append(dict(id=ident,candidate=False));candidate.append(dict(id=ident,candidate=bool(points),baseline=False,new_geometry_support=bool(points)))
            rows.append(row)
    return rows,es,geometry,baseline,candidate,dict(context,frames=frames,source_audit=audits),{}


class DenseEvaluatorTests(unittest.TestCase):
    def test_static_lateral_correct_rod_component_and_alert(self):
        result=evaluate_records(*fixture(4,points_per_frame=3))
        self.assertTrue(result['component_pass']);self.assertTrue(result['alert_pass'])
        self.assertEqual(result['geometry']['counts']['static_lateral_rod_points'],12)
        self.assertEqual(result['geometry']['accepted_point_containment_fraction'],1.)
        self.assertEqual(result['candidate']['metrics']['TP'],4)

    def test_context_false_near_and_controls_are_not_actor_witnesses(self):
        context=evaluate_records(*fixture(objects=False))
        self.assertEqual(context['geometry']['counts']['false_near_context_points'],1)
        self.assertEqual(context['geometry']['accepted_point_containment_fraction'],0.)
        self.assertFalse(context['alert_pass'])
        control=evaluate_records(*fixture(mode='rotation_only'))
        self.assertEqual(control['geometry']['counts']['control_finite_near_points'],1)
        self.assertEqual(control['geometry']['static_lateral_correct_actor_episodes'],[])

    def test_unknown_reference_stays_in_denominator(self):
        args=fixture();args[2][-1]['points'][0]['pixel']=[-1.,180.]
        result=evaluate_records(*args)
        self.assertEqual(result['geometry']['counts']['accepted_points'],1)
        self.assertEqual(result['geometry']['counts']['known_reference_points'],0)
        self.assertEqual(result['geometry']['accepted_point_containment_fraction'],0.)

    def test_future_reference_rejected_and_inputs_not_mutated(self):
        args=fixture();before=copy.deepcopy(args);evaluate_records(*args)
        self.assertEqual(args,before)
        args[2][-1]['points'][0]['reference_indices']=[3,1]
        with self.assertRaises(ValueError):evaluate_records(*args)


if __name__=='__main__':unittest.main()
