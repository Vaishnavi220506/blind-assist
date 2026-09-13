"""Synthetic causal-Doppler, noise-draw and source-contract checks; no UE run."""
import copy
import math
from types import SimpleNamespace
import unittest

import mz107_sensors as original
import mz113_dynamic_sensors as dynamic
import mz113_source_spec as source


class Vector:
    def __init__(self,x,y,z):self.x,self.y,self.z=x,y,z
    def __add__(self,other):return Vector(self.x+other.x,self.y+other.y,self.z+other.z)


class Actor:
    def __init__(self,center):
        self.center=Vector(*center)
        self.static_mesh_component=object()
    def get_actor_bounds(self,_):return self.center,Vector(10.,10.,30.)


def trace(world,origin,end,*args):
    for actor in world:
        if all(abs(getattr(end,k)-getattr(actor.center,k))<1e-9 for k in ('x','y','z')):
            values=[None]*11
            values[0]=True
            values[5]=Vector(actor.center.x-10.,actor.center.y,actor.center.z)
            values[10]=actor.static_mesh_component
            return SimpleNamespace(to_tuple=lambda:tuple(values))
    return None


UE=SimpleNamespace(Vector=Vector,SystemLibrary=SimpleNamespace(line_trace_single=trace),
                   TraceTypeQuery=SimpleNamespace(TRACE_TYPE_QUERY1=1),DrawDebugTrace=SimpleNamespace(NONE=0))


def frame(t,episode='episode'):
    return dict(id=f'{episode}_{t}',episode=episode,family='synthetic',time_s=t,sensor_seed=113017,
                camera=dict(x=.5*t,y=0.,z=1.7,yaw=0.,pitch=-3.,roll=0.),
                body_origin_m=[.5*t,0.,0.],wearer_speed=.5,
                # Deliberately unrelated audit fields: native actor state owns velocity.
                objects=[dict(name='shape0',center_m=[900.,900.,900.],velocity_mps=[999.,999.,999.])],
                radar_ghost=dict(x=-.1,z=2.77))


class DynamicSensor(unittest.TestCase):
    def test_static_object_and_ego_motion(self):
        v=dynamic.relative_radial_velocity([3.,0.,1.7],[.125,0.,1.7],
                                          [3.,0.,1.7],[0.,0.,1.7],.25)
        self.assertAlmostEqual(v,-.5)

    def test_receding_object_relative_to_moving_ego(self):
        v=dynamic.relative_radial_velocity([3.5,0.,1.7],[.125,0.,1.7],
                                          [3.,0.,1.7],[0.,0.,1.7],.25)
        self.assertAlmostEqual(v,1.5)

    def test_lateral_crossing_and_lateral_ego_are_in_radial_projection(self):
        v=dynamic.relative_radial_velocity([3.,.5,1.8],[.1,.1,1.7],
                                          [3.,1.,1.8],[0.,0.,1.7],.25)
        expected=(-.4*2.9-2.4*.4)/math.hypot(2.9,.4)
        self.assertAlmostEqual(v,expected)

    def test_first_invalid_history_and_zero_range_have_no_doppler(self):
        for dt in (None,0.,-.25,float('nan')):
            self.assertIsNone(dynamic.relative_radial_velocity([3.,0.,0.],[0.,0.,0.],
                                                               [3.,0.,0.],[0.,0.,0.],dt))
        self.assertIsNone(dynamic.relative_radial_velocity([3.,0.,0.],[0.,0.,0.],None,None,.25))
        self.assertIsNone(dynamic.relative_radial_velocity([0.,0.,0.],[0.,0.,0.],
                                                           [1.,0.,0.],[0.,0.,0.],.25))

    def test_native_centers_own_velocity_and_episode_first_is_missing(self):
        actor=Actor((300.,0.,170.));state={}
        first,evaluation,_=dynamic.sensors(UE,[actor],frame(0.),[actor],state)
        self.assertTrue(any(first['radar_valid']))
        self.assertTrue(all(v is None for v in first['radar_velocity']))
        self.assertIsNone(evaluation['native_bounds'][0]['native_relative_radial_velocity_mps'])
        actor.center=Vector(337.5,0.,170.)
        f=frame(.25);f['wearer_speed']=999.
        _,evaluation,_=dynamic.sensors(UE,[actor],f,[actor],state)
        self.assertAlmostEqual(evaluation['native_bounds'][0]['native_relative_radial_velocity_mps'],1.)
        reset,ev,_=dynamic.sensors(UE,[actor],frame(0.,'new_episode'),[actor],state)
        self.assertTrue(all(v is None for v in reset['radar_velocity']))
        self.assertIsNone(ev['native_motion_dt_s'])

    def test_noise_draw_sequence_and_other_channels_match_mz107(self):
        actor=Actor((300.,10.,170.));old_state={};new_state={}
        for t in (0.,.25,.5,.75):
            f=frame(t)
            before=copy.deepcopy(f)
            old,_,_=original.sensors(UE,[actor],f,[actor],old_state)
            new,_,prov=dynamic.sensors(UE,[actor],f,[actor],new_state)
            self.assertEqual(f,before)
            self.assertEqual(old_state['episode']['rng'].getstate(),new_state['episode']['rng'].getstate())
            for key in old:
                if key!='radar_velocity':self.assertEqual(old[key],new[key],key)
            if t>0:self.assertEqual(old['radar_velocity'],new['radar_velocity'])
            for slot in prov['radar_slots']:
                if slot is not None:self.assertIn('doppler_authority',slot)
            self.assertFalse(any(any(token in key for token in ('native','truth','actor','velocity_mps')) for key in new))

    def test_source_fixed_count_transitions_and_stable_actor_names(self):
        a=source.source()
        self.assertEqual(a,source.source())
        self.assertEqual(len(a['frames']),240)
        self.assertEqual(a['source_design_aabb_frame_counts'],dict(positive=88,negative=152))
        self.assertEqual(a['source_design_transition_counts'],dict(enter=13,exit=15))
        for audit in a['source_audit']:
            fs=[f for f in a['frames'] if f['episode']==audit['episode']]
            self.assertEqual([f['time_s'] for f in fs],[j*.25 for j in range(12)])
            for prev,cur in zip(fs,fs[1:]):
                self.assertEqual([o['name'] for o in prev['objects']],[o['name'] for o in cur['objects']])
                for old,new,definition in zip(prev['objects'],cur['objects'],audit['objects']):
                    for before,after,v in zip(old['center_m'],new['center_m'],definition['velocity_mps']):
                        self.assertAlmostEqual((after-before)/.25,v)


if __name__=='__main__':unittest.main()
