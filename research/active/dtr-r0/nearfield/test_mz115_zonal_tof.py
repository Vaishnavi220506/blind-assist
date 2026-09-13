"""Synthetic finite-footprint, measurement-boundary and channel-parity tests."""
import copy
import json
import math
import random
from types import SimpleNamespace
import unittest

import mz113_dynamic_sensors as previous_sensor
import mz115_zonal_sensors as wrapper
import mz115_zonal_tof as model
from test_mz113_dynamic_sensors import Actor, Vector, frame


class NoNoise:
    def __init__(self):self.calls=[]
    def gauss(self,mean,sigma):self.calls.append((mean,sigma));return mean


def hit(distance,rho=1.,**metadata):
    return dict(range_m=distance,reflectance_proxy=rho,**metadata)


def nine(*hits):return list(hits)+[dict(range_m=None)]*(9-len(hits))


def every_ray_hit(world,origin,end,*args):
    delta=[getattr(end,k)-getattr(origin,k) for k in ('x','y','z')]
    length=math.sqrt(sum(v*v for v in delta))
    point=Vector(*(getattr(origin,k)+delta[i]/length*200 for i,k in enumerate(('x','y','z'))))
    fields=[None]*11;fields[0]=True;fields[5]=point;fields[10]=world[0].static_mesh_component
    return SimpleNamespace(to_tuple=lambda:tuple(fields))


UE=SimpleNamespace(Vector=Vector,SystemLibrary=SimpleNamespace(line_trace_single=every_ray_hit),
                   TraceTypeQuery=SimpleNamespace(TRACE_TYPE_QUERY1=1),DrawDebugTrace=SimpleNamespace(NONE=0))


class ZonalToF(unittest.TestCase):
    def test_fixed_footprints_and_thin_pole_can_fall_between_quadrature_rays(self):
        geometry,rays=model.zone_geometry(27)
        self.assertEqual(geometry['theta_bounds_deg'],[-5.625,0.])
        self.assertEqual(geometry['phi_bounds_deg'],[0.,5.625])
        self.assertEqual(len(rays),9)
        # A visible narrow angular interval lies in this zone, but between its
        # fixed subrays. This is numerical aliasing, not a hardware miss rate.
        low,high=-2.0,-1.9
        self.assertTrue(geometry['theta_bounds_deg'][0]<low<high<geometry['theta_bounds_deg'][1])
        samples=[hit(1.2) if low<=r['theta_deg']<=high else dict(range_m=None) for r in rays]
        targets,private=model.measure_zone(samples,NoNoise())
        self.assertEqual(targets,[])
        self.assertEqual(private['output_state'],'SIM_NO_DETECTION')

    def test_far_wall_can_be_stronger_than_near_pole(self):
        targets,private=model.measure_zone([hit(1.,.1)]+[hit(3.)]*8,NoNoise())
        self.assertEqual(len(targets),2)
        self.assertAlmostEqual(targets[0]['distance_m'],3.)
        self.assertAlmostEqual(targets[1]['distance_m'],1.)
        self.assertGreater(targets[0]['signal_strength_proxy'],targets[1]['signal_strength_proxy'])
        self.assertEqual(private['returned_lineage'][0]['hit_indices'],list(range(1,9)))

    def test_weak_near_return_disappears_without_becoming_clearance(self):
        targets,private=model.measure_zone([hit(1.,.05)]+[hit(3.)]*8,NoNoise())
        self.assertEqual(len(targets),1)
        self.assertEqual(targets[0]['distance_m'],3.)
        near=next(c for c in private['components'] if c['pre_noise_range_m']==1.)
        self.assertFalse(near['detected'])
        self.assertLess(near['signal_strength_proxy'],model.SIGNAL_FLOOR)

    def test_merged_semantics_and_exact_separation_boundary(self):
        merged,private=model.measure_zone(nine(hit(1.),hit(1.5)),NoNoise())
        self.assertEqual(len(merged),1)
        self.assertEqual(merged[0]['status'],'SIM_MERGED')
        self.assertEqual(len(private['components'][0]['peak_bins']),2)
        separate,_=model.measure_zone(nine(hit(1.),hit(1.6)),NoNoise())
        self.assertEqual(len(separate),2)
        self.assertTrue(all(t['status']=='SIM_VALID' for t in separate))
        chain,_=model.measure_zone(nine(hit(1.),hit(1.5),hit(2.)),NoNoise())
        self.assertEqual(len(chain),1)
        self.assertEqual(chain[0]['status'],'SIM_MERGED')

    def test_peak_plateau_is_single_and_count_is_capped_strongest_first(self):
        self.assertEqual(model.peak_bins([0.,1.,1.,0.,2.,0.]),[1,4])
        targets,_=model.measure_zone(nine(hit(1.),hit(2.),hit(3.)),NoNoise())
        self.assertEqual([t['distance_m'] for t in targets],[1.,2.])
        self.assertEqual(len(targets),model.MAX_TARGETS)

    def test_noise_sigma_is_injected_measurement_noise_not_private_spread(self):
        noise=NoNoise()
        targets,private=model.measure_zone(nine(hit(1.),hit(1.5)),noise)
        self.assertEqual(noise.calls,[(0.,.04)])
        self.assertEqual(targets[0]['range_noise_sigma_m'],.04)
        self.assertAlmostEqual(private['components'][0]['private_hit_span_m'],.5)
        self.assertAlmostEqual(targets[0]['distance_m']/.02,round(targets[0]['distance_m']/.02))

    def test_packet_missing_and_no_detection_are_distinct_without_dummy_target(self):
        rng=NoNoise()
        targets,private=model.measure_zone([hit(2.)]*9,rng,packet_received=False)
        self.assertEqual(targets,[])
        self.assertEqual(rng.calls,[])
        self.assertEqual(private['output_state'],'SIM_PACKET_MISSING')
        self.assertTrue(private['components'][0]['detected'])

    def test_object_metadata_cannot_drive_peak_grouping_or_public_output(self):
        samples=[hit(1.,.1,actor_id='near')]+[hit(3.,actor_id='far')]*8
        changed=copy.deepcopy(samples)
        for i,h in enumerate(changed):h.update(actor_id=str(i),subray=999,hidden_weight=900.)
        first,_=model.measure_zone(samples,NoNoise());second,_=model.measure_zone(changed,NoNoise())
        self.assertEqual(first,second)
        for target in first:
            self.assertEqual(set(target),{'distance_m','range_noise_sigma_m','signal_strength_proxy','status'})
        with self.assertRaises(ValueError):model.measure_zone(nine(hit(1.,2.)),NoNoise())

    def test_wrapper_preserves_non_tof_channels_rng_and_only_exports_zone_tuples(self):
        actor=Actor((300.,10.,170.));old_state={};new_state={}
        for t in (0.,.25,.5):
            f=frame(t);f['sensor_seed']=1;f['tof_sensor_seed']=115017
            f['objects'][0]['tof_reflectance_proxy']=.7
            before=copy.deepcopy(f)
            old,_,_=previous_sensor.sensors(UE,[actor],f,[actor],old_state)
            new,ev,prov=wrapper.sensors(UE,[actor],f,[actor],new_state)
            self.assertEqual(f,before)
            self.assertEqual(old_state['episode']['rng'].getstate(),new_state['episode']['rng'].getstate())
            for key,value in old.items():
                if not key.startswith(('tof_','tof64_')) or key=='tof_packet_received':
                    self.assertEqual(value,new[key],key)
            self.assertEqual({key for key in new if key.startswith(('tof_','tof64_'))},
                             {'tof_packet_received','tof_zones','tof_max_targets','tof_target_order','tof_model'})
            self.assertEqual(len(new['tof_zones']),64)
            for zone in new['tof_zones']:
                self.assertEqual(set(zone),{'zone_id','theta_bounds_deg','phi_bounds_deg','target_count','targets'})
                self.assertEqual(zone['target_count'],len(zone['targets']))
                for target in zone['targets']:
                    self.assertEqual(set(target),{'distance_m','range_noise_sigma_m','signal_strength_proxy','status'})
            encoded=json.dumps(new)
            for token in ('subray','hit_indices','actor_id','reflectance_proxy','histogram','legacy_center'):
                self.assertNotIn(token,encoded)
            self.assertEqual(ev['legacy_center_tof_control']['tof64_range_m'],old['tof64_range_m'])
            self.assertEqual(len(ev['zonal_tof_native'][0]['private_rays']),9)
            self.assertEqual(prov['zonal_tof_lineage'][0]['observed_targets'],new['tof_zones'][0]['targets'])
            if t==0:self.assertTrue(any(z['target_count'] for z in new['tof_zones']))

    def test_wrapper_packet_loss_keeps_flag_and_empty_targets(self):
        actor=Actor((300.,0.,170.));f=frame(0.);f['sensor_seed']=31
        self.assertLess(random.Random(31).random(),.1)
        row,ev,_=wrapper.sensors(UE,[actor],f,[actor],{})
        self.assertFalse(row['tof_packet_received'])
        self.assertTrue(all(z['target_count']==0 and z['targets']==[] for z in row['tof_zones']))
        self.assertTrue(all(z['output_state']=='SIM_PACKET_MISSING' for z in ev['zonal_tof_native']))

    def test_private_environment_reflectance_override_and_default(self):
        environment=Actor((300.,0.,170.));f=frame(0.);f['sensor_seed']=1;f['objects']=[]
        for rho in (None,.30,.50):
            state={} if rho is None else {'mz115_environment_reflectance':[(environment.static_mesh_component,rho)]}
            row,ev,_=wrapper.sensors(UE,[environment],f,[],state)
            target=row['tof_zones'][0]['targets'][0]
            self.assertAlmostEqual(target['signal_strength_proxy'],(1. if rho is None else rho)/4.)
            ray=ev['zonal_tof_native'][0]['private_rays'][0]
            self.assertEqual(ray['reflectance_proxy'],1. if rho is None else rho)
            self.assertNotIn('reflectance_proxy',json.dumps(row))


if __name__=='__main__':unittest.main()
