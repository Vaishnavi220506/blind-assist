"""Native finite-footprint ToF wrapper; unchanged MZ113 Radar/IMU runs first.

Legacy centre-ray ToF is evaluator-only. Latent rays, reflectance and lineage
never enter raw observations. New ToF uses a separate per-episode RNG.
"""
import math
import random

import mz113_dynamic_sensors as original
from mz115_zonal_tof import MAX_TARGETS, MODEL, measure_zone, zone_geometry


def sensors(u,world,frame,actors,state):
    row,evaluation,provenance=original.sensors(u,world,frame,actors,state)
    legacy={key:row.pop(key) for key in list(row) if
            (key.startswith('tof_') or key.startswith('tof64_')) and key!='tof_packet_received'}
    ep=frame['episode'];camera=frame['camera'];st=state[ep]
    if 'mz115_zonal_rng' not in st:
        st['mz115_zonal_rng']=random.Random(frame.get('tof_sensor_seed',frame['sensor_seed']+1150003))
    rng=st['mz115_zonal_rng'];forward,right,up=original.basis(camera)
    origin=u.Vector(*(camera[k]*100 for k in ('x','y','z')))
    zones=[];native=[];lineages=[]
    for zone_id in range(64):
        geometry,rays=zone_geometry(zone_id);hits=[]
        for ray in rays:
            direction=[forward[k]+math.tan(math.radians(ray['theta_deg']))*right[k]+
                       math.tan(math.radians(ray['phi_deg']))*up[k] for k in range(3)]
            norm=math.sqrt(sum(v*v for v in direction))
            end=origin+u.Vector(*(v/norm*400 for v in direction))
            result=u.SystemLibrary.line_trace_single(world,origin,end,u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                                                     True,[],u.DrawDebugTrace.NONE)
            private_hit=dict(ray,range_m=None)
            if result and result.to_tuple()[0]:
                fields=result.to_tuple();point,component=fields[5],fields[10]
                matched=next((obj for actor,obj in zip(actors,frame['objects'])
                              if component==actor.static_mesh_component),None)
                environment=next((rho for known,rho in state.get('mz115_environment_reflectance',[])
                                  if component==known),None)
                rho=(1. if environment is None else environment) if matched is None else matched.get('tof_reflectance_proxy',1.)
                distance=math.sqrt((point.x-origin.x)**2+(point.y-origin.y)**2+(point.z-origin.z)**2)/100
                private_hit.update(range_m=distance,reflectance_proxy=rho,
                                   actor_id=None if matched is None else ep+'/'+matched['name'],
                                   hit_point_m=[point.x/100,point.y/100,point.z/100],
                                   material_authority=('DEFAULT_ENVIRONMENT_RHO_1' if environment is None else
                                                       'SOURCE_ENVIRONMENT_TOF_REFLECTANCE') if matched is None else 'SOURCE_LATENT_TOF_REFLECTANCE')
            hits.append(private_hit)
        targets,details=measure_zone(hits,rng,packet_received=row['tof_packet_received'])
        zones.append(dict(geometry,target_count=len(targets),targets=targets))
        native.append(dict(zone_id=zone_id,private_rays=hits,**details))
        lineages.append(dict(zone_id=zone_id,returned_lineage=details['returned_lineage'],
                             observed_targets=targets,output_state=details['output_state']))
    row.update(tof_zones=zones,tof_max_targets=MAX_TARGETS,tof_target_order='strongest',tof_model=MODEL)
    evaluation.update(legacy_center_tof_control=legacy,zonal_tof_native=native,
                      zonal_tof_authority='EVALUATOR_ONLY_PRIVATE_RAYS_HISTOGRAM_REFLECTANCE_AND_LINEAGE')
    provenance.update(zonal_tof_lineage=lineages,
                      zonal_tof_authority='EVALUATOR_ONLY_RETURN_TO_PRIVATE_HIT_LINEAGE_NOT_OBSERVATIONS')
    return row,evaluation,provenance
