"""EVALUATOR-ONLY returned-contributor ceilings; never observation inference."""
import math
import numpy as np
from mz115_spatial_allocation import slant_envelope, possible, certain
from mz128_zone_weighting import column_weights, FOUR, THRESHOLD

ARMS=('angular','pooled_native','ownership_extent','full_native')
METHOD=dict(authority='EVALUATOR_ONLY_ORACLE_NOT_OBSERVABLE_ALGORITHM',
    cohort='CONSUMED_MZ136_DEV48_RETURNED_CONTRIBUTORS_ONLY',
    angular='RETURN_NATIVE_THETA_PHI_ENVELOPE_WITH_ORIGINAL_RANGE_AND_ORIENTATION_INTERVALS',
    pooled_native='PER_RETURN_NATIVE_XYZ_SAMPLE_ENVELOPE_ATTRIBUTION_BRIDGE',
    ownership_extent='UNION_OF_ACTOR_FACE_NATIVE_XYZ_SAMPLE_ENVELOPES',
    full_native='EXACT_RETURNED_NATIVE_POINT_CORRIDOR_MEMBERSHIP',
    face_tolerance_m=1e-5,certainty_unit='ORIGINAL_RETURN_ALL_PARTS_INSIDE',
    final='FROZEN_BINARY_FOUR_ZONE_SCORE_GE1_OR_RETURN_CERTAINTY_OR_MZ129_RADAR',
    fallback='INCUMBENT_RETURN_ON_MISSING_LINEAGE',sweeps=0)


def envelope(points):
    values=np.asarray(points,float)
    return np.stack([values.min(0),values.max(0)],axis=1).tolist()


def contains(xyz,point):
    return all(lo-1e-9<=v<=hi+1e-9 for (lo,hi),v in zip(xyz,point))


def inside(point):
    return possible([[v,v] for v in point])


def surface_key(hit,bounds):
    actor=hit['actor_id']; name=actor.rsplit('/',1)[-1]
    obj=bounds.get(name)
    if obj is None:return (actor,'UNKNOWN_ACTOR_BOUND')
    p=np.asarray(hit['hit_point_m']);lo=np.asarray(obj['center_m'])-obj['extent_m'];hi=np.asarray(obj['center_m'])+obj['extent_m']
    eps=METHOD['face_tolerance_m']
    if np.any(p<lo-eps) or np.any(p>hi+eps):return (actor,'UNKNOWN_POINT_OUTSIDE_BOUND')
    faces=[f'{axis}:{side}' for axis in range(3) for side,bound in [('MIN',lo),('MAX',hi)] if abs(p[axis]-bound[axis])<=eps]
    if not faces:return (actor,'UNKNOWN_FACE')
    return (actor,faces[0] if len(faces)==1 else 'AMBIGUOUS:'+','.join(faces))


def linked_hits(item,native):
    zone=native.get(item['zone_id'])
    if zone is None:return []
    lineage=[v for v in zone['returned_lineage'] if v['target_index']==item['target_slot']]
    if len(lineage)!=1 or not lineage[0]['hit_indices']:return []
    rays={r['subray']:r for r in zone['private_rays']}
    hits=[rays.get(i) for i in lineage[0]['hit_indices']]
    if any(h is None or not all(k in h for k in ('hit_point_m','actor_id','theta_deg','phi_deg')) for h in hits):return []
    if any(not np.isfinite(h['hit_point_m']).all() for h in hits):return []
    return hits


def support(row,item,hits,evaluation,yaw):
    if not hits:
        old=dict(possible=possible(item['localized_xyz']),certain=certain(item['coarse_xyz']),
                 pieces=[item['localized_xyz']],fallback=True)
        return {a:dict(old) for a in ARMS},[],[]
    intr=row['rgb_intrinsics']; angles=np.asarray([[h['theta_deg'],h['phi_deg']] for h in hits])
    lo,hi=angles.min(0),angles.max(0)
    box=[intr['cx']+intr['fx']*math.tan(math.radians(lo[0])),intr['cy']-intr['fy']*math.tan(math.radians(hi[1])),
         intr['cx']+intr['fx']*math.tan(math.radians(hi[0])),intr['cy']-intr['fy']*math.tan(math.radians(lo[1]))]
    dy=.5+.2*row['time_s'];pitch=row['camera_pitch_deg']
    angular=slant_envelope(box,item['range_bounds'],intr,(pitch-.5,pitch+.5),(yaw-dy,yaw+dy),row['camera_in_body_m'][2])
    # The inherited enclosure routine accounts for height only. Preserve its
    # existing behavior: this fixed source must have zero x/y camera offsets.
    assert row['camera_in_body_m'][:2]==[0.,0.]
    points=[(np.asarray(h['hit_point_m'])-evaluation['body_origin_m']).tolist() for h in hits]
    bounds={o['name']:o for o in evaluation['native_bounds']};groups={};keys=[]
    for hit,point in zip(hits,points):
        key=surface_key(hit,bounds);keys.append(key);groups.setdefault(key,[]).append(point)
    pieces={'angular':[angular],'pooled_native':[envelope(points)],
            'ownership_extent':[envelope(v) for v in groups.values()],
            'full_native':[[[v,v] for v in p] for p in points]}
    result={a:dict(possible=any(possible(p) for p in pp),certain=all(certain(p) for p in pp),
                   pieces=pp,fallback=False) for a,pp in pieces.items()}
    # Tightening with identical native points cannot create possible support.
    assert not result['full_native']['possible'] or result['ownership_extent']['possible']
    assert not result['ownership_extent']['possible'] or result['pooled_native']['possible']
    assert len({result[a]['certain'] for a in ('pooled_native','ownership_extent','full_native')})==1
    return result,points,keys


def aggregate(row,bits,radar):
    weights=column_weights(row,FOUR)
    active=sorted({b['zone'] for b in bits if b['possible']})
    score=sum(weights[z] for z in active);sure=any(b['certain'] for b in bits)
    tof=bool(score>=THRESHOLD or sure)
    assert radar['candidate']==bool(radar['common_radar'] or radar['guard_events'])
    return dict(candidate=bool(tof or radar['candidate']),candidate_state='ALERT' if tof or radar['candidate'] else 'UNKNOWN',
        tof_candidate=tof,tof_score=score,certain=sure,active_zones=active,
        radar_candidate=radar['candidate'],common_radar=radar['common_radar'],guard_events=radar['guard_events'])


def frame_oracles(row,cached,baseline,radar,evaluation):
    native={z['zone_id']:z for z in evaluation['zonal_tof_native']}
    bits={a:[] for a in ('mz129',*ARMS)};returns=[];contributors=[]
    for item in cached['spatial_evidence']:
        hits=linked_hits(item,native);options,points,keys=support(row,item,hits,evaluation,cached['integrated_yaw_deg'])
        old=dict(possible=possible(item['localized_xyz']),certain=certain(item['coarse_xyz']))
        key=dict(zone=item['zone_id'],slot=item['target_slot'])
        bits['mz129'].append(dict(key,**old))
        for a,value in options.items():bits[a].append(dict(key,possible=value['possible'],certain=value['certain']))
        returns.append(dict(id=row['id'],**key,status=item['status'],hits=len(hits),
            actors=len({h['actor_id'] for h in hits}),surface_groups=len(set(keys)),surface_keys=keys,
            original=old,oracles=options))
        for i,(point,hit) in enumerate(zip(points,hits)):
            record=dict(id=row['id'],**key,subray=hit['subray'],actor=hit['actor_id'],surface=keys[i],
                body_point_m=point,corridor=inside(point),incumbent_contains=contains(item['localized_xyz'],point))
            for a,value in options.items():
                record[a+'_contains']=any(contains(p,point) for p in value['pieces'])
                record[a+'_possible']=value['possible']
            contributors.append(record)
    values={a:aggregate(row,b,radar) for a,b in bits.items()}
    assert values['mz129']['candidate']==baseline['candidate']
    assert values['mz129']['tof_score']==baseline['score']
    assert values['mz129']['certain']==baseline['certain_coarse']
    for a in ARMS:values[a]['changed_returns']=sum(b!=c for b,c in zip(bits[a],bits['mz129']))
    return dict(id=row['id'],arms=values,returns=returns,contributors=contributors)
