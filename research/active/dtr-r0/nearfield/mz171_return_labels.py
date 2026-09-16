"""Evaluator-only labels for returned native ToF witnesses, never inputs.

These labels concern sampled contributors to an actual public return. They do
not describe a complete object, an unsampled surface, or rendered RGB pixels.
Actor identity and saved actor AABBs are not required for a finite native hit.
"""
from collections import Counter
import math
import numpy as np


METHOD=dict(schema='MZ171_RETURNED_TOF_WITNESS_LABELS_V1',
    authority='OFFLINE_NATIVE_RETURN_LINEAGE_SUPERVISION_NOT_PREDICTOR_INPUT',
    point_authority='SAVED_UE_LINE_TRACE_HIT_POINT_WORLD_METERS',
    body_origin_authority='SOURCE_COMMANDED_BODY_ORIGIN',
    corridor_lo_m=[.2,-.3,.4],corridor_hi_m=[3.6,.3,2.05],
    positive='ANY_FINITE_UNIQUELY_RESOLVED_REFERENCED_POINT_INSIDE_CLOSED_CORRIDOR',
    negative='UNIQUE_NONEMPTY_LINEAGE_ALL_REFERENCED_POINTS_RESOLVED_AND_OUTSIDE',
    unresolved='UNKNOWN_IS_MASKED_NOT_NEGATIVE',radar='ALL_FOUR_LABELS_UNKNOWN',
    rgb_gate=False,actor_identity_required=False,rendered_pixel_labels=False)
LOW=np.asarray(METHOD['corridor_lo_m']);HIGH=np.asarray(METHOD['corridor_hi_m'])


def _finite(value):
    return isinstance(value,(int,float,np.integer,np.floating)) and math.isfinite(float(value))


def _usable(row,target):
    if not target or not row['tof_packet_received'] or target.get('status') not in ('SIM_VALID','SIM_MERGED'):
        return False
    values=[target.get(k) for k in ('distance_m','range_noise_sigma_m','signal_strength_proxy')]
    return all(_finite(v) for v in values) and values[0]>0 and values[1]>=0 and values[2]>=0


def make_witness_labels(row,evaluation):
    """Return float32 target[132], bool known[132], and JSON-compatible audit.

    Token index 2*zone_id+slot matches MZ161; last four tokens are Radar.
    Public validity is evaluated independently from native annotation validity.
    A unique lineage with one known positive and another unresolved point is
    positive. Incomplete outside-only lineages remain unknown. Duplicate native
    zone/lineage matches cannot establish which public return was referenced.
    Duplicate hit indices cannot establish a complete negative annotation.
    Inputs are read without mutation; caller owns permitted cohort selection.
    """
    if row.get('id') is not None and evaluation.get('id') is not None and row['id']!=evaluation['id']:
        raise ValueError('Public/native frame identity mismatch')
    origin=np.asarray(evaluation['body_origin_m'],dtype=np.float64)
    if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('Finite native body origin required')
    zones=row['tof_zones'];by_id={z['zone_id']:z for z in zones}
    if len(zones)!=64 or set(by_id)!=set(range(64)):raise ValueError('Exactly64 unique public zones required')
    if any(len(z['targets'])>2 for z in zones):raise ValueError('At most two public returns per zone')
    native=evaluation.get('zonal_tof_native',[])
    target=np.zeros(132,np.float32);known=np.zeros(132,bool);slots=[]
    for zid in range(64):
        for slot in range(2):
            public=by_id[zid]['targets'];t=public[slot] if slot<len(public) else None;index=2*zid+slot
            info=dict(zone_id=zid,slot=slot,token_index=index,present=t is not None,public_usable=_usable(row,t),
                status=t.get('status') if t else 'ABSENT',state='UNKNOWN',reasons=[],
                returned_contributors=0,corridor_contributors=0,unresolved_references=0,
                ownerless_contributors=0,ownerless_corridor_contributors=0)
            if not info['public_usable']:
                info['reasons'].append('MISSING_PACKET' if not row['tof_packet_received'] else 'ABSENT' if t is None else 'UNUSABLE_PUBLIC_RETURN')
                slots.append(info);continue
            matches=[z for z in native if z.get('zone_id')==zid]
            if len(matches)!=1:
                info['reasons'].append('NONUNIQUE_OR_MISSING_NATIVE_ZONE');slots.append(info);continue
            nz=matches[0]
            if nz.get('packet_received',True) is False:
                info['reasons'].append('NATIVE_PACKET_MISSING');slots.append(info);continue
            lineages=[v for v in nz.get('returned_lineage',[]) if v.get('target_index')==slot]
            if len(lineages)!=1:
                info['reasons'].append('NONUNIQUE_OR_MISSING_LINEAGE');slots.append(info);continue
            indices=lineages[0].get('hit_indices')
            if not isinstance(indices,list) or not indices:
                info['reasons'].append('EMPTY_OR_MALFORMED_LINEAGE');slots.append(info);continue
            if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in indices):
                info['reasons'].append('MALFORMED_HIT_INDEX');slots.append(info);continue
            unique=list(dict.fromkeys(indices));complete=len(unique)==len(indices)
            if not complete:info['reasons'].append('DUPLICATE_HIT_INDEX')
            rays=nz.get('private_rays',[])
            for ix in unique:
                hits=[h for h in rays if h.get('subray')==ix]
                if len(hits)!=1:
                    info['unresolved_references']+=1;complete=False
                    info['reasons'].append('NONUNIQUE_OR_MISSING_HIT');continue
                hit=hits[0];point=hit.get('hit_point_m')
                if not isinstance(point,(list,tuple,np.ndarray)) or len(point)!=3 or not all(_finite(v) for v in point):
                    info['unresolved_references']+=1;complete=False
                    info['reasons'].append('NONFINITE_OR_MISSING_HIT_POINT');continue
                body=np.asarray(point,dtype=np.float64)-origin
                inside=bool(np.all((body>=LOW)&(body<=HIGH)))
                info['returned_contributors']+=1;info['corridor_contributors']+=int(inside)
                if hit.get('actor_id') is None:
                    info['ownerless_contributors']+=1;info['ownerless_corridor_contributors']+=int(inside)
            if info['corridor_contributors']:
                target[index]=1.;known[index]=True;info['state']='POSITIVE'
            elif complete and info['returned_contributors']==len(indices):
                known[index]=True;info['state']='NEGATIVE'
            info['reasons']=sorted(set(info['reasons']));slots.append(info)
    audit=dict(**METHOD,slots=slots,states=dict(Counter(s['state'] for s in slots)),
        public_slots=sum(s['present'] for s in slots),public_usable=sum(s['public_usable'] for s in slots),
        returned_contributors=sum(s['returned_contributors'] for s in slots),
        corridor_contributors=sum(s['corridor_contributors'] for s in slots),
        known_tof_slots=int(known[:128].sum()),positive_tof_slots=int(target[:128].sum()),
        radar_slots=4,radar_known=0)
    return dict(target=target,known=known,audit=audit)
