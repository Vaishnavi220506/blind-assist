"""Lazy evaluator-only return/surface correspondence labels; no file access.

Inference must not receive this module's frame object, annotation states, native
identities or masks. Public usability is separate from label completeness.
"""
import math
from collections import Counter

import numpy as np

from evaluate_mz140_depthor import rasterize_native_bounds


METHOD = dict(schema='MZ162_RETURN_FACE_SET_RELATIVE_SLANT_LABELS_V1',
    authority='OFFLINE_CONSUMED_TRAIN_NATIVE_LABELS_ONLY',
    pose_authority='SOURCE_COMMANDED_REFERENCE', face_tolerance_m=1e-5,
    surface='UNION_OF_COMPLETE_RESOLVED_CONTRIBUTOR_FACES_AT_VISIBLE_FIRST_HIT',
    residual='(VISIBLE_NATIVE_SLANT-PUBLIC_MEASURED_RETURN_RANGE)/PUBLIC_MEASURED_RETURN_RANGE',
    mixed='SET_MEMBERSHIP_AND_PIXELWISE_RESIDUAL_NOT_UNIQUE_SLOT_IDENTITY',
    missing='UNKNOWN_OWNER_FACE_OR_LINEAGE_MASKS_ENTIRE_SLOT_LABEL_NOT_RAW_RETURN',
    unknown_pixels='NO_SAVED_NATIVE_AABB_FIRST_HIT_OR_NONUNIQUE_FACE',
    target_actor='ALL_SAVED_ACTORS_LABELS; SHAPE0_DIAGNOSTIC_ONLY',
    rendered_geometry=False, private_subray_layout_is_input=False)
LOW = np.array([.2, -.3, .4]); HIGH = np.array([3.6, .3, 2.05])


def _inside(points):
    return np.all((points >= LOW) & (points <= HIGH), axis=-1)


def _face(point, obj):
    point = np.asarray(point, float)
    low = np.asarray(obj['center_m'])-obj['extent_m']; high = np.asarray(obj['center_m'])+obj['extent_m']
    tolerance = METHOD['face_tolerance_m']
    if point.shape != (3,) or not np.isfinite(point).all(): return None, 'UNKNOWN_NONFINITE_HIT'
    if np.any(point < low-tolerance) or np.any(point > high+tolerance): return None, 'UNKNOWN_HIT_OUTSIDE_BOUND'
    choices = [axis*2+side for axis in range(3) for side, bound in enumerate((low, high))
               if abs(point[axis]-bound[axis]) <= tolerance]
    if len(choices) != 1: return None, 'UNKNOWN_AMBIGUOUS_FACE' if choices else 'UNKNOWN_FACE'
    return choices[0], None


def prepare_frame(row, evaluation, yaw):
    """Rasterize once, returning integer face IDs and all 128 slot metadata.

    Use slot_labels(frame, slot) for one temporary dense target at a time.
    face_id = native actor index * 6 + axis * 2 + side (MIN=0, MAX=1).
    No label-based slot filter is an inference eligibility rule.
    """
    if row.get('id') != evaluation.get('id'): raise ValueError('Frame identity mismatch')
    camera = evaluation.get('camera', {})
    if not all(k in camera and math.isfinite(float(camera[k])) for k in ('x','y','z','yaw','pitch')):
        raise ValueError('Complete source-commanded reference camera required')
    if not math.isfinite(float(camera.get('roll',0.))) or not math.isfinite(float(yaw)):
        raise ValueError('Finite orientation required')
    origin = np.asarray(evaluation['body_origin_m'],float)
    if origin.shape != (3,) or not np.isfinite(origin).all(): raise ValueError('Invalid body origin')
    ref = rasterize_native_bounds(row,evaluation,{'integrated_yaw_deg':float(yaw)})
    intr = row['rgb_intrinsics']; depth = ref['depth']; yy,xx = np.mgrid[:depth.shape[0],:depth.shape[1]]
    rays = np.stack([np.ones(depth.shape),(xx-intr['cx'])/intr['fx'],(intr['cy']-yy)/intr['fy']],-1)
    first_hit_known = (ref['owner'] >= 0) & np.isfinite(depth) & (depth > 0)
    directions = rays @ ref['camera_rotation'].T
    world = ref['camera_origin_world_m']+np.where(first_hit_known,depth,0.)[...,None]*directions
    slant = np.where(first_hit_known,depth*np.linalg.norm(rays,axis=-1),0.).astype(np.float32)
    face_ids = np.full(depth.shape,-1,np.int32)
    objects = evaluation['native_bounds']; names = {o['name']:i for i,o in enumerate(objects)}
    for index,obj in enumerate(objects):
        owned = first_hit_known & (ref['owner']==index)
        low = np.asarray(obj['center_m'])-obj['extent_m']; high = np.asarray(obj['center_m'])+obj['extent_m']
        face_count = np.zeros(depth.shape,np.uint8); chosen = np.full(depth.shape,-1,np.int32)
        for axis in range(3):
            for side,bound in enumerate((low,high)):
                hit = owned & (np.abs(world[...,axis]-bound[axis]) <= METHOD['face_tolerance_m'])
                face_count += hit; chosen[hit] = index*6+axis*2+side
        unique = owned & (face_count==1); face_ids[unique] = chosen[unique]
    known = first_hit_known & (face_ids>=0)
    risky = first_hit_known & _inside(world-origin)
    native = {z['zone_id']:z for z in evaluation['zonal_tof_native']}
    public = {z['zone_id']:z for z in row['tof_zones']}
    if set(public) != set(range(64)) or len(row['tof_zones']) != 64: raise ValueError('Exactly 64 public zones required')
    slots=[]
    for zid in range(64):
        zone=public[zid]
        if len(zone['targets'])>2: raise ValueError('At most two public targets required')
        for slot in range(2):
            target=zone['targets'][slot] if slot<len(zone['targets']) else None
            values=[target.get(k) for k in ('distance_m','range_noise_sigma_m','signal_strength_proxy')] if target else []
            usable=bool(target and row['tof_packet_received'] and target['status'] in ('SIM_VALID','SIM_MERGED')
                and all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
                and values[0]>0 and values[1]>=0 and values[2]>=0)
            meta=dict(zone_id=zid,target_slot=slot,present=target is not None,public_usable=usable,
                public_status=target['status'] if target else 'ABSENT',range_m=float(values[0]) if usable else None,
                complete_face_set=False,annotation_status='ABSENT' if target is None else 'UNUSABLE_PUBLIC_RETURN',
                face_ids=[],face_keys=[],owner_count=0,face_count=0,unknown_reasons=[],
                identity_unresolved=True,single_face_unresolved=True,native_contributors=0,
                native_corridor_contributors=0,native_inside_rgb_contributors=0,native_shape0_contributors=0)
            if usable:
                nz=native.get(zid,{})
                lineage=[v for v in nz.get('returned_lineage',[]) if v['target_index']==slot]
                hits={h['subray']:h for h in nz.get('private_rays',[])}
                reasons=[]; faces=[]; owners=[]
                if len(lineage)!=1 or not lineage[0]['hit_indices']: reasons.append('UNKNOWN_LINEAGE')
                else:
                    for hit_index in lineage[0]['hit_indices']:
                        hit=hits.get(hit_index)
                        if hit is None or 'hit_point_m' not in hit:
                            reasons.append('UNKNOWN_LINEAGE_HIT');continue
                        point=np.asarray(hit['hit_point_m'],float)
                        if point.shape!=(3,) or not np.isfinite(point).all():
                            reasons.append('UNKNOWN_NONFINITE_HIT');continue
                        meta['native_contributors']+=1;meta['native_corridor_contributors']+=int(_inside(point-origin))
                        cp=(point-ref['camera_origin_world_m'])@ref['camera_rotation']
                        if cp[0]>0:
                            u,v=intr['cx']+intr['fx']*cp[1]/cp[0],intr['cy']-intr['fy']*cp[2]/cp[0]
                            meta['native_inside_rgb_contributors']+=int(0<=u<=intr['width']-1 and 0<=v<=intr['height']-1)
                        actor=hit.get('actor_id')
                        if not isinstance(actor,str) or not actor:
                            reasons.append('UNKNOWN_NATIVE_OWNER');continue
                        name=actor.rsplit('/',1)[-1]
                        meta['native_shape0_contributors']+=int(name=='shape0')
                        if name not in names:
                            reasons.append('UNKNOWN_NATIVE_BOUND');continue
                        index=names[name]; face,reason=_face(point,objects[index])
                        owners.append(index)
                        if reason: reasons.append(reason)
                        else: faces.append(index*6+face)
                meta['unknown_reasons']=sorted(set(reasons)); meta['face_ids']=sorted(set(faces))
                meta['face_keys']=[dict(name=objects[f//6]['name'],axis=(f%6)//2,side='MIN' if f%2==0 else 'MAX') for f in meta['face_ids']]
                meta['owner_count']=len(set(owners));meta['face_count']=len(meta['face_ids'])
                complete=not reasons and bool(faces);meta['complete_face_set']=complete
                meta['annotation_status']=('UNKNOWN_INCOMPLETE_NATIVE_SET' if not complete else 'MIXED_OWNER_SET'
                    if meta['owner_count']>1 else 'MULTI_FACE_SINGLE_OWNER' if meta['face_count']>1 else 'SINGLE_FACE')
                meta['identity_unresolved']=not complete or meta['owner_count']!=1
                meta['single_face_unresolved']=not complete or meta['face_count']!=1
            slots.append(meta)
    return dict(reference=dict(owner=ref['owner'],face_ids=face_ids,known=known,
        first_hit_known=first_hit_known,slant_m=slant,risky=risky,target_visible=ref['target_visible'],
        target_risky=ref['target_visible']&risky),slots=slots,
        audit=dict(**METHOD,shape=list(depth.shape),slots=128,public_present=sum(s['present'] for s in slots),
            public_usable=sum(s['public_usable'] for s in slots),annotation_states=dict(Counter(s['annotation_status'] for s in slots)),
            known_first_hit_pixels=int(first_hit_known.sum()),known_unique_face_pixels=int(known.sum()),
            ambiguous_first_hit_face_pixels=int((first_hit_known&~known).sum()),
            unknown_first_hit_pixels=int((~first_hit_known).sum()),
            source_commanded_minus_public_yaw_deg=float(camera['yaw'])-float(yaw)))


def slot_labels(frame, slot):
    """Generate one slot's temporary HxW label arrays; mixed sets stay a union."""
    meta=frame['slots'][slot] if isinstance(slot,int) else slot
    ref=frame['reference']; known=ref['known'] if meta['complete_face_set'] else np.zeros_like(ref['known'])
    member=known & np.isin(ref['face_ids'],meta['face_ids'])
    residual=np.zeros(member.shape,np.float32)
    if member.any(): residual[member]=ref['slant_m'][member]/meta['range_m']-1.
    return dict(member=member,known=known,residual_known=member.copy(),relative_slant_residual=residual,
        audit=dict(visible_member_pixels=int(member.sum()),known_pixels=int(known.sum()),
                   visible_member_columns=int(member.any(axis=0).sum()),
                   residual_authority='UNIQUE_VISIBLE_FIRST_HIT_PER_PIXEL_NOT_UNIQUE_RETURN_SOURCE'))
