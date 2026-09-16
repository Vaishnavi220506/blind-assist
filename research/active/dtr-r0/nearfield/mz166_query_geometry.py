"""Public lateral-query adapter and separately scoped evaluator-only labels.

public_query never reads native/evaluator data. The label functions are offline
FIT supervision only: fixed source-commanded camera and saved native AABBs.
"""
from copy import deepcopy
import math

import numpy as np

from mz143_corridor_features import _public
from mz161_dense_task import ray_query

OFFSETS = (-.45, -.30, -.15, 0., .15, .30, .45)
CORRIDOR_LO = np.array([.2, -.3, .4], dtype=np.float64)
CORRIDOR_HI = np.array([3.6, .3, 2.05], dtype=np.float64)


def _offset(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('Finite fixed lateral offset required')
    for fixed in OFFSETS:
        if abs(value-fixed)<1e-7:return fixed
    raise ValueError('Offset outside the fixed seven-query design')


def public_query(row,yaw,base_image,base_tokens,base_valid,offset):
    """Only move query coordinates; all cached measurement evidence survives.

    Positive offset shifts the corridor toward positive body y. Subtracting it
    from the camera and return y coordinates expresses the same fixed rays and
    measurements relative to the queried corridor centre, never a new camera.
    """
    offset=_offset(offset);yaw=float(yaw)
    if not math.isfinite(yaw):raise ValueError('Finite public yaw required')
    public=_public(row)
    intr=public['rgb_intrinsics'];h,w=int(intr['height']),int(intr['width'])
    image=np.array(base_image,dtype=np.float32,copy=True)
    tokens=np.array(base_tokens,dtype=np.float32,copy=True)
    valid=np.array(base_valid,dtype=bool,copy=True)
    if image.shape!=(22,h,w) or tokens.shape!=(132,21) or valid.shape!=(132,):
        raise ValueError('Expected cached22xHxW,132x21 and132-mask tensors')
    if not np.isfinite(image).all() or not np.isfinite(tokens).all():
        raise ValueError('Public cache requires finite values with explicit masks')
    usable=valid & (tokens[:,0]>0)
    if offset != 0.:
        origin=np.asarray(public['camera_in_body_m'],dtype=np.float64).copy()
        if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('Invalid public camera offset')
        origin[1]-=offset
        public['camera_in_body_m']=origin.tolist()
        _,query=ray_query(public,yaw)
        image[13:16]=query.astype(np.float16).astype(np.float32)
        tokens[np.ix_(usable,[6,10,11])]-=np.float32(offset/2.)
    return dict(image=image,tokens=tokens,valid=valid,audit=dict(
        authority='PUBLIC_CACHED_OBSERVATIONS_AND_FIXED_QUERY_ONLY',offset_m=offset,
        corridor_lateral_bounds_m=[offset-.3,offset+.3],corridor_width_m=.6,
        zero_query_exact_copy=offset==0.,usable_tokens_shifted=int(usable.sum()) if offset else 0,
        changed_image_channels=[] if offset==0. else [13,14,15],
        changed_token_columns=[] if offset==0. else [6,10,11],
        query_quantization='FLOAT32_TO_FLOAT16_TO_FLOAT32_MATCHES_ORIGINAL_IMAGE_CACHE',
        camera_rays_and_measurement_planes_unchanged=True,all132_slots_preserved=True))


def make_query_references(row,evaluation,yaw):
    """EVALUATOR ONLY: freeze one native first-hit projection for all queries.

    IDs may be checked here by make_labels; they never become predictor inputs.
    No source-designed labels/objects are substituted for native saved bounds.
    """
    from mz161_dense_labels import make_labels
    from evaluate_mz140_depthor import rasterize_native_bounds
    central=make_labels(row,evaluation,yaw)
    raster=rasterize_native_bounds(row,evaluation,{'integrated_yaw_deg':float(yaw)})
    known=central['known'].copy();depth=raster['depth'];intr=row['rgb_intrinsics']
    yy,xx=np.mgrid[:depth.shape[0],:depth.shape[1]]
    rays=np.stack([np.ones(depth.shape),(xx-intr['cx'])/intr['fx'],(intr['cy']-yy)/intr['fy']],-1)
    directions=rays@raster['camera_rotation'].T
    body_origin=np.asarray(evaluation['body_origin_m'],np.float64)
    points=raster['camera_origin_world_m']-body_origin+np.where(known,depth,0.)[...,None]*directions
    lower=[];upper=[];names=[]
    for actor in evaluation['native_bounds']:
        center=np.asarray(actor['center_m'],np.float64)-body_origin
        extent=np.asarray(actor['extent_m'],np.float64)
        lower.append(center-extent);upper.append(center+extent);names.append(actor['name'])
    reference=dict(known=known,body_points=points,owner=raster['owner'].copy(),
        target_visible=central['target_visible'].copy(),central=central,
        native_lo=np.asarray(lower,np.float64).reshape(-1,3),
        native_hi=np.asarray(upper,np.float64).reshape(-1,3),names=names,
        audit=dict(authority='OFFLINE_FIT_NATIVE_AABB_REFERENCE_NOT_PREDICTOR_INPUT',
            pose_authority='SOURCE_COMMANDED_REFERENCE',rendered_labels=False,
            raster_unchanged_across_queries=True,body_origin_m=body_origin.tolist(),
            reference_camera_origin_world_m=raster['camera_origin_world_m'].tolist()))
    central_from_points=known & np.all((points>=CORRIDOR_LO)&(points<=CORRIDOR_HI),axis=-1)
    if not np.array_equal(central_from_points,central['target']):raise ValueError('Central raster/label mismatch')
    central_frame=bool(np.any(np.all((reference['native_hi']>=CORRIDOR_LO)&
                                  (reference['native_lo']<=CORRIDOR_HI),axis=-1)))
    if central_frame != central['audit']['native_aabb_corridor_truth']:raise ValueError('Central native volume mismatch')
    return reference


def label_for_query(reference,offset):
    """EVALUATOR ONLY: shifted corridor labels over the fixed first-hit raster.

    Return make_labels-compatible masks/weights/audit, plus boolean ``frame``
    for all-native-AABB volume overlap. Pixel and frame truth remain distinct.
    """
    from mz161_dense_labels import _column_counts
    offset=_offset(offset)
    shift=np.array([0.,offset,0.]);lo=CORRIDOR_LO+shift;hi=CORRIDOR_HI+shift
    if offset==0.:
        result=deepcopy(reference['central'])
    else:
        known=reference['known'].copy();points=reference['body_points']
        target=known & np.all((points>=lo)&(points<=hi),axis=-1)
        visible=reference['target_visible'].copy();risky=visible & target
        negative=known & ~target
        positive_count,negative_count=int(target.sum()),int(negative.sum())
        classes=int(positive_count>0)+int(negative_count>0)
        weights=np.zeros(known.shape,np.float32)
        if positive_count:weights[target]=1./(classes*positive_count)
        if negative_count:weights[negative]=1./(classes*negative_count)
        volumes=np.all((reference['native_hi']>=lo)&(reference['native_lo']<=hi),axis=-1)
        actors=[]
        for index,name in enumerate(reference['names']):
            actor_visible=known & (reference['owner']==index);actor_risk=actor_visible & target
            actors.append(dict(name=name,native_aabb_corridor_truth=bool(volumes[index]),
                visible_pixels=int(actor_visible.sum()),visible_corridor_pixels=int(actor_risk.sum()),
                visible_columns=int(actor_visible.any(axis=0).sum()),
                visible_corridor_columns=int(actor_risk.any(axis=0).sum()),columns=_column_counts(actor_visible,actor_risk)))
        frame=bool(volumes.any());audit=deepcopy(reference['central']['audit'])
        audit.update(positive_pixels=positive_count,known_negative_pixels=negative_count,
            positive_weight_mass=float(weights[target].sum(dtype=np.float64)),
            negative_weight_mass=float(weights[negative].sum(dtype=np.float64)),
            total_weight_mass=float(weights.sum(dtype=np.float64)),
            native_aabb_corridor_truth=frame,any_visible_first_hit_corridor=bool(target.any()),
            native_volume_truth_vs_visible_mismatch=frame!=bool(target.any()),
            target_corridor_pixels=int(risky.sum()),target_corridor_columns=int(risky.any(axis=0).sum()),
            target_columns=_column_counts(visible,risky),actors=actors)
        result=dict(target=target,known=known,target_visible=visible,target_corridor=risky,weights=weights,audit=audit)
    result['frame']=bool(result['audit']['native_aabb_corridor_truth'])
    result['audit'].update(query_offset_m=offset,query_corridor_lo_m=lo.tolist(),query_corridor_hi_m=hi.tolist(),
        query_geometry='SHIFT_CORRIDOR_ONLY_FIXED_BODY_POINTS_AND_CAMERA_RASTER',
        pose_authority='SOURCE_COMMANDED_REFERENCE',rendered_labels=False)
    return result
