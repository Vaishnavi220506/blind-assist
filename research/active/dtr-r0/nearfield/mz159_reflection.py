"""Reflect observed public geometry, then average a learned score over both views.

This is a synthetic symmetry constraint, not another camera or acquisition.
Retained Radar returns are re-sorted; discarded returns cannot be reconstructed.
"""
import copy
import math
import numpy as np
from mz143_corridor_features import _public,extract

METHOD=dict(schema='MZ159_OBSERVED_REFLECTION_V1',feature_dimension=2485,
    transform='BODY_Y_REFLECTION_RGB_PIXEL_CENTER_AND_ALL_PUBLIC_SENSOR_GEOMETRY',
    training='ORIGINAL_AND_REFLECTED_VIEWS_WEIGHT_HALF_SAME_SCENE_FOLD',
    minimum_leaf_view_count=16,original_minimum_leaf_count=8,
    score='ARITHMETIC_MEAN_OF_TWO_VIEW_PROBABILITIES_BEFORE_CAUSAL_READOUT',
    authority='SYNTHETIC_TRANSFORM_OF_OBSERVED_RETURNS_NOT_NEW_ACQUISITION',
    radar='RESORT_OBSERVED_RANGE_ANGLE_VELOCITY_TUPLES_AFTER_ANGLE_REFLECTION',
    backend_reason='GPU_BACKEND_UNAVAILABLE')


def _key(t):
    return tuple(float(v) if v is not None else math.inf for v in t[:3])


def reflected(row,image,yaw):
    public=copy.deepcopy(_public(row));intr=public['rgb_intrinsics']
    if image.dtype!=np.uint8 or image.shape!=(intr['height'],intr['width'],3):
        raise ValueError('Native BGR image matching public intrinsics required')
    if not math.isfinite(yaw):raise ValueError('Finite integrated public yaw required')
    public['camera_in_body_m'][1]=-public['camera_in_body_m'][1]
    intr['cx']=intr['width']-1-intr['cx']
    for zone in public['tof_zones']:
        zid=zone['zone_id'];zone['zone_id']=8*(zid//8)+7-zid%8
        lo,hi=zone['theta_bounds_deg'];zone['theta_bounds_deg']=[-hi,-lo]
    public['tof_zones'].sort(key=lambda z:z['zone_id'])
    keys=('radar_range_m','radar_angle','radar_velocity','radar_valid')
    slots=list(zip(*(public[k] for k in keys)))
    if slots!=sorted(slots,key=_key):
        raise ValueError('Expected inherited canonical observed Radar slot order')
    slots=sorted([(r,-a if a is not None else None,v,ok) for r,a,v,ok in slots],key=_key)
    for i,k in enumerate(keys):public[k]=[slot[i] for slot in slots]
    return public,np.ascontiguousarray(image[:,::-1]),-yaw


def views(row,image,yaw):
    original=extract(row,image,yaw)
    rr,ii,yy=reflected(row,image,yaw)
    mirror=extract(rr,ii,yy)
    assert original['sensor_names']==mirror['sensor_names']
    assert original['geometry_names']==mirror['geometry_names']
    assert original['audit']['public_slots']==mirror['audit']['public_slots']
    return np.stack([np.r_[v['sensor'],v['geometry']] for v in (original,mirror)]),dict(
        original=original['audit'],reflected=mirror['audit'],raw_slots_preserved=True)


def averaged_score(model,x):
    x=np.asarray(x)
    if x.ndim!=3 or x.shape[1]!=2 or not np.isfinite(x).all():
        raise ValueError('Expected finite [frames,2,features]')
    probabilities=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(-1,2)
    return probabilities.mean(1),probabilities
