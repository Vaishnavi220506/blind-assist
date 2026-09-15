"""Observation-preserving virtual query translations for supervised training.

Pixels and public sensor packets do not move. The query coordinate origin moves
laterally, so camera/return coordinates are expressed relative to that query.
Native geometry is used only by the separate TRAIN target builder.
"""
import copy
import math
import numpy as np
from mz143_corridor_features import extract as original_extract
from run_mz107_four_sensor import truth

OFFSETS=(-.45,-.30,-.15,0.,.15,.30,.45)
METHOD=dict(schema='MZ150_VIRTUAL_CORRIDOR_SUPERVISION_V1',offsets_m=list(OFFSETS),
 original_pixels_unchanged=True,physical_sensor_budget_unchanged=True,
 query_width_m=.6,default_inference_offset_m=0.,
 labels='TRAIN_NATIVE_AABB_INTERSECTION_WITH_TRANSLATED_QUERY_ONLY',
 predictor='MZ143_FEATURES_IN_QUERY_COORDINATES_NO_NATIVE_INPUT',
 radar_adapter='SUBTRACT_QUERY_OFFSET_FROM_VALID_RADAR_BODY_Y',
 no_native_support_pruning=True)


def extract(row,image,yaw,offset=0.):
    if not math.isfinite(offset):raise ValueError('Finite query offset required')
    public=copy.deepcopy(row)
    public['camera_in_body_m']=list(public['camera_in_body_m'])
    public['camera_in_body_m'][1]-=float(offset)
    value=original_extract(public,image,yaw)
    # Radar body coordinates in the inherited encoder do not use camera origin.
    # Keep raw range/angle and invalid sentinels; translate only valid body y.
    for slot in range(4):
        if value['sensor'][value['sensor_names'].index(f'radar{slot}.valid')]>0:
            index=value['sensor_names'].index(f'radar{slot}.body_y')
            value['sensor'][index]-=float(offset)
    value['audit']['virtual_query_offset_m']=float(offset)
    value['audit']['schema']=METHOD['schema']
    return value


def training_target(native_evaluator,offset):
    """Evaluator-only training supervision. Never called by inference."""
    shifted=copy.deepcopy(native_evaluator)
    shifted['body_origin_m']=list(shifted['body_origin_m'])
    shifted['body_origin_m'][1]+=float(offset)
    return bool(truth(shifted))
