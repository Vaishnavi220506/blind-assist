"""Additive ToF contained-envelope anchors; never a veto of other evidence."""
import numpy as np
from mz145_causal_confirmation import fit_onset,predict as learned_predict
from run_mz143_corridor_evidence import operating_threshold

METHOD=dict(schema='MZ152_CONTAINED_RETURN_ANCHOR_V1',
 support='UNCHANGED_PUBLIC_MZ143_3SIGMA_FULL_ZONE_ENVELOPE',
 forward_m=[.2,3.6],lateral_m=[-.3,.3],height_m=[.4,2.05],
 only_usable_unmerged_with_valid_imu=True,minimum_contained_slots=1,
 rgb_required=False,independent_anchor_or_learned=True,
 missing_anchor='UNKNOWN_NOT_CLEAR_SPACE',no_support_contraction=True)


def anchors(features,names):
    x=np.asarray(features);lookup={name:i for i,name in enumerate(names)}
    if x.ndim!=2 or x.shape[1]<len(names) or not np.isfinite(x).all():
        raise ValueError('Finite feature matrix with named sensor columns required')
    valid_imu=x[:,lookup['global.imu_valid']]>.5
    slots=[]
    for zone in range(64):
        for slot in range(2):
            prefix=f'zone{zone:02d}.slot{slot}.'
            at=lambda field:x[:,lookup[prefix+field]]
            inside=valid_imu&(at('usable')>.5)&(at('merged')<.5)
            for axis,lo,hi in (('x',.2,3.6),('y',-.3,.3),('z',.4,2.05)):
                a,b=at('support_'+axis+'_lo'),at('support_'+axis+'_hi')
                inside &= (a>=lo)&(b<=hi)&(a<=b)
            slots.append(inside)
    slots=np.stack(slots,axis=1)
    return slots.any(axis=1),slots


def calibrate(rows,scores,target,baseline,anchor):
    required=np.asarray(target,bool)&np.asarray(baseline,bool)&~np.asarray(anchor,bool)
    if not required.any():return dict(low=1.,high=1.)
    low=operating_threshold(scores,target,required)
    high=fit_onset(rows,scores,target,required,low)
    return dict(low=low,high=high)


def predict(rows,scores,anchor,low,high):
    return np.asarray(anchor,bool)|learned_predict(rows,scores,low,high)
