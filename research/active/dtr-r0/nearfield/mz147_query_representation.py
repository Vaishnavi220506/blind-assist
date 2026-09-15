"""Pixel evidence conditioned on each public regional range hypothesis.

The fixed body query supplies coordinates, never occupancy truth. All original
ToF/Radar slots remain in the separate MZ143 sensor vector. This is a learned
feature carrier, not point ownership, an independent-return veto, or clearance.
"""
import math
import cv2
import numpy as np
from mz143_corridor_features import _public
from mz125_observable_correction import foreground
from mz115_spatial_allocation import zone_box
from mz136_boundary_geometry import camera_to_body

REGIONS=('left','corridor','right')
DESCRIPTORS=('visible_fraction','mask_mean','contrast_abs_mean','contrast_abs_q95',
 'column_contrast_max','column_edge_max','column_positive_fraction_max','column_negative_fraction_max')
METHOD=dict(schema='MZ147_RANGE_CONDITIONED_BODY_QUERY_PIXELS_V1',
 range='VALID_REGIONAL_CENTER_FRONTPLANE_HYPOTHESIS_NOT_PIXEL_DEPTH',
 body_query=dict(forward=[.2,3.6],lateral=[-.3,.3],height=[.4,2.05]),
 pooling='REGION_DESCRIPTORS_REMAIN_CONDITIONED_ON_SAME_SLOT_AND_BODY_LATERAL_REGION',
 merged='RAW_SENSOR_FEATURES_RETAINED_NO_PRECISE_PIXEL_HYPOTHESIS',
 no_native_inputs=True,no_trained_parameters=True,backend_reason='GPU_BACKEND_UNAVAILABLE')


def query_masks(row,yaw,depth,box):
    intr=row['rgb_intrinsics'];h,w=intr['height'],intr['width']
    l,t,r,b=box;l=max(0,int(math.floor(l)));r=min(w,int(math.ceil(r)))
    t=max(0,int(math.floor(t)));b=min(h,int(math.ceil(b)))
    if r<=l or b<=t:return (l,t,r,b),None
    v,u=np.mgrid[t:b,l:r]
    rays=np.stack([np.ones_like(u),(u-intr['cx'])/intr['fx'],(intr['cy']-v)/intr['fy']],-1)
    points=depth*rays@camera_to_body(row,yaw).T+np.asarray(row['camera_in_body_m'])
    spatial=(points[...,0]>=.2)&(points[...,0]<=3.6)&(points[...,2]>=.4)&(points[...,2]<=2.05)
    lateral=points[...,1]
    return (l,t,r,b),[spatial&(lateral<-.3),spatial&(lateral>=-.3)&(lateral<=.3),spatial&(lateral>.3)]


def describe(region,mask,contrast,edge):
    if not region.any():return [0.]*len(DESCRIPTORS)
    count=region.sum(0);valid=count>0
    def column(x):return (np.where(region,x,0).sum(0)/np.maximum(1,count))[valid]
    absolute=np.abs(contrast)
    return [float(region.mean()),float(mask[region].mean()),float(absolute[region].mean()),
      float(np.quantile(absolute[region],.95)),float(column(absolute).max()),
      float(column(edge).max()),float(column(contrast>2/255).max()),
      float(column(contrast< -2/255).max())]


def extract(row,image,yaw):
    row=_public(row);intr=row['rgb_intrinsics']
    if image.dtype!=np.uint8 or image.shape!=(intr['height'],intr['width'],3):
        raise ValueError('Unchanged native BGR image required')
    if not math.isfinite(float(yaw)):raise ValueError('Finite causal yaw required')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    contrast=gray-cv2.GaussianBlur(gray,(31,1),7.,borderType=cv2.BORDER_REFLECT_101)
    edge=np.abs(cv2.Scharr(gray,cv2.CV_32F,1,0)/32.)
    _,binary=foreground(image);mask=(binary>0).astype(np.float32)
    values=[];names=[];joint=[];audit=[]
    for zone in sorted(row['tof_zones'],key=lambda z:z['zone_id']):
        theta,phi=[math.radians(sum(zone[k])/2) for k in ('theta_bounds_deg','phi_bounds_deg')]
        norm=math.sqrt(1+math.tan(theta)**2+math.tan(phi)**2)
        for slot in range(2):
            target=zone['targets'][slot] if slot<len(zone['targets']) else None
            active=bool(row['tof_packet_received'] and target and target['status']=='SIM_VALID'
                and np.isfinite(target['distance_m']) and target['distance_m']>0)
            features=np.zeros((3,len(DESCRIPTORS)),float)
            if active:
                box,regions=query_masks(row,yaw,target['distance_m']/norm,zone_box(zone,intr))
                if regions is not None:
                    l,t,r,b=box
                    for j,region in enumerate(regions):
                        features[j]=describe(region,mask[t:b,l:r],contrast[t:b,l:r],edge[t:b,l:r])
            prefix=f'zone{zone["zone_id"]:02d}.slot{slot}.'
            names.append(prefix+'valid_plane_hypothesis');values.append(float(active))
            for j,region in enumerate(REGIONS):
                names.extend(prefix+region+'.'+d for d in DESCRIPTORS);values.extend(features[j])
            if active:joint.append(features)
            if target:audit.append(dict(zone=zone['zone_id'],slot=slot,status=target['status'],
                valid_plane_hypothesis=active,visible_region_fractions=features[:,0].tolist()))
    pooled=np.asarray(joint,float) if joint else np.zeros((0,3,len(DESCRIPTORS)))
    for j,region in enumerate(REGIONS):
        valid=pooled[pooled[:,j,0]>0,j] if len(pooled) else np.zeros((0,len(DESCRIPTORS)))
        names.append(region+'.visible_slot_count');values.append(len(valid))
        for k,d in enumerate(DESCRIPTORS):
            stats=np.quantile(valid[:,k],[.5,.9,1.]) if len(valid) else np.zeros(3)
            for label,value in zip(('median','q90','max'),stats):
                names.append(region+'.'+d+'.'+label);values.append(float(value))
    values=np.asarray(values,np.float32)
    assert len(values)==len(names)==3275 and np.isfinite(values).all()
    return dict(values=values,names=names,audit=dict(schema=METHOD['schema'],slots=audit,
        target_plane_hypotheses=len(joint),encoded_slots=128,raw_sensor_vector_required=True,
        no_rgb_support_is_not_clear=True,no_native_inputs=True,merged_or_outside_support_not_deleted=True))
