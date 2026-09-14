"""Observation-only narrow foreground edge proposals tied to near ToF cohorts.

This exposes candidate geometry, not a certified interval or an alert veto.
Missing/merged/out-of-view measurements remain available in the original row.
"""
import math
import cv2
import numpy as np
from mz115_spatial_allocation import zone_box


def camera_to_body(row,yaw):
    p,y=math.radians(row['camera_pitch_deg']),math.radians(yaw)
    cp,sp,cy,sy=math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cp*cy,-sy,-sp*cy],[cp*sy,cy,-sp*sy],[sp,0,cp]])


def ray(row,u,v,yaw):
    intr=row['rgb_intrinsics']
    return camera_to_body(row,yaw)@np.array([1.,(u-intr['cx'])/intr['fx'],(intr['cy']-v)/intr['fy']])


def near_cohorts(row):
    items=[]
    if not row['tof_packet_received']:return []
    for z in row['tof_zones']:
        a,b=[math.radians(sum(z[k])/2) for k in ('theta_bounds_deg','phi_bounds_deg')]
        for slot,t in enumerate(z['targets']):
            if t['status']!='SIM_VALID' or not .2<t['distance_m']<4.2:continue
            depth=t['distance_m']/math.sqrt(1+math.tan(a)**2+math.tan(b)**2)
            items.append(dict(zone=z['zone_id'],slot=slot,depth=depth,range=t['distance_m'],
                sigma=t['range_noise_sigma_m'],box=zone_box(z,row['rgb_intrinsics'])))
    cohorts=[]
    for item in sorted(items,key=lambda v:v['depth']):
        if not cohorts or item['depth']-cohorts[-1][0]['depth']>.25:cohorts.append([])
        cohorts[-1].append(item)
    return cohorts


def proposals(row,image,yaw):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    contrast=gray-cv2.GaussianBlur(gray,(31,1),7.,borderType=cv2.BORDER_REFLECT_101)
    h,w=gray.shape;out=[]
    for cohort in near_cohorts(row):
        if len({v['zone'] for v in cohort})<2:continue
        support=np.zeros((h,w),bool)
        for item in cohort:
            a,b,c,d=item['box']
            # Search may extend across a measured zone's image border. Absence
            # of an adjacent return does not truncate independent RGB evidence.
            support[max(0,int(b)):min(h,int(math.ceil(d))),max(0,int(a)-8):min(w,int(math.ceil(c))+8)]=True
        valid_rows=support.sum(0)
        positive=(contrast>2.)&support
        ratio=positive.sum(0)/np.maximum(1,valid_rows)
        active=(valid_rows>=24)&(ratio>=.5)
        # Bridge small texture gaps inside a narrow candidate, after rejecting
        # low-persistence background tails. This is a proposal, not an extent bound.
        active=cv2.morphologyEx(active.astype(np.uint8)[None],cv2.MORPH_CLOSE,np.ones((1,7),np.uint8))[0].astype(bool)
        borders=np.diff(np.r_[False,active,False].astype(int))
        for left,right in zip(np.where(borders==1)[0],np.where(borders==-1)[0]):
            if right-left<2 or right-left>48:continue
            ys=np.where(positive[:,left:right].any(1))[0]
            if len(ys)<24:continue
            top,bottom=int(ys.min()),int(ys.max()+1)
            depth=float(np.median([v['depth'] for v in cohort]))
            # Diagnostic front-plane hypothesis, explicitly not full 3D extent.
            endpoints=[]
            for u in (float(left),float(right)):
                camera=np.array([1.,(u-row['rgb_intrinsics']['cx'])/row['rgb_intrinsics']['fx'],
                    (row['rgb_intrinsics']['cy']-(top+bottom)/2)/row['rgb_intrinsics']['fy']])
                endpoints.append((camera_to_body(row,yaw)@(depth*camera)+row['camera_in_body_m']).tolist())
            lo,hi=sorted(p[1] for p in endpoints)
            overlap=min(hi,.3)-max(lo,-.3)
            out.append(dict(box=[int(left),top,int(right),bottom],camera_depth_m=depth,
                depth_spread_m=float(np.ptp([v['depth'] for v in cohort])),
                side_interval_front_plane_m=[lo,hi],signed_overlap_front_plane_m=overlap,
                contrast_support=int(positive[:,left:right].sum()),
                zones=[v['zone'] for v in cohort],
                authority='OBSERVABLE_EDGE_PLUS_COHORT_MEDIAN_FRONT_PLANE_HYPOTHESIS_NOT_VOLUME_TRUTH'))
    return sorted(out,key=lambda p:-p['contrast_support']),contrast
