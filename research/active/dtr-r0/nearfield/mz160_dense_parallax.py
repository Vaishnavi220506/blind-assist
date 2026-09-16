"""Conditional dense displacement plus regional ToF scale; no private pose input.

Horizontal matching after IMU derotation is approximate for general motion.
Regional range-to-pixel correspondence and static anchors remain assumptions.
"""
import itertools
import math
import cv2
import numpy as np
from mz115_spatial_allocation import zone_box, slant_envelope, possible
from mz119_parallax import basis

METHOD=dict(schema='MZ160_DENSE_PUBLIC_DISPARITY_V1',lags=[2,3],
    matcher='SGBM_3WAY_SIGNED_CURRENT_MINUS_DEROTATED_PAST',min_disparity=-48,
    number_disparities=96,block_size=3,P1=72,P2=288,uniqueness_ratio=10,
    speckle_window=50,speckle_range=1,explicit_lr_error_px=1.,
    calibration='d=a/Z+b; independent single-valid current ToF zones; query-zone excluded',
    minimum_zones=8,inverse_depth_span=.03,minimum_scale_px_m=2.,
    residual_inlier_px=1.,minimum_inlier_fraction=.7,query_stride=8,
    range_interval_width_limit_m=1.,range_limit_m=4.,
    uncertainty='LEAVE_ONE_ZONE_COEFFICIENT_ENVELOPE_PLUS_1PX_NOT_CERTIFIED',
    assumptions=['STATIC_ANCHOR_SURFACES','REGIONAL_TOF_MATCHES_MEDIAN_RGB_DISPARITY',
                 'APPROXIMATE_HORIZONTAL_EPIPOLAR_GEOMETRY'],
    inference_inputs='PUBLIC_RGB_TOF_IMU_ONLY; Radar preserved in incumbent',
    zero_or_unsupported='UNKNOWN_NOT_CLEAR',backend_reason='GPU_BACKEND_UNAVAILABLE')


def matrix(intr):
    return np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]],float)


def dense_pair(row, past, gray, previous, yaw, past_yaw):
    intr=row['rgb_intrinsics'];h,w=gray.shape;K=matrix(intr)
    R=basis(row['camera_pitch_deg'],yaw).T@basis(past['camera_pitch_deg'],past_yaw)
    H=K@R@np.linalg.inv(K)
    warped=cv2.warpPerspective(previous,H,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
    border=cv2.warpPerspective(np.ones_like(previous),H,(w,h),flags=cv2.INTER_NEAREST,borderValue=0)
    border=cv2.erode(border,np.ones((5,5),np.uint8))>0
    def matcher():
        return cv2.StereoSGBM_create(minDisparity=-48,numDisparities=96,blockSize=3,
            P1=72,P2=288,disp12MaxDiff=1,preFilterCap=31,uniquenessRatio=10,
            speckleWindowSize=50,speckleRange=1,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d=matcher().compute(gray,warped).astype(np.float32)/16
    back=matcher().compute(warped,gray).astype(np.float32)/16
    yy,xx=np.indices(gray.shape);xr=xx-d
    mx=xr.astype(np.float32);my=yy.astype(np.float32)
    db=cv2.remap(back,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=-49)
    valid=cv2.remap(border.astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    valid &= (d>-48)&(d<47)&(db>-48)&(db<47)&(np.abs(d+db)<=1.)&(xr>=2)&(xr<w-2)
    valid[:2]=False;valid[-2:]=False
    return d,valid,dict(derotation_homography=H.tolist(),valid_disparity_pixels=int(valid.sum()))


def anchors(row, disparity, valid):
    if not row['tof_packet_received']:return []
    intr=row['rgb_intrinsics'];h,w=disparity.shape;result=[]
    for z in row['tof_zones']:
        if len(z['targets'])!=1:continue
        t=z['targets'][0];r=t['distance_m'];sigma=t['range_noise_sigma_m']
        if t['status']!='SIM_VALID' or not math.isfinite(r) or not math.isfinite(sigma) or sigma<0 or not .2<r or r+3*sigma>=4.:continue
        box=zone_box(z,intr)
        if not (2<=box[0]<box[2]<=w-3 and 2<=box[1]<box[3]<=h-3):continue
        l,u,rr,b=[int(v) for v in box];roi=valid[u:b+1,l:rr+1]
        if roi.sum()<20 or roi.mean()<.1:continue
        y,x=np.where(roi);x=x+l;y=y+u
        # Each zone contributes exactly one observation, not many independent pixels.
        u0=float(np.median(x));v0=float(np.median(y))
        raynorm=math.sqrt(1+((u0-intr['cx'])/intr['fx'])**2+((v0-intr['cy'])/intr['fy'])**2)
        result.append(dict(zone=int(z['zone_id']),inverse_depth=raynorm/r,
            disparity=float(np.median(disparity[y,x])),pixels=len(x),
            regional_range_m=r,range_sigma_m=sigma,pixel=[u0,v0]))
    return result


def calibration(records, excluded_zone=-1):
    aa=[a for a in records if a['zone']!=excluded_zone]
    if len(aa)<8:return None
    x=np.array([a['inverse_depth'] for a in aa]);y=np.array([a['disparity'] for a in aa])
    if np.ptp(x)<.03:return None
    pairs=list(itertools.combinations(range(len(x)),2))
    rng=np.random.default_rng(160016)
    if len(pairs)>64:pairs=[pairs[i] for i in rng.choice(len(pairs),64,replace=False)]
    best=None
    for i,j in pairs:
        if abs(x[i]-x[j])<.03:continue
        a=(y[i]-y[j])/(x[i]-x[j]);b=y[i]-a*x[i]
        residual=np.abs(y-a*x-b);good=residual<=1.
        key=(int(good.sum()),-float(np.median(residual[good])))
        if best is None or key>best[0]:best=(key,good)
    if best is None:return None
    good=best[1]
    if good.sum()<8 or good.mean()<.7 or np.ptp(x[good])<.03:return None
    xx=np.c_[x[good],np.ones(good.sum())];yy=y[good]
    estimates=[np.linalg.lstsq(xx,yy,rcond=None)[0]]
    for k in range(len(yy)):
        ix=np.arange(len(yy))!=k
        if np.ptp(xx[ix,0])<.03:return None
        estimates.append(np.linalg.lstsq(xx[ix],yy[ix],rcond=None)[0])
    values=np.array(estimates);lo=values.min(0);hi=values.max(0)
    if not np.isfinite(values).all() or lo[0]*hi[0]<=0 or np.min(np.abs(values[:,0]))<2:return None
    if hi[0]-lo[0]>.5*abs(values[0,0]):return None
    residual=np.abs(yy-xx@values[0])
    if np.quantile(residual,.9)>1.:return None
    return dict(a_bounds=lo[:1].tolist()+hi[:1].tolist(),b_bounds=[float(lo[1]),float(hi[1])],
        estimate=values[0].tolist(),disparity_error_px=1.,
        anchor_zones=[aa[i]['zone'] for i in np.flatnonzero(good)],
        excluded_zone=int(excluded_zone),residual_q90_px=float(np.quantile(residual,.9)),
        inverse_depth_span=float(np.ptp(x[good])))


def depth_bounds(d, model):
    if model is None or not math.isfinite(d):return None
    den=[d-model['disparity_error_px']-model['b_bounds'][1],
         d+model['disparity_error_px']-model['b_bounds'][0]]
    if den[0]*den[1]<=0:return None
    values=[a/q for a,q in itertools.product(model['a_bounds'],den)]
    if min(values)<=0 or not all(math.isfinite(z) for z in values):return None
    return [min(values),max(values)]


def query_zone(row,u,v):
    intr=row['rgb_intrinsics'];theta=math.degrees(math.atan((u-intr['cx'])/intr['fx']))
    phi=math.degrees(math.atan((intr['cy']-v)/intr['fy']))
    for z in row['tof_zones']:
        if z['theta_bounds_deg'][0]<=theta<z['theta_bounds_deg'][1] and z['phi_bounds_deg'][0]<=phi<z['phi_bounds_deg'][1]:return int(z['zone_id'])
    return -1


def predict(rows, loader, disparity_sink=None):
    output=[];history=[];episode=None;yaw=0.
    for i,row in enumerate(rows):
        if row['episode_id']!=episode or (history and abs(row['time_s']-history[-1]['row']['time_s']-.25)>1e-6):
            history=[];yaw=0.
        episode=row['episode_id']
        if not row['imu_valid']:history=[];yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        gray=cv2.cvtColor(loader(row),cv2.COLOR_BGR2GRAY)
        frame=dict(id=row['id'],points=[],pairs=[],state='UNKNOWN',integrated_yaw_deg=yaw)
        data=[]
        if row['imu_valid'] and len(history)>=3:
            for lag in (2,3):
                past=history[-lag]
                if not past['row']['imu_valid']:break
                d,valid,audit=dense_pair(row,past['row'],gray,past['gray'],yaw,past['yaw'])
                aa=anchors(row,d,valid);models={z:calibration(aa,z) for z in range(-1,64)}
                frame['pairs'].append(dict(reference_index=past['index'],anchors=aa,
                    models={str(k):v for k,v in models.items() if v},**audit))
                data.append((d,valid,models,past['index']))
                if disparity_sink is not None:disparity_sink(i,lag,d,valid)
        if len(data)==2:
            grad=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
            intr=row['rgb_intrinsics'];h,w=gray.shape
            for v in range(4,h-4,8):
                for u in range(4,w-4,8):
                    if abs(grad[v,u])<4 or not all(valid[v,u] for _,valid,_,_ in data):continue
                    zid=query_zone(row,u,v);bb=[depth_bounds(float(d[v,u]),models[zid]) for d,_,models,_ in data]
                    if any(b is None for b in bb):continue
                    bounds=[max(b[0] for b in bb),min(b[1] for b in bb)]
                    norm=math.sqrt(1+((u-intr['cx'])/intr['fx'])**2+((v-intr['cy'])/intr['fy'])**2)
                    ranges=[z*norm for z in bounds]
                    if not .2<ranges[0]<=ranges[1]<=4. or ranges[1]-ranges[0]>1.:continue
                    dy=.5+.2*row['time_s'];pitch=row['camera_pitch_deg']
                    xyz=slant_envelope([u-.5,v-.5,u+.5,v+.5],ranges,intr,
                        (pitch-.5,pitch+.5),(yaw-dy,yaw+dy),row['camera_in_body_m'][2])
                    frame['points'].append(dict(pixel=[u,v],range_bounds_m=ranges,
                        reference_indices=[p[3] for p in data],excluded_zone=zid,
                        alert_support=bool(possible(xyz,3.6)),localized_xyz=xyz))
            if frame['points']:frame['state']='CONDITIONAL_DENSE_GEOMETRY'
        output.append(frame);history.append(dict(row=row,gray=gray,yaw=yaw,index=i));history=history[-3:]
    return output
