"""Bounded CPU line/stripe geometry under supplied ideal metric poses.

No dense-depth network, native reference or object specification is accepted.
One optical-depth hypothesis per straight segment is a local frontoparallel
approximation, not an arbitrary 3D curve model. Scores are not probabilities.
"""
import math

import cv2
import numpy as np

CONFIG = dict(max_lines=128, tile_columns=8, tile_rows=4, lines_per_tile=4,
              minimum_length_px=8., points_per_line=17, strip_radius_px=4,
              inverse_min=.05, inverse_max=2., hypotheses=96,
              minimum_score=.75, minimum_margin=.05, alternative_fraction=.2,
              plausible_score_drop=.03, maximum_relative_inverse_width=.5,
              minimum_parallax_px=1., minimum_supported_points=13,
              algorithm_p95_budget_ms=200., peak_rss_budget_mib=256.)


def pose_matrix(pose):
    if pose.get('roll', 0) != 0:
        raise ValueError('Frozen probe supports roll-zero poses')
    p, y = math.radians(pose['pitch']), math.radians(pose['yaw'])
    cp, sp, cy, sy = math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[-sy, sp*cy, cp*cy], [cy, sp*sy, cp*sy], [0., -cp, sp]]), np.array([pose[k] for k in ('x','y','z')])


def project(pixels, inverse_depth, source_pose, endpoint_pose, calibration):
    """Project endpoint pixels to a past camera, H hypotheses x N points."""
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    inv = np.asarray(inverse_depth, np.float64).reshape(-1)
    rs, ts = pose_matrix(source_pose); rt, tt = pose_matrix(endpoint_pose)
    rays = np.column_stack(((pixels[:,0]-calibration['cx'])/calibration['fx'],
                            (pixels[:,1]-calibration['cy'])/calibration['fy'], np.ones(len(pixels))))
    direction = rays @ (rs.T @ rt).T
    xyz = direction[None,:,:]/inv[:,None,None]+(rs.T @ (tt-ts))[None,None,:]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = xyz[:,:,:2]/xyz[:,:,2,None]
    uv[:,:,0] = uv[:,:,0]*calibration['fx']+calibration['cx']
    uv[:,:,1] = uv[:,:,1]*calibration['fy']+calibration['cy']
    return uv, xyz[:,:,2]


def sample(gray, coordinates):
    shape = coordinates.shape[:-1]
    xy = np.asarray(coordinates, np.float32).reshape(-1,2)
    # The small fixed per-line sample count fits OpenCV's remap dimensions.
    return cv2.remap(gray, xy[:,0].reshape(-1,1), xy[:,1].reshape(-1,1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(shape)


def ncc(a, b):
    a = a-a.mean(axis=-1,keepdims=True); b = b-b.mean(axis=-1,keepdims=True)
    den = np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    return np.divide((a*b).sum(-1),den,out=np.full(den.shape,-1.,np.float32),where=den>1e-6)


def extract_lines(gray):
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 1.).detect(np.clip(gray*255,0,255).astype(np.uint8))[0]
    if detected is None:
        return [], dict(detected=0, eligible=0, selected=0)
    h,w = gray.shape; bins = {}; eligible=0
    for index, values in enumerate(detected.reshape(-1,4)):
        ends = values.reshape(2,2).astype(np.float64)
        # Canonical endpoint order removes detector direction from tie breaking.
        if tuple(ends[1]) < tuple(ends[0]): ends = ends[::-1].copy()
        vector = ends[1]-ends[0]; length=float(np.linalg.norm(vector))
        if length < CONFIG['minimum_length_px'] or (ends[:,0]<5).any() or (ends[:,0]>w-6).any() or (ends[:,1]<5).any() or (ends[:,1]>h-6).any():
            continue
        eligible+=1; middle=ends.mean(0)
        key=(min(int(middle[1]*CONFIG['tile_rows']/h),CONFIG['tile_rows']-1),
             min(int(middle[0]*CONFIG['tile_columns']/w),CONFIG['tile_columns']-1))
        bins.setdefault(key,[]).append((length,tuple(ends.ravel()),index,ends))
    selected=[]
    for key in sorted(bins):
        for length,_,_,ends in sorted(bins[key],key=lambda row:(-row[0],row[1],row[2]))[:CONFIG['lines_per_tile']]:
            selected.append((ends,length))
    assert len(selected)<=CONFIG['max_lines']
    return selected,dict(detected=len(detected),eligible=eligible,selected=len(selected))


def match_line(gray, poses, calibration, ends, length):
    count=CONFIG['points_per_line']; radius=CONFIG['strip_radius_px']
    along=(ends[1]-ends[0])/length; normal=np.array([-along[1],along[0]])
    points=ends[0]+np.linspace(0,1,count)[:,None]*(ends[1]-ends[0])
    offsets=np.arange(-radius,radius+1); width=len(offsets)
    stripe=points[:,None,:]+offsets[None,:,None]*normal
    # Endpoint patches retain along-contour evidence; long straight interiors
    # alone cannot resolve motion parallel to the edge (aperture ambiguity).
    oy,ox=np.meshgrid(offsets,offsets,indexing='ij')
    endcoords=ends[:,None,:]+ox.ravel()[None,:,None]*along+oy.ravel()[None,:,None]*normal
    coordinates=np.concatenate([stripe.reshape(-1,2),endcoords.reshape(-1,2)])
    reference=sample(gray[-1],coordinates)
    ref_stripe=reference[:count*width].reshape(1,count,width)
    ref_ends=reference[count*width:].reshape(1,2,width*width)
    inv=np.linspace(CONFIG['inverse_min'],CONFIG['inverse_max'],CONFIG['hypotheses'])
    all_score=[]; profiles=[]; normal_displacement=[]; endpoint_displacement=[]
    for image,pose in zip(gray[:2],poses[:2]):
        uv,z=project(coordinates,inv,pose,poses[-1],calibration)
        good=(z>0)&np.isfinite(uv).all(-1)&(uv[:,:,0]>=0)&(uv[:,:,0]<image.shape[1]-1)&(uv[:,:,1]>=0)&(uv[:,:,1]<image.shape[0]-1)
        samples=sample(image,uv)
        profile=ncc(ref_stripe,samples[:,:count*width].reshape(len(inv),count,width))
        endpoints=ncc(ref_ends,samples[:,count*width:].reshape(len(inv),2,width*width))
        # Both endpoints and the distributed profiles must explain one depth.
        score=np.minimum(profile.mean(-1),endpoints.min(-1))
        score[~good.all(-1)]=-1.
        all_score.append(score);profiles.append(profile)
        point_uv,_=project(points,inv,pose,poses[-1],calibration)
        rotation_uv,_=project(points,[1e-8],pose,poses[-1],calibration)
        tangent=point_uv[:,-1,:]-point_uv[:,0,:]
        unit_normal=np.column_stack([-tangent[:,1],tangent[:,0]])/np.maximum(np.linalg.norm(tangent,axis=-1)[:,None],1e-12)
        displacement=point_uv-rotation_uv
        normal_displacement.append(np.median(np.abs((displacement*unit_normal[:,None,:]).sum(-1)),axis=-1))
        endpoint_displacement.append(np.min(np.linalg.norm(displacement[:,[0,-1],:],axis=-1),axis=-1))
    scores=np.minimum(*all_score);best=int(np.argmax(scores));value=float(scores[best]);best_inv=inv[best]
    outside=np.abs(inv-best_inv)>CONFIG['alternative_fraction']*best_inv
    margin=value-float(scores[outside].max())
    plausible=scores>=value-CONFIG['plausible_score_drop']
    low,high=float(inv[plausible].min()),float(inv[plausible].max())
    # Include half-bin discretization uncertainty rather than report a point.
    step=float(inv[1]-inv[0]);low=max(CONFIG['inverse_min'],low-step/2);high=min(CONFIG['inverse_max'],high+step/2)
    normal_px=float(normal_displacement[0][best]);end_px=float(endpoint_displacement[0][best])
    support=np.minimum(profiles[0][best],profiles[1][best])>=CONFIG['minimum_score']
    baseline=float(np.linalg.norm(pose_matrix(poses[-1])[1]-pose_matrix(poses[0])[1]))
    checks=dict(metric_baseline=baseline>1e-6,correlation=value>=CONFIG['minimum_score'],
                unique=margin>=CONFIG['minimum_margin'],bounded_interval=(high-low)<=CONFIG['maximum_relative_inverse_width']*best_inv,
                parallax=max(normal_px,end_px)>=CONFIG['minimum_parallax_px'],
                distributed_support=int(support.sum())>=CONFIG['minimum_supported_points'],
                interior_grid=0<best<len(inv)-1)
    accepted=all(checks.values())
    return dict(endpoints=ends.tolist(),length_px=length,points=points.tolist(),normal=normal.tolist(),
                depth_m=float(1/best_inv),interval_m=[1/high,1/low],accepted=accepted,
                point_supported=support.tolist(),score=value,margin=margin,
                normal_parallax_px=normal_px,endpoint_parallax_px=end_px,baseline_m=baseline,
                reason='ACCEPTED' if accepted else ','.join(k for k,v in checks.items() if not v),checks=checks,
                score_curve=scores.tolist())


def match(images,poses,calibration):
    if len(images)!=3 or len(poses)!=3:
        raise ValueError('Exactly three chronological frames required')
    if any(image.shape!=(calibration['height'],calibration['width'],3) or image.dtype!=np.uint8 for image in images):
        raise ValueError('Calibrated BGR uint8 frames required')
    gray=[image.mean(-1).astype(np.float32)/255. for image in images]
    selected,counts=extract_lines(gray[-1])
    lines=[dict(id=i,**match_line(gray,poses,calibration,ends,length)) for i,(ends,length) in enumerate(selected)]
    return dict(lines=lines,candidate_count=len(lines),selection=counts,configuration=CONFIG,
                uses_ideal_metric_poses=True,model_calls=0,uses_native_depth=False)
