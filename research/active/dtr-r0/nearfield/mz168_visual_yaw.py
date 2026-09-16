"""Causal public-RGB homography-decomposition yaw proposals, with IMU fallback.

No source geometry, evaluator truth, actor identity, translation estimate or
plane normal supplies a yaw output. Branch translations/normals are retained
only to audit calibrated decomposition and positive-depth filtering.
"""
import math

import cv2
import numpy as np


CONSTANTS = dict(seed=168016, max_corners=1200, quality_level=.01, min_distance=6,
    block_size=7, lk_window=[21,21], lk_max_level=3, lk_iterations=30,
    lk_epsilon=.01, forward_backward_max_px=.5, ransac_threshold_px=1.,
    ransac_max_iterations=2000, ransac_confidence=.995, min_tracks=40,
    min_inliers=40, min_inlier_fraction=.6, min_bbox_width_fraction=.4,
    min_bbox_height_fraction=.4, duplicate_rotation_deg=1e-6,
    rotation_orthogonality_tolerance=1e-6, rotation_determinant_tolerance=1e-6,
    max_expected_rotation_difference_deg=1., max_branch_yaw_spread_deg=.10,
    frame_step_s=.25, frame_step_tolerance_s=1e-6, max_reference_age_s=1.5)
METHOD = dict(schema='MZ168_CAUSAL_CALIBRATED_HOMOGRAPHY_VISUAL_YAW_V1',
    authority='PUBLIC_RGB_INTRINSICS_PITCH_AND_VALID_IMU_INCREMENTS_ONLY',
    backend='CPU_OPENCV',cpu_reason='GPU_BACKEND_UNAVAILABLE',
    homography_direction='REFERENCE_PIXEL_TO_CURRENT_PIXEL',
    reference='FIRST_EPISODE_FRAME_AT_EXACT_TIME_ZERO_FORWARD_ALIGNED_YAW_ZERO',
    branch_selection='POSITIVE_DEPTH_FILTER_THEN_ALL_DISTINCT_YAWS_AGREE_THEN_NEAREST_EXPECTED_SO3',
    output='YAW_OF_REFERENCE_BASIS_TIMES_SELECTED_ROTATION_TRANSPOSE',
    fallback='RAW_VALID_IMU_DELTA_INTEGRAL_WITHOUT_VISUAL_FEEDBACK',
    translation_and_normal='DECOMPOSITION_AUDIT_AND_VISIBILITY_ONLY_NEVER_OUTPUT_POSE',
    invalidation='TIME_GAP_OR_INVALID_IMU_UNTIL_NEXT_EPISODE',constants=CONSTANTS)


def cv_to_body(pitch_deg, yaw_deg):
    """Columns map camera right/down/forward to project body coordinates.

    Public project rays are [forward,right,up]; the pixel y axis points down.
    Both absolute bases use this same convention, so B_current.T @ B_reference
    is a proper source-camera to current-camera SO(3) rotation.
    """
    p,y=math.radians(float(pitch_deg)),math.radians(float(yaw_deg))
    cp,sp,cy,sy=math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[-sy,sp*cy,cp*cy],[cy,sp*sy,cp*sy],[0.,-cp,sp]],np.float64)


def _angle_deg(a,b):
    r=np.asarray(a)@np.asarray(b).T
    sine=np.linalg.norm([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    cosine=np.clip((np.trace(r)-1)/2,-1.,1.)
    return math.degrees(math.atan2(float(sine),float(cosine)))


def _unwrap_yaw(rotation, reference_basis, imu_yaw):
    forward=(reference_basis@rotation.T)[:,2]
    if not np.isfinite(forward).all() or np.linalg.norm(forward[:2])<1e-12:
        return None
    raw=math.degrees(math.atan2(float(forward[1]),float(forward[0])))
    return raw+360.*round((float(imu_yaw)-raw)/360.)


def _yaw_spread(values):
    if len(values)<=1:return 0.
    v=np.sort(np.mod(values,360.))
    gaps=np.diff(np.r_[v,v[0]+360.])
    return float(360.-gaps.max())


def _result(imu_yaw, reason, audit=None, yaw=None):
    value=dict(audit or {})
    value['reason']=reason
    accepted=reason=='ACCEPTED'
    return dict(yaw_deg=float(yaw if accepted else imu_yaw),imu_yaw_deg=float(imu_yaw),
                accepted=accepted,audit=value)


def _json_array(value):
    a=np.asarray(value,dtype=np.float64)
    result=a.astype(object)
    result[~np.isfinite(a)]=None
    return result.tolist()


def _calibration(intrinsics):
    values=[float(intrinsics[k]) for k in ('fx','fy','cx','cy')]
    h,w=int(intrinsics['height']),int(intrinsics['width'])
    if not np.isfinite(values).all() or min(values[:2])<=0 or min(h,w)<=0:
        raise ValueError('Finite positive public intrinsics and native image dimensions required')
    fx,fy,cx,cy=values
    return np.array([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]],np.float64),(h,w)


def _decompose_yaw(homography, k, reference_points, current_points,
                   pitch_ref, pitch_current, ref_yaw, imu_yaw):
    """Calibrated decomposition helper; point arguments are inlier pixels."""
    h=np.asarray(homography,np.float64)
    audit=dict(homography=_json_array(h),candidates=[],distinct_rotation_count=0,
               positive_depth_filter='NOT_RUN',best_difference_deg=None,
               second_difference_deg=None,best_second_margin_deg=None,
               surviving_branch_yaw_spread_deg=None)
    if h.shape!=(3,3) or not np.isfinite(h).all():
        return _result(imu_yaw,'INVALID_HOMOGRAPHY',audit)
    expected=cv_to_body(pitch_current,imu_yaw).T@cv_to_body(pitch_ref,ref_yaw)
    ref_basis=cv_to_body(pitch_ref,ref_yaw)
    audit['expected_rotation']=expected.tolist()
    try:
        count,rotations,translations,normals=cv2.decomposeHomographyMat(h,k)
    except cv2.error as exc:
        audit['opencv_error']=str(exc)
        return _result(imu_yaw,'DECOMPOSITION_FAILED',audit)
    if not count:return _result(imu_yaw,'NO_DECOMPOSITION',audit)
    # Validate OpenCV's output, including its near-pure shortcut. That shortcut
    # can return normalized H instead of an orthogonal rotation when translation
    # is small. Do not silently replace that matrix with a polar projection.
    invalid=[]
    for index,(r,t,n) in enumerate(zip(rotations,translations,normals)):
        finite=bool(np.isfinite(r).all() and np.isfinite(t).all() and np.isfinite(n).all())
        error=float(np.linalg.norm(np.asarray(r).T@r-np.eye(3),'fro')) if finite else None
        determinant=float(np.linalg.det(r)) if finite else None
        so3=bool(finite and error<=CONSTANTS['rotation_orthogonality_tolerance']
                 and abs(determinant-1.)<=CONSTANTS['rotation_determinant_tolerance'])
        if not so3:invalid.append(index)
        audit['candidates'].append(dict(index=index,rotation=_json_array(r),
            translation_over_plane_distance=_json_array(np.asarray(t).reshape(-1)),
            plane_normal=_json_array(np.asarray(n).reshape(-1)),finite=finite,
            orthogonality_frobenius_error=error,determinant=determinant,valid_so3=so3,
            positive_depth_survivor=None,expected_difference_deg=_angle_deg(r,expected) if so3 else None,
            yaw_deg=_unwrap_yaw(r,ref_basis,imu_yaw) if so3 else None,rotation_group=None))
    audit['invalid_rotation_indices']=invalid
    if invalid:return _result(imu_yaw,'INVALID_ROTATION_SOLUTION',audit)
    # Exact pure rotation has no plane or translation; a zero plane normal does
    # not define a positive-depth half-space for OpenCV's filter.
    pure=all(np.linalg.norm(t)<1e-12 and np.linalg.norm(n)<1e-12
             for t,n in zip(translations,normals))
    surviving=list(range(count))
    if pure:
        audit['positive_depth_filter']='NOT_APPLICABLE_PURE_ROTATION_ZERO_NORMAL'
    elif callable(getattr(cv2,'filterHomographyDecompByVisibleRefpoints',None)):
        try:
            before=cv2.undistortPoints(np.asarray(reference_points,np.float32).reshape(-1,1,2),k,None)
            after=cv2.undistortPoints(np.asarray(current_points,np.float32).reshape(-1,1,2),k,None)
            possible=cv2.filterHomographyDecompByVisibleRefpoints(rotations,normals,before,after)
            surviving=[] if possible is None else np.asarray(possible).reshape(-1).astype(int).tolist()
            audit['positive_depth_filter']='NORMALIZED_VISIBLE_REFERENCE_POINTS'
        except cv2.error as exc:
            audit['positive_depth_filter']='FAILED_CLOSED'
            audit['positive_depth_filter_error']=str(exc)
            return _result(imu_yaw,'POSITIVE_DEPTH_FILTER_FAILED',audit)
    else:
        audit['positive_depth_filter']='API_UNAVAILABLE_FAILED_CLOSED'
        return _result(imu_yaw,'POSITIVE_DEPTH_FILTER_UNAVAILABLE',audit)
    audit['surviving_solution_indices']=surviving
    groups=[]
    for index,(r,t,n) in enumerate(zip(rotations,translations,normals)):
        item=audit['candidates'][index]
        item['positive_depth_survivor']=index in surviving
        difference,yaw=item['expected_difference_deg'],item['yaw_deg']
        if index in surviving and yaw is not None:
            group=next((j for j,g in enumerate(groups)
                        if _angle_deg(r,g['rotation'])<=CONSTANTS['duplicate_rotation_deg']),None)
            if group is None:
                group=len(groups)
                groups.append(dict(rotation=r,index=index,difference=difference,yaw=yaw))
            item['rotation_group']=group
    audit['distinct_rotation_count']=len(groups)
    if not groups:return _result(imu_yaw,'NO_VISIBLE_ROTATION_SOLUTION',audit)
    ranked=sorted(groups,key=lambda g:(g['difference'],g['index']))
    best=ranked[0]
    audit['best_solution_index']=best['index']
    audit['best_difference_deg']=best['difference']
    if len(ranked)>1:
        audit['second_difference_deg']=ranked[1]['difference']
        audit['best_second_margin_deg']=ranked[1]['difference']-best['difference']
    spread=_yaw_spread([g['yaw'] for g in groups])
    audit['surviving_branch_yaw_spread_deg']=spread
    audit['best_within_expected_tolerance']=best['difference']<=CONSTANTS['max_expected_rotation_difference_deg']
    audit['surviving_branches_agree_in_yaw']=spread<=CONSTANTS['max_branch_yaw_spread_deg']
    if not audit['best_within_expected_tolerance']:
        return _result(imu_yaw,'ROTATION_DISAGREES_WITH_IMU',audit)
    if not audit['surviving_branches_agree_in_yaw']:
        return _result(imu_yaw,'AMBIGUOUS_YAW_BRANCHES',audit)
    return _result(imu_yaw,'ACCEPTED',audit,best['yaw'])


def _inside(points, shape):
    h,w=shape
    return np.isfinite(points).all(axis=1)&(points[:,0]>=0)&(points[:,0]<=w-1)&(points[:,1]>=0)&(points[:,1]<=h-1)


def estimate_pair(reference,current,intrinsics,pitch_ref,pitch_current,ref_yaw,imu_yaw):
    """Reference/current are native uint8 BGR images; no metadata is accepted.

    RANSAC fits reference-to-current pixel homography. Decomposition uses the
    public K, and every surviving distinct rotation must agree in yaw <= .10
    degrees. The closest expected rotation must be within one degree. Failure
    returns the supplied raw IMU integral without updating any reference.
    """
    k,shape=_calibration(intrinsics)
    if not all(math.isfinite(float(v)) for v in (pitch_ref,pitch_current,ref_yaw,imu_yaw)):
        raise ValueError('Finite public pitch and yaw required')
    if any(not isinstance(im,np.ndarray) or im.dtype!=np.uint8 or im.shape!=(*shape,3)
           for im in (reference,current)):
        raise ValueError('Native-resolution uint8 BGR pair required')
    gray_ref=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
    gray_cur=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    audit=dict(authority='PUBLIC_RGB_INTRINSICS_PITCH_AND_IMU_ONLY',opencv=cv2.__version__,
        detected_points=0,forward_valid_points=0,fb_valid_points=0,inliers=0,
        inlier_fraction=0.,point_sample=[],homography=None)
    points=cv2.goodFeaturesToTrack(gray_ref,maxCorners=CONSTANTS['max_corners'],
        qualityLevel=CONSTANTS['quality_level'],minDistance=CONSTANTS['min_distance'],
        blockSize=CONSTANTS['block_size'])
    audit['detected_points']=0 if points is None else len(points)
    if points is None or len(points)<CONSTANTS['min_tracks']:
        return _result(imu_yaw,'INSUFFICIENT_CORNERS',audit)
    lk=dict(winSize=tuple(CONSTANTS['lk_window']),maxLevel=CONSTANTS['lk_max_level'],
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,
                  CONSTANTS['lk_iterations'],CONSTANTS['lk_epsilon']))
    forward,status,_=cv2.calcOpticalFlowPyrLK(gray_ref,gray_cur,points,None,**lk)
    if forward is None or status is None:return _result(imu_yaw,'FORWARD_TRACK_FAILED',audit)
    p0=points.reshape(-1,2);p1=forward.reshape(-1,2)
    good=status.reshape(-1).astype(bool)&_inside(p0,shape)&_inside(p1,shape)
    p0,p1=p0[good],p1[good];audit['forward_valid_points']=len(p0)
    if len(p0)<CONSTANTS['min_tracks']:return _result(imu_yaw,'INSUFFICIENT_FORWARD_TRACKS',audit)
    back,status,_=cv2.calcOpticalFlowPyrLK(gray_cur,gray_ref,p1.reshape(-1,1,2),None,**lk)
    if back is None or status is None:return _result(imu_yaw,'BACKWARD_TRACK_FAILED',audit)
    back=back.reshape(-1,2)
    fb=np.linalg.norm(back-p0,axis=1)
    good=status.reshape(-1).astype(bool)&_inside(back,shape)&np.isfinite(fb)&(fb<=CONSTANTS['forward_backward_max_px'])
    p0,p1,fb=p0[good],p1[good],fb[good];audit['fb_valid_points']=len(p0)
    if len(p0)<CONSTANTS['min_tracks']:return _result(imu_yaw,'INSUFFICIENT_FB_TRACKS',audit)
    cv2.setRNGSeed(CONSTANTS['seed'])
    h,inlier_mask=cv2.findHomography(p0,p1,cv2.RANSAC,CONSTANTS['ransac_threshold_px'],
        maxIters=CONSTANTS['ransac_max_iterations'],confidence=CONSTANTS['ransac_confidence'])
    if h is None or inlier_mask is None:return _result(imu_yaw,'HOMOGRAPHY_FAILED',audit)
    inliers=inlier_mask.reshape(-1).astype(bool)
    audit.update(homography=h.tolist(),inliers=int(inliers.sum()),inlier_fraction=float(inliers.mean()))
    selected=np.linspace(0,len(p0)-1,min(64,len(p0)),dtype=int)
    audit['point_sample']=[dict(reference=p0[i].tolist(),current=p1[i].tolist(),
        fb_error_px=float(fb[i]),inlier=bool(inliers[i])) for i in selected]
    if audit['inliers']<CONSTANTS['min_inliers']:return _result(imu_yaw,'INSUFFICIENT_INLIERS',audit)
    if audit['inlier_fraction']<CONSTANTS['min_inlier_fraction']:return _result(imu_yaw,'LOW_INLIER_FRACTION',audit)
    spans={name:(np.ptp(p[inliers],axis=0)/np.array([shape[1],shape[0]])).tolist()
           for name,p in (('reference',p0),('current',p1))}
    audit['inlier_bbox_fraction']=spans
    if any(s[0]<CONSTANTS['min_bbox_width_fraction'] or s[1]<CONSTANTS['min_bbox_height_fraction'] for s in spans.values()):
        return _result(imu_yaw,'INSUFFICIENT_INLIER_SPREAD',audit)
    result=_decompose_yaw(h,k,p0[inliers],p1[inliers],pitch_ref,pitch_current,ref_yaw,imu_yaw)
    result['audit']={**audit,**result['audit']}
    return result


class CausalVisualYaw:
    """One time-zero anchor per episode; visual proposals never feed the IMU.

    Uses only episode_id, time_s, imu_valid, delta_yaw, rgb_intrinsics and
    camera_pitch_deg from the public row. Time-zero anchor reference yaw is
    fixed at zero. Raw valid deltas are integrated even when a proposal fails.
    A gap or invalid IMU invalidates the reference until the next episode; a nonzero-time
    start never creates an anchor. All input arrays and row fields are read-only.
    """
    def __init__(self):
        self.episode=None;self.previous_time=None;self.imu_yaw=0.
        self.anchor=None;self.anchor_pitch=None;self.anchor_intrinsics=None
        self.anchor_unavailable_reason='NO_EPISODE'

    def update(self,public_row,image):
        episode=public_row['episode_id'];time_s=float(public_row['time_s'])
        delta=float(public_row['delta_yaw']);pitch=float(public_row['camera_pitch_deg'])
        imu_valid=bool(public_row['imu_valid'])
        k,shape=_calibration(public_row['rgb_intrinsics'])
        if not math.isfinite(time_s) or not math.isfinite(pitch) or (imu_valid and not math.isfinite(delta)):
            raise ValueError('Finite public time/pitch and valid IMU delta required')
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.shape!=(*shape,3):
            raise ValueError('Native-resolution uint8 BGR image required')
        new=episode!=self.episode or self.previous_time is None
        if new:
            self.episode=episode;self.previous_time=None;self.imu_yaw=0.
            self.anchor=None;self.anchor_pitch=None;self.anchor_intrinsics=None
            self.anchor_unavailable_reason='NONZERO_TIME_EPISODE_START'
        if imu_valid:self.imu_yaw+=delta
        dt=None if self.previous_time is None else time_s-self.previous_time
        audit=dict(time_s=time_s,imu_valid=imu_valid,raw_delta_yaw_deg=delta if math.isfinite(delta) else None,
            imu_yaw_deg=float(self.imu_yaw),reference_time_s=0.,reference_yaw_deg=0.,
            step_s=dt,new_episode=new,visual_feedback_into_imu=False)
        if not imu_valid:
            self.anchor=None;self.anchor_unavailable_reason='INVALID_IMU_INVALIDATED_ANCHOR'
            result=_result(self.imu_yaw,self.anchor_unavailable_reason,audit)
        elif new:
            if time_s==0.:
                self.anchor=image.copy();self.anchor_pitch=pitch
                self.anchor_intrinsics={key:public_row['rgb_intrinsics'][key] for key in ('width','height','fx','fy','cx','cy')}
                self.anchor_unavailable_reason=None
                result=_result(self.imu_yaw,'ANCHOR_ESTABLISHED',audit)
            else:result=_result(self.imu_yaw,self.anchor_unavailable_reason,audit)
        elif abs(dt-CONSTANTS['frame_step_s'])>CONSTANTS['frame_step_tolerance_s']:
            self.anchor=None;self.anchor_unavailable_reason='TIME_GAP_INVALIDATED_ANCHOR'
            result=_result(self.imu_yaw,self.anchor_unavailable_reason,audit)
        elif self.anchor is None:
            result=_result(self.imu_yaw,self.anchor_unavailable_reason,audit)
        elif time_s>CONSTANTS['max_reference_age_s']:
            self.anchor=None;self.anchor_unavailable_reason='REFERENCE_EXPIRED'
            result=_result(self.imu_yaw,self.anchor_unavailable_reason,audit)
        elif any(public_row['rgb_intrinsics'][key]!=value for key,value in self.anchor_intrinsics.items()):
            self.anchor=None;self.anchor_unavailable_reason='INTRINSICS_CHANGED'
            result=_result(self.imu_yaw,self.anchor_unavailable_reason,audit)
        else:
            result=estimate_pair(self.anchor,image,self.anchor_intrinsics,self.anchor_pitch,pitch,0.,self.imu_yaw)
            result['audit']={**audit,**result['audit']}
        self.previous_time=time_s
        return result
