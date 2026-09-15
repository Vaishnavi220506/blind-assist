"""Causal image-alignment residuals, without metric pose or ToF ownership.

A homography describes a dominant image layer only. Residuals can also come
from occlusion, illumination, moving objects or alignment error. The learner
receives validity/quality markers; this module never emits clearance or alarms.
"""
import cv2
import numpy as np

LAGS=(1,2,4)
GRID=(8,16)
CHANNELS=('valid_fraction','abs_mean','abs_q95','abs_max','column_abs_max',
          'positive_fraction','negative_fraction','edge_mean')
METHOD=dict(schema='MZ148_CAUSAL_BACKGROUND_RESIDUAL_V1',lags=list(LAGS),dt_s=.25,
 registration='320x180 ECC homography, 50 iterations, eps1e-5, Gaussian5',
 registration_authority='DOMINANT_IMAGE_LAYER_NOT_METRIC_CAMERA_POSE',
 grid=list(GRID),residual='SIGNED_CURRENT_MINUS_ALIGNED_REFERENCE_MEDIAN_BIAS_REMOVED',
 features='CURRENT_COORDINATE_CELL_AND_COLUMN_SUMMARIES',native_inputs=False,
 no_tof_pruning=True,backend_reason='GPU_BACKEND_UNAVAILABLE')


def align_pair(current,reference):
    h,w=current.shape
    if reference.shape!=(h,w):raise ValueError('Matching native image dimensions required')
    small_current=cv2.resize(current,(320,180),interpolation=cv2.INTER_AREA)
    small_reference=cv2.resize(reference,(320,180),interpolation=cv2.INTER_AREA)
    transform=np.eye(3,dtype=np.float32)
    try:
        correlation,transform=cv2.findTransformECC(small_current,small_reference,transform,
            cv2.MOTION_HOMOGRAPHY,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-5),None,5)
    except cv2.error:
        return None,None,dict(valid=False,reason='ECC_NOT_CONVERGED')
    scale=np.diag([320/w,180/h,1.]);transform=np.linalg.inv(scale)@transform@scale
    if not np.isfinite(transform).all() or not np.isfinite(correlation):
        return None,None,dict(valid=False,reason='NONFINITE_HOMOGRAPHY')
    warped=cv2.warpPerspective(reference,transform,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    valid=cv2.warpPerspective(np.ones_like(reference),transform,(w,h),
        flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>0
    overlap=float(valid.mean())
    accepted=bool(correlation>=.8 and overlap>=.6)
    audit=dict(valid=accepted,correlation=float(correlation),overlap=overlap,
        current_to_reference_homography=transform.tolist(),reason='IMAGE_ALIGNMENT_ONLY')
    return (warped,valid,audit) if accepted else (None,None,audit)


def summarize(current,reference,valid):
    h,w=current.shape;values=[]
    if valid is None or not valid.any():return np.zeros(GRID[0]*GRID[1]*len(CHANNELS),np.float32)
    residual=current-reference
    residual=residual-float(np.median(residual[valid]))
    noise=max(1/255,float(np.median(np.abs(residual[valid])))*1.4826)
    edge=np.abs(cv2.Scharr(residual,cv2.CV_32F,1,0)/32.)
    for ys,ye in zip(np.linspace(0,h,GRID[0]+1,dtype=int)[:-1],np.linspace(0,h,GRID[0]+1,dtype=int)[1:]):
        for xs,xe in zip(np.linspace(0,w,GRID[1]+1,dtype=int)[:-1],np.linspace(0,w,GRID[1]+1,dtype=int)[1:]):
            mask=valid[ys:ye,xs:xe];r=residual[ys:ye,xs:xe];a=np.abs(r)
            if not mask.any():values.extend([0.]*len(CHANNELS));continue
            n=mask.sum(0);columns=n>0
            column=(np.where(mask,a,0).sum(0)/np.maximum(1,n))[columns]
            values.extend([float(mask.mean()),float(a[mask].mean()),float(np.quantile(a[mask],.95)),
                float(a[mask].max()),float(column.max()),float((r[mask]>3*noise).mean()),
                float((r[mask]< -3*noise).mean()),float(edge[ys:ye,xs:xe][mask].mean())])
    return np.asarray(values,np.float32)


class CausalResidual:
    def __init__(self):self.history=[];self.episode=None;self.time=None

    def update(self,row,image):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Native uint8 BGR image required')
        if row['episode_id']!=self.episode or self.time is None or abs(row['time_s']-self.time-.25)>1e-6:
            self.history=[]
        self.episode=row['episode_id'];self.time=row['time_s']
        current=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        compensated=[];unregistered=[];audit=[]
        for lag in LAGS:
            available=len(self.history)>=lag
            if available:
                reference=self.history[-lag]
                if reference.shape!=current.shape:raise ValueError('Resolution changed within episode')
                warped,valid,a=align_pair(current,reference)
                comp=summarize(current,warped,valid) if a['valid'] else summarize(current,None,None)
                raw=summarize(current,reference,np.ones_like(current,dtype=bool))
                comp_head=[1.,float(a['valid']),a.get('correlation',0.),a.get('overlap',0.)]
                raw_head=[1.,1.,0.,1.]
            else:
                a=dict(valid=False,reason='NO_CAUSAL_REFERENCE');comp=summarize(current,None,None);raw=comp.copy()
                comp_head=raw_head=[0.,0.,0.,0.]
            compensated.extend(comp_head);compensated.extend(comp)
            unregistered.extend(raw_head);unregistered.extend(raw)
            audit.append(dict(lag_frames=lag,reference_available=available,**a))
        self.history.append(current.copy());self.history=self.history[-max(LAGS):]
        c=np.asarray(compensated,np.float32);u=np.asarray(unregistered,np.float32)
        assert c.shape==u.shape==(3084,) and np.isfinite(c).all() and np.isfinite(u).all()
        return dict(compensated=c,unregistered=u,audit=audit)
