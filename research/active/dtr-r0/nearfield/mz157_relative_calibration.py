"""Frozen relative geometry with public regional range calibration hypotheses.

Uniform pixel-mode arithmetic ranges approximate regional measurements; unknown
reflectance and sparse native ray sampling prevent exact return ownership claims.
"""
import math
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from mz115_spatial_allocation import zone_box

METHOD=dict(schema='MZ157_FROZEN_RELATIVE_REGIONAL_CALIBRATION_V1',input_size=518,
    model='ORIGINAL_DEPTHOR_CHECKPOINT_EMBEDDED_FROZEN_DA2_SMALL_239_KEYS',
    geometry='POSITIVE_AFFINE_CAMERA_INVERSE_DEPTH',range_proxy='ARITHMETIC_MEAN_SLANT_OF_UNIFORMLY_SAMPLED_PIXEL_MODE',
    mode_samples=32,kmeans_iterations=8,assignment_iterations=6,max_nfev=35,
    parameter_bounds=[[0.,.05],[100.,100.]],minimum_zones=6,minimum_jacobian_ratio=1e-3,
    huber_scale_m=.12,holdout='zone_id modulo4 equals0; diagnostic only, final fit uses all eligible zones',
    unknown='MISSING_MERGED_PARTIAL_RGB_OR_UNIDENTIFIABLE_IS_NOT_CLEAR',
    hypothesis='MODE_CORRESPONDENCE_NOT_NATIVE_IDENTITY_OR_CERTIFIED_SURFACE')


def load_prior(upstream,checkpoint):
    import torch
    sys.path.insert(0,str(Path(upstream).resolve()))
    from src.models.depth_anything_v2.dpt import DepthAnythingV2
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state={k[len('depth_anything.'):]:v for k,v in weights.items() if k.startswith('depth_anything.')}
    assert len(state)==239
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(state,strict=True);model.requires_grad_(False);model=model.cuda().eval()
    return model,dict(strict_keys=len(state),parameters=sum(p.numel() for p in model.parameters()),
        device=str(next(model.parameters()).device),weights_frozen=not any(p.requires_grad for p in model.parameters()))


def relative_prediction(model,bgr):
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        tensor,(h,w)=model.image2tensor(bgr,518)
        assert tensor.is_cuda
        output=model(tensor)
        assert isinstance(output,tuple) and len(output)==2
        relative=F.interpolate(output[1][:,None],(h,w),mode='bilinear',align_corners=True)[0,0]
        assert torch.isfinite(relative).all()
        return relative.cpu().numpy(),dict(network_shape=list(tensor.shape),dtype=str(tensor.dtype))


def pixel_modes(d,n):
    centers=np.quantile(d,[.1,.9]);labels=np.zeros(len(d),int)
    for _ in range(8):
        labels=np.argmin(np.abs(d[:,None]-centers[None]),axis=1)
        if len(np.unique(labels))<2:break
        centers=np.array([d[labels==i].mean() for i in range(2)])
    groups=[np.flatnonzero(labels==i) for i in range(2)]
    groups=[g for g in groups if len(g)]
    groups.sort(key=lambda g:float(d[g].mean()))
    if len(groups)==2 and abs(d[groups[1]].mean()-d[groups[0]].mean())<1e-5:groups=[np.arange(len(d))]
    result=[]
    for group in groups:
        ordered=group[np.argsort(d[group],kind='stable')]
        chosen=ordered[np.linspace(0,len(ordered)-1,32,dtype=int)]
        result.append((np.stack([d[chosen],n[chosen]],axis=1),len(group)))
    return result


def anchors(row,relative,arm):
    assert arm in ('median','regional_modes')
    rel=np.asarray(relative,float);intr=row['rgb_intrinsics'];h,w=rel.shape
    assert (h,w)==(intr['height'],intr['width']) and np.isfinite(rel).all()
    spread=float(np.ptp(rel));audit=dict(excluded=[],included=[],relative_min=float(rel.min()),relative_span=spread)
    if spread<1e-6:return None,[],audit
    d=(rel-rel.min())/spread
    yy,xx=np.mgrid[:h,:w];norm=np.sqrt(1+((xx-intr['cx'])/intr['fx'])**2+((intr['cy']-yy)/intr['fy'])**2)
    groups=[]
    for zone in row['tof_zones']:
        zid=zone['zone_id'];l,t,r,b=zone_box(zone,intr)
        reason=None
        if not row['tof_packet_received']:reason='MISSING_PACKET'
        elif not (0<=l<r<=w-1 and 0<=t<b<=h-1):reason='INCOMPLETE_RGB_ZONE_KEEP_RAW_SUPPORT'
        targets=[(i,v) for i,v in enumerate(zone['targets']) if v['status']=='SIM_VALID'
            and math.isfinite(v['distance_m']) and v['distance_m']>0]
        if not targets:reason=reason or 'NO_VALID_RETURN'
        if reason:audit['excluded'].append(dict(zone=zid,reason=reason));continue
        ys=slice(math.ceil(t),math.floor(b)+1);xs=slice(math.ceil(l),math.floor(r)+1)
        dv=d[ys,xs].ravel();nv=norm[ys,xs].ravel()
        if len(dv)<2:audit['excluded'].append(dict(zone=zid,reason='EMPTY_RASTER_ZONE'));continue
        if arm=='median':
            ix=np.linspace(0,len(nv)-1,32,dtype=int)
            modes=[(np.stack([np.full(32,np.median(dv)),nv[ix]],axis=1),len(dv))]
        else:modes=pixel_modes(dv,nv)
        if arm=='regional_modes' and len(targets)>len(modes):
            audit['excluded'].append(dict(zone=zid,reason='UNSEPARABLE_MULTIPLE_RETURNS',slots=[i for i,_ in targets]));continue
        group=dict(zone=zid,ranges=np.array([v['distance_m'] for _,v in targets]),
            slots=[i for i,_ in targets],modes=np.stack([v[0] for v in modes]))
        groups.append(group);audit['included'].append(dict(zone=zid,slots=group['slots'],mode_pixels=[v[1] for v in modes]))
    return d,groups,audit


def predicted_ranges(parameters,modes):
    a,b=parameters
    return np.mean(modes[...,1]/(a*modes[...,0]+b),axis=-1)


def assignments(groups,parameters,ordinal=False):
    choices=[]
    for g in groups:
        pred=predicted_ranges(parameters,g['modes']);count=len(g['ranges'])
        if len(pred)==1:choice=[0]*count
        elif count==1:
            choice=[int(np.argmin(pred)) if ordinal else int(np.argmin(abs(pred-g['ranges'][0])))]
        else:
            permutations=[[0,1],[1,0]]
            choice=min(permutations,key=lambda p:float(np.square(pred[p]-g['ranges']).sum()))
        choices.append(choice)
    return choices


def fitted(groups,*,latent):
    if len(groups)<6:return dict(valid=False,reason='INSUFFICIENT_DISTINCT_ZONES',zones=len(groups))
    target=np.concatenate([g['ranges'] for g in groups])
    weights=np.concatenate([np.full(len(g['ranges']),1/math.sqrt(len(g['ranges']))) for g in groups])
    starts=[False,True] if latent else [False];solutions=[]
    for ordinal in starts:
        parameter=np.array([.5,.2]);choice=assignments(groups,parameter,ordinal=ordinal)
        for _ in range(6 if latent else 1):
            selected=np.stack([g['modes'][c] for g,cs in zip(groups,choice) for c in cs])
            def residual(v):return (predicted_ranges(v,selected)-target)*weights
            fit=least_squares(residual,parameter,bounds=([0.,.05],[100.,100.]),loss='huber',f_scale=.12,max_nfev=35)
            parameter=fit.x
            choice=assignments(groups,parameter)
        # Refit to the last assignment so reported parameters and support agree.
        selected=np.stack([g['modes'][c] for g,cs in zip(groups,choice) for c in cs])
        def residual(v):return (predicted_ranges(v,selected)-target)*weights
        fit=least_squares(residual,parameter,bounds=([0.,.05],[100.,100.]),loss='huber',f_scale=.12,max_nfev=35)
        singular=np.linalg.svd(fit.jac,compute_uv=False);ratio=float(singular[-1]/singular[0]) if singular[0]>0 else 0.
        raw=predicted_ranges(fit.x,selected)-target
        valid=bool(fit.success and ratio>=1e-3 and fit.x[0]>1e-5 and np.isfinite(fit.x).all())
        solutions.append(dict(valid=valid,reason='PUBLIC_POINT_ESTIMATE' if valid else 'UNIDENTIFIABLE_OR_ZERO_SLOPE',
            parameters=fit.x.tolist(),jacobian_singular_ratio=ratio,cost=float(fit.cost),
            median_abs_range_residual_m=float(np.median(abs(raw))),max_abs_range_residual_m=float(abs(raw).max()),
            zones=len(groups),returns=len(target),solver_success=bool(fit.success),
            active_bounds=fit.active_mask.tolist(),assignments=[dict(zone=g['zone'],slots=g['slots'],modes=c) for g,c in zip(groups,choice)]))
    selected=min(solutions,key=lambda s:s['cost'])
    return dict(selected,starts=solutions)


def calibrate(row,relative,arm):
    d,groups,audit=anchors(row,relative,arm)
    shape=relative.shape;output=np.full(shape,np.nan,np.float32)
    train=[g for g in groups if g['zone']%4!=0];held=[g for g in groups if g['zone']%4==0]
    trial=fitted(train,latent=arm=='regional_modes');residual=[]
    if trial['valid']:
        cs=assignments(held,trial['parameters'])
        for g,c in zip(held,cs):residual.extend((predicted_ranges(trial['parameters'],g['modes'])[c]-g['ranges']).tolist())
    audit['withheld_zone_check']=dict(fit=trial,withheld_zones=len(held),compared_returns=len(residual),
        median_abs_range_residual_m=float(np.median(np.abs(residual))) if residual else None)
    fit=fitted(groups,latent=arm=='regional_modes');audit['fit']=fit
    if fit['valid'] and d is not None:
        a,b=fit['parameters'];output=(1/(a*d+b)).astype(np.float32)
    audit['raw_packets_preserved']=True;audit['valid_prediction_pixels']=int(np.isfinite(output).sum())
    return output,audit
