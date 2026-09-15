"""Frozen official DEPTHOR-ZJU-Small observation adapter; no evaluator inputs.

Public regional slant range is approximately assigned to its region center as
camera-axis depth. This is the pretrained interface, not a true pixel hit.
"""
import math
import sys
import types
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mz115_spatial_allocation import zone_box

METHOD = dict(model='DEPTHOR_ZJU_SMALL_OFFICIAL_FROZEN', n_bins=256,
    min_depth=.001,max_depth=10.,network_size=[480,640],training=False,
    rgb='FULL_RGB_0_1_INTERNAL_IMAGENET_NORMALIZATION',
    sparse='ONE_SINGLE_SIM_VALID_PUBLIC_ZONE_CENTER_AXIS_DEPTH_APPROXIMATION',
    missing='ZERO_MODEL_INPUT_NOT_CLEAR_SPACE; PRESERVE_RAW_SUPPORT',
    multiple_or_merged='OMIT_FROM_SPARSE_INPUT_KEEP_INCUMBENT',
    outside_rgb='OMIT_CENTER_OUTSIDE_RGB_KEEP_INCUMBENT',
    custom_op='BP_NET_LOCAL_CONV_FORWARD_EQUIVALENT_TORCH_UNFOLD',
    rgb_ablation='KEEP_PUBLIC_RECOVERED_BOX_PIXELS_MEAN_RGB_ELSEWHERE_SAME_TOF',
    output='CAMERA_AXIS_DEPTH_M_NOT_CERTIFIED_INTERVAL')


def local_conv(input,weight):
    """BP-Net conv2d_kernel_lf, zero padding, per-output-pixel weights.

    Inference only. Official order: row-major kernel, channel-wise local filter.
    This changes execution backend, not learned CSPN architecture or weights.
    """
    b,c,h,w=input.shape;k=math.isqrt(weight.shape[1]//c)
    assert k*k*c==weight.shape[1] and k%2==1
    patches=F.unfold(input,kernel_size=k,padding=k//2).reshape(b,c,k*k,h,w)
    return (patches*weight.reshape(b,c,k*k,h,w)).sum(2)


def load_model(upstream,checkpoint,device='cuda'):
    upstream=Path(upstream).resolve();sys.path.insert(0,str(upstream))
    # Only the exact inference operation is exposed; no training surrogate.
    bp=types.ModuleType('BpOps');bp.Conv2dLocal_F=local_conv
    sys.modules['BpOps']=bp
    from src.utils import set_mde
    from src.models.depth_anything_v2.dpt import DepthAnythingV2
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert isinstance(weights,dict) and any(k.startswith('depth_anything.') for k in weights)
    def initialize_embedded(encoder='vits'):
        assert encoder=='vits'
        return DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    # Full checkpoint supplies all frozen DA parameters; avoid author's absolute
    # initialization path. Strict loading below rejects absent/mismatched keys.
    original=set_mde.set_depthanything;set_mde.set_depthanything=initialize_embedded
    try:
        from src.models.depthor_s import Depthor
        model=Depthor(n_bins=256,min_val=.001,max_val=10.)
    finally:set_mde.set_depthanything=original
    model.load_state_dict(weights,strict=True)
    model.set_extra_param(device);model=model.to(device).eval()
    model.requires_grad_(False)
    return model,dict(checkpoint_keys=len(weights),parameters=sum(p.numel() for p in model.parameters()),
        strict_load=True,embedded_monocular_keys=sum(k.startswith('depth_anything.') for k in weights))


def make_input(row,bgr,device='cuda'):
    intr=row['rgb_intrinsics'];h,w=bgr.shape[:2]
    assert [h,w]==[intr['height'],intr['width']]==[360,640]
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    rgb=cv2.resize(rgb,(640,480),interpolation=cv2.INTER_LINEAR)
    sparse=np.zeros((480,640),np.float32);used=[];excluded=[]
    for z in row['tof_zones']:
        reason=None;targets=z['targets']
        if not row['tof_packet_received']:reason='MISSING_PACKET'
        elif len(targets)!=1:reason='MISSING_OR_MULTIPLE'
        elif targets[0]['status']!='SIM_VALID':reason='NON_SINGLE_VALID'
        if reason:excluded.append(dict(zone=z['zone_id'],reason=reason));continue
        box=zone_box(z,intr);l,t,r,b=box
        u,v=(l+r)/2,(t+b)/2
        if not (0<=u<w and 0<=v<h):
            excluded.append(dict(zone=z['zone_id'],reason='CENTER_OUTSIDE_RGB'));continue
        # Full-image cv2 resize maps pixel centers using this half-pixel rule.
        x,y=int((u+.5)*640/w-.5),int((v+.5)*480/h-.5)
        depth=float(targets[0]['distance_m'])/math.sqrt(1+((u-intr['cx'])/intr['fx'])**2+((intr['cy']-v)/intr['fy'])**2)
        if not (.001<depth<=10) or not math.isfinite(depth):
            excluded.append(dict(zone=z['zone_id'],reason='INVALID_DEPTH'));continue
        assert 0<=x<640 and 0<=y<480 and sparse[y,x]==0
        sparse[y,x]=depth
        used.append(dict(zone=z['zone_id'],slot=0,source_center=[u,v],network_pixel=[x,y],axis_depth_m=depth,
                         whole_zone_in_rgb=bool(0<=l<r<=w-1 and 0<=t<b<=h-1)))
    data=dict(image=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].to(device),
              sparse_depth=torch.from_numpy(sparse)[None,None].to(device))
    return data,dict(used=used,excluded=excluded,input_shape=[480,640],source_shape=[h,w])


def box_only_rgb(bgr,cached):
    mask=np.zeros(bgr.shape[:2],bool)
    for b in cached.get('recovered_boxes',cached.get('proposals',[])):
        l,t,r,bottom=b;l=max(0,int(math.floor(l)));t=max(0,int(math.floor(t)))
        r=min(bgr.shape[1],int(math.ceil(r)));bottom=min(bgr.shape[0],int(math.ceil(bottom)))
        if r>l and bottom>t:mask[t:bottom,l:r]=True
    other=np.broadcast_to(np.round(bgr.mean((0,1))).astype(np.uint8),bgr.shape).copy()
    other[mask]=bgr[mask]
    return other,float(mask.mean())


@torch.inference_mode()
def predict(model,data):
    raw=model(data)[1]
    assert torch.isfinite(raw).all(), 'Nonfinite official model output'
    clipped=raw.clamp(.001,10.)
    return F.interpolate(clipped,size=(360,640),mode='bilinear',align_corners=False)[0,0]
