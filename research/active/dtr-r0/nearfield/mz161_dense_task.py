"""Shared dense corridor-task classifier with independent raw-return tokens.

Pixel outputs are learned risk scores, not metric depth or certified surfaces.
All original ToF/Radar slots have a separate token path independent of RGB FOV.
"""
import math
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mz143_corridor_features import _public
from mz115_spatial_allocation import slant_envelope, zone_box
from mz136_boundary_geometry import camera_to_body

CHANNELS = 22
TOKEN_CHANNELS = 21
METHOD = dict(schema='MZ161_DENSE_CORRIDOR_TASK_V1', image_shape=[360,640],
    image_channels=CHANNELS, token_channels=TOKEN_CHANNELS, tokens=132,
    full_resolution=True, hidden_channels=24, dilations=[1,2,4,8,16],
    frame_reducer='MAX_DENSE_PIXEL_AND_INDEPENDENT_RAW_RETURN_LOGITS',
    supervision='FRAME_BCE_CONTROL_VS_FRAME_BCE_PLUS_BALANCED_KNOWN_PIXEL_BCE',
    unknown='NONALERT_IS_UNKNOWN; UNSAVED_REFERENCE_PIXELS_MASKED',
    input_authority='PUBLIC_RGB_TOF_RADAR_IMU_ONLY',
    claims='COORDINATE_CONDITIONED_SHARED_PROCESSING; NOT_CERTIFIED_GEOMETRY')


def ray_query(row, yaw):
    intr=row['rgb_intrinsics']; h,w=intr['height'],intr['width']
    yy,xx=np.mgrid[:h,:w]
    rays=np.stack([np.ones_like(xx),(xx-intr['cx'])/intr['fx'],(intr['cy']-yy)/intr['fy']],-1)
    rays=rays/np.linalg.norm(rays,axis=-1,keepdims=True)
    directions=rays@camera_to_body(row,yaw).T
    origin=np.asarray(row['camera_in_body_m'],float)
    lower=np.array([.2,-.3,.4]);upper=np.array([3.6,.3,2.05])
    enter=np.zeros((h,w));leave=np.full((h,w),np.inf)
    for axis in range(3):
        d=directions[...,axis];parallel=np.abs(d)<1e-12;safe=np.where(parallel,1.,d)
        aa=(lower[axis]-origin[axis])/safe;bb=(upper[axis]-origin[axis])/safe
        inside=lower[axis]<=origin[axis]<=upper[axis]
        lo=np.where(parallel,-np.inf if inside else np.inf,np.minimum(aa,bb))
        hi=np.where(parallel,np.inf if inside else -np.inf,np.maximum(aa,bb))
        enter=np.maximum(enter,lo);leave=np.minimum(leave,hi)
    valid=(leave>=enter)&(leave>0)&np.isfinite(enter)&np.isfinite(leave)
    query=np.stack([np.where(valid,enter/4,0),np.where(valid,leave/4,0),valid],0).astype(np.float32)
    return directions.astype(np.float32),query


def encode(row,image,yaw):
    row=_public(row);intr=row['rgb_intrinsics'];h,w=intr['height'],intr['width']
    if image.shape!=(h,w,3) or image.dtype!=np.uint8 or not math.isfinite(yaw):
        raise ValueError('Native BGR uint8 and finite public yaw required')
    direction,query=ray_query(row,yaw)
    maps=np.zeros((10,h,w),np.float32);tokens=[];present=[];outside=0
    rotation=camera_to_body(row,yaw);origin=np.asarray(row['camera_in_body_m'],float)
    packets=(float(row['imu_valid']),float(row['tof_packet_received']),float(row['radar_packet_received']))
    for z in sorted(row['tof_zones'],key=lambda z:z['zone_id']):
        box=zone_box(z,intr)
        fully_visible=0<=box[0]<=box[2]<=w-1 and 0<=box[1]<=box[3]<=h-1
        theta=math.radians(sum(z['theta_bounds_deg'])/2);phi=math.radians(sum(z['phi_bounds_deg'])/2)
        ray=np.array([1.,math.tan(theta),math.tan(phi)]);ray/=np.linalg.norm(ray)
        # Half-open pixel ownership: adjacent regions cannot overwrite a shared edge.
        l=max(0,int(math.ceil(box[0])));r=min(w-1,int(math.ceil(box[2]))-1)
        t=max(0,int(math.ceil(box[1])));b=min(h-1,int(math.ceil(box[3]))-1)
        for slot in range(2):
            target=z['targets'][slot] if slot<len(z['targets']) else None
            usable=bool(target and row['tof_packet_received'] and target['status'] in ('SIM_VALID','SIM_MERGED'))
            values=np.array([target[k] for k in ('distance_m','range_noise_sigma_m','signal_strength_proxy')],float) if target else np.zeros(3)
            usable=usable and np.isfinite(values).all() and values[0]>0 and min(values[1:])>=0
            distance,sigma,strength=values if usable else np.zeros(3)
            merged=bool(usable and target['status']=='SIM_MERGED')
            center=rotation@(distance*ray)+origin if usable else np.zeros(3)
            support=np.zeros((3,2))
            if usable:
                bounds=(.02,4.) if merged else (max(.02,distance-3*sigma),distance+3*sigma)
                support=np.asarray(slant_envelope(box,bounds,intr,(row['camera_pitch_deg'],)*2,(yaw,)*2,0.))+origin[:,None]
            tokens.append([usable,merged,distance/4,sigma/.2,math.log1p(strength*distance**2),
                center[0]/4,center[1]/2,center[2]/3,*list(support[0]/4),*list(support[1]/2),*list(support[2]/3),
                0.,0.,1.,0.,*packets])
            present.append(bool(usable));outside+=int(usable and not fully_visible)
            if r>=l and b>=t:
                maps[slot*5:(slot+1)*5,t:b+1,l:r+1]=np.array([distance/4,sigma/.2,
                    math.log1p(strength*distance**2),usable,merged],np.float32)[:,None,None]
    for r,a,v,valid in zip(row['radar_range_m'],row['radar_angle'],row['radar_velocity'],row['radar_valid']):
        valid=bool(valid and row['radar_packet_received'] and r is not None and a is not None and math.isfinite(r) and math.isfinite(a) and r>0)
        rr=float(r) if valid else 0.;angle=math.radians(float(a)+yaw) if valid else 0.
        x,y=rr*math.cos(angle),rr*math.sin(angle)
        vv=bool(valid and v is not None and math.isfinite(v))
        tokens.append([valid,0.,rr/4,0.,0.,x/4,y/2,0.,x/4,x/4,y/2,y/2,-1.,1.,
            float(v)/3 if vv else 0.,vv,0.,1.,*packets]);present.append(valid)
    planes=np.broadcast_to(np.array(packets,np.float32)[:,None,None],(3,h,w))
    features=np.concatenate([image[...,::-1].transpose(2,0,1).astype(np.float32)/127.5-1,
        maps,query,direction.transpose(2,0,1),planes],0)
    tokens=np.asarray(tokens,np.float32);present=np.asarray(present,bool)
    assert features.shape==(CHANNELS,h,w) and tokens.shape==(132,TOKEN_CHANNELS)
    if not np.isfinite(features).all() or not np.isfinite(tokens).all():raise ValueError('Nonfinite public tensor')
    return dict(image=features,tokens=tokens,valid=present,
        audit=dict(public_tof_slots_encoded=128,public_radar_slots_encoded=4,
            valid_tokens=int(present.sum()),outside_or_partial_rgb_tof_tokens=outside,
            all_slots_have_independent_path=True,public_ray_query_pixels=int(query[2].sum())))


class Residual(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.depth=nn.Conv2d(24,24,3,padding=dilation,dilation=dilation,groups=24)
        self.point=nn.Conv2d(24,24,1)
    def forward(self,x):return x+self.point(F.silu(self.depth(x)))


class DenseTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(CHANNELS,24,3,padding=1)
        self.blocks=nn.ModuleList([Residual(d) for d in METHOD['dilations']])
        self.pixel=nn.Sequential(nn.Conv2d(24,24,1),nn.SiLU(),nn.Conv2d(24,1,1))
        self.returns=nn.Sequential(nn.Linear(TOKEN_CHANNELS,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
    def forward(self,batch):
        x=F.silu(self.stem(batch['image']))
        for block in self.blocks:x=F.silu(block(x))
        pixel=self.pixel(x)[:,0]
        token=self.returns(batch['tokens']).squeeze(-1).masked_fill(~batch['valid'],-30.)
        local=pixel.masked_fill(batch['image'][:,15]<=0,-30.).flatten(1).amax(1)
        independent=token.amax(1)
        return dict(pixel=pixel,token=token,local=local,independent=independent,frame=torch.maximum(local,independent))


def loss(output,frame_target,pixel_target,pixel_weight,arm):
    frame=F.binary_cross_entropy_with_logits(output['frame'],frame_target)
    dense=(F.binary_cross_entropy_with_logits(output['pixel'],pixel_target,reduction='none')*pixel_weight).flatten(1).sum(1).mean()
    total=frame+(dense if arm=='dense' else 0.)
    return total,dict(frame=float(frame.detach()),dense=float(dense.detach()))
