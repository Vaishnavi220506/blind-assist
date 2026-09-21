"""Ordered frozen image features plus optional full-interval query coordinates."""
import numpy as np
import torch
from torch import nn
from spatial_bce_model import _boxes

RECIPE = dict(seed=20260921,steps=1200,batch_size=64,learning_rate=.001,weight_decay=.0001)


def ordered_grid(features):
    """Invert feature/subrow/subcol packing, interleave zone/subcell spatial axes."""
    f=np.asarray(features,np.float32)
    if f.ndim!=4 or f.shape[1:]!=(216,8,8):
        raise ValueError('Expected [N,216,8,8] ordered feature cache')
    return f.reshape(-1,24,3,3,8,8).transpose(0,1,4,2,5,3).reshape(-1,24,24,24)


def build_inputs(features,ranges,boxes):
    f=ordered_grid(features)
    values=np.asarray(ranges,np.float32)
    if values.shape!=(len(f),64) or not np.isfinite(f).all():
        raise ValueError('Expected finite RGB and matched [N,64] ranges')
    b=_boxes(np.asarray(boxes,np.float32)/[192,256,192,256])
    frac=(np.arange(3,dtype=np.float32)+.5)/3
    # [zone row,subrow,zone col,subcol] is the same order as ordered_grid.
    bx=b.reshape(8,8,4)
    x=bx[:,None,:,1,None]+(bx[:,None,:,3,None]-bx[:,None,:,1,None])*frac[None,None,None,:]
    y=bx[:,None,:,0,None]+(bx[:,None,:,2,None]-bx[:,None,:,0,None])*frac[None,:,None,None]
    x=np.broadcast_to(x,(8,3,8,3)).reshape(24,24)
    y=np.broadcast_to(y,(8,3,8,3)).reshape(24,24)
    focal=640/(2*np.tan(np.deg2rad(50)))
    a=(x*640-320)/focal; c=(y*360-180)/focal
    valid=np.isfinite(values)&(values>=.1)&(values<8)
    observed=np.where(valid,values,0)
    v=np.repeat(np.repeat(valid.reshape(-1,8,8),3,1),3,2)
    r=np.repeat(np.repeat(observed.reshape(-1,8,8),3,1),3,2)
    common=np.stack([r/8,v.astype(np.float32),np.broadcast_to(np.arctan(a)/np.deg2rad(45),r.shape),
                     np.broadcast_to(np.arctan(c)/np.deg2rad(45),r.shape)],1)
    radius=.1+3*(.01+.02*r)
    lo=np.maximum(.1,r-radius); hi=r+radius
    channels=[]
    for z in (lo,hi):
        channels.extend([a*z+.3,.3-a*z,c*z+.2,.9-c*z])
    geometry=np.clip(np.stack(channels,1),-4,4)/4
    geometry=np.where(v[:,None],geometry,0)
    result=np.concatenate([f,common,geometry],1).astype(np.float32)
    assert result.shape==(len(f),36,24,24) and np.isfinite(result).all()
    return result


def arm_inputs(inputs,arm):
    if arm not in ('U','G'):
        raise ValueError('Expected U or G')
    value=np.array(inputs,copy=True)
    if arm=='U': value[:,28:]=0
    return value


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(36,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Flatten(),nn.Linear(1152,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x):
        if x.ndim!=4 or x.shape[1:]!=(36,24,24):
            raise ValueError('Expected [N,36,24,24]')
        return self.net(x).squeeze(-1)
