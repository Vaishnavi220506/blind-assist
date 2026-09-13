"""Keep position-bound RGB/ToF samples until after local fusion.

RGB encoder, original ToF/Radar encoders and 45-cell readout remain MZ120.
Radar's cell-level branch is unchanged; only RGB/ToF fusion becomes point-local.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mz120_occupancy import OccupancyNet


def point_geometry(row, encoded):
    intr = row['rgb_intrinsics']; grid = encoded['grid']
    pixels = (grid+1)*np.array([intr['width']-1,intr['height']-1])/2
    boxes = []
    for zone in row['tof_zones']:
        a,b=zone['theta_bounds_deg'];c,d=zone['phi_bounds_deg']
        box=[intr['cx']+intr['fx']*math.tan(math.radians(a)),
             intr['cy']-intr['fy']*math.tan(math.radians(d)),
             intr['cx']+intr['fx']*math.tan(math.radians(b)),
             intr['cy']-intr['fy']*math.tan(math.radians(c))]
        boxes.extend([box,box])
    boxes=np.asarray(boxes)
    # Same 8px angular-neighborhood padding as MZ120, applied per sample.
    u=pixels[...,0,None];v=pixels[...,1,None]
    mask=(u>=boxes[:,0]-8)&(u<=boxes[:,2]+8)&(v>=boxes[:,1]-8)&(v<=boxes[:,3]+8)
    mask &= encoded['local'][:,None,:-1]
    return np.concatenate([mask,np.ones((*mask.shape[:-1],1),bool)],-1)


class LocalFusionNet(OccupancyNet):
    def __init__(self):
        super().__init__()
        self.position = nn.Linear(2,24)
        self.local_fusion = nn.Sequential(nn.Linear(24+24+2,24),nn.SiLU())

    def fuse_samples(self, sampled, batch):
        # sampled: B,Q,P,24. No position is pooled before RGB/ToF correspondence.
        b,q,p,_=sampled.shape
        cells=self.cells[None,:,None].expand(b,-1,p,-1)
        imu=batch['imu'][:,None,None].expand(-1,q,p,-1)
        position=batch['grid'].clamp(-4,4)
        query=self.query(torch.cat([sampled,sampled,cells,imu],-1))+self.position(position)
        tof=self.tof(batch['tof'])
        scores=torch.einsum('bqpc,btc->bqpt',query,tof)/math.sqrt(24)
        weights=scores.masked_fill(~batch['point_local'],-1e4).softmax(-1)
        returns=torch.einsum('bqpt,btc->bqpc',weights,tof)
        fused=self.local_fusion(torch.cat([sampled,returns,position],-1))
        rgb=torch.cat([fused.mean(2),fused.amax(2)],-1)
        return rgb,returns.mean(2)

    def forward(self,batch):
        features=self.rgb(batch['rgb'])
        sampled=F.grid_sample(features,batch['grid'],align_corners=True).permute(0,2,3,1)
        rgb,local_tof=self.fuse_samples(sampled,batch)
        b,q,_,_=sampled.shape
        cells=self.cells[None].expand(b,-1,-1);imu=batch['imu'][:,None].expand(-1,q,-1)
        # Keep Radar's original global-query recipe separate from the tested change.
        original_rgb=torch.cat([sampled.mean(2),sampled.amax(2)],-1)
        rq=self.query(torch.cat([original_rgb,cells,imu],-1));radar=self.radar(batch['radar'])
        rw=(rq@radar.transpose(1,2)/math.sqrt(24)).softmax(-1)
        return self.head(torch.cat([rgb,local_tof,rw@radar,cells,imu],-1)).squeeze(-1)
