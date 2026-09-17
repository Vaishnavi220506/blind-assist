"""Small observable corridor-query residual; evaluator margins are labels only."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_e1 import ROOT
from mz120_occupancy import encode,CELLS,CORE

QUERY=CELLS[CORE].copy()

def local_inputs(row,image,yaw):
    encoded=encode(row,yaw)
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    gradient=cv2.Scharr(gray,cv2.CV_32F,1,0)/16
    channels=np.stack([gray,gradient,np.ones_like(gray)],-1)
    patches=[]
    for grid in encoded['grid'][CORE]:
        lo,hi=grid.min(0),grid.max(0)
        xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],21),np.linspace(lo[1],hi[1],21))
        patch=cv2.remap(channels,((xx+1)*(w-1)/2).astype(np.float32),((yy+1)*(h-1)/2).astype(np.float32),
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        patches.append(patch.transpose(2,0,1))
    return dict(patch=np.stack(patches),tof=encoded['tof'],local=encoded['local'][CORE])

def native_margins(evaluation):
    """Signed minimum separating-face slack, metres; max over ALL native objects.

    Positive means intersection of that query and an object; negative means
    separation. This is an AABB intrusion surrogate, not Euclidean surface depth.
    It retains volume labels, including objects not visible in RGB.
    """
    origin=np.asarray(evaluation['body_origin_m'])
    lo=np.array([o['center_m'] for o in evaluation['native_bounds']])-np.array([o['extent_m'] for o in evaluation['native_bounds']])-origin
    hi=np.array([o['center_m'] for o in evaluation['native_bounds']])+np.array([o['extent_m'] for o in evaluation['native_bounds']])-origin
    margins=np.minimum(hi[None]-QUERY[:,None,:3],QUERY[:,None,3:]-lo[None]).min(-1).max(-1)
    return margins.astype(np.float32)

class IntrusionNet(nn.Module):
    def __init__(self,features=2485):
        super().__init__()
        self.base=nn.Sequential(nn.Linear(features,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.rgb=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.SiLU(),nn.AvgPool2d(3),
            nn.Conv2d(8,8,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(72,24),nn.SiLU())
        self.tof=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,16),nn.SiLU())
        self.query=nn.Sequential(nn.Linear(32+24+32+6,32),nn.SiLU())
        self.field=nn.Linear(32,1);self.margin=nn.Linear(32,1)
        self.frame=nn.Sequential(nn.Linear(32+8*32,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.frame[-1].weight);nn.init.zeros_(self.frame[-1].bias)
        self.register_buffer('queries',torch.from_numpy(QUERY)/torch.tensor([4.,1.,3.,4.,1.,3.]))

    def forward(self,b):
        n=len(b['base']);global_feature=self.base(b['base'])
        rgb=self.rgb(b['patch'].reshape(-1,3,21,21)).reshape(n,8,24)
        tof=self.tof(b['tof']);mask=b['local']
        mean=torch.einsum('bqt,btc->bqc',mask.float(),tof)/mask.sum(-1,keepdim=True).clamp_min(1)
        maximum=tof[:,None].expand(-1,8,-1,-1).masked_fill(~mask[:,:,:,None],-1e4).amax(2)
        query=self.query(torch.cat([global_feature[:,None].expand(-1,8,-1),rgb,mean,maximum,
            self.queries[None].expand(n,-1,-1)],-1))
        frame=b['A_logit']+self.frame(torch.cat([global_feature,query.flatten(1)],-1)).squeeze(-1)
        return dict(logit=frame,field=self.field(query).squeeze(-1),margin=self.margin(query).squeeze(-1))

def objective(pred,b,auxiliary=False,pairs=None,invariance=None):
    loss=F.binary_cross_entropy_with_logits(pred['logit'],b['truth'])
    if auxiliary:
        loss=loss+.25*F.binary_cross_entropy_with_logits(pred['field'],b['field'])
        loss=loss+.2*F.smooth_l1_loss(pred['margin'],b['margin'].clamp(-.3,.3)/.3,beta=.1)
        if pairs is not None and len(pairs):
            i,j=pairs.T;sign=b['truth'][i]-b['truth'][j]
            loss=loss+.1*F.relu(.5-(pred['logit'][i]-pred['logit'][j])*sign).mean()
        if invariance is not None and len(invariance):
            i,j=invariance.T
            loss=loss+.1*F.mse_loss(pred['logit'][i].sigmoid(),pred['logit'][j].sigmoid())
            # Background motion can change clearance of an empty query even
            # when frame truth is unchanged. Only equal geometric targets
            # justify forcing query-margin invariance.
            target=b['margin'].clamp(-.3,.3)/.3
            same=(target[i]-target[j]).abs()<1e-4
            if same.any():loss=loss+.05*F.mse_loss(pred['margin'][i][same],pred['margin'][j][same])
    return loss
