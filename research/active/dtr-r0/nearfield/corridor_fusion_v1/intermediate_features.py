"""Frozen DA-V2 positioned encoder tokens; label-free TRAIN PCA adapter."""
import numpy as np
import torch
from depth_features import query_box

LAYERS=(2,11)
POSITIONS=48
CHANNELS=8


def positions(row,yaw):
    intr=row['rgb_intrinsics'];w,h=intr['width'],intr['height']
    locations=[];names=[]
    def append(box,ny,nx,prefix):
        for iy in range(ny):
            for ix in range(nx):
                if box is None:u=v=-2.
                else:
                    l,t,r,b=box;u=(l+(ix+.5)/nx*(r-l))/w;v=(t+(iy+.5)/ny*(b-t))/h
                locations.append([u,v]);names.append(f'{prefix}_{iy}_{ix}')
    for distance in (.5,1.,2.,3.6):
        for band,low,high in [('body',.4,1.5),('head',1.5,2.05)]:
            append(query_box(row,yaw,distance,low,high),2,2,f'{band}_{distance}')
    append([0,0,w,h],4,4,'image')
    uv=np.array(locations,np.float32)
    valid=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<1)&(uv[:,1]>=0)&(uv[:,1]<1)
    assert uv.shape==(POSITIONS,2)
    return uv,valid,names


def sample_maps(maps,uv,valid):
    """maps: L,C,H,W. Native-normalized fractions map to resized token centers."""
    grid=torch.as_tensor(uv*2-1,device=maps.device,dtype=maps.dtype)[None,None]
    grid=grid.expand(maps.shape[0],1,-1,2)
    sampled=torch.nn.functional.grid_sample(maps,grid,align_corners=False,
        mode='bilinear',padding_mode='border')[:,:,0].transpose(1,2)
    mask=torch.as_tensor(valid,device=maps.device)
    sampled=sampled.masked_fill(~mask[None,:,None],0)
    return sampled


@torch.inference_mode()
def extract(model,image,row,yaw):
    tensor,_=model.image2tensor(image,518)
    features=model.pretrained.get_intermediate_layers(tensor,list(LAYERS),reshape=True,
        return_class_token=False,norm=True)
    maps=torch.cat(features,dim=0)
    assert maps.shape==(2,384,37,66),maps.shape
    uv,valid,names=positions(row,yaw)
    sampled=sample_maps(maps,uv,valid).cpu().numpy()
    assert sampled.shape==(2,48,384) and np.isfinite(sampled).all()
    return sampled,valid,names


def transform(tokens,masks,pcas):
    """Preserve positions; invalid measurements never acquire PCA mean offsets."""
    n=len(tokens);output=[]
    for layer,pca in enumerate(pcas):
        projected=pca.transform(tokens[:,layer].reshape(-1,384)).reshape(n,48,CHANNELS)
        projected[~masks]=0
        output.append(projected.reshape(n,-1))
    result=np.concatenate([*output,masks.astype(np.float32)],axis=1).astype(np.float32)
    assert result.shape==(n,816) and np.isfinite(result).all()
    return result
