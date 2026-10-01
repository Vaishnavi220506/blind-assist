"""CVR finite-volume projection; observations and rigid transforms only.

No scene generator, labels, boxes or family identifiers are accepted here.
Intersection volumes use fixed 3x3x3 midpoint quadrature within each voxel.
Sensor cells are angular rectangles intersected with radial shells.
"""
import numpy as np
import torch

LOW = np.array([-.6, -.5, 0.])
STEP = np.array([.05, .1, .1])
SHAPE = (24, 17, 33)
WIDTH = 8 * .0375348
EDGE = np.tan(np.pi/8)
SUB = 3


def grid():
    indices = np.stack(np.meshgrid(*[np.arange(n) for n in SHAPE], indexing='ij'), -1)
    centers = LOW + (indices+.5)*STEP
    off = np.stack(np.meshgrid(*[(np.arange(SUB)+.5)/SUB-.5]*3, indexing='ij'), -1).reshape(-1,3)
    return centers, centers.reshape(-1,1,3)+off[None]*STEP


def cell_volumes():
    edges = np.linspace(-EDGE, EDGE, 9)
    def primitive(x,y):
        return np.arctan2(x*y, np.sqrt(1+x*x+y*y))
    lo, hi = edges[:-1], edges[1:]
    angle = (primitive(hi[:,None],hi[None,:])-primitive(lo[:,None],hi[None,:])
             -primitive(hi[:,None],lo[None,:])+primitive(lo[:,None],lo[None,:]))
    radial = ((np.arange(1,17)*WIDTH)**3-(np.arange(16)*WIDTH)**3)/3
    return angle[:,:,None]*radial


def query_masks():
    # Exact box-voxel overlap fractions, including y=.42 crossing a voxel.
    centers,_ = grid()
    lower,upper = centers-STEP/2,centers+STEP/2
    masks=[]
    for yl,yh in [(-.2,.42),(.42,.9)]:
        length=np.maximum(0,np.minimum(upper,[.3,yh,3.])-np.maximum(lower,[-.3,yl,.3]))
        masks.append(np.prod(length/STEP,axis=-1))
    return np.array(masks,dtype=np.float32)


class Projector:
    def __init__(self, device='cuda'):
        self.device = device
        _,points=grid()
        self.points=torch.tensor(points.reshape(-1,3),dtype=torch.float64,device=device)
        self.volumes=torch.tensor(cell_volumes().reshape(-1),dtype=torch.float64,device=device)
        self.voxel_volume=float(np.prod(STEP))

    @torch.no_grad()
    def frame(self, z, corridor_from_sensor):
        """Return distributed z mass and observed fraction for a single exposure.

        Each bin contributes z * overlap_volume / complete_bin_volume.
        Geometric coverage counts all bins regardless of z, including empty.
        Cropped mass is not renormalized back into the query domain.
        """
        t=torch.as_tensor(corridor_from_sensor,dtype=torch.float64,device=self.device)
        # Row-vector inverse rigid transform (corridor sample -> sensor).
        p=(self.points-t[:3,3])@t[:3,:3]
        radius=torch.linalg.vector_norm(p,dim=1)
        xy=p[:,:2]/p[:,2:3].clamp_min(1e-30)
        ij=torch.floor((xy+EDGE)/(2*EDGE)*8).long()
        bins=torch.floor(radius/WIDTH).long()
        valid=(p[:,2]>0)&(ij>=0).all(1)&(ij<8).all(1)&(bins>=0)&(bins<16)
        index=((ij[:,1].clamp(0,7)*8+ij[:,0].clamp(0,7))*16+bins.clamp(0,15))
        weights=valid*self.voxel_volume/(SUB**3)/self.volumes[index]
        evidence=torch.as_tensor(z,dtype=torch.float64,device=self.device).reshape(-1)[index]*weights
        out=evidence.reshape(-1,SUB**3).sum(1).reshape(SHAPE)
        coverage=valid.reshape(-1,SUB**3).double().mean(1).reshape(SHAPE)
        return out.float(),coverage.float()

    @torch.no_grad()
    def sequence(self, z, corridor_from_sensor):
        assert len(z)==len(corridor_from_sensor) and 1<=len(z)<=8
        total=torch.zeros(SHAPE,device=self.device)
        count=torch.zeros_like(total)
        for values,transform in zip(z,corridor_from_sensor):
            current,seen=self.frame(values,transform)
            total+=current;count+=seen
        return torch.stack([total,count,current],0)
