"""Frozen diagnostic scenes and label-blind, equal-budget angular renderer.

The renderer accepts static physical boxes only; query labels never enter it.
All sensor/geometry dependencies are imported from the retained v5 snapshot.
"""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-track-a-v5-20260928/data/source'
sys.path.insert(0, str(SOURCE))
from cnh_route_sensor import angular_rays, synthesize_response
from cnh_track_a_geometry import box_mesh, clip_triangles
from cnh_track_a_fov import rx
from cnh_track_a_v13_sensor import reference_parameters

for _module in ('cnh_route_sensor','cnh_track_a_geometry','cnh_track_a_fov','cnh_track_a_v13_sensor'):
    if Path(sys.modules[_module].__file__).resolve().parent != SOURCE.resolve():
        raise ImportError(f'{_module} was already loaded from live code; use an isolated frozen-source process')

MARGINS = (-.12, -.045, -.015, .015, .045, .12)
QUERY_LOW = np.array([[-.30,-.2,.3],[-.30,.42,.3]])
QUERY_HIGH = np.array([[.30,.42,3.],[.30,.9,3.]])


def ry(degrees):
    a=np.deg2rad(degrees);c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def _pose(rotation, position):
    t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=position
    return t


def labels_for(boxes, query_poses):
    """Evaluator-only visible-or-occluded surface intersection, no renderer use."""
    triangles=np.concatenate([box_mesh(b['lo'],b['hi']) for b in boxes])
    labels=[]
    for pose in query_poses:
        local=(triangles-pose[:3,3])@pose[:3,:3]
        labels.append([int(len(clip_triangles(local,lo,hi))>0) for lo,hi in zip(QUERY_LOW,QUERY_HIGH)])
    return np.asarray(labels,dtype=np.int8)


def make_scenes(unit):
    rng=np.random.default_rng(2026092900+unit)
    mode=unit%3
    yaw=np.linspace(-20.,0.,16) if mode==2 else np.zeros(16)
    position=np.zeros((16,3))
    for t in range(1,16):
        position[t]=position[t-1]+.16*(ry((yaw[t-1]+yaw[t])/2)@np.array([0.,0.,1.]))
    position-=position[-1]
    sensor_yaw=20*np.sin(np.linspace(-np.pi/2,np.pi/2,16)) if mode==1 else np.full(16,15. if mode==0 else 0.)
    travel=np.array([_pose(ry(y),p) for y,p in zip(yaw,position)])
    head=np.array([_pose(ry(y+s),p) for y,s,p in zip(yaw,sensor_yaw,position)])
    poses=np.array([_pose(h[:3,:3]@rx(-10),p) for h,p in zip(head,position)])
    scenes=[]
    def append(family,margin,group):
        width=float(rng.uniform(.08,.12));side=int(rng.choice([-1,1]))
        z=float(rng.uniform(1.15,1.85));thick=float(rng.uniform(.06,.18))
        yy=(-.10,.26) if group==0 else (.50,.84)
        inside=.30+margin
        xx=(inside,inside+width) if side==1 else (-inside-width,-inside)
        boxes=[dict(lo=[xx[0],yy[0],z],hi=[xx[1],yy[1],z+thick],rho=float(rng.uniform(.22,.65)))]
        if family=='mixed_surface':
            # Brighter second surface on the same lateral side, outside corridor,
            # deeper than the boundary target; it never receives a danger mask.
            a,b=.34,.72
            if side<0:a,b=-b,-a
            boxes.append(dict(lo=[a,yy[0]-.08,z+.32],hi=[b,yy[1]+.08,z+.48],rho=.65))
        elif family=='sidewall':
            a,b=.44,1.0
            if side<0:a,b=-b,-a
            boxes.append(dict(lo=[a,-.30,.65],hi=[b,1.5,3.8],rho=.60))
        boxes.extend([dict(lo=[-8.,1.65,-8.],hi=[8.,1.80,9.],rho=.45),
                      dict(lo=[-8.,-3.,4.5],hi=[8.,2.,4.7],rho=.35)])
        scenes.append(dict(unit=int(unit),config=len(scenes),family=family,margin=float(margin),group=int(group),
            boxes=boxes,poses=poses.copy(),travel=travel.copy(),head=head.copy(),
            labels=labels_for(boxes,travel),mode=mode,dt=.2,speed=.8))
    for fi,family in enumerate(['boundary','mixed_surface','sidewall']):
        group=(unit+fi)%2
        for m in MARGINS:append(family,float(m+rng.uniform(-.006,.006)),group)
    for j,m in enumerate([-.22,-.18,.20,.30]):append('general',m,(unit+j)%2)
    return scenes


def raycast_boxes(origin, directions, boxes):
    """Analytic first-visible AABB slab intersections; normalized radial range."""
    origin=np.asarray(origin,float)
    shape=np.asarray(directions).shape[:-1]
    d=np.asarray(directions,float).reshape(-1,3)
    d=d/np.linalg.norm(d,axis=1)[:,None]
    distance=np.full(len(d),np.inf);rho=np.zeros(len(d));cosine=np.zeros(len(d));ids=np.full(len(d),-1)
    parallel=abs(d)<1e-14
    inverse=np.divide(1.,d,out=np.zeros_like(d),where=~parallel)
    for i,b in enumerate(boxes):
        lo,hi=np.asarray(b['lo'],float),np.asarray(b['hi'],float)
        x=(lo-origin)*inverse;y=(hi-origin)*inverse
        near=np.where(parallel,-np.inf,np.minimum(x,y))
        far=np.where(parallel,np.inf,np.maximum(x,y))
        enter=near.max(1);leave=far.min(1)
        t=np.where(enter>1e-10,enter,leave)
        ok=(leave>=np.maximum(enter,0))&(t>1e-10)&~(parallel&((origin<lo)|(origin>hi))).any(1)
        take=ok&(t<distance)
        axis=np.where(enter>1e-10,near.argmax(1),far.argmin(1))
        distance[take]=t[take];rho[take]=b['rho'];ids[take]=i
        cosine[take]=abs(d[np.arange(len(d)),axis])[take]
    return dict(distance=distance.reshape(shape),rho=rho.reshape(shape),cos=cosine.reshape(shape),
                object_id=ids.reshape(shape),valid=np.isfinite(distance).reshape(shape))


@lru_cache(maxsize=1)
def nominal_parameters():
    params,receipt=reference_parameters()
    return params[6],receipt


@lru_cache(maxsize=1)
def ray_grid():
    return angular_rays(16)


def render(scene,seed,static=False,*,noise_scale=None):
    """Return coarse[N,8,8,16] and four-way angular fine[N,16,16,16].

    static uses exactly eight independent final-pose exposures. noise_scale=0
    is a mechanical expected-energy test hook, never a readout feature.
    """
    directions,weights=ray_grid()
    poses=np.repeat(scene['poses'][-1:],8,axis=0) if static else np.asarray(scene['poses'])
    hits=[raycast_boxes(p[:3,3],directions@p[:3,:3].T,scene['boxes']) for p in poses]
    dd=np.stack([h['distance'] for h in hits]).reshape(len(poses),8,8,16,16)
    rr=np.stack([h['rho'] for h in hits]).reshape(len(poses),8,8,16,16)
    cc=np.stack([h['cos'] for h in hits]).reshape(len(poses),8,8,16,16)
    w=weights.reshape(8,8,16,16)
    p,_=nominal_parameters()
    if noise_scale is not None:p=replace(p,noise_scale=float(noise_scale))
    quarter=replace(p,signal_counts=p.signal_counts/4,ambient_counts=p.ambient_counts/4)
    fine=np.empty((len(poses),16,16,16));ambient=np.empty((len(poses),16,16))
    for qy in range(2):
        for qx in range(2):
            ys,xs=slice(8*qy,8*qy+8),slice(8*qx,8*qx+8)
            qw=w[:,:,ys,xs].reshape(8,8,64)
            fraction=qw.sum(-1)/weights.sum(-1)
            shape=(len(poses),8,8,64)
            reflectance=rr[:,:,:,ys,xs].reshape(shape)*(4*fraction[None,:,:,None])
            if reflectance.max()>1:raise ValueError('quadrant rho scaling exceeds sensor domain; do not silently clip')
            response=synthesize_response(dd[:,:,:,ys,xs].reshape(shape),reflectance,
                cc[:,:,:,ys,xs].reshape(shape),qw,params=quarter,
                seed=int(np.random.SeedSequence([int(seed),qy,qx]).generate_state(1)[0]))
            fine[:,qy::2,qx::2]=response['histogram'].reshape(len(poses),8,8,16,8).sum(-1)
            ambient[:,qy::2,qx::2]=response['ambient']
    # Exact paired coarse observation, not a separately sampled approximation.
    coarse=fine.reshape(len(poses),8,2,8,2,16).sum(axis=(2,4))
    coarseambient=ambient.reshape(len(poses),8,2,8,2).sum(axis=(2,4))
    return dict(hist=coarse,ambient=coarseambient,finehist=fine,fineambient=ambient)
