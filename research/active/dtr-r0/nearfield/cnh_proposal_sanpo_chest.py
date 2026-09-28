"""Cheap local geometry-only chest coverage and Z/radial falsification check."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def depth(path):
    a = np.frombuffer(gzip.decompress(Path(path).read_bytes()), dtype='<f2')
    return a[2:].reshape(int(a[0]), int(a[1]))


def rays(c):
    yy, xx = np.meshgrid(np.arange(4, c['image_height'], 8),
                         np.arange(4, c['image_width'], 8), indexing='ij')
    return np.stack([(xx-c['cx'])/c['fx'], (yy-c['cy'])/c['fy'], np.ones_like(xx)], -1)


def xyz(d, ray, hyp):
    scale = d if hyp == 'Z' else d / np.linalg.norm(ray, axis=-1)
    return ray * scale[..., None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    old = a.repo/'artifacts.local/work/cnh-proposal-diagnostics-20260929/sanpo/results.json'
    manifest = a.repo/'artifacts.local/work/ba-nfo-20260919/prepared-manifest.json'
    root = a.repo/'artifacts.local/datasets/sanpo-synthetic-ba-nfo'
    out = a.repo/'artifacts.local/work/cnh-attribution-diagnostic-20260929/sanpo'
    assert (out/'DECISIONS.md').exists()
    previous = json.loads(old.read_text())
    scenes = sorted(s for s,m in previous['metadata'].items() if m['camera']==['camera_chest'])
    rows = [r for r in json.loads(manifest.read_text()) if r.get('source')=='sanpo' and r['scene'] in scenes]
    assert len(scenes)==34 and len(rows)==1700
    by_scene = {s:sorted([r for r in rows if r['scene']==s],key=lambda r:r['rgb']) for s in scenes}
    meta = {}
    for s in scenes:
        c = previous['metadata'][s]['intrinsics']
        pp = next((root/s).glob('*/camera_poses.csv'))
        poses = np.genfromtxt(pp,delimiter=',',skip_header=1,usecols=range(1,8))
        meta[s] = (c, rays(c), poses[:,:3], Rotation.from_quat(poses[:,3:]).as_matrix())
    checks=[]
    for s in scenes[:6]:
        c, ray, pos, rot = meta[s]
        for t in [0,10,20]:
            da=depth(root/by_scene[s][t]['depth'])[4::8,4::8].astype(float)
            db=depth(root/by_scene[s][t+1]['depth'])
            compare={}
            for h in ['Z','RADIAL']:
                x=xyz(da,ray,h)
                b=(x@rot[t].T+pos[t]-pos[t+1])@rot[t+1]
                z=b[...,2]
                u=np.rint(c['fx']*b[...,0]/np.maximum(z,.001)+c['cx'])
                v=np.rint(c['fy']*b[...,1]/np.maximum(z,.001)+c['cy'])
                good=np.isfinite(da)&(da>0)&(da<=20)&np.isfinite(u)&np.isfinite(v)&(z>.1)
                good &= (u>=0)&(u<c['image_width'])&(v>=0)&(v<c['image_height'])
                ui=np.nan_to_num(u,nan=0,posinf=0,neginf=0).clip(0,c['image_width']-1).astype(int)
                vi=np.nan_to_num(v,nan=0,posinf=0,neginf=0).clip(0,c['image_height']-1).astype(int)
                observed=db[vi,ui].astype(float)
                good &= np.isfinite(observed)&(observed>0)&(observed<=20)
                if h=='RADIAL':
                    observed/=np.sqrt(1+((ui-c['cx'])/c['fx'])**2+((vi-c['cy'])/c['fy'])**2)
                compare[h]=(good,np.abs(z-observed)/np.maximum(observed,.1))
            common=compare['Z'][0]&compare['RADIAL'][0]
            record=dict(scene=s,frame=t)
            for name,mask in [('full',common),('off_axis',common&(np.linalg.norm(ray[...,:2],axis=-1)>=.35))]:
                record[name]=dict(pixels=int(mask.sum()),**{h:float(np.median(compare[h][1][mask])) if mask.any() else None for h in compare})
            checks.append(record)
    totals={h:collections.Counter() for h in ['Z','RADIAL']}
    supports={h:{n:set() for n in [1,2,3]} for h in totals}
    angles=[]
    with (out/'frames.jsonl').open('w') as dest:
        for i,r in enumerate(rows):
            c,ray,pos,rot=meta[r['scene']]
            t=int(Path(r['rgb']).stem)
            rr=rot[t]
            angles.append(float(rr[2,1]))
            d=depth(root/r['depth'])[4::8,4::8].astype(float)
            tx=np.tan(np.deg2rad(22.5))*16/np.sqrt(337)
            ty=tx*9/16
            fov=(abs(ray[...,0])<=tx)&(abs(ray[...,1])<=ty)
            valid=np.isfinite(d)&(d>0)
            forward=rr[:,2].copy();forward[2]=0
            forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,0,1])
            for hyp in totals:
                points=xyz(d,ray,hyp)@rr.T
                z=points[...,2]
                ground_candidates=valid&(d<=6)&(z>=-2.2)&(z<=-.6)&((ray@rr.T)[...,2]<0)
                hist,bins=np.histogram(z[ground_candidates],bins=np.linspace(-2.2,-.6,33))
                k=int(hist.argmax())
                floor_ok=bool(hist[k]>=.005*d.size)
                floor=float((bins[k]+bins[k+1])/2)
                along=points@forward
                lateral=points@right
                corridor=valid&fov&(abs(lateral)<=.6)&(z>=-.55)&(z<=.55)&(along>=.3)
                row=dict(id=r['id'],scene=r['scene'],hypothesis=hyp,sampled_pixels=d.size,
                    fov_pixels=int(fov.sum()),floor_evaluable=int(floor_ok),floor_relative_height=floor if floor_ok else None)
                for n in [1,2,3]:
                    raw=corridor&(along<=n)
                    kept=raw&(abs(z-floor)>.15) if floor_ok else np.zeros_like(raw)
                    row[f'raw_corridor_pixels_{n}m']=int(raw.sum())
                    row[f'floor_excluded_corridor_pixels_{n}m']=int(kept.sum())
                    hit=bool(floor_ok and kept.sum()>=.001*fov.sum())
                    row[f'support_frames_{n}m']=int(hit)
                    if hit:supports[hyp][n].add(r['scene'])
                for k,v in row.items():
                    if isinstance(v,int):totals[hyp][k]+=v
                dest.write(json.dumps(row)+'\n')
            if (i+1)%250==0:print('chest frames',i+1,flush=True)
    reductions=[]
    for r in checks:
        z,rad=r['off_axis']['Z'],r['off_axis']['RADIAL']
        if z is not None and rad is not None: reductions.append((rad-z)/max(rad,1e-12))
    z_wins=sum(x>0 for x in reductions)
    preference='Z_EMPIRICAL_ONLY' if z_wins>=.75*len(reductions) and np.median(reductions)>=.2 else 'AMBIGUOUS'
    # Symmetric rule for radial, without changing any pair or criterion.
    radial_reduction=[(r['off_axis']['Z']-r['off_axis']['RADIAL'])/max(r['off_axis']['Z'],1e-12) for r in checks if r['off_axis']['Z'] is not None]
    if sum(x>0 for x in radial_reduction)>=.75*len(radial_reduction) and np.median(radial_reduction)>=.2:preference='RADIAL_EMPIRICAL_ONLY'
    coverage=all(totals[h]['support_frames_2m']>=30 and len(supports[h][2])>=3 and totals[h]['floor_evaluable']>=.9*len(rows) for h in totals)
    result=dict(decision='CONDITIONAL_GO_GEOMETRY_ONLY' if coverage else 'STOP_AS_PRINCIPAL_CHEST_NEAR_SOURCE',
        frames=len(rows),sessions=len(scenes),depth_source_contract='UNRESOLVED',depth_empirical_preference=preference,
        depth_check_pairs=len(checks),depth_check_Z_wins=z_wins,median_offaxis_relative_error_reduction_Z=float(np.median(reductions)),
        checks=checks,totals={h:dict(v) for h,v in totals.items()},
        support_sessions={h:{str(n):len(s) for n,s in v.items()} for h,v in supports.items()},
        camera_down_worldZ_quantiles=np.quantile(angles,[0,.5,1]).tolist(),
        decisions_sha256=sha(out/'DECISIONS.md'),script_sha256=sha(__file__),input_manifest_sha256=sha(manifest),prior_results_sha256=sha(old),
        backend='TASK_NOT_GPU_SUITABLE; numpy CPU geometry and gzip I/O; no CNH/RGB models')
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
