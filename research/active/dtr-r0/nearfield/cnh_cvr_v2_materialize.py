"""Reuse cached CNH and deterministic pose metadata for frozen CVR v2."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from cnh_cvr_pilot import BASE, motion_metadata, relative_transforms
from cnh_cvr_projection import Projector, SHAPE, SUB, EDGE, WIDTH

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-cvr-v2-20260929'
SPLITS={'train':range(1000,1096),'calib':range(2000,2024),'evaluation':range(3000,3048)}


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8');temp.replace(path)


class BatchedProjector(Projector):
    @torch.no_grad()
    def sequence(self,z,transforms):
        t=torch.as_tensor(transforms,dtype=torch.float64,device=self.device)
        p=torch.bmm(self.points[None]-t[:,None,:3,3],t[:,:3,:3])
        radius=torch.linalg.vector_norm(p,dim=2)
        xy=p[:,:,:2]/p[:,:,2:3].clamp_min(1e-30)
        ij=torch.floor((xy+EDGE)/(2*EDGE)*8).long()
        bins=torch.floor(radius/WIDTH).long()
        valid=(p[:,:,2]>0)&(ij>=0).all(2)&(ij<8).all(2)&(bins>=0)&(bins<16)
        index=(ij[:,:,1].clamp(0,7)*8+ij[:,:,0].clamp(0,7))*16+bins.clamp(0,15)
        weights=valid*self.voxel_volume/(SUB**3)/self.volumes[index]
        values=torch.as_tensor(z,dtype=torch.float64,device=self.device).reshape(len(z),-1)
        mass=torch.gather(values,1,index)*weights
        evidence=mass.reshape(len(z),-1,SUB**3).sum(2).reshape(len(z),*SHAPE).float()
        coverage=valid.reshape(len(z),-1,SUB**3).double().mean(2).reshape(len(z),*SHAPE).float()
        # Preserve float32 sequential accumulation order from v1.
        total=torch.zeros_like(evidence[0]);count=torch.zeros_like(total)
        for e,c in zip(evidence,coverage):total+=e;count+=c
        return torch.stack([total,count,evidence[-1]])


def benchmark():
    with np.load(BASE/'features/train/unit1000.npz') as z:values=z['z1'][:16]
    p,q=Projector(),BatchedProjector()
    sensor,travel,noisy=motion_metadata(1000,0)
    errors=[]
    for f in [3,7,15]:
        window=values[max(0,f-7):f+1];t=relative_transforms(sensor,travel,noisy,f)
        a=p.sequence(window,t);b=q.sequence(window,t)
        errors.append(float((a-b).abs().max()))
    timings={}
    for name,method in [('serial',p),('batched',q)]:
        torch.cuda.synchronize();start=time.monotonic()
        for _ in range(5):method.sequence(values[-8:],relative_transforms(sensor,travel,noisy,15))
        torch.cuda.synchronize();timings[name]=(time.monotonic()-start)/5
    selected='batched' if max(errors)<=1e-6 and timings['batched']<timings['serial'] else 'serial'
    save(OUT/'projection_benchmark.json',dict(training_only=True,max_abs_errors=errors,timings_s=timings,selected=selected))
    del p,q;torch.cuda.empty_cache()
    return BatchedProjector() if selected=='batched' else Projector()


def main():
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.45)
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    source_paths=[Path(__file__),Path(__file__).with_name('cnh_cvr_pilot.py'),Path(__file__).with_name('cnh_cvr_projection.py'),OUT/'CVR_V2_PLAN.md']
    inputs=[BASE/'features'/s/f'unit{u}.npz' for s,ids in SPLITS.items() for u in ids]
    request=dict(sources={str(p):sha(p) for p in source_paths},inputs={str(p):sha(p) for p in inputs},
        device=torch.cuda.get_device_name(),dtype='float64 geometry / float32 accumulation / float16 cache',
        pid=os.getpid(),free_disk_bytes=shutil.disk_usage(OUT).free)
    requestpath=OUT/'materialization_request.json'
    if requestpath.exists():
        prior=json.loads(requestpath.read_text());assert prior['sources']==request['sources'] and prior['inputs']==request['inputs']
    else:save(requestpath,request)
    assert shutil.disk_usage(OUT).free>6*1024**3
    projector=benchmark()
    for split,units in SPLITS.items():
        folder=OUT/'data'/split;folder.mkdir(parents=True,exist_ok=True)
        frames=np.arange(3,16) if split=='train' else np.arange(11,16)
        n=len(units)*22*len(frames)
        specifications={'features':((n,3,*SHAPE),np.float16),'zone':((n,2,8,8,16),np.float16),'support':((n,2,8,8,16),bool)}
        arrays={}
        for name,(shape,dtype) in specifications.items():
            path=folder/f'{name}.npy'
            arrays[name]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
            assert arrays[name].shape==shape and arrays[name].dtype==dtype
        allmeta={k:[] for k in ['unit','config','frame','family','labels']}
        donepath=folder/'progress.json'
        completed=json.loads(donepath.read_text())['completed_units'] if donepath.exists() else []
        for ui,u in enumerate(units):
            with np.load(BASE/'features'/split/f'unit{u}.npz') as z:
                d={k:z[k] for k in ['z1','z4','support','scene','frame','labels','family']}
            for config in range(22):
                ids=np.flatnonzero(d['scene']==config)
                assert np.array_equal(d['frame'][ids],np.arange(16))
                selected=ids[frames]
                offset=(ui*22+config)*len(frames)
                dest=slice(offset,offset+len(frames))
                if u not in completed:
                    sensor,travel,noisy=motion_metadata(u,config)
                    for fi,f in enumerate(frames):
                        voxel=projector.sequence(d['z1'][ids[max(0,f-7):f+1]],relative_transforms(sensor,travel,noisy,int(f)))
                        assert bool(torch.isfinite(voxel).all())
                        arrays['features'][offset+fi]=voxel.cpu().numpy().astype(np.float16)
                    arrays['zone'][dest]=np.stack([d['z4'][selected],d['z1'][selected]],axis=1)
                    arrays['support'][dest]=d['support'][selected,2:4]
                allmeta['unit'].extend([u]*len(frames));allmeta['config'].extend([config]*len(frames))
                allmeta['frame'].extend(frames.tolist());allmeta['family'].extend([str(d['family'][config])]*len(frames))
                allmeta['labels'].extend(d['labels'][selected,2:4].tolist())
            if u not in completed:
                for arr in arrays.values():arr.flush()
                completed.append(u);save(donepath,dict(completed_units=completed,elapsed_s=time.monotonic()-start))
            save(OUT/'progress.json',dict(stage='materialize',split=split,unit=u,completed_units=len(completed),total_units=len(units),elapsed_s=time.monotonic()-start))
            print(split,u,round(time.monotonic()-start,1),flush=True)
        np.savez_compressed(folder/'metadata.npz',**{k:np.asarray(v) for k,v in allmeta.items()})
        for arr in arrays.values():arr.flush()
        del arrays
        save(OUT/'data'/f'complete_{split}.json',dict(split=split,samples=n,units=len(units),frames=frames.tolist(),
            hashes={p.name:sha(p) for p in folder.glob('*.np*')}))
    save(OUT/'data/complete.json',dict(status='complete',elapsed_s=time.monotonic()-start,training_seconds=0))
    save(OUT/'materialization_terminal.json',dict(status='complete',elapsed_s=time.monotonic()-start,training_seconds=0,pid=os.getpid()))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True)
        save(OUT/'materialization_terminal.json',dict(status='failed',error=repr(e),pid=os.getpid()))
        raise
