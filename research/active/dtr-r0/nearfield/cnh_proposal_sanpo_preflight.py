"""Local-only SANPO metadata/native-depth audit; never synthesizes sensor inputs."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    work = a.repo / 'artifacts.local/work/ba-nfo-20260919'
    root = a.repo / 'artifacts.local/datasets/sanpo-synthetic-ba-nfo'
    out = a.repo / 'artifacts.local/work/cnh-proposal-diagnostics-20260929/sanpo'
    criteria = out / 'DECISIONS.md'
    assert criteria.is_file(), 'freeze criteria before audit'
    rows = [r for r in json.loads((work/'prepared-manifest.json').read_text()) if r['source']=='sanpo']
    scenes = sorted(set(r['scene'] for r in rows))
    totals = collections.Counter()
    meta = {}
    masks = {}
    steps = []
    for scene in scenes:
        d = json.loads((root/scene/'description.json').read_text())
        c = d['session_camera_details'][0]['left_camera_params']
        poses = list(csv.DictReader(next((root/scene).glob('*/camera_poses.csv')).open()))
        xyz = np.array([[float(r[k]) for k in ['pos_x','pos_y','pos_z']] for r in poses])
        delta = np.linalg.norm(np.diff(xyz,axis=0),axis=1)
        steps.extend(delta.tolist())
        q = np.array([[float(r[k]) for k in ['q_x','q_y','q_z','q_w']] for r in poses])
        meta[scene] = dict(split=next(r['split'] for r in rows if r['scene']==scene),
            camera=d['session_camera_location'], fps=d['session_camera_details'][0]['fps'],
            intrinsics=c, pose_rows=len(poses), max_quaternion_norm_error=float(np.max(abs(np.linalg.norm(q,axis=1)-1))),
            pose_step_median=float(np.median(delta)), frames=0, supported_frames=0)
    frame_file = out/'frames.jsonl'
    with frame_file.open('w',encoding='utf-8') as f:
        for i,r in enumerate(rows):
            data = np.frombuffer(gzip.decompress((root/r['depth']).read_bytes()),dtype='<f2')
            h,w = map(int,data[:2])
            dep = data[2:].reshape(h,w).astype(np.float32)
            with Image.open(root/r['rgb']) as im:
                assert im.size == (w,h)
            c = meta[r['scene']]['intrinsics']
            assert (w,h)==(c['image_width'],c['image_height'])
            assert c['fx']>0 and c['fy']>0 and 0<c['cx']<w and 0<c['cy']<h
            key=tuple(c[k] for k in ['fx','fy','cx','cy','image_width','image_height'])
            if key not in masks:
                # Rectangular pinhole FOV: diagonal 45 degrees, aspect 16:9.
                tx=math.tan(math.radians(22.5))*16/math.sqrt(337)
                ty=math.tan(math.radians(22.5))*9/math.sqrt(337)
                x=abs((np.arange(w)-c['cx'])/c['fx'])<=tx
                y=abs((np.arange(h)-c['cy'])/c['fy'])<=ty
                masks[key]=y[:,None]&x[None,:]
            mask=masks[key]
            valid=np.isfinite(dep)&(dep>0)
            item=dict(id=r['id'],scene=r['scene'],split=r['split'],pixels=h*w,
                valid_pixels=int(valid.sum()),fov_pixels=int(mask.sum()),fov_valid_pixels=int((valid&mask).sum()))
            for limit in [.3,1,2,3]:
                near=valid&(dep<=limit)
                item[f'full_le_{limit}m']=int(near.sum())
                item[f'fov_le_{limit}m']=int((near&mask).sum())
                item[f'fov_frames_ge_0.1pct_le_{limit}m']=int(item[f'fov_le_{limit}m']>=.001*item['fov_pixels'])
            for k,v in item.items():
                if isinstance(v,int): totals[k]+=v
            meta[r['scene']]['frames']+=1
            meta[r['scene']]['supported_frames']+=item['fov_frames_ge_0.1pct_le_3m']
            f.write(json.dumps(item)+'\n')
            if (i+1)%500==0: print('audited',i+1,flush=True)
    hashes=collections.Counter(r['rgb_sha256'] for r in rows)
    split_scenes={s:{r['scene'] for r in rows if r['split']==s} for s in ['train','val','test']}
    support_sessions=sum(m['supported_frames']>0 for m in meta.values())
    coverage_pass=totals['fov_frames_ge_0.1pct_le_3m']>=30 and support_sessions>=3
    result=dict(decision='CONDITIONAL_GO_DIAGNOSTIC_ONLY' if coverage_pass else 'NO_GO_FOR_CURRENT_SUBSET',
        criteria_sha256=hashlib.sha256(criteria.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_manifest_sha256=hashlib.sha256((work/'prepared-manifest.json').read_bytes()).hexdigest(),
        frames=len(rows),sessions=len(scenes),support_sessions=support_sessions,
        split_frames=dict(collections.Counter(r['split'] for r in rows)),
        split_sessions={k:len(v) for k,v in split_scenes.items()},
        session_overlap=sum(len(split_scenes[x]&split_scenes[y]) for x,y in [('train','val'),('train','test'),('val','test')]),
        duplicate_rgb_hash_extra=sum(v-1 for v in hashes.values()),
        rgb_hash_scope='retained manifest identity, not rehashed raw RGB',
        totals=dict(totals),pose_step_quantiles=dict(zip(['min','p50','p90','max'],map(float,np.quantile(steps,[0,.5,.9,1])))),
        pose_step_count=len(steps),metadata=meta,
        semantics=dict(units='metres per retained official README',depth_axis='UNRESOLVED optical-Z versus radial',
            coordinates='UNRESOLVED world axes, pose direction and camera axes',visibility='single depth value per pixel; occluded surfaces unavailable',
            independence='session disjointness verified, physical layout independence UNKNOWN; consumed Development'),
        backend='TASK_NOT_GPU_SUITABLE: gzip/image headers and numpy counting on CPU')
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='metadata'},indent=2))


if __name__=='__main__':
    main()
