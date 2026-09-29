"""CVR frozen pilot pretraining checks on reused z1, without scene generation."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from cnh_cvr_projection import Projector, query_masks
from cnh_corridor_late_fusion import ranks, threshold, metrics

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work'
OUT=WORK/'cnh-cvr-pilot-20260929'
BASE=WORK/'cnh-corridor-retrain-20260929'
SOURCE=WORK/'cnh-track-a-v5-20260928/data/source'
FAMILIES=('boundary','mixed_surface','sidewall','general')


def save(name,obj):
    p=OUT/name
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8');temp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rotation(degrees,axis):
    a=np.deg2rad(degrees);c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]]) if axis=='y' else np.array([[1,0,0],[0,c,-s],[0,s,c]])


def motion_metadata(unit,config):
    """Replay only frozen motion schedule; never construct boxes or labels."""
    sys.path.insert(0,str(SOURCE)) if str(SOURCE) not in sys.path else None
    from cnh_track_a_readout import noisy_poses
    assert Path(sys.modules['cnh_track_a_readout'].__file__).resolve().parent==SOURCE.resolve()
    mode=unit%3
    yaw=np.linspace(-20.,0.,16) if mode==2 else np.zeros(16)
    position=np.zeros((16,3))
    for i in range(1,16):
        position[i]=position[i-1]+.16*(rotation((yaw[i-1]+yaw[i])/2,'y')@np.array([0.,0.,1.]))
    position-=position[-1]
    sensor_yaw=20*np.sin(np.linspace(-np.pi/2,np.pi/2,16)) if mode==1 else np.full(16,15. if mode==0 else 0.)
    sensor=np.repeat(np.eye(4)[None],16,0)
    travel=sensor.copy()
    for i in range(16):
        travel[i,:3,:3]=rotation(yaw[i],'y');travel[i,:3,3]=position[i]
        sensor[i,:3,:3]=rotation(yaw[i]+sensor_yaw[i],'y')@rotation(-10,'x');sensor[i,:3,3]=position[i]
    noisy=noisy_poses(sensor,2026092900+unit*1000+config*10+7,dt=.2)
    return sensor,travel,noisy


def relative_transforms(sensor,travel,noisy,frame):
    q=np.linalg.inv(travel[frame])@sensor[frame]
    return q@np.linalg.inv(noisy[frame])@noisy[max(0,frame-7):frame+1]


class CVR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body=torch.nn.Sequential(torch.nn.Conv3d(5,16,3,stride=2,padding=1),torch.nn.GELU(),
            torch.nn.Conv3d(16,32,3,padding=1),torch.nn.GELU(),torch.nn.Conv3d(32,32,3,padding=1),torch.nn.GELU())
        self.emb=torch.nn.Embedding(2,8)
        self.head=torch.nn.Sequential(torch.nn.Linear(72,32),torch.nn.GELU(),torch.nn.Linear(32,1))

    def forward(self,x):
        f=self.body(x)
        mask=torch.nn.functional.adaptive_avg_pool3d(x[:,3:5],f.shape[2:])[:,:,None]
        ff=f[:,None]
        mx=ff.masked_fill(mask==0,-1e4).flatten(3).max(-1).values
        mean=(ff*mask).flatten(3).sum(-1)/mask.flatten(3).sum(-1).clamp_min(1e-8)
        emb=self.emb.weight[None].expand(len(x),-1,-1)
        return self.head(torch.cat([mx,mean,emb],-1)).squeeze(-1)


def precheck():
    mechanical=json.loads((OUT/'mechanics.json').read_text())
    assert mechanical['status']=='PASS' and all(mechanical['checks'].values()), 'A failed; do not evaluate B'
    assert not (OUT/'precheck.json').exists(), 'frozen precheck already consumed'
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.5)
    start=time.monotonic()
    paths=[Path(__file__),Path(__file__).with_name('cnh_cvr_projection.py'),OUT/'CVR_PLAN.md',
           SOURCE/'cnh_track_a_readout.py',BASE/'source/cnh_proposal_attribution_scenes.py',
           Path(__file__).with_name('cnh_corridor_late_fusion.py')]
    for split,ids in [('calib',range(2000,2024)),('evaluation',range(3000,3048))]:
        for unit in ids:
            paths.extend([BASE/'features'/split/f'unit{unit}.npz',BASE/'predictions'/f'unit{unit}.npz'])
    identity={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save('request.json',dict(hashes=identity,started_unix=time.time(),device=torch.cuda.get_device_name(),
        backend='cuda float64 geometric integration; float32 outputs',training_budget_seconds=7200))
    projector=Projector()
    masks=torch.tensor(query_masks(),device='cuda')
    rows=[]
    for split,ids in [('calib',range(2000,2024)),('evaluation',range(3000,3048))]:
        for unit in ids:
            with np.load(BASE/'features'/split/f'unit{unit}.npz') as z:
                d={k:z[k] for k in ['z1','labels','scene','frame','family','margin','group']}
            with np.load(BASE/'predictions'/f'unit{unit}.npz') as z:
                old={k:z[k] for k in z.files}
            final=[]
            for config in range(22):
                keep=d['scene']==config
                assert np.array_equal(d['frame'][keep],np.arange(16))
                values=d['z1'][keep]
                sensor,travel,noisy=motion_metadata(unit,config)
                features=projector.sequence(values[-8:],relative_transforms(sensor,travel,noisy,15))
                scores=(features[0][None]*masks).flatten(1).sum(1).cpu().numpy()
                final.append(features.cpu().numpy().astype(np.float16))
                for q,group in enumerate(['HEAD','BODY']):
                    label=int(d['labels'][keep][-1,q+2])
                    assert label==int(old['labels'][config,q])
                    rows.append(dict(split=split,unit=unit,config=config,group=group,family=str(d['family'][config]),
                        target_group=int(d['group'][config]),margin=float(d['margin'][config]),label=label,
                        score=float(scores[q]),motion=float(old['scores'][config,4,q])))
            dest=OUT/'last_frame_voxels'/split;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'unit{unit}.npz',features=np.array(final),unit=unit)
            save('progress.json',dict(stage='precheck_projection',split=split,unit=unit,elapsed_s=time.monotonic()-start))
            print(split,unit,round(time.monotonic()-start,2),flush=True)
    correlations=[];evaluated=[];calibration=[]
    for family in ['pooled',*FAMILIES]:
        for group in ['HEAD','BODY']:
            ev=[r for r in rows if r['split']=='evaluation' and r['group']==group and (family=='pooled' or r['family']==family)]
            rho=float(spearmanr([r['score'] for r in ev],[r['motion'] for r in ev]).statistic)
            passed=bool(np.isfinite(rho) and (rho>=.5 if family=='pooled' else rho>0))
            correlations.append(dict(family=family,group=group,n=len(ev),rho=rho,passed=passed))
            if family=='pooled':continue
            cal=[r for r in rows if r['split']=='calib' and r['group']==group and r['family']!=family and not r['label']]
            for arm,key in [('CVR_SUM','score'),('MOTION8_PARTIAL','motion')]:
                ref=[r[key] for r in cal];cr=ranks(ref,ref);er=ranks(ref,[r[key] for r in ev]);k=threshold(cr,len(ref))
                calibration.append(dict(family=family,group=group,arm=arm,threshold=k,fp=int((cr>=k).sum()),negative=len(ref)))
                for r,p in zip(ev,er>=k):r.setdefault('predictions',{})[arm]=int(p)
            evaluated.extend(ev)
    (OUT/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    (OUT/'evaluation_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in evaluated),encoding='utf8')
    count=sum(p.numel() for p in CVR().parameters())
    assert 22417<=count<=67249
    passed=all(r['passed'] for r in correlations)
    result=dict(passed=passed,correlations=correlations,calibration=calibration,parameters=count,
        projection_elapsed_s=time.monotonic()-start,training_elapsed_s=0,trained_models=0,
        main_gates='NOT_RUN' if not passed else 'PENDING_TRAINING',deviations=[])
    save('precheck.json',result)
    for p in paths:assert sha(p)==identity[str(p.relative_to(ROOT))]
    save('terminal.json',dict(status='PRECHECK_PASSED' if passed else 'STOPPED_PRETRAINING_RANK_CHECK',
        training_seconds=0,training_models=0,process_exited=True))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--precheck',action='store_true');args=parser.parse_args()
    if args.precheck:precheck()
