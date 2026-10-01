"""Train-only symmetric sub-angle geometry targets; deployment stays Readout."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import cnh_corridor_retrain as T
import cnh_corridor_diagnostic as D
import cnh_proposal_attribution_scenes as S
from cnh_corridor_statistics import paired_cluster_ber

HERE=Path(__file__).resolve().parent
ROOT=D.ROOT
BASE=D.OUT
OUT=ROOT/'artifacts.local/work/cnh-subzone-aux-20260929'
L=T.frozen_module()

def targets_for_geometry(boxes,poses):
    directions,weights=S.ray_grid();w=weights.reshape(8,8,16,16)
    result=[]
    for pose in poses:
        hit=S.raycast_boxes(pose[:3,3],directions@pose[:3,:3].T,boxes)
        distance=hit['distance'].reshape(8,8,16,16)
        bins=np.floor(np.where(np.isfinite(distance),distance,100.)/(8*.0375348)).astype(int)
        bins=np.clip(bins,0,16)
        target=np.zeros((4,8,8,17),np.float32)
        for qy in range(2):
            for qx in range(2):
                sl=(slice(None),slice(None),slice(qy*8,(qy+1)*8),slice(qx*8,(qx+1)*8))
                ww=w[sl];bb=bins[sl];den=ww.sum((-2,-1))
                for b in range(17):target[qy*2+qx,:,:,b]=((bb==b)*ww).sum((-2,-1))/den
        result.append(target)
    result=np.asarray(result)
    assert np.allclose(result.sum(-1),1.,atol=1e-6)
    return result

class AuxiliaryReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout=L.Readout()
        self.angle_head=nn.Conv3d(32,4,1)
        self.escape_head=nn.Conv2d(32,4,1)
        self._features=None
        self.readout.body.register_forward_hook(self._capture)
    def _capture(self,module,args,result):self._features=result
    def forward(self,x,support):
        score=self.readout(x,support)
        f=self._features
        fine=self.angle_head(f)
        outside=self.escape_head(f.mean(-1))[...,None]
        return score,torch.cat([fine,outside],dim=-1)

def generate():
    folder=OUT/'targets';folder.mkdir(parents=True,exist_ok=True)
    identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'CNH_SUBZONE_AUX_PLAN_20260929.md',HERE/'cnh_proposal_attribution_scenes.py']}
    request=OUT/'target_request.json'
    if request.exists():assert json.loads(request.read_text())['identity']==identity
    else:
        D.old.save(request,dict(identity=identity,train_units=D.SPLITS['train'],input_fields=['boxes','sensor_poses','public_angular_grid']))
        snap=OUT/'source';snap.mkdir(exist_ok=True)
        for name in identity:(snap/name).write_bytes((HERE/name).read_bytes())
    start=time.time()
    for u in D.SPLITS['train']:
        dest=folder/f'unit{u}.npz'
        if dest.exists():continue
        scenes=S.make_scenes(u)
        target=np.concatenate([targets_for_geometry(sc['boxes'],sc['poses']) for sc in scenes])
        np.savez_compressed(dest,target=target.astype(np.float16),unit=u,scene=np.repeat(np.arange(22),16),frame=np.tile(np.arange(16),22))
        D.old.save(OUT/'target_progress.json',dict(unit=u,elapsed_s=time.time()-start))
        print('target',u,round(time.time()-start,1),flush=True)
    D.old.save(OUT/'target_terminal.json',dict(status='complete',elapsed_s=time.time()-start))

def train():
    torch.set_num_threads(2);assert torch.cuda.is_available()
    paths=[BASE/'features/train'/f'unit{u}.npz' for u in D.SPLITS['train']]
    x,support,y,meta=T.load_train(paths)
    targets=[];hashes=[]
    for u in D.SPLITS['train']:
        p=OUT/'targets'/f'unit{u}.npz'
        with np.load(p) as z:
            assert int(z['unit'])==u and np.array_equal(z['scene'],np.repeat(np.arange(22),16))
            mask=z['frame']>=3
            target=z['target'][mask].astype(np.float32)
            target/=target.sum(-1,keepdims=True)
            targets.append(target)
        hashes.append(dict(unit=u,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    target=np.concatenate(targets)
    assert len(target)==len(x)==27456
    folder=OUT/'models';folder.mkdir(exist_ok=True)
    assert not list(folder.glob('model_seed*.pt')),'do not overwrite frozen run'
    D.old.save(folder/'request.json',dict(recipe=T.RECIPE,lambda_aux=.1,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features=meta,targets=hashes,train_samples=len(x)))
    start=time.time();checkpoints=[]
    for seed in [0,1,2]:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=AuxiliaryReadout().cuda();rng=np.random.default_rng(seed)
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20);history=[]
        for ep in range(20):
            model.train();total=main_total=aux_total=0.;count=0
            for batch in T.training_batches(rng.permutation(len(x)),256,64):
                opt.zero_grad(set_to_none=True)
                for ids,weight in batch:
                    xb=torch.as_tensor(x[ids],device='cuda');sb=torch.as_tensor(support[ids],device='cuda');yb=torch.as_tensor(y[ids],device='cuda');tb=torch.as_tensor(target[ids],device='cuda')
                    score,logits=model(xb,sb)
                    main=nn.functional.binary_cross_entropy_with_logits(score,yb)
                    aux=-(tb*nn.functional.log_softmax(logits,dim=-1)).sum(-1).mean()
                    loss=main+.1*aux
                    assert torch.isfinite(loss)
                    (loss*weight).backward();total+=float(loss.detach())*len(ids);main_total+=float(main.detach())*len(ids);aux_total+=float(aux.detach())*len(ids);count+=len(ids)
                opt.step()
            sched.step();history.append(dict(epoch=ep+1,loss=total/count,main=main_total/count,aux=aux_total/count))
            D.old.save(folder/f'history_seed{seed}.json',history)
            print('aux',seed,ep+1,history[-1],flush=True)
        path=folder/f'model_seed{seed}.pt';torch.save({k:v.detach().cpu() for k,v in model.readout.state_dict().items()},path)
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},folder/f'train_only_seed{seed}.pt')
        checkpoints.append(dict(seed=seed,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),deployed_parameters=sum(p.numel() for p in model.readout.parameters()),training_only_parameters=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.readout.parameters())))
        del model,opt,sched;torch.cuda.empty_cache()
    D.old.save(folder/'receipt.json',dict(status='complete',models=checkpoints,elapsed_s=time.time()-start))

def infer():
    torch.set_num_threads(2)
    models=T.load_models([OUT/'models'/f'model_seed{s}.pt' for s in [0,1,2]])
    from cnh_learned_memory_fusion import causal_ewma
    folder=OUT/'predictions';folder.mkdir(exist_ok=True)
    for split in ['calib','evaluation']:
        for u in D.SPLITS[split]:
            with np.load(BASE/'features'/split/f'unit{u}.npz') as z:d={k:z[k] for k in z.files}
            score=T.predict(models,d['z4'],d['z1'],d['support'])
            score=causal_ewma(score,d['scene'],d['frame'],alpha=.5,window=5).reshape(22,16,6)[:,-1,[2,3]]
            np.savez_compressed(folder/f'unit{u}.npz',scores=score)

def evaluate():
    from sklearn.metrics import average_precision_score
    data={};aux={}
    for split in ['calib','evaluation']:
        for u in D.SPLITS[split]:
            with np.load(BASE/'predictions'/f'unit{u}.npz') as z:data[u]={k:z[k] for k in z.files}
            with np.load(OUT/'predictions'/f'unit{u}.npz') as z:aux[u]=z['scores']
    cat=lambda ids,key:np.concatenate([data[u][key] for u in ids])
    cal=D.SPLITS['calib'];ev=D.SPLITS['evaluation'];unitids=np.repeat(ev,22)
    cy=cat(cal,'labels').astype(bool);yy=cat(ev,'labels').astype(bool)
    original_thresholds=json.loads((BASE/'results.json').read_text())['thresholds']
    thresholds={};rows=[];comparisons=[];ledger=[]
    family=cat(ev,'family');margin=cat(ev,'margin');group=cat(ev,'group')
    for qi,name in enumerate(['HEAD','BODY']):
        cs=np.concatenate([aux[u] for u in cal])[:,qi];score=np.concatenate([aux[u] for u in ev])[:,qi]
        t=D.old.threshold(cs[~cy[:,qi]])
        thresholds[name]=t
        base_score=cat(ev,'scores')[:,2,qi]
        scores={'BASE':base_score,'AUX':score}
        pred={'BASE':base_score>=original_thresholds[name+'|A2_RETRAIN'],'AUX':score>=t}
        subsets={'all':np.ones(len(yy),bool),'mixed_surface':family=='mixed_surface','general':family=='general',
                 'inside_0_5cm':(group==qi)&(margin<0)&(abs(margin)<=.05),'outside_0_5cm':(group==qi)&(margin>0)&(margin<=.05)}
        for policy in ['strict','ignore5cm']:
            keep=np.ones(len(yy),bool) if policy=='strict' else ~((group==qi)&(abs(margin)<=.05))
            for subset,mask0 in subsets.items():
                mask=mask0&keep
                if not mask.any():continue
                metrics={a:D.old.binary_metrics(yy[mask,qi],p[mask]) for a,p in pred.items()}
                for a,m in metrics.items():
                    y=yy[mask,qi];ap=float(average_precision_score(y,scores[a][mask])) if 0<y.sum()<len(y) else None
                    rows.append(dict(group=name,policy=policy,subset=subset,arm=a,ap=ap,**m))
                boot=paired_cluster_ber(yy[mask,qi],pred['BASE'][mask],pred['AUX'][mask],unitids[mask])
                comparisons.append(dict(group=name,policy=policy,subset=subset,bootstrap=boot))
        for i in range(len(yy)):ledger.append(dict(unit=int(unitids[i]),config=i%22,group=name,family=str(family[i]),margin=float(margin[i]),target_group=int(group[i]),label=int(yy[i,qi]),base=int(pred['BASE'][i]),aux=int(pred['AUX'][i])))
    decisions={}
    for g in ['HEAD','BODY']:
        lookup=lambda subset,arm:next(r for r in rows if r['group']==g and r['policy']=='strict' and r['subset']==subset and r['arm']==arm)
        c=next(c['bootstrap'] for c in comparisons if c['group']==g and c['policy']=='strict' and c['subset']=='all')
        b=lookup('general','BASE');a=lookup('general','AUX')
        decisions[g]=dict(advance=bool(c['point']<=-.03 and c['ci'][1]<0 and lookup('mixed_surface','AUX')['ber']<=lookup('mixed_surface','BASE')['ber'] and lookup('inside_0_5cm','AUX')['tpr']>=lookup('inside_0_5cm','BASE')['tpr'] and a['tpr']>=b['tpr']-.02 and a['fpr']<=b['fpr']+.02),delta_ber=c['point'],ci95=c['ci'])
    result=dict(scope='consumed Development',thresholds=thresholds,rows=rows,comparisons=comparisons,decisions=decisions)
    D.old.save(OUT/'results.json',result)
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger),encoding='utf-8')
    print(json.dumps(decisions,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['generate','train','infer','evaluate','all'],required=True);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for stage in (['generate','train','infer','evaluate'] if a.stage=='all' else [a.stage]):globals()[stage]()
