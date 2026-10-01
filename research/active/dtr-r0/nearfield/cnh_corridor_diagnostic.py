"""Expanded Development corridor pilot: disjoint generation, fixed training, scoring."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import cnh_proposal_attribution as old

HERE=Path(__file__).resolve().parent
ROOT=old.ROOT
OUT=ROOT/'artifacts.local/work/cnh-corridor-retrain-20260929'
SPLITS={'train':list(range(1000,1096)), 'calib':list(range(2000,2024)), 'evaluation':list(range(3000,3048))}
ARMS=['SINGLE1','A2_TRANSFER','A2_RETRAIN','MOTION8_HARD','MOTION8_PARTIAL','MOTION8_EXACT']

def point_score(hist,ambient,bias,poses,query_from_sensor,require_all=False):
    hist=hist[-8:];ambient=ambient[-8:];poses=poses[-8:]
    residual=hist-bias;variance=16*ambient[...,None]+np.maximum(bias,0)
    out=[];coverage=[];counts=[]
    for yl,yh in [(-.2,.42),(.42,.9)]:
        xs=np.unique(np.r_[np.arange(-.275,.3,.05),-.299,.299])
        ys=np.unique(np.r_[np.linspace(yl+.025,yh-.025,5),yl+.001,yh-.001])
        zs=np.unique(np.r_[np.arange(.35,3,.1),.301,2.999])
        q=np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),-1).reshape(-1,3)
        local=(q-query_from_sensor[:3,3])@query_from_sensor[:3,:3]
        world=local@poses[-1,:3,:3].T+poses[-1,:3,3]
        total=np.zeros(len(q));var=np.zeros(len(q));seen=np.zeros(len(q),int)
        now=np.linalg.norm(local,axis=1)
        for h,a,p in zip(residual,variance,poses):
            v=(world-p[:3,3])@p[:3,:3];radius=np.linalg.norm(v,axis=1)
            xy=v[:,:2]/np.maximum(v[:,2,None],1e-12)
            ij=np.floor((xy+np.tan(np.pi/8))/(2*np.tan(np.pi/8))*8).astype(int)
            bins=np.floor(radius/(8*.0375348)).astype(int)
            valid=(v[:,2]>0)&(ij>=0).all(1)&(ij<8).all(1)&(bins>=0)&(bins<16)
            ii=np.flatnonzero(valid);cc=ij[ii,0];rr=ij[ii,1];bb=bins[ii]
            gain=(radius[ii]/now[ii])**2
            total[ii]+=gain*h[rr,cc,bb];var[ii]+=gain**2*a[rr,cc,bb];seen[ii]+=1
        good=seen==8 if require_all else seen>=1
        score=total/np.sqrt(np.maximum(var,1e-9))
        out.append(float(score[good].max()) if good.any() else -50.)
        coverage.append(float(good.mean()))
        counts.append(np.bincount(seen,minlength=9))
    return np.asarray(out),np.asarray(coverage),np.asarray(counts)

def generate():
    from cnh_proposal_attribution_scenes import make_scenes,render
    from cnh_corridor_labels import labels_for_all
    mod=old.setup();torch,g,F,L,smooth,noisy_fn,g2,models=mod
    bias=np.load(old.DATA/'readouts-gpu/primary-mount-10-snr6/bias.npy')
    source=[Path(__file__),HERE/'cnh_corridor_labels.py',HERE/'cnh_proposal_attribution_scenes.py',HERE/'CNH_CORRIDOR_RETRAIN_PLAN_20260929.md']
    identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    request=OUT/'generation_request.json'
    if request.exists():assert json.loads(request.read_text())['identity']==identity
    else:
        old.save(request,dict(identity=identity,splits=SPLITS,device=torch.cuda.get_device_name()))
        snap=OUT/'source';snap.mkdir(exist_ok=True)
        for p in source:(snap/p.name).write_bytes(p.read_bytes())
    start=time.time()
    for split,units in SPLITS.items():
        folder=OUT/'features'/split;folder.mkdir(parents=True,exist_ok=True)
        for u in units:
            dest=folder/f'unit{u}.npz'
            if dest.exists():continue
            rows=[]
            for scene in make_scenes(u):
                seed=2026092900+u*1000+scene['config']*10
                obs=render(scene,seed);h,a=obs['hist'],obs['ambient']
                noisy=noisy_fn(scene['poses'],seed+7,dt=.2)
                tq=np.linalg.inv(scene['travel'])@scene['poses']
                z4,z1,sup=F.sequence_features(h,a,bias,tq,noisy)
                labels=labels_for_all(scene['boxes'],scene['travel'])
                if not np.array_equal(labels[:,[2,3]],scene['labels']):
                    closed=labels_for_all(scene['boxes'],scene['travel'],boundary='closed')
                    assert np.array_equal(closed[:,[2,3]],scene['labels'])
                    assert (labels<=closed).all()
                assert np.array_equal(labels[-1,[2,3]],scene['labels'][-1])
                row=dict(z4=z4.astype(np.float16),z1=z1.astype(np.float16),support=sup,labels=labels,
                         scene=np.full(16,scene['config']),frame=np.arange(16),family=scene['family'],
                         margin=scene['margin'],group=scene['group'])
                if split!='train':
                    hard,hardcov,_=point_score(h,a,bias,noisy,tq[-1],True)
                    partial,coverage,counts=point_score(h,a,bias,noisy,tq[-1])
                    exact,_,_=point_score(h,a,bias,scene['poses'],tq[-1])
                    single=g.scan(g.T(h[-1:]-bias),g.T(16*a[-1:,...,None]+np.maximum(bias,0)),torch.as_tensor(sup[-1:],device=g.DEV)).cpu().numpy()[0,[2,3]]
                    row.update(motion=np.array([single,hard,partial,exact]),coverage=coverage,hardcov=hardcov,seen_counts=counts)
                rows.append(row)
            payload={k:np.concatenate([r[k] for r in rows]) for k in ['z4','z1','support','labels','scene','frame']}
            payload.update(unit=u,split=split,query_frame='travel',train_mask=payload['frame']>=3)
            for k in ['family','margin','group']+([] if split=='train' else ['motion','coverage','hardcov','seen_counts']):payload[k]=np.array([r[k] for r in rows])
            np.savez_compressed(dest,**payload)
            old.save(OUT/'generation_progress.json',dict(split=split,unit=u,elapsed_s=time.time()-start))
            print(split,u,round(time.time()-start,1),flush=True)
    old.save(OUT/'generation_terminal.json',dict(status='complete',elapsed_s=time.time()-start))

def infer():
    import cnh_corridor_retrain as train
    mod=old.setup();torch,g,F,L,smooth,noisy_fn,g2,original=mod
    learned=[]
    for seed in range(3):
        m=L.Readout().to(L.DEV);m.load_state_dict(torch.load(OUT/'models'/f'model_seed{seed}.pt',weights_only=True,map_location=L.DEV));learned.append(m.eval())
    predout=OUT/'predictions';predout.mkdir(exist_ok=True)
    for split in ['calib','evaluation']:
        for u in SPLITS[split]:
            with np.load(OUT/'features'/split/f'unit{u}.npz') as z:d={k:z[k] for k in z.files}
            x=np.stack([L.squash(d['z4'].astype(np.float32)),L.squash(d['z1'].astype(np.float32))],1)
            scores=[]
            for models in [original,learned]:
                seedpred=[]
                for net in models:
                    chunks=[]
                    with torch.no_grad():
                        for st in range(0,len(x),48):
                            chunks.append(net(torch.as_tensor(x[st:st+48],device=L.DEV),torch.as_tensor(d['support'][st:st+48],device=L.DEV)).cpu().numpy())
                    seedpred.append(np.concatenate(chunks))
                nn=np.mean(seedpred,axis=0)
                score=smooth(nn,d['scene'],d['frame'],alpha=.5,window=5).reshape(22,16,6)[:,-1,[2,3]]
                scores.append(score)
            final=np.stack([d['motion'][:,0],scores[0],scores[1],d['motion'][:,1],d['motion'][:,2],d['motion'][:,3]],axis=1)
            np.savez_compressed(predout/f'unit{u}.npz',scores=final,labels=d['labels'].reshape(22,16,6)[:,-1,[2,3]],family=d['family'],margin=d['margin'],group=d['group'],coverage=d['coverage'],hardcov=d['hardcov'],seen_counts=d['seen_counts'])
    old.save(OUT/'inference_terminal.json',dict(status='complete'))

def evaluate():
    from sklearn.metrics import average_precision_score
    from cnh_corridor_statistics import paired_cluster_ber
    units={}
    for split in ['calib','evaluation']:
        for u in SPLITS[split]:
            with np.load(OUT/'predictions'/f'unit{u}.npz') as z:units[u]={k:z[k] for k in z.files}
    cat=lambda keys,k:np.concatenate([units[u][k] for u in keys])
    calib=SPLITS['calib'];audit=SPLITS['evaluation']
    cy=cat(calib,'labels');cs=cat(calib,'scores');y=cat(audit,'labels').astype(bool);s=cat(audit,'scores')
    fam=cat(audit,'family');margin=cat(audit,'margin');target=cat(audit,'group')
    thresholds={};rows=[];ledger=[];comparisons=[]
    unitids=np.repeat(audit,22)
    for qi,group in enumerate(['HEAD','BODY']):
        pred={}
        for j,arm in enumerate(ARMS):
            thr=old.threshold(cs[:,j,qi][~cy[:,qi].astype(bool)])
            thresholds[group+'|'+arm]=thr;pred[arm]=s[:,j,qi]>=thr
        for policy in ['strict','ignore5cm']:
            keep=np.ones(len(y),bool) if policy=='strict' else ~((abs(margin)<=.05)&(target==qi))
            bins={'all':np.ones(len(y),bool),'off_group':target!=qi}
            for sign,name in [(-1,'inside'),(1,'outside')]:
                for lo,hi,label in [(0,.05,'0_5cm'),(.05,.15,'5_15cm'),(.15,float('inf'),'15cm_plus')]:
                    bins[name+'_'+label]=(target==qi)&(margin*sign>0)&(abs(margin)>lo)&(abs(margin)<=hi)
            for family in ['all','boundary','mixed_surface','sidewall','general']:
                for bname,bmask in bins.items():
                    mask=keep&bmask&(np.ones(len(y),bool) if family=='all' else fam==family)
                    if not mask.any():continue
                    for j,arm in enumerate(ARMS):
                        yy=y[mask,qi];row=dict(group=group,policy=policy,family=family,bin=bname,arm=arm,**old.binary_metrics(yy,pred[arm][mask]))
                        row['ap']=float(average_precision_score(yy,s[mask,j,qi])) if 0<yy.sum()<len(yy) else None
                        rows.append(row)
            for base,method in [('A2_TRANSFER','A2_RETRAIN'),('A2_RETRAIN','MOTION8_PARTIAL')]:
                for family in ['all','mixed_surface','general']:
                    mask=keep&(np.ones(len(y),bool) if family=='all' else fam==family)
                    b=old.binary_metrics(y[mask,qi],pred[base][mask]);m=old.binary_metrics(y[mask,qi],pred[method][mask])
                    boot=paired_cluster_ber(y[mask,qi],pred[base][mask],pred[method][mask],unitids[mask])
                    ci=boot['ci']
                    comparisons.append(dict(group=group,policy=policy,family=family,base=base,method=method,base_metrics=b,method_metrics=m,ber_delta=boot['point'],ci95=ci,paired_units=boot['units'],bootstrap=boot,numerical_gate=bool(ci is not None and m['ber']<=b['ber']-.05 and ci[1]<0 and m['tpr']>=b['tpr']-.02)))
        for i in range(len(y)):
            ledger.append(dict(unit=audit[i//22],config=i%22,group=group,family=str(fam[i]),margin=float(margin[i]),target_group=int(target[i]),label=int(y[i,qi]),predictions={a:int(p[i]) for a,p in pred.items()}))
    result=dict(scope='Development only',splits=SPLITS,arms=ARMS,thresholds=thresholds,rows=rows,comparisons=comparisons,
                coverage_mean=cat(audit,'coverage').mean(0).tolist(),hard_coverage_mean=cat(audit,'hardcov').mean(0).tolist(),seen_counts=cat(audit,'seen_counts').sum(0).tolist())
    for c in comparisons:
        if c['family']=='all':
            guard=next(x for x in comparisons if all(x[k]==c[k] for k in ['group','policy','base','method']) and x['family']=='general')
            c['advance']=bool(c['numerical_gate'] and (c['method']!='MOTION8_PARTIAL' or guard['method_metrics']['tpr']>=guard['base_metrics']['tpr']-.05))
    old.save(OUT/'results.json',result)
    old.save(OUT/'evaluation_receipt.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),statistics_sha256=hashlib.sha256((HERE/'cnh_corridor_statistics.py').read_bytes()).hexdigest(),criteria_sha256=hashlib.sha256((HERE/'CNH_CORRIDOR_RETRAIN_PLAN_20260929.md').read_bytes()).hexdigest()))
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','seen_counts']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['generate','infer','evaluate'],required=True);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    {'generate':generate,'infer':infer,'evaluate':evaluate}[a.stage]()
