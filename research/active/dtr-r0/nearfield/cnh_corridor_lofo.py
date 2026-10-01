"""Fixed four-family leave-one-family-out Development check; no old-file edits."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import cnh_corridor_retrain as T
from cnh_corridor_statistics import paired_cluster_ber

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'artifacts.local/work/cnh-corridor-retrain-20260929'
OUT=ROOT/'artifacts.local/work/cnh-corridor-lofo-20260929'
FAMILIES=('boundary','mixed_surface','sidewall','general')
SPLITS={'train':range(1000,1096),'calib':range(2000,2024),'evaluation':range(3000,3048)}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8');temp.replace(p)


def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def metrics(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool);pos=int(y.sum());neg=int((~y).sum())
    tp=int((y&p).sum());fp=int((~y&p).sum())
    return dict(positive=pos,negative=neg,tp=tp,fp=fp,fn=pos-tp,tn=neg-fp,
        tpr=tp/pos if pos else None,fpr=fp/neg if neg else None,
        ber=.5*((pos-tp)/pos+fp/neg) if pos and neg else None)


def threshold(scores):
    a=np.sort(np.asarray(scores,float))[::-1]
    if not len(a) or not np.isfinite(a).all():raise ValueError('invalid calibration negatives')
    return float(np.nextafter(a[int(np.floor(.1*len(a)))],np.inf))


def frozen_request():
    sources=[Path(__file__),HERE/'cnh_corridor_retrain.py',HERE/'cnh_corridor_statistics.py',T.SOURCE/'cnh_learned_readout.py',T.SOURCE/'cnh_learned_memory_fusion.py']
    inputs=[BASE/'features'/s/f'unit{u}.npz' for s,ids in SPLITS.items() for u in ids]
    inputs += [BASE/'predictions'/f'unit{u}.npz' for s in ['calib','evaluation'] for u in SPLITS[s]]
    inputs += [BASE/'results.json',BASE/'models/training_request.json',BASE/'models/training_receipt.json']
    request=dict(plan_sha256=sha(OUT/'LOFO_PLAN.md'),sources={str(p):sha(p) for p in sources},
        inputs={str(p):sha(p) for p in inputs},families=FAMILIES,splits={k:list(v) for k,v in SPLITS.items()},recipe=T.RECIPE)
    if (OUT/'request.json').exists():assert json.loads((OUT/'request.json').read_text())==json.loads(json.dumps(request))
    else:
        save(OUT/'request.json',request)
        for p in sources:
            dest=OUT/'source'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    return request


def train_all():
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('configured GPU is required')
    torch.cuda.set_per_process_memory_fraction(min(.9,1.2*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
    frozen_request();L=T.frozen_module()
    xs=[];ss=[];ys=[];families=[]
    for u in SPLITS['train']:
        d=read(BASE/'features/train'/f'unit{u}.npz');T.validate_arrays(d)
        assert int(d['unit'])==u and len(d['family'])==22
        mask=d['train_mask'];a=d['z4'][mask].astype(np.float32);b=d['z1'][mask].astype(np.float32)
        xs.append(np.stack([np.sign(a)*np.log1p(abs(a)),np.sign(b)*np.log1p(abs(b))],1))
        ss.append(d['support'][mask]);ys.append(d['labels'][mask].astype(np.float32))
        families.append(d['family'][d['scene'][mask]])
    x=np.concatenate(xs);sup=np.concatenate(ss);y=np.concatenate(ys);family=np.concatenate(families)
    del xs,ss,ys,families
    start=time.monotonic()
    for heldout in FAMILIES:
        mask=family!=heldout
        expected=96*(18 if heldout=='general' else 16)*13
        assert int(mask.sum())==expected and not (family[mask]==heldout).any()
        xx,ss,yy=x[mask],sup[mask],y[mask]
        fold=OUT/heldout;fold.mkdir(exist_ok=True)
        save(fold/'training_role.json',dict(heldout=heldout,train_samples=len(xx),train_units=96,excluded_samples=int((~mask).sum()),recipe=T.RECIPE))
        for seed in range(3):
            modelpath=fold/f'model_seed{seed}.pt';receiptpath=fold/f'seed{seed}_receipt.json'
            if receiptpath.exists():
                receipt=json.loads(receiptpath.read_text());assert sha(modelpath)==receipt['sha256'];continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
            net=L.Readout().to(L.DEV);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20);history=[]
            resume=fold/f'seed{seed}_resume.pt'
            if resume.exists():
                ck=torch.load(resume,map_location=L.DEV,weights_only=False)
                net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
                history=ck['history'];rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
            for epoch in range(len(history),20):
                epochstart=time.monotonic();net.train();total=0.;count=0;steps=0
                for batches in T.training_batches(rng.permutation(len(xx)),256,64):
                    opt.zero_grad(set_to_none=True)
                    for ids,weight in batches:
                        xb=torch.as_tensor(xx[ids],device=L.DEV);sb=torch.as_tensor(ss[ids],device=L.DEV);yb=torch.as_tensor(yy[ids],device=L.DEV)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(net(xb,sb),yb)
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                        (loss*weight).backward();total+=float(loss.detach())*len(ids);count+=len(ids)
                    opt.step();steps+=1
                scheduler.step();history.append(dict(epoch=epoch+1,loss=total/count,samples=count,steps=steps,elapsed_s=time.monotonic()-epochstart))
                save(fold/f'seed{seed}_history.json',history)
                torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),history=history,
                    rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),resume)
                save(OUT/'progress.json',dict(fold=heldout,seed=seed,epoch=epoch+1,elapsed_s=time.monotonic()-start))
                print(heldout,seed,epoch+1,round(history[-1]['elapsed_s'],2),round(history[-1]['loss'],5),flush=True)
            torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},modelpath)
            save(receiptpath,dict(seed=seed,epochs=20,sha256=sha(modelpath),samples=len(xx),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),final_loss=history[-1]['loss']))
            del net,opt,scheduler;torch.cuda.empty_cache()
        del xx,ss,yy
    save(OUT/'training_terminal.json',dict(status='complete',folds=4,seeds_per_fold=3,epochs_per_seed=20,elapsed_s=time.monotonic()-start))


def infer_all():
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(min(.9,1.2*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
    L=T.frozen_module()
    from cnh_learned_memory_fusion import causal_ewma
    for heldout in FAMILIES:
        models=T.load_models([OUT/heldout/f'model_seed{i}.pt' for i in range(3)],'cuda')
        for split in ['calib','evaluation']:
            for u in SPLITS[split]:
                dest=OUT/heldout/'predictions'/f'unit{u}.npz'
                if dest.exists():continue
                d=read(BASE/'features'/split/f'unit{u}.npz')
                scene_keep=d['family']!=heldout if split=='calib' else d['family']==heldout
                keep=scene_keep[d['scene']]
                nn=T.predict(models,d['z4'][keep],d['z1'][keep],d['support'][keep],48)
                sm=causal_ewma(nn,d['scene'][keep],d['frame'][keep],alpha=.5,window=5)
                score=sm.reshape(int(scene_keep.sum()),16,6)[:,-1,[2,3]]
                old=read(BASE/'predictions'/f'unit{u}.npz')
                assert np.array_equal(d['family'],old['family'])
                dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest,lofo=score,transfer=old['scores'][scene_keep,1],allfamily=old['scores'][scene_keep,2],
                    labels=old['labels'][scene_keep],family=old['family'][scene_keep],margin=old['margin'][scene_keep],group=old['group'][scene_keep],
                    config=np.flatnonzero(scene_keep),unit=u,split=split)
        del models;torch.cuda.empty_cache()
        print('inferred',heldout,flush=True)
    save(OUT/'inference_terminal.json',dict(status='complete'))


def evaluate_all():
    from sklearn.metrics import average_precision_score
    old=json.loads((BASE/'results.json').read_text())['thresholds']
    rows=[];comparisons=[];ledger=[];thresholds={};allfold=[]
    for heldout in FAMILIES:
        data={split:[read(OUT/heldout/'predictions'/f'unit{u}.npz') for u in SPLITS[split]] for split in ['calib','evaluation']}
        cal={k:np.concatenate([d[k] for d in data['calib']]) for k in ['lofo','transfer','allfamily','labels','family']}
        ev={k:np.concatenate([d[k] for d in data['evaluation']]) for k in ['lofo','transfer','allfamily','labels','margin','group','config']}
        units=np.concatenate([np.full(len(d['lofo']),int(d['unit'])) for d in data['evaluation']])
        assert not (cal['family']==heldout).any()
        for qi,g in enumerate(['HEAD','BODY']):
            pred={};scores={}
            for a in ['lofo','transfer','allfamily']:
                th=threshold(cal[a][:,qi][cal['labels'][:,qi]==0]);name={'lofo':'LOFO','transfer':'TRANSFER_SAME3','allfamily':'ALLFAMILY_SAME3'}[a]
                thresholds[f'{heldout}|{g}|{name}']=dict(value=th,calib_negative=int((cal['labels'][:,qi]==0).sum()),calib_scenes=len(cal[a]),excluded_family=heldout)
                pred[name]=ev[a][:,qi]>=th;scores[name]=ev[a][:,qi]
            for a,oldname,name in [('transfer','A2_TRANSFER','TRANSFER_OLD4'),('allfamily','A2_RETRAIN','ALLFAMILY_OLD4')]:
                th=old[g+'|'+oldname];pred[name]=ev[a][:,qi]>=th;scores[name]=ev[a][:,qi]
                thresholds[f'{heldout}|{g}|{name}']=dict(value=th,scope='existing all-four-family calibration; reference only')
            for i in range(len(units)):
                entry=dict(family=heldout,unit=int(units[i]),config=int(ev['config'][i]),group=g,target_group=int(ev['group'][i]),
                    margin=float(ev['margin'][i]),label=int(ev['labels'][i,qi]),predictions={a:int(v[i]) for a,v in pred.items()},scores={a:float(v[i]) for a,v in scores.items()})
                ledger.append(entry)
    # Each fold heldout evaluation plus an explicitly cross-fold pooled summary.
    for family in [*FAMILIES,'pooled_crossfold']:
        for g in ['HEAD','BODY']:
            base_rows=[r for r in ledger if r['group']==g and (family=='pooled_crossfold' or r['family']==family)]
            for policy in ['strict','ignore5cm']:
                rr=[r for r in base_rows if policy=='strict' or not(r['target_group']==['HEAD','BODY'].index(g) and abs(r['margin'])<=.05)]
                y=np.array([r['label'] for r in rr]);u=np.array([r['unit'] for r in rr])
                for arm in ['LOFO','TRANSFER_SAME3','ALLFAMILY_SAME3','TRANSFER_OLD4','ALLFAMILY_OLD4']:
                    p=np.array([r['predictions'][arm] for r in rr]);z=[r['scores'][arm] for r in rr]
                    rows.append(dict(family=family,group=g,policy=policy,arm=arm,**metrics(y,p),ap=float(average_precision_score(y,z)) if 0<y.sum()<len(y) else None))
                for baseline in ['TRANSFER_SAME3','ALLFAMILY_SAME3']:
                    b=np.array([r['predictions'][baseline] for r in rr]);p=np.array([r['predictions']['LOFO'] for r in rr]);bm=metrics(y,b);pm=metrics(y,p)
                    boot=paired_cluster_ber(y,b,p,u)
                    gate=bool(boot['point'] is not None and boot['point']<=-.05 and boot['ci'] is not None and boot['ci'][1]<0 and pm['tpr']>=bm['tpr']-.02)
                    comparisons.append(dict(family=family,group=g,policy=policy,baseline=baseline,bootstrap=boot,numerical_gate=gate,
                        repaired=int(((b!=y)&(p==y)).sum()),new_errors=int(((b==y)&(p!=y)).sum()),base_errors=int((b!=y).sum()),method_errors=int((p!=y).sum())))
    primary=[c for c in comparisons if c['baseline']=='TRANSFER_SAME3' and c['policy']=='strict']
    result=dict(scope='consumed Development; family-heldout model and calibration; no fresh confirmation',thresholds=thresholds,rows=rows,comparisons=comparisons,
        pooled_both_groups_pass=all(c['numerical_gate'] for c in primary if c['family']=='pooled_crossfold'),
        every_family_group_pass=all(c['numerical_gate'] for c in primary if c['family']!='pooled_crossfold'))
    save(OUT/'results.json',result)
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger),encoding='utf8')
    print(json.dumps(dict(pooled_both_groups_pass=result['pooled_both_groups_pass'],every_family_group_pass=result['every_family_group_pass'],primary=primary),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','infer','evaluate','all'],default='all');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    try:
        for name,f in [('train',train_all),('infer',infer_all),('evaluate',evaluate_all)]:
            if a.stage in (name,'all'):f()
        save(OUT/'terminal.json',dict(status='complete',stage=a.stage))
    except BaseException as e:
        save(OUT/'terminal.json',dict(status='failed',stage=a.stage,error=repr(e)));raise
