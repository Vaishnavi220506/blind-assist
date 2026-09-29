"""Frozen all-family equal-count LOFO control; consumed Development only."""
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
ORIGINAL=ROOT/'artifacts.local/work/cnh-corridor-lofo-20260929'
OUT=ROOT/'artifacts.local/work/cnh-corridor-equal-count-20260929'
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
    sources=[Path(__file__),HERE/'cnh_corridor_lofo.py',HERE/'cnh_corridor_retrain.py',HERE/'cnh_corridor_statistics.py',T.SOURCE/'cnh_learned_readout.py',T.SOURCE/'cnh_learned_memory_fusion.py']
    inputs=[BASE/'features'/s/f'unit{u}.npz' for s,ids in SPLITS.items() for u in ids]
    inputs += [BASE/'predictions'/f'unit{u}.npz' for s in ['calib','evaluation'] for u in SPLITS[s]]
    inputs += [BASE/'results.json',BASE/'models/training_request.json',BASE/'models/training_receipt.json']
    request=dict(plan_sha256=sha(OUT/'EQUAL_COUNT_PLAN.md'),sources={str(p):sha(p) for p in sources},
        inputs={str(p):sha(p) for p in inputs},families=FAMILIES,splits={k:list(v) for k,v in SPLITS.items()},recipe=T.RECIPE,sampling_seed=20260929)
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
    xs=[];ss=[];ys=[];families=[];identities=[]
    for u in SPLITS['train']:
        d=read(BASE/'features/train'/f'unit{u}.npz');T.validate_arrays(d)
        assert int(d['unit'])==u and len(d['family'])==22
        mask=d['train_mask'];a=d['z4'][mask].astype(np.float32);b=d['z1'][mask].astype(np.float32)
        xs.append(np.stack([np.sign(a)*np.log1p(abs(a)),np.sign(b)*np.log1p(abs(b))],1))
        ss.append(d['support'][mask]);ys.append(d['labels'][mask].astype(np.float32))
        families.append(d['family'][d['scene'][mask]])
        identities.append(np.stack([np.full(int(mask.sum()),u),d['scene'][mask],d['frame'][mask]],axis=1))
    x=np.concatenate(xs);sup=np.concatenate(ss);y=np.concatenate(ys);family=np.concatenate(families)
    identities=np.concatenate(identities)
    del xs,ss,ys,families
    start=time.monotonic()
    for heldout in FAMILIES:
        expected=96*(18 if heldout=='general' else 16)*13
        counts=np.array([(family==f).sum() for f in FAMILIES])
        ideal=counts*expected/len(family);allocated=np.floor(ideal).astype(int)
        order=sorted(range(4),key=lambda i:(-(ideal[i]-allocated[i]),i))
        for i in order[:expected-int(allocated.sum())]:allocated[i]+=1
        sample_rng=np.random.default_rng(20260929)
        indices=np.sort(np.concatenate([sample_rng.choice(np.flatnonzero(family==f),int(n),replace=False) for f,n in zip(FAMILIES,allocated)]))
        assert len(indices)==expected and len(np.unique(indices))==expected
        assert all(abs(allocated-ideal)<1)
        xx,ss,yy=x[indices],sup[indices],y[indices]
        fold=OUT/heldout;fold.mkdir(exist_ok=True)
        selection=fold/'selection.npz'
        if selection.exists():
            prior=read(selection);assert np.array_equal(prior['indices'],indices) and np.array_equal(prior['identities'],identities[indices])
        else:np.savez_compressed(selection,indices=indices,identities=identities[indices],family=family[indices])
        save(fold/'training_role.json',dict(heldout_evaluation=heldout,train_samples=len(xx),train_units=96,original_samples=len(x),family_counts=dict(zip(FAMILIES,allocated.tolist())),sampling_seed=20260929,selection_sha256=sha(selection),recipe=T.RECIPE))
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
                np.savez_compressed(dest,equal_count=score,transfer=old['scores'][scene_keep,1],allfamily=old['scores'][scene_keep,2],
                    labels=old['labels'][scene_keep],family=old['family'][scene_keep],margin=old['margin'][scene_keep],group=old['group'][scene_keep],
                    config=np.flatnonzero(scene_keep),unit=u,split=split)
        del models;torch.cuda.empty_cache()
        print('inferred',heldout,flush=True)
    save(OUT/'inference_terminal.json',dict(status='complete'))



def evaluate_all():
    import csv
    original=[json.loads(line) for line in (ORIGINAL/'sample_ledger.jsonl').read_text().splitlines()]
    lookup={(r['family'],r['unit'],r['config'],r['group']):r for r in original}
    ledger=[];thresholds={};comparisons=[];depth=[]
    for heldout in FAMILIES:
        cal=[read(OUT/heldout/'predictions'/f'unit{u}.npz') for u in SPLITS['calib']]
        labels=np.concatenate([d['labels'] for d in cal]);scores=np.concatenate([d['equal_count'] for d in cal])
        assert all(not (d['family']==heldout).any() for d in cal)
        th=[threshold(scores[:,q][labels[:,q]==0]) for q in range(2)]
        for q,g in enumerate(['HEAD','BODY']):thresholds[f'{heldout}|{g}']=dict(value=th[q],negative=int((labels[:,q]==0).sum()),calib_fpr=float((scores[:,q][labels[:,q]==0]>=th[q]).mean()))
        for u in SPLITS['evaluation']:
            d=read(OUT/heldout/'predictions'/f'unit{u}.npz')
            for i,config in enumerate(d['config']):
                for qi,g in enumerate(['HEAD','BODY']):
                    old=lookup[(heldout,u,int(config),g)]
                    assert old['label']==int(d['labels'][i,qi]) and old['margin']==float(d['margin'][i])
                    row={k:old[k] for k in ['family','unit','config','group','target_group','margin','label']}
                    row['predictions']={'LOFO':old['predictions']['LOFO'],'EQUAL_COUNT':int(d['equal_count'][i,qi]>=th[qi]),'ALLFAMILY':old['predictions']['ALLFAMILY_SAME3']}
                    row['scores']={'LOFO':old['scores']['LOFO'],'EQUAL_COUNT':float(d['equal_count'][i,qi]),'ALLFAMILY':old['scores']['ALLFAMILY_SAME3']}
                    if row['label']:role='positive_intrusion'
                    elif row['target_group']==qi:role='target_group_outside'
                    else:role='other_height_negative'
                    value=abs(row['margin']);band='<=5cm' if value<=.05 else ('5-15cm' if value<=.15 else '>15cm')
                    row.update(geometry_role=role,depth_band=band)
                    ledger.append(row)
        for g in ['HEAD','BODY']:
            rr=[r for r in ledger if r['family']==heldout and r['group']==g]
            y=np.array([r['label'] for r in rr]);u=np.array([r['unit'] for r in rr])
            pred={a:np.array([r['predictions'][a] for r in rr]) for a in ['LOFO','EQUAL_COUNT','ALLFAMILY']}
            boot=paired_cluster_ber(y,pred['LOFO'],pred['EQUAL_COUNT'],u)
            delta=boot['point'];ci=boot['ci']
            decision='缺结构经验' if delta<=-.05 and ci[1]<0 else ('样本量主导' if abs(delta)<.02 and ci[0]<=0<=ci[1] else '未分离')
            comparisons.append(dict(family=heldout,group=g,decision=decision,bootstrap=boot,metrics={a:metrics(y,p) for a,p in pred.items()}))
            for role in ['positive_intrusion','target_group_outside','other_height_negative']:
                for band in ['<=5cm','5-15cm','>15cm']:
                    rs=[r for r in rr if r['geometry_role']==role and r['depth_band']==band]
                    for arm in pred:
                        m=metrics([r['label'] for r in rs],[r['predictions'][arm] for r in rs])
                        depth.append(dict(family=heldout,group=g,geometry_role=role,depth_band=band,arm=arm,**m))
    assert len(ledger)==len(original)==2112
    save(OUT/'results.json',dict(scope='consumed Development; fixed equal-count subset; no causal proof or new formal confirmation',thresholds=thresholds,comparisons=comparisons,depth_rows=depth,deviations=[]))
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger),encoding='utf8')
    with (OUT/'depth_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(depth[0]));w.writeheader();w.writerows(depth)
    lines=['# B 等量全族对照（快速通道）','','|族|组|LOFO BER|等量全族 BER|全族 BER|差值及95%CI(pp)|预设判读|','|---|---|---:|---:|---:|---|---|']
    for c in comparisons:
        b=c['bootstrap'];m=c['metrics'];lines.append(f"|{c['family']}|{c['group']}|{m['LOFO']['ber']:.3%}|{m['EQUAL_COUNT']['ber']:.3%}|{m['ALLFAMILY']['ber']:.3%}|{100*b['point']:.2f} [{100*b['ci'][0]:.2f}, {100*b['ci'][1]:.2f}]|{c['decision']}|")
    lines+=['','每fold固定一次分层抽样；3训练seed组成集成。48单位配对bootstrap，仅消费Development上的诊断，不代表独立重复抽样或真实部署泛化。全族与LOFO均采用同三族calib阈值。FN/FP完整分母见 depth_table.csv；其他高度组负例单独列出，其绝对几何margin不称为查询内伸入。原冻结结论不变。']
    (OUT/'B_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(comparisons,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','infer','evaluate','all'],default='all');a=p.parse_args()
    OUT.mkdir(exist_ok=True);started=time.monotonic()
    try:
        for name,f in [('train',train_all),('infer',infer_all),('evaluate',evaluate_all)]:
            if a.stage in (name,'all'):f()
        save(OUT/'terminal.json',dict(status='complete',stage=a.stage,elapsed_s=time.monotonic()-started))
    except BaseException as e:
        save(OUT/'terminal.json',dict(status='failed',stage=a.stage,elapsed_s=time.monotonic()-started,error=repr(e)));raise
