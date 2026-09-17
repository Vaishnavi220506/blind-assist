"""Joint-data controls and coarse intrusion learning; fresh predictions seal first."""
import argparse
from collections import defaultdict
import itertools
import pickle
import time
import sys
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from run_e1 import ROOT,HERE,BASE,OLD,CAP,read,write,sha,dataset,labels,report,native_account,select_threshold
from intrusion_model import IntrusionNet,local_inputs,native_margins,objective
from mz143_corridor_features import extract

WORK=ROOT/'artifacts.local/work/corridor-intrusion-20260917'
NEW=ROOT/'artifacts.local/work/corridor-depth-confirmation-recovery-20260917/source/returned-v1/capture-v1'
TRAIN=ROOT/'artifacts.local/work/corridor-intrusion-train-20260917/source/returned-v1/capture-v1'
REPORT=ROOT/'artifacts.local/work/corridor-intrusion-report-20260917/source/returned-v1/capture-v1'
AP=ROOT/'artifacts.local/work/corridor-depth-e1-20260917/run-v2/A-model.pkl'
AT=.3917890013717321
SEED=182017
INPUTS=['base','patch','tof','local','A_logit']

def setup():
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    assert sha(AP)=='d1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef'
    return pickle.loads(AP.read_bytes())

def feature_cache(name,cap,split,labelled=True):
    folder=WORK/'cache';folder.mkdir(exist_ok=True,parents=True)
    path=folder/(name+'.npz');seal=folder/(name+'-seal.json')
    data=dataset(cap,split);rows=data['rows'];ids=[r['id'] for r in rows]
    bindings={str(cap/n):sha(cap/n) for n in ['raw.jsonl','spec.json','receipt.json']}
    bindings.update({str(HERE/'intrusion_model.py'):sha(HERE/'intrusion_model.py')})
    if seal.exists():
        s=read(seal);assert sha(path)==s['sha256'] and s['bindings']==bindings
        z=dict(np.load(path));assert list(z['ids'])==ids
        if labelled and 'truth' not in z:raise ValueError('Use distinct labelled cache name')
        return z,data
    m=setup();allvalues=defaultdict(list);yaw=0.;episode=None;times=[]
    for j,r in enumerate(rows):
        if r['episode_id']!=episode:yaw=0.
        if r['imu_valid']:yaw+=r['delta_yaw']
        episode=r['episode_id'];p=cap/r['rgb_path'];assert sha(p)==data['receipt']['hashes'][r['rgb_path']]
        tick=time.perf_counter();im=cv2.imread(str(p));v=extract(r,im,yaw)
        b=np.r_[v['sensor'],v['geometry']].astype(np.float32)
        a=float(m.predict_proba(b[None])[0,1]);a_seconds=time.perf_counter()-tick
        local=local_inputs(r,im,yaw)
        times.append(dict(id=r['id'],A_seconds=a_seconds,frontend_seconds=time.perf_counter()-tick))
        for key,value in dict(base=b,A_score=a,A_logit=np.log(np.clip(a,1e-5,1-1e-5)/(1-np.clip(a,1e-5,1-1e-5))),**local).items():
            allvalues[key].append(value)
        if (j+1)%48==0:print(dict(stage='cache',cohort=name,frames=j+1,total=len(rows)),flush=True)
    z={k:np.array(v) for k,v in allvalues.items()};z['ids']=np.array(ids)
    if labelled:
        es,y=labels(data);z['truth']=y.astype(np.float32);z['margin']=np.stack([native_margins(e) for e in es]);z['field']=(z['margin']>0).astype(np.float32)
        assert np.array_equal(z['field'].max(1).astype(bool),y),'query/whole corridor truth mismatch'
    np.savez_compressed(path,**z);write(folder/(name+'-latency.json'),times)
    write(seal,dict(sha256=sha(path),bindings=bindings,ids=ids,labelled=labelled,
        metadata_never_inputs=True,evaluator_used_only_for_labels=labelled))
    return z,data

def combine(items):return {k:np.concatenate([z[k] for z in items]) for k in items[0]}

def pair_indices(datasets):
    lateral=[];background=[];offset=0
    for data in datasets:
        fs={f['id']:f for f in data['spec']['frames']};lat=defaultdict(list);bg=defaultdict(list)
        for i,r in enumerate(data['rows']):
            f=fs[r['id']];lat[(f['pair_id'],f['time_s'])].append(i+offset)
            if 'target_group' in f:bg[(f['target_group'],f['pair_member'],f['time_s'])].append(i+offset)
        for ii in lat.values():
            assert len(ii)==2;lateral.append(ii)
        for ii in bg.values():background.extend(itertools.combinations(ii,2))
        offset+=len(data['rows'])
    return np.array(lateral,dtype=np.int64),np.array(background,dtype=np.int64)

def tensors(z,mean,std,device='cuda',labels=True):
    names=INPUTS+(['truth','field','margin'] if labels else [])
    result={k:torch.as_tensor(z[k],device=device,dtype=torch.bool if k=='local' else torch.float32) for k in names}
    result['base']=(result['base']-torch.as_tensor(mean,device=device))/torch.as_tensor(std,device=device)
    return result

@torch.inference_mode()
def infer(model,b,batch=96):
    values=defaultdict(list)
    for start in range(0,len(b['base']),batch):
        p=model({k:v[start:start+batch] for k,v in b.items()})
        for k,v in p.items():values[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in values.items()}

def metrics(y,s,t):
    y=np.asarray(y,bool);p=s>=t;tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum())
    return dict(TP=tp,FP=fp,FN=fn,f1=2*tp/max(1,2*tp+fp+fn),recall=tp/max(1,tp+fn),precision=tp/max(1,tp+fp))

def train():
    m=setup();out=WORK/'training-v1';assert not out.exists();out.mkdir(parents=True)
    config=dict(seed=SEED,steps=2000,batch=96,lr=.0003,weight_decay=.0001,
        selection='DEV_BCE_every100_earlier_tie; threshold_DEV_F1_then_FP_then_higher_threshold',
        ordinary='R frame BCE',intrusion='I paired plus auxiliary',margin_consistency='only equal clipped query labels',
        bindings={str(p):sha(p) for p in [Path(__file__),HERE/'intrusion_model.py',HERE/'INTRUSION_PROTOCOL_20260917.md']})
    write(out/'freeze.json',config)
    trainparts=[feature_cache(n,c,s) for n,c,s in [('original-train',OLD,'train'),('old-domain',CAP,'confirmation'),('new-domain',NEW,'confirmation'),('cf-train',TRAIN,'train')]]
    devparts=[feature_cache(n,c,s) for n,c,s in [('original-dev',OLD,'dev'),('cf-dev',TRAIN,'dev')]]
    tr=combine([x[0] for x in trainparts]);dv=combine([x[0] for x in devparts]);assert len(tr['ids'])==912 and len(dv['ids'])==192
    assert not set(tr['ids'])&set(dv['ids'])
    mean=tr['base'].mean(0).astype(np.float32);std=np.maximum(tr['base'].std(0),.01).astype(np.float32)
    np.savez(out/'normalization.npz',mean=mean,std=std)
    write(out/'split-seal.json',dict(train_ids=tr['ids'].tolist(),dev_ids=dv['ids'].tolist(),original_test_access=False,
        caches={str(p):sha(p) for p in (WORK/'cache').glob('*-seal.json')}))
    j=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=8,l2_regularization=1.,random_state=177017)
    j.fit(tr['base'],tr['truth']);(out/'J.pkl').write_bytes(pickle.dumps(j))
    selections={};scores={}
    for arm,model in [('A',m),('J',j)]:
        ts=model.predict_proba(tr['base'])[:,1];ds=model.predict_proba(dv['base'])[:,1]
        chosen,_=select_threshold(dv['truth'].astype(bool),ds);threshold=AT if arm=='A' else chosen['threshold']
        selections[arm]=dict(threshold=threshold,train=metrics(tr['truth'],ts,threshold),dev=metrics(dv['truth'],ds,threshold))
        scores[arm+'_train']=ts;scores[arm+'_dev']=ds
    bt=tensors(tr,mean,std);bd=tensors(dv,mean,std);lp,bp=pair_indices([x[1] for x in trainparts])
    assert np.all(tr['truth'][lp[:,0]]!=tr['truth'][lp[:,1]]) and np.all(tr['truth'][bp[:,0]]==tr['truth'][bp[:,1]])
    for arm in ['R','I']:
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);rng=np.random.default_rng(SEED)
        model=IntrusionNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        logs=[];best=float('inf');tick=time.perf_counter()
        for step in range(1,2001):
            a=rng.integers(0,len(tr['ids']),32);l=lp[rng.integers(0,len(lp),16)].flatten();b=bp[rng.integers(0,len(bp),16)].flatten()
            ix=np.r_[a,l,b];batch={k:v[ix] for k,v in bt.items()}
            pairs=torch.arange(32,64,device='cuda').reshape(-1,2);invariance=torch.arange(64,96,device='cuda').reshape(-1,2)
            model.train();opt.zero_grad(set_to_none=True);p=model(batch)
            loss=objective(p,batch,arm=='I',pairs,invariance);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            if step%100==0:
                model.eval();d=infer(model,bd);t=infer(model,bt)
                dbce=float(F.binary_cross_entropy_with_logits(torch.from_numpy(d['logit']),torch.from_numpy(dv['truth'])))
                tbce=float(F.binary_cross_entropy_with_logits(torch.from_numpy(t['logit']),torch.from_numpy(tr['truth'])))
                rec=dict(step=step,dev_bce=dbce,train_bce=tbce,seconds=time.perf_counter()-tick,loss=float(loss.detach()),
                    train_at05=metrics(tr['truth'],1/(1+np.exp(-t['logit'])),.5))
                logs.append(rec)
                if dbce<best:
                    best=dbce;beststep=step;torch.save(model.state_dict(),out/(arm+'.pt'))
                write(out/(arm+'-progress.json'),rec);print(dict(arm=arm,**rec),flush=True)
        write(out/(arm+'-training.json'),logs);model.load_state_dict(torch.load(out/(arm+'.pt'),weights_only=True));model.eval()
        tp=infer(model,bt);dp=infer(model,bd);ts=1/(1+np.exp(-tp['logit']));ds=1/(1+np.exp(-dp['logit']))
        selected,curve=select_threshold(dv['truth'].astype(bool),ds);threshold=selected['threshold'];write(out/(arm+'-threshold-curve.json'),curve)
        selections[arm]=dict(threshold=threshold,step=beststep,dev_bce=best,train=metrics(tr['truth'],ts,threshold),dev=metrics(dv['truth'],ds,threshold),
            dev_query_margin_mae_m=float(np.abs(dp['margin']*.3-np.clip(dv['margin'],-.3,.3)).mean()),
            train_query_margin_mae_m=float(np.abs(tp['margin']*.3-np.clip(tr['margin'],-.3,.3)).mean()))
        scores[arm+'_train']=ts;scores[arm+'_dev']=ds
        del model,opt;torch.cuda.empty_cache()
    np.savez(out/'development-scores.npz',**scores)
    write(out/'selection-seal.json',dict(selections=selections,models={str(p):sha(p) for p in [AP,out/'J.pkl',out/'R.pt',out/'I.pt',out/'normalization.npz']},
        training_freeze_sha256=sha(out/'freeze.json'),split_seal_sha256=sha(out/'split-seal.json'),scores_sha256=sha(out/'development-scores.npz'),report_access=False))
    write(out/'backend.json',dict(device='cuda',device_name=torch.cuda.get_device_name(0),torch=torch.__version__,parameters=176547,
        hgb_backend='CPU TASK_NOT_GPU_SUITABLE sklearn native implementation',threads=4,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    assert all(sha(p)==h for p,h in config['bindings'].items())
    write(out/'completion.json',dict(status='PASS',selection_sha256=sha(out/'selection-seal.json')));print(selections,flush=True)

def evaluate():
    modela=setup();training=WORK/'training-v1';out=WORK/'report-v1';assert not out.exists();out.mkdir()
    seal=read(training/'selection-seal.json')
    for p,h in seal['models'].items():assert sha(p)==h
    for p,h in read(training/'freeze.json')['bindings'].items():assert sha(p)==h
    assert read(REPORT/'spec.json')['recipe_binding']['sha256']==sha(training/'selection-seal.json')
    z,data=feature_cache('report-public',REPORT,'confirmation',False)
    norm=dict(np.load(training/'normalization.npz'));bt=tensors(z,**norm,labels=False)
    modelj=pickle.loads((training/'J.pkl').read_bytes());models={}
    for arm in ['R','I']:
        model=IntrusionNet().cuda();model.load_state_dict(torch.load(training/(arm+'.pt'),weights_only=True));models[arm]=model.eval()
    values={arm:infer(model,bt) for arm,model in models.items()}
    scores=dict(A=z['A_score'],J=modelj.predict_proba(z['base'])[:,1],**{a:1/(1+np.exp(-v['logit'])) for a,v in values.items()})
    flags={a:s>=seal['selections'][a]['threshold'] for a,s in scores.items()}
    np.savez(out/'predictions.npz',ids=z['ids'],**scores,**{a+'_margin':v['margin'] for a,v in values.items()})
    write(out/'prediction-seal.json',dict(sha256=sha(out/'predictions.npz'),selection_sha256=sha(training/'selection-seal.json'),
        cache_seal_sha256=sha(WORK/'cache/report-public-seal.json'),authority='PUBLIC_PREDICTIONS_BEFORE_REPORT_EVALUATOR_DECODE'))
    es,y=labels(data);margins=np.stack([native_margins(e) for e in es])
    reports={a:report(data,es,y,s,flags[a],flags['A']) for a,s in scores.items()}
    native={a:native_account(data['rows'],es,p,flags['A']) for a,p in flags.items()};write(out/'native.json',native)
    # Warm all heads and frontend, then compare complete per-frame paths.
    r=data['rows'][0];im=cv2.imread(str(REPORT/r['rgb_path']));extract(r,im,0.);local_inputs(r,im,0.)
    for model in models.values():model({k:v[:1] for k,v in bt.items()})
    torch.cuda.synchronize();times=[];yaw=0.;episode=None
    for i,r in enumerate(data['rows']):
        if r['episode_id']!=episode:yaw=0.
        if r['imu_valid']:yaw+=r['delta_yaw']
        episode=r['episode_id'];tick=time.perf_counter();im=cv2.imread(str(REPORT/r['rgb_path']));v=extract(r,im,yaw)
        base=np.r_[v['sensor'],v['geometry']].astype(np.float32);front=time.perf_counter()-tick
        tick=time.perf_counter();av=float(modela.predict_proba(base[None])[0,1])
        ahead=time.perf_counter()-tick
        tick=time.perf_counter();modelj.predict_proba(base[None]);jhead=time.perf_counter()-tick
        tick=time.perf_counter();loc=local_inputs(r,im,yaw)
        sample=dict(base=base[None],A_logit=np.array([np.log(np.clip(av,1e-5,1-1e-5)/(1-np.clip(av,1e-5,1-1e-5)))]),**{k:v[None] for k,v in loc.items()})
        b=tensors(sample,**norm,labels=False);torch.cuda.synchronize();extra=time.perf_counter()-tick
        rec=dict(id=r['id'],A=front+ahead,J=front+jhead)
        for arm,model in models.items():
            tick=time.perf_counter()
            with torch.inference_mode():p=model(b)
            torch.cuda.synchronize();rec[arm]=front+ahead+extra+time.perf_counter()-tick
            assert abs(float(p['logit'].sigmoid())-scores[arm][i])<1e-5
        times.append(rec)
    write(out/'latency-samples.json',times)
    latency={a:dict(mean_ms=float(np.mean([t[a] for t in times])*1000),p50_ms=float(np.quantile([t[a] for t in times],.5)*1000),
        p95_ms=float(np.quantile([t[a] for t in times],.95)*1000)) for a in scores}
    result=dict(reports=reports,latency=latency,query_margin_mae_m={a:float(np.abs(v['margin']*.3-np.clip(margins,-.3,.3)).mean()) for a,v in values.items()},
        scope='FRESH_TARGET_GROUPS_SAME_GENERATOR_CONTROLLED',frames=len(y),target_groups=12,source_configurations=24,original_test_access=False,
        latency_note='Resident warm models; decode/public frontend plus head; no sensor capture/transport. Every frame runs head.',
        thresholds={a:v['threshold'] for a,v in seal['selections'].items()},native_radar_lineage='NOT_EVALUABLE')
    write(out/'summary.json',result);write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),resources='process-local CUDA released on exit'))
    print({a:dict(v['metrics'],f1=v['f1']) for a,v in reports.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cache','train','report'])
    p.add_argument('--resume-deferred-design',action='store_true',help='Use only after a new user-authorized design revision; no training result exists')
    args=p.parse_args()
    if args.action in ['train','report'] and not args.resume_deferred_design:
        p.error('Intrusion pilot deferred by user; complete tolerance review first. No active training/report run.')
    with threadpool_limits(4):
        setup_A=setup()
        if args.action=='cache':
            for n,c,s in [('original-train',OLD,'train'),('original-dev',OLD,'dev'),('old-domain',CAP,'confirmation'),('new-domain',NEW,'confirmation')]:feature_cache(n,c,s)
        elif args.action=='train':train()
        else:evaluate()
