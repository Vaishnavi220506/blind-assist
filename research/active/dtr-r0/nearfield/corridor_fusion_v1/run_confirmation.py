"""Prospective frozen S1 confirmation. No fitting or operating-point selection."""
import argparse
import csv
import pickle
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from run_e1 import ROOT,HERE,BASE,read,write,sha,dataset,labels,report,native_account
from run_intermediate import E1,PARENT,load_model
from intermediate_features import extract,transform
from specialist_gate import features,geometry_columns,probability,policy
from evaluate_mz136_corridor_pair import retention
from run_mz139_surface_fit import local_dependencies

S1=ROOT/'artifacts.local/work/corridor-depth-specialist-20260917'
INTER=ROOT/'artifacts.local/work/corridor-depth-intermediate-20260917'
WORK=ROOT/'artifacts.local/work/corridor-depth-confirmation-20260917'
DEFAULT_CAPTURE=WORK/'source/returned-v1/capture-v1'
EXPECTED=dict(A='d1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef',
 C='62998d4b798220b1c02c0e4670a6d609d288f9238e0a851397a23c8d7dacafa6',
 pca='6af95e12601930a2ebe32f8d0cf1abdcb035ec8a9931b33b53193532bb6b0546',
 gate='9da38b95bdf8fe1c5937dea8dfbb04d553315a04acdc21f8ccd35ff6904d1be1')
AT=.3917890013717321
CT=.728388090293355
GT=.9
QROOT=3.930000066757202


def frozen():
    from mz143_corridor_features import extract as base_extract
    paths=dict(A=E1/'A-model.pkl',C=INTER/'C-model.pkl',pca=INTER/'pca.pkl',gate=S1/'gate.pkl')
    for key,path in paths.items():assert sha(path)==EXPECTED[key],key
    ss=read(S1/'selection-seal.json');af=read(E1/'selection-seal.json');cf=read(INTER/'selection-seal.json')
    assert ss['selected']['threshold']==GT and ss['gate_sha256']==EXPECTED['gate']
    assert af['selection']['A']['dev_best']['threshold']==AT
    assert cf['selection']['dev_best']['threshold']==CT
    inherited=read(S1/'freeze.json')['bindings'];bindings={}
    for p,h in inherited.items():
        if Path(p).suffix in ('.py','.pkl','.pth') or Path(p).name=='feature-names.json':
            assert sha(p)==h,p;bindings[p]=h
    for p in [Path(__file__),HERE/'test_confirmation.py',S1/'selection-seal.json',E1/'selection-seal.json',INTER/'selection-seal.json']:
        bindings[str(p)]=sha(p)
    # Static local import closure remains identical under CLI/unittest/import
    # invocation, and includes base geometry and evaluator dependencies.
    bindings.update(local_dependencies(Path(__file__)))
    models={k:pickle.loads(p.read_bytes()) for k,p in paths.items()}
    names=read(BASE/'feature-names.json');columns,gnames=geometry_columns(names)
    assert models['gate'].tree_.threshold[0]==QROOT
    assert gnames[models['gate'].tree_.feature[0]]=='nearest.range_m.q100'
    return models,columns,gnames,bindings


def compact(y,a,p,called):
    y,a,p,called=[np.asarray(v,bool) for v in (y,a,p,called)]
    assert not (p&~a).any(),'A veto cannot add alerts'
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum())
    return dict(frames=len(y),TP=tp,FP=fp,FN=fn,TN=int((~y&~p).sum()),
        precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),
        called=int(called.sum()),called_TP=int((called&y&a).sum()),
        called_TP_retained=int((called&y&a&p).sum()),
        called_TP_retention_rate=int((called&y&a&p).sum())/max(1,int((called&y&a).sum())),
        called_FP=int((called&~y&a).sum()),called_FP_retained=int((called&~y&a&p).sum()),
        FP_removed=int((~y&a&~p).sum()),harmful_veto=int((y&a&~p).sum()))


def qbin(value):
    if value is None or not np.isfinite(value):return 'missing'
    bounds=[3.3,3.6,3.8,QROOT,4.2,4.5]
    names=['<=3.3','(3.3,3.6]','(3.6,3.8]','(3.8,tree_threshold]',
           '(tree_threshold,4.2]','(4.2,4.5]']
    for bound,name in zip(bounds,names):
        if value<=bound:return name
    return '>4.5'


def latency(values):
    if not len(values):return None
    return dict(mean_ms=float(np.mean(values)*1000),p50_ms=float(np.quantile(values,.5)*1000),
        p95_ms=float(np.quantile(values,.95)*1000),max_ms=float(np.max(values)*1000),frames=len(values))


def physical_values(spec,rows,es,key):
    """Evaluation-only frame/group join; never passed to feature extraction."""
    frames={f['id']:f for f in spec['frames']}
    groups={g['scene_group']:g for g in spec['scene_groups']}
    return [e.get(key,frames[r['id']].get(key,groups[frames[r['id']]['scene_group']].get(key,'UNKNOWN')))
            for r,e in zip(rows,es)]


def cached_parity():
    m,columns,names,bindings=frozen()
    seal=read(S1/'prediction-seal.json');path=S1/'report-predictions.npz'
    assert sha(path)==seal['sha256'];old=np.load(path)
    bf=E1/'report-features.npz';cf=INTER/'report-features.npz'
    assert sha(bf)==read(E1/'feature-seal.json')['report']
    assert sha(cf)==read(INTER/'feature-seal.json')['features']['report']
    b=np.load(bf);c=np.load(cf)
    from mz136_incumbent import public_observations
    import json
    source=ROOT/'artifacts.local/work/mz170-mean-confirmation-20260916/source/returned-v1/capture-v1'
    assert sha(source/'raw.jsonl')==read(S1/'freeze.json')['bindings'][str(source/'raw.jsonl')]
    rows=public_observations([json.loads(s) for s in (source/'raw.jsonl').read_text().splitlines()])
    assert [r['id'] for r in rows]==list(b['ids'])==list(c['ids'])
    a=m['A'].predict_proba(b['base'])[:,1];cs=m['C'].predict_proba(np.c_[c['base'],c['intermediate']])[:,1]
    g=np.stack([features(r,v,s,columns) for r,v,s in zip(rows,b['base'],a)])
    gp=probability(m['gate'],g);final,called,veto=policy(a>=AT,gp,GT,cs,CT)
    for key,value in dict(A=a,C=cs,gate=gp,final=final,invoked=called,veto=veto).items():
        assert value.dtype==old[key].dtype and value.tobytes()==old[key].tobytes(),key
    return dict(status='PASS',frames=len(rows),bitwise_fields=['A','C','gate','final','invoked','veto'],
        actual_token_calls=0,fit_calls=0,input_hashes=bindings)


def run(capture,out):
    started=time.perf_counter();out=out.resolve();capture=capture.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    m,columns,names,bindings=frozen()
    pre=read(out.parent/'runner-freeze.json')
    assert pre['bindings']==bindings and pre['thresholds']==dict(A=AT,C=CT,gate=GT)
    out.mkdir();write(out/'freeze.json',pre)
    data=dataset(capture,'confirmation');rows=data['rows'];assert len(rows)==288
    assert sha(out.parent/'runner-freeze.json')==data['spec']['recipe_binding']['sha256'],'Capture bound to different inference recipe'
    ids=[r['id'] for r in rows];assert len(set(ids))==288
    inputs={str(capture/n):sha(capture/n) for n in ('raw.jsonl','evaluator.jsonl','spec.json','receipt.json')}
    for r in rows:
        p=(capture/r['rgb_path']).resolve();assert p.is_relative_to(capture)
        assert sha(p)==data['receipt']['hashes'][r['rgb_path']];inputs[str(p)]=sha(p)
    write(out/'input-seal.json',dict(inputs=inputs,ids=ids,authority='PUBLIC_ROWS_BEFORE_EVALUATOR_DECODE'))
    torch.set_num_threads(4);cv2.setNumThreads(4);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.perf_counter();model=load_model().cuda();load_seconds=time.perf_counter()-tick
    inherited_backend=read(INTER/'backend.json');assert inherited_backend['selected_device_type']=='cuda'
    write(out/'backend.json',dict(device='cuda',device_name=torch.cuda.get_device_name(0),
        torch=torch.__version__,opencv=cv2.__version__,numpy=np.__version__,
        inherited_encoder_backend=inherited_backend,inherited_receipt_sha256=sha(INTER/'backend.json'),
        placement='Same frozen encoder workload; sklearn heads and base frontend CPU; four CPU threads'))
    first=rows[0];im=cv2.imread(str(capture/first['rgb_path']))
    extract(model,im,first,first['delta_yaw'] if first['imu_valid'] else 0.)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    from mz143_corridor_features import extract as base_extract
    records=[];times=[];yaw=0.;episode=None
    allnames=read(BASE/'feature-names.json');allnames=allnames['sensor_names']+allnames['geometry_names']
    qidx=allnames.index('nearest.range_m.q100');nidx=allnames.index('nearest.count')
    for i,r in enumerate(rows):
        assert time.perf_counter()-started<3600,'One-pass inference budget exceeded'
        if r['episode_id']!=episode:yaw=0.
        if r['imu_valid']:yaw+=r['delta_yaw']
        episode=r['episode_id'];tick=time.perf_counter();im=cv2.imread(str(capture/r['rgb_path']))
        base=base_extract(r,im,yaw);b=np.r_[base['sensor'],base['geometry']]
        a=float(m['A'].predict_proba(b[None])[0,1]);a_seconds=time.perf_counter()-tick
        flag=a>=AT;called=False;veto=False;gp=None;cs=None
        if flag:
            g=features(r,b,a,columns);gp=float(probability(m['gate'],g[None])[0])
            if gp>=GT:
                called=True;t,mask,_=extract(model,im,r,yaw);v=transform(t[None],mask[None],m['pca'])[0]
                cs=float(m['C'].predict_proba(np.r_[b,v][None])[0,1]);veto=np.isfinite(cs) and cs<CT
                if not np.isfinite(cs):cs=None
        elapsed=time.perf_counter()-tick
        records.append(dict(id=r['id'],A_score=a,A_alert=flag,gate_probability=gp,C_score=cs,
            invoked=called,veto=bool(veto),S1_alert=bool(flag and not veto),
            nearest_q100=float(b[qidx]) if b[nidx]>0 else None))
        times.append(dict(id=r['id'],A_seconds=a_seconds,S1_seconds=elapsed,invoked=called))
        if (i+1)%48==0:print(dict(stage='conditional_public_inference',frames=i+1,total=288),flush=True)
    write(out/'predictions.json',records);write(out/'latency-samples.json',times)
    write(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.json'),
        latency_sha256=sha(out/'latency-samples.json'),input_seal_sha256=sha(out/'input-seal.json'),
        authority='ALL_PUBLIC_PREDICTIONS_SEALED_BEFORE_EVALUATOR_DECODE'))
    es,y=labels(data)
    a=np.array([r['A_alert'] for r in records]);p=np.array([r['S1_alert'] for r in records]);called=np.array([r['invoked'] for r in records])
    reports={key:report(data,es,y,flags.astype(float),flags,a,ranking=False) for key,flags in [('A',a),('S1',p)]}
    paired=retention(reports['S1'],reports['A'])
    native={key:native_account(rows,es,flags,a) for key,flags in [('A',a),('S1',p)]}
    write(out/'native.json',native)
    strata={}
    dimensions=['family','wall_distance_m','background_style','target_reflectance','texture_grid']
    values={key:physical_values(data['spec'],rows,es,key) for key in dimensions}
    values['q100_bin']=[qbin(r['nearest_q100']) for r in records]
    for key,vs in values.items():
        strata[key]={}
        categories=set(str(v) for v in vs)
        if key=='q100_bin':categories.update(qbin(v) for v in [None,3.3,3.6,3.8,QROOT,4.2,4.5,5.])
        for value in sorted(categories):
            ix=np.array([str(v)==value for v in vs]);strata[key][value]=compact(y[ix],a[ix],p[ix],called[ix])
    runtime={key:latency([t[field] for t in times]) for key,field in [('A','A_seconds'),('S1','S1_seconds')]}
    runtime.update(invoked=latency([t['S1_seconds'] for t in times if t['invoked']]),
        not_invoked=latency([t['S1_seconds'] for t in times if not t['invoked']]),cold_model_load_seconds=load_seconds)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        fieldnames=list(records[0])+['truth','family','episode_id','time_s']
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader()
        for r,e,record,gt in zip(rows,es,records,y):writer.writerow(dict(record,truth=int(gt),family=e['family'],episode_id=r['episode_id'],time_s=r['time_s']))
    result=dict(reports=reports,conditional=compact(y,a,p,called),retention=paired,strata=strata,
        latency=runtime,invocation_fraction=float(called.mean()),invocation_fraction_of_A_alerts=int(called.sum())/max(1,int(a.sum())),
        actual_token_calls=int(called.sum()),warmup_token_calls=1,q100_tree_threshold=QROOT,
        q100_missing_rule='nearest.count <= 0',source_groups=len({f['scene_group'] for f in data['spec']['frames']}),
        thresholds=dict(A=AT,C=CT,gate=GT),scope='PROSPECTIVE_TARGETED_SAME_GENERATOR_CONFIRMATION',
        native_radar_lineage='NOT_EVALUABLE',original_test_access=False,fit_calls=0,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started,
        runtime_note='Resident models; actual conditional token execution. A cost is same-frame prefix of S1. No capture/transport/event wait; one excluded warmup.',
        decision='FROZEN_S1_CONFIRMATION_COMPLETE_NO_AUTOMATIC_TUNING')
    write(out/'summary.json',result)
    assert all(sha(k)==v for k,v in {**bindings,**inputs}.items())
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        prediction_seal_sha256=sha(out/'prediction-seal.json'),predictions_csv_sha256=sha(out/'predictions.csv'),
        actual_token_calls=int(called.sum()),resources='process-local models; released at process exit'))
    print(result['conditional'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,default=DEFAULT_CAPTURE)
    parser.add_argument('--output',type=Path,default=WORK/'evaluation-v1')
    parser.add_argument('--check-frozen',action='store_true');parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(4):
        if args.check_frozen:print(cached_parity())
        elif args.freeze_only:
            _,_,_,bindings=frozen();path=args.output.resolve().parent/'runner-freeze.json'
            assert path.is_relative_to((ROOT/'artifacts.local').resolve()) and not path.exists()
            path.parent.mkdir(parents=True,exist_ok=True)
            write(path,dict(bindings=bindings,thresholds=dict(A=AT,C=CT,gate=GT),authority='RUNNER_FROZEN_BEFORE_NEW_CAPTURE'))
            print(path)
        else:
            try:run(args.capture,args.output)
            except BaseException:
                if args.output.exists():write(args.output/'failure.json',dict(status='FAIL',traceback=traceback.format_exc(),automatic_restart=False))
                raise
