"""One conditional frozen-C veto experiment; all report inference is sealed."""
import argparse
import csv
import pickle
import shutil
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier,export_text
from threadpoolctl import threadpool_limits
from run_e1 import ROOT,HERE,CAP,BASE,REF,read,write,sha,dataset,labels,report,native_account
from run_intermediate import load_model,PARENT,E1
from intermediate_features import extract,transform
from specialist_gate import features,geometry_columns,EXTRA_NAMES,probability,policy

INTER=ROOT/'artifacts.local/work/corridor-depth-intermediate-20260917'
META=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916/source/returned-v1/capture-v1'


def subset(d,indices):
    ids={d['rows'][i]['id'] for i in indices}
    return dict(d,rows=[d['rows'][i] for i in indices],ids=ids)


def event_preserved(rows,y,base,pred):
    events=[];active=None;episode=None
    for i,row in enumerate(rows):
        if row['episode_id']!=episode or not y[i]:active=None
        episode=row['episode_id']
        if y[i] and active is None:active=dict(base=None,candidate=None);events.append(active)
        if y[i] and base[i] and active['base'] is None:active['base']=row['time_s']
        if y[i] and pred[i] and active['candidate'] is None:active['candidate']=row['time_s']
    return all(e['base'] is None or (e['candidate'] is not None and e['candidate']<=e['base']+.25+1e-9) for e in events)


def run(out):
    started=time.perf_counter();out=out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(exist_ok=True);assert not (out/'completion.json').exists()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    aseal=read(E1/'selection-seal.json');cseal=read(INTER/'selection-seal.json')
    for folder,file,h in [(E1,'A-model.pkl',aseal['models']['A']),(INTER,'C-model.pkl',cseal['head_sha256']),(INTER,'pca.pkl',cseal['pca_sha256'])]:
        assert sha(folder/file)==h
    A=pickle.loads((E1/'A-model.pkl').read_bytes());C=pickle.loads((INTER/'C-model.pkl').read_bytes());pcas=pickle.loads((INTER/'pca.pkl').read_bytes())
    at=aseal['selection']['A']['dev_best']['threshold'];ct=cseal['selection']['dev_best']['threshold']
    names=read(BASE/'feature-names.json');columns,gnames=geometry_columns(names)
    write(out/'gate-feature-names.json',gnames+EXTRA_NAMES)
    meta=dataset(META,'confirmation');target=dataset(CAP,'confirmation')
    by_id={f['id']:f for f in meta['spec']['frames']}
    train=np.array([int(by_id[r['id']]['scene_group'].rsplit('scene',1)[1])<4 for r in meta['rows']])
    assert train.sum()==192 and (~train).sum()==96
    bindings={}
    for p in [Path(__file__),HERE/'specialist_gate.py',HERE/'intermediate_features.py',HERE/'run_e1.py',HERE/'run_intermediate.py',HERE/'SPECIALIST_PROTOCOL_20260917.md',
              E1/'A-model.pkl',INTER/'C-model.pkl',INTER/'pca.pkl',E1/'selection-seal.json',INTER/'selection-seal.json',BASE/'feature-names.json',PARENT/'depth_anything_v2_vits.pth']:
        bindings[str(p)]=sha(p)
    for d in (meta,target):
        for n in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json'):bindings[str(d['cap']/n)]=sha(d['cap']/n)
    # Authenticate report cache against original seals, then bind it to this run.
    for p,h in [(E1/'report-features.npz',read(E1/'feature-seal.json')['report']),
               (INTER/'report-features.npz',read(INTER/'feature-seal.json')['features']['report'])]:
        assert sha(p)==h;bindings[str(p)]=h
    for p,h in read(INTER/'freeze.json')['bindings'].items():
        if 'upstream' in p:assert sha(p)==h;bindings[p]=h
    freeze=dict(bindings=bindings,A_threshold=at,C_threshold=ct,train_ids=[r['id'] for r,t in zip(meta['rows'],train) if t],
          dev_ids=[r['id'] for r,t in zip(meta['rows'],train) if not t],report_ids=[r['id'] for r in target['rows']],
          policy='A AND NOT(cheap_gate AND C_negative)',tree=dict(max_depth=3,min_samples_leaf=6,random_state=178017,harm_weight=8),
          scope='CONSUMED_CONTROLLED_DEVELOPMENT',original_test_access=False)
    if (out/'freeze.json').exists():assert read(out/'freeze.json')==freeze
    else:write(out/'freeze.json',freeze)
    snapshot=out/'source-snapshot';snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__),HERE/'specialist_gate.py',HERE/'SPECIALIST_PROTOCOL_20260917.md']:shutil.copyfile(p,snapshot/p.name)
    tick=time.perf_counter();model=load_model().cuda();load_seconds=time.perf_counter()-tick
    # Reuse measured identical encoder backend; no model or workload change.
    backend=read(INTER/'backend.json');assert backend['selected_device_type']=='cuda'
    write(out/'backend.json',dict(reused_backend=backend,source_sha256=sha(INTER/'backend.json'),actual_device=torch.cuda.get_device_name(0),
                                torch=torch.__version__,reason='Identical frozen encoder token workload; small sklearn tree is CPU-only'))
    from mz143_corridor_features import extract as base_extract
    cache=out/'meta-features';cache.mkdir(exist_ok=True)
    bb=[];xx=[];gg=[];yaw=0.;episode=None
    for i,r in enumerate(meta['rows']):
        assert time.perf_counter()-started<3600
        if r['episode_id']!=episode:yaw=0.
        if r['imu_valid']:yaw+=r['delta_yaw']
        episode=r['episode_id'];path=META/r['rgb_path'];digest=sha(path)
        assert digest==meta['receipt']['hashes'][r['rgb_path']]
        p=cache/(r['id']+'.npz')
        if p.exists():
            z=np.load(p);assert str(z['rgb_sha256'])==digest;b=z['base'];v=z['intermediate']
        else:
            im=cv2.imread(str(path));b0=base_extract(r,im,yaw);b=np.r_[b0['sensor'],b0['geometry']]
            t,m,_=extract(model,im,r,yaw);v=transform(t[None],m[None],pcas)[0]
            np.savez_compressed(p,base=b,intermediate=v,rgb_sha256=digest)
        a=A.predict_proba(b[None])[0,1];bb.append(b);xx.append(np.r_[b,v]);gg.append(features(r,b,a,columns))
        if (i+1)%24==0:
            progress=dict(stage='frozen_experts_meta',completed=i+1,total=288,seconds=time.perf_counter()-started)
            write(out/'progress.json',progress);print(progress,flush=True)
    bmeta=np.stack(bb);xmeta=np.stack(xx);gmeta=np.stack(gg)
    sa=A.predict_proba(bmeta)[:,1];sc=C.predict_proba(xmeta)[:,1];af=sa>=at;cf=sc>=ct
    np.savez_compressed(out/'meta-inputs.npz',base=bmeta,C_input=xmeta,gate=gmeta,A=sa,C=sc,train=train)
    write(out/'meta-feature-seal.json',dict(sha256=sha(out/'meta-inputs.npz'),labels_not_yet_read=True))
    es,y=labels(meta);fitmask=train&af
    useful=(~y)&af&~cf;harm=y&af&~cf
    weights=np.where(harm,8.,1.)
    tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=6,random_state=178017)
    tree.fit(gmeta[fitmask],useful[fitmask].astype(int),sample_weight=weights[fitmask])
    (out/'gate.pkl').write_bytes(pickle.dumps(tree));(out/'gate.txt').write_text(export_text(tree,feature_names=gnames+EXTRA_NAMES),encoding='utf-8')
    gp=probability(tree,gmeta);dev=np.where(~train)[0];dd=subset(meta,dev);yd=y[dev];ad=af[dev];cd=sc[dev]
    thresholds=np.r_[1.0000001,np.unique(gp[dev])];curve=[]
    for threshold in thresholds:
        final,invoke,veto=policy(ad,gp[dev],threshold,cd,ct)
        lost=int((yd&ad&~final).sum());fp=int((~yd&final).sum());events_ok=event_preserved(dd['rows'],yd,ad,final)
        curve.append(dict(threshold=float(threshold),FP=fp,lost_A_TP=lost,invoked=int(invoke.sum()),
                          events_preserved=events_ok,eligible=lost<=1 and events_ok))
    eligible=[r for r in curve if r['eligible']]
    selected=min(eligible,key=lambda r:(r['FP'],r['lost_A_TP'],r['invoked'],-r['threshold']))
    write(out/'development-selection.json',dict(selected=selected,curve=curve,
        train_A_alerts=int(fitmask.sum()),train_useful_veto=int((train&useful).sum()),train_harmful_veto=int((train&harm).sum()),
        dev_A_TP=int((yd&ad).sum()),dev_A_FP=int((~yd&ad).sum()),feature_names=gnames+EXTRA_NAMES))
    write(out/'selection-seal.json',dict(selected=selected,gate_sha256=sha(out/'gate.pkl'),development_sha256=sha(out/'development-selection.json'),
          frozen_experts={k:bindings[k] for k in bindings if k.endswith(('.pkl','.pth'))},report_labels_used=False))
    # Report predictions use cached frozen experts only here; actual conditional latency below bypasses token caches.
    rb=np.load(E1/'report-features.npz');rc=np.load(INTER/'report-features.npz')
    assert list(rb['ids'])==list(rc['ids'])==freeze['report_ids']
    ar=A.predict_proba(rb['base'])[:,1];cr=C.predict_proba(np.c_[rc['base'],rc['intermediate']])[:,1]
    gr=np.stack([features(r,b,a,columns) for r,b,a in zip(target['rows'],rb['base'],ar)])
    pr=probability(tree,gr);fa=ar>=at;fc=cr>=ct
    final,invoke,veto=policy(fa,pr,selected['threshold'],cr,ct);gate_only=fa&~invoke
    np.savez_compressed(out/'report-predictions.npz',A=ar,C=cr,gate=pr,invoked=invoke,veto=veto,final=final,gate_only=gate_only)
    write(out/'prediction-seal.json',dict(sha256=sha(out/'report-predictions.npz'),selection_sha256=sha(out/'selection-seal.json')))
    er,yr=labels(target)
    # Privileged, explicitly post-seal: diagnostic upper bound only.
    rod=np.array([e['family']=='near_rod_farwall' for e in er]);oracle=fa&~(rod&~fc)
    arms=dict(A=fa,C_global=fc,conditional=final,gate_only=gate_only,rod_family_oracle=oracle)
    reports={a:report(target,er,yr,v.astype(float),v,fa,ranking=False) for a,v in arms.items()}
    native={a:native_account(target['rows'],er,v,fa) for a,v in arms.items()}
    write(out/'reports.json',reports);write(out/'native.json',native)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['id','episode','time_s','family','truth','A_score','C_score','gate_probability','invoked','veto',*arms])
        for i,(r,e) in enumerate(zip(target['rows'],er)):
            writer.writerow([r['id'],r['episode_id'],r['time_s'],e['family'],int(yr[i]),ar[i],cr[i],pr[i],int(invoke[i]),int(veto[i]),*[int(v[i]) for v in arms.values()]])
    # Warm model once; no all-frame hidden depth execution in timed routing.
    first=target['rows'][0];im=cv2.imread(str(CAP/first['rgb_path']))
    extract(model,im,first,first['delta_yaw'] if first['imu_valid'] else 0.)
    torch.cuda.reset_peak_memory_stats();timings=[];actual_calls=0;predicted=[];yaw=0.;episode=None
    video=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),4.,(640,480));assert video.isOpened()
    try:
        for i,r in enumerate(target['rows']):
            assert time.perf_counter()-started<3600
            if r['episode_id']!=episode:yaw=0.
            if r['imu_valid']:yaw+=r['delta_yaw']
            episode=r['episode_id'];path=CAP/r['rgb_path'];assert sha(path)==target['receipt']['hashes'][r['rgb_path']]
            tick=time.perf_counter();im=cv2.imread(str(path));b0=base_extract(r,im,yaw);b=np.r_[b0['sensor'],b0['geometry']]
            a=A.predict_proba(b[None])[0,1];assert a==ar[i]
            a_seconds=time.perf_counter()-tick;ff=bool(a>=at);called=False
            if ff:
                g=features(r,b,a,columns);prob=probability(tree,g[None])[0]
                if prob>=selected['threshold']:
                    called=True;actual_calls+=1;t,m,_=extract(model,im,r,yaw);v=transform(t[None],m[None],pcas)[0]
                    c=C.predict_proba(np.r_[b,v][None])[0,1];assert abs(c-cr[i])<1e-12
                    if np.isfinite(c) and c<ct:ff=False
            elapsed=time.perf_counter()-tick
            assert called==bool(invoke[i]) and ff==bool(final[i]);predicted.append(ff)
            timings.append(dict(id=r['id'],A_seconds=a_seconds,conditional_seconds=elapsed,invoked=called))
            canvas=np.zeros((480,640,3),np.uint8);canvas[:360]=im
            lines=[f"{er[i]['family']} t={r['time_s']:.2f}s truth={int(yr[i])}",
                   f"A={int(fa[i])} depth_called={int(called)} veto={int(veto[i])} final={int(ff)}",
                   'Consumed Development | conditional encoder specialist']
            for j,line in enumerate(lines):cv2.putText(canvas,line,(9,389+30*j),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
            video.write(canvas)
            if i==0:cv2.imwrite(str(out/'preview.png'),canvas)
            if (i+1)%48==0:print(dict(stage='online_conditional',completed=i+1,total=288,actual_calls=actual_calls),flush=True)
    finally:video.release()
    write(out/'latency-samples.json',timings)
    def latency(vals):
        return dict(mean_ms=float(np.mean(vals)*1000),p50_ms=float(np.quantile(vals,.5)*1000),p95_ms=float(np.quantile(vals,.95)*1000),max_ms=float(max(vals)*1000)) if len(vals) else None
    runtime={k:latency([r[key] for r in timings]) for k,key in [('A','A_seconds'),('conditional','conditional_seconds')]}
    runtime.update(invoked=latency([r['conditional_seconds'] for r in timings if r['invoked']]),
                   not_invoked=latency([r['conditional_seconds'] for r in timings if not r['invoked']]),cold_model_load_seconds=load_seconds)
    result=dict(reports=reports,selected=selected,latency=runtime,actual_calls=actual_calls,
        invocation_fraction=actual_calls/288,invocation_fraction_of_A_alerts=actual_calls/max(1,int(fa.sum())),
        invocations_by_family={f:int((invoke&np.array([e['family']==f for e in er])).sum()) for f in sorted({e['family'] for e in er})},
        veto_frames=int(veto.sum()),corrected_FP=int((veto&~yr).sum()),lost_A_TP=int((veto&yr).sum()),
        native_radar_lineage='NOT_EVALUABLE',scope='CONSUMED_CONTROLLED_DEVELOPMENT',original_test_access=False,
        seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        decision='ONE_SPECIALIST_RECIPE_COMPLETE_NO_REPORT_TUNING')
    write(out/'summary.json',result)
    assert all(sha(p)==h for p,h in bindings.items())
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),prediction_seal_sha256=sha(out/'prediction-seal.json'),
        predictions_sha256=sha(out/'predictions.csv'),video_sha256=sha(out/'comparison.mp4'),actual_token_calls=actual_calls,
        online_flags_exact=True,resources='process-local; released at exit'))
    print({a:(r['metrics'],r['f1'],r['events']['missed_positive_segments']) for a,r in reports.items()},flush=True)
    print(dict(runtime=runtime,actual_calls=actual_calls,selected=selected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:
        with threadpool_limits(4):run(args.output)
    except Exception:
        if args.output.exists():write(args.output/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()))
        raise
