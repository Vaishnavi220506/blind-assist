"""One fixed TRAIN-only query representation test, stopping on dev gate failure."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import shutil
import time
import cv2
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from mz147_query_representation import extract,METHOD
from mz145_causal_confirmation import fit_onset,predict
from run_mz143_corridor_evidence import (ROOT,CODE,CAP,INC,read,sha,write,truth,selected_jsonl,
 public_observations,augmented_score,native_account,classifier,operating_threshold,local_dependencies)
from evaluate_mz136_corridor_pair import retention
from research_backend import BackendCandidate,DeviceObservation,select_backend

OLD=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
PREVIOUS=ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'
CONSUMED=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'
ARMS=('query_hgb','sensor_hgb');CANDIDATE='query_hgb'


def new_features(rows,capture,out,label,start):
    receipt=read(capture/'receipt.json');features=[];audit=[];yaw=0.;episode=None;tick=time.perf_counter()
    for i,row in enumerate(rows):
        if row['episode_id']!=episode:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id'];p=(capture/row['rgb_path']).resolve()
        assert p.is_relative_to(capture.resolve()) and sha(p)==receipt['hashes'][row['rgb_path']]
        value=extract(row,cv2.imread(str(p)),yaw)
        if features:assert value['names']==names
        names=value['names'];features.append(value['values']);audit.append(dict(id=row['id'],**value['audit']))
        if (i+1)%48==0:print(json.dumps(dict(stage='query_features',split=label,frames=i+1)),flush=True)
        assert time.perf_counter()-start<600,'Fixed run time budget exceeded'
    x=np.stack(features);np.savez_compressed(out/(label+'-features.npz'),query=x)
    write(out/(label+'-feature-audit.json'),audit)
    write(out/(label+'-feature-seal.json'),dict(feature_sha256=sha(out/(label+'-features.npz')),
        ids=[r['id'] for r in rows],audit_sha256=sha(out/(label+'-feature-audit.json')),
        seconds=time.perf_counter()-tick,authority='PUBLIC_FEATURES_BEFORE_MATCHED_EVALUATOR_PARSE'))
    return x,names


def report(rows,es,gt,scores,low,high,baseline,spec,split):
    flags=predict(rows,scores,low,high)
    causal=augmented_score(rows,es,gt,flags.astype(float),.5,baseline,spec,split)
    causal['pairs']['ordering_semantics']='BINARY_CAUSAL_FLAGS_WITH_TIES'
    single=augmented_score(rows,es,gt,scores,low,baseline,spec,split)
    return dict(causal=causal,single=single)


def strict_retention(result,base):
    rr=retention(result,base);result['retention']=rr
    return bool(not result['lost_baseline_tp'] and rr['incumbent_events_and_timing_retained']
        and all(v['relative_delay_s'] is not None and v['relative_delay_s']<=1e-9 for v in rr['per_event_delta']))


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    source=local_dependencies(__file__);source.update(local_dependencies(CODE/'mz147_query_representation.py'))
    source[str(CODE/'MZ147_PROTOCOL_20260916.md')]=sha(CODE/'MZ147_PROTOCOL_20260916.md')
    for p in source:
        dest=out/'source-snapshot'/Path(p).relative_to(ROOT.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
    spec=read(CAP/'spec.json');receipt=read(CAP/'receipt.json');oldfreeze=read(OLD/'freeze.json')
    assert sha(CAP/'spec.json')==receipt['spec_sha256']
    for name in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/name)==receipt['hashes'][name]
    ids={s:{f['id'] for f in spec['frames'] if f['split']==s} for s in ('train','dev')}
    rows={s:public_observations(selected_jsonl(CAP/'raw.jsonl',ids[s])) for s in ids}
    assert len(rows['train'])==192 and len(rows['dev'])==48
    sensor={}
    for s in ids:
        seal=read(OLD/(s+'-feature-seal.json'))
        assert sha(OLD/(s+'-features.npz'))==seal['feature_sha256']
        assert [r['id'] for r in rows[s]]==oldfreeze['selected_frame_ids'][s]
        sensor[s]=np.load(OLD/(s+'-features.npz'))['sensor']
    incseal=read(INC/'prediction-seal.json')
    assert sha(INC/'nominal/predictions.json')==incseal['predictions_sha256']['nominal']
    base={p['id']:bool(p['candidate']) for p in read(INC/'nominal/predictions.json')['predictions']}
    write(out/'freeze.json',dict(sources=source,method=METHOD,candidate=CANDIDATE,arms=ARMS,
        inputs={str(p):sha(p) for p in [CAP/'spec.json',CAP/'raw.jsonl',CAP/'evaluator.jsonl',
        OLD/'train-features.npz',OLD/'dev-features.npz',OLD/'train-folds.json',INC/'nominal/predictions.json']},
        original_test_access=False,authority='NEW_REPRESENTATION_ON_CONSUMED_DEVELOPMENT_NOT_MZ146_RESCUE'))
    probe_row=rows['train'][0];probe_image=cv2.imread(str(CAP/probe_row['rgb_path']))
    probe_yaw=probe_row['delta_yaw'] if probe_row['imu_valid'] else 0.
    select_backend('batch-tensor',cpu=BackendCandidate('sklearn-opencv-cpu','cpu',lambda:extract(probe_row,probe_image,probe_yaw),
        lambda _:DeviceObservation('cpu','host CPU','sklearn '+sklearn.__version__+' OpenCV '+cv2.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Implemented HGB and OpenCV pixel features are CPU only'})
    q,names=new_features(rows['train'],CAP,out,'train',start);write(out/'feature-names.json',names)
    xx=dict(query_hgb=np.c_[sensor['train'],q],sensor_hgb=sensor['train'])
    es=selected_jsonl(CAP/'evaluator.jsonl',ids['train']);assert [e['id'] for e in es]==[r['id'] for r in rows['train']]
    y=np.array([truth(e) for e in es],bool);b=np.array([base[r['id']] for r in rows['train']],bool)
    folds0=read(OLD/'train-folds.json');assert [r['id'] for r in rows['train']]==[f['id'] for f in folds0]
    folds=np.array([f['heldout_fold'] for f in folds0]);assert dict(Counter(folds))=={0:48,1:48,2:48,3:48}
    models={};cuts={};oof={};train={}
    for arm in ARMS:
        scores=np.zeros(len(y))
        for fold in range(4):
            m=classifier('fused_hgb');m.fit(xx[arm][folds!=fold],y[folds!=fold])
            scores[folds==fold]=m.predict_proba(xx[arm][folds==fold])[:,1]
            assert time.perf_counter()-start<600
        low=operating_threshold(scores,y,b);high=fit_onset(rows['train'],scores,y,b,low)
        cuts[arm]=dict(low=low,high=high);oof[arm]=scores
        train[arm]=report(rows['train'],es,y,scores,low,high,b,spec,'train')
        m=classifier('fused_hgb');m.fit(xx[arm],y);models[arm]=m
        with (out/(arm+'.pkl')).open('wb') as f:pickle.dump(m,f)
        print(json.dumps(dict(stage='fit',arm=arm,low=low,high=high,oof=train[arm]['causal']['metrics'])),flush=True)
    write(out/'train-summary.json',train);np.savez_compressed(out/'oof.npz',**oof)
    write(out/'model-seal.json',dict(candidate=CANDIDATE,cutoffs=cuts,
        models={a:sha(out/(a+'.pkl')) for a in ARMS},oof_sha256=sha(out/'oof.npz'),
        train_summary_sha256=sha(out/'train-summary.json'),freeze_sha256=sha(out/'freeze.json'),
        authority='SEALED_TRAIN_ONLY_MODELS_AND_CUTOFFS_BEFORE_DEV_FEATURES_OR_PREDICTIONS'))
    q,nn=new_features(rows['dev'],CAP,out,'dev',start);assert names==nn
    xd=dict(query_hgb=np.c_[sensor['dev'],q],sensor_hgb=sensor['dev'])
    scores={a:models[a].predict_proba(xd[a])[:,1] for a in ARMS}
    write(out/'dev-predictions.json',[dict(id=r['id'],scores={a:float(scores[a][i]) for a in ARMS}) for i,r in enumerate(rows['dev'])])
    write(out/'dev-prediction-seal.json',dict(predictions_sha256=sha(out/'dev-predictions.json'),
        model_seal_sha256=sha(out/'model-seal.json'),feature_seal_sha256=sha(out/'dev-feature-seal.json')))
    ee=selected_jsonl(CAP/'evaluator.jsonl',ids['dev']);assert [e['id'] for e in ee]==[r['id'] for r in rows['dev']]
    yd=np.array([truth(e) for e in ee],bool);bd=np.array([base[r['id']] for r in rows['dev']],bool)
    br=augmented_score(rows['dev'],ee,yd,bd.astype(float),.5,bd,spec,'dev')
    reports={a:report(rows['dev'],ee,yd,scores[a],**cuts[a],baseline=bd,spec=spec,split='dev') for a in ARMS}
    cr=reports[CANDIDATE]['causal'];retained=strict_retention(cr,br)
    family_ok=all(cr['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
    passed=bool(retained and cr['metrics']['FP']<=9 and family_ok and cr['events']['false_alert_segments']<=4)
    write(out/'dev-native-contributors.json',native_account(rows['dev'],ee,predict(rows['dev'],scores[CANDIDATE],**cuts[CANDIDATE]),bd))
    result=dict(candidate=CANDIDATE,baseline=br,reports=reports,cutoffs=cuts,dev_gate=passed,
        family_fp_noninferior=family_ok,mz146_scored=False,mz145_dev_reference=dict(TP=24,FP=3,FN=0,false_segments=2),
        decision='QUERY_DEV_GAIN_NEEDS_SECOND_DEVELOPMENT_CHECK' if passed else 'QUERY_REPRESENTATION_DEV_GATE_NOT_MET',
        seconds=time.perf_counter()-start,authority='CONSUMED_DEVELOPMENT_ONLY',original_test_access=False)
    # The consumed MZ146 comparison is a separate explicit second stage, only
    # allowed when this sealed first-stage outcome passes; no rescue on failure.
    write(out/'summary.json',result)
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        model_seal_sha256=sha(out/'model-seal.json'),resource_state='CPU process exits'))
    print(json.dumps(dict(decision=result['decision'],dev_gate=passed,
        reports={a:v['causal']['metrics'] for a,v in reports.items()},seconds=result['seconds']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(a.output)
