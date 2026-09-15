"""Fixed causal residual representation with a predeclared staged falsifier."""
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
from mz148_background_residual import CausalResidual,align_pair,METHOD
from mz145_causal_confirmation import fit_onset,predict
from run_mz143_corridor_evidence import (ROOT,CODE,CAP,INC,read,sha,write,truth,selected_jsonl,
 public_observations,augmented_score,native_account,classifier,operating_threshold,local_dependencies)
from run_mz147_query_representation import strict_retention,report
from research_backend import BackendCandidate,DeviceObservation,select_backend

OLD=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
PREVIOUS=ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'
SECOND=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'
ARMS=('compensated','unregistered');CANDIDATE='compensated'


def features(rows,capture,out,label,start):
    receipt=read(capture/'receipt.json');model=CausalResidual();values={a:[] for a in ARMS};audits=[]
    tick=time.perf_counter()
    for i,row in enumerate(rows):
        p=(capture/row['rgb_path']).resolve();assert p.is_relative_to(capture.resolve())
        assert sha(p)==receipt['hashes'][row['rgb_path']]
        value=model.update(row,cv2.imread(str(p)))
        for arm in ARMS:values[arm].append(value[arm])
        audits.append(dict(id=row['id'],pairs=value['audit']))
        if (i+1)%48==0:print(json.dumps(dict(stage='temporal_features',split=label,frames=i+1,seconds=time.perf_counter()-tick)),flush=True)
        assert time.perf_counter()-start<600,'Fixed600s budget exceeded'
    values={a:np.stack(v) for a,v in values.items()}
    np.savez_compressed(out/(label+'-features.npz'),**values);write(out/(label+'-feature-audit.json'),audits)
    write(out/(label+'-feature-seal.json'),dict(ids=[r['id'] for r in rows],
        feature_sha256=sha(out/(label+'-features.npz')),audit_sha256=sha(out/(label+'-feature-audit.json')),
        seconds=time.perf_counter()-tick,authority='CAUSAL_OBSERVATIONS_BEFORE_MATCHED_LABEL_PARSE'))
    return values


def evaluate(rows,capture,cached_fused,out,label,start,models,cuts,spec,split,base,ids):
    temporal=features(rows,capture,out,label,start)
    scores={a:models[a].predict_proba(np.c_[cached_fused,temporal[a]])[:,1] for a in ARMS}
    flags={a:predict(rows,scores[a],**cuts[a]) for a in ARMS}
    write(out/(label+'-predictions.json'),[dict(id=r['id'],
        scores={a:float(scores[a][i]) for a in ARMS},flags={a:bool(flags[a][i]) for a in ARMS}) for i,r in enumerate(rows)])
    write(out/(label+'-prediction-seal.json'),dict(predictions_sha256=sha(out/(label+'-predictions.json')),
        model_seal_sha256=sha(out/'model-seal.json'),feature_seal_sha256=sha(out/(label+'-feature-seal.json')),
        authority='PREDICTIONS_BEFORE_MATCHED_EVALUATOR_PARSE'))
    es=selected_jsonl(capture/'evaluator.jsonl',ids);assert [e['id'] for e in es]==[r['id'] for r in rows]
    y=np.array([truth(e) for e in es],bool);b=np.array([base[r['id']] for r in rows],bool)
    br=augmented_score(rows,es,y,b.astype(float),.5,b,spec,split)
    reports={a:report(rows,es,y,scores[a],**cuts[a],baseline=b,spec=spec,split=split) for a in ARMS}
    cr=reports[CANDIDATE]['causal'];retained=strict_retention(cr,br)
    native=native_account(rows,es,flags[CANDIDATE],b)
    native['radar_native_lineage']='NOT_EVALUABLE: inherited capture has no per-return Radar mapping'
    write(out/(label+'-native-contributors.json'),native)
    family_ok=all(cr['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
    native_ok=not native['nonalert_with_native_corridor_contributors']
    passed=bool(retained and family_ok and native_ok and (
        cr['metrics']['TP']==24 and cr['metrics']['FP']<=3 and cr['events']['false_alert_segments']<=2
        if label=='dev' else cr['metrics']['TP']>=143 and cr['metrics']['FP']<57 and cr['events']['false_alert_segments']<=21))
    result=dict(baseline=br,reports=reports,gate=passed,retention=retained,family_fp_noninferior=family_ok,
        returned_tof_corridor_retention=native_ok,authority='CONSUMED_DEVELOPMENT_ONLY')
    write(out/(label+'-summary.json'),result)
    print(json.dumps(dict(stage='evaluated',panel=label,gate=passed,
        reports={a:v['causal']['metrics'] for a,v in reports.items()})),flush=True)
    return result


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    sources=local_dependencies(__file__);sources.update(local_dependencies(CODE/'mz148_background_residual.py'))
    sources[str(CODE/'MZ148_PROTOCOL_20260916.md')]=sha(CODE/'MZ148_PROTOCOL_20260916.md')
    for path in sources:
        dest=out/'source-snapshot'/Path(path).relative_to(ROOT.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    spec=read(CAP/'spec.json');receipt=read(CAP/'receipt.json');oldfreeze=read(OLD/'freeze.json')
    assert sha(CAP/'spec.json')==receipt['spec_sha256']
    for name in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/name)==receipt['hashes'][name]
    ids={s:{f['id'] for f in spec['frames'] if f['split']==s} for s in ('train','dev')}
    rows={s:public_observations(selected_jsonl(CAP/'raw.jsonl',ids[s])) for s in ids};cache={}
    for s in ids:
        assert [r['id'] for r in rows[s]]==oldfreeze['selected_frame_ids'][s]
        seal=read(OLD/(s+'-feature-seal.json'));assert sha(OLD/(s+'-features.npz'))==seal['feature_sha256']
        data=np.load(OLD/(s+'-features.npz'));cache[s]=np.c_[data['sensor'],data['geometry']]
    incseal=read(INC/'prediction-seal.json');assert sha(INC/'nominal/predictions.json')==incseal['predictions_sha256']['nominal']
    base={p['id']:bool(p['candidate']) for p in read(INC/'nominal/predictions.json')['predictions']}
    write(out/'freeze.json',dict(sources=sources,method=METHOD,candidate=CANDIDATE,arms=ARMS,
        inputs={str(p):sha(p) for p in [CAP/'raw.jsonl',CAP/'evaluator.jsonl',CAP/'spec.json',
        OLD/'train-features.npz',OLD/'dev-features.npz',OLD/'train-folds.json',INC/'nominal/predictions.json']},
        original_test_access=False,authority='NEW_TEMPORAL_OBSERVATION_CONSUMED_DEVELOPMENT'))
    probe=[cv2.cvtColor(cv2.imread(str(CAP/r['rgb_path'])),cv2.COLOR_BGR2GRAY).astype(np.float32)/255. for r in rows['train'][:2]]
    select_backend('batch-tensor',cpu=BackendCandidate('opencv-sklearn-cpu','cpu',lambda:align_pair(probe[1],probe[0]),
        lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__+' sklearn '+sklearn.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Implemented ECC and sklearn HGB are CPU only'})
    temporal=features(rows['train'],CAP,out,'train',start)
    es=selected_jsonl(CAP/'evaluator.jsonl',ids['train']);assert [e['id'] for e in es]==[r['id'] for r in rows['train']]
    y=np.array([truth(e) for e in es],bool);b=np.array([base[r['id']] for r in rows['train']],bool)
    ff=read(OLD/'train-folds.json');assert [r['id'] for r in rows['train']]==[f['id'] for f in ff]
    folds=np.array([f['heldout_fold'] for f in ff]);assert dict(Counter(folds))=={0:48,1:48,2:48,3:48}
    models={};cuts={};oof={};train={}
    for arm in ARMS:
        xx=np.c_[cache['train'],temporal[arm]];scores=np.empty(len(y))
        for fold in range(4):
            model=classifier('fused_hgb');model.fit(xx[folds!=fold],y[folds!=fold])
            scores[folds==fold]=model.predict_proba(xx[folds==fold])[:,1]
            assert time.perf_counter()-start<600
        low=operating_threshold(scores,y,b);high=fit_onset(rows['train'],scores,y,b,low)
        cuts[arm]=dict(low=low,high=high);oof[arm]=scores
        train[arm]=report(rows['train'],es,y,scores,low,high,b,spec,'train')
        model=classifier('fused_hgb');model.fit(xx,y);models[arm]=model
        with (out/(arm+'.pkl')).open('wb') as f:pickle.dump(model,f)
        print(json.dumps(dict(stage='fit',arm=arm,low=low,high=high,oof=train[arm]['causal']['metrics'])),flush=True)
    write(out/'train-summary.json',train);np.savez_compressed(out/'oof.npz',**oof)
    write(out/'model-seal.json',dict(candidate=CANDIDATE,cutoffs=cuts,models={a:sha(out/(a+'.pkl')) for a in ARMS},
        oof_sha256=sha(out/'oof.npz'),train_summary_sha256=sha(out/'train-summary.json'),freeze_sha256=sha(out/'freeze.json'),
        authority='SEALED_BEFORE_DEV_PREDICTION_AND_LABEL_PARSE'))
    dev=evaluate(rows['dev'],CAP,cache['dev'],out,'dev',start,models,cuts,spec,'dev',base,ids['dev'])
    second=None
    if dev['gate']:
        cap=SECOND/'source/returned-v1/capture-v1';s=read(cap/'spec.json');r=read(cap/'receipt.json')
        for name in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/name)==r['hashes'][name]
        assert sha(cap/'spec.json')==r['spec_sha256']
        rr=public_observations([json.loads(line) for line in (cap/'raw.jsonl').read_text(encoding='utf-8').splitlines()])
        saved=read(SECOND/'evaluation-v1/prediction-seal.json')
        assert sha(SECOND/'evaluation-v1/features.npz')==saved['features_sha256']
        assert [r['id'] for r in rr]==[p['id'] for p in read(SECOND/'evaluation-v1/predictions.json')]
        ss=read(SECOND/'incumbent-v1/prediction-seal.json')
        assert sha(SECOND/'incumbent-v1/predictions.json')==ss['predictions_sha256']
        bb={p['id']:bool(p['candidate']) for p in read(SECOND/'incumbent-v1/predictions.json')['predictions']}
        fused=np.load(SECOND/'evaluation-v1/features.npz')['features']
        write(out/'second-input-seal.json',dict(inputs={str(p):sha(p) for p in [cap/'spec.json',cap/'raw.jsonl',
            cap/'evaluator.jsonl',SECOND/'evaluation-v1/features.npz',SECOND/'incumbent-v1/predictions.json']},
            authority='MZ146_ALREADY_CONSUMED_NOT_FRESH_CONFIRMATION'))
        second=evaluate(rr,cap,fused,out,'mz146-consumed',start,models,cuts,s,'confirmation',bb,{r['id'] for r in rr})
    passed=bool(dev['gate'] and second and second['gate'])
    result=dict(dev=dev,second=second,second_scored=second is not None,cutoffs=cuts,development_gain=passed,
        decision='BACKGROUND_RESIDUAL_DEVELOPMENT_GAIN' if passed else 'BACKGROUND_RESIDUAL_DEV_GATE_NOT_MET',
        seconds=time.perf_counter()-start,original_test_access=False,authority='CONSUMED_DEVELOPMENT_NOT_FRESH')
    write(out/'summary.json',result);write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        model_seal_sha256=sha(out/'model-seal.json'),resource_state='CPU process exits, no allocations'))
    print(json.dumps(dict(decision=result['decision'],second_scored=result['second_scored'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(a.output)
