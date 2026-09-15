"""One fixed query-supervision experiment with whole-scene held-out folds."""
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
from mz150_virtual_corridors import extract,training_target,OFFSETS,METHOD
from mz145_causal_confirmation import fit_onset,predict
from run_mz143_corridor_evidence import (ROOT,CODE,CAP,INC,read,sha,write,truth,
 selected_jsonl,public_observations,augmented_score,native_account,classifier,
 operating_threshold,local_dependencies)
from run_mz147_query_representation import strict_retention,report
from research_backend import BackendCandidate,DeviceObservation,select_backend

OLD=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
SECOND=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'
ZERO=OFFSETS.index(0.)


def features(rows,capture,offsets,cached,out,label,start):
    receipt=read(capture/'receipt.json');values=[];audits=[];yaw=0.;episode=None
    tick=time.perf_counter();names=None
    for i,row in enumerate(rows):
        if row['episode_id']!=episode:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id'];path=(capture/row['rgb_path']).resolve()
        assert path.is_relative_to(capture.resolve())
        assert sha(path)==receipt['hashes'][row['rgb_path']]
        image=cv2.imread(str(path));assert image is not None
        vv=[];aa=[]
        for offset in offsets:
            value=extract(row,image,yaw,offset)
            nn=value['sensor_names']+value['geometry_names']
            if names is None:names=nn
            assert names==nn
            vector=np.r_[value['sensor'],value['geometry']]
            assert np.isfinite(vector).all() and vector.shape==(2485,)
            if offset==0.:
                assert vector.dtype==cached.dtype and vector.tobytes()==cached[i].tobytes(),row['id']
            vv.append(vector);aa.append(value['audit'])
        values.append(np.stack(vv));audits.append(dict(id=row['id'],queries=aa))
        if (i+1)%24==0:
            print(json.dumps(dict(stage='query_features',split=label,frames=i+1,
                queries=len(offsets)*(i+1),seconds=time.perf_counter()-tick)),flush=True)
        assert time.perf_counter()-start<600,'Fixed 600s run budget exceeded'
    xx=np.stack(values);np.savez_compressed(out/(label+'-features.npz'),queries=xx)
    write(out/(label+'-feature-audit.json'),audits)
    write(out/(label+'-feature-seal.json'),dict(ids=[r['id'] for r in rows],offsets_m=offsets,
        shape=list(xx.shape),feature_sha256=sha(out/(label+'-features.npz')),
        audit_sha256=sha(out/(label+'-feature-audit.json')),zero_query_bitwise_identical=True,
        seconds=time.perf_counter()-tick,authority='PUBLIC_FEATURES_BEFORE_MATCHED_LABEL_PARSE'))
    return xx,names


def evaluate(rows,capture,cached,out,label,start,model,cuts,spec,split,base):
    xx,_=features(rows,capture,(0.,),cached,out,label,start)
    scores=model.predict_proba(xx[:,0])[:,1];flags=predict(rows,scores,**cuts)
    write(out/(label+'-predictions.json'),[dict(id=r['id'],score=float(scores[i]),flag=bool(flags[i]))
        for i,r in enumerate(rows)])
    write(out/(label+'-prediction-seal.json'),dict(predictions_sha256=sha(out/(label+'-predictions.json')),
        model_seal_sha256=sha(out/'model-seal.json'),feature_seal_sha256=sha(out/(label+'-feature-seal.json')),
        authority='PREDICTIONS_BEFORE_MATCHED_EVALUATOR_PARSE'))
    es=selected_jsonl(capture/'evaluator.jsonl',{r['id'] for r in rows})
    assert [e['id'] for e in es]==[r['id'] for r in rows]
    y=np.array([truth(e) for e in es],bool);b=np.array([base[r['id']] for r in rows],bool)
    br=augmented_score(rows,es,y,b.astype(float),.5,b,spec,split)
    rr=report(rows,es,y,scores,**cuts,baseline=b,spec=spec,split=split)
    cr=rr['causal'];retained=strict_retention(cr,br)
    native=native_account(rows,es,flags,b)
    native['radar_native_lineage']='NOT_EVALUABLE: inherited capture has no per-return Radar mapping'
    write(out/(label+'-native-contributors.json'),native)
    family_ok=all(cr['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
    native_ok=not native['nonalert_with_native_corridor_contributors']
    passed=bool(retained and family_ok and native_ok and (
        cr['metrics']['TP']==24 and cr['metrics']['FP']<=3 and cr['events']['false_alert_segments']<=2
        if label=='dev' else cr['metrics']['TP']>=143 and cr['metrics']['FP']<57
        and cr['events']['false_alert_segments']<=21))
    result=dict(baseline=br,reports=rr,gate=passed,retention=retained,family_fp_noninferior=family_ok,
        returned_tof_corridor_retention=native_ok,authority='CONSUMED_DEVELOPMENT_ONLY')
    write(out/(label+'-summary.json'),result)
    print(json.dumps(dict(stage='evaluated',panel=label,gate=passed,metrics=cr['metrics'],
        families=cr['families'],events=cr['events'])),flush=True)
    return result


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    sources=local_dependencies(__file__);sources.update(local_dependencies(CODE/'mz150_virtual_corridors.py'))
    sources[str(CODE/'MZ150_PROTOCOL_20260916.md')]=sha(CODE/'MZ150_PROTOCOL_20260916.md')
    for path in sources:
        dest=out/'source-snapshot'/Path(path).relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    spec=read(CAP/'spec.json');receipt=read(CAP/'receipt.json');oldfreeze=read(OLD/'freeze.json')
    assert sha(CAP/'spec.json')==receipt['spec_sha256']
    for name in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/name)==receipt['hashes'][name]
    ids={s:{f['id'] for f in spec['frames'] if f['split']==s} for s in ('train','dev')}
    rows={s:public_observations(selected_jsonl(CAP/'raw.jsonl',ids[s])) for s in ids};cache={}
    assert len(rows['train'])==192 and len(rows['dev'])==48
    for s in ids:
        assert [r['id'] for r in rows[s]]==oldfreeze['selected_frame_ids'][s]
        seal=read(OLD/(s+'-feature-seal.json'));assert sha(OLD/(s+'-features.npz'))==seal['feature_sha256']
        with np.load(OLD/(s+'-features.npz')) as data:cache[s]=np.c_[data['sensor'],data['geometry']]
    incseal=read(INC/'prediction-seal.json')
    assert sha(INC/'nominal/predictions.json')==incseal['predictions_sha256']['nominal']
    base={p['id']:bool(p['candidate']) for p in read(INC/'nominal/predictions.json')['predictions']}
    write(out/'freeze.json',dict(sources=sources,method=METHOD,
        inputs={str(p):sha(p) for p in [CAP/'raw.jsonl',CAP/'evaluator.jsonl',CAP/'spec.json',
        OLD/'train-features.npz',OLD/'dev-features.npz',OLD/'train-folds.json',INC/'nominal/predictions.json']},
        original_test_access=False,authority='QUERY_SUPERVISION_CONSUMED_DEVELOPMENT'))
    row=rows['train'][0];path=CAP/row['rgb_path'];assert sha(path)==receipt['hashes'][row['rgb_path']]
    image=cv2.imread(str(path));yaw=row['delta_yaw'] if row['imu_valid'] else 0.
    select_backend('batch-tensor',cpu=BackendCandidate('opencv-sklearn-cpu','cpu',lambda:extract(row,image,yaw,.3),
        lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__+' sklearn '+sklearn.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Implemented MZ143 features and sklearn HGB are CPU only'})
    xx,names=features(rows['train'],CAP,OFFSETS,cache['train'],out,'train',start)
    write(out/'feature-names.json',names)
    es=selected_jsonl(CAP/'evaluator.jsonl',ids['train'])
    assert [e['id'] for e in es]==[r['id'] for r in rows['train']]
    yy=np.array([[training_target(e,o) for o in OFFSETS] for e in es],bool)
    y=np.array([truth(e) for e in es],bool);assert np.array_equal(y,yy[:,ZERO])
    b=np.array([base[r['id']] for r in rows['train']],bool)
    ff=read(OLD/'train-folds.json');assert [r['id'] for r in rows['train']]==[f['id'] for f in ff]
    folds=np.array([f['heldout_fold'] for f in ff]);assert dict(Counter(folds))=={0:48,1:48,2:48,3:48}
    np.savez_compressed(out/'train-targets.npz',targets=yy,folds=folds,offsets=OFFSETS)
    write(out/'training-query-seal.json',dict(targets_sha256=sha(out/'train-targets.npz'),
        features_sha256=sha(out/'train-features.npz'),ids=[r['id'] for r in rows['train']],
        target_positive_counts=yy.sum(axis=0).tolist(),
        same_image_different_targets=int(np.any(yy!=yy[:,ZERO,None],axis=1).sum()),
        folds=ff,all_queries_of_one_frame_and_scene_remain_in_one_fold=True,
        authority='TRAIN_NATIVE_LABELS_ONLY_NOT_PREDICTOR_INPUT'))
    scores=np.empty(len(y))
    for fold in range(4):
        model=classifier('fused_hgb')
        model.fit(xx[folds!=fold].reshape(-1,2485),yy[folds!=fold].ravel())
        scores[folds==fold]=model.predict_proba(xx[folds==fold,ZERO])[:,1]
        print(json.dumps(dict(stage='oof_fit',fold=fold,seconds=time.perf_counter()-start)),flush=True)
        assert time.perf_counter()-start<600,'Fixed 600s budget exceeded'
    low=operating_threshold(scores,y,b);high=fit_onset(rows['train'],scores,y,b,low)
    cuts=dict(low=low,high=high);train=report(rows['train'],es,y,scores,low,high,b,spec,'train')
    write(out/'train-summary.json',train);np.savez_compressed(out/'oof.npz',scores=scores,folds=folds)
    model=classifier('fused_hgb');model.fit(xx.reshape(-1,2485),yy.ravel())
    with (out/'virtual-query.pkl').open('wb') as f:pickle.dump(model,f)
    write(out/'model-seal.json',dict(cutoffs=cuts,model_sha256=sha(out/'virtual-query.pkl'),
        oof_sha256=sha(out/'oof.npz'),train_summary_sha256=sha(out/'train-summary.json'),
        training_query_seal_sha256=sha(out/'training-query-seal.json'),freeze_sha256=sha(out/'freeze.json'),
        authority='SEALED_BEFORE_DEV_PREDICTION_AND_LABEL_PARSE'))
    print(json.dumps(dict(stage='fit',cuts=cuts,oof=train['causal']['metrics'])),flush=True)
    assert time.perf_counter()-start<600
    dev=evaluate(rows['dev'],CAP,cache['dev'],out,'dev',start,model,cuts,spec,'dev',base)
    second=None
    if dev['gate']:
        cap=SECOND/'source/returned-v1/capture-v1';s=read(cap/'spec.json');r=read(cap/'receipt.json')
        for name in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/name)==r['hashes'][name]
        assert sha(cap/'spec.json')==r['spec_sha256']
        rr=public_observations([json.loads(line) for line in (cap/'raw.jsonl').read_text(encoding='utf-8').splitlines()])
        saved=read(SECOND/'evaluation-v1/prediction-seal.json')
        assert sha(SECOND/'evaluation-v1/features.npz')==saved['features_sha256']
        assert sha(SECOND/'evaluation-v1/predictions.json')==saved['predictions_sha256']
        assert [r['id'] for r in rr]==[p['id'] for p in read(SECOND/'evaluation-v1/predictions.json')]
        ii=SECOND/'incumbent-v1';ss=read(ii/'prediction-seal.json')
        assert sha(ii/'predictions.json')==ss['predictions_sha256']
        bb={p['id']:bool(p['candidate']) for p in read(ii/'predictions.json')['predictions']}
        with np.load(SECOND/'evaluation-v1/features.npz') as data:fused=data['features']
        write(out/'second-input-seal.json',dict(inputs={str(p):sha(p) for p in [cap/'spec.json',cap/'raw.jsonl',
            cap/'evaluator.jsonl',SECOND/'evaluation-v1/features.npz',ii/'predictions.json']},
            authority='MZ146_ALREADY_CONSUMED_NOT_FRESH_CONFIRMATION'))
        second=evaluate(rr,cap,fused,out,'mz146-consumed',start,model,cuts,s,'confirmation',bb)
    passed=bool(dev['gate'] and second and second['gate'])
    result=dict(dev=dev,second=second,second_scored=second is not None,cutoffs=cuts,development_gain=passed,
        decision='VIRTUAL_QUERY_DEVELOPMENT_GAIN' if passed else 'VIRTUAL_QUERY_DEV_GATE_NOT_MET',
        seconds=time.perf_counter()-start,original_test_access=False,authority='CONSUMED_DEVELOPMENT_NOT_FRESH')
    assert result['seconds']<600
    write(out/'summary.json',result)
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        model_seal_sha256=sha(out/'model-seal.json'),resource_state='CPU process exits, no allocations'))
    print(json.dumps(dict(decision=result['decision'],second_scored=result['second_scored'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(a.output)
