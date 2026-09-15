"""Fixed MZ143/MZ145 recipe on two consumed sources, dev-gated fresh capture."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import pickle
import shutil
import time
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from mz145_causal_confirmation import fit_onset,predict
from run_mz143_corridor_evidence import (ROOT,CODE,CAP,INC,read,sha,write,truth,selected_jsonl,
 public_observations,augmented_score,native_account,classifier,operating_threshold,local_dependencies)
from run_mz147_query_representation import strict_retention,report
from research_backend import BackendCandidate,DeviceObservation,select_backend

OLD=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
SECOND=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    sources=local_dependencies(__file__)
    sources[str(CODE/'MZ151_PROTOCOL_20260916.md')]=sha(CODE/'MZ151_PROTOCOL_20260916.md')
    for path in sources:
        dest=out/'source-snapshot'/Path(path).relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    cap2=SECOND/'source/returned-v1/capture-v1';caps=(CAP,cap2)
    specs=[read(c/'spec.json') for c in caps]
    for cap,spec in zip(caps,specs):
        receipt=read(cap/'receipt.json');assert receipt['status']=='PASS'
        assert sha(cap/'spec.json')==receipt['spec_sha256']
        for name in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/name)==receipt['hashes'][name]
    oldfreeze=read(OLD/'freeze.json');cache={}
    for split in ('train','dev'):
        seal=read(OLD/(split+'-feature-seal.json'))
        assert sha(OLD/(split+'-features.npz'))==seal['feature_sha256']
        with np.load(OLD/(split+'-features.npz')) as data:cache[split]=np.c_[data['sensor'],data['geometry']]
    seal2=read(SECOND/'evaluation-v1/prediction-seal.json')
    assert sha(SECOND/'evaluation-v1/features.npz')==seal2['features_sha256']
    assert sha(SECOND/'evaluation-v1/predictions.json')==seal2['predictions_sha256']
    with np.load(SECOND/'evaluation-v1/features.npz') as data:xx2=data['features']
    oldids=set(oldfreeze['selected_frame_ids']['train'])
    rows1=public_observations(selected_jsonl(CAP/'raw.jsonl',oldids))
    rows2=public_observations([json.loads(line) for line in (cap2/'raw.jsonl').read_text().splitlines()])
    assert [r['id'] for r in rows1]==oldfreeze['selected_frame_ids']['train']
    assert [r['id'] for r in rows2]==[r['id'] for r in read(SECOND/'evaluation-v1/predictions.json')]
    rows=rows1+rows2;assert len(rows)==len({r['id'] for r in rows})==480
    xx=np.r_[cache['train'],xx2];assert xx.shape==(480,2485) and np.isfinite(xx).all()
    incseal=read(INC/'prediction-seal.json')
    assert sha(INC/'nominal/predictions.json')==incseal['predictions_sha256']['nominal']
    inc2=SECOND/'incumbent-v1';iseal2=read(inc2/'prediction-seal.json')
    assert sha(inc2/'predictions.json')==iseal2['predictions_sha256']
    base={p['id']:bool(p['candidate']) for p in read(INC/'nominal/predictions.json')['predictions']}
    base.update({p['id']:bool(p['candidate']) for p in read(inc2/'predictions.json')['predictions']})
    ff=read(OLD/'train-folds.json');assert [f['id'] for f in ff]==[r['id'] for r in rows1]
    groupfold={g:i%4 for i,g in enumerate(sorted(g['scene_group'] for g in specs[1]['scene_groups']))}
    meta={f['id']:f for f in specs[1]['frames']}
    f2=[dict(id=r['id'],heldout_fold=groupfold[meta[r['id']]['scene_group']],
             scene_group=meta[r['id']]['scene_group'],source='mz146_consumed_training') for r in rows2]
    folds=np.array([f['heldout_fold'] for f in ff+f2]);assert dict(Counter(folds))=={0:120,1:120,2:120,3:120}
    for spec,rr,foldv in ((specs[0],rows1,folds[:192]),(specs[1],rows2,folds[192:])):
        mm={f['id']:f for f in spec['frames']};seen={}
        for row,fold in zip(rr,foldv):seen.setdefault(mm[row['id']]['scene_group'],set()).add(int(fold))
        assert all(len(v)==1 for v in seen.values())
    merged=dict(frames=[],pairs=[])
    for spec,ids in ((specs[0],oldids),(specs[1],{r['id'] for r in rows2})):
        included=[f for f in spec['frames'] if f['id'] in ids];episodes={f['episode'] for f in included}
        merged['frames'].extend(dict(copy.deepcopy(f),split='train') for f in included)
        merged['pairs'].extend(dict(copy.deepcopy(p),split='train') for p in spec['pairs'] if set(p['episodes'])<=episodes)
    write(out/'folds.json',ff+f2);write(out/'training-spec.json',merged)
    np.savez_compressed(out/'features.npz',features=xx)
    write(out/'freeze.json',dict(sources=sources,training_source_roles=dict(mz136='original TRAIN192 only',
        mz146='consumed confirmation reclassified as TRAIN288; cannot validate this model'),
        inputs={str(p):sha(p) for p in [CAP/'spec.json',CAP/'raw.jsonl',CAP/'evaluator.jsonl',cap2/'spec.json',
        cap2/'raw.jsonl',cap2/'evaluator.jsonl',OLD/'train-features.npz',OLD/'train-folds.json',
        OLD/'dev-features.npz',SECOND/'evaluation-v1/features.npz',INC/'nominal/predictions.json',inc2/'predictions.json']},
        ids=[r['id'] for r in rows],folds_sha256=sha(out/'folds.json'),features_sha256=sha(out/'features.npz'),
        original_test_access=False,method='UNCHANGED_MZ143_HGB_AND_MZ145_CAUSAL_RULE'))
    es1=selected_jsonl(CAP/'evaluator.jsonl',oldids)
    es2=[json.loads(line) for line in (cap2/'evaluator.jsonl').read_text().splitlines()];es=es1+es2
    assert [e['id'] for e in es]==[r['id'] for r in rows]
    y=np.array([truth(e) for e in es],bool);b=np.array([base[r['id']] for r in rows],bool)
    assert int(y.sum())==240
    # Probe existing frozen-model inference; do not add another candidate fit.
    oldmodelseal=read(OLD/'model-seal.json')
    assert sha(OLD/'fused_hgb.pkl')==oldmodelseal['models']['fused_hgb']
    for name in ('mz143_corridor_features.py','mz125_observable_correction.py','mz136_boundary_geometry.py'):
        assert sha(CODE/name)==oldfreeze['sources'][str(CODE/name)]
    with (OLD/'fused_hgb.pkl').open('rb') as f:probe_model=pickle.load(f)
    select_backend('batch-tensor',cpu=BackendCandidate('sklearn-hgb-cpu','cpu',
        lambda:probe_model.predict_proba(xx[:24]),
        lambda _:DeviceObservation('cpu','host CPU','sklearn '+sklearn.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Existing sklearn HGB implementation has no GPU backend'})
    scores=np.empty(len(y))
    for fold in range(4):
        m=classifier('fused_hgb');m.fit(xx[folds!=fold],y[folds!=fold])
        scores[folds==fold]=m.predict_proba(xx[folds==fold])[:,1]
        print(json.dumps(dict(stage='oof',fold=fold,seconds=time.perf_counter()-start)),flush=True)
        assert time.perf_counter()-start<300
    low=operating_threshold(scores,y,b);high=fit_onset(rows,scores,y,b,low);cuts=dict(low=low,high=high)
    train=report(rows,es,y,scores,low,high,b,merged,'train')
    by_source={}
    for name,sl,spec,split in (('mz136',slice(0,192),specs[0],'train'),('mz146',slice(192,480),specs[1],'confirmation')):
        by_source[name]=report(rows[sl],es[sl],y[sl],scores[sl],low,high,b[sl],spec,split)
    write(out/'train-summary.json',dict(all=train,by_source=by_source))
    np.savez_compressed(out/'oof.npz',scores=scores,folds=folds,target=y,baseline=b)
    m=classifier('fused_hgb');m.fit(xx,y)
    with (out/'expanded-hgb.pkl').open('wb') as f:pickle.dump(m,f)
    write(out/'model-seal.json',dict(model_sha256=sha(out/'expanded-hgb.pkl'),cutoffs=cuts,
        freeze_sha256=sha(out/'freeze.json'),oof_sha256=sha(out/'oof.npz'),
        train_summary_sha256=sha(out/'train-summary.json'),
        authority='TRAIN480_ONLY_MODEL_AND_CUTOFFS_BEFORE_DEV_PREDICTION'))
    print(json.dumps(dict(stage='trained',cutoffs=cuts,train_oof=train['causal']['metrics'])),flush=True)
    devids=set(oldfreeze['selected_frame_ids']['dev'])
    rr=public_observations(selected_jsonl(CAP/'raw.jsonl',devids))
    assert [r['id'] for r in rr]==oldfreeze['selected_frame_ids']['dev']
    ds=m.predict_proba(cache['dev'])[:,1];flags=predict(rr,ds,**cuts)
    write(out/'dev-predictions.json',[dict(id=r['id'],score=float(ds[i]),candidate=bool(flags[i])) for i,r in enumerate(rr)])
    write(out/'dev-prediction-seal.json',dict(predictions_sha256=sha(out/'dev-predictions.json'),
        model_seal_sha256=sha(out/'model-seal.json'),authority='BEFORE_DEV_EVALUATOR_PARSE'))
    ee=selected_jsonl(CAP/'evaluator.jsonl',devids);assert [e['id'] for e in ee]==[r['id'] for r in rr]
    yd=np.array([truth(e) for e in ee],bool);bd=np.array([base[r['id']] for r in rr],bool)
    br=augmented_score(rr,ee,yd,bd.astype(float),.5,bd,specs[0],'dev')
    dev=report(rr,ee,yd,ds,**cuts,baseline=bd,spec=specs[0],split='dev');cr=dev['causal']
    retained=strict_retention(cr,br);family_ok=all(cr['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
    native=native_account(rr,ee,flags,bd)
    native['radar_native_lineage']='NOT_EVALUABLE: no per-return mapping in inherited capture'
    write(out/'dev-native-contributors.json',native)
    native_ok=not native['nonalert_with_native_corridor_contributors']
    passed=bool(cr['metrics']['TP']==24 and cr['metrics']['FP']<=3 and cr['events']['false_alert_segments']<=2
        and retained and family_ok and native_ok)
    result=dict(baseline=br,reports=dev,dev_gate=passed,retention=retained,
        family_fp_noninferior=family_ok,native_retention=native_ok,cutoffs=cuts,
        decision='EXPANDED_TRAIN_DEV_PASS_CONFIRM_FRESH' if passed else 'EXPANDED_TRAIN_DEV_GATE_NOT_MET',
        fresh_capture_allowed=passed,seconds=time.perf_counter()-start,original_test_access=False,
        authority='CONSUMED_DEVELOPMENT_TRAIN480_AND_DEV48')
    assert result['seconds']<300
    write(out/'summary.json',result);write(out/'completion.json',dict(status='PASS',
        summary_sha256=sha(out/'summary.json'),model_seal_sha256=sha(out/'model-seal.json'),
        resource_state='CPU fit/inference exits; source capture separately gated'))
    print(json.dumps(dict(decision=result['decision'],dev=cr['metrics'],events=cr['events'],families=cr['families'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(a.output)
