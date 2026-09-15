"""One additive contained-envelope branch and fixed TRAIN-only calibration."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from mz152_contained_return import anchors,calibrate,predict,METHOD
from mz145_causal_confirmation import predict as learned_predict
from run_mz143_corridor_evidence import (ROOT,CODE,CAP,INC,read,sha,write,truth,
 selected_jsonl,public_observations,augmented_score,native_account,local_dependencies)
from run_mz147_query_representation import strict_retention

SOURCE=ROOT/'artifacts.local/work/mz151-expanded-training-20260916/train-v1'
OLD=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
CAP2=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916/source/returned-v1/capture-v1'


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    source=local_dependencies(__file__);source[str(CODE/'MZ152_PROTOCOL_20260916.md')]=sha(CODE/'MZ152_PROTOCOL_20260916.md')
    for p in source:
        dest=out/'source-snapshot'/Path(p).relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    model=read(SOURCE/'model-seal.json');done=read(SOURCE/'completion.json');freeze=read(SOURCE/'freeze.json')
    assert sha(SOURCE/'model-seal.json')==done['model_seal_sha256']
    assert sha(SOURCE/'expanded-hgb.pkl')==model['model_sha256']
    assert sha(SOURCE/'freeze.json')==model['freeze_sha256']
    assert sha(SOURCE/'oof.npz')==model['oof_sha256']
    assert sha(SOURCE/'features.npz')==freeze['features_sha256']
    for p,digest in freeze['inputs'].items():assert sha(Path(p))==digest
    ids=freeze['ids'];rows=public_observations(selected_jsonl(CAP/'raw.jsonl',set(ids[:192])))
    rows+=public_observations(selected_jsonl(CAP2/'raw.jsonl',set(ids[192:])))
    assert [r['id'] for r in rows]==ids
    data=np.load(SOURCE/'oof.npz');scores=data['scores'];y=data['target'];b=data['baseline']
    xx=np.load(SOURCE/'features.npz')['features'];names=read(OLD/'feature-names.json')['sensor_names']
    anchor,slots=anchors(xx,names)
    write(out/'freeze.json',dict(sources=source,method=METHOD,source_model_sha256=model['model_sha256'],
        source_model_seal_sha256=sha(SOURCE/'model-seal.json'),source_oof_sha256=sha(SOURCE/'oof.npz'),
        source_features_sha256=sha(SOURCE/'features.npz'),model_refit=False,original_test_access=False))
    np.savez_compressed(out/'train-anchors.npz',anchor=anchor,slots=slots)
    train_anchor=dict(TP=int((anchor&y).sum()),FP=int((anchor&~y).sum()),
        incumbent_true_frames_covered=int((anchor&y&b).sum()),
        previously_strong_start_support_covered=bool(anchor[ids.index('mz146_suspended_head_scene3_in_00')]),
        previously_low_support_covered=bool(anchor[ids.index('mz146_shallow_boundary_stress_scene3_exit_02')]))
    train_gate=bool(train_anchor['FP']==0 and train_anchor['incumbent_true_frames_covered']>0)
    write(out/'train-anchor-summary.json',dict(metrics=train_anchor,gate=train_gate))
    if not train_gate:
        result=dict(decision='CONTAINED_RETURN_TRAIN_GATE_NOT_MET',train_anchor=train_anchor,
            train_gate=False,dev_scored=False,seconds=time.perf_counter()-start)
    else:
        cuts=calibrate(rows,scores,y,b,anchor);flags=predict(rows,scores,anchor,**cuts)
        es=selected_jsonl(CAP/'evaluator.jsonl',set(ids[:192]))+selected_jsonl(CAP2/'evaluator.jsonl',set(ids[192:]))
        assert [e['id'] for e in es]==ids and np.array_equal(y,[truth(e) for e in es])
        train=augmented_score(rows,es,y,flags.astype(float),.5,b,read(SOURCE/'training-spec.json'),'train')
        write(out/'parameter-seal.json',dict(cutoffs=cuts,train=train,train_anchor=train_anchor,
            freeze_sha256=sha(out/'freeze.json'),train_anchors_sha256=sha(out/'train-anchors.npz'),
            backend=dict(device='CPU',reason='TASK_NOT_GPU_SUITABLE',work='cached feature bounds and saved-score comparisons'),
            authority='TRAIN480_ONLY_BEFORE_DEV_FEATURES_OR_SCORES'))
        original=read(OLD/'freeze.json');devids=set(original['selected_frame_ids']['dev'])
        rr=public_observations(selected_jsonl(CAP/'raw.jsonl',devids))
        seal=read(OLD/'dev-feature-seal.json');assert sha(OLD/'dev-features.npz')==seal['feature_sha256']
        cache=np.load(OLD/'dev-features.npz');aa,ss=anchors(cache['sensor'],names)
        predseal=read(SOURCE/'dev-prediction-seal.json')
        assert sha(SOURCE/'dev-predictions.json')==predseal['predictions_sha256']
        saved=read(SOURCE/'dev-predictions.json');assert [r['id'] for r in rr]==[p['id'] for p in saved]
        ds=np.array([p['score'] for p in saved]);df=predict(rr,ds,aa,**cuts)
        write(out/'dev-predictions.json',[dict(id=r['id'],score=float(ds[i]),anchor=bool(aa[i]),
            candidate=bool(df[i]),contained_slots=np.flatnonzero(ss[i]).tolist()) for i,r in enumerate(rr)])
        write(out/'dev-prediction-seal.json',dict(predictions_sha256=sha(out/'dev-predictions.json'),
            parameter_seal_sha256=sha(out/'parameter-seal.json'),authority='BEFORE_DEV_LABEL_PARSE'))
        ee=selected_jsonl(CAP/'evaluator.jsonl',devids);assert [e['id'] for e in ee]==[r['id'] for r in rr]
        yd=np.array([truth(e) for e in ee],bool)
        base={p['id']:p['candidate'] for p in read(INC/'nominal/predictions.json')['predictions']}
        bd=np.array([base[r['id']] for r in rr],bool);spec=read(CAP/'spec.json')
        br=augmented_score(rr,ee,yd,bd.astype(float),.5,bd,spec,'dev')
        cr=augmented_score(rr,ee,yd,df.astype(float),.5,bd,spec,'dev')
        sr=augmented_score(rr,ee,yd,ds,cuts['low'],bd,spec,'dev')
        ar=augmented_score(rr,ee,yd,aa.astype(float),.5,bd,spec,'dev')
        lr=augmented_score(rr,ee,yd,learned_predict(rr,ds,**cuts).astype(float),.5,bd,spec,'dev')
        cr['pairs']['ordering_semantics']='BINARY_OR_FLAGS_WITH_TIES';retained=strict_retention(cr,br)
        family_ok=all(cr['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
        native=native_account(rr,ee,df,bd);native['radar_native_lineage']='NOT_EVALUABLE'
        write(out/'dev-native-contributors.json',native)
        passed=bool(cr['metrics']['TP']==24 and cr['metrics']['FP']<=3 and cr['events']['false_alert_segments']<=2
            and retained and family_ok and not native['nonalert_with_native_corridor_contributors'])
        result=dict(train_anchor=train_anchor,train_gate=True,dev_scored=True,cutoffs=cuts,
            baseline=br,candidate=cr,single_score=sr,anchor_only=ar,learned_only=lr,dev_gate=passed,
            family_fp_noninferior=family_ok,retention=retained,
            decision='CONTAINED_RETURN_DEV_GAIN_NEEDS_FRESH' if passed else 'CONTAINED_RETURN_DEV_GATE_NOT_MET',
            seconds=time.perf_counter()-start)
    result.update(authority='CONSUMED_DEVELOPMENT_ONLY',model_refit=False,original_test_access=False)
    assert result['seconds']<120
    write(out/'summary.json',result);write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        resource_state='Scalar process exits, no allocations'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','candidate','single_score','anchor_only','learned_only')},indent=2),flush=True)
    if result['dev_scored']:print(json.dumps(dict(dev=result['candidate']['metrics'],events=result['candidate']['events'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
