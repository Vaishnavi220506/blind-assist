"""A TRAIN-only state model on frozen MZ148 classifier scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import numpy as np
from mz149_obstacle_state import fit_parameters,filter_scores
from run_mz143_corridor_evidence import (ROOT,CODE,CAP,INC,read,sha,write,truth,selected_jsonl,
 public_observations,augmented_score,native_account,operating_threshold,local_dependencies)
from run_mz147_query_representation import strict_retention

SOURCE=ROOT/'artifacts.local/work/mz148-background-residual-20260916/run-v1'
OLD=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
SECOND=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'


def evaluate(capture,spec,rows,base,label,split,source_label,parameters,threshold,out):
    seal=read(SOURCE/(source_label+'-prediction-seal.json'))
    path=SOURCE/(source_label+'-predictions.json');assert sha(path)==seal['predictions_sha256']
    saved=read(path);assert [p['id'] for p in saved]==[r['id'] for r in rows]
    scores=np.array([p['scores']['compensated'] for p in saved])
    filtered=filter_scores(rows,scores,parameters);flags=filtered>=threshold
    write(out/(label+'-predictions.json'),[dict(id=r['id'],raw_score=float(s),filtered_score=float(f),
        candidate=bool(a),candidate_state='ALERT' if a else 'UNKNOWN') for r,s,f,a in zip(rows,scores,filtered,flags)])
    write(out/(label+'-prediction-seal.json'),dict(predictions_sha256=sha(out/(label+'-predictions.json')),
        parameter_seal_sha256=sha(out/'parameter-seal.json'),source_prediction_sha256=sha(path),
        authority='NEW_FILTER_DECISIONS_BEFORE_MATCHED_EVALUATOR_PARSE'))
    es=selected_jsonl(capture/'evaluator.jsonl',{r['id'] for r in rows})
    assert [e['id'] for e in es]==[r['id'] for r in rows]
    y=np.array([truth(e) for e in es],bool);b=np.array([base[r['id']] for r in rows],bool)
    br=augmented_score(rows,es,y,b.astype(float),.5,b,spec,split)
    cr=augmented_score(rows,es,y,filtered,threshold,b,spec,split)
    cr['pairs']['ordering_semantics']='EMPIRICAL_CAUSAL_FILTERED_SCORE_NOT_CALIBRATED_PROBABILITY'
    retained=strict_retention(cr,br);family_ok=all(cr['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
    native=native_account(rows,es,flags,b);native['radar_native_lineage']='NOT_EVALUABLE'
    native_ok=not native['nonalert_with_native_corridor_contributors'];write(out/(label+'-native-contributors.json'),native)
    passed=bool(retained and family_ok and native_ok and (
        cr['metrics']['TP']==24 and cr['metrics']['FP']<=3 and cr['events']['false_alert_segments']<=2
        if label=='dev' else cr['metrics']['TP']>=143 and cr['metrics']['FP']<57 and cr['events']['false_alert_segments']<=21))
    result=dict(baseline=br,candidate=cr,gate=passed,retention=retained,family_fp_noninferior=family_ok,
        returned_tof_corridor_retention=native_ok,authority='CONSUMED_DEVELOPMENT_ONLY')
    write(out/(label+'-summary.json'),result)
    print(json.dumps(dict(panel=label,gate=passed,metrics=cr['metrics'],events=cr['events'],families=cr['families'])),flush=True)
    return result


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    sources=local_dependencies(__file__);sources.update(local_dependencies(CODE/'mz149_obstacle_state.py'))
    sources[str(CODE/'MZ149_PROTOCOL_20260916.md')]=sha(CODE/'MZ149_PROTOCOL_20260916.md')
    for path in sources:
        dest=out/'source-snapshot'/Path(path).relative_to(ROOT.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    model=read(SOURCE/'model-seal.json');done=read(SOURCE/'completion.json')
    assert sha(SOURCE/'model-seal.json')==done['model_seal_sha256'] and done['status']=='PASS'
    assert sha(SOURCE/'compensated.pkl')==model['models']['compensated']
    assert sha(SOURCE/'oof.npz')==model['oof_sha256']
    spec=read(CAP/'spec.json');receipt=read(CAP/'receipt.json')
    assert sha(CAP/'spec.json')==receipt['spec_sha256']
    for name in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/name)==receipt['hashes'][name]
    ids={s:{f['id'] for f in spec['frames'] if f['split']==s} for s in ('train','dev')}
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',ids['train']))
    es=selected_jsonl(CAP/'evaluator.jsonl',ids['train']);assert [e['id'] for e in es]==[r['id'] for r in rows]
    fold_rows=read(OLD/'train-folds.json');assert [r['id'] for r in rows]==[r['id'] for r in fold_rows]
    folds=np.array([r['heldout_fold'] for r in fold_rows]);assert dict(Counter(folds))=={0:48,1:48,2:48,3:48}
    incseal=read(INC/'prediction-seal.json');assert sha(INC/'nominal/predictions.json')==incseal['predictions_sha256']['nominal']
    base={p['id']:bool(p['candidate']) for p in read(INC/'nominal/predictions.json')['predictions']}
    y=np.array([truth(e) for e in es],bool);b=np.array([base[r['id']] for r in rows],bool)
    scores=np.load(SOURCE/'oof.npz')['compensated'];filtered=np.empty(len(rows));fold_parameters={}
    write(out/'freeze.json',dict(sources=sources,source_model_sha256=sha(SOURCE/'compensated.pkl'),
        source_model_seal_sha256=sha(SOURCE/'model-seal.json'),source_oof_sha256=sha(SOURCE/'oof.npz'),
        inputs={str(p):sha(p) for p in [CAP/'raw.jsonl',CAP/'evaluator.jsonl',CAP/'spec.json',
        OLD/'train-folds.json',INC/'nominal/predictions.json']},vision_model_refit=False,original_test_access=False))
    for fold in range(4):
        train_ix=np.where(folds!=fold)[0];valid_ix=np.where(folds==fold)[0]
        parameters=fit_parameters([rows[i] for i in train_ix],y[train_ix]);fold_parameters[str(fold)]=parameters
        filtered[valid_ix]=filter_scores([rows[i] for i in valid_ix],scores[valid_ix],parameters)
    threshold=operating_threshold(filtered,y,b);parameters=fit_parameters(rows,y)
    np.savez_compressed(out/'oof-state.npz',raw=scores,filtered=filtered,folds=folds)
    train=augmented_score(rows,es,y,filtered,threshold,b,spec,'train')
    write(out/'parameter-seal.json',dict(parameters=parameters,fold_parameters=fold_parameters,
        threshold=threshold,train=train,oof_state_sha256=sha(out/'oof-state.npz'),
        freeze_sha256=sha(out/'freeze.json'),backend=dict(device='CPU',reason='TASK_NOT_GPU_SUITABLE'),
        authority='TRAIN_ONLY_TRANSITION_AND_THRESHOLD_SEAL_BEFORE_DEV_SCORES'))
    print(json.dumps(dict(stage='state_fitted',threshold=threshold,parameters=parameters,train=train['metrics'])),flush=True)
    rr=public_observations(selected_jsonl(CAP/'raw.jsonl',ids['dev']))
    dev=evaluate(CAP,spec,rr,base,'dev','dev','dev',parameters,threshold,out);second=None
    if dev['gate']:
        cap=SECOND/'source/returned-v1/capture-v1';s=read(cap/'spec.json');r=read(cap/'receipt.json')
        for name in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/name)==r['hashes'][name]
        assert sha(cap/'spec.json')==r['spec_sha256']
        rr=public_observations([json.loads(line) for line in (cap/'raw.jsonl').read_text(encoding='utf-8').splitlines()])
        ii=SECOND/'incumbent-v1';ss=read(ii/'prediction-seal.json');assert sha(ii/'predictions.json')==ss['predictions_sha256']
        bb={p['id']:bool(p['candidate']) for p in read(ii/'predictions.json')['predictions']}
        write(out/'second-input-seal.json',dict(inputs={str(p):sha(p) for p in [cap/'spec.json',cap/'raw.jsonl',cap/'evaluator.jsonl',ii/'predictions.json']},
            authority='CONSUMED_MZ146_NOT_FRESH'))
        second=evaluate(cap,s,rr,bb,'mz146-consumed','confirmation','mz146-consumed',parameters,threshold,out)
    assert time.perf_counter()-start<120
    passed=bool(dev['gate'] and second and second['gate'])
    result=dict(dev=dev,second=second,second_scored=second is not None,parameters=parameters,threshold=threshold,
        development_gain=passed,decision='STATE_FILTER_DEVELOPMENT_GAIN' if passed else 'STATE_FILTER_DEV_GATE_NOT_MET',
        seconds=time.perf_counter()-start,authority='CONSUMED_DEVELOPMENT_ONLY',original_test_access=False)
    write(out/'summary.json',result);write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        parameter_seal_sha256=sha(out/'parameter-seal.json'),resource_state='Scalar process exits, no allocations'))
    print(json.dumps(dict(decision=result['decision'],second_scored=result['second_scored'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
