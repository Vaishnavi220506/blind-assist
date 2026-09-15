"""Seal a TRAIN-derived temporal rule before consumed-dev score/label access."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from mz145_causal_confirmation import fit_onset,predict
from run_mz143_corridor_evidence import (ROOT,CODE,CAP,INC,read,sha,write,truth,
    selected_jsonl,public_observations,augmented_score,native_account)
from evaluate_mz136_corridor_pair import retention

SOURCE=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    model_seal=read(SOURCE/'model-seal.json');original_seal=read(SOURCE/'prediction-seal.json')
    complete=read(SOURCE/'completion.json');assert complete['status']=='PASS'
    assert sha(SOURCE/'prediction-seal.json')==complete['prediction_seal_sha256']
    assert sha(SOURCE/'model-seal.json')==original_seal['model_seal_sha256']
    assert sha(SOURCE/'oof-scores.npz')==model_seal['oof_sha256']
    spec=read(CAP/'spec.json');receipt=read(CAP/'receipt.json')
    assert sha(CAP/'spec.json')==receipt['spec_sha256']
    for name in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/name)==receipt['hashes'][name]
    inc_seal=read(INC/'prediction-seal.json')
    assert sha(INC/'nominal/predictions.json')==inc_seal['predictions_sha256']['nominal']
    base={p['id']:bool(p['candidate']) for p in read(INC/'nominal/predictions.json')['predictions']}
    ids={s:{f['id'] for f in spec['frames'] if f['split']==s} for s in ('train','dev')}
    selected=model_seal['selected'];low=float(model_seal['thresholds'][selected])
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',ids['train']))
    es=selected_jsonl(CAP/'evaluator.jsonl',ids['train'])
    assert [r['id'] for r in rows]==[e['id'] for e in es]
    assert [r['id'] for r in rows]==read(SOURCE/'freeze.json')['selected_frame_ids']['train']
    target=np.array([truth(e) for e in es],bool);baseline=np.array([base[r['id']] for r in rows],bool)
    scores=np.load(SOURCE/'oof-scores.npz')[selected]
    high=fit_onset(rows,scores,target,baseline,low)
    files=[CODE/'mz145_causal_confirmation.py',Path(__file__),CODE/'MZ145_PROTOCOL_20260916.md']
    (out/'source-snapshot').mkdir()
    for f in files:shutil.copyfile(f,out/'source-snapshot'/f.name)
    train_flags=predict(rows,scores,low,high)
    write(out/'onset-seal.json',dict(low=low,high=high,source_arm=selected,
        sources={str(f):sha(f) for f in files},oof_sha256=sha(SOURCE/'oof-scores.npz'),
        model_seal_sha256=sha(SOURCE/'model-seal.json'),
        train=augmented_score(rows,es,target,train_flags.astype(float),.5,baseline,spec,'train'),
        backend=dict(actual_device='CPU',reason='TASK_NOT_GPU_SUITABLE',work='scalar causal score comparisons'),
        authority='ONSET_CUTOFF_SEALED_BEFORE_LOADING_DEV_SCORES'))
    assert sha(SOURCE/'dev-predictions.json')==original_seal['predictions_sha256']
    saved=read(SOURCE/'dev-predictions.json')
    rr=public_observations(selected_jsonl(CAP/'raw.jsonl',ids['dev']))
    assert [r['id'] for r in rr]==[p['id'] for p in saved]
    sc=np.array([p['scores'][selected] for p in saved])
    flags=predict(rr,sc,low,high)
    write(out/'dev-decisions.json',[dict(id=r['id'],candidate=bool(p),candidate_state='ALERT' if p else 'UNKNOWN') for r,p in zip(rr,flags)])
    write(out/'decision-seal.json',dict(onset_seal_sha256=sha(out/'onset-seal.json'),
        decisions_sha256=sha(out/'dev-decisions.json'),
        authority='CAUSAL_DECISIONS_SAVED_BEFORE_DEV_LABEL_PARSE'))
    ee=selected_jsonl(CAP/'evaluator.jsonl',ids['dev'])
    assert [r['id'] for r in rr]==[e['id'] for e in ee]
    gt=np.array([truth(e) for e in ee],bool);bb=np.array([base[r['id']] for r in rr],bool)
    result=augmented_score(rr,ee,gt,flags.astype(float),.5,bb,spec,'dev')
    # Ranking is not redefined by binary temporal decisions; report original
    # score ordering separately and label binary ties explicitly.
    result['pairs']['ordering_semantics']='BINARY_TEMPORAL_DECISIONS_WITH_TIES; original score ranking unchanged'
    br=augmented_score(rr,ee,gt,bb.astype(float),.5,bb,spec,'dev')
    single=augmented_score(rr,ee,gt,sc,low,bb,spec,'dev')
    assert br['metrics']==dict(TP=24,FP=13,FN=0,TN=11,UNKNOWN=11)
    assert single['metrics']==dict(TP=24,FP=8,FN=0,TN=16,UNKNOWN=16)
    result['retention']=retention(result,br)
    family_ok=all(result['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
    passed=bool(not result['lost_baseline_tp'] and result['metrics']['FP']<=10 and family_ok and
        result['events']['false_alert_segments']<=br['events']['false_alert_segments'] and
        result['retention']['incumbent_events_and_timing_retained'])
    write(out/'native-contributors.json',native_account(rr,ee,flags,bb))
    summary=dict(low=low,high=high,baseline=br,single_frame=single,candidate=result,
        family_fp_noninferior=family_ok,joint_gain=passed,
        decision='CAUSAL_CONFIRMATION_DEV_GAIN' if passed else 'CAUSAL_CONFIRMATION_NO_JOINT_DEV_GAIN',
        seconds=time.perf_counter()-start,original_test_access=False,
        authority='CONSUMED_DEVELOPMENT_NEW_CAUSAL_READOUT_NOT_FRESH_CONFIRMATION')
    write(out/'summary.json',summary)
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        decision_seal_sha256=sha(out/'decision-seal.json'),resource_state='Scalar process exits, no workers'))
    print(__import__('json').dumps(dict(decision=summary['decision'],low=low,high=high,
        candidate=result['metrics'],events=result['events'],families=result['families']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
