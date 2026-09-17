"""Resident single-model full replay, sealed before native scoring."""
import argparse
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from single_positive_inference import SinglePositiveSystem

ROOT=Path(__file__).resolve().parents[5]
HOME=ROOT/'artifacts.local/work/corridor-public-single-20260917'
OUT=HOME/'confirmation'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def latency(x):return dict(mean=float(np.mean(x)),p50=float(np.percentile(x,50)),p95=float(np.percentile(x,95)),observed_max=float(max(x)))


def evaluate(cap,rows,preds):
    # Import/decode evaluator data only after public predictions are sealed.
    from tolerance_eval import geometry,classify,metric,temporal
    from mz171_return_labels import make_witness_labels
    es=lines(cap/'evaluator.jsonl');spec=read(cap/'spec.json')
    fs={f['id']:f for f in spec['frames']}
    assert [r['id'] for r in rows]==[e['id'] for e in es]==[p['id'] for p in preds]
    labels=[make_witness_labels(r,e) for r,e in zip(rows,es)]
    g=[geometry(e) for e in es]
    states=np.array([classify(v,.05) for v in g]);clear=states!='boundary'
    truth=np.array([v['strict'] for v in g],bool)
    witness=np.array([bool(((l['target'][:128]>0)&l['known'][:128]).any()) for l in labels])
    a=np.array([p['A'] for p in preds],bool)
    flags={m:np.array([p[key] for p in preds],bool) for m,key in [('A','A'),('A_plus_public','alert'),('A_retrained','control')]}
    families=np.array([fs[r['id']]['family'] for r in rows])
    a_core=temporal(rows,states,a);a_strict=temporal(rows,np.where(truth,'positive','negative'),a)
    def changes_temporal(current,baseline):
        old={e['episode']:e for e in baseline['events']};result=[]
        for event in current['events']:
            before=old[event['episode']]['first_in_core_delay_s'];after=event['first_in_core_delay_s']
            if before!=after:result.append(dict(episode=event['episode'],A_first_s=before,method_first_s=after))
        return result
    summary={}
    for name,p in flags.items():
        t=temporal(rows,states,p);st=temporal(rows,np.where(truth,'positive','negative'),p)
        gained=~a&p;lost=a&~p
        def delta(mask):
            return dict(rescued_FN=int(sum(mask&gained&truth)),added_FP=int(sum(mask&gained&~truth)),
                lost_TP=int(sum(mask&lost&truth)),removed_FP=int(sum(mask&lost&~truth)))
        selected=[]
        if name=='A_plus_public':
            for i in np.flatnonzero(gained):
                slot=preds[i]['max_return_slot']
                assert slot is not None
                selected.append(dict(id=rows[i]['id'],truth=bool(truth[i]),clear=bool(clear[i]),
                    any_witness=bool(witness[i]),max_slot=int(slot),max_slot_witness=bool(labels[i]['known'][slot] and labels[i]['target'][slot]>0)))
            assert not lost.any()
            assert all(e['method_first_s'] is not None and (e['A_first_s'] is None or e['method_first_s']<=e['A_first_s'])
                for e in changes_temporal(st,a_strict))
        summary[name]=dict(clear=metric(truth[clear],p[clear]),coverage=float(clear.mean()),strict=metric(truth,p),
            boundary=metric(truth[~clear],p[~clear]),clear_changes=delta(clear),strict_changes=delta(np.ones(len(rows),bool)),
            temporal=t,strict_temporal=st,first_core_changes=changes_temporal(t,a_core),
            first_strict_changes=changes_temporal(st,a_strict),families={f:dict(
                clear=metric(truth[clear&(families==f)],p[clear&(families==f)]),
                strict=metric(truth[families==f],p[families==f])) for f in sorted(set(families))},
            changed_episodes={label:sorted({rows[i]['episode_id'] for i in range(len(rows)) if mask[i]})
                for label,mask in [('clear_rescue',clear&truth&gained),('clear_added_FP',clear&~truth&gained),
                    ('clear_lost_TP',clear&truth&lost),('clear_removed_FP',clear&~truth&lost)]},
            added_public_support=selected)
    miss=clear&truth&~a
    opportunities=dict(clear_A_FN=int(sum(miss)),clear_A_FN_with_sampled_return=int(sum(miss&witness)),
        supported_miss_episodes=sorted({rows[i]['episode_id'] for i in np.flatnonzero(miss&witness)}),
        clear_A_FN_without_sampled_support=int(sum(miss&~witness)),
        all_zero_return_frames=sum(p['usable_tof_returns']==0 for p in preds))
    cases=[dict(**p,family=fs[p['id']]['family'],group=fs[p['id']]['scene_group'],
        truth=bool(truth[i]),stratum=states[i],sampled_witness=bool(witness[i])) for i,p in enumerate(preds)]
    return summary,opportunities,cases


def main(cap):
    assert not OUT.exists(),'Preserve completed or failed confirmation'
    OUT.mkdir()
    evaluation_freeze=read(HOME/'evaluation-freeze.json')
    for p,h in evaluation_freeze['sources'].items():assert sha(p)==h,p
    recipe=read(HOME/'recipe-freeze.json')
    for p,h in recipe['bundle_files'].items():assert sha(p)==h,p
    for p,h in recipe['sources'].items():assert sha(p)==h,p
    receipt=read(cap/'receipt.json');assert receipt['status']=='PASS' and receipt['frames']==288
    for p,h in receipt['hashes'].items():assert sha(cap/p)==h,p
    assert read(HOME/'source/controller-verification.json')['status']=='PASS'
    assert sha(cap/'spec.json')==receipt['spec_sha256']
    torch.set_num_threads(4)
    tick=time.perf_counter();system=SinglePositiveSystem(HOME/'bundle');load_s=time.perf_counter()-tick
    # Whole path warmup uses prior consumed RGB/public rows only.
    old=ROOT/'artifacts.local/work/mz170-mean-confirmation-20260916/source/returned-v1/capture-v1'
    warm=lines(old/'raw.jsonl')[:3];yaw=0.
    for r in warm:
        if r['imu_valid']:yaw+=r['delta_yaw']
        system.predict(r,cv2.imread(str(old/r['rgb_path'])),yaw)
    rows=lines(cap/'raw.jsonl');assert len(rows)==288
    sources={str(Path(__file__).resolve()):sha(Path(__file__)),**recipe['sources']}
    write(OUT/'input-seal.json',dict(recipe_sha256=sha(HOME/'recipe-freeze.json'),sources=sources,
        evaluation_freeze_sha256=sha(HOME/'evaluation-freeze.json'),
        raw_sha256=sha(cap/'raw.jsonl'),receipt_sha256=sha(cap/'receipt.json'),
        native_outcomes_unread=True,warmup_source=str(old),warmup_frames=3))
    preds=[];yaw=0.;episode=None
    for r in rows:
        if r['episode_id']!=episode:yaw=0.
        if r['imu_valid']:yaw+=r['delta_yaw']
        episode=r['episode_id'];tick=time.perf_counter()
        image=cv2.imread(str(cap/r['rgb_path']))
        if image is None:raise ValueError('RGB decode failed: '+r['id'])
        decode_ms=(time.perf_counter()-tick)*1000
        result=system.predict(r,image,yaw)
        result.update(id=r['id'],episode_id=episode,time_s=r['time_s'],yaw=yaw,rgb_decode_ms=decode_ms,
            A_complete_ms=decode_ms+result['A_algorithm_ms'],
            public_complete_ms=decode_ms+result['A_algorithm_ms']+result['head_increment_ms'],
            control_complete_ms=decode_ms+result['A_algorithm_ms']+result['control_head_ms'])
        if result['usable_tof_returns']==0:assert result['alert']==result['A']
        assert not result['A'] or result['alert']
        preds.append(result)
    write(OUT/'predictions.json',preds)
    write(OUT/'prediction-seal.json',dict(predictions_sha256=sha(OUT/'predictions.json'),
        input_seal_sha256=sha(OUT/'input-seal.json'),authority='ALL_PUBLIC_OUTPUTS_SEALED_BEFORE_NATIVE_REDUCTION'))
    s,opportunities,cases=evaluate(cap,rows,preds)
    write(OUT/'summary.json',dict(methods=s,opportunities=opportunities,model_load_seconds=load_s,
        latency_ms={k:latency([p[k] for p in preds]) for k in ['A_complete_ms','public_complete_ms','head_increment_ms','control_complete_ms']},
        latency_scope='Host RGB decode plus full algorithm replay; excludes sensor capture/transport. A* timing conservatively also includes frozen-A head.',
        no_new_calibration=True,authority='FROZEN_SAME_SIMULATOR_NEW_CONFIGURATION_CONFIRMATION'))
    write(OUT/'cases.json',cases)
    assert all(sha(p)==h for p,h in sources.items())
    write(OUT/'completion.json',dict(status='PASS',summary_sha256=sha(OUT/'summary.json'),
        prediction_seal_sha256=sha(OUT/'prediction-seal.json'),resources='Resident models released on process exit'))
    print(json.dumps(dict(metrics={name:s[name]['clear'] for name in s},opportunities=opportunities)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);args=p.parse_args()
    with threadpool_limits(4):main(args.capture)
