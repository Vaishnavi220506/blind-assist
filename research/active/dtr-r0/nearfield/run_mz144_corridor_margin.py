"""One direct corridor margin readout with independent nonvisual evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from mz115_spatial_allocation import possible, certain, zone_box
from mz128_zone_weighting import column_weights, FOUR
from mz136_incumbent import public_observations
from run_mz139_surface_fit import selected_jsonl, local_dependencies
from run_mz143_corridor_evidence import (ROOT, CODE, CAP, INC, read, sha, write,
    truth, augmented_score, native_account, BackendCandidate, DeviceObservation,
    select_backend, operating_threshold)
from evaluate_mz136_corridor_pair import retention


def independent(row, corrected, radar):
    """RGB cannot veto frozen Radar, raw certain ToF, or outside-RGB support."""
    intr=row['rgb_intrinsics'];zones={z['zone_id']:z for z in row['tof_zones']}
    weights=column_weights(row,FOUR);outside=set();strong=[];records=[]
    for item in corrected['spatial_evidence']:
        zid=item['zone_id'];box=zone_box(zones[zid],intr)
        full=0<=box[0]<=box[2]<=intr['width']-1 and 0<=box[1]<=box[3]<=intr['height']-1
        yes=certain(item['coarse_xyz']);poss=possible(item['coarse_xyz'])
        if yes:strong.append([zid,item['target_slot']])
        if not full and poss:outside.add(zid)
        records.append(dict(zone=zid,slot=item['target_slot'],certain=bool(yes),
            partial_or_outside_rgb=not full,possible=bool(poss)))
    outside_score=float(sum(weights[z] for z in outside))
    value=bool(radar['candidate'] or strong or outside_score>=1.)
    return dict(candidate=value,radar=bool(radar['candidate']),
        raw_certain_slots=strong,outside_possible_zones=sorted(outside),
        outside_score=outside_score,raw_support_records=records)


def infer(rows,receipt,caches,out,label,started):
    from mz144_corridor_margin import extract_margin
    preds=[];hashes={};yaw=0.;previous=None;tick=time.perf_counter()
    for i,row in enumerate(rows):
        if row['episode_id']!=previous:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        previous=row['episode_id']
        path=(CAP/row['rgb_path']).resolve();assert path.is_relative_to(CAP.resolve())
        hashes[row['rgb_path']]=sha(path)
        assert hashes[row['rgb_path']]==receipt['hashes'][row['rgb_path']]
        image=cv2.imread(str(path));assert image is not None
        raw=extract_margin(row,image,yaw)
        assert np.isfinite(raw['score'])
        cached=caches[row['id']]
        fixed=independent(row,cached['corrected'],cached['radar'])
        preds.append(dict(id=row['id'],margin=raw,independent=fixed))
        if (i+1)%24==0:
            print(json.dumps(dict(stage='margin',split=label,frames=i+1,
                total=len(rows),seconds=time.perf_counter()-tick)),flush=True)
            assert time.perf_counter()-started<600,'10 minute margin run cap exhausted'
    write(out/(label+'-predictions.json'),preds)
    write(out/(label+'-prediction-seal.json'),dict(
        predictions_sha256=sha(out/(label+'-predictions.json')),
        rgb_sha256=hashes,seconds=time.perf_counter()-tick,
        authority='PUBLIC_MARGIN_PREDICTIONS_BEFORE_SELECTED_EVALUATOR_PARSE'))
    return preds,time.perf_counter()-tick


def combined_scores(preds):
    # Independent branch is an unconditional alarm at every finite fitted cutoff.
    return np.array([100. if p['independent']['candidate'] else p['margin']['score'] for p in preds],float)


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);started=time.perf_counter()
    receipt=read(CAP/'receipt.json');spec=read(CAP/'spec.json')
    assert receipt['status']=='PASS' and sha(CAP/'spec.json')==receipt['spec_sha256']
    for name in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/name)==receipt['hashes'][name]
    seal=read(INC/'prediction-seal.json');done=read(INC/'completion.json')
    assert done['status']=='PASS' and sha(INC/'prediction-seal.json')==done['prediction_seal_sha256']
    assert sha(INC/'nominal/predictions.json')==seal['predictions_sha256']['nominal']
    cached=read(INC/'nominal/predictions.json')
    caches={p['id']:dict(prediction=p,corrected=c,radar=r) for p,c,r in
        zip(cached['predictions'],cached['corrected'],cached['radar'])}
    ids={s:{f['id'] for f in spec['frames'] if f['split']==s} for s in ('train','dev')}
    assert len(ids['train'])==192 and len(ids['dev'])==48
    rows={s:public_observations(selected_jsonl(CAP/'raw.jsonl',ids[s])) for s in ids}
    assert all(rows[s]==selected_jsonl(INC/'nominal/raw.jsonl',ids[s]) for s in rows)
    sources=local_dependencies(__file__)
    sources.update(local_dependencies(CODE/'mz144_corridor_margin.py'))
    for source in sources:
        path=Path(source);dest=out/'source-snapshot'/path.relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    shutil.copyfile(CODE/'MZ144_PROTOCOL_20260916.md',out/'protocol-before-fit.md')
    write(out/'freeze.json',dict(sources=sources,protocol_sha256=sha(out/'protocol-before-fit.md'),
        inputs={str(p):sha(p) for p in [CAP/'raw.jsonl',CAP/'evaluator.jsonl',CAP/'spec.json',
            CAP/'receipt.json',INC/'nominal/predictions.json',INC/'nominal/raw.jsonl']},
        selected_frame_ids={s:[r['id'] for r in rows[s]] for s in rows},
        original_test_access=False,scope='CONSUMED_CONTROLLED_DEVELOPMENT'))
    from mz144_corridor_margin import extract_margin
    first=rows['train'][0];image=cv2.imread(str(CAP/first['rgb_path']))
    assert sha(CAP/first['rgb_path'])==receipt['hashes'][first['rgb_path']]
    select_backend('batch-tensor',cpu=BackendCandidate('numpy-opencv-cpu','cpu',
        lambda:extract_margin(first,image,first['delta_yaw']),
        lambda _:DeviceObservation('cpu','host CPU','NumPy '+np.__version__+' OpenCV '+cv2.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
        capabilities={'reason':'Implemented interval-consensus and robust foreground operators are CPU-only'})
    train,ttrain=infer(rows['train'],receipt,caches,out,'train',started)
    es_train=selected_jsonl(CAP/'evaluator.jsonl',ids['train'])
    assert [e['id'] for e in es_train]==[r['id'] for r in rows['train']]
    y=np.array([truth(e) for e in es_train],bool)
    base=np.array([caches[r['id']]['prediction']['candidate'] for r in rows['train']],bool)
    train_scores=combined_scores(train)
    needed=np.array([not p['independent']['candidate'] for p in train])&y&base
    cutoff=float(train_scores[needed].min()) if needed.any() else 0.
    train_result=augmented_score(rows['train'],es_train,y,train_scores,cutoff,base,spec,'train')
    write(out/'operating-point-seal.json',dict(cutoff=cutoff,
        train_prediction_sha256=sha(out/'train-predictions.json'),
        train_score=train_result,required_visual_true_frames=int(needed.sum()),
        missing_required_visual_frames=[rows['train'][i]['id'] for i in np.where(needed)[0] if train[i]['margin']['unknown']],
        authority='ONE_TRAIN_CUTOFF_SEALED_BEFORE_DEV_PREDICTION_AND_TRUTH_PARSE'))
    print(json.dumps(dict(stage='cutoff_sealed',cutoff=cutoff,train=train_result['metrics'])),flush=True)
    dev,tdev=infer(rows['dev'],receipt,caches,out,'dev',started)
    scores=combined_scores(dev);flags=scores>=cutoff
    write(out/'dev-decisions.json',[dict(id=r['id'],score=float(s),candidate=bool(p),
        candidate_state='ALERT' if p else 'UNKNOWN') for r,s,p in zip(rows['dev'],scores,flags)])
    write(out/'decision-seal.json',dict(decisions_sha256=sha(out/'dev-decisions.json'),
        operating_point_sha256=sha(out/'operating-point-seal.json'),
        dev_prediction_seal_sha256=sha(out/'dev-prediction-seal.json'),
        authority='DEV_DECISIONS_SEALED_BEFORE_DEV_EVALUATOR_PARSE'))
    es=selected_jsonl(CAP/'evaluator.jsonl',ids['dev'])
    assert [e['id'] for e in es]==[r['id'] for r in rows['dev']]
    gt=np.array([truth(e) for e in es],bool)
    baseline=np.array([caches[r['id']]['prediction']['candidate'] for r in rows['dev']],bool)
    br=augmented_score(rows['dev'],es,gt,baseline.astype(float),.5,baseline,spec,'dev')
    assert br['metrics']==dict(TP=24,FP=13,FN=0,TN=11,UNKNOWN=11)
    result=augmented_score(rows['dev'],es,gt,scores,cutoff,baseline,spec,'dev')
    result['retention']=retention(result,br)
    family_ok=all(result['families'][f]['FP']<=br['families'][f]['FP'] for f in br['families'])
    passed=bool(not result['lost_baseline_tp'] and result['metrics']['FP']<=10 and family_ok and
        result['events']['false_alert_segments']<=br['events']['false_alert_segments'] and
        result['retention']['incumbent_events_and_timing_retained'])
    write(out/'native-contributors.json',native_account(rows['dev'],es,flags,baseline))
    summary=dict(baseline=br,candidate=result,cutoff=cutoff,joint_gain=passed,
        family_fp_noninferior=family_ok,decision='DIRECT_MARGIN_DEV_GAIN' if passed else 'DIRECT_MARGIN_NO_JOINT_DEV_GAIN',
        geometry_unknown=dict(train=sum(p['margin']['unknown'] for p in train),dev=sum(p['margin']['unknown'] for p in dev)),
        independent_alert_frames=dict(train=sum(p['independent']['candidate'] for p in train),dev=sum(p['independent']['candidate'] for p in dev)),
        seconds=dict(train=ttrain,dev=tdev,total=time.perf_counter()-started),
        original_test_access=False,authority='CONSUMED_CONTROLLED_DEVELOPMENT_NOT_FRESH_CONFIRMATION')
    write(out/'summary.json',summary)
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        decision_seal_sha256=sha(out/'decision-seal.json'),resource_state='CPU process exits; no persistent allocation'))
    print(json.dumps(dict(decision=summary['decision'],cutoff=cutoff,baseline=br['metrics'],
        candidate=result['metrics'],events=result['events'],seconds=summary['seconds']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.output)
