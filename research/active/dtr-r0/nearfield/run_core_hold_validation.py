"""One fixed high-precision plus one-frame-hold complete-layout validation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
import time
import numpy as np
import run_core_transfer as source
from core_hold_validation_spec import specification, check_spec
from core_transfer_spec import bounds, classify, PROFILE
from tof_corridor_calibration import score_frame, decide
from audit_existing_strata_20260920 import tally, paired_ids
from audit_core_workpoint_20260920 import silence_runs

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/ba-core-hold-validation-20260920'
T0=.007085703945147101
T=.4071309640537889
ARMS=('calibration','strong','hold')
OLD_NAMES=('ba-core-transfer-20260920','ba-core-workpoint-transfer-20260920')
CODE=tuple(dict.fromkeys(source.CODE+('core_hold_validation_spec.py','run_core_hold_validation.py',
    'audit_existing_strata_20260920.py','audit_core_workpoint_20260920.py','causal_event_readout_20260920.py')))
read,sha,require,write=source.read,source.sha,source.require,source.write_new


def readout(score, previous, time_s):
    cal=decide(score,T0);strong=decide(score,T)
    held=bool(previous is not None and abs(time_s-previous[0]-.2)<1e-6 and previous[1])
    return dict(calibration=cal,strong=strong,
                hold=dict(alert=bool(strong['alert'] or held),unknown=strong['unknown'],
                          held_only=bool(held and not strong['alert']))),(time_s,strong['alert'])


def preflight():
    from causal_event_readout_20260920 import predict_sequence, selfcheck
    selfcheck()
    spec=specification()
    checked=check_spec(spec,[read(ROOT/'artifacts.local/work'/n/'spec.json') for n in OLD_NAMES])
    # Verify the copied causal readout against the frozen reference on both full old cohorts.
    count=0
    for name in OLD_NAMES:
        groups=defaultdict(list)
        for row in read(ROOT/'artifacts.local/work'/name/'predictions.json'):groups[row['clip_id']].append(row)
        for sequence in groups.values():
            scores=[]
            for r in sequence:
                with np.load(ROOT/'artifacts.local/work'/name/'observations'/f"{r['id']}.npz",allow_pickle=False) as data:
                    scores.append(score_frame(data['boxes'],data['values']))
            reference=predict_sequence([dict(score=s['score'],raw_alert=s['baseline']['alert'],
                definite=s['baseline']['definite_zones']>0,time_s=r['time_s']) for r,s in zip(sequence,scores)])
            prev=None
            for r,s,ref in zip(sequence,scores,reference):
                arms,prev=readout(s,prev,r['time_s'])
                assert all(arms[a]['alert']==ref['flags'][a] and arms[a]['unknown']==ref['unknown'] for a in ARMS)
                count+=1
    assert count==864
    # Reset at a missing sample and at each fresh clip; hold cannot renew itself.
    fake=lambda n: dict(score=n,baseline=dict(alert=n>0,possible_zones=int(n>0),definite_zones=0,
                                           valid_zones=64,ambiguous=n>0,unknown=True))
    s=fake(.8); weak=fake(0)
    a,p=readout(s,None,0);b,p=readout(weak,p,.2);c,p=readout(weak,p,.4)
    assert a['strong']['alert'] and b['hold']['held_only'] and not c['hold']['alert']
    assert not readout(weak,(0,True),.4)[0]['hold']['alert']
    assert not readout(weak,None,0)[0]['hold']['alert']
    return dict(status='PASS',reference_frames=count,spec=checked)


def freeze(out):
    require(not out.exists(),'New output only; no scientific rerun')
    checked=preflight();spec=specification();out.mkdir(parents=True)
    write(out/'spec.json',spec);write(out/'preflight.json',checked)
    report=Path(__file__).with_name('CORE_HOLD_VALIDATION_PROTOCOL_20260920.md')
    (out/'protocol-before-run.md').write_bytes(report.read_bytes())
    inputs=[ROOT/'artifacts.local/work'/n/f for n in OLD_NAMES for f in ('spec.json','prediction-seal.json','predictions.json')]
    inputs += [ROOT/'artifacts.local/work/ba-causal-event-readout-20260920'/f for f in ('protocol.json','results.json')]
    write(out/'protocol.json',dict(id='ba-core-hold-validation-20260920',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        threshold=T,baseline=T0,arms=ARMS,frames=432,clips=36,dt_s=.2,profile=PROFILE,
        spec_sha256=sha(out/'spec.json'),protocol_text_sha256=sha(out/'protocol-before-run.md'),
        code_hashes={n:sha(Path(__file__).with_name(n)) for n in CODE},
        input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        acceptance=dict(core_fp_reduction_fraction=.5,max_onset_delay_s=.2,max_silent_sampled_s=.2,min_event_alert_fraction=5/6),
        placement='Resident authenticated primary UE capture; CPU scalar readout TASK_NOT_GPU_SUITABLE',
        stop='One new432 capture/evaluation; no retuning or automatic successor'))
    print(json.dumps(dict(status='FROZEN',preflight=checked)))


def predict(out):
    source.verify(out);source.check_seal(out,'observation-seal.json')
    write(out/'prediction-start.json',dict(label_access=False,model_calls=0,training_updates=0,python=sys.executable,
        backend='CPU_SCALAR_GEOMETRY',placement='TASK_NOT_GPU_SUITABLE'))
    rows=[];previous={};times=[]
    for obs in read(out/'observations.json'):
        require(sha(out/obs['path'])==obs['sha256'],'Observation binding')
        with np.load(out/obs['path'],allow_pickle=False) as d:
            start=time.perf_counter();s=score_frame(d['boxes'],d['values'])
            arms,previous[obs['clip_id']]=readout(s,previous.get(obs['clip_id']),obs['time_s'])
            times.append(time.perf_counter()-start)
        rows.append({**{k:obs[k] for k in source.IDENTITY},'observation_sha256':obs['sha256'],
                     'raw':s['baseline'],'score':s['score'],'predictions':arms})
    require(len(rows)==432 and len({r['id'] for r in rows})==432,'Complete predictions')
    write(out/'predictions.json',rows)
    write(out/'prediction-seal.json',dict(status='COMPLETE',frames=432,protocol_sha256=sha(out/'protocol.json'),
        hashes={n:sha(out/n) for n in ('predictions.json','prediction-start.json','observation-seal.json')},
        host_score_and_three_readouts_ms=dict(mean=1000*float(np.mean(times)),p50=1000*float(np.median(times)),p95=1000*float(np.quantile(times,.95))),
        timing_scope='Host scalar scoring only; excludes sensing, image decoding, I/O and target-device execution'))
    print('PREDICTIONS_SEALED: all432, three frozen arms')


def evaluate(out):
    p=source.verify(out);source.check_seal(out,'prediction-seal.json');source.check_seal(out,'observation-seal.json')
    require(not (out/'results.json').exists(),'One evaluation only')
    spec,manifest,geometry=source.check_capture(out)
    observations=read(out/'observations.json');preds=read(out/'predictions.json');rows=[];prev={}
    require(len(spec['cases'])==len(geometry)==len(observations)==len(preds)==432,'Coverage')
    for case,g,obs,pred in zip(spec['cases'],geometry,observations,preds):
        require(all(pred[k]==obs[k] for k in source.IDENTITY),'Timeline binding')
        require(all(pred[k]==case[k] for k in ('clip_id','frame_in_clip','time_s')),'Source timeline binding')
        require(pred['observation_sha256']==obs['sha256'],'Observation hash binding')
        truth=source.primitive_truth(dict(case=case,geometry=g),PROFILE);declared=classify(*bounds(case))
        require(truth['truth']==declared['truth'] and truth['boundary']==declared['boundary'],'Native/declared geometry parity')
        raw=pred['raw'];strong=bool(raw['definite_zones'] or (raw['alert'] and pred['score']>=T))
        cal=bool(raw['definite_zones'] or (raw['alert'] and pred['score']>=T0))
        prior=prev.get(pred['clip_id']);held=bool(prior and abs(pred['time_s']-prior[0]-.2)<1e-6 and prior[1])
        require([pred['predictions'][a]['alert'] for a in ARMS]==[cal,strong,strong or held],'Decision parity')
        require(pred['predictions']['hold']['held_only']==bool(held and not strong),'Hold attribution')
        require(all(pred['predictions'][a]['unknown']==(raw['definite_zones']==0) for a in ARMS),'Current UNKNOWN retained')
        prev[pred['clip_id']]=(pred['time_s'],strong)
        target=next(o for o in g['objects'] if o['name']=='target')
        front=target['render_bounds_center_m'][0]-target['render_bounds_extent_m'][0]-g['actual_camera_location_m'][0]
        rows.append({**{k:pred[k] for k in source.IDENTITY},**declared,
            **{k:case[k] for k in ('layout_relation','layer','type_id','background','arrangement_id')},
            'predictions':pred['predictions'],'score':pred['score'],'target_front_axial_m':front})
    masks=dict(all432=lambda r:True,core288=lambda r:r['layout_relation']!='BOUNDARY',
        boundary144=lambda r:r['layout_relation']=='BOUNDARY',outside144=lambda r:r['layout_relation']=='OUTSIDE',
        inside_negative=lambda r:r['layout_relation']=='INSIDE' and not r['truth'])
    for key in ('layer','background','type_id'):
        for v in sorted({r[key] for r in rows}):masks[key+':'+v]=lambda r,k=key,v=v:r['layout_relation']!='BOUNDARY' and r[k]==v
    metrics={a:{n:tally(rows,lambda r,a=a:r['predictions'][a]['alert'],m,.2,'clip_id') for n,m in masks.items()} for a in ARMS}
    for a in ARMS:
        for n,m in masks.items():
            metrics[a][n].pop('zero_return_frames');metrics[a][n]['prediction_unknown']=sum(r['predictions'][a]['unknown'] for r in rows if m(r))
    events=[]
    for event in metrics['calibration']['core288']['events']:
        sequence=[r for r in rows if r['clip_id']==event['episode'] and r['truth']]
        detail={}
        for a in ARMS:
            flags=[r['predictions'][a]['alert'] for r in sequence]
            first=next((r for r in sequence if r['predictions'][a]['alert']),None)
            detail[a]=dict(alerted_frames=sum(flags),alert_fraction=sum(flags)/len(flags),
                first_s=None if first is None else first['time_s'],delay_s=None if first is None else round(first['time_s']-event['entry_s'],9),
                first_target_front_axial_m=None if first is None else first['target_front_axial_m'],**silence_runs(sequence,flags))
        events.append(dict(clip_id=event['episode'],entry_s=event['entry_s'],positive_frames=len(sequence),arms=detail))
    b,c=metrics['calibration']['core288'],metrics['hold']['core288'];limits=p['acceptance']
    checks=dict(nonempty_comparison=len(events)==12 and b['FP']>0,
        all_core_events_detected=c['events_detected']==12,
        core_onset_within_one_sample=all(e['arms']['hold']['delay_s'] is not None and e['arms']['hold']['delay_s']<=limits['max_onset_delay_s'] for e in events),
        material_core_FP_reduction=c['FP']<=(1-limits['core_fp_reduction_fraction'])*b['FP'],
        no_more_core_false_segments=c['false_segments']<=b['false_segments'],
        per_event_continuity=all(e['arms']['hold']['alert_fraction']>=limits['min_event_alert_fraction'] and e['arms']['hold']['max_silent_frames']<=1 for e in events))
    for phase in ('outside144','inside_negative'):
        for key in ('FP','false_segments'):checks[phase+'_'+key+'_nonincrease']=metrics['hold'][phase][key]<=metrics['calibration'][phase][key]
    changes={pair:{n:paired_ids(rows,lambda r,a=a:r['predictions'][a]['alert'],lambda r,b=b:r['predictions'][b]['alert'],m)
        for n,m in masks.items()} for pair,a,b in (('calibration_to_hold','calibration','hold'),('strong_to_hold','strong','hold'))}
    clips=sorted({r['clip_id'] for r in rows})
    clip_first={a:{clip:next((r['time_s'] for r in rows if r['clip_id']==clip and r['predictions'][a]['alert']),None) for clip in clips} for a in ARMS}
    write(out/'frame-results.json',rows)
    write(out/'results.json',dict(metrics=metrics,core_events=events,paired_changes=changes,clip_first_s=clip_first,
        checks=checks,passed=all(checks.values()),scope='ONE_NEW_CONTROLLED_ARRANGEMENT_HOLD_POLICY_VALIDATION',
        limits_are_development_informed_not_human_safety=True,no_retuning=True))
    write(out/'evaluation-seal.json',dict(status='COMPLETE',frames=432,protocol_sha256=sha(out/'protocol.json'),
        hashes={n:sha(out/n) for n in ('frame-results.json','results.json','prediction-seal.json')}))
    print(json.dumps(dict(status='COMPLETE',passed=all(checks.values()),checks=checks,
        core={a:{k:metrics[a]['core288'][k] for k in ('TP','FP','FN','false_segments')} for a in ARMS})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('preflight','freeze','materialize','predict','evaluate'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    if args.stage=='preflight':print(json.dumps(preflight()))
    elif args.stage=='materialize':source.materialize(args.out)
    else:globals()[args.stage](args.out)
