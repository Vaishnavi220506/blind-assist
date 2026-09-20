"""Freeze, construct, predict and evaluate one Core-only scalar candidate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import run_core_transfer as source
from core_workpoint_transfer_spec import specification, check_spec
from core_transfer_spec import bounds, classify, PROFILE
from tof_corridor_calibration import score_frame, decide
from ba_camera_corridor_metrics import evaluate_rows
from audit_existing_strata_20260920 import tally
from audit_core_workpoint_20260920 import silence_runs

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT/'artifacts.local/work/ba-core-workpoint-transfer-20260920'
DIAGNOSTIC = ROOT/'artifacts.local/work/ba-core-workpoint-20260920'
BASELINE = .007085703945147101
CANDIDATE = .4071309640537889
ARMS = ('baseline','candidate')
read, sha, require, write_new = source.read, source.sha, source.require, source.write_new
CODE = tuple(dict.fromkeys(source.CODE + ('core_workpoint_transfer_spec.py','run_core_workpoint_transfer.py',
    'audit_core_workpoint_20260920.py','audit_existing_strata_20260920.py')))


def freeze(out):
    require(not out.exists(), 'New immutable output required')
    spec = specification()
    checked = check_spec(spec, read(ROOT/'artifacts.local/work/ba-core-transfer-20260920/spec.json'))
    diag = read(DIAGNOSTIC/'results.json')
    require(diag['roles']['event_onset_ceiling']['threshold'] == CANDIDATE,'Candidate binding')
    out.mkdir(parents=True)
    write_new(out/'spec.json',spec)
    report=Path(__file__).with_name('CORE_WORKPOINT_TRANSFER_PROTOCOL_20260920.md')
    (out/'protocol-before-run.md').write_bytes(report.read_bytes())
    inputs=[DIAGNOSTIC/'results.json',DIAGNOSTIC/'protocol.json']
    protocol=dict(id='ba-core-workpoint-transfer-20260920',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        baseline=BASELINE,candidate=CANDIDATE,frames=432,clips=36,training_updates=0,
        spec_sha256=sha(out/'spec.json'),protocol_text_sha256=sha(out/'protocol-before-run.md'),
        code_hashes={n:sha(Path(__file__).with_name(n)) for n in CODE},
        input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        acceptance=dict(core_fp_reduction_fraction=.25,max_silent_sampled_s=.2,min_event_alert_fraction=5/6),
        placement='Resident authenticated primary UE capture; CPU scalar readout TASK_NOT_GPU_SUITABLE',
        stop='One fixed candidate, one432 capture/evaluation; no threshold repair or automatic successor')
    write_new(out/'protocol.json',protocol);write_new(out/'spec-check.json',checked)
    print(json.dumps(dict(status='FROZEN',frames=432,candidate=CANDIDATE)))


def predict(out):
    p=source.verify(out);source.check_seal(out,'observation-seal.json')
    require(not (out/'prediction-start.json').exists(),'One pass only')
    write_new(out/'prediction-start.json',dict(protocol_sha256=sha(out/'protocol.json'),backend='CPU_SCALAR_GEOMETRY',
        placement='TASK_NOT_GPU_SUITABLE',python=sys.executable,model_calls=0))
    predictions=[];started=time.perf_counter()
    for obs in read(out/'observations.json'):
        require(sha(out/obs['path'])==obs['sha256'],'Observation hash')
        with np.load(out/obs['path'],allow_pickle=False) as data:
            scored=score_frame(data['boxes'],data['values'])
        arms={arm:decide(scored,p[arm]) for arm in ARMS}
        predictions.append({**{k:obs[k] for k in source.IDENTITY},'observation_sha256':obs['sha256'],
            'predictions':arms,'raw':scored['baseline'],'score':scored['score'],
            'anchors':scored['anchors'],'zone_scores':scored['zone_scores']})
    require(len(predictions)==432,'Prediction coverage')
    write_new(out/'predictions.json',predictions)
    write_new(out/'prediction-seal.json',dict(status='COMPLETE',frames=432,protocol_sha256=sha(out/'protocol.json'),
        hashes={n:sha(out/n) for n in ('predictions.json','prediction-start.json','observation-seal.json')},
        elapsed_s=time.perf_counter()-started,model_calls=0,training_updates=0))
    print(json.dumps(dict(status='PREDICTIONS_SEALED',frames=432)))


def evaluate(out):
    p=source.verify(out);source.check_seal(out,'prediction-seal.json')
    source.check_seal(out,'observation-seal.json')
    spec,manifest,geometry=source.check_capture(out)
    require(not (out/'results.json').exists(),'One evaluation only')
    predictions=read(out/'predictions.json');rows=[]
    require(len(predictions)==len(geometry)==len(read(out/'observations.json'))==432,'Complete evaluation binding')
    for case,geo,obs,pred in zip(spec['cases'],geometry,read(out/'observations.json'),predictions):
        require(all(obs[k]==pred[k] for k in source.IDENTITY),'Timeline binding')
        require(obs['sha256']==pred['observation_sha256'],'Observation binding')
        actual=source.primitive_truth(dict(case=case,geometry=geo),PROFILE)
        declared=classify(*bounds(case))
        require(actual['truth']==declared['truth'] and actual['boundary']==declared['boundary'],'Geometry mismatch')
        for arm in ARMS:
            expected=bool(pred['raw']['definite_zones'] or (pred['raw']['alert'] and pred['score']>=p[arm]))
            require(expected==pred['predictions'][arm]['alert'],'Scalar decision parity')
            require(pred['predictions'][arm]['unknown']==(pred['raw']['definite_zones']==0),'UNKNOWN changed')
        require(not pred['predictions']['candidate']['alert'] or pred['predictions']['baseline']['alert'],'Monotonicity')
        rows.append({**{k:pred[k] for k in source.IDENTITY},**declared,
            **{k:case[k] for k in ('layout_relation','layer','type_id','background','arrangement_id')},
            'predictions':pred['predictions'],'score':pred['score']})
    masks={'all432':lambda r:True,'core288':lambda r:r['layout_relation']!='BOUNDARY',
        'inside144':lambda r:r['layout_relation']=='INSIDE','outside144':lambda r:r['layout_relation']=='OUTSIDE',
        'boundary144':lambda r:r['layout_relation']=='BOUNDARY',
        'inside_negative':lambda r:r['layout_relation']=='INSIDE' and not r['truth']}
    metrics={arm:{name:tally(rows,lambda r,a=arm:r['predictions'][a]['alert'],mask,.2,'clip_id')
                  for name,mask in masks.items()} for arm in ARMS}
    for arm in ARMS:
        for name,mask in masks.items():
            metrics[arm][name].pop('zero_return_frames')
            metrics[arm][name]['prediction_unknown']=sum(r['predictions'][arm]['unknown'] for r in rows if mask(r))
    conventional=evaluate_rows(rows,arms=ARMS)
    for arm in ARMS:
        old=conventional['arms'][arm]
        for key in ('TP','FP','FN'):
            require(old['frames']['all_known'][key]==metrics[arm]['all432'][key],'Metric parity '+key)
        require(old['false_alert_segment_count']==metrics[arm]['all432']['false_segments'],'Segment parity')
    events=[]
    for event in metrics['baseline']['core288']['events']:
        sequence=[r for r in rows if r['clip_id']==event['episode'] and r['truth']]
        flags=[r['predictions']['candidate']['alert'] for r in sequence]
        first=next((r['time_s'] for r,yes in zip(sequence,flags) if yes),None)
        events.append(dict(clip_id=event['episode'],positive_frames=len(sequence),alerted_frames=sum(flags),
            alert_fraction=sum(flags)/len(sequence),baseline_first_s=event['first_alert_s'],candidate_first_s=first,
            retained=first is not None and first==event['first_alert_s'],
            lost_ids=[r['id'] for r in sequence if r['predictions']['baseline']['alert'] and not r['predictions']['candidate']['alert']],
            **silence_runs(sequence,flags)))
    b,c=(metrics[a]['core288'] for a in ARMS)
    checks=dict(nonempty_core_events=len(events)>0 and b['FP']>0,
        retain_all_core_onsets=all(e['retained'] for e in events),
        material_core_FP_reduction=c['FP']<=.75*b['FP'],
        no_more_core_false_segments=c['false_segments']<=b['false_segments'],
        per_event_continuity=all(e['alert_fraction']>=5/6 and e['max_silent_frames']<=1 for e in events))
    for phase in ('outside144','inside_negative'):
        checks[phase+'_FP_nonincrease']=metrics['candidate'][phase]['FP']<=metrics['baseline'][phase]['FP']
        checks[phase+'_segments_nonincrease']=metrics['candidate'][phase]['false_segments']<=metrics['baseline'][phase]['false_segments']
    changed=[dict(id=r['id'],clip_id=r['clip_id'],truth=r['truth'],layout_relation=r['layout_relation'],score=r['score'])
             for r in rows if r['predictions']['baseline']['alert']!=r['predictions']['candidate']['alert']]
    subgroups={key:{value:{arm:tally(rows,lambda r,a=arm:r['predictions'][a]['alert'],
         lambda r,k=key,v=value:r['layout_relation']!='BOUNDARY' and r[k]==v,.2,'clip_id') for arm in ARMS}
         for value in sorted({r[key] for r in rows})} for key in ('layer','background','type_id')}
    clip_first={arm:{clip:next((r['time_s'] for r in rows if r['clip_id']==clip and r['predictions'][arm]['alert']),None)
                    for clip in sorted({r['clip_id'] for r in rows})} for arm in ARMS}
    write_new(out/'frame-results.json',rows)
    write_new(out/'results.json',dict(metrics=metrics,subgroups=subgroups,core_events=events,changes=changed,clip_first_s=clip_first,
        passed=all(checks.values()),checks=checks,baseline=p['baseline'],candidate=p['candidate'],
        boundary_reporting=conventional['arms'],scope='ONE_NEW_CONTROLLED_ARRANGEMENT_CORE_ONLY_TRANSFER',
        no_retuning=True,protocol_sha256=sha(out/'protocol.json')))
    write_new(out/'evaluation-seal.json',dict(status='COMPLETE',frames=432,protocol_sha256=sha(out/'protocol.json'),
        hashes={n:sha(out/n) for n in ('frame-results.json','results.json','prediction-seal.json')}))
    print(json.dumps(dict(status='COMPLETE',passed=all(checks.values()),checks=checks,
        core={a:{k:metrics[a]['core288'][k] for k in ('TP','FP','FN','false_segments')} for a in ARMS})))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('freeze','materialize','predict','evaluate'))
    parser.add_argument('--out',type=Path,default=DEFAULT)
    args=parser.parse_args()
    if args.stage=='materialize':
        source.materialize(args.out)
    else:
        globals()[args.stage](args.out)


if __name__=='__main__':main()
