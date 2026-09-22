"""One fixed causal scalar replay, with predictions sealed before label join."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from audit_existing_strata_20260920 import tally
from audit_core_workpoint_20260920 import silence_runs

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/ba-causal-event-readout-20260920'
SOURCES = {'primary': ROOT/'artifacts.local/work/ba-core-workpoint-transfer-20260920',
           'context': ROOT/'artifacts.local/work/ba-core-transfer-20260920'}
T = .4071309640537889
T0 = .007085703945147101
ARMS = ('calibration','strong','rise','hold','combined')
CODE = ('causal_event_readout_20260920.py','audit_existing_strata_20260920.py',
        'audit_core_workpoint_20260920.py')


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def predict_sequence(observations):
    """Only scalar observations enter this function; no identities or labels."""
    previous = None
    result=[]
    for obs in observations:
        score=obs['score'];assert math.isfinite(score) and 0 <= score <= 1
        raw=obs['raw_alert'];definite=obs['definite'];time=obs['time_s']
        contiguous=previous is not None and abs(time-previous['time_s']-.2)<1e-6
        p=previous['score'] if contiguous else None
        previous_strong=previous['strong'] if contiguous else False
        strong=bool(definite or (raw and score>=T))
        extrapolated=2*score-p if contiguous else None
        rise=bool(contiguous and raw and score<T and score>p and extrapolated>=T)
        hold=bool(previous_strong)
        flags=dict(calibration=bool(definite or (raw and score>=T0)),strong=strong,
                   rise=strong or rise,hold=strong or hold,combined=strong or rise or hold)
        result.append(dict(score=score,previous_score=p,pred_next=extrapolated,
                           rise_trigger=rise,hold_trigger=hold,previous_strong=previous_strong,
                           previous_raw_alert=previous['raw_alert'] if contiguous else None,
                           unknown=not definite,flags=flags))
        previous=dict(time_s=time,score=score,strong=strong,raw_alert=raw)
    return result


def selfcheck():
    def obs(s,i,raw=True,definite=False):return dict(score=s,time_s=.2*i,raw_alert=raw,definite=definite)
    s=[obs(.5,0),obs(0,1,False),obs(0,2,False)]
    assert [r['flags']['combined'] for r in predict_sequence(s)]==[True,True,False]
    r=predict_sequence([obs(.1,0),obs(.3,1),obs(.1,2)])
    assert r[1]['rise_trigger'] and not r[2]['flags']['combined']
    assert not predict_sequence([obs(.1,0),obs(.3,1,False)])[1]['rise_trigger']
    assert not predict_sequence([obs(.5,0),obs(0,2)])[1]['hold_trigger']
    assert predict_sequence([obs(0,0,False,True)])[0]['flags']['strong']
    assert predict_sequence([obs(T,0)])[0]['flags']['strong']
    assert not predict_sequence([obs(T,0)])[0]['hold_trigger']
    full=[obs(s,i) for i,s in enumerate([.1,.3,.5,.0,.0,.2])]
    complete=predict_sequence(full)
    for n in range(1,len(full)+1):assert predict_sequence(full[:n])==complete[:n]


def freeze():
    assert not OUT.exists(),'Immutable new output required'
    OUT.mkdir(parents=True)
    files={}
    for directory in SOURCES.values():
        for name in ('predictions.json','frame-results.json','prediction-seal.json','evaluation-seal.json'):
            p=directory/name;files[p.relative_to(ROOT).as_posix()]=sha(p)
    protocol=Path(__file__).with_name('CAUSAL_EVENT_READOUT_PROTOCOL_20260920.md')
    (OUT/'protocol-before-run.md').write_bytes(protocol.read_bytes())
    write(OUT/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        threshold=T,baseline=T0,arms=ARMS,input_hashes=files,
        code_hashes={n:sha(Path(__file__).with_name(n)) for n in CODE},
        protocol_text_sha256=sha(OUT/'protocol-before-run.md'),
        authority='CONSUMED_DEVELOPMENT_FIXED_CAUSAL_REPLAY',backend='TASK_NOT_GPU_SUITABLE'))
    print('FROZEN: one causal mechanism, two consumed432 cohorts, no tuning')


def verify():
    p=read(OUT/'protocol.json')
    assert sha(OUT/'protocol-before-run.md')==p['protocol_text_sha256']
    for n,h in p['code_hashes'].items():assert sha(Path(__file__).with_name(n))==h,n
    for n,h in p['input_hashes'].items():assert sha(ROOT/n)==h,n


def predict():
    verify();selfcheck()
    write(OUT/'prediction-start.json',dict(selfchecks='PASS',label_access=False,model_calls=0))
    output={}
    for name,directory in SOURCES.items():
        original=read(directory/'predictions.json')
        seal=read(directory/'prediction-seal.json')
        assert sha(directory/'predictions.json')==seal['hashes']['predictions.json']
        assert len(original)==432 and len({r['id'] for r in original})==432
        groups=defaultdict(list)
        for r in original:groups[r['clip_id']].append(r)
        assert len(groups)==36
        out=[]
        for clip,sequence in groups.items():
            sequence.sort(key=lambda r:r['time_s']);assert len(sequence)==12
            inputs=[]
            for r in sequence:
                raw=r['raw'] if name=='primary' else r['predictions']['raw']
                score=r['score'] if name=='primary' else r['predictions']['calibrated']['score']
                inputs.append(dict(time_s=r['time_s'],score=score,raw_alert=raw['alert'],definite=raw['definite_zones']>0))
            predictions=predict_sequence(inputs)
            for r,pred,obs in zip(sequence,predictions,inputs):
                old=r['predictions']['baseline' if name=='primary' else 'calibrated']
                assert pred['flags']['calibration']==old['alert'] and pred['unknown']==old['unknown']
                if name=='primary':assert pred['flags']['strong']==r['predictions']['candidate']['alert']
                out.append({**{k:r[k] for k in ('id','clip_id','time_s','frame_in_clip')},
                            **pred,'raw_alert':obs['raw_alert'],'definite':obs['definite']})
        output[name]=out
    write(OUT/'predictions.json',output)
    write(OUT/'prediction-seal.json',dict(status='COMPLETE',frames=864,protocol_sha256=sha(OUT/'protocol.json'),
        hashes={n:sha(OUT/n) for n in ('predictions.json','prediction-start.json')}))
    print('PREDICTIONS_SEALED 864; no evaluator labels loaded')


def evaluate():
    verify();seal=read(OUT/'prediction-seal.json')
    assert seal['protocol_sha256']==sha(OUT/'protocol.json')
    for n,h in seal['hashes'].items():assert sha(OUT/n)==h
    output=read(OUT/'predictions.json');results={};all_rows={}
    for name,directory in SOURCES.items():
        labels=read(directory/'frame-results.json');label={r['id']:r for r in labels}
        assert len(label)==432
        rows=[]
        for p in output[name]:
            r=label[p['id']]
            assert all(p[k]==r[k] for k in ('clip_id','time_s','frame_in_clip'))
            rows.append({**p,**{k:r[k] for k in ('truth','boundary','layout_relation','layer','background','type_id')}})
        masks=dict(all432=lambda r:True,core288=lambda r:r['layout_relation']!='BOUNDARY',
            boundary144=lambda r:r['layout_relation']=='BOUNDARY',outside144=lambda r:r['layout_relation']=='OUTSIDE',
            inside_negative=lambda r:r['layout_relation']=='INSIDE' and not r['truth'])
        for key in ('layer','background'):
            for value in sorted({r[key] for r in rows}):
                masks[key+':'+value]=lambda r,k=key,v=value:r['layout_relation']!='BOUNDARY' and r[k]==v
        metrics={a:{n:tally(rows,lambda r,arm=a:r['flags'][arm],mask,.2,'clip_id') for n,mask in masks.items()} for a in ARMS}
        for arm in ARMS:
            for n,mask in masks.items():
                metrics[arm][n].pop('zero_return_frames')
                metrics[arm][n]['prediction_unknown']=sum(r['unknown'] for r in rows if mask(r))
        groups=defaultdict(list)
        for r in rows:groups[r['clip_id']].append(r)
        events=[]
        for clip,seq in groups.items():
            positives=[r for r in seq if r['truth'] and r['layout_relation']!='BOUNDARY']
            if not positives:continue
            detail={a:dict(first_s=next((r['time_s'] for r in positives if r['flags'][a]),None),
                       alerted=sum(r['flags'][a] for r in positives),positive_frames=len(positives),
                       **silence_runs(positives,[r['flags'][a] for r in positives])) for a in ARMS}
            events.append(dict(clip_id=clip,arms=detail))
        changes={a:{phase:{kind:[r['id'] for r in rows if mask(r) and r['flags'][a] and not r['flags']['strong'] and r['truth']==truth]
                          for kind,truth in (('TP_recovered',True),('FP_added',False))} for phase,mask in masks.items()}
                 for a in ('rise','hold','combined')}
        assert all(not r['flags']['strong'] or r['flags']['combined'] for r in rows)
        max_rise=0
        for seq in groups.values():
            run=0
            for r in seq:
                run=run+1 if r['rise_trigger'] else 0;max_rise=max(max_rise,run)
        temporal=dict(rise_triggers=sum(r['rise_trigger'] for r in rows),
            rise_after_no_raw_support=sum(r['rise_trigger'] and r['previous_raw_alert'] is False for r in rows),
            max_consecutive_rise_frames=max_rise,
            hold_without_current_raw=sum(r['hold_trigger'] and not r['raw_alert'] for r in rows))
        results[name]=dict(metrics=metrics,core_events=events,changes_vs_strong=changes,
            temporal=temporal,
            clip_first={a:{clip:next((r['time_s'] for r in seq if r['flags'][a]),None) for clip,seq in groups.items()} for a in ARMS})
        all_rows[name]=rows
    primary=results['primary'];by_event={e['clip_id']:e['arms'] for e in primary['core_events']}
    delayed=[e for e in by_event.values() if e['strong']['first_s']!=e['calibration']['first_s']]
    worst=by_event['workpoint_b0_head_horizontal_inside']['combined']
    checks=dict(all_core_events=primary['metrics']['combined']['core288']['events_detected']==12,
        body36=primary['metrics']['combined']['layer:BODY']['TP']==36,
        both_delayed_onsets_restored=len(delayed)==2 and all(e['combined']['first_s']==e['calibration']['first_s'] for e in delayed),
        worst_HEAD_at_least6of7=worst['alerted']>=6 and worst['positive_frames']==7,
        strong_alerts_preserved=all(not r['flags']['strong'] or r['flags']['combined'] for r in all_rows['primary']))
    target_ids=('f0005','f0008','f0227','f0293')
    targets=[r for r in all_rows['primary'] if r['id'] in target_ids]
    write(OUT/'frame-results.json',all_rows)
    write(OUT/'results.json',dict(cohorts=results,necessary_checks=checks,necessary_targets_pass=all(checks.values()),
        target_frames=targets,threshold=T,scope='CONSUMED_DEVELOPMENT_NOT_FRESH_TRANSFER',fp_budget_not_assigned=True))
    write(OUT/'evaluation-seal.json',dict(status='COMPLETE',hashes={n:sha(OUT/n) for n in ('results.json','frame-results.json','prediction-seal.json')}))
    print(json.dumps(dict(necessary_checks=checks,primary={a:{k:primary['metrics'][a]['core288'][k] for k in ('TP','FP','FN','false_segments','false_sampled_s')} for a in ARMS})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('freeze','predict','evaluate'))
    globals()[parser.parse_args().stage]()
