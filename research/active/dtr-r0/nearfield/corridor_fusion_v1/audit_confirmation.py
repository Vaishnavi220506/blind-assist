"""Independent sealed-output accounting, without new model predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from run_e1 import read,write,sha,dataset,labels
from run_confirmation import WORK,DEFAULT_CAPTURE,AT,CT,GT,QROOT


def segments(rows,truth,flags):
    result=[];active=None;previous=None
    for row,y,p in zip(rows,truth,flags):
        if row['episode_id']!=previous or not y:active=None
        previous=row['episode_id']
        if y and active is None:
            active=dict(episode=previous,onset_s=row['time_s'],first_alert_s=None)
            result.append(active)
        if y and p and active['first_alert_s'] is None:active['first_alert_s']=row['time_s']
    return result


def audit(out,cap):
    done=read(out/'completion.json');assert done['status']=='PASS'
    assert sha(out/'summary.json')==done['summary_sha256']
    seal=read(out/'prediction-seal.json')
    assert sha(out/'prediction-seal.json')==done['prediction_seal_sha256']
    assert sha(out/'predictions.json')==seal['predictions_sha256']
    assert sha(out/'latency-samples.json')==seal['latency_sha256']
    for path,h in read(out/'freeze.json')['bindings'].items():assert sha(path)==h,path
    for path,h in read(out/'input-seal.json')['inputs'].items():assert sha(path)==h,path
    data=dataset(cap,'confirmation');es,y=labels(data);rows=data['rows']
    predictions=read(out/'predictions.json');times=read(out/'latency-samples.json')
    assert [p['id'] for p in predictions]==[r['id'] for r in rows]==[t['id'] for t in times]
    a=np.array([p['A_alert'] for p in predictions],bool)
    s=np.array([p['S1_alert'] for p in predictions],bool)
    called=np.array([p['invoked'] for p in predictions],bool)
    for p,t in zip(predictions,times):
        assert p['A_alert']==(p['A_score']>=AT)
        assert p['invoked']==(p['A_alert'] and p['gate_probability']>=GT)
        assert p['veto']==(p['invoked'] and p['C_score'] is not None and p['C_score']<CT)
        assert p['S1_alert']==(p['A_alert'] and not p['veto'])
        assert t['S1_seconds']>=t['A_seconds']>0 and t['invoked']==p['invoked']
    summary=read(out/'summary.json')
    for key,p in [('A',a),('S1',s)]:
        counts=dict(TP=int((y&p).sum()),FP=int((~y&p).sum()),FN=int((y&~p).sum()))
        assert all(summary['reports'][key]['metrics'][k]==v for k,v in counts.items())
    ae,se=segments(rows,y,a),segments(rows,y,s)
    assert len(ae)==len(se)==30
    event_changes=[dict(A=x,S1=z) for x,z in zip(ae,se) if x!=z]
    lost=[dict(A=x,S1=z) for x,z in zip(ae,se) if x['first_alert_s'] is not None and z['first_alert_s'] is None]
    tp_loss=int((y&a&~s).sum());fp_gain=int((~y&a&~s).sum())
    f1a=summary['reports']['A']['f1'];f1s=summary['reports']['S1']['f1']
    if fp_gain>0 and tp_loss==0 and not event_changes:
        decision='RETAIN_TARGETED_CONTROLLED_COMPONENT'
    elif 1<=tp_loss<=2 and f1s>f1a and fp_gain/max(1,int((~y&a).sum()))>=.10 and not lost:
        decision='RETAIN_PRECISION_ORIENTED_POINT_WITH_EXPLICIT_COST'
    elif fp_gain==0 or f1s<=f1a:
        decision='NO_USEFUL_TRANSFER_CLOSE_ONLINE_DAV2_RECIPE'
    else:
        decision='UNPROMOTED_TRADEOFF_REQUIRES_USER_INTERPRETATION'
    native=read(out/'native.json')
    by_id={p['id']:p for p in native['A']['cases']}
    specframes={f['id']:f for f in data['spec']['frames']}
    changed=[];invoked=[]
    for row,e,p,gt in zip(rows,es,predictions,y):
        f=specframes[row['id']]
        record=dict(p,truth=bool(gt),family=e['family'],episode=row['episode_id'],time_s=row['time_s'],
            wall_distance_m=f['wall_distance_m'],background_style=f['background_style'],
            native_corridor_contributors=by_id[row['id']].get('corridor_contributor_samples',0))
        if p['invoked']:invoked.append(record)
        if p['A_alert']!=p['S1_alert']:changed.append(record)
    result=dict(status='PASS',decision=decision,frames=len(rows),events=30,
        event_details_identical=ae==se,event_changes=event_changes,lost_A_events=lost,
        A_detected_events=sum(e['first_alert_s'] is not None for e in ae),
        S1_detected_events=sum(e['first_alert_s'] is not None for e in se),
        A_TP_lost=tp_loss,A_FP_removed=fp_gain,changed_frames=changed,called_frames=invoked,
        called_true_frames=int((called&y).sum()),called_false_frames=int((called&~y).sum()),
        positive_q100_above_root=sum(bool(gt) and p['nearest_q100'] is not None and p['nearest_q100']>QROOT for gt,p in zip(y,predictions)),
        removed_native_supported_frames=sum(c['native_corridor_contributors']>0 for c in changed),
        removed_native_contributors=sum(c['native_corridor_contributors'] for c in changed),
        native_radar_lineage='NOT_EVALUABLE',source_evaluator_boundary='Source labels are audit-only after prediction seal')
    write(out/'independent-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('called_frames','changed_frames')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'evaluation-v1');parser.add_argument('--capture',type=Path,default=DEFAULT_CAPTURE)
    args=parser.parse_args();audit(args.output,args.capture)
