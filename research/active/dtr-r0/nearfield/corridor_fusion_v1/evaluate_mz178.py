"""Offline audit of sealed MZ178 outputs against consumed MZ177 timelines."""
import argparse
import json
from pathlib import Path
from mz178_current_frame import sha,write,ART


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--mz177',type=Path,required=True)
    a=p.parse_args();out=a.predictions.resolve();old=a.mz177.resolve()
    assert out.is_relative_to(ART)
    c=json.loads((out/'completion.json').read_text());assert c['status']=='PASS' and c['evaluator_read'] is False
    for name,h in c['outputs'].items():assert sha(out/name)==h
    ec=json.loads((old/'completion.json').read_text());assert ec['status']=='PASS'
    for name,h in ec['outputs'].items():assert sha(old/name)==h
    rows=[json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()]
    seq=[json.loads(l) for l in (old/'timelines.jsonl').read_text().splitlines()]
    indexed={r['id']:r for r in rows};assert len(indexed)==1920
    arms=['head_control','body_current'];stats={};details=[]
    baseline_miss={f['id'] for s in seq if s['process']=='head_turn' for f in s['frames']
        if f['risk_truth'] is True and f['target_returned'] is True and not f['predictions']['astar']['alert']}
    assert len(baseline_miss)==106
    for arm in arms:
        totals=dict(TP=0,FP=0,FN=0,TN=0,unknown=0,head_supported_FN=0,head_transitions=0,side_pass_duration_s=0.,baseline_alert_lost=0)
        recovered=[];attributed=[];events=0
        for s in seq:
            alerts=[]
            for f in s['frames']:
                r=indexed[f['id']];assert r['episode_id']==s['sequence_id'] and r['time_s']==f['time_s']
                active=r['arms'][arm]['alert'];alerts.append(active)
                truth=f['risk_truth'];totals['unknown' if truth is None else 'TP' if truth and active else 'FN' if truth else 'FP' if active else 'TN']+=1
                totals['baseline_alert_lost']+=int(f['predictions']['astar']['alert'] and not active)
                if s['process']=='side_pass' and active and truth is False:totals['side_pass_duration_s']+=.1
                if s['process']=='head_turn' and truth is True and f['target_returned'] is True and not active:totals['head_supported_FN']+=1
                if f['id'] in baseline_miss and active:
                    recovered.append(f['id'])
                    targets={(v['zone_id'],v['target_index']) for v in f['target_tof']['target_slots']}
                    if any(v['sensor']=='tof' and (v['zone_id'],v['target_index']) in targets for v in r['arms'][arm]['contributors']):attributed.append(f['id'])
            transitions=sum(a!=b for a,b in zip(alerts,alerts[1:]))
            if s['process']=='head_turn':totals['head_transitions']+=transitions
            if s['process']!='head_turn' and s['contact_first_s'] is not None:
                events+=int(any(active and f['risk_truth'] is True and f['time_s']<s['contact_first_s'] for f,active in zip(s['frames'],alerts)))
            details.append(dict(sequence=s['sequence_id'],arm=arm,alert_frames=sum(alerts),transitions=transitions))
        totals.update(recovered=len(recovered),recovered_ids=recovered,target_contributor_recovered=len(attributed),target_contributor_recovered_ids=attributed,advance_contact_events=events)
        totals['precision']=totals['TP']/(totals['TP']+totals['FP']);totals['recall']=totals['TP']/(totals['TP']+totals['FN'])
        stats[arm]=totals
    b=stats['body_current'];h=stats['head_control']
    gates=dict(recover_60percent=b['recovered']>=64,target_attributed_60percent=b['target_contributor_recovered']>=64,
        total_FP=b['FP']<=61,side_pass=b['side_pass_duration_s']<=1.4+1e-8,transitions=b['head_transitions']<=93,
        no_baseline_loss=b['baseline_alert_lost']==0,contact_retention=b['advance_contact_events']==12,
        rotation_specific_gain=b['recovered']-h['recovered']>=11)
    result=dict(status='PASS',decision='A_GATE_MET_B_ELIGIBLE' if all(gates.values()) else 'A_GATE_FAILED_STOP_NO_MEMORY',
        arms=stats,gates=gates,sequence_details=details,inputs={str(out/'completion.json'):sha(out/'completion.json'),str(old/'timelines.jsonl'):sha(old/'timelines.jsonl')},evaluator_code_sha256=sha(__file__))
    write(out/'evaluation.json',result)
    print(json.dumps({**result,'arms':{k:{a:b for a,b in v.items() if not a.endswith('_ids')} for k,v in stats.items()},'sequence_details':len(details)},indent=2))


if __name__=='__main__':main()
