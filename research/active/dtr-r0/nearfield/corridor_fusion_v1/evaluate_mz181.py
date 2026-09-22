"""Offline consumed outcome audit after MZ181 public prediction sealing."""
import json
from collections import Counter
from pathlib import Path
from mz178_current_frame import ART, sha, write


def verify(folder):
    c=json.loads((folder/'completion.json').read_text())
    assert c['status']=='PASS'
    for name,h in c['outputs'].items(): assert sha(folder/name)==h


def main():
    out=ART/'work/mz181-set-intersection-20260918'
    pred=out/'predictions-v1'
    old=ART/'work/continuous-approach-20260918/evaluation-v1'
    verify(pred);verify(old)
    sealed=json.loads((pred/'input-seal.json').read_text());assert not sealed['evaluator_read']
    for name,h in sealed['inputs'].items(): assert sha(name)==h
    rows=[json.loads(l) for l in (pred/'predictions.jsonl').read_text().splitlines()]
    seq=[json.loads(l) for l in (old/'timelines.jsonl').read_text().splitlines()]
    indexed={r['id']:r for r in rows}
    assert len(indexed)==len(rows)==1920
    flat=[f for s in seq for f in s['frames']]
    assert [f['id'] for f in flat]==[r['id'] for r in rows]
    names=['astar','unknown']+[f'depth{i}' for i in range(1,7)]
    stats={};sets={};timing=[];native_removed=[];strata=Counter();fp_cases=[]
    def alert(r,name):
        return r['astar_alert'] if name=='astar' else r['unknown_alert'] if name=='unknown' else r['alerts'][int(name[5:])]
    for name in names:
        c=Counter(); ids=dict(new_tp=[],new_fp=[],lost_unknown_tp=[],removed_unknown_fp=[])
        for s in seq:
            flags=[];risk_times=[]
            for f in s['frames']:
                r=indexed[f['id']]
                assert r['time_s']==f['time_s'] and r['episode_id']==s['sequence_id']
                assert r['astar_alert']==f['predictions']['astar']['alert']
                a=alert(r,name);flags.append(a);truth=f['risk_truth']
                c['UNKNOWN_LABEL' if truth is None else 'TP' if truth and a else 'FN' if truth else 'FP' if a else 'TN']+=1
                c['baseline_alert_lost']+=r['astar_alert'] and not a
                if a and truth is True:
                    risk_times.append(f['time_s'])
                    if not r['astar_alert']: ids['new_tp'].append(f['id'])
                if a and truth is False and not r['astar_alert']: ids['new_fp'].append(f['id'])
                if r['unknown_alert'] and not a:
                    if truth is True: ids['lost_unknown_tp'].append(f['id'])
                    elif truth is False: ids['removed_unknown_fp'].append(f['id'])
                if s['process']=='head_turn':
                    c['head_supported_FN']+=truth is True and f['target_returned'] is True and not a
                if s['process']=='side_pass' and truth is False and a: c['side_pass_false_frames']+=1
            transitions=sum(a!=b for a,b in zip(flags,flags[1:]))
            c['head_transitions']+=transitions if s['process']=='head_turn' else 0
            if s['process']!='head_turn' and s['contact_first_s'] is not None:
                c['advance_contact_events']+=any(t<s['contact_first_s'] for t in risk_times)
            timing.append(dict(arm=name,sequence=s['sequence_id'],process=s['process'],
                first_risk_alert_s=min(risk_times) if risk_times else None,
                last_risk_alert_s=max(risk_times) if risk_times else None,
                alert_frames=sum(flags),transitions=transitions))
        c['new_tp']=len(ids['new_tp']);c['new_fp']=len(ids['new_fp'])
        c['precision']=c['TP']/(c['TP']+c['FP']);c['recall']=c['TP']/(c['TP']+c['FN'])
        c['F1']=2*c['TP']/(2*c['TP']+c['FP']+c['FN'])
        stats[name]=dict(c);sets[name]=ids
    assert stats['unknown']['new_tp']==142 and stats['unknown']['new_fp']==44
    for s in seq:
        for f in s['frames']:
            r=indexed[f['id']]
            if f['id'] in sets['unknown']['new_fp']:
                typ='both' if r['tof'] and r['radar'] else 'valid_ToF_only' if r['tof'] else 'Radar_only'
                strata[typ]+=1
                fp_cases.append(dict(id=f['id'],stratum=typ,retained=r['alerts'][-1],
                    tof_states=[d['state'] for d in r['tof']],
                    tof_witnesses=[d['witness'] for d in r['tof'] if d['witness']]))
            target={(v['zone_id'],v['target_index']) for v in f['target_tof']['target_slots']}
            for d in r['tof']:
                if not d['possible_by_depth'][-1]:
                    native_removed.append(dict(id=f['id'],zone_id=d['zone_id'],target_index=d['target_index'],
                        target_linked=(d['zone_id'],d['target_index']) in target,
                        risk_truth=f['risk_truth'],still_alert=r['alerts'][-1]))
    states=Counter(d['state'] for r in rows for d in r['tof'])
    lat=sorted(r['readout_s']*1000 for r in rows)
    primary=stats['depth6']
    gates=dict(new_tp=primary['new_tp']>=135,new_fp=primary['new_fp']<=25,
               baseline_preserved=primary.get('baseline_alert_lost',0)==0,
               contacts=primary['advance_contact_events']==12)
    times={(r['arm'],r['sequence']):r for r in timing}
    changes=[dict(sequence=s['sequence_id'],before=times['unknown',s['sequence_id']],after=times['depth6',s['sequence_id']])
             for s in seq if any(times['unknown',s['sequence_id']][k]!=times['depth6',s['sequence_id']][k]
                                for k in ('first_risk_alert_s','last_risk_alert_s','transitions','alert_frames'))]
    result=dict(status='PASS',decision='GATE_MET_DEVELOPMENT_ONLY' if all(gates.values()) else 'STOP_FIXED_ANGULAR_REFINEMENT_GATE_NOT_MET',
        gates=gates,stats=stats,sets=sets,new_fp_strata=strata,fp_cases=fp_cases,
        contributor_states=states,native_removed=native_removed,timing=timing,timing_changes=changes,
        geometry_unknown_frames=sum(r['geometry_state']=='UNKNOWN' for r in rows),
        readout_latency_ms=dict(mean=sum(lat)/len(lat),p50=lat[len(lat)//2],p95=lat[int(len(lat)*.95)],max=max(lat)),
        input_hashes={str(p):sha(p) for p in (pred/'completion.json',old/'timelines.jsonl',Path(__file__))})
    write(out/'evaluation.json',result)
    print(json.dumps({k:result[k] for k in ('decision','gates','stats','new_fp_strata','contributor_states','geometry_unknown_frames','readout_latency_ms')},indent=2))
    print('removed contributors',len(native_removed),'target linked',sum(v['target_linked'] for v in native_removed),'timing changes',len(changes))


if __name__=='__main__': main()
