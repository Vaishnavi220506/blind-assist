"""Outcome audit for sealed reconstructed visible-ownership predictions."""
import json
from collections import Counter
from pathlib import Path
from mz178_current_frame import ART, sha, write


def main():
    root=ART/'work/mz183-visible-ownership-20260918';pred=root/'predictions.jsonl'
    comp=json.loads((root/'completion.json').read_text()); assert comp['status']=='PASS' and not comp['task_truth_read']
    for n,h in comp['outputs'].items(): assert sha(root/n)==h
    seal=json.loads((root/'mask-seal.json').read_text()); assert sha(root/'association.json')==seal['association_sha256']
    for p,h in seal['inputs'].items(): assert sha(p)==h
    rows=[json.loads(l) for l in pred.read_text().splitlines()]
    seq=[json.loads(l) for l in (ART/'work/continuous-approach-20260918/evaluation-v1/timelines.jsonl').read_text().splitlines()]
    indexed={r['id']:r for r in rows}; assert len(indexed)==1920
    m181=json.loads((ART/'work/mz181-set-intersection-20260918/evaluation.json').read_text())
    m181fp=set(m181['sets']['depth6']['new_fp']); m181tp=set(m181['sets']['depth6']['new_tp'])
    stats={k:Counter() for k in ('astar','mz181','mz183')}; ids={k:{'new_tp':[],'new_fp':[],'lost_tp':[],'removed_fp':[]} for k in stats}
    strata=Counter(); ownership=Counter(); timing=[]; fp_details=[]
    for s in seq:
        alerts={k:[] for k in stats}
        for f in s['frames']:
            r=indexed[f['id']]; truth=f['risk_truth']; base=bool(f['predictions']['astar']['alert'])
            assert r['astar_alert']==base
            arms={'astar':base,'mz181':r['old_alert'],'mz183':r['alert']}
            for name,a in arms.items():
                alerts[name].append(a)
                stats[name]['UNKNOWN_LABEL' if truth is None else 'TP' if truth and a else 'FN' if truth else 'FP' if a else 'TN']+=1
                stats[name]['baseline_lost']+=bool(base and not a)
                if truth is True and a and not base: ids[name]['new_tp'].append(f['id'])
                if truth is False and a and not base: ids[name]['new_fp'].append(f['id'])
                if truth is True and name=='mz183' and f['id'] in m181tp and not a: ids[name]['lost_tp'].append(f['id'])
                if truth is False and name=='mz183' and f['id'] in m181fp and not a: ids[name]['removed_fp'].append(f['id'])
            if f['id'] in m181fp:
                tof=[d for d in r['tof'] if d['keep']]
                typ='Radar_only' if not tof and r['radar'] else 'both' if tof and r['radar'] else 'valid_ToF_only'
                strata[typ]+=1
                fp_details.append(dict(id=f['id'],stratum=typ,tof=r['tof'],radar=r['radar'],alert=r['alert']))
            for d in r['tof']:
                ownership[d['state']]+=1
            for d in r['tof']:
                ownership['kept' if d['keep'] else 'rejected']+=1
        for name,flags in alerts.items():
            timing.append(dict(arm=name,sequence=s['sequence_id'],first_true_s=min((f['time_s'] for f,a in zip(s['frames'],flags) if a and f['risk_truth'] is True),default=None),
                               alert_frames=sum(flags),transitions=sum(a!=b for a,b in zip(flags,flags[1:]))))
    for name,c in stats.items():
        c['new_tp']=len(ids[name]['new_tp']);c['new_fp']=len(ids[name]['new_fp']);c['precision']=c['TP']/(c['TP']+c['FP']);c['recall']=c['TP']/(c['TP']+c['FN']);c['F1']=2*c['TP']/(2*c['TP']+c['FP']+c['FN'])
    retained=len(set(ids['mz183']['new_tp']) & m181tp);removed=len(set(ids['mz183']['removed_fp']) & m181fp)
    base_extra_tp=len(m181tp);base_fp=len(m181fp)
    gate=dict(extra_tp=retained>=135,fp_removed=removed>=24,baseline_preserved=stats['mz183']['baseline_lost']==0)
    # Sequence timing changes are diagnostic only.
    t={(x['arm'],x['sequence']):x for x in timing}; changes=[]
    for s in seq:
        a,b=t['mz181',s['sequence_id']],t['mz183',s['sequence_id']]
        if any(a[k]!=b[k] for k in ('first_true_s','alert_frames','transitions')):changes.append(dict(sequence=s['sequence_id'],before=a,after=b))
    result=dict(status='PASS',decision='MZ183_GATE_MET_CONDITIONAL' if all(gate.values()) else 'MZ183_VISIBLE_OWNERSHIP_GATE_NOT_MET',gate=gate,stats=stats,sets=ids,
                baseline_mz181=dict(extra_tp=base_extra_tp,extra_fp=base_fp),retained_mz181_extra_tp=retained,removed_mz181_extra_fp=removed,
                fp_strata=strata,ownership=ownership,fp_details=fp_details,timing=timing,timing_changes=changes,
                mask_authority=seal.get('authority'),mask_counts=dict(frames=len(rows),mask_files=len(list((root/'masks').glob('*.npz')))),
                input_hashes={str(p):sha(p) for p in (pred,ART/'work/continuous-approach-20260918/evaluation-v1/timelines.jsonl',Path(__file__))})
    write(root/'evaluation.json',result)
    print(json.dumps({k:result[k] for k in ('decision','gate','stats','baseline_mz181','retained_mz181_extra_tp','removed_mz181_extra_fp','fp_strata','ownership','timing_changes')},indent=2))


if __name__=='__main__': main()
