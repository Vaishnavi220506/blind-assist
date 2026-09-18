"""Public UNKNOWN replay first; separately audit consumed saved outcomes."""
import copy
import json
from collections import Counter
from pathlib import Path
from mz178_current_frame import ART, sha, write, support_readout


def public_unknown(row, yaw):
    result = support_readout(row, yaw)
    ambiguous = [c for c in result['contributors'] if c.get('status') == 'SIM_MERGED']
    clean = [c for c in result['contributors'] if c.get('status') != 'SIM_MERGED']
    return dict(contributors=clean, ambiguous=ambiguous,
                state='POSSIBLE_OCCUPANCY' if clean else 'UNKNOWN')


def main():
    root = ART/'work/continuous-approach-20260918'
    old = ART/'work/mz180-disjunctive-geometry-20260918/predictions-v1'
    out = ART/'work/mz180-public-check-20260918'
    assert not out.exists()
    raw = root/'capture-complete/raw.jsonl'
    baseline = root/'baselines-v1/predictions.jsonl'
    inputs = {str(p): sha(p) for p in (raw, baseline, Path(__file__))}
    for folder in (root/'baselines-v1', old):
        completion = json.loads((folder/'completion.json').read_text())
        assert completion['status'] == 'PASS'
        for name, digest in completion['outputs'].items():
            assert sha(folder/name) == digest
    receipt = json.loads((root/'capture-complete/receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['hashes']['raw.jsonl'] == sha(raw)
    rows = [json.loads(l) for l in raw.read_text().splitlines()]
    base = [json.loads(l) for l in baseline.read_text().splitlines()]
    assert len(rows) == len(base) == 1920
    keys, counts = Counter(), Counter()
    predictions = []
    for r, b in zip(rows, base):
        assert all(r[k] == b[k] for k in ('id', 'episode_id', 'time_s'))
        for z in r['tof_zones']:
            counts['zones_with_multiple_slots'] += len(z['targets']) > 1
            for t in z['targets']:
                counts[t['status']] += 1
                if t['status'] == 'SIM_MERGED': keys.update(t.keys())
        v = public_unknown(r, b['yaw'])
        predictions.append(dict(id=r['id'], episode_id=r['episode_id'], time_s=r['time_s'],
                                alert=b['astar_alert'] or bool(v['contributors']), **v))
    # A mixed zone must retain its clean slot and independent Radar unchanged.
    example = next((r,b) for r,b in zip(rows,base) if any(c.get('status') == 'SIM_VALID'
                   for c in support_readout(r,b['yaw'])['contributors']))
    r,b = example; mixed = copy.deepcopy(r)
    z = next(z for z in mixed['tof_zones'] if any(t['status']=='SIM_VALID' for t in z['targets']))
    t = copy.deepcopy(next(t for t in z['targets'] if t['status']=='SIM_VALID'))
    t['status']='SIM_MERGED'; z['targets'].append(t)
    assert public_unknown(mixed,b['yaw'])['contributors'] == public_unknown(r,b['yaw'])['contributors']
    out.mkdir()
    (out/'public-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    write(out/'public-seal.json', dict(inputs=inputs,evaluator_read=False,counts=counts,
          merged_keys=keys,backend='CPU',backend_reason='TASK_NOT_GPU_SUITABLE',
          predictions_sha256=sha(out/'public-predictions.jsonl')))
    # Only now open the existing evaluation timeline and privileged diagnostic output.
    seq = [json.loads(l) for l in (root/'evaluation-v1/timelines.jsonl').read_text().splitlines()]
    saved = [json.loads(l) for l in (old/'predictions.jsonl').read_text().splitlines()]
    public = {r['id']:r for r in predictions}
    historical = {r['id']:r for r in saved}
    assert len(historical) == len(public) == 1920
    stats = {}; ids = {}; timing = []
    names = ('astar','broad','union','ambiguous','oracle','public_unknown')
    broad_recovered = set()
    for s in seq:
        for f in s['frames']:
            if (s['process']=='head_turn' and f['risk_truth'] is True and f['target_returned'] is True
                and not f['predictions']['astar']['alert'] and historical[f['id']]['arms']['broad']['alert']):
                broad_recovered.add(f['id'])
    assert len(broad_recovered)==103
    for name in names:
        v=Counter(); sets={'true':[], 'false':[], 'recovered103':[], 'new_fp':[]}
        for s in seq:
            alerts=[]
            for f in s['frames']:
                k=f['id']; h=historical[k]; p=public[k]; base_alert=f['predictions']['astar']['alert']
                assert all(p[x]==h[x] for x in ('id','episode_id','time_s'))
                assert h['baseline_alert']==base_alert
                alert = base_alert if name=='astar' else p['alert'] if name=='public_unknown' else h['arms'][name]['alert']
                truth=f['risk_truth']; alerts.append(alert)
                v['UNKNOWN' if truth is None else 'TP' if truth and alert else 'FN' if truth else 'FP' if alert else 'TN']+=1
                v['baseline_lost']+=bool(base_alert and not alert)
                if alert and truth is not None: sets['true' if truth else 'false'].append(k)
                if alert and k in broad_recovered: sets['recovered103'].append(k)
                if alert and truth is False and not base_alert: sets['new_fp'].append(k)
                if alert and truth is True and not base_alert: v['full_tp_increment']+=1
            times=[f['time_s'] for f,a in zip(s['frames'],alerts) if a and f['risk_truth'] is True]
            timing.append(dict(arm=name,sequence=s['sequence_id'],first_risk_alert_s=min(times) if times else None,
                               transitions=sum(a!=b for a,b in zip(alerts,alerts[1:])),alert_frames=sum(alerts)))
        stats[name]=dict(v,retained103=len(sets['recovered103']),new_fp=len(sets['new_fp']))
        ids[name]=sets
    mismatches=[k for k in public if public[k]['alert']!=historical[k]['arms']['ambiguous']['alert']]
    assert not mismatches
    diff={key:sorted(set(ids['union'][key])-set(ids['public_unknown'][key])) for key in ('true','false','recovered103')}
    report=dict(stats=stats,union_minus_unknown=diff,unknown_minus_union_true=sorted(set(ids['public_unknown']['true'])-set(ids['union']['true'])),
                public_old_unknown_mismatches=mismatches,ids=ids,timing=timing,
                decision='STOP_CURRENT_PUBLIC_DECOMPOSITION_NOT_ESTABLISHED',
                audit_inputs={str(p):sha(p) for p in (root/'evaluation-v1/timelines.jsonl',old/'predictions.jsonl')})
    write(out/'audit.json',report)
    print(json.dumps(dict(stats=stats,union_minus_unknown={k:len(v) for k,v in diff.items()},counts=counts,merged_keys=keys),indent=2))


if __name__ == '__main__': main()
