"""Two consumed four-column arms using sealed observable RGB proposals."""
import argparse
from collections import Counter
from pathlib import Path
import time

from mz126_central_tof import ROOT, read, write, sha, serial, alert, tof
from mz124_measurement_geometry import metrics
from research_backend import BackendCandidate, DeviceObservation, select_backend


def four_columns(row):
    bins=sorted({tuple(z['theta_bounds_deg']) for z in row['tof_zones']})
    assert len(bins)==8 and len(row['tof_zones'])==64
    chosen=set(bins[2:6])
    selected=[z for z in row['tof_zones'] if tuple(z['theta_bounds_deg']) in chosen]
    assert len(selected)==32
    assert {tuple(z['phi_bounds_deg']) for z in selected}=={tuple(z['phi_bounds_deg']) for z in row['tof_zones']}
    return selected


def predict(row, original, proposals, central):
    zones=four_columns(row) if central else row['tof_zones']
    cache=dict(original,proposals=proposals)
    evidence=serial(tof.allocate(dict(row,tof_zones=zones),cache))
    flag=alert(original,evidence)
    return dict(id=row['id'],candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN',
        selected_zone_ids=[z['zone_id'] for z in zones],proposals=proposals,spatial_evidence=evidence,
        common_radar=original['common_radar'],guard_events=original['guard_events'],
        integrated_yaw_deg=original['integrated_yaw_deg'])


def run(source, correction, attribution, out):
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(parents=True)
    raw=source/'capture-v1/raw.jsonl';basefile=source/'analysis-v1/baseline-predictions.json'
    oldseal=read(source/'analysis-v1/prediction-seal.json');rgbseal=read(correction/'prediction-seal.json')
    assert sha(raw)==oldseal['raw_sha256']==rgbseal['raw_sha256']
    assert sha(basefile)==oldseal['baseline_sha256']==rgbseal['baseline_sha256']
    assert sha(correction/'predictions.json')==rgbseal['predictions_sha256']
    rows=[__import__('json').loads(x) for x in raw.read_text().splitlines()]
    baseline=read(basefile);rgb=read(correction/'predictions.json')
    assert len(rows)==len(baseline)==len(rgb)==288
    assert [r['id'] for r in rows]==list(rgbseal['rgb_sha256'])
    select_backend('scalar-scoring',cpu=BackendCandidate('small-zone-association','cpu',
        lambda:four_columns(rows[0]),lambda _:DeviceObservation('cpu','host CPU','Python / NumPy')),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=out/'backend.json')
    start=time.perf_counter(); predictions={}
    for name,central,inputs in [('baseline',False,baseline),('mz125_full',False,rgb),
                               ('four_original',True,baseline),('four_mz125',True,rgb)]:
        predictions[name]=[predict(r,b,p['proposals'],central) for r,b,p in zip(rows,baseline,inputs)]
    for key,cached in [('baseline',baseline),('mz125_full',rgb)]:
        for a,b in zip(predictions[key],cached):
            assert a['spatial_evidence']==b['spatial_evidence'] and a['candidate']==b['candidate']
    for arm in predictions.values():
        for p,b in zip(arm,baseline):
            old={(e['zone_id'],e['target_slot']):e for e in b['spatial_evidence']}
            for e in p['spatial_evidence']:
                for k in ('zone_box','range_bounds','range_m','coarse_xyz','status'):
                    assert e[k]==old[e['zone_id'],e['target_slot']][k]
    seconds=time.perf_counter()-start
    write(out/'predictions.json',predictions)
    dependencies={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(tof.__file__),
        ROOT/'research/active/dtr-r0/nearfield/mz126_central_tof.py',ROOT/'research/active/dtr-r0/nearfield/mz124_measurement_geometry.py']}
    write(out/'prediction-seal.json',dict(raw_sha256=sha(raw),baseline_sha256=sha(basefile),
        mz125_predictions_sha256=sha(correction/'predictions.json'),dependencies=dependencies,
        predictions_sha256=sha(out/'predictions.json'),seconds=seconds,
        authority='CONSUMED_OBSERVABLE_OUTPUTS_SAVED_BEFORE_THIS_LABEL_PARSE'))
    report=read(source/'analysis-v1/frame-report.json');truth=[r['truth'] for r in report]
    assert [r['id'] for r in report]==[r['id'] for r in rows]
    returns=read(attribution/'returns.json');assert len(returns)==3954
    oldflags=[p['candidate'] for p in baseline]; result=dict(frames=288,seconds=seconds,arms={})
    chosen={p['id']:set(p['selected_zone_ids']) for p in predictions['four_original']}
    dropped=[e for e in returns if e['zone_id'] not in chosen[e['id']]]
    write(out/'discarded-native-returns.json',dropped)
    result['discarded']=dict(returns=len(dropped),active_returns=sum(e['active_support'] for e in dropped),
        hazardous_actor_samples=sum(e['hazardous_actor_contributors'] for e in dropped),
        actual_corridor_samples=sum(e['actual_corridor_hit_contributors'] for e in dropped),
        hazard_frames=len({e['id'] for e in dropped if e['hazardous_actor_contributors']}))
    for name,group in predictions.items():
        flags=[p['candidate'] for p in group]; arm=metrics(rows,truth,flags)
        arm['families']={}
        for family in sorted({r['family'] for r in report}):
            ix=[i for i,r in enumerate(report) if r['family']==family]
            arm['families'][family]=metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
        arm['changed_frames']=[dict(id=r['id'],family=r['family'],truth=t,baseline=a,candidate=b)
            for r,t,a,b in zip(report,truth,oldflags,flags) if a!=b]
        result['arms'][name]=arm
    for a,b in [('four_original','four_mz125'),('mz125_full','four_mz125')]:
        result[a+'_vs_'+b]=[dict(id=r['id'],truth=t,before=p['candidate'],after=q['candidate'])
            for r,t,p,q in zip(rows,truth,predictions[a],predictions[b]) if p['candidate']!=q['candidate']]
    write(out/'summary.json',result)
    assert sha(raw)==oldseal['raw_sha256'] and sha(basefile)==oldseal['baseline_sha256']
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),resources='No persistent CPU process, training or capture'))
    print(__import__('json').dumps({k:{n:v for n,v in a.items() if n not in ('families','changed_frames')} for k,a in result['arms'].items()},indent=2))
    print('Discarded:',result['discarded'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','correction','attribution','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();run(args.source,args.correction,args.attribution,args.output)
