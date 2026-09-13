"""Fixed heuristic zone evidence budgets; full spatial context unchanged."""
import argparse
from pathlib import Path
import time

from mz126_central_tof import ROOT,read,write,sha,serial,tof
from mz124_measurement_geometry import metrics
from research_backend import BackendCandidate,DeviceObservation,select_backend

SOFT=(.25,.5,.5,1.,1.,.5,.5,.25)
UNIT=(1.,)*8
FOUR=(0.,0.,1.,1.,1.,1.,0.,0.)
THRESHOLD=1.


def column_weights(row,profile):
    bins=sorted({tuple(z['theta_bounds_deg']) for z in row['tof_zones']})
    assert len(bins)==8 and len(profile)==8
    return {z['zone_id']:profile[bins.index(tuple(z['theta_bounds_deg']))] for z in row['tof_zones']}


def readout(cached,weights):
    evidence=cached['spatial_evidence']
    active={e['zone_id'] for e in evidence if tof.possible(e['localized_xyz'])}
    certain=any(tof.certain(e['coarse_xyz']) for e in evidence)
    contributions={str(z):weights[z] for z in sorted(active)}
    score=sum(contributions.values())
    flag=bool(cached['common_radar'] or cached['guard_events'] or certain or score>=THRESHOLD)
    return dict(candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN',score=score,
        active_zone_contributions=contributions,zone_weights={str(k):v for k,v in weights.items()},
        certain_coarse=certain,common_radar=cached['common_radar'],guard_events=cached['guard_events'],
        integrated_yaw_deg=cached['integrated_yaw_deg'])


def run(source,correction,attribution,out):
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(parents=True)
    raw=source/'capture-v1/raw.jsonl';basepath=source/'analysis-v1/baseline-predictions.json'
    bseal=read(source/'analysis-v1/prediction-seal.json');cseal=read(correction/'prediction-seal.json')
    assert sha(raw)==bseal['raw_sha256']==cseal['raw_sha256']
    assert sha(basepath)==bseal['baseline_sha256']==cseal['baseline_sha256']
    assert sha(correction/'predictions.json')==cseal['predictions_sha256']
    rows=[__import__('json').loads(x) for x in raw.read_text().splitlines()]
    base=read(basepath);rgb=read(correction/'predictions.json')
    assert len(rows)==len(base)==len(rgb)==288
    assert [r['id'] for r in rows]==list(cseal['rgb_sha256'])
    select_backend('scalar-scoring',cpu=BackendCandidate('zone-evidence-budget','cpu',
        lambda:column_weights(rows[0],SOFT),lambda _:DeviceObservation('cpu','host CPU','Python / NumPy')),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=out/'backend.json')
    start=time.perf_counter()
    for cached in (base,rgb):
        for r,p in zip(rows,cached):assert serial(tof.allocate(r,p))==p['spatial_evidence']
    arms=[('baseline',base,UNIT),('mz125_full',rgb,UNIT),('weighted_original',base,SOFT),
          ('weighted_mz125',rgb,SOFT),('binary_four_mz125',rgb,FOUR)]
    predictions={name:[dict(id=r['id'],**readout(p,column_weights(r,profile))) for r,p in zip(rows,cache)]
                 for name,cache,profile in arms}
    for name,cached in [('baseline',base),('mz125_full',rgb)]:
        assert [p['candidate'] for p in predictions[name]]==[p['candidate'] for p in cached]
    for name,cached,_ in arms:
        assert all(not p['candidate'] or old['candidate'] for p,old in zip(predictions[name],cached))
    seconds=time.perf_counter()-start
    write(out/'predictions.json',predictions)
    deps=[Path(__file__),Path(tof.__file__),ROOT/'research/active/dtr-r0/nearfield/mz126_central_tof.py',ROOT/'research/active/dtr-r0/nearfield/mz124_measurement_geometry.py']
    write(out/'prediction-seal.json',dict(raw_sha256=sha(raw),baseline_sha256=sha(basepath),
        mz125_predictions_sha256=sha(correction/'predictions.json'),predictions_sha256=sha(out/'predictions.json'),
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps},seconds=seconds,soft_weights=SOFT,
        threshold=THRESHOLD,authority='CONSUMED_OBSERVABLE_READOUT_SAVED_BEFORE_THIS_LABEL_PARSE'))
    # Evaluator-only parsing begins here.
    report=read(source/'analysis-v1/frame-report.json');truth=[r['truth'] for r in report]
    assert [r['id'] for r in rows]==[r['id'] for r in report]
    native=read(attribution/'returns.json');assert len(native)==3954
    result=dict(frames=288,seconds=seconds,arms={})
    baseflags=[p['candidate'] for p in base]
    for name,cached,_ in arms:
        group=predictions[name];flags=[p['candidate'] for p in group];m=metrics(rows,truth,flags)
        m['families']={}
        for family in sorted({r['family'] for r in report}):
            ix=[i for i,r in enumerate(report) if r['family']==family]
            m['families'][family]=metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
        m['changed_from_baseline']=[dict(id=r['id'],family=r['family'],truth=t,before=a,after=b)
            for r,t,a,b in zip(report,truth,baseflags,flags) if a!=b]
        m['changed_from_own_reference']=[dict(id=r['id'],family=r['family'],truth=t,before=a['candidate'],after=b)
            for r,t,a,b in zip(report,truth,cached,flags) if a['candidate']!=b]
        lost={r['id'] for r in m['changed_from_own_reference']}
        sources={(r['id'],e['zone_id'],e['target_slot']) for r,p in zip(rows,cached) if r['id'] in lost
                 for e in p['spatial_evidence'] if tof.possible(e['localized_xyz']) or tof.certain(e['coarse_xyz'])}
        exposed=[e for e in native if (e['id'],e['zone_id'],e['slot']) in sources]
        write(out/(name+'-suppressed-alert-support.json'),exposed)
        m['suppressed_alert_support']=dict(returns=len(exposed),hazardous_actor_samples=sum(e['hazardous_actor_contributors'] for e in exposed),
            actual_corridor_samples=sum(e['actual_corridor_hit_contributors'] for e in exposed))
        result['arms'][name]=m
    write(out/'summary.json',result)
    assert sha(raw)==bseal['raw_sha256'] and sha(basepath)==bseal['baseline_sha256'] and sha(correction/'predictions.json')==cseal['predictions_sha256']
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),resources='No persistent process or allocation; inputs unchanged'))
    print(__import__('json').dumps({n:{k:v for k,v in m.items() if k not in ('families','changed_from_baseline','changed_from_own_reference')} for n,m in result['arms'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','correction','attribution','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.correction,a.attribution,a.output)
