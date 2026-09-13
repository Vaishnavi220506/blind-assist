"""Consumed central ToF zone selection; no echo-footprint clipping or labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import mz115_spatial_allocation as tof
from mz124_measurement_geometry import metrics

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend

WIDTHS = (1., .5, .25)
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
serial = lambda v: json.loads(json.dumps(v))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def selected_zones(row, fraction):
    assert fraction in WIDTHS
    intr = row['rgb_intrinsics']; width = intr['width']
    left, right = width*(1-fraction)/2, width*(1+fraction)/2
    return [z for z in row['tof_zones'] if fraction == 1. or
            (tof.zone_box(z, intr)[0] >= left-1e-9 and tof.zone_box(z, intr)[2] <= right+1e-9)]


def alert(cached, allocated):
    return bool(cached['common_radar'] or cached['guard_events'] or
                any(tof.certain(e['coarse_xyz']) or tof.possible(e['localized_xyz']) for e in allocated))


def predict(row, cached, fraction):
    zones = selected_zones(row, fraction)
    allocated = serial(tof.allocate(dict(row, tof_zones=zones), cached))
    spans = [tof.zone_box(z, row['rgb_intrinsics']) for z in zones]
    bounds = [min(b[0] for b in spans), max(b[2] for b in spans)] if spans else None
    return dict(id=row['id'], candidate=alert(cached, allocated),
        candidate_state='ALERT' if alert(cached, allocated) else 'UNKNOWN',
        selected_zone_ids=[z['zone_id'] for z in zones], actual_horizontal_bounds_px=bounds,
        spatial_evidence=allocated, common_radar=cached['common_radar'],
        guard_events=cached['guard_events'], integrated_yaw_deg=cached['integrated_yaw_deg'])


def run(source, attribution, out):
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(parents=True)
    rawpath=source/'capture-v1/raw.jsonl'; basepath=source/'analysis-v1/baseline-predictions.json'
    oldseal=read(source/'analysis-v1/prediction-seal.json')
    assert sha(rawpath)==oldseal['raw_sha256'] and sha(basepath)==oldseal['baseline_sha256']
    rows=[json.loads(x) for x in rawpath.read_text().splitlines()]; baseline=read(basepath)
    assert len(rows)==len(baseline)==288
    select_backend('scalar-scoring', cpu=BackendCandidate('python-small-zone-arrays', 'cpu',
        lambda: selected_zones(rows[0], .5), lambda _: DeviceObservation('cpu','host CPU','Python / NumPy')),
        cpu_reason='TASK_NOT_GPU_SUITABLE', record_path=out/'backend.json')
    tick=time.perf_counter()
    predictions={str(f):[predict(r,p,f) for r,p in zip(rows,baseline)] for f in WIDTHS}
    for original, full in zip(baseline,predictions['1.0']):
        assert original['spatial_evidence']==full['spatial_evidence']
        assert original['candidate']==full['candidate']
    for group in predictions.values():
        for old,new in zip(baseline,group):
            mapping={(e['zone_id'],e['target_slot']):e for e in old['spatial_evidence']}
            for e in new['spatial_evidence']:
                before=mapping[e['zone_id'],e['target_slot']]
                for field in ('zone_box','range_m','range_bounds','coarse_xyz','status'):
                    assert e[field]==before[field], field
    seconds=time.perf_counter()-tick
    write(out/'predictions.json', predictions)
    write(out/'prediction-seal.json',dict(raw_sha256=sha(rawpath),baseline_sha256=sha(basepath),
        code_sha256=sha(Path(__file__)), allocation_sha256=sha(Path(tof.__file__)),
        predictions_sha256=sha(out/'predictions.json'), widths=WIDTHS, seconds=seconds,
        authority='CONSUMED_OBSERVABLE_OUTPUTS_SAVED_BEFORE_THIS_LABEL_PARSE'))
    # Evaluator-only inputs first opened below. No truth or lineage in selection.
    report=read(source/'analysis-v1/frame-report.json')
    assert [r['id'] for r in rows]==[r['id'] for r in report]
    truth=[r['truth'] for r in report]; oldflags=[p['candidate'] for p in baseline]
    returns=read(attribution/'returns.json'); nativeframes=read(attribution/'frames.json')
    assert [r['id'] for r in nativeframes]==[r['id'] for r in rows]
    assert [r['baseline_alert'] for r in nativeframes]==oldflags
    assert [r['truth'] for r in nativeframes]==truth
    assert len(returns)==3954 and all(not r['contract_violations'] for r in returns)
    summary=dict(authority='CONSUMED_CONSTRUCTED_DEVELOPMENT',frames=len(rows),seconds=seconds,arms={})
    for name,group in predictions.items():
        flags=[p['candidate'] for p in group]
        arm=metrics(rows,truth,flags)
        arm['families']={}
        for family in sorted({r['family'] for r in report}):
            ids=[i for i,r in enumerate(report) if r['family']==family]
            arm['families'][family]=metrics([rows[i] for i in ids],[truth[i] for i in ids],[flags[i] for i in ids])
        arm['changed_frames']=[dict(id=r['id'],family=r['family'],truth=t,baseline=a,candidate=b)
            for r,t,a,b in zip(report,truth,oldflags,flags) if a!=b]
        arm['zone_counts']=dict(Counter(len(p['selected_zone_ids']) for p in group))
        arm['actual_bounds_px']=sorted({tuple(p['actual_horizontal_bounds_px']) for p in group})
        chosen={p['id']:set(p['selected_zone_ids']) for p in group}
        dropped=[e for e in returns if e['zone_id'] not in chosen[e['id']]]
        write(out/('discarded-returns-'+name+'.json'),dropped)
        arm['discarded_returns']=len(dropped)
        arm['discarded_active_returns']=sum(e['active_support'] for e in dropped)
        arm['discarded_hazardous_actor_samples']=sum(e['hazardous_actor_contributors'] for e in dropped)
        arm['discarded_actual_corridor_samples']=sum(e['actual_corridor_hit_contributors'] for e in dropped)
        arm['frames_discarding_hazardous_actor_samples']=len({e['id'] for e in dropped if e['hazardous_actor_contributors']})
        retainedalerts={p['id'] for p in group if p['candidate']}
        arm['still_alerting_frames_discarding_hazardous_samples']=len({e['id'] for e in dropped if e['hazardous_actor_contributors'] and e['id'] in retainedalerts})
        summary['arms'][name]=arm
    write(out/'summary.json',summary)
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        source_raw_unchanged=sha(rawpath)==oldseal['raw_sha256'],source_baseline_unchanged=sha(basepath)==oldseal['baseline_sha256'],
        resources='Foreground CPU only; no persistent process, training or capture'))
    print(json.dumps({name:{k:v for k,v in a.items() if k not in ('families','changed_frames')} for name,a in summary['arms'].items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','attribution','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); run(args.source,args.attribution,args.output)
