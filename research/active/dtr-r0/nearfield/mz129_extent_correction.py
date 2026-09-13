"""Fixed consumed refinement of MZ125 ROIs within the MZ128 readout."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from mz128_zone_weighting import FOUR, THRESHOLD, column_weights, readout
from mz126_central_tof import ROOT, read, write, sha, serial, tof
from mz124_measurement_geometry import refine_support, slant_box, metrics

SOURCE = ROOT/'artifacts.local/work/mz123-frozen-early-20260913/returned-v1'
CORRECTION = ROOT/'artifacts.local/work/mz125-observable-correction-20260913/correction-v1r1'
INCUMBENT = ROOT/'artifacts.local/work/mz128-zone-weighting-20260913/replay-v1'
BRIEF = Path(__file__).with_name('MZ129_EXTENT_BRIEF_20260914.md')


def refine_frame(row, cached):
    weights = column_weights(row, FOUR)
    old = readout(cached, weights)
    records = []
    yaw = cached['integrated_yaw_deg']
    for e in cached['spatial_evidence']:
        p = row['camera_pitch_deg']; dy = .5+.2*row['time_s']
        outer = slant_box(e['roi'], e['range_bounds'], row['rgb_intrinsics'],
                         (p-.5,p+.5), (yaw-dy,yaw+dy), row['camera_in_body_m'][2])
        assert serial(outer) == e['localized_xyz']
        result = refine_support(e['roi'], e['range_bounds'], row, yaw)
        before = tof.possible(outer)
        assert not result['possible'] or before
        records.append(dict(zone_id=e['zone_id'],slot=e['target_slot'],status=e['status'],
                            allocation_state=e['state'],weight=weights[e['zone_id']],
                            coarse_possible=before,range_bounds=e['range_bounds'],**result))
    active = {e['zone_id'] for e in records if e['possible']}
    contributions = {str(z):weights[z] for z in sorted(active)}
    score = sum(contributions.values())
    flag = bool(old['common_radar'] or old['guard_events'] or old['certain_coarse'] or score>=THRESHOLD)
    assert not flag or old['candidate']
    pred = dict(old, candidate=flag, candidate_state='ALERT' if flag else 'UNKNOWN',
                score=score, active_zone_contributions=contributions)
    return pred, records


def events(rows, truth, flags):
    result = []; active = None; previous = None
    for row, target, flag in zip(rows, truth, flags):
        if row['episode_id'] != previous or not target:
            active = None
        previous = row['episode_id']
        if target and active is None:
            active = dict(episode=previous,onset_s=row['time_s'],first_alert_s=None)
            result.append(active)
        if target and flag and active['first_alert_s'] is None:
            active['first_alert_s'] = row['time_s']
    return result


def predict(output, radar_dir):
    output, radar_dir = output.absolute(), radar_dir.absolute()
    assert not output.exists() and output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    output.mkdir(parents=True)
    raw = SOURCE/'capture-v1/raw.jsonl'
    basepath = SOURCE/'analysis-v1/baseline-predictions.json'
    correctionpath = CORRECTION/'predictions.json'
    oldpath = INCUMBENT/'predictions.json'
    cs, bs = read(CORRECTION/'prediction-seal.json'), read(INCUMBENT/'prediction-seal.json')
    assert sha(raw)==cs['raw_sha256']==bs['raw_sha256']
    assert sha(basepath)==cs['baseline_sha256']==bs['baseline_sha256']
    assert sha(correctionpath)==cs['predictions_sha256']==bs['mz125_predictions_sha256']
    assert sha(oldpath)==bs['predictions_sha256']
    for path, digest in bs['dependencies'].items(): assert sha(ROOT/path)==digest
    rows = [json.loads(x) for x in raw.read_text().splitlines()]
    cached = read(correctionpath)
    old = read(oldpath)['binary_four_mz125']
    assert len(rows)==len(cached)==len(old)==288
    assert [r['id'] for r in rows]==[p['id'] for p in old]==list(cs['rgb_sha256'])
    for r,p,b in zip(rows,cached,old):
        assert serial(tof.allocate(r,p))==p['spatial_evidence']
        assert dict(id=r['id'],**readout(p,column_weights(r,FOUR)))==b
    start = time.perf_counter()
    refined = [refine_frame(r,p) for r,p in zip(rows,cached)]
    tof_seconds = time.perf_counter()-start
    write(output/'tof-records.json',[dict(id=r['id'],returns=p[1]) for r,p in zip(rows,refined)])
    # Radar producer validates complete original flow carry before correction.
    radar = read(radar_dir/'predictions.json')
    rs = read(radar_dir/'prediction-seal.json')
    assert sha(radar_dir/'predictions.json')==rs['predictions_sha256']
    assert rs['raw_sha256']==sha(raw) and rs['baseline_sha256']==sha(basepath)
    assert rs['correction_sha256']==sha(correctionpath)
    for path,digest in rs['source_hashes'].items(): assert sha(ROOT/path)==digest
    for path,digest in rs['input_hashes'].items(): assert sha(Path(path))==digest
    assert len(radar)==288 and [p['id'] for p in radar]==[r['id'] for r in rows]
    arms = dict(baseline=old,tof=[],radar=[],combined=[])
    for row, baseline, refined_pair, rp in zip(rows,old,refined,radar):
        narrow = dict(id=row['id'],**refined_pair[0]); arms['tof'].append(narrow)
        for arm, geom in [('radar',baseline),('combined',narrow)]:
            flag = bool(geom['score']>=THRESHOLD or geom['certain_coarse'] or rp['candidate'])
            arms[arm].append(dict(geom,candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN',
                                 common_radar=rp['common_radar'],guard_events=rp['guard_events']))
    write(output/'predictions.json',arms)
    deps = [Path(__file__),BRIEF,Path(__file__).with_name('mz129_radar_box_correction.py'),
            Path(__file__).with_name('mz124_measurement_geometry.py')]
    inputs = [raw,basepath,correctionpath,oldpath,radar_dir/'predictions.json',radar_dir/'prediction-seal.json']
    write(output/'prediction-seal.json',dict(predictions_sha256=sha(output/'predictions.json'),
        tof_records_sha256=sha(output/'tof-records.json'),dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},tof_seconds=tof_seconds,
        cpu_reason='TASK_NOT_GPU_SUITABLE',authority='CONSUMED_OBSERVABLE_OUTPUTS_BEFORE_SCORING'))
    return rows, arms


def score(output, rows, arms):
    seal = read(output/'prediction-seal.json')
    assert sha(output/'predictions.json')==seal['predictions_sha256']
    report = read(SOURCE/'analysis-v1/frame-report.json')
    assert [r['id'] for r in rows]==[r['id'] for r in report]
    truth = [r['truth'] for r in report]
    baseline = [p['candidate'] for p in arms['baseline']]
    old_events = events(rows,truth,baseline)
    result = dict(frames=len(rows),arms={})
    for name, predictions in arms.items():
        flags = [p['candidate'] for p in predictions]
        m = metrics(rows,truth,flags)
        ev = events(rows,truth,flags)
        m['events'] = ev
        m['later_or_lost_events'] = [dict(before=a,after=b) for a,b in zip(old_events,ev)
            if a['first_alert_s'] is not None and (b['first_alert_s'] is None or b['first_alert_s']>a['first_alert_s'])]
        m['changed'] = [dict(id=r['id'],family=label['family'],truth=t,before=a,after=b)
            for r,label,t,a,b in zip(rows,report,truth,baseline,flags) if a!=b]
        m['lost_TP'] = sum(t and a and not b for t,a,b in zip(truth,baseline,flags))
        m['added_FP'] = sum(not t and not a and b for t,a,b in zip(truth,baseline,flags))
        m['families'] = {}
        for family in sorted({r['family'] for r in report}):
            ix = [i for i,r in enumerate(report) if r['family']==family]
            m['families'][family]=metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
        m['retention_criterion'] = m['FP']<result['arms'].get('baseline',m)['FP'] and not m['lost_TP'] and not m['later_or_lost_events']
        result['arms'][name]=m
    nativepath = ROOT/'artifacts.local/work/mz125-observable-correction-20260913/attribution-v1/returns.json'
    native = {(r['id'],r['zone_id'],r['slot']):r for r in read(nativepath)}
    assert len(native)==3954
    allrecords = read(output/'tof-records.json')
    excluded = [dict(id=r['id'],**e,native=native[(r['id'],e['zone_id'],e['slot'])])
                for r in allrecords for e in r['returns'] if e['coarse_possible'] and not e['possible']]
    write(output/'excluded-tof-native.json',excluded)
    result['tof_refinement'] = dict(returns=sum(len(r['returns']) for r in allrecords),excluded_returns=len(excluded),
        positive_weight_exclusions=sum(e['weight']>0 for e in excluded),
        statuses=dict(Counter(e['status'] for e in excluded)),states=dict(Counter(e['allocation_state'] for e in excluded)),
        native_corridor_hit_samples=sum(e['native']['actual_corridor_hit_contributors'] for e in excluded),
        hazardous_actor_samples=sum(e['native']['hazardous_actor_contributors'] for e in excluded))
    result['label_sha256']=sha(SOURCE/'analysis-v1/frame-report.json')
    result['native_attribution_sha256']=sha(nativepath)
    write(output/'summary.json',result)
    for path,digest in seal['input_hashes'].items(): assert sha(ROOT/path)==digest
    write(output/'completion.json',dict(status='PASS',summary_sha256=sha(output/'summary.json'),
        resources='No persistent process or allocation; source inputs unchanged'))
    print(json.dumps({n:{k:m[k] for k in ('TP','FP','FN','precision','false_segments','max_detected_delay_s','lost_TP','added_FP','retention_criterion')} for n,m in result['arms'].items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--radar',type=Path,required=True)
    args=parser.parse_args()
    rows, arms=predict(args.output,args.radar)
    score(args.output,rows,arms)
