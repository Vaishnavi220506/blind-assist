"""Fixed ToF-only cohort envelope; center span is a hypothesis, not a bound."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from mz129_extent_correction import ROOT, SOURCE, CORRECTION, events
from mz126_central_tof import read, write, sha, serial, tof
from mz128_zone_weighting import column_weights, FOUR, THRESHOLD, readout
from mz124_measurement_geometry import slant_box, metrics
from tof_depth_connectivity import neighbors, depth_edge
from research_backend import BackendCandidate, DeviceObservation, select_backend

MZ129 = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'
BRIEF = Path(__file__).with_name('MZ131_NATIVE_ANGULAR_BRIEF_20260914.md')
GAP = .30


def native_cohorts(packet):
    """Deliberately receives ToF only, without image, proposals or pose."""
    eligible = {}; reasons = {}
    if not packet['tof_packet_received']:
        return {}, [], {}
    for zone in packet['tof_zones']:
        z = zone['zone_id']; targets = zone['targets']
        if len(targets) != 1:
            reasons[z] = 'MULTI_TARGET_OR_EMPTY'; continue
        target = targets[0]; r = target['distance_m']; sigma = target['range_noise_sigma_m']
        if target['status'] != 'SIM_VALID':
            reasons[z] = 'NON_VALID_OR_MERGED'; continue
        if not math.isfinite(r) or not math.isfinite(sigma) or r <= 0 or sigma < 0:
            reasons[z] = 'INVALID_RANGE'; continue
        eligible[z] = zone
    remaining = set(eligible); groups = []; chosen = {}
    while remaining:
        first = min(remaining); remaining.remove(first); group = [first]; stack = [first]
        while stack:
            i = stack.pop()
            linked = [j for j in sorted(remaining) if neighbors(i, j) and depth_edge(
                eligible[i]['targets'][0]['distance_m'], eligible[j]['targets'][0]['distance_m'], GAP, 0.)]
            for j in linked:
                remaining.remove(j); group.append(j); stack.append(j)
        group.sort(); zones = [eligible[z] for z in group]
        depths = [z['targets'][0]['distance_m'] for z in zones]
        reason = 'SINGLE_ZONE' if len(group) < 2 else 'GLOBAL_DEPTH_CONFLICT' if max(depths)-min(depths) >= GAP else 'NATIVE_CENTER_SPAN_HYPOTHESIS'
        rec = dict(zone_ids=group, range_span_m=[min(depths),max(depths)], reason=reason, accepted=False)
        groups.append(rec)
        for z in group: reasons[z] = reason
        if reason != 'NATIVE_CENTER_SPAN_HYPOTHESIS': continue
        angular = {}; centroids = {}; full = {}
        for axis in ('theta_bounds_deg','phi_bounds_deg'):
            centers = [sum(z[axis])/2 for z in zones]; center = sum(centers)/len(centers)
            extent = [min(z[axis][0] for z in zones),max(z[axis][1] for z in zones)]
            radius = max(abs(v-center) for v in centers)
            angular[axis] = [center-radius,center+radius] if radius > 1e-12 else extent
            centroids[axis] = center; full[axis] = extent
        rec.update(accepted=True, angular=angular, centroid_deg=centroids, native_full_span=full)
        for z in group: chosen[z] = dict(cohort=len(groups)-1, **angular)
    return chosen, groups, reasons


def predict_frame(row, cached, old, radar):
    chosen, groups, reasons = native_cohorts({k:row[k] for k in ('tof_packet_received','tof_zones')})
    weights = column_weights(row, FOUR); records = []; yaw = cached['integrated_yaw_deg']
    for e in cached['spatial_evidence']:
        z = e['zone_id']; before = tof.possible(e['localized_xyz'])
        rec = dict(zone_id=z, slot=e['target_slot'], status=e['status'], weight=weights[z],
            range_bounds=e['range_bounds'], old_roi=e['roi'], tiles=[e['roi']],
            old_possible=before, possible=before, narrowed=False, cohort=None,
            reason=reasons.get(z,'PACKET_OR_ZONE_UNAVAILABLE'))
        if z in chosen:
            angular = chosen[z]; box = tof.zone_box(angular, row['rgb_intrinsics'])
            tile = tof.intersection(e['roi'], box)
            rec['cohort'] = angular['cohort']; rec['native_envelope_px'] = box
            if tile is None: rec['reason'] = 'EMPTY_INTERSECTION_CONFLICT_FALLBACK'
            elif any(abs(a-b)>1e-9 for a,b in zip(tile,e['roi'])):
                pitch = row['camera_pitch_deg']; dy = .5+.2*row['time_s']
                xyz = slant_box(tile,e['range_bounds'],row['rgb_intrinsics'],
                    (pitch-.5,pitch+.5),(yaw-dy,yaw+dy),row['camera_in_body_m'][2])
                rec.update(tiles=[tile],narrowed=True,possible=tof.possible(xyz),xyz=serial(xyz))
                assert not rec['possible'] or before
        if e['status']=='SIM_MERGED':
            assert not rec['narrowed'] and rec['range_bounds']==[.02,4.]
        records.append(rec)
    active = {e['zone_id'] for e in records if e['possible']}
    score = sum(weights[z] for z in active)
    certain = any(tof.certain(e['coarse_xyz']) for e in cached['spatial_evidence'])
    assert certain == old['certain_coarse']
    flag = bool(score>=THRESHOLD or certain or radar['candidate'])
    assert not flag or old['candidate']
    pred = dict(old,candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN',score=score,
        active_zone_contributions={str(z):weights[z] for z in sorted(active)})
    return pred, dict(id=row['id'],returns=records,cohorts=groups)


def run(output):
    assert not output.exists() and output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    raw = SOURCE/'capture-v1/raw.jsonl'; cp = CORRECTION/'predictions.json'
    bp = MZ129/'replay-v1r2/predictions.json'; rp = MZ129/'radar-v1/predictions.json'
    for directory in (MZ129/'replay-v1r2',MZ129/'radar-v1'):
        seal = read(directory/'prediction-seal.json')
        assert sha(directory/'predictions.json') == seal['predictions_sha256']
        for key in ('input_hashes','dependencies','source_hashes'):
            for path,digest in seal.get(key,{}).items(): assert sha(ROOT/path)==digest, path
    cs = read(CORRECTION/'prediction-seal.json')
    assert sha(cp)==cs['predictions_sha256'] and sha(raw)==cs['raw_sha256']
    rows = [json.loads(s) for s in raw.read_text().splitlines()]
    cached = read(cp); baseline = read(bp)['radar']; radar = read(rp)
    assert len(rows)==len(cached)==len(baseline)==len(radar)==288
    assert [r['id'] for r in rows]==[p['id'] for p in baseline]==[p['id'] for p in radar]
    for r,c,b,rb in zip(rows,cached,baseline,radar):
        assert serial(tof.allocate(r,c))==c['spatial_evidence']
        rd = readout(c,column_weights(r,FOUR))
        assert rd['score']==b['score'] and rd['certain_coarse']==b['certain_coarse']
        assert bool(rd['score']>=THRESHOLD or rd['certain_coarse'] or rb['candidate'])==b['candidate']
    output.mkdir(parents=True)
    select_backend('scalar-scoring',cpu=BackendCandidate('native-angular-cohorts','cpu',
        lambda: neighbors(0,1),lambda _: DeviceObservation('cpu','host CPU','Python / NumPy')),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=output/'backend.json')
    started=time.perf_counter(); arms=dict(baseline=baseline,tof_native=[]); details=[]
    for row,c,old,rb in zip(rows,cached,baseline,radar):
        pred,detail=predict_frame(row,c,old,rb); arms['tof_native'].append(pred); details.append(detail)
    seconds=time.perf_counter()-started
    write(output/'predictions.json',arms); write(output/'angular-support.json',details)
    paths=[raw,cp,bp,rp,CORRECTION/'prediction-seal.json',MZ129/'replay-v1r2/prediction-seal.json',MZ129/'radar-v1/prediction-seal.json']
    deps=[Path(__file__),BRIEF]+[Path(__file__).with_name(n) for n in (
        'tof_depth_connectivity.py','tof_directional_readout.py','mz115_spatial_allocation.py',
        'mz124_measurement_geometry.py','mz128_zone_weighting.py','mz129_extent_correction.py','mz126_central_tof.py')]
    write(output/'prediction-seal.json',dict(predictions_sha256=sha(output/'predictions.json'),
        angular_support_sha256=sha(output/'angular-support.json'),seconds=seconds,
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps},
        authority='CONSUMED_OBSERVABLE_OUTPUTS_SAVED_BEFORE_SCORING'))
    score_run(output,rows,arms,details)


def score_run(output,rows,arms,details):
    from mz115_allocation_audit import native_truth, project_point, contains, possible, native_bounds
    report=read(SOURCE/'analysis-v1/frame-report.json')
    evaluator=[json.loads(s) for s in (SOURCE/'capture-v1/evaluator.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==[e['id'] for e in evaluator]==[r['id'] for r in report]
    truth=[native_truth(e) for e in evaluator]; assert truth==[r['truth'] for r in report]
    native=[]
    for row,ev,detail in zip(rows,evaluator,details):
        actors={row['episode_id']+'/'+o['name']:o for o in ev['native_bounds']}
        for record in detail['returns']:
            if not record['narrowed']: continue
            zone=next(z for z in ev['zonal_tof_native'] if z['zone_id']==record['zone_id'])
            lineage=next((l for l in zone['returned_lineage'] if l['target_index']==record['slot']),None)
            key=dict(id=row['id'],zone_id=record['zone_id'],slot=record['slot'])
            if lineage is None or not lineage['hit_indices']:
                native.append(dict(**key,evaluable=False,reason='MISSING_LINEAGE')); continue
            for k in lineage['hit_indices']:
                hit=zone['private_rays'][k]; point=hit['hit_point_m']
                pixel=project_point(point,ev['camera'],row['rgb_intrinsics'])
                if pixel is None:
                    native.append(dict(**key,sample=k,evaluable=False,reason='PROJECTION_UNKNOWN'));continue
                body=[p-o for p,o in zip(point,ev['body_origin_m'])]; actor=actors.get(hit.get('actor_id'))
                native.append(dict(**key,sample=k,evaluable=True,point=point,pixel=pixel,
                    previously_contained=contains(record['old_roi'],pixel),
                    locally_contained=any(contains(t,pixel) for t in record['tiles']),
                    corridor_point=possible([(x,x) for x in body]),
                    hazard_actor=bool(actor and possible(native_bounds(actor,ev['body_origin_m'])))))
    write(output/'native-retention.json',native)
    dropped=[n for n in native if n['evaluable'] and n['previously_contained'] and not n['locally_contained']]
    native_stats=dict(audited_samples=len(native),not_evaluable=sum(not n['evaluable'] for n in native),
        newly_dropped_samples=len(dropped),newly_dropped_corridor_samples=sum(n['corridor_point'] for n in dropped),
        newly_dropped_hazard_actor_samples=sum(n['hazard_actor'] for n in dropped))
    before=[p['candidate'] for p in arms['baseline']]; oldevents=events(rows,truth,before)
    impact=[]
    for r,label,t,b,a,detail in zip(rows,report,truth,arms['baseline'],arms['tof_native'],details):
        if t or not b['candidate']:continue
        narrowed=[e for e in detail['returns'] if e['narrowed']]
        triggering=[e for e in detail['returns'] if e['old_possible'] and e['weight']>0]
        changed=[e for e in triggering if e['narrowed']]
        impact.append(dict(id=r['id'],family=label['family'],narrowed_returns=len(narrowed),
            positive_weight_possible_returns=len(triggering),narrowed_triggering_returns=len(changed),
            removed_triggering_possible_bits=sum(not e['possible'] for e in changed),
            old_score=b['score'],new_score=a['score'],score_changed=b['score']!=a['score'],
            alert_removed=not a['candidate'],protected_radar=bool(b['common_radar'] or b['guard_events']),
            certain_coarse=b['certain_coarse']))
    write(output/'fp-trigger-impact.json',impact)
    result=dict(frames=len(rows),arms={},native_retention=native_stats,
        narrowed_returns=sum(e['narrowed'] for d in details for e in d['returns']),
        cohort_reasons=dict(Counter(g['reason'] for d in details for g in d['cohorts'])),
        return_reasons=dict(Counter(e['reason'] for d in details for e in d['returns'])))
    for name,predictions in arms.items():
        flags=[p['candidate'] for p in predictions]; m=metrics(rows,truth,flags); m['events']=events(rows,truth,flags)
        m['lost_TP_ids']=[r['id'] for r,t,b,a in zip(rows,truth,before,flags) if t and b and not a]
        m['event_times_identical']=m['events']==oldevents
        m['changed']=[dict(id=r['id'],family=l['family'],truth=t,before=b,after=a) for r,l,t,b,a in zip(rows,report,truth,before,flags) if b!=a]
        m['families']={}
        for family in sorted({r['family'] for r in report}):
            ix=[i for i,r in enumerate(report) if r['family']==family]
            m['families'][family]=metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
        result['arms'][name]=m
    def summarize(items):
        return dict(frames=len(items),any_narrowed=sum(e['narrowed_returns']>0 for e in items),
            triggering_narrowed=sum(e['narrowed_triggering_returns']>0 for e in items),
            triggering_possible_changed=sum(e['removed_triggering_possible_bits']>0 for e in items),
            score_changed=sum(e['score_changed'] for e in items),alert_removed=sum(e['alert_removed'] for e in items),
            triggering_return_count=sum(e['positive_weight_possible_returns'] for e in items),
            narrowed_triggering_return_count=sum(e['narrowed_triggering_returns'] for e in items),
            removed_triggering_possible_bits=sum(e['removed_triggering_possible_bits'] for e in items))
    result['fp_impact']=summarize(impact)
    result['fp_impact_families']={f:summarize([r for r in impact if r['family']==f]) for f in sorted({r['family'] for r in impact})}
    candidate=result['arms']['tof_native']
    result['retention_criterion']=bool(candidate['FP']<93 and not candidate['lost_TP_ids'] and candidate['event_times_identical']
        and not native_stats['newly_dropped_corridor_samples'] and not native_stats['not_evaluable'])
    write(output/'summary.json',result)
    seal=read(output/'prediction-seal.json')
    assert sha(output/'predictions.json')==seal['predictions_sha256']
    assert sha(output/'angular-support.json')==seal['angular_support_sha256']
    for k in ('input_hashes','dependencies'):
        for path,digest in seal[k].items():assert sha(ROOT/path)==digest,path
    write(output/'completion.json',dict(status='PASS',summary_sha256=sha(output/'summary.json'),
        evaluator_sha256=sha(SOURCE/'capture-v1/evaluator.jsonl'),label_sha256=sha(SOURCE/'analysis-v1/frame-report.json'),
        resources='No persistent process or allocation; source inputs unchanged'))
    print(json.dumps(dict(arms={n:{k:m[k] for k in ('TP','FP','FN','false_segments','max_detected_delay_s')} for n,m in result['arms'].items()},
        native_retention=native_stats,fp_impact=result['fp_impact'],retention_criterion=result['retention_criterion']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
