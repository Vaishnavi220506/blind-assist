"""Read-only, evaluator-side attribution of the frozen MZ128 binary-arm errors.

No fitting or new deployable predictor. Native-range substitution and branch
removal are explicitly diagnostic controls, never proposed clearance rules.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from mz115_allocation_audit import audit_target, native_truth, native_bounds, possible, certain
from mz124_measurement_geometry import slant_box


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/mz123-frozen-early-20260913/returned-v1'
MZ125 = ROOT/'artifacts.local/work/mz125-observable-correction-20260913/correction-v1r1'
MZ128 = ROOT/'artifacts.local/work/mz128-zone-weighting-20260913/replay-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def tally(truth, flags):
    return dict(TP=sum(t and f for t,f in zip(truth,flags)),
                FP=sum(not t and f for t,f in zip(truth,flags)),
                FN=sum(t and not f for t,f in zip(truth,flags)),
                TN=sum(not t and not f for t,f in zip(truth,flags)))


def sealed_binary_readout(row, cached):
    """Independent scalar reconstruction; checked against every sealed output."""
    bins = sorted({tuple(z['theta_bounds_deg']) for z in row['tof_zones']})
    assert len(bins) == 8
    profile = (0.,0.,1.,1.,1.,1.,0.,0.)
    weights = {z['zone_id']:profile[bins.index(tuple(z['theta_bounds_deg']))] for z in row['tof_zones']}
    evidence = cached['spatial_evidence']
    active = {e['zone_id'] for e in evidence if possible(e['localized_xyz'])}
    contributions = {str(z):weights[z] for z in sorted(active)}
    coarse = any(certain(e['coarse_xyz']) for e in evidence)
    score = sum(contributions.values())
    flag = bool(cached['common_radar'] or cached['guard_events'] or coarse or score >= 1.)
    return dict(id=row['id'],candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN',score=score,
        active_zone_contributions=contributions,zone_weights={str(k):v for k,v in weights.items()},
        certain_coarse=coarse,common_radar=cached['common_radar'],guard_events=cached['guard_events'],
        integrated_yaw_deg=cached['integrated_yaw_deg'])


def run(output):
    assert not output.exists() and output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    output.mkdir(parents=True)
    cap, ana = SOURCE/'capture-v1', SOURCE/'analysis-v1'
    paths = [cap/n for n in ('raw.jsonl','evaluator.jsonl','spec.json','receipt.json')]
    paths += [ana/'frame-report.json', MZ125/'predictions.json', MZ125/'prediction-seal.json',
              MZ128/'predictions.json', MZ128/'prediction-seal.json', MZ128/'summary.json', MZ128/'completion.json']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    seal, receipt, correction_seal = read(MZ128/'prediction-seal.json'), read(cap/'receipt.json'), read(MZ125/'prediction-seal.json')
    assert sha(MZ128/'predictions.json') == seal['predictions_sha256']
    assert sha(MZ128/'summary.json') == read(MZ128/'completion.json')['summary_sha256']
    assert sha(MZ125/'predictions.json') == seal['mz125_predictions_sha256'] == correction_seal['predictions_sha256']
    assert sha(cap/'raw.jsonl') == seal['raw_sha256'] == correction_seal['raw_sha256'] == receipt['hashes']['raw.jsonl']
    assert sha(cap/'evaluator.jsonl') == receipt['hashes']['evaluator.jsonl']
    assert sha(cap/'spec.json') == receipt['spec_sha256']
    for name, digest in seal['dependencies'].items():
        assert sha(ROOT/name) == digest, name
    rows = [json.loads(s) for s in (cap/'raw.jsonl').read_text().splitlines()]
    ev = [json.loads(s) for s in (cap/'evaluator.jsonl').read_text().splitlines()]
    report, cached = read(ana/'frame-report.json'), read(MZ125/'predictions.json')
    predictions = read(MZ128/'predictions.json')['binary_four_mz125']
    design = {f['id']:f for f in read(cap/'spec.json')['frames']}
    assert len(rows)==len(ev)==len(report)==len(cached)==len(predictions)==288
    assert [r['id'] for r in rows]==[r['id'] for r in ev]==[r['id'] for r in report]==[r['id'] for r in predictions]
    records, targets, controls = [], [], {'frozen':[], 'drop_merged_diagnostic':[], 'native_merged_range_oracle':[]}
    truth = []
    for row, evaluator, label, cache, pred in zip(rows,ev,report,cached,predictions):
        current = sealed_binary_readout(row, cache)
        assert current == pred, row['id']
        gt = native_truth(evaluator); assert gt == label['truth']; truth.append(gt)
        branches = dict(TOF=pred['score']>=1, RADAR=bool(pred['common_radar']),
                        GUARD=bool(pred['guard_events']), CERTAIN=pred['certain_coarse'])
        assert pred['candidate'] == any(branches.values())
        other = branches['RADAR'] or branches['GUARD'] or branches['CERTAIN']
        valid_zones, merged_zones, oracle_zones = set(), set(), set()
        active = []
        for item in cache['spatial_evidence']:
            contributes = possible(item['localized_xyz']) and pred['zone_weights'][str(item['zone_id'])] > 0
            if not contributes:
                continue
            audit = audit_target(row,evaluator,cache,item)
            assert not audit['contract_violations']
            (valid_zones if item['status']=='SIM_VALID' else merged_zones).add(item['zone_id'])
            ranges = [c['native_range_m'] for c in audit['contributors']]
            assert ranges
            oracle_xyz = item['localized_xyz']
            if item['status']=='SIM_MERGED':
                pitch, yaw = row['camera_pitch_deg'], cache['integrated_yaw_deg']
                dy = .5+.2*row['time_s']
                oracle_xyz = slant_box(item['roi'], (min(ranges),max(ranges)), row['rgb_intrinsics'],
                    (pitch-.5,pitch+.5),(yaw-dy,yaw+dy),row['camera_in_body_m'][2])
            if possible(oracle_xyz):
                oracle_zones.add(item['zone_id'])
            contributors = audit['contributors']
            source_ids = sorted({c['actor_id'] for c in contributors if c['actor_id'] is not None})
            roles = {row['episode_id']+'/'+o['name']:o['source_role'] for o in design[row['id']]['objects']}
            private = next(z for z in evaluator['zonal_tof_native'] if z['zone_id']==item['zone_id'])
            lineage = next(x for x in private['returned_lineage'] if x['target_index']==item['target_slot'])
            sigma = row['tof_zones'][item['zone_id']]['targets'][item['target_slot']]['range_noise_sigma_m']
            range_error = item['range_m']-lineage['pre_noise_range_m']
            sides = Counter()
            for c in contributors:
                point=[x-b for x,b in zip(c['native_hit_point_m'],evaluator['body_origin_m'])]
                if abs(point[1]) > .3: sides['outside_lateral'] += 1
                if not .2 <= point[0] <= 3.6: sides['outside_forward_range'] += 1
                if not .4 <= point[2] <= 2.05: sides['outside_height'] += 1
            rec=dict(id=row['id'],zone_id=item['zone_id'],slot=item['target_slot'],status=item['status'],
                state=item['state'],range_m=item['range_m'],range_bounds=item['range_bounds'],signal=item['signal'],
                localized_xyz=item['localized_xyz'],native_range_span=[min(ranges),max(ranges)],
                native_range_oracle_possible=possible(oracle_xyz),native_range_oracle_xyz=oracle_xyz,
                source_actor_ids=source_ids,source_actor_count=len(source_ids),contributors=len(contributors),
                source_roles=[roles.get(i,'UNRESOLVED_ROLE') for i in source_ids],
                pre_noise_range_m=lineage['pre_noise_range_m'],range_error_m=range_error,range_sigma_m=sigma,
                exceeds_three_sigma=abs(range_error)>3*sigma,
                actual_corridor_hits=sum(c['native_hit_inside_corridor'] for c in contributors),
                hazardous_actor_hits=sum(c['native_actor_intersects_corridor'] for c in contributors),
                unknown_actor_hits=sum(c['actor_status']!='KNOWN_NATIVE_ACTOR' for c in contributors),
                geometry_exclusions=dict(sides),evidence_age_s=item['evidence_age_s'])
            active.append(rec)
            if not gt and pred['candidate']: targets.append(rec)
        controls['frozen'].append(pred['candidate'])
        controls['drop_merged_diagnostic'].append(bool(other or valid_zones))
        controls['native_merged_range_oracle'].append(bool(other or oracle_zones))
        if gt or not pred['candidate']:
            continue
        status='BOTH_STATUSES' if valid_zones and merged_zones else 'VALID_ONLY' if valid_zones else 'MERGED_ONLY' if merged_zones else 'NO_TOF_TRIGGER'
        bounds=[native_bounds(o,evaluator['body_origin_m']) for o in evaluator['native_bounds']]
        records.append(dict(id=row['id'],episode_id=row['episode_id'],time_s=row['time_s'],family=label['family'],
            branches=branches,status_support=status,positive_weight_tof_returns=len(active),
            valid_zones=sorted(valid_zones),merged_zones=sorted(merged_zones),
            drop_merged_still_alerts=controls['drop_merged_diagnostic'][-1],
            native_range_oracle_still_alerts=controls['native_merged_range_oracle'][-1],
            all_active_tof_from_known_single_actor=bool(active) and all(t['source_actor_count']==1 and not t['unknown_actor_hits'] for t in active),
            active_native_corridor_hits=sum(t['actual_corridor_hits'] for t in active),
            active_native_hazard_actor_hits=sum(t['hazardous_actor_hits'] for t in active),
            native_object_bounds=bounds,
            all_native_objects_lateral_out=all(y[1]<-.3 or y[0]>.3 for _,y,_ in bounds),
            active_tof_native_range_min=min((t['native_range_span'][0] for t in active),default=None),
            active_tof_native_range_max=max((t['native_range_span'][1] for t in active),default=None),
            tof_packet_received=row['tof_packet_received'],radar_packet_received=row['radar_packet_received']))
    assert len(records)==103 and tally(truth,controls['frozen'])==dict(TP=139,FP=103,FN=5,TN=41)
    control_results = {}
    for name,flags in controls.items():
        control_results[name]=dict(metrics=tally(truth,flags),
            removed_fp_ids=[r['id'] for r,t,a,b in zip(rows,truth,controls['frozen'],flags) if a and not b and not t],
            lost_tp_ids=[r['id'] for r,t,a,b in zip(rows,truth,controls['frozen'],flags) if a and not b and t],
            authority='DIAGNOSTIC_ONLY_NOT_A_RUNNABLE_CANDIDATE' if name!='frozen' else 'SEALED_REPRODUCTION')
    summary=dict(scope='CONSUMED_EVALUATOR_DIAGNOSIS_NO_ALGORITHM_CHANGE',frames=288,fp_frames=103,
        branch_partition=dict(Counter('+'.join(k for k,v in f['branches'].items() if v) for f in records)),
        family=dict(Counter(f['family'] for f in records)),status_support=dict(Counter(f['status_support'] for f in records)),
        positive_weight_tof_returns=len(targets),active_return_status=dict(Counter(t['status'] for t in targets)),
        active_return_state=dict(Counter(t['state'] for t in targets)),
        active_single_actor_returns=sum(t['source_actor_count']==1 and not t['unknown_actor_hits'] for t in targets),
        active_multi_actor_returns=sum(t['source_actor_count']>1 for t in targets),
        active_unknown_actor_returns=sum(t['unknown_actor_hits']>0 for t in targets),
        valid_return_error_max_m=max(abs(t['range_error_m']) for t in targets if t['status']=='SIM_VALID'),
        valid_return_three_sigma_exceedances=sum(t['exceeds_three_sigma'] for t in targets if t['status']=='SIM_VALID'),
        source_roles=dict(Counter('+'.join(sorted(set(t['source_roles']))) for t in targets)),
        actual_corridor_hits=sum(t['actual_corridor_hits'] for t in targets),
        hazardous_actor_hits=sum(t['hazardous_actor_hits'] for t in targets),
        native_objects_lateral_out_frames=sum(f['all_native_objects_lateral_out'] for f in records),
        stale_tof_support_returns=sum(t['evidence_age_s']!=0 for t in targets),
        controls=control_results,code_sha256=sha(Path(__file__)),inputs=hashes,
        backend='CPU: TASK_NOT_GPU_SUITABLE',
        topology_scope='MZ128 has no ToF angular connected-component/centroid alert stage. ToF association already groups valid forward depths within .25m; per-return interval intersection drives warnings. The later .30m direction graph has no measured integration benefit here.',
        limits='Do not infer absence of a hazard from missing samples; FP truth is cross-checked against native object bounds. Native-range oracle and branch deletion are causal diagnostics with preserved FN counts, never deployable rules. Actor labels identify evaluator provenance only.')
    for key,data in [('frames',records),('support-returns',targets),('summary',summary)]:write(output/(key+'.json'),data)
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(output/'completion.json',dict(status='PASS',summary_sha256=sha(output/'summary.json'),frames_sha256=sha(output/'frames.json'),
        supports_sha256=sha(output/'support-returns.json'),inputs_unchanged=True,resources='No persistent allocation or process'))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('inputs','controls')},indent=2))
    print(json.dumps({k:v['metrics'] for k,v in control_results.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
