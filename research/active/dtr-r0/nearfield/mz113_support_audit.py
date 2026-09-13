"""Evaluator-only MZ113 support audit, including explicit unknown-Doppler hold.

Consumes only sealed predictions. Never imported by an observable predictor.
Truth determines scoring and correspondence diagnostics, never a candidate.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from run_mz107_four_sensor import ROOT, readrows, sha, truth, write
from mz111_support_audit import owners, observed_return, identity_status, audit_spatial

COMPARISONS = {
    'flow_vs_nominal': ('flow', 'nominal'),
    'angular_vs_nominal': ('angular_unknown_velocity', 'nominal'),
    'flow_vs_incumbent': ('flow', 'combined'),
    'angular_vs_incumbent': ('angular_unknown_velocity', 'combined'),
    'combined_flow_vs_incumbent': ('combined_flow', 'combined'),
    'combined_angular_vs_incumbent': ('combined_angular', 'combined'),
    'flow_vs_angular': ('flow', 'angular_unknown_velocity'),
    'combined_flow_vs_combined_angular': ('combined_flow', 'combined_angular'),
}


def audit_transport(index, trace, rows, es, prov, object_cache):
    source_index = trace['measurement_index']; slot = trace['measurement_slot']; failures = []
    if not isinstance(source_index, int) or not 0 <= source_index < index:
        return dict(status='MOTION_AGE_FAILURE', failures=['noncausal_source_index'], trace=trace)
    old = rows[source_index]; row = rows[index]; age = row['time_s']-old['time_s']
    if old['episode_id'] != row['episode_id'] or not 0 < age <= .5 or not math.isclose(age, trace['age_s'], abs_tol=1e-8):
        failures.append('episode_or_age')
    segment = rows[source_index:index+1]
    if not all(r['imu_valid'] and r['episode_id'] == row['episode_id'] for r in segment):
        failures.append('imu_or_episode_gap')
    if not all(a['time_s'] < b['time_s'] for a, b in zip(segment, segment[1:])):
        failures.append('nonmonotonic_time')
    observed = observed_return(old, slot)
    if observed is None:
        failures.append('invalid_source_range')
    else:
        velocity = observed['velocity_mps']; finite_velocity = velocity is not None and math.isfinite(velocity)
        if finite_velocity:
            expected = observed['range_m']+velocity*age
            if trace.get('velocity_mps') is None or not math.isclose(velocity, trace['velocity_mps'], abs_tol=1e-8):
                failures.append('measured_velocity_mismatch')
            if trace.get('velocity_state', 'MEASURED') != 'MEASURED':
                failures.append('finite_doppler_mislabeled_unknown')
        else:
            expected = observed['range_m']
            if trace.get('velocity_state') != 'UNKNOWN' or trace.get('velocity_mps') is not None or trace.get('range_assumption') != 'HOLD_RANGE_VELOCITY_UNKNOWN':
                failures.append('missing_doppler_without_explicit_hold_assumption')
        if expected <= 0 or not math.isclose(expected, trace['range_m'], abs_tol=1e-8):
            failures.append('transported_range_mismatch')
    visual = trace.get('visual', {})
    if visual.get('state') == 'FLOW_MATCHABLE':
        error = visual.get('max_fb_error_px'); overlap = visual.get('match_iou')
        if visual.get('valid_points', 0) < 3 or error is None or not math.isfinite(error) or error > 1.5 or overlap is None or overlap < .2:
            failures.append('flow_consistency_contract')
    slots = prov[source_index]['radar_slots']
    source = slots[slot] if isinstance(slot, int) and 0 <= slot < len(slots) else None
    previous_objects, _ = object_cache[source_index]; current_objects, matches = object_cache[index]
    box_owners = matches.get(trace['proposal'], [])
    status = 'MOTION_AGE_FAILURE' if failures else identity_status(source, previous_objects, current_objects, box_owners)
    actor = source.get('actor_id') if source else None
    return dict(status=status, failures=failures, trace=trace, source=source,
        source_frame_id=old['id'], source_index=source_index, source_episode=old['episode_id'], elapsed_s=age,
        source_measurement=observed, current_box_owners=box_owners,
        actor_currently_hazard=bool(actor in current_objects and truth(dict(es[index], native_bounds=[current_objects[actor]]))))


def supporting_evidence(arm, index, rows, values, es, prov, object_cache):
    nominal = values['nominal']; n = nominal[index]; result = []
    temporal_arm = {'combined': 'temporal', 'combined_flow': 'flow', 'combined_angular': 'angular_unknown_velocity'}.get(arm)
    if temporal_arm:
        spatial = values['filtered_plane'][index]
        if spatial['candidate']:
            for ret in spatial['spatial_evidence']:
                if ret['support']:
                    entry = audit_spatial(index, ret, rows, nominal, es, prov, object_cache)
                    result.append(dict(entry, component='filtered_plane'))
            if n['tof_support']:
                result.append(dict(status='INDEPENDENT_TOF', component='filtered_plane'))
        if not n['candidate'] and values[temporal_arm][index]['candidate']:
            result.extend(supporting_evidence(temporal_arm, index, rows, values, es, prov, object_cache))
        assert bool(spatial['candidate'] or (values[temporal_arm][index]['candidate'] and not n['candidate'])) == bool(values[arm][index]['candidate'])
    elif arm in ('flow', 'angular_unknown_velocity', 'temporal'):
        prediction = values[arm][index]
        if n['candidate']:
            result.extend(supporting_evidence('nominal', index, rows, values, es, prov, object_cache))
        if prediction['diagnostics']['added_support']:
            for trace in prediction['diagnostics']['propagated']:
                if trace['refined_support']:
                    result.append(dict(audit_transport(index, trace, rows, es, prov, object_cache), component=arm))
    elif arm == 'nominal':
        if n['tof_support']:
            result.append(dict(status='INDEPENDENT_TOF', component='nominal'))
        for association in n['associations']:
            if association['refined_support']:
                if association.get('proposal') is None:
                    source = prov[index]['radar_slots'][association['slot']]
                    result.append(dict(status='INDEPENDENT_RADAR_NO_RGB_CORRESPONDENCE', source=source, component='nominal'))
                else:
                    entry = audit_spatial(index, association, rows, nominal, es, prov, object_cache)
                    result.append(dict(entry, component='nominal'))
    else:
        raise ValueError('Unsupported audit arm: '+arm)
    return result


def summarize(records):
    counts = Counter(r['change'] for r in records)
    by_change = {}
    for change in sorted(counts):
        subset = [r for r in records if r['change'] == change]
        by_change[change] = dict(frames=len(subset), statuses_nonexclusive=dict(Counter(
            s for r in subset for s in {e['status'] for e in r['alerting_evidence']})))
    sources = [(r['panel'], e['source_index'], e['trace']['measurement_slot']) for r in records
               for e in r['alerting_evidence'] if 'source_index' in e]
    return dict(changed_frames=len(records), changes=dict(counts), by_change=by_change,
        evidence_status=dict(Counter(e['status'] for r in records for e in r['alerting_evidence'])),
        no_attributed_evidence=sum(not r['alerting_evidence'] for r in records),
        temporal_frame_supports=len(sources), distinct_temporal_measurements=len(set(sources)))


def audit_panel(panel, rows, values, es, prov):
    object_cache = [owners(row, nominal, e) for row, nominal, e in zip(rows, values['nominal'], es)]
    result = {}
    for name, (candidate_arm, reference_arm) in COMPARISONS.items():
        records = []
        for index, row in enumerate(rows):
            candidate = bool(values[candidate_arm][index]['candidate']); reference = bool(values[reference_arm][index]['candidate'])
            if candidate == reference:
                continue
            gt = truth(es[index]); alerting_arm = candidate_arm if candidate else reference_arm
            evidence = supporting_evidence(alerting_arm, index, rows, values, es, prov, object_cache)
            change = ('GAINED_TP' if gt else 'ADDED_FP') if candidate else ('LOST_TP' if gt else 'REMOVED_FP')
            records.append(dict(panel=panel, frame_index=index, frame_id=row['id'], episode_id=row['episode_id'], time_s=row['time_s'],
                truth=gt, candidate_arm=candidate_arm, reference_arm=reference_arm, candidate=candidate, reference=reference,
                change=change, alerting_arm=alerting_arm, alerting_evidence=evidence,
                candidate_diagnostics=values[candidate_arm][index].get('diagnostics'),
                reference_diagnostics=values[reference_arm][index].get('diagnostics')))
        result[name] = records
    return result


def run(analysis, output):
    analysis = analysis.resolve(); output = output.resolve()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    seal_path = analysis/'prediction-seal.json'; seal = json.loads(seal_path.read_text())
    assert seal['status'] == 'ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE'
    for name, digest in seal['code_sha256'].items():
        assert sha(analysis/name) == digest, name
    packets = {}; seen = set()
    for panel, item in seal['panels'].items():
        capture = Path(item['capture']).resolve(); prediction = analysis/panel/'predictions.json'
        assert sha(capture/'receipt.json') == item['receipt_sha256']
        assert sha(capture/'raw.jsonl') == item['raw_sha256']
        assert sha(prediction) == item['predictions_sha256']
        assert item['raw_sha256'] not in seen, 'Duplicate raw source panel'
        seen.add(item['raw_sha256'])
        receipt = json.loads((capture/'receipt.json').read_text()); assert receipt['status'] == 'PASS'
        for name, digest in receipt['hashes'].items():
            assert sha(capture/name) == digest, name
        rows = readrows(capture/'raw.jsonl'); values = json.loads(prediction.read_text())
        assert len(rows) == receipt['frames'] and len({r['id'] for r in rows}) == len(rows)
        assert all(len(preds) == len(rows) for preds in values.values())
        assert all('id' not in n or n['id'] == r['id'] for r, n in zip(rows, values['nominal']))
        packets[panel] = (capture, rows, values)
    output.mkdir(parents=True); shutil.copyfile(Path(__file__), output/Path(__file__).name)
    summaries = {}; overall = {}
    for panel, (capture, rows, values) in packets.items():
        es = readrows(capture/'evaluator.jsonl'); prov = readrows(capture/'provenance.jsonl')
        assert [r['id'] for r in rows] == [e['id'] for e in es] == [p['id'] for p in prov]
        records = audit_panel(panel, rows, values, es, prov); directory = output/panel; directory.mkdir(); summaries[panel] = {}
        for name, items in records.items():
            write(directory/(name+'.json'), items); summaries[panel][name] = summarize(items)
            overall.setdefault(name, []).extend(items)
    summary = dict(status='EVALUATOR_ONLY_SUPPORT_AUDIT_COMPLETE', unique_source_frames=sum(len(rows) for _, rows, _ in packets.values()),
        panels=summaries, overall={name: summarize(items) for name, items in overall.items()},
        scope='POSTHOC_SEALED_PREDICTION_AUDIT', limits=[
            'Unique projected AABB IoU >= 0.5 supports correspondence, not exact pixel visibility or identity.',
            'Missing Doppler hold-range is a declared assumption, not measured zero velocity.',
            'The alerting side is audited for both gained and lost decisions; a removed FP may be a wrong prior association.',
            'Comparisons overlap; distinct source measurements and current frames are separate denominators.',
            'Real actor origin is hypothetical Radar provenance, not RF/sensor ground truth.'])
    write(output/'summary.json', summary)
    write(output/'completion.json', dict(status='PASS', analysis_seal_sha256=sha(seal_path),
        dependency_hashes={name: sha(Path(__file__).with_name(name)) for name in ('mz111_support_audit.py', 'run_mz107_four_sensor.py', 'run_mz108_competitive_association.py')},
        output_hashes={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    print(json.dumps(summary['overall'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--analysis', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.analysis, args.output)
