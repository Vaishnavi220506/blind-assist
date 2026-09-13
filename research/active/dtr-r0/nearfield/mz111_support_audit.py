"""Evaluator-only audit of frozen MZ111-format decisions; never prediction input.

Projected AABB correspondence is conditional support, not pixel identity truth.
Each changed frame retains every contributing/suppressed trace; frame category
counts can overlap and must not be summed as unique frames or sensor identities.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from run_mz107_four_sensor import ROOT, readrows, sha, truth, write
from run_mz108_competitive_association import projected_objects, iou

MAX_AGE_S = .5
IOU_THRESHOLD = .5


def owners(row, nominal, evaluation):
    objects = {row['episode_id']+'/'+o['name']: o for o in evaluation['native_bounds']}
    result = {}
    for j, box in enumerate(nominal['proposals']):
        result[j] = []
        for actor_id, obj in objects.items():
            projected = projected_objects(row, dict(evaluation, native_bounds=[obj]))
            if projected:
                overlap = iou(box, projected[0])
                if overlap >= IOU_THRESHOLD:
                    result[j].append(dict(actor_id=actor_id, iou=overlap))
    return objects, result


def observed_return(row, slot, need_velocity=False):
    if not isinstance(slot, int) or not 0 <= slot < len(row['radar_valid']):
        return None
    r = row['radar_range_m'][slot]; a = row['radar_angle'][slot]
    if not row['radar_packet_received'] or not row['radar_valid'][slot] or r is None or a is None or not math.isfinite(r+a) or r <= 0:
        return None
    v = row['radar_velocity'][slot]
    if need_velocity and (v is None or not math.isfinite(v)):
        return None
    return dict(range_m=r, velocity_mps=v, angle_deg=a)


def identity_status(source, old_objects, current_objects, box_owners):
    name = source.get('actor_id') if source else None
    if not source or source.get('kind') != 'real_actor' or name not in old_objects:
        return 'GHOST_OR_UNMATCHED_SOURCE'
    if name not in current_objects or [o['actor_id'] for o in box_owners] != [name]:
        return 'CURRENT_BBOX_UNSUPPORTED'
    return 'CORRECT_SUPPORTED_PROXY'


def audit_temporal(index, trace, rows, nominal, evaluations, provenance, object_cache):
    source_index = trace['measurement_index']; slot = trace['measurement_slot']
    row = rows[index]; failures = []
    if not isinstance(source_index, int) or not 0 <= source_index < index:
        return dict(mechanism='temporal', status='MOTION_AGE_FAILURE', failures=['noncausal_measurement_index'], trace=trace)
    old = rows[source_index]; age = row['time_s']-old['time_s']
    if old['episode_id'] != row['episode_id'] or not 0 < age <= MAX_AGE_S:
        failures.append('episode_or_age')
    if not math.isclose(age, trace['age_s'], abs_tol=1e-8):
        failures.append('reported_age_mismatch')
    segment = rows[source_index:index+1]
    if not all(r['imu_valid'] and r['episode_id'] == row['episode_id'] for r in segment):
        failures.append('imu_or_episode_gap')
    if not all(a['time_s'] < b['time_s'] for a, b in zip(segment, segment[1:])):
        failures.append('nonmonotonic_time')
    observed = observed_return(old, slot, need_velocity=True)
    if observed is None:
        failures.append('invalid_source_measurement')
    else:
        predicted = observed['range_m']+observed['velocity_mps']*age
        if predicted <= 0 or not math.isclose(predicted, trace['range_m'], abs_tol=1e-8):
            failures.append('transported_range_mismatch')
        if not math.isclose(observed['velocity_mps'], trace['velocity_mps'], abs_tol=1e-8):
            failures.append('source_velocity_mismatch')
    slots = provenance[source_index]['radar_slots']
    source = slots[slot] if isinstance(slot, int) and 0 <= slot < len(slots) else None
    old_objects, _ = object_cache[source_index]; current_objects, matches = object_cache[index]
    box_owners = matches.get(trace['proposal'], [])
    status = 'MOTION_AGE_FAILURE' if failures else identity_status(source, old_objects, current_objects, box_owners)
    name = source.get('actor_id') if source else None
    return dict(mechanism='temporal', status=status, failures=failures, trace=trace,
        source_index=source_index, source_frame_id=old['id'], source_episode=old['episode_id'], source=source,
        observed_measurement=observed, elapsed_s=age, current_box_owners=box_owners,
        actor_currently_hazard=bool(name in current_objects and truth(dict(evaluations[index], native_bounds=[current_objects[name]]))))


def audit_spatial(index, evidence, rows, nominal, evaluations, provenance, object_cache):
    slot = evidence['slot']; row = rows[index]
    source = provenance[index]['radar_slots'][slot]
    objects, matches = object_cache[index]
    box_owners = matches.get(evidence['proposal'], [])
    valid = observed_return(row, slot) is not None
    status = identity_status(source, objects, objects, box_owners) if valid else 'GHOST_OR_UNMATCHED_SOURCE'
    name = source.get('actor_id') if source else None
    return dict(mechanism='spatial', status=status, trace=evidence, source=source,
        valid_observed_return=valid, current_box_owners=box_owners,
        actor_currently_hazard=bool(name in objects and truth(dict(evaluations[index], native_bounds=[objects[name]]))))


def summarize(records):
    result = dict(changed_frames=len(records), decisions=dict(Counter(r['change'] for r in records)),
        evidence_status=dict(Counter(e['status'] for r in records for e in r['evidence'])),
        frame_status_nonexclusive=dict(Counter(status for r in records for status in {e['status'] for e in r['evidence']})),
        all_supports_correct_proxy=sum(bool(r['evidence']) and all(e['status']=='CORRECT_SUPPORTED_PROXY' for e in r['evidence']) for r in records),
        no_attributed_evidence=sum(not r['evidence'] for r in records), by_change={})
    for change in sorted({r['change'] for r in records}):
        subset = [r for r in records if r['change'] == change]
        result['by_change'][change] = dict(frames=len(subset),
            frame_status_nonexclusive=dict(Counter(status for r in subset for status in {e['status'] for e in r['evidence']})))
    return result


def audit_panel(panel, rows, values, es, prov):
    nominal = values['nominal']
    object_cache = [owners(r, n, e) for r, n, e in zip(rows, nominal, es)]
    result = {}
    arms = [arm for arm, preds in values.items() if arm in ('temporal', 'combined') or any('spatial_evidence' in p for p in preds)]
    for arm in arms:
        records = []
        for index, (row, n, prediction) in enumerate(zip(rows, nominal, values[arm])):
            if bool(n['candidate']) == bool(prediction['candidate']):
                continue
            added = bool(prediction['candidate']); gt = truth(es[index]); evidence = []
            if arm in ('temporal', 'combined'):
                t = values['temporal'][index]
                if added and t['diagnostics']['added_support']:
                    for trace in t['diagnostics']['propagated']:
                        if trace['refined_support']:
                            evidence.append(audit_temporal(index, trace, rows, nominal, es, prov, object_cache))
            spatial = values['filtered_plane'][index] if arm == 'combined' else prediction
            if 'spatial_evidence' in spatial:
                old_support = {a['slot']: bool(a['refined_support']) for a in n['associations']}
                for ret in spatial['spatial_evidence']:
                    relevant = bool(ret['support']) if added else old_support.get(ret['slot'], False) and not ret['support']
                    if relevant:
                        entry = audit_spatial(index, ret, rows, nominal, es, prov, object_cache)
                        entry['nominal_slot_support'] = old_support.get(ret['slot'], False)
                        evidence.append(entry)
            if arm == 'combined':
                expected = bool(values['filtered_plane'][index]['candidate'] or
                    (values['temporal'][index]['candidate'] and not n['candidate']))
                assert expected == added, 'Combined decision does not match frozen component outputs'
            change = ('GAINED_TP' if gt else 'ADDED_FP') if added else ('LOST_TP' if gt else 'REMOVED_FP')
            records.append(dict(panel=panel, frame_index=index, frame_id=row['id'], episode_id=row['episode_id'],
                time_s=row['time_s'], truth=gt, nominal=bool(n['candidate']), candidate=added,
                change=change, evidence=evidence))
        result[arm] = records
    return result


def run(analysis, output):
    analysis = analysis.resolve(); output = output.resolve()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    seal_path = analysis/'prediction-seal.json'; seal = json.loads(seal_path.read_text())
    assert seal['status'] == 'ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE'
    for name, digest in seal['code_sha256'].items():
        assert sha(analysis/name) == digest, name
    packets = {}; seen_sources = set()
    # Authenticate every panel and prediction before parsing any evaluator data.
    for panel, item in seal['panels'].items():
        capture = Path(item['capture']).resolve(); prediction_path = analysis/panel/'predictions.json'
        assert sha(capture/'receipt.json') == item['receipt_sha256']
        assert sha(capture/'raw.jsonl') == item['raw_sha256']
        assert sha(prediction_path) == item['predictions_sha256']
        receipt = json.loads((capture/'receipt.json').read_text()); assert receipt['status'] == 'PASS'
        for name, digest in receipt['hashes'].items():
            assert sha(capture/name) == digest, name
        rows = readrows(capture/'raw.jsonl'); values = json.loads(prediction_path.read_text())
        assert len(rows) == receipt['frames'] and len({r['id'] for r in rows}) == len(rows)
        assert all(len(p) == len(rows) for p in values.values())
        assert all('id' not in n or r['id'] == n['id'] for r, n in zip(rows, values['nominal']))
        identity = (str(capture), item['raw_sha256'])
        assert identity not in seen_sources, 'Duplicate source panel; do not inflate frame denominator'
        seen_sources.add(identity); packets[panel] = (capture, rows, values)
    output.mkdir(parents=True); shutil.copyfile(Path(__file__), output/Path(__file__).name)
    all_records = {}; panels = {}
    for panel, (capture, rows, values) in packets.items():
        es = readrows(capture/'evaluator.jsonl'); prov = readrows(capture/'provenance.jsonl')
        assert [r['id'] for r in rows] == [e['id'] for e in es] == [p['id'] for p in prov]
        records = audit_panel(panel, rows, values, es, prov); panels[panel] = {}
        directory = output/panel; directory.mkdir()
        for arm, items in records.items():
            write(directory/(arm+'-changed-supports.json'), items)
            panels[panel][arm] = summarize(items); all_records.setdefault(arm, []).extend(items)
    result = dict(status='EVALUATOR_ONLY_SUPPORT_AUDIT_COMPLETE', panels=panels,
        overall={arm: summarize(items) for arm, items in all_records.items()},
        unique_source_frames=sum(len(rows) for _, rows, _ in packets.values()),
        scope='POSTHOC_AUDIT_OF_SEALED_PREDICTIONS',
        limits=['Projected AABB IoU >= 0.5 unique correspondence is not pixel identity or visibility truth.',
                'Hypothetical actor-origin Radar provenance is not RF or hardware validation.',
                'Frame status categories overlap when a decision has multiple evidence sources.',
                'Measurements reused by multiple later frames remain one measured source, not independent confirmations.',
                'Correct identity support does not establish correct proxy geometry or solve ghosts.'])
    write(output/'summary.json', result)
    write(output/'completion.json', dict(status='PASS', analysis_seal_sha256=sha(seal_path),
        source_panels=seal['panels'], output_hashes={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    print(json.dumps(result['overall'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--analysis', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.analysis, args.output)
