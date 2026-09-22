"""Evaluate sealed lateral vote attribution on consumed camera-corridor96.

No model or score pass runs here. Native contributor and whole-target truth are
evaluation-only authority, accessed after all prediction bindings are checked.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from ba_camera_corridor import WIDTH, LOW_W, HFOV, rays, sample_native
from ba_camera_corridor_metrics import evaluate_rows
from evaluate_ba_camera_corridor import read, require, sha, write_new

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/ba-tof-lateral-attribution-20260920'
CALIBRATED = ROOT / 'artifacts.local/work/ba-tof-corridor-calibration-20260920'
NOMINAL = ROOT / 'artifacts.local/work/ba-tof-fov45-20260920'
CAMERA = ROOT / 'artifacts.local/work/ba-camera-corridor-20260919'
ARMS = ('A', 'B', 'C', 'C_no_rgb')
CHALLENGERS = ARMS[1:]
IDENTITY = ('id', 'clip_id', 'frame_in_clip', 'time_s')


def bound_hashes(record, bindings):
    for key, path in bindings.items():
        require(record[key] == sha(path), 'Hash binding differs: ' + key)


def lateral_eligible(box, value, anchor):
    """Public near-return gate over the unchanged complete interval support."""
    if not np.isfinite(value) or not 0 < value < 3 or not anchor['possible']:
        return False
    _, x0, _, x1 = box
    focal = WIDTH / (2 * np.tan(np.deg2rad(HFOV / 2)))
    slopes = ((x0 * WIDTH / LOW_W - WIDTH / 2) / focal,
              (x1 * WIDTH / LOW_W - WIDTH / 2) / focal)
    envelope = [slope * depth for slope in slopes for depth in anchor['interval_m']]
    return any(min(envelope) <= face <= max(envelope) for face in (-.3, .3))


def fp_subtype(row):
    return '+'.join(name for name, value in zip(('lateral', 'vertical', 'depth'),
                    row['axis_penetration_m']) if value < -1e-9) or 'unresolved'


def whole_target_class(row):
    """Closed original target bounds; exact contact stays CROSSING."""
    low = row['target_camera_bounds_m']['lower'][0]
    high = row['target_camera_bounds_m']['upper'][0]
    if high < -.3 - 1e-9 or low > .3 + 1e-9:
        return 'OUTSIDE'
    if low > -.3 + 1e-9 and high < .3 - 1e-9:
        return 'INSIDE'
    return 'CROSSING'


def verify_predictions(out):
    """Validate sealed public predictions before loading evaluator truth."""
    from run_tof_lateral import verify
    # Reuse exact code/input checks, including the recorded pre-fit backend
    # API repair. Original protocol and public/supervision seals stay immutable.
    protocol = verify(out)
    seal = read(out / 'prediction-seal.json')
    require(seal['status'] == 'COMPLETE' and seal['frames'] == 96, 'Predictions incomplete')
    bound_hashes(seal, dict(protocol_sha256=out / 'protocol.json',
                           predictions_sha256=out / 'predictions.json'))
    parent_seal = read(CALIBRATED / 'prediction-seal.json')
    require(parent_seal['status'] == 'COMPLETE' and parent_seal['frames'] == 96,
            'Parent predictions incomplete')
    bound_hashes(parent_seal, dict(protocol_sha256=CALIBRATED / 'protocol.json',
        score_seal_sha256=CALIBRATED / 'score-seal.json',
        label_sha256=CAMERA / 'frame-results.json',
        point_sha256=CALIBRATED / 'operating-point.json',
        predictions_sha256=CALIBRATED / 'predictions.json'))
    bound_hashes(read(CALIBRATED / 'score-seal.json'),
        dict(scores_sha256=CALIBRATED / 'scores.json',
             observation_seal_sha256=NOMINAL / 'observation-seal.json'))
    bound_hashes(read(NOMINAL / 'observation-seal.json'),
        dict(protocol_sha256=NOMINAL / 'protocol.json',
             observations_sha256=NOMINAL / 'observations.json',
             private_lineage_sha256=NOMINAL / 'private-lineage.json'))
    predictions = read(out / 'predictions.json')
    parent = read(CALIBRATED / 'predictions.json')
    scores = read(CALIBRATED / 'scores.json')
    observations = read(NOMINAL / 'observations.json')
    point = read(CALIBRATED / 'operating-point.json')
    require(len(predictions) == len(parent) == len(scores) == len(observations) == 96,
            'Public cohort count differs')
    observable_audits = []
    for index, (pred, previous, scored, obs) in enumerate(zip(predictions, parent, scores, observations)):
        require(pred['id'] == f'f{index:04d}', 'Prediction order differs')
        require(all(pred[key] == previous[key] == scored[key] == obs[key]
                    for key in IDENTITY), 'Public temporal identity differs')
        require(set(pred['arms']) == set(ARMS)
                and set(pred['suppressed_zones']) == set(CHALLENGERS), 'Arm schema differs')
        require(previous['anchors'] == scored['anchors'], 'Original anchors differ')
        require(previous['baseline'] == scored['baseline'], 'Original raw baseline differs')
        require(scored['observation_sha256'] == previous['observation_sha256'] == obs['sha256'],
                'Parent scalar input binding differs')
        require(sha(NOMINAL / obs['path']) == obs['sha256'], 'Original scalar vector changed')
        anchors = {a['zone']: a for a in previous['anchors']}
        zone_scores = {s['zone']: s['joint'] for s in scored['zone_scores']}
        require(set(zone_scores) == set(anchors), 'Original zone scores differ')
        supported = any(a['definite'] for a in anchors.values())
        for key in ('alert', 'unknown', 'ambiguous'):
            require(pred['arms']['A'][key] == previous['candidate'][key], 'A differs from frozen baseline')
        with np.load(NOMINAL / obs['path'], allow_pickle=False) as data:
            eligibility = {z: bool(lateral_eligible(data['boxes'][z], float(data['values'][z]), a))
                           for z, a in anchors.items()}
        per_arm = {}
        for arm in ARMS:
            decision = pred['arms'][arm]
            require(all(type(decision[key]) is bool for key in ('alert', 'unknown', 'ambiguous')),
                    'Decision flags must be booleans')
            suppressed = pred['suppressed_zones'][arm] if arm != 'A' else []
            require(isinstance(suppressed, list) and all(type(z) is int for z in suppressed)
                    and len(set(suppressed)) == len(suppressed), 'Invalid suppressed-zone list')
            require(set(suppressed) <= set(anchors), 'Suppressed zone lacks an observed anchor')
            require(all(eligibility[z] for z in suppressed), 'Suppression exceeds public near/lateral scope')
            retained = set(anchors) - set(suppressed)
            expected = bool(supported or any(anchors[z]['possible'] and zone_scores[z] >= point['threshold']
                                             for z in retained))
            require(decision['alert'] == expected, 'Saved decision does not match fixed votes: ' + arm)
            require(decision['unknown'] == previous['candidate']['unknown'], 'UNKNOWN changed: ' + arm)
            require(decision['ambiguous'] == bool(expected and not supported), 'Ambiguity changed: ' + arm)
            require(not expected or pred['arms']['A']['alert'], 'Suppression created a new alert')
            per_arm[arm] = dict(suppressed_zones=suppressed,
                suppressed_definite_zones=[z for z in suppressed if anchors[z]['definite']],
                eligible_zones=[z for z in anchors if eligibility[z]])
        observable_audits.append(per_arm)
    return protocol, predictions, parent, scores, observations, observable_audits


def evaluate(out=OUT):
    out = Path(out).resolve()
    require(not any((out / name).exists() for name in
        ('results.json', 'frame-results.json', 'native-support-audit.json')), 'Evaluation outputs already exist')
    began = time.perf_counter()
    protocol, predictions, parent, scores, observations, observable = verify_predictions(out)
    # Private evaluator authority starts only after every public binding above.
    labels = read(CAMERA / 'frame-results.json')
    sources = read(CAMERA / 'evaluator-source.json')
    lineage = read(NOMINAL / 'private-lineage.json')
    nominal_rows = read(NOMINAL / 'frame-results.json')
    require(len(labels) == len(sources) == len(lineage) == len(nominal_rows) == 96,
            'Evaluator cohort count differs')
    rows, native_rows = [], []
    a, b = rays()
    for pred, previous, label, source, trace_row, nominal, public in zip(
            predictions, parent, labels, sources, lineage, nominal_rows, observable):
        require(all(pred[key] == label[key] for key in IDENTITY), 'Evaluator temporal identity differs')
        require(pred['id'] == source['id'] == trace_row['id'] == nominal['id'], 'Native identity differs')
        require(type(label['truth']) is bool, 'Original known-truth denominator changed')
        rows.append({**label, 'predictions': pred['arms'],
            'fp_subtype': fp_subtype(label) if label['truth'] is False else None,
            'whole_target_lateral_class': whole_target_class(label)})
        native_path = CAMERA / 'capture/evaluator' / source['geometry']['native_path']
        require(sha(native_path) == source['sensor_native_sha256'], 'Original native depth changed')
        depth = sample_native(np.load(native_path, allow_pickle=False))
        inside = (np.isfinite(depth) & (depth >= .3) & (depth <= 3.) & (abs(a * depth) <= .3)
                  & (b * depth >= -.2) & (b * depth <= .9)).ravel()
        traces = {t['zone_id']: t for t in trace_row['traces'] if t['observed']}
        anchors = {an['zone']: an for an in previous['anchors']}
        require(set(traces) == set(anchors), 'Observed raw anchors are not all retained by reference')
        ownership = {z['zone_id']: z['ownership'] for z in nominal['new_lineage']['zones']}
        corridor_counts = {}
        for zone, trace in traces.items():
            indices = np.asarray(trace['pixel_indices'], dtype=int)
            require(len(indices) > 0 and len(np.unique(indices)) == len(indices)
                    and np.all((indices >= 0) & (indices < inside.size)), 'Invalid inherited native contributors')
            corridor_counts[zone] = int(inside[indices].sum())
        arm_audits = {}
        for arm in CHALLENGERS:
            suppressed = []
            for zone in public[arm]['suppressed_zones']:
                trace = traces[zone]
                indices = np.asarray(trace['pixel_indices'], dtype=int)
                suppressed.append(dict(zone=zone, distance_m=trace['distance_m'],
                    original_interval_m=anchors[zone]['interval_m'],
                    raw_possible=anchors[zone]['possible'], raw_definite=anchors[zone]['definite'],
                    contributor_pixels=len(indices), corridor_contributor_samples=corridor_counts[zone],
                    corridor_contributor_indices=indices[inside[indices]].tolist(),
                    ownership=ownership[zone],
                    whole_target_class=whole_target_class(label) if ownership[zone] == 'pure' else 'UNKNOWN',
                    semantics='Raw support retained by reference; its alert vote is suppressed'))
            arm_audits[arm] = dict(suppressed_zones=suppressed,
                suppressed_corridor_contributor_samples=sum(z['corridor_contributor_samples'] for z in suppressed),
                frame_alert_retained=pred['arms'][arm]['alert'],
                suppressed_definite_zones=public[arm]['suppressed_definite_zones'])
        native_rows.append(dict(id=pred['id'], clip_id=pred['clip_id'], frame_in_clip=pred['frame_in_clip'],
            observed_anchor_count=len(anchors),
            original_possible_corridor_contributors=sum(corridor_counts[z] for z in anchors if anchors[z]['possible']),
            arms=arm_audits))
    metrics = evaluate_rows(rows, .2, ARMS)
    base = metrics['arms']['A']
    parent_metrics = read(CALIBRATED / 'results.json')['metrics']['arms']['candidate']
    require(base == parent_metrics, 'Frozen A metric/event replay differs')
    require(tuple(base['frames']['all_known'][k] for k in ('TP', 'FP', 'FN', 'TN')) == (30, 15, 6, 0),
            'Baseline constants changed')
    original_tp = [r['id'] for r in rows if r['truth'] and r['predictions']['A']['alert']]
    lateral_fp = [r['id'] for r in rows if r['truth'] is False and r['predictions']['A']['alert']
                  and r['fp_subtype'] == 'lateral']
    require(len(lateral_fp) == 6, 'Original six lateral false-alert frames changed')
    comparisons = {}
    for arm in CHALLENGERS:
        current = metrics['arms'][arm]
        paired = {}
        for name, truth, before, after in (
                ('lost_tp', True, True, False), ('rescued_fn', True, False, True),
                ('removed_fp', False, True, False), ('added_fp', False, False, True)):
            paired[name] = [r['id'] for r in rows if r['truth'] is truth
                and r['predictions']['A']['alert'] is before and r['predictions'][arm]['alert'] is after]
        events = []
        for old, new in zip(base['events'], current['events']):
            require(all(old[k] == new[k] for k in ('clip_id', 'start_frame', 'end_frame', 'entry_time_s',
                                                  'interior_frames', 'boundary_frames')), 'Event identity changed')
            delayed = old['detected'] and (not new['detected'] or new['first_alert_time_s'] > old['first_alert_time_s'])
            events.append(dict(baseline=old, candidate=new, delayed=bool(delayed)))
        native_arm = [n['arms'][arm] for n in native_rows]
        affected_native_frames = [n['id'] for n in native_rows
            if n['arms'][arm]['suppressed_corridor_contributor_samples'] > 0]
        native_summary = dict(suppressed_zone_instances=sum(len(n['suppressed_zones']) for n in native_arm),
            suppressed_corridor_contributor_samples=sum(n['suppressed_corridor_contributor_samples'] for n in native_arm),
            suppressed_corridor_frame_ids=affected_native_frames,
            suppressed_corridor_frames_still_alerting=[n['id'] for n in native_rows
                if n['arms'][arm]['suppressed_corridor_contributor_samples'] > 0 and n['arms'][arm]['frame_alert_retained']],
            suppressed_definite_zone_instances=sum(len(n['suppressed_definite_zones']) for n in native_arm))
        checks = dict(original30_tp_ids_retained=not paired['lost_tp'] and current['frames']['all_known']['TP'] == 30,
            no_new_alerts=all(not r['predictions'][arm]['alert'] or r['predictions']['A']['alert'] for r in rows),
            no_new_fp=not paired['added_fp'], no_missing_event_rescue=not paired['rescued_fn'],
            interior_events_retained=current['interior_detected_events'] == 4,
            all_events_retained=current['detected_events'] == 5,
            no_event_first_delay=not any(e['delayed'] for e in events),
            fewer_fp=current['frames']['all_known']['FP'] < 15,
            false_segments_not_increased=current['false_alert_segment_count'] <= base['false_alert_segment_count'],
            false_duration_not_increased=current['false_alert_sampled_duration_s'] <= base['false_alert_sampled_duration_s'],
            no_suppressed_native_corridor_contributors=native_summary['suppressed_corridor_contributor_samples'] == 0,
            no_suppressed_definite_votes=native_summary['suppressed_definite_zone_instances'] == 0,
            unknown_retained=all(r['predictions'][arm]['unknown'] == r['predictions']['A']['unknown'] for r in rows),
            nonalert_not_clear=current['frames']['all_known']['TN'] == 0)
        comparisons[arm] = dict(paired=paired, events=events,
            original_lateral_fp_removed=[i for i in lateral_fp if i in paired['removed_fp']],
            original_lateral_fp_retained=[i for i in lateral_fp if i not in paired['removed_fp']],
            native_support=native_summary, gate=dict(passed=all(checks.values()), checks=checks))
    result = dict(status='COMPLETE', scope='ALL96_CONSUMED_DEVELOPMENT_IN_SAMPLE_ONLY', frames=96,
        metrics=metrics, baseline_exact_parity=True, original30_tp_ids=original_tp,
        original6_lateral_fp_ids=lateral_fp, comparisons=comparisons,
        fp_subtypes={arm: dict(Counter(r['fp_subtype'] for r in rows
            if r['truth'] is False and r['predictions'][arm]['alert'])) for arm in ARMS},
        by_clip={clip: evaluate_rows([r for r in rows if r['clip_id'] == clip], .2, ARMS)['arms']
                 for clip in sorted({r['clip_id'] for r in rows})},
        raw_anchor_reference=dict(path=(CALIBRATED / 'predictions.json').relative_to(ROOT).as_posix(),
            sha256=sha(CALIBRATED / 'predictions.json'), anchors_unchanged_by_reference=True,
            observed_anchor_instances=sum(n['observed_anchor_count'] for n in native_rows)),
        limitations=['All96 consumed; learned results are fitting diagnostics, not held-out/generalization evidence.',
            'Native support counts are exact observed winning-bin sampled points, not full surfaces or physical sensor guarantees.',
            'Raw anchors retained by reference does not mean alert votes retained; all suppressed votes are explicitly audited.',
            'Whole-target closed bounds remain evaluator truth; local outside samples do not certify full-target exclusion.',
            'One frozen comparison; no model, score, calibration or alternative inference runs in this evaluator.'],
        evaluator_sha256=sha(__file__), evaluator_backend=dict(device='CPU', reason='TASK_NOT_GPU_SUITABLE'),
        elapsed_s=time.perf_counter() - began)
    write_new(out / 'frame-results.json', rows)
    write_new(out / 'native-support-audit.json', native_rows)
    result['hashes'] = {name: sha(out / name) for name in ('protocol.json', 'prediction-seal.json',
        'predictions.json', 'frame-results.json', 'native-support-audit.json')}
    write_new(out / 'results.json', result)
    summary = {arm: {key: metrics['arms'][arm]['frames']['all_known'][key]
                    for key in ('TP', 'FP', 'FN', 'TN')} for arm in ARMS}
    for arm in CHALLENGERS:
        summary[arm].update(gate_passed=comparisons[arm]['gate']['passed'],
            removed_fp=comparisons[arm]['paired']['removed_fp'],
            lost_tp=comparisons[arm]['paired']['lost_tp'],
            suppressed_native_corridor_samples=comparisons[arm]['native_support']['suppressed_corridor_contributor_samples'])
    print(json.dumps(dict(status=result['status'], arms=summary), ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    evaluate(args.out)
