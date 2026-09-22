"""Independent saved-output verification of the frozen projection sensitivity run.

No runner, scorer or metric helpers are imported. Public geometric support and
all decisions are reconstructed; sealed score integrals are inputs, not rerun.
Metrics are rebuilt from predictions plus the original evaluator files, never
from the runner's joined frame-results. Only the requested receipt is written.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT = ROOT / 'artifacts.local/work/ba-core-projection-stress-20260921'
OFFSETS = {'left': -2, 'nominal': 0, 'right': 2}
POLICIES = ('calibration', 'strong', 'hold')
ARMS = tuple(c + '_' + p for c in OFFSETS for p in POLICIES)
DT, T0, T = .2, .007085703945147101, .4071309640537889
COUNTS = Counter()
HASHES = []


def require(condition, message):
    COUNTS['assertions'] += 1
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def check_hash(path, expected):
    actual = digest(path)
    require(actual == expected, 'Hash mismatch: ' + str(path))
    HASHES.append({'path': str(path), 'sha256': actual})


def check_seal(directory, name):
    receipt = read(directory / name)
    check_hash(directory / 'protocol.json', receipt['protocol_sha256'])
    for name, expected in receipt['hashes'].items():
        check_hash(directory / name, expected)
    COUNTS['seals'] += 1


def compare(saved, rebuilt, path):
    """All discrete values exact; independently accumulated floats within 1e-12."""
    if isinstance(rebuilt, dict):
        require(isinstance(saved, dict) and saved.keys() == rebuilt.keys(), path + ': keys')
        for key, value in rebuilt.items():
            compare(saved[key], value, path + '/' + key)
    elif isinstance(rebuilt, list):
        require(isinstance(saved, list) and len(saved) == len(rebuilt), path + ': length')
        for index, value in enumerate(rebuilt):
            compare(saved[index], value, path + '/' + str(index))
    elif isinstance(rebuilt, float):
        require(type(saved) in (int, float) and math.isfinite(saved)
                and math.isclose(saved, rebuilt, rel_tol=0., abs_tol=1e-12), path + ': float')
    else:
        require(type(saved) is type(rebuilt) and saved == rebuilt, path + ': exact value')
    COUNTS['compared_nodes'] += 1


def spans(values):
    """Inclusive ranges, derived independently with run grouping."""
    return [(group[0][0], group[-1][0])
            for state, members in groupby(enumerate(values), key=lambda item: bool(item[1]))
            if state for group in [list(members)]]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def segment(clip, first, last):
    length = last - first + 1
    return {'clip_id': clip[0]['clip_id'], 'start_frame': first, 'end_frame': last,
            'start_time_s': clip[first]['time_s'], 'end_time_s': clip[last]['time_s'],
            'samples': length, 'sampled_s': length * DT}


def event_and_clip(clip, arm):
    alert = [row['flags'][arm] for row in clip]
    episodes = spans(alert)
    first_clip = episodes[0][0] if episodes else None
    detail = {'clip_id': clip[0]['clip_id'],
              'whole_clip_first_alert_time_s': clip[first_clip]['time_s'] if first_clip is not None else None,
              'alert_episode_count': len(episodes),
              'alert_episode_start_times_s': [clip[first]['time_s'] for first, _ in episodes],
              'event': None}
    positive = spans([row['truth'] for row in clip])
    require(len(positive) == (0 if clip[0]['layout_relation'] == 'OUTSIDE' else 1),
            clip[0]['clip_id'] + ': contiguous event contract')
    if not positive:
        return detail, 0
    begin, finish = positive[0]
    observed = alert[begin:finish + 1]
    on = [i for i, value in enumerate(observed) if value]
    silent = spans([not value for value in observed])
    longest = max((last - first + 1 for first, last in silent), default=0)
    first = on[0] if on else None
    last = on[-1] if on else None
    initial = first if first is not None else len(observed)
    terminal = len(observed) - last - 1 if last is not None else 0
    internal = [segment(clip, begin + a, begin + b) for a, b in silent
                if first is not None and a > first and b < last]
    after = alert[finish + 1:]
    post_spans = spans(after)
    first_off = next((i for i, value in enumerate(after) if not value), None)
    prefix = next((b + 1 for a, b in post_spans if a == 0), 0)
    post_new = [first for first, _ in episodes if first > finish]
    within = [first for first, _ in episodes if begin <= first <= finish]
    entry = clip[begin]['time_s']
    carryover = prefix if observed[-1] else 0
    detail['event'] = {
        'entry_frame': begin, 'last_positive_frame': finish, 'entry_time_s': entry,
        'positive_frames': len(observed), 'positive_alert_frames': len(on),
        'positive_coverage': ratio(len(on), len(observed)), 'detected': bool(on),
        'entry_left_censored': begin == 0,
        'first_in_event_alert_time_s': clip[begin + first]['time_s'] if first is not None else None,
        'first_in_event_alert_delay_s': first * DT if first is not None else None,
        'whole_clip_first_alert_relative_to_entry_s': clip[first_clip]['time_s'] - entry if first_clip is not None else None,
        'preentry_false_alert_frames': sum(alert[:begin]),
        'preexisting_alert_at_entry': bool(begin and alert[begin - 1] and alert[begin]),
        'in_event_new_alert_episode_count': len(within),
        'in_event_new_alert_episode_start_times_s': [clip[i]['time_s'] for i in within],
        'initial_silent_frames': initial, 'initial_silent_sampled_s': initial * DT,
        'internal_interruption_count': len(internal), 'internal_interruptions': internal,
        'internal_silent_frames': sum(item['samples'] for item in internal),
        'internal_silent_sampled_s': sum(item['samples'] for item in internal) * DT,
        'terminal_silent_frames': terminal, 'terminal_silent_sampled_s': terminal * DT,
        'total_silent_frames': len(observed) - len(on),
        'exit_observed': bool(after),
        'first_negative_exit_time_s': clip[finish + 1]['time_s'] if after else None,
        'postexit_observed_frames': len(after),
        'postexit_leading_alert_frames': prefix if after else None,
        'postexit_carryover_alert_frames': carryover if after else None,
        'postexit_carryover_sampled_s': carryover * DT if after else None,
        'first_silent_relative_to_exit_s': first_off * DT if first_off is not None else None,
        'release_right_censored': first_off is None,
        'postexit_false_alert_frames': sum(after), 'postexit_false_alert_sampled_s': sum(after) * DT,
        'postexit_false_alert_episode_count': len(post_spans),
        'postexit_false_alert_episodes': [segment(clip, finish + 1 + a, finish + 1 + b) for a, b in post_spans],
        'postexit_new_alert_episode_count': len(post_new),
        'postexit_new_alert_episode_start_times_s': [clip[i]['time_s'] for i in post_new],
    }
    require(initial + terminal + sum(x['samples'] for x in internal) == len(observed) - len(on),
            clip[0]['clip_id'] + ': silence partitions')
    return detail, longest


def aggregate(clips, longest_by_arm):
    rows = [row for clip in clips for row in clip]
    result = {'frames': len(rows), 'clips': len(clips), 'arms': {}}
    for arm in ARMS:
        positive = sum(row['truth'] for row in rows)
        negative = len(rows) - positive
        tp = sum(row['truth'] and row['flags'][arm] for row in rows)
        fp = sum(not row['truth'] and row['flags'][arm] for row in rows)
        fn = positive - tp
        unknown = sum(row['current_unknown'][arm] for row in rows)
        counts = {
            'TP': tp, 'FP': fp, 'FN': fn,
            'TN': sum(not r['truth'] and not r['flags'][arm] and not r['current_unknown'][arm] for r in rows),
            'abstained_positive': sum(r['truth'] and not r['flags'][arm] and r['current_unknown'][arm] for r in rows),
            'abstained_negative': sum(not r['truth'] and not r['flags'][arm] and r['current_unknown'][arm] for r in rows),
            'current_unknown': unknown,
            'current_unknown_positive': sum(r['truth'] and r['current_unknown'][arm] for r in rows),
            'current_unknown_negative': sum(not r['truth'] and r['current_unknown'][arm] for r in rows),
            'frames': len(rows), 'positive_frames': positive, 'negative_frames': negative,
            'precision': ratio(tp, tp + fp), 'recall': ratio(tp, positive),
            'FPR': ratio(fp, negative), 'F1': ratio(2 * tp, 2 * tp + fp + fn)}
        # Keep every positive/negative time position; do not concatenate negatives.
        false = [segment(clip, a, b) for clip in clips
                 for a, b in spans([not r['truth'] and r['flags'][arm] for r in clip])]
        descriptions = []
        for clip in clips:
            detail, longest = event_and_clip(clip, arm)
            descriptions.append(detail)
            longest_by_arm[arm][clip[0]['clip_id']] = longest
        events = [{'clip_id': c['clip_id'], **c['event']} for c in descriptions if c['event'] is not None]
        detected = sum(e['detected'] for e in events)
        result['arms'][arm] = {
            'frames': counts, 'current_unknown': unknown,
            'alert_episode_count': sum(c['alert_episode_count'] for c in descriptions),
            'false_alert_segments': false, 'false_alert_segment_count': len(false),
            'false_alert_sampled_s': fp * DT, 'clips': descriptions, 'events': events,
            'event_count': len(events), 'detected_events': detected,
            'event_recall': ratio(detected, len(events))}
    return result


def rebuild_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['clip_id']].append(row)
    clips = []
    for name in sorted(grouped):
        clip = sorted(grouped[name], key=lambda r: r['frame_in_clip'])
        require(len(clip) == 24, name + ': complete 24-frame clip')
        for i, row in enumerate(clip):
            require(row['frame_in_clip'] == i and abs(row['time_s'] - i * DT) < 1e-8,
                    name + ': contiguous timeline')
            for key in ('layout_relation', 'layer', 'background', 'base_group_id'):
                require(row[key] == clip[0][key], name + ': constant stratum ' + key)
        clips.append(clip)
    require(len(clips) == 48, '48 clips')
    longest = {arm: {} for arm in ARMS}
    report = {'dt_s': DT, 'duration_convention': 'sample count * dt_s; not wall-clock latency',
              'Core_definition': 'Complete INSIDE and OUTSIDE clips',
              'overall': aggregate(clips, longest),
              'Core': aggregate([c for c in clips if c[0]['layout_relation'] != 'BOUNDARY'], longest),
              'Boundary': aggregate([c for c in clips if c[0]['layout_relation'] == 'BOUNDARY'], longest),
              'subgroups': {}}
    for key in ('layout_relation', 'layer', 'background'):
        report['subgroups'][key] = {value: aggregate([c for c in clips if c[0][key] == value], longest)
                                  for value in sorted({c[0][key] for c in clips})}
    return report, longest


def check_public_support(boxes, values, saved):
    """Reconstruct possible/definite ray-volume support, without score integration."""
    focal = 640 / (2 * np.tan(np.deg2rad(50.)))
    possible = definite = valid = 0
    for (y0, x0, y1, x1), measured in zip(boxes, values):
        if not np.isfinite(measured) or measured <= 0:
            continue
        valid += 1
        a = [(float(x) * 2.5 - 320) / focal for x in (x0, x1)]
        b = [(float(y) * (360 / 192) - 180) / focal for y in (y0, y1)]
        uncertainty = .1 + 3 * (.01 + .02 * float(measured))
        lo, hi = max(.1, float(measured) - uncertainty), float(measured) + uncertainty

        def far_z(horizontal, vertical):
            limits = [3.]
            if horizontal:
                limits.append(.3 / abs(horizontal))
            if vertical:
                limits.append((.9 if vertical > 0 else -.2) / vertical)
            return min(limits)

        center_a = 0. if a[0] <= 0 <= a[1] else min(a, key=abs)
        center_b = 0. if b[0] <= 0 <= b[1] else min(b, key=abs)
        possible += int(max(lo, .3) <= min(hi, far_z(center_a, center_b)))
        definite += int(lo >= .3 and all(hi <= far_z(x, y) for x in a for y in b))
    require((saved['valid_zones'], saved['possible_zones'], saved['definite_zones']) == (valid, possible, definite),
            'Independent public support counts')
    require(saved['raw_alert'] is bool(possible) and saved['unknown'] is (definite == 0),
            'Public raw alert/UNKNOWN')
    require(math.isfinite(saved['score']) and 0 <= saved['score'] <= 1, 'Sealed score range')
    require(possible > 0 or saved['score'] == 0., 'No possible support implies score zero')
    COUNTS['independent_geometry_frames'] += 1


def verify(out):
    protocol = read(out / 'protocol.json')
    require(protocol['id'] == 'ba-core-projection-stress-20260921', 'Experiment identity')
    require(protocol['offsets'] == OFFSETS and protocol['dt_s'] == DT, 'Fixed perturbation/time')
    require(protocol['thresholds'] == {'calibration': T0, 'strong': T}, 'Frozen thresholds')
    require(protocol['acceptance'] == {'core_events': 16, 'onset_delay_s': .2, 'coverage': 5/6,
                                      'max_silent_samples': 1, 'fp_reduction_fraction': .5}, 'Frozen criteria')
    source = ROOT / protocol['source']
    require(source == ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921', 'Consumed source only')
    check_hash(out / 'protocol-before-run.md', protocol['protocol_text_sha256'])
    check_hash(HERE / 'CORE_PROJECTION_STRESS_PROTOCOL_20260921.md', protocol['protocol_text_sha256'])
    for name, expected in protocol['code_hashes'].items():
        check_hash(HERE / name, expected)
    for name, expected in protocol['input_hashes'].items():
        check_hash(source / name, expected)
    for directory in (source, out):
        for name in ('observation-seal.json', 'prediction-seal.json', 'evaluation-seal.json'):
            if (directory / name).exists() and not (directory == source and name == 'evaluation-seal.json'):
                check_seal(directory, name)
    source_protocol = read(source / 'protocol.json')
    check_hash(source / 'protocol-before-run.md', source_protocol['protocol_text_sha256'])
    # Source generation code has a separate historical seal. The decoder files
    # used now must retain their source-generation identities as well.
    for name in ('tof_corridor_calibration.py', 'ba_camera_corridor.py', 'tof_fov45_core.py'):
        check_hash(HERE / name, source_protocol['code_hashes'][name])
    started, evaluated = read(out / 'prediction-start.json'), read(out / 'evaluation-start.json')
    require(datetime.fromisoformat(protocol['frozen_at_utc']) <= datetime.fromisoformat(started['time_utc'])
            <= datetime.fromisoformat(evaluated['time_utc']), 'Stage time order')
    require(evaluated['prediction_seal_sha256'] == digest(out / 'prediction-seal.json'), 'Join bound to prediction seal')
    require(started['model_calls'] == 0 and started['evaluator_values_read'] is False
            and started['native_depth_read'] is False and started['raw_ranges_changed'] is False,
            'Sealed prediction declaration')
    protected = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
    for name in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'):
        require(not (protected / name).exists(), 'Original reserved test remains unactivated: ' + name)

    with np.load(source / 'observations.npz', allow_pickle=False) as archive:
        ranges, original_boxes = archive['ranges'].copy(), archive['boxes'].copy()
    require(ranges.shape == (1152, 64) and ranges.dtype == np.float32, 'Original range layout/dtype')
    require(original_boxes.shape == (64, 4) and original_boxes.dtype == np.int64, 'Original box layout/dtype')
    valid = ranges[np.isfinite(ranges)]
    require(bool(np.all((valid >= .1) & (valid < 8))), 'Original public validity range')
    receipt = read(out / 'prediction-receipt.json')
    range_bytes = hashlib.sha256(ranges.tobytes(order='C')).hexdigest()
    require(range_bytes == receipt['range_bytes_sha256'] and receipt['range_bytes_unchanged'] is True,
            'Range bytes unchanged from sealed source')
    require(receipt['public_source_sha256'] == digest(source / 'observations.npz')
            and receipt['nominal_parity_frames'] == 1152, 'Source/zero-offset receipt')
    projections = read(out / 'reported-projections.json')
    require(projections.keys() == OFFSETS.keys(), 'All and only declared projections')
    for condition, offset in OFFSETS.items():
        shifted = np.asarray(projections[condition])
        expected = original_boxes + np.asarray([0, offset, 0, offset], dtype=np.int64)
        require(shifted.dtype.kind in 'iu' and np.array_equal(shifted, expected), condition + ': exact shifted edges')
        require(np.array_equal(shifted[:, 2:] - shifted[:, :2], original_boxes[:, 2:] - original_boxes[:, :2]),
                condition + ': all zone sizes retained')
        require(bool(np.all(shifted[:, 1] >= 0) and np.all(shifted[:, 3] <= 256)), condition + ': no clipping')
        projections[condition] = shifted

    identities = read(source / 'identities.json')
    baseline = read(source / 'baseline.json')
    parent = read(source / 'predictions.json')
    predictions = read(out / 'predictions.json')
    metadata = read(source / 'evaluator/metadata.json')
    labels = read(source / 'evaluator/transfer-labels.json')
    require(all(len(items) == 1152 for items in (identities, baseline, parent, predictions, metadata, labels)), 'Full coverage')
    require(len({p['id'] for p in predictions}) == 1152, 'Unique prediction identities')
    rows, prior = [], {}
    for i, (identity, old, original, pred, meta, label) in enumerate(zip(identities, baseline, parent, predictions, metadata, labels)):
        require(all(item['index'] == i for item in (identity, old, original, pred, meta, label)), 'Index alignment')
        require(all(item['id'] == identity['id'] for item in (original, pred, meta)), 'ID alignment')
        for key in ('clip_id', 'frame_in_clip', 'time_s'):
            require(pred[key] == identity[key] == meta[key], 'Public identity/metadata alignment: ' + key)
        require(type(label['truth']) is bool, 'Boolean source truth')
        require(pred['flags'].keys() == pred['current_unknown'].keys() == set(ARMS), 'Nine declared arms')
        require(pred['support'].keys() == OFFSETS.keys(), 'Three support conditions')
        for condition in OFFSETS:
            support = pred['support'][condition]
            check_public_support(projections[condition], ranges[i], support)
            definite = support['definite_zones'] > 0
            strong = definite or (support['raw_alert'] and support['score'] >= T)
            calibration = definite or (support['raw_alert'] and support['score'] >= T0)
            previous = prior.get((condition, identity['clip_id']))
            inherited = bool(previous is not None and abs(identity['time_s'] - previous[0] - DT) < 1e-8 and previous[1])
            decisions = {'strong': strong, 'calibration': calibration, 'hold': strong or inherited}
            for policy in POLICIES:
                arm = condition + '_' + policy
                require(type(pred['flags'][arm]) is bool and pred['flags'][arm] == decisions[policy], arm + ': decision')
                require(type(pred['current_unknown'][arm]) is bool and pred['current_unknown'][arm] == (not definite),
                        arm + ': UNKNOWN is current geometric support')
                COUNTS['decision_rows'] += 1
            prior[condition, identity['clip_id']] = (identity['time_s'], strong)
        nominal = pred['support']['nominal']
        require(nominal['score'] == old['score'], 'Exact nominal score')
        require(pred['flags']['nominal_strong'] == old['current'] == original['flags']['A_current'], 'Exact nominal current')
        require(pred['flags']['nominal_hold'] == original['flags']['A_hold'], 'Exact nominal held alert')
        require(nominal['unknown'] == old['unknown'] == original['current_unknown']['A_hold'], 'Exact nominal UNKNOWN')
        require((nominal['valid_zones'], nominal['definite_zones']) == (old['valid_zones'], old['definite_zones']),
                'Exact nominal valid/definite counts')
        COUNTS['nominal_parity_frames'] += 1
        rows.append({**meta, 'truth': label['truth'], 'flags': pred['flags'],
                     'current_unknown': pred['current_unknown'], 'support': pred['support']})
    compare(read(out / 'frame-results.json'), rows, 'independent_evaluator_join')
    metrics, longest = rebuild_metrics(rows)
    compare(read(out / 'metrics.json'), metrics, 'all_metric_fields')

    summaries, changes, checks = {}, {}, {}
    for stratum in ('Core', 'Boundary'):
        subset = [r for r in rows if (r['layout_relation'] == 'BOUNDARY') == (stratum == 'Boundary')]
        summaries[stratum], changes[stratum] = {}, {}
        for arm in ARMS:
            calculated = metrics[stratum]['arms'][arm]
            events = calculated['events']
            condition = arm.rsplit('_', 1)[0]
            summaries[stratum][arm] = {
                **{name: calculated['frames'][name] for name in ('TP', 'FP', 'FN', 'precision', 'recall', 'FPR', 'current_unknown')},
                'events': calculated['detected_events'], 'event_count': calculated['event_count'],
                'false_segments': calculated['false_alert_segment_count'], 'false_sampled_s': calculated['false_alert_sampled_s'],
                'max_detected_onset_delay_s': max((e['first_in_event_alert_delay_s'] for e in events if e['detected']), default=None),
                'min_event_coverage': min((e['positive_coverage'] for e in events), default=None),
                'preentry_FP': sum(e['preentry_false_alert_frames'] for e in events),
                'postexit_FP': sum(e['postexit_false_alert_frames'] for e in events),
                'false_definite_frames': sum(not r['truth'] and r['support'][condition]['definite_zones'] > 0 for r in subset)}
        for condition in ('left', 'right'):
            candidate, reference = condition + '_hold', 'nominal_hold'
            changes[stratum][condition] = {name: [] for name in ('lost_nominal_TP', 'added_TP', 'added_FP', 'removed_FP', 'unknown_changed')}
            for row in subset:
                before, after = row['flags'][reference], row['flags'][candidate]
                if row['truth'] and before and not after:
                    changes[stratum][condition]['lost_nominal_TP'].append(row['id'])
                if row['truth'] and not before and after:
                    changes[stratum][condition]['added_TP'].append(row['id'])
                if not row['truth'] and not before and after:
                    changes[stratum][condition]['added_FP'].append(row['id'])
                if not row['truth'] and before and not after:
                    changes[stratum][condition]['removed_FP'].append(row['id'])
                if row['current_unknown'][reference] != row['current_unknown'][candidate]:
                    changes[stratum][condition]['unknown_changed'].append(row['id'])
    compare(read(out / 'changed-ids.json'), changes, 'all_changed_ids')
    for condition in OFFSETS:
        base, arm = condition + '_calibration', condition + '_hold'
        b, c = (metrics['Core']['arms'][key] for key in (base, arm))
        events = c['events']
        checks[condition] = {
            'all_core_events': c['event_count'] == 16 and c['detected_events'] == 16,
            'onset_within_budget': all(e['first_in_event_alert_delay_s'] is not None and e['first_in_event_alert_delay_s'] <= .2 + 1e-8 for e in events),
            'coverage': all(e['positive_coverage'] >= 5/6 for e in events),
            'fp_reduction': b['frames']['FP'] > 0 and c['frames']['FP'] * 2 <= b['frames']['FP'],
            'core_false_segments': c['false_alert_segment_count'] <= b['false_alert_segment_count'],
            'max_silence': all(longest[arm][e['clip_id']] <= 1 for e in events)}
        for relation in ('INSIDE', 'OUTSIDE'):
            group = metrics['subgroups']['layout_relation'][relation]['arms']
            checks[condition][relation + '_FP'] = group[arm]['frames']['FP'] <= group[base]['frames']['FP']
            checks[condition][relation + '_false_alert_segment_count'] = group[arm]['false_alert_segment_count'] <= group[base]['false_alert_segment_count']
    result = read(out / 'result.json')
    compare(result['summary'], summaries, 'result_summaries')
    compare(result['checks'], checks, 'all_acceptance_checks')
    passed = {condition: all(values.values()) for condition, values in checks.items()}
    compare(result['passed_by_condition'], passed, 'pass_by_condition')
    compare(result['both_signed_conditions_pass'], passed['left'] and passed['right'], 'both_signs')
    require(result['original_test_activated'] is False and result['automatic_successor'] is False, 'No successor/test promotion')
    events = {stratum: {arm: [{**e, 'max_silent_samples': longest[arm][e['clip_id']]}
                             for e in metrics[stratum]['arms'][arm]['events']] for arm in ARMS}
              for stratum in ('Core', 'Boundary')}
    deltas = {condition: {key: summaries['Core'][condition + '_hold'][key] - summaries['Core']['nominal_hold'][key]
                          for key in ('TP', 'FP', 'FN', 'false_segments', 'current_unknown')}
              for condition in ('left', 'right')}
    return {
        'status': 'PASS', 'experiment_id': protocol['id'], 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': digest(Path(__file__)), 'python_executable': sys.executable,
        'backend': 'TASK_NOT_GPU_SUITABLE', 'counts': dict(COUNTS), 'checked_hashes': HASHES,
        'range_bytes_sha256': range_bytes, 'range_dtype': str(ranges.dtype), 'range_shape': list(ranges.shape),
        'input_coordinates': '64 integer y0,x0,y1,x1 boxes; horizontal edges both shifted by -2/0/+2 at LOW_W=256',
        'public_geometry_recomputed': True, 'score_integrals_recomputed': False,
        'all_metrics_rebuilt_independently': True, 'evaluator_join_uses_runner_frame_results': False,
        'summary': summaries, 'checks': checks, 'passed_by_condition': passed,
        'per_event_details': events, 'hold_changes_from_nominal': changes, 'core_hold_deltas_from_nominal': deltas,
        'interpretation': [
            'Passing is relative to Calibration under the same shifted input, not lossless retention of nominal performance.',
            'Left and right have asymmetric false-alert costs on this consumed simulated cohort; no hardware bias cause is identified.',
            'Current UNKNOWN is recomputed from the reported geometry; it is not a hardware confidence certificate.',
            'Sealed score integrals are trusted inputs; this verification independently rebuilds support, decisions and metrics, not the integral.',
            'Range-byte and source hashes verify retained inputs; sealed staged code/receipts support, but do not independently observe, historical execution access.',
            'Controlled consumed simulation only; no original reserved test, model training, hardware evidence or automatic successor.'],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--stdout-only', action='store_true', help='Recheck without overwriting the exclusive verification receipt.')
    args = parser.parse_args()
    report = verify(args.out)
    if not args.stdout_only:
        with (args.out / 'independent-verification.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'status': report['status'], 'counts': report['counts'],
                      'passed': report['passed_by_condition'],
                      'Core_hold': {c: {k: report['summary']['Core'][c + '_hold'][k]
                                       for k in ('TP', 'FP', 'FN', 'events', 'false_segments')}
                                    for c in OFFSETS}}))


if __name__ == '__main__':
    main()
