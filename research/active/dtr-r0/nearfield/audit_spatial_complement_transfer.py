"""Independent fixed-policy recount/native audit; no model execution or old test.

Uses the prior independent auditor's hashing and closed-volume geometry helpers,
but never the primary transfer runner, evaluator, or training/prediction functions.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from audit_spatial_bce import read, digest, seal_check, rendered_truth

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts.local/work/ba-spatial-complement-transfer-20260921'
SOURCE = ROOT/'artifacts.local/work/ba-spatial-bce-20260920'
T = .4071309640537889
HIGH = 7.6612162590026855
ARMS = ('A_current', 'C_current', 'A_hold', 'C_hold')


def reconstruct(logits, baseline, metadata):
    """Scalar float64 threshold; explicit previous-RAW reset at clip boundaries."""
    scores = np.asarray(logits, dtype=np.float64)
    assert scores.ndim == 1 and len(scores) == len(baseline) == len(metadata)
    flags = {arm: [] for arm in ARMS}
    previous_clip, previous_a, previous_c = None, False, False
    for i, row in enumerate(metadata):
        if row['clip_id'] != previous_clip:
            previous_a = previous_c = False
        a = bool(baseline[i])
        c = a or bool(float(scores[i]) >= HIGH)
        now = dict(A_current=a, C_current=c, A_hold=a or previous_a, C_hold=c or previous_c)
        for arm in ARMS:
            flags[arm].append(now[arm])
        previous_clip, previous_a, previous_c = row['clip_id'], a, c
    return flags


def count(rows):
    """Independent counts, false segments, positive coverage and first delays."""
    answer = {}
    for stratum in ('Core', 'Boundary'):
        subset = [r for r in rows if (r['layout_relation'] == 'BOUNDARY') == (stratum == 'Boundary')]
        clips = defaultdict(list)
        for row in subset:
            clips[row['clip_id']].append(row)
        answer[stratum] = {}
        for arm in ARMS:
            counts = dict.fromkeys(('TP', 'FP', 'FN', 'TN', 'positive_frames', 'negative_frames',
                'current_unknown', 'segments', 'events', 'detected_events', 'preentry_FP', 'postexit_FP'), 0)
            events = {}
            for name, clip in sorted(clips.items()):
                clip.sort(key=lambda r: r['frame_in_clip'])
                positive = [i for i, r in enumerate(clip) if r['truth']]
                false_previous = False
                for r in clip:
                    truth, active, unknown = r['truth'], r['flags'][arm], r['current_unknown'][arm]
                    counts['positive_frames' if truth else 'negative_frames'] += 1
                    counts['current_unknown'] += int(unknown)
                    if active:
                        counts['TP' if truth else 'FP'] += 1
                    elif truth:
                        counts['FN'] += 1
                    elif not unknown:
                        counts['TN'] += 1
                    false = active and not truth
                    counts['segments'] += int(false and not false_previous)
                    false_previous = false
                if positive:
                    first, last = positive[0], positive[-1]
                    assert positive == list(range(first, last + 1))
                    detections = [i for i in positive if clip[i]['flags'][arm]]
                    pre = sum(r['flags'][arm] for r in clip[:first])
                    post = sum(r['flags'][arm] for r in clip[last + 1:])
                    counts['events'] += 1
                    counts['detected_events'] += bool(detections)
                    counts['preentry_FP'] += pre
                    counts['postexit_FP'] += post
                    events[name] = dict(positive_frames=len(positive), positive_alert_frames=len(detections),
                        coverage=len(detections)/len(positive), first_delay=(detections[0]-first)*.2 if detections else None,
                        preentry_FP=pre, postexit_FP=post)
            counts['precision'] = counts['TP']/(counts['TP']+counts['FP']) if counts['TP']+counts['FP'] else None
            counts['recall'] = counts['TP']/counts['positive_frames'] if counts['positive_frames'] else None
            counts['FPR'] = counts['FP']/counts['negative_frames'] if counts['negative_frames'] else None
            counts['false_sampled_s'] = counts['FP']*.2
            answer[stratum][arm] = dict(counts=counts, event_details=events)
    return answer


def compare(report, independent):
    for stratum, arms in independent.items():
        for arm, result in arms.items():
            observed, expected = result['counts'], report[stratum]['arms'][arm]
            for key in ('TP', 'FP', 'FN', 'TN', 'positive_frames', 'negative_frames', 'precision', 'recall', 'FPR'):
                assert observed[key] == expected['frames'][key], (stratum, arm, key)
            for a, b in (('current_unknown', 'current_unknown'), ('segments', 'false_alert_segment_count'),
                         ('events', 'event_count'), ('detected_events', 'detected_events'), ('false_sampled_s', 'false_alert_sampled_s')):
                assert observed[a] == expected[b], (stratum, arm, a)
            assert len(expected['events']) == len(result['event_details'])
            for event in expected['events']:
                actual = result['event_details'][event['clip_id']]
                for a, b in (('positive_frames', 'positive_frames'), ('positive_alert_frames', 'positive_alert_frames'),
                             ('coverage', 'positive_coverage'), ('preentry_FP', 'preentry_false_alert_frames'),
                             ('postexit_FP', 'postexit_false_alert_frames')):
                    assert actual[a] == event[b], (stratum, arm, a)
                delay = event['first_in_event_alert_delay_s']
                assert delay is None if actual['first_delay'] is None else math.isclose(delay, actual['first_delay'], abs_tol=1e-8)


def old_test_unactivated():
    names = ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json', 'test-frame-results.json')
    assert all(not (SOURCE/name).exists() for name in names), 'Original test unexpectedly activated'
    return names


def audit(out):
    out = Path(out).resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve())
    assert not (out/'independent-audit.json').exists(), 'Write-once audit'
    original_absent = old_test_unactivated()
    p = read(out/'protocol.json')
    assert p['frames'] == 1152 and p['groups'] == 16 and p['clips'] == 48
    assert p['strong_threshold'] == T and p['high_logit'] == HIGH
    assert digest(out/'spec.json') == p['spec_sha256']
    assert digest(out/'protocol-before-run.md') == p['protocol_text_sha256']
    for name, value in p['code_hashes'].items():
        assert digest(HERE/name) == value, name
    for name, value in p['input_hashes'].items():
        assert digest(ROOT/name) == value, name
    for name in ('observation-seal.json', 'prediction-seal.json', 'evaluation-seal.json'):
        seal_check(out, name)
    prior_protocol = read(SOURCE/'protocol.json')
    prior_fit = read(SOURCE/'fit/train_receipt.json')
    assert digest(SOURCE/'fit/head_last.pt') == prior_fit['checkpoint_sha256']
    unchanged = ('spatial_bce_model.py', 'tof_fov45_core.py', 'tof_corridor_calibration.py',
                 'ba_camera_corridor.py', 'core_transfer_spec.py')
    assert all(digest(HERE/name) == prior_protocol['code_hashes'][name] for name in unchanged)
    receipt = read(out/'inference-receipt.json')
    assert receipt['head_sha256'] == prior_fit['checkpoint_sha256']
    assert receipt['rows'] == 1152 and receipt['high_logit'] == HIGH and not receipt['evaluator_labels_read']
    feature_receipt = read(out/'features/feature_receipt.json')
    assert digest(out/'features/rgb_features.npy') == feature_receipt['cache_sha256']
    assert feature_receipt['checkpoint_sha256'] == receipt['encoder_sha256']
    assert feature_receipt['recipe'] == prior_fit['recipe'] and feature_receipt['shape'] == [1152, 216, 8, 8]
    for path in ('features/encoder_backend.json', 'head_backend.json'):
        backend = read(out/path)
        assert backend['selection_status'] == 'SELECTED'
        assert backend['selected_device_type'] in ('cpu', 'cuda')
        assert backend['selected_device_type'] == 'cuda' or backend['selection_reason'] in (
            'CPU_FASTER_MEASURED', 'ACCELERATOR_UNAVAILABLE', 'GPU_BACKEND_UNAVAILABLE')

    spec, meta = read(out/'spec.json'), read(out/'evaluator/metadata.json')
    identities, baseline = read(out/'identities.json'), read(out/'baseline.json')
    predictions, rows = read(out/'predictions.json'), read(out/'frame-results.json')
    labels, geometry = read(out/'evaluator/transfer-labels.json'), read(out/'capture/evaluator/geometry.json')
    admission, lineage = read(out/'source-admission.json'), read(out/'private-lineage.json')
    assert admission['status'] == 'PASS'
    assert all(len(a) == 1152 for a in (spec['cases'], meta, identities, baseline, predictions, rows,
                                      labels, geometry, admission['frames'], lineage))
    assert all(r['split'] == 'transfer' for r in meta)
    groups = Counter(r['base_group_id'] for r in meta)
    assert len(groups) == 16 and set(groups.values()) == {72}
    assert len({r['clip_id'] for r in meta}) == 48
    with np.load(out/'observations.npz', allow_pickle=False) as data:
        ranges, boxes = data['ranges'], data['boxes']
    logits = np.load(out/'logits.npy', allow_pickle=False)
    assert ranges.shape == (1152, 64) and boxes.shape == (64, 4)
    assert logits.shape == (1152,) and logits.dtype == np.float64 and np.isfinite(logits).all()
    from tof_corridor_calibration import score_frame, decide
    from tof_fov45_core import boxes45, simulate
    from ba_camera_corridor import sample_native
    import run_core_transfer as source
    np.testing.assert_array_equal(boxes, boxes45())
    for i, (case, m, obs, prior, pred, row, label, geo, adm, lin) in enumerate(zip(
            spec['cases'], meta, identities, baseline, predictions, rows, labels, geometry, admission['frames'], lineage)):
        assert i == m['index'] == obs['index'] == prior['index'] == pred['index'] == row['index'] == label['index']
        assert obs['id'] == m['id'] == pred['id'] == row['id'] == adm['id'] == lin['id']
        for key in ('clip_id', 'frame_in_clip', 'time_s'):
            assert case[key] == m[key] == obs[key] == row[key]
        assert m['frame_in_clip'] == i % 24 and abs(m['time_s'] - .2*(i % 24)) < 1e-8
        assert digest(out/obs['rgb_path']) == obs['rgb_sha256'] == geo['rgb_sha256']
        assert digest(out/'capture/evaluator'/geo['native_path']) == geo['native_sha256'] == lin['native_sha256']
        assert lin['identity'] == 'spatial-bce-v1/'+case['sensor_noise_key']
        assert rendered_truth(case, geo) == label['truth'] == row['truth'] == adm['truth']
        assert adm['visible_target_pixels'] > 0 and adm['competing_corridor_pixels'] < 4
        s = score_frame(boxes, ranges[i]); a = decide(s, T)
        assert s['score'] == prior['score']
        assert a['alert'] == prior['current'] and a['unknown'] == prior['unknown']
        assert a['valid_zones'] == prior['valid_zones'] and a['definite_zones'] == prior['definite_zones']
        assert pred['logit'] == row['logit'] == float(logits[i])
        assert all(row['current_unknown'][arm] == pred['current_unknown'][arm] == a['unknown'] for arm in ARMS)
        if i % 384 == 0:
            print('TRANSFER_INDEPENDENT_AUDIT', i, '/1152', flush=True)
    flags = reconstruct(logits, [r['current'] for r in baseline], meta)
    for i, (pred, row) in enumerate(zip(predictions, rows)):
        assert pred['flags'] == row['flags'] == {a: flags[a][i] for a in ARMS}
    independent = count(rows)
    compare(read(out/'metrics.json'), independent)
    per_group, stored_groups = {}, read(out/'group-metrics.json')
    for group in sorted(groups):
        per_group[group] = count([r for r in rows if r['base_group_id'] == group])
        compare(stored_groups[group], per_group[group])
    added, onset_retained = {}, True
    for stratum in ('Core', 'Boundary'):
        subset = [r for r in rows if (r['layout_relation'] == 'BOUNDARY') == (stratum == 'Boundary')]
        for suffix in ('current', 'hold'):
            added[stratum+'_'+suffix] = {key: [r['id'] for r in subset if condition(r)] for key, condition in (
                ('added_TP', lambda r: r['truth'] and r['flags']['C_'+suffix] and not r['flags']['A_'+suffix]),
                ('added_FP', lambda r: not r['truth'] and r['flags']['C_'+suffix] and not r['flags']['A_'+suffix]),
                ('lost_A_flags', lambda r: r['flags']['A_'+suffix] and not r['flags']['C_'+suffix]))}
            assert not added[stratum+'_'+suffix]['lost_A_flags']
            aa, cc = independent[stratum]['A_'+suffix], independent[stratum]['C_'+suffix]
            for clip, event in aa['event_details'].items():
                if event['first_delay'] is not None:
                    other = cc['event_details'][clip]['first_delay']
                    onset_retained &= other is not None and other <= event['first_delay'] + 1e-8
    assert onset_retained and added == read(out/'incremental-ids.json')
    native_audits = []
    added_ids = {name for value in added.values() for key in ('added_TP', 'added_FP') for name in value[key]}
    for i, row in enumerate(rows):
        if row['id'] not in added_ids:
            continue
        case, geo = spec['cases'][i], geometry[i]
        native = np.load(out/'capture/evaluator'/geo['native_path'], allow_pickle=False)
        target, corridor = source.masks(native, case, geo)
        recreated, traces = simulate(sample_native(native), 'spatial-bce-v1/'+case['sensor_noise_key'], boxes)
        np.testing.assert_allclose(recreated, ranges[i], rtol=0, atol=0, equal_nan=True)
        target_corridor = sample_native((target & corridor).astype(np.float32)).astype(bool).ravel()
        support = sum(int(target_corridor[t['pixel_indices']].sum()) for t in traces)
        assert support == row['native_target_corridor_samples'] == admission['frames'][i]['returned_target_corridor_samples']
        assert [{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in t.items()} for t in traces] == lineage[i]['traces']
        native_audits.append(dict(id=row['id'],truth=row['truth'],current_added=flags['C_current'][i] and not flags['A_current'][i],
                                 held_added=flags['C_hold'][i] and not flags['A_hold'][i],native_support=support))
    result = read(out/'result.json')
    gain = independent['Boundary']['C_current']['counts']['recall']-independent['Boundary']['A_current']['counts']['recall']
    gaining = sum(g['Boundary']['C_current']['counts']['TP'] > g['Boundary']['A_current']['counts']['TP'] for g in per_group.values())
    signal = gain >= .1 and gaining >= 8 and onset_retained
    def no_cost(suffix):
        return all(independent[g]['C_'+suffix]['counts']['FP'] == independent[g]['A_'+suffix]['counts']['FP']
            and independent[g]['C_'+suffix]['counts']['segments'] <= independent[g]['A_'+suffix]['counts']['segments']
            for g in ('Core', 'Boundary'))
    assert result['prospective_rescue_signal'] == signal and result['boundary_gaining_groups'] == gaining
    assert result['strict_current_upgrade'] == (signal and no_cost('current'))
    assert result['strict_complete_upgrade'] == (signal and no_cost('current') and no_cost('hold'))
    assert result['all_A_flags_retained'] and not result['original_test_activated']
    assert old_test_unactivated() == original_absent
    report = dict(status='PASS', audit_utc=datetime.now(timezone.utc).isoformat(), audit_code_sha256=digest(__file__),
        helper_code_sha256=digest(HERE/'audit_spatial_bce.py'), protocol_sha256=digest(out/'protocol.json'),
        frames=1152, groups=16, exact_head_and_preprocessing_retained=True,
        original_test_label_file_opened=False, original_test_unactivated=True,
        all_RGB_and_native_hashes_verified=True, float64_OR_and_nonrecursive_hold_verified=True,
        independent_counts=independent, independent_group_counts=per_group, incremental_ids=added,
        original_onsets_retained=onset_retained, native_added_frame_audits=native_audits,
        boundary_recall_gain=gain, boundary_gaining_groups=gaining, prospective_rescue_signal=signal,
        strict_current_upgrade=signal and no_cost('current'),
        strict_complete_upgrade=signal and no_cost('current') and no_cost('hold'))
    with (out/'independent-audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('independent_counts','independent_group_counts','incremental_ids','native_added_frame_audits')}), flush=True)


def self_test():
    meta = [dict(clip_id='a' if i < 4 else 'b',frame_in_clip=i%4,time_s=.2*(i%4),
                 layout_relation='BOUNDARY') for i in range(8)]
    values = [HIGH, -100., -100., -100., -100., math.nextafter(HIGH,-math.inf),HIGH,-100.]
    flags = reconstruct(values, [False,False,False,True,False,False,False,False], meta)
    assert flags['C_current'] == [True,False,False,True,False,False,True,False]
    assert flags['C_hold'] == [True,True,False,True,False,False,True,True]
    rows=[dict(**m,truth=i%4 in (1,2),flags={a:flags[a][i] for a in ARMS},
               current_unknown={a:False for a in ARMS}) for i,m in enumerate(meta)]
    result=count(rows)['Boundary']
    assert result['C_current']['counts']['TP'] == 1 and result['C_current']['counts']['FP'] == 2
    assert result['C_hold']['counts']['TP'] == 2 and result['C_hold']['counts']['FP'] == 3
    assert result['C_hold']['counts']['segments'] == 3
    print('SYNTHETIC_SELF_TEST_PASS; no experiment data accessed')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    self_test() if args.self_test else audit(args.output)
