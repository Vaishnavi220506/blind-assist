"""Post-seal independent arithmetic and preservation audit for data coverage.

Never imports or reads original protected test outcomes. Evaluation labels are
opened only after every public-input prediction and evaluation seal validates.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
import time

import numpy as np

from run_data_coverage import OUT, SOURCE, read, write, verify, check, sha
from tof_corridor_calibration import score_frame, decide


def hold(flags, rows):
    result = list(flags)
    for i in range(1, len(rows)):
        if rows[i]['clip_id'] == rows[i-1]['clip_id']:
            assert abs(rows[i]['time_s']-rows[i-1]['time_s']-.2) < 1e-8
            result[i] = bool(flags[i] or flags[i-1])
    return result


def segments(flags, truth, rows, mask):
    previous, previous_clip, count = False, None, 0
    for flag, positive, row, include in zip(flags, truth, rows, mask):
        current = bool(flag and not positive and include)
        count += int(current and (not previous or previous_clip != row['clip_id']))
        previous, previous_clip = current, row['clip_id']
    return count


def replay_public_n(rows, predictions, ranges, boxes, cutoff):
    """64 fixed public-input frames; no fitting or labels used for replay."""
    import torch
    import spatial_bce_model as model
    from run_data_coverage import CHECKPOINT

    model._precision()
    positions = [i for i, row in enumerate(rows)
                 if row['layout_relation'] == 'BOUNDARY' and row['frame_in_clip'] in (0, 8, 13, 23)]
    assert len(positions) == 64
    assert set(Counter(rows[i]['base_group_id'] for i in positions).values()) == {4}
    identities = read(OUT/'identities.json')
    indices = [rows[i]['index'] for i in positions]
    public = [identities[i] for i in indices]
    paths = [OUT/p['rgb_path'] for p in public]
    assert all(p['index'] == i for p, i in zip(public, indices))
    assert all(sha(path) == p['rgb_sha256'] for path, p in zip(paths, public))
    normalized = boxes/np.array([192, 256, 192, 256], np.float32)
    started = time.perf_counter()
    rgb = model.encode_rgb(paths, CHECKPOINT, OUT/'replay-encoder', boxes=normalized)
    inputs = model.build_inputs(rgb, ranges[indices], normalized)
    head = model.load_head(OUT/'fit/head_last.pt')
    device, backend = model._select(head, torch.from_numpy(inputs.copy()), OUT/'N-replay-backend.json')
    head.to(device).eval()
    actual = model.predict(head, inputs)
    expected = np.array([predictions[i]['logits']['N'] for i in positions], np.float32)
    assert actual.shape == expected.shape == (64,)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    actual_p = torch.sigmoid(torch.from_numpy(actual)).numpy()
    expected_p = torch.sigmoid(torch.from_numpy(expected)).numpy()
    # Probability, not saturated-logit drift, is the numerical acceptance gate.
    assert np.allclose(actual_p, expected_p, atol=1e-6, rtol=1e-5)
    assert np.array_equal(actual.astype(float) >= cutoff, expected.astype(float) >= cutoff)
    return dict(status='PASS', frames=64, groups=16,
                frame_in_clip=[0, 8, 13, 23], relation='BOUNDARY', indices=indices,
                max_logit_diff=float(np.max(np.abs(actual-expected))),
                max_probability_diff=float(np.max(np.abs(actual_p-expected_p))),
                probability_atol=1e-6, probability_rtol=1e-5,
                exact_supplement_decisions=True, refit=False,
                actual_device=str(next(head.parameters()).device), backend=backend,
                elapsed_s=time.perf_counter()-started)


def audit():
    protocol = verify()
    # These checks hash sealed files; no label content is read at this point.
    for name in ('observation', 'features', 'fit', 'selection', 'prediction', 'evaluation'):
        check(name+'-seal.json')
    predictions = read(OUT/'predictions.json')
    routing = read(OUT/'split-routing.json')
    assert {s: len(v) for s, v in routing.items()} == dict(train=1728, dev=576, evaluation=1152)
    assert sorted(sum(routing.values(), [])) == list(range(3456))
    assert len(predictions) == 1152 and [p['index'] for p in predictions] == routing['evaluation']
    labels = read(OUT/'evaluator/evaluation-labels.json')
    metadata = read(OUT/'evaluator/metadata.json')
    rows = [metadata[p['index']] for p in predictions]
    assert [l['index'] for l in labels] == [r['index'] for r in rows]
    assert {r['split'] for r in rows} == {'evaluation'}
    assert len({r['base_group_id'] for r in rows}) == 16
    assert set(Counter(r['base_group_id'] for r in rows).values()) == {72}
    assert len({r['clip_id'] for r in rows}) == 48
    assert set(Counter(r['clip_id'] for r in rows).values()) == {24}
    assert Counter(r['layout_relation'] for r in rows) == dict(INSIDE=384, BOUNDARY=384, OUTSIDE=384)
    truth = [bool(l['truth']) for l in labels]
    baseline = read(OUT/'baseline.json')
    selection = read(OUT/'selection.json')['arms']
    thresholds = dict(B_frozen=7.6612162590026855, R_frozen=5.128307342529297,
                      B_control=selection['B_control']['threshold'], N=selection['N']['threshold'])
    with np.load(OUT/'evaluation-logits.npz', allow_pickle=False) as data:
        logits = {k: data[k].astype(float) for k in ('B', 'R', 'N')}
    assert all(v.shape == (1152,) and np.isfinite(v).all() for v in logits.values())
    a, unknown = [], []
    with np.load(OUT/'observations.npz', allow_pickle=False) as data:
        ranges, boxes = data['ranges'], data['boxes']
        assert ranges.shape == (3456, 64)
        for p in predictions:
            index = p['index']
            got = decide(score_frame(boxes, ranges[index]), protocol['strong_threshold'])
            assert bool(got['alert']) == bool(baseline[index]['current'])
            assert bool(got['unknown']) == bool(baseline[index]['unknown'])
            a.append(bool(got['alert'])); unknown.append(bool(got['unknown']))
    flags = {'A_current': a, 'A_hold': hold(a, rows)}
    for name, cutoff in thresholds.items():
        score = logits['B' if name.startswith('B') else name[0]]
        current = [bool(aa or value >= cutoff) for aa, value in zip(a, score)]
        flags[name+'_current'] = current
        flags[name+'_hold'] = hold(current, rows)
    for i, prediction in enumerate(predictions):
        assert set(prediction['flags']) == set(flags) == set(prediction['current_unknown'])
        assert all(bool(prediction['flags'][arm]) == values[i] for arm, values in flags.items())
        assert all(bool(v) == unknown[i] for v in prediction['current_unknown'].values())
        assert prediction['logits'] == {k: float(v[i]) for k, v in logits.items()}
    metrics, result = read(OUT/'metrics.json'), read(OUT/'result.json')
    calculated_costs = {name: {} for name in thresholds}
    for region in ('Core', 'Boundary'):
        mask = [(r['layout_relation'] == 'BOUNDARY') == (region == 'Boundary') for r in rows]
        negatives = sum(keep and not y for keep, y in zip(mask, truth))
        clips = len({r['clip_id'] for r, keep in zip(rows, mask) if keep})
        for arm, values in flags.items():
            expected = {key: 0 for key in ('TP', 'FP', 'FN', 'TN', 'current_unknown')}
            for keep, y, flag, unk in zip(mask, truth, values, unknown):
                if keep:
                    expected['current_unknown'] += int(unk)
                    if flag:
                        expected['TP' if y else 'FP'] += 1
                    elif y:
                        expected['FN'] += 1
                    elif not unk:
                        expected['TN'] += 1
            saved = metrics[region]['arms'][arm]
            assert all(saved['frames'][k] == v for k, v in expected.items()), (region, arm)
            assert saved['frames']['negative_frames'] == negatives
            assert saved['frames']['positive_frames'] == sum(keep and y for keep, y in zip(mask, truth))
            assert saved['frames']['recall'] == expected['TP']/saved['frames']['positive_frames']
            assert saved['frames']['FPR'] == expected['FP']/negatives
            assert saved['frames']['precision'] == (expected['TP']/(expected['TP']+expected['FP']) if expected['TP']+expected['FP'] else None)
            assert saved['false_alert_segment_count'] == segments(values, truth, rows, mask)
        for name in thresholds:
            for mode, fraction in (('current', .01), ('hold', .02)):
                aa, cc = flags['A_'+mode], flags[name+'_'+mode]
                added = sum(keep and not y and c for keep, y, c in zip(mask, truth, cc)) - sum(keep and not y and a0 for keep, y, a0 in zip(mask, truth, aa))
                added_segments = segments(cc, truth, rows, mask)-segments(aa, truth, rows, mask)
                fp_cap, segment_cap = math.floor(fraction*negatives), math.floor(.125*clips)
                cost = dict(negative_frames=negatives, clips=clips, added_FP=added, FP_cap=fp_cap,
                            added_segments=added_segments, segment_cap=segment_cap,
                            pass_cost=added <= fp_cap and added_segments <= segment_cap)
                calculated_costs[name][region+'_'+mode] = cost
    assert result['cost'] == calculated_costs
    by_clip = defaultdict(list)
    for i, row in enumerate(rows):
        by_clip[row['clip_id']].append(i)
    for region in ('Core', 'Boundary'):
        eligible = {clip: indices for clip, indices in by_clip.items()
                    if (rows[indices[0]]['layout_relation'] == 'BOUNDARY') == (region == 'Boundary')}
        for arm, values in flags.items():
            saved = metrics[region]['arms'][arm]
            events = {event['clip_id']: event for event in saved['events']}
            expected_events = {clip: idx for clip, idx in eligible.items() if any(truth[i] for i in idx)}
            assert set(events) == set(expected_events) and saved['event_count'] == len(expected_events)
            detected = 0
            for clip, indices in expected_events.items():
                positives = [i for i in indices if truth[i]]
                first = next((i for i in positives if values[i]), None)
                event = events[clip]
                detected += int(first is not None)
                assert event['detected'] == (first is not None)
                assert event['positive_frames'] == len(positives)
                assert event['positive_alert_frames'] == sum(values[i] for i in positives)
                assert event['exit_observed'] and not event['entry_left_censored']
                if first is None:
                    assert event['first_in_event_alert_delay_s'] is None
                else:
                    assert math.isclose(event['first_in_event_alert_delay_s'], rows[first]['time_s']-rows[positives[0]]['time_s'], abs_tol=1e-9)
            assert saved['detected_events'] == detected
    for mode in ('current', 'hold'):
        aa, nn = flags['A_'+mode], flags['N_'+mode]
        assert all(not av or nv for av, nv in zip(aa, nn))
        for indices in by_clip.values():
            old = next((i for i in indices if truth[i] and aa[i]), None)
            new = next((i for i in indices if truth[i] and nn[i]), None)
            assert old is None or (new is not None and rows[new]['time_s'] <= rows[old]['time_s'])
    gains = []
    for group in sorted({r['base_group_id'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['base_group_id'] == group and r['layout_relation'] == 'BOUNDARY' and truth[i]]
        if sum(flags['N_hold'][i] for i in idx) > sum(flags['A_hold'][i] for i in idx):
            gains.append(group)
    nrec = metrics['Boundary']['arms']['N_hold']['frames']['recall']
    brec = metrics['Boundary']['arms']['B_control_hold']['frames']['recall']
    gates = dict(cost=all(v['pass_cost'] for v in calculated_costs['N'].values()),
                 retain_A_events_and_onset=True, boundary_recall=nrec >= .5,
                 broad_gain=len(gains) >= 8, matched_control_gain=nrec-brec >= .10-1e-12)
    assert result['gates'] == gates and result['boundary_gain_groups'] == gains
    assert result['status'] == ('PASS' if all(gates.values()) else 'NO_GO')
    usable = all(value for key, value in gates.items() if key != 'matched_control_gain')
    assert result['candidate_usable'] == usable
    assert result['data_condition_contribution_supported'] == all(gates.values())
    assert read(OUT/'local-inheritance.json')['inheritance_role'] == ('COMPONENT_OR_CHALLENGER' if usable else 'NEGATIVE_CONTROL')
    train = read(OUT/'fit/train_receipt.json')
    inherited = read(SOURCE/'fit/train_receipt.json')
    assert train['recipe'] == inherited['recipe'] and train['train_rows'] == inherited['train_rows'] == 1728
    assert train['recipe']['steps'] == 1200 and train['recipe']['batch_size'] == 64
    replay = replay_public_n(rows, predictions, ranges, boxes, thresholds['N'])
    report = dict(status='PASS', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        prediction_seal_sha256=sha(OUT/'prediction-seal.json'), evaluation_seal_sha256=sha(OUT/'evaluation-seal.json'),
        frames=1152, groups=16, clips=48, original_recipe_preserved=True,
        flags_costs_and_terminal_recomputed=True, A_recomputed_from_public_ToF=True,
        A_current_hold_onset_preserved=True, UNKNOWN_preserved=True,
        all_arm_event_counts_and_first_delays_verified=True, public_N_replay=replay,
        evaluation_labels_opened_after_prediction_seal=True, original_protected_test_read=False)
    write(OUT/'independent-audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    audit()
