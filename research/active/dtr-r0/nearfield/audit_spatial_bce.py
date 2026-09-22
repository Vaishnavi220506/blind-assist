"""Independent source/count/operating-point audit; never fits or predicts a model.

Only the original scalar A scorer is reused for baseline replay. Geometry truth,
current/hold counts, false segments and operating-point reconstruction below do
not import the primary runner or its metric/selection functions.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DEFAULT = ROOT/'artifacts.local/work/ba-spatial-bce-20260920'
ARMS = ('A_current', 'B_current', 'A_hold', 'B_hold')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def seal_check(out, name):
    obj = read(out/name)
    assert obj['protocol_sha256'] == digest(out/'protocol.json'), name
    for relative, expected in obj['hashes'].items():
        path = (out/relative).resolve()
        assert path.is_relative_to(out.resolve()) and digest(path) == expected, relative


def rendered_truth(case, geometry):
    """Independent level-camera AABB overlap, including closed-volume contact."""
    target = next(o for o in geometry['objects'] if o['name'] == 'target')
    center, half = target['render_bounds_center_m'], target['render_bounds_extent_m']
    camera = case['camera']
    assert all(abs(camera[a]) < 1e-10 for a in ('pitch', 'yaw', 'roll'))
    lo = [center[1]-half[1]-camera['y'], camera['z']-center[2]-half[2], center[0]-half[0]-camera['x']]
    hi = [center[1]+half[1]-camera['y'], camera['z']-center[2]+half[2], center[0]+half[0]-camera['x']]
    query = ((-.3, .3), (-.2, .9), (.3, 3.))
    return all(lo[i] <= query[i][1]+1e-9 and hi[i] >= query[i][0]-1e-9 for i in range(3))


def replay(logits, threshold, truth, metadata, baseline, unknown=None):
    """Independent per-clip iteration; hold refers only to preceding RAW output."""
    assert len(logits) == len(truth) == len(metadata) == len(baseline)
    unknown = [False]*len(truth) if unknown is None else unknown
    clips = defaultdict(list)
    for i, row in enumerate(metadata):
        clips[row['clip_id']].append(i)
    flags = {arm: [False]*len(truth) for arm in ARMS}
    groups = {stratum: {arm: dict(TP=0, FP=0, FN=0, TN=0, positives=0, negatives=0,
               unknown=0, segments=0, events=0, detected_events=0, first_delays={}) for arm in ARMS}
              for stratum in ('Core', 'Boundary')}
    lost = {'current': [], 'hold': []}
    for clip_id, positions in clips.items():
        positions.sort(key=lambda i: metadata[i]['frame_in_clip'])
        relation = metadata[positions[0]]['layout_relation']
        stratum = 'Boundary' if relation == 'BOUNDARY' else 'Core'
        prev_a = prev_b = False
        prev_fp = dict.fromkeys(ARMS, False)
        positives = [i for i in positions if truth[i]]
        first_positive_time = metadata[positives[0]]['time_s'] if positives else None
        first_detection = {}
        for local, i in enumerate(positions):
            if local:
                assert math.isclose(metadata[i]['time_s']-metadata[positions[local-1]]['time_s'], .2, abs_tol=1e-8)
            a, b = bool(baseline[i]), float(logits[i]) >= float(threshold)
            current = dict(A_current=a, B_current=b, A_hold=a or prev_a, B_hold=b or prev_b)
            for arm, alert in current.items():
                flags[arm][i] = alert
                g = groups[stratum][arm]
                y = bool(truth[i])
                g['positives' if y else 'negatives'] += 1
                g['unknown'] += int(unknown[i])
                if alert:
                    g['TP' if y else 'FP'] += 1
                elif y:
                    g['FN'] += 1
                elif not unknown[i]:
                    g['TN'] += 1
                fp = alert and not y
                g['segments'] += int(fp and not prev_fp[arm])
                prev_fp[arm] = fp
                if alert and y and arm not in first_detection:
                    first_detection[arm] = round(metadata[i]['time_s']-first_positive_time, 9)
            if stratum == 'Core' and truth[i]:
                for suffix in ('current', 'hold'):
                    if current['A_'+suffix] and not current['B_'+suffix]:
                        lost[suffix].append(metadata[i]['id'])
            prev_a, prev_b = a, b
        for arm in ARMS:
            if positives:
                groups[stratum][arm]['events'] += 1
                groups[stratum][arm]['detected_events'] += int(arm in first_detection)
                groups[stratum][arm]['first_delays'][clip_id] = first_detection.get(arm)
    checks, summary = {}, {}
    for suffix in ('current', 'hold'):
        checks['retain_Core_'+suffix] = not lost[suffix]
        for stratum in ('Core', 'Boundary'):
            a, b = groups[stratum]['A_'+suffix], groups[stratum]['B_'+suffix]
            key = stratum+'_'+suffix
            checks[key+'_FP'] = b['FP'] <= a['FP']
            checks[key+'_segments'] = b['segments'] <= a['segments']
            summary[key] = dict(A_TP=a['TP'], B_TP=b['TP'], A_FP=a['FP'], B_FP=b['FP'],
                A_segments=a['segments'], B_segments=b['segments'], positives=a['positives'])
    record = dict(threshold=float(threshold), admissible=all(checks.values()), checks=checks, summary=summary)
    return record, groups, flags


def compare_metrics(report, groups):
    for stratum, arms in groups.items():
        for arm, observed in arms.items():
            stored = report[stratum]['arms'][arm]
            for key in ('TP', 'FP', 'FN', 'TN'):
                assert observed[key] == stored['frames'][key], (stratum, arm, key)
            assert observed['positives'] == stored['frames']['positive_frames']
            assert observed['negatives'] == stored['frames']['negative_frames']
            assert observed['unknown'] == stored['current_unknown']
            assert observed['segments'] == stored['false_alert_segment_count']
            assert observed['events'] == stored['event_count']
            assert observed['detected_events'] == stored['detected_events']
            for event in stored['events']:
                expected = observed['first_delays'][event['clip_id']]
                actual = event['first_in_event_alert_delay_s']
                assert expected is actual if expected is None else math.isclose(expected, actual, abs_tol=1e-8)


def audit(out):
    out = out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve())
    assert not (out/'independent-audit.json').exists(), 'Write-once audit already exists'
    protocol = read(out/'protocol.json')
    assert digest(out/'spec.json') == protocol['spec_sha256']
    assert digest(out/'protocol-before-run.md') == protocol['protocol_text_sha256']
    for name, expected in protocol['code_hashes'].items():
        assert digest(HERE/name) == expected, name
    for name, expected in protocol['input_hashes'].items():
        assert digest(ROOT/name) == expected, name
    for name in ('observation-seal.json', 'fit-seal.json', 'development-seal.json'):
        seal_check(out, name)
    op = read(out/'operating-point.json')
    assert op['status'] in ('DEV_ADMISSIBLE', 'DEV_NO_ADMISSIBLE_OPERATING_POINT')
    test_active = op['status'] == 'DEV_ADMISSIBLE'
    if test_active:
        assert (out/'test-prediction-seal.json').is_file()
        seal_check(out, 'test-prediction-seal.json')
        seal_check(out, 'evaluation-seal.json')
    else:
        for name in ('test-logits.npy', 'test-start.json', 'test-prediction-seal.json', 'test-frame-results.json', 'result.json'):
            assert not (out/name).exists(), 'Test activated after development failure: '+name
    spec, metadata = read(out/'spec.json'), read(out/'evaluator/metadata.json')
    identities, baseline = read(out/'identities.json'), read(out/'baseline.json')
    geometry = read(out/'capture/evaluator/geometry.json')
    manifest = read(out/'capture/observations/manifest.json')['frames']
    lineage = read(out/'private-lineage.json')
    admission = read(out/'source-admission.json')
    assert admission['status'] == 'PASS'
    assert all(len(x) == 2880 for x in (spec['cases'], metadata, identities, baseline, geometry, manifest, lineage, admission['frames']))
    groups, clips = defaultdict(list), defaultdict(list)
    for row in metadata:
        groups[row['base_group_id']].append(row)
        clips[row['clip_id']].append(row)
    assert len(groups) == 40 and len(clips) == 120
    splits = Counter()
    family_splits = defaultdict(Counter)
    for rows in groups.values():
        assert len(rows) == 72 and len({r['split'] for r in rows}) == 1
        assert Counter(r['layout_relation'] for r in rows) == dict(INSIDE=24, BOUNDARY=24, OUTSIDE=24)
        splits[rows[0]['split']] += 1
        family_splits[rows[0]['type_id']][rows[0]['split']] += 1
    assert splits == dict(train=24, dev=8, test=8)
    assert len(family_splits) == 4 and all(c == dict(train=6, dev=2, test=2) for c in family_splits.values())
    for rows in clips.values():
        assert [r['frame_in_clip'] for r in rows] == list(range(24))
        assert all(math.isclose(r['time_s'], .2*i, abs_tol=1e-8) for i, r in enumerate(rows))
    with np.load(out/'observations.npz', allow_pickle=False) as packet:
        ranges, boxes = packet['ranges'], packet['boxes']
    assert ranges.shape == (2880, 64) and boxes.shape == (64, 4)
    from tof_corridor_calibration import score_frame, decide
    geometry_truth = {}
    for i, (case, meta, obs, geo, image, prior, lineage_row, admission_row) in enumerate(zip(
            spec['cases'], metadata, identities, geometry, manifest, baseline, lineage, admission['frames'])):
        assert i == meta['index'] == obs['index'] == geo['sample_index'] == image['sample_index'] == prior['index']
        assert case['name'] == geo['id'] == image['id']
        assert obs['id'] == meta['id'] == lineage_row['id'] == admission_row['id']
        for key in ('clip_id', 'frame_in_clip'):
            assert case[key] == meta[key] == obs[key] == geo[key] == image[key]
        assert case['time_s'] == meta['time_s'] == obs['time_s'] == geo['nominal_time_s'] == image['time_s']
        for key in ('base_group_id', 'split', 'layout_relation', 'type_id'):
            assert case[key] == meta[key]
        assert geo['declared_camera'] == case['camera']
        assert all(abs(geo['actual_camera_location_m'][j]-case['camera'][key]) <= .002
                   for j, key in enumerate(('x', 'y', 'z')))
        assert all(abs(v) <= 1e-6 for v in geo['actual_camera_rotation'])
        assert len(geo['objects']) == len(case['objects']) == 2
        for declared, actual in zip(case['objects'], geo['objects']):
            assert declared['name'] == actual['name']
            assert all(abs(a-b) <= .002 for a, b in zip(declared['center_m'], actual['render_bounds_center_m']))
            assert all(abs(a-2*b) <= .002 for a, b in zip(declared['size_m'], actual['render_bounds_extent_m']))
        truth = rendered_truth(case, geo)
        geometry_truth[i] = truth
        assert truth == admission_row['truth']
        rgb_path = out/obs['rgb_path']
        assert rgb_path.resolve() == (out/'capture/observations'/image['rgb_path']).resolve()
        assert digest(rgb_path) == obs['rgb_sha256'] == image['rgb_sha256'] == geo['rgb_sha256']
        assert digest(out/'capture/evaluator'/geo['native_path']) == geo['native_sha256'] == lineage_row['native_sha256']
        assert lineage_row['identity'] == 'spatial-bce-v1/'+case['sensor_noise_key']
        assert len(lineage_row['traces']) == 64
        for zone, trace in enumerate(lineage_row['traces']):
            assert trace['zone_id'] == zone
            if trace['observed']:
                assert float(ranges[i, zone]) == trace['distance_m']
            else:
                assert np.isnan(ranges[i, zone])
        score = score_frame(boxes, ranges[i])
        decision = decide(score, protocol['strong_threshold'])
        assert score['score'] == prior['score']
        assert decision['alert'] == prior['current'] and decision['unknown'] == prior['unknown']
        assert decision['valid_zones'] == prior['valid_zones'] and decision['definite_zones'] == prior['definite_zones']
        if i % 480 == 0:
            print('INDEPENDENT_SOURCE_AUDIT', i, '/2880', flush=True)
    labels, split_meta, scores = {}, {}, {}
    for split in ('train', 'dev') + (('test',) if test_active else ()):
        split_meta[split] = [r for r in metadata if r['split'] == split]
        label_rows = read(out/f'evaluator/{split}-labels.json')
        assert [r['index'] for r in label_rows] == [r['index'] for r in split_meta[split]]
        labels[split] = [r['truth'] for r in label_rows]
        assert all(type(r['truth']) is bool and r['truth'] == geometry_truth[r['index']] for r in label_rows)
        scores[split] = np.load(out/f'{split}-logits.npy', allow_pickle=False)
        assert scores[split].shape == (len(label_rows),) and np.isfinite(scores[split]).all()
    fit = read(out/'fit/train_receipt.json')
    train_indices = np.asarray([r['index'] for r in split_meta['train']], np.int64)
    train_labels = np.asarray(labels['train'], np.float32)
    assert fit['train_rows'] == 1728
    assert fit['train_indices_sha256'] == hashlib.sha256(train_indices.tobytes()).hexdigest()
    assert fit['train_labels_sha256'] == hashlib.sha256(train_labels.tobytes()).hexdigest()
    assert fit['checkpoint_sha256'] == digest(out/'fit/head_last.pt')
    summaries = {}
    def recount(split, threshold):
        meta = split_meta[split]
        return replay(scores[split], threshold, labels[split], meta,
            [baseline[r['index']]['current'] for r in meta], [baseline[r['index']]['unknown'] for r in meta])
    for split in ('train', 'dev'):
        _, g, _ = recount(split, 0.)
        compare_metrics(read(out/f'{split}-zero-logit-metrics.json'), g)
        summaries[split+'_zero'] = g
    thresholds = sorted(set(float(x) for x in scores['dev']))
    thresholds.append(math.nextafter(thresholds[-1], math.inf))
    reconstructed = [recount('dev', t)[0] for t in thresholds]
    assert reconstructed == read(out/'development-thresholds.json'), 'Exact development curve mismatch'
    eligible = [r for r in reconstructed if r['admissible']]
    selected = sorted(eligible, key=lambda r: (r['summary']['Boundary_current']['B_TP'],
        r['summary']['Core_current']['B_TP'], r['threshold']), reverse=True)[0] if eligible else None
    assert selected == op['selected'] and len(eligible) == op['admissible'] and len(thresholds) == op['thresholds']
    assert bool(selected) == test_active == op['test_activated']
    assert op['dev_logits_sha256'] == digest(out/'dev-logits.npy')
    if selected:
        _, g, _ = recount('dev', selected['threshold'])
        compare_metrics(read(out/'dev-selected-metrics.json'), g)
        summaries['dev_selected'] = g
        record, g, flags = recount('test', selected['threshold'])
        compare_metrics(read(out/'test-metrics.json'), g)
        rows = read(out/'test-frame-results.json')
        assert len(rows) == len(split_meta['test'])
        for i, row in enumerate(rows):
            assert row['id'] == split_meta['test'][i]['id'] and row['truth'] == labels['test'][i]
            assert row['logit'] == float(scores['test'][i])
            assert row['flags'] == {arm: values[i] for arm, values in flags.items()}
        result = read(out/'result.json')
        boundary = record['summary']['Boundary_current']
        gain = (boundary['B_TP']-boundary['A_TP'])/boundary['positives']
        checks = dict(record['checks'], boundary_current_gain_10pp=gain >= .1-1e-12)
        assert checks == result['checks'] and gain == result['boundary_current_recall_gain']
        assert result['status'] == ('PASS' if all(checks.values()) else 'NEGATIVE_CONTROL')
        summaries['test_selected'] = g
    report = dict(status='PASS', audit_utc=datetime.now(timezone.utc).isoformat(),
        audit_code_sha256=digest(__file__), protocol_sha256=digest(out/'protocol.json'),
        groups=40, group_splits=dict(splits), frames=2880, source_identity_hashes='ALL_RGB_AND_NATIVE_DEPTH_VERIFIED',
        truth='Independent closed-camera-volume intersection of authenticated rendered full extents',
        baseline='Replayed existing scalar A scorer on identical saved ranges; no model inference',
        exact_dev_curve_verified=True, selected_threshold=selected['threshold'] if selected else None,
        test_status='ADMITTED_SEALED_RESULTS_RECOUNTED' if test_active else 'UNCONSUMED_NO_TEST_START_OR_LOGITS',
        test_label_file_opened=test_active, summaries=summaries,
        limitation='New procedural groups share renderer, cuboid families and hypothetical sensor law; source admission geometry inspected for all splits, not protected blind evidence.')
    with (out/'independent-audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'summaries'}), flush=True)


def self_test():
    meta = [dict(id=str(i), clip_id='inside' if i < 4 else 'boundary', frame_in_clip=i % 4,
                 time_s=.2*(i % 4), layout_relation='INSIDE' if i < 4 else 'BOUNDARY') for i in range(8)]
    truth = [False, True, True, False]*2
    a = [False, True, True, False]+[False]*4
    logits = [-2., 2., 2., -2., -2., 2., -2., -2.]
    record, g, flags = replay(logits, 2., truth, meta, a)
    assert record['admissible'] and g['Boundary']['B_current']['TP'] == 1
    assert g['Boundary']['B_hold']['TP'] == 2 and g['Boundary']['B_hold']['FP'] == 0
    assert g['Core']['A_hold']['segments'] == 1 and g['Core']['A_hold']['FP'] == 1
    assert flags['B_hold'][6] and not flags['B_hold'][7]
    assert not replay([0.]*8, 1., truth, meta, a)[0]['admissible']
    case = dict(camera=dict(x=0., y=0., z=1.82, pitch=0., yaw=0., roll=0.))
    geo = dict(objects=[dict(name='target', render_bounds_center_m=[2., .4, 1.6], render_bounds_extent_m=[.1, .11, .1])])
    assert rendered_truth(case, geo)
    geo['objects'][0]['render_bounds_extent_m'][1] = .09
    assert not rendered_truth(case, geo)
    print('SYNTHETIC_SELF_TEST_PASS; no experiment files opened or written')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else audit(args.output)
