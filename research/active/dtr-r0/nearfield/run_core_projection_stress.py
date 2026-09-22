"""One frozen projection-error check; public prediction precedes evaluator join."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tof_corridor_calibration import score_frame, decide
from tof_fov45_core import boxes45

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
OUT = ROOT / 'artifacts.local/work/ba-core-projection-stress-20260921'
NAME = 'ba-core-projection-stress-20260921'
PROTOCOL = 'CORE_PROJECTION_STRESS_PROTOCOL_20260921.md'
OFFSETS = {'left': -2, 'nominal': 0, 'right': 2}
POLICIES = ('calibration', 'strong', 'hold')
ARMS = tuple(f'{condition}_{policy}' for condition in OFFSETS for policy in POLICIES)
T0, T, DT = .007085703945147101, .4071309640537889, .2
CODE = ('run_core_projection_stress.py', 'tof_corridor_calibration.py',
        'tof_fov45_core.py', 'ba_camera_corridor.py', 'ba_camera_corridor_metrics.py',
        'full_event_metrics_20260920.py', 'test_core_projection_stress.py')
INPUTS = ('observations.npz', 'identities.json', 'baseline.json', 'predictions.json',
          'observation-seal.json', 'prediction-seal.json',
          'evaluator/metadata.json', 'evaluator/transfer-labels.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def seal(out, name, files):
    write(out / name, {'protocol_sha256': sha(out / 'protocol.json'),
                      'hashes': {n: sha(out / n) for n in files}})


def check_seal(out, name):
    receipt = read(out / name)
    assert receipt['protocol_sha256'] == sha(out / 'protocol.json')
    for file, digest in receipt['hashes'].items():
        assert sha(out / file) == digest, file


def shifted_boxes(boxes, dx):
    boxes = np.asarray(boxes)
    if boxes.shape != (64, 4) or boxes.dtype.kind not in 'iu' or type(dx) is not int:
        raise ValueError('Expected 64 integer boxes and an integer projection shift')
    result = boxes.copy()
    result[:, [1, 3]] += dx
    if not (np.all(result[:, 0] >= 0) and np.all(result[:, 2] <= 192)
            and np.all(result[:, 1] >= 0) and np.all(result[:, 3] <= 256)
            and np.all(result[:, 2] > result[:, 0]) and np.all(result[:, 3] > result[:, 1])):
        raise ValueError('Shifted projection outside image or empty; no clipping allowed')
    return result


def readout(scored, previous, time_s):
    calibration, strong = decide(scored, T0), decide(scored, T)
    held = bool(previous is not None and abs(time_s - previous[0] - DT) < 1e-8 and previous[1])
    flags = dict(calibration=calibration['alert'], strong=strong['alert'],
                 hold=bool(strong['alert'] or held))
    return flags, bool(strong['unknown']), (time_s, strong['alert'])


def freeze(out):
    assert not out.exists(), 'New output only; no rerun/overwrite'
    observation, prediction = read(SOURCE / 'observation-seal.json'), read(SOURCE / 'prediction-seal.json')
    for name in INPUTS:
        if name in observation['hashes']:
            assert sha(SOURCE / name) == observation['hashes'][name], name
        if name in prediction['hashes']:
            assert sha(SOURCE / name) == prediction['hashes'][name], name
    assert observation['protocol_sha256'] == prediction['protocol_sha256'] == sha(SOURCE / 'protocol.json')
    out.mkdir(parents=True)
    (out / 'protocol-before-run.md').write_bytes((HERE / PROTOCOL).read_bytes())
    write(out / 'protocol.json', dict(id=NAME, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source=SOURCE.relative_to(ROOT).as_posix(), frames=1152, clips=48, dt_s=DT,
        offsets=OFFSETS, thresholds=dict(calibration=T0, strong=T),
        code_hashes={n: sha(HERE / n) for n in CODE},
        input_hashes={n: sha(SOURCE / n) for n in INPUTS},
        protocol_text_sha256=sha(out / 'protocol-before-run.md'),
        scope='CONSUMED_SIMULATION_PROJECTION_ERROR_SENSITIVITY',
        interpretation='Reported box offset only; no sensor movement, new returns, RGB model, training or hardware claim',
        acceptance=dict(core_events=16, onset_delay_s=.2, coverage=5/6,
                        max_silent_samples=1, fp_reduction_fraction=.5),
        original_test_activated=False, automatic_successor=False))
    print('FROZEN', NAME, flush=True)


def verify(out):
    p = read(out / 'protocol.json')
    assert p['offsets'] == OFFSETS and p['thresholds'] == dict(calibration=T0, strong=T)
    assert sha(out / 'protocol-before-run.md') == p['protocol_text_sha256']
    for name, digest in p['code_hashes'].items():
        assert sha(HERE / name) == digest, name
    for name, digest in p['input_hashes'].items():
        assert sha(SOURCE / name) == digest, name
    return p


def predict(out):
    verify(out)
    assert not (out / 'prediction-start.json').exists(), 'One prediction pass'
    write(out / 'prediction-start.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        backend='CPU_SCALAR_GEOMETRY', placement='TASK_NOT_GPU_SUITABLE',
        executable=sys.executable, evaluator_values_read=False, model_calls=0,
        native_depth_read=False, raw_ranges_changed=False))
    ids, baseline, parent = (read(SOURCE / name) for name in ('identities.json', 'baseline.json', 'predictions.json'))
    with np.load(SOURCE / 'observations.npz', allow_pickle=False) as data:
        ranges, boxes = data['ranges'].copy(), data['boxes'].copy()
    assert ranges.shape == (1152, 64) and np.array_equal(boxes, boxes45())
    assert len(ids) == len(baseline) == len(parent) == 1152
    range_digest = hashlib.sha256(ranges.tobytes()).hexdigest()
    projections = {key: shifted_boxes(boxes, dx) for key, dx in OFFSETS.items()}
    states, rows, timings = {}, [], []
    for i, identity in enumerate(ids):
        assert identity['index'] == baseline[i]['index'] == parent[i]['index'] == i
        assert identity['id'] == parent[i]['id']
        row = {k: identity[k] for k in ('id', 'index', 'clip_id', 'frame_in_clip', 'time_s')}
        row.update(flags={}, current_unknown={}, support={})
        for condition, projection in projections.items():
            start = time.perf_counter()
            scored = score_frame(projection, ranges[i])
            key = (condition, identity['clip_id'])
            flags, unknown, states[key] = readout(scored, states.get(key), identity['time_s'])
            timings.append(time.perf_counter() - start)
            row['flags'].update({condition + '_' + policy: flag for policy, flag in flags.items()})
            row['current_unknown'].update({condition + '_' + policy: unknown for policy in POLICIES})
            raw = scored['baseline']
            row['support'][condition] = dict(score=scored['score'], valid_zones=raw['valid_zones'],
                definite_zones=raw['definite_zones'], possible_zones=raw['possible_zones'],
                raw_alert=raw['alert'], unknown=unknown)
            if condition == 'nominal':
                b, old = baseline[i], parent[i]
                assert scored['score'] == b['score'], ('Nominal score mismatch', i)
                assert flags['strong'] == b['current'] == old['flags']['A_current']
                assert flags['hold'] == old['flags']['A_hold']
                assert unknown == b['unknown'] == old['current_unknown']['A_hold']
                assert raw['valid_zones'] == b['valid_zones'] and raw['definite_zones'] == b['definite_zones']
        rows.append(row)
        if (i + 1) % 384 == 0:
            print('PREDICT', i + 1, '/1152', flush=True)
    assert hashlib.sha256(ranges.tobytes()).hexdigest() == range_digest
    write(out / 'reported-projections.json', {k: v.tolist() for k, v in projections.items()})
    write(out / 'predictions.json', rows)
    write(out / 'prediction-receipt.json', dict(nominal_parity_frames=len(rows), range_bytes_sha256=range_digest,
        range_bytes_unchanged=True, public_source_sha256=sha(SOURCE / 'observations.npz'),
        host_score_three_readouts_ms=dict(mean=1000*float(np.mean(timings)), p95=1000*float(np.quantile(timings, .95))),
        timing_scope='One condition geometry and three decisions; excludes sensing/RGB/IO, not target-device latency'))
    seal(out, 'prediction-seal.json', ['predictions.json', 'prediction-start.json',
        'reported-projections.json', 'prediction-receipt.json'])
    verify(out)
    print('PREDICTIONS_SEALED', len(rows), flush=True)


def metric_module():
    spec = importlib.util.spec_from_file_location('projection_event_metrics', HERE / 'full_event_metrics_20260920.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARMS = ARMS
    return module


def max_silence(rows, arm):
    longest = run = 0
    for row in rows:
        run = run + 1 if row['truth'] and not row['flags'][arm] else 0
        longest = max(longest, run)
    return longest


def checks_for(rows, report, condition, limits):
    a, c = condition + '_calibration', condition + '_hold'
    baseline, candidate = (report['Core']['arms'][arm] for arm in (a, c))
    events = candidate['events']
    checks = dict(all_core_events=len(events) == limits['core_events'] and candidate['detected_events'] == len(events),
        onset_within_budget=all(e['first_in_event_alert_delay_s'] is not None and e['first_in_event_alert_delay_s'] <= limits['onset_delay_s'] + 1e-8 for e in events),
        coverage=all(e['positive_coverage'] >= limits['coverage'] for e in events),
        fp_reduction=baseline['frames']['FP'] > 0 and candidate['frames']['FP'] <= (1-limits['fp_reduction_fraction'])*baseline['frames']['FP'],
        core_false_segments=candidate['false_alert_segment_count'] <= baseline['false_alert_segment_count'])
    clips = defaultdict(list)
    for row in rows:
        if row['layout_relation'] != 'BOUNDARY':
            clips[row['clip_id']].append(row)
    checks['max_silence'] = all(max_silence(clip, c) <= limits['max_silent_samples'] for clip in clips.values())
    for relation in ('INSIDE', 'OUTSIDE'):
        group = report['subgroups']['layout_relation'][relation]['arms']
        for metric in ('FP', 'false_alert_segment_count'):
            value = lambda v: v['frames']['FP'] if metric == 'FP' else v[metric]
            checks[relation + '_' + metric] = value(group[c]) <= value(group[a])
    return checks


def evaluate(out):
    p = verify(out)
    check_seal(out, 'prediction-seal.json')
    assert not (out / 'evaluation-start.json').exists(), 'One evaluator join'
    write(out / 'evaluation-start.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        prediction_seal_sha256=sha(out / 'prediction-seal.json')))
    predictions, meta, truth = read(out / 'predictions.json'), read(SOURCE / 'evaluator/metadata.json'), read(SOURCE / 'evaluator/transfer-labels.json')
    assert len(predictions) == len(meta) == len(truth) == 1152
    rows = []
    for pred, m, y in zip(predictions, meta, truth):
        assert pred['index'] == m['index'] == y['index'] and pred['id'] == m['id']
        assert all(pred[k] == m[k] for k in ('clip_id', 'frame_in_clip', 'time_s'))
        rows.append({**m, 'truth': y['truth'], 'flags': pred['flags'],
                     'current_unknown': pred['current_unknown'], 'support': pred['support']})
    metrics = metric_module().evaluate(rows, DT)
    checks = {c: checks_for(rows, metrics, c, p['acceptance']) for c in OFFSETS}
    summary, changes = {}, {}
    for stratum in ('Core', 'Boundary'):
        group = [r for r in rows if (r['layout_relation'] != 'BOUNDARY') == (stratum == 'Core')]
        summary[stratum] = {}
        for arm, m in metrics[stratum]['arms'].items():
            events = m['events']
            summary[stratum][arm] = dict(**{k: m['frames'][k] for k in ('TP', 'FP', 'FN', 'precision', 'recall', 'FPR', 'current_unknown')},
                events=m['detected_events'], event_count=m['event_count'],
                false_segments=m['false_alert_segment_count'], false_sampled_s=m['false_alert_sampled_s'],
                max_detected_onset_delay_s=max((e['first_in_event_alert_delay_s'] for e in events if e['detected']), default=None),
                min_event_coverage=min((e['positive_coverage'] for e in events), default=None),
                preentry_FP=sum(e['preentry_false_alert_frames'] for e in events),
                postexit_FP=sum(e['postexit_false_alert_frames'] for e in events),
                false_definite_frames=sum(not r['truth'] and r['support'][arm.rsplit('_', 1)[0]]['definite_zones'] > 0 for r in group))
        changes[stratum] = {}
        for condition in ('left', 'right'):
            c, n = condition + '_hold', 'nominal_hold'
            changes[stratum][condition] = {
                kind: [r['id'] for r in group if predicate(r)] for kind, predicate in (
                    ('lost_nominal_TP', lambda r: r['truth'] and r['flags'][n] and not r['flags'][c]),
                    ('added_TP', lambda r: r['truth'] and not r['flags'][n] and r['flags'][c]),
                    ('added_FP', lambda r: not r['truth'] and not r['flags'][n] and r['flags'][c]),
                    ('removed_FP', lambda r: not r['truth'] and r['flags'][n] and not r['flags'][c]),
                    ('unknown_changed', lambda r: r['current_unknown'][n] != r['current_unknown'][c]))}
    result = dict(id=NAME, status='COMPLETE', scope=p['scope'], summary=summary, checks=checks,
        passed_by_condition={c: all(v.values()) for c, v in checks.items()},
        both_signed_conditions_pass=all(all(checks[c].values()) for c in ('left', 'right')),
        inheritance_role='COMPONENT_OR_CHALLENGER', inheritance_mode='COMPONENT',
        original_test_activated=False, automatic_successor=False)
    write(out / 'frame-results.json', rows)
    write(out / 'metrics.json', metrics)
    write(out / 'changed-ids.json', changes)
    write(out / 'result.json', result)
    seal(out, 'evaluation-seal.json', ['frame-results.json', 'metrics.json', 'changed-ids.json', 'result.json', 'evaluation-start.json'])
    verify(out)
    print(json.dumps(dict(passed=result['passed_by_condition'], Core={c:summary['Core'][c+'_hold'] for c in OFFSETS})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('freeze', 'predict', 'evaluate'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.stage](args.out)
