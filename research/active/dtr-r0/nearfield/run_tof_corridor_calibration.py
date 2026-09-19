"""One fixed score and one in-sample cutoff on sealed nominal-FOV Development96."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from ba_camera_corridor import rays, sample_native
from ba_camera_corridor_metrics import evaluate_rows
from evaluate_ba_camera_corridor import read, sha, require, write_new
from tof_corridor_calibration import score_frame, calibrate, decide

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'artifacts.local/work/ba-camera-corridor-20260919'
SOURCE = ROOT / 'artifacts.local/work/ba-tof-fov45-20260920'
OUT = ROOT / 'artifacts.local/work/ba-tof-corridor-calibration-20260920'
REPORT = Path(__file__).with_name('TOF_CORRIDOR_CALIBRATION_20260920.md')
CODE = ('tof_corridor_calibration.py', 'run_tof_corridor_calibration.py',
        'test_tof_corridor_calibration.py', 'ba_camera_corridor.py',
        'ba_camera_corridor_metrics.py', 'evaluate_ba_camera_corridor.py')
IDENTITY = ('id', 'clip_id', 'frame_in_clip', 'time_s')


def freeze(out):
    require(not out.exists(), 'Run path already exists; no overwrite or automatic retry')
    # The prior run's code and source identities must still reproduce its scope.
    from run_tof_fov45 import verify as verify_parent
    verify_parent(SOURCE)
    inputs = [SOURCE / name for name in ('protocol.json', 'observation-seal.json',
        'observations.json', 'prediction-seal.json', 'predictions.json',
        'results.json', 'frame-results.json', 'private-lineage.json')]
    inputs += [OLD / name for name in ('frame-results.json', 'evaluator-source.json')]
    protocol = dict(id='ba-tof-corridor-calibration-20260920',
        phase='EXPLORE_ALL96_CONSUMED_DEVELOPMENT_IN_SAMPLE',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        question='Can one support-overlap score reduce FP while preserving every original30TP and event timing?',
        hypothesis='Boolean possible overlap over-alerts on small depth/angular support intersections.',
        budget=dict(frames=96, clips=8, fixed_scores=1, cutoff_calibrations=1,
                    new_observations=0, model_calls=0, training_updates=0, rgb_calls=0),
        score='max zone exact integral of angular corridor fraction over unchanged full axial-depth interval; uniform support measure, not probability',
        cutoff='largest inclusive threshold preserving each baseline positive alert; definite supports always alert',
        stop='One score/cutoff only; no post-outcome weights, alternative scores, RGB, training, capture or successor.',
        gates=['same30TP IDs', 'lower19FP', 'retain4of5 interior and5of6 all events',
               'no later in-event first alerts', 'no increased false segments/duration',
               'all original anchors and intervals retained', 'all definite alerts retained',
               'no withheld native corridor contributors', 'UNKNOWN never relabelled clear'],
        timing='In-event onset gate; first-within-clip and removed pre-entry FP separately reported.',
        limits='Uncalibrated single-return optical-Z simulation; no hardware likelihood, holdout, generalization, runtime or safety claim.',
        code_hashes={name: sha(Path(__file__).with_name(name)) for name in CODE},
        input_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        protocol_text_sha256=sha(REPORT), synthetic_tests_passed=6)
    out.mkdir(parents=True)
    (out / 'protocol-before-run.md').write_bytes(REPORT.read_bytes())
    (out / 'current-before-run.md').write_bytes((ROOT / 'research/active/dtr-r0/CURRENT.md').read_bytes())
    write_new(out / 'protocol.json', protocol)
    print(json.dumps(dict(status='FROZEN', protocol_sha256=sha(out / 'protocol.json'))))


def verify(out):
    protocol = read(out / 'protocol.json')
    for name, digest in protocol['code_hashes'].items():
        actual = sha(Path(__file__).with_name(name))
        if actual != digest:
            repair = read(out / 'evaluator-repair.json')
            require(name == 'run_tof_corridor_calibration.py'
                    and repair['original_sha256'] == digest
                    and repair['repaired_sha256'] == actual
                    and sha(out / 'runner-before-evaluator-repair.py') == digest
                    and repair['scores_sha256'] == sha(out / 'scores.json')
                    and repair['predictions_sha256'] == sha(out / 'predictions.json')
                    and repair['protocol_sha256'] == sha(out / 'protocol.json')
                    and repair['reason'] == 'PARENT_EVENT_SCHEMA_ADDS_BOUNDARY_DETECTED',
                    'Unrecorded code change: ' + name)
    for name, digest in protocol['input_hashes'].items():
        require(sha(ROOT / name) == digest, 'Input changed: ' + name)
    require(sha(out / 'protocol-before-run.md') == protocol['protocol_text_sha256'], 'Protocol text changed')
    return protocol


def score(out):
    verify(out)
    require(not (out / 'scores.json').exists(), 'Score pass already exists')
    os, ps = read(SOURCE / 'observation-seal.json'), read(SOURCE / 'prediction-seal.json')
    require(os['status'] == ps['status'] == 'COMPLETE' and os['frames'] == ps['frames'] == 96, 'Parent incomplete')
    require(os['protocol_sha256'] == ps['protocol_sha256'] == sha(SOURCE / 'protocol.json'), 'Parent protocol differs')
    require(os['observations_sha256'] == sha(SOURCE / 'observations.json'), 'Parent observation index differs')
    require(ps['observation_seal_sha256'] == sha(SOURCE / 'observation-seal.json'), 'Parent observation seal differs')
    require(ps['predictions_sha256'] == sha(SOURCE / 'predictions.json'), 'Parent predictions differ')
    observations, parent = read(SOURCE / 'observations.json'), read(SOURCE / 'predictions.json')
    require(len(observations) == len(parent) == 96, 'Cohort count differs')
    sys.path.insert(0, str(ROOT / 'tools'))
    from research_backend import BackendCandidate, DeviceObservation, select_backend
    select_backend('scalar-scoring', cpu=BackendCandidate('numpy-cpu', 'cpu', lambda: np.sum(np.arange(8)),
        lambda _: DeviceObservation('cpu', platform.processor(), 'NumPy', ('CPU',))),
        cpu_reason='TASK_NOT_GPU_SUITABLE', capabilities={'scope': '64 scalar geometry supports'},
        record_path=out / 'backend.json')
    started = time.perf_counter()
    rows, times = [], []
    for obs, previous in zip(observations, parent):
        require(all(obs[k] == previous[k] for k in IDENTITY), 'Parent identity differs')
        path = SOURCE / obs['path']
        require(sha(path) == obs['sha256'] == previous['observation_sha256'], 'Observation changed')
        with np.load(path, allow_pickle=False) as data:
            began = time.perf_counter()
            result = score_frame(data['boxes'], data['values'])
            times.append(time.perf_counter() - began)
        require(result['baseline'] == previous['decision'] and result['anchors'] == previous['anchors'],
                'Exact baseline decision/anchor replay failed')
        rows.append({**{k: obs[k] for k in IDENTITY}, 'observation_sha256': obs['sha256'], **result})
    write_new(out / 'scores.json', rows)
    write_new(out / 'score-seal.json', dict(status='COMPLETE', frames=96,
        protocol_sha256=sha(out / 'protocol.json'), scores_sha256=sha(out / 'scores.json'),
        observation_seal_sha256=sha(SOURCE / 'observation-seal.json'),
        exact_baseline_replay=True, elapsed_s=time.perf_counter() - started,
        scalar_compute_ms=dict(mean=float(np.mean(times) * 1000), p95=float(np.percentile(times, 95) * 1000)),
        scope='One desktop pass including baseline and score; excludes input IO and sensor acquisition, not target-device latency'))
    print(json.dumps(dict(status='SCORES_SEALED', frames=96)))


def load_scores(out):
    verify(out)
    seal = read(out / 'score-seal.json')
    require(seal['status'] == 'COMPLETE' and seal['frames'] == 96, 'Scores incomplete')
    require(seal['protocol_sha256'] == sha(out / 'protocol.json'), 'Score protocol differs')
    require(seal['scores_sha256'] == sha(out / 'scores.json'), 'Scores changed')
    require(seal['observation_seal_sha256'] == sha(SOURCE / 'observation-seal.json'), 'Score input differs')
    return read(out / 'scores.json')


def select(out):
    scored = load_scores(out)
    # Labels enter only this explicit in-sample calibration stage. Scoring has
    # no clip/target/native truth features; identities merely bind saved rows.
    labels = read(OLD / 'frame-results.json')
    require(len(labels) == len(scored) == 96, 'Calibration cohort differs')
    require(all(all(a[k] == b[k] for k in IDENTITY) for a, b in zip(scored, labels)), 'Calibration identities differ')
    point = calibrate(scored, [row['truth'] for row in labels])
    require(point['required_tp'] == 30, 'Baseline30 TP differs')
    predictions = [{**{k: s[k] for k in IDENTITY}, 'observation_sha256': s['observation_sha256'],
                    'baseline': s['baseline'], 'candidate': decide(s, point['threshold']),
                    'anchors': s['anchors']} for s in scored]
    write_new(out / 'operating-point.json', point)
    write_new(out / 'predictions.json', predictions)
    write_new(out / 'prediction-seal.json', dict(status='COMPLETE', frames=96,
        protocol_sha256=sha(out / 'protocol.json'), score_seal_sha256=sha(out / 'score-seal.json'),
        label_sha256=sha(OLD / 'frame-results.json'), point_sha256=sha(out / 'operating-point.json'),
        predictions_sha256=sha(out / 'predictions.json'), scope=point['scope']))
    print(json.dumps(dict(status='CUTOFF_SELECTED_PREDICTIONS_SEALED', **point)))


def fp_type(row):
    """Evaluator-only decomposition. Every subtype remains a strict FP."""
    penetration = row['axis_penetration_m']
    separated = [name for name, p in zip(('lateral', 'vertical', 'depth'), penetration) if p < -1e-9]
    return '+'.join(separated) or 'unresolved'


def evaluate(out):
    started = time.perf_counter()
    scored = load_scores(out)
    seal = read(out / 'prediction-seal.json')
    require(seal['status'] == 'COMPLETE' and seal['frames'] == 96, 'Predictions incomplete')
    for key, path in (('protocol_sha256', out / 'protocol.json'),
                      ('score_seal_sha256', out / 'score-seal.json'),
                      ('label_sha256', OLD / 'frame-results.json'),
                      ('point_sha256', out / 'operating-point.json'),
                      ('predictions_sha256', out / 'predictions.json')):
        require(seal[key] == sha(path), 'Prediction seal binding differs: ' + key)
    preds, old = read(out / 'predictions.json'), read(OLD / 'frame-results.json')
    point = read(out / 'operating-point.json')
    rows = []
    for s, p, label in zip(scored, preds, old):
        require(all(s[k] == p[k] == label[k] for k in IDENTITY), 'Evaluation identity differs')
        require(p['candidate'] == decide(s, point['threshold']) and p['baseline'] == s['baseline'], 'Decision differs')
        require(p['anchors'] == s['anchors'], 'Support was removed or changed')
        rows.append({**label, 'predictions': {arm: p[arm] for arm in ('baseline', 'candidate')},
                     'fp_subtype': fp_type(label) if label['truth'] is False else None})
    require(len(rows) == len(preds) == 96, 'Evaluation budget changed')
    metrics = evaluate_rows(rows, dt_s=.2, arms=('baseline', 'candidate'))
    base, candidate = metrics['arms']['baseline'], metrics['arms']['candidate']
    parent = read(SOURCE / 'results.json')['candidate']
    expected = {k: parent[k] for k in base}
    # The parent's summarize wrapper adds boundary_detected to each event.
    # Compare every core evaluator field; preserve the parent report untouched.
    require(len(parent['events']) == len(base['events']), 'Parent event count differs')
    expected['events'] = [{k: previous[k] for k in event}
                          for previous, event in zip(parent['events'], base['events'])]
    require(base == expected, 'Parent metric replay differs')
    paired = {}
    for name, condition in (
            ('lost_tp', lambda r: r['truth'] and r['predictions']['baseline']['alert'] and not r['predictions']['candidate']['alert']),
            ('rescued_fn', lambda r: r['truth'] and not r['predictions']['baseline']['alert'] and r['predictions']['candidate']['alert']),
            ('removed_fp', lambda r: not r['truth'] and r['predictions']['baseline']['alert'] and not r['predictions']['candidate']['alert']),
            ('added_fp', lambda r: not r['truth'] and not r['predictions']['baseline']['alert'] and r['predictions']['candidate']['alert'])):
        paired[name] = [r['id'] for r in rows if condition(r)]
    events = []
    for before, after in zip(base['events'], candidate['events']):
        require(all(before[k] == after[k] for k in ('clip_id', 'start_frame', 'end_frame', 'interior_frames')), 'Events changed')
        events.append(dict(baseline=before, candidate=after,
            delayed=before['detected'] and (not after['detected'] or after['first_alert_time_s'] > before['first_alert_time_s'])))
    clip_first = {}
    for clip in sorted({r['clip_id'] for r in rows}):
        group = [r for r in rows if r['clip_id'] == clip]
        clip_first[clip] = {arm: next((dict(time_s=r['time_s'], truth=r['truth'], left_censored=r['frame_in_clip'] == 0)
                           for r in group if r['predictions'][arm]['alert']), None) for arm in ('baseline', 'candidate')}
    types = {arm: dict(Counter(r['fp_subtype'] for r in rows if r['truth'] is False and r['predictions'][arm]['alert']))
             for arm in ('baseline', 'candidate')}
    # Native contributor audit is downstream of sealed predictions. It never
    # changes a score/cutoff. Every observed winning-bin index is inherited.
    lineage = read(SOURCE / 'private-lineage.json')
    require(sha(SOURCE / 'private-lineage.json') == read(SOURCE / 'observation-seal.json')['private_lineage_sha256'], 'Native lineage differs')
    sources = read(OLD / 'evaluator-source.json')
    a, b = rays()
    audit = []
    for r, p, lin, source in zip(rows, preds, lineage, sources):
        require(r['id'] == lin['id'] == source['id'], 'Native audit identity differs')
        path = OLD / 'capture/evaluator' / source['geometry']['native_path']
        require(sha(path) == source['sensor_native_sha256'], 'Native depth differs')
        depth = sample_native(np.load(path, allow_pickle=False))
        inside = np.isfinite(depth) & (depth >= .3) & (depth <= 3.) & (abs(a * depth) <= .3) & (b * depth >= -.2) & (b * depth <= .9)
        anchors = {v['zone']: v for v in p['anchors']}
        observed = [t for t in lin['traces'] if t['observed']]
        require(set(anchors) == {t['zone_id'] for t in observed}, 'Observed anchors dropped')
        relevant = [t for t in observed if anchors[t['zone_id']]['possible']]
        contributors = sum(int(inside.ravel()[np.asarray(t['pixel_indices'], int)].sum()) for t in relevant)
        withheld = p['baseline']['alert'] and not p['candidate']['alert']
        audit.append(dict(id=r['id'], observed_zones=len(observed), observed_lt3m_zones=sum(t['distance_m'] < 3 for t in observed),
                          possible_native_corridor_contributors=contributors,
                          withheld=withheld, withheld_corridor_contributors=contributors if withheld else 0))
    counts = candidate['frames']['all_known']
    native_lost = sum(r['withheld_corridor_contributors'] for r in audit)
    checks = dict(same_tp_ids=not paired['lost_tp'] and counts['TP'] == 30,
        fewer_fp=counts['FP'] < 19, no_new_fp=not paired['added_fp'],
        events_retained=candidate['interior_detected_events'] == 4 and candidate['detected_events'] == 5,
        event_first_not_delayed=not any(e['delayed'] for e in events),
        false_segments_not_increased=candidate['false_alert_segment_count'] <= base['false_alert_segment_count'],
        false_duration_not_increased=candidate['false_alert_sampled_duration_s'] <= base['false_alert_sampled_duration_s'],
        anchors_exact=True, definite_alerts_retained=all(not p['baseline']['definite_zones'] or p['candidate']['alert'] for p in preds),
        no_withheld_native_corridor_contributors=native_lost == 0,
        unknown_retained=all(p['candidate']['unknown'] == p['baseline']['unknown'] for p in preds),
        no_free_space_claim=counts['TN'] == 0)
    result = dict(status='COMPLETE', frames=96, operating_point=point, metrics=metrics,
        exact_parent_metric_parity=True, paired=paired, events=events, first_within_clip=clip_first,
        fp_subtypes=types, native_audit=dict(observed_zones=sum(a['observed_zones'] for a in audit),
        observed_lt3m_zones=sum(a['observed_lt3m_zones'] for a in audit), withheld_corridor_contributors=native_lost,
        scope='Exact inherited winning-bin sampled points; not full unsampled surfaces or physical sensor support'),
        gate=dict(passed=all(checks.values()), checks=checks),
        decision='RETAIN_IN_SAMPLE_READOUT_COMPONENT' if all(checks.values()) else 'FIXED_SCORE_JOINT_TARGET_NOT_MET_STOP',
        by_clip={clip: evaluate_rows([r for r in rows if r['clip_id'] == clip], .2, ('baseline', 'candidate'))['arms']
                 for clip in sorted({r['clip_id'] for r in rows})},
        by_pair={pair: evaluate_rows([r for r in rows if r['pair_id'] == pair], .2, ('baseline', 'candidate'))['arms']
                 for pair in sorted({r['pair_id'] for r in rows})},
        score_timing=read(out / 'score-seal.json')['scalar_compute_ms'],
        interpretation='In-sample optimum for this single monotone score only. No split, holdout, independent confirmation or hardware confidence.',
        elapsed_s=time.perf_counter() - started)
    write_new(out / 'frame-results.json', rows)
    write_new(out / 'native-support-audit.json', audit)
    result['hashes'] = {name: sha(out / name) for name in ('protocol.json', 'score-seal.json', 'scores.json',
        'operating-point.json', 'prediction-seal.json', 'predictions.json', 'frame-results.json', 'native-support-audit.json')}
    write_new(out / 'results.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'decision', 'paired', 'fp_subtypes', 'gate', 'native_audit', 'score_timing')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=('freeze', 'score', 'select', 'evaluate'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.phase](args.out)
