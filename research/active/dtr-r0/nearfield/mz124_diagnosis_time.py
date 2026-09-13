"""Consumed MZ123 spatial diagnosis and fixed causal alert replay; no inference.

Temporal prediction accepts only episode/time and frozen alerts. Geometry truth is
parsed after temporal predictions are saved. Threshold enumeration is explicitly
posthoc diagnosis and never produces a selected threshold or checkpoint.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend

# Exact MZ120 cell contract, kept NumPy-only to avoid loading model frameworks.
D = (.2, 1.4, 2.6, 3.2, 3.6, 4.2)
S = (-1.5, -.3, .3, 1.5)
H = (.4, 1.5, 2.05, 2.7)
CELLS = np.array([[D[d], S[s], H[h], D[d+1], S[s+1], H[h+1]]
                  for h, s, d in itertools.product(range(3), range(3), range(5))], np.float32)
CORE = np.array([h < 2 and s == 1 and d < 4
                 for h, s, d in itertools.product(range(3), range(3), range(5))])
THRESHOLD = .58


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def readrows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal(rows, alerts, mode):
    """Missing startup votes are false. Reset only at observable episode boundary.

    hold1 uses the previous INPUT, so held outputs never recursively persist.
    No future samples, truth, family, sensor identities, or evaluation metadata.
    """
    assert len(rows) == len(alerts)
    assert mode in ('instant', 'hold1', 'two_of_three')
    out = []; history = []; previous = None; last_time = None; seen = set()
    for row, value in zip(rows, alerts):
        episode = row['episode_id']; stamp = float(row['time_s'])
        if episode != previous:
            assert episode not in seen, 'Episodes must be complete and contiguous'
            seen.add(episode); history = []; last_time = None
        assert last_time is None or stamp > last_time
        current = bool(value)
        if mode == 'instant': answer = current
        elif mode == 'hold1': answer = current or bool(history and history[-1])
        else: answer = sum(history[-2:])+int(current) >= 2
        out.append(answer); history.append(current)
        previous = episode; last_time = stamp
    return np.array(out, bool)


def runs(values):
    start = None
    for i, value in enumerate(list(values)+[False]):
        if value and start is None: start = i
        if not value and start is not None:
            yield start, i
            start = None


def metrics(gt, pred):
    gt = np.asarray(gt, bool); pred = np.asarray(pred, bool)
    tp = int((gt & pred).sum()); fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum()); tn = int((~gt & ~pred).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def evaluate(rows, gt, pred):
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r['episode_id']].append(i)
    events = []; false_segments = []; missed_runs = []
    for episode, idx in groups.items():
        stamps = np.array([rows[i]['time_s'] for i in idx], float)
        assert len(idx) > 1 and np.allclose(np.diff(stamps), .25), 'MZ123 fixed cadence'
        y = gt[idx]; p = pred[idx]
        for a, b in runs(y):
            hits = np.flatnonzero(p[a:b]); hit = len(hits) > 0
            events.append(dict(episode=episode, start_s=float(stamps[a]),
                end_exclusive_s=float(stamps[b-1]+.25), duration_s=(b-a)*.25,
                detected=hit, delay_s=float(stamps[a+hits[0]]-stamps[a]) if hit else None,
                missed_delay_lower_bound_s=None if hit else (b-a)*.25))
        for a, b in runs(~y & p):
            false_segments.append(dict(episode=episode, start_s=float(stamps[a]), frames=b-a, duration_s=(b-a)*.25))
        missed_runs.extend([b-a for a, b in runs(y & ~p)])
    detected = sum(e['detected'] for e in events)
    delays = [e['delay_s'] for e in events if e['detected']]
    return dict(**metrics(gt, pred), positive_events=len(events), detected_events=detected,
        event_recall=detected/len(events) if events else None,
        missed_events=len(events)-detected, maximum_detected_delay_s=max(delays) if delays else None,
        mean_detected_delay_s=float(np.mean(delays)) if delays else None,
        all_events_detected_delay_bound_s=max(delays) if events and detected == len(events) else None,
        false_duration_s=sum(x['duration_s'] for x in false_segments),
        false_segment_count=len(false_segments),
        longest_false_segment_frames=max([x['frames'] for x in false_segments], default=0),
        longest_missed_positive_run_frames=max(missed_runs, default=0),
        events=events, false_segments=false_segments)


def labels(e):
    result = np.zeros(len(CELLS), bool)
    for obj in e['native_bounds']:
        lo = np.array(obj['center_m'])-obj['extent_m']-np.array(e['body_origin_m'])
        hi = np.array(obj['center_m'])+obj['extent_m']-np.array(e['body_origin_m'])
        result |= ((hi >= CELLS[:, :3]) & (lo <= CELLS[:, 3:])).all(1)
    return result


def extrema(values):
    v = [float(x) for x in values if x is not None]
    return dict(n=len(v), minimum=min(v) if v else None,
                median=float(np.median(v)) if v else None, maximum=max(v) if v else None)


def maxmask(p, mask):
    return float(p[mask].max()) if mask.any() else None


def run(source, output):
    started = time.perf_counter()
    assert output.resolve().is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    output.mkdir(parents=True)
    capture = source/'capture-v1'; analysis = source/'analysis-v1'
    seal = read(analysis/'prediction-seal.json')
    assert sha(capture/'raw.jsonl') == seal['raw_sha256']
    assert sha(analysis/'early-probabilities.npy') == seal['early_sha256']
    assert sha(analysis/'baseline-predictions.json') == seal['baseline_sha256']
    receipt = read(capture/'receipt.json')
    # Returned local payload intentionally contains only a subset of RGB images.
    # This replay consumes none; verify every capture input actually consumed.
    for name in ('raw.jsonl', 'evaluator.jsonl'):
        assert sha(capture/name) == receipt['hashes'][name], name
    assert sha(capture/'spec.json') == receipt['spec_sha256']
    rows = readrows(capture/'raw.jsonl')
    p = np.load(analysis/'early-probabilities.npy')
    base_rows = read(analysis/'baseline-predictions.json')
    base = np.array([r['candidate'] for r in base_rows], bool)
    assert p.shape == (len(rows), 45) and len(rows) == 288 and np.isfinite(p).all()
    score = p[:, CORE].max(1)
    observable_rows = [{k:r[k] for k in ('episode_id', 'time_s')} for r in rows]
    arms = {f'{name}_{mode}': causal(observable_rows, alerts, mode)
            for name, alerts in [('baseline', base), ('early', score >= THRESHOLD)]
            for mode in ('instant', 'hold1', 'two_of_three')}
    write(output/'temporal-predictions.json', dict(ids=[r['id'] for r in rows], arms={k:v.tolist() for k,v in arms.items()}))
    write(output/'prediction-seal.json', dict(authority='CONSUMED_DEVELOPMENT_NO_FRESH_CONFIRMATION',
        prediction_sha256=sha(output/'temporal-predictions.json'),
        original_prediction_seal_sha256=sha(analysis/'prediction-seal.json'),
        code_sha256=sha(Path(__file__)), cpu_reason='TASK_NOT_GPU_SUITABLE'))
    # Prediction is now immutable; evaluator-only access starts here.
    es = readrows(capture/'evaluator.jsonl'); spec = read(capture/'spec.json')
    assert [e['id'] for e in es] == [r['id'] for r in rows]
    y = np.stack([labels(e) for e in es]); gt = y[:, CORE].any(1)
    previous = read(analysis/'frame-report.json')
    assert gt.tolist() == [r['truth'] for r in previous]
    assert np.allclose(score, [r['early_score'] for r in previous], rtol=0, atol=0)
    assert base.tolist() == [r['baseline'] for r in previous]
    select_backend('scalar-scoring', cpu=BackendCandidate('numpy-python-cpu','cpu',
        lambda: metrics(gt, arms['early_instant']),
        lambda _: DeviceObservation('cpu', 'host CPU', 'NumPy '+np.__version__)),
        record_path=output/'backend.json', capabilities={'cpu_reason':'TASK_NOT_GPU_SUITABLE', 'gpu_work':False})
    summary = {k:evaluate(rows, gt, v) for k,v in arms.items()}
    family_results = {}
    for family in sorted({e['family'] for e in es}):
        idx = np.array([i for i,e in enumerate(es) if e['family'] == family])
        family_results[family] = {k:evaluate([rows[i] for i in idx], gt[idx], v[idx]) for k,v in arms.items()}
    frames = []
    for i, (r,e) in enumerate(zip(rows,es)):
        maxima = {name:maxmask(p[i], mask) for name,mask in
            [('true_core',y[i]&CORE), ('wrong_core',~y[i]&CORE), ('true_offcore',y[i]&~CORE), ('wrong_offcore',~y[i]&~CORE)]}
        targets = [t for z in r['tof_zones'] for t in z['targets'] if t['status'] in ('SIM_VALID','SIM_MERGED')]
        frames.append(dict(id=r['id'], episode_id=r['episode_id'], time_s=r['time_s'], family=e['family'],
            truth=bool(gt[i]), baseline=bool(base[i]), early=bool(arms['early_instant'][i]), score=float(score[i]),
            **maxima, true_core_cells=np.flatnonzero(y[i]&CORE).tolist(),
            predicted_core_cells=np.flatnonzero((p[i]>=THRESHOLD)&CORE).tolist(),
            top_cell=int(p[i].argmax()), top_cell_true=bool(y[i,p[i].argmax()]),
            valid_tof_targets=len(targets), tof_ranges_m=[t['distance_m'] for t in targets],
            radar_ranges_m=[x for x,v in zip(r['radar_range_m'],r['radar_valid']) if v and x is not None],
            imu_valid=bool(r['imu_valid']), delta_yaw=r['delta_yaw'], delta_pitch=r['delta_pitch']))
    ep_diag = {}
    for ep in dict.fromkeys(r['episode_id'] for r in rows):
        subset = [f for f in frames if f['episode_id']==ep]
        ep_diag[ep] = dict(family=subset[0]['family'], frames=len(subset), positive_frames=sum(f['truth'] for f in subset),
            early_alert_frames=sum(f['early'] for f in subset), baseline_alert_frames=sum(f['baseline'] for f in subset),
            **{k:extrema([f[k] for f in subset]) for k in ('score','true_core','wrong_core','true_offcore','wrong_offcore','valid_tof_targets')})
    pairs = []
    for pair in spec['pairs']:
        a,b = [np.array([i for i,r in enumerate(rows) if r['episode_id']==ep]) for ep in pair['episodes']]
        assert np.all(gt[a] != gt[b])
        changed = y[a] != y[b]; signed = np.where(y[a],1.,-1.)*(p[a]-p[b])
        pairs.append(dict(pair_id=pair['pair_id'], category=pair['category'], episodes=pair['episodes'],
            both_alert_answers_correct={k:int(((v[a]==gt[a])&(v[b]==gt[b])).sum()) for k,v in arms.items()},
            paired_frames=len(a), changed_truth_cells=int(changed.sum()),
            changed_cells_score_moved_correct_direction=int((changed&(signed>0)).sum()),
            changed_cells_signed_score_delta=extrema(signed[changed]),
            matched_positive_minus_negative_score=extrema(np.where(gt[a],1.,-1.)*(score[a]-score[b])),
            full_grid_probability_absolute_delta_mean=float(np.abs(p[a]-p[b]).mean()),
            caveat='Matched geometry/appearance/time; independent sensor noise; not isolated sensor intervention'))
    # Every attainable global-threshold prediction; include all-positive and all-negative.
    curve = []
    for threshold in np.unique(np.r_[0.,score.astype(float),np.nextafter(float(score.max()),np.inf)]):
        prediction = score.astype(float) >= threshold
        m = metrics(gt,prediction)
        curve.append(dict(threshold=float(threshold), **m,
            HEAD_TP=int((prediction & gt & np.array([e['family']=='suspended_head' for e in es])).sum())))
    retention = [x for x in curve if x['TP']>=int((gt&base).sum())]
    fp_budget = [x for x in curve if x['FP']<=int((~gt&base).sum())]
    feasibility = dict(authority='POSTHOC_CONSUMED_DIAGNOSTIC_ONLY_NO_THRESHOLD_SELECTED',
        points=len(curve), thresholds_matching_baseline_TP_and_FP=[x for x in curve if x['TP']>=139 and x['FP']<=116],
        minimum_FP_at_baseline_TP=min(x['FP'] for x in retention),
        maximum_TP_at_baseline_FP=max(x['TP'] for x in fp_budget),
        positive_scores=extrema(score[gt]), negative_scores=extrema(score[~gt]))
    write(output/'frame-diagnosis.json',frames); write(output/'episode-diagnosis.json',ep_diag)
    write(output/'pair-diagnosis.json',pairs); write(output/'threshold-curve.json',curve)
    write(output/'temporal-results.json',dict(overall=summary,families=family_results))
    result = dict(status='CONSUMED_DIAGNOSTIC_COMPLETE',frames=len(rows), episodes=len(ep_diag),
        temporal_arms={k:{q:v for q,v in m.items() if q not in ('events','false_segments')} for k,m in summary.items()},
        threshold_diagnosis=feasibility, cpu_reason='TASK_NOT_GPU_SUITABLE', elapsed_seconds=time.perf_counter()-started,
        source=source.as_posix(), input_hashes={str(path.relative_to(source)):sha(path) for path in
            [capture/'raw.jsonl',capture/'evaluator.jsonl',capture/'spec.json',analysis/'early-probabilities.npy',analysis/'baseline-predictions.json']},
        imu_grid_arm='NOT_TESTED: grid projections already use accumulated yaw in fixed episode axes; IMU supplies no translation, and 45-cell extents cannot uniquely reproject unknown within-cell occupancy',
        limitations=['Consumed constructed simulation only; no new independent validation',
            'False duration counts 0.25 seconds per sampled false frame including last frame; not actual notification count',
            'Temporal hold is previous alert persistence, not verified same-object tracking',
            'Majority startup uses missing votes as false; missed delays remain null with duration lower bound',
            'No learned temporal network or IMU-only metric-translation claim'])
    write(output/'summary.json',result)
    write(output/'completion.json',dict(status='PASS',summary_sha256=sha(output/'summary.json'), resources='No persistent processes or accelerator allocations'))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); run(args.source,args.output)
