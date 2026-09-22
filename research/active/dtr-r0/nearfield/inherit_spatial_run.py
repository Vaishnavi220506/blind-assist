"""One governed consumed-Development fit/selection/seal/evaluation, no sweep."""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from inherit_spatial_model import ARMS, CENTRE, FEATURE_SIZE, PARAMS, QUERIES, SEED, extract
from query_occupancy_data import read, write, sha, new_stage_directory
from tof_corridor_calibration import score_frame, decide
from ba_camera_corridor_metrics import evaluate_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from tools.research_backend import BackendCandidate, DeviceObservation, select_backend

REPORT_ARMS = ('A_current', 'A_hold', 'raw_standalone', 'local_standalone', 'raw', 'local')


def frame_truth(labels):
    return (labels['classes'][:, CENTRE] < 6).any(1), labels['valid'][:, CENTRE].all(1)


def counts(flags, truth, valid, identities):
    p, y, v = np.asarray(flags, bool), np.asarray(truth, bool), np.asarray(valid, bool)
    false = p & ~y & v
    previous, segments = {}, 0
    for i, row in enumerate(identities):
        segments += int(false[i] and not previous.get(row['clip_id'], False))
        previous[row['clip_id']] = bool(false[i])
    return dict(TP=int((p & y & v).sum()), FP=int(false.sum()),
                FN=int((~p & y & v).sum()), false_segments=segments)


def select_threshold(score, labels, identities):
    y, valid = frame_truth(labels)
    if not (y & valid).any() or not (~y & valid).any():
        raise ValueError('NOT_EVALUABLE: dev needs known positives and negatives')
    baseline = np.array([r['baseline']['alert'] for r in identities], bool)
    b = counts(baseline, y, valid, identities)
    thresholds = np.r_[np.nextafter(float(max(score)), np.inf), np.unique(score)[::-1]]
    curve, best = [], None
    for cutoff in thresholds:
        metrics = counts(baseline | (score >= cutoff), y, valid, identities)
        record = dict(threshold=float(cutoff), **metrics,
                      added_TP=metrics['TP']-b['TP'], added_FP=metrics['FP']-b['FP'],
                      added_false_segments=metrics['false_segments']-b['false_segments'])
        record['admissible'] = record['added_FP'] <= 1 and record['added_false_segments'] <= 1
        curve.append(record)
        key = (record['added_TP'], -record['false_segments'], -record['FP'], float(cutoff))
        if record['admissible'] and (best is None or key > best[0]):
            best = key, record
    return dict(selection=best[1], baseline=b, curve=curve,
                rule='MAX_ADDED_TP_FP_LE_1_SEGMENTS_LE_1_THEN_SEGMENTS_FP_LARGER_CUTOFF')


def metric_rows(metas, labels, probabilities, selection):
    y, valid = frame_truth(labels)
    rows, previous = [], {}
    for k, meta in enumerate(metas):
        base = meta['baseline']
        flags = dict(A_current=base['alert'],
                     A_hold=base['alert'] or previous.get(meta['clip_id'], False))
        previous[meta['clip_id']] = base['alert']
        for arm in ARMS:
            positive = float(probabilities[arm][k, CENTRE].max()) >= selection[arm]['selection']['threshold']
            flags[arm+'_standalone'] = positive
            flags[arm] = bool(base['alert'] or positive)
        rows.append(dict(id=meta['id'], clip_id=meta['clip_id'],
            frame_in_clip=meta['frame_in_clip'], time_s=meta['time_s'],
            base_group_id=meta['base_group_id'], type_id=meta['type_id'],
            layer=meta['layer'], layout_relation=meta['layout_relation'],
            truth=bool(y[k]) if valid[k] else None, boundary=meta['layout_relation']=='BOUNDARY',
            predictions={a:dict(alert=bool(flag), unknown=bool(base['unknown']),
                                 ambiguous=bool(flag and base['unknown'])) for a, flag in flags.items()}))
    return rows


def event_differences(metrics, arm, reference):
    base = {(e['clip_id'], e['start_frame']): e for e in metrics['arms'][reference]['events']}
    result = []
    for event in metrics['arms'][arm]['events']:
        before = base[(event['clip_id'], event['start_frame'])]
        old, new = before['first_alert_time_s'], event['first_alert_time_s']
        result.append(dict(clip_id=event['clip_id'], start_frame=event['start_frame'],
            baseline_detected=before['detected'], candidate_detected=event['detected'],
            baseline_first_s=old, candidate_first_s=new,
            delay_s=None if old is None or new is None else new-old,
            gained=not before['detected'] and event['detected'],
            lost=before['detected'] and not event['detected']))
    return result


def bootstrap(rows, arm, reference):
    rng = np.random.default_rng(SEED)
    groups = sorted({r['base_group_id'] for r in rows})
    totals = []
    for g in groups:
        group = [r for r in rows if r['base_group_id']==g and r['truth'] is not None]
        totals.append([sum(r['truth'] for r in group)] + [
            sum(r['truth'] and r['predictions'][a]['alert'] for r in group) for a in (arm, reference)])
    totals = np.array(totals)
    values = []
    for _ in range(1000):
        t = totals[rng.integers(0, len(groups), len(groups))].sum(0)
        values.append((t[1]-t[2])/max(1, t[0]))
    return dict(unit='whole base layout with all clips and queries', groups=len(groups),
                resamples=1000, seed=SEED, recall_delta_95=np.quantile(values, [.025, .975]).tolist())


def query_counts(probabilities, labels, threshold):
    y, valid = labels['classes'] < 6, labels['valid'].astype(bool)
    p = probabilities >= threshold
    def one(mask):
        mask = valid & mask
        tp, fp, fn, tn = (int(x.sum()) for x in (p & y & mask, p & ~y & mask, ~p & y & mask, ~p & ~y & mask))
        return dict(TP=tp, FP=fp, FN=fn, TN=tn, known_queries=int(mask.sum()),
                    recall=tp/(tp+fn) if tp+fn else None,
                    precision=tp/(tp+fp) if tp+fp else None)
    return dict(all=one(np.ones_like(valid)), by_query={str(q):one(np.arange(6)[None]==q) for q in range(6)},
                not_observed_free_space=True)


def report(rows, probabilities, labels, selection):
    metrics = evaluate_rows(rows, arms=REPORT_ARMS)
    strata = {key:{str(value):evaluate_rows([r for r in rows if r[key]==value], arms=REPORT_ARMS)
                   for value in sorted({r[key] for r in rows})}
              for key in ('type_id', 'layer', 'layout_relation', 'base_group_id')}
    comparisons = {}
    for arm, ref in [('raw', 'A_current'), ('local', 'A_current'), ('local', 'raw')]:
        a, b = (metrics['arms'][name] for name in (arm, ref))
        ac, bc = a['frames']['all_known'], b['frames']['all_known']
        diffs = event_differences(metrics, arm, ref)
        gain_groups = sum(v['arms'][arm]['frames']['all_known']['TP'] > v['arms'][ref]['frames']['all_known']['TP']
                          for v in strata['base_group_id'].values())
        d = dict(recall_delta=ac['recall']-bc['recall'], TP_delta=ac['TP']-bc['TP'],
                 FP_delta=ac['FP']-bc['FP'], false_segment_delta=a['false_alert_segment_count']-b['false_alert_segment_count'],
                 lost_true_frames=[r['id'] for r in rows if r['truth'] is True and r['predictions'][ref]['alert'] and not r['predictions'][arm]['alert']],
                 gained_true_frames=[r['id'] for r in rows if r['truth'] is True and not r['predictions'][ref]['alert'] and r['predictions'][arm]['alert']],
                 added_false_frames=[r['id'] for r in rows if r['truth'] is False and not r['predictions'][ref]['alert'] and r['predictions'][arm]['alert']],
                 improved_layouts=gain_groups, event_differences=diffs,
                 bootstrap=bootstrap(rows, arm, ref))
        layer_ok = all(v['arms'][arm]['frames']['all_known']['recall'] >= v['arms'][ref]['frames']['all_known']['recall']-.05-1e-12
                       for v in strata['layer'].values())
        costs = d['FP_delta'] <= 2 and d['false_segment_delta'] <= 1
        if ref == 'A_current':
            d['gate_met'] = bool(d['recall_delta'] >= .10-1e-12 and costs and not d['lost_true_frames']
                and not any(e['lost'] or (e['delay_s'] is not None and e['delay_s'] > 1e-9) for e in diffs))
        else:
            d['gate_met'] = bool(d['recall_delta'] >= .05-1e-12 and costs and gain_groups >= 4
                                 and d['bootstrap']['recall_delta_95'][0] > 0 and layer_ok)
        d['layer_recall_noninferior'] = layer_ok
        comparisons[arm+'_vs_'+ref] = d
    decisions = {}
    for arm in ARMS:
        decisions[arm] = ('COMPONENT_OR_CHALLENGER' if comparisons[arm+'_vs_A_current']['gate_met']
            else 'COMPONENT_OR_CHALLENGER' if arm=='local' and comparisons['local_vs_raw']['gate_met']
            else 'NEGATIVE_CONTROL')
    return dict(status='PASS', scope='CONSUMED_CONTROLLED_SAME_GENERATOR_DEVELOPMENT',
                metrics=metrics, strata=strata, comparisons=comparisons, inheritance_recommendations=decisions,
                query_metrics={a:query_counts(probabilities[a], labels, selection[a]['selection']['threshold']) for a in ARMS},
                selection=selection, mask_IoU='NOT_APPLICABLE_SCALAR_CLASSIFIER',
                physical_return_ownership='NOT_ESTABLISHED', range_estimation='NOT_OUTPUT')


def bind_inputs(args, out):
    manifest = read(args.materialization)
    files = {f'observations/{name}':args.observations/name for name in ('rgb.npy','tof.npy','identities.json')}
    files.update({'labels/train.npz':args.train_labels, 'labels/dev.npz':args.dev_labels})
    for key, path in files.items():
        if sha(path) != manifest['hashes'][key]:
            raise ValueError('Input identity mismatch: '+key)
    sources = [HERE/name for name in ('inherit_spatial_model.py', 'inherit_spatial_run.py',
        'INHERIT_SPATIAL_PROTOCOL_20260922.md', 'query_occupancy_data.py',
        'tof_corridor_calibration.py', 'ba_camera_corridor.py', 'ba_camera_corridor_metrics.py')]
    sources.append(ROOT/'tools/research_backend.py')
    snapshots = out/'source-snapshot'
    for path in sources:
        dest = snapshots/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    freeze = dict(status='PASS', sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        input_hashes={k:manifest['hashes'][k] for k in files},
        expected_evaluation_labels_sha256=manifest['hashes']['labels/evaluation.npz'],
        materialization_sha256=sha(args.materialization), params=PARAMS,
        feature_width=FEATURE_SIZE, queries=QUERIES.tolist(), evaluation_labels_opened=False,
        source_status='ALREADY_CONSUMED_DEVELOPMENT_NO_FRESHNESS_RESTORED')
    write(out/'freeze.json', freeze)
    return freeze


def run(args):
    out = args.result.parent.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Artifact routing violation')
    new_stage_directory(out)
    begun = time.perf_counter()
    freeze = bind_inputs(args, out)
    rgb = np.load(args.observations/'rgb.npy', mmap_mode='r')
    tof = np.load(args.observations/'tof.npy', mmap_mode='r')
    identities = read(args.observations/'identities.json')
    if rgb.shape != (1728,3,180,320) or tof.shape != (1728,64,6) or len(identities) != 1728:
        raise ValueError('Unexpected source dimensions')
    splits = {s:np.array([i for i,m in enumerate(identities) if m['split']==s], dtype=np.int64)
              for s in ('train','dev','evaluation')}
    if [len(splits[s]) for s in splits] != [864,288,576]:
        raise ValueError('Frozen split changed')
    groups = {s:{identities[i]['base_group_id'] for i in ix} for s,ix in splits.items()}
    if [len(groups[s]) for s in splits] != [24,8,16] or any(groups[a] & groups[b] for a,b in (('train','dev'),('train','evaluation'),('dev','evaluation'))):
        raise ValueError('Layout groups are not disjoint')
    select_backend('batch-tensor', cpu=BackendCandidate('numpy-sklearn-cpu', 'cpu',
        lambda:extract(rgb[0],tof[0]), lambda _:DeviceObservation('cpu',platform.processor() or 'host CPU',
        'numpy '+np.__version__+' sklearn '+sklearn.__version__)), cpu_reason='TASK_NOT_GPU_SUITABLE',
        record_path=out/'backend.json', capabilities=dict(reason='Small tabular histogram trees and ragged pixel statistics; no implemented GPU tree backend',
        python_executable=sys.executable, threads=4))
    x = np.empty((1728,6,2,FEATURE_SIZE), np.float32)
    feature_times, baseline_times = [], []
    for i in range(1728):
        tick = time.perf_counter()
        x[i] = extract(rgb[i], tof[i])
        feature_times.append(time.perf_counter()-tick)
        tick = time.perf_counter()
        values = np.where(tof[i,:,1] == 1, tof[i,:,0]*8, np.nan)
        boxes = np.rint(tof[i,:,2:]*[192,256,192,256]).astype(int)
        baseline = decide(score_frame(boxes,values), .4071309640537889)
        prior = identities[i]['baseline']
        if any(baseline[k] != prior[k] for k in ('alert','unknown','ambiguous','valid_zones','definite_zones')):
            raise ValueError('Frozen A parity failure at '+str(i))
        if abs(baseline['score']-prior['score']) > 1e-12:
            raise ValueError('Frozen A score mismatch')
        identities[i] = dict(identities[i], baseline=baseline)
        baseline_times.append(time.perf_counter()-tick)
        if (i+1)%144 == 0:
            print(json.dumps(dict(stage='public_features', frames=i+1, seconds=time.perf_counter()-begun)),flush=True)
    np.savez_compressed(out/'features.npz', raw=x[:,:,0], local=x[:,:,1])
    write(out/'feature-seal.json', dict(sha256=sha(out/'features.npz'), freeze_sha256=sha(out/'freeze.json'),
        indices={s:ix.tolist() for s,ix in splits.items()}, groups={s:sorted(g) for s,g in groups.items()},
        A_parity_frames=1728, evaluation_labels_opened=False, features_from_arrays_only=True))
    labels = {s:dict(np.load(p, allow_pickle=False)) for s,p in (('train',args.train_labels),('dev',args.dev_labels))}
    for split in labels:
        if not np.array_equal(labels[split]['indices'],splits[split]):
            raise ValueError('Label identity mismatch')
    train_y = (labels['train']['classes'] < 6).ravel()
    valid = labels['train']['valid'].astype(bool).ravel()
    if not train_y[valid].any() or train_y[valid].all():
        raise ValueError('NOT_EVALUABLE: no train class contrast')
    probabilities, selection, training, models = {}, {}, {}, {}
    fit_times, inference_times, single_inference = {}, {}, {}
    for ai, arm in enumerate(ARMS):
        model = HistGradientBoostingClassifier(**PARAMS)
        tick = time.perf_counter()
        model.fit(x[splits['train'],:,ai].reshape(-1,FEATURE_SIZE)[valid], train_y[valid])
        fit_times[arm] = time.perf_counter()-tick
        if model.n_iter_ != PARAMS['max_iter']:
            raise ValueError('Unexpected fit budget')
        with (out/(arm+'.pkl')).open('xb') as stream:
            pickle.dump(model, stream)
        models[arm] = model
        dev_score = model.predict_proba(x[splits['dev'],:,ai].reshape(-1,FEATURE_SIZE))[:,1].reshape(-1,6)
        dev_metas = [identities[i] for i in splits['dev']]
        selection[arm] = select_threshold(dev_score[:,CENTRE].max(1),labels['dev'],dev_metas)
        train_score = model.predict_proba(x[splits['train'],:,ai].reshape(-1,FEATURE_SIZE))[:,1].reshape(-1,6)
        training[arm] = query_counts(train_score, labels['train'], .5)
        np.savez_compressed(out/(arm+'-development-scores.npz'), train=train_score, dev=dev_score)
        print(json.dumps(dict(stage='fit', arm=arm, fit_s=fit_times[arm], selection=selection[arm]['selection'])),flush=True)
    write(out/'selection.json',selection)
    write(out/'training.json',training)
    write(out/'model-seal.json',dict(models={a:sha(out/(a+'.pkl')) for a in ARMS},
        selection_sha256=sha(out/'selection.json'), features_sha256=sha(out/'feature-seal.json'),
        training_sha256=sha(out/'training.json'), evaluation_labels_opened=False))
    for ai, arm in enumerate(ARMS):
        tick = time.perf_counter()
        probabilities[arm] = models[arm].predict_proba(x[splits['evaluation'],:,ai].reshape(-1,FEATURE_SIZE))[:,1].reshape(-1,6)
        inference_times[arm] = time.perf_counter()-tick
        samples = []
        for i in splits['evaluation'][:20]:
            tick = time.perf_counter()
            models[arm].predict_proba(x[i,:,ai])
            samples.append(time.perf_counter()-tick)
        single_inference[arm] = dict(samples=20,p50_s=float(np.median(samples)),p95_s=float(np.quantile(samples,.95)))
        np.savez_compressed(out/(arm+'-predictions.npz'), indices=splits['evaluation'],probability=probabilities[arm])
    write(out/'prediction-seal.json',dict(status='PASS', models_sha256=sha(out/'model-seal.json'),
        source_seal_sha256=sha(out/'freeze.json'), selection_sha256=sha(out/'selection.json'),
        predictions={a:sha(out/(a+'-predictions.npz')) for a in ARMS},
        evaluation_labels_opened=False, frames=576, queries=3456))
    # The sole evaluation-label join occurs after every model/selection/prediction seal.
    if sha(args.evaluation_labels) != freeze['expected_evaluation_labels_sha256']:
        raise ValueError('Evaluation label hash mismatch')
    evaluation = dict(np.load(args.evaluation_labels,allow_pickle=False))
    if not np.array_equal(evaluation['indices'],splits['evaluation']):
        raise ValueError('Evaluation indices mismatch')
    rows = metric_rows([identities[i] for i in splits['evaluation']],evaluation,probabilities,selection)
    results = report(rows, probabilities, evaluation, selection)
    write(out/'frame-results.json',rows)
    write(out/'metrics.json',results)
    cost = dict(feature_frames=1728, feature_seconds=sum(feature_times),
        feature_p50_s=float(np.median(feature_times)), feature_p95_s=float(np.quantile(feature_times,.95)),
        A_reference_seconds=sum(baseline_times), fit_seconds=fit_times, batch_inference_seconds=inference_times,
        single_frame_cached_feature_inference=single_inference,
        total_elapsed_s=time.perf_counter()-begun, model_bytes={a:(out/(a+'.pkl')).stat().st_size for a in ARMS},
        scope='Host CPU, shared array-to-features plus HGB; excludes camera capture and PNG decode, not endpoint latency')
    write(out/'costs.json',cost)
    result = dict(status='PASS', frames=576, queries=3456, train_frames=864, dev_frames=288,
        inheritance_recommendations=results['inheritance_recommendations'],
        arms={a:dict(**m['frames']['all_known'], detected_events=m['detected_events'], events=m['event_count'],
                    false_segments=m['false_alert_segment_count']) for a,m in results['metrics']['arms'].items()},
        metrics_sha256=sha(out/'metrics.json'), prediction_seal_sha256=sha(out/'prediction-seal.json'),
        frame_results_sha256=sha(out/'frame-results.json'), costs_sha256=sha(out/'costs.json'),
        resource_state='CPU process completes; no capture/worker/allocation created',
        scope='CONSUMED_CONTROLLED_DEVELOPMENT_NO_PROMOTION')
    write(args.result,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--materialization', type=Path, required=True)
    parser.add_argument('--train-labels', type=Path, required=True)
    parser.add_argument('--dev-labels', type=Path, required=True)
    parser.add_argument('--evaluation-labels', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)
