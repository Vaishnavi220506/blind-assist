"""Frozen transfer predict/seal then evaluate; zero fitting and zero selection."""
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
from threadpoolctl import threadpool_limits

from inherit_spatial_model import ARMS, FEATURE_SIZE, QUERIES, extract
from query_occupancy_data import read, write, sha, new_stage_directory
from tof_corridor_calibration import score_frame, decide
from local_transfer_metrics import GATES, make_rows, report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from tools.research_backend import BackendCandidate, DeviceObservation, select_backend

A_THRESHOLD = .4071309640537889
FROZEN_MODELS = {'raw': 'c87050130956f6142298d9701e8f445eba9d3712912f361080fe732e267f72eb',
                 'local': '1e6a4345353887effebb2aa798ace9dcdc5a46111166486179402bfcd4cc610e'}
FROZEN_SELECTION = 'fa4ab0845aab5eb62a2c3dadacc82bc073c73b2e52da198527609384e6b4a753'


def frozen_bundle(root):
    seal, freeze = read(root/'model-seal.json'), read(root/'freeze.json')
    if seal['models'] != FROZEN_MODELS or seal['selection_sha256'] != FROZEN_SELECTION:
        raise ValueError('Frozen model authority changed')
    for arm, digest in FROZEN_MODELS.items():
        if sha(root/(arm+'.pkl')) != digest:
            raise ValueError('Frozen model payload changed: '+arm)
    if sha(root/'selection.json') != FROZEN_SELECTION:
        raise ValueError('Frozen cutoff payload changed')
    for name, digest in freeze['sources'].items():
        if sha(ROOT/name) != digest or sha(root/'source-snapshot'/name) != digest:
            raise ValueError('Original implementation changed: '+name)
    selection = read(root/'selection.json')
    return freeze, {arm: selection[arm]['selection']['threshold'] for arm in ARMS}


def validate_identities(ids, old_groups):
    if len(ids) != 576 or len({r['id'] for r in ids}) != 576:
        raise ValueError('Expected 576 unique frames')
    groups = {r['base_group_id'] for r in ids}
    if len(groups) != 8 or groups & old_groups:
        raise ValueError('Need eight new base-group identifiers')
    if {r['appearance'] for r in ids} != {'base', 'changed'}:
        raise ValueError('Unexpected appearances')
    if len({r['type_id'] for r in ids}) != 4:
        raise ValueError('Need all four families')
    for family in {r['type_id'] for r in ids}:
        if len({r['base_group_id'] for r in ids if r['type_id'] == family}) != 2:
            raise ValueError('Need two geometries per family')
    if any(r.get('index', i) != i for i, r in enumerate(ids)):
        raise ValueError('Identity index mismatch')
    clips = {}
    for row in ids:
        clips.setdefault(row['clip_id'], []).append(row)
    if len(clips) != 48:
        raise ValueError('Need 48 clips')
    for clip in clips.values():
        ordered = sorted(clip, key=lambda r: r['frame_in_clip'])
        if [r['frame_in_clip'] for r in ordered] != list(range(12)):
            raise ValueError('Need twelve contiguous frames')
        if any(abs(r['time_s']-.2*r['frame_in_clip']) > 1e-8 for r in ordered):
            raise ValueError('Unexpected sampled timestamps')
        for key in ('base_group_id', 'appearance', 'layout_relation', 'type_id', 'layer'):
            if len({r[key] for r in clip}) != 1:
                raise ValueError('Clip identity changes: '+key)
    for group in groups:
        rows = [r for r in ids if r['base_group_id'] == group]
        if len(rows) != 72 or {(r['appearance'], r['layout_relation']) for r in rows} != {
                (a, rel) for a in ('base', 'changed') for rel in ('INSIDE', 'BOUNDARY', 'OUTSIDE')}:
            raise ValueError('Incomplete factorial group')
    return sorted(groups)


def output_dir(result):
    out = result.parent.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Artifact routing violation')
    new_stage_directory(out)
    return out


def public_pair_checks(ids, rgb, tof):
    groups, issues, pairs = {}, [], []
    for i, meta in enumerate(ids):
        members = groups.setdefault(meta['appearance_pair_id'], {})
        if meta['appearance'] in members:
            issues.append('Duplicate public pair member: '+meta['id'])
        members[meta['appearance']] = i
    for key, members in sorted(groups.items()):
        if set(members) != {'base', 'changed'}:
            issues.append('Incomplete public pair: '+key)
            continue
        b, c = members['base'], members['changed']
        same = bool(np.array_equal(tof[b], tof[c], equal_nan=True))
        changed = not np.array_equal(rgb[b], rgb[c])
        l1 = float(np.abs(rgb[b].astype(np.float64)-rgb[c].astype(np.float64)).mean()/255)
        pairs.append(dict(pair_id=key, base_index=b, changed_index=c, tof_exactly_equal=same,
                          rgb_changed=changed, normalized_rgb_L1=l1))
        if not same or not changed:
            issues.append('Public appearance pair invalid: '+key)
    if len(pairs) != 288:
        issues.append('Expected 288 public appearance pairs')
    mean_l1 = float(np.mean([p['normalized_rgb_L1'] for p in pairs])) if pairs else None
    if mean_l1 is None or mean_l1 < GATES['minimum_mean_normalized_rgb_L1']:
        issues.append('Mean normalized RGB L1 is below 1/255')
    return dict(pairs=pairs, mean_normalized_rgb_L1=mean_l1, issues=issues,
                status='PASS' if not issues else 'NOT_EVALUABLE')


def predict(args):
    start = time.perf_counter()
    old, cutoffs = frozen_bundle(args.frozen_run)
    manifest = read(args.materialization)
    if not np.allclose(manifest['queries'], QUERIES, rtol=0, atol=3e-8):
        raise ValueError('Original fixed queries changed')
    input_hashes = {}
    for name in ('rgb.npy', 'tof.npy', 'identities.json'):
        key = 'observations/'+name
        input_hashes[key] = sha(args.observations/name)
        if input_hashes[key] != manifest['hashes'][key]:
            raise ValueError('Public input identity mismatch: '+name)
    ids = read(args.observations/'identities.json')
    old_group_sets = read(args.frozen_run/'feature-seal.json')['groups']
    groups = validate_identities(ids, set().union(*map(set, old_group_sets.values())))
    rgb, tof = (np.load(args.observations/name, mmap_mode='r', allow_pickle=False) for name in ('rgb.npy', 'tof.npy'))
    if rgb.shape != (576, 3, 180, 320) or rgb.dtype != np.uint8 or tof.shape != (576, 64, 6):
        raise ValueError('Unexpected public array shape or dtype')
    out = output_dir(args.result)
    own = [HERE/name for name in ('local_transfer_inference.py', 'local_transfer_metrics.py',
                                  'local_transfer_audit.py', 'test_local_transfer.py')]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in own}
    sources.update(old['sources'])
    for name, digest in sources.items():
        dest = out/'source-snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        if sha(dest) != digest:
            raise ValueError('Source snapshot failed')
    freeze = dict(source_hashes=sources, input_hashes=input_hashes,
        materialization_sha256=sha(args.materialization), protocol_sha256=sha(args.protocol),
        expected_evaluation_labels_sha256=manifest['hashes']['labels/evaluation.npz'],
        frozen_model_hashes=FROZEN_MODELS, frozen_selection_sha256=FROZEN_SELECTION,
        frozen_model_seal_sha256=sha(args.frozen_run/'model-seal.json'),
        thresholds=cutoffs, A_threshold=A_THRESHOLD, query_boxes=QUERIES.tolist(), groups=groups,
        feature_width=FEATURE_SIZE, evaluation_labels_opened=False, fits=0, cutoff_selections=0)
    freeze['gate_configuration'] = GATES
    write(out/'freeze.json', freeze)
    shutil.copyfile(args.frozen_run/'selection.json', out/'selection.json')
    select_backend('model-inference', cpu=BackendCandidate('frozen-hgb-cpu', 'cpu',
        lambda: extract(rgb[0], tof[0]), lambda _: DeviceObservation('cpu', platform.processor() or 'host CPU',
        'numpy '+np.__version__+' sklearn '+sklearn.__version__)), cpu_reason='TASK_NOT_GPU_SUITABLE',
        record_path=out/'backend.json', capabilities=dict(reason='Frozen tabular HGB and interval/RGB statistics',
        python_executable=sys.executable, threads=4))
    x = np.empty((576, 6, 2, FEATURE_SIZE), np.float32)
    baseline, feature_times, baseline_times = [], [], []
    for i in range(576):
        tick = time.perf_counter()
        x[i] = extract(rgb[i], tof[i])
        feature_times.append(time.perf_counter()-tick)
        tick = time.perf_counter()
        values = np.where(tof[i, :, 1] == 1, tof[i, :, 0]*8, np.nan)
        boxes = np.rint(tof[i, :, 2:]*[192, 256, 192, 256]).astype(int)
        base = decide(score_frame(boxes, values), A_THRESHOLD)
        if 'baseline' in ids[i]:
            prior = ids[i]['baseline']
            if any(base[k] != prior[k] for k in ('alert', 'unknown', 'ambiguous', 'valid_zones', 'definite_zones')) or abs(base['score']-prior['score']) > 1e-12:
                raise ValueError('Materialization A parity failed')
        baseline.append(base)
        baseline_times.append(time.perf_counter()-tick)
    np.savez_compressed(out/'features.npz', raw=x[:, :, 0], local=x[:, :, 1])
    probabilities, times = {}, {}
    with threadpool_limits(limits=4):
        for j, arm in enumerate(ARMS):
            with (args.frozen_run/(arm+'.pkl')).open('rb') as stream:
                model = pickle.load(stream)
            if list(model.classes_) != [0, 1] or model.n_features_in_ != FEATURE_SIZE:
                raise ValueError('Unexpected frozen classifier schema')
            tick = time.perf_counter()
            probabilities[arm] = model.predict_proba(x[:, :, j].reshape(-1, FEATURE_SIZE))[:, 1].reshape(576, 6)
            times[arm] = time.perf_counter()-tick
    np.savez_compressed(out/'probabilities.npz', indices=np.arange(576), **probabilities)
    write(out/'baseline.json', baseline)
    write(out/'identities.json', ids)
    write(out/'public-pair-checks.json', public_pair_checks(ids, rgb, tof))
    costs = dict(frames=576, fits=0, cutoff_selections=0,
        feature_seconds=sum(feature_times), feature_p50_s=float(np.median(feature_times)),
        feature_p95_s=float(np.quantile(feature_times, .95)), baseline_seconds=sum(baseline_times),
        batch_inference_seconds=times, total_seconds=time.perf_counter()-start,
        scope='Host CPU; excludes capture and PNG decode, no endpoint latency claim')
    write(out/'costs.json', costs)
    seal = dict(status='PASS', frames=576, queries=3456, evaluation_labels_opened=False,
        fits=0, cutoff_selections=0, thresholds=cutoffs,
        hashes={name: sha(out/name) for name in ('freeze.json', 'features.npz', 'probabilities.npz',
               'baseline.json', 'identities.json', 'selection.json', 'costs.json', 'public-pair-checks.json')})
    write(out/'prediction-seal.json', seal)
    answer = dict(status='PASS', phase='SEALED_PREDICTIONS', frames=576, fits=0,
        prediction_seal_sha256=sha(out/'prediction-seal.json'), thresholds=cutoffs,
        evaluation_labels_opened=False, resource_state='CPU command completed; no worker or capture created')
    write(args.result, answer)
    print(json.dumps(answer), flush=True)


def evaluate(args):
    seal = read(args.predictions/'prediction-seal.json')
    freeze = read(args.predictions/'freeze.json')
    if seal['evaluation_labels_opened'] or freeze['fits'] or freeze['cutoff_selections']:
        raise ValueError('Frozen prediction contract violated')
    for name, digest in seal['hashes'].items():
        if sha(args.predictions/name) != digest:
            raise ValueError('Prediction seal mismatch: '+name)
    for name, digest in freeze['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Implementation changed after prediction sealing: '+name)
    if sha(args.protocol) != freeze['protocol_sha256'] or sha(args.labels) != freeze['expected_evaluation_labels_sha256']:
        raise ValueError('Protocol or evaluator identity mismatch')
    # No evaluator payload is parsed before the prediction seal is checked above.
    labels = dict(np.load(args.labels, allow_pickle=False))
    probabilities = dict(np.load(args.predictions/'probabilities.npz', allow_pickle=False))
    identities = read(args.predictions/'identities.json')
    rows = make_rows(identities, read(args.predictions/'baseline.json'), probabilities, labels, freeze['thresholds'])
    if GATES != freeze['gate_configuration']:
        raise ValueError('Gate configuration changed after prediction sealing')
    measured = report(rows, freeze['thresholds'], labels, read(args.predictions/'public-pair-checks.json'))
    out = output_dir(args.result)
    write(out/'frame-results.json', rows)
    write(out/'metrics.json', measured)
    answer = dict(status='PASS', phase='EVALUATED', frames=len(rows),
        prediction_seal_sha256=sha(args.predictions/'prediction-seal.json'),
        evaluator_labels_sha256=sha(args.labels), protocol_sha256=sha(args.protocol),
        metrics_sha256=sha(out/'metrics.json'), frame_results_sha256=sha(out/'frame-results.json'),
        thresholds=freeze['thresholds'], fits=0, cutoff_selections=0,
        source_admissibility=measured['source_admissibility']['status'], gates=measured['gates'],
        scope='Frozen transfer, new controlled geometries in the same generator, not external validation')
    write(args.result, answer)
    print(json.dumps(answer), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='phase', required=True)
    p = commands.add_parser('predict')
    for name in ('observations', 'materialization', 'frozen-run', 'protocol', 'result'):
        p.add_argument('--'+name, type=Path, required=True)
    p = commands.add_parser('evaluate')
    for name in ('predictions', 'labels', 'protocol', 'result'):
        p.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    (predict if args.phase == 'predict' else evaluate)(args)
