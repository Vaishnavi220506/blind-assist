"""One fixed-B fit under a changed data condition, with sealed public inference.

New development labels calibrate B_control and N by the same finite rule. Legacy
B_frozen/R_frozen working points are immutable references. Evaluation truth is
joined only after all five arms' predictions have been sealed.
"""
from datetime import datetime, timezone
import importlib.util
import time

import numpy as np
import torch

import spatial_bce_model as model
import corridor_relative_model as relative
from run_spatial_bce import held
from run_corridor_relative import costs
from run_data_coverage import OUT, ROOT, HERE, read, write, sha, seal, verify, check

ORIGINAL = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
RELATIVE = ROOT / 'artifacts.local/work/ba-corridor-relative-20260921'
ENCODER = ROOT / 'artifacts.local/work/ba-nfo-20260919/torch-cache/checkpoints/mobilenet_v3_small-047dcff4.pth'
B_THRESHOLD = 7.6612162590026855
R_THRESHOLD = 5.128307342529297
NAMES = ('A', 'B_frozen', 'R_frozen', 'B_control', 'N')
ARMS = tuple(name + '_' + mode for mode in ('current', 'hold') for name in NAMES)
SPLITS = dict(train=(1728, 24), dev=(576, 8), evaluation=(1152, 16))


def _start(name, **fields):
    write(OUT / (name + '-start.json'), dict(time_utc=datetime.now(timezone.utc).isoformat(), **fields))


def _public(indices):
    """Read only public identities, RGB and original single-return observations."""
    identities = read(OUT / 'identities.json')
    selected = [identities[i] for i in indices]
    assert [r['index'] for r in selected] == indices
    paths = [OUT / r['rgb_path'] for r in selected]
    for path, row in zip(paths, selected):
        assert sha(path) == row['rgb_sha256'], str(path)
    with np.load(OUT / 'observations.npz', allow_pickle=False) as data:
        ranges, boxes = data['ranges'][indices].copy(), data['boxes'].copy()
    assert ranges.shape == (len(indices), 64) and boxes.shape == (64, 4)
    return selected, paths, ranges, boxes


def _inputs(split):
    indices = read(OUT / 'split-routing.json')[split]
    _, _, ranges, boxes = _public(indices)
    features = np.load(OUT / (split + '-encoder/rgb_features.npy'), mmap_mode='r')
    return model.build_inputs(features, ranges, boxes / np.array([192, 256, 192, 256], np.float32))


def prepare():
    verify(); check('observation-seal.json')
    _start('features', evaluation_images_read=False, labels_read=False)
    metadata = read(OUT / 'evaluator/metadata.json')
    routing, groups = {}, {}
    for split, (count, group_count) in SPLITS.items():
        rows = [r for r in metadata if r['split'] == split]
        routing[split] = [r['index'] for r in rows]
        groups[split] = {r['base_group_id'] for r in rows}
        assert len(rows) == count and len(groups[split]) == group_count
        assert all(sum(r['base_group_id'] == g for r in rows) == 72 for g in groups[split])
    assert sorted(sum(routing.values(), [])) == list(range(3456))
    assert not (groups['train'] & groups['dev'] or groups['train'] & groups['evaluation'] or groups['dev'] & groups['evaluation'])
    write(OUT / 'split-routing.json', routing)
    files = ['split-routing.json']
    started = time.perf_counter()
    for split in ('train', 'dev'):
        _, paths, _, boxes = _public(routing[split])
        model.encode_rgb(paths, ENCODER, OUT / (split + '-encoder'),
                         boxes=boxes / np.array([192, 256, 192, 256], np.float32))
        files += [split + '-encoder/' + n for n in ('rgb_features.npy', 'feature_receipt.json', 'encoder_backend.json')]
    write(OUT / 'features-receipt.json', dict(elapsed_s=time.perf_counter() - started,
        rows=2304, evaluation_images_read=0, labels_read=False, native_depth_read=False,
        routing_metadata_use='Split membership only; no truth or geometry enters features'))
    seal('features-seal.json', files + ['features-receipt.json'])


def fit():
    verify(); check('features-seal.json'); model._precision()
    _start('fit', recipe=model.RECIPE, automatic_retry=False)
    indices = read(OUT / 'split-routing.json')['train']
    labels = read(OUT / 'evaluator/train-labels.json')
    assert [r['index'] for r in labels] == indices and len(labels) == 1728
    head = model.train_fit(_inputs('train'), np.arange(1728), [r['truth'] for r in labels], OUT / 'fit')
    dev_inputs = _inputs('dev')
    np.save(OUT / 'N-dev-logits.npy', model.predict(head, dev_inputs))
    old = model.load_head(ORIGINAL / 'fit/head_last.pt', device=str(next(head.parameters()).device))
    np.save(OUT / 'B_control-dev-logits.npy', model.predict(old, dev_inputs))
    seal('fit-seal.json', ['fit/' + n for n in ('head_last.pt', 'train_receipt.json', 'train_backend.json', 'training.jsonl')]
         + ['N-dev-logits.npy', 'B_control-dev-logits.npy'])


def choose_threshold(logits, truth, metadata, baseline):
    """Max Boundary held TP, min Core held FP, min Boundary held FP, max cut."""
    logits = np.asarray(logits, float)
    y, a = np.asarray(truth, bool), np.asarray(baseline, bool)
    if logits.ndim != 1 or len(logits) != len(metadata) or y.shape != logits.shape or a.shape != logits.shape:
        raise ValueError('Aligned nonempty development vectors required')
    if not len(logits) or not np.isfinite(logits).all():
        raise ValueError('Finite nonempty development logits required')
    disable = float(logits.max() + 1.)
    if not np.isfinite(disable) or disable <= logits.max():
        raise ValueError('Cannot construct frozen finite disabled cutoff')
    boundary = np.array([r['layout_relation'] == 'BOUNDARY' for r in metadata])
    candidates, best = [], None
    for threshold in sorted(set(logits.tolist()) | {disable}):
        cost = costs(logits, threshold, y, metadata, a)
        flags = held(a | (logits >= threshold), metadata)
        tp = int((boundary & y & flags).sum())
        core_fp = int((~boundary & ~y & flags).sum())
        boundary_fp = int((boundary & ~y & flags).sum())
        admissible = all(v['pass_cost'] for v in cost.values())
        row = dict(threshold=threshold, cost=cost, admissible=admissible,
                   Boundary_hold_TP=tp, Core_hold_FP=core_fp, Boundary_hold_FP=boundary_fp)
        candidates.append(row)
        key = (tp, -core_fp, -boundary_fp, threshold)
        if admissible and (best is None or key > best[0]):
            best = (key, row)
    assert best is not None  # The finite disabled branch exactly retains A.
    return dict(**best[1], disabled=best[1]['threshold'] == disable,
                candidate_count=len(candidates)), candidates


def select():
    verify(); check('fit-seal.json'); check('features-seal.json')
    _start('selection', selection_split='dev', evaluation_labels_read=False)
    indices = read(OUT / 'split-routing.json')['dev']
    all_metadata = read(OUT / 'evaluator/metadata.json')
    metadata = [all_metadata[i] for i in indices]
    labels = read(OUT / 'evaluator/dev-labels.json')
    assert [r['index'] for r in labels] == indices and len(labels) == 576
    baseline = read(OUT / 'baseline.json')
    selection, candidates = {}, {}
    for name in ('B_control', 'N'):
        selection[name], candidates[name] = choose_threshold(np.load(OUT / (name + '-dev-logits.npy')),
            [r['truth'] for r in labels], metadata, [baseline[i]['current'] for i in indices])
    write(OUT / 'selection.json', dict(arms=selection,
        rule='Admissible fixed costs; max Boundary held TP, min Core held FP, min Boundary held FP, highest cutoff',
        legacy_thresholds_unchanged=dict(B_frozen=B_THRESHOLD, R_frozen=R_THRESHOLD),
        interpretation='B_control is a separately calibrated experimental control; original working point is not revised'))
    write(OUT / 'selection-candidates.json', candidates)
    seal('selection-seal.json', ['selection.json', 'selection-candidates.json'])
    print('SELECTED', {n: r['threshold'] for n, r in selection.items()}, flush=True)


def _predict_head(head, inputs, name):
    device, backend = model._select(head, torch.from_numpy(inputs[:64].copy()), OUT / (name + '-inference-backend.json'))
    head.to(device).eval()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = np.concatenate([head(torch.from_numpy(inputs[s:s + 64].copy()).to(device)).cpu().numpy()
                                 for s in range(0, len(inputs), 64)])
    assert np.isfinite(logits).all()
    return logits, dict(backend=backend, elapsed_s=time.perf_counter() - started,
                       actual_device=str(next(head.parameters()).device))


def predict():
    verify(); check('features-seal.json'); check('fit-seal.json'); check('selection-seal.json'); model._precision()
    _start('prediction', evaluation_truth_read=False, native_depth_read=False)
    indices = read(OUT / 'split-routing.json')['evaluation']
    ids, paths, ranges, boxes = _public(indices)
    assert len(indices) == 1152
    normalized = boxes / np.array([192, 256, 192, 256], np.float32)
    features = model.encode_rgb(paths, ENCODER, OUT / 'evaluation-encoder', boxes=normalized)
    inputs = model.build_inputs(features, ranges, normalized)
    b, b_receipt = _predict_head(model.load_head(ORIGINAL / 'fit/head_last.pt'), inputs, 'B')
    n, n_receipt = _predict_head(model.load_head(OUT / 'fit/head_last.pt'), inputs, 'N')
    started = time.perf_counter()
    r_features = np.empty((len(indices), 64, 60), np.float32)
    for j, path in enumerate(paths):
        r_features[j], valid = relative.extract(path, ranges[j], boxes)
        assert np.array_equal(valid, r_features[j, :, -1].astype(bool))
        if j % 288 == 0:
            print('R_FROZEN_FEATURES', j, '/', len(indices), flush=True)
    r_feature_seconds = time.perf_counter() - started
    payload = torch.load(RELATIVE / 'head_last.pt', map_location='cpu', weights_only=True)
    assert payload['recipe'] == relative.RECIPE
    r_head = relative.Head(); r_head.load_state_dict(payload['state_dict'], strict=True)
    r, r_receipt = _predict_head(r_head, r_features, 'R')
    np.savez_compressed(OUT / 'evaluation-logits.npz', B=b, R=r, N=n)
    selection = read(OUT / 'selection.json')['arms']
    baseline = read(OUT / 'baseline.json')
    a = np.array([baseline[i]['current'] for i in indices], bool)
    flags = dict(A_current=a, B_frozen_current=a | (b.astype(float) >= B_THRESHOLD),
        R_frozen_current=a | (r.astype(float) >= R_THRESHOLD),
        B_control_current=a | (b.astype(float) >= selection['B_control']['threshold']),
        N_current=a | (n.astype(float) >= selection['N']['threshold']))
    flags.update({name + '_hold': held(flags[name + '_current'], ids) for name in NAMES})
    rows = []
    for j, identity in enumerate(ids):
        i = indices[j]
        assert baseline[i]['index'] == i
        rows.append(dict(**{k: identity[k] for k in ('id', 'index', 'clip_id', 'time_s', 'frame_in_clip')},
            logits=dict(B=float(b[j]), R=float(r[j]), N=float(n[j])),
            flags={arm: bool(flags[arm][j]) for arm in ARMS},
            current_unknown={arm: bool(baseline[i]['unknown']) for arm in ARMS}))
    write(OUT / 'predictions.json', rows)
    write(OUT / 'inference-receipt.json', dict(frames=len(rows), B=b_receipt, R=r_receipt, N=n_receipt,
        R_feature_elapsed_s=r_feature_seconds, R_feature_placement='TASK_NOT_GPU_SUITABLE: ragged per-zone image statistics',
        evaluation_truth_read=False, native_depth_read=False, protected_original_test_activated=False))
    seal('prediction-seal.json', ['predictions.json', 'evaluation-logits.npz', 'inference-receipt.json']
         + [n + '-inference-backend.json' for n in ('B', 'R', 'N')]
         + ['evaluation-encoder/' + n for n in ('rgb_features.npy', 'feature_receipt.json', 'encoder_backend.json')])
    print('PREDICTIONS_SEALED', len(rows), flush=True)


def metrics(rows):
    spec = importlib.util.spec_from_file_location('coverage_metrics', HERE / 'full_event_metrics_20260920.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ARMS = ARMS
    return module.evaluate(rows)


def event_retention(result, candidate='N'):
    details = []
    for region in ('Core', 'Boundary'):
        for mode in ('current', 'hold'):
            old = result[region]['arms']['A_' + mode]['events']
            new = {r['clip_id']: r for r in result[region]['arms'][candidate + '_' + mode]['events']}
            for event in old:
                got = new[event['clip_id']]
                passed = (not event['detected'] or
                          (got['detected'] and got['first_in_event_alert_delay_s'] <= event['first_in_event_alert_delay_s'] + 1e-9))
                details.append(dict(region=region, mode=mode, clip_id=event['clip_id'], pass_retention=passed))
    return dict(pass_retention=all(r['pass_retention'] for r in details), events=details,
                interpretation='A OR branch plus identical hold structurally preserves A; not independent model benefit')


def evaluate():
    verify(); check('prediction-seal.json'); check('selection-seal.json')
    _start('evaluation', prediction_seal_sha256=sha(OUT / 'prediction-seal.json'))
    predictions = read(OUT / 'predictions.json')
    indices = read(OUT / 'split-routing.json')['evaluation']
    metadata = read(OUT / 'evaluator/metadata.json')
    labels = read(OUT / 'evaluator/evaluation-labels.json')
    native = read(OUT / 'source-admission.json')['frames']
    assert [r['index'] for r in labels] == indices == [r['index'] for r in predictions]
    rows = []
    for p, y in zip(predictions, labels):
        i = p['index']; m = metadata[i]; source = native[i]
        assert m['index'] == i and p['id'] == m['id'] == source['id']
        assert m['split'] == 'evaluation'
        rows.append(dict(**m, truth=y['truth'], flags=p['flags'], current_unknown=p['current_unknown'],
                         native_target_corridor_samples=source['returned_target_corridor_samples']))
    assert len(rows) == 1152 and len({r['base_group_id'] for r in rows}) == 16
    result = metrics(rows)
    groups = {g: metrics([r for r in rows if r['base_group_id'] == g]) for g in sorted({r['base_group_id'] for r in rows})}
    types = {t: metrics([r for r in rows if r['type_id'] == t]) for t in sorted({r['type_id'] for r in rows})}
    with np.load(OUT / 'evaluation-logits.npz', allow_pickle=False) as data:
        logits = {name: data[name].copy() for name in ('B', 'R', 'N')}
    selection = read(OUT / 'selection.json')['arms']
    thresholds = dict(B_frozen=B_THRESHOLD, R_frozen=R_THRESHOLD,
                      B_control=selection['B_control']['threshold'], N=selection['N']['threshold'])
    cost = {name: costs(logits['B' if name.startswith('B') else name[0]], threshold,
        [r['truth'] for r in rows], rows, [r['flags']['A_current'] for r in rows]) for name, threshold in thresholds.items()}
    gains = [g for g, value in groups.items() if value['Boundary']['arms']['N_hold']['frames']['TP'] > value['Boundary']['arms']['A_hold']['frames']['TP']]
    retention = event_retention(result)
    retention['all_A_flags_retained'] = all(not r['flags']['A_' + mode] or r['flags']['N_' + mode]
        for r in rows for mode in ('current', 'hold'))
    n_recall = result['Boundary']['arms']['N_hold']['frames']['recall']
    b_recall = result['Boundary']['arms']['B_control_hold']['frames']['recall']
    usable = dict(cost=all(v['pass_cost'] for v in cost['N'].values()),
                  retain_A_events_and_onset=retention['pass_retention'] and retention['all_A_flags_retained'],
                  boundary_recall=n_recall >= .5, broad_gain=len(gains) >= 8)
    contribution = dict(**usable, matched_control_gain=n_recall - b_recall >= .10 - 1e-12)
    increments = [r for r in rows if r['truth'] and r['flags']['N_current'] and not r['flags']['A_current']]
    terminal = dict(status='PASS' if all(contribution.values()) else 'NO_GO', gates=contribution,
        candidate_usable=all(usable.values()), data_condition_contribution_supported=all(contribution.values()),
        Boundary_held_recall_gain_over_B_control=n_recall - b_recall, boundary_gain_groups=gains, cost=cost,
        new_current_TP=len(increments), new_current_TP_without_native_corridor_contributor=sum(r['native_target_corridor_samples'] == 0 for r in increments),
        scope='PROSPECTIVE_SAME_GENERATOR_SIMULATION_DEVELOPMENT', protected_original_test_activated=False,
        automatic_successor=False, attribution_limit='Bundled geometry and appearance distribution change; individual causes not isolated')
    write(OUT / 'metrics.json', result); write(OUT / 'frame-results.json', rows)
    write(OUT / 'group-metrics.json', groups); write(OUT / 'type-metrics.json', types)
    write(OUT / 'retention.json', retention); write(OUT / 'result.json', terminal)
    role = 'COMPONENT_OR_CHALLENGER' if all(usable.values()) else 'NEGATIVE_CONTROL'
    write(OUT / 'local-inheritance.json', dict(terminal_id=OUT.name, inheritance_role=role,
        inheritance_mode='CHALLENGER' if role == 'COMPONENT_OR_CHALLENGER' else None,
        role_scope='This one fixed-B fit under the sealed data-coverage condition and fixed dev selection',
        retained_surface='A and legacy B/R dispositions unchanged; original UNKNOWN and ranges unchanged',
        failure_signature=terminal['gates'], revisit_trigger='Separate authorization and materially different hypothesis',
        assignment_basis='Sealed prospective simulation Development evaluation; no protected test or deployment claim'))
    seal('evaluation-seal.json', ['metrics.json', 'frame-results.json', 'group-metrics.json', 'type-metrics.json',
                                 'retention.json', 'result.json', 'local-inheritance.json'])
    print('EVALUATED', terminal, flush=True)
