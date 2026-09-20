"""One grouped spatial BCE fit with a development gate before test access."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
CHECKPOINT = ROOT / 'artifacts.local/work/ba-nfo-20260919/torch-cache/checkpoints/mobilenet_v3_small-047dcff4.pth'
T = .4071309640537889
ARMS = ('A_current', 'B_current', 'A_hold', 'B_hold')
OLD = ('ba-core-transfer-20260920', 'ba-core-workpoint-transfer-20260920',
       'ba-core-hold-validation-20260920', 'ba-full-event-transfer-20260920')
CODE = ('run_spatial_bce.py', 'spatial_bce_spec.py', 'spatial_bce_capture.py',
        'launch_spatial_bce.py', 'spatial_bce_model.py', 'core_transfer_spec.py',
        'ue_capture_readiness.py', 'run_core_transfer.py', 'full_event_metrics_20260920.py',
        'tof_fov45_core.py', 'tof_corridor_calibration.py', 'ba_camera_corridor.py',
        'ba_camera_corridor_spec.py', 'evaluate_ba_camera_corridor.py', 'tof_lateral_core.py',
        'ba_camera_corridor_metrics.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def seal(out, name, files):
    write(out / name, dict(protocol_sha256=sha(out / 'protocol.json'),
                          hashes={str(p): sha(out / p) for p in files}))


def check_seal(out, name):
    obj = read(out / name)
    assert obj['protocol_sha256'] == sha(out / 'protocol.json')
    for p, digest in obj['hashes'].items():
        assert sha(out / p) == digest, p


def verify(out):
    p = read(out / 'protocol.json')
    assert sha(out / 'spec.json') == p['spec_sha256']
    assert sha(out / 'protocol-before-run.md') == p['protocol_text_sha256']
    for name, digest in p['code_hashes'].items():
        assert sha(HERE / name) == digest, name
    for name, digest in p['input_hashes'].items():
        assert sha(ROOT / name) == digest, name
    return p


def freeze(out):
    from spatial_bce_spec import specification, check_spec
    from spatial_bce_model import RECIPE
    assert not out.exists(), 'One new output only; no overwrite'
    spec = specification()
    old = [ROOT / 'artifacts.local/work' / n / 'spec.json' for n in OLD]
    receipt = check_spec(spec, [read(p) for p in old])
    assert CHECKPOINT.is_file()
    out.mkdir(parents=True)
    write(out / 'spec.json', spec)
    write(out / 'preflight.json', receipt)
    (out / 'protocol-before-run.md').write_bytes((HERE / 'SPATIAL_BCE_PROTOCOL_20260920.md').read_bytes())
    deps = old + [CHECKPOINT, ROOT / 'tools/research_backend.py', ROOT / 'tools/run_obstacle_research.py',
                  ROOT / 'research/active/dtr-r0/unreal/street_process_lifecycle.py']
    write(out / 'protocol.json', dict(id='ba-spatial-bce-20260920',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        frames=2880, clips=120, frames_per_clip=24, capture_timeout_s=3600, dt_s=.2,
        spec_sha256=sha(out / 'spec.json'), protocol_text_sha256=sha(out / 'protocol-before-run.md'),
        code_hashes={n: sha(HERE / n) for n in CODE},
        input_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in deps},
        model_recipe=RECIPE, strong_threshold=T, arms=list(ARMS),
        scope='NEW_PROCEDURAL_GROUPS_SAME_SIMULATOR_DEVELOPMENT',
        stop='One fit; stop before test if no admissible development threshold; no successor'))
    print(json.dumps(dict(status='FROZEN', receipt=receipt)), flush=True)


def materialize(out):
    import run_core_transfer as source
    from core_transfer_spec import bounds, classify
    from tof_fov45_core import boxes45, simulate
    from ba_camera_corridor import sample_native
    from tof_corridor_calibration import score_frame, decide
    p = verify(out)
    write(out / 'materialization-start.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        backend='TASK_NOT_GPU_SUITABLE', input_law='UNCHANGED_SINGLE_RETURN'))
    cap = out / 'capture'
    receipt, launch = read(cap / 'receipt.json'), read(cap / 'launch-receipt.json')
    assert receipt['status'] == 'PASS' and receipt['frame_count'] == p['frames']
    assert receipt['source_unchanged'] and receipt['task_actors_released']
    assert read(cap / 'process-release.json')['released']
    assert launch['protocol_sha256'] == receipt['protocol_sha256'] == sha(out / 'protocol.json')
    assert receipt['spec_sha256'] == launch['spec_sha256'] == p['spec_sha256']
    assert receipt['script_sha256'] == launch['capture_script_sha256'] == p['code_hashes']['spatial_bce_capture.py']
    assert launch['launcher_sha256'] == p['code_hashes']['launch_spatial_bce.py']
    assert receipt['readiness_helper_sha256'] == p['code_hashes']['ue_capture_readiness.py']
    spec = read(out / 'spec.json')
    assert receipt['map_sha256_before'] == receipt['map_sha256_after'] == spec['expected_map_sha256']
    assert len(receipt['view_readiness']) == p['frames']
    manifest = read(cap / 'observations/manifest.json')['frames']
    geometry = read(cap / 'evaluator/geometry.json')
    assert len(manifest) == len(geometry) == len(spec['cases']) == p['frames']
    values, identities, baseline, admission, private, metadata = [], [], [], [], [], []
    labels = {s: [] for s in ('train', 'dev', 'test')}
    boxes = boxes45()
    for i, (case, rgb, geo) in enumerate(zip(spec['cases'], manifest, geometry)):
        source.source_check(case, geo, rgb, i)
        source.validate_readiness(receipt['view_readiness'][i])
        native_path = cap / 'evaluator' / geo['native_path']
        rgb_path = cap / 'observations' / rgb['rgb_path']
        assert sha(native_path) == geo['native_sha256']
        assert sha(rgb_path) == rgb['rgb_sha256'] == geo['rgb_sha256']
        native = np.load(native_path, allow_pickle=False)
        target, corridor = source.masks(native, case, geo)
        # Same group/time RNG identity for lateral interventions, never a label.
        identity = 'spatial-bce-v1/' + case['sensor_noise_key']
        vector, traces = simulate(sample_native(native), identity, boxes)
        score = score_frame(boxes, vector)
        a = decide(score, T)
        ids = dict(id=f'f{i:04d}', index=i, clip_id=rgb['clip_id'],
            frame_in_clip=rgb['frame_in_clip'], time_s=rgb['time_s'],
            rgb_path=rgb_path.relative_to(out).as_posix(), rgb_sha256=rgb['rgb_sha256'])
        identities.append(ids)
        values.append(vector)
        baseline.append(dict(index=i, current=a['alert'], unknown=a['unknown'], score=score['score'],
            valid_zones=a['valid_zones'], definite_zones=a['definite_zones']))
        # Provenance-bound evaluator label from rendered bounds, checked against declaration.
        target_geo = next(o for o in geo['objects'] if o['name'] == 'target')
        center = np.asarray(target_geo['render_bounds_center_m'])
        half = np.asarray(target_geo['render_bounds_extent_m'])
        camera = case['camera']
        cc = np.array([center[1] - camera['y'], camera['z'] - center[2], center[0] - camera['x']])
        hh = half[[1, 2, 0]]
        actual = classify(cc - hh, cc + hh)
        declared = classify(*bounds(case))
        assert actual['truth'] == declared['truth']
        meta = {k: case[k] for k in ('base_group_id', 'split', 'clip_id', 'frame_in_clip',
                                   'time_s', 'phase', 'layer', 'background', 'layout_relation', 'type_id')}
        meta.update(index=i, id=ids['id'])
        metadata.append(meta)
        labels[case['split']].append(dict(index=i, truth=bool(actual['truth'])))
        sampled_target = sample_native(target.astype(np.float32)).astype(bool)
        sampled_corridor = sample_native(corridor.astype(np.float32)).astype(bool)
        native_support = sum(int((sampled_target & sampled_corridor).ravel()[t['pixel_indices']].sum()) for t in traces)
        admission.append(dict(id=ids['id'], visible_target_pixels=int(target.sum()),
            competing_corridor_pixels=int((corridor & ~target).sum()),
            returned_target_corridor_samples=native_support, truth=bool(actual['truth'])))
        private.append(dict(id=ids['id'], identity=identity, native_sha256=geo['native_sha256'],
            traces=[{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in t.items()} for t in traces]))
        if i % 480 == 0:
            print('MATERIALIZE', i, '/', p['frames'], flush=True)
    passed = all(r['visible_target_pixels'] > 0 and r['competing_corridor_pixels'] < 4 for r in admission)
    write(out / 'source-admission.json', dict(status='PASS' if passed else 'NOT_EVALUABLE', frames=admission))
    assert passed, 'Source failure; no sample exclusions or replacement'
    np.savez_compressed(out / 'observations.npz', ranges=np.stack(values), boxes=boxes)
    write(out / 'identities.json', identities)
    write(out / 'baseline.json', baseline)
    write(out / 'private-lineage.json', private)
    write(out / 'evaluator/metadata.json', metadata)
    for split, rows in labels.items():
        write(out / f'evaluator/{split}-labels.json', rows)
    files = ['observations.npz', 'identities.json', 'baseline.json', 'private-lineage.json',
             'source-admission.json', 'evaluator/metadata.json', 'capture/observations/manifest.json',
             'capture/evaluator/geometry.json'] + [f'evaluator/{s}-labels.json' for s in labels]
    seal(out, 'observation-seal.json', files)
    print('OBSERVATIONS_SEALED', p['frames'], flush=True)


def held(flags, metadata):
    flags = np.asarray(flags, bool)
    answer = flags.copy()
    for i in range(1, len(flags)):
        if metadata[i]['clip_id'] == metadata[i - 1]['clip_id']:
            assert abs(metadata[i]['time_s'] - metadata[i - 1]['time_s'] - .2) < 1e-8
            answer[i] |= flags[i - 1]  # Never recursively propagate a held alert.
    return answer


def fp_segments(flags, truth, metadata, mask):
    count, previous_clip, previous_fp = 0, None, False
    for a, y, r, keep in zip(flags, truth, metadata, mask):
        fp = bool(a and not y and keep)
        count += int(fp and (r['clip_id'] != previous_clip or not previous_fp))
        previous_clip, previous_fp = r['clip_id'], fp
    return count


def threshold_record(threshold, logits, truth, metadata, a):
    truth, a = np.asarray(truth, bool), np.asarray(a, bool)
    b = np.asarray(logits) >= threshold
    core = np.array([r['layout_relation'] != 'BOUNDARY' for r in metadata])
    flags = dict(A_current=a, B_current=b, A_hold=held(a, metadata), B_hold=held(b, metadata))
    checks, summary = {}, {}
    for suffix in ('current', 'hold'):
        aa, bb = flags['A_' + suffix], flags['B_' + suffix]
        checks['retain_Core_' + suffix] = not bool((core & truth & aa & ~bb).any())
        for name, mask in (('Core', core), ('Boundary', ~core)):
            key = name + '_' + suffix
            afp, bfp = int((mask & ~truth & aa).sum()), int((mask & ~truth & bb).sum())
            aseg, bseg = fp_segments(aa, truth, metadata, mask), fp_segments(bb, truth, metadata, mask)
            checks[key + '_FP'] = bfp <= afp
            checks[key + '_segments'] = bseg <= aseg
            summary[key] = dict(A_TP=int((mask & truth & aa).sum()), B_TP=int((mask & truth & bb).sum()),
                A_FP=afp, B_FP=bfp, A_segments=aseg, B_segments=bseg, positives=int((mask & truth).sum()))
    return dict(threshold=float(threshold), admissible=all(checks.values()), checks=checks, summary=summary)


def select_threshold(logits, truth, metadata, baseline):
    logits = np.asarray(logits, dtype=np.float64)
    assert logits.ndim == 1 and len(logits) and np.isfinite(logits).all()
    candidates = np.r_[np.unique(logits), np.nextafter(logits.max(), np.inf)]
    records = [threshold_record(t, logits, truth, metadata, baseline) for t in candidates]
    admissible = [r for r in records if r['admissible']]
    selected = max(admissible, key=lambda r: (r['summary']['Boundary_current']['B_TP'],
        r['summary']['Core_current']['B_TP'], r['threshold'])) if admissible else None
    return selected, records


def metrics(rows):
    # Separate module instance: preserve the frozen original evaluator and its ARMS.
    module_spec = importlib.util.spec_from_file_location('spatial_event_evaluator', HERE / 'full_event_metrics_20260920.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.ARMS = ARMS
    return module.evaluate(rows)


def joined_rows(out, split, logits, threshold):
    meta = [r for r in read(out / 'evaluator/metadata.json') if r['split'] == split]
    label_rows = read(out / f'evaluator/{split}-labels.json')
    assert [r['index'] for r in meta] == [r['index'] for r in label_rows]
    baseline = read(out / 'baseline.json')
    a = np.array([baseline[r['index']]['current'] for r in meta])
    b = np.asarray(logits) >= threshold
    assert len(b) == len(meta)
    predictions = dict(A_current=a, B_current=b, A_hold=held(a, meta), B_hold=held(b, meta))
    rows = []
    for i, (r, label) in enumerate(zip(meta, label_rows)):
        rows.append(dict(**r, truth=label['truth'], logit=float(logits[i]),
            flags={k: bool(v[i]) for k, v in predictions.items()},
            current_unknown={k: bool(baseline[r['index']]['unknown']) for k in ARMS}))
    return rows


def fit(out):
    from spatial_bce_model import encode_rgb, build_inputs, train_fit, predict
    verify(out)
    check_seal(out, 'observation-seal.json')
    write(out / 'fit-start.json', dict(recipe='ONE_BCE_FIT', seed=20260920, test_labels_access=False))
    rows = read(out / 'identities.json')
    for r in rows:
        assert sha(out / r['rgb_path']) == r['rgb_sha256']
    features = encode_rgb([out / r['rgb_path'] for r in rows], CHECKPOINT, out / 'features')
    with np.load(out / 'observations.npz') as data:
        inputs = build_inputs(features, data['ranges'])
    train = read(out / 'evaluator/train-labels.json')
    indices = np.array([r['index'] for r in train], np.int64)
    head = train_fit(inputs, indices, np.array([r['truth'] for r in train], np.float32), out / 'fit')
    metadata = read(out / 'evaluator/metadata.json')
    # Test features are observation-only cache; test logits remain uncomputed.
    for split in ('train', 'dev'):
        idx = np.array([r['index'] for r in metadata if r['split'] == split])
        logits = predict(head, inputs[idx])
        np.save(out / f'{split}-logits.npy', logits)
    seal(out, 'fit-seal.json', ['train-logits.npy', 'dev-logits.npy', 'fit-start.json',
         'features/rgb_features.npy', 'features/feature_receipt.json', 'features/encoder_backend.json'] +
         [p.relative_to(out).as_posix() for p in (out / 'fit').iterdir() if p.is_file()])
    print('ONE_FIT_COMPLETE_TEST_NOT_ACTIVATED', flush=True)


def calibrate(out):
    verify(out)
    check_seal(out, 'fit-seal.json')
    metadata = [r for r in read(out / 'evaluator/metadata.json') if r['split'] == 'dev']
    labels = read(out / 'evaluator/dev-labels.json')
    assert [r['index'] for r in metadata] == [r['index'] for r in labels]
    baseline = read(out / 'baseline.json')
    a = [baseline[r['index']]['current'] for r in metadata]
    logits = np.load(out / 'dev-logits.npy')
    selected, records = select_threshold(logits, [r['truth'] for r in labels], metadata, a)
    write(out / 'development-thresholds.json', records)
    receipt = dict(status='DEV_ADMISSIBLE' if selected else 'DEV_NO_ADMISSIBLE_OPERATING_POINT',
        selected=selected, thresholds=len(records), admissible=sum(r['admissible'] for r in records),
        test_activated=bool(selected), dev_logits_sha256=sha(out / 'dev-logits.npy'))
    write(out / 'operating-point.json', receipt)
    # Logit zero is a prespecified train-fit/development diagnostic, never fallback selection.
    for split in ('train', 'dev'):
        scores = np.load(out / f'{split}-logits.npy')
        rows = joined_rows(out, split, scores, 0.)
        write(out / f'{split}-zero-logit-metrics.json', metrics(rows))
    if selected:
        rows = joined_rows(out, 'dev', logits, selected['threshold'])
        write(out / 'dev-selected-metrics.json', metrics(rows))
    seal(out, 'development-seal.json', ['operating-point.json', 'development-thresholds.json',
        'train-zero-logit-metrics.json', 'dev-zero-logit-metrics.json'] +
         (['dev-selected-metrics.json'] if selected else []))
    print(json.dumps(receipt), flush=True)


def evaluate(out):
    from spatial_bce_model import build_inputs, load_head, predict
    verify(out)
    check_seal(out, 'observation-seal.json')
    check_seal(out, 'fit-seal.json')
    check_seal(out, 'development-seal.json')
    op = read(out / 'operating-point.json')
    assert op['status'] == 'DEV_ADMISSIBLE', 'No test access after development failure'
    write(out / 'test-start.json', dict(threshold=op['selected']['threshold'], test_labels_access=False))
    metadata = [r for r in read(out / 'evaluator/metadata.json') if r['split'] == 'test']
    idx = np.array([r['index'] for r in metadata])
    features = np.load(out / 'features/rgb_features.npy', mmap_mode='r')
    with np.load(out / 'observations.npz') as obs:
        inputs = build_inputs(features[idx], obs['ranges'][idx])
    device = read(out / 'fit/train_backend.json')['selected_device_type']
    head = load_head(out / 'fit/head_last.pt', device=device)
    logits = predict(head, inputs)
    np.save(out / 'test-logits.npy', logits)
    seal(out, 'test-prediction-seal.json', ['test-logits.npy', 'test-start.json', 'operating-point.json'])
    check_seal(out, 'test-prediction-seal.json')
    rows = joined_rows(out, 'test', logits, op['selected']['threshold'])
    report = metrics(rows)
    a = np.array([r['flags']['A_current'] for r in rows])
    gate = threshold_record(op['selected']['threshold'], logits, [r['truth'] for r in rows], metadata, a)
    boundary = gate['summary']['Boundary_current']
    gain = (boundary['B_TP'] - boundary['A_TP']) / boundary['positives']
    checks = dict(**gate['checks'], boundary_current_gain_10pp=gain >= .1 - 1e-12)
    paired = {}
    for suffix in ('current', 'hold'):
        paired[suffix] = {key: [r['id'] for r in rows if condition(r)] for key, condition in {
            'rescued': lambda r: r['truth'] and r['flags']['B_' + suffix] and not r['flags']['A_' + suffix],
            'lost': lambda r: r['truth'] and r['flags']['A_' + suffix] and not r['flags']['B_' + suffix],
            'new_fp': lambda r: not r['truth'] and r['flags']['B_' + suffix] and not r['flags']['A_' + suffix],
            'removed_fp': lambda r: not r['truth'] and r['flags']['A_' + suffix] and not r['flags']['B_' + suffix]}.items()}
    write(out / 'test-frame-results.json', rows)
    write(out / 'test-metrics.json', report)
    write(out / 'result.json', dict(status='PASS' if all(checks.values()) else 'NEGATIVE_CONTROL',
        checks=checks, boundary_current_recall_gain=gain, paired=paired,
        scope='CONTROLLED_NEW_GROUP_DEVELOPMENT_NOT_DEVICE_OR_SAFETY'))
    seal(out, 'evaluation-seal.json', ['test-frame-results.json', 'test-metrics.json', 'result.json'])
    print(json.dumps(read(out / 'result.json')), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('freeze', 'materialize', 'fit', 'calibrate', 'evaluate'))
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to((ROOT / 'artifacts.local').resolve())
    started = time.monotonic()
    globals()[args.stage](args.output)
    print('STAGE_SECONDS', args.stage, round(time.monotonic() - started, 3), flush=True)


if __name__ == '__main__':
    main()
