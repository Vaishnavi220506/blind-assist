"""Predeclared paired rotation-seeded LK candidate; no benefit claimed before run.

Granted camera rotation initializes LK only. Translation/depth/visibility are
evaluator-only. This is correspondence readiness, not incremental ToF or alarm
performance. prepare freezes a geometry-selected cohort before RGB tracking
outcomes (the supplier already decoded image dimensions for input readiness).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HYP = ROOT / 'artifacts.local/datasets/hypersim-ba-nfo'
META = ROOT / 'artifacts.local/work/ba-nfo-20260919'
HELPER = ROOT / 'artifacts.local/work/cnh-rgb-pair-correspondence-20261002/run.py'
OLD_PLAN = HELPER.with_name('PLAN.json')
OUT = ROOT / 'artifacts.local/work/cnh-rgb-pose-seeded-flow-20261002'
METHOD = dict(maxCorners=1000, qualityLevel=.01, minDistance=8, blockSize=7,
              useHarrisDetector=False, winSize=[31, 31], maxLevel=4,
              iterations=30, epsilon=.01, minEigThreshold=.0001,
              fb_max_px=1., correct_max_px=3.)
PROPOSAL = {
    'question': 'Does granted rotation-only LK initialization increase correct near-field correspondences without increasing wrong accepted ones?',
    'role': 'Synthetic consumed Development, metadata/depth-selected train cohort; not confirmation.',
    'selection': 'Exactly first up-to-five distinct-scene selected_diagnostic_cohort from supply result, frozen before RGB outcomes. Require at least three. No replacement after outcome access.',
    'arms': {'A': 'Old Shi-Tomasi plus LK, both directions with no initial flow.',
             'B': 'Same detected points and parameters; rotation-only projection initializes forward LK; inverse rotation of forward endpoints initializes backward LK. OPTFLOW_USE_INITIAL_FLOW.'},
    'method': METHOD,
    'inputs': 'Method: two grayscale JPEGs, exact M_cam_from_uv, granted R0 and R1. No depth, translation, instance, target mask or evaluator-selected points.',
    'evaluator': 'Inherited float64 ray/project/evaluate_points. Nearest-pixel radial depth; visibility depth agreement <=max(5cm,2% target range). Not exact surface identity.',
    'primary': 'All detected corners with source AND projected destination radial range in [1.2,2.1]m and GT visible proxy. Correct=accepted and endpoint error<=3px; wrong=accepted and error>3px; rejected retained as a failure count.',
    'support_rule': 'Descriptive SUPPORT iff A primary correct>0, B primary correct>=1.10*A primary correct and B primary wrong<=A primary wrong. A zero or no primary visible points => NOT_EVALUABLE. Otherwise NOT_SUPPORTED. Report pair heterogeneity, losses and all denominators; no posthoc exclusions.',
    'secondary': 'All-range visible correct/wrong/rejected; accepted but GT invisible or undefined separately; source-range-only near counts and actual common-near visible counts. No primary substitution.',
    'boundaries': 'Ideal rotation input is not estimated pose, deployed sensing, hardware, occupancy, incremental ToF accumulation or timely collision warning evidence.',
    'compute': 'Local CPU, cv2 one thread; <=10 JPEGs and small projections. No downloads, renders, training or ToF synthesis.',
    'stop': 'One frozen cohort, one A/B comparison; no parameter sweep or old52 static-edge mechanisms. Missing/corrupt inputs abort, never silently replace pairs.',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def helper():
    spec = importlib.util.spec_from_file_location('prior_pair_projection', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paths_for(pair):
    scene, cam, f0, f1 = pair
    base = HYP / scene
    return {
        'rgb': [base / 'images' / f'scene_{cam}_final_preview' / f'frame.{f:04d}.tonemap.jpg' for f in (f0, f1)],
        'depth': [base / 'images' / f'scene_{cam}_geometry_hdf5' / f'frame.{f:04d}.depth_meters.hdf5' for f in (f0, f1)],
        'indices': base / '_detail' / cam / 'camera_keyframe_frame_indices.hdf5',
        'orientations': base / '_detail' / cam / 'camera_keyframe_orientations.hdf5',
        'positions': base / '_detail' / cam / 'camera_keyframe_positions.hdf5',
        'scale': base / '_detail' / 'metadata_scene.csv',
    }


def input_paths(pairs):
    values = [Path(__file__), HELPER, OLD_PLAN,
              META / 'metadata_camera_parameters.csv', META / 'official_split.csv',
              META / 'hypersim-selection.json']
    for pair in pairs:
        for value in paths_for(pair).values():
            values.extend(value if isinstance(value, list) else [value])
    return sorted(set(values), key=str)


def prepare(out, supply_plan, supply_result):
    supply = json.loads(supply_result.read_text(encoding='utf-8'))
    # Explicit shared schema; no fallback selection from RGB-derived artifacts.
    assert supply['plan_sha256'] == digest(supply_plan)
    by_id = {row['id']: row for row in supply['pairs']}
    selected = [by_id[key] for key in supply['selected_diagnostic_cohort'][:5]]
    assert all(row['data_ready'] for row in selected)
    pairs = [[row['metadata']['scene'], row['metadata']['camera'], *row['metadata']['frames']]
             for row in selected]
    assert 3 <= len(pairs) <= 5, 'Require >=3 metadata/depth-selected pairs.'
    assert len({p[0] for p in pairs}) == len(pairs), 'Distinct scenes required.'
    assert all(len(p) == 4 and isinstance(p[2], int) and isinstance(p[3], int) for p in pairs)
    roles = json.loads((META / 'hypersim-selection.json').read_text())['families']
    official = {(r['scene_name'], r['camera_name'], int(r['frame_id'])): r['split_partition_name']
                for r in csv.DictReader((META / 'official_split.csv').open())}
    assert all(roles[p[0][:6]] == 'train' and
               all(official[(p[0], p[1], f)] == 'train' for f in p[2:]) for p in pairs)
    paths = input_paths(pairs) + [supply_plan, supply_result]
    manifest = [{'path': str(p), 'sha256': digest(p)} for p in paths]
    out.mkdir(parents=True, exist_ok=True)
    assert not any((out / name).exists() for name in ('PLAN.json', 'RUN_STARTED.json', 'result.json'))
    write_new(out / 'PLAN.json', dict(PROPOSAL, pairs=pairs, supply_selected_rows=selected, manifest=manifest,
                                    imported_helper_functions=['ray', 'project', 'evaluate_points', 'inside'],
                                    prepared_before_rgb_outcomes=True))
    print(json.dumps({'stage': 'prepared', 'pairs': pairs, 'plan': str(out / 'PLAN.json')}))


def initial_projection(points, M, rotation, shape, ev):
    h, w = shape
    camera = ev.ray(points, M, w, h) @ rotation
    projected = ev.project(camera, M, w, h)
    good = np.isfinite(projected).all(axis=1) & (camera[:, 2] < 0)
    # Unprojectable seeds cannot help LK. Keep a finite input for OpenCV, but
    # explicitly reject these outputs; preserve every source in the ledger.
    return np.where(good[:, None], projected, points).astype(np.float32), good


def track(a, b, points, M, rotation, seeded, ev):
    if not len(points):
        return dict(dest=points.copy(), back=points.copy(), accepted=np.zeros(0, bool),
                    forward=np.zeros(0, bool), both=np.zeros(0, bool), seed_valid=np.zeros(0, bool))
    kwargs = dict(winSize=tuple(METHOD['winSize']), maxLevel=METHOD['maxLevel'],
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            METHOD['iterations'], METHOD['epsilon']),
                  minEigThreshold=METHOD['minEigThreshold'])
    p = points.astype(np.float32).reshape(-1, 1, 2)
    seed, good = initial_projection(points, M, rotation, a.shape, ev) if seeded else (None, np.ones(len(p), bool))
    flags = cv2.OPTFLOW_USE_INITIAL_FLOW if seeded else 0
    q, ok1, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None if seed is None else seed.reshape(-1, 1, 2), flags=flags, **kwargs)
    q = q.reshape(-1, 2)
    finite = np.isfinite(q).all(axis=1)
    safe_q = np.where(finite[:, None], q, points).astype(np.float32)
    seed_back, good_back = initial_projection(safe_q, M, rotation.T, a.shape, ev) if seeded else (None, np.ones(len(p), bool))
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(b, a, safe_q.reshape(-1, 1, 2),
                                         None if seed_back is None else seed_back.reshape(-1, 1, 2), flags=flags, **kwargs)
    back = back.reshape(-1, 2)
    h, w = a.shape
    forward = ok1.ravel().astype(bool) & ev.inside(q, w, h)
    both = forward & ok2.ravel().astype(bool) & ev.inside(back, w, h)
    accepted = both & good & good_back & (np.linalg.norm(back - points, axis=1) <= METHOD['fb_max_px'])
    return dict(dest=q, back=back, forward=forward, both=both, accepted=accepted, seed_valid=good & good_back)


def preflight():
    """Synthetic identity-rotation fixture; no dataset images are read."""
    ev = helper()
    rng = np.random.default_rng(20261002)
    a = rng.integers(0, 256, (96, 128), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (5, 5), 0)
    b = np.roll(a, 2, axis=1)
    points = cv2.goodFeaturesToTrack(a, 80, .01, 8, blockSize=7).reshape(-1, 2)
    M = np.array([[1., .03, 0], [0, .75, 0], [0, 0, -1.]])
    A = track(a, b, points, M, np.eye(3), False, ev)
    B = track(a, b, points, M, np.eye(3), True, ev)
    assert np.array_equal(A['accepted'], B['accepted'])
    assert np.allclose(A['dest'], B['dest'], atol=1e-4, rtol=0)
    assert np.allclose(A['back'], B['back'], atol=1e-4, rtol=0)
    return dict(identity_rotation_fixture='PASS', points=len(points))


def hdf(path):
    with h5py.File(path, 'r') as stream:
        return stream['dataset'][:]


def counts(mask, accepted, correct):
    wrong = accepted & ~correct
    return dict(denominator=int(mask.sum()), correct=int((mask & correct).sum()),
                wrong=int((mask & wrong).sum()), rejected=int((mask & ~accepted).sum()))


def finite_list(value):
    a = np.asarray(value)
    return a.tolist() if np.isfinite(a).all() else None


def run_pair(pair, cameras, ev):
    paths = paths_for(pair)
    scene, cam, f0, f1 = pair
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths['rgb']]
    assert all(a is not None for a in imgs) and imgs[0].shape == imgs[1].shape
    indices, orientations = hdf(paths['indices']), hdf(paths['orientations'])
    ids = [np.flatnonzero(indices == f) for f in (f0, f1)]
    assert all(len(i) == 1 for i in ids)
    i, j = [int(x[0]) for x in ids]
    R0, R1 = [np.asarray(orientations[k], dtype=np.float64) for k in (i, j)]
    assert all(np.max(np.abs(r.T @ r - np.eye(3))) < 1e-4 and abs(np.linalg.det(r) - 1) < 1e-4 for r in (R0, R1))
    row = cameras[scene]
    M = np.array([[float(row[f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)])
    raw = cv2.goodFeaturesToTrack(imgs[0], **{k: METHOD[k] for k in ('maxCorners', 'qualityLevel', 'minDistance', 'blockSize', 'useHarrisDetector')})
    p = np.empty((0, 2), np.float32) if raw is None else raw.reshape(-1, 2)
    # Both arms finish before any evaluator depth or translation is loaded.
    arms = {name: track(*imgs, p, M, R0.T @ R1, seeded, ev) for name, seeded in (('A', False), ('B', True))}
    depth0, depth1 = [hdf(x) for x in paths['depth']]
    assert depth0.shape == depth1.shape == imgs[0].shape
    positions = hdf(paths['positions'])
    scale = float(next(r['parameter_value'] for r in csv.DictReader(paths['scale'].open()) if r['parameter_name'] == 'meters_per_asset_unit'))
    gt = ev.evaluate_points(p, depth0, depth1, M, R0, R1, positions[i] * scale, positions[j] * scale)
    source_near = gt['valid'] & (gt['range0'] >= 1.2) & (gt['range0'] <= 2.1)
    near = source_near & gt['visible'] & (gt['range1'] >= 1.2) & (gt['range1'] <= 2.1)
    output = dict(pair=pair, detected=len(p), source_near_detected=int(source_near.sum()),
                  near_visible_detected=int(near.sum()), all_visible_detected=int(gt['visible'].sum()), arms={})
    for name, arm in arms.items():
        arm['error'] = np.linalg.norm(arm['dest'] - gt['expected'], axis=1)
        arm['correct'] = arm['accepted'] & (arm['error'] <= METHOD['correct_max_px'])
        output['arms'][name] = dict(near=counts(near, arm['accepted'], arm['correct']),
                                   all_visible=counts(gt['visible'], arm['accepted'], arm['correct']),
                                   accepted_total=int(arm['accepted'].sum()),
                                   accepted_not_gt_visible=int((arm['accepted'] & ~gt['visible']).sum()),
                                   invalid_rotation_seed=int((~arm['seed_valid']).sum()))
    for key, mask in (('near', near), ('all_visible', gt['visible'])):
        output[key + '_paired'] = dict(correct_gained=int((mask & arms['B']['correct'] & ~arms['A']['correct']).sum()),
                                      correct_lost=int((mask & arms['A']['correct'] & ~arms['B']['correct']).sum()))
    ledger = []
    for k in range(len(p)):
        record = dict(source=p[k].tolist(), source_range_m=float(gt['range0'][k]) if np.isfinite(gt['range0'][k]) else None,
                      target_range_m=float(gt['range1'][k]) if np.isfinite(gt['range1'][k]) else None,
                      gt_visible=bool(gt['visible'][k]), primary_near=bool(near[k]), gt_projected=finite_list(gt['expected'][k]))
        record['arms'] = {name: dict(endpoint=finite_list(a['dest'][k]), backward=finite_list(a['back'][k]),
                                    accepted=bool(a['accepted'][k]), forward=bool(a['forward'][k]),
                                    both=bool(a['both'][k]), seed_valid=bool(a['seed_valid'][k]),
                                    error_px=float(a['error'][k]) if np.isfinite(a['error'][k]) else None)
                          for name, a in arms.items()}
        ledger.append(record)
    return output, ledger


def run(out):
    plan_path = out / 'PLAN.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert all(digest(row['path']) == row['sha256'] for row in plan['manifest']), 'Frozen input/code changed.'
    assert plan['method'] == METHOD and plan['support_rule'] == PROPOSAL['support_rule']
    assert 3 <= len(plan['pairs']) <= 5
    assert not any((out / x).exists() for x in ('RUN_STARTED.json', 'result.json', 'point-ledger.json', 'REPORT.md'))
    write_new(out / 'RUN_STARTED.json', dict(plan_sha256=digest(plan_path), started_unix=time.time()))
    start = time.perf_counter()
    fixture = preflight()
    cameras = {r['scene_name']: r for r in csv.DictReader((META / 'metadata_camera_parameters.csv').open())}
    ev = helper()
    results, ledgers = [], []
    for pair in plan['pairs']:
        result, ledger = run_pair(pair, cameras, ev)
        results.append(result)
        ledgers.append(dict(pair=pair, points=ledger))
    aggregate = {name: {key: sum(r['arms'][name]['near'][key] for r in results)
                        for key in ('denominator', 'correct', 'wrong', 'rejected')} for name in ('A', 'B')}
    A, B = aggregate['A'], aggregate['B']
    verdict = ('NOT_EVALUABLE' if A['correct'] == 0 else
               'SUPPORT' if B['correct'] >= 1.1 * A['correct'] and B['wrong'] <= A['wrong'] else 'NOT_SUPPORTED')
    result = dict(verdict=verdict, aggregate_primary=aggregate, pairs=results, fixture=fixture,
                  scope=plan['boundaries'], plan_sha256=digest(plan_path),
                  wall_seconds=time.perf_counter() - start, opencv_version=cv2.__version__)
    write_new(out / 'point-ledger.json', ledgers)
    write_new(out / 'result.json', result)
    lines = ['# Rotation-seeded RGB correspondence diagnostic', '',
             'Synthetic consumed Development; exact rotation granted. No ToF increment or alarm claim.', '',
             f'Descriptive verdict: **{verdict}**. Primary range: source and destination both 1.2–2.1m, GT visible proxy.', '',
             '| Pair | Detected | Source near | Near visible | A correct/wrong/rejected | B correct/wrong/rejected | Gained/lost correct |',
             '|---|---:|---:|---:|---|---|---|']
    for r in results:
        c = [r['arms'][name]['near'] for name in ('A', 'B')]
        cells = ['/'.join(str(x[k]) for k in ('correct', 'wrong', 'rejected')) for x in c]
        paired = r['near_paired']
        lines.append(f"| {r['pair']} | {r['detected']} | {r['source_near_detected']} | {r['near_visible_detected']} | {cells[0]} | {cells[1]} | {paired['correct_gained']}/{paired['correct_lost']} |")
    lines += ['', f'Aggregate primary: `{aggregate}`.', '',
              'All-range secondary, accepted but not GT-visible and every source corner including failures are retained in result.json and point-ledger.json. No pair was excluded for poor RGB results.', '',
              'Visibility uses nearest-pixel depth agreement, not exact identity. Source geometry selected the cohort; it never selected or filtered method corners. Camera translation and depth entered only the evaluator after both arms finished. Ideal rotation availability and sparse corner coverage are unresolved deployment assumptions.', '',
              f"Local CPU wall time: {result['wall_seconds']:.3f}s; OpenCV one thread. No owned background process."]
    with (out / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('proposal', 'preflight', 'prepare', 'run'), required=True)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--supply-plan', type=Path)
    parser.add_argument('--supply-result', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == 'proposal':
        print(json.dumps(PROPOSAL, indent=2))
    elif args.stage == 'preflight':
        print(json.dumps(preflight()))
    elif args.stage == 'prepare':
        if args.supply_plan is None or args.supply_result is None:
            parser.error('prepare requires --supply-plan and --supply-result')
        prepare(args.out, args.supply_plan, args.supply_result)
    else:
        run(args.out)


if __name__ == '__main__':
    main()
