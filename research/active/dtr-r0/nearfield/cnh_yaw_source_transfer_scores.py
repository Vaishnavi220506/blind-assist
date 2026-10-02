"""Frozen M3 yaw inputs for retained source evaluation units 96000..96095.

Reuse Engine/yaw explicitly without changing imported globals. Read only retained
z1/scene/frame observations; no truth, rendering, training or threshold changes.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_yaw_envelope_scores as E

OUT = E.SS.WORK / 'cnh-yaw-source-transfer-20261002'
UNITS = list(range(96000, 96096))
ANGLES = {'minus2': -2., 'minus1': -1., 'plus1': 1., 'plus2': 2.}
FRAMES = E.FRAMES.copy()
BATCH = E.BATCH
RUN_ID = 'CNH_YAW_SOURCE_TRANSFER_20261002'


def frozen_plan():
    path = OUT / 'PLAN.json'
    runs = Path(__file__).resolve().parents[1] / 'RUNS.md'
    if not path.is_file() or RUN_ID not in runs.read_text(encoding='utf8'):
        raise RuntimeError('Parent PLAN and RUNS pre-registration required')
    return E.SS.sha(path)


def observation_path(unit):
    if unit not in UNITS:
        raise ValueError('Unit not in fixed source evaluation cohort')
    return E.MC.OUT / 'features/evaluation' / f'unit{unit}.npz'


def observation(unit):
    with np.load(observation_path(unit), allow_pickle=False) as data:
        result = {key: data[key] for key in ('z1', 'scene', 'frame')}
    if result['scene'].shape != (640,) or result['frame'].shape != (640,) or len(result['z1']) != 640:
        raise ValueError('Expected exactly 40 configurations x 16 retained frames')
    if not np.isfinite(result['z1']).all():
        raise ValueError('Nonfinite standardized observations')
    for config in range(40):
        if not np.array_equal(result['frame'][result['scene'] == config], np.arange(16)):
            raise ValueError('Retained scene/frame ordering mismatch')
    return result


def inherited_proof():
    path = E.OUT / 'canary.json'
    old = E.read(path)
    selected = {'calib': sorted({row['unit'] for row in old['zero_cases']})}
    if old['status'] != 'PASS' or old['inputs'] != E.identity(selected):
        raise ValueError('Unchanged Engine and old angle/backend evidence must remain valid')
    if len(old['angle_cases']) != 12 or not old['angles_half_bit_exact']:
        raise ValueError('Expected retained twelve exact +/-2 projection comparisons')
    return dict(envelope_canary_sha256=E.SS.sha(path),
        fused_54_canary_sha256=old['inputs']['retained']['fused_canary_v3_sha256'],
        same_angle_backend_proof_reused=True, angle_canary_repeated=False)


def identity(units):
    import cupy
    sources = [__file__, E.__file__, E.SS.__file__, E.NR.__file__, E.MC.__file__,
               E.CP.__file__, E.PR.__file__, E.MP.__file__, E.FP.__file__,
               E.CP.SOURCE / 'cnh_track_a_readout.py']
    caches = {tag: E.MC.OUT / name for tag, name in
              [('early', 'frame_scores_M3_early.npz'), ('late', 'frame_scores_M3.npz')]}
    return dict(source_sha256={str(Path(p)): E.SS.sha(p) for p in sources},
        model_sha256={str(p): E.SS.sha(p) for p in E.model_paths()},
        observation_sha256={str(u): E.SS.sha(observation_path(u)) for u in units},
        zero_cache_sha256={tag: dict(path=str(p), sha256=E.SS.sha(p)) for tag, p in caches.items()},
        inherited_proof=inherited_proof(), cupy_version=cupy.__version__, backend='unchanged fused v3',
        observation_route='MC.OUT/features/evaluation; source 96000..96095')


def canary_cases():
    return [dict(unit=next(u for u in UNITS if u % 3 == mode), mode=mode, config=c, frame=f)
            for mode in range(3) for c in (0, 20) for f in (3, 8, 15)]


def canary():
    plan_sha = frozen_plan()
    path = OUT / 'canary.json'
    if path.exists():
        raise FileExistsError('Preserve existing canary; no silent repetition')
    cases = canary_cases()
    selected = sorted({r['unit'] for r in cases})
    inputs = identity(selected)
    E.create_json(OUT / 'canary_plan.json', dict(cases=cases, inputs=inputs, plan_sha256=plan_sha,
        rule='18 preselected source evaluation zero M3 mean raw logits versus old early/late caches; abs<1e-4',
        angles='Reuse unchanged same-angle/backend numerical proof; no repeated angle canary'))
    report = dict(status='NOT_RUN', inputs=inputs, zero_cases=[], truth_read=False,
        plan_sha256=plan_sha, canary_plan_sha256=E.SS.sha(OUT / 'canary_plan.json'))
    engine = None
    try:
        engine = E.Engine()
        loaded = {u: observation(u) for u in selected}
        expected, voxels = [], []
        with np.load(E.MC.OUT / 'frame_scores_M3_early.npz', allow_pickle=False) as early, \
                np.load(E.MC.OUT / 'frame_scores_M3.npz', allow_pickle=False) as late:
            for case in cases:
                u, c, f = case['unit'], case['config'], case['frame']
                data = loaded[u]
                ids = np.flatnonzero(data['scene'] == c)
                sensor, travel, noisy = E.CP.motion_metadata(u, c)
                transforms = E.CP.relative_transforms(sensor, travel, noisy, f)
                z = data['z1'][ids[max(0, f-7):f+1]]
                voxels.append(engine.project(z, E.yaw(0.) @ transforms))
                expected.append(early[str(u)][c, f-3] if f <= 10 else late[str(u)][c, f-11])
        predicted, expected = engine.score(voxels), np.asarray(expected)
        if expected.shape != (18, 2) or not np.isfinite(expected).all():
            raise ValueError('Invalid retained raw cache values')
        for i, case in enumerate(cases):
            report['zero_cases'].append(dict(**case,
                m3_raw_logit_max_abs=float(np.abs(predicted[i]-expected[i]).max())))
        error = float(np.abs(predicted-expected).max())
        if error >= 1e-4:
            raise ValueError(f'Source zero M3 raw score parity failed: {error}')
        if identity(selected) != inputs or frozen_plan() != plan_sha:
            raise ValueError('Canary input/plan identity changed during execution')
        report.update(status='PASS', zero_max_abs=error, runtime=engine.runtime,
                      angles_canary_repeated=False)
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        E.create_json(path, report)
        raise
    finally:
        if engine is not None:
            engine.close()
    E.create_json(path, report)
    print('PASS source evaluation zero routing canary', report['zero_max_abs'], flush=True)
    return report


def request(prepare=False):
    plan_sha = frozen_plan()
    checked = E.read(OUT / 'canary.json')
    selected = sorted({row['unit'] for row in checked['zero_cases']})
    if checked['status'] != 'PASS' or checked['inputs'] != identity(selected) or checked['plan_sha256'] != plan_sha:
        raise ValueError('Source input routing canary no longer valid')
    value = dict(inputs=identity(UNITS), units=UNITS, angles=ANGLES, frames=FRAMES.tolist(), batch=BATCH,
        plan_sha256=plan_sha, canary_sha256=E.SS.sha(OUT / 'canary.json'),
        role='Frozen M3 five seed raw logits; source evaluation missing absolute yaw angles; no truth or thresholds')
    path = OUT / 'request.json'
    if path.exists():
        if E.read(path) != value:
            raise ValueError('Frozen request changed; cannot resume checkpoints')
    elif prepare:
        E.create_json(path, value)
    else:
        raise RuntimeError('Complete serial --stage prepare before launching inference chunks')
    return value, E.SS.sha(path)


def checkpoint(unit, digest):
    path = OUT / 'checkpoints' / f'unit{unit}.npz'
    receipt = path.with_suffix('.json')
    if not receipt.exists():
        if path.exists():
            raise RuntimeError('Partial checkpoint requires inspection: '+str(path))
        return None
    old = E.read(receipt)
    if old['status'] != 'COMPLETE' or old['request_sha256'] != digest or old['sha256'] != E.SS.sha(path):
        raise ValueError('Checkpoint hash/request mismatch')
    with np.load(path, allow_pickle=False) as z:
        if int(z['unit']) != unit or not np.array_equal(z['frames'], FRAMES):
            raise ValueError('Checkpoint identities changed')
        arrays = {tag: z[tag] for tag in ANGLES}
    if any(x.shape != (40, 13, 2) or not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Incomplete checkpoint')
    return arrays


def chunk_units(chunk):
    k, n = map(int, chunk.split('/'))
    if not 1 <= n <= len(UNITS) or not 0 <= k < n:
        raise ValueError('Invalid chunk')
    return k, n, UNITS[k::n]


def infer(chunk):
    frozen, digest = request()
    k, n, units = chunk_units(chunk)
    (OUT / 'checkpoints').mkdir(parents=True, exist_ok=True)
    engine = None
    begun = time.monotonic()
    try:
        for unit in units:
            if checkpoint(unit, digest) is not None:
                print('reuse', unit, flush=True)
                continue
            if engine is None:
                engine = E.Engine()
            start = time.monotonic()
            data = observation(unit)
            output = {tag: np.full((40, 13, 2), np.nan, np.float32) for tag in ANGLES}
            buffer, positions = [], []
            def flush():
                logits = engine.score(buffer)
                for i, (tag, config, frame) in enumerate(positions):
                    output[tag][config, frame-3] = logits[i]
                buffer.clear()
                positions.clear()
            for config in range(40):
                ids = np.flatnonzero(data['scene'] == config)
                sensor, travel, noisy = E.CP.motion_metadata(unit, config)
                for frame in FRAMES:
                    f = int(frame)
                    transforms = E.CP.relative_transforms(sensor, travel, noisy, f)
                    z = data['z1'][ids[max(0, f-7):f+1]]
                    for tag, bias in ANGLES.items():
                        buffer.append(engine.project(z, E.yaw(bias) @ transforms))
                        positions.append((tag, config, f))
                        if len(buffer) == BATCH:
                            flush()
            if buffer:
                flush()
            if any(not np.isfinite(x).all() for x in output.values()):
                raise ValueError('Missing raw logits in unit output')
            if E.SS.sha(observation_path(unit)) != frozen['inputs']['observation_sha256'][str(unit)]:
                raise ValueError('Unit observation changed during inference')
            path = OUT / 'checkpoints' / f'unit{unit}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, **output, unit=np.asarray(unit), frames=FRAMES)
            E.create_json(path.with_suffix('.json'), dict(status='COMPLETE', unit=unit, request_sha256=digest,
                sha256=E.SS.sha(path), elapsed_s=time.monotonic()-start, runtime=engine.runtime))
            progress = OUT / f'progress_c{k}of{n}.json'
            tmp = progress.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(unit=unit, unit_seconds=time.monotonic()-start,
                                          elapsed_s=time.monotonic()-begun))+'\n', encoding='utf8')
            tmp.replace(progress)
            print('complete', unit, round(time.monotonic()-start, 2), flush=True)
        if identity(UNITS) != frozen['inputs'] or frozen_plan() != frozen['plan_sha256']:
            raise ValueError('Inputs/plan changed during worker execution')
    finally:
        if engine is not None:
            engine.close()


def assemble():
    frozen, digest = request()
    receipt = OUT / 'scores_receipt.json'
    if receipt.exists():
        old = E.read(receipt)
        if old['status'] != 'COMPLETE' or old['request_sha256'] != digest:
            raise ValueError('Existing receipt differs from frozen request')
        for name, checksum in old['output_sha256'].items():
            if E.SS.sha(OUT / name) != checksum:
                raise ValueError('Completed score file changed')
        print('Existing COMPLETE source transfer scores verified', flush=True)
        return old
    arrays = {u: checkpoint(u, digest) for u in UNITS}
    if any(value is None for value in arrays.values()):
        raise RuntimeError('Source checkpoints incomplete')
    hashes = {}
    for tag, bias in ANGLES.items():
        name = f'frame_scores_{tag}_M3.npz'
        with (OUT / name).open('xb') as f:
            np.savez_compressed(f, logit=np.stack([arrays[u][tag] for u in UNITS]),
                                units=UNITS, frames=FRAMES, bias_deg=np.asarray(bias))
        hashes[name] = E.SS.sha(OUT / name)
    result = dict(status='COMPLETE', request_sha256=digest, plan_sha256=frozen['plan_sha256'],
        output_sha256=hashes, shape=[len(UNITS), 40, 13, 2], units=UNITS, angles=ANGLES, frames=FRAMES.tolist(),
        input_sha256=frozen['inputs'], canary_sha256=frozen['canary_sha256'],
        role='Frozen M3 source transfer raw input scores only; no calibration or truth')
    E.create_json(receipt, result)
    print('COMPLETE source transfer raw scores', flush=True)
    return result


def check():
    cases = canary_cases()
    assert len(cases) == 18 and {r['unit'] for r in cases} == {96000, 96001, 96002}
    assert {r['frame'] for r in cases} == {3, 8, 15} and {r['config'] for r in cases} == {0, 20}
    assert len(UNITS) == 96 and not set(UNITS).intersection(sum(E.UNITS.values(), []))
    assert set(ANGLES.values()) == {-2., -1., 1., 2.}
    a, b = chunk_units('0/2')[2], chunk_units('1/2')[2]
    assert len(a) == len(b) == 48 and not set(a).intersection(b) and sorted(a+b) == UNITS
    for u in sorted({r['unit'] for r in cases}):
        data = observation(u)
        assert len(data['frame']) == 640
        assert observation_path(u).parent == E.MC.OUT / 'features/evaluation'
    with np.load(E.MC.OUT / 'frame_scores_M3_early.npz', allow_pickle=False) as early, \
            np.load(E.MC.OUT / 'frame_scores_M3.npz', allow_pickle=False) as late:
        for u in UNITS:
            assert early[str(u)].shape == (40, 8, 2) and late[str(u)].shape == (40, 5, 2)
    np.testing.assert_allclose(E.yaw(2.) @ E.yaw(-2.), np.eye(4), atol=1e-15)
    print('PASS CPU source input/cache routing, 18-case plan, fixed angles, disjoint chunks; no GPU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'canary', 'prepare', 'infer', 'assemble'))
    parser.add_argument('--chunk', default='0/1')
    args = parser.parse_args()
    if args.stage == 'check':
        check()
    elif args.stage == 'canary':
        canary()
    elif args.stage == 'prepare':
        _, digest = request(prepare=True)
        print('Prepared immutable source transfer request', digest, flush=True)
    elif args.stage == 'infer':
        infer(args.chunk)
    else:
        assemble()
