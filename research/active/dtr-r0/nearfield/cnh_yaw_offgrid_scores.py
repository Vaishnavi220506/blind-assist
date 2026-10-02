"""Fixed off-grid M3 input scores using the retained envelope implementation.

No global monkeypatching, truth access, recalibration, rendering or training.
Each split adds only -1.5,-0.5,+0.5,+1.5 degree body-frame alignment inputs.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import cnh_yaw_envelope_scores as E

OUT = E.SS.WORK / 'cnh-yaw-offgrid-20261002'
UNITS = {split: list(units) for split, units in E.UNITS.items()}
ANGLES = {'minus1p5': -1.5, 'minus0p5': -.5, 'plus0p5': .5, 'plus1p5': 1.5}
FRAMES = E.FRAMES.copy()
BATCH = E.BATCH


def inherited_proof():
    path = E.OUT / 'canary.json'
    old = E.read(path)
    selected = {'calib': sorted({row['unit'] for row in old['zero_cases']})}
    if old['status'] != 'PASS' or old['inputs'] != E.identity(selected):
        raise ValueError('Unchanged M3 zero model path and old projector canary must remain valid')
    return dict(envelope_canary_sha256=E.SS.sha(path),
        zero_model_canary_reused=True, old_zero_canary_not_repeated=True,
        old_fused_54_canary_sha256=old['inputs']['retained']['fused_canary_v3_sha256'])


def identity(units_by_split):
    inputs = E.identity(units_by_split)
    inputs['source_sha256'][str(Path(__file__))] = E.SS.sha(__file__)
    inputs['inherited_numerical_proof'] = inherited_proof()
    inputs['wrapper'] = 'Explicit Engine/observation/yaw calls; no old globals modified'
    return inputs


def canary_cases():
    return [dict(unit=next(u for u in UNITS['calib'] if u % 3 == mode), mode=mode, config=c, frame=8,
                 angle_tag=tag, bias_deg=bias)
            for mode in range(3) for c in (0, 20) for tag, bias in ANGLES.items()]


def canary():
    if not (OUT / 'PLAN.json').is_file():
        raise RuntimeError('Parent scoring PLAN/RUNS must exist before the new mechanical canary')
    path = OUT / 'canary.json'
    if path.exists():
        raise FileExistsError('Keep existing canary evidence; do not rerun silently')
    cases = canary_cases()
    selected = {'calib': sorted({case['unit'] for case in cases})}
    inputs = identity(selected)
    E.create_json(OUT / 'canary_plan.json', dict(cases=cases, inputs=inputs,
        rule='24 preselected projection pairs at frame8; original and unchanged fused-v3 float16 bits exactly equal',
        model_validation='Inherited frozen M3/zero numerical canary, no new model inference or truth',
        prior_proof=inputs['inherited_numerical_proof']))
    torch, original, fused = None, None, None
    report = dict(status='NOT_RUN', inputs=inputs, cases=[], model_inference=False, truth_read=False,
                  canary_plan_sha256=E.SS.sha(OUT / 'canary_plan.json'))
    try:
        import torch as torch_runtime
        torch = torch_runtime
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA projector required')
        original, fused = E.MP.BatchedProjector(), E.FP.FusedProjector()
        loaded = {u: E.observation('calib', u) for u in selected['calib']}
        for number, case in enumerate(cases):
            data = loaded[case['unit']]
            ids = np.flatnonzero(data['scene'] == case['config'])
            sensor, travel, noisy = E.CP.motion_metadata(case['unit'], case['config'])
            f = case['frame']
            transform = E.yaw(case['bias_deg']) @ E.CP.relative_transforms(sensor, travel, noisy, f)
            z = data['z1'][ids[max(0, f-7):f+1]]
            order = [('original', original), ('fused', fused)]
            if number % 2:
                order.reverse()
            values, elapsed = {}, {}
            for backend, projector in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                values[backend] = projector.sequence(z, transform).cpu().numpy().astype(np.float16)
                elapsed[backend] = time.perf_counter()-start
                if not np.isfinite(values[backend]).all():
                    raise ValueError('Nonfinite canary voxel values')
            a, b = values['original'], values['fused']
            mismatch = int(np.count_nonzero(a.view(np.uint16) != b.view(np.uint16)))
            report['cases'].append(dict(**case, half_bit_mismatches=mismatch,
                max_abs=float(np.abs(a.astype(float)-b.astype(float)).max()),
                order=[name for name, _ in order], elapsed_s=elapsed))
            if mismatch:
                raise ValueError(f'Offgrid half parity failed: {case}, count {mismatch}')
        if identity(selected) != inputs:
            raise ValueError('Canary input identity changed')
        report.update(status='PASS', runtime=dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
            cuda=torch.version.cuda, cupy=fused.cp.__version__, threads=torch.get_num_threads(), tf32=False),
            inherited_proof=inputs['inherited_numerical_proof'], zero_model_repeated=False)
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        E.create_json(path, report)
        raise
    finally:
        original = fused = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    E.create_json(path, report)
    print('PASS offgrid mechanical canary: 24 exact half pairs; no model/cohort scoring', flush=True)
    return report


def request(prepare=False):
    if not (OUT / 'PLAN.json').exists():
        raise RuntimeError('Frozen scoring plan required')
    checked = E.read(OUT / 'canary.json')
    selected = {'calib': sorted({row['unit'] for row in checked['cases']})}
    if checked['status'] != 'PASS' or checked['inputs'] != identity(selected):
        raise ValueError('Offgrid numerical canary is not valid for current inputs')
    value = dict(inputs=identity(UNITS), units=UNITS, angles=ANGLES, frames=FRAMES.tolist(), batch=BATCH,
        plan_sha256=E.SS.sha(OUT / 'PLAN.json'), canary_sha256=E.SS.sha(OUT / 'canary.json'),
        role='Frozen M3 raw logits only; missing absolute offgrid angles; thresholds/labels never read')
    path = OUT / 'request.json'
    if path.exists():
        if E.read(path) != value:
            raise ValueError('Frozen request changed; cannot resume these checkpoints')
    elif prepare:
        E.create_json(path, value)
    else:
        raise RuntimeError('Complete serial --stage prepare before launching inference chunks')
    return value, E.SS.sha(path)


def checkpoint(split, unit, digest):
    path = OUT / 'checkpoints' / split / f'unit{unit}.npz'
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


def infer(split, chunk):
    frozen, digest = request()
    k, n = map(int, chunk.split('/'))
    if n < 1 or not 0 <= k < n:
        raise ValueError('Invalid chunk')
    (OUT / 'checkpoints' / split).mkdir(parents=True, exist_ok=True)
    engine = None
    begun = time.monotonic()
    try:
        for unit in UNITS[split][k::n]:
            if checkpoint(split, unit, digest) is not None:
                print('reuse', split, unit, flush=True)
                continue
            if engine is None:
                engine = E.Engine()
            start = time.monotonic()
            data = E.observation(split, unit)
            output = {tag: np.full((40, 13, 2), np.nan, np.float32) for tag in ANGLES}
            buffer, positions = [], []
            def flush():
                logits = engine.score(buffer)
                for i, (tag, config, f) in enumerate(positions):
                    output[tag][config, f-3] = logits[i]
                buffer.clear(); positions.clear()
            for config in range(40):
                ids = np.flatnonzero(data['scene'] == config)
                sensor, travel, noisy = E.CP.motion_metadata(unit, config)
                for f in FRAMES:
                    transforms = E.CP.relative_transforms(sensor, travel, noisy, int(f))
                    z = data['z1'][ids[max(0, f-7):f+1]]
                    for tag, bias in ANGLES.items():
                        buffer.append(engine.project(z, E.yaw(bias) @ transforms))
                        positions.append((tag, config, int(f)))
                        if len(buffer) == BATCH:
                            flush()
            if buffer:
                flush()
            if any(not np.isfinite(x).all() for x in output.values()):
                raise ValueError('Missing raw logits in unit output')
            if E.SS.sha(E.observation_path(split, unit)) != frozen['inputs']['observation_sha256'][split][str(unit)]:
                raise ValueError('Retained unit observation changed during inference')
            path = OUT / 'checkpoints' / split / f'unit{unit}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, **output, unit=np.asarray(unit), frames=FRAMES)
            E.create_json(path.with_suffix('.json'), dict(status='COMPLETE', split=split, unit=unit, request_sha256=digest,
                sha256=E.SS.sha(path), elapsed_s=time.monotonic()-start, runtime=engine.runtime))
            progress = OUT / f'progress_{split}_c{k}of{n}.json'
            tmp = progress.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(split=split, unit=unit, unit_seconds=time.monotonic()-start,
                                           elapsed_s=time.monotonic()-begun))+'\n', encoding='utf8')
            tmp.replace(progress)
            print('complete', split, unit, round(time.monotonic()-start, 2), flush=True)
        if identity(UNITS) != frozen['inputs']:
            raise ValueError('Inputs changed during worker execution')
    finally:
        if engine is not None:
            engine.close()


def assemble(split):
    frozen, digest = request()
    receipt = OUT / f'scores_receipt_{split}.json'
    if receipt.exists():
        old = E.read(receipt)
        if old['status'] != 'COMPLETE' or old['request_sha256'] != digest:
            raise ValueError('Existing receipt differs from frozen request')
        for name, checksum in old['output_sha256'].items():
            if E.SS.sha(OUT / name) != checksum:
                raise ValueError('Completed score file changed')
        print('Existing COMPLETE offgrid scores verified', split, flush=True)
        return old
    units = UNITS[split]
    arrays = {u: checkpoint(split, u, digest) for u in units}
    if any(value is None for value in arrays.values()):
        raise RuntimeError(f'{split} checkpoints incomplete')
    hashes = {}
    for tag, bias in ANGLES.items():
        name = f'frame_scores_{split}_{tag}_M3.npz'
        with (OUT / name).open('xb') as f:
            np.savez_compressed(f, logit=np.stack([arrays[u][tag] for u in units]),
                                units=units, frames=FRAMES, bias_deg=np.asarray(bias))
        hashes[name] = E.SS.sha(OUT / name)
    result = dict(status='COMPLETE', split=split, request_sha256=digest, plan_sha256=frozen['plan_sha256'],
        output_sha256=hashes, shape=[len(units), 40, 13, 2], units=units, angles=ANGLES, frames=FRAMES.tolist(),
        input_sha256=frozen['inputs'], canary_sha256=frozen['canary_sha256'],
        role='Fixed M3 offgrid raw input scores only, no calibration or labels')
    E.create_json(receipt, result)
    print('COMPLETE offgrid raw scores', split, flush=True)
    return result


def check():
    cases = canary_cases()
    assert len(cases) == 24 and {r['mode'] for r in cases} == {0, 1, 2}
    assert {r['frame'] for r in cases} == {8} and {r['config'] for r in cases} == {0, 20}
    assert set(ANGLES.values()) == {-1.5, -.5, .5, 1.5}
    assert UNITS == E.UNITS and UNITS is not E.UNITS
    assert OUT != E.OUT and E.ANGLES['target'] == {'minus2': -2., 'plus2': 2.}
    np.testing.assert_allclose(E.yaw(1.5) @ E.yaw(-1.5), np.eye(4), atol=1e-15)
    print('PASS offgrid metadata-only 24-case plan, fixed angles and explicit reuse without old globals changing; no GPU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'canary', 'prepare', 'infer', 'assemble'))
    parser.add_argument('--split', choices=tuple(UNITS))
    parser.add_argument('--chunk', default='0/1')
    args = parser.parse_args()
    if args.stage == 'check':
        check()
    elif args.stage == 'canary':
        canary()
    elif args.stage == 'prepare':
        _, digest = request(prepare=True)
        print('Prepared immutable offgrid request', digest, flush=True)
    else:
        if args.split is None:
            parser.error('--split required for infer/assemble')
        infer(args.split, args.chunk) if args.stage == 'infer' else assemble(args.split)
