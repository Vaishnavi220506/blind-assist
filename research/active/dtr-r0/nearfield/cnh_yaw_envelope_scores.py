"""M3 evidence for one fixed three-point yaw envelope; no truth or scoring gate.

Only missing angle inputs are reprojected from retained z1=r/sqrt(v): calib
95000..95047 at -2,-1,+1,+2 degrees, target94000..94095 at -2,+2. Zero and
target +/-1 inputs remain the already validated caches. No rendering/training.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_near_range as NR
import cnh_margin_confirm as MC
import cnh_cvr_pilot as CP
import cnh_cvr_projection as PR
import cnh_cvr_v2_materialize as MP
import cnh_cvr_fused_projector as FP

OUT = SS.WORK / 'cnh-yaw-envelope-20261002'
EXTRINSIC = SS.WORK / 'cnh-sequence-extrinsic-yaw-20261002'
TRANSFER = SS.WORK / 'cnh-sequence-transfer-20261002'
MARGIN = SS.WORK / 'cnh-margin-labels-20261002'
UNITS = {'calib': list(range(95000, 95048)), 'target': list(range(94000, 94096))}
ANGLES = {'calib': {'minus2': -2., 'minus1': -1., 'plus1': 1., 'plus2': 2.},
          'target': {'minus2': -2., 'plus2': 2.}}
FRAMES = np.arange(3, 16)
BATCH = 32


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def create_json(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def model_paths():
    return [MARGIN / 'models/M3' / f'model_seed{seed}.pt' for seed in range(5)]


def observation_path(split, unit):
    if unit not in UNITS[split]:
        raise ValueError('Unit is not in the fixed split')
    return ((MC.OUT / 'features/calib') if split == 'calib' else (NR.OUT / 'features/evaluation')) / f'unit{unit}.npz'


def retained_caches():
    """Bind old observation scores/engineering receipts, never evaluation truth."""
    fused_receipt = EXTRINSIC / 'fused_canary_v3.json'
    check = read(fused_receipt)
    source = next(v for k, v in check['inputs']['source_sha256'].items() if Path(k).name == Path(FP.__file__).name)
    if check['status'] != 'PASS' or source != SS.sha(FP.__file__):
        raise ValueError('The unchanged fused v3 backend must retain its 54-case PASS')
    old = read(EXTRINSIC / 'scores_receipt.json')
    zero = read(TRANSFER / 'scores_receipt.json')
    if old['status'] != 'COMPLETE' or zero['status'] != 'COMPLETE':
        raise ValueError('Retained target signed/zero scores must be COMPLETE')
    references = {'calib_early': MC.OUT / 'frame_scores_M3_early.npz',
                  'calib_late': MC.OUT / 'frame_scores_M3.npz',
                  'target_zero': TRANSFER / 'frame_scores_M3.npz',
                  'target_minus1': EXTRINSIC / 'frame_scores_minus1_M3.npz',
                  'target_plus1': EXTRINSIC / 'frame_scores_plus1_M3.npz'}
    hashes = {key: dict(path=str(path), sha256=SS.sha(path)) for key, path in references.items()}
    if hashes['target_zero']['sha256'] != zero['output_sha256']['M3']:
        raise ValueError('Retained target zero cache changed')
    for tag in ('minus1', 'plus1'):
        if hashes[f'target_{tag}']['sha256'] != old['output_sha256'][f'frame_scores_{tag}_M3.npz']:
            raise ValueError('Retained target signed cache changed')
    return dict(caches=hashes, fused_canary_v3_sha256=SS.sha(fused_receipt),
                extrinsic_scores_receipt_sha256=SS.sha(EXTRINSIC / 'scores_receipt.json'),
                transfer_scores_receipt_sha256=SS.sha(TRANSFER / 'scores_receipt.json'))


def identity(units_by_split):
    import cupy
    paths = [__file__, SS.__file__, NR.__file__, MC.__file__, CP.__file__, PR.__file__, MP.__file__, FP.__file__,
             CP.SOURCE / 'cnh_track_a_readout.py']
    return dict(source_sha256={str(Path(path)): SS.sha(path) for path in paths},
        model_sha256={str(path): SS.sha(path) for path in model_paths()},
        observation_sha256={split: {str(u): SS.sha(observation_path(split, u)) for u in units}
                           for split, units in units_by_split.items()},
        cupy_version=cupy.__version__, backend='unchanged fused v3', retained=retained_caches())


def yaw(degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    matrix = np.eye(4)
    matrix[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return matrix


def observation(split, unit):
    with np.load(observation_path(split, unit), allow_pickle=False) as data:
        result = {key: data[key] for key in ('z1', 'scene', 'frame')}
    if not np.isfinite(result['z1']).all():
        raise ValueError('Nonfinite standardized observations')
    for config in range(40):
        if not np.array_equal(result['frame'][result['scene'] == config], np.arange(16)):
            raise ValueError('Retained 40x16 scene/frame ordering mismatch')
    return result


class Engine:
    def __init__(self):
        import torch
        self.torch, self.nets, self.projector, self.masks = torch, [], None, None
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required')
        try:
            self.projector = FP.FusedProjector()
            self.masks = torch.as_tensor(PR.query_masks(), dtype=torch.float32, device='cuda')
            for path in model_paths():
                net = CP.CVR().cuda()
                net.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
                self.nets.append(net.eval())
            self.runtime = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                cupy=self.projector.cp.__version__, threads=torch.get_num_threads(), tf32=False, batch=BATCH, m3_seeds=5)
        except BaseException:
            self.close()
            raise

    def project(self, z, transforms):
        x = self.projector.sequence(z, transforms).cpu().numpy().astype(np.float16)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite float16 projected input')
        return x

    def score(self, voxels):
        with self.torch.no_grad():
            x = SS.prep(self.torch, np.asarray(voxels, np.float16), self.masks)
            y = self.torch.stack([net(x) for net in self.nets]).mean(0).cpu().numpy()
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite M3 logits')
        return y

    def close(self):
        self.nets.clear()
        self.projector, self.masks = None, None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def canary_cases():
    return [dict(unit=next(u for u in UNITS['calib'] if u % 3 == mode), mode=mode, config=c, frame=f)
            for mode in range(3) for c in (0, 20) for f in (3, 8, 15)]


def canary():
    if not (OUT / 'PLAN.json').exists():
        raise RuntimeError('Parent PLAN/RUNS must be frozen before new canary')
    if (OUT / 'canary.json').exists():
        raise FileExistsError('Preserve existing canary; no silent repetition')
    cases = canary_cases()
    selected = {'calib': sorted({r['unit'] for r in cases})}
    inputs = identity(selected)
    create_json(OUT / 'canary_plan.json', dict(cases=cases, inputs=inputs,
        zero_rule='18 metadata-preselected calibration cases, fused zero reprojected M3 mean raw logits versus retained early/late caches; abs<1e-4',
        new_angle_rule='Six frame8 cases at each of -2/+2, original/fused float16 bit exact; no truth or alarm calculation',
        old_fused_54_pass_sha256=inputs['retained']['fused_canary_v3_sha256']))
    engine, original = None, None
    report = dict(status='NOT_RUN', inputs=inputs, zero_cases=[], angle_cases=[], truth_read=False,
                  canary_plan_sha256=SS.sha(OUT / 'canary_plan.json'))
    try:
        engine, original = Engine(), MP.BatchedProjector()
        loaded = {u: observation('calib', u) for u in selected['calib']}
        expected, voxels = [], []
        with np.load(MC.OUT / 'frame_scores_M3_early.npz', allow_pickle=False) as early, \
                np.load(MC.OUT / 'frame_scores_M3.npz', allow_pickle=False) as late:
            for case in cases:
                u, c, f = case['unit'], case['config'], case['frame']
                data = loaded[u]
                ids = np.flatnonzero(data['scene'] == c)
                sensor, travel, noisy = CP.motion_metadata(u, c)
                transform = CP.relative_transforms(sensor, travel, noisy, f)
                z = data['z1'][ids[max(0, f-7):f+1]]
                voxels.append(engine.project(z, yaw(0.) @ transform))
                expected.append(early[str(u)][c, f-3] if f <= 10 else late[str(u)][c, f-11])
                if f == 8:
                    for bias in (-2., 2.):
                        transforms = yaw(bias) @ transform
                        before = time.perf_counter()
                        a = original.sequence(z, transforms).cpu().numpy().astype(np.float16)
                        original_s = time.perf_counter()-before
                        before = time.perf_counter()
                        b = engine.project(z, transforms)
                        fused_s = time.perf_counter()-before
                        count = int(np.count_nonzero(a.view(np.uint16) != b.view(np.uint16)))
                        report['angle_cases'].append(dict(**case, bias_deg=bias, half_mismatches=count,
                            max_abs=float(np.abs(a.astype(float)-b.astype(float)).max()), original_s=original_s, fused_s=fused_s))
                        if count:
                            raise ValueError(f'New-angle half parity failed: {case}, bias={bias}, count={count}')
        predicted = engine.score(voxels)
        expected = np.asarray(expected)
        for i, case in enumerate(cases):
            error = float(np.abs(predicted[i]-expected[i]).max())
            report['zero_cases'].append(dict(**case, m3_raw_logit_max_abs=error))
        error = float(np.abs(predicted-expected).max())
        if error >= 1e-4:
            raise ValueError(f'Calibration zero M3 raw score parity failed: {error}')
        if identity(selected) != inputs:
            raise ValueError('Canary input identity changed during execution')
        report.update(status='PASS', zero_max_abs=error, runtime=engine.runtime, angles_half_bit_exact=True)
    except BaseException as exception:
        report.update(status='FAILED', error=repr(exception))
        create_json(OUT / 'canary.json', report)
        raise
    finally:
        original = None
        if engine is not None:
            engine.close()
    create_json(OUT / 'canary.json', report)
    print('PASS new envelope canary', json.dumps(dict(zero_max_abs=error, zero_items=len(cases), angle_items=len(report['angle_cases']))), flush=True)
    return report


def request(prepare=False):
    if not (OUT / 'PLAN.json').is_file():
        raise RuntimeError('Parent PLAN/RUNS required before inference')
    checked = read(OUT / 'canary.json')
    selected = {'calib': sorted({r['unit'] for r in checked['zero_cases']})}
    if checked['status'] != 'PASS' or identity(selected) != checked['inputs']:
        raise ValueError('Current sources/models/observations must pass the new canary')
    value = dict(inputs=identity(UNITS), units=UNITS, frames=FRAMES.tolist(), angles=ANGLES, batch=BATCH,
        plan_sha256=SS.sha(OUT / 'PLAN.json'), canary_sha256=SS.sha(OUT / 'canary.json'),
        operation='M3 five-seed raw logits from missing fixed yaw inputs; B@relative_transforms; no envelope aggregation or truth access')
    path = OUT / 'request.json'
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen request changed; do not resume checkpoints')
    elif prepare:
        create_json(path, value)
    else:
        raise RuntimeError('Sequential --stage prepare must finish before parallel inference workers')
    return value, SS.sha(path)


def checkpoint(split, unit, digest):
    path = OUT / 'checkpoints' / split / f'unit{unit}.npz'
    receipt = path.with_suffix('.json')
    if not receipt.exists():
        if path.exists():
            raise RuntimeError('Partial checkpoint requires inspection: '+str(path))
        return None
    old = read(receipt)
    if old['status'] != 'COMPLETE' or old['request_sha256'] != digest or old['sha256'] != SS.sha(path):
        raise ValueError('Checkpoint provenance mismatch')
    with np.load(path, allow_pickle=False) as z:
        if int(z['unit']) != unit or not np.array_equal(z['frames'], FRAMES):
            raise ValueError('Checkpoint identities differ')
        arrays = {tag: z[tag] for tag in ANGLES[split]}
    if any(x.shape != (40, 13, 2) or not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Incomplete raw-logit checkpoint')
    return arrays


def infer(split, chunk):
    frozen, digest = request()
    k, n = map(int, chunk.split('/'))
    if n < 1 or not 0 <= k < n:
        raise ValueError('Invalid unit chunk')
    (OUT / 'checkpoints' / split).mkdir(parents=True, exist_ok=True)
    engine = None
    begun = time.monotonic()
    try:
        for unit in UNITS[split][k::n]:
            if checkpoint(split, unit, digest) is not None:
                print('reuse', split, unit, flush=True)
                continue
            if engine is None:
                engine = Engine()
            start = time.monotonic()
            data = observation(split, unit)
            output = {tag: np.full((40, 13, 2), np.nan, np.float32) for tag in ANGLES[split]}
            buffer, positions = [], []
            def flush():
                predictions = engine.score(buffer)
                for j, (tag, config, f) in enumerate(positions):
                    output[tag][config, f-3] = predictions[j]
                buffer.clear(); positions.clear()
            for config in range(40):
                ids = np.flatnonzero(data['scene'] == config)
                sensor, travel, noisy = CP.motion_metadata(unit, config)
                for f in FRAMES:
                    transform = CP.relative_transforms(sensor, travel, noisy, int(f))
                    z = data['z1'][ids[max(0, f-7):f+1]]
                    for tag, bias in ANGLES[split].items():
                        buffer.append(engine.project(z, yaw(bias) @ transform))
                        positions.append((tag, config, int(f)))
                        if len(buffer) == BATCH:
                            flush()
            if buffer:
                flush()
            if any(not np.isfinite(x).all() for x in output.values()):
                raise ValueError('Unfilled unit raw scores')
            if SS.sha(observation_path(split, unit)) != frozen['inputs']['observation_sha256'][split][str(unit)]:
                raise ValueError('Source observation changed during inference')
            path = OUT / 'checkpoints' / split / f'unit{unit}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, **output, unit=np.asarray(unit), frames=FRAMES)
            create_json(path.with_suffix('.json'), dict(status='COMPLETE', split=split, unit=unit, request_sha256=digest,
                sha256=SS.sha(path), elapsed_s=time.monotonic()-start, runtime=engine.runtime))
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
        old = read(receipt)
        if old['status'] != 'COMPLETE' or old['request_sha256'] != digest:
            raise ValueError('Existing scores receipt does not match frozen request')
        for name, checksum in old['output_sha256'].items():
            if SS.sha(OUT / name) != checksum:
                raise ValueError('Completed score artifact changed')
        print('Existing COMPLETE envelope-input scores verified; no rerun', flush=True)
        return old
    units = UNITS[split]
    arrays = {u: checkpoint(split, u, digest) for u in units}
    if any(x is None for x in arrays.values()):
        raise RuntimeError(f'All fixed {split} unit checkpoints must complete before its assembly')
    hashes = {}
    for tag, bias in ANGLES[split].items():
        name = f'frame_scores_{split}_{tag}_M3.npz'
        with (OUT / name).open('xb') as f:
            np.savez_compressed(f, logit=np.stack([arrays[u][tag] for u in units]),
                                units=units, frames=FRAMES, bias_deg=np.asarray(bias))
        hashes[name] = SS.sha(OUT / name)
    result = dict(status='COMPLETE', split=split, request_sha256=digest, plan_sha256=frozen['plan_sha256'],
        output_sha256=hashes, shape=[len(units), 40, 13, 2],
        units=units, frames=FRAMES.tolist(), angles=ANGLES[split], input_sha256=frozen['inputs'],
        reused_score_caches=frozen['inputs']['retained'], canary_sha256=frozen['canary_sha256'],
        role='Frozen M3 raw evidence only; no truth, threshold calibration or outcome calculation')
    create_json(receipt, result)
    print('COMPLETE envelope input scores', split, flush=True)
    return result


def check():
    cases = canary_cases()
    assert len(cases) == 18 and len([r for r in cases if r['frame'] == 8]) == 6
    assert {r['mode'] for r in cases} == {0, 1, 2}
    assert len(UNITS['calib']) == 48 and len(UNITS['target']) == 96
    np.testing.assert_array_equal(yaw(0.), np.eye(4))
    np.testing.assert_allclose(yaw(2.) @ yaw(-2.), np.eye(4), atol=1e-15)
    assert set(ANGLES['calib'].values()) == {-2, -1, 1, 2} and set(ANGLES['target'].values()) == {-2, 2}
    assert observation_path('calib', 95000) == MC.OUT / 'features/calib/unit95000.npz'
    assert observation_path('target', 94000) == NR.OUT / 'features/evaluation/unit94000.npz'
    print('PASS fixed splits/angles, metadata-only canary and body-frame yaw; no real observations scored')


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
        _, checksum = request(prepare=True)
        print('Prepared immutable envelope score request', checksum, flush=True)
    elif args.stage == 'infer':
        if args.split is None:
            parser.error('--split required for infer')
        infer(args.split, args.chunk)
    else:
        if args.split is None:
            parser.error('--split required for assemble')
        assemble(args.split)
