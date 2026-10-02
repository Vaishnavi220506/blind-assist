"""Fixed output-side body-frame yaw alignment bias on retained z1 observations.

For every retained relative transform T, Tpert=B@T. Neither noisy poses nor
truth is changed. This reprojects old observations, never rays or new scenes.
"""
import argparse
import ast
import gc
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_near_range as NR
import cnh_cvr_pilot as CP
import cnh_cvr_projection as PR
import cnh_cvr_v2_materialize as MP

OUT = SS.WORK / 'cnh-sequence-extrinsic-yaw-20261002'
ZERO = SS.WORK / 'cnh-sequence-transfer-20261002'
MARGIN = SS.WORK / 'cnh-margin-labels-20261002'
UNITS = list(range(94000, 94096))
FRAMES = np.arange(3, 16)
ARMS = ('NEAR', 'M3')
SIGNS = {'minus1': -1., 'plus1': 1.}
BATCH = 32


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def progress(path, value):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def model_paths(arm):
    return [(NR.OUT if arm == 'NEAR' else MARGIN) / 'models' / arm / f'model_seed{s}.pt' for s in range(5)]


def identity(units, backend='original'):
    sources = [__file__, SS.__file__, NR.__file__, CP.__file__, PR.__file__, MP.__file__,
               CP.SOURCE / 'cnh_track_a_readout.py']
    backend_info = dict(name=backend)
    if backend == 'fused':
        import cupy
        sources.append(Path(__file__).with_name('cnh_cvr_fused_projector.py'))
        backend_info['cupy_version'] = cupy.__version__
    elif backend != 'original':
        raise ValueError('Unknown projector backend')
    return dict(source_sha256={str(Path(p)): SS.sha(p) for p in sources}, backend=backend_info,
        model_sha256={str(p): SS.sha(p) for arm in ARMS for p in model_paths(arm)},
        observations_sha256={str(u): SS.sha(NR.OUT / 'features/evaluation' / f'unit{u}.npz') for u in units})


def baseline_receipt():
    """Retain the old full source; verify only the original numerical path here."""
    old = read(OUT / 'canary.json')
    snapshot = OUT / 'baseline_scores_source.py'
    old_key = next(key for key in old['inputs']['source_sha256'] if Path(key).name == Path(__file__).name)
    if old['status'] != 'PASS' or SS.sha(snapshot) != old['inputs']['source_sha256'][old_key]:
        raise ValueError('Original canary source snapshot does not match its recorded identity')
    new = identity(sorted({r['unit'] for r in old['cases']}))
    for key, digest in old['inputs']['source_sha256'].items():
        if key != old_key and new['source_sha256'].get(key) != digest:
            raise ValueError('An original canary numerical dependency changed')
    for key in ('model_sha256', 'observations_sha256'):
        if new[key] != old['inputs'][key]:
            raise ValueError('Original canary observations/models changed')
    def numerical_nodes(path):
        tree = ast.parse(Path(path).read_text(encoding='utf8'))
        nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        engine = {n.name: n for n in nodes['Engine'].body if isinstance(n, ast.FunctionDef)}
        wanted = {k: nodes[k] for k in ('yaw_bias', 'data_for', 'model_paths', 'canary_cases')}
        wanted.update({f'Engine.{k}': engine[k] for k in ('project', 'score')})
        return {k: ast.dump(v, include_attributes=False) for k, v in wanted.items()}
    if numerical_nodes(snapshot) != numerical_nodes(__file__):
        raise ValueError('Retained baseline numerical functions changed; old canary is insufficient')
    return dict(canary_sha256=SS.sha(OUT / 'canary.json'), snapshot_path=str(snapshot),
        snapshot_sha256=SS.sha(snapshot), validated_functions=list(numerical_nodes(snapshot)),
        scope='Old receipt validates original projector path and retained numerical functions, not the entire revised scoring script')


def yaw_bias(degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    matrix = np.eye(4)
    matrix[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return matrix


def data_for(unit):
    with np.load(NR.OUT / 'features/evaluation' / f'unit{unit}.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in ('z1', 'scene', 'frame')}
    for config in range(40):
        rows = np.flatnonzero(data['scene'] == config)
        if not np.array_equal(data['frame'][rows], np.arange(16)):
            raise ValueError('Retained scene frame order changed')
    if not np.isfinite(data['z1']).all():
        raise ValueError('Nonfinite original observations')
    return data


class Engine:
    def __init__(self, backend='original'):
        import torch
        self.torch, self.projector, self.masks = torch, None, None
        self.nets = {a: [] for a in ARMS}
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA projector/model backend required')
        try:
            if backend == 'original':
                self.projector = MP.BatchedProjector()
            elif backend == 'fused':
                from cnh_cvr_fused_projector import FusedProjector
                self.projector = FusedProjector()
            else:
                raise ValueError('Unknown projector backend')
            self.masks = torch.as_tensor(PR.query_masks(), dtype=torch.float32, device='cuda')
            for arm in ARMS:
                for path in model_paths(arm):
                    net = CP.CVR().cuda()
                    net.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
                    self.nets[arm].append(net.eval())
            self.runtime = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                                threads=torch.get_num_threads(), tf32=False, batch=BATCH, seeds_per_arm=5, backend=backend)
            if backend == 'fused':
                self.runtime['cupy_version'] = self.projector.cp.__version__
        except BaseException:
            self.close()
            raise

    def project(self, observations, transforms):
        voxel = self.projector.sequence(observations, transforms)
        value = voxel.cpu().numpy().astype(np.float16)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite reprojected float16 voxels')
        return value

    def score(self, voxels):
        with self.torch.no_grad():
            batch = SS.prep(self.torch, np.asarray(voxels, dtype=np.float16), self.masks)
            result = {arm: self.torch.stack([net(batch) for net in self.nets[arm]]).mean(0).cpu().numpy()
                      for arm in ARMS}
        if any(not np.isfinite(x).all() for x in result.values()):
            raise ValueError('Nonfinite ensemble scores')
        return result

    def close(self):
        for values in self.nets.values():
            values.clear()
        self.projector, self.masks = None, None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def canary_cases():
    return [dict(unit=next(u for u in UNITS if u % 3 == mode), mode=mode, config=config, frame=frame)
            for mode in range(3) for config in (0, 20) for frame in (3, 8, 15)]


def canary():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'canary.json'
    if path.exists():
        raise FileExistsError('Existing canary evidence must be inspected, never silently overwritten')
    cases = canary_cases()
    inputs = identity(sorted({r['unit'] for r in cases}))
    plan = dict(cases=cases, selection='First retained unit in each unit%3 mode; config0/20; frames3/8/15; no outcome selection',
        bias_deg=0., voxel_acceptance='bitwise equal after float16', logit_abs_tolerance=1e-4, inputs=inputs)
    p = OUT / 'canary_plan.json'
    if p.exists():
        if read(p) != plan:
            raise ValueError('Canary plan/inputs changed')
    else:
        write_new(p, plan)
    locations, mmaps, cache_records = {}, [], []
    engine = None
    try:
        for part in ('evaluation_early', 'evaluation'):
            for features in sorted((NR.OUT / 'data' / part).glob('features_c*.npy')):
                metadata = features.with_name(features.name.replace('features_', 'metadata_').replace('.npy', '.npz'))
                array = np.load(features, mmap_mode='r')
                mmaps.append(array)
                with np.load(metadata, allow_pickle=False) as meta:
                    assert len(meta['unit']) == len(array)
                    for row in cases:
                        ids = np.flatnonzero((meta['unit'] == row['unit']) & (meta['config'] == row['config']) & (meta['frame'] == row['frame']))
                        if len(ids):
                            key = (row['unit'], row['config'], row['frame'])
                            if len(ids) != 1 or key in locations:
                                raise ValueError('Duplicate canary metadata identity')
                            locations[key] = (array, int(ids[0]))
                stat = features.stat()
                cache_records.append(dict(path=str(features), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                          metadata_sha256=SS.sha(metadata)))
        if len(locations) != len(cases):
            raise ValueError('Canary cache lookup incomplete')
        reference = {}
        reference_hashes = {}
        for arm in ARMS:
            file = ZERO / f'frame_scores_{arm}.npz'
            reference_hashes[arm] = SS.sha(file)
            if reference_hashes[arm] != read(ZERO / 'scores_receipt.json')['output_sha256'][arm]:
                raise ValueError('Retained zero-score cache changed')
            with np.load(file, allow_pickle=False) as data:
                reference[arm] = np.asarray([data['logit'][r['unit']-94000, r['config'], r['frame']-3] for r in cases])
        engine = Engine()
        voxels, timings, checks = [], [], []
        current_unit, data = None, None
        for row in cases:
            if row['unit'] != current_unit:
                current_unit, data = row['unit'], data_for(row['unit'])
            ids = np.flatnonzero(data['scene'] == row['config'])
            sensor, travel, noisy = CP.motion_metadata(row['unit'], row['config'])
            f = row['frame']
            transform = CP.relative_transforms(sensor, travel, noisy, f)
            engine.torch.cuda.synchronize()
            started = time.monotonic()
            value = engine.project(data['z1'][ids[max(0, f-7):f+1]], yaw_bias(0.) @ transform)
            elapsed = time.monotonic()-started
            cached, index = locations[(row['unit'], row['config'], f)]
            equal = bool(np.array_equal(value, cached[index]))
            error = float(np.abs(value.astype(float)-cached[index].astype(float)).max())
            checks.append(dict(**row, voxel_equal=equal, max_abs_error=error, projection_s=elapsed))
            timings.append(elapsed)
            if not equal:
                raise ValueError(f'Zero-bias voxel canary failed: {row}, error {error}')
            voxels.append(value)
        start = time.monotonic()
        prediction = engine.score(voxels)
        score_seconds = time.monotonic()-start
        errors = {arm: float(np.abs(prediction[arm]-reference[arm]).max()) for arm in ARMS}
        if any(error >= 1e-4 for error in errors.values()):
            raise ValueError(f'Zero-bias frozen score canary failed: {errors}')
        # Metadata-preselected canary only: no +/-1 score has been generated.
        warm = np.asarray(timings[1:])
        estimate = float(warm.mean()) * 96*40*13*2
        result = dict(status='PASS', cases=checks, model_logit_max_abs_error=errors, inputs=inputs,
            baseline_caches=cache_records, baseline_score_sha256=reference_hashes, runtime=engine.runtime,
            benchmark=dict(projection_total_s=float(np.sum(timings)), per_frame_mean_s=float(np.mean(timings)),
                warm_projection_mean_s=float(warm.mean()), warm_projection_median_s=float(np.median(warm)),
                warm_projection_p95_s=float(np.percentile(warm, 95)), models_both_arms_18_rows_s=score_seconds,
                projection_99840_frames_estimated_s=estimate, estimated_minutes=estimate/60,
                caveat='Small zero-bias canary extrapolation, includes CPU transfer/float16 but excludes full raw I/O/model batches; not a measured full-run runtime'),
            model_inputs_changed=False, formal_signed_bias_scores_generated=False)
        if identity(sorted({r['unit'] for r in cases})) != inputs:
            raise ValueError('Canary inputs changed during check')
        write_new(path, result)
        print('PASS canary', json.dumps(dict(errors=errors, benchmark=result['benchmark'], runtime=engine.runtime)), flush=True)
        return result
    finally:
        if engine is not None:
            engine.close()
        for value in mmaps:
            value._mmap.close()


def fused_canary():
    """Numerical backend equivalence only; never inspect truth or alarm scores."""
    target = OUT / 'fused_canary_v3.json'
    if target.exists():
        raise FileExistsError('Preserve existing fused canary; no automatic rerun')
    baseline = baseline_receipt()
    synthetic_path = OUT / 'fused_check_v3.json'
    synthetic = read(synthetic_path)
    fused_path = Path(__file__).with_name('cnh_cvr_fused_projector.py')
    required_cases = {f'n{n}_bias{bias:+.0f}' for n in (1, 4, 8) for bias in (-1., 0., 1.)}
    required_cases.update(('identity', 'retained94002_c0_f3_bias0'))
    if synthetic['source_sha256'] != SS.sha(fused_path) or not synthetic.get('numerical_equivalence') or \
            len(synthetic['cases']) != 11 or {case['name'] for case in synthetic['cases']} != required_cases or \
            not all(case['raw_pass'] and case['half_exact'] and case['max_abs'] == 0 for case in synthetic['cases']):
        raise ValueError('Current fused backend must pass ten synthetic plus the retained failure-case exact checks first')
    cases = [dict(**case, bias_deg=bias) for case in canary_cases() for bias in (0., -1., 1.)]
    selected_units = sorted({case['unit'] for case in cases})
    inputs = identity(selected_units, 'fused')
    prior_receipts = {name: SS.sha(OUT / name) for name in
                      ('fused_check.json', 'fused_check_v2.json', 'fused_timing_diagnostic.json',
                       'fused_canary.json', 'fused_canary_plan.json', 'fused_canary_failed_scores_source.py',
                       'fused_real_failure_diagnostic.json', 'fused_check_v3.json')}
    plan = dict(cases=cases, inputs=inputs, baseline=baseline, prior_receipts=prior_receipts,
        synthetic_status_unchanged=synthetic['status'],
        rule='Each original/fused projection must match bitwise after float16; all 10 frozen model logits compared on paired identical inputs; no truth/threshold access',
        speed_gate='Alternate original-first/fused-first by case; retain every pair timing; exclude first pair from warm summary. Require warm mean original/fused >1 overall AND median speedup >1 separately for history4 and history8.',
        logit_abs_tolerance=1e-4, status='NUMERICAL_CANARY_ONLY')
    write_new(OUT / 'fused_canary_plan_v3.json', plan)
    engine, candidate = None, None
    report = dict(status='NOT_RUN', inputs=inputs, baseline=baseline, cases=[], plan_sha256=SS.sha(OUT / 'fused_canary_plan_v3.json'),
                  truth_or_threshold_read=False, scientific_perturbation_result=False, prior_receipts=prior_receipts)
    try:
        from cnh_cvr_fused_projector import FusedProjector
        engine, candidate = Engine('original'), FusedProjector()
        original_voxels, fused_voxels, timings = [], [], {'original': [], 'fused': []}
        loaded = {u: data_for(u) for u in selected_units}
        for case_index, case in enumerate(cases):
            data = loaded[case['unit']]
            ids = np.flatnonzero(data['scene'] == case['config'])
            sensor, travel, noisy = CP.motion_metadata(case['unit'], case['config'])
            f = case['frame']
            transforms = yaw_bias(case['bias_deg']) @ CP.relative_transforms(sensor, travel, noisy, f)
            z = data['z1'][ids[max(0, f-7):f+1]]
            values = {}
            order = [('original', engine.projector), ('fused', candidate)]
            if case_index % 2:
                order.reverse()
            paired_timing = {}
            for backend, projector in order:
                engine.torch.cuda.synchronize()
                begin = time.perf_counter()
                values[backend] = projector.sequence(z, transforms).cpu().numpy().astype(np.float16)
                elapsed = time.perf_counter()-begin
                timings[backend].append(elapsed)
                paired_timing[backend] = elapsed
            a, b = values['original'], values['fused']
            mismatch = int(np.count_nonzero(a.view(np.uint16) != b.view(np.uint16)))
            report['cases'].append(dict(**case, history=len(z), timing_order=[x[0] for x in order], paired_timing_s=paired_timing,
                                       half_bit_mismatch_elements=mismatch,
                                       max_abs=float(np.abs(a.astype(float)-b.astype(float)).max())))
            if mismatch:
                raise ValueError(f'Retained biased geometry half parity failed: {case}, count {mismatch}')
            original_voxels.append(a); fused_voxels.append(b)
        model_checks = []
        torch = engine.torch
        with torch.no_grad():
            for arm in ARMS:
                for seed, net in enumerate(engine.nets[arm]):
                    maximum = 0.; exact = True
                    for start in range(0, len(cases), BATCH):
                        x = SS.prep(torch, original_voxels[start:start+BATCH], engine.masks)
                        y = SS.prep(torch, fused_voxels[start:start+BATCH], engine.masks)
                        if not torch.equal(x, y):
                            raise ValueError('Equal half tensors changed during common SS.prep')
                        actual, expected = net(y), net(x)
                        maximum = max(maximum, float((actual-expected).abs().max()))
                        exact = exact and torch.equal(actual, expected)
                    model_checks.append(dict(arm=arm, seed=seed, max_abs=maximum, exact=exact))
                    if maximum >= 1e-4:
                        raise ValueError('Equivalent backend inputs failed frozen model logit parity')
        if identity(selected_units, 'fused') != inputs:
            raise ValueError('Fused canary sources/models/observations changed during execution')
        warm = {key: np.asarray(value[1:]) for key, value in timings.items()}
        overall_speedup = float(warm['original'].mean()/warm['fused'].mean())
        by_history = {}
        for history in (4, 8):
            paired = [r for r in report['cases'][1:] if r['history'] == history]
            median = {backend: float(np.median([r['paired_timing_s'][backend] for r in paired]))
                      for backend in ('original', 'fused')}
            by_history[str(history)] = dict(pairs=len(paired), median_s=median, speedup=median['original']/median['fused'])
        faster = overall_speedup > 1 and all(r['speedup'] > 1 for r in by_history.values())
        report.update(status='PASS' if faster else 'NOT_FASTER', models=model_checks,
            runtime=dict(**engine.runtime, candidate_backend='fused', cupy_version=candidate.cp.__version__),
            timings={key: dict(warm_mean_s=float(value.mean()), warm_median_s=float(np.median(value)),
                warm_p95_s=float(np.percentile(value, 95)), projection_99840_estimate_s=float(value.mean())*99840)
                for key, value in warm.items()},
            speedup_mean=overall_speedup, by_history=by_history, numerical_equivalence=True, faster=faster,
            receipt_scope='54 numerical original/fused comparisons only; no alarm rates or timing outcomes calculated')
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        write_new(target, report)
        raise
    finally:
        candidate = None
        if engine is not None:
            engine.close()
    write_new(target, report)
    print(report['status'], 'fused retained canary', json.dumps(dict(timings=report['timings'], speedup=report['speedup_mean'], by_history=by_history)), flush=True)
    return report


def request(backend='original', prepare=False):
    if not (OUT / 'PLAN.json').is_file():
        raise RuntimeError('Parent PLAN/RUNS required before signed inference')
    baseline = baseline_receipt()
    backend_canary = OUT / ('fused_canary_v3.json' if backend == 'fused' else 'canary.json')
    if backend == 'fused':
        checked = read(backend_canary)
        if checked['status'] != 'PASS' or checked['inputs'] != identity(sorted({r['unit'] for r in checked['cases']}), 'fused'):
            raise ValueError('Current fused backend must pass retained 54-case numerical canary')
        if checked['baseline'] != baseline:
            raise ValueError('Baseline numerical receipt changed')
    value = dict(inputs=identity(UNITS, backend), backend=backend, units=UNITS, frames=FRAMES.tolist(), signs=SIGNS,
        batch=BATCH, transform='Tpert=B@relative_transforms; fixed body-yaw left multiplication',
        plan_sha256=SS.sha(OUT / 'PLAN.json'), canary_sha256=SS.sha(backend_canary), baseline=baseline)
    path = OUT / 'request.json'
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen request changed; checkpoints cannot be resumed')
    elif prepare:
        # Only the sequential parent prepare stage writes this file. Workers
        # require it to exist, avoiding concurrent readers of a partial request.
        write_new(path, value)
    else:
        raise RuntimeError('Run sequential --stage prepare before launching worker chunks')
    return value, SS.sha(path)


def checkpoint(unit, digest):
    file = OUT / 'checkpoints' / f'unit{unit}.npz'
    receipt = file.with_suffix('.json')
    if not receipt.exists():
        if file.exists():
            raise RuntimeError(f'Partial checkpoint requires inspection: {file}')
        return None
    r = read(receipt)
    if r['status'] != 'COMPLETE' or r['request_sha256'] != digest or r['sha256'] != SS.sha(file):
        raise ValueError(f'Invalid/changed checkpoint: {unit}')
    with np.load(file, allow_pickle=False) as cache:
        result = {f'{s}_{a}': cache[f'{s}_{a}'] for s in SIGNS for a in ARMS}
    if any(x.shape != (40, 13, 2) or not np.isfinite(x).all() for x in result.values()):
        raise ValueError('Checkpoint raw score tensor incomplete')
    return result


def infer(chunk, backend='original'):
    frozen, digest = request(backend)
    k, n = map(int, chunk.split('/'))
    if n < 1 or not 0 <= k < n:
        raise ValueError('Invalid unit chunk')
    (OUT / 'checkpoints').mkdir(exist_ok=True)
    engine = None
    started = time.monotonic()
    try:
        for unit in UNITS[k::n]:
            if checkpoint(unit, digest) is not None:
                print('reuse checkpoint', unit, flush=True)
                continue
            if engine is None:
                engine = Engine(backend)
            unit_start = time.monotonic()
            data = data_for(unit)
            outputs = {f'{s}_{a}': np.full((40, 13, 2), np.nan, np.float32) for s in SIGNS for a in ARMS}
            voxels, positions = [], []
            project_seconds, model_seconds = 0., 0.
            def flush():
                nonlocal model_seconds
                begin = time.monotonic()
                result = engine.score(voxels)
                model_seconds += time.monotonic()-begin
                for j, (sign, config, frame) in enumerate(positions):
                    for arm in ARMS:
                        outputs[f'{sign}_{arm}'][config, frame-3] = result[arm][j]
                voxels.clear(); positions.clear()
            for config in range(40):
                ids = np.flatnonzero(data['scene'] == config)
                sensor, travel, noisy = CP.motion_metadata(unit, config)
                for f in FRAMES:
                    transform = CP.relative_transforms(sensor, travel, noisy, int(f))
                    observations = data['z1'][ids[max(0, f-7):f+1]]
                    for sign, bias in SIGNS.items():
                        begin = time.monotonic()
                        voxels.append(engine.project(observations, yaw_bias(bias) @ transform))
                        project_seconds += time.monotonic()-begin
                        positions.append((sign, config, int(f)))
                        if len(voxels) == BATCH:
                            flush()
            if voxels:
                flush()
            if any(not np.isfinite(x).all() for x in outputs.values()):
                raise ValueError('Incomplete unit score output')
            if SS.sha(NR.OUT / 'features/evaluation' / f'unit{unit}.npz') != frozen['inputs']['observations_sha256'][str(unit)]:
                raise ValueError('Unit observations changed during inference')
            path = OUT / 'checkpoints' / f'unit{unit}.npz'
            with path.open('xb') as file:
                np.savez_compressed(file, **outputs, unit=np.asarray(unit), frames=FRAMES)
            write_new(path.with_suffix('.json'), dict(status='COMPLETE', unit=unit, request_sha256=digest,
                sha256=SS.sha(path), projection_s=project_seconds, models_s=model_seconds,
                elapsed_s=time.monotonic()-unit_start, runtime=engine.runtime))
            progress(OUT / f'progress_c{k}of{n}.json', dict(last_unit=unit, elapsed_s=time.monotonic()-started,
                                                         unit_elapsed_s=time.monotonic()-unit_start))
            print('complete unit', unit, 'seconds', round(time.monotonic()-unit_start, 2), flush=True)
        # Source/model/data drift invalidates continuation/publication; no silent reset.
        if identity(UNITS, backend) != frozen['inputs']:
            raise ValueError('Input identity changed during signed projection')
    finally:
        if engine is not None:
            engine.close()


def assemble(backend='original'):
    frozen, digest = request(backend)
    receipt = OUT / 'scores_receipt.json'
    if receipt.exists():
        old = read(receipt)
        if old['status'] != 'COMPLETE' or old['request_sha256'] != digest:
            raise ValueError('Existing receipt does not match frozen request')
        for name, checksum in old['output_sha256'].items():
            if SS.sha(OUT / name) != checksum:
                raise ValueError('Completed signed score cache changed')
        print('Existing COMPLETE signed scores verified', flush=True)
        return old
    units = {u: checkpoint(u, digest) for u in UNITS}
    if any(value is None for value in units.values()):
        raise RuntimeError('All 96 unit checkpoints must complete before assembly')
    hashes = {}
    for sign, bias in SIGNS.items():
        for arm in ARMS:
            logit = np.stack([units[u][f'{sign}_{arm}'] for u in UNITS])
            name = f'frame_scores_{sign}_{arm}.npz'
            with (OUT / name).open('xb') as file:
                np.savez_compressed(file, logit=logit, units=UNITS, frames=FRAMES, bias_deg=np.asarray(bias))
            hashes[name] = SS.sha(OUT / name)
    result = dict(status='COMPLETE', request_sha256=digest, plan_sha256=frozen['plan_sha256'],
        output_sha256=hashes, shape=[96, 40, 13, 2], units=UNITS, frames=FRAMES.tolist(),
        input_sha256=frozen['inputs'], canary_sha256=frozen['canary_sha256'], backend=backend,
        meaning='Fixed output-side body-frame yaw alignment bias applied to retained r/sqrt(v) standardized observations; no new rays, scenes or physical sensor data')
    write_new(receipt, result)
    print('COMPLETE signed scores', flush=True)
    return result


def check():
    assert len(canary_cases()) == 18 and {r['mode'] for r in canary_cases()} == {0, 1, 2}
    sensor, travel, noisy = CP.motion_metadata(94000, 0)
    original = noisy.copy()
    t = CP.relative_transforms(sensor, travel, noisy, 8)
    np.testing.assert_array_equal(yaw_bias(0.) @ t, t)
    np.testing.assert_array_equal(noisy, original)
    assert (yaw_bias(1.) @ np.asarray([0., 0., 1., 1.]))[0] > 0
    np.testing.assert_allclose(yaw_bias(1.) @ yaw_bias(-1.), np.eye(4), atol=1e-15)
    assert np.max(np.abs(yaw_bias(1.) @ t - t)) > 0
    print('PASS left body-yaw transform, zero identity, unchanged noisy poses and metadata-only canary selection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'canary', 'fused-canary', 'prepare', 'infer', 'assemble'))
    parser.add_argument('--chunk', default='0/1')
    parser.add_argument('--backend', default='original', choices=('original', 'fused'))
    args = parser.parse_args()
    if args.stage == 'check':
        check()
    elif args.stage == 'canary':
        canary()
    elif args.stage == 'fused-canary':
        fused_canary()
    elif args.stage == 'prepare':
        _, digest = request(args.backend, prepare=True)
        print('Prepared immutable request', args.backend, digest, flush=True)
    elif args.stage == 'infer':
        infer(args.chunk, args.backend)
    else:
        assemble(args.backend)
