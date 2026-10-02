"""Retained per-frame CVR evidence/coverage, with unchanged old-three parity.

No labels, geometry truth, models, scores or alarm metrics are read. Only old
z1, frozen noisy-motion reconstruction and existing three-channel voxel caches.
"""
import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_near_range as NR
import cnh_margin_confirm as MC
import cnh_cvr_pilot as CP
import cnh_cvr_projection as PR
import cnh_cvr_fused_projector as FP

OUT = SS.WORK / 'cnh-frame-history-20261002'
RUN_ID = 'CNH_FRAME_HISTORY_20261002'
SPLITS = {'train': list(range(93000, 93096)), 'calib': list(range(95000, 95048)),
          'evaluation': list(range(96000, 96096))}
CONFIGS = {'train': 22, 'calib': 40, 'evaluation': 40}
FRAMES = np.arange(3, 16)
RESERVE = 20_000_000_000
SHAPE = PR.SHAPE


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def create_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Preserve completed receipt: '+str(path))
    partial = path.with_name(path.name+'.partial')
    with partial.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    partial.rename(path)


def plan_sha():
    path = OUT / 'PLAN.json'
    if not path.is_file() or RUN_ID not in (Path(__file__).resolve().parents[1] / 'RUNS.md').read_text(encoding='utf8'):
        raise RuntimeError('Parent PLAN/RUNS required before canary/prepare/run')
    plan = read(path)
    expected = dict(run=RUN_ID, train_units=SPLITS['train'], calibration_units=SPLITS['calib'],
                    evaluation_units=SPLITS['evaluation'], decision_frames=FRAMES.tolist())
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Cache cohort/frame routing differs from frozen PLAN')
    raw_bytes = sum(CONFIGS[s]*len(u)*13*8*int(np.prod(SHAPE))*3 for s, u in SPLITS.items())
    if plan['cache']['reserve_bytes'] != RESERVE or plan['cache']['size_bytes'] != raw_bytes:
        raise ValueError('Cache byte budget differs from frozen PLAN')
    return sha(path)


def observation_path(split, unit):
    if unit not in SPLITS[split]:
        raise ValueError('Unit outside fixed split')
    root = NR.OUT if split == 'train' else MC.OUT
    return root / 'features' / split / f'unit{unit}.npz'


def observation(split, unit):
    with np.load(observation_path(split, unit), allow_pickle=False) as data:
        result = {key: data[key] for key in ('z1', 'scene', 'frame')}
    count = CONFIGS[split]*16
    if result['scene'].shape != (count,) or result['frame'].shape != (count,) or len(result['z1']) != count:
        raise ValueError('Observation row count mismatch')
    if not np.isfinite(result['z1']).all():
        raise ValueError('Nonfinite retained standardized observations')
    for config in range(CONFIGS[split]):
        if not np.array_equal(result['frame'][result['scene'] == config], np.arange(16)):
            raise ValueError('Original frame identity/order mismatch')
    return result


def old_parts(split):
    names = ['train'] if split == 'train' else [split+'_early', split]
    root = NR.OUT if split == 'train' else MC.OUT
    paths = []
    for name in names:
        found = sorted(p for p in (root / 'data' / name).glob('features_c*.npy')
                       if re.fullmatch(r'features_c\d+', p.stem))
        if not found:
            raise FileNotFoundError('Old voxel parts missing: '+name)
        paths.extend(found)
    if len(paths) != {'train': 3, 'calib': 2, 'evaluation': 4}[split]:
        raise ValueError('Expected original three train / six early-late evaluation chunks')
    return [(p, p.with_name(p.name.replace('features_', 'metadata_').replace('.npy', '.npz'))) for p in paths]


class OldCache:
    def __init__(self, split):
        self.arrays, self.index = [], {}
        try:
            for path, metadata in old_parts(split):
                array = np.load(path, mmap_mode='r')
                self.arrays.append(array)
                if array.dtype != np.float16 or array.shape[1:] != (3, *SHAPE):
                    raise ValueError('Old voxel shape/dtype changed')
                with np.load(metadata, allow_pickle=False) as data:
                    fields = [data[key] for key in ('unit', 'config', 'frame')]
                if any(field.shape != (len(array),) or not np.issubdtype(field.dtype, np.integer) for field in fields):
                    raise ValueError('Old metadata length/type mismatch')
                for row, key in enumerate(zip(*fields)):
                    key = tuple(int(v) for v in key)
                    if key in self.index:
                        raise ValueError('Duplicate old voxel unit/config/frame')
                    self.index[key] = (len(self.arrays)-1, row)
            expected = {(u, c, int(f)) for u in SPLITS[split] for c in range(CONFIGS[split]) for f in FRAMES}
            if set(self.index) != expected:
                raise ValueError('Old voxel cache not exactly the fixed split/config/frame set')
        except BaseException:
            self.close()
            raise

    def get(self, unit, config, frame):
        part, row = self.index[(unit, config, frame)]
        result = self.arrays[part][row]
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite old voxel reference')
        return result

    def close(self):
        for array in self.arrays:
            array._mmap.close()
        self.arrays.clear()


def record(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(path))


def inherited_proof():
    path = SS.WORK / 'cnh-sequence-extrinsic-yaw-20261002/fused_canary_v3.json'
    old = read(path)
    source = next(v for p, v in old['inputs']['source_sha256'].items() if Path(p).name == Path(FP.__file__).name)
    if old['status'] != 'PASS' or source != sha(FP.__file__):
        raise ValueError('Unchanged fused-v3 54-case proof required')
    return dict(path=str(path), sha256=sha(path), scope='retained engineering proof; not rerun')


def identity(units):
    sources = [Path(__file__), Path(SS.__file__), Path(NR.__file__), Path(MC.__file__), Path(CP.__file__),
               Path(PR.__file__), Path(FP.__file__), Path(__file__).with_name('cnh_cvr_v2_materialize.py'),
               CP.SOURCE / 'cnh_track_a_readout.py']
    return dict(source_sha256={str(p): sha(p) for p in sources},
        observations={split: {str(u): record(observation_path(split, u)) for u in ids} for split, ids in units.items()},
        old_caches={split: [dict(voxel=record(p), metadata=record(m)) for p, m in old_parts(split)] for split in units},
        inherited_fused_proof=inherited_proof())


def validate_inputs(inputs, full=False):
    for path, digest in inputs['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Bound source changed: '+path)
    if inherited_proof() != inputs['inherited_fused_proof']:
        raise ValueError('Inherited numerical proof changed')
    files = [value for split in inputs['observations'].values() for value in split.values()]
    files += [value for split in inputs['old_caches'].values() for part in split for value in part.values()]
    for value in files:
        path = Path(value['path'])
        stat = path.stat()
        if stat.st_size != value['bytes'] or stat.st_mtime_ns != value['mtime_ns'] or (full and sha(path) != value['sha256']):
            raise ValueError('Bound input changed: '+str(path))


class HistoryProjector:
    def __init__(self):
        import torch
        self.torch = torch
        self.projector = None
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required')
        try:
            self.projector = FP.FusedProjector()
            self.runtime = dict(device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                cuda=torch.version.cuda, cupy=self.projector.cp.__version__, threads=2, tf32=False)
        except BaseException:
            self.close()
            raise

    def project(self, z, transforms):
        torch = self.torch
        n = len(z)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            mass, valid = self.projector.point_terms(z, transforms)
            e = mass.sum(2).reshape(n, *SHAPE).float()
            c = valid.double().mean(2).reshape(n, *SHAPE).float()
            k = valid.sum(2).reshape(n, *SHAPE).to(torch.uint8)
            total, count = torch.zeros_like(e[0]), torch.zeros_like(c[0])
            for current, seen in zip(e, c):
                total += current
                count += seen
            old = torch.stack((total, count, e[-1]))
            torch.cuda.synchronize()
            projected = time.perf_counter()
            eh = e.half().cpu().numpy()
            kh = k.cpu().numpy()
            ch = c.half().cpu().numpy()
            old_half = old.half().cpu().numpy()
            transferred = time.perf_counter()
        if not np.isfinite(eh).all() or not np.isfinite(old_half).all() or kh.max() > 27:
            raise ValueError('Nonfinite history or invalid coverage count')
        recovered = (kh.astype(np.float64)/27).astype(np.float32).astype(np.float16)
        if not np.array_equal(recovered.view(np.uint16), ch.view(np.uint16)):
            raise ValueError('K/27 did not recover original half coverage exactly')
        padded_e = np.zeros((8, *SHAPE), np.float16)
        padded_k = np.zeros((8, *SHAPE), np.uint8)
        padded_e[-n:] = eh
        padded_k[-n:] = kh
        return padded_e, padded_k, old_half, dict(project_s=projected-start, copy_s=transferred-projected)

    def close(self):
        self.projector = None
        gc.collect()
        if self.torch.cuda.is_initialized():
            self.torch.cuda.empty_cache()


def verify_case(e, k, old, expected, n):
    if e.shape != (8, *SHAPE) or k.shape != e.shape or old.shape != (3, *SHAPE):
        raise ValueError('History/old3 tensor shape mismatch')
    if e.dtype != np.float16 or k.dtype != np.uint8 or old.dtype != np.float16:
        raise ValueError('History cache dtype mismatch')
    if not np.isfinite(e).all() or not np.isfinite(old).all() or k.max() > 27:
        raise ValueError('Nonfinite history or invalid K range')
    mismatches = int(np.count_nonzero(old.view(np.uint16) != expected.view(np.uint16)))
    if mismatches:
        raise ValueError(f'Old three-channel half bit mismatch: {mismatches}')
    if not np.array_equal(e[-1].view(np.uint16), expected[2].view(np.uint16)):
        raise ValueError('Latest frame E differs from old current channel')
    if n < 8 and (np.any(e[:-n].view(np.uint16)) or np.any(k[:-n])):
        raise ValueError('Absent-history slots must be positive zero/zero coverage')
    return dict(old3_half_bit_mismatches=0, latest_exact=True, padding_exact=True, coverage_levels_exact=True)


def canary_cases():
    return [dict(split=split, unit=next(u for u in ids if u % 3 == mode), mode=mode, config=c, frame=f)
            for split, ids in SPLITS.items() for mode in range(3) for c, f in ((0, 3), (20, 15))]


def canary():
    plan = plan_sha()
    target = OUT / 'canary.json'
    if target.exists():
        raise FileExistsError('Preserve existing canary; no silent repetition')
    cases = canary_cases()
    selected = {split: sorted({r['unit'] for r in cases if r['split'] == split}) for split in SPLITS}
    inputs = identity(selected)
    create_json(OUT / 'canary_plan.json', dict(plan_sha256=plan, inputs=inputs, cases=cases,
        rule='18 preselected observations only; old3/current half bit exact, left zero padding, K/27 exact; no models/truth/metrics'))
    report = dict(status='NOT_RUN', plan_sha256=plan, inputs=inputs, cases=[],
                  canary_plan_sha256=sha(OUT / 'canary_plan.json'))
    engine, caches = None, {}
    try:
        engine = HistoryProjector()
        # One mechanical warmup; no model and no old 54-case rerun.
        warm = cases[0]
        d = observation(warm['split'], warm['unit'])
        ids = np.flatnonzero(d['scene'] == warm['config'])
        sensor, travel, noisy = CP.motion_metadata(warm['unit'], warm['config'])
        engine.project(d['z1'][ids[:4]], CP.relative_transforms(sensor, travel, noisy, 3))
        samples_e, samples_k = [], []
        for split in SPLITS:
            caches[split] = OldCache(split)
        for case in cases:
            split, unit, config, frame = case['split'], case['unit'], case['config'], case['frame']
            data = observation(split, unit)
            ids = np.flatnonzero(data['scene'] == config)
            sensor, travel, noisy = CP.motion_metadata(unit, config)
            z = data['z1'][ids[max(0, frame-7):frame+1]]
            e, k, old, elapsed = engine.project(z, CP.relative_transforms(sensor, travel, noisy, frame))
            checks = verify_case(e, k, old, caches[split].get(unit, config, frame), len(z))
            samples_e.append(e)
            samples_k.append(k)
            report['cases'].append(dict(**case, **checks, history=len(z), **elapsed))
        before = time.perf_counter()
        for name, data in [('canary_mass.npy', np.stack(samples_e)), ('canary_coverage.npy', np.stack(samples_k))]:
            with (OUT / name).open('xb') as stream:
                np.save(stream, data, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        write_s = time.perf_counter()-before
        files = {name: record(OUT / name) for name in ('canary_mass.npy', 'canary_coverage.npy')}
        validate_inputs(inputs)
        if plan_sha() != plan:
            raise ValueError('PLAN changed during canary')
        report.update(status='PASS', runtime=engine.runtime, write_s=write_s, files=files,
            written_bytes=sum(v['bytes'] for v in files.values()),
            project_mean_s=float(np.mean([r['project_s'] for r in report['cases']])),
            copy_mean_s=float(np.mean([r['copy_s'] for r in report['cases']])),
            scope='Mechanical history representation/cache check only; no task outcomes')
        means = {n: float(np.mean([r['project_s']+r['copy_s'] for r in report['cases'] if r['history'] == n]))
                 for n in (4, 8)}
        per_frame = float(np.mean([np.interp(min(int(f)+1, 8), [4, 8], [means[4], means[8]]) for f in FRAMES]))
        windows = sum(len(ids)*CONFIGS[s]*13 for s, ids in SPLITS.items())
        bytes_expected = sum(unit_bytes(s)*len(ids) for s, ids in SPLITS.items())
        project_copy_estimate = per_frame*windows
        write_estimate = write_s*bytes_expected/report['written_bytes']
        report['throughput_estimate'] = dict(windows=windows, expected_bytes=bytes_expected,
            measured_history4_project_copy_s=means[4], measured_history8_project_copy_s=means[8],
            weighted_project_copy_s_per_window=per_frame, project_copy_s=project_copy_estimate,
            write_s=write_estimate, total_s=project_copy_estimate+write_estimate,
            method='Measured18cases project+copy, linear h5..7 interpolation, actual13frame history mix; measured fsync writes scaled by bytes',
            limitations='Estimate excludes source/output hashing, old-cache reads, observation loading and per-unit overhead; small-write timing may not predict33GB sustained throughput')
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        create_json(target, report)
        raise
    finally:
        for cache in caches.values():
            cache.close()
        if engine is not None:
            engine.close()
    create_json(target, report)
    print('PASS 18-case history cache canary', report['project_mean_s'], report['copy_mean_s'], report['write_s'],
          'estimated_pipeline_s', report['throughput_estimate']['total_s'], flush=True)
    return report


def request(prepare=False):
    plan = plan_sha()
    checked = read(OUT / 'canary.json')
    if checked['status'] != 'PASS' or checked['plan_sha256'] != plan:
        raise ValueError('Matching successful history canary required')
    validate_inputs(checked['inputs'])
    path = OUT / 'request.json'
    if path.exists():
        value = read(path)
        if value['plan_sha256'] != plan or value['canary_sha256'] != sha(OUT / 'canary.json'):
            raise ValueError('Frozen request plan/canary changed')
        validate_inputs(value['inputs'])
        return value, sha(path)
    if not prepare:
        raise RuntimeError('Serial --stage prepare required before run chunks')
    value = dict(plan_sha256=plan, canary_sha256=sha(OUT / 'canary.json'), inputs=identity(SPLITS),
        splits=SPLITS, configs=CONFIGS, frames=FRAMES.tolist(), slots=8, shape=list(SHAPE),
        mass='float16 per-frame E; raw float32 reductions; not clipped',
        coverage='uint8 K in0..27; C=(K.float64()/27).float32().float16',
        channel_order='e0..e7,c0..c7; oldest to latest, left zero padding',
        old_three='Original cache retained; each window checked bitwise against old3 before publication',
        reserve_bytes=RESERVE, expected_payload_bytes=sum(unit_bytes(s)*len(ids) for s, ids in SPLITS.items()))
    create_json(path, value)
    return value, sha(path)


def unit_paths(split, unit):
    folder = OUT / 'cache' / split / f'unit{unit}'
    return folder / 'mass.npy', folder / 'coverage.npy', folder / 'receipt.json'


def unit_bytes(split):
    return int(CONFIGS[split]*13*8*np.prod(SHAPE)*3 + 512)


def completed_unit(split, unit, digest, verify_hash=True):
    mass, coverage, receipt = unit_paths(split, unit)
    if not receipt.exists():
        return None
    old = read(receipt)
    if old['status'] != 'COMPLETE' or old['request_sha256'] != digest or old['unit'] != unit or old['split'] != split:
        raise ValueError('Unit receipt does not match frozen request')
    if verify_hash:
        for path in (mass, coverage):
            if sha(path) != old['output_sha256'][path.name]:
                raise ValueError('Completed unit output changed')
    return old


def disk_guard(digest):
    remaining, allocated_partial = 0, 0
    for split, units in SPLITS.items():
        for unit in units:
            if completed_unit(split, unit, digest, verify_hash=False) is None:
                remaining += unit_bytes(split)
                mass, coverage, _ = unit_paths(split, unit)
                # In-flight partial arrays already consume free disk; don't
                # charge their allocated bytes a second time as future writes.
                paths = (mass, coverage, mass.with_name('mass.partial.npy'), coverage.with_name('coverage.partial.npy'))
                allocated_partial += min(unit_bytes(split), sum(allocated_bytes(p) for p in paths if p.exists()))
    future = max(0, remaining-allocated_partial)
    free = shutil.disk_usage(OUT).free
    if free < RESERVE + future:
        raise RuntimeError(f'Insufficient disk: free={free}, reserve={RESERVE}, remaining_to_allocate={future}; preserve partial files')
    return dict(free_bytes=free, reserve_bytes=RESERVE, remaining_expected_bytes=remaining,
                incomplete_allocated_bytes=allocated_partial, remaining_to_allocate_bytes=future)


def allocated_bytes(path):
    """Physical allocation for in-flight files, including Windows sparse files."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        get_size = kernel.GetCompressedFileSizeW
        get_size.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        get_size.restype = wintypes.DWORD
        high = wintypes.DWORD()
        ctypes.set_last_error(0)
        low = get_size(str(Path(path).resolve()), ctypes.byref(high))
        if low == 0xffffffff and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        return (high.value << 32) + low
    stat = Path(path).stat()
    return getattr(stat, 'st_blocks', (stat.st_size+511)//512)*512


def finalize():
    """Parent-only serial publication; workers never race aggregate receipts."""
    frozen, digest = request()
    target = OUT / 'cache_receipt.json'
    hashes, splits, total_bytes = {}, {}, 0
    for split, units in SPLITS.items():
        rows, count = 0, 0
        for unit in units:
            old = completed_unit(split, unit, digest, verify_hash=True)
            if old is None:
                raise RuntimeError(f'Cache incomplete: {split} unit {unit}')
            expected = CONFIGS[split]*len(FRAMES)
            if old['old3_windows_bit_exact'] != expected or old['shape'] != [CONFIGS[split], 13, 8, *SHAPE]:
                raise ValueError('Unit cache coverage differs from fixed contract')
            mass, coverage, receipt = unit_paths(split, unit)
            hashes[f'{split}/unit{unit}/receipt.json'] = sha(receipt)
            total_bytes += mass.stat().st_size + coverage.stat().st_size
            rows += expected
            count += 1
        splits[split] = dict(units=units, completed_units=count, rows=rows, configs=CONFIGS[split])
    value = dict(status='COMPLETE', plan_sha256=frozen['plan_sha256'], request_sha256=digest,
        canary_sha256=frozen['canary_sha256'], splits=splits, unit_receipt_sha256=hashes,
        cache_bytes=total_bytes, old3_windows_bit_exact=sum(v['rows'] for v in splits.values()),
        mass_dtype='float16', coverage_dtype='uint8', frames=FRAMES.tolist(), slots=8,
        scope='All240units complete; all per-window old3/current half bit exact; no truth or scores used')
    if target.exists():
        if read(target) != value:
            raise ValueError('Existing aggregate cache receipt changed')
        print('Existing COMPLETE history cache verified', flush=True)
        return value
    create_json(target, value)
    print('COMPLETE history cache', value['old3_windows_bit_exact'], 'windows', total_bytes, 'bytes', flush=True)
    return value


def run(split, chunk):
    frozen, digest = request()
    k, n = map(int, chunk.split('/'))
    if not 1 <= n <= len(SPLITS[split]) or not 0 <= k < n:
        raise ValueError('Invalid unit chunk')
    engine = cache = mass = coverage = None
    try:
        for unit in SPLITS[split][k::n]:
            if completed_unit(split, unit, digest) is not None:
                print('reuse', split, unit, flush=True)
                continue
            final_mass, final_coverage, receipt = unit_paths(split, unit)
            final_mass.parent.mkdir(parents=True, exist_ok=True)
            partial_mass = final_mass.with_name('mass.partial.npy')
            partial_coverage = final_coverage.with_name('coverage.partial.npy')
            if any(p.exists() for p in (final_mass, final_coverage, partial_mass, partial_coverage)):
                raise RuntimeError('Partial/unpublished unit requires inspection; no overwrite: '+str(final_mass.parent))
            disk = disk_guard(digest)
            source = frozen['inputs']['observations'][split][str(unit)]
            if sha(source['path']) != source['sha256']:
                raise ValueError('Original unit observations changed')
            if engine is None:
                engine = HistoryProjector()
                cache = OldCache(split)
            start = time.monotonic()
            shape = (CONFIGS[split], 13, 8, *SHAPE)
            mass = np.lib.format.open_memmap(partial_mass, mode='w+', dtype=np.float16, shape=shape)
            coverage = np.lib.format.open_memmap(partial_coverage, mode='w+', dtype=np.uint8, shape=shape)
            data = observation(split, unit)
            project_s = copy_s = write_s = 0.
            checked = 0
            for config in range(CONFIGS[split]):
                ids = np.flatnonzero(data['scene'] == config)
                sensor, travel, noisy = CP.motion_metadata(unit, config)
                for frame in FRAMES:
                    f = int(frame)
                    z = data['z1'][ids[max(0, f-7):f+1]]
                    e, counts, old, elapsed = engine.project(z, CP.relative_transforms(sensor, travel, noisy, f))
                    verify_case(e, counts, old, cache.get(unit, config, f), len(z))
                    before = time.perf_counter()
                    mass[config, f-3] = e
                    coverage[config, f-3] = counts
                    write_s += time.perf_counter()-before
                    project_s += elapsed['project_s']
                    copy_s += elapsed['copy_s']
                    checked += 1
            before = time.perf_counter()
            mass.flush()
            coverage.flush()
            mass._mmap.close()
            coverage._mmap.close()
            mass = coverage = None
            write_s += time.perf_counter()-before
            if sha(source['path']) != source['sha256'] or plan_sha() != frozen['plan_sha256']:
                raise ValueError('Original observations/PLAN changed during unit')
            validate_inputs(frozen['inputs'])
            hashes = {'mass.npy': sha(partial_mass), 'coverage.npy': sha(partial_coverage)}
            # Same-volume atomic rename per completed array; receipt publishes the pair.
            partial_mass.rename(final_mass)
            partial_coverage.rename(final_coverage)
            result = dict(status='COMPLETE', split=split, unit=unit, request_sha256=digest,
                plan_sha256=frozen['plan_sha256'], output_sha256=hashes, shape=list(shape),
                mass_dtype='float16', coverage_dtype='uint8', frames=FRAMES.tolist(), configs=CONFIGS[split],
                slots=8, old3_windows_bit_exact=checked, latest_exact=True, coverage_recovery_exact=True,
                observation_sha256=source['sha256'], runtime=engine.runtime, disk_at_start=disk,
                elapsed_s=time.monotonic()-start, project_s=project_s, copy_s=copy_s, write_s=write_s)
            create_json(receipt, result)
            print('complete', split, unit, round(result['elapsed_s'], 2), 's', flush=True)
        print('COMPLETE history cache chunk', split, chunk, flush=True)
    except BaseException as error:
        failure = OUT / f'failure_{split}_c{k}of{n}_{time.time_ns()}.json'
        create_json(failure, dict(status='FAILED', split=split, chunk=chunk, error=repr(error),
                                  request_sha256=digest, partial_files='Preserved for inspection; no automatic deletion'))
        raise
    finally:
        for array in (mass, coverage):
            if array is not None:
                array.flush()
                array._mmap.close()
        if cache is not None:
            cache.close()
        if engine is not None:
            engine.close()


def check():
    cases = canary_cases()
    assert len(cases) == 18
    assert set(SPLITS['train']) == set(NR.SPLITS['train'])
    assert SPLITS['calib'] == MC.SPLITS['calib'] and SPLITS['evaluation'] == MC.SPLITS['evaluation']
    k = np.arange(28, dtype=np.uint8)
    restored = (k.astype(np.float64)/27).astype(np.float32).astype(np.float16)
    expected = np.asarray([np.ones(int(v)).sum()/27 for v in k], np.float64).astype(np.float32).astype(np.float16)
    np.testing.assert_array_equal(restored.view(np.uint16), expected.view(np.uint16))
    for split in SPLITS:
        cache = OldCache(split)
        cache.close()
    assert allocated_bytes(old_parts('train')[0][1]) > 0
    print('PASS CPU old9chunk identity coverage and18case route plan; all28 K levels exact; no GPU')
    print('Planned payload bytes including conservative headers', sum(unit_bytes(s)*len(u) for s, u in SPLITS.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'canary', 'prepare', 'run', 'finalize'))
    parser.add_argument('--split', choices=tuple(SPLITS))
    parser.add_argument('--chunk', default='0/1')
    args = parser.parse_args()
    if args.stage == 'check':
        check()
    elif args.stage == 'canary':
        canary()
    elif args.stage == 'prepare':
        _, digest = request(prepare=True)
        print('Prepared immutable history cache request', digest, flush=True)
    elif args.stage == 'finalize':
        finalize()
    else:
        if args.split is None:
            parser.error('--split required for run')
        run(args.split, args.chunk)
