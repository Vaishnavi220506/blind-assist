"""Reproduce existing v2 train / v4 calib observations while adding oracle.

CPU-only replay of identical geometry/family/seeds/rate, never new sampling.
Every observation key/shape/dtype/value must match the existing recording exactly.
Checkpoints only attest complete paired outputs plus exact comparison receipts.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

FAMILY = 'cnh-track-a-scale-v2-20260926'
V4_FAMILY = 'cnh-track-a-scale-v4-20260926'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def compare_observations(original, reproduced):
    with np.load(original) as old, np.load(reproduced) as new:
        if set(old.files) != set(new.files):
            raise AssertionError('Observation key mismatch')
        shapes = {}
        for key in old.files:
            a, b = old[key], new[key]
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                raise AssertionError(f'Observation mismatch: {key}')
            shapes[key] = list(a.shape)
        if int(old['rate']) != 5:
            raise AssertionError('Only existing 5 Hz train stream')
    return shapes


def prepare_unit(job):
    geometry, original, output, unit = Path(job[0]), Path(job[1]), Path(job[2]), job[3]
    family = job[4] if len(job) > 4 else FAMILY
    import cnh_track_a_v13_sensor as sensor
    sensor.FAMILY, sensor.FORCE_RATE = family, 5
    stem = f'unit{unit:02d}-mount-10'
    meta = json.loads((geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    expected_split, limit = ('train', 96) if family == FAMILY else ('calib', 32)
    if family not in (FAMILY, V4_FAMILY) or meta['split'] != expected_split or not 0 <= unit < limit:
        raise ValueError('Only v2 train 0..95 / v4 calib 0..31')
    before = time.monotonic()
    existing = original/f'{stem}-observations.npz'
    generated = output/f'{stem}-observations.npz'
    oracle = output/f'{stem}-oracle.npz'
    receipt = output/f'{stem}-exact.json'
    source_hash = sha(existing)
    geometry_hash = sha(geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json')
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if (prior.get('original_sha256') == source_hash and prior.get('geometry_sha256') == geometry_hash
                and generated.exists() and oracle.exists()
                and prior.get('generated_sha256') == sha(generated) and prior.get('oracle_sha256') == sha(oracle)):
            return dict(prior, resumed=True)
        raise RuntimeError(f'Existing checkpoint identity mismatch for {unit}; preserve evidence')
    rate, frames = sensor.run(geometry, output, unit, -10,
                              oracle_train=expected_split == 'train', oracle_calib=expected_split == 'calib')
    shapes = compare_observations(existing, generated)
    with np.load(oracle) as f:
        if f['object_id'].shape != (frames, 8, 8, 256):
            raise AssertionError('Incorrect oracle shape')
    result = dict(unit=unit, exact=True, compared_keys=shapes, family=family, split=expected_split, rate=rate,
                  frames=frames, mount=-10, backend='numpy-cpu', seconds=time.monotonic()-before,
                  original_sha256=source_hash, generated_sha256=sha(generated),
                  oracle_sha256=sha(oracle), geometry_sha256=geometry_hash)
    receipt.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--units', nargs='+', type=int)
    p.add_argument('--family', choices=(FAMILY, V4_FAMILY), default=FAMILY)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    allowed = set(range(96 if a.family == FAMILY else 32))
    a.units = a.units or sorted(allowed)
    if a.output.resolve() == a.original.resolve() or not set(a.units) <= allowed:
        raise ValueError('Require separate output and allowed units')
    source_name = a.family+('-dev-repaired' if a.family == FAMILY else '-v1')
    if a.geometry.parent.name != source_name:
        raise ValueError('Only repaired-v2 / recorded-v4 geometry')
    if not 1 <= a.workers <= 16:
        raise ValueError('Workers must be 1..16')
    a.output.mkdir(parents=True, exist_ok=True)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    progress = dict(status='running', total=len(a.units), complete=0, workers=a.workers,
                    family=a.family, backend='numpy-cpu', geometry=str(a.geometry), original=str(a.original),
                    script_sha256=sha(__file__))
    path = a.output/'progress.json'
    start = time.monotonic()
    try:
        path.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        with ProcessPoolExecutor(a.workers) as pool:
            pending = [pool.submit(prepare_unit, (str(a.geometry), str(a.original), str(a.output), u, a.family)) for u in a.units]
            for future in as_completed(pending):
                try:
                    result = future.result()
                except BaseException:
                    for remaining in pending:
                        remaining.cancel()
                    raise
                progress.update(complete=progress['complete']+1, last=result['unit'], elapsed_s=time.monotonic()-start)
                path.write_text(json.dumps(progress, indent=2), encoding='utf-8')
                with (a.output/'receipts.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(result)+'\n')
                print(json.dumps({k:result[k] for k in ('unit', 'exact', 'frames', 'seconds')}), flush=True)
        progress['status'] = 'complete'
    except BaseException as exc:
        progress.update(status='failed', error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(progress, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
