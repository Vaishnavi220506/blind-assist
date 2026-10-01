"""Privileged foreground union channel; geometry identity only, no range/labels.

Coverage is occupied subray count /256. Perturbed object masks are UNIONED,
never added (overlaps remain <=1). Fixed evaluation shifts share a direction
throughout a sequence and across objects, representing registration bias.
Training MIXED redraws error per frame/epoch. All seeds share the same 20
augmentation banks, just as they share the original training observations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import maximum_filter
from cnh_candidate_quality_readout import fake_footprints

CONDITIONS = ('IDEAL', 'SHIFT025', 'SHIFT05', 'SHIFT1', 'DILATE05', 'DILATE1',
              'DROP20', 'DROP40', 'FA1', 'FA3', 'MIXED')


def rng_for(family, unit, config, frame, purpose):
    key = f'20260928|{family}|{unit}|{config}|{frame}|{purpose}'
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))


def raster(hit):
    return hit.reshape(8, 8, 16, 16).transpose(0, 2, 1, 3).reshape(128, 128)


def coverage(mask):
    return mask.reshape(8, 16, 8, 16).mean((1, 3)).astype(np.float32)


def shift_mask(mask, dx, dy):
    """Integer subray translation, zero outside FOV, never wraps."""
    out = np.zeros_like(mask)
    if abs(dx) < 128 and abs(dy) < 128:
        sx, sy = slice(max(0, -dx), min(128, 128-dx)), slice(max(0, -dy), min(128, 128-dy))
        tx, ty = slice(max(0, dx), min(128, 128+dx)), slice(max(0, dy), min(128, 128+dy))
        out[ty, tx] = mask[sy, sx]
    return out


def perturb(base, pool, rng, condition, fixed_angle=None):
    shift = {'SHIFT025': .25, 'SHIFT05': .5, 'SHIFT1': 1}.get(condition, 0.)
    dilation = {'DILATE05': .5, 'DILATE1': 1}.get(condition, 0.)
    drop = {'DROP20': .2, 'DROP40': .4}.get(condition, 0.)
    count = {'FA1': 1, 'FA3': 3}.get(condition, 0)
    if condition == 'MIXED':
        shift, dilation, drop, count = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, .4), int(rng.integers(4))
    angle = rng.uniform(0, 2*np.pi) if fixed_angle is None else fixed_angle
    dx, dy = np.rint(16*shift*np.array([np.cos(angle), np.sin(angle)])).astype(int)
    radius = int(np.rint(dilation*16))
    result = []
    for ident, mask in base:
        if rng.random() < drop:
            continue
        m = shift_mask(mask, dx, dy) if shift else mask.copy()
        if radius:
            m = maximum_filter(m, size=2*radius+1, mode='constant', cval=0)
        if m.any():
            result.append(dict(id=ident, mask=m, coverage=coverage(m)))
    fake, _ = fake_footprints(pool, rng, count)
    for i, m in enumerate(fake):
        pixels = np.repeat(np.repeat(m, 16, 0), 16, 1)
        result.append(dict(id=-1-i, mask=pixels, coverage=coverage(pixels)))
    return result


def base_frames(oid, objects):
    ids = [int(o['id']) for o in objects if o['category'] != 'BACKGROUND']
    return [[(i, raster(frame == i)) for i in ids if (frame == i).any()] for frame in oid]


def eval_candidates(oid, objects, unit, config, family, pool):
    base = base_frames(oid, objects)
    angle = rng_for(family, unit, config, -1, 'registration').uniform(0, 2*np.pi)
    return {condition: [perturb(b, pool, rng_for(family, unit, config, t, condition), condition,
                                 angle if condition.startswith('SHIFT') else None)
                        for t, b in enumerate(base)] for condition in CONDITIONS}


def union_coverage(candidates):
    m = np.zeros((128, 128), bool)
    for c in candidates:
        m |= c['mask']
    return coverage(m)


def build_unit(job):
    geometry, oracle_dirs, features, output, pool_path, family, unit, training = job
    started = time.monotonic()
    target = Path(output)/f'unit{unit:03d}.npz'
    if target.exists():
        return dict(unit=unit, resumed=True)
    source = next(Path(s)/f'unit{unit:02d}-mount-10-oracle.npz' for s in oracle_dirs
                  if (Path(s)/f'unit{unit:02d}-mount-10-oracle.npz').exists())
    with np.load(source) as f:
        oid = f['object_id']  # no other oracle array is opened
    data = json.loads((Path(geometry)/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    with np.load(Path(features)/f'unit{unit:03d}.npz') as f:
        cfg, frame, split = f['config'], f['frame'], str(f['split'])
    pool = [np.asarray(x, int) for x in json.loads(Path(pool_path).read_text())]
    if len(cfg) != len(oid) or len(oid) != len(data['configs'])*12:
        raise ValueError('Require identical 5Hz oracle/feature ordering')
    arrays = {k: np.empty((len(cfg), 8, 8), np.float16) for k in (('IDEAL',) if training else CONDITIONS)}
    if training:
        if split != 'train':
            raise ValueError('Augmentation bank only for train')
        arrays['AUG'] = np.empty((20, len(cfg), 8, 8), np.float16)
    for c in data['configs']:
        ix = np.flatnonzero(cfg == c['config'])
        if not np.array_equal(frame[ix], np.arange(12)):
            raise ValueError('Frame mismatch')
        if training:
            for t, base in enumerate(base_frames(oid[ix], c['objects'])):
                arrays['IDEAL'][ix[t]] = coverage(np.logical_or.reduce([m for _, m in base])) if base else 0
                for ep in range(20):
                    candidates = perturb(base, pool, rng_for(family, unit, c['config'], t, f'train-{ep}'), 'MIXED')
                    arrays['AUG'][ep, ix[t]] = union_coverage(candidates)
        else:
            candidates = eval_candidates(oid[ix], c['objects'], unit, c['config'], family, pool)
            for cond in CONDITIONS:
                arrays[cond][ix] = np.asarray([union_coverage(cs) for cs in candidates[cond]])
    np.savez_compressed(target, **arrays, config=cfg, frame=frame, split=np.array(split), family=np.array(family))
    return dict(unit=unit, seconds=time.monotonic()-started, split=split, training=training)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry', 'features', 'output', 'pool', 'family'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--oracle', nargs='+', required=True)
    p.add_argument('--units', nargs='+', type=int, required=True)
    p.add_argument('--train', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.geometry, a.oracle, a.features, a.output, a.pool, a.family, u, a.train) for u in a.units]
    progress = dict(status='running', total=len(jobs), complete=0)
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(a.workers) as executor:
            for future in as_completed([executor.submit(build_unit, j) for j in jobs]):
                r = future.result(); print(json.dumps(r), flush=True)
                progress.update(complete=progress['complete']+1, last=r, elapsed_s=time.monotonic()-start)
                (out/'progress.json').write_text(json.dumps(progress), encoding='utf-8')
        progress['status'] = 'complete'
    except BaseException as e:
        progress.update(status='failed', error=str(e)); raise
    finally:
        (out/'progress.json').write_text(json.dumps(progress), encoding='utf-8')


if __name__ == '__main__':
    main()
