"""Evaluation-only privileged mask controls; no range, outcomes or training.

BG contains every visible nonnegative first-hit identity, including BACKGROUND.
False masks use the existing calib footprint pool, rasterized as whole zones.
FA5 is the exact first-five prefix of FA10. Composite controls first union BG
and fake masks, then shift the entire union by 0.5 zone using one registration
angle per config sequence. Count/256 coverage matches learned-mask channels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np
from cnh_candidate_quality_readout import fake_footprints
from cnh_segment_learned_masks import raster, coverage, shift_mask, rng_for

CONDITIONS = ('IDEAL', 'FA5', 'FA10', 'BG', 'BG_FA5_SHIFT05', 'BG_FA10_SHIFT05')


def sequence_masks(oid, objects, unit, config, family, pool):
    if oid.ndim != 4 or oid.shape[1:] != (8, 8, 256):
        raise ValueError('Expected [T,8,8,256] identity array')
    foreground = [int(o['id']) for o in objects if o['category'] != 'BACKGROUND']
    angle = rng_for(family, unit, config, -1, 'registration').uniform(0, 2*np.pi)
    dx, dy = np.rint(8*np.array([np.cos(angle), np.sin(angle)])).astype(int)
    output = {name: np.empty((len(oid), 8, 8), np.float16) for name in CONDITIONS}
    for t, frame in enumerate(oid):
        ideal = raster(np.isin(frame, foreground))
        bg = raster(frame >= 0)
        fake, _ = fake_footprints(pool, rng_for(family, unit, config, t, 'controls-fake-prefix'), 10)
        first5 = np.logical_or.reduce(fake[:5])
        all10 = np.logical_or.reduce(fake)
        fake5 = np.repeat(np.repeat(first5, 16, 0), 16, 1)
        fake10 = np.repeat(np.repeat(all10, 16, 0), 16, 1)
        pixels = dict(IDEAL=ideal, FA5=ideal | fake5, FA10=ideal | fake10, BG=bg,
                      BG_FA5_SHIFT05=shift_mask(bg | fake5, dx, dy),
                      BG_FA10_SHIFT05=shift_mask(bg | fake10, dx, dy))
        for name, mask in pixels.items():
            output[name][t] = coverage(mask)
    return output


def coverage_stats(arrays):
    return {key: dict(mean_coverage=float(x.astype(np.float64).mean()),
                     fraction_zones_nonzero=float((x > 0).mean()),
                     fraction_zones_saturated=float((x == 1).mean()),
                     fraction_frames_fully_saturated=float((x == 1).all((1, 2)).mean()))
            for key, x in arrays.items()}


def build_unit(job):
    geometry, oracle_dirs, metadata, output, pool_path, family, unit = job
    started = time.monotonic()
    target = Path(output)/f'unit{unit:03d}.npz'
    receipt = target.with_suffix('.json')
    if receipt.exists() and target.exists():
        return dict(json.loads(receipt.read_text()), resumed=True)
    source = next(Path(s)/f'unit{unit:02d}-mount-10-oracle.npz' for s in oracle_dirs
                  if (Path(s)/f'unit{unit:02d}-mount-10-oracle.npz').exists())
    with np.load(source) as f:
        oid = f['object_id']
    data = json.loads((Path(geometry)/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    with np.load(Path(metadata)/f'unit{unit:03d}.npz') as f:
        cfg, frame, split, old_ideal = f['config'], f['frame'], str(f['split']), f['IDEAL']
        if str(f['family']) != family:
            raise ValueError('Mask metadata family mismatch')
    if split not in ('calib', 'audit') or split != data['split']:
        raise ValueError('Only matching calib/audit split')
    if len(oid) != len(cfg) or not np.array_equal(cfg, np.repeat([c['config'] for c in data['configs']], 12)):
        raise ValueError('Geometry/oracle/config order mismatch')
    if not np.array_equal(frame, np.tile(np.arange(12), len(data['configs']))):
        raise ValueError('Frame order mismatch')
    pool = [np.asarray(x, int) for x in json.loads(Path(pool_path).read_text())]
    arrays = {name: np.empty((len(cfg), 8, 8), np.float16) for name in CONDITIONS}
    for c in data['configs']:
        ix = np.flatnonzero(cfg == c['config'])
        values = sequence_masks(oid[ix], c['objects'], unit, c['config'], family, pool)
        for key in CONDITIONS:
            arrays[key][ix] = values[key]
    if not np.array_equal(arrays['IDEAL'], old_ideal):
        raise AssertionError('IDEAL differs from existing learned masks')
    np.savez_compressed(target, **arrays, config=cfg, frame=frame, split=np.array(split), family=np.array(family))
    result = dict(unit=unit, frames=len(cfg), split=split, seconds=time.monotonic()-started,
                  ideal_exact=True, source=str(source), stats=coverage_stats(arrays))
    receipt.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry', 'metadata', 'output', 'pool', 'family'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--oracle', nargs='+', required=True)
    p.add_argument('--units', type=int, nargs='+', required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    allowed = (set(range(96, 192))-{143} if a.family == 'cnh-track-a-scale-v2-20260926'
               else set(range(96)) if a.family == 'cnh-track-a-scale-v4-20260926' else set())
    if not set(a.units) <= allowed or not a.units:
        raise ValueError('Only existing v2/v4 evaluation units')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.geometry, a.oracle, a.metadata, a.output, a.pool, a.family, u) for u in a.units]
    progress = dict(status='running', total=len(jobs), complete=0)
    started = time.monotonic()
    rows = []
    try:
        with ProcessPoolExecutor(a.workers) as executor:
            pending = [executor.submit(build_unit, job) for job in jobs]
            for future in as_completed(pending):
                try:
                    result = future.result()
                except BaseException:
                    for other in pending:
                        other.cancel()
                    raise
                rows.append(result)
                progress.update(complete=len(rows), last=result['unit'], elapsed_s=time.monotonic()-started)
                (out/'progress.json').write_text(json.dumps(progress), encoding='utf-8')
                print(json.dumps({k:result[k] for k in ('unit', 'frames', 'ideal_exact', 'seconds')}), flush=True)
        summary = dict(units=len(rows), frames=sum(x['frames'] for x in rows), ideal_exact=True, stats={})
        for key in CONDITIONS:
            summary['stats'][key] = {stat: float(np.average([x['stats'][key][stat] for x in rows],
                                                           weights=[x['frames'] for x in rows]))
                                    for stat in rows[0]['stats'][key]}
        (out/'coverage_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        progress['status'] = 'complete'
    except BaseException as exc:
        progress.update(status='failed', error=str(exc)); raise
    finally:
        (out/'progress.json').write_text(json.dumps(progress, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
