"""Heading uncertainty vs the 0-2 cm contact target (no rendering, no training).

The simulator corridor follows the known travel direction. A real walker's
future path deviates from the corridor the sensor assumes. Model: constant
heading error delta ~ N(0, sigma) over the remaining forward distance d, so the
actual intrusion is a = o + d*tan(delta) for nominal intrusion o (>0 inside).
Sensor alarms are fixed per row (the sensor sees the same geometry); only the
contact truth moves. Data: near-range evaluation units 94000-94095, frozen NEAR
5-seed smoothed final-frame scores, thresholds at 10%/30% clear-row alarm rate
per group (clear = other height band, label 0), as in cnh_near_range.evaluate.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

import cnh_structure_space as SS
import cnh_near_range as NR

OUT = SS.WORK/'cnh-heading-uncertainty-20261002'
SIGMAS_DEG = (0., .5, 1., 2., 4.)
RANGES = ((.6, 1.2), (1.2, 1.6), (1.6, 2.1), (2.1, 2.6))
BINS = {'contact 0-2': (0., .02), 'contact 2-5': (.02, .05), 'contact 5-10': (.05, .10),
        'pass 0-5': (-.05, 0.), 'pass 5-10': (-.10, -.05)}
OFFSET_SUPPORT = (-.10, .15)  # nominal offsets sampled by cnh_near_range evaluation scenes
DRAWS = 400
RULE = ('Reading fixed before results: for each range bin, sigma* = smallest sigma at which a PERFECT nominal '
        'sensor (alarm iff nominal intrusion > 0) recalls fewer actual 0-2 cm contacts than the deployed NEAR '
        'readout does at sigma=0 and 10% clear alarms. If sigma* <= 1 deg in ranges >= 1.2 m, report that the 0-2 cm '
        'target at those ranges is limited by path uncertainty at least as much as by sensing; otherwise not. '
        'sigma is a sweep, not a measured value for any user or device. Cells whose kernel puts >5% of mass '
        'outside the sampled nominal offsets are flagged TRUNCATED and not interpreted.')


def rows():
    scores = SS.read(NR.OUT/'scores_NEAR.npz'); out = []
    for u in NR.SPLITS['evaluation']:
        d = SS.read(NR.OUT/'features/evaluation'/f'unit{u}.npz'); scenes = NR.scenes_for(u)
        lab = d['labels'].reshape(40, 16, 6)[:, -1, 2:4]
        for c in range(40):
            tq = int(d['group'][c])
            for q in (0, 1):
                out.append(dict(unit=u, g=q, cond=str(d['family'][c]), rng=float(scenes[c]['meta']['range']),
                                lab=int(lab[c, q]), off=-float(d['margin'][c]) if q == tq else None,
                                s=float(scores[str(u)][c, q])))
    return out


def truncated(sigma_lat, lo, hi):
    """Kernel mass outside the sampled offsets for an actual intrusion at the bin centre."""
    if sigma_lat == 0:
        return 0.
    a = (lo+hi)/2
    return float(norm.cdf((OFFSET_SUPPORT[0]-a)/sigma_lat)+norm.sf((OFFSET_SUPPORT[1]-a)/sigma_lat))


def sigma_star(cells, range_key, sigmas):
    """Find a crossing only among cells interpretable under the frozen rule."""
    base = cells[f'{range_key}|0.0']['contact 0-2']['alarm']['NEAR@10%']
    star = next((s for s in sigmas if s > 0 and
                 cells[f'{range_key}|{s}']['contact 0-2']['truncated_mass'] <= .05 and
                 cells[f'{range_key}|{s}']['contact 0-2']['alarm']['perfect m=0cm'] < base), None)
    return dict(sigma_star=star, near10_at_sigma0=base)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    SS.save(OUT/'PLAN.json', dict(rule=RULE, sigmas_deg=SIGMAS_DEG, draws=DRAWS, ranges=RANGES, bins=BINS,
            model='NEAR 5-seed smoothed final-frame (scores_NEAR.npz)', script_sha256=SS.sha(__file__),
            scores_sha256=SS.sha(NR.OUT/'scores_NEAR.npz'),
            limits=['constant heading error only; no gait sway, curvature, reaction or replanning',
                    'sensor decision at one distance; no sequence/stop timing',
                    'consumed near-range Development; clear rows are other-height-band objects']))
    R = rows(); target = [r for r in R if r['off'] is not None]
    thr = {(fa, q): float(np.quantile([r['s'] for r in R if r['g'] == q and r['off'] is None and r['lab'] == 0], 1-fa))
           for fa in (.10, .30) for q in (0, 1)}
    o = np.array([r['off'] for r in target]); d = np.array([r['rng'] for r in target])
    alarm = {fa: np.array([r['s'] >= thr[(fa, r['g'])] for r in target]) for fa in (.10, .30)}
    rng = np.random.default_rng(2026100201)
    z = rng.standard_normal((len(target), DRAWS))
    result = dict(status='COMPLETE', rule=RULE, n_target_rows=len(target), thresholds={f'{k[0]}|{k[1]}': v for k, v in thr.items()},
                  cells={}, sigma_star_deg={})
    for lo_r, hi_r in RANGES:
        keep = (d >= lo_r) & (d < hi_r); dm = (lo_r+hi_r)/2
        for sig in SIGMAS_DEG:
            actual = o[keep, None]+d[keep, None]*np.tan(np.deg2rad(sig)*z[keep])
            arms = {'NEAR@10%': alarm[.10][keep], 'NEAR@30%': alarm[.30][keep],
                    **{f'perfect m={m}cm': o[keep] > -m/100 for m in (0, 5, 10)}}
            cell = {}
            for name, (lo, hi) in BINS.items():
                inbin = (actual > lo) & (actual <= hi) if lo >= 0 else (actual > lo) & (actual <= hi)
                n_eff = int(inbin.sum())
                cell[name] = dict(pairs=n_eff, rows=int(inbin.any(1).sum()),
                                  truncated_mass=truncated(dm*np.tan(np.deg2rad(sig)), lo, hi),
                                  alarm={a: float((np.broadcast_to(v[:, None], inbin.shape)[inbin]).mean()) if n_eff else None
                                         for a, v in arms.items()})
            result['cells'][f'{lo_r}-{hi_r}|{sig}'] = cell
        result['sigma_star_deg'][f'{lo_r}-{hi_r}'] = sigma_star(result['cells'], f'{lo_r}-{hi_r}', SIGMAS_DEG)
    far = [v['sigma_star'] for k, v in result['sigma_star_deg'].items() if float(k.split('-')[0]) >= 1.2]
    result['reading'] = ('PATH_LIMITED_AT_<=1DEG' if all(s is not None and s <= 1 for s in far)
                         else 'NOT_PATH_LIMITED_AT_<=1DEG')
    SS.save(OUT/'result.json', result)
    for k, cell in result['cells'].items():
        line = ' '.join(f"{b}:" + '/'.join('--' if v is None else f'{v*100:.0f}' for v in c['alarm'].values())
                        + ('T' if c['truncated_mass'] > .05 else '') for b, c in cell.items())
        print(f'{k:12s} {line}')
    print('arms order:', list(next(iter(result['cells'].values()))['contact 0-2']['alarm']))
    print(json.dumps(result['sigma_star_deg']), result['reading'])


if __name__ == '__main__':
    main()
