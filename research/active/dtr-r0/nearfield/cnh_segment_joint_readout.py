"""Privileged segmentation-guided cross-zone readout, repaired-v2 Development.

No oracle range enters scoring. Original S2 residuals, transport and covariance
are unchanged. M1 uses f-weighted one/two-bin sums. M2 contrasts candidate and
observed neighbouring background, fitting nonnegative foreground and bounded
occlusion amplitudes in a two-statistic Gaussian GLRT. This is projected evidence,
not full histogram maximum likelihood or ST firmware emulation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cnh_segment_joint_candidates import CONDITIONS, candidate_frames

FAMILY = 'cnh-track-a-scale-v2-20260926'
WINDOWS_R = [(b, width) for width in (1, 2) for b in range(1, 17-width)]


def glrt(zf, zo, correlation, cap):
    """Two correlated unit-variance measurements with mean (a,b), a>=0,
    0<=b<=cap. zf is positive foreground contrast; zo negative background
    contrast. Exact rectangle-constrained quadratic optimum, no numerical fit.
    Require fitted a>0 AND observed zf>0: pure disappearance has no range.
    """
    zf, zo, rho, cap = np.broadcast_arrays(zf, zo, np.clip(correlation, -.999, .999), cap)
    candidates = [(np.maximum(0, zf-rho*zo), np.zeros_like(zo)),
                  (np.zeros_like(zf), np.clip(zo-rho*zf, 0, cap)),
                  (np.maximum(0, zf-rho*(zo-cap)), cap),
                  (np.maximum(0, zf), np.clip(zo, 0, cap))]
    best = np.zeros_like(zf)
    best_a = np.zeros_like(zf)
    for a, b in candidates:
        gain = (2*(a*zf+b*zo-rho*(a*zo+b*zf))-(a*a+b*b-2*rho*a*b))/(1-rho*rho)
        take = gain > best
        best, best_a = np.where(take, gain, best), np.where(take, a, best_a)
    return np.where((best_a > 0) & (zf > 0), np.sqrt(np.maximum(best, 0)), -np.inf)


def query_membership(mask, tq):
    """Candidate's angular subrays at hypothesized radius -> public query boxes.
    No true range, contributors, positive labels or size class accepted here.
    """
    from cnh_route_sensor import angular_rays
    from cnh_track_a_readout import BOXES, WIDTH
    rays, _ = angular_rays(16)
    # Raster order is row-zone, row-subray, col-zone, col-subray.
    image_rays = rays.reshape(8, 8, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(128, 128, 3)
    selected = image_rays[mask]
    ranges = np.array([(b+w/2)*WIDTH for b, w in WINDOWS_R])
    if not len(selected):
        return np.zeros((len(ranges), 6), bool), ranges
    points = ranges[:, None, None]*selected[None]
    points = points@tq[:3, :3].T+tq[:3, 3]
    return np.stack([((points >= lo) & (points <= hi)).all(-1).any(-1) for lo, hi in BOXES], 1), ranges


def observable_scope(candidates, absolute, memberships):
    """M2 bounded scene gate using masks/observations, never oracle categories.
    Small=f<=.5 in every zone and <=4 zone-equivalent area. Ring=8-neighbour
    dilation of occupied zones excluding ALL candidate zones; >=3 ring zones.
    Far background=strongest positive ring bin>=2; its +/-1-bin window has
    >=50% of positive nonzero-bin ring energy. Hypotheses must be wholly before
    that window. Query has exactly one angular candidate within possible range.
    """
    from scipy.ndimage import binary_dilation
    union = np.zeros((8, 8), bool)
    for c in candidates:
        union |= c['coverage'] > 0
    counts = np.sum([m.any(0) for m in memberships], axis=0) if memberships else np.zeros(6)
    rows = []
    for c, member in zip(candidates, memberships):
        f = c['coverage']
        ring = binary_dilation(f > 0, structure=np.ones((3, 3))) & ~union
        bg = absolute[ring].mean(0) if ring.any() else np.zeros(16)
        positive = np.maximum(bg, 0); positive[0] = 0
        peak = int(np.argmax(positive))
        bins = np.arange(max(1, peak-1), min(16, peak+2))
        concentration = float(positive[bins].sum()/max(positive.sum(), 1e-12))
        eligible = bool(f.max() <= .5 and f.sum() <= 4 and ring.sum() >= 3
                        and peak >= 2 and concentration >= .5)
        queries = member.any(0) & (counts == 1) & eligible
        hypotheses = np.array([b+w <= bins[0] for b, w in WINDOWS_R])
        rows.append(dict(ring=ring, background=bg, bins=bins, eligible=queries,
                         hypotheses=hypotheses, concentration=concentration))
    return rows


def linear_moments(weights, total, variance, past, torch):
    """Exact S2 covariance of arbitrary linear functionals, includes splat cov.
    past contains (A[current<-j], v[j]); no diagonal-variance approximation.
    """
    mean = weights@total
    var = (weights*weights)@variance
    transformed = []
    for A, v in past:
        a = weights@A
        var += (a*a)@v
        transformed.append((a, v))
    return mean, var.clamp_min(1e-9), transformed


def sequence_scores(hist, ambient, bias, tq, noisy, fields):
    import torch
    import torch.nn.functional as F
    import cnh_track_a_gpu_readout as g
    from cnh_scan_development import WINDOWS
    n = len(hist)
    r = g.T(hist)-g.T(bias)
    v = 16*g.T(ambient)[..., None]+g.T(bias).clamp_min(0)
    rf, vf = r.reshape(n, 1024), v.reshape(n, 1024)
    support = (g.query_weights(torch.as_tensor(tq, dtype=g.D64, device=g.DEV)) >= .75).reshape(n, 6, 8, 8, 16)
    pairs = [(i, j) for i in range(1, n) for j in range(max(0, i-3), i)]
    I = torch.tensor([i for i, j in pairs], device=g.DEV)
    J = torch.tensor([j for i, j in pairs], device=g.DEV)
    p = torch.as_tensor(noisy, dtype=g.D64, device=g.DEV)
    A = g.transport(torch.linalg.inv(p[I])@p[J], 1).to(g.DT)
    total = rf.clone().index_add_(0, I, torch.bmm(A, rf[J].unsqueeze(-1)).squeeze(-1))
    b = g.T(bias).flatten()
    bt = b.expand(n, -1).clone().index_add_(0, I, (A@b))
    absolute = (total+bt).reshape(n, 8, 8, 16).cpu().numpy()
    # Identical S2 covariance and windows; validate against saved G0 each unit.
    At = A.transpose(1, 2).reshape(len(pairs)*1024, 1, 8, 8, 16)
    baseline = torch.full((n, 6), -50., device=g.DEV, dtype=g.DT)
    for shape in WINDOWS:
        size = int(np.prod(shape))
        coeff = F.avg_pool3d(At, shape, stride=1)*size
        coeff = coeff.reshape(len(pairs), 1024, *coeff.shape[-3:])
        variance = g.boxsum(v, shape).clone().index_add_(0, I, torch.einsum('ps,psxyz->pxyz', vf[J], coeff*coeff))
        z = g.boxsum(total.reshape(n, 8, 8, 16), shape)/variance.clamp_min(1e-9).sqrt()
        admitted = g.boxsum(support.to(g.DT), shape) > size-.5
        baseline = torch.maximum(baseline, torch.where(admitted, z[:, None], -float('inf')).flatten(2).max(-1).values)
    base = baseline.double().cpu().numpy()
    del At, coeff
    out, diagnostics, object_results = {'M0': base}, {}, {}
    for condition in CONDITIONS:
        for method in ('M1', 'M2'):
            key = f'{method}__{condition}'
            out[key] = np.full((n, 6), -np.inf) if method == 'M1' else base.copy()
            diagnostics[key] = dict(eligible=np.zeros((n, 6), bool), candidate=np.zeros((n, 6), bool),
                                    distance=np.full((n, 6), np.nan), selected_id=np.full((n, 6), -999, int))
            object_results[key] = []
    for t in range(n):
        past = [(A[k], vf[j]) for k, (i, j) in enumerate(pairs) if i == t]
        for condition in CONDITIONS:
            candidates = fields[condition][t]
            if not candidates:
                continue
            maps = [query_membership(c['mask'], tq[t]) for c in candidates]
            scope = observable_scope(candidates, absolute[t], [x[0] for x in maps])
            # All candidate/range template linear statistics in one GPU batch.
            weights, layouts = [], []
            for c, sc in zip(candidates, scope):
                f = c['coverage'].astype(float)
                contrast = f.copy()
                if sc['ring'].any():
                    contrast[sc['ring']] -= f.sum()/sc['ring'].sum()
                d = np.zeros((8, 8, 16))
                # Negative ring-contrast sum detects missing farther return.
                d[:, :, sc['bins']] = -contrast[..., None]
                for b0, width in WINDOWS_R:
                    u = np.zeros((8, 8, 16)); u[:, :, b0:b0+width] = f[..., None]
                    w = np.zeros_like(u); w[:, :, b0:b0+width] = contrast[..., None]
                    layouts.append(len(weights)); weights.extend((u.flatten(), w.flatten(), d.flatten()))
            U = g.T(np.array(weights))
            mean, var, moved = linear_moments(U, total[t], vf[t], past, torch)
            # Covariance between foreground contrast and negative-background contrast.
            aidx = torch.arange(1, len(U), 3, device=g.DEV)
            bidx = aidx+1
            cov = (U[aidx]*U[bidx])@vf[t]
            for transformed, vv in moved:
                cov += (transformed[aidx]*transformed[bidx])@vv
            mean, var, cov = mean.cpu().numpy().reshape(-1, 3), var.cpu().numpy().reshape(-1, 3), cov.cpu().numpy()
            for ci, (c, sc, (member, ranges)) in enumerate(zip(candidates, scope, maps)):
                sl = slice(ci*len(WINDOWS_R), (ci+1)*len(WINDOWS_R))
                mm, vv = mean[sl], var[sl]
                m1 = mm[:, 0]/np.sqrt(vv[:, 0])
                zf, zo = mm[:, 1]/np.sqrt(vv[:, 1]), mm[:, 2]/np.sqrt(vv[:, 2])
                rho = cov[sl]/np.sqrt(vv[:, 1]*vv[:, 2])
                # Lost scene return is f*B per zone; the statistic itself uses f,
                # hence its maximum mean is sum(f^2)*B, not sum(f)*B.
                expected_missing = np.square(c['coverage']).sum()*max(0., sc['background'][sc['bins']].sum())
                cap = expected_missing/np.sqrt(vv[:, 2])
                m2 = glrt(zf, zo, rho, cap)
                for method, scores in (('M1', m1), ('M2', m2)):
                    key = f'{method}__{condition}'
                    adm = member.copy()
                    if method == 'M2':
                        adm &= sc['hypotheses'][:, None] & sc['eligible'][None]
                    s = np.where(adm, scores[:, None], -np.inf)
                    best = s.argmax(0)
                    ss = s[best, np.arange(6)]
                    dd = np.where(np.isfinite(ss), ranges[best], np.nan)
                    eligible = adm.any(0)
                    if method == 'M2':
                        first = eligible & ~diagnostics[key]['eligible'][t]
                        out[key][t, first] = -np.inf  # eligible but unconfirmed never silently falls back
                    diagnostics[key]['eligible'][t] |= eligible
                    diagnostics[key]['candidate'][t] |= member.any(0)
                    take = ss > out[key][t]
                    out[key][t, take] = ss[take]
                    diagnostics[key]['distance'][t, take] = dd[take]
                    diagnostics[key]['selected_id'][t, take] = c['id']
                    object_results[key].append(dict(frame=t, id=c['id'], score=ss, distance=dd))
    return out, diagnostics, object_results


def score_unit(job):
    source, output, unit = Path(job[0]), Path(job[1]), job[2]
    import torch
    import cnh_track_a_scale_evaluate as se
    from cnh_track_a_readout import noisy_poses
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent heavy CPU fallback')
    torch.cuda.reset_peak_memory_stats()
    se.sensor_module.FAMILY = FAMILY
    start = time.monotonic()
    split, records, step = se.unit_records(source/'geometry', source/'sensor', unit, -10, 1)
    bias = np.load(source/'bias.npy')
    pool = [np.asarray(x, int) for x in json.loads((source/'angular_templates.json').read_text())]
    geometry = json.loads((source/'geometry'/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    objects = {c['config']: c['objects'] for c in geometry['configs']}
    sizes = {c['config']: c['size_classes'] for c in geometry['configs']}
    with np.load(source/'sensor'/f'unit{unit:02d}-mount-10-oracle.npz') as f:
        oid = f['object_id']  # exclusively for privileged segmentation
    with np.load(source/'sensor'/f'unit{unit:02d}-mount-10-observations.npz') as f:
        assert int(f['rate']) == 5
        config = f['config']
    arrays, details, object_rows = {}, {}, {}
    # Evaluator opens oracle distance separately; sequence_scores never sees it.
    with np.load(source/'sensor'/f'unit{unit:02d}-mount-10-oracle.npz') as f:
        evaluator_range = f['raydistance']
    for rec in records:
        rows = np.flatnonzero(config == rec['config'])[::step]
        fields = candidate_frames(oid[rows], objects[rec['config']], unit, rec['config'], pool)
        scores, diag, cand = sequence_scores(rec['hist'], rec['ambient'], bias, rec['tq'],
                                             noisy_poses(rec['poses'], rec['ego_seed'], dt=.2), fields)
        truth = {}
        for t, frame in enumerate(oid[rows]):
            for c in fields['IDEAL'][t]:
                hit = frame == c['id']
                truth[t, c['id']] = float(np.median(evaluator_range[rows[t]][hit]))
        for key, value in scores.items():
            arrays.setdefault(key, []).append(value)
        for key, d in diag.items():
            d['truth_distance'] = np.array([[truth.get((t, int(i)), np.nan) for i in ids]
                                             for t, ids in enumerate(d['selected_id'])])
            for label in ('eligible', 'candidate', 'distance', 'truth_distance'):
                details.setdefault(key+'__'+label, []).append(d[label])
            mapping = {(row['frame'], row['id']): row for row in cand[key]}
            # Include missed/dropped real candidates in unknown denominator.
            for t, cs in enumerate(fields['IDEAL']):
                for c in cs:
                    mapping.setdefault((t, c['id']), dict(frame=t, id=c['id'], score=np.full(6, -np.inf), distance=np.full(6, np.nan)))
            for (t, ident), row in sorted(mapping.items()):
                row = dict(row, truth=truth.get((t, ident), np.nan),
                           size=sizes[rec['config']].get(str(ident), 'false' if ident < 0 else 'realistic'),
                           config=rec['config'])
                object_rows.setdefault(key, []).append(row)
    result = {k: np.concatenate(x) for k, x in {**arrays, **details}.items()}
    with np.load(source/'scores'/f'unit{unit:02d}.npz') as old:
        rel = float(np.max(np.abs(result['M0']-old['G0'])/np.maximum(1., np.abs(old['G0']))))
        if rel > 1e-5:
            raise AssertionError(f'S2 parity failed {unit}: {rel}')
        result['M0'] = old['G0'].copy()  # saved exact baseline after checked recomputation
        for condition in CONDITIONS:
            key = f'M2__{condition}'
            fallback = ~result[key+'__eligible']
            result[key][fallback] = result['M0'][fallback]
        for k in ('config', 'frame', 'labels', 'witness', 'strata', 'split', 'family'):
            result[k] = old[k]
    np.savez_compressed(output/'scores'/f'unit{unit:02d}.npz', **result)
    obj = {}
    for key, rows in object_rows.items():
        for label, field in [('score', 'score'), ('distance', 'distance'), ('truth', 'truth'),
                             ('frame', 'frame'), ('class', 'size'), ('id', 'id'), ('config', 'config')]:
            obj[key+'__object_'+label] = np.asarray([r[field] for r in rows])
    np.savez_compressed(output/'scores'/f'objects{unit:02d}.npz', **obj)
    receipt = dict(unit=unit, split=split, seconds=time.monotonic()-start, baseline_relative_parity=rel,
                   backend='cuda', device=torch.cuda.get_device_name(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (output/'scores'/f'unit{unit:02d}.json').write_text(json.dumps(receipt), encoding='utf-8')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--units', type=int, nargs='+')
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.source.name != FAMILY+'-dev-repaired' or a.source.resolve() == a.output.resolve():
        raise ValueError('Only repaired-v2 Development and separate output allowed')
    allowed = [u for u in range(96, 192) if u != 143]
    units = a.units or allowed
    if not set(units) <= set(allowed):
        raise ValueError('Unsupported cohort')
    (a.output/'scores').mkdir(parents=True, exist_ok=True)
    jobs = [(str(a.source), str(a.output), u) for u in units if not (a.output/'scores'/f'unit{u:02d}.json').exists()]
    progress = dict(status='running', total=len(units), complete=len(units)-len(jobs),
                    source=str(a.source), code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record = a.output/'progress.json'
    start = time.monotonic()
    try:
        record.write_text(json.dumps(progress), encoding='utf-8')
        with ProcessPoolExecutor(a.workers) as pool:
            for future in as_completed([pool.submit(score_unit, j) for j in jobs]):
                r = future.result()
                progress.update(complete=progress['complete']+1, last=r, elapsed_s=time.monotonic()-start)
                record.write_text(json.dumps(progress), encoding='utf-8')
                print(json.dumps(r), flush=True)
        progress['status'] = 'complete'
    except BaseException as exc:
        progress.update(status='failed', error=str(exc)); raise
    finally:
        record.write_text(json.dumps(progress, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
