"""Frozen signed observation-consistency proxy; no scene/truth/model access.

CVR's finite-volume B defines A = row_normalize(B.T), not a surface renderer.
Only z1, scene and frame are loaded from retained observations. No labels,
boxes, ray casting, de-standardization, training or alarm score is used.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits

import cnh_cvr_projection as P
import cnh_cvr_pilot as CP

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-multiframe-observation-probe-20261002'
SOURCE = ROOT/'artifacts.local/work/cnh-near-range-20261001/features/train'
RUN = 'CNH_MULTIFRAME_OBSERVATION_PROBE_20261002'
UNITS = list(range(93000, 93006))
FRAMES = np.arange(8, 16)
ARMS = ('backprojection', 'ridge', 'ridge_reversed')
BOOT_SEED = 2026100231
NVOX = int(np.prod(P.SHAPE))
LOW, HIGH = P.LOW, P.LOW+P.STEP*np.asarray(P.SHAPE)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def envelope(transform, ulo, uhi, vlo, vhi, rlo, rhi):
    """Enclose EVERY shell/cone point analytically, not just its corners.

    q_i=t_i+r*(R_i0*u+R_i1*v+R_i2)/sqrt(1+u²+v²).
    Interval arithmetic deliberately discards numerator/denominator dependence,
    hence overestimates the range. Outward rounding plus 1e-12 protects double
    arithmetic; acceptance also requires that safety margin inside the grid.
    """
    def square_min(lo, hi):
        return np.where((lo <= 0) & (hi >= 0), 0., np.minimum(lo*lo, hi*hi))
    denlo = np.sqrt(1+square_min(ulo, uhi)+square_min(vlo, vhi))
    denhi = np.sqrt(1+np.maximum(ulo*ulo, uhi*uhi)+np.maximum(vlo*vlo, vhi*vhi))
    lower, upper = [], []
    for row in transform[:3]:
        a, b, c, translation = row
        nlo = np.minimum(a*ulo, a*uhi)+np.minimum(b*vlo, b*vhi)+c
        nhi = np.maximum(a*ulo, a*uhi)+np.maximum(b*vlo, b*vhi)+c
        ratios = np.stack([nlo/denlo, nlo/denhi, nhi/denlo, nhi/denhi])
        dlo, dhi = ratios.min(0), ratios.max(0)
        products = np.stack([rlo*dlo, rlo*dhi, rhi*dlo, rhi*dhi])
        lower.append(np.nextafter(translation+products.min(0), -np.inf)-1e-12)
        upper.append(np.nextafter(translation+products.max(0), np.inf)+1e-12)
    return np.stack(lower, -1), np.stack(upper, -1)


def contained_cells(transform):
    yy, xx, rr = np.indices((8, 8, 16)).reshape(3, -1)
    width = 2*P.EDGE/8
    lo, hi = envelope(transform, -P.EDGE+xx*width, -P.EDGE+(xx+1)*width,
        -P.EDGE+yy*width, -P.EDGE+(yy+1)*width, rr*P.WIDTH, (rr+1)*P.WIDTH)
    return ((lo >= LOW+1e-12) & (hi <= HIGH-1e-12)).all(1)


def projection(transform):
    """Exact original grid, 27 midpoint memberships and volume coefficients."""
    points = P.grid()[1].reshape(-1, 3)
    local = (points-transform[:3, 3])@transform[:3, :3]
    radius = np.linalg.norm(local, axis=1)
    ij = np.floor((local[:, :2]/np.maximum(local[:, 2:3], 1e-30)+P.EDGE)/(2*P.EDGE)*8).astype(np.int64)
    bins = np.floor(radius/P.WIDTH).astype(np.int64)
    valid = (local[:, 2] > 0) & (ij >= 0).all(1) & (ij < 8).all(1) & (bins >= 0) & (bins < 16)
    index = (ij[:, 1]*8+ij[:, 0])*16+bins
    voxel = np.repeat(np.arange(NVOX), P.SUB**3)
    # Original torch valid*Python float happens in float32 before /double volumes.
    numerator = float(np.float32(np.prod(P.STEP))/np.float32(P.SUB**3))
    weight = numerator/P.cell_volumes().reshape(-1)[index[valid]]
    b = sparse.coo_matrix((weight, (voxel[valid], index[valid])), shape=(NVOX, 1024)).tocsr()
    b.sum_duplicates()
    a = b.T.tocsr()
    total = np.asarray(a.sum(1)).ravel()
    a = sparse.diags(np.divide(1., total, out=np.zeros_like(total), where=total > 0))@a
    return a.tocsr(), b, contained_cells(transform) & (total > 0)


def matrices(transforms):
    built = [projection(t) for t in transforms]
    operators, inside = [x[0] for x in built], np.stack([x[2] for x in built])
    # Correct/reversed arms get the SAME sensor-bin IDs at each observed frame.
    train_mask = inside[:7] & inside[:7][::-1]
    correct = sparse.vstack([operators[i][train_mask[i]] for i in range(7)], format='csr')
    wrong = sparse.vstack([operators[6-i][train_mask[i]] for i in range(7)], format='csr')
    support = (np.asarray(correct.sum(0)).ravel() > 0) & (np.asarray(wrong.sum(0)).ravel() > 0)
    heldout = inside[7] & (np.asarray(operators[7][:, ~support].getnnz(axis=1)).ravel() == 0)
    return correct, wrong, operators[7][heldout], train_mask, heldout, dict(
        analytically_contained_per_frame=inside.sum(1).tolist(), training_common_per_frame=train_mask.sum(1).tolist(),
        heldout_common=int(heldout.sum()), supported_voxels=int(support.sum()))


def reconstruct(a, y):
    """Fixed spectral scaling and zero-start CG; a signed field, not occupancy."""
    x = np.ones(a.shape[1], dtype=np.float64)
    x /= np.linalg.norm(x)
    for _ in range(30):
        next_x = a.T@(a@x)
        norm = np.linalg.norm(next_x)
        if norm == 0:
            raise ValueError('No nonzero observation operator')
        x = next_x/norm
    scale = float(np.dot(x, a.T@(a@x)))
    ridge = .01*scale
    rhs = a.T@y
    op = LinearOperator((a.shape[1], a.shape[1]), matvec=lambda q: a.T@(a@q)+ridge*q, dtype=np.float64)
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    solution, info = cg(op, rhs, x0=np.zeros_like(rhs), rtol=1e-6, atol=0., maxiter=50, callback=callback)
    residual = float(np.linalg.norm(op@solution-rhs)/max(np.linalg.norm(rhs), 1e-30))
    if info < 0 or not np.isfinite(solution).all():
        raise RuntimeError('CG failed numerically')
    coverage = np.asarray(a.sum(0)).ravel()
    baseline = np.divide(rhs, coverage, out=np.zeros_like(rhs), where=coverage > 0)
    return solution, baseline, dict(spectral_scale_30_iterations=scale, lambda_absolute=ridge,
        cg_info=int(info), cg_iterations=iterations[0], relative_normal_equation_residual=residual)


def source_identity():
    # motion_metadata imports only frozen noisy_poses, never the scene generator.
    CP.motion_metadata(UNITS[0], 0)
    import cnh_track_a_readout as readout
    paths = [Path(__file__), Path(P.__file__), Path(CP.__file__), Path(readout.__file__)]
    return {str(p): sha(p) for p in paths}


def plan():
    return dict(run=RUN, units=UNITS, configurations=list(range(22)), frames=FRAMES.tolist(),
        fitting_frames=list(range(8, 15)), heldout_frame=15, observation_keys=['z1', 'scene', 'frame'],
        source_sha256=source_identity(), observation_sha256={str(SOURCE/f'unit{u}.npz'): sha(SOURCE/f'unit{u}.npz') for u in UNITS},
        operator='Original CVR grid/27midpoints B; A=row_normalize(B.T); normalized signed intensity volume-average surrogate',
        mask='Analytic interval enclosure of complete angular/radial cells within grid. Correct/reversed per-frame binID intersection. Heldout nonzero voxel columns all supported by common seven-frame training operators.',
        inverse=dict(lambda_relative=.01, power_iterations=30, power_start='normalized all ones', cg_maxiter=50, cg_rtol=1e-6, cg_atol=0., cg_start='zeros'),
        baseline='A.T@y / (A.T@ones), zero where unsupported; no fitted gain',
        wrong_pose='Reverse seven fitting transforms; keep observations and heldout transform fixed; same row masks',
        aggregation='Per unit pool squared errors over all common heldout rows in22 scenes, divide total row count; then equal weight six units',
        bootstrap=dict(seed=BOOT_SEED, replicates=1000, pairing='common whole-unit draws for two differences'),
        decision='Both ridge-minus-backprojection and ridge-minus-reversed unit-bootstrap95% upper bounds<0 => PROXY_VIEW_PREDICTION_SUPPORTED_DEV; else NOT_SUPPORTED; zero-row unit => NOT_EVALUABLE',
        limits=['Consumed Development training observations; not fresh confirmation or alarm benefit',
            'No physical surface/visibility/IRF/photon likelihood; z1 kept signed and standardized',
            'Discrete training support is not continuous visibility; conservative masks exclude some valid cells',
            'Noisy ego-motion; true current sensor-to-body transform remains inherited',
            'Finite CG recipe retained even when iteration budget stops before tolerance; report residuals'])


def canary():
    target = OUT/'canary.json'
    if target.exists():
        raise FileExistsError('Preserve existing canary; inspect instead of rerunning')
    frozen = plan()
    plan_path = OUT/'PLAN.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf8')) != frozen:
            raise ValueError('Existing PLAN differs')
    else:
        save(plan_path, frozen)
    started = time.monotonic()
    rng = np.random.default_rng(2026100230)
    transform = np.eye(4)
    transform[:3, :3] = CP.rotation(11, 'y')@CP.rotation(-10, 'x')
    transform[:3, 3] = [.03, -.01, -.18]
    a, b, inside = projection(transform)
    x, y = rng.normal(size=NVOX), rng.normal(size=1024)
    adjoint_error = abs(float((a@x)@y-x@(a.T@y)))
    assert adjoint_error <= 1e-10*max(1., abs(float((a@x)@y)))
    assert np.allclose(np.asarray(a.sum(1)).ravel()[a.getnnz(axis=1) > 0], 1., atol=1e-12)
    # Mechanical parity with the original implementation, CPU only, synthetic z.
    import torch
    torch.set_num_threads(2)
    old, _ = P.Projector(device='cpu').frame(y.reshape(8, 8, 16), transform)
    projection_error = float(np.max(np.abs((b@y).astype(np.float32)-old.numpy().ravel())))
    assert projection_error <= 1e-6
    # Independent dense interior samples check the analytic bound implementation.
    yy, xx, rr = np.indices((8, 8, 16)).reshape(3, -1)
    width = 2*P.EDGE/8
    for _ in range(20):
        u = -P.EDGE+(xx+rng.random(1024))*width
        v = -P.EDGE+(yy+rng.random(1024))*width
        r = (rr+rng.random(1024))*P.WIDTH
        points = np.stack([u, v, np.ones(1024)], -1)
        points *= (r/np.linalg.norm(points, axis=1))[:, None]
        points = points@transform[:3, :3].T+transform[:3, 3]
        assert ((points[inside] >= LOW) & (points[inside] <= HIGH)).all()
    # Rotated cone has an interior directional maximum missed by corner-only tests.
    t = np.eye(4); t[:3, :3] = CP.rotation(np.rad2deg(np.arctan(.05)), 'y').T
    lo, hi = envelope(t, np.array([0.]), np.array([.1]), np.array([0.]), np.array([.1]), np.array([1.]), np.array([1.]))
    center = np.array([.05, 0., 1.]); center /= np.linalg.norm(center)
    assert hi[0, 2] >= (t[:3, :3]@center)[2]
    # Synthetic linear constant field predicts one exactly; sparse identity ridge
    # has a closed-form solution independent of CG implementation.
    small = sparse.eye(12, format='csr')
    truth = np.linspace(-2, 2, 12)
    estimate, baseline, diagnostics = reconstruct(small, truth)
    assert np.allclose(estimate, truth/1.01, rtol=1e-10, atol=1e-12)
    assert np.array_equal(baseline, truth)
    opportunities = []
    for unit in UNITS[:3]:
        sensor, travel, noisy = CP.motion_metadata(unit, 0)
        transforms = CP.relative_transforms(sensor, travel, noisy, 15)
        ac, aw, held, mask, hm, count = matrices(transforms)
        assert np.allclose(held@np.ones(NVOX), 1., atol=1e-12)
        assert np.array_equal(mask, mask[::-1])
        opportunities.append(dict(unit=unit, config=0, mode=unit % 3, **count))
    result = dict(status='PASS', plan_sha256=sha(plan_path), source_sha256=frozen['source_sha256'],
        adjoint_abs_error=adjoint_error, original_B_float32_max_abs_error=projection_error,
        synthetic_ridge=diagnostics, geometry_opportunities=opportunities,
        real_observation_values_read=False, real_heldout_errors_computed=False,
        elapsed_s=time.monotonic()-started)
    save(target, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run():
    frozen = json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    if frozen != plan():
        raise ValueError('Frozen inputs/recipe changed')
    proof = json.loads((OUT/'canary.json').read_text(encoding='utf8'))
    if proof['status'] != 'PASS' or proof['plan_sha256'] != sha(OUT/'PLAN.json'):
        raise ValueError('Matching canary required')
    if RUN not in (Path(__file__).resolve().parents[1]/'RUNS.md').read_text(encoding='utf8'):
        raise ValueError('Parent RUNS preregistration required')
    if (OUT/'result.json').exists():
        raise FileExistsError('Completed run preserved; do not rerun')
    started = time.monotonic()
    rows, arrays, totals = [], {}, []
    for unit in UNITS:
        with np.load(SOURCE/f'unit{unit}.npz', allow_pickle=False) as cache:
            z1, scene, frame = (cache[k] for k in ('z1', 'scene', 'frame'))
        if z1.shape != (22*16, 8, 8, 16) or not np.isfinite(z1).all():
            raise ValueError('Unexpected observation cache')
        sse, n = np.zeros(3), 0
        for config in range(22):
            ids = np.flatnonzero(scene == config)
            if not np.array_equal(frame[ids], np.arange(16)):
                raise ValueError('Frame identity/order mismatch')
            y = z1[ids[FRAMES]].astype(np.float64).reshape(8, 1024)
            sensor, travel, noisy = CP.motion_metadata(unit, config)
            ac, aw, held, mask, hm, count = matrices(CP.relative_transforms(sensor, travel, noisy, 15))
            prefix = f'{unit}_{config}'
            arrays[prefix+'_mask'] = hm
            arrays[prefix+'_training_mask'] = mask
            target = y[7, hm]
            arrays[prefix+'_target'] = target
            if len(target):
                rc, base, dc = reconstruct(ac, y[:7][mask])
                rw, _, dw = reconstruct(aw, y[:7][mask])
                pred = np.stack([held@base, held@rc, held@rw])
                error = np.square(pred-target).sum(1)
                sse += error; n += len(target)
                details = dict(mse=(error/len(target)).tolist(), correct_solver=dc, reversed_solver=dw)
            else:
                pred = np.empty((3, 0))
                details = dict(mse=[None]*3, status='NO_COMMON_HELDOUT_ROWS')
            arrays[prefix+'_predictions'] = pred
            rows.append(dict(unit=unit, config=config, mode=unit % 3, **count, **details))
        totals.append(dict(unit=unit, mode=unit % 3, heldout_rows=n, sse=sse.tolist(), mse=(sse/n).tolist() if n else None))
        print('complete unit', unit, 'heldout rows', n, 'elapsed', round(time.monotonic()-started, 2), flush=True)
    if not all(t['heldout_rows'] for t in totals):
        verdict, comparison = 'NOT_EVALUABLE', None
    else:
        values = np.asarray([t['mse'] for t in totals])
        differences = values[:, 1:2]-values[:, [0, 2]]
        draw = np.random.default_rng(BOOT_SEED).integers(0, 6, size=(1000, 6))
        ci = np.percentile(differences[draw].mean(1), [2.5, 97.5], axis=0).T
        comparison = {name: dict(delta_mse=float(differences[:, i].mean()), paired_unit_ci95=ci[i].tolist())
            for i, name in enumerate(('ridge_minus_backprojection', 'ridge_minus_reversed'))}
        verdict = 'PROXY_VIEW_PREDICTION_SUPPORTED_DEV' if (ci[:, 1] < 0).all() else 'NOT_SUPPORTED'
    with (OUT/'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    save(OUT/'rows.json', rows)
    result = dict(status='COMPLETE', verdict=verdict, comparisons=comparison, units=totals, arms=ARMS,
        mode_denominators={str(m): dict(units=sum(t['mode'] == m for t in totals),
            scenes=sum(r['mode'] == m for r in rows), heldout_rows=sum(t['heldout_rows'] for t in totals if t['mode'] == m)) for m in range(3)},
        plan_sha256=sha(OUT/'PLAN.json'), canary_sha256=sha(OUT/'canary.json'),
        input_sha256={**frozen['source_sha256'], **frozen['observation_sha256']},
        output_sha256={name: sha(OUT/name) for name in ('rows.json', 'predictions.npz')},
        elapsed_s=time.monotonic()-started, limits=frozen['limits'])
    # Detect concurrent source/data modifications before publishing a terminal.
    if frozen != plan():
        raise ValueError('Frozen inputs changed during run; retain nonterminal evidence')
    save(OUT/'result.json', result)
    print(verdict, json.dumps(comparison), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('canary', 'run'), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {'canary': canary, 'run': run}[args.stage]()
