"""Consumed-Development conditional rotational coarse-cell unmixing pilot.

This simulates same-optical-center rotation from first-visible radial depth.
No translated views, measured ToF, hardware response, or full radiance model.
The decoder gets coarse histograms and a mask predicted from cached Depth Pro;
reference range and instance/semantic truth remain in simulation/evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import nnls

from cnh_rgb_clearance_geometry import reference_frame

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
PRIOR = ROOT/'artifacts.local/work/cnh-rgb-zone-association-20261001-v2'
EDGE_PRIOR = ROOT/'artifacts.local/work/cnh-rgb-clearance-probe-20261001'
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-tof-dither-20261002'
YAW_DEG = np.array([-2.8125, -1.40625, 0., 1.40625, 2.8125])
PERMUTATION = np.array([2, 4, 1, 0, 3])
BIN_WIDTH = .05
N_BINS = 256
N_QUADRATURE = 32
NOISE_DRAWS = 20
PHOTON_SCALE = 4000.
AMBIENT_PER_BIN = .1


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_mask(depth, prediction):
    """Connected relative-depth region seeded 2..5 pixels inside old edge.

    The 25% multiplicative band, 4-connectivity, and seed policy are fixed.
    No absolute reference range, label, GT outline, or target mask is accepted.
    """
    depth = np.asarray(depth, dtype=float)
    y, xhalf = prediction['edge_pixel']
    x = int(np.floor(xhalf))
    side = prediction['side']
    prior = prediction['foreground_depth_m']
    valid = np.isfinite(depth) & (depth > 0)
    candidate = valid & (depth >= prior / 1.25) & (depth <= prior * 1.25)
    labels, _ = ndimage.label(candidate)
    xs = x + (np.arange(2, 6) if side > 0 else -np.arange(1, 5))
    ys = np.arange(max(0, y-3), min(depth.shape[0], y+4))
    yy, xx = np.meshgrid(ys, xs[(xs >= 0) & (xs < depth.shape[1])], indexing='ij')
    allowed = candidate[yy, xx]
    if not allowed.any():
        return np.zeros(depth.shape, bool), dict(status='NO_PREDICTED_SEED')
    distance = np.where(allowed, abs(np.log(np.maximum(depth[yy, xx], 1e-9)/prior)), np.inf)
    k = np.argmin(distance)
    sy, sx = int(yy.flat[k]), int(xx.flat[k])
    mask = labels == labels[sy, sx]
    return mask, dict(status='OK', seed_yx=[sy, sx], pixels=int(mask.sum()),
                     depth_ratio_band=1.25, connectivity=4)


def projected_samples(matrix, yaw_degrees, n=N_QUADRATURE):
    """All 64 full-zone midpoint rays, exactly rotated about graphics +Y.

    R_y yaw acts on sensor rays in the fixed source-camera frame. Rotation
    keeps radial distance and visibility valid at the same optical center.
    Full inverse M projection retains source tilt/shift and invalid support.
    """
    half = np.deg2rad(22.5)
    step = 2*half/(8*n)
    angles = -half + (np.arange(8*n)+.5)*step
    v, h = np.meshgrid(angles, angles, indexing='ij')
    a, b = np.tan(h), np.tan(v)
    norm = np.sqrt(1+a*a+b*b)
    rays = np.stack([a, -b, -np.ones_like(a)], -1)/norm[..., None]
    omega = (1+a*a)*(1+b*b)/norm**3*step**2
    theta = np.deg2rad(yaw_degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    rotated = rays @ rotation.T
    uvh = rotated @ np.linalg.inv(np.asarray(matrix)).T
    forward = uvh[..., 2] > 1e-12
    uv = np.zeros(uvh.shape[:-1]+(2,))
    np.divide(uvh[..., :2], uvh[..., 2:], out=uv, where=forward[..., None])
    xx = np.floor((uv[..., 0]+1)*1024/2).astype(int)
    yy = np.floor((1-uv[..., 1])*768/2).astype(int)
    inside = forward & (xx >= 0) & (xx < 1024) & (yy >= 0) & (yy < 768)

    def group(x):
        return x.reshape(8, n, 8, n).transpose(0, 2, 1, 3).reshape(64, n*n)

    weights = group(omega)
    weights /= weights.sum(-1, keepdims=True)
    return dict(yy=group(yy), xx=group(xx), inside=group(inside), weights=weights,
                rotation=rotation)


def simulate(radial, predicted_mask, camera, yaw=YAW_DEG, n=N_QUADRATURE):
    """Return only coarse observations and observable predicted overlaps.

    All first-visible samples in every zone contribute; the mask never
    filters sensor photons. Missing/image-exterior samples keep denominator.
    Uniform solid-angle geometry and inverse-square energy are separate.
    """
    uniform, inverse_square, overlap, coverage = [], [], [], []
    for phase in yaw:
        p = projected_samples(camera, phase, n)
        yy, xx, inside, weights = (p[k] for k in ('yy', 'xx', 'inside', 'weights'))
        r = np.full(inside.shape, np.nan)
        mask = np.zeros(inside.shape, bool)
        r[inside] = radial[yy[inside], xx[inside]]
        mask[inside] = predicted_mask[yy[inside], xx[inside]]
        eligible = inside & np.isfinite(r) & (r > 0) & (r < BIN_WIDTH*N_BINS)
        safe = np.where(eligible, r, 1.)
        index = np.floor(safe/BIN_WIDTH).astype(int)
        rows = np.broadcast_to(np.arange(64)[:, None], index.shape)
        for output, energy in ((uniform, weights), (inverse_square, weights*.5/safe**2)):
            hist = np.zeros((64, N_BINS))
            np.add.at(hist, (rows[eligible], index[eligible]), energy[eligible])
            output.append(hist)
        overlap.append((weights*mask).sum(-1))
        coverage.append((weights*inside).sum(-1))
    return dict(uniform=np.array(uniform), inverse_square=np.array(inverse_square),
                overlap=np.array(overlap), image_coverage=np.array(coverage))


def decode(hist, overlap, zones, holdout=False):
    """NNLS H[t,z,b] = w[t,z]F[b] + (1-w[t,z])B[z,b].

    Shared foreground F, independent zone backgrounds B_z stable over phase.
    Rank checks concern this explicitly assumed low-dimensional model only.
    Nonnegative spectra have free masses, including the inverse-square arm.
    """
    if not zones:
        return dict(status='NO_IMAGE_SUPPORTED_ZONE',radial_m=None)
    h, w = np.asarray(hist)[:, zones], np.asarray(overlap)[:, zones]
    phases, count = w.shape
    design = np.zeros((phases, count, count+1))
    design[..., 0] = w
    for z in range(count):
        design[:, z, z+1] = 1-w[:, z]
    design = design.reshape(-1, count+1)
    # A pure-foreground zone has an unneeded all-zero background column.
    # Its absent B cannot make F unidentifiable: retain F, drop only zero B.
    active=np.linalg.norm(design,axis=0)>1e-12
    active[0]=True
    design=design[:,active]
    columns=design.shape[1]
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design, tol=1e-10))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    result = dict(status='MODEL_NOT_IDENTIFIABLE', rank=rank, columns=columns,
                  dropped_zero_nuisance_columns=int((~active[1:]).sum()),
                  condition=condition, modulation_max=float(np.ptp(w, axis=0).max()),
                  radial_m=None, foreground_mass=None, residual_relative=None)
    if rank < columns or condition is None or condition > 1e6:
        return result
    target = h.reshape(-1, N_BINS)
    fit = np.empty((columns, N_BINS))
    for b in range(N_BINS):
        fit[:, b], _ = nnls(design, target[:, b], maxiter=1000)
    f = fit[0]
    residual = design @ fit-target
    result.update(status='OK' if f.sum() > 1e-10 else 'NO_FOREGROUND_SPECTRUM',
                  foreground_mass=float(f.sum()),
                  residual_relative=float(np.linalg.norm(residual)/max(np.linalg.norm(target), 1e-12)),
                  foreground_histogram=f.tolist())
    if f.sum() > 1e-10:
        index = int(np.argmax(f))
        result.update(radial_m=float((index+.5)*BIN_WIDTH), peak_bin=index)
    if holdout:
        folds=[]
        shaped_design=design.reshape(phases,count,columns)
        for held in range(phases):
            kept=[t for t in range(phases) if t!=held]
            train=shaped_design[kept].reshape(-1,columns)
            targets=h[kept].reshape(-1,N_BINS)
            if np.linalg.matrix_rank(train,tol=1e-10)<columns:
                folds.append(None)
                continue
            fitted=np.stack([nnls(train,targets[:,b],maxiter=1000)[0] for b in range(N_BINS)],axis=1)
            difference=shaped_design[held]@fitted-h[held]
            folds.append(float(np.linalg.norm(difference)/max(np.linalg.norm(h[held]),1e-12)))
        result['held_phase_relative_errors']=folds
    return result


def static_readouts(hist, zone):
    h = np.asarray(hist).mean(0)[zone]
    mass = h.sum()
    if mass <= 1e-12:
        return dict(q10=None, peak=None)
    return dict(q10=float((np.searchsorted(np.cumsum(h), .1*mass)+.5)*BIN_WIDTH),
                peak=float((np.argmax(h)+.5)*BIN_WIDTH))


def summarize(rows, arm):
    errors = [r['errors_m'][arm] for r in rows if r['errors_m'][arm] is not None]
    abs_cm = np.abs(errors)*100
    return dict(n=len(rows), available=len(errors), missing=len(rows)-len(errors),
                within2cm=int(sum(e <= 2 for e in abs_cm)),
                median_abs_cm=float(np.median(abs_cm)) if len(errors) else None,
                p95_abs_cm=float(np.percentile(abs_cm, 95)) if len(errors) else None)


def self_check():
    w = np.array([[.1, .3], [.3, .2], [.6, .1], [.8, .5], [.5, .8]])
    f = np.zeros(N_BINS); f[30] = 1.
    bg = np.zeros((2, N_BINS)); bg[0, 90] = 1.; bg[1, 140] = 1.
    h = w[..., None]*f+(1-w[..., None])*bg
    recovered = decode(h, w, [0, 1])
    np.testing.assert_allclose(recovered['foreground_histogram'], f, atol=1e-12)
    assert recovered['residual_relative'] < 1e-12
    assert decode(np.repeat(h[2:3], 5, axis=0), np.repeat(w[2:3], 5, axis=0), [0, 1])['status'] == 'MODEL_NOT_IDENTIFIABLE'
    assert decode(h, w[PERMUTATION], [0, 1])['residual_relative'] > .1
    pure_w=w.copy(); pure_w[:,0]=1.
    pure_h=pure_w[...,None]*f+(1-pure_w[...,None])*bg
    pure=decode(pure_h,pure_w,[0,1])
    np.testing.assert_allclose(pure['foreground_histogram'],f,atol=1e-12)
    assert pure['dropped_zero_nuisance_columns']==1
    assert decode(np.repeat(pure_h[2:3],5,axis=0),np.repeat(pure_w[2:3],5,axis=0),[0,1])['status']=='OK'
    assert decode(h,w,[])['status']=='NO_IMAGE_SUPPORTED_ZONE'
    camera = np.diag([.6, .45, -1.])
    p = projected_samples(camera, 2.)
    np.testing.assert_allclose(p['rotation'].T@p['rotation'], np.eye(3), atol=1e-12)
    radial = np.full((768, 1024), 2.025)
    observations = simulate(radial, np.ones(radial.shape, bool), camera)
    np.testing.assert_allclose(observations['uniform'].sum(-1), observations['image_coverage'], atol=1e-12)
    np.testing.assert_allclose(observations['inverse_square'].sum(-1), observations['image_coverage']*.5/2.025**2, atol=1e-12)
    depth = np.full((20, 40), 4.); depth[:, 20:]=2.
    mask, meta = prediction_mask(depth, dict(edge_pixel=[10, 19.5], side=1, foreground_depth_m=2.))
    np.testing.assert_array_equal(mask, depth == 2.)
    assert meta['status'] == 'OK'
    return dict(status='PASS', checks=['different_zone_background_exact_recovery', 'static_nonidentifiability',
                'phase_shuffle_model_mismatch', 'proper_rotation', 'all_zone_counts_and_inverse_square',
                'predicted_mask_no_labels','pure_foreground_anchor_and_zero_background_column',
                'empty_zone_explicit'])


def run(out):
    started = time.perf_counter()
    if out.exists():
        raise RuntimeError('Preserve outputs; choose a fresh --out directory')
    out.mkdir(parents=True)
    source_inputs = [Path(__file__), PRIOR/'case-ledger.json', EDGE_PRIOR/'evaluation-input-seal.json',
                     CACHE/'manifest.json', CACHE/'observations.json']
    plan = dict(scope='fixed15/8scene consumed Hypersim Development; conditional geometry pilot',
        yaw_deg=YAW_DEG.tolist(), rotation='exact graphics Y; same optical center; no translation/disocclusion',
        quadrature_per_axis=N_QUADRATURE, bin_width_m=BIN_WIDTH, bins=N_BINS,
        mask='cached DepthPro relative25percent connected region, prescribed original edge seed',
        estimator='NNLS H_tzb=w_tz F_b+(1-w_tz)B_zb; sharedF, per-zone background stable across phases',
        estimator_limits='rank is only conditional on correct mask, sharedF and stable backgrounds; not arbitrary-scene identifiability',
        zone_scope='original coarsecell and immediate3x3 neighbours with full image coverage across all phases',
        controls='same5exposure static q10/peak; phase-permuted weights; old pixel-q10; no-motion model rank check',
        photon_sensitivity=dict(reflectance=.5, weight='1/radial^2', draws=NOISE_DRAWS,
            photon_scale=PHOTON_SCALE, ambient_per_bin=AMBIENT_PER_BIN,
            role='unconvolved arbitrary Poisson geometric energy; no H3/ST/hardware calibration'),
        claim_limit='No contact recall, hardware precision, deployability, or new independent confirmation',
        inputs={str(p.relative_to(ROOT)):sha(p) for p in source_inputs})
    save(out/'PLAN.json', plan)
    save(out/'self-check.json', self_check())
    (out/'source').mkdir()
    (out/'observations').mkdir()
    shutil.copyfile(__file__, out/'source'/Path(__file__).name)
    prior = read(PRIOR/'case-ledger.json')
    manifest = {r['id']: r for r in read(CACHE/'manifest.json')}
    cameras = {r['id']: r['camera_matrix'] for r in read(CACHE/'observations.json')}
    seal = read(EDGE_PRIOR/'evaluation-input-seal.json')['input_sha256']
    assert all(sha(ROOT/p) == digest for p, digest in seal.items())
    sys.path.insert(0, str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-scipy-cpu','cpu',lambda:np.arange(128).mean(),
            lambda _:DeviceObservation('cpu','host CPU','numpy-scipy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},
        record_path=out/'backend.json')
    ledger=[]
    for i, old in enumerate(prior):
        row = manifest[old['frame_id']]
        camera = cameras[old['frame_id']]
        with np.load(CACHE/'predictions/native'/f'{row["id"]}.npz', allow_pickle=False) as blob:
            predicted_depth = blob['native_depth']
        mask, mask_info = prediction_mask(predicted_depth, old['prediction'])
        ref = reference_frame(ROOT, row, camera)
        simulated = simulate(ref['radial'], mask, camera)
        static = simulate(ref['radial'], mask, camera, yaw=np.zeros(5))
        zr, zc = divmod(old['zone_id'], 8)
        local = [r*8+c for r in range(max(0,zr-1),min(8,zr+2))
                 for c in range(max(0,zc-1),min(8,zc+2))]
        zones = [z for z in local if simulated['image_coverage'][:, z].min() >= .999999]
        decoded={}
        ranges=dict(old_q10=old['simulated_radial_ranges_m']['zone_q10'])
        noise_ranges=[]
        if zones:
            for geometry in ('uniform', 'inverse_square'):
                h=simulated[geometry]
                decoded[geometry]=decode(h, simulated['overlap'], zones, holdout=True)
                decoded[geometry+'_shuffled']=decode(h, simulated['overlap'][PERMUTATION], zones)
                decoded[geometry+'_static_identifiability']=decode(static[geometry], static['overlap'], zones)
                ranges[geometry]=decoded[geometry]['radial_m']
                ranges[geometry+'_shuffled']=decoded[geometry+'_shuffled']['radial_m']
                ranges['static_'+geometry+'_unmix']=decoded[geometry+'_static_identifiability']['radial_m']
                for name, value in static_readouts(static[geometry], old['zone_id']).items():
                    ranges['static_'+geometry+'_'+name]=value
            seed=int(hashlib.sha256(old['id'].encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed)
            for draw in range(NOISE_DRAWS):
                mean=simulated['inverse_square']*PHOTON_SCALE+AMBIENT_PER_BIN
                noisy=(rng.poisson(mean)-AMBIENT_PER_BIN)/PHOTON_SCALE
                fitted=decode(noisy, simulated['overlap'], zones)
                noise_ranges.append(fitted['radial_m'])
            ranges['inverse_square_noise_first']=noise_ranges[0]
        else:
            for arm in ('uniform','uniform_shuffled','inverse_square','inverse_square_shuffled',
                        'static_uniform_q10','static_uniform_peak','static_inverse_square_q10',
                        'static_inverse_square_peak','static_uniform_unmix','static_inverse_square_unmix',
                        'inverse_square_noise_first'):
                ranges[arm]=None
        factor=old['radial_edge_factor']
        estimates={arm:factor*r-.30 if r is not None else None for arm,r in ranges.items()}
        errors={arm:x-old['gt_clearance_m'] if x is not None else None for arm,x in estimates.items()}
        known=old['instance_id'] >= 0
        # Truth enters only this post-decoder audit and final error calculation.
        target=(ref['instance']==old['instance_id'])&(ref['semantic']==old['semantic']) if known else None
        predicted_precision=float(target[mask].mean()) if known and mask.any() else None
        noise_errors=[factor*r-.30-old['gt_clearance_m'] if r is not None else None for r in noise_ranges]
        rec=dict(id=old['id'], scene=old['scene'], frame_id=old['frame_id'], zone_id=old['zone_id'],
            gt_clearance_m=old['gt_clearance_m'], radial_edge_factor=factor,
            edge_pixel=old['prediction']['edge_pixel'], mask=mask_info,
            mask_target_pixel_precision=predicted_precision, identity_known=known,
            retained_zones=zones, excluded_image_zones=sorted(set(local)-set(zones)),
            predicted_overlaps=simulated['overlap'][:, zones].tolist(),
            decoded=decoded, radial_ranges_m=ranges, estimates=estimates, errors_m=errors,
            noise_errors_m=noise_errors,
            noise_within2cm=sum(e is not None and abs(e)<=.02 for e in noise_errors))
        ledger.append(rec)
        np.savez_compressed(out/'observations'/f'case-{i:02d}.npz', **simulated,
                            static_uniform=static['uniform'], static_inverse_square=static['inverse_square'])
        save(out/'case-ledger.json',ledger)
        print(f'case {i+1}/{len(prior)}: uniform={errors["uniform"]}, inverse_square={errors["inverse_square"]}',flush=True)
    arms=list(ledger[0]['errors_m'])
    result=dict(status='COMPLETE',n=len(ledger),scenes=len({r['scene'] for r in ledger}),
        arms={arm:summarize(ledger,arm) for arm in arms},
        noisy_within2cm=dict(success=sum(r['noise_within2cm'] for r in ledger),
            n=len(ledger)*NOISE_DRAWS, role='20 repeated synthetic draws on same15cases, not300independentcases'),
        seconds=time.perf_counter()-started,
        limits=['same15selectedcleanedges; known coarsecell/floor prerequisites',
            'predicted foreground masks may merge surfaces; shared foreground and phase-stable background assumptions',
            'pure samecenter yaw; no real head translation, motion blur, sequential scan, pose noise or radiometry',
            '5cm synthetic bins not L8CH/CNH measurement resolution',
            'Poisson scale arbitrary sensitivity, not a hardware success rate',
            '2of15near body line; no contact recall or alarm tradeoff'])
    save(out/'result.json',result)
    lines=['# 多相位ToF粗格条件解混小试','',
        '固定15例/8场景，既有Depth Pro预测边缘与对象盲相对深度连通mask；首可见深度只用于模拟和评估。纯旋转同光心，不是新增真实测量。','',
        '|条件|净距误差≤2cm|缺失|P50/P95绝对误差(cm)|','|---|---:|---:|---:|']
    for arm,metric in result['arms'].items():
        lines.append(f'|{arm}|{metric["within2cm"]}/15|{metric["missing"]}|{metric["median_abs_cm"]}/{metric["p95_abs_cm"]}|')
    lines += ['',f'逆平方+Poisson敏感性：{result["noisy_within2cm"]["success"]}/{15*NOISE_DRAWS}次case-draw满足2cm；仍仅15例，不能写成300例硬件成功率。','',
        '解码模型：H(t,z,b)=w(t,z)F(b)+(1-w(t,z))B(z,b)。前景谱跨格/相位共享，各格背景谱可不同但跨相位稳定。w仅来自预测mask几何投影；F和B均非负但不强制单位质量。原始range/身份不进入拟合。秩满只证明这个条件模型可识别，不证明真实场景符合它。','',
        '无移动重复本身没有新增解混信息；若预测mask提供纯前景anchor则静态已可条件识别F，故同时报告静态同mask解混与相同5次曝光q10/peak。phase shuffle将同一hist与固定错误相位mask配对；它是机制诊断，不是新方法。','',
        '限制：'+ '；'.join(result['limits'])+'。','',
        '逐例相位覆盖、矩阵条件数、拟合残差、预测mask目标组成及误差见case-ledger.json。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    verify(out)


def verify(out):
    ledger,result=read(out/'case-ledger.json'),read(out/'result.json')
    assert len(ledger)==15 and len({r['id'] for r in ledger})==15
    for arm in result['arms']:
        assert summarize(ledger,arm)==result['arms'][arm]
        for row in ledger:
            radial=row['radial_ranges_m'][arm]
            if radial is not None:
                assert abs(row['estimates'][arm]-(row['radial_edge_factor']*radial-.30))<1e-12
                assert abs(row['errors_m'][arm]-(row['estimates'][arm]-row['gt_clearance_m']))<1e-12
    save(out/'verification.json',dict(status='PASS',cases=15,
        checks=['exact15case_denominator','all_radial_clearance_arithmetic','summary_recomputed',
                'zero_motion_identifiability_fixture','paired_source_input_seal'],source_sha256=sha(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    else:
        run(args.out)
