"""Fixed material-weight sensitivity of the consumed 52-edge ordinal rule.

This does not select new edges or supply an observation-driven target cell.
Visible-rendered albedo and ideal normals are not calibrated near-IR returns.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from cnh_rgb_association_expanded import DEFAULT_OUT, cpu_backend, paired_difference
from cnh_rgb_clearance_probe import ROOT, CACHE, read, save, sha, summarize
from cnh_rgb_clearance_geometry import hdf
from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_ordinal_stress import histogram_quantile

OUT = ROOT/'artifacts.local/work/cnh-rgb-physical-weights-20261002'
DATA = ROOT/'artifacts.local/datasets/hypersim-ba-nfo'
DOWNLOAD = ROOT/'artifacts.local/work/cnh-rgb-physical-download-20261002'
MANIFEST = ROOT/'artifacts.local/work/cnh-rgb-physical-download-plan-20261002/manifest.json'
DOC = ROOT/'artifacts.local/work/cnh-rgb-candidate-supplement-20261001/hypersim/official-sources/README.md'
WEIGHTS = ('uniform', 'inv_r2_pixel', 'albedo_cos_inv_r2')


def rays_from_matrix(matrix, shape):
    h, w = shape
    yy, xx = np.indices(shape)
    uv = np.stack(((xx+.5)*2/w-1, 1-(yy+.5)*2/h, np.ones(shape)), -1)
    rays = uv @ np.asarray(matrix, dtype=np.float64).T
    assert np.isfinite(rays).all() and np.all(rays[..., 2] < 0)
    return rays/np.linalg.norm(rays, axis=-1, keepdims=True)


def physical_weights(radial, rays, normals, reflectance, valid):
    normals = np.asarray(normals, dtype=np.float64)
    reflectance = np.asarray(reflectance, dtype=np.float64)
    assert normals.shape == reflectance.shape == radial.shape+(3,)
    norm = np.linalg.norm(normals, axis=-1)
    normal_ok = np.isfinite(normals).all(-1) & np.isfinite(norm) & (norm > 0)
    albedo_ok = np.isfinite(reflectance).all(-1)
    usable = valid & normal_ok & albedo_ok
    cosine = np.zeros(radial.shape, dtype=np.float64)
    albedo = np.zeros(radial.shape, dtype=np.float64)
    cosine[usable] = np.clip(np.sum(-rays[usable]*normals[usable]/norm[usable, None], -1), 0, 1)
    albedo[usable] = np.maximum(reflectance[usable], 0).mean(-1)
    weights = np.zeros(radial.shape, dtype=np.float64)
    weights[usable] = albedo[usable]*cosine[usable]/radial[usable]**2
    assert np.isfinite(weights).all() and np.all(weights >= 0)
    return weights, dict(depth_valid=int(valid.sum()), invalid_normal=int((valid & ~normal_ok).sum()),
        invalid_albedo=int((valid & ~albedo_ok).sum()), positive_mass_pixels=int((weights > 0).sum()),
        usable_nonpositive_cosine=int((usable & (cosine == 0)).sum()),
        usable_zero_albedo=int((usable & (albedo == 0)).sum()),
        usable_negative_albedo_channels=int((reflectance[usable] < 0).sum()),
        usable_above_one_albedo_channels=int((reflectance[usable] > 1).sum()))


def histogram(radial, zones, valid, weights):
    # Native radial admission and 5cm floor-bin semantics match the frozen rule.
    codes = zones[valid]*256+np.floor(radial[valid]/.05).astype(int)
    return np.bincount(codes, weights=weights[valid], minlength=64*256).reshape(64, 256)


def fixtures():
    rays = rays_from_matrix(np.diag([1., 1., -1.]), (1, 1))
    np.testing.assert_array_equal(rays, [[[0., 0., -1.]]])
    radial = np.array([[2., 2., 2., 2., 2.]])
    ray = np.broadcast_to(rays, (1, 5, 3))
    normals = np.array([[[0., 0., 2.], [0., 0., -2.], [1., 0., 0.], [0., 0., 0.], [0., 0., 1.]]])
    refl = np.ones((1, 5, 3))*.6
    refl[0, 4] = np.nan
    weight, stats = physical_weights(radial, ray, normals, refl, np.ones((1, 5), bool))
    np.testing.assert_allclose(weight, [[.15, 0, 0, 0, 0]], atol=1e-15)
    assert stats['invalid_normal'] == stats['invalid_albedo'] == 1
    assert histogram_quantile(np.zeros(256), .1, .05) is None
    assert histogram_quantile(np.array([0., 1., 1.]), .5, .05) == .1
    return dict(status='PASS', checks=['negative-z camera ray', 'front/back/tangent normals',
        'normal magnitude invariant', 'nonfinite albedo/zero normal yield zero mass', 'zero histogram abstains'])


def prepare(out):
    if (out/'PLAN.json').exists():
        raise FileExistsError('Preserve frozen plan')
    prior = read(DEFAULT_OUT/'ordinal-ledger.json')
    assert len(prior) == 52 and len({r['id'] for r in prior}) == 52
    frames = sorted({r['frame_id'] for r in prior})
    assert len(frames) == 19
    inventory = {r['id']: r for r in read(CACHE/'manifest.json')}
    receipts = {r['path']: r for r in (read(p) for p in (DOWNLOAD/'receipts').glob('*.json'))}
    members = [r for r in read(MANIFEST)['files'] if r['package'] == 'A_static_19frames']
    assert len(members) == 38 and {r['frame_id'] for r in members} == set(frames)
    source_paths = [Path(__file__), DEFAULT_OUT/'ordinal-ledger.json', DEFAULT_OUT/'selection.json',
        CACHE/'manifest.json', CACHE/'observations.json', MANIFEST, DOC]
    source_paths += [Path(__file__).with_name(name) for name in (
        'cnh_rgb_association_expanded.py', 'cnh_rgb_clearance_probe.py',
        'cnh_rgb_clearance_geometry.py', 'cnh_rgb_clearance_edge.py', 'cnh_rgb_ordinal_stress.py')]
    source_paths += [DATA/inventory[f]['depth'] for f in frames]
    materials = []
    for member in members:
        receipt = receipts[member['path']]
        assert receipt['crc32_hex'] == member['crc32_hex']
        assert receipt['bytes'] == member['uncompressed_bytes']
        assert receipt['shape'] == [768, 1024, 3]
        materials.append({k: member[k] for k in ('path', 'frame_id', 'channel')} | {'sha256': receipt['sha256']})
    plan = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='User-requested fixed physical sensitivity, consumed synthetic Development; not a new mechanism',
        role='Same 52 edges / 19 frames / 8 scenes; gifted target cell, floor and clean GT edge selection retained',
        material_access_at_freeze='Receipt metadata only; this script has not opened material HDF arrays',
        weights={
            'uniform': 'One per admitted native depth pixel; no material-validity filtering',
            'inv_r2_pixel': '1/r^2 at exact native per-pixel radial range, not prior bin-centre approximation',
            'albedo_cos_inv_r2': 'mean(max(linear diffuse_reflectance RGB,0))*max(0,n_hat dot -ray_hat)/r^2'},
        invalid_handling='Depth finite 0<r<12.8 and zone>=0. Physical: any nonfinite RGB/normal or zero normal -> zero mass. Negative RGB channels clip to0; no upper clipping; normalize normals. Never absolute cosine or flip normals. Empty histogram -> None, remains in all-case denominator.',
        normalization='Raw nonnegative mass within each cell; CDF divides by total implicitly; no per-surface or reflectance compensation',
        camera_convention='Official camera x right/y up/z backward; cached M_cam_from_uv gives camera-to-scene ray, so front-facing normal dot -ray is positive; use exact M including tilt shift',
        frozen_rule='Reuse original52 IDs, prediction edge, predicted_rank and radial_factor exactly; 5cm bins, piecewise-uniform inverseCDF; clearance=factor*radius-.30; q10 from identical histogram',
        metrics='<=2cm error is diagnostic only; paired scene bootstrap, per-edge ledger, signed body-line crossings and missing counts. Not three-level alarm recall or clear false-alert burden.',
        limitations=['Visible albedo is not nearIR reflectance; ideal normal ignores bump mapping',
            'No ambient, detector noise, multi-return timing or calibrated active emitter model',
            'No observation-driven candidate, fresh confirmation, or hardware/safety result',
            'Positive sensitivity result does not bypass target-cell selection and new-scene confirmation'],
        frames=frames, material_members=materials,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_paths}, fixtures=fixtures())
    save(out/'PLAN.json', plan)
    print('PREPARED; no material channels opened', flush=True)


def paired_hits(rows, candidate, baseline):
    def hit(r, arm):
        v = r['errors_m'][arm]
        return v is not None and abs(v) <= .02
    return dict(rescued=sum(hit(r, candidate) and not hit(r, baseline) for r in rows),
                lost=sum(not hit(r, candidate) and hit(r, baseline) for r in rows),
                **paired_difference(rows, candidate, baseline))


def run(out):
    if (out/'result.json').exists() or (out/'case-ledger.json').exists():
        raise FileExistsError('Preserve completed result')
    started = time.perf_counter()
    plan = read(out/'PLAN.json')
    # Verify immutable inputs, including material bytes, before opening channels.
    for path, expected in plan['input_sha256'].items():
        assert sha(ROOT/path) == expected, path
    for member in plan['material_members']:
        assert sha(DATA/member['path']) == member['sha256'], member['path']
    cpu_backend(out)
    fixtures()
    inventory = {r['id']: r for r in read(CACHE/'manifest.json')}
    cameras = {r['id']: r['camera_matrix'] for r in read(CACHE/'observations.json')}
    previous = read(DEFAULT_OUT/'ordinal-ledger.json')
    ledger, frame_stats = [], []
    for frame in plan['frames']:
        # Original reference_frame casts native float16 before 5cm binning.
        radial = hdf(DATA/inventory[frame]['depth']).astype(np.float32)
        camera = cameras[frame]
        zones = zone_map(camera)
        valid = np.isfinite(radial) & (radial > 0) & (radial < 12.8) & (zones >= 0)
        channels = {m['channel']: DATA/m['path'] for m in plan['material_members'] if m['frame_id'] == frame}
        physical, stats = physical_weights(radial.astype(np.float64), rays_from_matrix(camera, radial.shape),
            hdf(channels['normal_cam']), hdf(channels['diffuse_reflectance']), valid)
        inverse = np.zeros(radial.shape, dtype=np.float64)
        inverse[valid] = 1/radial[valid].astype(np.float64)**2
        histograms = {name: histogram(radial, zones, valid, w) for name, w in
            zip(WEIGHTS, (np.ones(radial.shape), inverse, physical))}
        frame_stats.append(dict(frame_id=frame, **stats))
        for old in (r for r in previous if r['frame_id'] == frame):
            rank, factor = old['details'].get('predicted_rank'), old['details'].get('radial_factor')
            estimates, masses = {}, {}
            for name in WEIGHTS:
                counts = histograms[name][old['zone_id']]
                masses[name] = dict(total=float(counts.sum()), occupied_bins=int(np.count_nonzero(counts)))
                for method, q in (('ordinal', rank), ('q10', .1)):
                    radius = histogram_quantile(counts, q, .05) if q is not None and factor is not None else None
                    estimates[f'{method}_{name}'] = float(factor*radius-.30) if radius is not None else None
            if old['estimates']['ordinal_transport'] is not None:
                np.testing.assert_allclose(estimates['ordinal_uniform'], old['estimates']['ordinal_transport'], rtol=0, atol=1e-12)
            row = {k: v for k, v in old.items() if k not in ('errors_m', 'estimates')}
            row.update(estimates=estimates, histogram_mass=masses,
                errors_m={a: v-old['gt_clearance_m'] if v is not None else None for a, v in estimates.items()})
            ledger.append(row)
        print(frame, len(ledger), '/52', flush=True)
    ledger.sort(key=lambda r: r['id'])
    assert len(ledger) == 52 and {r['id'] for r in ledger} == {r['id'] for r in previous}
    arms = list(ledger[0]['estimates'])
    subsets = {'all52': ledger, 'original15': [r for r in ledger if r['original_record']],
        'additional37': [r for r in ledger if not r['original_record']],
        'body_line_2cm': [r for r in ledger if abs(r['gt_clearance_m']) <= .02]}
    summaries = {name: {arm: summarize(rows, arm) for arm in arms} for name, rows in subsets.items()}
    result = dict(status='COMPLETE_SENSITIVITY_ONLY', n=52, frames=19, scenes=8,
        summaries=summaries, paired={name: paired_hits(ledger, 'ordinal_'+name, 'q10_'+name) for name in WEIGHTS},
        frame_stats=frame_stats, seconds=time.perf_counter()-started, plan_sha256=sha(out/'PLAN.json'),
        note='Per-pixel inv_r2 differs from historical bin-centre inv_r2; all signs are local nominal edge diagnostics, not alarm errors',
        limits=plan['limitations'])
    save(out/'case-ledger.json', ledger)
    save(out/'result.json', result)
    lines = ['# 冻结排序传递：物理权重敏感性复核', '',
        '同一52条已消费Development边缘，19帧、8场景；仍赠送目标格和干净边缘。没有新候选、训练或新场景确认。', '',
        '物理代理固定为线性RGB反射率非负均值 × 正面余弦 / 逐像素径向距离²；可见光albedo不是近红外标定。', '',
        '|权重|排序≤2cm|同直方图q10≤2cm|补回/丢失|差值pp（场景95%区间）|', '|---|---|---|---|---|']
    for name in WEIGHTS:
        a, b = (summaries['all52'][m+'_'+name] for m in ('ordinal', 'q10'))
        paired = result['paired'][name]
        lines.append(f"|{name}|{round(a['within_cm_all']['2']*52)}/52|{round(b['within_cm_all']['2']*52)}/52|{paired['rescued']}/{paired['lost']}|{paired['delta_pp']:.2f} [{paired['ci95_pp'][0]:.2f},{paired['ci95_pp'][1]:.2f}]|")
    lines += ['', '≤2cm仅为诊断；未计算三级真值擦碰召回或清晰误报。全部缺失和零质量均保留。', '',
        '|方法|有效/总数|接触→畅通/接触数|身体外→接触/身体外数|', '|---|---|---|---|']
    for arm in arms:
        s = summaries['all52'][arm]; c = s['sign_counts']
        lines.append(f"|{arm}|{s['valid']}/{s['n']}|{c['contact_wrongly_clear']}/{c['contact']}|{c['pass_by_wrongly_contact']}/{c['pass_by']}|")
    lines += ['', '身体外→接触不是当前三级真值下的误报率，0–10cm擦身允许报警。全部逐例真值、估计、固定rank/edge及缺失见case-ledger.json。', '',
        '本轮1/r²按逐像素距离计算，历史压力试验按5cm箱中心计算，两者明确区分。均匀排序臂逐例复现原输出。', '',
        '任何正结果也不绕过观测驱动选格、未消费新场景和任务报警验证；不宣称物理上限、硬件效果或安全性。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print({w: result['paired'][w] for w in WEIGHTS}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run', 'fixtures'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == 'fixtures':
        print(fixtures())
    else:
        {'prepare': prepare, 'run': run}[args.stage](args.out)
