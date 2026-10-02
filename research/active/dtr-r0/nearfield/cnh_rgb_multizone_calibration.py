"""Consumed fixed-edge assay: cross-zone distribution calibration, no target truth.

GT radial depth is reduced by simulator_summary before an estimator sees it.
The estimator accepts 64 coarse marginal distributions and cached RGB depth
summaries; target identities, pixel correspondences and clearance are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

from cnh_rgb_clearance_geometry import camera_geometry, hdf, self_check as geometry_check
from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_clearance_probe import summarize
from cnh_rgb_zone_range import summarize_distribution

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT/'artifacts.local/work/cnh-rgb-clearance-probe-20261001'
ASSOC = ROOT/'artifacts.local/work/cnh-rgb-zone-association-20261001-v2'
CACHE = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
OUT = ROOT/'artifacts.local/work/cnh-rgb-multizone-calibration-20261002'
QUANTILES = [.1, .25, .5, .75, .9]
ARMS = ['depthpro', 'zone_q10', 'rgb_guided_mode', 'scale_direct', 'affine_direct',
        'affine_guided_mode', 'oracle_target']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = [Path(__file__), HERE/'cnh_rgb_clearance_geometry.py', HERE/'cnh_rgb_clearance_edge.py',
             HERE/'cnh_rgb_clearance_probe.py', HERE/'cnh_rgb_zone_range.py',
             PRIOR/'selection.json', PRIOR/'case-ledger.json', PRIOR/'evaluation-input-seal.json',
             ASSOC/'case-ledger.json', ASSOC/'result.json', CACHE/'manifest.json', CACHE/'observations.json']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def quantile_summary(radial, zones):
    """Marginal ranges only: pixel locations are discarded within each zone."""
    out = []
    for z in range(64):
        values = radial[zones == z]
        values = values[np.isfinite(values) & (values > 0) & (values < 12.8)]
        out.append(dict(zone_id=z, n=int(values.size),
                        quantiles_m=np.quantile(values, QUANTILES).tolist() if values.size >= 32 else None))
    return out


def simulator_summary(radial, zones, query_zone):
    """Simulator-owned compression; no target masking or semantic input."""
    return dict(zones=quantile_summary(radial, zones),
                query_distribution=summarize_distribution(radial[zones == query_zone]))


def calibrate(observed, predicted, query_zone):
    """Robust global radial affine fit on non-query-zone marginal quantiles.

    Matching ranks does not assert pixel correspondences. Same fixed 5 ranks
    per zone give equal zone weights. Query zone is excluded from fitting.
    Scale control uses median ratios; primary soft-L1 affine fit has 0.10 m
    transition, a in [0.1,10], b in [-5,5] m, max 200 function evaluations.
    """
    pairs = [(p['quantiles_m'], o['quantiles_m']) for p, o in zip(predicted, observed)
             if p['zone_id'] == o['zone_id'] != query_zone
             and p['quantiles_m'] is not None and o['quantiles_m'] is not None]
    if len(pairs) < 8:
        return dict(status='INSUFFICIENT_ZONES', zones=len(pairs))
    x = np.asarray([p[0] for p in pairs]).ravel()
    y = np.asarray([p[1] for p in pairs]).ravel()
    scale = float(np.clip(np.median(y/x), .1, 10.))
    if np.ptp(x) < .05:
        return dict(status='INSUFFICIENT_DEPTH_SPREAD', zones=len(pairs))
    fit = least_squares(lambda ab: ab[0]*x+ab[1]-y, [scale, 0.],
                        bounds=([.1, -5.], [10., 5.]), loss='soft_l1', f_scale=.10, max_nfev=200)
    if not fit.success:
        return dict(status='FIT_FAILED', zones=len(pairs), message=fit.message)
    a, b = map(float, fit.x)
    return dict(status='OK', zones=len(pairs), quantile_pairs=len(x), a=a, b_m=b, scale=scale,
                residual_median_abs_m=float(np.median(abs(a*x+b-y))),
                optimizer_nfev=int(fit.nfev), at_bound=bool(np.any(fit.active_mask != 0)))


def estimate(observation, prediction_summary, query_zone, prior_radial, radial_edge_factor):
    fit = calibrate(observation['zones'], prediction_summary, query_zone)
    ranges = dict(scale_direct=None, affine_direct=None, affine_guided_mode=None)
    selected = None
    if fit['status'] == 'OK':
        ranges['scale_direct'] = fit['scale']*prior_radial
        corrected = fit['a']*prior_radial+fit['b_m']
        if corrected > 0 and np.isfinite(corrected):
            ranges['affine_direct'] = corrected
            modes = observation['query_distribution']['modes']
            if modes:
                selected = min(range(len(modes)), key=lambda i: abs(modes[i]['radial_m']-corrected))
                ranges['affine_guided_mode'] = modes[selected]['radial_m']
    return dict(fit=fit, prior_radial_m=prior_radial, selected_mode_index=selected, ranges_m=ranges,
                estimates={k: radial_edge_factor*v-.30 if v is not None else None for k,v in ranges.items()})


def self_check():
    x = np.linspace(.5, 6., 64*5).reshape(64,5)
    p = [dict(zone_id=i, n=100, quantiles_m=q.tolist()) for i,q in enumerate(x)]
    o = [dict(zone_id=i, n=100, quantiles_m=(1.25*q+.15).tolist()) for i,q in enumerate(x)]
    fit = calibrate(o,p,3)
    np.testing.assert_allclose([fit['a'],fit['b_m']], [1.25,.15], atol=1e-6)
    corrupt = [dict(row) for row in o]
    corrupt[3] = dict(zone_id=3, n=100, quantiles_m=[1000.]*5)
    assert calibrate(corrupt,p,3) == fit  # Query sensor cannot influence calibration.
    corrupt[60] = dict(zone_id=60, n=100, quantiles_m=[10.]*5)
    robust = calibrate(corrupt,p,3)
    np.testing.assert_allclose([robust['a'],robust['b_m']], [1.25,.15], atol=.015)
    invalid = [dict(zone_id=i, n=0, quantiles_m=None) for i in range(64)]
    assert calibrate(invalid,p,3)['status'] == 'INSUFFICIENT_ZONES'
    sensor = dict(zones=o,query_distribution=dict(modes=[dict(radial_m=2.65),dict(radial_m=4.)]))
    got = estimate(sensor,p,3,2.,.1)
    assert got['selected_mode_index'] == 0
    np.testing.assert_allclose(got['estimates']['affine_direct'], -.035, atol=1e-6)
    sample = np.repeat(np.arange(1.,65.),40).reshape(64,40)
    zones = np.repeat(np.arange(64),40).reshape(64,40)
    summary = quantile_summary(sample,zones)
    np.testing.assert_allclose(summary[10]['quantiles_m'], 11.)
    return dict(status='PASS', geometry=geometry_check(), checks=['known_scale_bias_recovery',
        'query_zone_exclusion', 'outlier_robustness', 'missing_support', 'mode_projection', 'marginal_compression'])


def prepare(out):
    assert not (out/'PLAN.json').exists(), 'Preserve prior plans/results'
    save(out/'PLAN.json', dict(scope='EXPLORE, consumed Hypersim Development; unchanged 15 cases/8 scenes',
        question='Does across-zone marginal calibration improve metric edge range or mode association?',
        method='63 non-query zones, ranks .1/.25/.5/.75/.9; robust positive-affine radial least-squares soft-L1 delta .10m; nearest target-cell geometry mode after correction',
        ablations='median-ratio scale direct; affine direct; inherited DepthPro/q10/uncalibrated guided/oracle',
        bounds='a [0.1,10], b [-5,5]m; >=8 valid zones and >=5cm prediction spread; per-zone >=32 positive finite samples below12.8m',
        observations='ideal full native visible GT radial maps reduced to per-zone quantiles and old uniform-pixel modes; not measured ToF and not photon histograms',
        no_truth='estimator accepts only coarse summaries, RGB depth summaries, prior depth, known query zone and public edge projection; no target GT/identity/pixel correspondences',
        selection='original target cell, edge and side frozen; preexisting truth-selected clean visible-edge cohort retained',
        limits=['15 selected cases/8 scenes; only2 cases within2cm of body boundary; no grazing recall claim',
                'five exact marginal quantiles per zone are gifted rich measurements; not ST scalar output',
                'uniform pixels, perfect synchronization/calibration, no physical multipath/noise/reflectance',
                'global affine relation may fail spatially or across mixed-surface distributions',
                'same-frame unlabeled calibration is transductive consumed Development, not independent validation',
                'oracle target is evaluator-only; mode proximity is not proof of correct object association'],
        input_sha256=source_hashes()))


def run(out):
    started=time.perf_counter(); plan=read(out/'PLAN.json')
    assert source_hashes() == plan['input_sha256']
    assert not (out/'result.json').exists(), 'Preserve result'
    seal=read(PRIOR/'evaluation-input-seal.json')
    assert all(sha(ROOT/p)==h for p,h in seal['input_sha256'].items())
    save(out/'input-seal.json',dict(parent_input_sha256=seal['input_sha256'],plan_sha256=sha(out/'PLAN.json')))
    save(out/'self-check.json', self_check())
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    backend=select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-scipy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy/scipy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=out/'backend.json')
    old=read(ASSOC/'case-ledger.json')
    selection=read(PRIOR/'selection.json')
    assert len(old)==len(selection)==15 and {r['id'] for r in old}=={r['id'] for r in selection}
    manifest={r['id']:r for r in read(CACHE/'manifest.json')}
    cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    ledger=[]
    for case in old:
        fid=case['frame_id']; row=manifest[fid]; camera=cameras[fid]
        geometry=camera_geometry(ROOT,row,camera); zones=zone_map(camera)
        radial=hdf(ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/row['depth'])
        observation=simulator_summary(radial,zones,case['zone_id'])
        with np.load(CACHE/'predictions/native'/f'{fid}.npz',allow_pickle=False) as data:
            predicted_radial=data['native_depth']/geometry['optical_z_per_radial']
        pred=quantile_summary(predicted_radial,zones)
        computed=estimate(observation,pred,case['zone_id'],case['prediction']['foreground_depth_m']/case['optical_z_per_radial_at_edge'],case['radial_edge_factor'])
        # Evaluator joins scalar labels only after the estimator returns.
        estimates={a:case['estimates'][a] for a in ('depthpro','zone_q10','rgb_guided_mode','oracle_target')}
        estimates.update(computed['estimates'])
        result={k:case[k] for k in ('id','frame_id','scene','zone_id','range_bin','gt_clearance_m','prediction')}
        result.update(estimates=estimates,errors_m={a:v-case['gt_clearance_m'] if v is not None else None for a,v in estimates.items()},
                      estimator=computed,observation=observation,predicted_zone_summary=pred,radial_edge_factor=case['radial_edge_factor'])
        ledger.append(result)
        print(f'evaluated {len(ledger)}/15',flush=True)
    save(out/'case-ledger.json',ledger)
    paired={}
    for a in ('affine_direct','affine_guided_mode','scale_direct'):
        success=lambda r,k: r['errors_m'][k] is not None and abs(r['errors_m'][k])<=.02
        paired[a]=dict(both_success=sum(success(r,a) and success(r,'zone_q10') for r in ledger),
            rescued_vs_q10=[r['id'] for r in ledger if success(r,a) and not success(r,'zone_q10')],
            lost_vs_q10=[r['id'] for r in ledger if not success(r,a) and success(r,'zone_q10')],
            both_fail=sum(not success(r,a) and not success(r,'zone_q10') for r in ledger))
    result=dict(status='COMPLETE',n=len(ledger),scenes=len({r['scene'] for r in ledger}),
                arms={a:summarize(ledger,a) for a in ARMS},seconds=time.perf_counter()-started,backend=backend,
                paired_vs_q10=paired,
                fit_status={s:sum(r['estimator']['fit']['status']==s for r in ledger) for s in {r['estimator']['fit']['status'] for r in ledger}},limits=plan['limits'])
    save(out/'result.json',result)
    for path in source_hashes():
        p=ROOT/path
        if p.suffix=='.py':
            dest=out/'source'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    verify(out)
    lines=['# 跨粗格分布校正小试','',
           '固定原15例/8场景、原预测边缘。主机制用非查询粗格的5个理想径向分位数校正DepthPro，再选择原格最近几何峰；仅排除query_zone，同一目标物体跨到其他格的观测仍可参与拟合。估计器不读取目标真值标签。','',
           '|方法|≤2cm /15|绝对误差中位/P95 cm|漏判接触/7|误判接触/8|','|---|---|---|---|---|']
    for a in ARMS:
        s=result['arms'][a]; q=s['absolute_error_cm_quantiles']; signs=s['sign_counts']
        qs=f"{q['p50']:.2f}/{q['p95']:.2f}" if q else 'UNKNOWN'
        lines.append(f"|{a}|{round(s['within_cm_all']['2']*15)}/15|{qs}|{signs['contact_wrongly_clear']}|{signs['pass_by_wrongly_contact']}|")
    lines+=['','相对q10的≤2cm成败配对：']
    for a,p in paired.items():
        lines.append(f"- {a}：共同成功{p['both_success']}；补回{len(p['rescued_vs_q10'])}，丢失{len(p['lost_vs_q10'])}，共同失败{p['both_fail']}；完整ID见result.json。")
    lines+=['','这是理想分布观测下的机制检查，不是实测ToF或新增模型推理。全帧校正的假设是各格深度偏差可以共享一个尺度和偏差；本试没有从净距结果调参。','',
            '即使净距成功也不能断言距离归属正确。详细逐例拟合、传感摘要、预测摘要、误差、场景bootstrap区间在JSON中。','',
            '限制：']+['- '+x for x in plan['limits']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def verify(out):
    plan=read(out/'PLAN.json'); assert source_hashes()==plan['input_sha256']
    seal=read(out/'input-seal.json');assert seal['plan_sha256']==sha(out/'PLAN.json')
    assert all(sha(ROOT/p)==h for p,h in seal['parent_input_sha256'].items())
    ledger=read(out/'case-ledger.json'); result=read(out/'result.json')
    assert len(ledger)==15 and len({r['scene'] for r in ledger})==8
    for a in ARMS: assert summarize(ledger,a)==result['arms'][a]
    for r in ledger:
        estimate_again=estimate(r['observation'],r['predicted_zone_summary'],r['zone_id'],r['estimator']['prior_radial_m'],r['radial_edge_factor'])
        assert estimate_again==r['estimator']
        for a,v in r['estimates'].items():
            if v is not None: assert abs(v-r['gt_clearance_m']-r['errors_m'][a])<1e-12
    assert result['arms']['depthpro']['within_cm_all']['2']==4/15
    assert result['arms']['zone_q10']['within_cm_all']['2']==8/15
    save(out/'verification.json',dict(status='PASS',checks=['input_hashes','15cases8scenes','baseline4and8of15',
            'estimator_replay_from_coarse_summary_without_labels','summary_recompute','error_arithmetic']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','run','verify','self-check'],required=True)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    if args.stage=='self-check': print(json.dumps(self_check()));return
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='verify':verify(args.output)
    else:
        try:
            run(args.output);save(args.output/'terminal.json',dict(status='complete'))
        except BaseException as error:
            save(args.output/'terminal.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':main()
