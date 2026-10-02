"""Object-blind coarse-cell distribution calibration of cached RGB depth.

Public functions accept unordered per-cell radial summaries only. Official
visible depth supplies a *perfect geometric sensor proxy*, not measured ToF.
The executable joins the old selected-case truth only after public prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT / 'artifacts.local/work/cnh-rgb-clearance-probe-20261001'
ASSOC = ROOT / 'artifacts.local/work/cnh-rgb-zone-association-20261001-v2'
CACHE = ROOT / 'artifacts.local/work/ba-nfo-depthpro-20260919'
OUT = ROOT / 'artifacts.local/work/cnh-rgb-distribution-alignment-20261002'
QUANTILES = np.array([.10, .25, .50, .75, .90])
ARMS = ('depthpro', 'zone_q10', 'rgb_guided_mode', 'affine',
        'scale_only', 'affine_leave_cell_out', 'inverse_affine', 'affine_leave_3x3_out')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_quantiles(radial, zones):
    """64 by 5 summaries; each field uses its own validity, never paired pixels.

    Uniform pixels, first-visible geometry in (0,12.8m), at least 16 per cell.
    A permutation within any cell leaves these observations unchanged.
    """
    radial, zones = np.asarray(radial), np.asarray(zones)
    if radial.shape != zones.shape:
        raise ValueError('radial and zones shapes differ')
    result = np.full((64, len(QUANTILES)), np.nan)
    for cell in range(64):
        values = radial[zones == cell]
        values = values[np.isfinite(values) & (values > 0) & (values < 12.8)]
        if len(values) >= 16:
            result[cell] = np.quantile(values, QUANTILES)
    return result


def robust_fit(x, y, scale_only=False):
    """Huber IRLS on quantile correspondences, without point correspondences.

    25 fixed iterations; MAD scale floor .02 in the fitted units; Huber 1.345.
    A nonpositive scale is an explicit unavailable result, never a GT fallback.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    design = x[:, None] if scale_only else np.column_stack([x, np.ones(len(x))])
    if len(x) < 10 or np.ptp(x) < 1e-6:
        return None
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(25):
        residual = y - design @ coef
        scale = max(.02, float(1.4826 * np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1., 1.345 * scale / np.maximum(np.abs(residual), 1e-12))
        weighted = design * np.sqrt(weights[:, None])
        coef = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)[0]
    slope, intercept = float(coef[0]), 0. if scale_only else float(coef[1])
    if not np.isfinite(coef).all() or slope <= 0:
        return None
    return dict(slope=slope, intercept=intercept, pairs=len(x),
                quantile_residual_median_abs=float(np.median(np.abs(y-design@coef))),
                quantile_residual_p95_abs=float(np.quantile(np.abs(y-design@coef), .95)))


def predict_ranges(predicted_quantiles, sensor_quantiles, prior_radial_m, zone_id):
    """Public selector. No case records, pixel GT, semantic or instance inputs.

    All-cell affine is the pre-result main arm; the remaining arms are fixed
    mechanism ablations. Missing/unphysical transformed distances stay None.
    """
    predicted_quantiles = np.asarray(predicted_quantiles, float)
    sensor_quantiles = np.asarray(sensor_quantiles, float)
    if predicted_quantiles.shape != (64, 5) or sensor_quantiles.shape != (64, 5):
        raise ValueError('expected 64 by 5 quantiles')
    if not 0 <= zone_id < 64 or not np.isfinite(prior_radial_m) or prior_radial_m <= 0:
        raise ValueError('invalid public target input')
    result = {}
    for arm in ('affine', 'scale_only', 'affine_leave_cell_out', 'inverse_affine', 'affine_leave_3x3_out'):
        cells = (np.isfinite(predicted_quantiles).all(axis=1)
                 & np.isfinite(sensor_quantiles).all(axis=1)
                 & (predicted_quantiles > 0).all(axis=1)
                 & (sensor_quantiles > 0).all(axis=1))
        if arm == 'affine_leave_cell_out':
            cells[zone_id] = False
        if arm == 'affine_leave_3x3_out':
            indices = np.arange(64)
            cells[(np.abs(indices//8-zone_id//8)<=1) & (np.abs(indices%8-zone_id%8)<=1)] = False
        x, y = predicted_quantiles[cells].ravel(), sensor_quantiles[cells].ravel()
        inverse = arm == 'inverse_affine'
        if inverse:
            x, y = 1/x, 1/y
        fit = robust_fit(x, y, scale_only=arm == 'scale_only')
        value = None
        if fit is not None:
            value = fit['slope'] * (1/prior_radial_m if inverse else prior_radial_m) + fit['intercept']
            if value <= 0 or not np.isfinite(value):
                value = None
            elif inverse:
                value = 1/value
        result[arm] = dict(radial_m=value, fit=fit, cells=int(cells.sum()),
                           fitted_units='inverse_m' if inverse else 'm')
    return result


def self_check():
    x = np.linspace(.5, 5., 64*5).reshape(64, 5)
    a, b, prior, target = 1.2, .15, 1.8, 12
    result = predict_ranges(x, a*x+b, prior, target)
    for arm in ('affine', 'affine_leave_cell_out', 'affine_leave_3x3_out'):
        np.testing.assert_allclose(result[arm]['radial_m'], a*prior+b, atol=1e-10)
    inverse = predict_ranges(x, 1/(.8/x+.1), prior, target)
    np.testing.assert_allclose(inverse['inverse_affine']['radial_m'], 1/(.8/prior+.1), atol=1e-10)
    scaled = predict_ranges(x, 1.3*x, prior, target)
    np.testing.assert_allclose(scaled['scale_only']['radial_m'], 1.3*prior, atol=1e-10)
    contaminated = a*x+b
    contaminated[target] += 5
    poisoned = predict_ranges(x, contaminated, prior, target)
    assert poisoned['affine_leave_cell_out'] == result['affine_leave_cell_out']
    assert poisoned['affine_leave_3x3_out'] == result['affine_leave_3x3_out']
    zones = np.repeat(np.arange(64), 32).reshape(64,32)
    values = np.linspace(.2, 10., zones.size).reshape(zones.shape)
    np.testing.assert_allclose(cell_quantiles(values,zones), cell_quantiles(values[:,::-1],zones))
    blank = predict_ranges(np.full((64,5),np.nan),x,prior,target)
    assert all(v['radial_m'] is None for v in blank.values())
    return dict(status='PASS', checks=['affine synthetic transform recovered',
        'scale-only transform recovered', 'inverse transform recovered',
        'withheld cell poisoning invariant', 'within-cell permutation invariant',
        'empty observations explicitly unavailable',
        'public selector has only numeric summaries, prior and zone input'])


def sources():
    paths = [Path(__file__), HERE/'cnh_rgb_clearance_geometry.py',
             HERE/'cnh_rgb_clearance_edge.py', HERE/'cnh_rgb_clearance_probe.py',
             PRIOR/'selection.json', PRIOR/'case-ledger.json',
             PRIOR/'evaluation-input-seal.json', ASSOC/'case-ledger.json',
             CACHE/'manifest.json', CACHE/'observations.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def prepare(out):
    assert not (out/'PLAN.json').exists(), 'Preserve previous run'
    save(out/'PLAN.json', dict(lane='EXPLORE', scope='consumed synthetic Hypersim Development; fixed15 cases/8 scenes/old edges',
        question='Can cross-cell distribution calibration repair DepthPro metric-scale bias without selecting one local range mode?',
        main_arm='affine', ablations=['scale_only','affine_leave_cell_out','inverse_affine','affine_leave_3x3_out'],
        selection='arms and constants chosen by mechanism before this run; no post-result parameter sweep',
        sensor='perfect first-visible radial geometry, independently aggregated uniform pixels in64 equal-angle cells, not measured ToF or photons',
        summaries='q10,q25,q50,q75,q90; independently filter(0,12.8m); >=16 pixels per cell; no shared pixel-validity mask; no semantic mask',
        fit='paired same-cell quantile ranks; equal cell weight; affine Huber IRLS25,1.345*MAD floor.02 in fitted units; positive scale required',
        controls='old DepthPro,zone_q10,rgb_guided_mode; preserve all15 denominator and original predicted edge',
        evaluator='GT clean-edge clearance and labels accessed only after all public observations and predictions written',
        limits=['known target cell granted by prior GT selection', 'tiny consumed8-scene Development, no independent confirmation',
                'perfect geometry distributions are stronger than measured coarse sensor',
                'radial affine assumes spatially coherent RGB depth bias; nonlinear/local errors may remain',
                'only2/15 near body line; cannot estimate grazing-contact recall',
                'no training, download, new inference, hardware or real-user claims'],
        input_sha256=sources()))


def run(out):
    from cnh_rgb_clearance_geometry import camera_geometry, hdf
    from cnh_rgb_clearance_edge import zone_map
    from cnh_rgb_clearance_probe import summarize
    started = time.perf_counter()
    plan=read(out/'PLAN.json')
    assert plan['input_sha256']==sources()
    assert not (out/'result.json').exists(), 'Preserve existing result'
    parent=read(PRIOR/'evaluation-input-seal.json')
    assert all(sha(ROOT/p)==h for p,h in parent['input_sha256'].items())
    save(out/'input-seal.json',dict(plan_sha256=sha(out/'PLAN.json'),parent_input_sha256=parent['input_sha256']))
    save(out/'self-check.json',self_check())
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    backend=select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=out/'backend.json')
    old=read(PRIOR/'case-ledger.json')
    # Strip evaluator-only case fields before constructing public observations.
    public_cases=[dict(id=r['id'],frame_id=r['frame_id'],zone_id=r['zone_id'],prediction=r['prediction']) for r in old]
    del old
    manifest={r['id']:r for r in read(CACHE/'manifest.json')}
    cameras={r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    predictions=[]
    for case in public_cases:
        row=manifest[case['frame_id']]; camera=cameras[case['frame_id']]
        geometry=camera_geometry(ROOT,row,camera)
        zones=zone_map(camera)
        # Sensor adapter reads only depth geometry and returns coarse aggregates.
        radial=hdf(ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/row['depth'])
        sensor=cell_quantiles(radial,zones)
        del radial
        with np.load(CACHE/'predictions/native'/f'{row["id"]}.npz',allow_pickle=False) as cache:
            depth=cache['native_depth']
        observed=cell_quantiles(depth/geometry['optical_z_per_radial'],zones)
        p=case['prediction']; y,xhalf=p['edge_pixel']; x=int(np.floor(xhalf))
        optical=float(np.mean(geometry['optical_z_per_radial'][y,x:x+2]))
        prior=p['foreground_depth_m']/optical
        arms=predict_ranges(observed,sensor,prior,case['zone_id'])
        factor=p['side']*p['lateral_factor']*optical
        predictions.append(dict(**case,prior_radial_m=prior,radial_edge_factor=factor,
            arms=arms,estimates={a:(v['radial_m']*factor-.30 if v['radial_m'] is not None else None) for a,v in arms.items()}))
        save(out/'observations'/f'{case["frame_id"]}.json',dict(
            frame_id=case['frame_id'],quantiles=QUANTILES.tolist(),
            sensor=[[float(v) if np.isfinite(v) else None for v in q] for q in sensor],
            predicted=[[float(v) if np.isfinite(v) else None for v in q] for q in observed]))
        print('predicted',len(predictions),'/',len(public_cases),flush=True)
    save(out/'public-predictions.json',predictions)
    # Evaluation-only join begins after all predictions have been persisted.
    old={r['id']:r for r in read(PRIOR/'case-ledger.json')}
    assoc={r['id']:r for r in read(ASSOC/'case-ledger.json')}
    ledger=[]
    for prediction in predictions:
        ref=old[prediction['id']]
        estimate=prediction['estimates']
        estimate.update({a:assoc[ref['id']]['estimates'][a] for a in ARMS[:3]})
        ledger.append(dict(id=ref['id'],frame_id=ref['frame_id'],scene=ref['scene'],
            zone_id=ref['zone_id'],gt_clearance_m=ref['gt_clearance_m'],range_bin=ref['range_bin'],
            estimates=estimate,errors_m={a:(v-ref['gt_clearance_m'] if v is not None else None) for a,v in estimate.items()},
            fits=prediction['arms']))
    assert len(ledger)==15 and len({r['scene'] for r in ledger})==8
    result=dict(status='COMPLETE',main_arm='affine',n=len(ledger),scenes=8,backend=backend,
        seconds=time.perf_counter()-started,arms={a:summarize(ledger,a) for a in ARMS},
        scene_pairs={scene:{a:dict(mean_abs_cm=float(np.mean([abs(r['errors_m'][a])*100 for r in ledger if r['scene']==scene and r['errors_m'][a] is not None])),
            within2=sum(r['errors_m'][a] is not None and abs(r['errors_m'][a])<=.02 for r in ledger if r['scene']==scene),
            n=sum(r['scene']==scene for r in ledger)) for a in ARMS} for scene in sorted({r['scene'] for r in ledger})})
    save(out/'case-ledger.json',ledger);save(out/'result.json',result)
    report(out,result,ledger)
    (out/'source').mkdir(exist_ok=True)
    for p in (Path(__file__),HERE/'cnh_rgb_clearance_geometry.py',HERE/'cnh_rgb_clearance_edge.py',HERE/'cnh_rgb_clearance_probe.py'):
        (out/'source'/p.name).write_bytes(p.read_bytes())
    verify(out)


def verify(out):
    from cnh_rgb_clearance_probe import summarize
    plan=read(out/'PLAN.json');seal=read(out/'input-seal.json')
    assert sources()==plan['input_sha256'] and sha(out/'PLAN.json')==seal['plan_sha256']
    ledger=read(out/'case-ledger.json');result=read(out/'result.json')
    for a in ARMS:
        assert summarize(ledger,a)==result['arms'][a]
    public={r['id']:r for r in read(out/'public-predictions.json')}
    for row in ledger:
        for arm in ARMS[3:]:
            assert row['estimates'][arm]==public[row['id']]['estimates'][arm]
        for arm,estimate in row['estimates'].items():
            if estimate is not None:
                assert abs(estimate-row['gt_clearance_m']-row['errors_m'][arm])<1e-12
    save(out/'verification.json',dict(status='PASS',checks=[
        'source and parent input seal', 'same15case denominator','all metrics recomputed',
        'persisted public prediction equals evaluated prediction', 'ledger arithmetic'],self_check=self_check()))


def report(out,result,ledger):
    lines=['# 跨粗格距离分布校准RGB深度小试','',
        '固定15例/8场景/原预测边缘。主方法在结果前指定为affine；其他新增臂是固定消融，没有按结果切换主方法。粗格输入为对象盲的完美首可见几何分布，不是实测ToF。','',
        '|方法|≤2cm（全部15例）|P50/P95绝对误差(cm)|缺失|','|---|---|---|---|']
    for arm,s in result['arms'].items():
        q=s['absolute_error_cm_quantiles']
        lines.append(f'|{arm}|{round(s["within_cm_all"]["2"]*s["n"])}/{s["n"]}|{q["p50"]:.3f}/{q["p95"]:.3f}|{s["missing"]}|')
    lines+=['','逐场景配对：每格为“≤2cm例数/该场景例数；MAE cm”，保留失败场景。','',
        '|场景|DepthPro|q10|affine主臂|leave-cell-out|','|---|---|---|---|---|']
    for scene,arms in result['scene_pairs'].items():
        entries=[f'{arms[a]["within2"]}/{arms[a]["n"]}; {arms[a]["mean_abs_cm"]:.2f}' for a in ('depthpro','zone_q10','affine','affine_leave_cell_out')]
        lines.append('|'+scene+'|'+'|'.join(entries)+'|')
    lines+=['','主臂失败逐例（>2cm）：','', '|事件|DepthPro误差cm|affine误差cm|拟合尺度/偏置m|分布残差中位m|','|---|---|---|---|---|']
    for r in ledger:
        if r['errors_m']['affine'] is None or abs(r['errors_m']['affine'])>.02:
            fit=r['fits']['affine']['fit']
            lines.append(f'|{r["id"]}|{r["errors_m"]["depthpro"]*100:.3f}|{r["errors_m"]["affine"]*100:.3f}|{fit["slope"]:.3f}/{fit["intercept"]:.3f}|{fit["quantile_residual_median_abs"]:.3f}|')
    lines+=['','机制解释：全图跨格的同分位对应只约束整体尺度和偏置；局部边缘的前景预测误差、前背景排序或占比不一致仍会留在输出中。leave-cell-out检验是否主要依赖目标格本身。分位匹配没有逐像素真值对应，原目标选择仍是已知真值格的条件任务。',
        '', '运行成本：CPU NumPy，小量HDF/NPZ读取与320对分位拟合；无训练/下载/新模型推理。', '',
        '局限：8个已消费合成Development场景、15例；完美几何代理未模拟反射率/遮挡混合/噪声/串扰；不能据此宣称真实硬件融合、碰撞/擦碰识别或取消报警能力。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','run','verify','self-check'),required=True)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='self-check':print(json.dumps(self_check()))
    elif args.stage=='verify':verify(args.output)
    else:
        try:
            run(args.output);save(args.output/'terminal.json',dict(status='complete'))
        except BaseException as e:
            save(args.output/'terminal.json',dict(status='failed',error=repr(e)));raise


if __name__=='__main__':main()
