"""Conditional coarse-range association assay on the previous fixed 15 edges.

Visible reference depth is simulator geometry, never measured ToF. Instance
truth audits source attribution after object-blind range selection only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from cnh_rgb_clearance_geometry import reference_frame
from cnh_rgb_clearance_edge import zone_map
from cnh_rgb_clearance_probe import summarize
from cnh_rgb_zone_range import summarize_distribution, self_check as range_check
from cnh_rgb_zone_sensor import sensor_readouts, self_check as sensor_check

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT/'artifacts.local/work/cnh-rgb-clearance-probe-20261001'
CACHE = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
OUT = ROOT/'artifacts.local/work/cnh-rgb-zone-association-20261001'
ARMS = {
    'depthpro': '预测边缘+Depth Pro距离（已有）',
    'oracle_target': '同一边缘+正确目标GT光学Z（已有）',
    'zone_median': '整格径向中位数',
    'zone_q10': '整格径向10%分位',
    'strongest_mode': '最强几何模式',
    'rgb_guided_mode': 'RGB预测距离引导模式',
    'oracle_best_mode': '真值选最小误差模式（上限）',
    'h3_clean': 'H3简化电子代理最强标量',
    'h3_stress': 'H3默认压力代理最强标量',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    files = [Path(__file__), HERE/'cnh_rgb_zone_range.py', HERE/'cnh_rgb_zone_sensor.py',
             HERE/'cnh_route_sensor.py', HERE/'cnh_rgb_clearance_geometry.py',
             HERE/'cnh_rgb_clearance_edge.py', HERE/'cnh_rgb_clearance_probe.py',
             PRIOR/'selection.json', PRIOR/'case-ledger.json', PRIOR/'result.json',
             PRIOR/'evaluation-input-seal.json', CACHE/'observations.json', CACHE/'manifest.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def prepare(out):
    assert not (out/'PLAN.json').exists(), 'Retain old run; use a new directory'
    save(out/'PLAN.json', dict(
        scope='consumed Hypersim Development, fixed 15 edges/8 scenes from prior precision assay',
        question='Does an object-blind coarse-cell range support the RGB edge target?',
        observations='perfect first-visible reference radial depth used as simulator input; no measured ToF exists',
        invariants='same 15 events, same predicted edge and coarse equal-angle45deg/8x8 cell; no selection/threshold/model changes',
        pixel_proxy='whole original cell; uniform pixel radial samples in(0,12.8m); no height/range/query/semantic masking',
        modes='5cm bins; Gaussian sigma1bin truncate4; local maxima, leftmost plateau; >=5% raw support in+-1bin; strength sorted NMS<=3bins; at most4; support median range',
        association='RGB prior=DepthPro foreground opticalZ converted to radial at predicted edge; nearest retained mode by absolute range; no truth/identity passed to selector',
        geometry='radial readout projected along predicted edge ray: side*lateral_factor*opticalZ_per_radial*r-0.30; no axis/scale fit',
        controls='existing target oracle; median/q10/strongest; GT best-mode chosen by clearance error is explicitly oracle and may select another object',
        electronics='original uncalibrated synthesize_response/derive_readout H3, adapted to same equal-angle grid using16x16 angular quadrature and full inverse camera matrix; rho0.5,cos1; clean disables noise/tail/crosstalk/neighbour leak; stress defaults, one fixed frame seed',
        attribution='evaluator-only semantic/instance shares in chosen mode support; >=50% target is majority. Instance-1 not individually attributable; wall semantic share descriptive only. Never fed back to selection',
        metrics='same all-case1/2/5cm including missing; bias/tails/sign error and fixed-policy scene-bootstrap1000. Association majority count and target-mode availability separate from metric coincidence',
        limits=['known target cell and previous clean visible-edge/floor/roll scope',
                'only2/15 within2cm of body line; cannot estimate grazing-contact recall',
                'uniform-pixel geometry modes are not photon histogram peaks',
                'H3 pulse/reflectance/noise/binning assumptions are not hardware calibration',
                'radial transfer from cell to edge assumes common target/range; slant and angle variation retained',
                'one synthetic stress draw; no physical success-rate estimate',
                'only8 scene clusters; bootstrap may degenerate at all-success/all-failure'],
        stop='finish fixed assay; no training, model inference, rendering, new data, protected/final access or hardware deployment',
        input_sha256=sources()))


def mode_attribution(mode, radial, target, known):
    mask = (np.isfinite(radial) & (radial > 0)
            & (radial >= mode['support_lower_radial_m'])
            & (radial < mode['support_upper_radial_exclusive_m']))
    n = int(mask.sum())
    assert n == mode['support_samples']
    fraction = float(target[mask].mean()) if n else None
    return dict(target_fraction=fraction, target_majority=bool(fraction >= .5) if known and fraction is not None else None,
                identity_known=known, support_samples=n)


def attribution_check():
    # The source distribution drops invalid zeros even when support starts at0.
    radial = np.array([0., np.nan, -.01, .025, .025])
    mode = summarize_distribution(radial)['modes'][0]
    audit = mode_attribution(mode, radial, np.array([False, False, False, True, True]), True)
    assert audit['support_samples'] == 2 and audit['target_fraction'] == 1.
    assert mode_attribution(mode, radial, np.ones(5, bool), False)['target_majority'] is None
    return dict(status='PASS',checks=['zero-invalid mode support excluded','unknown identity not asserted'])


def evaluate_case(event, old, row, camera):
    ref = reference_frame(ROOT, row, camera)
    cell = zone_map(camera) == event['zone_id']
    pixels = np.asarray(event['support_yx'])
    yy, xx = pixels.T
    np.testing.assert_allclose(np.median(event['side']*ref['lateral'][yy, xx]-.30), event['gt_clearance_m'], atol=1e-8)
    prediction = old['prediction']
    assert prediction['status'] == 'OK'
    y, xhalf = prediction['edge_pixel']
    x = int(np.floor(xhalf))
    optical_per_radial = float(np.mean(ref['optical_z_per_radial'][y, x:x+2]))
    radial_factor = event['side']*prediction['lateral_factor']*optical_per_radial
    prior = prediction['foreground_depth_m']/optical_per_radial
    # Only unmasked visible radial geometry and RGB prior enter range selection.
    radial = ref['radial'][cell].astype(np.float64)
    distribution = summarize_distribution(radial, radial_prior=prior)
    seed = int(hashlib.sha256(event['frame_id'].encode()).hexdigest()[:8], 16)
    electronic = sensor_readouts(ref['radial'], camera, seed)
    zone = event['zone_id']
    ranges = dict(zone_median=distribution['median_radial_m'], zone_q10=distribution['q10_radial_m'],
                  strongest_mode=distribution['strongest_radial_m'], rgb_guided_mode=distribution['guided_radial_m'],
                  h3_clean=electronic['clean']['distance_m'][zone], h3_stress=electronic['stress']['distance_m'][zone])
    modes = distribution['modes']
    selected_modes = {}
    if modes:
        selected_modes = dict(strongest_mode=0,
            rgb_guided_mode=min(range(len(modes)), key=lambda i: abs(modes[i]['radial_m']-prior)),
            oracle_best_mode=min(range(len(modes)), key=lambda i: abs(radial_factor*modes[i]['radial_m']-.30-event['gt_clearance_m'])))
        ranges['oracle_best_mode'] = modes[selected_modes['oracle_best_mode']]['radial_m']
    else:
        ranges['oracle_best_mode'] = None
    estimates = dict(depthpro=old['estimates']['local_edge_depthpro'],
                     oracle_target=old['estimates']['local_edge_oracle_range'])
    estimates.update({arm:float(radial_factor*r-.30) if r is not None else None for arm,r in ranges.items()})
    # Truth joins only after all object-blind selectors are finished.
    sem, inst = ref['semantic'][cell], ref['instance'][cell]
    known = event['instance_id'] >= 0
    target = (sem == event['semantic']) & (inst == event['instance_id']) if known else sem == event['semantic']
    eligible = np.isfinite(radial) & (radial > 0) & (radial < 12.8)
    mode_audits = [mode_attribution(mode, radial, target, known) for mode in modes]
    composition_keys, counts = np.unique(np.stack([sem[eligible],inst[eligible]],axis=1),axis=0,return_counts=True)
    order = np.argsort(-counts,kind='stable')
    composition = [dict(semantic=int(composition_keys[i,0]),instance_id=int(composition_keys[i,1]),
                        fraction=float(counts[i]/eligible.sum())) for i in order[:10]]
    result = {k:v for k,v in event.items() if k!='support_yx'}
    result.update(prediction=prediction, radial_edge_factor=radial_factor, optical_z_per_radial_at_edge=optical_per_radial,
                  gt_support_radial_median_m=float(np.median(ref['radial'][yy,xx])), distribution=distribution,
                  simulated_radial_ranges_m=ranges, estimates=estimates,
                  errors_m={arm:v-event['gt_clearance_m'] if v is not None else None for arm,v in estimates.items()},
                  identity_known=known, cell_target_fraction=float(target[eligible].mean()) if eligible.any() else None,
                  cell_other_instance_fraction=float(((inst!=event['instance_id']) & (inst>=0))[eligible].mean()) if known and eligible.any() else None,
                  cell_semantic_composition=composition, mode_attribution=mode_audits,
                  selected_mode_indices=selected_modes,
                  selected_mode_attribution={arm:mode_audits[i] for arm,i in selected_modes.items()},
                  electronics=electronic)
    return result


def run(out):
    plan = read(out/'PLAN.json')
    assert sources() == plan['input_sha256']
    assert not (out/'result.json').exists(), 'Preserve existing results'
    parent_seal = read(PRIOR/'evaluation-input-seal.json')
    assert all(sha(ROOT/p) == value for p,value in parent_seal['input_sha256'].items())
    selection, old = read(PRIOR/'selection.json'), read(PRIOR/'case-ledger.json')
    assert len(selection) == len(old) == 15 and {e['id'] for e in selection} == {e['id'] for e in old}
    manifest = {r['id']:r for r in read(CACHE/'manifest.json')}
    cameras = {r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    old = {r['id']:r for r in old}
    save(out/'input-seal.json', dict(plan_sha256=sha(out/'PLAN.json'), parent_input_sha256=parent_seal['input_sha256']))
    save(out/'self-check.json', dict(range=range_check(),sensor=sensor_check(),attribution=attribution_check()))
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    backend = select_backend(Workload.SCALAR_SCORING,
        cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(128).mean(),lambda _:DeviceObservation('cpu','host CPU','numpy',('CPU',))),
        capabilities={'python_executable':sys.executable,'reason_code':'TASK_NOT_GPU_SUITABLE'},record_path=out/'backend.json')
    ledger = []
    for e in selection:
        ledger.append(evaluate_case(e,old[e['id']],manifest[e['frame_id']],cameras[e['frame_id']]))
        print('evaluated',len(ledger),'/',len(selection),flush=True)
    save(out/'case-ledger.json',ledger)
    association = {}
    for arm in ('strongest_mode','rgb_guided_mode','oracle_best_mode'):
        eligible_rows = [r for r in ledger if r['identity_known']]
        majority = sum(r.get('selected_mode_attribution',{}).get(arm,{}).get('target_majority') is True for r in eligible_rows)
        association[arm] = dict(n=len(eligible_rows),target_majority=majority,
            missing=sum(arm not in r['selected_mode_attribution'] for r in eligible_rows))
    result = dict(status='COMPLETE',scope=plan['scope'],n=len(ledger),scenes=len({r['scene'] for r in ledger}),
        plan_sha256=sha(out/'PLAN.json'),arms={arm:dict(overall=summarize(ledger,arm),
            by_range={s:summarize(ledger,arm,lambda r,name=s:r['range_bin']==name) for s in ('1.15-1.6m','1.6-2.1m')}) for arm in ARMS},
        association=association,mode_availability=dict(
            cases_with_modes=sum(bool(r['distribution']['modes']) for r in ledger),
            identity_known=sum(r['identity_known'] for r in ledger),
            cases_with_target_majority_mode=sum(any(m['target_majority'] is True for m in r['mode_attribution']) for r in ledger if r['identity_known'])),
        limits=plan['limits'],backend=backend)
    save(out/'result.json',result)
    report(out,result,ledger)
    verify(out)
    snapshot = out/'source'; snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__),HERE/'cnh_rgb_zone_range.py',HERE/'cnh_rgb_zone_sensor.py',HERE/'cnh_route_sensor.py']:
        (snapshot/p.name).write_bytes(p.read_bytes())


def verify(out):
    plan,seal = read(out/'PLAN.json'),read(out/'input-seal.json')
    assert plan['input_sha256'] == sources() and sha(out/'PLAN.json') == seal['plan_sha256']
    assert all(sha(ROOT/p) == h for p,h in seal['parent_input_sha256'].items())
    ledger,result = read(out/'case-ledger.json'),read(out/'result.json')
    for arm in ARMS:
        assert summarize(ledger,arm) == result['arms'][arm]['overall']
        for r in ledger:
            v=r['estimates'][arm]
            if v is not None: assert abs(v-r['gt_clearance_m']-r['errors_m'][arm])<1e-12
    for r in ledger:
        for arm in ('zone_median','zone_q10','strongest_mode','rgb_guided_mode','oracle_best_mode','h3_clean','h3_stress'):
            radial = r['simulated_radial_ranges_m'][arm]
            if radial is not None: assert abs(radial*r['radial_edge_factor']-.30-r['estimates'][arm])<1e-12
    save(out/'verification.json',dict(status='PASS',cases=len(ledger),arms=len(ARMS),
        checks=['original frame input receipts','unchanged15case/edge selection','radial-edge projection arithmetic',
                'all-case denominators/scene-bootstrap','object-blind selector contract mathematical checks']))


def report(out,result,ledger):
    lines=['# 粗格距离与RGB边缘的目标归属小试','',
        '固定上一轮15例/8场景/原预测边缘和名义等角粗格。所有新增距离来自官方首可见深度的几何或未标定电子代理，不是实测ToF。语义/实例只在对象盲选距离后做归属审计。','',
        '|距离条件|≤2cm（全部15例）及95%场景区间|绝对误差中位/P95(cm)|缺失|',
        '|---|---|---|---|']
    for arm,label in ARMS.items():
        s=result['arms'][arm]['overall']; q=s['absolute_error_cm_quantiles'];ci=s['within_cm_all_ci95']['2']
        qs=f"{q['p50']:.2f}/{q['p95']:.2f}" if q else 'UNKNOWN'
        lines.append(f"|{label}|{round(s['within_cm_all']['2']*s['n'])}/{s['n']} [{ci[0]:.1%},{ci[1]:.1%}]|{qs}|{s['missing']}|")
    lines+=['','归属审计（实例未知的墙体不纳入独立对象归属分母；不删除净距误差）：']
    for arm,a in result['association'].items():
        lines.append(f"- {ARMS[arm]}：所选模式支持中目标占比≥50%为{a['target_majority']}/{a['n']}，无模式{a['missing']}。")
    av=result['mode_availability']
    lines += [f"- 有模式{av['cases_with_modes']}/15；存在目标占多数模式{av['cases_with_target_majority_mode']}/{av['identity_known']}。",'',
        '整格统计和几何模式均按像素等权，5cm几何箱、平滑、5%支持门槛和最多4模式是固定诊断假设，不是传感器物理峰数或分辨率。RGB引导只按Depth Pro前景径向距离选最近模式。oracle_best_mode用真实净距选误差最小模式，是上限且可能选到别的物体；净距碰巧接近不等于归属正确。', '',
        'H3代理使用相同等角格的16×16角度采样和完整inverse M投影、固体角权重；原CNH库为等tan格，本轮适配不复用其zone索引。rho=.5、incidence=1。clean关闭噪声、尾巴、串扰和邻格泄漏，仍保留脉冲及H3的16×30.02784cm箱中心最强标量读出。stress使用原默认参数和固定逐帧seed的一次合成读数，不能称硬件性能；bin宽不能称实际测距精度。默认压力的串扰可能形成近距伪峰；本轮不在结果后屏蔽或改阈值。','',
        '将一格的径向读数沿RGB边缘射线投影，不代表峰本来就在该射线上；对象归属、斜面和角度变化均保留在误差中。详细每格组成、模式支持/身份占比、距离、电子参数及逐例错误见case-ledger.json。','',
        '局限：']+['- '+t for t in result['limits']]
    lines += ['','建议只依据本轮对象盲距离及关联结果判断下一验证优先级；不做训练、部署或硬件能力结论。', '',
              '![净距精度](association-errors.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    short=['DepthPro','target oracle','cell median','cell q10','strongest mode','RGB guided','GT best mode','H3 clean','H3 stress']
    rates=[result['arms'][a]['overall']['within_cm_all']['2'] for a in ARMS]
    axes[0].barh(short,rates);axes[0].set(xlim=(0,1),xlabel='fraction of all15 cases within2cm');axes[0].invert_yaxis()
    for arm,label in zip(('depthpro','zone_median','strongest_mode','rgb_guided_mode','oracle_target'),('DepthPro','cell median','strongest mode','RGB guided','target oracle')):
        err=np.sort([abs(r['errors_m'][arm])*100 for r in ledger if r['errors_m'][arm] is not None])
        if len(err): axes[1].step(err,np.arange(1,len(err)+1)/15,where='post',label=label)
    axes[1].set(xlim=(0,30),ylim=(0,1),xlabel='absolute clearance error(cm)',ylabel='fraction of all15 cases')
    axes[1].axvline(2,color='black',ls='--');axes[1].legend(fontsize=8)
    fig.suptitle('Synthetic coarse-cell range association; known target cell; consumed Development')
    fig.tight_layout();fig.savefig(out/'association-errors.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('prepare','run','verify'),required=True)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='verify':verify(args.output)
    else:
        try:
            run(args.output);save(args.output/'terminal.json',dict(status='complete'))
        except BaseException as error:
            save(args.output/'terminal.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':main()
