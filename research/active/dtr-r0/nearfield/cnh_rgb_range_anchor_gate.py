"""One fixed global range anchor; all arms evaluated on whole-range queries.

Exploratory consumed Development check, not M3/hardware fusion confirmation.
"""
import cnh_rgb_visible_query_gate as P  # Sets bounded BLAS threads before numpy.
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import time

import h5py
import numpy as np
import cnh_rgb_visible_query as V
import cnh_rgb_range_anchor as A

ROOT = P.ROOT
OUT = ROOT/'artifacts.local/work/cnh-rgb-range-anchor-20261002'
RUN_ID = 'CNH_RGB_RANGE_ANCHOR_20261002'
ARMS = ('depthpro', 'coarse_q10', 'coarse_median', 'anchored_depthpro')
BOOT_SEED = 2026100212


def prepare(out):
    if (out/'PLAN.json').exists():
        raise FileExistsError('Preserve existing plan')
    rows = [s for s in P.RUNS.read_text(encoding='utf8').splitlines() if RUN_ID in s]
    assert len(rows) == 1
    previous = P.read(P.OUT/'PLAN.json')
    plan = dict(run_id=RUN_ID, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        preregistration_row=rows[0], inputs=previous['inputs'], role=previous['role'],
        split_rule=previous['split_rule'], excluded_scenes=previous['excluded_scenes'],
        source_sha256={str(p.relative_to(ROOT)): P.sha(p) for p in (
            Path(__file__), Path(A.__file__), Path(V.__file__), Path(P.__file__), P.OUT/'PLAN.json')},
        arms=ARMS, budgets=P.BUDGETS,
        anchor='Global exp(median(log(sensor median / predicted median))) over >=8 valid zones; equal zone weight, no clipping/search. Otherwise scale1 and unavailable. Predicted zone median uses its own valid points only. Sensor64 is object-blind reference-radial r^-2 median, no target/query/GT-error selection.',
        whole_query='HEAD/BODY original height bands, optical Z[0.6,2.1); D16 from complete cloud, not OR of range labels. Recalibrate every arm/query/budget independently.',
        truth=previous['truth'], score=previous['score'], coarse=previous['coarse'],
        thresholds=previous['thresholds'], missing=previous['missing'],
        tof_selection='Calibration whole all-contact recall, then lower clear rate, then q10 tie; no eval selection.',
        interpretation='Exploratory effect check, no new promotion gate. Report anchored-minus-raw and anchored-minus-calbestcoarse, actual evaluation clear cost, all shallow denominators/families. Whole contact gain cannot establish shallow clearance ability. Do not subtract old segmented-task scores.',
        bootstrap=dict(cluster='family', eval_clusters=3, draws=1000, seed=BOOT_SEED,
            resample='paired multinomial family weights; fixed thresholds; exclude and report zero-denominator draws'),
        execution='CPU maximum4 workers/BLAS1; no inference/training/download',
        limitations=previous['limitations']+['Same343 consumed Development frames; no M3 observation or hardware noise model',
            'Global scale anchoring is a reused method, not an algorithm novelty claim'])
    P.save(out/'PLAN.json', plan)
    print('PREPARED343frames; fixed scale, two whole queries, four arms', flush=True)


def process_frame(item):
    for kind in ('depth', 'prediction'):
        assert P.sha(ROOT/item[kind+'_path']) == item[kind+'_sha256'], (item['id'], kind)
    with h5py.File(ROOT/item['depth_path'], 'r') as handle:
        radial = handle['dataset'][:].astype(np.float32)
    with np.load(ROOT/item['prediction_path'], allow_pickle=False) as handle:
        predicted = handle['native_depth']
    assert radial.shape == predicted.shape == (768, 1024)
    geometry = V.ray_geometry(radial.shape, item['camera_matrix'])
    whole = A.whole_geometry(geometry)
    truth = A.visible_truth(radial/geometry['radial_factor'], whole)
    scores = {'depthpro': A.query_scores(predicted, whole)}
    coarse_meta = {}
    for arm, quantile in (('coarse_q10', .1), ('coarse_median', .5)):
        coarse = V.coarse_returns(radial, geometry, quantile, weight_power=2)
        scores[arm] = A.query_scores(coarse.pop('expanded_optical_z'), whole)
        coarse_meta[arm] = coarse
    anchored = A.anchor_depth(predicted, geometry, coarse_meta['coarse_median']['zone_return_radial'])
    scores['anchored_depthpro'] = A.query_scores(anchored.pop('scaled_depth'), whole)
    def slice_query(data, index):
        return {k: v[index] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == 2 else v
                for k, v in data.items() if k != 'query_names'}
    rows = []
    for index, query in enumerate(whole['queries']):
        row = {k: item[k] for k in ('id', 'scene', 'family', 'split')}
        row.update(query=index, query_name=query['name'], distance='whole',
            category=str(truth['category'][index]), contact_bin=str(truth['contact_bin'][index]),
            truth=slice_query(truth, index), scores={a: slice_query(d, index) for a,d in scores.items()})
        rows.append(P.plain(row))
    return dict(id=item['id'], rows=rows, anchor=P.plain(anchored), coarse=P.plain(coarse_meta),
        geometry=P.plain({k:v for k,v in whole.items() if not isinstance(v,np.ndarray) or v.size <=1000}))


def select_tof(cal, policies):
    choices = []
    for index, arm in enumerate(('coarse_q10', 'coarse_median')):
        contact = P.tally(P.subset(cal, 'whole', 'contact'), arm, policies)
        clear = P.tally(P.subset(cal, 'whole', 'clear'), arm, policies)
        choices.append(dict(arm=arm, contact=contact, clear=clear,
            key=(contact['rate'] if contact['rate'] is not None else -1,
                 -clear['rate'] if clear['rate'] is not None else -np.inf, -index)))
    return max(choices, key=lambda x:x['key'])['arm'], P.plain(choices)


def summarize(rows, policies, best):
    families = sorted({r['family'] for r in rows})
    assert len(families) == 3
    draws = np.random.default_rng(BOOT_SEED).multinomial(3, [1/3]*3, size=1000)
    output = {}
    for stratum in P.STRATA:
        group = P.subset(rows, 'whole', stratum)
        stats = {a:P.tally(group,a,policies) for a in ARMS}
        sample_rates = {}
        for arm in ARMS:
            cells = [P.tally([r for r in group if r['family']==f],arm,policies) for f in families]
            n = draws @ np.array([c['n'] for c in cells])
            alarms = draws @ np.array([c['alarms'] for c in cells])
            missing = draws @ np.array([c['uncalibrated_query_rows'] for c in cells])
            valid = (n>0) & (missing==0)
            rates = np.divide(alarms,n,out=np.full(1000,np.nan),where=valid)
            sample_rates[arm] = rates
            stats[arm].update(ci95=np.percentile(rates[valid],[2.5,97.5]).tolist() if valid.any() else None,
                bootstrap_valid_draws=int(valid.sum()), bootstrap_excluded_draws=int((~valid).sum()))
        differences = {}
        for reference in ('depthpro', best):
            delta = (sample_rates['anchored_depthpro']-sample_rates[reference])*100
            valid = np.isfinite(delta)
            aa,bb=stats['anchored_depthpro']['rate'],stats[reference]['rate']
            differences[reference] = dict(pp=(aa-bb)*100 if aa is not None and bb is not None else None,
                ci95_pp=np.percentile(delta[valid],[2.5,97.5]).tolist() if valid.any() else None,
                paired_valid_draws=int(valid.sum()), paired_excluded_draws=int((~valid).sum()))
        output[stratum] = dict(arms=stats, anchored_minus=differences)
    return output


def run(out, workers):
    if (out/'result.json').exists() or (out/'frame-ledger.json').exists():
        raise FileExistsError('Preserve existing scientific output')
    plan = P.read(out/'PLAN.json')
    for path,expected in plan['source_sha256'].items():
        assert P.sha(ROOT/path)==expected,path
    started=time.perf_counter()
    frames=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,frame in enumerate(pool.map(process_frame,plan['inputs']),1):
            frames.append(frame)
            if i%25==0 or i==len(plan['inputs']):
                print('scored',i,'/',len(plan['inputs']),flush=True)
    P.save(out/'frame-ledger.json',frames)
    rows=[r for f in frames for r in f['rows']]
    assert len(rows)==686
    cal,evaluation=([r for r in rows if r['split']==s] for s in ('cal','eval'))
    scales=np.array([f['anchor']['scale'] for f in frames])
    result=dict(role=plan['role'], verdict='EXPLORATORY_SCALE_CHECK_COMPLETE', plan_sha256=P.sha(out/'PLAN.json'),
        frame_counts=dict(Counter(i['split'] for i in plan['inputs'])), queries=len(rows),
        truth_counts={s:dict(Counter(r['category'] for r in rows if r['split']==s)) for s in ('cal','eval')},
        anchor=dict(available=sum(f['anchor']['available'] for f in frames),n=len(frames),
            scale_quantiles=dict(zip(('min','q10','median','q90','max'),np.quantile(scales,[0,.1,.5,.9,1]).tolist()))),
        budgets={},limits=plan['limitations'])
    for budget in P.BUDGETS:
        policies={a:[P.calibrate(cal,a,q,budget) for q in range(2)] for a in ARMS}
        best,selection=select_tof(cal,policies)
        result['budgets'][str(budget)]=dict(policies=policies,selected_tof=best,calibration_selection=selection,
            evaluation=summarize(evaluation,policies,best),
            calibration={s:{a:P.tally(P.subset(cal,'whole',s),a,policies) for a in ARMS} for s in P.STRATA})
    result['seconds']=time.perf_counter()-started
    P.save(out/'result.json',result)
    lines=['# 固定粗测距尺度锚：全距离可见查询','',result['verdict'],'',
        '343帧/44场景，cal193帧3family、eval150帧3family；已消费Development。HEAD/BODY全距离0.6–2.1m，重新校准各方法。',
        '64格参考径向深度r^-2 median模拟观测约束RGB全局尺度；不是M3或硬件。预测器不读GT像素或目标身份。','',
        '|cal清晰预算|真值|原RGB报警/n|尺度RGB报警/n|cal选粗测距报警/n|尺度−原RGB pp|尺度−粗测距 pp|',
        '|---|---|---|---|---|---|---|']
    for budget,data in result['budgets'].items():
        for name,s in data['evaluation'].items():
            stats=s['arms']; best=data['selected_tof']
            counts=[f"{stats[a]['alarms']}/{stats[a]['n']}" for a in ('depthpro','anchored_depthpro',best)]
            delta=[s['anchored_minus'][a]['pp'] for a in ('depthpro',best)]
            displays=['NOT_EVALUABLE' if d is None else f'{d:.2f}' for d in delta]
            lines.append('|'+ '|'.join([budget,name,*counts,*displays])+'|')
    lines+=['','完整4臂、16阈值、所有UNKNOWN/缺测、64格残差与尺度、FOV/遮挡诊断和family配对区间保留JSON。',
        'cal预算不保证eval误报相等。pass报警不计代价；UNKNOWN保留但不作清晰。',
        '全距离任务修正不等于尺度贡献；只比较本轮同口径各臂，不减旧分段数字。',
        '整体接触改善不能替代浅擦碰分母；可见表面不能排除隐藏碰撞；不作RGB融合、M3、实机或新单位确认。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('EXPLORATORY_SCALE_CHECK_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,choices=range(1,5),default=4)
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare(args.out)
    else:
        run(args.out,args.workers)
