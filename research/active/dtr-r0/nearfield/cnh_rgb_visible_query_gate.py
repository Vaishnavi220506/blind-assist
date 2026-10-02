"""Frozen camera-relative visible-query comparison on consumed Development.

No new inference, training, downloaded input, privileged target cell or floor.
The coarse reference-depth simulator is a geometry proxy, not deployed M3.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

import cnh_rgb_visible_query as adapter

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
DATA = ROOT/'artifacts.local/datasets/hypersim-ba-nfo'
OUT = ROOT/'artifacts.local/work/cnh-rgb-visible-query-20261002'
EXCLUDE = ROOT/'artifacts.local/work/cnh-rgb-clearance-probe-20261001/selection.json'
RUNS = ROOT/'research/active/dtr-r0/RUNS.md'
RUN_ID = 'CNH_RGB_VISIBLE_QUERY_20261002'
ARMS = ('depthpro', 'coarse_q10', 'coarse_median')
STRATA = ('contact0-2', 'contact2-5', 'contact>5', 'contact', 'pass', 'clear', 'UNKNOWN')
BUDGETS = (.1, .2)
BOOT_SEED = 2026100211


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def prepare(out):
    if (out/'PLAN.json').exists():
        raise FileExistsError('Preserve existing plan')
    prereg = [line for line in RUNS.read_text(encoding='utf8').splitlines() if RUN_ID in line]
    assert len(prereg) == 1, 'Exactly one preregistration RUNS row required before input access'
    manifest = read(SOURCE/'manifest.json')
    observations = {r['id']: r for r in read(SOURCE/'observations.json')}
    excluded = sorted({r['scene'] for r in read(EXCLUDE)})
    assert len(manifest) == 500 and len(excluded) == 8
    rows = [r for r in manifest if r['scene'] not in excluded]
    assert len(rows) == 343 and len({r['scene'] for r in rows}) == 44
    assert all(r['source'] == 'hypersim' and r['split'] == 'val' for r in rows)
    families = sorted({r['family'] for r in rows}, key=lambda f: hashlib.sha256(('rgb-visible-query-20261002|'+f).encode()).hexdigest())
    assert families == ['ai_051', 'ai_033', 'ai_023', 'ai_017', 'ai_047', 'ai_035']
    inputs = []
    for row in sorted(rows, key=lambda r: r['id']):
        depth = DATA/row['depth']
        pred = SOURCE/'predictions/native'/(row['id']+'.npz')
        depth_hash = sha(depth)
        assert depth_hash == row['depth_sha256'], row['id']
        inputs.append(dict(id=row['id'], scene=row['scene'], family=row['family'],
            split='cal' if row['family'] in families[:3] else 'eval',
            depth_path=str(depth.relative_to(ROOT)), depth_sha256=depth_hash,
            prediction_path=str(pred.relative_to(ROOT)), prediction_sha256=sha(pred),
            camera_matrix=observations[row['id']]['camera_matrix']))
    assert Counter(r['split'] for r in inputs) == {'cal': 193, 'eval': 150}
    plan = dict(run_id=RUN_ID, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        preregistration_row=prereg[0], preregistration_sha256=hashlib.sha256(prereg[0].encode()).hexdigest(),
        role='Consumed Hypersim Development; no fresh confirmation; original52-edge mechanisms remain stopped',
        inputs=inputs, families_order=families, excluded_scenes=excluded,
        split_rule='sha256(rgb-visible-query-20261002|family), first3cal/last3eval; no reshuffling for outcomes',
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__), Path(adapter.__file__), SOURCE/'manifest.json', SOURCE/'observations.json', EXCLUDE)},
        arms=ARMS, queries=adapter.QUERY_SPECS,
        score='0.30 minus16th smallest absX among finite positive optical-Z points in query y/z and common45degree FOV; fewer16 -> no finite alarm score; coverage<95% or no possible rays -> explicit abstain',
        truth='All visible surfaces;16pixel support; coverage>=95%;1..15 contact/expanded pixels UNKNOWN; no floor/object/pose truth',
        coarse='64 equiangular zones; native radial q10/median weighted1/r², all object-blind valid zone pixels; one return broadcast across entire zone as constant radial distance. This is a reconstruction prior, not independent pixel measurements or M3.',
        thresholds='Per arm/query/budget exact finite observed score breakpoints plus all-off; choose lowest threshold satisfying calibration-clear alarm budget. score>=threshold alarms. No calibration clear -> NOT_EVALUABLE.',
        tof_selection='Per budget choose among q10/median using calibration far0-2 recall, then2-5,allcontact,-clear,false index(q10 first). Missing recall criterion=-1.',
        missing='Reference UNKNOWN excluded from known-label metrics but all rows retained. Prediction abstention counts missed contact and not certified clear. Missing calibrated query makes aggregate metric NOT_EVALUABLE, never silently drops query.',
        budgets=BUDGETS, bootstrap=dict(cluster='family', eval_clusters=3, draws=1000, seed=BOOT_SEED,
            resample='paired shared multinomial family weights, fixed thresholds/model selection; zero-denominator draws excluded and counted'),
        primary_verdict='At10% budget far eval: if either0-2/2-5 n<20 or representedfamilies<3 => DESCRIPTIVE_LOW_SHALLOW_SUPPORT; else both DepthPro-minus-calbestToF>=3pp => PROMISING_VISIBLE_QUERY_CONTROL; otherwise NO_STABLE_SHALLOW_GAIN. Uncalibrated needed query => NOT_EVALUABLE.',
        execution='CPU maximum4frameworkers, BLAS1; no new inference/training/download',
        limitations=['Camera-relative virtual query, not calibrated wearer or hidden complete collision volume',
            'Visible clear cannot rule out geometry behind foreground occluders',
            'Geometry-derived coarse proxies, not M3 or hardware; uniform material/r^-2 proxy',
            'Three evaluation families and previously consumed data limit generalization'])
    save(out/'PLAN.json', plan)
    print('PREPARED343frames44scenes; no scientific scoring', flush=True)


def process_frame(item):
    for kind in ('depth', 'prediction'):
        assert sha(ROOT/item[kind+'_path']) == item[kind+'_sha256'], (item['id'], kind)
    with h5py.File(ROOT/item['depth_path'], 'r') as handle:
        radial = handle['dataset'][:].astype(np.float32)
    with np.load(ROOT/item['prediction_path'], allow_pickle=False) as handle:
        predicted = handle['native_depth']
    assert radial.shape == predicted.shape == (768, 1024)
    geometry = adapter.ray_geometry(radial.shape, item['camera_matrix'])
    truth = adapter.visible_truth(radial/geometry['radial_factor'], geometry)
    scores = {'depthpro': adapter.query_scores(predicted, geometry)}
    coarse_meta = {}
    for arm, q in (('coarse_q10', .1), ('coarse_median', .5)):
        coarse = adapter.coarse_returns(radial, geometry, q, weight_power=2)
        scores[arm] = adapter.query_scores(coarse.pop('expanded_optical_z'), geometry)
        coarse_meta[arm] = coarse
    rows = []
    for index, query in enumerate(geometry['queries']):
        row = {k: item[k] for k in ('id', 'scene', 'family', 'split')}
        row.update(query=index, query_name=query['name'], distance='near' if query['z_near'] < 1 else 'far',
            category=str(truth['category'][index]), contact_bin=str(truth['contact_bin'][index]),
            truth={k: v[index] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == 4 else v
                for k, v in truth.items() if k not in ('query_names',)}, scores={})
        for arm, data in scores.items():
            row['scores'][arm] = {k: v[index] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == 4 else v
                for k, v in data.items() if k != 'query_names'}
            row['scores'][arm]['abstain'] = bool(data['abstain'][index])
        rows.append(plain(row))
    return dict(id=item['id'], rows=rows, coarse=plain(coarse_meta),
        geometry=plain({k: v for k, v in geometry.items()
            if not isinstance(v, np.ndarray) or v.size <= 1000}))


def calibrate(rows, arm, query, budget):
    group = [r for r in rows if r['query'] == query]
    clear = [r for r in group if r['category'] == 'clear']
    if not clear:
        return dict(status='NOT_EVALUABLE_NO_CAL_CLEAR', threshold=None, clear_n=0)
    finite = sorted({r['scores'][arm]['score'] for r in group if r['scores'][arm]['score'] is not None})
    for threshold in finite+[None]:
        alarms = sum(r['scores'][arm]['score'] is not None and threshold is not None and r['scores'][arm]['score'] >= threshold for r in clear)
        if alarms <= budget*len(clear)+1e-12:
            return dict(status='READY', threshold=threshold, all_off=threshold is None,
                clear_n=len(clear), clear_alarms=alarms, clear_rate=alarms/len(clear), breakpoint_count=len(finite)+1)
    raise AssertionError('All-off must be feasible')


def alarm(row, arm, policies):
    policy = policies[arm][row['query']]
    if policy['status'] != 'READY':
        return None
    score = row['scores'][arm]['score']
    return bool(score is not None and policy['threshold'] is not None and score >= policy['threshold'])


def subset(rows, distance, stratum):
    return [r for r in rows if r['distance'] == distance and
        (r['contact_bin'] == stratum if stratum.startswith('contact') and stratum != 'contact' else r['category'] == stratum)]


def tally(rows, arm, policies, weights=None):
    weights = weights or {r['family']: 1 for r in rows}
    n = sum(weights.get(r['family'], 0) for r in rows)
    uncalibrated = sum(weights.get(r['family'], 0) for r in rows if alarm(r, arm, policies) is None)
    alarms = sum(weights.get(r['family'], 0) for r in rows if alarm(r, arm, policies) is True)
    return dict(n=n, alarms=alarms, rate=alarms/n if n and not uncalibrated else None,
        uncalibrated_query_rows=uncalibrated,
        prediction_abstain=sum(weights.get(r['family'], 0) for r in rows if r['scores'][arm]['abstain']),
        families=len({r['family'] for r in rows if weights.get(r['family'], 0) > 0}))


def choose_tof(cal, policies):
    choices = []
    for index, arm in enumerate(ARMS[1:]):
        rates = [tally(subset(cal, 'far', s), arm, policies)['rate'] for s in ('contact0-2','contact2-5','contact','clear')]
        key = tuple((-1 if rate is None else rate) for rate in rates[:3]) + ((-np.inf if rates[3] is None else -rates[3]), -index)
        choices.append(dict(arm=arm, key=key, rates=rates))
    best = max(choices, key=lambda r: r['key'])['arm']
    return best, plain(choices)


def summarize(rows, policies, best):
    families = sorted({r['family'] for r in rows})
    assert len(families) == 3
    draws = np.random.default_rng(BOOT_SEED).multinomial(3, [1/3]*3, size=1000)
    output = {}
    for distance in ('near', 'far'):
        output[distance] = {}
        for stratum in STRATA:
            group = subset(rows, distance, stratum)
            stats = {arm: tally(group, arm, policies) for arm in ARMS}
            delta = stats['depthpro']['rate']-stats[best]['rate'] if stats['depthpro']['rate'] is not None and stats[best]['rate'] is not None else None
            deltas, rates = [], {arm: [] for arm in ARMS}
            for draw in draws:
                weights = dict(zip(families, draw.tolist()))
                samples = {arm: tally(group, arm, policies, weights) for arm in ARMS}
                for arm in ARMS:
                    if samples[arm]['rate'] is not None:
                        rates[arm].append(samples[arm]['rate'])
                if samples['depthpro']['rate'] is not None and samples[best]['rate'] is not None:
                    deltas.append((samples['depthpro']['rate']-samples[best]['rate'])*100)
            for arm in ARMS:
                stats[arm].update(ci95=np.percentile(rates[arm], [2.5,97.5]).tolist() if rates[arm] else None,
                    bootstrap_valid_draws=len(rates[arm]), bootstrap_excluded_draws=1000-len(rates[arm]))
            output[distance][stratum] = dict(arms=stats, depthpro_minus_selected_tof_pp=None if delta is None else delta*100,
                difference_ci95_pp=np.percentile(deltas,[2.5,97.5]).tolist() if deltas else None,
                paired_valid_draws=len(deltas), paired_excluded_draws=1000-len(deltas))
    return output


def run(out, workers):
    if (out/'result.json').exists() or (out/'frame-ledger.json').exists():
        raise FileExistsError('Preserve result')
    plan = read(out/'PLAN.json')
    for path, expected in plan['source_sha256'].items():
        assert sha(ROOT/path) == expected, path
    started = time.perf_counter()
    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, frame in enumerate(pool.map(process_frame, plan['inputs']), 1):
            frames.append(frame)
            if index % 25 == 0 or index == len(plan['inputs']):
                print('scored', index, '/', len(plan['inputs']), flush=True)
    save(out/'frame-ledger.json', frames)
    rows = [r for frame in frames for r in frame['rows']]
    assert len(rows) == 343*4
    cal, evaluation = ([r for r in rows if r['split'] == split] for split in ('cal','eval'))
    result = dict(role=plan['role'], frame_counts=dict(Counter(r['split'] for r in plan['inputs'])),
        queries=len(rows), plan_sha256=sha(out/'PLAN.json'), budgets={}, limits=plan['limitations'])
    for budget in BUDGETS:
        policies = {arm: [calibrate(cal, arm, q, budget) for q in range(4)] for arm in ARMS}
        best, selection = choose_tof(cal, policies)
        result['budgets'][str(budget)] = dict(policies=policies, selected_tof=best,
            calibration_selection=selection, evaluation=summarize(evaluation, policies, best),
            calibration={distance: {s: {a: tally(subset(cal,distance,s),a,policies) for a in ARMS}
                for s in STRATA} for distance in ('near','far')})
    primary = result['budgets']['0.1']['evaluation']['far']
    bins = [primary[b] for b in ('contact0-2','contact2-5')]
    if any(x['arms']['depthpro']['n'] < 20 or x['arms']['depthpro']['families'] < 3 for x in bins):
        verdict = 'DESCRIPTIVE_LOW_SHALLOW_SUPPORT'
    elif any(x['depthpro_minus_selected_tof_pp'] is None for x in bins):
        verdict = 'NOT_EVALUABLE'
    elif all(x['depthpro_minus_selected_tof_pp'] >= 3 for x in bins):
        verdict = 'PROMISING_VISIBLE_QUERY_CONTROL'
    else:
        verdict = 'NO_STABLE_SHALLOW_GAIN'
    result.update(verdict=verdict, seconds=time.perf_counter()-started)
    save(out/'result.json', result)
    lines = ['# RGB全图可见表面查询：冻结检查', '', verdict, '',
        '343帧/44场景；cal193帧3family、eval150帧3family。排除旧52边缘的8场景，仍为已消费Development。',
        '相机坐标虚拟HEAD/BODY、近0.6–1.2m/远1.2–2.1m。全图观测驱动，无目标格、地面或物体身份赠送。', '',
        '粗格q10/median来自参考深度的r^-2几何代理，不是已训练M3或硬件；全格广播是固定重建先验，不是多个独立返回。', '',
        '|预算|校准选择ToF|距离/真值|DepthPro报警/n|ToF报警/n|差pp（family条件95%区间）|', '|---|---|---|---|---|---|']
    for budget, data in result['budgets'].items():
        for distance, strata in data['evaluation'].items():
            for name, s in strata.items():
                a,b=s['arms']['depthpro'],s['arms'][data['selected_tof']]
                delta=s['depthpro_minus_selected_tof_pp'];ci=s['difference_ci95_pp']
                display='NOT_EVALUABLE' if delta is None else f'{delta:.2f} [{ci[0]:.2f},{ci[1]:.2f}]'
                lines.append(f"|{budget}|{data['selected_tof']}|{distance}/{name}|{a['alarms']}/{a['n']}|{b['alarms']}/{b['n']}|{display}|")
    lines += ['', 'pass报警不计代价；UNKNOWN仅报告分母。预测缺测在已知contact计漏检，在clear不称正确清晰。',
        '参考clear只涵盖图像可见表面，不能排除前方遮挡后的碰撞。FOV裁切、有效支持和前方遮挡诊断保留在完整frame-ledger。',
        '所有3方法、工作点和1000次family配对bootstrap的有效/排除draw均保留result.json；仅3评估family，区间条件于已定阈值。',
        '不称佩戴者安全、隐藏完整真值、NFO500外确认或RGB融合收益；旧52机制和大位移机制继续停止。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps(dict(verdict=verdict,seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','run'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, choices=range(1,5), default=4)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.out)
    else:
        run(args.out,args.workers)
