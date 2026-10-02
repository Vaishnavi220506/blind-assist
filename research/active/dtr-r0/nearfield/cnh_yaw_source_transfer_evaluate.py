"""Frozen M3/envelope transfer to prior source evaluation units96000-96095.

No calibration or early stop. Nominal all-object observed geometry is fixed;
clear subgroup descriptions use nominal target identity/gap, not dynamic gap.
"""
import argparse
import json

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_sequence_extrinsic_evaluate as XE
import cnh_sequence_observed_evaluate as OE
import cnh_three_level_sequence as SE
import cnh_yaw_envelope_evaluate as YE

OUT = MC.SS.WORK/'cnh-yaw-source-transfer-20261002'
UNITS = list(range(96000, 96096))
FIXED = {'BASE': .8557642486787612, 'ENV': 1.569529990996084}
BIASES = (-1, 0, 1)
BOOT_SEED = 2026100220
N_BOOT = 1000
CATEGORIES = OE.CONTACTS+('pass0-10cm', 'clear')


def load_inputs():
    plan_path, receipt_path, request_path = OUT/'PLAN.json', OUT/'scores_receipt.json', OUT/'request.json'
    plan, receipt = OE.read(plan_path), OE.read(receipt_path)
    if plan['fixed_thresholds'] != FIXED or plan['bootstrap']['seed'] != BOOT_SEED:
        raise ValueError('Frozen thresholds or bootstrap seed differ')
    YE.verify_hashes(plan['prior_sha256'])
    if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != ME.sha(plan_path) or receipt.get('request_sha256') != ME.sha(request_path):
        raise ValueError('Score producer is incomplete or plan/request digest differs')
    full, old_scores, old_provenance = OE.load_inputs()
    keep = full['split'] == 'evaluation'
    fields = ('unit', 'config', 'query', 'split', 'frame_ranges', 'frame_category', 'ref_category', 'covered', 'clear_all', 'last_category')
    g = {key: full[key][keep] for key in fields}
    if np.unique(g['unit']).tolist() != UNITS or len(g['unit']) != 7680:
        raise ValueError('Expected exactly source96 evaluation units/config40/query2')
    rows_path = OE.OUT/'rows.json'
    rows = [row for row in OE.read(rows_path) if row['split'] == 'evaluation']
    keys = [(r['unit'], r['config'], r['query']) for r in rows]
    if keys != list(zip(g['unit'].tolist(), g['config'].tolist(), g['query'].tolist())):
        raise ValueError('Source row metadata identity differs from geometry')
    g['target_group'] = np.asarray([r['target_group'] for r in rows])
    g['target_off'] = np.asarray([r['target_off'] for r in rows])
    hashes = dict(plan['prior_sha256']); hashes.update(old_provenance['input_sha256'])
    hashes.update({str(p): ME.sha(p) for p in (plan_path, receipt_path, request_path, rows_path)})
    scores = {0: old_scores['M3'][keep]}
    for angle in (-2, -1, 1, 2):
        tag = ('minus' if angle < 0 else 'plus')+str(abs(angle))
        path = OUT/f'frame_scores_{tag}_M3.npz'
        digest = ME.sha(path)
        if receipt.get('output_sha256', {}).get(path.name) != digest:
            raise ValueError('Raw score receipt hash mismatch: '+path.name)
        with np.load(path, allow_pickle=False) as cache:
            raw = cache['logit']
            if not np.array_equal(cache['units'], UNITS) or not np.array_equal(cache['frames'], np.arange(3, 16)):
                raise ValueError('Raw cache axes mismatch: '+path.name)
            if np.asarray(cache['bias_deg']).shape != () or float(cache['bias_deg']) != angle:
                raise ValueError('Raw cache angle mismatch: '+path.name)
        if raw.shape != (96, 40, 13, 2) or not np.isfinite(raw).all():
            raise ValueError('Raw score shape/nonfinite data mismatch: '+path.name)
        scores[angle] = SE.smooth(raw).transpose(0, 1, 3, 2).reshape(-1, 13)
        hashes[str(path)] = digest
    prior_path = OE.OUT/'result.json'
    prior = OE.read(prior_path)
    if prior.get('status') != 'COMPLETE' or prior['cells']['M3']['threshold'] != FIXED['BASE']:
        raise ValueError('Original nominal source M3 result/threshold differs')
    hashes[str(prior_path)] = ME.sha(prior_path)
    return g, scores, prior, dict(input_sha256=hashes, plan=plan, score_producer_receipt=receipt,
        evaluator_sha256=ME.sha(__file__), helper_sha256={p: ME.sha(p) for p in (OE.__file__, SE.__file__, XE.__file__, YE.__file__)})


def clear_groups(g):
    same = g['query'] == g['target_group']
    clear = g['clear_all']
    outside = same & (g['target_off'] >= -.20) & (g['target_off'] < -.10)
    groups = dict(same_height_nominal_outside10_20cm=clear & outside,
        other_height=clear & ~same,
        remaining_same_height=clear & same & ~outside)
    if not np.array_equal(sum(m.astype(int) for m in groups.values()), clear.astype(int)):
        raise ValueError('Descriptive clear strata must partition the sole full-clear budget')
    return groups


def summarize_clear_groups(groups, stopped, ui, boot):
    result, samples = {}, {}
    for name, keep in groups.items():
        den = SE.unit_totals(keep.astype(float), ui, boot.shape[1])
        num = SE.unit_totals((keep & stopped).astype(float), ui, boot.shape[1])
        minutes = den*13*.2/60
        rate, samples[name] = SE.rates(num, minutes, boot)
        result[name] = dict(n=int(keep.sum()), first_stops=int(num.sum()), clear_minutes=float(minutes.sum()),
            false_stops_per_min=rate, contributing_units=int((den > 0).sum()))
    return result, samples


def decide(primary, task_guards, budget_guards):
    lower = primary['paired_unit_ci95'][0]
    decisions = dict(primary_pass=lower is not None and lower > 0,
        taskguards_pass=all(all(checks.values()) for checks in task_guards.values()),
        absolute_budget_pass=all(budget_guards.values()))
    return ('YAW_SOURCE_TRANSFER_SUPPORTED_DEV' if all(decisions.values()) else 'YAW_SOURCE_TRANSFER_NOT_ESTABLISHED_DEV'), decisions


def nominal_parity(cells, prior):
    count = 0
    for category in CATEGORIES:
        new, old = cells['0|BASE']['metrics'][category], prior['cells']['M3']['metrics'][category]
        fields = ['expected_episodes', 'expected_stops', 'contributing_episodes', 'contributing_units']
        fields += ['clear_minutes'] if category == 'clear' else ['expected_timely_stops']
        for name in fields:
            if new[name] != old[name]:
                raise ValueError('Old nominal M3 point count mismatch: '+category+'/'+name)
            count += 1
        rates = ['false_stops_per_min'] if category == 'clear' else ['timely_rate', 'stop_rate', 'median_lead_to_0p5m_s']
        for name in rates:
            if new[name]['value'] != old[name]['value']:
                raise ValueError('Old nominal M3 point metric mismatch: '+category+'/'+name)
            count += 1
    return dict(status='EXACT_NOMINAL_M3_POINT_REPRODUCTION', point_fields_checked=count,
        intervals='New common bootstrap seed; original CIs and old M3-vs-NEAR failed judgment are preserved')


def analyze(g, scores):
    if not np.all(g['split'] == 'evaluation'):
        raise ValueError('This run has no calibration rows')
    units, ui = np.unique(g['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    weights = {c: (g['covered'] & (g['ref_category'] == c)).astype(float) for c in CATEGORIES[:4]}
    weights['clear'] = g['clear_all'].astype(float)
    groups = clear_groups(g)
    cells, samples, flags, comparisons, task_guards, budgets = {}, {}, {}, {}, {}, {}
    subgroup_samples, subgroup_changes = {}, {}
    for bias in BIASES:
        for method in ('BASE', 'ENV'):
            score = scores[bias] if method == 'BASE' else YE.envelope(scores, bias)
            if score.shape != g['frame_ranges'].shape or not np.isfinite(score).all():
                raise ValueError('Score/geometry shape or finite values mismatch')
            key = f'{bias}|{method}'
            cells[key], samples[key], flags[key] = XE.summarize_cell(g, score, FIXED[method], weights, ui, boot)
            cells[key]['clear_subgroups'], subgroup_samples[key] = summarize_clear_groups(groups, flags[key]['stopped'], ui, boot)
        comparison = XE.paired_changes(cells, samples, flags, weights, f'{bias}|ENV', f'{bias}|BASE')
        comparisons[str(bias)] = comparison
        task_guards[str(bias)] = YE.per_bias_guards(comparison, nominal=bias == 0)
        clear_rate = cells[f'{bias}|ENV']['metrics']['clear']['false_stops_per_min']['value']
        budgets[str(bias)] = clear_rate is not None and clear_rate <= 1.
        subgroup_changes[str(bias)] = {}
        for name, keep in groups.items():
            env_key, base_key = f'{bias}|ENV', f'{bias}|BASE'
            av, bv = (cells[k]['clear_subgroups'][name]['false_stops_per_min']['value'] for k in (env_key, base_key))
            ea, ba = flags[env_key]['stopped'], flags[base_key]['stopped']
            subgroup_changes[str(bias)][name] = dict(delta_per_min=av-bv if av is not None and bv is not None else None,
                paired_unit_ci95=SE.interval(subgroup_samples[env_key][name]-subgroup_samples[base_key][name]),
                added_first_stops=int((keep & ea & ~ba).sum()), removed_first_stops=int((keep & ba & ~ea).sum()))
    points = {m: [cells[f'{b}|{m}']['metrics']['contact0-2cm']['timely_rate']['value'] for b in BIASES] for m in FIXED}
    draws = {m: [samples[f'{b}|{m}']['contact0-2cm'] for b in BIASES] for m in FIXED}
    primary = YE.worst_bootstrap(points['BASE'], points['ENV'], draws['BASE'], draws['ENV'])
    verdict, decisions = decide(primary, task_guards, budgets)
    return dict(status='COMPLETE', verdict=verdict, decisions=decisions, primary_worst_bias=primary,
        cells=cells, comparisons_ENV_minus_BASE=comparisons, per_bias_guards=task_guards,
        per_bias_absolute_budget=budgets, fixed_thresholds=FIXED, evaluation_units=units.tolist(),
        clear_subgroup_counts={name: int(keep.sum()) for name, keep in groups.items()},
        clear_subgroup_changes_ENV_minus_BASE=subgroup_changes,
        n=dict(evaluation_units=len(units), query_episodes=len(g['unit']), covered=int(g['covered'].sum()),
            right_censored=int((~g['covered']).sum()), all_clear=int(g['clear_all'].sum()),
            shallow_episodes=int(weights['contact0-2cm'].sum()),
            shallow_contributing_units=int(len(np.unique(g['unit'][weights['contact0-2cm'] > 0])))),
        clear_subgroup_definition=dict(role='DESCRIPTIVE ONLY; all clear remains the sole budget',
            same_height_nominal_outside10_20cm='query == target_group AND -0.20 <= nominal target_off < -0.10 AND clear_all',
            other_height='query != target_group AND clear_all',
            remaining_same_height='remaining same-target-group clear_all rows',
            semantics='target_off is nominal intrusion, positive=inside and negative=outside; strata are not dynamic actual-gap classes'),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, clusters=len(units),
            pairing='common unit multiplicities across BASE/ENV and all3biases', conditional_on='fixed thresholds and nominal geometry'))


def number(value, scale=1.):
    return '--' if value is None else f'{value*scale:.2f}'


def write_report(result):
    p, d, n = result['primary_worst_bias'], result['decisions'], result['n']
    lines = [result['verdict'], '', '# 冻结yaw并集的原source评估单位迁移', '',
        '评估96000–96095，共96单位；不使用本轮数据校准。M3、三修正角度−1/0/+1、BASE和ENV阈值均冻结；每个bias下两方法使用相同偏差输入，名义真值不变。',
        f"三类决定：最坏浅及时增益={d['primary_pass']}；任务护栏={d['taskguards_pass']}；ENV绝对清晰预算={d['absolute_budget_pass']}。无早停，三bias均完整评估。", '',
        f"主差：最坏ENV {number(p['ENV_worst_rate'],100)}% − 最坏BASE {number(p['BASE_worst_rate'],100)}% = {number(p['delta'],100)}pp，95%区间[{number(p['paired_unit_ci95'][0],100)}, {number(p['paired_unit_ci95'][1],100)}]。每次bootstrap内分别重取三bias的min。", '']
    for bias in BIASES:
        lines += [f'## 固定bias={bias:+d}°', '', '| 指标 | BASE | ENV |', '| --- | --- | --- |']
        for category in CATEGORIES:
            lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][f'{bias}|{method}']['metrics'][category], category == 'clear', category == 'pass0-10cm') for method in ('BASE', 'ENV'))+' |')
        lines += ['', f"任务护栏：{result['per_bias_guards'][str(bias)]}；ENV实际清晰≤1次/代理分钟：{result['per_bias_absolute_budget'][str(bias)]}。", '',
            '| 方法/擦碰档 | 已报警条件提前量中位数及95%区间，秒 |', '| --- | --- |']
        for method in FIXED:
            for category in OE.CONTACTS:
                m = result['cells'][f'{bias}|{method}']['metrics'][category]['median_lead_to_0p5m_s']
                lines.append(f"| {method}/{category} | {number(m['value'])} [{number(m['ci95'][0])}, {number(m['ci95'][1])}] |")
        lines += ['', '| 名义清晰子组（只描述） | BASE首停/分钟 | ENV首停/分钟 | ENV−BASE配对差及95%区间 |', '| --- | --- | --- | --- |']
        for group in result['cells'][f'{bias}|BASE']['clear_subgroups']:
            texts = []
            for method in FIXED:
                m = result['cells'][f'{bias}|{method}']['clear_subgroups'][group]
                rate = m['false_stops_per_min']
                texts.append(f"{m['first_stops']}/{m['clear_minutes']:.2f}={number(rate['value'])} [{number(rate['ci95'][0])}, {number(rate['ci95'][1])}]（n={m['n']}）")
            change = result['clear_subgroup_changes_ENV_minus_BASE'][str(bias)][group]
            texts.append(f"{number(change['delta_per_min'])} [{number(change['paired_unit_ci95'][0])}, {number(change['paired_unit_ci95'][1])}]")
            lines.append('| '+group+' | '+' | '.join(texts)+' |')
    lines += ['', '清晰分层按名义target_group和target_off，且都与全13采样时刻clear_all相交。target_off为伸入量，身体外>10cm对应小于−0.10；这些不是随转向计算的真实净距类别。所有清晰行仍是唯一预算分母，子组不单独调阈值。', '',
        f"主浅样本仅{n['shallow_episodes']}条、{n['shallow_contributing_units']}单位。原source M3对NEAR的失败判读保留；本轮是新方法与固定M3对照，不追认旧失败。", '',
        f"原几何有{n['query_episodes']}条查询序列，{n['covered']}覆盖0.9m截止点，{n['right_censored']}右删失；未观察到截止点无报警不算漏报。1000次整单位配对bootstrap，seed2026100220，条件于固定阈值和几何。", '',
        '通过需主95%下界>0，每bias深下降≤2pp、清晰增幅≤0.2次/分钟，名义浅下降≤2pp，且三bias的ENV实际清晰都≤1次/代理分钟。任务护栏与绝对预算分别存，不重校准。', '',
        '已消费Development的有限仿真场景；格点假设含正确抵消分支，不是连续yaw认证。清晰只在13采样时刻检验，2.6秒/查询暴露不是实际用户步行时间。提前量条件于报警，按0.8m/s相对0.5m名义换算，不是人安全停住。', '',
        '源阈值来自95000–95047，本轮96000–96095不参与拟合；旧成功/失败结果不改。完整配对补回丢失、单位分母、区间及输入哈希见result.json。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    path = OUT/'result.json'
    old = YE.existing_complete(path)
    if old is not None:
        print('Existing COMPLETE result verified; no repeated evaluation', flush=True)
        return old
    g, scores, prior, provenance = load_inputs()
    result = analyze(g, scores)
    result['nominal_M3_point_reproduction'] = nominal_parity(result['cells'], prior)
    result['old_source_verdict_retained'] = prior['verdict']
    result['provenance'] = provenance
    YE.verify_hashes(provenance['input_sha256'])
    write_report(result)
    OE.save(path, result)
    print(result['verdict'], json.dumps(result['decisions']), flush=True)
    return result


def check():
    from unittest.mock import patch
    # Positive target_off is intrusion; outside10cm must use the negative sign.
    g = dict(query=np.array([0, 0, 1, 0, 0]), target_group=np.zeros(5, int),
        target_off=np.array([-.11, -.10, .11, .11, -.15]), clear_all=np.array([True, True, True, True, False]))
    groups = clear_groups(g)
    assert groups['same_height_nominal_outside10_20cm'].tolist() == [True, False, False, False, False]
    assert groups['other_height'].tolist() == [False, False, True, False, False]
    good = dict(paired_unit_ci95=[.001, .1])
    tasks = {'0': {'nominal_shallow': True}, '-1': {'deep': True}, '1': {'clear': True}}
    assert decide(good, tasks, {'-1': True, '0': True, '1': True})[0] == 'YAW_SOURCE_TRANSFER_SUPPORTED_DEV'
    assert decide(good, tasks, {'-1': True, '0': False, '1': True})[1]['absolute_budget_pass'] is False
    ref = np.array(['contact0-2cm', 'contact>5cm', 'pass0-10cm', 'clear', 'censored'])
    g.update(split=np.array(['evaluation']*5), unit=np.array([96000, 96000, 96001, 96001, 96001]),
        frames=np.arange(3, 16), covered=np.array([True]*4+[False]), ref_category=ref,
        clear_all=ref == 'clear', frame_ranges=np.tile(np.linspace(1.2, .7, 13), (5, 1)))
    g['frame_ranges'][-1] += .3
    scores = {a: np.zeros((5, 13)) for a in (-2, -1, 0, 1, 2)}
    for values in scores.values():
        values[:3, 1] = 2.
    with patch.object(SE, 'calibrate', side_effect=AssertionError('Source evaluation must not calibrate')):
        result = analyze(g, scores)
    assert result['primary_worst_bias']['delta'] == 0.
    assert result['n']['shallow_episodes'] == 1
    assert 'episode_draw_pairs' not in result['cells']['0|BASE']['metrics']['contact0-2cm']
    prior = dict(cells={'M3': result['cells']['0|BASE']})
    assert nominal_parity(result['cells'], prior)['status'] == 'EXACT_NOMINAL_M3_POINT_REPRODUCTION'
    json.dumps(result, allow_nan=False)
    print('PASS: frozen thresholds/no fitting, nominal intrusion sign, clear partition, independent budget guard and nominal point identity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'evaluate'))
    args = parser.parse_args()
    check() if args.stage == 'check' else evaluate()
