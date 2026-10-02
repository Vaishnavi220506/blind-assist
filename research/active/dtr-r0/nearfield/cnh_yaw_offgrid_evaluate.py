"""Off-grid retention of the frozen M3 yaw envelope; no threshold fitting.

New +/-0.5deg observations never contain the exact cancelling0deg branch in
the unchanged {-1,0,+1} hypothesis bank. Prior and new extrema are recomputed
inside one shared whole-unit bootstrap. Failed source budget is not early stop.
"""
import argparse
import json

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_sequence_extrinsic_evaluate as XE
import cnh_sequence_observed_evaluate as OE
import cnh_sequence_transfer_evaluate as TE
import cnh_three_level_sequence as SE
import cnh_yaw_envelope_evaluate as YE

OUT = MC.SS.WORK/'cnh-yaw-offgrid-20261002'
FIXED = {'BASE': .8557642486787612, 'ENV': 1.569529990996084}
OFFGRID = (-.5, .5)
PRIORGRID = (-1, 0, 1)
ANGLES = (-1.5, -.5, .5, 1.5)
CATEGORIES = OE.CONTACTS+('pass0-10cm', 'clear')
BOOT_SEED = 2026100219
N_BOOT = 1000


def read_plan():
    path = OUT/'PLAN.json'
    plan = OE.read(path)
    if plan['fixed_thresholds'] != FIXED or plan['bootstrap']['seed'] != BOOT_SEED:
        raise ValueError('Frozen threshold/bootstrap definition differs')
    if tuple(plan['correction_hypotheses']) != (-1, 0, 1) or tuple(plan['offgrid_biases']) != OFFGRID:
        raise ValueError('Hypothesis/offgrid set changed')
    YE.verify_hashes(plan['prior_sha256'])
    hashes = dict(plan['prior_sha256']); hashes[str(path)] = ME.sha(path)
    return plan, hashes


def load_new_scores(split, units):
    receipt_path, request_path = OUT/f'scores_receipt_{split}.json', OUT/'request.json'
    receipt = OE.read(receipt_path)
    if receipt.get('status') != 'COMPLETE' or receipt.get('split') != split:
        raise ValueError('Required score split is not COMPLETE: '+split)
    if receipt.get('plan_sha256') != ME.sha(OUT/'PLAN.json') or receipt.get('request_sha256') != ME.sha(request_path):
        raise ValueError('Score receipt plan/request mismatch')
    hashes = {str(receipt_path): ME.sha(receipt_path), str(request_path): ME.sha(request_path)}
    scores = {}
    for angle in ANGLES:
        tag = ('minus' if angle < 0 else 'plus')+str(abs(angle)).replace('.', 'p')
        path = OUT/f'frame_scores_{split}_{tag}_M3.npz'
        digest = ME.sha(path)
        if receipt.get('output_sha256', {}).get(path.name) != digest:
            raise ValueError('Score receipt digest mismatch: '+path.name)
        with np.load(path, allow_pickle=False) as data:
            raw = data['logit']
            if not np.array_equal(data['units'], units) or not np.array_equal(data['frames'], np.arange(3, 16)):
                raise ValueError('Score axis identity mismatch: '+path.name)
            if np.asarray(data['bias_deg']).shape != () or float(data['bias_deg']) != angle:
                raise ValueError('Score bias identity mismatch: '+path.name)
        if raw.shape != (len(units), 40, 13, 2) or not np.isfinite(raw).all():
            raise ValueError('Invalid raw score shape/values: '+path.name)
        scores[angle] = SE.smooth(raw).transpose(0, 1, 3, 2).reshape(-1, 13)
        hashes[str(path)] = digest
    return scores, hashes, receipt


def source_budget():
    path = OUT/'source_budget.json'
    previous = YE.existing_complete(path)
    if previous is not None:
        print('Existing source budget audit verified; no threshold fitting', flush=True)
        return previous
    plan, hashes = read_plan()
    g, _, original_provenance = OE.load_inputs()
    cal = g['split'] == 'calib'
    units = np.unique(g['unit'][cal]).tolist()
    if units != plan['calibration_units']:
        raise ValueError('Source calibration unit population differs')
    scores, score_hashes, receipt = load_new_scores('calib', units)
    clear = g['clear_all'][cal]
    per_bias = {}
    for bias in OFFGRID:
        env = YE.rate_at(YE.envelope(scores, bias), clear, FIXED['ENV'])
        base = YE.rate_at(scores[bias], clear, FIXED['BASE'])
        per_bias[str(bias)] = dict(ENV=env, BASE=base, source_budget_pass=env['first_stops_per_min'] <= 1.)
    hashes.update(original_provenance['input_sha256']); hashes.update(score_hashes)
    result = dict(status='COMPLETE', stage='SOURCE_BUDGET_AUDIT_ONLY', fixed_thresholds=FIXED,
        per_bias=per_bias, sourcebudget_pass=all(p['source_budget_pass'] for p in per_bias.values()),
        calibration_units=units, no_early_stop=True,
        note='Verify original source48 empirical clear budget only; no fitting or threshold change; target still evaluated if budget fails',
        provenance=dict(input_sha256=hashes, score_receipt=receipt, evaluator_sha256=ME.sha(__file__)))
    YE.verify_hashes(hashes)
    OE.save(path, result)
    print('Source budget audit COMPLETE; pass=', result['sourcebudget_pass'], '; target still required', flush=True)
    return result


def prior_target():
    g, scores, hashes = YE.target_inputs()
    missing, missing_hashes, _ = YE.new_scores('target', TE.UNITS, (-2, 2))
    scores.update(missing); hashes.update(missing_hashes)
    path = YE.OUT/'result.json'
    prior = OE.read(path)
    if prior.get('status') != 'COMPLETE' or prior['calibration']['base_threshold'] != FIXED['BASE'] or prior['calibration']['envelope_threshold'] != FIXED['ENV']:
        raise ValueError('Completed prior envelope or retained thresholds differ')
    hashes[str(path)] = ME.sha(path)
    if len(g['unit']) != 7680 or int(g['covered'].sum()) != 1200:
        raise ValueError('Prior target population/coverage changed')
    return g, scores, prior, hashes


def extreme_difference(new_points, prior_points, new_samples, prior_samples, which='min'):
    operation = np.min if which == 'min' else np.max
    ns, ps = operation(np.asarray(new_samples), axis=0), operation(np.asarray(prior_samples), axis=0)
    valid = all(x is not None for x in list(new_points)+list(prior_points))
    new = float(operation(new_points)) if valid else None
    prior = float(operation(prior_points)) if valid else None
    return dict(new_extreme=new, prior_extreme=prior, delta=new-prior if valid else None,
        paired_unit_ci95=SE.interval(ns-ps), new_extreme_ci95=SE.interval(ns), prior_extreme_ci95=SE.interval(ps),
        offgrid_extreme_biases=[b for b, v in zip(OFFGRID, new_points) if v == new] if valid else [],
        prior_extreme_biases=[b for b, v in zip(PRIORGRID, prior_points) if v == prior] if valid else [],
        operation=f'{which} over new2 and prior3 separately recomputed inside every common unit resample')


def decide(primary, target_guards, source_pass):
    lower = primary['paired_unit_ci95'][0]
    ni = lower is not None and lower >= -.02
    guards = all(all(checks.values()) for checks in target_guards.values())
    decisions = dict(timely_NI_pass=ni, targetguards_pass=guards, sourcebudget_pass=bool(source_pass))
    return ('OFFGRID_RETENTION_SUPPORTED_DEV' if all(decisions.values()) else 'OFFGRID_RETENTION_NOT_ESTABLISHED_DEV'), decisions


def nominal_reference_parity(cells, prior):
    count = 0
    for bias in PRIORGRID:
        for category in CATEGORIES:
            new, old = cells[f'prior|{bias}|ENV']['metrics'][category], prior['cells'][f'{bias}|ENV']['metrics'][category]
            fields = ['expected_episodes', 'expected_stops', 'contributing_episodes', 'contributing_units']
            fields += ['clear_minutes'] if category == 'clear' else ['expected_timely_stops']
            for field in fields:
                if new[field] != old[field]:
                    raise ValueError(f'Prior grid count mismatch: {bias}/{category}/{field}')
                count += 1
            names = ['false_stops_per_min'] if category == 'clear' else ['timely_rate', 'stop_rate', 'median_lead_to_0p5m_s']
            for field in names:
                if new[field]['value'] != old[field]['value']:
                    raise ValueError(f'Prior grid point metric mismatch: {bias}/{category}/{field}')
                count += 1
    return dict(status='EXACT_PRIOR_GRID_POINT_REPRODUCTION', point_fields_checked=count,
        interval_note='New common bootstrap seed for the new comparison; old CIs and verdict stay unchanged')


def analyze(g, prior_scores, offgrid_scores):
    if not np.all(g['split'] == 'evaluation'):
        raise ValueError('Target must contain evaluation rows only')
    units, ui = np.unique(g['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    weights = {c: (g['covered'] & (g['ref_category'] == c)).astype(float) for c in CATEGORIES[:4]}
    weights['clear'] = g['clear_all'].astype(float)
    cells, samples, flags = {}, {}, {}
    for grid, biases, scores in [('prior', PRIORGRID, prior_scores), ('offgrid', OFFGRID, offgrid_scores)]:
        for bias in biases:
            for method in (('ENV',) if grid == 'prior' else ('BASE', 'ENV')):
                score = YE.envelope(scores, bias) if method == 'ENV' else scores[bias]
                if score.shape != g['frame_ranges'].shape or not np.isfinite(score).all():
                    raise ValueError('Scores do not align with original geometry')
                key = f'{grid}|{bias}|{method}'
                cells[key], samples[key], flags[key] = XE.summarize_cell(g, score, FIXED[method], weights, ui, boot)
    target_comparisons, target_guards = {}, {}
    for bias in OFFGRID:
        comparison = XE.paired_changes(cells, samples, flags, weights, f'offgrid|{bias}|ENV', f'offgrid|{bias}|BASE')
        target_comparisons[str(bias)] = comparison
        target_guards[str(bias)] = YE.per_bias_guards(comparison, nominal=False)
    def retained(category, which):
        metric = 'false_stops_per_min' if category == 'clear' else 'timely_rate'
        nk = [f'offgrid|{b}|ENV' for b in OFFGRID]
        pk = [f'prior|{b}|ENV' for b in PRIORGRID]
        points = lambda keys: [cells[k]['metrics'][category][metric]['value'] for k in keys]
        draws = lambda keys: [samples[k][category] for k in keys]
        return extreme_difference(points(nk), points(pk), draws(nk), draws(pk), which)
    primary = retained('contact0-2cm', 'min')
    descriptions = dict(deep_worst_timely=retained('contact>5cm', 'min'), clear_worst_burden=retained('clear', 'max'),
        role='Descriptive own-prior ENV comparisons only; not extra decision gates')
    return dict(cells=cells, primary_timely_retention=primary, own_prior_descriptive=descriptions,
        offgrid_ENV_minus_BASE=target_comparisons, target_guards=target_guards,
        fixed_thresholds=FIXED, evaluation_units=units.tolist(),
        n=dict(evaluation_units=len(units), query_episodes=len(g['unit']), covered=int(g['covered'].sum()),
            right_censored=int((~g['covered']).sum()), clear_all=int(g['clear_all'].sum()),
            shallow_episodes=int(weights['contact0-2cm'].sum()), shallow_contributing_units=int(len(np.unique(g['unit'][weights['contact0-2cm'] > 0])))),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, cluster='whole target units',
            pairing='same multiplicities across every new and prior bias/method', conditional_on='fixed thresholds and nominal geometry'))


def number(value, scale=1.):
    return '--' if value is None else f'{scale*value:.2f}'


def write_report(result):
    d, p = result['decisions'], result['primary_timely_retention']
    lines = [result['verdict'], '', '# 固定yaw并集报警的非格点保持检查', '',
        f"及时率非劣性：{d['timely_NI_pass']}；目标护栏：{d['targetguards_pass']}；source预算：{d['sourcebudget_pass']}。三者分别保留，全部通过才判整体保持。", '',
        f"BASE阈值固定{FIXED['BASE']!r}，ENV阈值固定{FIXED['ENV']!r}，不重新校准。纠正假设仍为−1/0/+1，新bias为−0.5/+0.5，四个绝对输入角度均不包含0°精确抵消分支。", '',
        f"主D：新两点ENV最坏浅及时{number(p['new_extreme'],100)}% − 旧三点ENV最坏浅及时{number(p['prior_extreme'],100)}% = {number(p['delta'],100)}pp；95%区间[{number(p['paired_unit_ci95'][0],100)}, {number(p['paired_unit_ci95'][1],100)}]，跑前非劣性要求下界≥−2pp。每次bootstrap内部同时重新取两个min，旧46/51不是无误差常数。", '']
    for bias in OFFGRID:
        lines += [f'## target bias={bias:+g}°', '', '| 指标 | BASE | ENV |', '| --- | --- | --- |']
        for category in CATEGORIES:
            lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][f'offgrid|{bias}|{m}']['metrics'][category], category == 'clear', category == 'pass0-10cm') for m in ('BASE', 'ENV'))+' |')
        c = result['offgrid_ENV_minus_BASE'][str(bias)]
        lines += ['', f"ENV−BASE深差{number(c['contact>5cm']['delta_timely'],100)}pp，清晰差{number(c['clear']['delta_per_min'])}次/代理分钟；该bias护栏{result['target_guards'][str(bias)]}。浅ENV−BASE差{number(c['contact0-2cm']['delta_timely'],100)}pp仅作描述，不要求ENV在半度显著优于BASE。", '',
            '| 方法/擦碰档 | 已报警条件提前量中位数及95%区间，秒 |', '| --- | --- |']
        for method in ('BASE', 'ENV'):
            for category in OE.CONTACTS:
                m = result['cells'][f'offgrid|{bias}|{method}']['metrics'][category]['median_lead_to_0p5m_s']
                lines.append(f"| {method}/{category} | {number(m['value'])} [{number(m['ci95'][0])}, {number(m['ci95'][1])}] |")
    lines += ['', '## 旧ENV三点复现', '', '| prior bias | 浅及时 | 深及时 | 清晰首停 |', '| --- | --- | --- | --- |']
    for bias in PRIORGRID:
        m = result['cells'][f'prior|{bias}|ENV']['metrics']
        lines.append(f"| {bias:+d} | {OE.rate_text(m['contact0-2cm'])} | {OE.rate_text(m['contact>5cm'])} | {OE.rate_text(m['clear'], clear=True)} |")
    lines += ['', '## source48预算核验（无阈值拟合）', '', '| offgrid bias | BASE首停/分钟 | ENV首停/分钟 | ENV≤1/min |', '| --- | --- | --- | --- |']
    for bias in OFFGRID:
        b = result['source_budget']['per_bias'][str(bias)]
        display = lambda m: f"{m['first_stops']}/{m['clear_minutes']:.2f}={m['first_stops_per_min']:.3f}"
        lines.append(f"| {bias:+g} | {display(b['BASE'])} | {display(b['ENV'])} | {b['source_budget_pass']} |")
    for name, label, scale in [('deep_worst_timely', 'ENV自身最坏深及时率变化', 100.), ('clear_worst_burden', 'ENV自身最坏清晰率变化', 1.)]:
        m = result['own_prior_descriptive'][name]
        lines += ['', f"{label}（仅描述）：{number(m['delta'],scale)}，95%区间[{number(m['paired_unit_ci95'][0],scale)}, {number(m['paired_unit_ci95'][1],scale)}]；分别在每次bootstrap内取深率min或清晰率max。单位分别为pp和次/代理分钟。"]
    n = result['n']
    lines += ['', f"浅擦碰仅{n['shallow_episodes']}条、来自{n['shallow_contributing_units']}单位；2pp约一条事件。保持尚未建立不等于证明方法无用，不能事后放宽容限。两个新增有限点通过也不证明连续区间或真实误差分布。", '',
        f"名义几何不变，{n['query_episodes']}查询中{n['covered']}覆盖0.9m截止点、{n['right_censored']}右删失；未到截止点无报警不算漏报。全13采样时刻清晰不保证帧间连续清晰，2.6秒/查询序列含首停后暴露，不是实际用户步行分钟。", '',
        '1000次整目标单位共同bootstrap，seed2026100219，条件于冻结阈值和几何。提前量相对0.5m按0.8m/s换算且条件于已报警，不是实际人停住。source曾用于先前校准，本轮仍为已消费Development，不是新独立确认。', '',
        '输入是已有z1=r/sqrt(v)的传感器到身体映射输出侧yaw扰动，不是新光子数、物理重装或真实安全性证据。source预算不通过也完成target，不校准救结果，不改旧成功/失败判读。完整补回丢失、分母和哈希见result.json。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    path = OUT/'result.json'
    previous = YE.existing_complete(path)
    if previous is not None:
        print('Existing COMPLETE offgrid result verified; no repeated evaluation', flush=True)
        return previous
    plan, hashes = read_plan()
    budget = source_budget()  # Deliberately no gate/return when the source budget fails.
    g, old_scores, old_result, old_hashes = prior_target()
    new_scores, new_hashes, receipt = load_new_scores('target', TE.UNITS)
    result = analyze(g, old_scores, new_scores)
    result['prior_point_reproduction'] = nominal_reference_parity(result['cells'], old_result)
    verdict, decisions = decide(result['primary_timely_retention'], result['target_guards'], budget['sourcebudget_pass'])
    hashes.update(old_hashes); hashes.update(new_hashes); hashes.update(budget['provenance']['input_sha256'])
    hashes[str(OUT/'source_budget.json')] = ME.sha(OUT/'source_budget.json')
    result.update(status='COMPLETE', verdict=verdict, decisions=decisions, source_budget=budget,
        old_verdict_retained=old_result['verdict'], provenance=dict(input_sha256=hashes, plan=plan,
            target_score_receipt=receipt, evaluator_sha256=ME.sha(__file__),
            helper_sha256={p: ME.sha(p) for p in (SE.__file__, OE.__file__, TE.__file__, XE.__file__, YE.__file__)}))
    YE.verify_hashes(hashes)
    write_report(result)
    OE.save(path, result)
    print(verdict, json.dumps(decisions), flush=True)
    return result


def check():
    from unittest.mock import patch
    # Each draw has a different worst reference/offgrid point; never fix the old46/51.
    old = np.array([[.2, .9], [.5, .5], [.9, .2]])
    new = np.array([[.3, .8], [.8, .3]])
    d = extreme_difference([.3, .8], [.2, .5, .9], new, old)
    assert np.allclose(d['paired_unit_ci95'], [.1, .1])
    guard = {'-0.5': {'deep': True, 'clear': True}, '0.5': {'deep': True, 'clear': True}}
    assert decide({'paired_unit_ci95': [-.02, .1]}, guard, True)[0] == 'OFFGRID_RETENTION_SUPPORTED_DEV'
    assert decide({'paired_unit_ci95': [-.020001, .1]}, guard, True)[1]['timely_NI_pass'] is False
    assert decide({'paired_unit_ci95': [0., .1]}, guard, False)[1] == dict(timely_NI_pass=True, targetguards_pass=True, sourcebudget_pass=False)
    ref = np.array(['contact0-2cm', 'contact>5cm', 'pass0-10cm', 'clear', 'censored'])
    g = dict(split=np.array(['evaluation']*5), unit=np.array([1, 1, 2, 2, 2]), frames=np.arange(3, 16),
        covered=np.array([True]*4+[False]), clear_all=ref == 'clear', ref_category=ref,
        frame_ranges=np.tile(np.linspace(1.2, .7, 13), (5, 1)))
    g['frame_ranges'][-1] += .3
    scores = {angle: np.zeros((5, 13)) for angle in (-2, -1, 0, 1, 2)}
    half = {angle: np.zeros((5, 13)) for angle in ANGLES}
    for data in (scores, half):
        for value in data.values():
            value[:3, 1] = 2.
    with patch.object(SE, 'calibrate', side_effect=AssertionError('Offgrid threshold fitting forbidden')):
        result = analyze(g, scores, half)
    assert result['primary_timely_retention']['delta'] == 0.
    assert result['n']['shallow_episodes'] == 1
    assert 'episode_draw_pairs' not in result['cells']['offgrid|0.5|ENV']['metrics']['contact0-2cm']
    old_result = dict(cells={f'{bias}|ENV': result['cells'][f'prior|{bias}|ENV'] for bias in PRIORGRID})
    assert nominal_reference_parity(result['cells'], old_result)['status'] == 'EXACT_PRIOR_GRID_POINT_REPRODUCTION'
    json.dumps(result, allow_nan=False)
    print('PASS: paired new/prior extrema per draw, NI boundary, separate source guard, no calibration, fixed finite bank and prior point parity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'source-budget', 'evaluate'))
    args = parser.parse_args()
    {'check': check, 'source-budget': source_budget, 'evaluate': evaluate}[args.stage]()
