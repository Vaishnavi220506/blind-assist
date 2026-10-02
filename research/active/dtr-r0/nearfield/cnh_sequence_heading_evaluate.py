"""Fixed-score/threshold sequence sensitivity to counterfactual heading truth.

One shared scene heading draw applies to every object, query and observed frame.
TARGET sigma1 alone has a new decision. SOURCE and analytic integration remain
descriptive. No calibration, rendering, inference, trajectory change or tuning.
"""
import argparse
import json

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_sequence_observed_evaluate as OE
import cnh_sequence_transfer_evaluate as TE
import cnh_three_level_sequence as SE

OUT = MC.SS.WORK/'cnh-sequence-heading-20261002'
ARMS = ('NEAR', 'M3')
DOMAINS = ('target', 'source')
SIGMAS = (0., 1.)
CATEGORIES = ('contact0-2cm', 'contact2-5cm', 'contact>5cm', 'pass0-10cm', 'clear')
BOOT_SEEDS = {'target': 2026100215, 'source': 2026100216}
N_BOOT = 1000
DRAWS = 400


def validate_truth(data, geometry):
    n = len(geometry['unit'])
    for field in ('unit', 'config', 'query', 'covered'):
        if not np.array_equal(data[field], geometry[field]):
            raise ValueError('Truth/original geometry row identity mismatch: '+field)
    if not np.array_equal(data['sigmas'], SIGMAS) or not np.array_equal(data['categories'], CATEGORIES):
        raise ValueError('Truth sigma/category axis order mismatch')
    for field in ('weights', 'analytic_weights'):
        weights = data[field]
        if weights.shape != (2, n, 5) or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
            raise ValueError('Invalid probability weight tensor: '+field)
        expected = np.column_stack([(geometry['covered'] & (geometry['ref_category'] == c)).astype(float)
                                    if c != 'clear' else geometry['clear_all'].astype(float) for c in CATEGORIES])
        if not np.array_equal(weights[0], expected):
            raise ValueError(field+': sigma0 truth must exactly reproduce prior nominal geometry')
        if np.any(weights[:, ~geometry['covered'], :4] != 0):
            raise ValueError('Right-censored reference contact/pass cannot enter deadline denominators')
    overlap = data['sampled_clear_reference_contact_overlap']
    if overlap.shape != (2, n) or not np.isfinite(overlap).all() or (overlap < 0).any() or (overlap > 1).any():
        raise ValueError('Invalid sampled-clear/reference-contact overlap diagnostic')
    # Clear is a whole-window probability, not the complement of reference bins.


def load_inputs():
    plan_path, receipt_path = OUT/'PLAN.json', OUT/'truth_receipt.json'
    plan, receipt = OE.read(plan_path), OE.read(receipt_path)
    if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != ME.sha(plan_path):
        raise ValueError('Truth receipt is incomplete or plan hash changed')
    hashes = {str(plan_path): ME.sha(plan_path), str(receipt_path): ME.sha(receipt_path)}
    source_result_path, target_result_path = OE.OUT/'result.json', TE.OUT/'result.json'
    prior = {'source': OE.read(source_result_path), 'target': OE.read(target_result_path)}
    if ME.sha(source_result_path) != plan['source_result_sha256'] or ME.sha(target_result_path) != plan['target_result_sha256']:
        raise ValueError('Prior source/target result differs from the frozen heading plan')
    if any(p.get('status') != 'COMPLETE' for p in prior.values()):
        raise ValueError('Both prior source/target sequence results must be COMPLETE')
    thresholds = {a: float(prior['source']['cells'][a]['threshold']) for a in ARMS}
    if thresholds != {a: float(prior['target']['cells'][a]['threshold']) for a in ARMS}:
        raise ValueError('Source/target were not evaluated at exactly the same source thresholds')
    if 'thresholds' in plan and plan['thresholds'] != thresholds:
        raise ValueError('Frozen heading plan thresholds differ from the source result')
    bundles, loader_provenance = {}, {}
    target_g, target_scores, target_thresholds, target_prov = TE.load_inputs()
    if target_thresholds != thresholds:
        raise ValueError('Transfer loader threshold differs')
    source_g, source_scores, source_prov = OE.load_inputs()
    source_keep = source_g['split'] == 'evaluation'
    fields = ('unit', 'config', 'query', 'split', 'frame_ranges', 'frame_category',
              'ref_category', 'covered', 'clear_all', 'last_category')
    source_g = {k: source_g[k][source_keep] for k in fields}
    source_scores = {a: source_scores[a][source_keep] for a in ARMS}
    for domain, geometry, scores, provenance, result_path in [
            ('target', target_g, target_scores, target_prov, target_result_path),
            ('source', source_g, source_scores, source_prov, source_result_path)]:
        expected_units = list(range(94000, 94096)) if domain == 'target' else list(range(96000, 96096))
        if len(geometry['unit']) != 7680 or np.unique(geometry['unit']).tolist() != expected_units:
            raise ValueError(domain+': unexpected evaluation unit population')
        path = OUT/f'truth_{domain}.npz'
        digest = ME.sha(path)
        if receipt.get('output_sha256', {}).get(domain) != digest:
            raise ValueError('Truth output digest mismatch: '+domain)
        with np.load(path, allow_pickle=False) as cache:
            truth = {k: cache[k] for k in cache.files}
        validate_truth(truth, geometry)
        hashes.update(provenance['input_sha256'])
        hashes[str(path)], hashes[str(result_path)] = digest, ME.sha(result_path)
        bundles[domain] = dict(geometry=geometry, scores=scores, truth=truth)
        loader_provenance[domain] = provenance
    return bundles, thresholds, prior, dict(input_sha256=hashes, plan=plan, truth_receipt=receipt,
        frozen_thresholds=thresholds, prior_verdicts={d: prior[d]['verdict'] for d in DOMAINS},
        loader_provenance=loader_provenance, evaluator_sha256=ME.sha(__file__),
        helper_sha256={p: ME.sha(p) for p in (SE.__file__, OE.__file__, TE.__file__)})


def point_summary(weights, stopped, timely, lead, covered):
    metrics = {}
    for j, category in enumerate(CATEGORIES):
        p = weights[:, j] if category == 'clear' else weights[:, j]*covered
        denominator, num = float(p.sum()), float(np.dot(p, stopped))
        common = dict(expected_episodes=denominator, expected_stops=num,
                      contributing_episodes=int((p > 0).sum()))
        if category == 'clear':
            minutes = denominator*13*.2/60
            metrics[category] = dict(**common, clear_minutes=minutes,
                false_stops_per_min=num/minutes if minutes else None)
        else:
            timely_num = float(np.dot(p, timely))
            metrics[category] = dict(**common, expected_timely_stops=timely_num,
                timely_rate=timely_num/denominator if denominator else None,
                stop_rate=num/denominator if denominator else None,
                median_lead_to_0p5m_s=SE.weighted_median(lead, p*stopped))
    return metrics


def point_differences(cells):
    result = {}
    for c in CATEGORIES[:3]:
        a, b = (cells[arm][c]['timely_rate'] for arm in ('M3', 'NEAR'))
        result[c] = a-b if a is not None and b is not None else None
    a, b = (cells[arm]['clear']['false_stops_per_min'] for arm in ('M3', 'NEAR'))
    result['clear_delta_per_min'] = a-b if a is not None and b is not None else None
    return result


def point_guards(differences):
    shallow, deep, clear = (differences[c] for c in ('contact0-2cm', 'contact>5cm', 'clear_delta_per_min'))
    return dict(shallow_point_positive=shallow is not None and shallow > 0,
        deep_point_decline_le_2pp=deep is not None and deep >= -.02,
        clear_point_increase_le_0p2_per_min=clear is not None and clear <= .2)


def truth_transitions(geometry, truth):
    covered = geometry['covered']
    weights = truth['weights'][1]
    reference_clear = covered.astype(float)-weights[:, :4].sum(1)
    if (reference_clear < -1e-6).any():
        raise ValueError('Reference contact/pass mass exceeds covered population')
    reference_clear = np.clip(reference_clear, 0., 1.)
    destination = {c: weights[:, i] for i, c in enumerate(CATEGORIES[:4])}
    destination['reference-clear'] = reference_clear
    origins = CATEGORIES[:4]+('clear',)
    reference = {}
    for origin in origins:
        keep = covered & (geometry['ref_category'] == origin)
        reference[origin] = dict(nominal_episodes=int(keep.sum()),
            sigma1_expected_episodes={name: float(p[keep].sum()) for name, p in destination.items()})
    window = {}
    for old_clear in (True, False):
        keep = geometry['clear_all'] == old_clear
        window['nominal_clear' if old_clear else 'nominal_nonclear'] = dict(nominal_episodes=int(keep.sum()),
            sigma1_expected_clear=float(weights[keep, 4].sum()),
            sigma1_expected_nonclear=float((1.-weights[keep, 4]).sum()))
    overlap = {str(s): dict(expected_episodes=float(truth['sampled_clear_reference_contact_overlap'][i].sum()),
                contributing_episodes=int((truth['sampled_clear_reference_contact_overlap'][i] > 0).sum()))
               for i, s in enumerate(SIGMAS)}
    return dict(reference_category=reference, sampled_window_clear=window,
        sampled_clear_reference_contact_overlap=overlap,
        note='Reference-clear is covered minus reference contact/pass mass; sampled-window clear is a distinct property and may overlap reference contact')


def verify_sigma0(cells, previous):
    """Compare points/counts only: new bootstrap seeds intentionally change CIs."""
    checks = 0
    for arm in ARMS:
        for category in CATEGORIES:
            now, old = cells[arm+'|0.0']['metrics'][category], previous['cells'][arm]['metrics'][category]
            names = ['expected_episodes', 'expected_stops', 'contributing_episodes', 'contributing_units']
            names += ['clear_minutes'] if category == 'clear' else ['expected_timely_stops']
            for name in names:
                if now[name] != old[name]:
                    raise ValueError(f'Sigma0 point mismatch: {arm}/{category}/{name}: {now[name]} vs {old[name]}')
                checks += 1
            rates = ['false_stops_per_min'] if category == 'clear' else ['timely_rate', 'stop_rate', 'median_lead_to_0p5m_s']
            for name in rates:
                if now[name]['value'] != old[name]['value']:
                    raise ValueError(f'Sigma0 rate/lead mismatch: {arm}/{category}/{name}')
                checks += 1
    return dict(status='EXACT_POINT_REPRODUCTION', point_fields_checked=checks,
                interval_note='New bootstrap seeds; CIs are newly computed and do not replace prior verdicts')


def evaluate_domain(domain, geometry, scores, truth, thresholds):
    units, ui = np.unique(geometry['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEEDS[domain])
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    flags = {a: SE.first_stops(scores[a], thresholds[a], geometry['frame_ranges']) for a in ARMS}
    cells, samples, clear_samples, comparisons, analytic, discrepancy = {}, {}, {}, {}, {}, {}
    for si, sigma in enumerate(SIGMAS):
        w = truth['weights'][si]
        weights = {c: w[:, j] for j, c in enumerate(CATEGORIES)}
        analytic_cells = {}
        for arm in ARMS:
            stopped, timely, lead = flags[arm]
            metrics, sample = SE.summarize(weights, geometry['covered'], stopped, timely, lead, ui, boot)
            for metric in metrics.values():
                metric.pop('episode_draw_pairs', None)
                # Fractional expected counts must not be rounded into independent n.
            key = f'{arm}|{sigma}'
            p = weights['clear']
            minutes = SE.unit_totals(p, ui, len(units))*13*.2/60
            count = SE.unit_totals(p*stopped, ui, len(units))
            _, clear_samples[key] = SE.rates(count, minutes, boot)
            diagnostics = {}
            for j, category in enumerate(CATEGORIES[:4]):
                p = w[:, j]*geometry['covered']
                diagnostics[category] = dict(expected_episodes=float(p.sum()),
                    expected_timely_before_final_drop=float(np.dot(p, timely & (scores[arm][:, -1] < thresholds[arm]))),
                    expected_only_late=float(np.dot(p, stopped & ~timely)))
            cells[key] = dict(threshold=thresholds[arm], metrics=metrics, timing_diagnostics=diagnostics)
            samples[key] = sample
            analytic_cells[arm] = point_summary(truth['analytic_weights'][si], stopped, timely, lead, geometry['covered'])
        a, b = f'M3|{sigma}', f'NEAR|{sigma}'
        comparison = {}
        for j, category in enumerate(CATEGORIES[:3]):
            p = w[:, j]*geometry['covered']
            av, bv = (cells[k]['metrics'][category]['timely_rate']['value'] for k in (a, b))
            mt, nt = flags['M3'][1], flags['NEAR'][1]
            comparison[category] = dict(delta_timely=av-bv if av is not None and bv is not None else None,
                paired_unit_ci95=SE.interval(samples[a][category]-samples[b][category]),
                expected_rescues=float(np.dot(p, mt & ~nt)), expected_losses=float(np.dot(p, nt & ~mt)))
        av, bv = (cells[k]['metrics']['clear']['false_stops_per_min']['value'] for k in (a, b))
        comparison['clear'] = dict(delta_per_min=av-bv if av is not None and bv is not None else None,
            paired_unit_ci95=SE.interval(clear_samples[a]-clear_samples[b]))
        comparisons[str(sigma)] = comparison
        analytic[str(sigma)] = dict(role='INTEGRATION_DIAGNOSTIC_ONLY; never used for verdict or method selection',
            cells=analytic_cells, point_differences=point_differences(analytic_cells))
        mc_point_differences = {c: comparison[c]['delta_timely'] for c in CATEGORIES[:3]}
        mc_point_differences['clear_delta_per_min'] = comparison['clear']['delta_per_min']
        mc_guards = point_guards(mc_point_differences)
        integral_guards = point_guards(analytic[str(sigma)]['point_differences'])
        analytic[str(sigma)]['point_guard_diagnostics'] = dict(monte_carlo=mc_guards, analytic=integral_guards,
            agreement={name: mc_guards[name] == integral_guards[name] for name in mc_guards},
            limitation='Point-sign/point-guard diagnostics only; analytic shallow CI is not computed and cannot replace MC primary CI')
        difference = truth['weights'][si]-truth['analytic_weights'][si]
        discrepancy[str(sigma)] = {c: dict(max_abs_probability_error=float(np.abs(difference[:, j]).max()),
            mc_expected_episodes=float(w[:, j].sum()), analytic_expected_episodes=float(truth['analytic_weights'][si, :, j].sum()))
            for j, c in enumerate(CATEGORIES)}
    censored = ~geometry['covered']
    return dict(cells=cells, comparisons=comparisons, analytic_integration=analytic,
        mc_integration_diagnostics=discrepancy, evaluation_units=units.tolist(),
        truth_transitions=truth_transitions(geometry, truth),
        observed_window=dict(query_episodes=len(geometry['unit']), covered=int(geometry['covered'].sum()),
            right_censored=int(censored.sum()), censored_by_arm={a: dict(already_alarm=int((censored & flags[a][0]).sum()),
                no_alarm_not_missed_deadline=int((censored & ~flags[a][0]).sum())) for a in ARMS}),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEEDS[domain], clusters=len(units),
            paired='same unit resamples across sigma and arm', conditional_on='fixed source thresholds and heading integration draws'),
        denominator='Expected episodes averaged over400 shared scene draws; contributing episodes/units reported separately')


def heading_decision(comparison):
    _, checks = TE.decision(comparison['contact0-2cm']['paired_unit_ci95'],
        comparison['contact>5cm']['delta_timely'], comparison['clear']['delta_per_min'])
    return ('HEADING_SENSITIVITY_SUPPORTED_DEV' if all(checks.values())
            else 'NOT_SUPPORTED_UNDER_HEADING_HYPOTHESIS'), checks


def number(value, scale=1.):
    return '--' if value is None else f'{value*scale:.2f}'


def metric_text(m, category):
    if category == 'clear':
        num, den, rate, scale, unit = m['expected_stops'], m['clear_minutes'], m['false_stops_per_min'], 1., '次/分钟'
    else:
        stop = category == 'pass0-10cm'
        num, den = m['expected_stops' if stop else 'expected_timely_stops'], m['expected_episodes']
        rate, scale, unit = m['stop_rate' if stop else 'timely_rate'], 100., '%'
    return (f"{number(num)}/{number(den)} = {number(rate['value'], scale)}{unit} "
            f"[{number(rate['ci95'][0], scale)}, {number(rate['ci95'][1], scale)}]；"
            f"贡献{m['contributing_episodes']}序列/{m['contributing_units']}单位")


def write_report(result):
    lines = [result['verdict'], '', '# 共享方向偏差假设下的序列敏感性', '',
        '只改变评价真值，分数、模型、观测窗口与源σ0阈值固定。每场景400个共同高斯方向draw用于所有物体、两个查询及13帧；主判读仅TARGET σ1°，SOURCE σ1°仅描述。',
        'TARGET=94000–94095；SOURCE=96000–96095。两域不合并，阈值来自此前source校准；期望分母不四舍五入为独立样本。', '']
    for domain in DOMAINS:
        data = result['domains'][domain]
        for sigma in SIGMAS:
            lines += [f'## {domain.upper()} σ={sigma:g}°', '', '| 指标 | NEAR | M3 |', '| --- | --- | --- |']
            for category in CATEGORIES:
                lines.append('| '+category+' | '+' | '.join(metric_text(data['cells'][f'{a}|{sigma}']['metrics'][category], category) for a in ARMS)+' |')
            c = data['comparisons'][str(sigma)]
            shallow, deep, clear = c['contact0-2cm'], c['contact>5cm'], c['clear']
            lines += ['', f"M3−NEAR浅及时差 {number(shallow['delta_timely'], 100)}pp [{number(shallow['paired_unit_ci95'][0], 100)}, {number(shallow['paired_unit_ci95'][1], 100)}]；深差 {number(deep['delta_timely'], 100)}pp；清晰差 {number(clear['delta_per_min'])}次/分钟 [{number(clear['paired_unit_ci95'][0])}, {number(clear['paired_unit_ci95'][1])}]。", '',
                '| 臂/擦碰档 | 已报警条件提前量中位数及95%区间，秒 |', '| --- | --- |']
            for arm in ARMS:
                for category in CATEGORIES[:3]:
                    m = data['cells'][f'{arm}|{sigma}']['metrics'][category]['median_lead_to_0p5m_s']
                    lines.append(f"| {arm}/{category} | {number(m['value'])} [{number(m['ci95'][0])}, {number(m['ci95'][1])}] |")
        analytic = data['analytic_integration']['1.0']['point_differences']
        guard_agreement = data['analytic_integration']['1.0']['point_guard_diagnostics']['agreement']
        lines += ['', f"{domain.upper()}积分核对（仅描述）：σ1浅及时差{number(analytic['contact0-2cm'], 100)}pp，深差{number(analytic['contact>5cm'], 100)}pp，清晰差{number(analytic['clear_delta_per_min'])}次/分钟。不据此替代Monte Carlo主结果或择优。", '',
            f"积分与Monte Carlo的浅点差正负、深点护栏、清晰点护栏一致性：{guard_agreement}。积分未另算浅差区间，不能替代主区间判据。", '',
            f"σ0点指标、事件数和分母精确复现旧结果，共核对{data['sigma0_reproduction']['point_fields_checked']}字段；新bootstrap种子可改变区间，不改旧判读。"]
    lines += ['', '主判据：TARGET σ1浅及时差95%下界>0、深及时下降≤2pp、清晰首停增幅≤0.2次/代理分钟。Source1和σ0均无新确认判读。', '',
        '区间为各域1000次整单位配对bootstrap，跨σ与两臂共用抽样，条件于固定源阈值。400方向draw是同一场景真值的数值积分，不是400倍独立实验。', '',
        '这是反事实相对方向偏差敏感性：名义0.9m参考pose固定，用xnew=x−z×tan(delta)剪切物理表面顶点，保持y/z及固定z下的横向宽度；并非刚体旋转、实际行走另一条轨迹或实测方向误差。未观测到名义截止点的序列仍右删失，其无报警不计漏报。', '',
        '清晰概率要求13个采样时刻都清晰，不保证帧间连续清晰。每查询序列至多一次首停，计满2.6秒代理暴露；不是现场步行负担。提前量条件于报警，相对0.5m按0.8m/s名义换算。', '',
        '参考点接触/擦身迁移质量表与全观察窗清晰迁移分别保存在truth_transitions。参考点可能位于两采样帧之间，reference contact与sampled-window clear允许重叠，不能当作互斥分区或把后者当作reference-clear。', '',
        '已消费Development；target缺少source的同高度外10–20cm目标，清晰组成不同。旧source0失败及其他旧判读保留。完整期望分母、贡献单位、诊断与输入哈希见result.json。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    path = OUT/'result.json'
    if path.exists():
        old = OE.read(path)
        if old.get('status') != 'COMPLETE':
            raise ValueError('Existing incomplete output requires inspection; no overwrite')
        for name, digest in old['provenance']['input_sha256'].items():
            if ME.sha(name) != digest:
                raise ValueError('Completed input changed: '+name)
        print('Existing COMPLETE result verified; not evaluated again', flush=True)
        return old
    bundles, thresholds, prior, provenance = load_inputs()
    domains = {}
    for domain in DOMAINS:
        data = bundles[domain]
        domains[domain] = evaluate_domain(domain, data['geometry'], data['scores'], data['truth'], thresholds)
        domains[domain]['sigma0_reproduction'] = verify_sigma0(domains[domain]['cells'], prior[domain])
        print('heading analyzed', domain, 'sigma0 point reproduction passed', flush=True)
    verdict, checks = heading_decision(domains['target']['comparisons']['1.0'])
    result = dict(status='COMPLETE', verdict=verdict, primary='TARGET sigma1 only', decision_checks=checks,
        thresholds=thresholds, domains=domains, provenance=provenance,
        truth_sampling=dict(draws_per_scene=DRAWS, seed=2026100214,
            keyed='SeedSequence([2026100214,unit,config]); shared across objects/query/frame',
            meaning='Counterfactual relative heading bias, no changed actual path or measured heading'),
        old_verdicts_retained={d: prior[d]['verdict'] for d in DOMAINS},
        analytic_role='Diagnostic integration only, no alternative verdict or result selection')
    for name, digest in provenance['input_sha256'].items():
        if ME.sha(name) != digest:
            raise ValueError('Input changed during analysis: '+name)
    write_report(result)
    OE.save(path, result)
    print(verdict, flush=True)
    return result


def check():
    from unittest.mock import patch
    ref = np.array(['contact0-2cm', 'contact>5cm', 'pass0-10cm', 'clear', 'censored'])
    geometry = dict(unit=np.array([1, 1, 2, 2, 2]), config=np.arange(5), query=np.zeros(5, int),
        covered=np.array([True]*4+[False]), ref_category=ref, clear_all=ref == 'clear',
        frame_ranges=np.tile(np.linspace(1.2, .7, 13), (5, 1)))
    geometry['frame_ranges'][-1] += .3
    nominal = np.column_stack([(geometry['covered'] & (ref == c)).astype(float) if c != 'clear'
                              else geometry['clear_all'].astype(float) for c in CATEGORIES])
    weights = np.stack([nominal, nominal*.625])
    truth = dict(unit=geometry['unit'], config=geometry['config'], query=geometry['query'],
        covered=geometry['covered'], sigmas=np.array(SIGMAS), categories=np.array(CATEGORIES),
        weights=weights, analytic_weights=weights.copy(), sampled_clear_reference_contact_overlap=np.zeros((2, 5)))
    validate_truth(truth, geometry)
    scores = {a: np.zeros((5, 13)) for a in ARMS}
    scores['NEAR'][1, 1] = 1.
    scores['M3'][:3, 1] = 1.
    with patch.object(SE, 'calibrate', side_effect=AssertionError('Calibration forbidden')):
        result = evaluate_domain('target', geometry, scores, truth, {'NEAR': .3, 'M3': .7})
    m = result['cells']['M3|1.0']['metrics']['contact0-2cm']
    assert m['expected_episodes'] == .625 and m['expected_timely_stops'] == .625
    assert 'n' not in m and 'episode_draw_pairs' not in m
    assert m['contributing_episodes'] == 1 and m['contributing_units'] == 1
    prior = {'cells': {a: result['cells'][f'{a}|0.0'] for a in ARMS}}
    assert verify_sigma0(result['cells'], prior)['status'] == 'EXACT_POINT_REPRODUCTION'
    comparison = dict(result['comparisons']['1.0'])
    comparison['contact0-2cm'] = dict(paired_unit_ci95=[.001, .1])
    comparison['contact>5cm'] = dict(delta_timely=-.02)
    comparison['clear'] = dict(delta_per_min=.2)
    assert heading_decision(comparison)[0] == 'HEADING_SENSITIVITY_SUPPORTED_DEV'
    comparison['contact0-2cm']['paired_unit_ci95'][0] = 0.
    assert heading_decision(comparison)[0] == 'NOT_SUPPORTED_UNDER_HEADING_HYPOTHESIS'
    json.dumps(result, allow_nan=False)
    print('PASS: floating expected denominators, no calibration, sigma0 point identity, paired bootstrap and heading-only guards')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'evaluate'))
    args = parser.parse_args()
    check() if args.stage == 'check' else evaluate()
