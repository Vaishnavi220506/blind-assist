"""Fixed nominal truth/thresholds, paired sequence evaluation of +/-1deg input yaw.

Each nonzero sign is a systematic bias shared by the whole target dataset and
both arms. Both signs must support the relative M3 gain. Within-arm degradation
from nominal remains descriptive and is reported separately.
"""
import argparse
import json

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_sequence_observed_evaluate as OE
import cnh_sequence_transfer_evaluate as TE
import cnh_three_level_sequence as SE

OUT = MC.SS.WORK/'cnh-sequence-extrinsic-yaw-20261002'
ARMS = ('NEAR', 'M3')
SIGNS = {'nominal0': 0., 'minus1': -1., 'plus1': 1.}
BOOT_SEED = 2026100217
N_BOOT = 1000
CATEGORIES = OE.CONTACTS+('pass0-10cm', 'clear')


def load_inputs():
    plan_path, receipt_path = OUT/'PLAN.json', OUT/'scores_receipt.json'
    plan, receipt = OE.read(plan_path), OE.read(receipt_path)
    if receipt.get('status') != 'COMPLETE':
        raise ValueError('Perturbed score producer is not COMPLETE')
    if receipt.get('plan_sha256') != ME.sha(plan_path):
        raise ValueError('Perturbed score receipt is not bound to the current frozen plan')
    geometry, nominal_scores, thresholds, base_provenance = TE.load_inputs()
    old_path = TE.OUT/'result.json'
    prior = OE.read(old_path)
    if prior.get('status') != 'COMPLETE':
        raise ValueError('Prior transfer result must be COMPLETE')
    if {a: prior['cells'][a]['threshold'] for a in ARMS} != thresholds:
        raise ValueError('Prior transfer thresholds changed')
    if 'thresholds' in plan and plan['thresholds'] != thresholds:
        raise ValueError('Perturbation plan source thresholds differ')
    if 'target_result_sha256' in plan and plan['target_result_sha256'] != ME.sha(old_path):
        raise ValueError('Prior transfer result differs from the perturbation plan')
    if 'source_result_sha256' in plan and plan['source_result_sha256'] != base_provenance['source_result_sha256']:
        raise ValueError('Original source threshold result differs from the perturbation plan')
    hashes = dict(base_provenance['input_sha256'])
    hashes.update({str(plan_path): ME.sha(plan_path), str(receipt_path): ME.sha(receipt_path), str(old_path): ME.sha(old_path)})
    scores = {'nominal0': nominal_scores}
    for sign in ('minus1', 'plus1'):
        scores[sign] = {}
        for arm in ARMS:
            path = OUT/f'frame_scores_{sign}_{arm}.npz'
            digest = ME.sha(path)
            if receipt.get('output_sha256', {}).get(path.name) != digest:
                raise ValueError('Perturbed cache digest mismatch: '+path.name)
            with np.load(path, allow_pickle=False) as data:
                if not np.array_equal(data['units'], TE.UNITS) or not np.array_equal(data['frames'], np.arange(3, 16)):
                    raise ValueError('Perturbed cache axis identity mismatch: '+path.name)
                if np.asarray(data['bias_deg']).shape != () or float(data['bias_deg']) != SIGNS[sign]:
                    raise ValueError('Perturbed cache sign mismatch: '+path.name)
                logit = data['logit']
            if logit.shape != (96, 40, 13, 2) or not np.isfinite(logit).all():
                raise ValueError('Invalid perturbed raw logit shape/values: '+path.name)
            scores[sign][arm] = SE.smooth(logit).transpose(0, 1, 3, 2).reshape(-1, 13)
            hashes[str(path)] = digest
    # The geometry is never recomputed from the biased model inputs.
    if len(geometry['unit']) != 7680 or int(geometry['covered'].sum()) != 1200:
        raise ValueError('Expected the fixed prior target population:7680 rows,1200 covered,6480 censored')
    return geometry, scores, thresholds, prior, dict(input_sha256=hashes, plan=plan,
        frozen_thresholds=thresholds, source_result_sha256=base_provenance['source_result_sha256'],
        target_result_sha256=ME.sha(old_path), nominal_input_provenance=base_provenance,
        score_producer_receipt=receipt, evaluator_sha256=ME.sha(__file__),
        helper_sha256={p: ME.sha(p) for p in (SE.__file__, OE.__file__, TE.__file__)})


def nominal_parity(cells, prior):
    count = 0
    for arm in ARMS:
        for category in CATEGORIES:
            new, old = cells[f'nominal0|{arm}']['metrics'][category], prior['cells'][arm]['metrics'][category]
            fields = ['expected_episodes', 'expected_stops', 'contributing_episodes', 'contributing_units']
            fields += ['clear_minutes'] if category == 'clear' else ['expected_timely_stops']
            for field in fields:
                if new[field] != old[field]:
                    raise ValueError(f'Nominal point reproduction mismatch: {arm}/{category}/{field}')
                count += 1
            rates = ['false_stops_per_min'] if category == 'clear' else ['timely_rate', 'stop_rate', 'median_lead_to_0p5m_s']
            for field in rates:
                if new[field]['value'] != old[field]['value']:
                    raise ValueError(f'Nominal point reproduction mismatch: {arm}/{category}/{field}')
                count += 1
    return dict(status='EXACT_POINT_REPRODUCTION', point_fields_checked=count,
                note='New paired bootstrap seed may change CIs; old transfer verdict is not rewritten')


def summarize_cell(g, score, threshold, weights, ui, boot):
    stopped, timely, lead = SE.first_stops(score, threshold, g['frame_ranges'])
    metrics, samples = SE.summarize(weights, g['covered'], stopped, timely, lead, ui, boot)
    for m in metrics.values():
        m.pop('episode_draw_pairs', None)
        m['n'] = int(round(m['expected_episodes']))
        m['stops'] = int(round(m['expected_stops']))
        if 'expected_timely_stops' in m:
            m['timely_stops'] = int(round(m['expected_timely_stops']))
    clear_p = weights['clear']
    minutes = SE.unit_totals(clear_p, ui, boot.shape[1])*13*.2/60
    stops = SE.unit_totals(clear_p*stopped, ui, boot.shape[1])
    _, samples['clear'] = SE.rates(stops, minutes, boot)
    diagnostics = {category: OE.extra_counts(weights[category] > 0, stopped, timely, score, threshold)
                   for category in CATEGORIES[:4]}
    censored = ~g['covered']
    cell = dict(threshold=threshold, metrics=metrics, timing_diagnostics=diagnostics,
        censored=dict(n=int(censored.sum()), already_alarm=int((censored & stopped).sum()),
            no_alarm_right_censored=int((censored & ~stopped).sum()),
            interpretation='No alarm before an unobserved deadline is not a missed deadline'))
    return cell, samples, dict(stopped=stopped, timely=timely)


def paired_changes(cells, samples, flags, weights, left, right):
    result = {}
    for category in OE.CONTACTS:
        a, b = (cells[k]['metrics'][category]['timely_rate']['value'] for k in (left, right))
        keep = weights[category] > 0
        la, ra = flags[left]['timely'], flags[right]['timely']
        result[category] = dict(delta_timely=a-b if a is not None and b is not None else None,
            paired_unit_ci95=SE.interval(samples[left][category]-samples[right][category]), n=int(keep.sum()),
            rescues=int((keep & la & ~ra).sum()), losses=int((keep & ra & ~la).sum()),
            both_timely=int((keep & la & ra).sum()), neither_timely=int((keep & ~la & ~ra).sum()))
    a, b = (cells[k]['metrics']['clear']['false_stops_per_min']['value'] for k in (left, right))
    result['clear'] = dict(delta_per_min=a-b if a is not None and b is not None else None,
        paired_unit_ci95=SE.interval(samples[left]['clear']-samples[right]['clear']))
    return result


def joint_decision(comparisons):
    per_sign = {}
    for sign in ('minus1', 'plus1'):
        c = comparisons[sign]
        _, checks = TE.decision(c['contact0-2cm']['paired_unit_ci95'], c['contact>5cm']['delta_timely'], c['clear']['delta_per_min'])
        per_sign[sign] = dict(supported=all(checks.values()), checks=checks)
    verdict = ('M3_RELATIVE_GAIN_SURVIVES_PLUS_MINUS_1DEG' if all(v['supported'] for v in per_sign.values())
               else 'NOT_SUPPORTED_UNDER_YAW_BIAS')
    return verdict, per_sign


def evaluate_arrays(g, scores, thresholds):
    row_fields = ('unit', 'frame_ranges', 'ref_category', 'clear_all', 'covered', 'last_category', 'frame_category')
    n = len(g['unit'])
    if any(g[k].shape[0] != n for k in row_fields):
        raise ValueError('Geometry row fields do not align')
    if not np.all(g['split'] == 'evaluation'):
        raise ValueError('No target calibration rows are allowed')
    units, ui = np.unique(g['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    weights = {c: (g['covered'] & (g['ref_category'] == c)).astype(float) for c in CATEGORIES[:4]}
    weights['clear'] = g['clear_all'].astype(float)
    cells, samples, flags = {}, {}, {}
    for sign in SIGNS:
        for arm in ARMS:
            score = scores[sign][arm]
            if score.shape != g['frame_ranges'].shape or not np.isfinite(score).all():
                raise ValueError('Score/geometry shape or finite-value mismatch')
            key = f'{sign}|{arm}'
            cells[key], samples[key], flags[key] = summarize_cell(g, score, thresholds[arm], weights, ui, boot)
            cells[key]['input_yaw_bias_deg'] = SIGNS[sign]
    relative = {sign: paired_changes(cells, samples, flags, weights, f'{sign}|M3', f'{sign}|NEAR') for sign in SIGNS}
    absolute = {sign: {arm: paired_changes(cells, samples, flags, weights, f'{sign}|{arm}', f'nominal0|{arm}')
                      for arm in ARMS} for sign in ('minus1', 'plus1')}
    verdict, checks = joint_decision(relative)
    transitions = (g['frame_category'] != g['frame_category'][:, :1]).any(1)
    return dict(status='COMPLETE', verdict=verdict, per_sign_decisions=checks, cells=cells,
        relative_M3_minus_NEAR=relative, within_arm_biased_minus_nominal=absolute,
        thresholds=thresholds, evaluation_units=units.tolist(),
        n=dict(evaluation_units=len(units), query_episodes=n, covered=int(g['covered'].sum()),
            right_censored=int((~g['covered']).sum()), clear_all=int(g['clear_all'].sum()),
            category_transitions=int(transitions.sum()),
            reference_categories={c: int((g['ref_category'] == c).sum()) for c in OE.CATEGORIES}),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, cluster='whole evaluation units',
            pairing='one common bootstrap matrix across all signs and arms', conditional_on='fixed nominal truth, source thresholds and cached input perturbation'),
        interpretation='Both signs must support relative M3 gain; this does not claim either arm is absolutely unaffected by bias',
        limits=['One systematic yaw bias per complete dataset and both arms; not independent random per-scene noise or measured calibration error',
            'Nominal all-object reference truth, coverage, geometry and target-front deadline remain fixed across all score arms',
            'Source48 original pooled thresholds unchanged; no target or sign calibration, no sign selection',
            'Target consumed Development94000-94095; no same-height target10-20cm outside support, no hardware or protected confirmation',
            'Clear at13 sampled instants only; each query exposure2.6s including after first stop; not continuous real walking burden',
            'Conditional lead to0.5m divided by0.8m/s is a nominal metric, not human reaction or actual stopping time',
            'No unobserved-deadline misses; all old successful/failed result judgments remain unchanged'])


def fmt(value, scale=1.):
    return '--' if value is None else f'{scale*value:+.2f}'


def write_report(result):
    n = result['n']
    lines = [result['verdict'], '', '# 固定真值和阈值的系统性外参偏航输入扰动', '',
        f"目标96单位、{n['query_episodes']}条查询序列；固定覆盖截止点{n['covered']}条，右删失{n['right_censored']}条。每个符号是全数据集共同的固定偏差，同一符号下NEAR/M3都接收同方向扰动输入。", '',
        '仅比较缓存输入扰动后的分数。名义全物体真值、参考pose、0.9m截止点和原源48单位阈值均不变。正负两侧相对增益须同时通过，不挑符号，不声称绝对性能不退化。', '']
    lines += ['扰动限定为传感器到身体映射输出侧yaw对齐偏差（受限orientation-extrinsic）：左乘固定B同时转动历史变换的旋转与平移。复用每帧z1的r/sqrt(v)表示与原运动估计；z1不是原始光子数。这不是硬件实际转动、物理重装传感器、新光线观测或另走一条身体路径。', '']
    for sign, degrees in SIGNS.items():
        lines += [f'## {sign}（{degrees:+g}°）', '', '| 指标 | NEAR | M3 |', '| --- | --- | --- |']
        for category in CATEGORIES:
            lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][f'{sign}|{a}']['metrics'][category], category == 'clear', category == 'pass0-10cm') for a in ARMS)+' |')
        c = result['relative_M3_minus_NEAR'][sign]
        a, d, f = c['contact0-2cm'], c['contact>5cm'], c['clear']
        lines += ['', f"M3−NEAR浅及时差{fmt(a['delta_timely'], 100)}pp [{fmt(a['paired_unit_ci95'][0], 100)}, {fmt(a['paired_unit_ci95'][1], 100)}]；深差{fmt(d['delta_timely'], 100)}pp；清晰差{fmt(f['delta_per_min'])}次/代理分钟 [{fmt(f['paired_unit_ci95'][0])}, {fmt(f['paired_unit_ci95'][1])}]。", '',
            '| 臂/擦碰 | 已报警条件提前量中位数及95%区间，秒 | 及时首报但末帧低于阈值 / 仅迟报 |', '| --- | --- | --- |']
        for arm in ARMS:
            for category in OE.CONTACTS:
                m = result['cells'][f'{sign}|{arm}']['metrics'][category]['median_lead_to_0p5m_s']
                t = result['cells'][f'{sign}|{arm}']['timing_diagnostics'][category]
                lines.append(f"| {arm}/{category} | {fmt(m['value'])} [{fmt(m['ci95'][0])}, {fmt(m['ci95'][1])}] | {t['timely_before_final_drop']} / {t['only_late']}（n={t['n']}） |")
        for arm in ARMS:
            c = result['cells'][f'{sign}|{arm}']['censored']
            lines += ['', f"{arm}右删失{c['n']}中已报{c['already_alarm']}、未报{c['no_alarm_right_censored']}；未报不计漏报。"]
    lines += ['', '## 同一臂相对自身0°的变化（诊断，不是另一工作点）', '',
        '| 符号/臂 | 浅及时差及95%区间 | 深及时差及95%区间 | 清晰首停差及95%区间，次/分钟 |', '| --- | --- | --- | --- |']
    for sign in ('minus1', 'plus1'):
        for arm in ARMS:
            c = result['within_arm_biased_minus_nominal'][sign][arm]
            a, d, f = c['contact0-2cm'], c['contact>5cm'], c['clear']
            cell = lambda m, name, scale: f"{fmt(m[name], scale)} [{fmt(m['paired_unit_ci95'][0], scale)}, {fmt(m['paired_unit_ci95'][1], scale)}]"
            lines.append(f"| {sign}/{arm} | {cell(a, 'delta_timely', 100)}pp | {cell(d, 'delta_timely', 100)}pp | {cell(f, 'delta_per_min', 1.)} |")
    lines += ['', '每个符号须同时满足：M3−NEAR浅及时差95%下界>0、深下降≤2pp、清晰首停增幅≤0.2次/代理分钟；两符号全部通过才判相对增益保留。该护栏不保证各臂实际负担≤1次/分钟。', '',
        '1000次整单位bootstrap，seed=2026100217，共同抽样用于所有臂与符号；条件于固定真值、阈值和输入。0°点分母、事件数和率精确复现旧transfer，新CI可因种子改变，不改旧判读。', '',
        '全13采样时刻清晰不保证帧间连续清晰；每查询序列最多一次首停，计满13×0.2=2.6秒代理暴露，不是用户级实际步行误停。提前量条件于已报警，相对0.5m按0.8m/s换算。', '',
        '本轮为已消费Development固定输入偏差敏感性；不是实测外参误差、真实安全性或绝对不退化证明。完整源阈值、输入/几何哈希、各档配对补回丢失及区间见result.json。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    path = OUT/'result.json'
    if path.exists():
        old = OE.read(path)
        if old.get('status') != 'COMPLETE':
            raise ValueError('Existing incomplete evidence requires inspection; no overwrite')
        for name, digest in old['provenance']['input_sha256'].items():
            if ME.sha(name) != digest:
                raise ValueError('Completed input changed: '+name)
        print('Existing COMPLETE result verified; no repeated evaluation', flush=True)
        return old
    geometry, scores, thresholds, prior, provenance = load_inputs()
    result = evaluate_arrays(geometry, scores, thresholds)
    result['nominal_point_reproduction'] = nominal_parity(result['cells'], prior)
    result['old_transfer_verdict_retained'] = prior['verdict']
    result['provenance'] = provenance
    for name, digest in provenance['input_sha256'].items():
        if ME.sha(name) != digest:
            raise ValueError('Input changed during evaluation: '+name)
    write_report(result)
    OE.save(path, result)
    print(result['verdict'], flush=True)
    return result


def check():
    from unittest.mock import patch
    ref = np.array(['contact0-2cm', 'contact>5cm', 'pass0-10cm', 'clear', 'censored'])
    g = dict(split=np.array(['evaluation']*5), unit=np.array([94000, 94000, 94001, 94001, 94001]),
        frames=np.arange(3, 16), ref_category=ref, last_category=np.where(ref == 'censored', 'contact0-2cm', ref),
        covered=np.array([True]*4+[False]), clear_all=ref == 'clear',
        frame_ranges=np.tile(np.linspace(1.2, .7, 13), (5, 1)))
    g['frame_ranges'][-1] += .3
    g['frame_category'] = np.repeat(g['last_category'][:, None], 13, axis=1)
    scores = {sign: {a: np.zeros((5, 13)) for a in ARMS} for sign in SIGNS}
    for sign in SIGNS:
        scores[sign]['NEAR'][1, 1] = 1.
        scores[sign]['M3'][:3, 1] = 1.
    # A paired absolute deterioration can coexist with a remaining relative gain.
    scores['plus1']['M3'][1, 1] = 0.
    with patch.object(SE, 'calibrate', side_effect=AssertionError('Calibration forbidden')):
        result = evaluate_arrays(g, scores, {'NEAR': .3, 'M3': .7})
    m = result['cells']['plus1|M3']['metrics']['contact0-2cm']
    assert m['n'] == 1 and 'episode_draw_pairs' not in m
    assert result['within_arm_biased_minus_nominal']['plus1']['M3']['contact>5cm']['delta_timely'] == -1.
    assert result['within_arm_biased_minus_nominal']['minus1']['NEAR']['clear']['delta_per_min'] == 0.
    prior = dict(cells={a: result['cells'][f'nominal0|{a}'] for a in ARMS})
    assert nominal_parity(result['cells'], prior)['status'] == 'EXACT_POINT_REPRODUCTION'
    good = {'contact0-2cm': {'paired_unit_ci95': [.001, .1]}, 'contact>5cm': {'delta_timely': -.02}, 'clear': {'delta_per_min': .2}}
    assert joint_decision({'minus1': good, 'plus1': good})[0] == 'M3_RELATIVE_GAIN_SURVIVES_PLUS_MINUS_1DEG'
    bad = dict(good, clear={'delta_per_min': .201})
    assert joint_decision({'minus1': good, 'plus1': bad})[0] == 'NOT_SUPPORTED_UNDER_YAW_BIAS'
    json.dumps(result, allow_nan=False)
    print('PASS: common paired bootstrap, fixed truth/thresholds, integer counts, nominal parity, both-sign gate and within-arm degradation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'evaluate'))
    args = parser.parse_args()
    check() if args.stage == 'check' else evaluate()
