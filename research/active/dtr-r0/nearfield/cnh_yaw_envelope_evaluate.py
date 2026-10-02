"""One frozen three-hypothesis M3 yaw envelope, with a predeclared nominal gate.

BASE retains its old threshold. ENV smooths each fixed projection hypothesis
before max and calibrates one threshold as max_b(T_b) on source48 only.
"""
import argparse
import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_sequence_extrinsic_evaluate as XE
import cnh_sequence_observed_evaluate as OE
import cnh_sequence_transfer_evaluate as TE
import cnh_three_level_sequence as SE

OUT = MC.SS.WORK/'cnh-yaw-envelope-20261002'
BIASES = (-1, 0, 1)
BOOT_SEED = 2026100218
N_BOOT = 1000
OLD_THRESHOLD = .8557642486787612
CATEGORIES = OE.CONTACTS+('pass0-10cm', 'clear')


def verify_hashes(hashes):
    for name, digest in hashes.items():
        if ME.sha(name) != digest:
            raise ValueError('Frozen input changed: '+name)


def plan_inputs():
    path = OUT/'PLAN.json'
    plan = OE.read(path)
    verify_hashes(plan['prior_sha256'])
    if plan['baseline']['threshold'] != OLD_THRESHOLD or plan['bootstrap']['seed'] != BOOT_SEED:
        raise ValueError('Plan baseline threshold/bootstrap contract differs')
    hashes = dict(plan['prior_sha256'])
    hashes[str(path)] = ME.sha(path)
    return plan, hashes


def existing_complete(path):
    if not path.exists():
        return None
    result = OE.read(path)
    if result.get('status') != 'COMPLETE':
        raise ValueError('Existing incomplete evidence requires inspection: '+str(path))
    verify_hashes(result['provenance']['input_sha256'])
    return result


def new_scores(split, units, angles):
    receipt_path, request_path = OUT/f'scores_receipt_{split}.json', OUT/'request.json'
    receipt = OE.read(receipt_path)
    if receipt.get('status') != 'COMPLETE' or receipt.get('split') != split:
        raise ValueError('Requested score split is not COMPLETE: '+split)
    if receipt.get('plan_sha256') != ME.sha(OUT/'PLAN.json') or receipt.get('request_sha256') != ME.sha(request_path):
        raise ValueError('Score receipt is not bound to the frozen plan/request')
    hashes = {str(receipt_path): ME.sha(receipt_path), str(request_path): ME.sha(request_path)}
    result = {}
    for angle in angles:
        tag = ('minus' if angle < 0 else 'plus')+str(abs(angle))
        path = OUT/f'frame_scores_{split}_{tag}_M3.npz'
        digest = ME.sha(path)
        if receipt.get('output_sha256', {}).get(path.name) != digest:
            raise ValueError('Score output hash differs: '+path.name)
        with np.load(path, allow_pickle=False) as data:
            raw = data['logit']
            if not np.array_equal(data['units'], units) or not np.array_equal(data['frames'], np.arange(3, 16)):
                raise ValueError('Raw score unit/frame axes differ: '+path.name)
            if np.asarray(data['bias_deg']).shape != () or float(data['bias_deg']) != angle:
                raise ValueError('Raw score angle differs: '+path.name)
        if raw.shape != (len(units), 40, 13, 2) or not np.isfinite(raw).all():
            raise ValueError('Raw scores have incorrect shape/values: '+path.name)
        result[angle] = SE.smooth(raw).transpose(0, 1, 3, 2).reshape(-1, 13)
        hashes[str(path)] = digest
    return result, hashes, receipt


def envelope(scores, bias):
    """All candidates are used; no true-bias-dependent selection of one branch."""
    return np.maximum.reduce([scores[bias+c] for c in (-1, 0, 1)])


def rate_at(score, clear, threshold):
    n = int(clear.sum())
    if not n:
        raise ValueError('No all-frame-clear calibration episodes')
    minutes = n*13*.2/60
    count = int((score[clear].max(1) >= threshold).sum())
    return dict(clear_episodes=n, clear_minutes=minutes, first_stops=count, first_stops_per_min=count/minutes)


def calibrate_arrays(scores, clear, base_threshold=OLD_THRESHOLD):
    local, local_receipts = {}, {}
    for bias in BIASES:
        local[bias], local_receipts[bias] = SE.calibrate(envelope(scores, bias), clear, 1.)
    threshold = max(local.values())
    if threshold < base_threshold:
        raise ValueError('Envelope threshold cannot fall below retained BASE threshold: every bank includes nominal0')
    per_bias = {}
    for bias in BIASES:
        env = rate_at(envelope(scores, bias), clear, threshold)
        if env['first_stops_per_min'] > 1.+1e-12:
            raise ValueError('Fixed envelope threshold violates source per-bias budget')
        per_bias[str(bias)] = dict(local_threshold=local[bias], local_calibration=local_receipts[bias],
            ENV_at_frozen_threshold=env, BASE_at_original_threshold=rate_at(scores[bias], clear, base_threshold))
    return dict(base_threshold=base_threshold, envelope_threshold=threshold, per_bias=per_bias,
        active_calibration_biases=[b for b in BIASES if local[b] == threshold],
        rule='One T=max_b T_b, each fixed global bias calibrated separately at1 clear first alarm/proxy minute; BASE never recalibrated')


def calibrate():
    path = OUT/'calibration.json'
    previous = existing_complete(path)
    if previous is not None:
        print('Existing frozen calibration verified; not recalibrated', flush=True)
        return previous
    plan, hashes = plan_inputs()
    geometry, old_scores, old_provenance = OE.load_inputs()
    source = OE.read(OE.OUT/'result.json')
    if source['cells']['M3']['threshold'] != OLD_THRESHOLD:
        raise ValueError('Original M3 threshold changed')
    cal = geometry['split'] == 'calib'
    units = np.unique(geometry['unit'][cal]).tolist()
    if units != plan['calibration_units']:
        raise ValueError('Calibration unit identities differ')
    scores, score_hashes, receipt = new_scores('calib', units, (-2, -1, 1, 2))
    scores[0] = old_scores['M3'][cal]
    hashes.update(old_provenance['input_sha256']); hashes.update(score_hashes)
    result = calibrate_arrays(scores, geometry['clear_all'][cal])
    result.update(status='COMPLETE', stage='SOURCE_CALIBRATION_FROZEN', calibration_units=units,
        provenance=dict(input_sha256=hashes, source_score_receipt=receipt, evaluator_sha256=ME.sha(__file__)))
    verify_hashes(hashes)
    OE.save(path, result)
    print('Calibration frozen', result['envelope_threshold'], flush=True)
    return result


def target_inputs():
    plan, hashes = plan_inputs()
    geometry, old_scores, thresholds, prior, provenance = XE.load_inputs()
    if thresholds['M3'] != OLD_THRESHOLD or np.unique(geometry['unit']).tolist() != plan['evaluation_units']:
        raise ValueError('Retained target threshold/unit identity differs')
    scores = {b: old_scores[sign]['M3'] for b, sign in [(-1, 'minus1'), (0, 'nominal0'), (1, 'plus1')]}
    hashes.update(provenance['input_sha256'])
    return geometry, scores, hashes


def baseline_parity(result, biases):
    old = OE.read(XE.OUT/'result.json')
    sign = {-1: 'minus1', 0: 'nominal0', 1: 'plus1'}
    checked = 0
    for bias in biases:
        for category in CATEGORIES:
            new = result['cells'][f'{bias}|BASE']['metrics'][category]
            prior = old['cells'][f'{sign[bias]}|M3']['metrics'][category]
            fields = ['expected_episodes', 'expected_stops']
            fields += ['clear_minutes'] if category == 'clear' else ['expected_timely_stops']
            for field in fields:
                if new[field] != prior[field]:
                    raise ValueError(f'Retained BASE count changed: {bias}/{category}/{field}')
                checked += 1
            rates = ['false_stops_per_min'] if category == 'clear' else ['timely_rate', 'stop_rate', 'median_lead_to_0p5m_s']
            for field in rates:
                if new[field]['value'] != prior[field]['value']:
                    raise ValueError(f'Retained BASE point changed: {bias}/{category}/{field}')
                checked += 1
    return dict(status='EXACT_OLD_M3_POINT_REPRODUCTION', point_fields_checked=checked,
        intervals='New common bootstrap seed; old intervals/verdicts retained')


def per_bias_guards(comparison, nominal=False):
    deep, clear = comparison['contact>5cm']['delta_timely'], comparison['clear']['delta_per_min']
    checks = dict(deep_decline_le_2pp=deep is not None and deep >= -.02,
        clear_increase_le_0p2_per_min=clear is not None and clear <= .2)
    if nominal:
        shallow = comparison['contact0-2cm']['delta_timely']
        checks['nominal_shallow_decline_le_2pp'] = shallow is not None and shallow >= -.02
    return checks


def worst_bootstrap(base_rates, env_rates, base_samples, env_samples):
    """Re-select the worst bias separately inside every paired unit resample."""
    bs, es = np.asarray(base_samples), np.asarray(env_samples)
    base_min, env_min = np.min(bs, axis=0), np.min(es, axis=0)
    valid_points = all(v is not None for v in list(base_rates)+list(env_rates))
    base, env = (min(base_rates), min(env_rates)) if valid_points else (None, None)
    return dict(status='MEASURED', BASE_worst_rate=base, ENV_worst_rate=env,
        delta=env-base if valid_points else None, paired_unit_ci95=SE.interval(env_min-base_min),
        BASE_worst_rate_ci95=SE.interval(base_min), ENV_worst_rate_ci95=SE.interval(env_min),
        BASE_worst_biases=[b for b, v in zip(BIASES, base_rates) if v == base] if valid_points else [],
        ENV_worst_biases=[b for b, v in zip(BIASES, env_rates) if v == env] if valid_points else [],
        bootstrap_operation='min_ENV and min_BASE independently recomputed over all3biases inside every common whole-unit draw')


def analyze(g, scores, threshold, biases):
    units, ui = np.unique(g['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    weights = {c: (g['covered'] & (g['ref_category'] == c)).astype(float) for c in CATEGORIES[:4]}
    weights['clear'] = g['clear_all'].astype(float)
    cells, samples, flags, comparisons, guards = {}, {}, {}, {}, {}
    for bias in biases:
        for method, score, t in [('BASE', scores[bias], OLD_THRESHOLD), ('ENV', envelope(scores, bias), threshold)]:
            key = f'{bias}|{method}'
            cells[key], samples[key], flags[key] = XE.summarize_cell(g, score, t, weights, ui, boot)
        comparisons[str(bias)] = XE.paired_changes(cells, samples, flags, weights, f'{bias}|ENV', f'{bias}|BASE')
        guards[str(bias)] = per_bias_guards(comparisons[str(bias)], nominal=bias == 0)
    worst = dict(status='NOT_EVALUATED_NOMINAL_GATE_ONLY', delta=None, paired_unit_ci95=[None, None])
    if tuple(biases) == BIASES:
        points = {m: [cells[f'{b}|{m}']['metrics']['contact0-2cm']['timely_rate']['value'] for b in BIASES] for m in ('BASE', 'ENV')}
        draws = {m: [samples[f'{b}|{m}']['contact0-2cm'] for b in BIASES] for m in ('BASE', 'ENV')}
        worst = worst_bootstrap(points['BASE'], points['ENV'], draws['BASE'], draws['ENV'])
    return dict(cells=cells, comparisons_ENV_minus_BASE=comparisons, per_bias_guards=guards,
        primary_worst_bias=worst, evaluation_units=units.tolist(),
        n=dict(query_episodes=len(g['unit']), covered=int(g['covered'].sum()), right_censored=int((~g['covered']).sum()),
            clear_all=int(g['clear_all'].sum())),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, clusters=len(units), pairing='same resamples across method and bias'))


def final_verdict(worst, guards):
    primary = worst['status'] == 'MEASURED' and worst['paired_unit_ci95'][0] is not None and worst['paired_unit_ci95'][0] > 0
    guard_pass = all(all(checks.values()) for checks in guards.values())
    return ('YAW_ENVELOPE_SUPPORTED_DEV' if primary and guard_pass else 'NOT_SUPPORTED_YAW_ENVELOPE'), dict(primary_pass=primary, guards_pass=guard_pass)


def bind_stage(result, hashes, calibration):
    hashes = dict(hashes)
    hashes.update(calibration['provenance']['input_sha256'])
    hashes[str(OUT/'calibration.json')] = ME.sha(OUT/'calibration.json')
    result.update(calibration=calibration, provenance=dict(input_sha256=hashes,
        evaluator_sha256=ME.sha(__file__), helper_sha256={p: ME.sha(p) for p in (SE.__file__, OE.__file__, XE.__file__)}))
    verify_hashes(hashes)
    return result


def nominal():
    terminal = existing_complete(OUT/'result.json')
    if terminal is not None:
        print('Existing terminal verified; nominal stage not repeated', flush=True)
        return terminal
    previous = existing_complete(OUT/'nominal_result.json')
    if previous is not None:
        print('Existing nominal gate verified; not repeated', flush=True)
        return previous
    calibration = existing_complete(OUT/'calibration.json')
    if calibration is None:
        raise ValueError('Freeze source calibration before nominal target gate')
    g, scores, hashes = target_inputs()
    result = analyze(g, scores, calibration['envelope_threshold'], (0,))
    result['baseline_point_reproduction'] = baseline_parity(result, (0,))
    passed = all(result['per_bias_guards']['0'].values())
    result.update(status='COMPLETE', stage='NOMINAL_GATE_PASSED' if passed else 'EARLY_STOP_NOMINAL_GUARD_FAILED',
        nominal_gate_passed=passed, verdict='PENDING_FULL_EVALUATION' if passed else 'NOT_SUPPORTED_YAW_ENVELOPE',
        scope='Nominal target only; full worst-bias primary not measured; target +/-2 not required if gate fails')
    bind_stage(result, hashes, calibration)
    OE.save(OUT/'nominal_result.json', result)
    if not passed:
        result['primary_worst_bias']['status'] = 'NOT_EVALUATED_EARLY_STOP'
        OE.save(OUT/'result.json', result)
    write_report(result)
    print(result['stage'], flush=True)
    return result


def evaluate():
    terminal = existing_complete(OUT/'result.json')
    if terminal is not None:
        print('Existing terminal verified; not repeated', flush=True)
        return terminal
    gate = existing_complete(OUT/'nominal_result.json')
    calibration = existing_complete(OUT/'calibration.json')
    if gate is None or not gate.get('nominal_gate_passed') or calibration is None:
        raise ValueError('Full evaluation requires the completed nominal gate to have passed')
    g, scores, hashes = target_inputs()
    added, added_hashes, receipt = new_scores('target', TE.UNITS, (-2, 2))
    scores.update(added); hashes.update(added_hashes)
    hashes[str(OUT/'nominal_result.json')] = ME.sha(OUT/'nominal_result.json')
    result = analyze(g, scores, calibration['envelope_threshold'], BIASES)
    result['baseline_point_reproduction'] = baseline_parity(result, BIASES)
    # The nominal gate must remain byte-for-byte equivalent in its numerical content.
    for method in ('BASE', 'ENV'):
        if result['cells'][f'0|{method}'] != gate['cells'][f'0|{method}']:
            raise ValueError('Nominal statistics changed between gate and full evaluation')
    outcome, checks = final_verdict(result['primary_worst_bias'], result['per_bias_guards'])
    result.update(status='COMPLETE', stage='FULL_EVALUATION', verdict=outcome, decision_checks=checks,
        scope='All3fixed global biases evaluated; no sign selection', target_score_receipt=receipt)
    bind_stage(result, hashes, calibration)
    write_report(result)
    OE.save(OUT/'result.json', result)
    print(outcome, flush=True)
    return result


def number(value, scale=1.):
    return '--' if value is None else f'{scale*value:.2f}'


def write_report(result):
    calibration = result['calibration']
    lines = [result['verdict'], '', '# 固定三假设yaw并集报警', '',
        f"阶段：{result['stage']}。BASE保留原M3阈值{calibration['base_threshold']!r}；ENV唯一阈值{calibration['envelope_threshold']!r}。每个角度先独立作原因果平滑，再取三个假设max；不利用真值选择纠正方向。", '',
        '| source固定bias | 单独预算阈值T_b | ENV冻结T后清晰首停/分钟 | BASE原阈值清晰首停/分钟 |', '| --- | --- | --- | --- |']
    for bias in BIASES:
        r = calibration['per_bias'][str(bias)]
        e, b = r['ENV_at_frozen_threshold'], r['BASE_at_original_threshold']
        lines.append(f"| {bias:+d} | {r['local_threshold']!r} | {e['first_stops']}/{e['clear_minutes']:.2f}={e['first_stops_per_min']:.3f} | {b['first_stops']}/{b['clear_minutes']:.2f}={b['first_stops_per_min']:.3f} |")
    lines += ['', 'T=max_b(T_b)只控制每个固定全局bias的source经验预算；不声称逐序列任意变化bias的联合预算、目标≤1次/分钟或真实步行负担。', '']
    for bias in result['comparisons_ENV_minus_BASE']:
        lines += [f'## target固定bias={int(bias):+d}°', '', '| 指标 | BASE | ENV |', '| --- | --- | --- |']
        for category in CATEGORIES:
            lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][f'{bias}|{m}']['metrics'][category], category == 'clear', category == 'pass0-10cm') for m in ('BASE', 'ENV'))+' |')
        lines += ['', '| 方法/擦碰档 | 已报警条件提前量中位数及95%区间，秒 |', '| --- | --- |']
        for method in ('BASE', 'ENV'):
            for category in OE.CONTACTS:
                m = result['cells'][f'{bias}|{method}']['metrics'][category]['median_lead_to_0p5m_s']
                lines.append(f"| {method}/{category} | {number(m['value'])} [{number(m['ci95'][0])}, {number(m['ci95'][1])}] |")
        lines += ['', f"该bias护栏：{result['per_bias_guards'][bias]}。"]
    worst = result['primary_worst_bias']
    if worst['status'] == 'MEASURED':
        lines += ['', f"最坏bias浅及时率：BASE {number(worst['BASE_worst_rate'],100)}%（bias {worst['BASE_worst_biases']}）；ENV {number(worst['ENV_worst_rate'],100)}%（bias {worst['ENV_worst_biases']}）。差{number(worst['delta'],100)}pp，95%区间[{number(worst['paired_unit_ci95'][0],100)}, {number(worst['paired_unit_ci95'][1],100)}]。每次bootstrap内重新对各自三bias取min，没有固定选择某个符号。"]
    else:
        lines += ['', '完整worst-bias主指标尚未测量，不填造数值。名义护栏失败时按预定AND判据早停，不补target±2，不把未测量写作零收益。']
    lines += ['', '已消费Development，1000次整目标单位共同bootstrap，seed2026100218。所有名义真值、覆盖/右删失、0.9m目标前缘参照固定；未观察到截止点的未报警不计漏报。', '',
        '三点假设在测试−1/0/+1格点包含精确抵消分支，不代表连续±1°区间认证或实测误差分布。传感器到身体映射输出侧yaw对齐偏差作用于已有z1=r/sqrt(v)表示；不是新光子观测或硬件实际转动。', '',
        '全13采样时刻清晰不保证帧间连续清晰；2.6秒/查询序列代理暴露包含首停后时间，不等于用户实际步行分钟。提前量相对0.5m按0.8m/s转换、条件于已报警。三分支计算成本另见评分凭据，桌面运行不能证明手机实时性。', '',
        '目标没有source的同高度身体外10–20cm清晰支持。旧失败全部保留，不重训、不改BASE阈值、不在目标调阈值、不失败后增加角度或权重。完整配对差、分母、区间及输入凭据见JSON。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def check():
    # Max after independent smoothing differs from switching branches before smoothing.
    a = np.array([10., 0.]).reshape(1, 1, 2, 1)
    b = np.array([0., 10.]).reshape(1, 1, 2, 1)
    after = np.maximum(SE.smooth(a), SE.smooth(b))
    before = SE.smooth(np.maximum(a, b))
    assert after[0, 0, -1, 0] < before[0, 0, -1, 0]
    scores = {i: np.full((2, 13), float(i)) for i in range(-2, 3)}
    assert np.array_equal(envelope(scores, -1), np.zeros((2, 13)))
    assert np.array_equal(envelope(scores, 1), np.full((2, 13), 2.))
    # Distinct high-clear cases in two global biases: calibrate each bias, not their union.
    calibration_scores = {a: np.zeros((100, 13)) for a in range(-2, 3)}
    calibration_scores[-2][:4] = 2.
    calibration_scores[2][4:8] = 2.
    frozen = calibrate_arrays(calibration_scores, np.ones(100, bool), base_threshold=0.)
    assert 0 < frozen['envelope_threshold'] < 2.
    assert frozen['per_bias']['-1']['ENV_at_frozen_threshold']['first_stops'] == 4
    assert frozen['per_bias']['1']['ENV_at_frozen_threshold']['first_stops'] == 4
    # Resample0 and1 have different worst signs; fixed-sign bootstrap would be wrong.
    base = np.array([[.2, .9], [.5, .5], [.9, .2]])
    env = np.array([[.4, .8], [.6, .6], [.8, .4]])
    worst = worst_bootstrap([.2, .5, .9], [.4, .6, .8], base, env)
    assert np.allclose(worst['paired_unit_ci95'], [.2, .2])
    cmp = {'contact0-2cm': {'delta_timely': -.02}, 'contact>5cm': {'delta_timely': -.02}, 'clear': {'delta_per_min': .2}}
    assert all(per_bias_guards(cmp, nominal=True).values())
    bad = dict(cmp, clear={'delta_per_min': .20001})
    assert not all(per_bias_guards(bad, nominal=True).values())
    good_worst = dict(status='MEASURED', paired_unit_ci95=[.001, .1])
    assert final_verdict(good_worst, {'0': per_bias_guards(cmp, True)})[0] == 'YAW_ENVELOPE_SUPPORTED_DEV'
    assert final_verdict(dict(status='NOT_EVALUATED_EARLY_STOP', paired_unit_ci95=[None, None]), {'0': per_bias_guards(bad, True)})[0] == 'NOT_SUPPORTED_YAW_ENVELOPE'
    print('PASS: smooth-before-max, fixed finite bank/no oracle branch, min inside bootstrap, gate boundaries and explicit unmeasured early stop')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'calibrate', 'nominal', 'evaluate'))
    args = parser.parse_args()
    {'check': check, 'calibrate': calibrate, 'nominal': nominal, 'evaluate': evaluate}[args.stage]()
