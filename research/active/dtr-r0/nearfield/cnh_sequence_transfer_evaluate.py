"""Frozen source-threshold NEAR/M3 sequence transfer on existing 94000 units.

Evaluation only: no target calibration, rendering, inference or temporal policy
selection. Source and target cohorts are reported separately, never pooled.
"""
import argparse
from pathlib import Path

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_sequence_observed_evaluate as OE
import cnh_three_level_sequence as SE

OUT = MC.SS.WORK/'cnh-sequence-transfer-20261002'
SOURCE = MC.SS.WORK/'cnh-observed-sequence-20261002'
UNITS = list(range(94000, 94096))
ARMS = ('NEAR', 'M3')
BOOT_SEED = 2026100213
N_BOOT = 1000
CONTACTS = OE.CONTACTS
CATEGORIES = OE.CATEGORIES


def validate_geometry(g, rows):
    expected = [('evaluation', u, c, q) for u in UNITS for c in range(40) for q in (0, 1)]
    actual = list(zip(g['split'].tolist(), g['unit'].tolist(), g['config'].tolist(), g['query'].tolist()))
    row_keys = [(r['split'], r['unit'], r['config'], r['query']) for r in rows]
    if actual != expected or row_keys != expected:
        raise ValueError('Geometry rows must exactly follow 94000-94095/config0-39/query0-1 evaluation order')
    n = len(expected)
    if g['frame_ranges'].shape != (n, 13) or g['frame_category'].shape != (n, 13):
        raise ValueError('Expected 13 observed frames for each of 7680 query episodes')
    if 'frames' in g and not np.array_equal(g['frames'], np.arange(3, 16)):
        raise ValueError('Geometry frame identities must be3-15')
    if not np.isfinite(g['frame_ranges']).all() or (np.diff(g['frame_ranges'], axis=1) > 1e-8).any():
        raise ValueError('Observed distances must be finite and non-increasing')
    for key in ('ref_category', 'last_category', 'covered', 'clear_all'):
        if g[key].shape != (n,):
            raise ValueError('Geometry row dimension mismatch: '+key)
    if not np.isin(g['ref_category'], CATEGORIES).all() or not np.isin(g['frame_category'], CATEGORIES[:-1]).all():
        raise ValueError('Unrecognized geometry category')
    covered = (g['frame_ranges'][:, 0] >= .9) & (g['frame_ranges'] <= .9).any(1)
    if not np.array_equal(g['covered'], covered) or not np.array_equal(g['ref_category'] == 'censored', ~covered):
        raise ValueError('Coverage requires an observed crossing; unreached references must be censored')
    if not np.array_equal(g['clear_all'], (g['frame_category'] == 'clear').all(1)):
        raise ValueError('Clear exposure includes a non-clear observed frame')
    if not np.array_equal(g['last_category'], g['frame_category'][:, -1]):
        raise ValueError('Last-frame category mismatch')


def load_inputs():
    plan_path = OUT/'PLAN.json'
    plan = OE.read(plan_path)
    geometry_receipt = OUT/'geometry_receipt.json'
    receipt = OE.read(geometry_receipt)
    if receipt.get('status') != 'COMPLETE':
        raise ValueError('Geometry receipt is not COMPLETE')
    hashes = {}
    for name, key in [('geometry.npz', 'geometry_sha256'), ('rows.json', 'rows_sha256'), ('PLAN.json', 'plan_sha256')]:
        digest = ME.sha(OUT/name)
        if receipt.get(key) != digest:
            raise ValueError('Geometry receipt hash mismatch: '+name)
        hashes[str(OUT/name)] = digest
    hashes[str(geometry_receipt)] = ME.sha(geometry_receipt)
    with np.load(OUT/'geometry.npz', allow_pickle=False) as cache:
        g = {k: cache[k] for k in cache.files}
    validate_geometry(g, OE.read(OUT/'rows.json'))
    source_path = SOURCE/'result.json'
    source = OE.read(source_path)
    if source.get('status') != 'COMPLETE':
        raise ValueError('Source operating point result must be COMPLETE')
    source_sha = ME.sha(source_path)
    thresholds = {a: float(source['cells'][a]['threshold']) for a in ARMS}
    if source_sha != plan['source_result_sha256'] or thresholds != plan['thresholds']:
        raise ValueError('Source result or frozen thresholds differ from the prewritten transfer plan')
    if not all(np.isfinite(t) for t in thresholds.values()):
        raise ValueError('Nonfinite frozen source threshold')
    hashes[str(source_path)] = source_sha
    scores_receipt_path = OUT/'scores_receipt.json'
    # Receipt structure is producer-owned; retain its complete content and hash.
    scores_receipt = OE.read(scores_receipt_path)
    if scores_receipt.get('status') != 'COMPLETE':
        raise ValueError('Score cache receipt must be COMPLETE')
    hashes[str(scores_receipt_path)] = ME.sha(scores_receipt_path)
    scores, parity = {}, {}
    for arm in ARMS:
        path = OUT/f'frame_scores_{arm}.npz'
        with np.load(path, allow_pickle=False) as cache:
            raw = cache['logit']
            if not np.array_equal(cache['units'], UNITS) or not np.array_equal(cache['frames'], np.arange(3, 16)):
                raise ValueError(arm+': raw cache axis identities differ')
        if raw.shape != (96, 40, 13, 2) or not np.isfinite(raw).all():
            raise ValueError(arm+': raw scores must be finite [96,40,13,2]')
        smoothed = SE.smooth(raw)
        final_path = (MC.NR.OUT/'scores_NEAR.npz' if arm == 'NEAR' else MC.MARGIN/'scores_M3.npz')
        with np.load(final_path, allow_pickle=False) as final:
            error = max(float(np.max(np.abs(smoothed[i, :, -1]-final[str(u) if arm == 'NEAR' else f'near|{u}']))) for i, u in enumerate(UNITS))
        if error >= 1e-4:
            raise ValueError(f'{arm}: frozen final-score parity failed: {error}')
        scores[arm] = smoothed.transpose(0, 1, 3, 2).reshape(-1, 13)
        hashes[str(path)], hashes[str(final_path)] = ME.sha(path), ME.sha(final_path)
        if scores_receipt.get('output_sha256', {}).get(arm) != hashes[str(path)]:
            raise ValueError('Score producer output hash mismatch: '+arm)
        parity[arm] = dict(final_score_max_abs_error=error, final_reference=str(final_path))
    return g, scores, thresholds, dict(input_sha256=hashes, plan=plan,
        source_result_sha256=source_sha, frozen_thresholds=thresholds,
        source_calibration={a: source['cells'][a]['calibration'] for a in ARMS},
        source_verdict_retained=source['verdict'], score_producer_receipt=scores_receipt, final_parity=parity,
        evaluator_sha256=ME.sha(__file__), sequence_helper_sha256=ME.sha(SE.__file__),
        observed_helper_sha256=ME.sha(OE.__file__))


def decision(shallow_ci, deep_delta, clear_delta):
    checks = dict(shallow_ci_lower_gt_zero=shallow_ci[0] is not None and shallow_ci[0] > 0,
                  deep_drop_le_2pp=deep_delta is not None and deep_delta >= -.02,
                  clear_increase_le_0p2_per_min=clear_delta is not None and clear_delta <= .2)
    return ('TRANSFER_SUPPORTED_AT_FROZEN_THRESHOLDS' if all(checks.values())
            else 'NOT_SUPPORTED_AT_FROZEN_THRESHOLDS'), checks


def evaluate_arrays(g, scores, thresholds):
    # Explicit row fields keep global metadata such as frames out of row indexing.
    fields = ('unit', 'frame_ranges', 'frame_category', 'ref_category', 'clear_all', 'covered', 'last_category')
    n = len(g['unit'])
    if any(g[k].shape[0] != n for k in fields):
        raise ValueError('Required geometry row fields do not align')
    if not np.all(g['split'] == 'evaluation'):
        raise ValueError('Transfer has no target calibration rows')
    e = {k: g[k] for k in fields}
    units, ui = np.unique(e['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    weights = {name: (e['covered'] & (e['ref_category'] == name)).astype(float) for name in CONTACTS+('pass0-10cm',)}
    weights['clear'] = e['clear_all'].astype(float)
    cells, sampled, flags, clear_samples = {}, {}, {}, {}
    for arm in ARMS:
        threshold, score = thresholds[arm], scores[arm]
        if score.shape != e['frame_ranges'].shape or not np.isfinite(score).all():
            raise ValueError('Score/geometry mismatch: '+arm)
        stopped, timely, lead = SE.first_stops(score, threshold, e['frame_ranges'])
        metrics, samples = SE.summarize(weights, e['covered'], stopped, timely, lead, ui, boot)
        for metric in metrics.values():
            metric.pop('episode_draw_pairs', None)
            metric['n'] = int(round(metric['expected_episodes']))
            metric['stops'] = int(round(metric['expected_stops']))
            if 'expected_timely_stops' in metric:
                metric['timely_stops'] = int(round(metric['expected_timely_stops']))
        clear_den = SE.unit_totals(weights['clear'], ui, len(units))*13*.2/60
        clear_num = SE.unit_totals(weights['clear']*stopped, ui, len(units))
        _, clear_samples[arm] = SE.rates(clear_num, clear_den, boot)
        diagnostics = {name: OE.extra_counts(weights[name] > 0, stopped, timely, score, threshold)
                       for name in CONTACTS+('pass0-10cm',)}
        censored = ~e['covered']
        censored_counts = dict(n=int(censored.sum()), already_alarm=int((censored & stopped).sum()),
            no_alarm_right_censored=int((censored & ~stopped).sum()),
            interpretation='No-alarm right-censored episodes are not missed deadlines',
            by_last_category={name: dict(n=int((censored & (e['last_category'] == name)).sum()),
                already_alarm=int((censored & (e['last_category'] == name) & stopped).sum())) for name in CATEGORIES[:-1]})
        cells[arm] = dict(threshold=threshold, threshold_origin='SOURCE observed-sequence 1/min pooled HEAD/BODY; no target recalibration',
            metrics=metrics, timing_diagnostics=diagnostics, censored=censored_counts)
        sampled[arm], flags[arm] = samples, dict(stopped=stopped, timely=timely)
    comparisons = {}
    for category in CONTACTS:
        a, b = (cells[arm]['metrics'][category]['timely_rate']['value'] for arm in ('M3', 'NEAR'))
        keep = weights[category] > 0
        nt, mt = flags['NEAR']['timely'], flags['M3']['timely']
        comparisons[category] = dict(delta_timely=a-b if a is not None and b is not None else None,
            paired_unit_ci95=SE.interval(sampled['M3'][category]-sampled['NEAR'][category]), n=int(keep.sum()),
            rescues=int((keep & mt & ~nt).sum()), losses=int((keep & nt & ~mt).sum()),
            both_timely=int((keep & nt & mt).sum()), neither_timely=int((keep & ~nt & ~mt).sum()))
    a, b = (cells[arm]['metrics']['clear']['false_stops_per_min']['value'] for arm in ('M3', 'NEAR'))
    clear_comparison = dict(delta_per_min=a-b if a is not None and b is not None else None,
        paired_unit_ci95=SE.interval(clear_samples['M3']-clear_samples['NEAR']))
    outcome, checks = decision(comparisons['contact0-2cm']['paired_unit_ci95'],
        comparisons['contact>5cm']['delta_timely'], clear_comparison['delta_per_min'])
    transition = np.any(e['frame_category'] != e['frame_category'][:, :1], axis=1)
    counts = dict(target_calibration_units=0, evaluation_units=len(units), evaluation_query_episodes=n,
        evaluation_covered=int(e['covered'].sum()), evaluation_censored=int((~e['covered']).sum()),
        evaluation_clear_all=int(e['clear_all'].sum()), evaluation_not_all_clear=int((~e['clear_all']).sum()),
        evaluation_category_transition=int(transition.sum()),
        reference_clear_but_not_all_frames_clear=int(((e['ref_category'] == 'clear') & ~e['clear_all']).sum()),
        last_clear_but_not_all_frames_clear=int(((e['last_category'] == 'clear') & ~e['clear_all']).sum()),
        reference_category={name: int((e['ref_category'] == name).sum()) for name in CATEGORIES})
    return dict(status='COMPLETE', verdict=outcome, decision_checks=checks, cells=cells,
        comparison_M3_minus_NEAR=comparisons, clear_comparison_M3_minus_NEAR=clear_comparison,
        n=counts, evaluation_units=units.tolist(), bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED,
            cluster='whole target evaluation units paired across arms', conditional_on='frozen source thresholds'),
        limits=['Consumed Development 94000-94095; source and target not pooled, not protected or hardware confirmation',
                'Target nominal intrusion[-.10,+.15] lacks the source same-height10-20cm outside target population; clear mixtures differ',
                'One source threshold per arm from 1/min calibration; no target recalibration or operating-point selection',
                'Sigma0 only, shared target-front 0.9m anchor under actual poses, not individual background collision deadlines',
                'Clear at all13 observed instants only; no guarantee over continuous between-frame time',
                '2.6s decision-frame query exposure per episode even after first stop; at most one first alarm, not actual walking burden',
                'Covered contacts/pass only; right-censored no-alarm episodes are not counted as missed deadlines',
                'Lead to0.5m uses0.8m/s nominal conversion conditional on alarm, not actual reaction/braking time',
                'Intervals omit source calibration, model training and simulator uncertainty; no400 heading-draw pseudo denominator',
                'Old failed judgments remain unchanged'])


def write_report(result):
    n = result['n']
    lines = [result['verdict'], '', '# 冻结源阈值的既有单位序列迁移', '',
        '目标为94000–94095的96单位、每单位40场景、每场景两个查询。NEAR/M3沿用源域observed-sequence在1次/代理分钟预算校准的各自单阈值；目标集不重新校准，不合并源域分母。',
        f"已观测至0.9m截止点 {n['evaluation_covered']}/{n['evaluation_query_episodes']} 条查询序列；另有{n['evaluation_censored']}条右删失，其未报警不当漏报。", '',
        '| 指标 | NEAR | M3 |', '| --- | --- | --- |']
    for category in CONTACTS+('pass0-10cm', 'clear'):
        lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][a]['metrics'][category], category == 'clear', category == 'pass0-10cm') for a in ARMS)+' |')
    lines += ['', '主判据：M3−NEAR浅擦碰及时率配对95%区间下界>0、深擦碰下降≤2pp、清晰首报警率增幅≤0.2次/代理分钟，三项均须满足。0.2是源1次/分钟预算20%的探索性增幅容限，不保证目标实际负担≤1次/分钟。', '',
        '| 擦碰 | 及时率差及95%区间 | 补回 / 丢失 |', '| --- | --- | --- |']
    pp = lambda value: '--' if value is None else f'{100*value:+.2f}pp'
    dec = lambda value: '--' if value is None else f'{value:.2f}'
    for category, m in result['comparison_M3_minus_NEAR'].items():
        lines.append(f"| {category} | {pp(m['delta_timely'])} [{pp(m['paired_unit_ci95'][0])}, {pp(m['paired_unit_ci95'][1])}] | {m['rescues']} / {m['losses']}（分母{m['n']}） |")
    c = result['clear_comparison_M3_minus_NEAR']
    lines += ['', f"清晰首报警率M3−NEAR={dec(c['delta_per_min'])}次/代理分钟，95%区间[{dec(c['paired_unit_ci95'][0])}, {dec(c['paired_unit_ci95'][1])}]。", '',
        '| 臂/类别 | 及时首报但末帧低于阈值 | 仅迟报 | 已报警者提前量中位数及95%区间，秒 |', '| --- | --- | --- | --- |']
    for arm in ARMS:
        for category in CONTACTS:
            d = result['cells'][arm]['timing_diagnostics'][category]
            m = result['cells'][arm]['metrics'][category]['median_lead_to_0p5m_s']
            lines.append(f"| {arm}/{category} | {d['timely_before_final_drop']}/{d['n']} | {d['only_late']}/{d['n']} | {dec(m['value'])} [{dec(m['ci95'][0])}, {dec(m['ci95'][1])}] |")
    for arm in ARMS:
        c = result['cells'][arm]['censored']
        lines += ['', f"{arm}固定阈值={result['cells'][arm]['threshold']!r}；右删失{c['n']}条中已报警{c['already_alarm']}、尚未报警{c['no_alarm_right_censored']}；后者不计漏报。"]
    lines += ['', f"全13个采样时刻清晰{n['evaluation_clear_all']}条；其余{n['evaluation_not_all_clear']}条不计清晰暴露。帧间类别变化{n['evaluation_category_transition']}条，参考点清晰但曾非清晰{n['reference_clear_but_not_all_frames_clear']}条，末帧清晰但曾非清晰{n['last_clear_but_not_all_frames_clear']}条。", '',
        '区间为1000次整目标单位配对bootstrap，条件于源阈值，seed=2026100213。每查询序列最多计一次首报，停止后仍计满13×0.2=2.6秒代理暴露；采样首尾跨度2.4秒。全采样时刻清晰不保证帧间连续清晰，不代表用户级连续步行负担。', '',
        '仅σ0及既有观测帧，使用实际travel几何、目标前缘0.9m场景共同参照。提前量相对0.5m，按0.8m/s折算且条件于已报警；不是人的实际停步能力或每背景物体碰撞截止点。', '',
        '目标名义伸入范围−10至+15cm，不含源域−20至−10cm的同高度清晰目标；两域清晰组成不同，不能声称等支持范围的清晰负担迁移。', '',
        '源域与目标域结果分开，本轮为已消费Development上的冻结阈值迁移；不重训、不选择新时间策略、不改写旧失败。完整阈值来源、输入哈希、分母与区间见result.json。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    path = OUT/'result.json'
    if path.exists():
        prior = OE.read(path)
        if prior.get('status') != 'COMPLETE':
            raise RuntimeError('Existing incomplete evidence requires inspection; no overwrite')
        for name, digest in prior['provenance']['input_sha256'].items():
            if ME.sha(name) != digest:
                raise ValueError('Completed input changed: '+name)
        print('Existing COMPLETE result verified; no repeated evaluation', flush=True)
        return prior
    geometry, scores, thresholds, provenance = load_inputs()
    result = evaluate_arrays(geometry, scores, thresholds)
    result['provenance'] = provenance
    # Inputs are immutable throughout this evaluator; reject concurrent changes.
    for name, digest in provenance['input_sha256'].items():
        if ME.sha(name) != digest:
            raise ValueError('Input changed during evaluation: '+name)
    write_report(result)
    OE.save(path, result)
    print(result['verdict'], flush=True)
    return result


def check():
    import json
    from unittest.mock import patch
    assert decision([.001, .1], -.02, .2)[0] == 'TRANSFER_SUPPORTED_AT_FROZEN_THRESHOLDS'
    for ci, deep, clear in [([0, .1], 0., 0.), ([.01, .1], -.02001, 0.), ([.01, .1], 0., .20001)]:
        assert decision(ci, deep, clear)[0] == 'NOT_SUPPORTED_AT_FROZEN_THRESHOLDS'
    ref = np.array(['contact0-2cm', 'contact>5cm', 'pass0-10cm', 'clear', 'censored'])
    g = dict(split=np.array(['evaluation']*5), unit=np.array([94000, 94000, 94001, 94001, 94001]),
        frames=np.arange(3, 16), ref_category=ref, last_category=np.where(ref == 'censored', 'contact0-2cm', ref),
        covered=np.array([True]*4+[False]), clear_all=ref == 'clear',
        frame_ranges=np.tile(np.linspace(1.2, .7, 13), (5, 1)))
    g['frame_ranges'][-1] += .3
    g['frame_category'] = np.repeat(g['last_category'][:, None], 13, axis=1)
    scores = {a: np.zeros((5, 13)) for a in ARMS}
    scores['NEAR'][1, 1] = 1.
    scores['M3'][:3, 1] = 1.
    thresholds = {'NEAR': .3, 'M3': .7}
    # A trap proves the entire statistic path cannot recalibrate on target data.
    with patch.object(SE, 'calibrate', side_effect=AssertionError('Target calibration forbidden')):
        result = evaluate_arrays(g, scores, thresholds)
    assert {a: result['cells'][a]['threshold'] for a in ARMS} == thresholds
    assert result['n']['evaluation_censored'] == 1 and result['n']['target_calibration_units'] == 0
    assert result['comparison_M3_minus_NEAR']['contact0-2cm']['rescues'] == 1
    assert result['cells']['M3']['timing_diagnostics']['contact0-2cm']['timely_before_final_drop'] == 1
    assert 'episode_draw_pairs' not in result['cells']['M3']['metrics']['contact0-2cm']
    assert result['clear_comparison_M3_minus_NEAR']['delta_per_min'] == 0
    json.dumps(result, allow_nan=False)
    print('PASS: frozen thresholds/no target calibration, metadata separation, censoring, paired counts, clear delta and guards')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'evaluate'))
    args = parser.parse_args()
    check() if args.stage == 'check' else evaluate()
