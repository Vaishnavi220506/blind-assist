"""Read-only cached NEAR/M3 evaluation at an observed geometric deadline.

Sigma0 only. No rendering, inference, new temporal readout or changed old result.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_margin_confirm_sequence as CS
import cnh_three_level_sequence as SE

OUT = MC.SS.WORK/'cnh-observed-sequence-20261002'
ARMS = ('NEAR', 'M3')
CONTACTS = ('contact0-2cm', 'contact2-5cm', 'contact>5cm')
CATEGORIES = CONTACTS+('pass0-10cm', 'clear', 'censored')
BOOT_SEED = 2026100206
N_BOOT = 1000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def validate_geometry(g, rows):
    expected = [(s, u, c, q) for s, units in MC.SPLITS.items() for u in units for c in range(40) for q in (0, 1)]
    keys = list(zip(g['split'].tolist(), g['unit'].tolist(), g['config'].tolist(), g['query'].tolist()))
    row_keys = [(r['split'], r['unit'], r['config'], r['query']) for r in rows]
    if keys != expected or row_keys != expected:
        raise ValueError('Geometry and rows must have exact planned split/unit/config/query order')
    n = len(expected)
    if g['frame_ranges'].shape != (n, 13) or g['frame_category'].shape != (n, 13):
        raise ValueError('Expected exactly 13 observed frames per query episode')
    if not np.isfinite(g['frame_ranges']).all():
        raise ValueError('Nonfinite observed distance')
    if (np.diff(g['frame_ranges'], axis=1) > 1e-8).any():
        raise ValueError('Approach distances are not monotone; first-crossing semantics need inspection')
    for field in ('ref_category', 'clear_all', 'covered', 'last_category'):
        if g[field].shape != (n,):
            raise ValueError('Wrong geometry field shape: '+field)
    if not np.isin(g['ref_category'], CATEGORIES).all() or not np.isin(g['frame_category'], CATEGORIES[:-1]).all():
        raise ValueError('Unknown truth category')
    covered = (g['frame_ranges'][:, 0] >= .9) & (g['frame_ranges'] <= .9).any(1)
    if not np.array_equal(g['covered'], covered):
        raise ValueError('Covered must require observed first crossing of 0.9m')
    if not np.array_equal(g['clear_all'], (g['frame_category'] == 'clear').all(1)):
        raise ValueError('Clear exposure includes a contact/pass frame')
    if not np.array_equal(g['last_category'], g['frame_category'][:, -1]):
        raise ValueError('Last-frame truth mismatch')
    if not np.array_equal(g['ref_category'] == 'censored', ~covered):
        raise ValueError('Unreached reference must be censored; no extrapolated reference labels')


def load_inputs():
    plan_path, receipt_path = OUT/'PLAN.json', OUT/'geometry_receipt.json'
    plan, receipt = read(plan_path), read(receipt_path)
    if receipt.get('status') != 'COMPLETE':
        raise ValueError('Geometry must be COMPLETE before evaluation')
    hashes = {}
    for name, field in [('geometry.npz', 'geometry_sha256'), ('rows.json', 'rows_sha256'), ('PLAN.json', 'plan_sha256')]:
        digest = ME.sha(OUT/name)
        if receipt.get(field) != digest:
            raise ValueError('Geometry receipt hash mismatch: '+name)
        hashes[str(OUT/name)] = digest
    hashes[str(receipt_path)] = ME.sha(receipt_path)
    with np.load(OUT/'geometry.npz', allow_pickle=False) as data:
        geometry = {k: data[k] for k in data.files}
    rows = read(OUT/'rows.json')
    validate_geometry(geometry, rows)
    old_path = MC.OUT/'sequence_result.json'
    old = read(old_path)
    if old.get('status') != 'COMPLETE':
        raise ValueError('Original cached sequence receipt is incomplete')
    if old['units'] != MC.SPLITS:
        raise ValueError('Original sequence unit split differs')
    if old['provenance']['reused_sequence_sha256'] != ME.sha(SE.__file__):
        raise ValueError('Original causal-smoothing/statistic helper changed')
    hashes[str(old_path)] = ME.sha(old_path)
    scores, cache_receipts = {}, {}
    for arm in ARMS:
        names = [f'scores_{arm}.npz', f'frame_scores_{arm}.npz', f'frame_scores_{arm}_early.npz']
        for name in names:
            digest = ME.sha(MC.OUT/name)
            if digest != old['provenance']['input_sha256'][name]:
                raise ValueError('Original sequence cache changed: '+name)
            hashes[str(MC.OUT/name)] = digest
        with np.load(MC.OUT/names[0], allow_pickle=False) as final:
            scores[arm], cache_receipts[arm] = CS.frame_scores(MC.OUT, arm, MC.SPLITS, final)
        if scores[arm].shape != geometry['frame_ranges'].shape:
            raise ValueError('Geometry/score shape mismatch')
    return geometry, scores, dict(input_sha256=hashes, frame_cache_receipts=cache_receipts,
        evaluator_sha256=ME.sha(__file__), cache_helper_sha256=ME.sha(CS.__file__),
        smoothing_and_metrics_sha256=ME.sha(SE.__file__), plan=plan)


def verdict(ci, deep):
    return ('M3_TIMELY_SUPPORTED_DEV' if ci[0] is not None and ci[0] > 0 and deep is not None and deep >= -.02
            else 'NOT_SUPPORTED_IN_THIS_RECHECK')


def extra_counts(keep, stopped, timely, score, threshold):
    return dict(n=int(keep.sum()), stopped=int((keep & stopped).sum()), no_alarm=int((keep & ~stopped).sum()),
                timely=int((keep & timely).sum()), only_late=int((keep & stopped & ~timely).sum()),
                timely_before_final_drop=int((keep & timely & (score[:, -1] < threshold)).sum()))


def evaluate_arrays(g, scores):
    cal, ev = g['split'] == 'calib', g['split'] == 'evaluation'
    row_fields = ('unit', 'frame_ranges', 'frame_category', 'ref_category',
                  'clear_all', 'covered', 'last_category')
    if any(g[k].shape[0] != len(ev) for k in row_fields):
        raise ValueError('Required geometry row field does not align with split rows')
    e = {k: g[k][ev] for k in row_fields}
    units, ui = np.unique(e['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    weights = {name: (e['covered'] & (e['ref_category'] == name)).astype(float) for name in CONTACTS+('pass0-10cm',)}
    weights['clear'] = e['clear_all'].astype(float)
    cells, sampled, flags = {}, {}, {}
    for arm in ARMS:
        threshold, calibration = SE.calibrate(scores[arm], cal & g['clear_all'], 1.)
        score = scores[arm][ev]
        stopped, timely, lead = SE.first_stops(score, threshold, e['frame_ranges'])
        metrics, samples = SE.summarize(weights, e['covered'], stopped, timely, lead, ui, boot)
        # Sigma0 uses integer episodes, not heading draws; remove inherited draw-pair field.
        for m in metrics.values():
            m.pop('episode_draw_pairs', None)
            m['n'] = int(round(m['expected_episodes']))
            m['stops'] = int(round(m['expected_stops']))
            if 'expected_timely_stops' in m:
                m['timely_stops'] = int(round(m['expected_timely_stops']))
        diagnostics = {name: extra_counts(weights[name] > 0, stopped, timely, score, threshold)
                       for name in CONTACTS+('pass0-10cm',)}
        censored = ~e['covered']
        censored_counts = dict(n=int(censored.sum()), already_alarm=int((censored & stopped).sum()),
                               no_alarm_right_censored=int((censored & ~stopped).sum()),
                               interpretation='No-alarm right-censored episodes are not missed deadlines',
                               by_last_category={name: dict(n=int((censored & (e['last_category'] == name)).sum()),
                                  already_alarm=int((censored & (e['last_category'] == name) & stopped).sum()))
                                  for name in CATEGORIES[:-1]})
        cells[arm] = dict(threshold=threshold, calibration=calibration, metrics=metrics,
                         timing_diagnostics=diagnostics, censored=censored_counts)
        sampled[arm], flags[arm] = samples, dict(stopped=stopped, timely=timely)
    comparisons = {}
    for category in CONTACTS:
        a, b = (cells[arm]['metrics'][category]['timely_rate']['value'] for arm in ('M3', 'NEAR'))
        keep = weights[category] > 0
        nt, mt = flags['NEAR']['timely'], flags['M3']['timely']
        comparisons[category] = dict(delta_timely=a-b if a is not None and b is not None else None,
            paired_unit_ci95=SE.interval(sampled['M3'][category]-sampled['NEAR'][category]),
            n=int(keep.sum()), rescues=int((keep & mt & ~nt).sum()), losses=int((keep & nt & ~mt).sum()),
            both_timely=int((keep & nt & mt).sum()), neither_timely=int((keep & ~nt & ~mt).sum()))
    outcome = verdict(comparisons['contact0-2cm']['paired_unit_ci95'], comparisons['contact>5cm']['delta_timely'])
    transition = np.any(e['frame_category'] != e['frame_category'][:, :1], axis=1)
    counts = dict(calibration_units=int(len(np.unique(g['unit'][cal]))), evaluation_units=len(units),
        calibration_query_episodes=int(cal.sum()), evaluation_query_episodes=int(ev.sum()),
        evaluation_covered=int(e['covered'].sum()), evaluation_censored=int((~e['covered']).sum()),
        calibration_clear_all=int((cal & g['clear_all']).sum()), evaluation_clear_all=int(e['clear_all'].sum()),
        evaluation_not_all_clear=int((~e['clear_all']).sum()), evaluation_category_transition=int(transition.sum()),
        reference_clear_but_not_all_frames_clear=int(((e['ref_category'] == 'clear') & ~e['clear_all']).sum()),
        last_clear_but_not_all_frames_clear=int(((e['last_category'] == 'clear') & ~e['clear_all']).sum()),
        reference_category={name: int((e['ref_category'] == name).sum()) for name in CATEGORIES})
    return dict(status='COMPLETE', verdict=outcome, cells=cells, comparison_M3_minus_NEAR=comparisons,
        n=counts, evaluation_units=units.tolist(), bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED,
            cluster='whole evaluation units paired across arms', conditional_on='fixed calibration thresholds'),
        limits=['Consumed Development; new evaluation definition on existing source episodes, not new confirmation',
                'Sigma0 only; truth at actual recorded/interpolated travel pose, no score interpolation',
                'Target-front 0.9m is a common scene anchor, not each background surface collision deadline',
                '13 decision frames times0.2s=2.6s proxy exposure, including time after first stop; at most one first stop/query episode',
                'HEAD/BODY query exposure is not physical continuous walking time or user-level alarm burden',
                'Conditional lead=(target-front range at first alarm-0.5)/0.8 is nominal seconds, not actual stop latency',
                'Unobserved frames before3 are excluded; censored no-alarm episodes are not missed deadlines',
                'Old failed and successful run results remain unchanged'])


def rate_text(metric, clear=False, stop=False):
    if clear:
        n, d, rate = metric['expected_stops'], metric['clear_minutes'], metric['false_stops_per_min']
        scale, suffix = 1., '次/分钟'
    else:
        n = metric['expected_stops' if stop else 'expected_timely_stops']
        d, rate = metric['expected_episodes'], metric['stop_rate' if stop else 'timely_rate']
        scale, suffix = 100., '%'
    show = lambda v: '--' if v is None else f'{scale*v:.2f}'
    denominator = f'{d:.2f}' if clear else f'{d:g}'
    return f"{n:g}/{denominator} = {show(rate['value'])}{suffix} [{show(rate['ci95'][0])}, {show(rate['ci95'][1])}]"


def write_report(result):
    n = result['n']
    lines = [result['verdict'], '', '# 实际观察截止点与全观察窗清晰序列复核', '',
        '只复用 NEAR/M3 的既有13帧五种子分数及原因果平滑；48单位校准每臂单一HEAD/BODY阈值，预算1次清晰首停/代理分钟，96单位评估。',
        f"观察窗已覆盖0.9m截止点 {n['evaluation_covered']}/{n['evaluation_query_episodes']} 条查询序列；未到截止点 {n['evaluation_censored']} 条右删失，不算漏报。", '',
        '| 指标 | NEAR | M3 |', '| --- | --- | --- |']
    for category in CONTACTS+('pass0-10cm', 'clear'):
        lines.append('| '+category+' | '+' | '.join(rate_text(result['cells'][a]['metrics'][category], category == 'clear', category == 'pass0-10cm') for a in ARMS)+' |')
    lines += ['', '主比较为已覆盖截止点的0–2cm及时率M3−NEAR配对单位95%区间下界>0，且>5cm及时率下降≤2pp；清晰实际负担完整报告。', '',
              '| 擦碰 | 及时率差及95%区间 | 补回 / 丢失 |', '| --- | --- | --- |']
    fmt = lambda v: '--' if v is None else f'{100*v:+.2f}pp'
    for name, d in result['comparison_M3_minus_NEAR'].items():
        lines.append(f"| {name} | {fmt(d['delta_timely'])} [{fmt(d['paired_unit_ci95'][0])}, {fmt(d['paired_unit_ci95'][1])}] | {d['rescues']} / {d['losses']}（分母{d['n']}） |")
    lines += ['', '| 臂/类别 | 及时首报但末帧已低于阈值 | 仅迟报 | 已报警者提前量中位数及95%区间（秒） |', '| --- | --- | --- | --- |']
    seconds = lambda value: '--' if value is None else f'{value:.2f}'
    for arm in ARMS:
        cell = result['cells'][arm]
        for category in CONTACTS:
            d = cell['timing_diagnostics'][category]
            m = cell['metrics'][category]['median_lead_to_0p5m_s']
            lines.append(f"| {arm}/{category} | {d['timely_before_final_drop']}/{d['n']} | {d['only_late']}/{d['n']} | {seconds(m['value'])} [{seconds(m['ci95'][0])}, {seconds(m['ci95'][1])}] |")
    for arm in ARMS:
        c = result['cells'][arm]['censored']
        lines += ['', f"{arm}：右删失{c['n']}条中，截至末帧已报警{c['already_alarm']}，尚未报警{c['no_alarm_right_censored']}；后者不计漏报。"]
    lines += ['', f"全观察窗清晰评估序列 {n['evaluation_clear_all']}；非全清晰 {n['evaluation_not_all_clear']}；帧间类别变化 {n['evaluation_category_transition']}；参考点清晰但曾非清晰 {n['reference_clear_but_not_all_frames_clear']}，末帧清晰但曾非清晰 {n['last_clear_but_not_all_frames_clear']}。后两类不计清晰暴露。", '',
        '区间为1000次整单位配对bootstrap，条件于固定校准阈值。这里全观察窗清晰只表示13个已观测采样时刻均清晰，不保证帧间连续时间清晰。每查询序列最多计一次首停，始终计满13×0.2=2.6秒代理暴露；13采样时刻首尾实际跨度2.4秒。不是用户级连续步行误停率。', '',
        '仅σ0。几何距离使用实际travel变换后的目标前缘，0.9m为场景共同参照，不是每个背景物体的独立碰撞截止点。提前量=(首报距离−0.5)/0.8为已报警条件下的名义秒数，不是人实际制动时间。', '',
        '本轮为已消费Development上的定义修正复核，无新渲染、推理或训练；不改写旧判读，不使用未观测帧/分数插值。完整分母、区间、缓存哈希见result.json。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def run():
    if (OUT/'result.json').exists():
        old = read(OUT/'result.json')
        if old.get('status') != 'COMPLETE':
            raise RuntimeError('Inspect existing incomplete output; no silent overwrite')
        for path, digest in old['provenance']['input_sha256'].items():
            if ME.sha(path) != digest:
                raise ValueError('Completed run input changed: '+path)
        print('Existing COMPLETE result verified; no repeated evaluation', flush=True)
        return old
    geometry, scores, provenance = load_inputs()
    result = evaluate_arrays(geometry, scores)
    result['provenance'] = provenance
    write_report(result)
    save(OUT/'result.json', result)
    print(result['verdict'], flush=True)
    return result


def check():
    assert verdict([.001, .1], -.02) == 'M3_TIMELY_SUPPORTED_DEV'
    assert verdict([0., .1], 0.) == 'NOT_SUPPORTED_IN_THIS_RECHECK'
    assert verdict([.01, .1], -.02001) == 'NOT_SUPPORTED_IN_THIS_RECHECK'
    ranges = np.array([[1.2, 1., .8], [1.3, 1.1, .95], [1.2, 1., .8], [1.2, 1., .8]])
    score = np.array([[0., 2., 0.], [0., 0., 0.], [0., 0., 2.], [0., 0., 0.]])
    stopped, timely, lead = SE.first_stops(score, 1., ranges)
    covered = (ranges[:, 0] >= .9) & (ranges <= .9).any(1)
    assert covered.tolist() == [True, False, True, True]
    counts = extra_counts(covered, stopped, timely, score, 1.)
    assert counts['n'] == 3 and counts['timely_before_final_drop'] == 1 and counts['only_late'] == 1
    assert counts['no_alarm'] == 1  # censored unalarmed row is not a missed deadline
    assert np.isclose(lead[0], .625)
    categories = np.array([['clear', 'clear'], ['pass0-10cm', 'clear'], ['contact0-2cm', 'clear']])
    assert (categories == 'clear').all(1).tolist() == [True, False, False]
    # Full statistic path on a tiny synthetic cohort; never reads production scores.
    ref = np.array(['clear', 'clear', 'contact0-2cm', 'contact>5cm', 'pass0-10cm', 'clear', 'censored'])
    g = dict(split=np.array(['calib']*2+['evaluation']*5), unit=np.array([1, 2, 3, 3, 4, 4, 4]),
             frames=np.arange(3, 16),
             ref_category=ref, last_category=np.where(ref == 'censored', 'contact0-2cm', ref),
             covered=np.array([True]*6+[False]), clear_all=ref == 'clear',
             frame_ranges=np.tile(np.linspace(1.2, .7, 13), (7, 1)))
    g['frame_ranges'][-1] += .3
    g['frame_category'] = np.repeat(g['last_category'][:, None], 13, axis=1)
    scores = {a: np.zeros((7, 13)) for a in ARMS}
    scores['NEAR'][3, 1] = 1.
    scores['M3'][2:5, 1] = 1.
    result = evaluate_arrays(g, scores)
    assert result['n']['evaluation_censored'] == 1
    assert result['cells']['M3']['metrics']['contact0-2cm']['timely_stops'] == 1
    assert result['comparison_M3_minus_NEAR']['contact0-2cm']['rescues'] == 1
    assert 'episode_draw_pairs' not in result['cells']['M3']['metrics']['contact0-2cm']
    json.dumps(result, allow_nan=False)
    print('PASS: deadline coverage/censoring, timely-then-drop, late-only, all-frame clear and decision guards')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'run'))
    args = parser.parse_args()
    check() if args.stage == 'check' else run()
