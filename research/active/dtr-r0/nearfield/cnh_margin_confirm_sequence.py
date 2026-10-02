"""Conditional descriptive sequence follow-up after M3_CONFIRMED; no inference.

Reuses the fixed final-frame comparison's new units and the early-frame cache.
This description never changes the primary confirmation or old failed verdicts.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_three_level_sequence as SE

BUDGET = 1.
BOOT_SEED = 2026100206
N_BOOT = 1000


def merge_frames(early, late):
    early, late = np.asarray(early), np.asarray(late)
    if early.shape != (40, 8, 2) or late.shape != (40, 5, 2):
        raise ValueError('Expected per-unit raw early (40,8,2), late (40,5,2) scores')
    raw = np.concatenate([early, late], axis=1)
    if not np.isfinite(raw).all():
        raise ValueError('Frame cache contains non-finite scores')
    return SE.smooth(raw[None])[0]


def frame_scores(out, arm, units, final_scores):
    paths = [out/f'frame_scores_{arm}_early.npz', out/f'frame_scores_{arm}.npz']
    joined = units['calib']+units['evaluation']
    values, largest_error = [], 0.
    with np.load(paths[0], allow_pickle=False) as early, np.load(paths[1], allow_pickle=False) as late:
        expected = set(map(str, joined))
        if set(early.files) != expected or set(late.files) != expected:
            raise ValueError(f'{arm}: frame cache does not exactly match planned units')
        for u in joined:
            smoothed = merge_frames(early[str(u)], late[str(u)])
            discrepancy = float(np.max(np.abs(smoothed[:, -1]-final_scores[str(u)])))
            largest_error = max(largest_error, discrepancy)
            values.append(smoothed.transpose(0, 2, 1).reshape(80, 13))
    if largest_error >= 1e-4:
        raise ValueError(f'{arm}: sequence/final score mismatch {largest_error}')
    return np.concatenate(values), dict(input_sha256={p.name: ME.sha(p) for p in paths},
                                       final_score_max_abs_error=largest_error)


def truth_weights(offsets, labels, z, sigma):
    actual = offsets[:, None]+SE.REFERENCE_M*np.tan(np.deg2rad(sigma)*z)
    classes = ME.classifications(offsets, labels, actual)
    names = {'contact0-2cm': 'contact_0_2cm', 'contact2-5cm': 'contact_2_5cm',
             'contact>5cm': 'contact_gt5cm', 'pass0-10cm': 'pass_0_10cm', 'clear': 'clear_merged'}
    return {a: classes[b] for a, b in names.items()}, classes


def clear_breakdown(classes, stopped, ui, boot):
    result = {}
    for name in ('clear_other_height', 'clear_same_height_gt10cm'):
        weights = classes[name]
        den = SE.unit_totals(weights, ui, boot.shape[1])
        num = SE.unit_totals(weights*stopped, ui, boot.shape[1])
        minutes = den*len(SE.GA.ALL)*SE.GA.FRAME_S/60
        rate, _ = SE.rates(num, minutes, boot)
        result[name] = dict(expected_episodes=float(den.sum()), expected_stops=float(num.sum()),
                            clear_minutes=float(minutes.sum()), false_stops_per_min=rate,
                            contributing_episodes=int((weights > 0).sum()),
                            contributing_units=int((den > 0).sum()))
    return result


def fmt(metric, category):
    if category == 'clear':
        n, d, rate = metric['expected_stops'], metric['clear_minutes'], metric['false_stops_per_min']
        scale, suffix = 1., '次/分钟'
    else:
        stop = category == 'pass0-10cm'
        n = metric['expected_stops' if stop else 'expected_timely_stops']
        d, rate = metric['expected_episodes'], metric['stop_rate' if stop else 'timely_rate']
        scale, suffix = 100., '%'
    val = lambda x: '--' if x is None else f'{scale*x:.2f}'
    return f"{n:.3f}/{d:.3f} = {val(rate['value'])}{suffix} [{val(rate['ci95'][0])}, {val(rate['ci95'][1])}]"


def write_report(result, path):
    lines = [result['primary_verdict']+'; DESCRIPTIVE', '', '# 新单位三级序列停步描述', '',
             '固定 NEAR、M3、M8；48 单位校准每臂一个合并 HEAD/BODY 阈值，预算为每分钟 1 次合并清晰误停；96 单位评估。',
             '13 帧使用因果五分数平滑。擦碰/擦身指标限末帧距离≤1m的走近子集；及时指距离≥0.9m时报警，不是人已经停住。',
             '清晰为另一高度 label=0 与同高度身体外>10cm的并集，使用全距离序列；另一高度 label>0 不列为清晰。',
             '1000 次配对整单位 bootstrap 95% 区间，条件于固定校准阈值；本描述不改变主判读。', '']
    for sigma in ('0.0', '1.0'):
        lines += [f'## σ={sigma}°', '', '| 指标 | NEAR | M3 | M8 |', '| --- | --- | --- | --- |']
        for category in ('contact0-2cm', 'contact2-5cm', 'contact>5cm', 'pass0-10cm', 'clear'):
            lines.append('| '+category+' | '+' | '.join(fmt(result['cells'][f'{a}|{sigma}']['metrics'][category], category) for a in MC.ARMS)+' |')
        lines += ['', '| 浅擦碰已报警者提前量中位数，秒 | NEAR | M3 | M8 |', '| --- | --- | --- | --- |']
        entries = []
        for arm in MC.ARMS:
            m = result['cells'][f'{arm}|{sigma}']['metrics']['contact0-2cm']['median_lead_to_0p5m_s']
            entries.append(f"{m['value']} [{m['ci95'][0]}, {m['ci95'][1]}]")
        lines.append('| 相对0.5m、按0.8m/s | '+' | '.join(entries)+' |')
    lines += ['', '## 边界', '',
              '既有已消费 Development 配方下的新仿真单位；不是保护集、实机或人实际停步证据。',
              'σ1 在统一0.9m截止点按名义伸入+0.9×tan(恒定方向误差)改变真值；400共同draw的期望分母不是新增独立样本，观测及阈值不变。',
              '清晰分钟为13帧×0.2秒的序列×查询暴露，每段至多计一次首停，停止后仍计全段暴露；不能换算现场连续步行负担。',
              '提前量条件于已经报警，相对到达0.5m按0.8m/s计算，不含人的反应与制动。名义伸入支持−20至+15cm，开放端点统计受支持限制。',
              '完整分母、区间、清晰组成、配对差、缓存一致性及输入哈希见 sequence_result.json。', '']
    path.write_text('\n'.join(lines), encoding='utf8')


def run(out=None):
    out = MC.OUT if out is None else Path(out)
    primary_path = out/'result.json'
    primary = json.loads(primary_path.read_text(encoding='utf8'))
    if primary.get('status') != 'COMPLETE' or primary.get('verdict') != 'M3_CONFIRMED':
        raise RuntimeError('Sequence follow-up is authorized only after COMPLETE M3_CONFIRMED')
    destination = out/'sequence_result.json'
    if destination.exists():
        previous = json.loads(destination.read_text(encoding='utf8'))
        if previous.get('status') != 'COMPLETE':
            raise RuntimeError('Existing incomplete sequence result requires inspection')
        for name, digest in previous['provenance']['input_sha256'].items():
            if ME.sha(out/name) != digest:
                raise ValueError('Completed sequence input changed: '+name)
        if not (out/'sequence_REPORT.md').exists():
            write_report(previous, out/'sequence_REPORT.md')
        print('Existing COMPLETE sequence result verified; no repeated evaluation', flush=True)
        return previous
    rows, units, hashes = ME.load_rows(out, MC.SPLITS)
    for name, digest in primary['provenance']['input_sha256'].items():
        if hashes.get(name) != digest:
            raise ValueError('Primary/sequence input mismatch: '+name)
    if ME.sha(out/'PLAN.json') != primary['provenance']['plan_sha256']:
        raise ValueError('Primary plan changed')
    hashes.update({'result.json': ME.sha(primary_path), 'PLAN.json': ME.sha(out/'PLAN.json')})
    cal, ev = rows['split'] == 'calib', rows['split'] == 'evaluation'
    clear_cal = cal & (ME.classifications(rows['off'], rows['label'])['clear_merged'] > 0)
    offsets, labels = rows['off'][ev], rows['label'][ev]
    z = np.zeros((len(offsets), SE.DRAWS))
    target = np.isfinite(offsets)
    z[target] = np.random.default_rng(SE.HEADING_SEED).standard_normal((int(target.sum()), SE.DRAWS))
    truths = {s: truth_weights(offsets, labels, z, s) for s in (0., 1.)}
    lookup = {u: i for i, u in enumerate(units['evaluation'])}
    ui = np.asarray([lookup[int(u)] for u in rows['unit'][ev]])
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(lookup), size=len(lookup)), minlength=len(lookup)) for _ in range(N_BOOT)])
    ranges = rows['range'][ev, None]+SE.GA.STEP*(15-SE.GA.ALL)
    approach = rows['range'][ev] <= 1.
    cells, sampled, cache_provenance = {}, {}, {}
    for arm in MC.ARMS:
        with np.load(out/f'scores_{arm}.npz', allow_pickle=False) as final:
            scores, receipt = frame_scores(out, arm, units, final)
        cache_provenance[arm] = receipt
        hashes.update(receipt['input_sha256'])
        threshold, calibration = SE.calibrate(scores, clear_cal, BUDGET)
        stopped, timely, lead = SE.first_stops(scores[ev], threshold, ranges)
        for sigma, (weights, classes) in truths.items():
            metrics, samples = SE.summarize(weights, approach, stopped, timely, lead, ui, boot)
            if sigma == 0:
                for m in metrics.values():
                    m['n'] = int(round(m['expected_episodes']))
                    m['stops'] = int(round(m['expected_stops']))
                    if 'expected_timely_stops' in m:
                        m['timely_stops'] = int(round(m['expected_timely_stops']))
            key = f'{arm}|{sigma}'
            cells[key] = dict(threshold=threshold, calibration=calibration, metrics=metrics,
                              clear_components=clear_breakdown(classes, stopped, ui, boot))
            sampled[key] = samples
        print('sequence analyzed', arm, flush=True)
    comparisons = {}
    for sigma in truths:
        for arm in ('M3', 'M8'):
            key, base = f'{arm}|{sigma}', f'NEAR|{sigma}'
            comparison = {}
            for category in ('contact0-2cm', 'contact2-5cm', 'contact>5cm'):
                a, n = (cells[k]['metrics'][category]['timely_rate']['value'] for k in (key, base))
                comparison[category] = dict(delta_timely=a-n if a is not None and n is not None else None,
                    paired_unit_ci95=SE.interval(sampled[key][category]-sampled[base][category]))
            comparisons[key] = dict(role='DESCRIPTIVE_ONLY; no sequence confirmation verdict', metrics=comparison)
    result = dict(status='COMPLETE', primary_verdict=primary['verdict'], role='DESCRIPTIVE_ONLY', cells=cells,
        comparisons=comparisons, frame_scores=cache_provenance, units=units,
        n=dict(calibration_units=len(units['calib']), evaluation_units=len(lookup),
               evaluation_query_episodes=int(ev.sum()), approach_query_episodes=int(approach.sum()),
               other_height_nonzero_label_excluded_from_clear=int((~target & (labels != 0)).sum())),
        protocol=dict(budget_clear_stops_per_min=BUDGET, thresholds='one per arm, pooled HEAD/BODY, nominal calibration clear; held fixed for sigma1',
            frames=SE.GA.ALL.tolist(), frame_seconds=SE.GA.FRAME_S, smoothing_weights=SE.SS.WEIGHTS.tolist(),
            approach_final_range_max_m=1., timely_min_range_m=SE.GA.SAFE, truth_reference_m=SE.REFERENCE_M,
            heading_draws=SE.DRAWS, heading_seed=SE.HEADING_SEED, bootstrap_seed=BOOT_SEED, unit_bootstraps=N_BOOT,
            heading='nominal intrusion + 0.9*tan(delta), one fixed delta per sequence/draw; label sensitivity only',
            exposure='full 13-frame sequence/query; maximum one first stop; not real walking minutes',
            lead='(first alarm nominal distance-0.5)/0.8 seconds, conditional on stopping'),
        provenance=dict(input_sha256=hashes, evaluator_sha256=ME.sha(__file__), reused_sequence_sha256=ME.sha(SE.__file__),
                        reused_classification_sha256=ME.sha(ME.__file__)),
        limits=['New simulator units under consumed Development recipe; not protected confirmation or hardware evidence',
                'Primary M3/M8 verdicts and old failed conclusions remain unchanged',
                'Intervals resample evaluation units only; no threshold-calibration, retraining, or simulator uncertainty',
                'Other-height labels use the fixed final-frame manifest; nonzero labels are excluded from clear',
                'Bounded nominal offset support [-0.20,+0.15] m; heading is assumed, not measured'])
    write_report(result, out/'sequence_REPORT.md')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(result['primary_verdict']+'; DESCRIPTIVE sequence COMPLETE', flush=True)
    return result


def check():
    from contextlib import nullcontext
    from unittest.mock import patch

    early = np.broadcast_to(np.arange(3., 11.)[None, :, None], (40, 8, 2))
    late = np.broadcast_to(np.arange(11., 16.)[None, :, None], (40, 5, 2))
    smooth = merge_frames(early, late)
    assert smooth.shape == (40, 13, 2) and smooth[0, 0, 0] == 3.
    assert np.isclose(smooth[0, -1, 0], np.dot(np.arange(11., 16.), SE.SS.WEIGHTS)/SE.SS.WEIGHTS.sum())
    class FixtureCache(dict):
        @property
        def files(self):
            return list(self)
    early_cache = FixtureCache({'95000': early, '96000': early+100})
    late_cache = FixtureCache({'95000': late, '96000': late+100})
    final_cache = {'95000': smooth[:, -1], '96000': smooth[:, -1]+100}
    with patch.object(np, 'load', side_effect=[nullcontext(early_cache), nullcontext(late_cache)]), patch.object(ME, 'sha', return_value='synthetic'):
        joined, receipt = frame_scores(Path('synthetic'), 'NEAR', {'calib': [95000], 'evaluation': [96000]}, final_cache)
    assert joined.shape == (160, 13) and np.allclose(joined[80:]-joined[:80], 100)
    assert receipt['final_score_max_abs_error'] < 1e-4
    offsets = np.array([-.2, -.1, 0., .01, np.nan, np.nan])
    labels = np.array([0, 0, 0, 1, 0, 1])
    weights, _ = truth_weights(offsets, labels, np.ones((6, SE.DRAWS)), 0.)
    assert weights['clear'].tolist() == [1, 0, 0, 0, 1, 0]
    assert weights['pass0-10cm'].tolist() == [0, 1, 1, 0, 0, 0]
    shifted, _ = truth_weights(offsets, labels, np.ones((6, SE.DRAWS)), 1.)
    assert shifted['contact2-5cm'][3] == 1 and shifted['clear'][5] == 0
    score = np.repeat(np.arange(6.)[:, None], 13, axis=1)
    threshold, receipt = SE.calibrate(score, weights['clear'] > 0, 1.)
    assert receipt['clear_episodes'] == 2 and receipt['stops'] == 0 and threshold > 4
    print('PASS: early/late frame ordering, causal smoothing parity, merged clear membership and calibration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'evaluate'))
    args = parser.parse_args()
    check() if args.stage == 'check' else run()
