"""Conditional RGB-clearance opportunity screen on consumed Development caches.

No rendering, training, inference or future-heading input to either policy.
Exact Gaussian integration replaces repeated pseudo-samples. This is a limited
vertical-side geometry oracle, not an achievable RGB or sensor upper bound.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
SOURCE = WORK / 'cnh-boundary-oracle-20261001'
OUT = WORK / 'cnh-rgb-threelevel-gate-20261002'
BINS = {'contact0-2': (0., .02), 'contact2-5': (.02, .05),
        'contact>5': (.05, .15), 'pass0-10': (-.10, 0.)}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(exist_ok=False)
    margins = WORK / 'cnh-margin-labels-20261002/result.json'
    save(OUT/'PLAN.json', dict(
        scope='consumed synthetic Development; same-table conditional opportunity, not confirmation',
        source=str(SOURCE.relative_to(ROOT)), units=[92000, 92095], rows=7680,
        model='cached deployed BASE five-seed ensemble', primary_budget=.10, sensitivity_budget=.20,
        budget='each query separately, all-distance other-height label=0 rows; near-pass unpenalized',
        primary_range=[1.2, 2.1], report_ranges=[[1.2, 1.6], [1.6, 2.1], [1.2, 2.1]],
        heading_deg=[0., 1.], clearance_sigma_cm=2.,
        integration='analytic independent Gaussian probabilities; heading maps target intrusion o to o+d*tan(delta)',
        observation='fixed nominal XZ all-vertical-side clearance plus independent N(0,2cm); no heading leakage',
        policies='per-query exact observed-score threshold scan; oracle also scans nominal clearance margins -30..20 cm step1 and bypass',
        selection='maximize pooled contact0-2 recall, then contact2-5, then lower clear, then lower margin index; one common margin for both queries',
        secondary='separately maximize contact2-5 over same family as conservative opportunity envelope; report policy, not a joint operating point',
        rule='At heading1 and budget10%, point PAUSE if best gains for BOTH contact0-2 and contact2-5 <3pp; else CONTINUE. Evidence-qualified PAUSE additionally requires both fixed-policy paired unit-bootstrap upper95 <3pp and valid support. Otherwise UNCERTAIN. Any >=3pp is an exploration opportunity only.',
        support='Uniform sampled nominal intrusion [-.10,.15]m. Primary-bin maximum missing kernel mass across bin endpoints and range endpoints must <=5%; not independent real observations.',
        bootstrap='1000 paired unit draws, seed2026100209, policies fixed after same-table selection; conditional only',
        margin_status='PENDING_NO_RESULT' if not margins.exists() else 'RESULT_EXISTS_REQUIRES_MATCHED_92000_SCORES',
        decision_scope='BASE threshold family only; if margin training HELPS, matched same-row scores required before final continuation decision',
        limits=['other-height clear is not same-height >10cm clear; latter absent',
                'heading changes target-relative truth only, not future collisions with background panels',
                'constant unmeasured heading distribution; no gait, replanning or stopping-time evidence',
                'Gaussian scalar error is an assumption, not RGB calibration',
                'contact>5 reports 5-15cm only due finite sampled support'],
        backend='TASK_NOT_GPU_SUITABLE: small cached scalar arrays on CPU',
        hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), SOURCE/'rows.npz', SOURCE/'PLAN.json',
            WORK/'cnh-detectability-envelope-20261001/base_ensemble.npz']}))


def bin_weights(offset, distance, lo, hi, sigma_deg):
    if sigma_deg == 0:
        return ((offset > lo) & (offset <= hi)).astype(float)
    sigma = np.deg2rad(sigma_deg)
    return ndtr(np.arctan((hi-offset)/distance)/sigma)-ndtr(np.arctan((lo-offset)/distance)/sigma)


def missing_mass(distance, actual, sigma_deg):
    if sigma_deg == 0:
        return 0.
    sigma = np.deg2rad(sigma_deg)
    # Prior nominal support [-.10,.15]; symmetric heading-induced displacement.
    return float(ndtr(np.arctan((-.10-actual)/distance)/sigma)
                 + ndtr(-np.arctan((.15-actual)/distance)/sigma))


def best_thresholds(data, veto, budget):
    """Lowest feasible threshold per query; nested score sets maximize every recall.

    Group all equal scores before evaluating budget; thresholds can use target
    score values but never target labels. Include disabled policy if necessary.
    """
    probability = np.zeros(len(veto)); thresholds = []; clear_rates = []
    for q in (0, 1):
        ids = np.flatnonzero(data['q'] == q)
        order = ids[np.argsort(-data['s'][ids], kind='stable')]
        scores = data['s'][order]
        clear = data['cat'][order] == 'clear'
        ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores)-1]
        burden = np.cumsum(veto[order]*clear)[ends] / clear.sum()
        feasible = np.flatnonzero(burden <= budget+1e-12)
        if len(feasible):
            k = feasible[-1]; end = ends[k]
            probability[order[:end+1]] = veto[order[:end+1]]
            thresholds.append(float(scores[end])); clear_rates.append(float(burden[k]))
        else:
            thresholds.append(None); clear_rates.append(0.)
    return probability, dict(thresholds=thresholds, comparator='>= (None disables query)', clear_per_query=clear_rates)


def fixtures():
    w = bin_weights(np.array([-.01, .01, .03]), np.ones(3), 0., .02, 0.)
    np.testing.assert_array_equal(w, [0, 1, 0])
    np.testing.assert_allclose(bin_weights(np.array([.01]), np.ones(1), 0., .02, 1.),
                               2*ndtr(np.arctan(.01)/np.deg2rad(1.))-1)
    d = dict(q=np.array([0, 0, 0, 1, 1, 1]), s=np.array([3., 2., 2., 3., 2., 2.]),
             cat=np.array(['contact', 'clear', 'clear']*2))
    p, _ = best_thresholds(d, np.ones(6), .5)
    np.testing.assert_array_equal(p, [1, 0, 0, 1, 0, 0])
    # Brute force every distinct score threshold, including ties and zero veto.
    rng = np.random.default_rng(23)
    for _ in range(30):
        d = dict(q=np.repeat([0, 1], 20), s=rng.integers(0, 8, 40).astype(float),
                 cat=np.array(['clear']*10+['contact']*10+['clear']*10+['contact']*10))
        v = rng.uniform(size=40); got, _ = best_thresholds(d, v, .2)
        for q in (0, 1):
            ids = d['q'] == q; clear = ids & (d['cat'] == 'clear')
            options = [v*ids*(d['s'] >= t) for t in [np.inf, *np.unique(d['s'][ids])]]
            valid = [p for p in options if p[clear].mean() <= .2+1e-12]
            np.testing.assert_allclose(got[ids].sum(), max(p.sum() for p in valid))
    return dict(heading_zero=True, heading_symmetry=True, threshold_ties=True, threshold_bruteforce=30)


def run():
    plan = json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'))
    for p, digest in plan['hashes'].items():
        assert sha(ROOT/p) == digest, f'input changed after plan: {p}'
    checks = fixtures()
    with np.load(SOURCE/'rows.npz') as z:
        data = {k: z[k] for k in z.files}
    assert np.array_equal(np.unique(data['unit']), np.arange(92000, 92096))
    assert len(data['s']) == 7680 and np.isfinite(data['s']).all()
    # Cache score association verified against the original deployed ensemble.
    with np.load(WORK/'cnh-detectability-envelope-20261001/base_ensemble.npz') as scores:
        for u in np.unique(data['unit']):
            ids = data['unit'] == u
            np.testing.assert_array_equal(data['s'][ids], scores[str(u)][data['config'][ids], data['q'][ids]])
    checks['source_score_rows_exact'] = len(data['s'])
    rng = np.random.default_rng(2026100209)
    boot = np.array([np.bincount(rng.integers(0, 96, 96), minlength=96) for _ in range(1000)])
    unit = data['ui']; target = np.isfinite(data['offset'])
    all_p = {}; ledger = {}; result = dict(plan=plan, checks=checks, runs={})
    margins = [None, *range(-30, 21)]
    for budget in (.10, .20):
        ps, defs = [], []
        for m in margins:
            v = np.ones(len(target)) if m is None else ndtr((m/100-data['vertical_sides_xz'])/.02)
            p, definition = best_thresholds(data, v, budget)
            definition['margin_cm'] = m; ps.append(p); defs.append(definition)
        ps = np.array(ps)
        all_p[str(budget)] = ps
        ledger[str(budget)] = defs
        for sig in (0., 1.):
            weights = {}; support = {}
            for lo_r, hi_r in plan['report_ranges']:
                mask = target & (data['range'] >= lo_r) & (data['range'] < hi_r)
                for name, (lo, hi) in BINS.items():
                    key = f'{lo_r}-{hi_r}|{name}'
                    w = np.zeros(len(target))
                    w[mask] = bin_weights(data['offset'][mask], data['range'][mask], lo, hi, sig)
                    weights[key] = w
                    support[key] = max(missing_mass(d, a, sig) for d in (lo_r, hi_r) for a in (lo, hi))
            keys = list(weights); ww = np.array(list(weights.values())); den = ww.sum(1)
            rates = ps @ ww.T / den
            primary = keys.index('1.2-2.1|contact0-2'); secondary = keys.index('1.2-2.1|contact2-5')
            best = min(range(len(ps)), key=lambda i: (-rates[i, primary], -rates[i, secondary], sum(defs[i]['clear_per_query']), i))
            other = min(range(len(ps)), key=lambda i: (-rates[i, secondary], -rates[i, primary], sum(defs[i]['clear_per_query']), i))
            summaries = {}
            for name, pi in [('ToF', 0), ('oracle_primary', best), ('oracle_secondary_envelope', other)]:
                metrics = {}
                for ki, key in enumerate(keys):
                    w = ww[ki]
                    cluster_den = np.bincount(unit, weights=w, minlength=96)
                    cluster_diff = np.bincount(unit, weights=w*(ps[pi]-ps[0]), minlength=96)
                    bs = (boot @ cluster_diff)/(boot @ cluster_den)*100
                    metrics[key] = dict(expected_n=float(den[ki]), source_rows=int(np.sum(w > 1e-12)),
                                        source_units=int(np.sum(cluster_den > 1e-12)),
                                        recall=float(rates[pi, ki]), delta_pp=float((rates[pi, ki]-rates[0, ki])*100),
                                        delta_pp_ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                                        maximum_missing_kernel_mass=support[key], support_valid=support[key] <= .05)
                summaries[name] = dict(policy_index=pi, policy=defs[pi], metrics=metrics)
            a = summaries['oracle_primary']['metrics']['1.2-2.1|contact0-2']
            b = summaries['oracle_secondary_envelope']['metrics']['1.2-2.1|contact2-5']
            point_pause = a['delta_pp'] < 3 and b['delta_pp'] < 3
            qualified = ('NOT_EVALUABLE' if not (a['support_valid'] and b['support_valid']) else
                         'CONTINUE_OPPORTUNITY' if not point_pause else
                         'PAUSE_BASE_FAMILY' if all(x['delta_pp_ci95'][1] < 3 for x in (a, b)) else 'UNCERTAIN')
            result['runs'][f'{budget}|{sig}'] = dict(arms=summaries,
                point_reading='PAUSE' if point_pause else 'CONTINUE', evidence_reading=qualified)
    result['decision'] = dict(base_only=result['runs']['0.1|1.0']['evidence_reading'],
        final_rgb_line='PENDING_MARGIN_RESULT_AND_MATCHED_SCORES',
        note='If widening is not supported, BASE-family screen applies; a supported widened arm requires a matched comparator. No physical follow-up started.')
    save(OUT/'policies.json', ledger)
    np.savez_compressed(OUT/'policy_probabilities.npz', **all_p)
    save(OUT/'result.json', result)
    report(result)
    print(json.dumps({k: {x: v[x] for x in ['point_reading', 'evidence_reading']} for k, v in result['runs'].items()}))


def report(result):
    lines = ['# RGB 净距三级真值机会检查', '',
             '已消费合成 Development；BASE 五种子缓存，92000–92095，共96单位、3840场景、7680查询行。',
             '另一高度带清晰行约束预算；身体外0–10cm报警不计代价。误差解析积分，概率质量不是新增独立样本。',
             '同表选择策略，区间为固定策略的条件单位bootstrap；不是确认性区间。', '',
             '|清晰预算|方向σ|0–2cm ToF→oracle|2–5cm ToF→同一oracle|2–5cm单独最优|判读|',
             '|---|---|---|---|---|---|']
    for key, run_ in result['runs'].items():
        budget, sig = key.split('|'); arms = run_['arms']
        m = lambda a, b: arms[a]['metrics'][f'1.2-2.1|{b}']['recall']*100
        lines.append(f"|{float(budget):.0%}|{sig}°|{m('ToF','contact0-2'):.2f}→{m('oracle_primary','contact0-2'):.2f}%|{m('ToF','contact2-5'):.2f}→{m('oracle_primary','contact2-5'):.2f}%|{m('oracle_secondary_envelope','contact2-5'):.2f}%|{run_['evidence_reading']}|")
    lines += ['', '## 主比较详细分母与条件区间', '', '|范围与区间|期望分母|源单位|ToF召回|oracle召回|增量pp与95%区间|', '|---|---|---|---|---|---|']
    arms = result['runs']['0.1|1.0']['arms']
    for key, m in arms['oracle_primary']['metrics'].items():
        if 'contact' not in key or not m['support_valid']:
            continue
        base = arms['ToF']['metrics'][key]
        ci = m['delta_pp_ci95']
        lines.append(f"|{key.replace(chr(124), chr(47))}|{m['expected_n']:.2f}|{m['source_units']}|{base['recall']:.2%}|{m['recall']:.2%}|{m['delta_pp']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]|")
    lines += ['', '## 决定与边界', '',
              f"BASE阈值族：`{result['decision']['base_only']}`。外扩训练尚未完成匹配比较，因此整条净距线决定暂记PENDING；不启动新52边缘机制或下载。",
              '竖直侧面XZ加2cm独立高斯误差仅是限定策略族的理想几何参考，含不可见面，无高度选择；不是真实RGB效果或全局数学上限。',
              'σheading=1°是假设压力条件；只移动目标接触标签，不向观察或阈值传递未来方向。没有重新计算背景panel与未来路径的交碰。',
              '本数据没有同高度身体外超过10cm的清晰负例；预算不能外推真实清晰误报。>5cm仅报告5–15cm，尾部截断须看result.json。',
              '主判读为1.2–2.1m合并；分段结果保留。2–5cm单独最优是另一策略的机会包络，不能冒充同一可用操作点。',
              '解析积分和全阈值扫描取代有限随机抽样、离散分位阈值，旧实验与失败结果保持。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run()
