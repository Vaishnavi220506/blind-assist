"""Calibrate-only selection of ideal XZ policies after same-unit M3 confirmation.

Uses existing metadata and score caches; no rendering, model inference or RGB
mechanism. Heading scenarios recalibrate symmetric weighted clear budgets.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

import cnh_rgb_threelevel_gate as G
import cnh_rgb_confirm_rows as R

OUT = G.WORK / 'cnh-rgb-confirmed-m3-gate-20261002'
SOURCE = G.WORK / 'cnh-margin-confirm-20261002'
MODELS = ['NEAR', 'M3', 'M8']
MARGINS = [None, *range(-30, 21)]
BINS = ['contact0-2', 'contact2-5']


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def prepare():
    assert not (OUT / 'PLAN.json').exists()
    row = (OUT / 'prerun-row.txt').read_text(encoding='utf-8-sig').strip()
    assert row in (G.ROOT / 'research/active/dtr-r0/RUNS.md').read_text(encoding='utf-8')
    assert read(SOURCE / 'rgb_handoff.json')['primary_verdict'] == 'M3_CONFIRMED'
    files = [Path(__file__), Path(G.__file__), Path(R.__file__),
             SOURCE / 'PLAN.json', SOURCE / 'rgb_handoff.json', SOURCE / 'scene_manifest.json',
             SOURCE / 'result.json', OUT / 'prerun-row.txt']
    files += [SOURCE / f'scores_{a}.npz' for a in MODELS]
    files += [SOURCE / f'scores96000_{a}.npz' for a in MODELS]
    files += [Path(__file__).with_name(n) for n in ['cnh_boundary_oracle_geometry.py',
              'cnh_margin_confirm_evaluate.py', 'cnh_cvr_pilot.py', 'cnh_proposal_attribution_scenes.py']]
    files = list(dict.fromkeys(files + R.input_paths(SOURCE)))
    G.save(OUT / 'PLAN.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        prerun_row=row, models=MODELS, primary_model='M3', margins_cm=MARGINS,
        budgets=[.1, .2], heading_deg=[0., 1.], clearance_sigma_cm=2.,
        splits=read(SOURCE / 'PLAN.json')['splits'], primary_range=[1.2, 2.1], clear_range=[.6, 2.6],
        selection='CALIB ONLY: exact score breakpoints per query, maximize contact0-2 then contact2-5 then lower merged-clear burden then index. Separate contact2-5 policy swaps first two objectives. One common margin across queries. Evaluator never selects.',
        truth='Analytic heading label probabilities for both contacts and same-height clear actual offset<-.10m; other-height label0 stays clear. Neither actual future heading nor shifted clearance enters observations.',
        calibration='Each heading scenario separately calibrates its expected merged-clear budget on48 units, not sigma0-fixed sensitivity. No claim actual evaluation clear rates are equal.',
        rule='At10%/heading1, both CALIB-selected single-bin M3 oracle policies have eval gain<3pp => PAUSED_AFTER_TOF_CONFIRMATION; otherwise CONTINUE_OPPORTUNITY. Invalid primary support => NOT_EVALUABLE. No CI or posthoc guard threshold added.',
        family_secondary='Choose ONE ToF model oncalib, ONE oracle model/margin oncalib; report eval comparison, never per-row switching or evaluation-based model selection.',
        support='Nominal offsets uniform[-.20,.15]; primary-bin missing mass<=5% as prior diagnostic. Same-height clear is finite source sampling, not deployment prevalence.',
        bootstrap='1000 paired96-unit resamples seed2026100209, conditional on fixed calib selection.',
        limits=['Consumed synthetic Development in one generator family, not protected or hardware confirmation',
                'Ideal XZ uses all vertical box surfaces including occluded surfaces, ignores height',
                'Same-height clear follows target offset; background panel clearance and future collisions are not redefined',
                'Separate-bin optimum policies are not one deployable operating point'],
        hashes={str(p.relative_to(G.ROOT)): G.sha(p) for p in files}))
    print('Prepared: prerequisite RUNS row and M3 confirmation verified.')


def memberships(data, sig):
    off, distance = data['off'], data['range']
    target = np.isfinite(off)
    result = {}
    for name, (lo, hi) in G.BINS.items():
        w = np.zeros(len(off))
        w[target] = G.bin_weights(off[target], distance[target], lo, hi, sig)
        result[name] = w
    same = np.zeros(len(off))
    same[target] = ((off[target] < -.10).astype(float) if sig == 0 else
                    ndtr(np.arctan((-.10 - off[target]) / distance[target]) / np.deg2rad(sig)))
    other = ((~target) & (data['label'] == 0)).astype(float)
    result.update(clear_same=same, clear_other=other, clear_merged=same + other)
    return result


def fit_thresholds(scores, veto, qids, clear, budget):
    """Inputs are calibration rows only; select lowest feasible tied score."""
    thresholds, rates = [], []
    for q in (0, 1):
        ids = np.flatnonzero(qids == q)
        order = ids[np.argsort(-scores[ids], kind='stable')]
        values = scores[order]
        ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values)-1]
        denominator = clear[ids].sum()
        assert denominator > 0
        burden = np.cumsum(veto[order] * clear[order])[ends] / denominator
        feasible = np.flatnonzero(burden <= budget + 1e-12)
        if len(feasible):
            k = feasible[-1]
            thresholds.append(float(values[ends[k]])); rates.append(float(burden[k]))
        else:
            thresholds.append(None); rates.append(0.)
    return thresholds, rates


def apply_thresholds(scores, veto, qids, thresholds):
    p = np.zeros(len(scores))
    for q, threshold in enumerate(thresholds):
        if threshold is not None:
            mask = (qids == q) & (scores >= threshold)
            p[mask] = veto[mask]
    return p


def choose(cal_rates, indices, primary, secondary, defs):
    return min(indices, key=lambda i: (-cal_rates[i, primary], -cal_rates[i, secondary],
                                      sum(defs[i]['calibration_clear_rates']), i))


def missing_mass(actual, distance, sig):
    if sig == 0:
        return 0.
    return float(ndtr(np.arctan((-.20 - actual) / distance) / np.deg2rad(sig)) +
                 ndtr(-np.arctan((.15 - actual) / distance) / np.deg2rad(sig)))


def fixtures():
    scores = np.array([3., 2., 2., 3., 2., 2.])
    q = np.repeat([0, 1], 3)
    w = np.array([0., .25, .75, 0., .25, .75])
    t, rates = fit_thresholds(scores, np.ones(6), q, w, .5)
    np.testing.assert_array_equal(apply_thresholds(scores, np.ones(6), q, t), [1, 0, 0, 1, 0, 0])
    assert rates == [0., 0.]
    d = dict(off=np.array([-.10, -.10, np.nan]), range=np.ones(3), label=np.zeros(3))
    np.testing.assert_allclose(memberships(d, 1)['clear_merged'], [.5, .5, 1.])
    return dict(weighted_threshold_ties=True, heading_moves_same_height_clear=True)


def run():
    plan = read(OUT / 'PLAN.json')
    assert not (OUT / 'result.json').exists()
    for rel, digest in plan['hashes'].items():
        assert G.sha(G.ROOT / rel) == digest, rel
    checks = fixtures()
    data, units, receipt = R.load_rows(SOURCE, plan['splits'])
    assert data['scores'].shape == (11520, 3)
    G.save(OUT / 'input-check.json', receipt)
    np.savez_compressed(OUT / 'rows.npz', **data)
    cal = data['split'] == 'calib'; ev = data['split'] == 'evaluation'
    assert cal.sum() == 3840 and ev.sum() == 7680 and not np.any(cal & ev)
    inrange = (data['range'] >= 1.2) & (data['range'] < 2.1)
    ui = np.searchsorted(units['evaluation'], data['unit'][ev])
    rng = np.random.default_rng(2026100209)
    boot = np.array([np.bincount(rng.integers(0, 96, 96), minlength=96) for _ in range(1000)])
    result = dict(plan_sha256=G.sha(OUT / 'PLAN.json'), checks=checks,
                  counts=dict(calibration_rows=int(cal.sum()), evaluation_rows=int(ev.sum()), evaluation_units=96), runs={})
    all_p, all_defs = {}, {}
    for sig in plan['heading_deg']:
        truth = memberships(data, sig)
        contact_weights = np.array([truth[b] * inrange for b in BINS])
        support = [max(missing_mass(a, d, sig) for a in G.BINS[b] for d in (1.2, 2.1)) for b in BINS]
        for budget in plan['budgets']:
            key = f'{budget}|{sig}'
            definitions, pp = [], []
            for mi, model in enumerate(MODELS):
                score = data['scores'][:, mi]
                for margin in MARGINS:
                    veto = np.ones(len(cal)) if margin is None else ndtr((margin/100 - data['vertical_sides_xz'])/.02)
                    thresholds, clear_rates = fit_thresholds(score[cal], veto[cal], data['group'][cal], truth['clear_merged'][cal], budget)
                    p = apply_thresholds(score, veto, data['group'], thresholds)
                    definitions.append(dict(model=model, margin_cm=margin, thresholds=thresholds,
                                            calibration_clear_rates=clear_rates))
                    pp.append(p)
            ps = np.asarray(pp)
            all_p[key], all_defs[key] = ps, definitions
            den_cal = contact_weights[:, cal].sum(1)
            assert np.all(den_cal > 0)
            cal_rates = ps[:, cal] @ contact_weights[:, cal].T / den_cal
            eval_weights = {b: contact_weights[j, ev] for j, b in enumerate(BINS)}
            for c in ('clear_merged', 'clear_same', 'clear_other'):
                eval_weights[c] = truth[c][ev]
                for q in (0, 1):
                    eval_weights[f'{c}_q{q}'] = truth[c][ev] * (data['group'][ev] == q)

            def compare(i, ref):
                metrics = {}
                for name, w in eval_weights.items():
                    den = float(w.sum())
                    assert den > 0
                    cd = np.bincount(ui, weights=w, minlength=96)
                    diff = np.bincount(ui, weights=w * (ps[i, ev] - ps[ref, ev]), minlength=96)
                    rate = float(w @ ps[i, ev] / den); base = float(w @ ps[ref, ev] / den)
                    metrics[name] = dict(expected_n=den, source_rows=int((w > 1e-12).sum()),
                        source_units=int((cd > 1e-12).sum()), expected_alarms=float(w @ ps[i, ev]),
                        recall=rate, reference_recall=base, delta_pp=100*(rate-base),
                        delta_pp_ci95=np.percentile(100*(boot @ diff)/(boot @ cd), [2.5, 97.5]).tolist())
                return dict(policy_index=i, policy=definitions[i], reference_index=ref,
                            reference=definitions[ref], metrics=metrics,
                            calibration_contact_recalls=cal_rates[i].tolist())

            models = {}
            for model in MODELS:
                ids = [i for i, d in enumerate(definitions) if d['model'] == model]
                ref = next(i for i in ids if definitions[i]['margin_cm'] is None)
                primary = choose(cal_rates, ids, 0, 1, definitions)
                secondary = choose(cal_rates, ids, 1, 0, definitions)
                models[model] = dict(tof=compare(ref, ref), shared_primary=compare(primary, ref),
                                     separate_contact2_5=compare(secondary, ref))
            tof_ids = [i for i, d in enumerate(definitions) if d['margin_cm'] is None]
            ref = choose(cal_rates, tof_ids, 0, 1, definitions)
            primary = choose(cal_rates, range(len(ps)), 0, 1, definitions)
            family = compare(primary, ref)
            gains = [models['M3']['shared_primary']['metrics'][BINS[0]]['delta_pp'],
                     models['M3']['separate_contact2_5']['metrics'][BINS[1]]['delta_pp']]
            verdict = ('NOT_EVALUABLE' if max(support) > .05 else
                       'PAUSED_AFTER_TOF_CONFIRMATION' if all(g < 3 for g in gains) else 'CONTINUE_OPPORTUNITY')
            result['runs'][key] = dict(models=models, family_secondary=family,
                verdict=verdict, primary_support_missing_mass=support,
                calibration_bin_denominators=den_cal.tolist(), all_calibration_recalls=cal_rates.tolist())
    result['decision'] = result['runs']['0.1|1.0']['verdict']
    G.save(OUT / 'policies.json', all_defs)
    np.savez_compressed(OUT / 'policy_probabilities.npz', **all_p)
    G.save(OUT / 'result.json', result)
    report(result)
    print(json.dumps({'decision': result['decision'], 'runs': {k: v['verdict'] for k, v in result['runs'].items()}}))


def report(result):
    lines = [result['decision'], '', '# M3确认后同单位的理想XZ增量', '',
        '校准95000–95047的48单位选择全部阈值/余量/次要模型；评估96000–96095的96单位只应用策略。',
        '名义侵入支持−20…+15cm；清晰合并另一高度带与同高度身体外>10cm。每个heading情境分别按解析期望清晰权重校准，与owner固定σ0阈值的σ1压力描述不同。',
        '以下为相同校准预算，实际评估误报并不强制相等。仅使用缓存/元数据，无渲染、训练、推理、下载。', '',
        '|预算/heading|M3召回0–2 / 2–5|M3+XZ共同策略增量pp|单独2–5策略增量pp|评估合并清晰M3→XZ|',
        '|---|---|---|---|---|']
    for key, v in result['runs'].items():
        a = v['models']['M3']; t = a['tof']['metrics']; m = a['shared_primary']['metrics']
        extra = a['separate_contact2_5']['metrics']['contact2-5']['delta_pp']
        lines.append(f'|{key.replace(chr(124), "/")}|{t[BINS[0]]["recall"]:.2%} / {t[BINS[1]]["recall"]:.2%}|{m[BINS[0]]["delta_pp"]:+.2f} / {m[BINS[1]]["delta_pp"]:+.2f}|{extra:+.2f}|{m["clear_merged"]["reference_recall"]:.2%}→{m["clear_merged"]["recall"]:.2%}|')
    lines += ['', '主10%/1°的完整模型对照（全部calib选定）：', '',
              '|模型|0–2 / 2–5共同策略增量pp|2–5单独策略增量pp|同高度clear→XZ|另一高度clear→XZ|',
              '|---|---|---|---|---|']
    main = result['runs']['0.1|1.0']
    for model, a in main['models'].items():
        m = a['shared_primary']['metrics']; e = a['separate_contact2_5']['metrics']['contact2-5']
        c, o = m['clear_same'], m['clear_other']
        lines.append(f'|{model}|{m[BINS[0]]["delta_pp"]:+.2f} / {m[BINS[1]]["delta_pp"]:+.2f}|{e["delta_pp"]:+.2f}|{c["reference_recall"]:.2%}→{c["recall"]:.2%}|{o["reference_recall"]:.2%}→{o["recall"]:.2%}|')
    lines += ['', '主M3共同策略分母/条件95%区间：']
    for name, v in main['models']['M3']['shared_primary']['metrics'].items():
        lines.append(f'- {name}: 期望n={v["expected_n"]:.6f}，源行{v["source_rows"]}，单位{v["source_units"]}；Δ={v["delta_pp"]:.4f}pp，CI={v["delta_pp_ci95"]}。')
    f = main['family_secondary']
    lines += ['', f'次要模型族：calib选择ToF={f["reference"]["model"]}、XZ={f["policy"]["model"]}/余量{f["policy"]["margin_cm"]}cm。eval不择优。',
        '判读沿用3pp点门槛，未新增CI门槛；分别为两个bin选定的策略不是同一个操作点。实际清晰成本和条件区间须与增量一起解释。',
        'XZ含不可见表面且忽略高度；同高度清晰是目标净距定义，不表示背景panel也在10cm外。heading没有重算背景碰撞。',
        '已有同一模拟生成族Development，非硬件或保护盲测确认；理想几何收益不等于实际RGB收益。全部失败臂保留。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run()
