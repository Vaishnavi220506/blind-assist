"""Cached ideal-XZ opportunity with shared-heading, all-surface scene truth.

CPU metadata only. All thresholds, margins and model choices use calibration.
The range filter identifies the generated-target scene cohort, not the distance
of every background surface that can become a contact after heading changes.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

import cnh_rgb_confirmed_m3_gate as C
import cnh_rgb_scene_heading_truth as T

G, R = C.G, C.R
OUT = G.WORK / 'cnh-rgb-scene-heading-gate-20261002'
SOURCE = C.SOURCE


def prepare():
    assert not (OUT / 'PLAN.json').exists()
    row = (OUT / 'prerun-row.txt').read_text(encoding='utf-8').strip()
    assert row in (G.ROOT / 'research/active/dtr-r0/RUNS.md').read_text(encoding='utf-8')
    files = list(dict.fromkeys([Path(__file__), Path(T.__file__), Path(C.__file__),
        Path(G.__file__), OUT / 'prerun-row.txt', *R.input_paths(SOURCE)]))
    G.save(OUT / 'PLAN.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        prerun_row=row, role='EXPLORE consumed Development evaluator repair',
        models=C.MODELS, margins_cm=C.MARGINS, heading_deg=[0., 1.], budgets=[.1, .2],
        clearance_sigma_cm=2., source_splits=C.read(SOURCE / 'PLAN.json')['splits'],
        truth='Shared future center x=z*tan(delta), fixed query y/z; union physical-face slope intervals, analytic Gaussian integral. Contact uses strict x halfwidth .30; pass includes closed .40; shallow bins are P(.30)-P(.28), P(.28)-P(.25), hence deepest all-scene lateral intrusion. y/z interior epsilon1e-8.',
        range='Primary cohort: generated target range 1.2<=r<2.1; background contacts can be at other z. Clear calibration/evaluation uses entire .6..2.6 generated-target cohort.',
        observations='Frozen nominal all-vertical-surface XZ plus independent Gaussian2cm; actual future heading never enters scores or oracle observations.',
        selection='Reuse exact tied-score thresholds per query; only calibration selects common margin by bin0-2 then bin2-5 then clear cost then index. Separate bin2-5 swaps bin objectives. Secondary family selects one ToF model and one XZ model on calibration, never evaluation or per-row.',
        rule='10%/1degree: both calibration-selected single-bin M3 gains<3pp => PAUSED_AFTER_SCENE_TRUTH_REPAIR, else CONTINUE_OPPORTUNITY. Geometry integrity failure stops as NOT_EVALUABLE, never retune.',
        uncertainty='1000 paired evaluation-unit bootstrap seed2026100209, conditional on fixed calibration choices; old target-only sampling missing-mass heuristic is not a full-scene support guarantee.',
        limits=['Finite synthetic corpus, not fresh confirmation or deployment prevalence',
                'Fixed-z lateral-width path scenario, not rigid yaw normal-width corridor',
                'All physical surfaces include occluded geometry; not achievable RGB performance',
                'Old ordinal recipe and large-baseline temporal mechanisms remain stopped'],
        hashes={str(p.relative_to(G.ROOT)): G.sha(p) for p in files}))
    print('Prepared all-scene shared-heading gate.')


def run():
    plan = C.read(OUT / 'PLAN.json')
    assert not (OUT / 'result.json').exists()
    for rel, digest in plan['hashes'].items():
        assert G.sha(G.ROOT / rel) == digest, rel
    data, units, receipt = R.load_rows(SOURCE, plan['source_splits'])
    G.save(OUT / 'input-check.json', receipt)
    np.savez_compressed(OUT / 'rows.npz', **data)
    manifest = C.read(SOURCE / 'scene_manifest.json')
    cal = data['split'] == 'calib'; ev = data['split'] == 'evaluation'
    cohort = (data['range'] >= 1.2) & (data['range'] < 2.1)
    ui = np.searchsorted(units['evaluation'], data['unit'][ev])
    rng = np.random.default_rng(2026100209)
    boot = np.array([np.bincount(rng.integers(0, 96, 96), minlength=96) for _ in range(1000)])
    result = dict(plan_sha256=G.sha(OUT / 'PLAN.json'), runs={}, geometry={},
        counts=dict(calibration_rows=int(cal.sum()), evaluation_rows=int(ev.sum()), evaluation_units=96))
    all_p, all_defs, all_truth = {}, {}, {}
    for sig in plan['heading_deg']:
        truth, geometry_receipt = T.build_truth(data, manifest, sig)
        result['geometry'][str(sig)] = geometry_receipt
        for name, w in truth.items():
            assert len(w) == len(cal) and np.isfinite(w).all()
            assert np.all(w >= -1e-12) and np.all(w <= 1+1e-12), name
            all_truth[f'{sig}|{name}'] = w
        contact = np.array([truth[b] * cohort for b in C.BINS])
        for budget in plan['budgets']:
            key = f'{budget}|{sig}'
            definitions, probabilities = [], []
            for mi, model in enumerate(C.MODELS):
                score = data['scores'][:, mi]
                for margin in C.MARGINS:
                    veto = np.ones(len(cal)) if margin is None else ndtr((margin/100-data['vertical_sides_xz'])/.02)
                    thresholds, rates = C.fit_thresholds(score[cal], veto[cal], data['group'][cal], truth['clear_merged'][cal], budget)
                    definitions.append(dict(model=model, margin_cm=margin, thresholds=thresholds, calibration_clear_rates=rates))
                    probabilities.append(C.apply_thresholds(score, veto, data['group'], thresholds))
            ps = np.asarray(probabilities)
            all_p[key], all_defs[key] = ps, definitions
            den_cal = contact[:, cal].sum(1)
            assert np.all(den_cal > 0)
            cal_rates = ps[:, cal] @ contact[:, cal].T / den_cal
            weights = {b: contact[j, ev] for j, b in enumerate(C.BINS)}
            for name, w in truth.items():
                if name not in C.BINS:
                    weights[name] = w[ev]
            for name in ('clear_merged', 'clear_same', 'clear_other'):
                for q in (0, 1):
                    weights[f'{name}_q{q}'] = truth[name][ev] * (data['group'][ev] == q)

            def compare(i, ref):
                metrics = {}
                for name, w in weights.items():
                    den = float(w.sum())
                    cd = np.bincount(ui, weights=w, minlength=96)
                    diff = np.bincount(ui, weights=w*(ps[i, ev]-ps[ref, ev]), minlength=96)
                    n, bn = float(w @ ps[i, ev]), float(w @ ps[ref, ev])
                    bd = boot @ cd; ok = bd > 1e-12
                    metrics[name] = dict(expected_n=den, source_rows=int((w > 1e-12).sum()),
                        source_units=int((cd > 1e-12).sum()), expected_alarms=n, reference_alarms=bn,
                        recall=n/den if den else None, reference_recall=bn/den if den else None,
                        delta_pp=100*(n-bn)/den if den else None,
                        bootstrap_valid=int(ok.sum()),
                        delta_pp_ci95=np.percentile(100*(boot @ diff)[ok]/bd[ok], [2.5,97.5]).tolist() if ok.any() else None)
                return dict(policy_index=i, reference_index=ref, policy=definitions[i], reference=definitions[ref],
                    metrics=metrics, calibration_contact_recalls=cal_rates[i].tolist())

            models = {}
            for model in C.MODELS:
                ids = [i for i, d in enumerate(definitions) if d['model'] == model]
                ref = next(i for i in ids if definitions[i]['margin_cm'] is None)
                primary = C.choose(cal_rates, ids, 0, 1, definitions)
                secondary = C.choose(cal_rates, ids, 1, 0, definitions)
                models[model] = dict(tof=compare(ref, ref), shared_primary=compare(primary, ref),
                    separate_contact2_5=compare(secondary, ref))
            refs = [i for i,d in enumerate(definitions) if d['margin_cm'] is None]
            ref = C.choose(cal_rates, refs, 0, 1, definitions)
            best = C.choose(cal_rates, range(len(ps)), 0, 1, definitions)
            gains = [models['M3']['shared_primary']['metrics'][C.BINS[0]]['delta_pp'],
                     models['M3']['separate_contact2_5']['metrics'][C.BINS[1]]['delta_pp']]
            verdict = 'PAUSED_AFTER_SCENE_TRUTH_REPAIR' if all(g < 3 for g in gains) else 'CONTINUE_OPPORTUNITY'
            result['runs'][key] = dict(models=models, family_secondary=compare(best,ref), verdict=verdict,
                calibration_bin_denominators=den_cal.tolist(), all_calibration_recalls=cal_rates.tolist())
    for rel,digest in plan['hashes'].items():
        assert G.sha(G.ROOT / rel) == digest, rel
    result['decision'] = result['runs']['0.1|1.0']['verdict']
    G.save(OUT / 'policies.json', all_defs)
    np.savez_compressed(OUT / 'policy_probabilities.npz', **all_p)
    np.savez_compressed(OUT / 'truth_probabilities.npz', **all_truth)
    G.save(OUT / 'result.json', result)
    report(result)
    print(json.dumps(dict(decision=result['decision'], counts=result['counts'])))


def report(result):
    lines = [result['decision'], '', '# 全场景共享方向误差下的理想XZ增量', '',
        '复用95000–95047校准48单位、96000–96095评估96单位；只读元数据与冻结分数。',
        '全部物理面共享未来路径中心x=z tanδ，y/z范围保持原query。真值只在横向扩展10cm，另一高度不自动清晰；浅接触按全场景最大横向侵入。',
        '主范围仍为生成目标1.2–2.1m的场景队列，背景接触可能在其他距离。此固定z横向宽度假设不等于真实yaw法向宽度。',
        '同校准清晰预算，评估实际清晰率并不强制相等；每个heading情境独立校准。',
        'clear_same/other按原目标高度身份分层，但清晰概率均由全场景表面计算，绝非另一高度自动计clear。', '',
        '|预算/heading|M3召回0–2 / 2–5|M3+XZ共同策略增量pp|单独2–5增量pp|评估清晰M3→XZ|',
        '|---|---|---|---|---|']
    for key, v in result['runs'].items():
        a=v['models']['M3']; t=a['tof']['metrics']; m=a['shared_primary']['metrics']; e=a['separate_contact2_5']['metrics']['contact2-5']
        lines.append(f'|{key.replace(chr(124), "/")}|{t[C.BINS[0]]["recall"]:.2%} / {t[C.BINS[1]]["recall"]:.2%}|{m[C.BINS[0]]["delta_pp"]:+.2f} / {m[C.BINS[1]]["delta_pp"]:+.2f}|{e["delta_pp"]:+.2f}|{m["clear_merged"]["reference_recall"]:.2%}→{m["clear_merged"]["recall"]:.2%}|')
    main=result['runs']['0.1|1.0']
    lines += ['', '主10%/1°共同M3策略的全部分母与条件95%区间：']
    for name,m in main['models']['M3']['shared_primary']['metrics'].items():
        lines.append(f'- {name}: 期望n={m["expected_n"]:.6f}，源行{m["source_rows"]}，单位{m["source_units"]}；报警={m["expected_alarms"]:.6f} vs {m["reference_alarms"]:.6f}；Δ={m["delta_pp"]}pp，CI={m["delta_pp_ci95"]}。')
    f=main['family_secondary']
    lines += ['', f'校准选定最强ToF族={f["reference"]["model"]}；XZ族={f["policy"]["model"]} / 余量{f["policy"]["margin_cm"]}cm。',
        f'族共同策略0–2/2–5增量：{f["metrics"][C.BINS[0]]["delta_pp"]:.4f}/{f["metrics"][C.BINS[1]]["delta_pp"]:.4f}pp。',
        '全部624策略和失败保留；两项单独择优并不是一个共同部署策略。',
        '这是已消费模拟Development中的理想几何机会，不是实际RGB或硬件收益。物理面含不可见面。旧排序配方与大位移图像对继续停止；原报告不改写。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.stage == 'prepare' else run()
