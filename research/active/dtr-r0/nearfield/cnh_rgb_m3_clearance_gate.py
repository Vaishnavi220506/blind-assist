"""Cached M3-referenced clearance opportunity check; no RGB mechanism run.

Same consumed rows, threshold family and analytic integration as the previous
screen. M3 is included irrespective of its separate same-height guard failure:
that negative class is absent for both competitors here. No training/download.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

import cnh_rgb_threelevel_gate as G
import cnh_rgb_threelevel_matched as M

OUT = G.WORK / 'cnh-rgb-m3-clearance-gate-20261002'
FOLLOWUP = ('Final line decision awaits ToF M3 confirmation on new units including '
            'same-height clear targets 10-20cm outside the body; recompute ideal '
            'clearance increment on those exact new units.')


def prepare():
    assert not (OUT / 'PLAN.json').exists()
    row = (OUT / 'prerun-row.txt').read_text(encoding='utf-8-sig').strip()
    assert row in (G.ROOT / 'research/active/dtr-r0/RUNS.md').read_text(encoding='utf-8')
    files = [Path(__file__), Path(G.__file__), Path(M.__file__),
             G.SOURCE / 'rows.npz', M.MARGIN / 'scores92000_M3.npz',
             M.ENVELOPE / 'base_ensemble.npz', G.OUT / 'result.json',
             M.ENVELOPE / 'data/evaluation/metadata.npz', OUT / 'prerun-row.txt']
    G.save(OUT / 'PLAN.json', dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(), prerun_row=row,
        role='Consumed synthetic Development; same-table selection; not confirmation',
        models=['BASE', 'M3'], units=[92000, 92095], rows=7680,
        budgets=[.1, .2], primary_budget=.1, heading_deg=[0., 1.],
        clearance_sigma_cm=2., margins_cm=M.MARGINS, range_m=[1.2, 2.1],
        bins=['contact0-2', 'contact2-5'],
        selection='Same shared primary: maximize contact0-2, then contact2-5, then lower clear, then index. Also preserve separate-bin best envelopes; not a single deployable policy.',
        threshold='Exact score breakpoints separately for each query, all-distance other-height clear rows; near-pass unpenalized',
        noise='Inherited analytic Gaussian integration. Heading moves target labels only, no future-heading leak into observed clearance.',
        rule='At primary budget10% and heading1deg, PAUSED_PENDING_TOF_CONFIRMATION iff BOTH separate-bin best M3+XZ minus M3 increments <3pp; else CONTINUE_OPPORTUNITY. Invalid support => NOT_EVALUABLE. No CI condition added.',
        sensitivity='20% results reported under same rule separately; primary decision remains10%',
        comparisons=['M3+XZ minus M3', 'BASE+XZ minus M3'],
        bootstrap='1000 paired96-unit draws seed2026100209; fixed selected policies, descriptive conditional intervals',
        limits=['No same-height >10cm clear negatives for either competitor',
                'No background-panel future collisions recomputed',
                'Ideal nominal XZ, unmeasured heading noise, not RGB or hardware performance',
                'M3 original failed guard retained; this screen neither confirms nor promotes M3'],
        followup=FOLLOWUP, mechanism_and_download='PAUSED_BEFORE_THIS_CHECK',
        hashes={str(p.relative_to(G.ROOT)): M.sha(p) for p in files}))
    print('Prepared M3 check; prerequisite RUNS row verified.')


def run():
    plan = M.read(OUT / 'PLAN.json')
    assert not (OUT / 'result.json').exists()
    for rel, digest in plan['hashes'].items():
        assert M.sha(G.ROOT / rel) == digest, rel
    data, _ = M.validate_rows()
    scores = {'BASE': data['s']}
    z = M.load_npz(M.MARGIN / 'scores92000_M3.npz')
    assert set(z) == set(map(str, M.UNITS))
    assert all(a.shape == (40, 2) and np.isfinite(a).all() for a in z.values())
    scores['M3'] = np.array([z[str(u)][c, q] for u, c, q in zip(data['unit'], data['config'], data['q'])])
    rng = np.random.default_rng(2026100209)
    boot = np.array([np.bincount(rng.integers(0, 96, 96), minlength=96) for _ in range(1000)])
    target = np.isfinite(data['offset']) & (data['range'] >= 1.2) & (data['range'] < 2.1)
    result = dict(plan_sha256=M.sha(OUT / 'PLAN.json'), rows=len(target), units=96,
                  clear_rows_per_query=[int(((data['q'] == q) & (data['cat'] == 'clear')).sum()) for q in (0, 1)],
                  checks=G.fixtures(), runs={}, followup=FOLLOWUP)
    policies, probabilities = {}, {}
    original = M.read(G.OUT / 'result.json')
    for budget in plan['budgets']:
        definitions, pp = [], []
        for model, score in scores.items():
            for margin in M.MARGINS:
                veto = np.ones(len(target)) if margin is None else ndtr((margin / 100 - data['vertical_sides_xz']) / .02)
                p, d = G.best_thresholds(dict(data, s=score), veto, budget)
                d.update(model=model, margin_cm=margin)
                pp.append(p); definitions.append(d)
        ps = np.asarray(pp)
        policies[str(budget)], probabilities[str(budget)] = definitions, ps
        m3ref = next(i for i, d in enumerate(definitions) if d['model'] == 'M3' and d['margin_cm'] is None)
        for sig in plan['heading_deg']:
            weights, support = [], []
            for b in plan['bins']:
                lo, hi = G.BINS[b]
                w = np.zeros(len(target))
                w[target] = G.bin_weights(data['offset'][target], data['range'][target], lo, hi, sig)
                weights.append(w)
                support.append(max(G.missing_mass(r, a, sig) for r in (1.2, 2.1) for a in (lo, hi)))
            ww = np.asarray(weights); den = ww.sum(1)
            assert np.all(den > 0)
            rates = ps @ ww.T / den

            def compare(i):
                metrics = {}
                for j, b in enumerate(plan['bins']):
                    w = ww[j]
                    cd = np.bincount(data['ui'], weights=w, minlength=96)
                    diff = np.bincount(data['ui'], weights=w * (ps[i] - ps[m3ref]), minlength=96)
                    assert np.all(boot @ cd > 0)
                    metrics[b] = dict(expected_n=float(den[j]), source_rows=int((w > 1e-12).sum()),
                        source_units=int((cd > 1e-12).sum()), recall=float(rates[i, j]),
                        reference_recall=float(rates[m3ref, j]), delta_pp=float(100 * (rates[i, j] - rates[m3ref, j])),
                        delta_pp_ci95=np.percentile(100 * (boot @ diff) / (boot @ cd), [2.5, 97.5]).tolist(),
                        maximum_missing_kernel_mass=support[j], support_valid=support[j] <= .05)
                return dict(policy_index=i, policy=definitions[i], reference_index=m3ref,
                            reference=definitions[m3ref], metrics=metrics)

            models = {}
            for model in scores:
                ids = [i for i, d in enumerate(definitions) if d['model'] == model]
                first = M.choose(rates, ids, 0, 1, definitions)
                second = M.choose(rates, ids, 1, 0, definitions)
                models[model] = dict(shared_primary=compare(first), separate_contact2_5=compare(second))
            for name, oldname in [('shared_primary', 'oracle_primary'), ('separate_contact2_5', 'oracle_secondary_envelope')]:
                old = original['runs'][f'{budget}|{sig}']['arms'][oldname]
                new = models['BASE'][name]
                for field in ('margin_cm', 'thresholds', 'clear_per_query'):
                    assert new['policy'][field] == old['policy'][field]
                for b in plan['bins']:
                    np.testing.assert_allclose(new['metrics'][b]['recall'], old['metrics'][f'1.2-2.1|{b}']['recall'], rtol=0, atol=1e-12)
            selected = [models['M3']['shared_primary']['metrics']['contact0-2'],
                        models['M3']['separate_contact2_5']['metrics']['contact2-5']]
            verdict = ('NOT_EVALUABLE' if not all(v['support_valid'] for v in selected) else
                       'PAUSED_PENDING_TOF_CONFIRMATION' if all(v['delta_pp'] < 3 for v in selected) else 'CONTINUE_OPPORTUNITY')
            result['runs'][f'{budget}|{sig}'] = dict(reference=compare(m3ref), models=models,
                verdict=verdict, all_policy_recalls=rates.tolist(), metric_keys=plan['bins'])
    result['decision'] = result['runs']['0.1|1.0']['verdict']
    result['checks']['original_BASE_policies_and_recalls_reproduced'] = True
    G.save(OUT / 'policies.json', policies)
    np.savez_compressed(OUT / 'policy_probabilities.npz', **probabilities)
    G.save(OUT / 'result.json', result)
    lines = [result['decision'], '', '# M3之上的理想净距增量', '',
        '同92000–92095已消费合成Development，96单位、3840场景、7680查询行；每query清晰分母1920。',
        '主10%预算，20%敏感性；清晰仅另一高度带，双方均缺同高度身体外>10cm负例。', '',
        '|预算/heading|M3召回0–2 / 2–5|M3+XZ共同策略召回|增量pp|BASE+XZ−M3 pp|M3+XZ单独2–5最优增量pp|',
        '|---|---|---|---|---|---|']
    for key, value in result['runs'].items():
        ref = value['reference']['metrics']; mm = value['models']['M3']['shared_primary']['metrics']
        bm = value['models']['BASE']['shared_primary']['metrics']
        pair = lambda obj, field, scale=1: ' / '.join(f'{obj[b][field]*scale:.2f}' for b in plan['bins'])
        extra = value['models']['M3']['separate_contact2_5']['metrics']['contact2-5']['delta_pp']
        lines.append(f'|{key.replace(chr(124), "/")}|{pair(ref,"recall",100)}%|{pair(mm,"recall",100)}%|{pair(mm,"delta_pp")}|{pair(bm,"delta_pp")}|{extra:.2f}|')
    lines += ['', '主10%/1°详细分母和固定策略条件95%区间：', '']
    for b, v in result['runs']['0.1|1.0']['models']['M3']['shared_primary']['metrics'].items():
        lines.append(f'- {b}: 期望n={v["expected_n"]:.6f}，源行{v["source_rows"]}，源单位{v["source_units"]}，增量{v["delta_pp"]:.4f}pp，CI={v["delta_pp_ci95"]}。')
    lines += ['', '最终去留等ToF线在新单位上的M3确认（含身体外10–20cm同高度负例），届时在同一新单位上复算理想净距增量。',
              'M3旧guard失败保留；本次仅使缺失负例条件下的对照对称，不宣称M3通过原门槛。',
              '同表选策略，区间条件于已选策略；两个bin分别最优包络不是同一可部署策略。理想XZ/方向噪声非真实RGB/硬件性能。',
              '缓存检查没有渲染、训练、模型推理或下载。暂停期间已完成的历史图像对报告保持原样。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'decision': result['decision'], 'runs': {k: v['verdict'] for k, v in result['runs'].items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run()
