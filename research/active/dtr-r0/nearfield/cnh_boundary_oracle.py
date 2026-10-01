"""Consumed-Development geometric-oracle veto envelope, never an RGB method.

Reuses cached ToF scores; reconstructs metadata only. No rendering, training,
inference, or protected/final data. Selection and conditional intervals use the
same consumed table and do not estimate prospective deployment performance.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import cnh_detectability_envelope as E
import cnh_structure_space as SS
from cnh_boundary_oracle_geometry import (scene_distances,
                                         scene_full3d_with_x_margin, self_check)

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'artifacts.local/work/cnh-boundary-oracle-20261001'
SIGMAS_CM = [0, 1, 2, 3, 5, 10]
GATE_FAS = [.02, .05, .10, .20, .30, .50, .75, .90, None]
MARGINS_CM = list(range(-30, 21)) + [None]  # None = bypass veto
BUDGETS = [.10, .20]
ARMS = ['full3d', 'vertical_sides_xz']
CATS = ['contact0-2cm', 'contact2-5cm', 'contact>5cm',
        'pass0-5cm', 'pass5-10cm']
RANGES = [(1.15, 1.6), (1.6, 2.1)]
METRICS = ['clear_HEAD_all', 'clear_BODY_all'] + [
    f'{lo}-{hi}|{c}' for lo, hi in RANGES for c in CATS] + [
    f'{lo}-{hi}|clear_{g}' for lo, hi in RANGES for g in SS.GROUPS]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    files = [E.OUT/'base_ensemble.npz', E.OUT/'contact_tradeoff.json',
             E.OUT/'contact_tradeoff.py', Path(__file__),
             HERE/'cnh_boundary_oracle_geometry.py', Path(E.__file__),
             Path(SS.__file__), HERE/'cnh_proposal_attribution_scenes.py']
    files += [E.OUT/'features/evaluation'/f'unit{u}.npz' for u in E.UNITS]
    import cnh_proposal_attribution_scenes as S
    files += [S.SOURCE/'cnh_track_a_geometry.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def prepare(out, reuse_output=None, dense_gates=False):
    if (out/'PLAN.json').exists():
        raise FileExistsError('Plan already exists; use run to resume, not replace it')
    gate_levels = [round(i/100, 2) for i in range(1, 100)]+[None] if dense_gates else GATE_FAS
    plan = dict(scope='consumed synthetic Development; evaluator-truth oracle',
                units=E.UNITS, score='cached deployed five-seed BASE ensemble',
                source='54d77185 contact_tradeoff rows, 40 scenes/unit, two queries',
                sigmas_cm=SIGMAS_CM, draws=20, noise_seed=2026100107,
                gate_clear_fa_levels=gate_levels, margins_cm=MARGINS_CM,
                budgets=BUDGETS, arms=ARMS, report_ranges_m=RANGES,
                geometry='min query Linf signed-distance over all physical box surfaces at final travel pose',
                full3d='all six faces, includes perfect height and occluded geometry',
                vertical_sides_xz='all boxes four vertical sides projected to xz; no height/group selection; explicit vertical-side prior',
                noise='one N(0,sigma) on scalar scene clearance per row; 20 common random-number draws shared across arms/sigmas',
                budget='per query, all-distance 1.15-2.6m other-height clear rows; constraint on mean of 20 draws; report actual segmented FA',
                gate='s >= per-query empirical clear quantile; ties repaired upward only if actual FA exceeds nominal; plus open gate',
                veto='estimated scalar clearance < m; None is bypass',
                selection='max pooled 0-2cm contact recall at 1.15-2.1m; tie-break lower pass0-5cm then clear FA then more selective gate then smaller m; both queries must satisfy budget',
                intervals='1000 paired unit bootstrap draws, seed2026100108; fixed selected policies, conditional CI; no correction for same-table selection',
                decision_margin_control='zero-noise all-surface oracle query x bounds expanded 0/5/10cm; y,z fixed; open gate and selected pure-ToF gate; no relabelling/training',
                prohibited=['rendering', 'training', 'new inference', 'protected/final access', 'RGB accuracy or hardware claims'],
                backend='CPU metadata/scalar scoring: TASK_NOT_GPU_SUITABLE',
                input_sha256=inputs(),
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    if reuse_output is not None:
        reuse_output = reuse_output.resolve()
        prior = json.loads((reuse_output/'PLAN.json').read_text(encoding='utf8'))
        current = plan['input_sha256']
        for path, digest in prior['input_sha256'].items():
            if path != str(Path(__file__).relative_to(ROOT)):
                assert current[path] == digest, ('Reuse source/data changed', path)
        plan['reuse_rows_from'] = str(reuse_output)
        plan['reused_rows_sha256'] = sha(reuse_output/'rows.npz')
        plan['prior_result_sha256'] = sha(reuse_output/'result.json')
        plan['scan_refinement_reason'] = 'Coarse gate20% at lateral sigma0 missed 10% budget by one clear row (0.1005208); gate1%-step sensitivity, same objective/budget/margin/noise; prior scan preserved'
    save(out/'PLAN.json', plan)
    print('Prepared', out/'PLAN.json', flush=True)


def category(offset, label):
    if offset is None:
        return 'clear' if label == 0 else 'other'
    if offset > .05:
        return CATS[2]
    if offset > .02:
        return CATS[1]
    if offset > 0:
        return CATS[0]
    return CATS[3] if offset > -.05 else CATS[4]


def rows(out):
    plan = json.loads((out/'PLAN.json').read_text(encoding='utf8'))
    if plan.get('reuse_rows_from'):
        prior = Path(plan['reuse_rows_from'])
        assert sha(prior/'rows.npz') == plan['reused_rows_sha256']
        receipt = json.loads((prior/'input-check.json').read_text(encoding='utf8'))
        receipt['reused_from'] = str(prior)
        save(out/'input-check.json', receipt)
        return SS.read(prior/'rows.npz'), receipt
    cached = out/'rows.npz'
    if cached.exists():
        return SS.read(cached), json.loads((out/'input-check.json').read_text(encoding='utf8'))
    pred = SS.read(E.OUT/'base_ensemble.npz')
    records = []
    receipt = dict(geometry=self_check(), unit_count=len(E.UNITS),
                   label_matches=0, generated_label_matches=0)
    for ui, u in enumerate(E.UNITS):
        data = SS.read(E.OUT/'features/evaluation'/f'unit{u}.npz')
        scenes = E.scenes_for(u)
        labels = data['labels'].reshape(40, 16, 6)[:, -1, 2:4]
        for c, sc in enumerate(scenes):
            ds = scene_distances(sc)
            expanded = np.array([scene_full3d_with_x_margin(sc, x/100)
                                 for x in (0, 5, 10)])
            np.testing.assert_array_equal(ds[0] < 0, labels[c].astype(bool))
            np.testing.assert_array_equal(sc['labels'][-1], labels[c])
            np.testing.assert_allclose(expanded[0], ds[0], atol=1e-12, rtol=0)
            assert np.isfinite(ds).all() and np.isfinite(expanded).all()
            receipt['label_matches'] += 2
            receipt['generated_label_matches'] += 2
            tq = int(data['group'][c])
            assert tq == sc['group']
            assert abs(float(data['margin'][c])-sc['margin']) < 1e-12
            for q in (0, 1):
                offset = -float(data['margin'][c]) if q == tq else None
                records.append(dict(ui=ui, unit=u, config=c, q=q,
                                    range=sc['meta']['range'],
                                    offset=np.nan if offset is None else offset,
                                    label=int(labels[c, q]), cat=category(offset, labels[c, q]),
                                    family=str(data['family'][c]), s=float(pred[str(u)][c, q]),
                                    full3d=float(ds[0, q]), vertical_sides_xz=float(ds[1, q]),
                                    expanded=expanded[:, q]))
        if ui % 12 == 0:
            save(out/'progress.json', dict(stage='metadata', completed=ui+1, total=len(E.UNITS)))
            print(f'metadata {ui+1}/{len(E.UNITS)}', flush=True)
    data = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    np.savez_compressed(cached, **data)
    receipt['rows'] = len(records)
    receipt['categories'] = {c: int(np.sum(data['cat'] == c)) for c in CATS+['clear', 'other']}
    # Reproduce every learned-score entry in the retained descriptive table.
    original = json.loads((E.OUT/'contact_tradeoff.json').read_text())['table']
    max_error = 0.
    for name, (lo, hi) in zip(('<=1.6m', '1.6-2.1m'), RANGES):
        for fa in (.02, .05, .10, .20, .30):
            gate = make_gate(data, fa, repair=False)[0]
            old = original[f's|{name}|clearFA{fa}']
            for c in CATS:
                ids = (data['cat'] == c) & (data['range'] >= lo) & (data['range'] < hi)
                got = round(float(gate[ids].mean()), 3)
                max_error = max(max_error, abs(got-old[c]))
    assert max_error < 1e-12
    receipt['contact_tradeoff_reproduction_max_error'] = max_error
    save(out/'input-check.json', receipt)
    return data, receipt


def make_gate(data, fa, repair=True):
    if fa is None:
        return np.ones(len(data['s']), dtype=bool), [None, None]
    gate = np.zeros(len(data['s']), dtype=bool)
    thresholds = []
    for q in (0, 1):
        clear = data['s'][(data['q'] == q) & (data['cat'] == 'clear')]
        threshold = float(np.quantile(clear, 1-fa))
        if repair and (clear >= threshold).mean() > fa+1e-12:
            limit = int(np.floor(fa*len(clear)+1e-10))
            threshold = float(np.nextafter(np.sort(clear)[len(clear)-limit-1], np.inf))
        ids = data['q'] == q
        gate[ids] = data['s'][ids] >= threshold
        if repair:
            assert (clear >= threshold).mean() <= fa+1e-12
        thresholds.append(threshold)
    return gate, thresholds


def masks(data):
    columns = [(data['cat'] == 'clear') & (data['q'] == q) for q in (0, 1)]
    for lo, hi in RANGES:
        in_range = (data['range'] >= lo) & (data['range'] < hi)
        columns += [in_range & (data['cat'] == c) for c in CATS]
    for lo, hi in RANGES:
        in_range = (data['range'] >= lo) & (data['range'] < hi)
        columns += [in_range & (data['cat'] == 'clear') & (data['q'] == q) for q in (0, 1)]
    return np.asarray(columns).T


def aggregate(values, data, metric_masks):
    """Policies x rows -> units x policies x metrics, clustered denominators."""
    values = np.atleast_2d(values)
    result = np.zeros((len(E.UNITS), len(values), len(METRICS)), dtype=np.float64)
    for ui in range(len(E.UNITS)):
        ids = data['ui'] == ui
        result[ui] = values[:, ids] @ metric_masks[ids].astype(float)
    return result


def choose(totals, denominators, budget, policy_defs):
    rates = totals/denominators[None, :]
    feasible = np.flatnonzero(np.all(rates[:, :2] <= budget+1e-12, axis=1))
    if not len(feasible):
        raise ValueError('No feasible policy, including disabled output')
    # Use pooled close-contact recall in the two requested ranges as the objective.
    def key(i):
        recall = (totals[i, 2]+totals[i, 7])/(denominators[2]+denominators[7])
        passed = (totals[i, 5]+totals[i, 10])/(denominators[5]+denominators[10])
        clear = float(totals[i, :2].sum()/denominators[:2].sum())
        d = policy_defs[i]
        gate = 2. if d['gate_fa'] is None else d['gate_fa']
        margin = 1000. if d['m_cm'] is None else d['m_cm']
        return (-recall, passed, clear, gate, margin)
    return min(feasible, key=key)


def ci(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def summarize(counts, den, baseline, boot_weights, gate_counts, definition,
              votes, data, metric_masks, budget):
    totals = counts.sum(0)
    denom = den.sum(0)
    rate = totals/denom
    base_rate = baseline.sum(0)/denom
    boot_den = boot_weights @ den
    boot_rate = (boot_weights @ counts)/boot_den
    boot_base = (boot_weights @ baseline)/boot_den
    all_clear = data['cat'] == 'clear'
    draw_clear = [votes[:, all_clear & (data['q'] == q)].mean(1) for q in (0, 1)]
    detail = {}
    for i, name in enumerate(METRICS):
        ids = metric_masks[:, i]
        draw = votes[:, ids].mean(1)
        detail[name] = dict(n=int(denom[i]), expected_count=float(totals[i]), rate=float(rate[i]),
                            ci95=ci(boot_rate[:, i]), baseline_rate=float(base_rate[i]),
                            delta_pp=float((rate[i]-base_rate[i])*100),
                            delta_pp_ci95=ci((boot_rate[:, i]-boot_base[:, i])*100),
                            noise_draw_min_max=[float(draw.min()), float(draw.max())])
    pooled_draw_fa = votes[:, all_clear].mean(1)
    return dict(policy=definition, metrics=detail,
                mean_clear_fa=float(totals[:2].sum()/denom[:2].sum()),
                noise_draw_clear_fa_min_max=[float(pooled_draw_fa.min()), float(pooled_draw_fa.max())],
                draw_over_budget_per_query={g: int(np.sum(a > budget+1e-12)) for g, a in zip(SS.GROUPS, draw_clear)},
                gate_metrics={name: float(v) for name, v in zip(METRICS, gate_counts.sum(0)/denom)},
                gate_recall_ceiling_verified=bool(np.all(totals <= gate_counts.sum(0)+1e-10)),
                ci_scope='paired unit bootstrap of fixed, same-table-selected policies; conditional, selection optimism uncorrected')


def run(out):
    global GATE_FAS
    started = time.perf_counter()
    plan = json.loads((out/'PLAN.json').read_text(encoding='utf8'))
    GATE_FAS = plan['gate_clear_fa_levels']
    if inputs() != plan['input_sha256']:
        raise ValueError('Inputs/code differ from prepared plan; use a new output directory')
    if (out/'result.json').exists():
        raise FileExistsError('Completed result exists; verify it rather than overwrite')
    sys.path.insert(0, str(ROOT/'tools'))
    from research_backend import select_backend, Workload, BackendCandidate, DeviceObservation
    probe = BackendCandidate(name='numpy-cpu', expected_device_type='cpu',
                             run_probe=lambda: np.arange(80, dtype=float).mean(),
                             observe=lambda _: DeviceObservation('cpu', 'host CPU', 'numpy', ('CPU',)))
    backend = select_backend(Workload.SCALAR_SCORING, cpu=probe,
                             record_path=out/'backend.json', capabilities={
                                 'python_executable': sys.executable, 'numpy': np.__version__,
                                 'reason_code': 'TASK_NOT_GPU_SUITABLE'})
    data, receipt = rows(out)
    mm = masks(data)
    den = aggregate(np.ones((1, len(data['s']))), data, mm)[:, 0]
    assert (den.sum(0) > 0).all()
    rng = np.random.default_rng(plan['noise_seed'])
    noise = rng.normal(size=(plan['draws'], len(data['s'])))
    np.savez_compressed(out/'noise.npz', standard_normal=noise)
    bootstrap_rng = np.random.default_rng(2026100108)
    boot_weights = np.array([np.bincount(bootstrap_rng.integers(0, len(E.UNITS), len(E.UNITS)),
                                       minlength=len(E.UNITS)) for _ in range(1000)])
    np.savez_compressed(out/'bootstrap.npz', unit_weights=boot_weights, denominators=den)
    gates, gate_defs = [], []
    for fa in GATE_FAS:
        g, t = make_gate(data, fa)
        gates.append(g)
        gate_defs.append(dict(gate_fa=fa, thresholds=t, comparator='>=', m_cm=None))
    gates = np.asarray(gates)
    gate_cluster = aggregate(gates.astype(float), data, mm)
    baseline, results = {}, {}
    policy_ledger = []
    selected_cluster = {}
    selected_votes = {}
    for budget in BUDGETS:
        k = GATE_FAS.index(budget)
        baseline[budget] = gate_cluster[:, k]
        results[str(budget)] = dict(baseline=summarize(
            baseline[budget], den, baseline[budget], boot_weights, baseline[budget],
            gate_defs[k], np.tile(gates[k], (20, 1)), data, mm, budget), oracle={})
    for ai, arm in enumerate(ARMS):
        for sigma in SIGMAS_CM:
            estimate = data[arm][None, :] + noise*sigma/100
            veto_draws = np.array([estimate < m/100 if m is not None else np.ones_like(estimate, dtype=bool)
                                  for m in MARGINS_CM])
            veto_probability = veto_draws.mean(1)
            # 20-draw average is the selection criterion; never select a noise draw.
            probabilities = (gates[:, None, :] * veto_probability[None, :, :]).reshape(-1, len(data['s']))
            definitions = [dict(gd, m_cm=m) for gd in gate_defs for m in MARGINS_CM]
            # Explicit no-output policy guarantees feasibility without a tie hack.
            probabilities = np.vstack([probabilities, np.zeros(len(data['s']))])
            definitions.append(dict(gate_fa=0., thresholds=[None, None], m_cm=None, disabled=True))
            cluster = aggregate(probabilities, data, mm)
            totals, denominator = cluster.sum(0), den.sum(0)
            for pi, definition in enumerate(definitions):
                policy_ledger.append(dict(arm=arm, sigma_cm=sigma, **definition,
                                          clear_fa_per_query=(totals[pi, :2]/denominator[:2]).tolist(),
                                          pooled_contact0_2_recall=float((totals[pi, 2]+totals[pi, 7])/(denominator[2]+denominator[7]))))
            for budget in BUDGETS:
                pi = choose(totals, denominator, budget, definitions)
                gi, mi = divmod(pi, len(MARGINS_CM))
                if definitions[pi].get('disabled'):
                    votes = np.zeros_like(noise, dtype=bool)
                    gate_c = np.zeros_like(den)
                else:
                    votes = veto_draws[mi] & gates[gi][None, :]
                    gate_c = gate_cluster[:, gi]
                point = summarize(cluster[:, pi], den, baseline[budget], boot_weights,
                                  gate_c, definitions[pi], votes, data, mm, budget)
                # Keep a full-open gate reference, regardless of its feasibility.
                open_ids = range((len(GATE_FAS)-1)*len(MARGINS_CM), len(GATE_FAS)*len(MARGINS_CM))
                open_defs = [definitions[i] for i in open_ids]
                oi_local = choose(totals[list(open_ids)], denominator, budget, open_defs)
                oi = list(open_ids)[oi_local]
                open_votes = veto_draws[oi % len(MARGINS_CM)]
                point['open_gate_reference'] = summarize(cluster[:, oi], den, baseline[budget], boot_weights,
                                                        gate_cluster[:, -1], definitions[oi], open_votes, data, mm, budget)
                # Family sensitivity is descriptive at the globally selected point.
                point['family_sensitivity'] = {}
                for family in ('none', 'corner'):
                    family_mask = data['family'] == family
                    stats = {}
                    for cat in ('contact0-2cm', 'pass0-5cm', 'clear'):
                        ids = family_mask & (data['cat'] == cat) & ((data['range'] < 2.1) if cat != 'clear' else True)
                        stats[cat] = dict(n=int(ids.sum()), rate=float(votes[:, ids].mean()))
                    point['family_sensitivity'][family] = stats
                results[str(budget)]['oracle'][f'{arm}|sigma{sigma}'] = point
                key = f'{budget}|{arm}|sigma{sigma}'
                selected_cluster[key] = cluster[:, pi]
                selected_votes[key] = votes
                print(f'{key}: gate={definitions[pi]["gate_fa"]} m={definitions[pi]["m_cm"]}cm near0-2={point["metrics"][METRICS[2]]["rate"]:.3f} clear={point["mean_clear_fa"]:.3f}', flush=True)
            save(out/'progress.json', dict(stage='oracle_grid', completed=ai*len(SIGMAS_CM)+SIGMAS_CM.index(sigma)+1, total=len(ARMS)*len(SIGMAS_CM)))
    controls = {}
    for budget in BUDGETS:
        bk = GATE_FAS.index(budget)
        for xi, margin in enumerate((0, 5, 10)):
            for gate_name, gi in (('open', len(GATE_FAS)-1), ('pure_tof_budget_gate', bk)):
                votes = np.tile((data['expanded'][:, xi] < 0) & gates[gi], (20, 1))
                cluster = aggregate(votes[:1].astype(float), data, mm)[:, 0]
                key = f'{budget}|x_margin{margin}|{gate_name}'
                controls[key] = summarize(cluster, den, baseline[budget], boot_weights, gate_cluster[:, gi],
                                          dict(gate_defs[gi], x_margin_cm=margin, m_cm=0), votes, data, mm, budget)
    np.savez_compressed(out/'selected-cluster.npz', **selected_cluster)
    np.savez_compressed(out/'selected-votes.npz', **selected_votes)
    save(out/'policy-summary.json', policy_ledger)
    result = dict(status='COMPLETE', scope=plan['scope'], plan_sha256=sha(out/'PLAN.json'),
                  source_checks=receipt, metrics=METRICS, working_points=results,
                  x_only_decision_margin_controls=controls,
                  pure_tof_curve={str(fa): dict(policy=gd, rates=dict(zip(METRICS, (gc.sum(0)/den.sum(0)).tolist())))
                                  for fa, gd, gc in zip(GATE_FAS, gate_defs, gate_cluster.transpose(1, 0, 2))},
                  limits=['evaluates perfect hidden/visible scene geometry, not actual RGB',
                          'full3d gains include perfect height truth; second arm has vertical-side prior',
                          'sigma is iid zero-mean scalar clearance error, not a measured RGB edge coordinate error',
                          'mean of 20 draws constrained; individual draws can exceed budget',
                          'fixed-policy paired unit intervals do not correct same-table policy selection',
                          'clear means other-height rows, not an operational false-alert distribution',
                          'decision x-margin control does not test expanded-label training',
                          'cached ToF score aggregates five frames; oracle evaluated at final query time',
                          'no severe occlusion, correspondence, calibration, latency or biased/tail error validation'],
                  backend=backend, elapsed_seconds=time.perf_counter()-started)
    save(out/'result.json', result)
    report(out, result)
    verify(out)
    save(out/'terminal.json', dict(status='complete', elapsed_seconds=time.perf_counter()-started))


def report(out, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    lines = ['# ToF门控与几何真值边界否决：已消费Development条件上限', '',
             '完整三维臂获得完美高度、全场景物体与遮挡表面真值；竖直侧面XZ臂忽略高度，但有竖直面先验。σ是标量有符号净距的零均值独立误差，不是实测RGB边缘误差。', '',
             '预算按HEAD/BODY各自在全部1.15–2.6 m另一高度带clear行计算，约束20次噪声的平均值。下表分距离报告；同表选择最优策略，区间为固定选点的单位配对bootstrap，未校正选择偏乐观。', '',
             '|全距离clear预算|距离(m)|σ(cm)|纯ToF 0–2cm召回|完整3D召回|增益pp及95%条件区间|竖直侧面XZ召回|3D擦身0–5cm报警|3D门控名义clearFA / m(cm)|',
             '|---|---|---|---|---|---|---|---|---|']
    for bi, budget in enumerate(BUDGETS):
        wp = result['working_points'][str(budget)]
        for ri, (lo, hi) in enumerate(RANGES):
            ax = axes[bi, ri]
            key = f'{lo}-{hi}|contact0-2cm'
            baseline = wp['baseline']['metrics'][key]['rate']
            ax.axhline(baseline*100, color='black', ls='--', label='ToF only')
            for arm, color, label in (('full3d', 'C0', 'all-surface 3D truth'),
                                      ('vertical_sides_xz', 'C1', 'vertical sides XZ prior')):
                points = [wp['oracle'][f'{arm}|sigma{s}'] for s in SIGMAS_CM]
                y = [p['metrics'][key]['rate']*100 for p in points]
                low = [p['metrics'][key]['ci95'][0]*100 for p in points]
                high = [p['metrics'][key]['ci95'][1]*100 for p in points]
                ax.plot(SIGMAS_CM, y, 'o-', color=color, label=label)
                ax.fill_between(SIGMAS_CM, low, high, color=color, alpha=.13)
            ax.set_title(f'{lo}-{hi} m; all-range clear budget {budget:.0%}')
            ax.grid(alpha=.25)
            ax.set_ylim(0, 105)
            if bi == 1:
                ax.set_xlabel('scalar clearance noise sigma (cm)')
            if ri == 0:
                ax.set_ylabel('0-2 cm contact recall (%)')
            if bi == 0 and ri == 0:
                ax.legend(fontsize=8)
            for sigma in SIGMAS_CM:
                full = wp['oracle'][f'full3d|sigma{sigma}']
                lateral = wp['oracle'][f'vertical_sides_xz|sigma{sigma}']
                metric = full['metrics'][key]
                ci_text = ', '.join(f'{x:+.1f}' for x in metric['delta_pp_ci95'])
                passed = full['metrics'][f'{lo}-{hi}|pass0-5cm']['rate']
                gate = '全开' if full['policy']['gate_fa'] is None else f"{full['policy']['gate_fa']:.0%}"
                lines.append(f"|{budget:.0%}|{lo}–{hi}|{sigma}|{baseline:.1%}|{metric['rate']:.1%}|{metric['delta_pp']:+.1f} [{ci_text}]|{lateral['metrics'][key]['rate']:.1%}|{passed:.1%}|{gate} / {full['policy']['m_cm']}|")
    fig.suptitle('Evaluated-truth oracle; same-table selected envelope, conditional unit-bootstrap CI', fontsize=11)
    fig.tight_layout()
    fig.savefig(out/'oracle-envelope.png', dpi=160)
    plt.close(fig)
    lines += ['', '0/5/10 cm余量对照仅扩张查询盒的横向边界，保持高度/纵向边界；没有重新标注训练模型。完整分母、各距离实际clearFA、逐噪声预算超限次数、门控召回天花板和全开门控参照见result.json。', '',
              '|10%纯ToF门控后的横向判定余量|近段0–2cm召回|近段擦身0–5cm报警|近段擦身5–10cm报警|全距离clearFA|',
              '|---|---|---|---|---|']
    for margin in (0, 5, 10):
        point = result['x_only_decision_margin_controls'][f'0.1|x_margin{margin}|pure_tof_budget_gate']
        metrics = point['metrics']
        lines.append(f"|{margin} cm|{metrics[METRICS[2]]['rate']:.1%}|{metrics[METRICS[5]]['rate']:.1%}|{metrics[METRICS[6]]['rate']:.1%}|{point['mean_clear_fa']:.1%}|")
    lines += ['', '局限：', ''] + ['- '+s for s in result['limits']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


def verify(out):
    global GATE_FAS
    result = json.loads((out/'result.json').read_text(encoding='utf8'))
    plan = json.loads((out/'PLAN.json').read_text(encoding='utf8'))
    GATE_FAS = plan['gate_clear_fa_levels']
    assert sha(out/'PLAN.json') == result['plan_sha256']
    assert inputs() == plan['input_sha256']
    row_path = Path(plan['reuse_rows_from'])/'rows.npz' if plan.get('reuse_rows_from') else out/'rows.npz'
    data = SS.read(row_path)
    noise = SS.read(out/'noise.npz')['standard_normal']
    stored = SS.read(out/'selected-votes.npz')
    check = dict(status='PASS', selected_policies_checked=0, max_direct_rate_error=0.,
                 checks=['source/code hashes', 'all selected budgets by query',
                         'direct row-level alarm reconstruction', 'veto <= gate ceiling',
                         '20 draw means, denominators and stored bootstrap inputs'])
    mm = masks(data)
    for bs, working in result['working_points'].items():
        budget = float(bs)
        for key, point in working['oracle'].items():
            arm, sigma_s = key.split('|sigma')
            definition = point['policy']
            gate = make_gate(data, definition['gate_fa'])[0]
            if definition.get('disabled'):
                votes = np.zeros_like(noise, dtype=bool)
                gate[:] = False
            elif definition['m_cm'] is None:
                votes = np.tile(gate, (20, 1))
            else:
                votes = gate[None, :] & (data[arm][None, :]+noise*int(sigma_s)/100 < definition['m_cm']/100)
            np.testing.assert_array_equal(votes, stored[f'{bs}|{key}'])
            assert (votes <= gate[None, :]).all()
            for q in (0, 1):
                ids = (data['cat'] == 'clear') & (data['q'] == q)
                assert votes[:, ids].mean() <= budget+1e-12
            for i, name in enumerate(METRICS):
                direct = float(votes[:, mm[:, i]].mean())
                error = abs(direct-point['metrics'][name]['rate'])
                assert error < 1e-12
                check['max_direct_rate_error'] = max(check['max_direct_rate_error'], error)
                assert point['metrics'][name]['n'] == int(mm[:, i].sum())
            check['selected_policies_checked'] += 1
    assert result['source_checks']['label_matches'] == 7680
    assert result['source_checks']['contact_tradeoff_reproduction_max_error'] == 0
    save(out/'verification.json', check)
    print(json.dumps(check), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'run', 'verify'], required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--reuse-output', type=Path)
    parser.add_argument('--dense-gates', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare(args.output, args.reuse_output, args.dense_gates)
    elif args.stage == 'verify':
        verify(args.output)
    else:
        try:
            run(args.output)
        except BaseException as error:
            save(args.output/'terminal.json', dict(status='failed', error=repr(error)))
            raise


if __name__ == '__main__':
    main()
