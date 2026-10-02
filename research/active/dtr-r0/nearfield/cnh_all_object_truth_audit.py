"""Fixed-score/fixed-threshold audit of evaluator-only all-object truth.

No inference, calibration, training, scene generation or original verdict edits.
Only this audit's outputs are written under its own preplanned artifact folder.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME

OUT = MC.SS.WORK / 'cnh-all-object-truth-20261002'
CONTEXT = MC.SS.WORK / 'cnh-margin-context-transfer-20261002'
ARMS = ('NEAR', 'M3', 'M8')
CATEGORIES = ('contact', 'pass', 'clear')
SEEDS = {'source': 2026100205, 'context': 2026100210}
CONTACTS = ('contact_0_2cm', 'contact_2_5cm', 'contact_gt5cm')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def input_data():
    plan_path = OUT / 'PLAN.json'
    if not plan_path.is_file():
        raise RuntimeError('Write the audit PLAN and RUNS entry before running')
    receipt = read(OUT / 'geometry_receipt.json')
    if receipt.get('status') != 'COMPLETE' or receipt['query_rows'] != 18432:
        raise ValueError('Geometry ledger must be complete, with 18432 query rows')
    if ME.sha(OUT / 'ledger.json') != receipt['ledger_sha256']:
        raise ValueError('Geometry ledger checksum differs from completion receipt')
    if ME.sha(OUT / 'geometry.json') != receipt['geometry_sha256']:
        raise ValueError('Geometry payload checksum differs from completion receipt')
    if ME.sha(plan_path) != receipt['plan_sha256']:
        raise ValueError('Audit plan changed since geometry reconstruction')
    ledger = read(OUT / 'ledger.json')
    source_result, context_result = read(MC.OUT / 'result.json'), read(CONTEXT / 'result.json')
    if source_result['status'] != 'COMPLETE' or context_result['status'] != 'COMPLETE':
        raise ValueError('Original source and context results must be COMPLETE')
    manifest_hash = ME.sha(MC.OUT / 'scene_manifest.json')
    if manifest_hash != source_result['provenance']['input_sha256']['scene_manifest.json'] or \
            manifest_hash != receipt['inputs']['source_manifest_sha256']:
        raise ValueError('Source scene manifest changed since original scores or geometry replay')
    thresholds = source_result['thresholds']
    if context_result['thresholds'] != thresholds:
        raise ValueError('Context and source must use exactly the same frozen thresholds')
    source_plan, context_plan = read(MC.OUT / 'PLAN.json'), read(CONTEXT / 'PLAN.json')
    expected = {(dataset, split, int(u), c, q)
                for dataset, splits, configs in [('source', MC.SPLITS, range(40)),
                    ('context', {'evaluation': list(range(70000, 70096))}, range(22, 58))]
                for split, units in splits.items() for u in units for c in configs for q in (0, 1)}
    identities = [(r['dataset'], r['split'], r['unit'], r['config'], r['query']) for r in ledger]
    if len(ledger) != len(expected) or set(identities) != expected:
        raise ValueError('Ledger identities do not match all planned source/context rows')
    for r in ledger:
        if r['old_category'] not in CATEGORIES or r['all_category'] not in CATEGORIES:
            raise ValueError('Unknown geometry category')
        if (r['old_category'] == 'contact') != (r['all_category'] == 'contact'):
            raise ValueError('Contact truth must be unchanged')
        if r['old_category'] == 'pass' and r['all_category'] != 'pass':
            raise ValueError('Original acceptable target pass must remain acceptable')
    scores, scores_hashes = {}, {}
    for dataset, folder in [('source', MC.OUT), ('context', CONTEXT)]:
        scores[dataset], scores_hashes[dataset] = {}, {}
        units = sorted({r['unit'] for r in ledger if r['dataset'] == dataset})
        for arm in ARMS:
            path = folder / f'scores_{arm}.npz'
            digest = ME.sha(path)
            old_hash = (source_result['provenance']['input_sha256'][path.name] if dataset == 'source'
                        else context_result['provenance']['scores_sha256'][arm])
            if digest != old_hash:
                raise ValueError(f'{dataset}/{arm}: cached scores changed since original result')
            with np.load(path, allow_pickle=False) as z:
                if set(z.files) != set(map(str, units)):
                    raise ValueError(f'{dataset}/{arm}: unexpected score unit keys')
                scores[dataset][arm] = {str(u): np.asarray(z[str(u)], float) for u in units}
            shape = (40 if dataset == 'source' else 36, 2)
            if any(s.shape != shape or not np.isfinite(s).all() for s in scores[dataset][arm].values()):
                raise ValueError(f'{dataset}/{arm}: invalid score shape or nonfinite scores')
            scores_hashes[dataset][arm] = digest
    models = {str(p.relative_to(MC.SS.ROOT)): ME.sha(p) for arm in ARMS for p in MC.model_paths(arm)}
    source_models = {p: digest for arm in ARMS for p, digest in source_plan['model_sha256'][arm].items()}
    if models != source_models or models != context_plan['model_sha256']:
        raise ValueError('Frozen model checkpoint hashes changed')
    provenance = dict(ledger_sha256=ME.sha(OUT / 'ledger.json'),
        geometry_receipt_sha256=ME.sha(OUT / 'geometry_receipt.json'), geometry_sha256=receipt['geometry_sha256'],
        scores_sha256=scores_hashes, model_sha256=models,
        source_manifest_sha256=manifest_hash,
        source_result_sha256=ME.sha(MC.OUT / 'result.json'), context_result_sha256=ME.sha(CONTEXT / 'result.json'),
        source_plan_sha256=ME.sha(MC.OUT / 'PLAN.json'), context_plan_sha256=ME.sha(CONTEXT / 'PLAN.json'),
        plan_sha256=ME.sha(plan_path), evaluator_sha256=ME.sha(__file__), metric_helper_sha256=ME.sha(ME.__file__))
    return ledger, scores, thresholds, source_result, context_result, provenance


def boot_weights(n, seed):
    rng = np.random.default_rng(seed)
    return np.asarray([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(1000)])


def masks(rows):
    old = np.asarray([r['old_category'] for r in rows])
    new = np.asarray([r['all_category'] for r in rows])
    off = np.asarray([r['target_off'] for r in rows])
    target = np.asarray([r['query'] == r['target_group'] for r in rows])
    result = {f'{version}_{category}': values == category
              for version, values in [('old', old), ('new', new)] for category in CATEGORIES}
    for name, lo, hi in [('contact_0_2cm', 0, .02), ('contact_2_5cm', .02, .05), ('contact_gt5cm', .05, np.inf)]:
        result[name] = target & (old == 'contact') & (off > lo) & (off <= hi)
    result['old_clear_to_new_pass'] = (old == 'clear') & (new == 'pass')
    return result


def scope_statistics(rows, scores, thresholds, units, seed, select):
    """Bootstrap whole units, including units with zero rows in a category."""
    lookup = {u: i for i, u in enumerate(units)}
    ui = np.asarray([lookup[r['unit']] for r in rows])
    boot = boot_weights(len(units), seed)
    membership = masks(rows)
    alarms = {arm: np.asarray([scores[arm][str(r['unit'])][r['config'] - (22 if r['dataset'] == 'context' else 0), r['query']]
                               >= thresholds[arm][str(r['query'])] for r in rows]) for arm in ARMS}
    statistics, bootstrap = {}, {}
    for arm in ARMS:
        statistics[arm], bootstrap[arm] = {}, {}
        for category, mask in membership.items():
            statistics[arm][category], bootstrap[arm][category] = ME.summarize(mask & select, alarms[arm], ui, boot)
    differences = {}
    for arm in ('M3', 'M8'):
        differences[arm] = {}
        for category in membership:
            a, n = statistics[arm][category]['rate'], statistics['NEAR'][category]['rate']
            differences[arm][category] = dict(delta=a-n if a is not None and n is not None else None,
                ci95=ME.ci95(bootstrap[arm][category] - bootstrap['NEAR'][category]))
    transitions = {}
    for old in CATEGORIES:
        for new in CATEGORIES:
            transition = select & membership[f'old_{old}'] & membership[f'new_{new}']
            transitions[f'{old}->{new}'] = dict(rows=int(transition.sum()),
                alarms={arm: int((alarms[arm] & transition).sum()) for arm in ARMS})
    return dict(query_rows=int(select.sum()), units=len(units), statistics=statistics,
                differences=differences, transitions=transitions)


def audit_arrays(ledger, scores, thresholds):
    scopes = {}
    for dataset, split in [('source', 'calib'), ('source', 'evaluation'), ('context', 'evaluation')]:
        rows = [r for r in ledger if r['dataset'] == dataset and r['split'] == split]
        units = sorted({r['unit'] for r in rows})
        all_rows = np.ones(len(rows), bool)
        selections = {'all': all_rows}
        if dataset == 'source' and split == 'calib':
            for q, group in enumerate(('HEAD', 'BODY')):
                selections[f'group:{group}'] = np.asarray([r['query'] == q for r in rows])
        families = sorted({r['family'] for r in rows})
        for family in families:
            selections[f'family:{family}'] = np.asarray([r['family'] == family for r in rows])
        if dataset == 'source' and split == 'evaluation':
            primary = np.asarray([.6 <= r['range'] < 2.1 for r in rows])
            selections['primary_0.6-2.1'] = primary
            for family in families:
                selections[f'primary_0.6-2.1|family:{family}'] = primary & selections[f'family:{family}']
        for name, select in selections.items():
            key = f'{dataset}|{split}|{name}'
            scopes[key] = scope_statistics(rows, scores[dataset], thresholds, units, SEEDS[dataset], select)
    return scopes


def verify_original(scopes, source, context):
    """Require original contact numerators, denominators and CI to reproduce."""
    checks = []
    pairs = [('source|evaluation|all', source['statistics']['sigma0']['all_0.6-2.6']),
             ('source|evaluation|primary_0.6-2.1', source['statistics']['sigma0']['primary_0.6-2.1']),
             ('context|evaluation|all', context['statistics']['pooled'])]
    pairs.extend((f'context|evaluation|family:{family}', context['statistics'][family]) for family in context['families'])
    for scope, original in pairs:
        for arm in ARMS:
            mapping = {k: k for k in CONTACTS}
            mapping.update(old_clear='clear_merged', old_pass='pass_0_10cm')
            for current, old in mapping.items():
                actual = scopes[scope]['statistics'][arm][current]
                for field in ('alarms', 'denominator', 'rate', 'ci95'):
                    if actual[field] != original[arm][old][field]:
                        raise ValueError(f'Original metric mismatch: {scope}/{arm}/{current}/{field}')
                checks.append(f'{scope}/{arm}/{current}')
    for arm in ARMS:
        old_calibration = source['calibration'][arm]
        metric = scopes['source|calib|all']['statistics'][arm]['old_clear']
        if metric['alarms'] != sum(g['alarms'] for g in old_calibration.values()) or \
                metric['denominator'] != sum(g['clear_rows'] for g in old_calibration.values()):
            raise ValueError(f'Original calibration count mismatch: {arm}')
        checks.append(f'source|calib|all/{arm}/old_clear')
        for q, group in enumerate(('HEAD', 'BODY')):
            metric = scopes[f'source|calib|group:{group}']['statistics'][arm]['old_clear']
            old = old_calibration[str(q)]
            if metric['alarms'] != old['alarms'] or metric['denominator'] != old['clear_rows']:
                raise ValueError(f'Original calibration group count mismatch: {arm}/{group}')
            checks.append(f'source|calib|group:{group}/{arm}/old_clear')
    return dict(status='EXACT_MATCH', checked_metrics=len(checks), metrics=checks)


def descriptive_recheck(scopes, source, context):
    result = {}
    for dataset, original in [('source', source), ('context', context)]:
        full = scopes[f'{dataset}|evaluation|all']
        contact = scopes['source|evaluation|primary_0.6-2.1'] if dataset == 'source' else full
        result[dataset] = {}
        for arm in ('M3', 'M8'):
            shallow = contact['differences'][arm]['contact_0_2cm']
            deep = contact['differences'][arm]['contact_gt5cm']['delta']
            clear = full['differences'][arm]['new_clear']['delta']
            actual_clear = full['statistics'][arm]['new_clear']['rate']
            checks = dict(shallow_ci_lower_gt_zero=shallow['ci95'][0] is not None and shallow['ci95'][0] > 0,
                          deep_drop_le_2pp=deep is not None and deep >= -.02,
                          clear_increase_le_2pp=clear is not None and clear <= .02)
            if dataset == 'context':
                checks['actual_clear_le_12pct'] = actual_clear is not None and actual_clear <= .12
            old = original['comparisons']['sigma0' if dataset == 'source' else 'pooled'][arm]['verdict']
            result[dataset][arm] = dict(role='DESCRIPTIVE_RECHECK_ONLY; not a new confirmatory verdict',
                original_verdict_unchanged=old, checks=checks, all_checks_met=all(checks.values()),
                new_clear_delta=clear, new_clear_actual=actual_clear,
                new_clear_delta_ci95=full['differences'][arm]['new_clear']['ci95'])
    return result


def write_report(result):
    lines = ['AUDIT_COMPLETE', '', '# 全物体三级真值固定阈值审计', '',
             '只重算评价真值；分数、模型、HEAD/BODY阈值保持原值，不推理、不校准、不重训。',
             '几何接触优先；所有物体在同高度、同深度范围内进入横向外扩10cm区域视为擦身。原接触率及分母精确复现，旧确认/迁移判读保留。',
             '1000次配对单位bootstrap区间，源单位种子2026100205、迁移单位种子2026100210；校准单位统计仅描述冻结工作点，不用于重新选阈值。', '']
    for name, scope in result['scopes'].items():
        lines += [f'## {name}', '', f"{scope['query_rows']}查询行，{scope['units']}单位。", '',
                  '| 类别 | NEAR 报警数/分母及95%CI | M3 | M8 |', '| --- | --- | --- | --- |']
        for category in ('old_clear', 'new_clear', 'old_pass', 'new_pass', 'old_clear_to_new_pass') + CONTACTS:
            lines.append('| ' + category + ' | ' + ' | '.join(ME.metric_text(scope['statistics'][arm][category]) for arm in ARMS) + ' |')
        lines += ['', '| 配对差及95%CI | M3−NEAR | M8−NEAR |', '| --- | --- | --- |']
        for category in ('old_clear', 'new_clear', 'old_pass', 'new_pass'):
            cells = []
            for arm in ('M3', 'M8'):
                d = scope['differences'][arm][category]
                cells.append(f"{ME.pp(d['delta'])} [{ME.pp(d['ci95'][0])}, {ME.pp(d['ci95'][1])}]")
            lines.append('| ' + category + ' | ' + ' | '.join(cells) + ' |')
        lines += ['', '| 类别转移 | 行数 | NEAR报警 | M3报警 | M8报警 |', '| --- | --- | --- | --- | --- |']
        for transition, values in scope['transitions'].items():
            lines.append(f"| {transition} | {values['rows']} | " + ' | '.join(str(values['alarms'][a]) for a in ARMS) + ' |')
        lines.append('')
    lines += ['## 原判据在新清晰定义下的描述性复核', '',
              '| 数据 | 臂 | 原判读保持 | 各条件布尔值 | 全部条件满足（仅描述） |', '| --- | --- | --- | --- | --- |']
    for dataset, arms in result['descriptive_recheck'].items():
        for arm, item in arms.items():
            lines.append(f"| {dataset} | {arm} | {item['original_verdict_unchanged']} | {json.dumps(item['checks'])} | {item['all_checks_met']} |")
    lines += ['', '已消费Development仿真几何审计，不是重新确认，也不是实机效果。新清晰集合更窄，率的分母变化需与报警计数一起解释；擦身不计误报不等于没有提醒负担。',
              '使用既有表面与查询体相交的标签语义，不推断实体体积占据、人体实际停止或场景安全。完整输入与模型SHA、精确复现检查见result.json。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def run():
    if (OUT / 'result.json').exists():
        old = read(OUT / 'result.json')
        if old.get('status') == 'AUDIT_COMPLETE':
            print('Existing AUDIT_COMPLETE; not repeated', flush=True)
            return old
        raise RuntimeError('Existing incomplete audit requires inspection; no silent overwrite')
    ledger, scores, thresholds, source, context, provenance = input_data()
    scopes = audit_arrays(ledger, scores, thresholds)
    verification = verify_original(scopes, source, context)
    result = dict(status='AUDIT_COMPLETE', thresholds_unchanged=thresholds,
        role='Fixed-threshold evaluator truth audit only; no original verdict changes',
        scopes=scopes, original_metric_verification=verification,
        descriptive_recheck=descriptive_recheck(scopes, source, context), provenance=provenance,
        bootstrap=dict(draws=1000, seeds=SEEDS, unit='whole units paired across arms',
                       conditioning='fixed original thresholds; no recalibration uncertainty'),
        operations=dict(inference=False, calibration=False, training=False, rendering=False))
    # Guard publication against concurrent changes to the exact inputs we used.
    if input_data()[-1] != provenance:
        raise ValueError('Audit inputs changed during statistics')
    write_report(result)
    save(OUT / 'result.json', result)
    print('AUDIT_COMPLETE', json.dumps(result['descriptive_recheck']), flush=True)
    return result


def check():
    rows = []
    for u in (0, 1):
        for c, (old, new, off) in enumerate([('clear', 'pass', -.15), ('clear', 'clear', -.2),
                                            ('pass', 'pass', -.05), ('contact', 'contact', .01)]):
            rows.append(dict(dataset='source', split='evaluation', unit=u, config=c, query=0,
                             target_group=0, target_off=off, old_category=old, all_category=new,
                             family='fixture', range=1.))
    scores = {a: {str(u): np.asarray([[1, 0], [0, 0], [1, 0], [1, 0]], float) for u in (0, 1)} for a in ARMS}
    s = scope_statistics(rows, scores, {a: {'0': .5, '1': .5} for a in ARMS}, [0, 1], SEEDS['source'], np.ones(8, bool))
    assert s['transitions']['clear->pass'] == dict(rows=2, alarms={a: 2 for a in ARMS})
    for a in ARMS:
        m = s['statistics'][a]
        assert m['old_clear']['denominator'] == 4 and m['old_clear']['alarms'] == 2
        assert m['new_clear']['denominator'] == 2 and m['new_clear']['alarms'] == 0
        assert m['old_pass']['denominator'] == 2 and m['new_pass']['denominator'] == 4
        assert m['contact_0_2cm']['denominator'] == 2 and m['contact_0_2cm']['rate'] == 1
    assert s['differences']['M3']['new_clear'] == dict(delta=0., ci95=[0., 0.])
    assert s['statistics']['NEAR']['contact_gt5cm']['rate'] is None
    json.dumps(s, allow_nan=False)
    print('PASS synthetic transition alarm counts, denominator migration, contact preservation and paired-unit intervals; no real scores evaluated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('check', 'run'), required=True)
    args = parser.parse_args()
    check() if args.stage == 'check' else run()
