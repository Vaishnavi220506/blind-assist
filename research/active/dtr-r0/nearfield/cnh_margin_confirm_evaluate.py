"""Frozen-model fresh-unit three-level confirmation; no training or rendering.

The independent calibration split sets one 90th-percentile threshold per arm
and HEAD/BODY query on the union of both clear-negative types. Confidence
intervals resample whole evaluation units, conditional on those thresholds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('NEAR', 'M3', 'M8')
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 2026100205
HEADING_SEED = 2026100201
HEADING_DRAWS = 400
RANGES = {'primary_0.6-2.1': (.6, 2.1), 'all_0.6-2.6': (.6, 2.6),
          '0.6-1.2': (.6, 1.2), '1.2-1.6': (1.2, 1.6),
          '1.6-2.1': (1.6, 2.1), '2.1-2.6': (2.1, 2.6)}
RULE = ('Primary M3 minus NEAR contact 0-2 cm alarm rate, evaluation units, '
        '0.6 <= final range < 2.1 m, both conditions pooled: paired unit '
        'bootstrap 1000 percentile 95% lower bound > 0, contact >5 cm '
        'alarm-rate difference >= -0.02, and actual merged-clear alarm-rate '
        'difference over all evaluation ranges <= 0.02. All three conditions '
        'are required. M8 is assessed separately with the same rule; no arm '
        'selection. Heading sigma=1 degree is descriptive only.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def classifications(off, labels, actual=None):
    """Return row membership weights; offset NaN means the other query band."""
    off = np.asarray(off, dtype=float)
    target = np.isfinite(off)
    actual = off[:, None] if actual is None else np.asarray(actual, dtype=float)
    if actual.ndim != 2 or actual.shape[0] != len(off):
        raise ValueError('actual offsets must have shape (rows, draws)')
    target2 = target[:, None]
    other_clear = (~target) & (np.asarray(labels) == 0)
    weight = lambda mask: np.mean(mask & target2, axis=1)
    result = {
        'contact_0_2cm': weight((actual > 0) & (actual <= .02)),
        'contact_2_5cm': weight((actual > .02) & (actual <= .05)),
        'contact_gt5cm': weight(actual > .05),
        'pass_0_10cm': weight((actual >= -.10) & (actual <= 0)),
        'clear_same_height_gt10cm': weight(actual < -.10),
        'clear_other_height': other_clear.astype(float),
    }
    result['clear_merged'] = (result['clear_same_height_gt10cm'] +
                              result['clear_other_height'])
    return result


def load_rows(out, splits):
    """Validate complete disjoint inputs before reading any performance outcome."""
    units = {s: [int(u) for u in splits[s]] for s in ('calib', 'evaluation')}
    if len(units['calib']) != 48 or len(units['evaluation']) != 96:
        raise ValueError('Expected exactly 48 calibration and 96 evaluation units')
    joined = units['calib'] + units['evaluation']
    if len(joined) != len(set(joined)):
        raise ValueError('Calibration and evaluation units must be unique and disjoint')
    manifest_path = out / 'scene_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    lookup = {}
    for row in manifest:
        key = (str(row['split']), int(row['unit']), int(row['config']))
        if key in lookup:
            raise ValueError(f'Duplicate manifest configuration {key}')
        lookup[key] = row
    expected = {(s, u, c) for s, us in units.items() for u in us for c in range(40)}
    if set(lookup) != expected:
        raise ValueError('Manifest must contain exactly 40 configurations for each planned unit')
    scores = {}
    for arm in ARMS:
        with np.load(out / f'scores_{arm}.npz', allow_pickle=False) as data:
            if set(data.files) != set(map(str, joined)):
                raise ValueError(f'{arm}: score units differ from the planned splits')
            scores[arm] = {u: np.asarray(data[str(u)], dtype=np.float64) for u in joined}
        if any(x.shape != (40, 2) or not np.isfinite(x).all() for x in scores[arm].values()):
            raise ValueError(f'{arm}: expected finite scores shaped (40,2) per unit')
    rows = []
    for split, us in units.items():
        for u in us:
            for c in range(40):
                m = lookup[(split, u, c)]
                target_group = int(m['group'])
                labels = np.asarray(m['labels'])
                if target_group not in (0, 1) or labels.shape != (2,) or not np.isin(labels, (0, 1)).all():
                    raise ValueError(f'Invalid group/labels for {split}/{u}/{c}')
                off, distance = float(m['off']), float(m['range'])
                if not (-.20 <= off <= .15 and .6 <= distance < 2.6):
                    raise ValueError(f'Unexpected offset/range for {split}/{u}/{c}: {off}, {distance}')
                for q in (0, 1):
                    rows.append(dict(split=split, unit=u, config=c, group=q,
                                     off=off if q == target_group else np.nan,
                                     label=int(labels[q]), range=distance, cond=str(m['cond']),
                                     scores=[float(scores[a][u][c, q]) for a in ARMS]))
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    hashes = {'scene_manifest.json': sha(manifest_path),
              **{f'scores_{a}.npz': sha(out / f'scores_{a}.npz') for a in ARMS}}
    return arrays, units, hashes


def calibrate(rows):
    clear = classifications(rows['off'], rows['label'])['clear_merged'] > 0
    thresholds, counts = {}, {}
    for ai, arm in enumerate(ARMS):
        thresholds[arm], counts[arm] = {}, {}
        for q in (0, 1):
            keep = (rows['split'] == 'calib') & (rows['group'] == q) & clear
            if not keep.any():
                raise ValueError(f'No calibration clear rows for group {q}')
            threshold = float(np.quantile(rows['scores'][keep, ai], .9))
            thresholds[arm][str(q)] = threshold
            counts[arm][str(q)] = dict(clear_rows=int(keep.sum()),
                                      alarms=int((rows['scores'][keep, ai] >= threshold).sum()))
    return thresholds, counts


def bootstrap_weights(n_units):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return np.asarray([np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
                       for _ in range(BOOTSTRAPS)])


def ci95(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return ([float(x) for x in np.percentile(finite, [2.5, 97.5])]
            if len(finite) else [None, None])


def summarize(weights, alarm, unit_index, boot):
    """One row can carry fractional expected membership under heading draws."""
    den = np.bincount(unit_index, weights=weights, minlength=boot.shape[1])
    num = np.bincount(unit_index, weights=weights * alarm, minlength=boot.shape[1])
    bden = boot @ den
    brate = np.divide(boot @ num, bden, out=np.full(len(boot), np.nan), where=bden > 0)
    result = dict(alarms=float(num.sum()), denominator=float(den.sum()),
                  contributing_rows=int((weights > 0).sum()),
                  contributing_units=int((den > 0).sum()),
                  rate=float(num.sum() / den.sum()) if den.sum() else None,
                  ci95=ci95(brate), bootstrap_valid=int(np.isfinite(brate).sum()))
    return result, brate


def comparison_verdict(arm, shallow_delta_ci, deep_delta, clear_delta):
    checks = dict(shallow_ci_lower_gt_zero=shallow_delta_ci[0] is not None and shallow_delta_ci[0] > 0,
                  deep_drop_le_2pp=deep_delta is not None and deep_delta >= -.02,
                  clear_increase_le_2pp=clear_delta is not None and clear_delta <= .02)
    return (f'{arm}_CONFIRMED' if all(checks.values()) else 'NOT_CONFIRMED'), checks


def evaluate_arrays(rows, units):
    thresholds, calibration = calibrate(rows)
    keep = rows['split'] == 'evaluation'
    ev = {k: v[keep] for k, v in rows.items()}
    unit_lookup = {u: i for i, u in enumerate(units['evaluation'])}
    ui = np.asarray([unit_lookup[int(u)] for u in ev['unit']])
    boot = bootstrap_weights(len(unit_lookup))
    alarms = {a: ev['scores'][:, ai] >= np.asarray([thresholds[a][str(q)] for q in ev['group']])
              for ai, a in enumerate(ARMS)}
    target = np.isfinite(ev['off'])
    actual = np.broadcast_to(ev['off'][:, None], (len(target), HEADING_DRAWS)).copy()
    z = np.random.default_rng(HEADING_SEED).standard_normal((int(target.sum()), HEADING_DRAWS))
    actual[target] += ev['range'][target, None] * np.tan(np.deg2rad(1.) * z)
    memberships = {'sigma0': classifications(ev['off'], ev['label']),
                   'sigma1': classifications(ev['off'], ev['label'], actual)}
    statistics, bootstrap_rates = {}, {}
    for version, categories in memberships.items():
        statistics[version], bootstrap_rates[version] = {}, {}
        for range_name, (lo, hi) in RANGES.items():
            range_keep = (ev['range'] >= lo) & (ev['range'] < hi)
            statistics[version][range_name], bootstrap_rates[version][range_name] = {}, {}
            for arm in ARMS:
                metrics, brates = {}, {}
                for name, membership in categories.items():
                    metrics[name], brates[name] = summarize(membership * range_keep, alarms[arm], ui, boot)
                statistics[version][range_name][arm] = metrics
                bootstrap_rates[version][range_name][arm] = brates
    comparisons = {}
    for version in memberships:
        comparisons[version] = {}
        for arm in ('M3', 'M8'):
            deltas = {}
            for name, range_name in [('contact_0_2cm', 'primary_0.6-2.1'),
                                      ('contact_2_5cm', 'primary_0.6-2.1'),
                                      ('contact_gt5cm', 'primary_0.6-2.1'),
                                      ('pass_0_10cm', 'primary_0.6-2.1'),
                                      ('clear_merged', 'all_0.6-2.6')]:
                a = statistics[version][range_name][arm][name]['rate']
                n = statistics[version][range_name]['NEAR'][name]['rate']
                br = bootstrap_rates[version][range_name]
                deltas[name] = dict(delta=a - n if a is not None and n is not None else None,
                                    ci95=ci95(br[arm][name] - br['NEAR'][name]), range=range_name)
            verdict, checks = comparison_verdict(arm, deltas['contact_0_2cm']['ci95'],
                                                  deltas['contact_gt5cm']['delta'],
                                                  deltas['clear_merged']['delta'])
            comparisons[version][arm] = dict(differences=deltas)
            if version == 'sigma0':
                comparisons[version][arm].update(verdict=verdict, checks=checks)
            else:
                comparisons[version][arm]['role'] = 'DESCRIPTIVE_ONLY; no secondary confirmation verdict'
    return dict(status='COMPLETE', verdict=comparisons['sigma0']['M3']['verdict'],
                rule=RULE, thresholds=thresholds, calibration=calibration,
                units=units, n=dict(calibration_units=48, evaluation_units=96,
                                   calibration_query_rows=int((rows['split'] == 'calib').sum()),
                                   evaluation_query_rows=len(ev['off']),
                                   evaluation_target_rows=int(target.sum()),
                                   other_height_nonzero_label_excluded_from_clear=int((~target & (ev['label'] != 0)).sum())),
                statistics=statistics, comparisons=comparisons,
                bootstrap=dict(draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED, resampling='whole evaluation units; paired across arms',
                               calibration_uncertainty='not resampled; intervals conditional on calibrated thresholds'),
                heading=dict(sigma_deg=1, draws=HEADING_DRAWS, seed=HEADING_SEED,
                             model='actual intrusion = nominal intrusion + final range * tan(delta); delta ~ Normal(0,1 degree)',
                             thresholds='unchanged sigma0 thresholds; shared draws across arms',
                             denominator='expected row counts averaged over 400 draws; not 400 independent units',
                             support='nominal intrusion support [-0.20,+0.15] m; shifted endpoint/open-ended categories are support-limited, not an unbounded population estimate'),
                limits=['EXPLORE on newly generated simulator units under a previously consumed Development recipe; not protected confirmation or hardware evidence',
                        'Final-frame alarm rates, not timely sequence stopping or real walking false-alert burden',
                        'Heading perturbation changes evaluator truth only; no changed observations, physical trajectory, reaction or replanning',
                        'Clear budget merges other-height label-zero queries and same-height targets more than 10 cm outside; near-pass alarms are descriptive',
                        'Old failed margin-label and heading verdicts are unchanged'])


def percentage(value):
    return '--' if value is None else f'{100 * value:.2f}%'


def pp(value):
    return '--' if value is None else f'{100 * value:+.2f}pp'


def metric_text(metric):
    num, den = metric['alarms'], metric['denominator']
    return (f"{num:.3f}/{den:.3f} = {percentage(metric['rate'])} "
            f"[{percentage(metric['ci95'][0])}, {percentage(metric['ci95'][1])}]")


def write_report(result, path):
    lines = [result['verdict'], '', '# 新单位三级真值冻结模型确认', '',
             '48 个单位校准，96 个单位评估；每臂每 HEAD/BODY 组在合并清晰行上取 90% 分位阈值。',
             '区间为 1000 次配对单位 bootstrap 的 95% 百分位区间，条件于已定校准阈值。', '',
             '主判据：0.6–2.1m、两条件合并，浅擦碰 M3−NEAR 区间下界>0，>5cm 降幅≤2pp，且全距离实际清晰报警增幅≤2pp。M8 独立同规则，不择优。', '',
             '| 臂 | 判读 | 浅擦碰差及95%区间 | >5cm差 | 全距离清晰差 |',
             '| --- | --- | --- | --- | --- |']
    for arm in ('M3', 'M8'):
        comparison = result['comparisons']['sigma0'][arm]
        d = comparison['differences']
        shallow = d['contact_0_2cm']
        lines.append(f"| {arm} | {comparison['verdict']} | {pp(shallow['delta'])} [{pp(shallow['ci95'][0])}, {pp(shallow['ci95'][1])}] | {pp(d['contact_gt5cm']['delta'])} | {pp(d['clear_merged']['delta'])} |")
    for version in ('sigma0', 'sigma1'):
        lines += ['', f'## {version}：报警数/分母、报警率及95%区间', '',
                  '| 指标 | NEAR | M3 | M8 |', '| --- | --- | --- | --- |']
        for name, label, scope in [
                ('contact_0_2cm', '0–2cm擦碰；0.6–2.1m', 'primary_0.6-2.1'),
                ('contact_2_5cm', '2–5cm擦碰；0.6–2.1m', 'primary_0.6-2.1'),
                ('contact_gt5cm', '>5cm擦碰；0.6–2.1m', 'primary_0.6-2.1'),
                ('pass_0_10cm', '0–10cm擦身；0.6–2.1m，不计误报', 'primary_0.6-2.1'),
                ('clear_merged', '合并清晰；全0.6–2.6m，主护栏', 'all_0.6-2.6'),
                ('clear_other_height', '另一高度清晰；全距离', 'all_0.6-2.6'),
                ('clear_same_height_gt10cm', '同高度外>10cm清晰；全距离', 'all_0.6-2.6'),
                ('clear_merged', '合并清晰；0.6–2.1m，描述', 'primary_0.6-2.1')]:
            lines.append('| ' + label + ' | ' + ' | '.join(metric_text(result['statistics'][version][scope][a][name]) for a in ARMS) + ' |')
    lines += ['', '## 边界', '',
              'σ1°仅作次要描述：沿用σ0阈值与固定模型，按末帧距离扰动真值，每目标400个共同抽样；期望加权分母不代表新独立案例。名义伸入支持为−20至+15cm，端点及开放区间受支持范围限制。',
              '本轮是既有Development配方下的新仿真单位EXPLORE，不是保护测试或实机证据；这里为末帧报警率，不是及时停步率或现场误停/分钟。旧失败判读不变。',
              '完整各距离分母、区间、阈值、配对差、输入哈希和独立单位列表见 result.json。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def evaluate(out=None, splits=None):
    if out is None or splits is None:
        import cnh_margin_confirm as MC
        out, splits = MC.OUT, MC.SPLITS
    out = Path(out)
    if (out / 'result.json').exists():
        previous = json.loads((out / 'result.json').read_text(encoding='utf-8'))
        if previous.get('status') == 'COMPLETE':
            print('Existing COMPLETE result; evaluation not repeated:', out, flush=True)
            return previous
        raise RuntimeError('Existing incomplete result requires inspection, not overwrite')
    if not (out / 'PLAN.json').is_file():
        raise RuntimeError('Run plan must exist before evaluation')
    rows, units, hashes = load_rows(out, splits)
    result = evaluate_arrays(rows, units)
    result['provenance'] = dict(input_sha256=hashes, plan_sha256=sha(out / 'PLAN.json'),
                                evaluator_sha256=sha(__file__))
    write_report(result, out / 'REPORT.md')
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(result['verdict'], flush=True)
    for arm in ('M3', 'M8'):
        print(arm, json.dumps(result['comparisons']['sigma0'][arm], ensure_ascii=False), flush=True)
    return result


def check():
    """Synthetic checks for boundary semantics, guards and independent calibration."""
    off = np.asarray([-.20, -.10, 0., .02, .05, .15, np.nan, np.nan])
    labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1])
    groups = classifications(off, labels)
    assert groups['clear_merged'].tolist() == [1, 0, 0, 0, 0, 0, 1, 0]
    assert groups['pass_0_10cm'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert groups['contact_0_2cm'].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert groups['contact_2_5cm'][4] == 1 and groups['contact_gt5cm'][5] == 1
    assert comparison_verdict('M3', [.001, .1], -.02, .02)[0] == 'M3_CONFIRMED'
    for ci, deep, clear in [([0, .1], 0, 0), ([.01, .1], -.02001, 0), ([.01, .1], 0, .02001)]:
        assert comparison_verdict('M3', ci, deep, clear)[0] == 'NOT_CONFIRMED'
    rows = dict(off=np.asarray([np.nan] * 8), label=np.zeros(8),
                split=np.asarray(['calib'] * 4 + ['evaluation'] * 4),
                group=np.asarray([0, 0, 1, 1] * 2),
                scores=np.repeat(np.asarray([0., 1., 2., 3., 90., 91., 92., 93.])[:, None], 3, axis=1))
    thresholds, _ = calibrate(rows)
    assert thresholds['NEAR'] == {'0': .9, '1': 2.9}
    rows['scores'][4:] = -1000
    assert calibrate(rows)[0] == thresholds
    boot = bootstrap_weights(2)
    assert np.all(boot.sum(axis=1) == 2)
    metric, rates = summarize(np.asarray([1., 1., .5, .5]), np.asarray([1, 1, 0, 0]),
                              np.asarray([0, 0, 1, 1]), boot)
    assert np.isclose(metric['rate'], 2 / 3) and metric['contributing_units'] == 2
    assert np.all(rates[(boot[:, 0] == 2)] == 1) and np.all(rates[(boot[:, 1] == 2)] == 0)
    print('PASS: three-level boundaries, decision guards, independent calibration, whole-unit bootstrap')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'evaluate'))
    args = parser.parse_args()
    check() if args.stage == 'check' else evaluate()
