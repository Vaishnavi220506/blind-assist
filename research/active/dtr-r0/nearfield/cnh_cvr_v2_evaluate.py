"""Frozen CVR v2 evaluation; no training, selection, or changes to prior evidence.

Run only after the root runner has terminated training. Incomplete predictions are
reported as missing coverage, never as a successful main-method decision.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cnh_corridor_late_fusion import FAMILIES, metrics, ranks, threshold
from cnh_corridor_statistics import paired_cluster_ber

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
DEFAULT_OUT = WORK / 'cnh-cvr-v2-20260929'
GROUPS = ('HEAD', 'BODY')
REFERENCES = ('LOFO', 'MOTION8_PARTIAL', 'EQUAL_COUNT')


def calibrate(cal_scores, cal_labels, scores):
    """Calibrate last-frame negatives with exact integer false-positive budget."""
    reference = np.asarray(cal_scores)[np.asarray(cal_labels) == 0]
    n = len(reference)
    cr = ranks(reference, reference)
    k = threshold(cr, n)
    er = ranks(reference, scores)
    fp = int((cr >= k).sum())
    assert 10 * fp <= n
    assert k == 0 or 10 * int((cr >= k - 1).sum()) > n
    return er >= k, dict(negative=n, fp=fp, rank_threshold=k, fpr=fp/n), er


def decide(family, bootstrap):
    if bootstrap['point'] is None or bootstrap['ci'] is None:
        return False
    if family in ('sidewall', 'mixed_surface'):
        return bool(bootstrap['point'] <= -.05 and bootstrap['ci'][1] < 0)
    return bool(bootstrap['point'] <= .02)


def validate_prediction(d, expected):
    """Demand full split coverage and exact identities before using any score."""
    n = len(d['scores'])
    assert d['scores'].shape == (n, 2) and np.isfinite(d['scores']).all()
    assert d['labels'].shape == (n, 2) and np.isin(d['labels'], [0, 1]).all()
    assert all(d[k].shape == (n,) for k in ('unit', 'config', 'family'))
    keys = list(zip(d['unit'].astype(int), d['config'].astype(int)))
    assert len(set(keys)) == n and set(keys) == set(expected), 'missing/duplicate split identities'
    for i, key in enumerate(keys):
        b = expected[key]
        assert str(d['family'][i]) == b['family']
        assert np.array_equal(d['labels'][i], b['labels']), 'label identity mismatch'
    return keys


def main(out=DEFAULT_OUT):
    out = Path(out)
    assert (out/'CVR_V2_PLAN.md').exists(), 'freeze plan before evaluation'
    assert not (out/'results.json').exists(), 'refusing to overwrite consumed evaluation'
    terminal_path = out/'training_terminal.json'
    terminal = json.loads(terminal_path.read_text(encoding='utf8'))
    assert terminal['status'] in ('COMPLETE', 'STOPPED_BUDGET'), 'training must terminate before evaluation'
    hashes = {}

    def record(path):
        path = Path(path)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read(path):
        record(path)
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                       allow_nan=False)+'\n', encoding='utf8')

    for p in (Path(__file__), Path(__file__).with_name('cnh_corridor_late_fusion.py'),
              Path(__file__).with_name('cnh_corridor_statistics.py'), out/'CVR_V2_PLAN.md', terminal_path):
        record(p)
    old_path = WORK/'cnh-corridor-late-fusion-20260929/sample_ledger.jsonl'
    record(old_path)
    old_rows = [json.loads(line) for line in old_path.read_text(encoding='utf8').splitlines()]
    old = {(r['unit'], r['config'], r['group']): r for r in old_rows}
    assert len(old) == len(old_rows) == 2112
    base = {}
    for split, units in [('calib', range(2000, 2024)), ('evaluation', range(3000, 3048))]:
        base[split] = {}
        for u in units:
            b = read(WORK/'cnh-corridor-retrain-20260929/predictions'/f'unit{u}.npz')
            # Original retrain NPZ carries unit identity only in its filename.
            if 'unit' in b:
                assert int(b['unit']) == u
            assert b['labels'].shape == (22, 2) and b['scores'].shape == (22, 6, 2)
            assert set(b['family']) == set(FAMILIES) and np.isin(b['labels'], [0, 1]).all()
            for i in range(len(b['labels'])):
                row = dict(family=str(b['family'][i]), labels=b['labels'][i],
                           score=b['scores'][i, 2], margin=float(b['margin'][i]),
                           target_group=int(b['group'][i]))
                base[split][u, i] = row
                if split == 'evaluation':
                    for q, group in enumerate(GROUPS):
                        r = old[u, i, group]
                        assert r['label'] == int(row['labels'][q])
                        assert r['family'] == row['family'] and r['margin'] == row['margin']
                        assert r['target_group'] == row['target_group']

    coverage, calibration, comparisons, pooled, ledger, depth = [], [], [], [], [], []
    for arm, folds in [('CVR', (*FAMILIES, 'allfamily')), ('DUAL', FAMILIES)]:
        for fold in folds:
            paths = {s: out/'predictions'/arm/fold/f'{s}.npz' for s in base}
            available = all(p.exists() for p in paths.values())
            coverage.append(dict(arm=arm, fold=fold, status='COMPLETE' if available else 'NOT_COMPLETED',
                                 splits={s: p.exists() for s, p in paths.items()}))
            if not available:
                continue
            sets = {}
            for split, path in paths.items():
                expected = {k: r for k, r in base[split].items() if fold == 'allfamily' or
                            ((r['family'] != fold) if split == 'calib' else (r['family'] == fold))}
                d = read(path)
                keys = validate_prediction(d, expected)
                sets[split] = d, keys
            cal, _ = sets['calib']
            ev, keys = sets['evaluation']
            for q, group in enumerate(GROUPS):
                y, units = ev['labels'][:, q], ev['unit']
                assert len(np.unique(units)) == 48
                pred, c, er = calibrate(cal['scores'][:, q], cal['labels'][:, q], ev['scores'][:, q])
                calibration.append(dict(arm=arm, fold=fold, group=group, **c))
                if fold == 'allfamily':
                    ck = sets['calib'][1]
                    baseline, bc, _ = calibrate([base['calib'][k]['score'][q] for k in ck],
                                               cal['labels'][:, q], [base['evaluation'][k]['score'][q] for k in keys])
                    calibration.append(dict(arm='A2_RETRAIN', fold=fold, group=group, **bc))
                    boot = paired_cluster_ber(y, baseline, pred, units, draws=5000)
                    pooled.append(dict(group=group, metrics={arm: metrics(y, pred),
                        'A2_RETRAIN': metrics(y, baseline)}, bootstrap=boot, passed=bool(boot['point'] <= .02)))
                else:
                    refs = {a: np.array([old[*k, group]['predictions'][a] for k in keys]) for a in REFERENCES}
                    boot = paired_cluster_ber(y, refs['LOFO'], pred, units, draws=5000)
                    comparisons.append(dict(arm=arm, family=fold, group=group,
                        metrics={arm: metrics(y, pred), **{a: metrics(y, refs[a]) for a in REFERENCES}},
                        bootstrap=boot, passed=decide(fold, boot) if arm == 'CVR' else None,
                        descriptive_only=arm == 'DUAL'))
                for i, k in enumerate(keys):
                    r = old[*k, group]
                    ps = {arm: int(pred[i])}
                    if fold == 'allfamily':
                        ps['A2_RETRAIN'] = int(baseline[i])
                    else:
                        ps.update({a: int(refs[a][i]) for a in REFERENCES})
                    ledger.append(dict(arm=arm, fold=fold, unit=int(k[0]), config=int(k[1]),
                        family=r['family'], group=group, label=int(y[i]), target_group=r['target_group'],
                        margin=r['margin'], role=r['role'], band=r['band'], score=float(ev['scores'][i,q]),
                        rank_numerator=int(er[i]), rank_denominator=c['negative'], predictions=ps))

    for arm, folds in [('CVR', (*FAMILIES, 'allfamily')), ('DUAL', FAMILIES)]:
        for fold in folds:
            for family in (FAMILIES if fold == 'allfamily' else (fold,)):
                for group in GROUPS:
                    selected = [r for r in ledger if r['arm'] == arm and r['fold'] == fold and
                                r['family'] == family and r['group'] == group]
                    if not selected:
                        continue
                    for role in ('positive_intrusion', 'target_group_outside', 'other_height_negative'):
                        for band in ('<=5cm', '5-15cm', '>15cm', 'all'):
                            rows = [r for r in selected if r['role'] == role and (band == 'all' or r['band'] == band)]
                            for method in selected[0]['predictions']:
                                depth.append(dict(arm=arm, fold=fold, family=family, group=group, role=role,
                                    band=band, method=method, **metrics([r['label'] for r in rows],
                                                                     [r['predictions'][method] for r in rows])))
    primary_complete = all(c['status'] == 'COMPLETE' for c in coverage if c['arm'] == 'CVR')
    all_complete = all(c['status'] == 'COMPLETE' for c in coverage)
    gate_a = [c for c in comparisons if c['arm'] == 'CVR' and c['family'] in ('sidewall', 'mixed_surface')]
    gate_b = [c for c in comparisons if c['arm'] == 'CVR' and c['family'] in ('boundary', 'general')]
    gates = dict(a=all(c['passed'] for c in gate_a) if len(gate_a) == 4 else None,
                 b=all(c['passed'] for c in gate_b) if len(gate_b) == 4 else None,
                 c=all(c['passed'] for c in pooled) if len(pooled) == 2 else None)
    passed = all(gates.values()) if primary_complete else None
    result = dict(status=('PASS' if passed else 'FAIL') if primary_complete else 'NOT_COMPLETED',
                  main_pass=passed, all_requested_predictions_complete=all_complete,
                  gates=gates, coverage=coverage, comparisons=comparisons, allfamily=pooled,
                  next_action='PAUSE_METHOD_EXPLORATION_AND_PREPARE_PROPOSAL',
                  formal_reproduction_plan_draft_required=passed is True,
                  limitations=['Consumed Development; v2 authorized after observing v1 zero-training outcomes.',
                    'Shared current query alignment uses simulator true travel direction.',
                    'MOTION8_PARTIAL hyperparameters were set using all-family Development.',
                    'DUAL is descriptive only and cannot rescue main CVR criteria.'])
    save('results.json', result)
    save('calibration.json', calibration)
    save('input_source_hashes.json', hashes)
    (out/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger), encoding='utf8')
    if depth:
        with (out/'depth_table.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(depth[0])); w.writeheader(); w.writerows(depth)
    table = ['# CVR v2 evaluation tables', '', f"Main decision: **{result['status']}**", '',
             '|Arm|Fold|Coverage|', '|---|---|---|']
    table += [f"|{c['arm']}|{c['fold']}|{c['status']}|" for c in coverage]
    table += ['', '|Arm|Family|Group|CVR/DUAL BER|LOFO|PARTIAL|Equal-count|Delta pp [95% CI]|Gate|',
              '|---|---|---|---:|---:|---:|---:|---|---|']
    for c in comparisons:
        m, b = c['metrics'], c['bootstrap']
        table.append('|'+ '|'.join([c['arm'], c['family'], c['group'],
            *[f"{100*m[a]['ber']:.2f}%" for a in (c['arm'], *REFERENCES)],
            f"{100*b['point']:.2f} [{100*b['ci'][0]:.2f}, {100*b['ci'][1]:.2f}]",
            str(c['passed']) if not c['descriptive_only'] else 'descriptive'])+'|')
    table += ['', '|Pooled group|CVR BER|All-family A2 BER|Delta pp|Gate (c)|', '|---|---:|---:|---:|---|']
    for c in pooled:
        table.append(f"|{c['group']}|{100*c['metrics']['CVR']['ber']:.2f}%|"
                     f"{100*c['metrics']['A2_RETRAIN']['ber']:.2f}%|{100*c['bootstrap']['point']:.2f}|{c['passed']}|")
    table += ['', '## Complete depth counts', '',
              '|Arm|Fold|Family|Group|Role|Band|Method|FN / positive|FP / negative|',
              '|---|---|---|---|---|---|---|---:|---:|']
    for r in depth:
        table.append('|'+ '|'.join(str(r[k]) for k in ('arm','fold','family','group','role','band','method'))+
                     f"|{r['fn']}/{r['positive']}|{r['fp']}/{r['negative']}|")
    (out/'EVALUATION_TABLES.md').write_text('\n'.join(table)+'\n', encoding='utf8')
    print(json.dumps({k: result[k] for k in ('status', 'main_pass', 'gates', 'coverage')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().out)
