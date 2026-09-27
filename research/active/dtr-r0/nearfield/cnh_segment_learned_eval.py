"""Frozen segmentation-conditioned readout evaluation; no model/audit selection.

Prediction NPZ: labels/main/witness/strata/config/frame/split, S2, NN,
IDEAL__condition, AUG__condition, M1__condition. Calib thresholds are dataset-
specific; trained models must be frozen on v2 before any v4 prediction access.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
import cnh_learned_readout as L

BOOTSTRAPS, SEED = 10000, 20260928
AP_FLOOR = -1e6
BUDGETS = (.05, .10, .20)
CONDITIONS = ('IDEAL', 'SHIFT025', 'SHIFT05', 'SHIFT1', 'DILATE05', 'DILATE1',
              'DROP20', 'DROP40', 'FA1', 'FA3', 'MIXED')


def expected_cohort(dataset):
    if dataset not in ('v2', 'v4'):
        raise ValueError('Dataset must be v2 or v4')
    if dataset == 'v4':
        return {'calib': list(range(32)), 'audit': list(range(32, 96))}
    return {'calib': list(range(96, 128)),
            'audit': [u for u in range(128, 192) if u != 143]}


def load(root, dataset, conditions):
    if len(set(conditions)) != len(conditions) or set(conditions) != set(CONDITIONS):
        raise ValueError('All eleven fixed conditions required; no partial condition evaluation')
    arms = ('S2', 'NN', *(f'{kind}__{c}' for kind in ('IDEAL', 'AUG', 'M1') for c in conditions))
    expected = expected_cohort(dataset)
    units, splits = {}, {'calib': [], 'audit': []}
    for path in sorted(Path(root).glob('unit*.npz'), key=lambda p: int(p.stem[4:])):
        u = int(path.stem[4:])
        if u in units:
            raise ValueError('Duplicate unit filename identity')
        with np.load(path, allow_pickle=False) as f:
            split = str(f['split'])
            if split not in splits or u not in expected[split]:
                raise ValueError(f'{path}: wrong dataset cohort/split')
            n = len(f['frame'])
            for key in ('labels', 'witness', 'strata', *arms):
                if f[key].shape != (n, 6):
                    raise ValueError(f'{path}: invalid {key} shape')
            if f['main'].shape != (n,) or f['main'].dtype != np.bool_ or f['config'].shape != (n,):
                raise ValueError('main must be boolean[N], config/frame aligned')
            if not np.isin(f['labels'], (0, 1)).all():
                raise ValueError('Labels must be binary')
            for cfg in np.unique(f['config']):
                frames = np.sort(f['frame'][f['config'] == cfg])
                if not np.array_equal(frames, np.arange(12)):
                    raise ValueError('Each config needs exact frames0..11')
            item = dict(y=f['labels'], main=f['main'], w=f['witness'], strata=f['strata'],
                        config=f['config'], frame=f['frame'], split=split)
            for arm in arms:
                score = f[arm]
                if np.isnan(score).any() or np.isposinf(score).any() or (not arm.startswith('M1__') and not np.isfinite(score).all()):
                    raise ValueError(f'Invalid scores {arm}; only M1 may have -inf')
                if np.any(np.isfinite(score) & (score <= AP_FLOOR)):
                    raise ValueError('Finite scores must exceed fixed AP floor')
                item[arm] = score
            units[u] = item
            splits[split].append(u)
    if splits != expected:
        raise ValueError('Incomplete cohort; no partial audit evaluation')
    return units, splits, arms


def unit_ap(units, keys, arms):
    out = {g: {} for g, _ in L.GROUPS}
    for group, boxes in L.GROUPS:
        for u in keys:
            d, row = units[u], {}
            y = d['y'][d['main']][:, boxes].ravel()
            if not 0 < y.sum() < len(y):
                continue
            for arm in arms:
                raw = d[arm][d['main']][:, boxes].ravel()
                score = np.where(np.isneginf(raw), AP_FLOOR, raw)
                row[arm] = float(average_precision_score(y, score))
            out[group][str(u)] = row
    return out


def ap_comparisons(per_unit, arms):
    macro, paired = {}, {}
    for group, rows in per_unit.items():
        ids = sorted(rows, key=int)
        if not ids:
            raise ValueError(f'{group}: no units with both classes')
        arrays = {a: np.array([rows[u][a] for u in ids]) for a in arms}
        macro[group] = {a: float(v.mean()) for a, v in arrays.items()}
        draws = np.random.default_rng(SEED).integers(len(ids), size=(BOOTSTRAPS, len(ids)))
        paired[group] = {}
        for arm in arms:
            for baseline in ('NN', 'S2'):
                difference = arrays[arm]-arrays[baseline]
                boot = difference[draws].mean(1)
                paired[group][f'{arm}-minus-{baseline}'] = dict(delta=float(difference.mean()),
                    ci95=np.quantile(boot, [.025, .975]).tolist(), units=len(ids),
                    units_positive=int((difference > 0).sum()), unit_ids=[int(u) for u in ids])
    return macro, paired


def clean_threshold(threshold):
    return float(threshold) if np.isfinite(threshold) else ('disabled' if threshold > 0 else 'all')


def evaluate(units, splits, arms, dataset, conditions):
    per_unit = unit_ap(units, splits['audit'], arms)
    macro, paired = ap_comparisons(per_unit, arms)
    report = dict(scope='Consumed v2 Development' if dataset == 'v2' else 'Frozen v2-trained models evaluated on v4; no v4 model selection',
        dataset=dataset, unit_ids=splits, conditions=list(conditions), arms=list(arms),
        AP_definition='Per-unit group AP on saved main mask, then equal-weight unit macro; -inf M1 scores replaced by fixed -1e6 for AP only',
        uncertainty=dict(bootstrap_units=BOOTSTRAPS, seed=SEED, paired=True, selection='none on audit',
                         note='Descriptive percentile intervals; no multiplicity-adjusted significance claim'),
        threshold_definition='Unmodified L.threshold_for_budget on dataset-own calib, each group/arm/condition separately; original scores retained; L.pairs and L.summary rule1/1',
        per_unit_AP=per_unit, audit_macro_AP=macro, paired=paired, alerts={})
    cal = L.sequences(units, splits['calib'], arms)
    aud = L.sequences(units, splits['audit'], arms)
    for group, boxes in L.GROUPS:
        for arm in arms:
            for budget in BUDGETS:
                threshold = L.threshold_for_budget(cal, arm, boxes, budget)
                cr = L.pairs(cal, arm, (1, 1), boxes, threshold)
                ar = L.pairs(aud, arm, (1, 1), boxes, threshold)
                report['alerts'][f'{group}|{arm}|{budget:.2f}'] = dict(threshold=clean_threshold(threshold),
                    calib=L.summary(cr), **{s: L.summary(ar, None if s == 'all' else s) for s in ('all', 'tiny', 'realistic', 'wide')})
    gate = {g: paired[g]['AUG__SHIFT05-minus-NN']['ci95'][0] > 0 for g, _ in L.GROUPS}
    report['primary_robustness'] = dict(criterion='AUG SHIFT05 paired macro-AP CI lower bound >0 vs NN in both groups',
                                      by_group=gate, passes_both_groups=all(gate.values()),
                                      alarm_tradeoffs='Report all three budgets; AP gate does not imply equal actual false alerts')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--dataset', choices=('v2', 'v4'), required=True)
    p.add_argument('--conditions', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    units, splits, arms = load(args.predictions, args.dataset, args.conditions)
    report = evaluate(units, splits, arms, args.dataset, args.conditions)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'AP': report['audit_macro_AP'], 'primary': report['primary_robustness']}, indent=2))


if __name__ == '__main__':
    main()
