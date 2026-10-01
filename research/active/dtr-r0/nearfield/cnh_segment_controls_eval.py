"""Frozen foreground-coverage controls: paired AP and two threshold policies."""
import argparse
import json
from pathlib import Path
import numpy as np
import cnh_segment_learned_eval as E

L = E.L
CONDITIONS = ('IDEAL', 'FA5', 'FA10', 'BG', 'BG_FA5_SHIFT05', 'BG_FA10_SHIFT05')
FAMILIES = ('IDEAL', 'AUG', 'ZERO_IDEAL', 'ZERO_AUG', 'MASK')
ARMS = ('S2', 'NN', *(f'{f}__{c}' for f in FAMILIES for c in CONDITIONS))


def load(root, dataset):
    expected = E.expected_cohort(dataset)
    units, splits = {}, {'calib': [], 'audit': []}
    for path in sorted(Path(root).glob('unit*.npz'), key=lambda p: int(p.stem[4:])):
        u = int(path.stem[4:])
        if u in units:
            raise ValueError('Duplicate unit')
        with np.load(path, allow_pickle=False) as f:
            split = str(f['split'])
            if split not in splits or u not in expected[split]:
                raise ValueError('Wrong dataset cohort/split')
            n = len(f['frame'])
            for k in ('labels', 'witness', 'strata', *ARMS):
                if f[k].shape != (n, 6):
                    raise ValueError(f'Invalid shape: {k}')
            if f['main'].shape != (n,) or f['main'].dtype != np.bool_ or f['config'].shape != (n,):
                raise ValueError('Invalid main/config metadata')
            if not np.isin(f['labels'], (0, 1)).all():
                raise ValueError('Labels must be binary')
            if np.any((f['labels'] == 1) & ~np.isfinite(f['witness'])):
                raise ValueError('Positive frames need finite witness')
            for c in np.unique(f['config']):
                if not np.array_equal(np.sort(f['frame'][f['config'] == c]), np.arange(12)):
                    raise ValueError('Require exact frame0..11 for each config')
            d = dict(y=f['labels'], main=f['main'], w=f['witness'], strata=f['strata'],
                     config=f['config'], frame=f['frame'], split=split)
            for a in ARMS:
                d[a] = f[a]
                if not np.isfinite(d[a]).all():
                    raise ValueError(f'Nonfinite score: {a}')
            units[u] = d
            splits[split].append(u)
    if splits != expected:
        raise ValueError('Full cohort required')
    return units, splits


def comparisons(arms=ARMS):
    pairs = {(a, 'NN') for a in arms}
    for a in arms:
        if '__' not in a:
            continue
        family, condition = a.split('__')
        pairs.add((a, f'MASK__{condition}'))
        if family in ('IDEAL', 'AUG'):
            pairs.add((a, f'ZERO_{family}__{condition}'))
    return sorted(pairs)


def ap_comparisons(per_unit, arms=ARMS):
    macro, paired = {}, {}
    for group, rows in per_unit.items():
        ids = sorted(rows, key=int)
        if not ids:
            raise ValueError('No units with both classes')
        values = {a: np.array([rows[u][a] for u in ids]) for a in arms}
        macro[group] = {a: float(v.mean()) for a, v in values.items()}
        draws = np.random.default_rng(E.SEED).integers(len(ids), size=(E.BOOTSTRAPS, len(ids)))
        paired[group] = {}
        for a, baseline in comparisons(arms):
            delta = values[a]-values[baseline]
            paired[group][f'{a}-minus-{baseline}'] = dict(delta=float(delta.mean()),
                ci95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                units=len(ids), unit_ids=[int(u) for u in ids], units_positive=int((delta > 0).sum()))
    return macro, paired


def threshold_source(arm, policy):
    if policy == 'per_condition' or '__' not in arm:
        return arm
    if policy != 'frozen_ideal':
        raise ValueError('Unknown threshold policy')
    return arm.split('__')[0]+'__IDEAL'


def alarm_results(units, splits, arms=ARMS):
    ca, au = [L.sequences(units, splits[s], arms) for s in ('calib', 'audit')]
    thresholds = {(g, a, b): L.threshold_for_budget(ca, a, boxes, b)
                  for g, boxes in L.GROUPS for a in arms for b in E.BUDGETS}
    out = {policy: {} for policy in ('per_condition', 'frozen_ideal')}
    for policy in out:
        for g, boxes in L.GROUPS:
            for a in arms:
                source = threshold_source(a, policy)
                for b in E.BUDGETS:
                    t = thresholds[g, source, b]
                    calib, audit = [L.pairs(seqs, a, (1, 1), boxes, t) for seqs in (ca, au)]
                    out[policy][f'{g}|{a}|{b:.2f}'] = dict(threshold=E.clean_threshold(t),
                        threshold_calibrated_on=source, calib=L.summary(calib),
                        **{s: L.summary(audit, None if s == 'all' else s) for s in ('all', 'tiny', 'realistic', 'wide')})
    return out


def evaluate(units, splits, dataset):
    ap = E.unit_ap(units, splits['audit'], ARMS)
    macro, paired = ap_comparisons(ap)
    alarms = alarm_results(units, splits)
    compact = []
    for group, _ in L.GROUPS:
        for arm in ARMS:
            compact.append(dict(group=group, arm=arm, AP=macro[group][arm],
                AP_vs_NN=paired[group][f'{arm}-minus-NN'],
                per_condition_10pct=alarms['per_condition'][f'{group}|{arm}|0.10'],
                frozen_ideal_10pct=alarms['frozen_ideal'][f'{group}|{arm}|0.10']))
    return dict(scope='Privileged segmentation controls; frozen models; consumed Development; no audit tuning',
        dataset=dataset, unit_ids=splits, conditions=list(CONDITIONS), arms=list(ARMS),
        AP_definition='Saved main mask, HEAD/BODY AP per eligible unit then equal-unit macro; paired unit bootstrap10000 seed20260928; descriptive intervals without multiplicity correction',
        alarm_definition='Original L.threshold_for_budget / pairs / summary, rule1/1, frames>=3. Primary each condition calibrated separately on own dataset calib; secondary same-family IDEAL calib thresholds retained for all stress conditions. Secondary stress calib FA need not meet nominal budget.',
        control_definition='ZERO is inference-time ToF removal from frozen fusion model, not retraining; MASK is separately trained mask-only control. Compare each against fusion in same condition.',
        per_unit_AP=ap, audit_macro_AP=macro, paired=paired, alerts=alarms, compact_10pct=compact)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--dataset', choices=('v2', 'v4'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    units, splits = load(args.predictions, args.dataset)
    report = evaluate(units, splits, args.dataset)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['audit_macro_AP'], indent=2))


if __name__ == '__main__':
    main()
