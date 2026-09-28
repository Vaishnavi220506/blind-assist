"""Consumed Development: frozen NN + existing outside-FOV memory, no retraining.

Fixed fusion is max of per-group empirical negative-frame percentile scores.
Reference distributions use calib negatives only; alarm thresholds also use calib.
NN_RANK exposes ties introduced by finite empirical distributions. NN_S2 is the
redundant-input control. Audit is never used to fit a transform or pick an arm.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
import cnh_learned_readout as L

ARMS = ('S2', 'S3', 'NN', 'NN_RANK', 'NN_M', 'NN_S2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_score(values, reference):
    """Finite-sample upper-tail surprise; equal raw values always map equally."""
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(ref) or not np.isfinite(ref).all():
        raise ValueError('invalid calibration reference')
    ranks = np.searchsorted(ref, values, side='right')
    return -np.log((len(ref) + .5 - ranks) / (len(ref) + 1.))


def fit_fusion(units, calib):
    refs = {}
    for group, ix in L.GROUPS:
        refs[group] = {}
        for arm in ('NN', 'M', 'S2'):
            vals = []
            for u in calib:
                d = units[u]
                mask = d['main'][:, None] & (d['y'][:, ix] == 0)
                vals.append(d[arm][:, ix][mask])
            refs[group][arm] = np.concatenate(vals)
    return refs


def apply_fusion(units, refs):
    for d in units.values():
        for arm in ('NN_RANK', 'NN_M', 'NN_S2'):
            d[arm] = np.zeros_like(d['NN'], dtype=np.float64)
        for group, ix in L.GROUPS:
            n, m, s = [rank_score(d[a][:, ix], refs[group][a]) for a in ('NN', 'M', 'S2')]
            d['NN_RANK'][:, ix] = n
            d['NN_M'][:, ix] = np.maximum(n, m)
            d['NN_S2'][:, ix] = np.maximum(n, s)


def ap_rows(units, keys, arm):
    result = {}
    for group, ix in L.GROUPS:
        rows = []
        for u in keys:
            d = units[u]
            mask = np.broadcast_to(d['main'][:, None], (len(d['main']), len(ix))).copy()
            y, s = d['y'][:, ix][mask], d[arm][:, ix][mask]
            if 0 < y.sum() < len(y):
                rows.append(dict(unit=u, ap=float(average_precision_score(y, s)),
                                 positives=int(y.sum()), negatives=int((y == 0).sum())))
        result[group] = rows
    return result


def paired_ap(a, b):
    by_id = {r['unit']: r['ap'] for r in b}
    delta = np.array([r['ap'] - by_id[r['unit']] for r in a])
    rng = np.random.default_rng(20260928)
    boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
    return dict(delta=float(delta.mean()), ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                units=len(delta), units_positive=int((delta > 0).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--readouts', type=Path, required=True)
    p.add_argument('--models', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    started = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    pred_dir = a.out / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    model_hashes = {str(m): sha(m) for m in a.models}
    source_hashes = {str(Path(__file__)): sha(__file__), str(Path(L.__file__)): sha(L.__file__)}
    provenance = dict(models=model_hashes, source=source_hashes, features=str(a.features),
                      readouts=str(a.readouts), backend=str(L.DEV), torch=torch.__version__,
                      device=torch.cuda.get_device_name() if L.DEV.type == 'cuda' else 'CPU',
                      cpu_reason='TASK_NOT_GPU_SUITABLE for rank calibration and scalar metrics',
                      scope='Consumed v4 Development; fixed score fusion; no audit selection',
                      inputs={})
    (a.out / 'plan.json').write_text(json.dumps(dict(arms=ARMS, budgets=[.05, .1, .2],
        fusion='max of calib-negative-frame empirical upper-tail scores, separately HEAD/BODY',
        selection='none; no weights, horizons or fusion choices tuned', **provenance), indent=2))
    units = L.load(a.features)
    models = []
    for m in a.models:
        net = L.Readout().to(L.DEV)
        net.load_state_dict(torch.load(m, map_location=L.DEV, weights_only=True))
        models.append(net)
    keys = sorted(units)
    for i, u in enumerate(keys):
        d = units[u]
        f = next(f for f in (a.readouts / f'unit{u:02d}.npz', a.readouts / f'unit{u:03d}.npz') if f.exists())
        feature = a.features / f'unit{u:03d}.npz'
        signature = dict(feature=sha(feature), readout=sha(f), models=model_hashes,
                         inference_source=sha(L.__file__))
        provenance['inputs'][str(u)] = signature
        with np.load(f) as r:
            for rk, dk in (('labels', 'y'), ('config', 'config'), ('frame', 'frame'), ('main', 'main')):
                assert np.array_equal(r[rk], d[dk]), (u, rk)
            assert str(r['split']) == d['split']
            assert np.allclose(r['witness'], d['w'], equal_nan=True)
            d['S2'], d['M'] = r['S2__noisy@0.75'], r['memory__noisy']
            d['S3'] = np.maximum(d['S2'], d['M'])
        target = pred_dir / f'unit{u:03d}.npz'
        receipt = pred_dir / f'unit{u:03d}.json'
        if target.exists() and receipt.exists() and json.loads(receipt.read_text()) == signature:
            with np.load(target) as cached:
                d['NN'] = cached['NN']
        else:
            predictions = []
            for net in models:
                L.predict(net, units, [u])
                predictions.append(d['NN'])
            d['NN'] = np.mean(predictions, axis=0)
            np.savez_compressed(target, **{k: d[k] for k in ('NN', 'S2', 'M', 'S3', 'y', 'main', 'w', 'config', 'frame', 'strata')}, split=np.array(d['split']))
            receipt.write_text(json.dumps(signature, indent=2))
        assert np.isfinite(d['NN']).all()
        if (i + 1) % 8 == 0 or i + 1 == len(keys):
            progress = dict(stage='inference', completed=i+1, total=len(keys), seconds=time.time()-started)
            (a.out / 'progress.json').write_text(json.dumps(progress))
            print(progress, flush=True)
    split = {s: [u for u in keys if units[u]['split'] == s] for s in ('calib', 'audit')}
    assert len(split['calib']) == 32 and len(split['audit']) == 64
    refs = fit_fusion(units, split['calib'])
    np.savez_compressed(a.out / 'calib_references.npz', **{f'{g}_{arm}': v for g, arms in refs.items() for arm, v in arms.items()})
    apply_fusion(units, refs)
    aps = {arm: ap_rows(units, split['audit'], arm) for arm in ARMS}
    report = dict(provenance=provenance, units={k: len(v) for k, v in split.items()},
                  macro_ap={arm: {g: float(np.mean([r['ap'] for r in rows])) for g, rows in groups.items()} for arm, groups in aps.items()},
                  ap_per_unit=aps, paired_ap={}, alerts={})
    for arm, base in (('NN_M', 'NN'), ('NN_M', 'NN_RANK'), ('NN_M', 'NN_S2'), ('NN_M', 'S3')):
        report['paired_ap'][f'{arm}-{base}'] = {g: paired_ap(aps[arm][g], aps[base][g]) for g, _ in L.GROUPS}
    cal = L.sequences(units, split['calib'], ARMS)
    aud = L.sequences(units, split['audit'], ARMS)
    from cnh_track_a_v4_analysis import unit_counts, paired_test
    alert_rows = {}
    for g, ix in L.GROUPS:
        for arm in ARMS:
            for budget in (.05, .1, .2):
                thr = L.threshold_for_budget(cal, arm, ix, budget)
                cr, rows = L.pairs(cal, arm, (1, 1), ix, thr), L.pairs(aud, arm, (1, 1), ix, thr)
                key = f'{g}|{arm}|{budget:.2f}'
                alert_rows[key] = rows
                report['alerts'][key] = dict(threshold=thr, calib=L.summary(cr), all=L.summary(rows), tiny=L.summary(rows, 'tiny'))
    report['paired_event_diagnostic'] = {}
    for g, _ in L.GROUPS:
        for budget in (.05, .1, .2):
            for metric in ('timely', 'never'):
                rows_a = alert_rows[f'{g}|NN_M|{budget:.2f}']
                rows_b = alert_rows[f'{g}|NN|{budget:.2f}']
                result = paired_test(unit_counts(rows_a, metric, split['audit']), unit_counts(rows_b, metric, split['audit']), split['audit'])
                result.pop('p_one_sided')  # Descriptive CI only; no confirmatory testing.
                report['paired_event_diagnostic'][f'{g}|NN_M-NN|{budget:.2f}|{metric}'] = result
    report['seconds'] = time.time()-started
    assert source_hashes == {str(Path(__file__)): sha(__file__), str(Path(L.__file__)): sha(L.__file__)}, 'source changed during run'
    (a.out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (a.out / 'terminal.json').write_text(json.dumps(dict(status='complete', seconds=report['seconds'], units=len(keys))))
    print(json.dumps(report['macro_ap'], indent=2), flush=True)
    for g, _ in L.GROUPS:
        for arm in ARMS:
            r = report['alerts'][f'{g}|{arm}|0.10']
            print(g, arm, 'FA', r['all']['false_alert_rate'], 'tiny', r['tiny'], flush=True)


if __name__ == '__main__':
    main()
