"""Bounded learned score fusion, disjoint fit/calibration on consumed v4.

Units 0..15 fit fixed-C logistic models, 16..31 set thresholds, 32..95 only
evaluate. No model or hyperparameter selection. This learns memory weights,
not memory construction or forgetting. EWMA is a fixed heuristic, not SPRT.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import cnh_learned_readout as L
from cnh_learned_memory_diagnostic import ap_rows, paired_ap, sha

ARMS = ('NN', 'S3', 'NN_M_LINEAR', 'NN_S2_LINEAR', 'NN_EWMA', 'NN_EWMA5', 'NN_EWMA5_M_LINEAR')


def features(d, ix, memory, base='NN'):
    n = d[base][:, ix]
    if memory:
        m = d['M'][:, ix]
        return np.stack((n, np.log1p(np.maximum(m, 0)), (m > -49).astype(float)), axis=-1)
    return np.stack((n, L.squash(d['S2'][:, ix])), axis=-1)


def causal_ewma(scores, config, frame, alpha=.5, window=None):
    out = np.empty_like(scores)
    for c in np.unique(config):
        indices = np.flatnonzero(config == c)
        indices = indices[np.argsort(frame[indices])]
        for t, i in enumerate(indices):
            if window is None:
                out[i] = scores[i] if t == 0 else alpha*scores[i] + (1-alpha)*out[indices[t-1]]
            else:
                past = indices[max(0, t-window+1):t+1]
                weights = (1-alpha)**np.arange(len(past)-1, -1, -1)
                out[i] = (scores[past]*weights[:, None]).sum(0)/weights.sum()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    plan = dict(scope='consumed v4 Development; selected after fixed-max diagnostic; not fresh confirmation',
                fit=list(range(16)), calib=list(range(16, 32)), audit=list(range(32, 96)),
                arms=ARMS, model='StandardScaler + LogisticRegression C=1, no class weighting, no search',
                memory_features=['NN logit', 'log1p(max(memory_z,0))', 'memory_z > -49'],
                sequential='causal EWMA of NN logits, alpha .5, reset at each configuration; not SPRT. Additional finite5-score-window control: at most 8 raw frames including 4-frame NN inputs, same as S3 memory. Combined EWMA5+M tests incremental memory value.',
                backend='CPU', placement_reason='TASK_NOT_GPU_SUITABLE', source_hash=sha(__file__))
    (a.out / 'plan.json').write_text(json.dumps(plan, indent=2))
    units, hashes = {}, {}
    for f in sorted(a.predictions.glob('unit*.npz')):
        with np.load(f) as d:
            units[int(f.stem[4:])] = {k: d[k] for k in d.files}
        hashes[str(f)] = sha(f)
    assert sorted(units) == list(range(96))
    assert all(str(units[u]['split']) == ('calib' if u < 32 else 'audit') for u in units)
    for d in units.values():
        for arm in ('NN_M_LINEAR', 'NN_S2_LINEAR', 'NN_EWMA5_M_LINEAR'):
            d[arm] = np.zeros_like(d['NN'], dtype=np.float64)
        d['NN_EWMA'] = causal_ewma(d['NN'], d['config'], d['frame'])
        d['NN_EWMA5'] = causal_ewma(d['NN'], d['config'], d['frame'], window=5)
    fitted = {}
    for g, ix in L.GROUPS:
        for arm, memory, base in (('NN_M_LINEAR', True, 'NN'), ('NN_S2_LINEAR', False, 'NN'), ('NN_EWMA5_M_LINEAR', True, 'NN_EWMA5')):
            x = np.concatenate([features(units[u], ix, memory, base)[units[u]['main']].reshape(-1, 3 if memory else 2) for u in plan['fit']])
            y = np.concatenate([units[u]['y'][units[u]['main']][:, ix].ravel() for u in plan['fit']])
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=20260928))
            model.fit(x, y)
            scale, lr = model.steps[0][1], model.steps[1][1]
            assert lr.n_iter_[0] < 2000
            fitted[f'{g}|{arm}'] = dict(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                coefficients=lr.coef_.tolist(), intercept=lr.intercept_.tolist(), iterations=lr.n_iter_.tolist(),
                fit_frames_queries=len(y), fit_positive=int(y.sum()))
            for d in units.values():
                x = features(d, ix, memory, base)
                d[arm][:, ix] = model.decision_function(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])
    aps = {arm: ap_rows(units, plan['audit'], arm) for arm in ARMS}
    report = dict(plan=plan, inputs=hashes, fit_models=fitted,
        macro_ap={arm: {g: float(np.mean([r['ap'] for r in rr])) for g, rr in groups.items()} for arm, groups in aps.items()},
        paired_ap={arm: {g: paired_ap(aps[arm][g], aps['NN'][g]) for g, _ in L.GROUPS} for arm in ARMS if arm != 'NN'},
        alerts={})
    report['memory_increment_over_equal_history'] = {g: paired_ap(aps['NN_EWMA5_M_LINEAR'][g], aps['NN_EWMA5'][g]) for g, _ in L.GROUPS}
    cal, aud = L.sequences(units, plan['calib'], ARMS), L.sequences(units, plan['audit'], ARMS)
    for g, ix in L.GROUPS:
        for arm in ARMS:
            for budget in (.05, .1, .2):
                thr = L.threshold_for_budget(cal, arm, ix, budget)
                crows, rows = L.pairs(cal, arm, (1, 1), ix, thr), L.pairs(aud, arm, (1, 1), ix, thr)
                report['alerts'][f'{g}|{arm}|{budget:.2f}'] = dict(threshold=thr, calib=L.summary(crows), all=L.summary(rows), tiny=L.summary(rows, 'tiny'))
    report['seconds'] = time.time()-started
    assert sha(__file__) == plan['source_hash']
    (a.out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (a.out/'terminal.json').write_text(json.dumps(dict(status='complete', seconds=report['seconds'])))
    print(json.dumps(report['macro_ap'], indent=2))
    for g, _ in L.GROUPS:
        for arm in ARMS:
            r = report['alerts'][f'{g}|{arm}|0.10']
            print(g, arm, 'FA', r['all']['false_alert_rate'], 'tiny', r['tiny'], flush=True)


if __name__ == '__main__':
    main()
