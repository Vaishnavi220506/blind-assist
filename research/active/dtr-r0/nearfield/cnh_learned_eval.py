"""Evaluate frozen learned-readout models on another dataset (no training, no model selection).

Loads features from cnh_learned_features, S2 scores from a readout directory (key given),
averages the logits of the saved seed models, picks alert thresholds on that dataset's
calib units and scores its audit units: per-unit macro AP with a paired unit bootstrap,
and v4-style alert metrics at calib false-alert budgets.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
import cnh_learned_readout as L


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--readouts', type=Path, required=True)
    p.add_argument('--s2-key', default='S2__noisy@0.75')
    p.add_argument('--models', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    units = L.load(a.features)
    for u, d in units.items():
        f = next(c for c in (a.readouts/f'unit{u:02d}.npz', a.readouts/f'unit{u:03d}.npz') if c.exists())
        s = np.load(f)
        assert np.array_equal(s['labels'].astype(np.float32), d['y'])
        d['S2'] = s[a.s2_key]
    split = {s: sorted(u for u, d in units.items() if d['split'] == s) for s in ('calib', 'audit')}
    models = []
    for m in a.models:
        net = L.Readout().to(L.DEV)
        net.load_state_dict(torch.load(m, map_location=L.DEV))
        models.append(net)
    keys = split['calib']+split['audit']
    for u in keys:
        preds = []
        for net in models:
            L.predict(net, units, [u])
            preds.append(units[u]['NN'])
        units[u]['NN'] = np.mean(preds, 0)
    report = dict(scope='frozen models; thresholds from this dataset calib; audit only scored',
                  units={k: len(v) for k, v in split.items()}, models=[str(m) for m in a.models],
                  audit_macro_AP={arm: L.macro_ap(units, split['audit'], arm) for arm in ('S2', 'NN')}, paired={})
    rng = np.random.default_rng(20260928)
    for g, ix in L.GROUPS:
        diff = []
        for u in split['audit']:
            m = units[u]['main']
            y = units[u]['y'][m][:, ix].ravel()
            if 0 < y.sum() < len(y):
                diff.append(average_precision_score(y, units[u]['NN'][m][:, ix].ravel())
                            - average_precision_score(y, units[u]['S2'][m][:, ix].ravel()))
        diff = np.array(diff)
        boot = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1)
        report['paired'][g] = dict(delta=float(diff.mean()), ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                                   units_positive=int((diff > 0).sum()), units=len(diff))
    cal, aud = L.sequences(units, split['calib'], ('S2', 'NN')), L.sequences(units, split['audit'], ('S2', 'NN'))
    report['alerts'] = {}
    for g, boxes in L.GROUPS:
        for arm in ('S2', 'NN'):
            for b in (.05, .10, .20):
                thr = L.threshold_for_budget(cal, arm, boxes, b)
                rows = L.pairs(aud, arm, (1, 1), boxes, thr)
                report['alerts'][f'{g}|{arm}|{b:.2f}'] = dict(threshold=thr, all=L.summary(rows), tiny=L.summary(rows, 'tiny'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('units', 'audit_macro_AP', 'paired')}, indent=1))
    for k, v in report['alerts'].items():
        x, t = v['all'], v['tiny']
        print(f"{k:16s} FA {x['false_alert_rate']:.3f} recall {x['event_recall']:.3f} near timely {x['timely']:.2f}"
              f"  tiny timely/never {t['timely']:.2f}/{t['never']:.2f} (near {t['near']})")


if __name__ == '__main__':
    main()
