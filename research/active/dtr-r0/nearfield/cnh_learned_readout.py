"""Learned CNH readout vs S2 (Development, fast lane; repaired-v2 dev data, not a formal result).

Input per frame: z4 (S2 four-frame transported per-cell z) and z1 (current-frame z), 8x8x16,
squashed by sign*log1p; six tau=0.75 query supports. A small 3D CNN produces a feature
volume; each query pools it (masked max and mean) and a shared MLP with a query embedding
outputs a logit. Train split trains, calib selects the epoch and alert thresholds, audit
is only scored. Metrics mirror v4: per-unit macro AP per group, and per calib false-alert
budget the empty-pair false-alert rate, event recall and near-event timely/late/never.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from cnh_track_a_v4_analysis import pairs, summary

GROUPS = (('HEAD', (0, 2, 4)), ('BODY', (1, 3, 5)))
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def squash(z):
    return np.sign(z)*np.log1p(np.abs(z))


def load(root, scores=None):
    units = {}
    for f in sorted(Path(root).glob('unit*.npz')):
        d = np.load(f)
        u = int(f.stem[4:])
        x = np.stack([squash(d['z4'].astype(np.float32)), squash(d['z1'].astype(np.float32))], 1)
        sup = np.unpackbits(d['sup'], axis=-1)[..., :16].astype(bool)
        item = dict(x=x, sup=sup, y=d['labels'].astype(np.float32), main=d['main'], w=d['witness'],
                    strata=d['strata'], config=d['config'], frame=d['frame'], split=str(d['split']))
        cand = [Path(scores)/f'unit{u:02d}.npz', Path(scores)/f'unit{u:03d}.npz'] if scores is not None else []
        found = next((c for c in cand if c.exists()), None)
        if found is not None:
            s = np.load(found)
            assert np.array_equal(s['labels'], d['labels'])
            item['S2'] = s['G0']
        units[u] = item
    return units


class Readout(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(2, 16, 3, padding=1), nn.GELU(),
                                  nn.Conv3d(16, width, 3, padding=1), nn.GELU(),
                                  nn.Conv3d(width, width, 3, padding=1), nn.GELU())
        self.emb = nn.Embedding(6, 8)
        self.head = nn.Sequential(nn.Linear(2*width+8, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x, sup):
        f = self.body(x)                                            # [B,C,8,8,16]
        m = sup[:, :, None]                                         # [B,6,1,8,8,16]
        fq = f[:, None]
        mx = torch.where(m, fq, torch.full_like(fq, -1e4)).flatten(3).max(-1).values
        mean = (fq*m).flatten(3).sum(-1)/m.flatten(3).sum(-1).clamp_min(1)
        e = self.emb.weight[None].expand(len(x), -1, -1)
        return self.head(torch.cat([mx, mean, e], -1)).squeeze(-1)  # [B,6]


def batches(units, keys, main_only, bs, shuffle, rng=None):
    x = np.concatenate([units[u]['x'][units[u]['main']] if main_only else units[u]['x'] for u in keys])
    s = np.concatenate([units[u]['sup'][units[u]['main']] if main_only else units[u]['sup'] for u in keys])
    y = np.concatenate([units[u]['y'][units[u]['main']] if main_only else units[u]['y'] for u in keys])
    idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        yield (torch.as_tensor(x[j], device=DEV), torch.as_tensor(s[j], device=DEV), torch.as_tensor(y[j], device=DEV))


@torch.no_grad()
def predict(model, units, keys):
    model.eval()
    for u in keys:
        out = []
        for xb, sb, _ in batches({u: units[u]}, [u], False, 512, False):
            out.append(model(xb, sb).float().cpu().numpy())
        units[u]['NN'] = np.concatenate(out)


def macro_ap(units, keys, arm):
    res = {}
    for g, ix in GROUPS:
        aps = []
        for u in keys:
            m = units[u]['main']
            y, s = units[u]['y'][m][:, ix].ravel(), units[u][arm][m][:, ix].ravel()
            if 0 < y.sum() < len(y):
                aps.append(average_precision_score(y, s))
        res[g] = float(np.mean(aps))
    return res


def sequences(units, keys, arms):
    seqs = []
    for u in keys:
        d = units[u]
        for c in np.unique(d['config']):
            m = np.flatnonzero(d['config'] == c)
            m = m[np.argsort(d['frame'][m])]
            seqs.append(dict(unit=u, frame=d['frame'][m], y=d['y'][m].astype(int), w=d['w'][m], strata=d['strata'][m],
                             s={a: d[a][m] for a in arms}))
    return seqs


def threshold_for_budget(seqs, arm, boxes, budget):
    """Lowest threshold whose empty-pair false-alert rate on these sequences is <= budget (rule 1/1)."""
    peaks = []
    for sq in seqs:
        t = sq['frame'] >= 3
        for q in boxes:
            if not (sq['y'][t, q] == 1).any():
                peaks.append(sq['s'][arm][t, q].max())
    peaks = np.sort(np.array(peaks))[::-1]
    k = int(np.floor(budget*len(peaks)))
    return float(np.nextafter(peaks[k], np.inf)) if k < len(peaks) else float(peaks[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    units = load(a.features, a.scores)
    split = {s: sorted(u for u, d in units.items() if d['split'] == s) for s in ('train', 'calib', 'audit')}
    print({k: len(v) for k, v in split.items()}, flush=True)
    model = Readout().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.BCEWithLogitsLoss()
    best, history = None, []
    for ep in range(a.epochs):
        model.train()
        tot = 0.
        for xb, sb, yb in batches(units, split['train'], True, 256, True, rng):
            loss = lossf(model(xb, sb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())*len(xb)
        sched.step()
        predict(model, units, split['calib'])
        ap = macro_ap(units, split['calib'], 'NN')
        history.append(dict(epoch=ep, calib_macro_AP=ap))
        print(ep, round(tot, 1), ap, flush=True)
        score = ap['HEAD']+ap['BODY']
        if best is None or score > best[0]:
            best = (score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    predict(model, units, split['calib']+split['audit'])
    report = dict(scope='repaired-v2 Development; learned readout prototype; not a formal result',
                  units={k: len(v) for k, v in split.items()}, best_epoch=best[1], history=history,
                  params=sum(p.numel() for p in model.parameters()))
    report['audit_macro_AP'] = {arm: macro_ap(units, split['audit'], arm) for arm in ('S2', 'NN')}
    report['calib_macro_AP'] = {arm: macro_ap(units, split['calib'], arm) for arm in ('S2', 'NN')}
    cal, aud = sequences(units, split['calib'], ('S2', 'NN')), sequences(units, split['audit'], ('S2', 'NN'))
    report['alerts'] = {}
    for g, boxes in GROUPS:
        for arm in ('S2', 'NN'):
            for b in (.05, .10, .20):
                thr = threshold_for_budget(cal, arm, boxes, b)
                rows = pairs(aud, arm, (1, 1), boxes, thr)
                report['alerts'][f'{g}|{arm}|{b:.2f}'] = dict(threshold=thr, all=summary(rows),
                                                             tiny=summary(rows, 'tiny'))
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(best[2], a.out/f'model_seed{a.seed}.pt')
    np.savez_compressed(a.out/f'predictions_seed{a.seed}.npz',
                        **{f'u{u}_{k}': units[u][k] for u in split['calib']+split['audit'] for k in ('NN', 'S2')})
    (a.out/f'report_seed{a.seed}.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('best_epoch', 'params', 'audit_macro_AP', 'calib_macro_AP')}, indent=1))
    for k, v in report['alerts'].items():
        x, t = v['all'], v['tiny']
        print(f"{k:16s} FA {x['false_alert_rate']:.3f} recall {x['event_recall']:.3f} near t/l/n {x['timely']:.2f}/{x['late']:.2f}/{x['never']:.2f}"
              f"  tiny t/l/n {t['timely']:.2f}/{t['late']:.2f}/{t['never']:.2f} (near {t['near']})")


if __name__ == '__main__':
    main()
