"""Post-plan descriptive add-on: sidewall-held-out models on the novel-structure units.

Not in the frozen plan. Frozen LOFO A2 (sidewall fold) and CVR v2 (sidewall fold)
are calibrated on calib units' boundary/mixed/general negatives only and scored on
every evaluation family. No training or selection.
"""
import json
from pathlib import Path

import numpy as np

import cnh_novel_structure as B

OUT = B.OUT/'nosidewall'
SEEN = ('boundary', 'mixed_surface', 'general')


def infer():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import cnh_corridor_retrain as T
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    T.frozen_module()
    from cnh_learned_memory_fusion import causal_ewma
    OUT.mkdir(parents=True, exist_ok=True)
    a2files = [B.WORK/'cnh-corridor-lofo-20260929/sidewall'/f'model_seed{s}.pt' for s in range(3)]
    cvrfiles = [B.WORK/'cnh-cvr-v2-20260929/models/CVR/sidewall'/f'model_seed{s}.pt' for s in range(3)]
    B.save(OUT/'model_hashes.json', {str(p.relative_to(B.ROOT)): B.sha(p) for p in a2files+cvrfiles})
    models = T.load_models(a2files, 'cuda'); out = {}
    for split, units in B.SPLITS.items():
        for u in units:
            d = B.read(B.OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            nn = T.predict(models, d['z4'], d['z1'], d['support'], 48)
            out[str(u)] = causal_ewma(nn, d['scene'], d['frame'], alpha=.5, window=5).reshape(k, 16, 6)[:, -1, [2, 3]]
    np.savez_compressed(OUT/'A2.npz', **out); del models; torch.cuda.empty_cache()
    nets = []
    for f in cvrfiles:
        net = CVR().cuda(); net.load_state_dict(torch.load(f, map_location='cuda', weights_only=True)); nets.append(net.eval())
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda'); out = {}
    for split in B.SPLITS:
        x = np.load(B.OUT/'data'/split/'features.npy', mmap_mode='r'); m = B.read(B.OUT/'data'/split/'metadata.npz'); raw = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                b = torch.as_tensor(np.array(x[i:i+64], dtype=np.float32), device='cuda')
                b[:, 0] = b[:, 0].sign()*b[:, 0].abs().log1p(); b[:, 2] = b[:, 2].sign()*b[:, 2].abs().log1p(); b[:, 1] /= 8
                b = torch.cat((b, masks[None].expand(len(b), -1, -1, -1, -1)), 1)
                raw.append(torch.stack([n(b) for n in nets]).mean(0).cpu().numpy())
        raw = np.concatenate(raw)
        for u in B.SPLITS[split]:
            sel = np.flatnonzero(m['unit'] == u); scores = []
            for config in range(int(m['config'][sel].max())+1):
                loc = sel[m['config'][sel] == config]; loc = loc[np.argsort(m['frame'][loc])]
                assert np.array_equal(m['frame'][loc], B.FRAMES)
                scores.append((raw[loc].astype(np.float64)*np.array([1, 2, 4, 8, 16])[:, None]).sum(0)/31)
            out[str(u)] = np.asarray(scores)
    np.savez_compressed(OUT/'CVR.npz', **out)


def evaluate():
    from cnh_corridor_late_fusion import metrics, ranks, threshold
    from cnh_corridor_statistics import paired_cluster_ber
    from sklearn.metrics import roc_auc_score
    nos = {a: B.read(OUT/f'{a}.npz') for a in ('A2', 'CVR')}
    allf = {a: B.read(B.OUT/'predictions'/f'{a}.npz') for a in ('A2', 'CVR')}
    arms = ('A2_NOSW', 'CVR_NOSW', 'A2_ALL', 'CVR_ALL')

    def rows(split, keep):
        out = []
        for u in B.SPLITS[split]:
            d = B.read(B.OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
            for c in range(k):
                fam = str(d['family'][c])
                if not keep(fam): continue
                for q, g in enumerate(B.GROUPS):
                    out.append(dict(unit=u, family=fam, group=g, label=int(labels[c, q]),
                        scores=dict(A2_NOSW=float(nos['A2'][str(u)][c, q]), CVR_NOSW=float(nos['CVR'][str(u)][c, q]),
                                    A2_ALL=float(allf['A2'][str(u)][c, q]), CVR_ALL=float(allf['CVR'][str(u)][c, q]))))
        return out

    cal_all = rows('calib', lambda f: f in SEEN); ev_all = rows('evaluation', lambda f: True)
    table = []
    for g in B.GROUPS:
        cal = [r for r in cal_all if r['group'] == g and r['label'] == 0]; n = len(cal)
        ev = [r for r in ev_all if r['group'] == g]; p = {}; er = {}
        for arm in arms:
            ref = [r['scores'][arm] for r in cal]; cr = ranks(ref, ref); er[arm] = ranks(ref, [r['scores'][arm] for r in ev])
            k = threshold(cr, n); p[arm] = er[arm] >= k
        for fam in B.KNOWN + B.NOVEL:
            idx = [i for i, r in enumerate(ev) if r['family'] == fam]
            y = np.array([ev[i]['label'] for i in idx]); units = np.array([ev[i]['unit'] for i in idx])
            row = dict(family=fam, group=g, calib_negative=n)
            for arm in arms:
                row[arm] = dict(metrics(y, p[arm][idx]), auc=float(roc_auc_score(y, er[arm][idx].astype(float))))
            for base in ('A2', 'CVR'):
                row[f'{base}_NOSW_minus_ALL'] = paired_cluster_ber(y, p[f'{base}_ALL'][idx], p[f'{base}_NOSW'][idx], units)
            table.append(row)
    fam = []
    for f in B.KNOWN + B.NOVEL:
        rr = [t for t in table if t['family'] == f]
        fam.append(dict(family=f, **{f'{a}_{m}': float(np.mean([t[a][m] for t in rr])) for a in arms for m in ('ber', 'auc')}))
    B.save(OUT/'results.json', dict(status='POSTHOC_DESCRIPTIVE_NOT_IN_PLAN', calibration='same-3 (no sidewall) calib negatives for all arms',
                                    script_sha256=B.sha(__file__), family=fam, table=table))
    for f in fam:
        print(f"{f['family']:13s} " + ' '.join(f"{a}:{f[a+'_ber']*100:5.1f}/{f[a+'_auc']:.3f}" for a in arms))


if __name__ == '__main__':
    import sys
    {'infer': infer, 'evaluate': evaluate}[sys.argv[1]]()
