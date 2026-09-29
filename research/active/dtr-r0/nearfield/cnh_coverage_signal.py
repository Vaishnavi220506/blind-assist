"""Label-free coverage-signal feasibility on the consumed novel-structure units.

Plan: artifacts.local/work/cnh-coverage-signal-20260929/COVERAGE_SIGNAL_PLAN.md
Per-seed frozen predictions only; no training, no fusion test.
"""
import json
import sys
from pathlib import Path

import numpy as np

import cnh_novel_structure as B

OUT = B.WORK/'cnh-coverage-signal-20260929'
PLAN = OUT/'COVERAGE_SIGNAL_PLAN.md'
SEEN = ('boundary', 'mixed_surface', 'general')
SETS = {
    'A2_NOSW': ('A2', B.WORK/'cnh-corridor-lofo-20260929/sidewall'),
    'CVR_NOSW': ('CVR', B.WORK/'cnh-cvr-v2-20260929/models/CVR/sidewall'),
    'A2_ALL': ('A2', B.WORK/'cnh-corridor-retrain-20260929/models'),
    'CVR_ALL': ('CVR', B.WORK/'cnh-cvr-v2-20260929/models/CVR/allfamily'),
}


def infer():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import cnh_corridor_retrain as T
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    T.frozen_module()
    from cnh_learned_memory_fusion import causal_ewma
    assert PLAN.exists()
    hashes = {}
    for name, (kind, folder) in SETS.items():
        dest = OUT/f'{name}_seeds.npz'
        if dest.exists(): continue
        files = [folder/f'model_seed{s}.pt' for s in range(3)]
        hashes.update({str(p.relative_to(B.ROOT)): B.sha(p) for p in files})
        out = {}
        if kind == 'A2':
            models = T.load_models(files, 'cuda')
            for split, units in B.SPLITS.items():
                for u in units:
                    d = B.read(B.OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
                    per = [causal_ewma(T.predict([m], d['z4'], d['z1'], d['support'], 48), d['scene'], d['frame'],
                                       alpha=.5, window=5).reshape(k, 16, 6)[:, -1, [2, 3]] for m in models]
                    out[str(u)] = np.stack(per, 1)  # [k, seed, group]
            del models
        else:
            nets = []
            for f in files:
                net = CVR().cuda(); net.load_state_dict(torch.load(f, map_location='cuda', weights_only=True)); nets.append(net.eval())
            masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
            w = np.array([1, 2, 4, 8, 16])[None, :, None]/31
            for split in B.SPLITS:
                x = np.load(B.OUT/'data'/split/'features.npy', mmap_mode='r'); m = B.read(B.OUT/'data'/split/'metadata.npz'); raw = []
                with torch.no_grad():
                    for i in range(0, len(x), 64):
                        b = torch.as_tensor(np.array(x[i:i+64], dtype=np.float32), device='cuda')
                        b[:, 0] = b[:, 0].sign()*b[:, 0].abs().log1p(); b[:, 2] = b[:, 2].sign()*b[:, 2].abs().log1p(); b[:, 1] /= 8
                        b = torch.cat((b, masks[None].expand(len(b), -1, -1, -1, -1)), 1)
                        raw.append(torch.stack([n(b) for n in nets], 1).cpu().numpy())  # [batch, seed, group]
                raw = np.concatenate(raw)
                for u in B.SPLITS[split]:
                    sel = np.flatnonzero(m['unit'] == u); per = []
                    for config in range(int(m['config'][sel].max())+1):
                        loc = sel[m['config'][sel] == config]; loc = loc[np.argsort(m['frame'][loc])]
                        assert np.array_equal(m['frame'][loc], B.FRAMES)
                        per.append((raw[loc].astype(np.float64).transpose(1, 0, 2)*w).sum(1))  # [seed, group]
                    out[str(u)] = np.asarray(per)
            del nets
        np.savez_compressed(dest, **out); torch.cuda.empty_cache(); print('infer', name, flush=True)
    B.save(OUT/'model_hashes.json', hashes) if hashes else None
    # Guard: seed-mean must reproduce the ensemble predictions already on disk.
    check = {}
    for name, ref in [('A2_ALL', B.OUT/'predictions/A2.npz'), ('CVR_ALL', B.OUT/'predictions/CVR.npz'),
                      ('A2_NOSW', B.OUT/'nosidewall/A2.npz'), ('CVR_NOSW', B.OUT/'nosidewall/CVR.npz')]:
        s = B.read(OUT/f'{name}_seeds.npz'); r = B.read(ref)
        check[name] = float(max(abs(s[u].mean(1)-r[u]).max() for u in r))
    B.save(OUT/'seed_mean_check.json', check); print(check)


def evaluate():
    from sklearn.metrics import roc_auc_score
    from cnh_corridor_late_fusion import ranks
    seeds = {n: B.read(OUT/f'{n}_seeds.npz') for n in SETS}
    rows = {s: [] for s in B.SPLITS}
    for split, units in B.SPLITS.items():
        for u in units:
            d = B.read(B.OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
            for c in range(k):
                for q in range(2):
                    rows[split].append(dict(unit=u, family=str(d['family'][c]), group=q, label=int(labels[c, q]),
                        partial=float(d['motion'][c, 2, q]),
                        **{n: seeds[n][str(u)][c, :, q] for n in SETS}))
    result = dict(status='DIAGNOSTIC', plan='COVERAGE_SIGNAL_PLAN.md', models={}, reading={})
    fams = B.KNOWN + B.NOVEL
    for n in SETS:
        cal = [r for r in rows['calib'] if r['family'] in SEEN]; ev = rows['evaluation']
        sig = {}
        for part, rr in (('calib', cal), ('evaluation', ev)):
            sig[part] = {'seed_std': np.array([r[n].std() for r in rr])}
        # Learned-geometry rank disagreement against seen-family calib negatives, per group.
        for part, rr in (('calib', cal), ('evaluation', ev)):
            v = np.zeros(len(rr))
            for q in range(2):
                ref = [r for r in cal if r['group'] == q and r['label'] == 0]; nref = len(ref)
                idx = [i for i, r in enumerate(rr) if r['group'] == q]
                lr = ranks([r[n].mean() for r in ref], [rr[i][n].mean() for i in idx])/nref
                gr = ranks([r['partial'] for r in ref], [rr[i]['partial'] for i in idx])/nref
                v[idx] = abs(lr-gr)
            sig[part]['learn_geo'] = v
        fam = np.array([r['family'] for r in ev]); lab = np.array([r['label'] for r in ev])
        entry = {}
        for s in ('seed_std', 'learn_geo'):
            x = sig['evaluation'][s]; target = fam == 'sidewall'; base = np.isin(fam, SEEN)
            keep = target | base
            aucs = dict(all=float(roc_auc_score(target[keep], x[keep])))
            for name, lm in (('positives_only', lab == 1), ('negatives_only', lab == 0)):
                kk = keep & lm; aucs[name] = float(roc_auc_score(target[kk], x[kk]))
            th = float(np.quantile(sig['calib'][s], .9))
            flag = {f: float((x[fam == f] > th).mean()) for f in fams}
            entry[s] = dict(auc_sidewall_vs_seen=aucs, calib_p90=th, flag_rate=flag,
                            verdict=None if 'ALL' in n else ('usable' if aucs['all'] >= .75 else 'weak' if aucs['all'] >= .60 else 'not_usable'))
        result['models'][n] = entry
    result['reading'] = {f'{n}|{s}': result['models'][n][s]['verdict'] for n in ('A2_NOSW', 'CVR_NOSW') for s in ('seed_std', 'learn_geo')}
    B.save(OUT/'results.json', result)
    for n, e in result['models'].items():
        for s, v in e.items():
            a = v['auc_sidewall_vs_seen']
            print(f"{n:9s} {s:9s} AUC all {a['all']:.3f} pos {a['positives_only']:.3f} neg {a['negatives_only']:.3f} verdict={v['verdict']} | "
                  + ' '.join(f"{f[:6]}:{v['flag_rate'][f]*100:.0f}" for f in fams))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    {'infer': infer, 'evaluate': evaluate}[sys.argv[1]]()
