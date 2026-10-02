"""Margin-label training under the user-adopted three-level truth (2026-10-02).

Three-level truth: contact (intrusion > 0, must alarm), near-pass (0-10 cm
outside the body line, alarm acceptable, not counted as false alarm), clear
(counted as false alarm). Arms reuse the near-range train voxels (93000-93095)
and NEAR recipe; only the corridor labels change: half-width 0.30 (NEAR,
existing), 0.33 (M3), 0.38 (M8). Training targets sit at +-1.5/4.5/12 cm, so
M3 flips the 1.5 cm-outside targets and M8 also the 4.5 cm ones; any other
widening in (2.1, 3.9) or (5.1, 11.4) cm gives identical target labels.
Evaluation: near-range units 94000-94095 (clear = other-height-band rows) and,
for laterally clear same-height objects, readout-fix units 87000-87143 with
thresholds from its calib units 88000-88047.
"""
import argparse
import json
import time

import numpy as np

import cnh_structure_space as SS
import cnh_near_range as NR
import cnh_readout_fix as RF

OUT = SS.WORK/'cnh-margin-labels-20261002'
HALF = {'NEAR': .30, 'M3': .33, 'M8': .38}
FAS = (.02, .05, .10, .20, .30)
RANGES = ((.6, 1.2), (1.2, 1.6), (1.6, 2.1), (2.1, 2.6))
BINS = {'contact 0-2': (0., .02), 'contact 2-5': (.02, .05), 'contact >5': (.05, 1.),
        'pass 0-5': (-.05, 0.), 'pass 5-10': (-.10, -.05)}
RULE = ('Fixed before training. Primary: contact 0-2 cm alarm rate at 10% clear-row alarm rate (per-group '
        'thresholds on 94000-94095 other-height rows), pooled 0.6-2.1 m and both conditions; arm minus NEAR, '
        '1000 unit-bootstrap. An arm HELPS if CI lower > 0 AND contact >5 cm change >= -2 pp AND, on readout-fix '
        'evaluation (thresholds at 10% of its calib clear rows), the alarm rate on same-height targets 12 cm '
        'outside rises by <= 5 pp. Otherwise THRESHOLD_SUFFICES. Both arms reported; no arm chosen from results. '
        'Near-pass alarms are acceptable under three-level truth and reported only.')


def labels_for(boxes, poses, half):
    import cnh_proposal_attribution_scenes as S
    lo, hi = S.QUERY_LOW.copy(), S.QUERY_HIGH.copy(); lo[:, 0] = -half; hi[:, 0] = half
    tri = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in boxes]); out = []
    for pose in poses:
        local = (tri-pose[:3, 3])@pose[:3, :3]
        out.append([int(len(S.clip_triangles(local, a, b)) > 0) for a, b in zip(lo, hi)])
    return np.asarray(out, np.int8)


def relabel():
    _, m = NR.load_split('train'); key = {}
    for u in NR.SPLITS['train']:
        for c, sc in enumerate(NR.scenes_for(u)):
            poses = sc['travel'][SS.TRAIN_FRAMES]
            key[(u, c)] = {a: labels_for(sc['boxes'], poses, h) for a, h in HALF.items()}
        print('relabel', u, flush=True)
    frame_pos = {int(f): i for i, f in enumerate(SS.TRAIN_FRAMES)}
    Y = {a: np.stack([key[(int(u), int(c))][a][frame_pos[int(f)]] for u, c, f in zip(m['unit'], m['config'], m['frame'])])
         for a in HALF}
    assert np.array_equal(Y['NEAR'], m['labels']), 'half-width 0.30 must reproduce stored training labels'
    np.savez_compressed(OUT/'train_labels.npz', **Y)
    SS.save(OUT/'label_counts.json', {a: dict(pos=int(y.sum()), flipped_vs_NEAR=int((y != Y['NEAR']).sum()), rows=int(y.size)) for a, y in Y.items()})


def train(arm):
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    xs, _ = NR.load_split('train'); n = sum(len(x) for x in xs)
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    X = torch.empty((n, 3, *xs[0].shape[2:]), dtype=torch.float16, device='cuda'); o = 0
    for x in xs:
        for c in range(0, len(x), 1024): X[o+c:o+c+len(x[c:c+1024])] = torch.as_tensor(np.array(x[c:c+1024]), device='cuda')
        o += len(x)
    Y = torch.as_tensor(SS.read(OUT/'train_labels.npz')[arm].astype(np.float32), device='cuda'); ids = np.arange(n)

    def prep_gpu(xb):
        x = xb.float(); x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p(); x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p(); x[:, 1] /= 8
        return torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)
    for seed in range(5):  # identical to cnh_near_range.train except the label tensor
        final = OUT/'models'/arm/f'model_seed{seed}.pt'
        if final.exists(): continue
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
        net = CVR().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); history = []
        for epoch in range(20):
            t0 = time.monotonic(); net.train(); order = rng.permutation(ids); total = 0.
            for c in range(0, n, 256):
                pos = torch.as_tensor(np.sort(order[c:c+256]), device='cuda'); opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(prep_gpu(X[pos])), Y[pos])
                loss.backward(); opt.step(); total += float(loss.detach())*len(pos)
            sched.step(); history.append(dict(epoch=epoch+1, loss=total/n, elapsed_s=time.monotonic()-t0))
            print('train', arm, seed, history[-1], flush=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, final); SS.save(final.parent/f'seed{seed}_history.json', history)
        del net, opt; torch.cuda.empty_cache()


def model_paths(arm):
    return [(NR.OUT if arm == 'NEAR' else OUT)/'models'/arm/f'model_seed{s}.pt' for s in range(5)]


def infer():
    """Smoothed 5-seed ensemble scores on near-range evaluation and readout-fix calib/evaluation."""
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    xs, m = NR.load_split('evaluation')
    sets = {'near': (xs, m)}
    for split in ('calib', 'evaluation'):
        sets[f'rf_{split}'] = ([np.load(RF.OUT/'data'/split/'features.npy', mmap_mode='r')], SS.read(RF.OUT/'data'/split/'metadata.npz'))
    for arm in HALF:
        path = OUT/f'scores_{arm}.npz'
        if path.exists(): continue
        nets = []
        for p in model_paths(arm):
            net = CVR().cuda(); net.load_state_dict(torch.load(p, map_location='cuda', weights_only=True)); nets.append(net.eval())
        out = {}
        for name, (xx, mm) in sets.items():
            raw = []
            with torch.no_grad():
                for x in xx:
                    for i in range(0, len(x), 128): raw.append(torch.stack([n_(SS.prep(torch, x[i:i+128], masks)) for n_ in nets]).mean(0).cpu().numpy())
            raw = np.concatenate(raw)
            for u in np.unique(mm['unit']):
                sel = np.flatnonzero(mm['unit'] == u); sc = []
                for c in range(int(mm['config'][sel].max())+1):
                    loc = sel[mm['config'][sel] == c]; loc = loc[np.argsort(mm['frame'][loc])]
                    assert np.array_equal(mm['frame'][loc], SS.EVAL_FRAMES)
                    sc.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None]).sum(0)/SS.WEIGHTS.sum())
                out[f'{name}|{int(u)}'] = np.asarray(sc)
            print('infer', arm, name, flush=True)
        np.savez_compressed(path, **out); del nets; torch.cuda.empty_cache()
    old = SS.read(NR.OUT/'scores_NEAR.npz'); new = SS.read(OUT/'scores_NEAR.npz')
    assert max(float(np.abs(old[str(u)]-new[f'near|{u}']).max()) for u in NR.SPLITS['evaluation']) < 1e-4


def infer92():
    """Same-format scores on detectability-envelope units 92000-92095 (for the RGB three-level gate)."""
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    import cnh_detectability_envelope as E
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    x = np.load(E.OUT/'data/evaluation/features.npy', mmap_mode='r'); m = SS.read(E.OUT/'data/evaluation/metadata.npz')
    for arm in HALF:
        nets = []
        for p in model_paths(arm):
            net = CVR().cuda(); net.load_state_dict(torch.load(p, map_location='cuda', weights_only=True)); nets.append(net.eval())
        raw = []
        with torch.no_grad():
            for i in range(0, len(x), 128): raw.append(torch.stack([n_(SS.prep(torch, x[i:i+128], masks)) for n_ in nets]).mean(0).cpu().numpy())
        raw = np.concatenate(raw); out = {}
        for u in E.UNITS:
            sel = np.flatnonzero(m['unit'] == u); sc = []
            for c in range(40):
                loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
                assert np.array_equal(m['frame'][loc], SS.EVAL_FRAMES)
                sc.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None]).sum(0)/SS.WEIGHTS.sum())
            out[str(u)] = np.asarray(sc)
        np.savez_compressed(OUT/f'scores92000_{arm}.npz', **out); print('infer92', arm, flush=True)
        del nets; torch.cuda.empty_cache()


def near_rows(scores):
    rows = []
    for u in NR.SPLITS['evaluation']:
        d = SS.read(NR.OUT/'features/evaluation'/f'unit{u}.npz'); scenes = NR.scenes_for(u)
        lab = d['labels'].reshape(40, 16, 6)[:, -1, 2:4]
        for c in range(40):
            tq = int(d['group'][c])
            for q in (0, 1):
                rows.append(dict(unit=u, g=q, cond=str(d['family'][c]), rng=float(scenes[c]['meta']['range']), lab=int(lab[c, q]),
                                 off=-float(d['margin'][c]) if q == tq else None, s={a: float(scores[a][f'near|{u}'][c, q]) for a in HALF}))
    return rows


def rf_rows(scores, split):
    man = {(r['unit'], r['config']): r for r in json.loads((RF.OUT/'scene_manifest.json').read_text()) if r['split'] == split}
    rows = []
    for u in RF.SPLITS[split]:
        d = SS.read(RF.OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
        lab = d['labels'].reshape(k, 16, 6)[:, -1, 2:4]
        for c in range(k):
            mm = man[(u, c)]
            for q in (0, 1):
                rows.append(dict(unit=u, g=q, lab=int(lab[c, q]), off=-float(mm['margin']) if q == mm['group'] else None,
                                 s={a: float(scores[a][f'rf_{split}|{u}'][c, q]) for a in HALF}))
    return rows


def evaluate():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    scores = {a: SS.read(OUT/f'scores_{a}.npz') for a in HALF}
    R = near_rows(scores); T = [r for r in R if r['off'] is not None]
    clear = lambda r: r['off'] is None and r['lab'] == 0
    thr = {(a, fa, q): float(np.quantile([r['s'][a] for r in R if r['g'] == q and clear(r)], 1-fa)) for a in HALF for fa in FAS for q in (0, 1)}
    units = np.array(NR.SPLITS['evaluation']); ui = {u: i for i, u in enumerate(units)}
    tu = np.array([ui[r['unit']] for r in T]); off = np.array([r['off'] for r in T]); rng_ = np.array([r['rng'] for r in T])
    alarm = {(a, fa): np.array([r['s'][a] >= thr[(a, fa, r['g'])] for r in T]) for a in HALF for fa in FAS}
    curves = {}
    for a in HALF:
        for fa in FAS:
            for lo_r, hi_r in RANGES+((.6, 2.1),):
                for b, (lo, hi) in BINS.items():
                    k = (rng_ >= lo_r) & (rng_ < hi_r) & (off > lo) & (off <= hi)
                    curves[f'{a}|{fa}|{lo_r}-{hi_r}|{b}'] = dict(n=int(k.sum()), alarm=float(alarm[(a, fa)][k].mean()))
    pool = (rng_ >= .6) & (rng_ < 2.1)
    sel = {b: pool & (off > lo) & (off <= hi) for b, (lo, hi) in BINS.items()}
    boot_rng = np.random.default_rng(2026100202); W = np.array([np.bincount(boot_rng.integers(0, len(units), len(units)), minlength=len(units)) for _ in range(1000)])

    def rate(a, b, w=None):
        k = sel[b]; y = alarm[(a, .10)][k].astype(float)
        if w is None: return float(y.mean())
        ww = w[tu[k]]; return float((ww*y).sum()/ww.sum())
    # same-height objects 12 cm outside the body line (clear under three-level truth): readout-fix set
    cal, ev = rf_rows(scores, 'calib'), rf_rows(scores, 'evaluation')
    rthr = {(a, q): float(np.quantile([r['s'][a] for r in cal if r['g'] == q and r['off'] is None and r['lab'] == 0], .9)) for a in HALF for q in (0, 1)}
    rf = {}
    for a in HALF:
        for name, (lo, hi) in {'outside 12cm (clear)': (-.14, -.10), 'outside 4.5cm (pass)': (-.06, -.03), 'outside 1.5cm (pass)': (-.03, 0.),
                               'inside 0-3cm': (0., .03), 'inside 3-6cm': (.03, .06), 'inside 12cm': (.10, .14), 'other-height clear': (None, None)}.items():
            rr = [r for r in ev if r['off'] is None and r['lab'] == 0] if lo is None else \
                 [r for r in ev if r['off'] is not None and lo < r['off'] <= hi and (r['lab'] == 0 or r['off'] > 0)]
            rf[f'{a}|{name}'] = dict(n=len(rr), alarm=float(np.mean([r['s'][a] >= rthr[(a, r['g'])] for r in rr])))
    primary = {}
    for a in ('M3', 'M8'):
        d0 = rate(a, 'contact 0-2')-rate('NEAR', 'contact 0-2'); bs = [rate(a, 'contact 0-2', w)-rate('NEAR', 'contact 0-2', w) for w in W]
        d5 = rate(a, 'contact >5')-rate('NEAR', 'contact >5')
        d12 = rf[f'{a}|outside 12cm (clear)']['alarm']-rf['NEAR|outside 12cm (clear)']['alarm']
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        primary[a] = dict(delta_contact02=d0, ci=ci, delta_contact_gt5=d5, delta_rf_outside12=d12,
                          verdict='HELPS' if ci[0] > 0 and d5 >= -.02 and d12 <= .05 else 'THRESHOLD_SUFFICES')
    result = dict(status='COMPLETE', rule=RULE, primary=primary, curves=curves, readout_fix=rf,
                  n=dict(target_rows=len(T), pooled_contact02=int(sel['contact 0-2'].sum()), clear_rows=sum(clear(r) for r in R)))
    SS.save(OUT/'result.json', result)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, (lo_r, hi_r) in zip(axes, RANGES[:3]):
        for a, col in zip(HALF, ('C0', 'C1', 'C2')):
            for b, ls in (('contact 0-2', '-'), ('contact 2-5', '--'), ('pass 0-5', ':')):
                ax.plot([fa*100 for fa in FAS], [curves[f'{a}|{fa}|{lo_r}-{hi_r}|{b}']['alarm']*100 for fa in FAS], color=col, ls=ls, marker='o', ms=3,
                        label=f'{a} {b}')
        ax.set_title(f'{lo_r}-{hi_r} m'); ax.set_xlabel('clear-row alarm rate (%)'); ax.grid(alpha=.3)
    axes[0].set_ylabel('alarm rate (%)'); axes[0].legend(fontsize=7, ncol=1); fig.suptitle('Three-level truth: contact must alarm, near-pass acceptable, clear = false alarm (sim, consumed Development)')
    fig.tight_layout(); fig.savefig(OUT/'margin_curves.png', dpi=150)
    for a, p in primary.items():
        print(a, {k: (round(v*100, 1) if isinstance(v, float) else v) for k, v in p.items() if k != 'ci'}, [round(x*100, 1) for x in p['ci']])
    for (lo_r, hi_r) in RANGES+((.6, 2.1),):
        print(f'{lo_r}-{hi_r} @10%:', ' | '.join(f"{a} " + ' '.join(f"{b.split()[0][0]}{b.split()[1]}={curves[f'{a}|0.1|{lo_r}-{hi_r}|{b}']['alarm']*100:.0f}" for b in BINS) for a in HALF))
    for k, v in rf.items(): print('RF', k, v['n'], round(v['alarm']*100, 1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', required=True, choices=['plan', 'relabel', 'train', 'infer', 'infer92', 'evaluate']); p.add_argument('--arm')
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'plan':
        assert not (OUT/'PLAN.json').exists()
        SS.save(OUT/'PLAN.json', dict(rule=RULE, half_widths=HALF, script_sha256=SS.sha(__file__), role='EXPLORE; consumed Development; user adopted three-level truth 2026-10-02',
                                      limits=['training targets only at +-1.5/4.5/12 cm: M3/M8 are the only distinct widenings available',
                                              'near-range evaluation has no same-height objects beyond 10 cm outside; readout-fix set supplies 12 cm',
                                              'clear rows are other-height objects and panels, not real walking false-alarm burden',
                                              'thresholds set on the same evaluation clear rows (near-range) as in prior runs']))
    elif a.stage == 'relabel': relabel()
    elif a.stage == 'train': train(a.arm)
    elif a.stage == 'infer': infer()
    elif a.stage == 'infer92': infer92()
    else: evaluate()
