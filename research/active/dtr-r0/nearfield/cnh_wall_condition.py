"""Privileged background-wall conditioning diagnostic (fast lane, Development).

CVR readouts receive a per-frame descriptor of the context panel (side, inner
lateral face, thickness, top, height, forward extent, reflectance) in the
current query frame, never the target. Arms: true vs shuffled (fixed
derangement across panel scenes) descriptors, at FULL and UB2 coverage; the
original descriptor-free models are the third reference. Evaluation reuses the
consumed coverage-sweep units. Plan: artifacts.local/work/cnh-wall-condition-20260930/WALL_PLAN.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_coverage_sweep as CS

OUT = SS.WORK/'cnh-wall-condition-20260930'
PLAN = OUT/'WALL_PLAN.md'
PILOT = SS.OUT
SWEEP = CS.OUT
COVERAGE = {'FULL': (10, (0, 1, 2, 3, 4)), 'UB2': (22, (0, 1, 2))}  # pilot train config block, support bins
ORIGINAL = {'FULL': PILOT/'models/FULL', 'UB2': PILOT/'models/EXTRAP'}
VARIANTS = ('true', 'shuf')
DIM = 9
GROUPS = SS.GROUPS


def descriptor(panel, travel):
    """[present, side, x_inner, thickness, y_top, height, z_lo, z_hi, rho] of the panel's bounding box in each query frame."""
    lo, hi = np.asarray(panel['lo'], float), np.asarray(panel['hi'], float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    out = []
    for t in travel:
        local = (corners-t[:3, 3])@t[:3, :3]; a, b = local.min(0), local.max(0)
        side = 1. if a[0]+b[0] > 0 else -1.
        inner = a[0] if side > 0 else -b[0]
        out.append([1., side, inner, b[0]-a[0], a[1], b[1]-a[1], a[2], b[2], panel['rho']])
    return np.asarray(out, np.float32)


def scene_descriptors(scenes, frames):
    """Per scene: descriptors at the given frames (zeros when no context panel)."""
    out = {}
    for s in scenes:
        panel = s['boxes'][1] if s.get('family') == 'panel' else None
        out[s['config']] = descriptor(panel, s['travel'][frames]) if panel is not None else np.zeros((len(frames), DIM), np.float32)
    return out


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any(): return p


def build(split):
    """Descriptor arrays aligned to the CVR feature rows: true and shuffled."""
    import cnh_proposal_attribution_scenes as S
    if split == 'train':
        m = SS.read(PILOT/'data/train/metadata.npz'); frames = SS.TRAIN_FRAMES; make = SS.scenes_for
    else:
        m = SS.read(SWEEP/'data'/split/'metadata.npz'); frames = SS.EVAL_FRAMES; make = CS.scenes_for
    per = {}
    for u in np.unique(m['unit']):
        for c, d in scene_descriptors(make(int(u)), frames).items(): per[(int(u), c)] = d
    keys = sorted(per); panel_keys = [k for k in keys if per[k][0, 0] == 1]
    # Train: derange within each arm's own panel block so a shuffled arm never sees descriptors outside its support.
    block = (lambda k: (k[1]-10)//12) if split == 'train' else (lambda k: 0)
    shuffled = dict(per)
    for g in sorted({block(k) for k in panel_keys}):
        pk = [k for k in panel_keys if block(k) == g]
        perm = derangement(len(pk), {'train': 11, 'calib': 12, 'evaluation': 13}[split]+100*g)
        shuffled.update({k: per[pk[j]] for k, j in zip(pk, perm)})
    true = np.zeros((len(m['unit']), DIM), np.float32); shuf = np.zeros_like(true)
    fi = {int(f): i for i, f in enumerate(frames)}
    for r, (u, c, f) in enumerate(zip(m['unit'], m['config'], m['frame'])):
        true[r] = per[(int(u), int(c))][fi[int(f)]]; shuf[r] = shuffled[(int(u), int(c))][fi[int(f)]]
    np.savez_compressed(OUT/f'descriptors_{split}.npz', true=true, shuf=shuf, panel_scenes=len(panel_keys))
    print('descriptors', split, len(true), 'panel scenes', len(panel_keys), flush=True)


def model(torch):
    from cnh_cvr_pilot import CVR

    class WallCVR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = CVR(); self.body = base.body; self.emb = base.emb
            self.wall = torch.nn.Sequential(torch.nn.Linear(DIM, 16), torch.nn.GELU())
            self.head = torch.nn.Sequential(torch.nn.Linear(88, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))

        def forward(self, x, w):
            f = self.body(x)
            mask = torch.nn.functional.adaptive_avg_pool3d(x[:, 3:5], f.shape[2:])[:, :, None]; ff = f[:, None]
            mx = ff.masked_fill(mask == 0, -1e4).flatten(3).max(-1).values
            mean = (ff*mask).flatten(3).sum(-1)/mask.flatten(3).sum(-1).clamp_min(1e-8)
            emb = self.emb.weight[None].expand(len(x), -1, -1)
            ww = self.wall(w)[:, None].expand(-1, 2, -1)
            return self.head(torch.cat([mx, mean, emb, ww], -1)).squeeze(-1)
    return WallCVR()


def train():
    import torch
    from cnh_cvr_projection import query_masks
    assert PLAN.exists()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    x = np.load(PILOT/'data/train/features.npy', mmap_mode='r'); m = SS.read(PILOT/'data/train/metadata.npz')
    desc = SS.read(OUT/'descriptors_train.npz'); labels = m['labels'].astype(np.float32)
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    for cov, (off, _) in COVERAGE.items():
        ids = np.flatnonzero((m['config'] < 10) | ((m['config'] >= off) & (m['config'] < off+12))); assert len(ids) == 96*22*13
        for var in VARIANTS:
            w_all = desc[var]
            for seed in range(3):
                final = OUT/'models'/f'{cov}_{var}'/f'model_seed{seed}.pt'
                if final.exists(): continue
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
                net = model(torch).cuda(); opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); history = []
                for epoch in range(20):
                    start = time.monotonic(); net.train(); order = rng.permutation(ids); total = 0.
                    for c in range(0, len(order), 256):
                        block = order[c:c+256]; opt.zero_grad(set_to_none=True)
                        for i in range(0, len(block), 64):
                            sub = np.sort(block[i:i+64])
                            out = net(SS.prep(torch, x[sub], masks), torch.as_tensor(w_all[sub], device='cuda'))
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.as_tensor(labels[sub], device='cuda'))
                            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                            (loss*(len(sub)/len(block))).backward(); total += float(loss.detach())*len(sub)
                        opt.step()
                    sched.step(); history.append(dict(epoch=epoch+1, loss=total/len(order), elapsed_s=time.monotonic()-start))
                    print('train', cov, var, seed, history[-1], flush=True)
                final.parent.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, final)
                SS.save(final.parent/f'seed{seed}_history.json', history)
                SS.save(final.parent/f'seed{seed}_receipt.json', dict(seed=seed, parameters=sum(p.numel() for p in net.parameters()), sha256=SS.sha(final), final_loss=history[-1]['loss']))
                del net, opt; torch.cuda.empty_cache()


def infer():
    import torch
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    dest = OUT/'predictions'; dest.mkdir(parents=True, exist_ok=True)
    for cov in COVERAGE:
        for var in VARIANTS:
            path = dest/f'{cov}_{var}.npz'
            if path.exists(): continue
            nets = []
            for s in range(3):
                net = model(torch).cuda(); net.load_state_dict(torch.load(OUT/'models'/f'{cov}_{var}'/f'model_seed{s}.pt', map_location='cuda', weights_only=True)); nets.append(net.eval())
            out = {}
            for split in ('calib', 'evaluation'):
                x = np.load(SWEEP/'data'/split/'features.npy', mmap_mode='r'); m = SS.read(SWEEP/'data'/split/'metadata.npz')
                w = SS.read(OUT/f'descriptors_{split}.npz')[var]; raw = []
                with torch.no_grad():
                    for i in range(0, len(x), 64):
                        b = SS.prep(torch, x[i:i+64], masks); wb = torch.as_tensor(w[i:i+64], device='cuda')
                        raw.append(torch.stack([n(b, wb) for n in nets]).mean(0).cpu().numpy())
                raw = np.concatenate(raw)
                for u in np.unique(m['unit']):
                    sel = np.flatnonzero(m['unit'] == u); scores = []
                    for c in range(int(m['config'][sel].max())+1):
                        loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
                        assert np.array_equal(m['frame'][loc], SS.EVAL_FRAMES)
                        scores.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None]).sum(0)/SS.WEIGHTS.sum())
                    out[f'{split}|{int(u)}'] = np.asarray(scores)
            np.savez_compressed(path, **out); print('infer', cov, var, flush=True); del nets; torch.cuda.empty_cache()


def evaluate():
    from cnh_corridor_late_fusion import ranks, threshold
    from scipy.stats import rankdata
    assert PLAN.exists() and not (OUT/'results.json').exists()
    orig = {'FULL': SS.read(SWEEP/'predictions/UB4.npz'), 'UB2': SS.read(SWEEP/'predictions/UB2.npz')}
    pred = {f'{c}_{v}': SS.read(OUT/'predictions'/f'{c}_{v}.npz') for c in COVERAGE for v in VARIANTS}
    for c in COVERAGE: pred[f'{c}_orig'] = orig[c]
    man = {(r['unit'], r['config']): r for r in json.loads((SWEEP/'scene_manifest.json').read_text()) if r['split'] != 'train'}
    arms = tuple(pred)

    def rows(split):
        out = []
        for u in CS.SPLITS[split]:
            d = SS.read(SWEEP/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
            for c in range(k):
                mm = man[(u, c)]
                for q, g in enumerate(GROUPS):
                    out.append(dict(unit=u, group=g, kind=mm['kind'], bin=mm['bin'], label=int(labels[c, q]),
                                    scores={a: float(pred[a][f'{split}|{u}'][c, q]) for a in arms}))
        return out
    cal, ev = rows('calib'), rows('evaluation')
    for a in arms:
        sup = COVERAGE[a.split('_')[0]][1]
        for g in GROUPS:
            ref = np.sort([r['scores'][a] for r in cal if r['group'] == g and r['label'] == 0 and r['bin'] in sup])
            k = threshold(ranks(ref, ref), len(ref))
            for r in ev:
                if r['group'] == g: r.setdefault('pred', {})[a] = int(np.searchsorted(ref, r['scores'][a], side='right') >= k)
    units = np.array(CS.SPLITS['evaluation']); ui = {u: i for i, u in enumerate(units)}
    cells = {}
    for kind in CS.KINDS:
        for b in range(5):
            for g in GROUPS:
                rr = [r for r in ev if r['kind'] == kind and r['bin'] == b and r['group'] == g]
                cells[(kind, b, g)] = dict(u=np.array([ui[r['unit']] for r in rr]), y=np.array([r['label'] for r in rr], bool),
                                           s={a: np.array([r['scores'][a] for r in rr]) for a in arms},
                                           p={a: np.array([r['pred'][a] for r in rr], bool) for a in arms})

    def auc(y, s):
        r = rankdata(s); p = int(y.sum()); n = len(y)-p
        return (r[y].sum()-p*(p+1)/2)/(p*n)

    def stat(w):
        out = {}
        for key, c in cells.items():
            keep = np.repeat(np.arange(len(c['u'])), w[c['u']]); y = c['y'][keep]
            for a in arms:
                p = c['p'][a][keep]
                out[('auc', a)+key] = auc(y, c['s'][a][keep])
                out[('ber', a)+key] = .5*((y & ~p).sum()/y.sum()+(~y & p).sum()/(~y).sum())
        return out
    point = stat(np.ones(len(units), int)); rng = np.random.default_rng(20260930)
    boots = [stat(np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))) for _ in range(2000)]
    fam = lambda st, what, a, kind, b: float(np.mean([st[(what, a, kind, b, g)] for g in GROUPS]))
    ci = lambda fn: [float(np.percentile([fn(bt) for bt in boots], 2.5)), float(np.percentile([fn(bt) for bt in boots], 97.5))]
    table, readings = [], []
    for cov in COVERAGE:
        for kind in CS.KINDS:
            for b in range(5):
                row = dict(coverage=cov, kind=kind, bin=b)
                for v in ('orig', *VARIANTS):
                    a = f'{cov}_{v}'; c0 = cells[(kind, b, 'HEAD')]; c1 = cells[(kind, b, 'BODY')]
                    row[v] = dict(auc=fam(point, 'auc', a, kind, b), ber=fam(point, 'ber', a, kind, b),
                                  fn=int(sum(((c['y'] & ~c['p'][a]).sum()) for c in (c0, c1))), fp=int(sum(((~c['y'] & c['p'][a]).sum()) for c in (c0, c1))))
                d_ts = lambda st, what='auc': fam(st, what, f'{cov}_true', kind, b)-fam(st, what, f'{cov}_shuf', kind, b)
                d_so = lambda st: fam(st, 'auc', f'{cov}_shuf', kind, b)-fam(st, 'auc', f'{cov}_orig', kind, b)
                row.update(auc_true_minus_shuf=d_ts(point), auc_true_minus_shuf_ci=ci(d_ts),
                           ber_true_minus_shuf=d_ts(point, 'ber'), ber_true_minus_shuf_ci=ci(lambda st: d_ts(st, 'ber')),
                           auc_shuf_minus_orig=d_so(point), auc_shuf_minus_orig_ci=ci(d_so))
                table.append(row)
                if kind == 'corner' and b in (3, 4):
                    lo, hi = row['auc_true_minus_shuf_ci']; bl, bh = row['ber_true_minus_shuf_ci']
                    if row['auc_true_minus_shuf'] >= .03 and lo > 0 and abs(row['auc_shuf_minus_orig']) <= .02: v = 'BACKGROUND_HELPS'
                    elif abs(row['auc_true_minus_shuf']) <= .01 and (bh < 0 or bl > 0): v = 'OPERATING_POINT_ONLY'
                    elif -.02 <= lo and hi <= .02: v = 'NO_IMPROVEMENT'
                    else: v = 'UNCERTAIN'
                    readings.append(dict(coverage=cov, region=f'corner B{b}', reading=v))
    SS.save(OUT/'results.json', dict(scope='privileged-wall Development diagnostic on consumed sweep units', plan_sha256=SS.sha(PLAN),
                                     script_sha256=SS.sha(__file__), readings=readings, table=table))
    for r in readings: print(r)
    for t in table:
        print(f"{t['coverage']:4s} {t['kind']:6s} B{t['bin']} AUC orig/true/shuf {t['orig']['auc']:.3f}/{t['true']['auc']:.3f}/{t['shuf']['auc']:.3f} "
              f"true-shuf {t['auc_true_minus_shuf']:+.3f}{np.round(t['auc_true_minus_shuf_ci'], 3).tolist()} shuf-orig {t['auc_shuf_minus_orig']:+.3f} "
              f"BER {t['orig']['ber']*100:.1f}/{t['true']['ber']*100:.1f}/{t['shuf']['ber']*100:.1f} FN {t['orig']['fn']}/{t['true']['fn']}/{t['shuf']['fn']} FP {t['orig']['fp']}/{t['true']['fp']}/{t['shuf']['fp']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['build', 'train', 'infer', 'evaluate'], required=True)
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    try:
        if a.stage == 'build':
            for s in ('train', 'calib', 'evaluation'): build(s)
        else:
            {'train': train, 'infer': infer, 'evaluate': evaluate}[a.stage]()
        SS.save(OUT/f'terminal_{a.stage}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_{a.stage}.json', dict(status='failed', error=repr(e))); raise
