"""Boundary-aware (lateral-profile) CVR readout vs the global max/mean pooled readout.

PROF keeps the lateral axis: features are pooled only over each query's height/depth
band (across the full grid width, i.e. both sides of the corridor boundary), then a
small 1-D conv over x and a position-specific linear layer decide. PROF_RES also keeps
lateral 5 cm resolution in the first conv. Trained end-to-end on the pilot FULL data
(27,456 rows, CVR recipe, 5 seeds); evaluated on the readout-fix benchmark units
(88000-88047 calib, 87000-87143 evaluation) against the existing 5-seed BASE.
Primary metric: shallow-intrusion (<=5 cm) miss rate at the frozen 10% calib FP budget.
"""
import argparse
import json
import time

import numpy as np

import cnh_structure_space as SS
import cnh_readout_fix as RF

OUT = SS.WORK/'cnh-boundary-readout-20261001'
ARMS = ('PROF', 'PROF_RES')
SEEDS = range(5)
GROUPS = SS.GROUPS


def model(torch, arm):
    from cnh_cvr_pilot import CVR

    class Profile(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = CVR().body
            if arm == 'PROF_RES': self.body[0] = torch.nn.Conv3d(5, 16, 3, stride=(1, 2, 2), padding=1)
            width = 24 if arm == 'PROF_RES' else 12
            self.emb = torch.nn.Embedding(2, 8)
            self.lat = torch.nn.Sequential(torch.nn.Conv1d(72, 32, 3, padding=1), torch.nn.GELU(),
                                           torch.nn.Conv1d(32, 32, 3, padding=1), torch.nn.GELU())
            self.out = torch.nn.Linear(32*width, 1)

        def forward(self, x):
            f = self.body(x)                                                   # [N,32,X,Y,Z]
            m = torch.nn.functional.adaptive_avg_pool3d(x[:, 3:5], f.shape[2:])  # [N,2,X,Y,Z] query masks
            band = m.amax(2, keepdim=True).expand_as(m)                        # height/depth band, all x
            ff = f[:, None]; bb = band[:, :, None]                             # [N,1,32,X,Y,Z], [N,2,1,X,Y,Z]
            mx = ff.masked_fill(bb == 0, -1e4).flatten(4).max(-1).values       # [N,2,32,X]
            mean = (ff*bb).flatten(4).sum(-1)/bb.flatten(4).sum(-1).clamp_min(1e-8)
            emb = self.emb.weight[None, :, :, None].expand(len(x), -1, -1, mx.shape[-1])
            h = torch.cat([mx, mean, emb], 2).flatten(0, 1)                    # [N*2,72,X]
            return self.out(self.lat(h).flatten(1)).view(len(x), 2)
    return Profile()


def train():
    import torch
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    xo = np.load(SS.OUT/'data/train/features.npy', mmap_mode='r'); mo = SS.read(SS.OUT/'data/train/metadata.npz')
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    ids = np.flatnonzero(mo['config'] < 22); assert len(ids) == 96*22*13
    X = torch.empty((len(ids), 3, *xo.shape[2:]), dtype=torch.float16, device='cuda')
    for c in range(0, len(ids), 1024): X[c:c+1024] = torch.as_tensor(np.asarray(xo[ids[c:c+1024]]), device='cuda')
    Y = torch.as_tensor(mo['labels'][ids].astype(np.float32), device='cuda')

    def prep_gpu(xb):
        x = xb.float()
        x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p(); x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p(); x[:, 1] /= 8
        return torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)
    for arm in ARMS:
        for seed in SEEDS:
            final = OUT/'models'/arm/f'model_seed{seed}.pt'
            if final.exists(): continue
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
            net = model(torch, arm).cuda(); opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); history = []
            for epoch in range(20):
                start = time.monotonic(); net.train(); order = rng.permutation(ids); total = 0.
                for c in range(0, len(order), 256):
                    block = order[c:c+256]; opt.zero_grad(set_to_none=True)
                    pos = torch.as_tensor(np.searchsorted(ids, np.concatenate([np.sort(block[i:i+64]) for i in range(0, len(block), 64)])), device='cuda')
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(prep_gpu(X[pos])), Y[pos])
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                    loss.backward(); total += float(loss.detach())*len(block); opt.step()
                sched.step(); history.append(dict(epoch=epoch+1, loss=total/len(order), elapsed_s=time.monotonic()-start))
                print('train', arm, seed, history[-1], flush=True)
            final.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, final)
            SS.save(final.parent/f'seed{seed}_history.json', dict(parameters=sum(p.numel() for p in net.parameters()), history=history))
            del net, opt; torch.cuda.empty_cache()


def infer():
    import torch
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    dest = OUT/'predictions'; dest.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        nets = []
        for s in SEEDS:
            net = model(torch, arm).cuda(); net.load_state_dict(torch.load(OUT/'models'/arm/f'model_seed{s}.pt', map_location='cuda', weights_only=True)); nets.append(net.eval())
        out = {}
        for split in ('calib', 'evaluation'):
            x = np.load(RF.OUT/'data'/split/'features.npy', mmap_mode='r'); m = SS.read(RF.OUT/'data'/split/'metadata.npz'); raw = []
            with torch.no_grad():
                for i in range(0, len(x), 128):
                    b = SS.prep(torch, x[i:i+128], masks); raw.append(torch.stack([n(b) for n in nets], -1).cpu().numpy())
            raw = np.concatenate(raw)
            for u in np.unique(m['unit']):
                sel = np.flatnonzero(m['unit'] == u); scores = []
                for c in range(int(m['config'][sel].max())+1):
                    loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
                    assert np.array_equal(m['frame'][loc], SS.EVAL_FRAMES)
                    scores.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None, None]).sum(0)/SS.WEIGHTS.sum())
                out[f'{split}|{int(u)}'] = np.asarray(scores)
        np.savez_compressed(dest/f'{arm}.npz', **out); print('infer', arm, flush=True)
        del nets; torch.cuda.empty_cache()


def evaluate():
    from cnh_corridor_late_fusion import ranks, threshold
    pred = {'BASE': SS.read(RF.OUT/'predictions/BASE.npz'), **{a: SS.read(OUT/'predictions'/f'{a}.npz') for a in ARMS}}
    arms = tuple(pred)
    man = {(r['unit'], r['config']): r for r in json.loads((RF.OUT/'scene_manifest.json').read_text()) if r['split'] != 'train'}

    def rows(split):
        out = []
        for u in RF.SPLITS[split]:
            d = SS.read(RF.OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
            for c in range(k):
                mm = man[(u, c)]
                for q, g in enumerate(GROUPS):
                    depth = -float(d['margin'][c]) if int(d['group'][c]) == q else None  # intrusion depth of the target in this query
                    band = None if not labels[c, q] or depth is None else ('<=2cm' if depth <= .02 else '2-5cm' if depth <= .05 else '>5cm')
                    out.append(dict(unit=u, group=g, kind=mm['kind'], bin=mm['bin'], label=int(labels[c, q]), band=band,
                                    s={a: np.asarray(pred[a][f'{split}|{u}'][c, q]) for a in arms}))
        return out
    cal, ev = rows('calib'), rows('evaluation')
    units = np.array(RF.SPLITS['evaluation']); ui = {u: i for i, u in enumerate(units)}
    U = np.array([ui[r['unit']] for r in ev]); y = np.array([r['label'] for r in ev], bool)
    band = np.array([r['band'] or '' for r in ev]); grp = np.array([r['group'] for r in ev])
    S_ev = {a: np.stack([r['s'][a] for r in ev]) for a in arms}                 # [rows, seeds]
    S_cal = {a: np.stack([r['s'][a] for r in cal]) for a in arms}
    cal_neg = {g: np.array([r['label'] == 0 and r['group'] == g for r in cal]) for g in GROUPS}

    def decide(a, seed_pick):
        """Frozen per-group threshold on calib negatives (10% budget) for the chosen seed ensemble."""
        p = np.zeros(len(ev), bool)
        for g in GROUPS:
            ref = np.sort(S_cal[a][cal_neg[g]][:, seed_pick].mean(1)); k = threshold(ranks(ref, ref), len(ref))
            sel = grp == g; p[sel] = np.searchsorted(ref, S_ev[a][sel][:, seed_pick].mean(1), side='right') >= k
        return p

    def stats(w, picks):
        keep = np.repeat(np.arange(len(ev)), w[U]); out = {}
        for a in arms:
            p = decide(a, picks[a])[keep]; yy = y[keep]; bb = band[keep]
            for b in ('<=2cm', '2-5cm', '>5cm'):
                sel = yy & (bb == b); out[(a, 'miss', b)] = float((~p[sel]).mean())
            sh = yy & ((bb == '<=2cm') | (bb == '2-5cm')); out[(a, 'miss', 'shallow')] = float((~p[sh]).mean())
            out[(a, 'fpr',)] = float(p[~yy].mean())
        return out
    full = {a: np.arange(5) for a in arms}
    point = stats(np.ones(len(units), int), full); rng = np.random.default_rng(20261001); boots = []
    for _ in range(1000):
        boots.append(stats(np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)), {a: rng.integers(0, 5, 5) for a in arms}))
    ci = lambda fn: [float(np.percentile([fn(b) for b in boots], 2.5)), float(np.percentile([fn(b) for b in boots], 97.5))]
    res = dict(benchmark='readout-fix units (consumed Development, reused per fast-lane rule)', point={'|'.join(k): v for k, v in point.items()})
    for a in ARMS:
        d_sh = lambda st: st[(a, 'miss', 'shallow')]-st[('BASE', 'miss', 'shallow')]
        d_deep = lambda st: st[(a, 'miss', '>5cm')]-st[('BASE', 'miss', '>5cm')]
        d_fpr = lambda st: st[(a, 'fpr')]-st[('BASE', 'fpr')]
        c_sh, c_deep, c_fpr = ci(d_sh), ci(d_deep), ci(d_fpr)
        if c_sh[1] < -.05 or (d_sh(point) <= -.05 and c_sh[1] < 0):
            v = 'EFFECTIVE' if c_fpr[1] < .01 and c_deep[1] < .02 else 'GAIN_WITH_COST'
        elif -.03 <= c_sh[0] and c_sh[1] <= .03: v = 'NO_EFFECT'
        else: v = 'UNCERTAIN'
        res[a] = dict(reading=v, shallow_miss_delta=d_sh(point), shallow_ci=c_sh, deep_miss_delta=d_deep(point), deep_ci=c_deep, fpr_delta=d_fpr(point), fpr_ci=c_fpr)
    SS.save(OUT/'results.json', res)
    for a in arms:
        print(a, ' '.join(f"miss{b}={point[(a, 'miss', b)]*100:.1f}%" for b in ('<=2cm', '2-5cm', 'shallow', '>5cm')), f"FPR={point[(a, 'fpr')]*100:.1f}%")
    for a in ARMS: print(a, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[a].items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['train', 'infer', 'evaluate'], required=True); a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        {'train': train, 'infer': infer, 'evaluate': evaluate}[a.stage](); SS.save(OUT/f'terminal_{a.stage}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_{a.stage}.json', dict(status='failed', error=repr(e))); raise
