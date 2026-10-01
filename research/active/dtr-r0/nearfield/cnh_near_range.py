"""Near-range detectability: does lateral boundary resolution improve as the walker approaches?

NEAR: CVR trained (5 seeds, FULL-pilot recipe and sample count) on new units whose
final-frame target range is uniform 0.6-2.6 m (10 no-context + 12 random-panel scenes
per unit). Evaluation: envelope-style scenes (20 none + 20 near same-side panel) with
continuous signed offset and range 0.6-2.6 m. Thresholds: 10% alarm rate on clear rows
(other height band) per group. Logistic psychometric fits per (model, condition, range).
Rendering and voxelisation run in parallel chunks; voxel chunks are separate files.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_readout_fix as RF

OUT = SS.WORK/'cnh-near-range-20261001'
SPLITS = {'train': list(range(93000, 93096)), 'evaluation': list(range(94000, 94096))}
PER = {'train': 22, 'evaluation': 40}
RANGE_BINS = ((.6, .9), (.9, 1.2), (1.2, 1.6), (1.6, 2.1), (2.1, 2.6))
ZONE_TAN = 2*np.tan(np.pi/8)/8
_ORIGINAL = {}


def known(unit):
    import cnh_proposal_attribution_scenes as S
    return _ORIGINAL.get('make_scenes', S.make_scenes)(unit)


def _none_scene(S, unit, config, ref, t):
    boxes = [t['box'], SS.FLOOR, SS.BACK]
    return dict(unit=int(unit), config=config, family='none', margin=t['margin'], group=t['group'], boxes=boxes,
                poses=ref['poses'].copy(), travel=ref['travel'].copy(), head=ref['head'].copy(),
                labels=S.labels_for(boxes, ref['travel']), mode=ref['mode'], dt=.2, speed=.8, meta={})


def _target(rng, margin, group):
    t = SS.draw_target(rng, margin, group); z = float(rng.uniform(.6, 2.6)); dz = z-t['z']
    t['box']['lo'][2] += dz; t['box']['hi'][2] += dz; t['z'] = z
    return t


def scenes_for(unit):
    import cnh_proposal_attribution_scenes as S
    ref = known(unit)[0]; out = []
    if unit in SPLITS['train']:
        rng = np.random.default_rng([2026100104, int(unit)])
        for s in range(22):
            t = _target(rng, S.MARGINS[s % 6]+rng.uniform(-.006, .006), (unit+s) % 2)
            if s < 10: sc = _none_scene(S, unit, s, ref, t)
            else:
                sh = SS.draw_shape(rng)
                sc = SS.panel_scene(unit, s, ref, t, sh, SS.EDGES[0]+float(rng.uniform())*(SS.EDGES[5]-SS.EDGES[0]))
            sc['meta'] = dict(sc.get('meta', {}), range=t['z']); out.append(sc)
        return out
    rng = np.random.default_rng([2026100105, int(unit)])
    for s in range(40):
        cond = 'none' if s < 20 else 'corner'
        t = _target(rng, float(rng.uniform(-.15, .10)), (unit//6+s) % 2)
        if cond == 'none': sc = _none_scene(S, unit, s, ref, t)
        else:
            sh = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.08, .15)))
            sc = SS.panel_scene(unit, s, ref, t, sh, SS.EDGES[0]+float(rng.uniform())*(SS.EDGES[5]-SS.EDGES[0]))
        sc['family'] = cond; sc['meta'] = dict(sc.get('meta', {}), cond=cond, range=t['z']); out.append(sc)
    return out


def generate(split, chunk):
    import cnh_proposal_attribution_scenes as S
    import cnh_corridor_diagnostic as D
    shared_save = D.old.save; tag = f"{split}_c{chunk.replace('/', 'of')}"

    def split_save(path, obj):
        path = Path(path)
        if path.name in ('generation_progress.json', 'generation_terminal.json'):
            return SS.save(path.with_name(path.stem+f'_{tag}.json'), obj)
        return shared_save(path, obj)
    D.old.save = split_save
    _ORIGINAL['make_scenes'] = S.make_scenes; S.make_scenes = scenes_for
    k, n = map(int, chunk.split('/'))
    D.OUT = OUT; D.SPLITS = {split: SPLITS[split][k::n]}
    D.generate()


def materialize(split, chunk):
    """Voxelise units[k::n] into data/<split>/features_c<k>.npy (+ metadata) as soon as each unit file is complete."""
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    torch.set_num_threads(2)
    k, n = map(int, chunk.split('/')); units = SPLITS[split][k::n]
    frames = SS.TRAIN_FRAMES if split == 'train' else SS.EVAL_FRAMES
    folder = OUT/'data'/split; folder.mkdir(parents=True, exist_ok=True)
    total = len(units)*PER[split]*len(frames); projector = BatchedProjector()
    feats = np.lib.format.open_memmap(folder/f'features_c{k}.npy', mode='w+', dtype=np.float16, shape=(total, 3, *SHAPE))
    meta = {key: [] for key in ('unit', 'config', 'frame', 'labels')}; offset = 0; start = time.time()
    for u in units:
        d = RF.wait_complete(OUT/'features'/split/f'unit{u}.npz'); ref = known(u)[0]
        for config in range(PER[split]):
            ids = np.flatnonzero(d['scene'] == config); assert np.array_equal(d['frame'][ids], np.arange(16))
            sensor, travel, noisy = motion_metadata(u, config)
            assert np.allclose(sensor, ref['poses'], atol=1e-9) and np.allclose(travel, ref['travel'], atol=1e-9)
            for f in frames:
                vox = projector.sequence(d['z1'][ids[max(0, f-7):f+1]], relative_transforms(sensor, travel, noisy, int(f)))
                feats[offset] = vox.cpu().numpy().astype(np.float16); offset += 1
            meta['unit'] += [u]*len(frames); meta['config'] += [config]*len(frames); meta['frame'] += frames.tolist()
            meta['labels'] += d['labels'][ids[frames], 2:4].tolist()
        print('materialize', split, u, round(time.time()-start, 1), flush=True)
    assert offset == total; feats.flush(); del feats
    np.savez_compressed(folder/f'metadata_c{k}.npz', **{key: np.asarray(v) for key, v in meta.items()})


def load_split(split):
    parts = sorted((OUT/'data'/split).glob('features_c*.npy'))
    xs = [np.load(p, mmap_mode='r') for p in parts]
    ms = [SS.read(p.with_name(p.name.replace('features_', 'metadata_').replace('.npy', '.npz'))) for p in parts]
    return xs, {key: np.concatenate([m[key] for m in ms]) for key in ms[0]}


def train():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    xs, m = load_split('train'); n = sum(len(x) for x in xs); assert n == 96*22*13
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    X = torch.empty((n, 3, *xs[0].shape[2:]), dtype=torch.float16, device='cuda'); o = 0
    for x in xs:
        for c in range(0, len(x), 1024): X[o+c:o+c+len(x[c:c+1024])] = torch.as_tensor(np.asarray(x[c:c+1024]), device='cuda')
        o += len(x)
    Y = torch.as_tensor(m['labels'].astype(np.float32), device='cuda'); ids = np.arange(n)

    def prep_gpu(xb):
        x = xb.float(); x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p(); x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p(); x[:, 1] /= 8
        return torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)
    for seed in range(5):
        final = OUT/'models/NEAR'/f'model_seed{seed}.pt'
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
            print('train NEAR', seed, history[-1], flush=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, final); SS.save(final.parent/f'seed{seed}_history.json', history)
        del net, opt; torch.cuda.empty_cache()


def infer():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    xs, m = load_split('evaluation')
    for name, paths in (('NEAR', [OUT/'models/NEAR'/f'model_seed{s}.pt' for s in range(5)]), ('BASE', [RF.model_path('BASE', s) for s in range(5)])):
        nets = []
        for p in paths:
            net = CVR().cuda(); net.load_state_dict(torch.load(p, map_location='cuda', weights_only=True)); nets.append(net.eval())
        raw = []
        with torch.no_grad():
            for x in xs:
                for i in range(0, len(x), 128): raw.append(torch.stack([n_(SS.prep(torch, x[i:i+128], masks)) for n_ in nets]).mean(0).cpu().numpy())
        raw = np.concatenate(raw); out = {}
        for u in SPLITS['evaluation']:
            sel = np.flatnonzero(m['unit'] == u); sc = []
            for c in range(40):
                loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
                sc.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None]).sum(0)/SS.WEIGHTS.sum())
            out[str(u)] = np.asarray(sc)
        np.savez_compressed(OUT/f'scores_{name}.npz', **out); del nets; torch.cuda.empty_cache(); print('infer', name, flush=True)


def evaluate():
    from sklearn.linear_model import LogisticRegression
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sc = {a: SS.read(OUT/f'scores_{a}.npz') for a in ('NEAR', 'BASE')}; rows = []
    for u in SPLITS['evaluation']:
        d = SS.read(OUT/'features/evaluation'/f'unit{u}.npz'); scenes = scenes_for(u); lab = d['labels'].reshape(40, 16, 6)[:, -1, 2:4]
        for c in range(40):
            tq = int(d['group'][c])
            for q in (0, 1):
                rows.append(dict(unit=u, g=q, cond=str(d['family'][c]), rng=scenes[c]['meta']['range'], lab=int(lab[c, q]),
                                 off=-float(d['margin'][c]) if q == tq else None,
                                 s={'NEAR': float(sc['NEAR'][str(u)][c, q]), 'BASE': float(sc['BASE'][str(u)][c, q]), 'PARTIAL': float(d['motion'][c, 2, q])}))
    arms = ('NEAR', 'BASE', 'PARTIAL'); thr = {}
    for a in arms:
        for q in (0, 1):
            thr[(a, q)] = float(np.quantile([r['s'][a] for r in rows if r['g'] == q and r['off'] is None and r['lab'] == 0], .9))
    fits = {}; fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ai, cond in enumerate(('none', 'corner')):
        for bi, (lo, hi) in enumerate(RANGE_BINS):
            rr = [r for r in rows if r['off'] is not None and r['cond'] == cond and lo <= r['rng'] < hi]
            o = np.array([r['off'] for r in rr]); uid = np.array([r['unit'] for r in rr])
            for a in arms:
                y = np.array([r['s'][a] >= thr[(a, r['g'])] for r in rr], int)

                def fit(idx):
                    lr = LogisticRegression(C=1e6, max_iter=2000).fit(o[idx, None], y[idx]); b1, b0 = lr.coef_[0, 0], lr.intercept_[0]
                    return -b0/b1, 1/b1
                mu, s = fit(np.arange(len(o))); boot = []
                g = np.random.default_rng(9)
                for _ in range(200):
                    idx = np.concatenate([np.flatnonzero(uid == p) for p in g.choice(SPLITS['evaluation'], 96)])
                    try: boot.append(fit(idx))
                    except Exception: pass
                boot = np.array(boot); band = 2*s*np.log(9)
                fits[f'{a}|{cond}|{lo}-{hi}'] = dict(n=len(o), mu_cm=mu*100, s_cm=s*100, band_cm=band*100,
                                                    band_ci_cm=[float(np.percentile(2*boot[:, 1]*np.log(9), q))*100 for q in (2.5, 97.5)],
                                                    d90_cm=(mu+s*np.log(9))*100, zone_cm=ZONE_TAN*(lo+hi)/2*100)
                if a != 'PARTIAL':
                    xs_ = np.linspace(-.10, .15, 200)
                    axes[ai].plot(xs_*100, 1/(1+np.exp(-(xs_-mu)/s)), color=f'C{bi}', ls='-' if a == 'NEAR' else ':',
                                  label=f'{a} {lo}-{hi} m')
        axes[ai].axvline(0, color='k', lw=.6); axes[ai].set_title({'none': 'no context', 'corner': 'near same-side panel'}[cond])
        axes[ai].set_xlabel('target intrusion into corridor (cm; <0 = outside)'); axes[ai].grid(alpha=.3); axes[ai].legend(fontsize=7, ncol=2)
    axes[0].set_ylabel('P(alarm) at 10% clear-row alarm rate'); fig.tight_layout(); fig.savefig(OUT/'near_range_envelope.png', dpi=150)
    SS.save(OUT/'results.json', dict(scope='Development characterisation; thresholds at 10% alarm on clear (other-band) rows of this evaluation set',
                                     fits=fits, script_sha256=SS.sha(__file__)))
    for k, v in fits.items():
        print(f"{k:24s} n={v['n']:4d} mu={v['mu_cm']:+6.1f} s={v['s_cm']:5.1f} band={v['band_cm']:6.1f} [{v['band_ci_cm'][0]:.1f},{v['band_ci_cm'][1]:.1f}] d90={v['d90_cm']:+6.1f} zone={v['zone_cm']:.0f}cm")


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', required=True, choices=['generate', 'materialize', 'train', 'infer', 'evaluate'])
    p.add_argument('--split'); p.add_argument('--chunk'); a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    tag = a.stage+(f'_{a.split}' if a.split else '')+(f"_c{a.chunk.replace('/', 'of')}" if a.chunk else '')
    try:
        if a.stage in ('generate', 'materialize'): {'generate': generate, 'materialize': materialize}[a.stage](a.split, a.chunk)
        else: {'train': train, 'infer': infer, 'evaluate': evaluate}[a.stage]()
        SS.save(OUT/f'terminal_{tag}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_{tag}.json', dict(status='failed', error=repr(e))); raise
