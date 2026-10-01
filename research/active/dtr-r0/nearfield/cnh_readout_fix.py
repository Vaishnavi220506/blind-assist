"""Readout-fix pilot: lateral resolution vs corner-rich data for strong-background targets.

Three FULL-coverage CVR arms at equal sample count (27,456), 5 seeds each:
BASE (original; pilot FULL seeds 0-2 + new seeds 3-4), RES (first conv keeps
lateral 5 cm resolution, stride (1,2,2); same parameter count), REBAL (half of
the panel slots replaced by near same-side panels in the two largest area
bins). Fresh calib/evaluation units with random and corner panels.
Plan: artifacts.local/work/cnh-readout-fix-20261001/FIX_PLAN.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_coverage_sweep as CS

OUT = SS.WORK/'cnh-readout-fix-20261001'
PLAN = OUT/'FIX_PLAN.md'
PILOT = SS.OUT
SPLITS = {'train': SS.SPLITS['train'], 'calib': list(range(88000, 88048)), 'evaluation': list(range(87000, 87144))}
USED = CS.USED | set(CS.SPLITS['calib']) | set(CS.SPLITS['evaluation']) | set(range(85000, 85024)) | set(range(86000, 86024))
ARMS = ('BASE', 'RES', 'REBAL')
SEEDS = range(5)
REBAL_OFFSET = 82
GROUPS = SS.GROUPS
_ORIGINAL = {}


def known(unit):
    import cnh_proposal_attribution_scenes as S
    return _ORIGINAL.get('make_scenes', S.make_scenes)(unit)


def scenes_for(unit):
    import cnh_proposal_attribution_scenes as S
    if unit in SPLITS['train']:
        ref = known(unit)[0]; rng = np.random.default_rng([2026100101, int(unit)]); out = []
        for j in range(6):
            m = S.MARGINS[j]+rng.uniform(-.006, .006)
            t = SS.draw_target(rng, m, (unit+j) % 2)
            sh = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.08, .15)))
            out.append(SS.panel_scene(unit, REBAL_OFFSET+j, ref, t, sh, SS.support_log_area(float(rng.uniform()), (3, 4))))
        return out
    CS._ORIGINAL['make_scenes'] = known
    return CS.scenes_for(unit)  # fresh unit ids -> new 40-panel random/corner evaluation scenes


def manifest():
    assert not USED & (set(SPLITS['calib']) | set(SPLITS['evaluation']))
    rows = []
    for split, units in SPLITS.items():
        for u in units:
            for s in scenes_for(u):
                rows.append(dict(split=split, unit=u, config=s['config'], margin=s['margin'], group=s['group'],
                                 label_final=s['labels'][-1].tolist(), **s['meta']))
    SS.save(OUT/'scene_manifest.json', rows)
    ev = [r for r in rows if r['split'] == 'evaluation']
    counts = {f"{k}|B{b}|{g}": dict(pos=sum(r['label_final'][q] for r in ev if r['kind'] == k and r['bin'] == b),
                                   rows=sum(1 for r in ev if r['kind'] == k and r['bin'] == b))
              for k in CS.KINDS for b in range(5) for q, g in enumerate(GROUPS)}
    tr = [r for r in rows if r['split'] == 'train']
    SS.save(OUT/'cell_counts.json', dict(evaluation=counts, rebal_train=dict(scenes=len(tr), bins=np.bincount([r['bin'] for r in tr], minlength=5).tolist(),
                                                                          positives=sum(sum(r['label_final']) for r in tr))))
    print(json.dumps(json.loads((OUT/'cell_counts.json').read_text())['rebal_train']), sorted({(v['pos'], v['rows']) for v in counts.values()}))


def generate(split, chunk=None):
    """chunk='k/n' renders units[k::n] so several processes can share one split (existing units are skipped)."""
    assert PLAN.exists()
    import cnh_proposal_attribution_scenes as S
    import cnh_corridor_diagnostic as D
    shared_save = D.old.save
    tag = split+(f"_c{chunk.replace('/', 'of')}" if chunk else '')

    def split_save(path, obj):
        path = Path(path)
        if path.name in ('generation_progress.json', 'generation_terminal.json'):
            return SS.save(path.with_name(path.stem+f'_{tag}.json'), obj)
        return shared_save(path, obj)
    D.old.save = split_save
    _ORIGINAL['make_scenes'] = S.make_scenes
    S.make_scenes = scenes_for
    units = SPLITS[split]
    if chunk:
        k, n = map(int, chunk.split('/')); units = units[k::n]
    D.OUT = OUT; D.SPLITS = {split: units}
    D.generate()


def wait_complete(path, timeout=3600):
    """A unit npz is complete once it loads: the zip central directory is written last."""
    start = time.time()
    while True:
        if path.exists():
            try:
                return SS.read(path)
            except Exception:
                pass
        if time.time()-start > timeout: raise TimeoutError(f'{path} not complete after {timeout}s')
        time.sleep(3)


def materialize(split):
    """Same projector, frames and ordering as the coverage sweep; streams units as generation writes them."""
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    torch.set_num_threads(2)
    projector = BatchedProjector(); frames = SS.TRAIN_FRAMES if split == 'train' else SS.EVAL_FRAMES
    folder = OUT/'data'/split
    if (folder/'metadata.npz').exists(): return
    folder.mkdir(parents=True, exist_ok=True)
    units = SPLITS[split]; per_unit = 6 if split == 'train' else 40
    n = len(units)*per_unit*len(frames)
    feats = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype=np.float16, shape=(n, 3, *SHAPE))
    meta = {k: [] for k in ['unit', 'config', 'frame', 'family', 'labels']}
    offset = 0; start = time.time()
    for i, u in enumerate(units):
        d = wait_complete(OUT/'features'/split/f'unit{u}.npz'); ref = known(u)[0]
        assert len(d['family']) == per_unit
        configs = sorted(set(d['scene'].tolist()))
        for config in configs:
            ids = np.flatnonzero(d['scene'] == config)
            assert np.array_equal(d['frame'][ids], np.arange(16))
            sensor, travel, noisy = motion_metadata(u, config)
            assert np.allclose(sensor, ref['poses'], atol=1e-9) and np.allclose(travel, ref['travel'], atol=1e-9)
            for f in frames:
                voxel = projector.sequence(d['z1'][ids[max(0, f-7):f+1]], relative_transforms(sensor, travel, noisy, int(f)))
                assert bool(torch.isfinite(voxel).all())
                feats[offset] = voxel.cpu().numpy().astype(np.float16); offset += 1
            meta['unit'] += [u]*len(frames); meta['config'] += [config]*len(frames)
            meta['frame'] += frames.tolist(); meta['family'] += [str(d['family'][configs.index(config)])]*len(frames)
            meta['labels'] += d['labels'][ids[frames], 2:4].tolist()
        print('materialize', split, u, round(time.time()-start, 1), flush=True)
    assert offset == n
    feats.flush(); del feats
    np.savez_compressed(folder/'metadata.npz', **{k: np.asarray(v) for k, v in meta.items()})


def model(torch, arm):
    from cnh_cvr_pilot import CVR
    net = CVR()
    if arm == 'RES': net.body[0] = torch.nn.Conv3d(5, 16, 3, stride=(1, 2, 2), padding=1)
    return net


def train():
    import torch
    from cnh_cvr_projection import query_masks
    assert PLAN.exists()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    xo = np.load(PILOT/'data/train/features.npy', mmap_mode='r'); mo = SS.read(PILOT/'data/train/metadata.npz')
    ready = (OUT/'data/train/metadata.npz').exists()  # REBAL waits for its rendered corner panels; rerun train() later
    if ready:
        xn = np.load(OUT/'data/train/features.npy', mmap_mode='r'); mn = SS.read(OUT/'data/train/metadata.npz')
    else:
        xn = np.zeros((0, *xo.shape[1:]), xo.dtype); mn = dict(labels=np.zeros((0, 2)))
    no = len(xo); masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    labels = np.concatenate([mo['labels'], mn['labels']]).astype(np.float32)
    full = np.flatnonzero(mo['config'] < 22)
    ids_by_arm = {'BASE': full, 'RES': full,
                  'REBAL': np.concatenate([np.flatnonzero(mo['config'] < 16), no+np.arange(len(xn))])}
    def prep_gpu(xb):  # identical transform to SS.prep, applied to GPU-resident float16 rows
        x = xb.float()
        x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p(); x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p(); x[:, 1] /= 8
        return torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)

    for arm in ARMS:
        if arm == 'REBAL' and not ready: continue
        ids = ids_by_arm[arm]; assert len(ids) == 96*22*13, (arm, len(ids)); assert np.all(np.diff(ids) > 0)
        todo = [s for s in SEEDS if not (OUT/'models'/arm/f'model_seed{s}.pt').exists() and not (arm == 'BASE' and s < 3)]
        if not todo: continue
        # Efficiency (no scientific change): the arm's 27,456 rows live in GPU memory; one 256-row forward per
        # optimizer step equals the former 4x64 accumulation (same rows, same order, same mean loss).
        X = torch.empty((len(ids), 3, *xo.shape[2:]), dtype=torch.float16, device='cuda'); t0 = time.monotonic()
        for c in range(0, len(ids), 1024):
            sub = ids[c:c+1024]; a = sub[sub < no]; b = sub[sub >= no]-no
            X[c:c+len(sub)] = torch.as_tensor(np.concatenate([np.asarray(xo[a]), np.asarray(xn[b])]) if len(b) else np.asarray(xo[a]), device='cuda')
        Y = torch.as_tensor(labels[ids], device='cuda'); print('loaded', arm, round(time.monotonic()-t0, 1), 's', flush=True)
        for seed in todo:
            final = OUT/'models'/arm/f'model_seed{seed}.pt'
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
                    loss.backward(); total += float(loss.detach())*len(block)
                    opt.step()
                sched.step(); history.append(dict(epoch=epoch+1, loss=total/len(order), elapsed_s=time.monotonic()-start))
                print('train', arm, seed, history[-1], flush=True)
            final.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, final)
            SS.save(final.parent/f'seed{seed}_history.json', history)
            SS.save(final.parent/f'seed{seed}_receipt.json', dict(arm=arm, seed=seed, sha256=SS.sha(final), final_loss=history[-1]['loss']))
            del net, opt; torch.cuda.empty_cache()
        del X, Y; torch.cuda.empty_cache()


def model_path(arm, seed):
    return PILOT/'models/FULL'/f'model_seed{seed}.pt' if arm == 'BASE' and seed < 3 else OUT/'models'/arm/f'model_seed{seed}.pt'


def infer():
    """Per-seed smoothed scores (frames 11-15, weights 1..16) for calib and evaluation."""
    import torch
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    dest = OUT/'predictions'; dest.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        path = dest/f'{arm}.npz'
        if path.exists(): continue
        nets = []
        for s in SEEDS:
            net = model(torch, arm).cuda(); net.load_state_dict(torch.load(model_path(arm, s), map_location='cuda', weights_only=True)); nets.append(net.eval())
        out = {}
        for split in ('calib', 'evaluation'):
            x = np.load(OUT/'data'/split/'features.npy', mmap_mode='r'); m = SS.read(OUT/'data'/split/'metadata.npz'); raw = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    b = SS.prep(torch, x[i:i+64], masks); raw.append(torch.stack([n(b) for n in nets], -1).cpu().numpy())
            raw = np.concatenate(raw)  # [N, 2, seeds]
            for u in np.unique(m['unit']):
                sel = np.flatnonzero(m['unit'] == u); scores = []
                for c in range(int(m['config'][sel].max())+1):
                    loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
                    assert np.array_equal(m['frame'][loc], SS.EVAL_FRAMES)
                    scores.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None, None]).sum(0)/SS.WEIGHTS.sum())
                out[f'{split}|{int(u)}'] = np.asarray(scores)  # [configs, 2, seeds]
            print('infer', arm, split, flush=True)
        np.savez_compressed(path, **out); del nets; torch.cuda.empty_cache()
    SS.save(OUT/'model_hashes.json', {arm: [SS.sha(model_path(arm, s)) for s in SEEDS] for arm in ARMS})


def evaluate():
    from cnh_corridor_late_fusion import ranks, threshold
    from scipy.stats import rankdata
    assert PLAN.exists() and not (OUT/'results.json').exists()
    pred = {a: SS.read(OUT/'predictions'/f'{a}.npz') for a in ARMS}
    man = {(r['unit'], r['config']): r for r in json.loads((OUT/'scene_manifest.json').read_text()) if r['split'] != 'train'}

    def rows(split):
        out = []
        for u in SPLITS[split]:
            d = SS.read(OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
            for c in range(k):
                mm = man[(u, c)]
                for q, g in enumerate(GROUPS):
                    out.append(dict(unit=u, group=g, kind=mm['kind'], bin=mm['bin'], label=int(labels[c, q]),
                                    seeds={a: pred[a][f'{split}|{u}'][c, q].tolist() for a in ARMS}))
        return out
    cal, ev = rows('calib'), rows('evaluation')
    thr = {}
    for a in ARMS:
        for g in GROUPS:
            ref = np.sort([np.mean(r['seeds'][a]) for r in cal if r['group'] == g and r['label'] == 0])
            thr[(a, g)] = (ref, threshold(ranks(ref, ref), len(ref)))
    for r in ev:
        r['pred'] = {a: int(np.searchsorted(thr[(a, r['group'])][0], np.mean(r['seeds'][a]), side='right') >= thr[(a, r['group'])][1]) for a in ARMS}
    units = np.array(SPLITS['evaluation']); ui = {u: i for i, u in enumerate(units)}
    cells = {}
    for kind in CS.KINDS:
        for b in range(5):
            for g in GROUPS:
                rr = [r for r in ev if r['kind'] == kind and r['bin'] == b and r['group'] == g]
                cells[(kind, b, g)] = dict(u=np.array([ui[r['unit']] for r in rr]), y=np.array([r['label'] for r in rr], bool),
                                           s={a: np.array([r['seeds'][a] for r in rr]) for a in ARMS},
                                           p={a: np.array([r['pred'][a] for r in rr], bool) for a in ARMS})
    primary = [('corner', 3), ('corner', 4)]
    guard = [(k, b) for k in CS.KINDS for b in range(5) if (k, b) not in primary]

    def auc(y, s):
        r = rankdata(s); p = int(y.sum()); n = len(y)-p
        return (r[y].sum()-p*(p+1)/2)/(p*n)

    def metric(w_units, seed_pick):
        out = {}
        for key, c in cells.items():
            keep = np.repeat(np.arange(len(c['u'])), w_units[c['u']]); y = c['y'][keep]
            for a in ARMS:
                out[(a,)+key] = auc(y, c['s'][a][keep][:, seed_pick[a]].mean(1))
        return out
    region = lambda st, a, regs: float(np.mean([st[(a, k, b, g)] for k, b in regs for g in GROUPS]))
    full_pick = {a: np.arange(5) for a in ARMS}
    point = metric(np.ones(len(units), int), full_pick)
    rng = np.random.default_rng(20261001); boots = []
    for _ in range(2000):  # hierarchical: resample evaluation units and, per arm, the 5 training seeds
        boots.append(metric(np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)),
                            {a: rng.integers(0, 5, 5) for a in ARMS}))
    ci = lambda fn: [float(np.percentile([fn(bt) for bt in boots], 2.5)), float(np.percentile([fn(bt) for bt in boots], 97.5))]
    readings = {}
    for a in ('RES', 'REBAL'):
        dp = lambda st: region(st, a, primary)-region(st, 'BASE', primary)
        dg = lambda st: region(st, a, guard)-region(st, 'BASE', guard)
        cp, cg = ci(dp), ci(dg)
        if cg[1] < -.01: v = 'HARMFUL_GUARD'
        elif dp(point) >= .02 and cp[0] > 0 and cg[0] > -.01: v = 'EFFECTIVE'
        elif -.02 <= cp[0] and cp[1] <= .02: v = 'NO_EFFECT'
        else: v = 'UNCERTAIN'
        readings[a] = dict(reading=v, primary_delta=dp(point), primary_ci=cp, guard_delta=dg(point), guard_ci=cg)
    seed_auc = {a: [region(metric(np.ones(len(units), int), {**full_pick, a: np.array([s])}), a, primary) for s in SEEDS] for a in ARMS}
    table = []
    for kind in CS.KINDS:
        for b in range(5):
            row = dict(kind=kind, bin=b)
            for a in ARMS:
                c0, c1 = cells[(kind, b, 'HEAD')], cells[(kind, b, 'BODY')]
                fn = sum(int((c['y'] & ~c['p'][a]).sum()) for c in (c0, c1)); fp = sum(int((~c['y'] & c['p'][a]).sum()) for c in (c0, c1))
                P = int(c0['y'].sum()+c1['y'].sum()); N = int((~c0['y']).sum()+(~c1['y']).sum())
                row[a] = dict(auc=float(np.mean([point[(a, kind, b, g)] for g in GROUPS])), fn=fn, fp=fp, P=P, N=N, ber=.5*(fn/P+fp/N))
            table.append(row)
    SS.save(OUT/'results.json', dict(scope='Development pilot, fresh units 87000-87143 / 88000-88047', plan_sha256=SS.sha(PLAN), script_sha256=SS.sha(__file__),
                                     readings=readings, primary_auc_by_seed=seed_auc, table=table))
    print(json.dumps(readings, indent=1)); print('seed-level primary AUC', {a: np.round(v, 3).tolist() for a, v in seed_auc.items()})
    for t in table:
        print(f"{t['kind']:6s} B{t['bin']} " + ' '.join(f"{a}: AUC {t[a]['auc']:.3f} BER {t[a]['ber']*100:.1f} FN {t[a]['fn']}/{t[a]['P']} FP {t[a]['fp']}/{t[a]['N']} |" for a in ARMS))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['manifest', 'generate', 'materialize', 'train', 'infer', 'evaluate'], required=True)
    p.add_argument('--split', choices=list(SPLITS))
    p.add_argument('--chunk', help="k/n: render units[k::n] (generate only)")
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    tag = a.stage+(f'_{a.split}' if a.split else '')+(f"_c{a.chunk.replace('/', 'of')}" if a.chunk else '')
    try:
        fn = {'manifest': manifest, 'generate': generate, 'materialize': materialize, 'train': train, 'infer': infer, 'evaluate': evaluate}[a.stage]
        fn(a.split, a.chunk) if a.stage == 'generate' else fn(a.split) if a.stage == 'materialize' else fn()
        SS.save(OUT/f'terminal_{tag}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_{tag}.json', dict(status='failed', error=repr(e))); raise
