"""Structure-space coverage pilot: interpolation vs extrapolation hold-outs.

One parameterised context primitive (a floor-standing lateral panel beside the
corridor) whose lateral-facing area is the coverage axis. Three training arms
see the same targets, poses and panel placements and differ only in which
area bins their panels come from (FULL / EXTRAP / INTERP), at equal scene
count. CVR v2 recipe, sidewall-free generator families (boundary, general).
Plan: artifacts.local/work/cnh-structure-space-20260929/STRUCTURE_SPACE_PLAN.md
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'artifacts.local/work'
OUT = WORK/'cnh-structure-space-20260929'
PLAN = OUT/'STRUCTURE_SPACE_PLAN.md'
NOVEL = WORK/'cnh-novel-structure-20260929'
SPLITS = {'train': list(range(80000, 80096)), 'calib': list(range(81000, 81024)), 'evaluation': list(range(82000, 82096))}
USED = (set(range(0, 18)) | set(range(1000, 1096)) | set(range(2000, 2024)) | set(range(3000, 3048))
        | set(range(40000, 40024)) | set(range(50000, 50048)) | set(range(60000, 60048)) | set(range(70000, 70096)))
LOG_AREA = (np.log10(.03), np.log10(7.))
EDGES = np.linspace(*LOG_AREA, 6)  # five equal-width log10-area bins B0..B4
ARMS = {'FULL': (0, 1, 2, 3, 4), 'EXTRAP': (0, 1, 2), 'INTERP': (0, 1, 4)}
ARM_OFFSET = {'FULL': 10, 'EXTRAP': 22, 'INTERP': 34}  # train config blocks of 12 panel scenes
GROUPS = ('HEAD', 'BODY')
TRAIN_FRAMES = np.arange(3, 16)
EVAL_FRAMES = np.arange(11, 16)
WEIGHTS = np.array([1, 2, 4, 8, 16], dtype=np.float64)
FLOOR = dict(lo=[-8., 1.65, -8.], hi=[8., 1.80, 9.], rho=.45)
BACK = dict(lo=[-8., -3., 4.5], hi=[8., 2., 4.7], rho=.35)
_ORIGINAL = {}  # unpatched make_scenes, captured before generate() patches the module


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    for attempt in range(7):
        try:
            tmp.replace(path); return
        except PermissionError:
            if attempt == 6: raise
            time.sleep(.02*2**attempt)


def read(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ scenes ---
def area_bin(log_area):
    return int(np.clip(np.searchsorted(EDGES, log_area, side='right')-1, 0, 4))


def support_log_area(u, bins):
    """Map a shared uniform draw onto the union of equal-width support bins."""
    k = len(bins); i = min(int(u*k), k-1); frac = u*k-i
    return EDGES[bins[i]]+frac*(EDGES[bins[i]+1]-EDGES[bins[i]])


def panel_size(log_area, log_aspect):
    """Height H (floor-standing) and forward length L with H*L = area exactly."""
    area = 10**log_area; r = 10**log_aspect
    h = float(np.clip(np.sqrt(area*r), .30, 1.95)); l = area/h
    if l > 3.6: l = 3.6; h = area/l
    if l < .10: l = .10; h = area/l
    assert .30-1e-9 <= h <= 1.95+1e-9 and .10-1e-9 <= l <= 3.6+1e-9 and abs(h*l-area) < 1e-9*area
    return h, l


def panel_scene(unit, config, ref, target, shape, log_area):
    import cnh_proposal_attribution_scenes as S
    from cnh_corridor_labels import labels_for_all
    h, l = panel_size(log_area, shape['log_aspect'])
    x0 = .30+shape['gap']; x1 = x0+shape['thick']
    if shape['same_side']*target['side'] < 0: x0, x1 = -x1, -x0
    zc = target['z']+shape['z_offset']; z0 = max(.65, zc-l/2); z0 = min(z0, 4.4-l)
    panel = dict(lo=[x0, 1.65-h, z0], hi=[x1, 1.65, z0+l], rho=shape['rho'])
    alone = labels_for_all([panel, FLOOR, BACK], ref['travel'][-1:], boundary='closed')[0, [2, 3]]
    assert not alone.any(), (unit, config)
    boxes = [target['box'], panel, FLOOR, BACK]
    return dict(unit=int(unit), config=int(config), family='panel', margin=target['margin'], group=target['group'],
                boxes=boxes, poses=ref['poses'].copy(), travel=ref['travel'].copy(), head=ref['head'].copy(),
                labels=S.labels_for(boxes, ref['travel']), mode=ref['mode'], dt=.2, speed=.8,
                meta=dict(log_area=float(log_area), area=float(h*l), bin=area_bin(log_area), height=h, length=l,
                          same_side=int(shape['same_side']), gap=shape['gap'], thick=shape['thick'], rho=shape['rho']))


def draw_target(rng, margin, group):
    width = float(rng.uniform(.08, .12)); side = int(rng.choice([-1, 1]))
    z = float(rng.uniform(1.15, 1.85)); thick = float(rng.uniform(.06, .18))
    yy = (-.10, .26) if group == 0 else (.50, .84)
    inside = .30+margin
    xx = (inside, inside+width) if side == 1 else (-inside-width, -inside)
    return dict(box=dict(lo=[xx[0], yy[0], z], hi=[xx[1], yy[1], z+thick], rho=float(rng.uniform(.22, .65))),
                margin=float(margin), group=int(group), side=side, z=z)


def draw_shape(rng):
    return dict(same_side=int(rng.choice([-1, 1])), gap=float(rng.uniform(.08, .30)), thick=float(rng.uniform(.05, .50)),
                log_aspect=float(rng.uniform(np.log10(.3), np.log10(3.))), z_offset=float(rng.uniform(-.4, .4)),
                rho=float(rng.uniform(.45, .65)))


def scenes_for(unit):
    """Train: 10 shared known scenes + 3 arms x 12 paired panels. Calib/eval: 20 stratified panels."""
    import cnh_proposal_attribution_scenes as S
    known = _ORIGINAL.get('make_scenes', S.make_scenes)(unit); ref = known[0]
    rng = np.random.default_rng([2026092902, int(unit)])
    if unit in SPLITS['train']:
        shared = [dict(s, config=i) for i, s in enumerate(known[:6]+known[18:22])]
        assert [s['family'] for s in shared] == ['boundary']*6+['general']*4
        slots = []
        for j in range(12):
            m = S.MARGINS[j % 6]+rng.uniform(-.006, .006)
            slots.append((draw_target(rng, m, (unit+j) % 2), draw_shape(rng), float(rng.uniform())))
        panels = [panel_scene(unit, ARM_OFFSET[arm]+j, ref, t, sh, support_log_area(u, ARMS[arm]))
                  for arm in ARMS for j, (t, sh, u) in enumerate(slots)]
        return shared+panels
    scenes = []
    for s in range(20):
        b = s//4; m = S.MARGINS[(unit+s) % 6]+rng.uniform(-.006, .006)
        t = draw_target(rng, m, (unit//6+s) % 2); sh = draw_shape(rng)
        la = EDGES[b]+float(rng.uniform())*(EDGES[b+1]-EDGES[b])
        scenes.append(panel_scene(unit, s, ref, t, sh, la))
    return scenes


def arm_of(unit, config):
    if unit not in SPLITS['train'] or config < 10: return 'shared'
    return next(a for a, o in ARM_OFFSET.items() if o <= config < o+12)


def manifest():
    """Scene metadata only (no rendering); used for smoke checks and the plan."""
    rows = []
    for split, units in SPLITS.items():
        for u in units:
            for s in scenes_for(u):
                rows.append(dict(split=split, unit=u, config=s['config'], family=s['family'], arm=arm_of(u, s['config']),
                                 margin=s['margin'], group=s['group'], label_final=s['labels'][-1].tolist(), **s.get('meta', {})))
    save(OUT/'scene_manifest.json', rows)
    summary = {}
    for split in SPLITS:
        for arm in ('shared', *ARMS):
            rr = [r for r in rows if r['split'] == split and r['arm'] == arm]
            if not rr: continue
            panels = [r for r in rr if r['family'] == 'panel']
            summary[f'{split}|{arm}'] = dict(scenes=len(rr), families=sorted({r['family'] for r in rr}),
                bins=np.bincount([r['bin'] for r in panels], minlength=5).tolist() if panels else None,
                area_range=[min(r['area'] for r in panels), max(r['area'] for r in panels)] if panels else None,
                same_side=int(sum(r['same_side'] > 0 for r in panels)),
                final_positive_rows=int(sum(sum(r['label_final']) for r in rr)))  # central HEAD/BODY labels
    save(OUT/'scene_summary.json', summary); print(json.dumps(summary, indent=1))


# -------------------------------------------------------------- pipeline ---
def generate(split):
    assert PLAN.exists(), 'freeze plan first'
    assert not USED & set(sum(SPLITS.values(), []))
    import cnh_proposal_attribution_scenes as S
    import cnh_corridor_diagnostic as D
    shared_save = D.old.save

    def split_save(path, obj):
        # Parallel split processes must not replace the same progress/terminal temp file (Windows lock).
        path = Path(path)
        if path.name in ('generation_progress.json', 'generation_terminal.json'):
            return save(path.with_name(path.stem+f'_{split}.json'), obj)
        return shared_save(path, obj)
    D.old.save = split_save
    _ORIGINAL['make_scenes'] = S.make_scenes
    S.make_scenes = scenes_for  # D.generate imports make_scenes from S at call time
    D.OUT = OUT; D.SPLITS = {split: SPLITS[split]}
    D.generate()


def materialize(split):
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    import cnh_proposal_attribution_scenes as S
    torch.set_num_threads(2)
    projector = BatchedProjector(); frames = TRAIN_FRAMES if split == 'train' else EVAL_FRAMES
    folder = OUT/'data'/split
    if (folder/'metadata.npz').exists(): return
    folder.mkdir(parents=True, exist_ok=True)
    units = SPLITS[split]; per_unit = 46 if split == 'train' else 20
    n = len(units)*per_unit*len(frames)
    feats = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype=np.float16, shape=(n, 3, *SHAPE))
    meta = {k: [] for k in ['unit', 'config', 'frame', 'family', 'labels']}
    offset = 0; start = time.time(); term = OUT/f'terminal_generate_{split}.json'
    for i, u in enumerate(units):
        # Generation writes units sequentially; unit i is complete once unit i+1 exists or generation finished.
        nxt = OUT/'features'/split/f'unit{units[i+1]}.npz' if i+1 < len(units) else None
        while not ((nxt is not None and nxt.exists()) or (term.exists() and '"complete"' in term.read_text())):
            if term.exists() and '"failed"' in term.read_text(): raise RuntimeError(f'generation of {split} failed')
            time.sleep(5)
        d = read(OUT/'features'/split/f'unit{u}.npz'); ref = S.make_scenes(u)[0]
        assert len(d['family']) == per_unit
        for config in range(len(d['family'])):
            ids = np.flatnonzero(d['scene'] == config)
            assert np.array_equal(d['frame'][ids], np.arange(16))
            sensor, travel, noisy = motion_metadata(u, config)
            assert np.allclose(sensor, ref['poses'], atol=1e-9) and np.allclose(travel, ref['travel'], atol=1e-9)
            for f in frames:
                voxel = projector.sequence(d['z1'][ids[max(0, f-7):f+1]], relative_transforms(sensor, travel, noisy, int(f)))
                assert bool(torch.isfinite(voxel).all())
                feats[offset] = voxel.cpu().numpy().astype(np.float16); offset += 1
            meta['unit'] += [u]*len(frames); meta['config'] += [config]*len(frames)
            meta['frame'] += frames.tolist(); meta['family'] += [str(d['family'][config])]*len(frames)
            meta['labels'] += d['labels'][ids[frames], 2:4].tolist()
        print('materialize', split, u, round(time.time()-start, 1), flush=True)
    assert offset == n
    feats.flush(); del feats
    np.savez_compressed(folder/'metadata.npz', **{k: np.asarray(v) for k, v in meta.items()})


def prep(torch, x, masks):
    x = torch.as_tensor(np.array(x, dtype=np.float32), device='cuda')
    x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p(); x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p(); x[:, 1] /= 8
    x = torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)
    assert torch.isfinite(x).all()
    return x


def train():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    assert PLAN.exists()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    x = np.load(OUT/'data/train/features.npy', mmap_mode='r'); m = read(OUT/'data/train/metadata.npz')
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    arm = np.array([arm_of(int(u), int(c)) for u, c in zip(m['unit'], m['config'])])
    labels = m['labels'].astype(np.float32)
    for name in ARMS:
        ids = np.flatnonzero((arm == 'shared') | (arm == name)); assert len(ids) == 96*22*13
        for seed in range(3):
            final = OUT/'models'/name/f'model_seed{seed}.pt'
            if final.exists(): continue
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
            net = CVR().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); history = []
            for epoch in range(20):
                start = time.monotonic(); net.train(); order = rng.permutation(ids); total = 0.
                for c in range(0, len(order), 256):
                    block = order[c:c+256]; opt.zero_grad(set_to_none=True)
                    for i in range(0, len(block), 64):
                        sub = np.sort(block[i:i+64])
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(prep(torch, x[sub], masks)), torch.as_tensor(labels[sub], device='cuda'))
                        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                        (loss*(len(sub)/len(block))).backward(); total += float(loss.detach())*len(sub)
                    opt.step()
                sched.step(); history.append(dict(epoch=epoch+1, loss=total/len(order), elapsed_s=time.monotonic()-start))
                print('train', name, seed, history[-1], flush=True)
            final.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, final)
            save(final.parent/f'seed{seed}_history.json', history)
            save(final.parent/f'seed{seed}_receipt.json', dict(arm=name, seed=seed, samples=int(len(ids)), sha256=sha(final), final_loss=history[-1]['loss']))
            del net, opt; torch.cuda.empty_cache()


def infer():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    dest = OUT/'predictions'; dest.mkdir(parents=True, exist_ok=True)
    sources = {'calib': OUT/'data/calib', 'evaluation': OUT/'data/evaluation', 'novel_evaluation': NOVEL/'data/evaluation'}
    for name in ARMS:
        path = dest/f'{name}.npz'
        if path.exists(): continue
        nets = []
        for s in range(3):
            net = CVR().cuda(); net.load_state_dict(torch.load(OUT/'models'/name/f'model_seed{s}.pt', map_location='cuda', weights_only=True)); nets.append(net.eval())
        out = {}
        for src, folder in sources.items():
            x = np.load(folder/'features.npy', mmap_mode='r'); m = read(folder/'metadata.npz'); raw = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    raw.append(torch.stack([n(prep(torch, x[i:i+64], masks)) for n in nets]).mean(0).cpu().numpy())
            raw = np.concatenate(raw)
            for u in np.unique(m['unit']):
                sel = np.flatnonzero(m['unit'] == u); scores = []
                for c in range(int(m['config'][sel].max())+1):
                    loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
                    assert np.array_equal(m['frame'][loc], EVAL_FRAMES)
                    scores.append((raw[loc].astype(np.float64)*WEIGHTS[:, None]).sum(0)/WEIGHTS.sum())
                out[f'{src}|{int(u)}'] = np.asarray(scores)
            print('infer', name, src, flush=True)
        np.savez_compressed(path, **out); del nets; torch.cuda.empty_cache()
    save(OUT/'model_hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted((OUT/'models').rglob('model_seed*.pt'))})


# --------------------------------------------------------------- evaluate ---
def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); r = rankdata(np.asarray(s, np.float64)); p = int(y.sum()); n = len(y)-p
    return float((r[y].sum()-p*(p+1)/2)/(p*n))


def family_auc_delta(rows_by_group, arm, base, draws=2000, seed=20260929):
    """Unit-cluster bootstrap of mean(HEAD,BODY) AUC(arm) - AUC(base)."""
    units = np.unique(rows_by_group['HEAD']['unit'])
    idx = {g: {u: np.flatnonzero(rows_by_group[g]['unit'] == u) for u in units} for g in GROUPS}

    def stat(pick):
        vals = []
        for g in GROUPS:
            r = rows_by_group[g]; ii = np.concatenate([idx[g][u] for u in pick])
            vals.append(auc(r['label'][ii], r[arm][ii])-auc(r['label'][ii], r[base][ii]))
        return float(np.mean(vals))
    rng = np.random.default_rng(seed); boot = []
    for _ in range(draws):
        pick = rng.choice(units, len(units))
        try: boot.append(stat(pick))
        except ZeroDivisionError: pass
    return dict(point=stat(units), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], draws=len(boot))


def evaluate():
    from cnh_corridor_late_fusion import metrics, ranks, threshold
    assert PLAN.exists() and not (OUT/'results.json').exists(), 'refusing to overwrite consumed evaluation'
    pred = {a: read(OUT/'predictions'/f'{a}.npz') for a in ARMS}
    man = {(r['unit'], r['config']): r for r in json.loads((OUT/'scene_manifest.json').read_text())}

    def rows(split):
        out = []
        for u in SPLITS[split]:
            d = read(OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
            for c in range(k):
                meta = man[(u, c)]; assert abs(meta['margin']-float(d['margin'][c])) < 1e-12
                for q, g in enumerate(GROUPS):
                    out.append(dict(unit=u, config=c, group=g, bin=meta['bin'], area=meta['area'], label=int(labels[c, q]),
                                    margin=float(d['margin'][c]), target_group=int(d['group'][c]),
                                    scores={**{a: float(pred[a][f'{split}|{u}'][c, q]) for a in ARMS}, 'PARTIAL': float(d['motion'][c, 2, q])}))
        return out
    cal, ev = rows('calib'), rows('evaluation')
    arms = (*ARMS, 'PARTIAL'); table = []; ledger = []
    for g in GROUPS:
        e = [r for r in ev if r['group'] == g]
        pr = {}
        for a in arms:
            support = ARMS.get(a, (0, 1, 2, 3, 4))
            ref = [r['scores'][a] for r in cal if r['group'] == g and r['label'] == 0 and r['bin'] in support]
            k = threshold(ranks(ref, ref), len(ref)); pr[a] = ranks(ref, [r['scores'][a] for r in e]) >= k
        for i, r in enumerate(e):
            r['prediction'] = {a: int(pr[a][i]) for a in arms}; ledger.append(r)
        for b in range(5):
            idx = [i for i, r in enumerate(e) if r['bin'] == b]; y = np.array([e[i]['label'] for i in idx])
            table.append(dict(group=g, bin=b, **{a: dict(metrics(y, pr[a][idx]), auc=auc(y, [e[i]['scores'][a] for i in idx])) for a in arms}))
    fam = {b: {a: float(np.mean([t[a]['auc'] for t in table if t['bin'] == b])) for a in arms} for b in range(5)}
    by = {}
    for b in range(5):
        by[b] = {}
        for g in GROUPS:
            rr = [r for r in ev if r['group'] == g and r['bin'] == b]
            by[b][g] = dict(unit=np.array([r['unit'] for r in rr]), label=np.array([r['label'] for r in rr]),
                            **{a: np.array([r['scores'][a] for r in rr]) for a in arms})
    deltas = {f'{a}|B{b}': family_auc_delta(by[b], a, 'FULL') for a in ('EXTRAP', 'INTERP') for b in range(5)}
    di = [deltas['INTERP|B2']['point'], deltas['INTERP|B3']['point']]; de = deltas['EXTRAP|B4']['point']
    interp_ok = min(di) >= -.03; interp_fail = min(di) <= -.10
    extrap_ok = de >= -.03; extrap_fail = de <= -.10
    if interp_fail: reading = 'TEMPLATE_INTERPOLATION_FAILS'
    elif extrap_ok: reading = 'AREA_AXIS_EXTRAPOLATES'
    elif interp_ok and extrap_fail: reading = 'LOCAL_GEOMETRY_INTERPOLATES_NOT_EXTRAPOLATES'
    else: reading = 'INTERMEDIATE'
    # Descriptive transfer to the consumed hand-designed families (not in the reading).
    transfer = {}
    for u_key in [k for k in pred['FULL'] if k.startswith('novel_evaluation|')]:
        u = int(u_key.split('|')[1]); d = read(NOVEL/'features/evaluation'/f'unit{u}.npz'); k = len(d['family'])
        labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
        for c in range(k):
            for q, g in enumerate(GROUPS):
                t = transfer.setdefault((str(d['family'][c]), g), dict(y=[], **{a: [] for a in arms}))
                t['y'].append(int(labels[c, q])); t['PARTIAL'].append(float(d['motion'][c, 2, q]))
                for a in ARMS: t[a].append(float(pred[a][u_key][c, q]))
    fams = sorted({f for f, _ in transfer})
    transfer_auc = {f: {a: float(np.mean([auc(transfer[(f, g)]['y'], transfer[(f, g)][a]) for g in GROUPS])) for a in arms} for f in fams}
    save(OUT/'results.json', dict(scope='Development pilot on fresh units 80000-82095; transfer rows are consumed descriptive',
                                  plan_sha256=sha(PLAN), script_sha256=sha(__file__), reading=reading, bin_edges_log10=EDGES.tolist(),
                                  family_auc_by_bin=fam, deltas_vs_full=deltas, table=table, transfer_auc_consumed_novel=transfer_auc))
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger), encoding='utf8')
    print('reading', reading)
    for b in range(5):
        print(f'B{b} ' + ' '.join(f'{a}={v:.3f}' for a, v in fam[b].items()) + '  ' +
              ' '.join(f"{a}-FULL={deltas[f'{a}|B{b}']['point']:+.3f}{np.round(deltas[f'{a}|B{b}']['ci'], 3).tolist()}" for a in ('EXTRAP', 'INTERP')))
    for f, v in transfer_auc.items():
        print(f'  {f:14s} ' + ' '.join(f'{a}={x:.3f}' for a, x in v.items()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['manifest', 'generate', 'materialize', 'train', 'infer', 'evaluate'], required=True)
    p.add_argument('--split', choices=list(SPLITS))
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.stage+(f'_{a.split}' if a.split else '')
    try:
        fn = {'manifest': manifest, 'generate': generate, 'materialize': materialize, 'train': train, 'infer': infer, 'evaluate': evaluate}[a.stage]
        fn(a.split) if a.stage in ('generate', 'materialize') else fn()
        save(OUT/f'terminal_{tag}.json', dict(status='complete'))
    except BaseException as e:
        save(OUT/f'terminal_{tag}.json', dict(status='failed', error=repr(e)))
        raise
