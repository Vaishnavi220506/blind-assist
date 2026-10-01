"""Coverage-distance sweep along the panel-area axis (fast lane, Development).

Five CVR arms differing only in the upper bound of panel-area coverage
(UB0..UB4; UB2 = EXTRAP and UB4 = FULL from the structure-space pilot, reused
frozen). New arms reuse the pilot's training units and paired panel slots.
Fresh calib/evaluation units carry 4 random + 4 near same-side ("corner")
panels per area bin. Plan: artifacts.local/work/cnh-coverage-sweep-20260930/SWEEP_PLAN.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'artifacts.local/work'
OUT = WORK/'cnh-coverage-sweep-20260930'
PILOT = SS.OUT
PLAN = OUT/'SWEEP_PLAN.md'
SPLITS = {'train': SS.SPLITS['train'], 'calib': list(range(84000, 84048)), 'evaluation': list(range(83000, 83144))}
USED = SS.USED | set(SS.SPLITS['calib']) | set(SS.SPLITS['evaluation']) | set(SS.SPLITS['train'])
TOP = {'UB0': 0, 'UB1': 1, 'UB2': 2, 'UB3': 3, 'UB4': 4}
NEW_ARMS = {'UB0': 46, 'UB1': 58, 'UB3': 70}  # train config offsets of the 12 paired panel slots
REUSED = {'UB2': 'EXTRAP', 'UB4': 'FULL'}
KINDS = ('random', 'corner')
GROUPS = SS.GROUPS
_ORIGINAL = {}


def support(arm):
    return tuple(range(TOP[arm]+1))


def slots(unit):
    """Re-draw the pilot's 12 paired train slots with the identical RNG sequence."""
    import cnh_proposal_attribution_scenes as S
    rng = np.random.default_rng([2026092902, int(unit)])
    out = []
    for j in range(12):
        m = S.MARGINS[j % 6]+rng.uniform(-.006, .006)
        out.append((SS.draw_target(rng, m, (unit+j) % 2), SS.draw_shape(rng), float(rng.uniform())))
    return out


def known(unit):
    import cnh_proposal_attribution_scenes as S
    return _ORIGINAL.get('make_scenes', S.make_scenes)(unit)


def scenes_for(unit):
    ref = known(unit)[0]
    if unit in SPLITS['train']:
        sl = slots(unit)
        return [SS.panel_scene(unit, off+j, ref, t, sh, SS.support_log_area(u, support(arm)))
                for arm, off in NEW_ARMS.items() for j, (t, sh, u) in enumerate(sl)]
    import cnh_proposal_attribution_scenes as S
    rng = np.random.default_rng([2026093001, int(unit)])
    scenes = []
    for s in range(40):
        kind = KINDS[s//20]; b = (s % 20)//4
        m = S.MARGINS[(unit+s) % 6]+rng.uniform(-.006, .006)
        t = SS.draw_target(rng, m, (unit//6+s) % 2); sh = SS.draw_shape(rng)
        if kind == 'corner': sh = dict(sh, same_side=1, gap=float(rng.uniform(.08, .15)))
        la = SS.EDGES[b]+float(rng.uniform())*(SS.EDGES[b+1]-SS.EDGES[b])
        sc = SS.panel_scene(unit, s, ref, t, sh, la); sc['meta']['kind'] = kind
        scenes.append(sc)
    return scenes


def manifest():
    import cnh_proposal_attribution_scenes as S
    assert not USED & (set(SPLITS['calib']) | set(SPLITS['evaluation']))
    rows = []
    for split, units in SPLITS.items():
        for u in units:
            sc = scenes_for(u)
            if split == 'train':
                # Paired-slot identity: rebuilt FULL panels must equal the pilot's FULL panels.
                pilot = SS.scenes_for(u)[10:22]; ref = S.make_scenes(u)[0]
                rebuilt = [SS.panel_scene(u, 10+j, ref, t, sh, SS.support_log_area(x, SS.ARMS['FULL'])) for j, (t, sh, x) in enumerate(slots(u))]
                assert all(a['boxes'] == b['boxes'] for a, b in zip(pilot, rebuilt)), u
            for s in sc:
                arm = next((a for a, o in NEW_ARMS.items() if o <= s['config'] < o+12), None) if split == 'train' else None
                rows.append(dict(split=split, unit=u, config=s['config'], arm=arm, margin=s['margin'], group=s['group'],
                                 label_final=s['labels'][-1].tolist(), **s['meta']))
    SS.save(OUT/'scene_manifest.json', rows)
    cells = {}
    for split in ('calib', 'evaluation'):
        for kind in KINDS:
            for b in range(5):
                rr = [r for r in rows if r['split'] == split and r.get('kind') == kind and r['bin'] == b]
                for q, g in enumerate(GROUPS):
                    pos = sum(r['label_final'][q] for r in rr)
                    cells[f'{split}|{kind}|B{b}|{g}'] = dict(units=len({r['unit'] for r in rr}), rows=len(rr), positives=int(pos), negatives=len(rr)-int(pos))
    tr = {a: np.bincount([r['bin'] for r in rows if r['arm'] == a], minlength=5).tolist() for a in NEW_ARMS}
    SS.save(OUT/'cell_counts.json', dict(eval_cells=cells, new_train_arm_bins=tr))
    print(json.dumps(tr)); print(json.dumps({k: v for k, v in cells.items() if k.startswith('evaluation')}, indent=0)[:3000])


def generate(split):
    assert PLAN.exists(), 'freeze plan first'
    import cnh_proposal_attribution_scenes as S
    import cnh_corridor_diagnostic as D
    shared_save = D.old.save

    def split_save(path, obj):
        path = Path(path)
        if path.name in ('generation_progress.json', 'generation_terminal.json'):
            return SS.save(path.with_name(path.stem+f'_{split}.json'), obj)
        return shared_save(path, obj)
    D.old.save = split_save
    _ORIGINAL['make_scenes'] = S.make_scenes
    S.make_scenes = scenes_for
    D.OUT = OUT; D.SPLITS = {split: SPLITS[split]}
    D.generate()


def materialize(split):
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    import cnh_proposal_attribution_scenes as S
    torch.set_num_threads(2)
    projector = BatchedProjector(); frames = SS.TRAIN_FRAMES if split == 'train' else SS.EVAL_FRAMES
    folder = OUT/'data'/split
    if (folder/'metadata.npz').exists(): return
    folder.mkdir(parents=True, exist_ok=True)
    units = SPLITS[split]; per_unit = 36 if split == 'train' else 40
    n = len(units)*per_unit*len(frames)
    feats = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype=np.float16, shape=(n, 3, *SHAPE))
    meta = {k: [] for k in ['unit', 'config', 'frame', 'family', 'labels']}
    offset = 0; start = time.time(); term = OUT/f'terminal_generate_{split}.json'
    for i, u in enumerate(units):
        nxt = OUT/'features'/split/f'unit{units[i+1]}.npz' if i+1 < len(units) else None
        while not ((nxt is not None and nxt.exists()) or (term.exists() and '"complete"' in term.read_text())):
            if term.exists() and '"failed"' in term.read_text(): raise RuntimeError(f'generation of {split} failed')
            time.sleep(5)
        d = SS.read(OUT/'features'/split/f'unit{u}.npz'); ref = S.make_scenes(u)[0]
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


def train():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    assert PLAN.exists()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    xo = np.load(PILOT/'data/train/features.npy', mmap_mode='r'); mo = SS.read(PILOT/'data/train/metadata.npz')
    xn = np.load(OUT/'data/train/features.npy', mmap_mode='r'); mn = SS.read(OUT/'data/train/metadata.npz')
    no = len(xo); masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    labels = np.concatenate([mo['labels'], mn['labels']]).astype(np.float32)
    shared = np.flatnonzero(mo['config'] < 10)
    for arm, off in NEW_ARMS.items():
        ids = np.concatenate([shared, no+np.flatnonzero((mn['config'] >= off) & (mn['config'] < off+12))])
        assert len(ids) == 96*22*13
        for seed in range(3):
            final = OUT/'models'/arm/f'model_seed{seed}.pt'
            if final.exists(): continue
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
            net = CVR().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); history = []
            for epoch in range(20):
                start = time.monotonic(); net.train(); order = rng.permutation(ids); total = 0.
                for c in range(0, len(order), 256):
                    block = order[c:c+256]; opt.zero_grad(set_to_none=True)
                    for i in range(0, len(block), 64):
                        sub = np.sort(block[i:i+64]); a = sub[sub < no]; b = sub[sub >= no]-no
                        x = np.concatenate([np.asarray(xo[a]), np.asarray(xn[b])])
                        y = torch.as_tensor(np.concatenate([labels[a], labels[no+b]]), device='cuda')
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(SS.prep(torch, x, masks)), y)
                        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                        (loss*(len(sub)/len(block))).backward(); total += float(loss.detach())*len(sub)
                    opt.step()
                sched.step(); history.append(dict(epoch=epoch+1, loss=total/len(order), elapsed_s=time.monotonic()-start))
                print('train', arm, seed, history[-1], flush=True)
            final.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, final)
            SS.save(final.parent/f'seed{seed}_history.json', history)
            SS.save(final.parent/f'seed{seed}_receipt.json', dict(arm=arm, seed=seed, samples=int(len(ids)), sha256=SS.sha(final), final_loss=history[-1]['loss']))
            del net, opt; torch.cuda.empty_cache()


def model_dir(arm):
    return PILOT/'models'/REUSED[arm] if arm in REUSED else OUT/'models'/arm


def infer():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    dest = OUT/'predictions'; dest.mkdir(parents=True, exist_ok=True)
    for arm in TOP:
        path = dest/f'{arm}.npz'
        if path.exists(): continue
        nets = []
        for s in range(3):
            net = CVR().cuda(); net.load_state_dict(torch.load(model_dir(arm)/f'model_seed{s}.pt', map_location='cuda', weights_only=True)); nets.append(net.eval())
        out = {}
        for split in ('calib', 'evaluation'):
            x = np.load(OUT/'data'/split/'features.npy', mmap_mode='r'); m = SS.read(OUT/'data'/split/'metadata.npz'); raw = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    raw.append(torch.stack([n(SS.prep(torch, x[i:i+64], masks)) for n in nets]).mean(0).cpu().numpy())
            raw = np.concatenate(raw)
            for u in np.unique(m['unit']):
                sel = np.flatnonzero(m['unit'] == u); scores = []
                for c in range(int(m['config'][sel].max())+1):
                    loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
                    assert np.array_equal(m['frame'][loc], SS.EVAL_FRAMES)
                    scores.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None]).sum(0)/SS.WEIGHTS.sum())
                out[f'{split}|{int(u)}'] = np.asarray(scores)
            print('infer', arm, split, flush=True)
        np.savez_compressed(path, **out); del nets; torch.cuda.empty_cache()
    SS.save(OUT/'model_hashes.json', {arm: {p.name: SS.sha(p) for p in sorted(model_dir(arm).glob('model_seed*.pt'))} for arm in TOP})


# --------------------------------------------------------------- evaluate ---
def evaluate():
    from cnh_corridor_late_fusion import metrics, ranks, threshold
    assert PLAN.exists() and not (OUT/'results.json').exists(), 'refusing to overwrite consumed evaluation'
    pred = {a: SS.read(OUT/'predictions'/f'{a}.npz') for a in TOP}
    man = {(r['unit'], r['config']): r for r in json.loads((OUT/'scene_manifest.json').read_text()) if r['split'] != 'train'}
    arms = tuple(TOP)

    def rows(split):
        out = []
        for u in SPLITS[split]:
            d = SS.read(OUT/'features'/split/f'unit{u}.npz'); k = len(d['family'])
            labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
            for c in range(k):
                mm = man[(u, c)]; assert abs(mm['margin']-float(d['margin'][c])) < 1e-12
                for q, g in enumerate(GROUPS):
                    out.append(dict(unit=u, config=c, group=g, kind=mm['kind'], bin=mm['bin'], area=mm['area'], label=int(labels[c, q]),
                                    scores={**{a: float(pred[a][f'{split}|{u}'][c, q]) for a in arms}, 'PARTIAL': float(d['motion'][c, 2, q])}))
        return out
    cal, ev = rows('calib'), rows('evaluation')
    # Frozen per-arm thresholds: arm and its paired geometry threshold share the arm's in-support calib negatives.
    thr = {}
    for a in arms:
        for g in GROUPS:
            ref = [r for r in cal if r['group'] == g and r['label'] == 0 and r['bin'] in support(a)]
            for s in (a, 'PARTIAL'):
                v = [r['scores'][s] for r in ref]; k = threshold(ranks(v, v), len(v))
                thr[(a, s, g)] = (np.sort(np.asarray(v, np.float64)), k, len(v))
    for r in ev:
        r['pred'] = {}
        for a in arms:
            for s, key in ((a, a), ('PARTIAL', f'PARTIAL@{a}')):
                ref, k, n = thr[(a, s, r['group'])]
                r['pred'][key] = int(np.searchsorted(ref, r['scores'][s], side='right') >= k)
    units = np.array(SPLITS['evaluation']); uindex = {u: i for i, u in enumerate(units)}
    cell = {}
    for kind in KINDS:
        for b in range(5):
            for g in GROUPS:
                rr = [r for r in ev if r['kind'] == kind and r['bin'] == b and r['group'] == g]
                cell[(kind, b, g)] = dict(u=np.array([uindex[r['unit']] for r in rr]), y=np.array([r['label'] for r in rr], bool),
                                          s={a: np.array([r['scores'][a] for r in rr]) for a in (*arms, 'PARTIAL')},
                                          p={key: np.array([r['pred'][key] for r in rr], bool) for key in rr[0]['pred']})
    from scipy.stats import rankdata

    def auc(y, s):
        r = rankdata(s); p = int(y.sum()); n = len(y)-p
        return (r[y].sum()-p*(p+1)/2)/(p*n)

    def ber(y, p):
        return .5*((y & ~p).sum()/y.sum()+(~y & p).sum()/(~y).sum())

    def stats(weights):
        """All cell statistics for one unit-resampling (integer unit weights)."""
        out = {}
        for key, c in cell.items():
            w = weights[c['u']]; keep = np.repeat(np.arange(len(w)), w)
            y = c['y'][keep]
            for a in (*arms, 'PARTIAL'):
                out[('auc', a)+key] = auc(y, c['s'][a][keep])
            for pk in c['p']:
                out[('ber', pk)+key] = ber(y, c['p'][pk][keep])
        return out
    point = stats(np.ones(len(units), int))
    rng = np.random.default_rng(20260930); boots = []
    for _ in range(2000):
        boots.append(stats(np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))))

    def fam(st, what, a, kind, b):
        return float(np.mean([st[(what, a, kind, b, g)] for g in GROUPS]))

    def interval(fn, level=.95):
        vals = np.array([fn(bt) for bt in boots]); lo = (1-level)/2*100
        return [float(np.percentile(vals, lo)), float(np.percentile(vals, 100-lo))]
    results = dict(scope='Development pilot on fresh units 83000-83143 / 84000-84047; UB2/UB4 frozen pilot models',
                   plan_sha256=SS.sha(PLAN), script_sha256=SS.sha(__file__))
    for kind in KINDS:
        rl = lambda st, a, b: fam(st, 'auc', 'UB4', kind, b)-fam(st, 'auc', a, kind, b)  # loss relative to FULL
        table = []
        for a in arms:
            for b in range(5):
                d = b-TOP[a]
                table.append(dict(arm=a, bin=b, distance=d, auc=fam(point, 'auc', a, kind, b), auc_partial=fam(point, 'auc', 'PARTIAL', kind, b),
                                  relative_loss=rl(point, a, b), relative_loss_ci=interval(lambda st: rl(st, a, b)),
                                  ber=fam(point, 'ber', a, kind, b), ber_geometry_same_calib=fam(point, 'ber', f'PARTIAL@{a}', kind, b)))
        # Analysis 1: equal-distance equivalence of relative loss across arms (held-out cells only).
        pairs = []
        for d in (1, 2, 3):
            cells = [(a, TOP[a]+d) for a in arms if TOP[a]+d <= 4]
            for i in range(len(cells)):
                for j in range(i+1, len(cells)):
                    (a1, b1), (a2, b2) = cells[i], cells[j]
                    diff = lambda st: rl(st, a1, b1)-rl(st, a2, b2)
                    ci = interval(diff)
                    verdict = 'EQUIVALENT' if -.03 <= ci[0] and ci[1] <= .03 else ('DIFFERENT' if ci[0] > .03 or ci[1] < -.03 else 'UNCERTAIN')
                    pairs.append(dict(distance=d, cell_a=f'{a1}@B{b1}', cell_b=f'{a2}@B{b2}', diff=diff(point), ci95=ci, verdict=verdict))
        vs = [p['verdict'] for p in pairs]
        alignment = 'ALIGNED' if all(v == 'EQUIVALENT' for v in vs) else ('NOT_ALIGNED' if 'DIFFERENT' in vs else 'UNCERTAIN')
        # Analysis 2: geometry vs arm BER at the arm's frozen operating point, held-out cells, Bonferroni over 20 cells.
        geo = []
        for a in arms:
            for b in range(TOP[a]+1, 5):
                delta = lambda st: fam(st, 'ber', f'PARTIAL@{a}', kind, b)-fam(st, 'ber', a, kind, b)
                ci_adj = interval(delta, 1-.05/20); ci95 = interval(delta)
                verdict = 'GEOMETRY_BETTER' if ci_adj[1] < 0 else ('LEARNED_BETTER' if ci_adj[0] > 0 else 'NO_CLEAR_DIFFERENCE')
                m_arm = {g: dict(fn=int((cell[(kind, b, g)]['y'] & ~cell[(kind, b, g)]['p'][a]).sum()), fp=int((~cell[(kind, b, g)]['y'] & cell[(kind, b, g)]['p'][a]).sum()),
                                 P=int(cell[(kind, b, g)]['y'].sum()), N=int((~cell[(kind, b, g)]['y']).sum())) for g in GROUPS}
                m_geo = {g: dict(fn=int((cell[(kind, b, g)]['y'] & ~cell[(kind, b, g)]['p'][f'PARTIAL@{a}']).sum()), fp=int((~cell[(kind, b, g)]['y'] & cell[(kind, b, g)]['p'][f'PARTIAL@{a}']).sum())) for g in GROUPS}
                geo.append(dict(arm=a, bin=b, distance=b-TOP[a], ber_geo_minus_arm=delta(point), ci95=ci95, ci_bonferroni=ci_adj, verdict=verdict, arm_counts=m_arm, geo_counts=m_geo))
        results[kind] = dict(table=table, equal_distance_pairs=pairs, alignment=alignment, geometry_vs_arm=geo)
    SS.save(OUT/'results.json', results)
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps({k: v for k, v in r.items()})+'\n' for r in ev), encoding='utf8')
    for kind in KINDS:
        res = results[kind]; print('==', kind, 'alignment', res['alignment'])
        for t in res['table']:
            print(f"  {t['arm']} B{t['bin']} d={t['distance']:+d} AUC={t['auc']:.3f} geo={t['auc_partial']:.3f} RL={t['relative_loss']:+.3f}{np.round(t['relative_loss_ci'], 3).tolist()} BER={t['ber']*100:.1f} geoBER={t['ber_geometry_same_calib']*100:.1f}")
        for p in res['equal_distance_pairs']:
            print(f"  d={p['distance']} {p['cell_a']} vs {p['cell_b']}: {p['diff']:+.3f} {np.round(p['ci95'], 3).tolist()} {p['verdict']}")
        for q in res['geometry_vs_arm']:
            print(f"  geo-arm {q['arm']} B{q['bin']} d={q['distance']}: {q['ber_geo_minus_arm']*100:+.1f}pp adj{np.round(np.array(q['ci_bonferroni'])*100, 1).tolist()} {q['verdict']}")


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
        SS.save(OUT/f'terminal_{tag}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_{tag}.json', dict(status='failed', error=repr(e)))
        raise
