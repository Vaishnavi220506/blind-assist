"""Geometric-residual corridor readout pilot (fast lane, Development only).

logit_q = a_q * g(PARTIAL_q) + b_q + r_q(x). a, b are fixed by a train-fold
logistic fit of PARTIAL alone; r is the CVR v2 backbone with a zero-initialised
last layer, so r == 0 reproduces the PARTIAL ranking exactly. Sidewall-held-out
fold only; evaluation reuses consumed fresh-fusion and novel-structure units.
Plan: artifacts.local/work/cnh-residual-readout-20260929/RESIDUAL_PLAN.md
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
OUT = WORK/'cnh-residual-readout-20260929'
PLAN = OUT/'RESIDUAL_PLAN.md'
CVR2 = WORK/'cnh-cvr-v2-20260929'
RETRAIN = WORK/'cnh-corridor-retrain-20260929'
FRESH = WORK/'cnh-fresh-fusion-20260929'
NOVEL = WORK/'cnh-novel-structure-20260929'
TRAIN_UNITS = list(range(1000, 1096))
FOLD = 'sidewall'
KNOWN = ('boundary', 'mixed_surface', 'sidewall', 'general')
NOVEL_FAMILIES = ('pole', 'railing', 'low_beam', 'opposite_wall', 'bollards', 'vehicle')
SEEN = ('boundary', 'mixed_surface', 'general')
GROUPS = ('HEAD', 'BODY')
TRAIN_FRAMES = np.arange(3, 16)
EVAL_FRAMES = np.arange(11, 16)
WEIGHTS = np.array([1, 2, 4, 8, 16], dtype=np.float64)
FLOOR = -5.0  # g(P) = max(P, FLOOR); fixed in the plan from train-fold PARTIAL only
DATASETS = {
    'fresh': dict(root=FRESH, splits={'calib': list(range(40000, 40024)), 'evaluation': list(range(50000, 50048))}),
    'novel': dict(root=NOVEL, splits={'calib': list(range(60000, 60048)), 'evaluation': list(range(70000, 70096))}),
}


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


def g(p):
    return np.maximum(np.asarray(p, dtype=np.float64), FLOOR)


# ---------------------------------------------------------------- partial ---
def partial():
    """Re-render training units and score PARTIAL at every training frame."""
    import cnh_proposal_attribution as old
    from cnh_proposal_attribution_scenes import make_scenes, render
    from cnh_corridor_diagnostic import point_score
    from cnh_corridor_labels import labels_for_all
    torch, _, F, _, _, noisy_fn, _, _ = old.setup()
    bias = np.load(old.DATA/'readouts-gpu/primary-mount-10-snr6/bias.npy')
    dest = OUT/'partial_train'; dest.mkdir(parents=True, exist_ok=True)
    checks = {}

    def observe(scene, u):
        seed = 2026092900+u*1000+scene['config']*10
        obs = render(scene, seed); h, a = obs['hist'], obs['ambient']
        noisy = noisy_fn(scene['poses'], seed+7, dt=.2)
        tq = np.linalg.inv(scene['travel'])@scene['poses']
        return h, a, noisy, tq

    # Replay check: stored last-frame PARTIAL of consumed calib units must reproduce.
    diffs = []
    for u in (2000, 2001):
        d = read(RETRAIN/'features/calib'/f'unit{u}.npz')
        for scene in make_scenes(u):
            h, a, noisy, tq = observe(scene, u)
            p, _, _ = point_score(h, a, bias, noisy, tq[-1])
            diffs.append(float(np.abs(p-d['motion'][scene['config'], 2]).max()))
    checks['calib_last_frame_partial_max_abs_diff'] = max(diffs)
    assert max(diffs) <= 1e-9, f'PARTIAL replay mismatch {max(diffs)}'
    zdiff = []; start = time.time()
    for u in TRAIN_UNITS:
        path = dest/f'unit{u}.npz'
        if path.exists(): continue
        d = read(RETRAIN/'features/train'/f'unit{u}.npz')
        P = np.zeros((22, len(TRAIN_FRAMES), 2)); labels = np.zeros((22, len(TRAIN_FRAMES), 2), np.int8)
        for scene in make_scenes(u):
            c = scene['config']; ids = np.flatnonzero(d['scene'] == c)
            assert np.array_equal(d['frame'][ids], np.arange(16))
            h, a, noisy, tq = observe(scene, u)
            assert np.array_equal(labels_for_all(scene['boxes'], scene['travel']), d['labels'][ids])
            if c == 0:
                _, z1, _ = F.sequence_features(h, a, bias, tq, noisy)
                zdiff.append(float(np.abs(z1.astype(np.float16).astype(np.float32)-d['z1'][ids].astype(np.float32)).max()))
                assert zdiff[-1] == 0.0, f'z1 replay mismatch unit {u}'
            for fi, t in enumerate(TRAIN_FRAMES):
                P[c, fi], _, _ = point_score(h[:t+1], a[:t+1], bias, noisy[:t+1], tq[t])
            labels[c] = d['labels'][ids[TRAIN_FRAMES]][:, 2:4]
        np.savez_compressed(path, P=P, labels=labels, family=d['family'])
        save(OUT/'partial_progress.json', dict(unit=u, elapsed_s=time.time()-start))
        print('partial', u, round(time.time()-start, 1), flush=True)
    checks['train_config0_z1_max_abs_diff'] = max(zdiff) if zdiff else None
    prior = json.loads((OUT/'partial_checks.json').read_text()) if (OUT/'partial_checks.json').exists() else {}
    save(OUT/'partial_checks.json', {**prior, **{k: v for k, v in checks.items() if v is not None}})


def train_partial_aligned():
    """PARTIAL for CVR v2 train rows, in the cached metadata order."""
    m = read(CVR2/'data/train/metadata.npz')
    P = np.zeros((len(m['unit']), 2))
    for ui, u in enumerate(TRAIN_UNITS):
        d = read(OUT/'partial_train'/f'unit{u}.npz')
        for c in range(22):
            off = (ui*22+c)*len(TRAIN_FRAMES); sl = slice(off, off+len(TRAIN_FRAMES))
            assert (m['unit'][sl] == u).all() and (m['config'][sl] == c).all()
            assert np.array_equal(m['frame'][sl], TRAIN_FRAMES)
            assert np.array_equal(m['labels'][sl], d['labels'][c]) and m['family'][off] == d['family'][c]
            P[sl] = d['P'][c]
    return m, P


def describe():
    """Train-fold PARTIAL distribution only (used to fix FLOOR before training)."""
    m, P = train_partial_aligned(); keep = m['family'] != FOLD
    out = {}
    for q, grp in enumerate(GROUPS):
        p = P[keep, q]; y = m['labels'][keep, q]
        out[grp] = dict(n=int(len(p)), positives=int(y.sum()), sentinel_minus50=int((p <= -49.9).sum()),
                        quantiles={str(k): float(np.quantile(p[p > -49.9], k)) for k in (0, .001, .01, .05, .25, .5, .75, .95, .99, 1)})
    save(OUT/'train_partial_distribution.json', out); print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ model ---
def fit_geometry():
    from sklearn.linear_model import LogisticRegression
    m, P = train_partial_aligned(); keep = m['family'] != FOLD
    geo = {}
    for q, grp in enumerate(GROUPS):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(g(P[keep, q])[:, None], m['labels'][keep, q])
        geo[grp] = dict(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))
        assert geo[grp]['a'] > 0
    save(OUT/'geometry_fit.json', dict(floor=FLOOR, fold=FOLD, samples=int(keep.sum()), **geo))
    return geo


def residual_model(torch, geo):
    from cnh_cvr_pilot import CVR

    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = CVR()
            torch.nn.init.zeros_(self.net.head[-1].weight); torch.nn.init.zeros_(self.net.head[-1].bias)
            self.register_buffer('a', torch.tensor([geo[k]['a'] for k in GROUPS], dtype=torch.float32))
            self.register_buffer('b', torch.tensor([geo[k]['b'] for k in GROUPS], dtype=torch.float32))

        def forward(self, x, p):
            r = self.net(x)
            return self.a*p+self.b+r, r
    return Residual()


def prep(torch, x, masks):
    x = torch.as_tensor(np.array(x, dtype=np.float32), device='cuda')
    x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p(); x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p(); x[:, 1] /= 8
    x = torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)
    assert torch.isfinite(x).all()
    return x


def train():
    import torch
    from cnh_cvr_projection import query_masks
    assert PLAN.exists(), 'freeze plan first'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    geo = fit_geometry(); m, P = train_partial_aligned()
    x = np.load(CVR2/'data/train/features.npy', mmap_mode='r')
    assert x.shape[0] == len(P)
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    ids = np.flatnonzero(m['family'] != FOLD); assert len(ids) == 96*16*13
    gp = g(P).astype(np.float32); labels = m['labels'].astype(np.float32)
    for seed in range(3):
        final = OUT/'models'/f'model_seed{seed}.pt'
        if final.exists(): continue
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
        net = residual_model(torch, geo).cuda()
        opt = torch.optim.AdamW(net.net.parameters(), lr=.002, weight_decay=.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); history = []
        for epoch in range(20):
            start = time.monotonic(); net.train(); order = rng.permutation(ids); total = 0.; rabs = 0.
            for c in range(0, len(order), 256):
                block = order[c:c+256]; opt.zero_grad(set_to_none=True)
                for i in range(0, len(block), 64):
                    sub = np.sort(block[i:i+64])
                    logit, r = net(prep(torch, x[sub], masks), torch.as_tensor(gp[sub], device='cuda'))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.as_tensor(labels[sub], device='cuda'))
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                    (loss*(len(sub)/len(block))).backward()
                    total += float(loss.detach())*len(sub); rabs += float(r.detach().abs().mean())*len(sub)
                opt.step()
            sched.step()
            history.append(dict(epoch=epoch+1, loss=total/len(order), mean_abs_r=rabs/len(order), elapsed_s=time.monotonic()-start))
            print('train', seed, history[-1], flush=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, final)
        save(OUT/'models'/f'seed{seed}_history.json', history)
        save(OUT/'models'/f'seed{seed}_receipt.json', dict(seed=seed, samples=int(len(ids)), sha256=sha(final), final_loss=history[-1]['loss']))
        del net, opt; torch.cuda.empty_cache()


def infer():
    import torch
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    geo = json.loads((OUT/'geometry_fit.json').read_text())
    nets = []
    for s in range(3):
        net = residual_model(torch, geo).cuda()
        net.load_state_dict(torch.load(OUT/'models'/f'model_seed{s}.pt', map_location='cuda', weights_only=True)); nets.append(net.eval())
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    (OUT/'predictions').mkdir(parents=True, exist_ok=True)
    for name, spec in DATASETS.items():
        dest = OUT/'predictions'/f'{name}_R.npz'
        if dest.exists(): continue
        out = {}
        for split, units in spec['splits'].items():
            x = np.load(spec['root']/'data'/split/'features.npy', mmap_mode='r'); m = read(spec['root']/'data'/split/'metadata.npz')
            raw = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    b = prep(torch, x[i:i+64], masks); zero = torch.zeros(len(b), 2, device='cuda')
                    raw.append(torch.stack([n(b, zero)[1] for n in nets]).mean(0).cpu().numpy())
            raw = np.concatenate(raw)
            for u in units:
                sel = np.flatnonzero(m['unit'] == u); R = []
                for c in range(int(m['config'][sel].max())+1):
                    loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
                    assert np.array_equal(m['frame'][loc], EVAL_FRAMES)
                    R.append((raw[loc].astype(np.float64)*WEIGHTS[:, None]).sum(0)/WEIGHTS.sum())
                out[str(u)] = np.asarray(R)
            print('infer', name, split, flush=True)
        np.savez_compressed(dest, **out)
    save(OUT/'model_hashes.json', {str((OUT/'models'/f'model_seed{s}.pt').relative_to(ROOT)): sha(OUT/'models'/f'model_seed{s}.pt') for s in range(3)})


# --------------------------------------------------------------- evaluate ---
def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); r = rankdata(np.asarray(s, np.float64))
    p = int(y.sum()); n = len(y)-p
    return float((r[y].sum()-p*(p+1)/2)/(p*n))


def cluster_auc_delta(y, a, b, units, draws=2000, seed=20260929):
    """Paired unit bootstrap of AUC(b) - AUC(a); percentile 95% interval."""
    y, a, b, units = map(np.asarray, (y, a, b, units))
    uniq = np.unique(units); index = {u: np.flatnonzero(units == u) for u in uniq}
    rng = np.random.default_rng(seed); out = []
    for _ in range(draws):
        idx = np.concatenate([index[u] for u in rng.choice(uniq, len(uniq))])
        if 0 < y[idx].sum() < len(idx): out.append(auc(y[idx], b[idx])-auc(y[idx], a[idx]))
    return dict(point=auc(y, b)-auc(y, a), ci=[float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], draws=len(out))


def threshold_any(r, n):
    r = np.asarray(r, np.int64)
    for k in range(int(r.max())+2):
        if 10*int((r >= k).sum()) <= n: return k
    raise AssertionError


def evaluate():
    from cnh_corridor_late_fusion import metrics, ranks
    assert PLAN.exists() and not (OUT/'results.json').exists(), 'refusing to overwrite consumed evaluation'
    geo = json.loads((OUT/'geometry_fit.json').read_text())
    results = {}; ledger = []
    for name, spec in DATASETS.items():
        root = spec['root']; R = read(OUT/'predictions'/f'{name}_R.npz')
        if name == 'fresh':
            pred = {'LOFO': read(root/'predictions/LOFO_sidewall.npz'), 'CVR': read(root/'predictions/CVR_sidewall.npz'),
                    'EQUAL_COUNT': read(root/'predictions/EQUAL_COUNT_sidewall.npz')}
            families = KNOWN
        else:
            pred = {'LOFO': read(root/'nosidewall/A2.npz'), 'CVR': read(root/'nosidewall/CVR.npz'),
                    'CVR_ALL': read(root/'predictions/CVR.npz')}
            families = KNOWN+NOVEL_FAMILIES
        base = tuple(pred)

        def rows(split):
            out = []
            for u in spec['splits'][split]:
                d = read(root/'features'/split/f'unit{u}.npz'); k = len(d['family'])
                labels = d['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
                for c in range(k):
                    for q, grp in enumerate(GROUPS):
                        p = float(d['motion'][c, 2, q]); r = float(R[str(u)][c, q])
                        sc = {a: float(pred[a][str(u)][c, q]) for a in base}
                        sc.update(PARTIAL=p, RESID=geo[grp]['a']*float(g(p))+geo[grp]['b']+r, R=r)
                        out.append(dict(dataset=name, unit=u, config=c, group=grp, family=str(d['family'][c]),
                                        label=int(labels[c, q]), margin=float(d['margin'][c]), scores=sc))
            return out
        cal_all = [r for r in rows('calib') if r['family'] in SEEN]; ev_all = rows('evaluation')
        arms = base+('PARTIAL', 'RESID', 'FUSION', 'SUM')
        table = []
        for grp in GROUPS:
            cal = [r for r in cal_all if r['group'] == grp and r['label'] == 0]; n = len(cal)
            ev = [r for r in ev_all if r['group'] == grp]; cr, er, pr = {}, {}, {}
            for a in base+('PARTIAL', 'RESID'):
                ref = [r['scores'][a] for r in cal]; cr[a] = ranks(ref, ref); er[a] = ranks(ref, [r['scores'][a] for r in ev])
            cr['FUSION'] = np.maximum(cr['CVR'], cr['PARTIAL']); er['FUSION'] = np.maximum(er['CVR'], er['PARTIAL'])
            cr['SUM'] = cr['CVR']+cr['PARTIAL']; er['SUM'] = er['CVR']+er['PARTIAL']
            thr = {}
            for a in arms:
                thr[a] = threshold_any(cr[a], n); assert 10*int((cr[a] >= thr[a]).sum()) <= n
                pr[a] = er[a] >= thr[a]
            for i, r in enumerate(ev):
                r['calib_rank'] = {a: int(er[a][i]) for a in arms}; r['prediction'] = {a: int(pr[a][i]) for a in arms}
                r['rank_denominator'] = n; ledger.append(r)
            for fam in families:
                idx = np.array([i for i, r in enumerate(ev) if r['family'] == fam])
                y = np.array([ev[i]['label'] for i in idx]); units = np.array([ev[i]['unit'] for i in idx])
                Rv = np.array([ev[i]['scores']['R'] for i in idx])
                row = dict(family=fam, group=grp, calib_negative=n,
                           arms={a: dict(metrics(y, pr[a][idx]), auc=auc(y, er[a][idx].astype(float)), rank_threshold=int(thr[a])) for a in arms})
                row['R'] = {lab: dict(median_abs=float(np.median(np.abs(Rv[y == v]))), std=float(np.std(Rv[y == v])),
                                      mean=float(np.mean(Rv[y == v])), n=int((y == v).sum())) for lab, v in (('positive', 1), ('negative', 0))}
                row['R']['auc'] = auc(y, Rv)
                ok_p = pr['PARTIAL'][idx] == y.astype(bool); ok_r = pr['RESID'][idx] == y.astype(bool)
                row['flips_vs_partial'] = dict(partial_right_resid_wrong=int((ok_p & ~ok_r).sum()), partial_wrong_resid_right=int((~ok_p & ok_r).sum()))
                ra = {a: er[a][idx].astype(float) for a in ('PARTIAL', 'CVR', 'FUSION', 'LOFO', 'RESID')}
                row['auc_delta'] = {f'RESID-{b}': cluster_auc_delta(y, ra[b], ra['RESID'], units) for b in ('PARTIAL', 'CVR', 'FUSION')}
                table.append(row)
        fam_auc = {f: {a: float(np.mean([t['arms'][a]['auc'] for t in table if t['family'] == f])) for a in arms} for f in families}
        sw = fam_auc['sidewall']; mid = (sw['LOFO']+sw['PARTIAL'])/2
        covered_ok = {f: fam_auc[f]['RESID'] >= fam_auc[f]['CVR']-.02 for f in SEEN}
        if sw['RESID'] <= mid: reading = 'TEMPLATE'
        elif sw['RESID'] >= sw['PARTIAL']-.02: reading = 'FALLBACK' if all(covered_ok.values()) else 'COLLAPSED_TO_GEOMETRY'
        else: reading = 'PARTIAL_FALLBACK'
        rmed = lambda fs: float(np.median([abs(r['scores']['R']) for r in ev_all if r['family'] in fs]))
        results[name] = dict(reading=reading, sidewall_auc=sw, template_midpoint=mid, covered_auc_ok=covered_ok,
                             beats_max_fusion_on_sidewall=bool(sw['RESID'] > sw['FUSION']),
                             median_abs_R=dict(sidewall=rmed(('sidewall',)), seen=rmed(SEEN),
                                               **{f: rmed((f,)) for f in NOVEL_FAMILIES if name == 'novel'}),
                             family_auc=fam_auc, table=table)
    save(OUT/'results.json', dict(scope='Development diagnostic on consumed units; not a method test',
                                  plan_sha256=sha(PLAN), script_sha256=sha(__file__), geometry=geo, results=results))
    (OUT/'sample_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger), encoding='utf8')
    for name, res in results.items():
        print(name, res['reading'], 'median|R| sidewall', round(res['median_abs_R']['sidewall'], 3), 'seen', round(res['median_abs_R']['seen'], 3))
        for f, v in res['family_auc'].items():
            print(f'  {f:14s} ' + ' '.join(f'{a}={x:.3f}' for a, x in v.items()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['partial', 'describe', 'train', 'infer', 'evaluate'], required=True)
    stage = p.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        {'partial': partial, 'describe': describe, 'train': train, 'infer': infer, 'evaluate': evaluate}[stage]()
        save(OUT/f'terminal_{stage}.json', dict(status='complete'))
    except BaseException as e:
        save(OUT/f'terminal_{stage}.json', dict(status='failed', error=repr(e)))
        raise
