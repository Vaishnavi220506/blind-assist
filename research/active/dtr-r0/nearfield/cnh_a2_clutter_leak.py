"""Does frozen A2 leak out-of-box clutter into HEAD scores? (EXPLORE fast lane, Development)

Input-swap test on the v5 clutter diagnostic (K0 clean vs K10). For each audit unit and HEAD
query q, rebuild the frozen learned-readout features for K0 and K10 and score hybrids:
  IN  = K10 features only on q's support voxels (weight >= .75), K0 elsewhere
  OUT = K10 features only off q's support, K0 on it
  RING_d = K10 on support plus zones within Chebyshev distance d of the support footprint
A2 = frozen causal EWMA (alpha .5, window 5) of the 3-seed mean NN logit, applied per config
to each hybrid sequence. Threshold = A2 HEAD clean-calib nominal budget-2 threshold from the
clutter results. Frozen reading rules: artifacts.local/work/cnh-a2-leak-20260929/DECISIONS.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
V5 = ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data'
CLUTTER = ROOT/'artifacts.local/work/cnh-v5-clutter-20260928'
MODELS = [ROOT/f'artifacts.local/work/cnh-learned-readout-20260928/seeds/model_seed{i}.pt' for i in range(3)]
HEAD_Q = (0, 2, 4)
RINGS = (1, 2, 3)
FRAMES = range(3, 12)


def setup():
    sys.path.insert(0, str(V5/'source'))
    sys.path.append(str(HERE))
    import cnh_track_a_scale_evaluate as se
    se.sensor_module.FAMILY = json.loads((V5/'request.json').read_text())['family']
    return se


def features(se, folder, u, bias):
    import cnh_learned_features as F
    import cnh_learned_readout as L
    from cnh_track_a_readout import noisy_poses
    _, records, _ = se.unit_records(folder/'geometry', folder/'sensor', u, -10, 1)
    xs, sups, ys, cfg, frm = [], [], [], [], []
    for rec in records:
        noisy = noisy_poses(rec['poses'], rec['ego_seed'], dt=.2)
        z4, z1, sup = F.sequence_features(rec['hist'], rec['ambient'], bias, rec['tq'], noisy)
        xs.append(np.stack([L.squash(z4.astype(np.float16).astype(np.float32)),
                            L.squash(z1.astype(np.float16).astype(np.float32))], 1))
        sups.append(sup.astype(bool)); ys.append(rec['labels']); cfg.append(np.full(len(z4), rec['config'])); frm.append(np.arange(len(z4)))
    return np.concatenate(xs), np.concatenate(sups), np.concatenate(ys), np.concatenate(cfg), np.concatenate(frm)


def score(nets, x, sup):
    import torch
    import cnh_learned_readout as L
    out = []
    with torch.no_grad():
        for net in nets:
            parts = []
            for i in range(0, len(x), 256):
                xb = torch.as_tensor(x[i:i+256], device=L.DEV)
                sb = torch.as_tensor(sup[i:i+256], device=L.DEV)
                parts.append(net(xb, sb).float().cpu().numpy())
            out.append(np.concatenate(parts))
    return np.mean(out, 0)


def zone_distance(sup_q):
    """[n,8,8] Chebyshev zone distance to the support footprint (any bin); 0 on the footprint."""
    foot = sup_q.any(-1)
    yy, xx = np.mgrid[0:8, 0:8]
    out = np.full(foot.shape, 99, int)
    for i in range(len(foot)):
        ys, xs = np.nonzero(foot[i])
        if len(ys):
            out[i] = np.max(np.abs(np.stack([yy[..., None]-ys, xx[..., None]-xs])), 0).min(-1)
    return out


def unit_job(u, nets, bias, se):
    import cnh_learned_memory_fusion as MF
    k0 = features(se, V5, u, bias)
    k10 = features(se, CLUTTER/'K10', u, bias)
    x0, s0, y0, c0, f0 = k0
    x1, s1, y1, c1, f1 = k10
    assert np.array_equal(y0, y1) and np.array_equal(c0, c1) and np.array_equal(f0, f1) and np.array_equal(s0, s1)
    stored0 = np.load(V5/'predictions'/f'unit{u:03d}.npz')['NN'] if (V5/'predictions'/f'unit{u:03d}.npz').exists() else None
    stored1 = np.load(CLUTTER/'K10'/'predictions'/f'unit{u:03d}.npz')['NN']
    nn0, nn1 = score(nets, x0, s0), score(nets, x1, s1)
    ident = dict(k10_max_abs=float(np.abs(nn1-stored1).max()),
                 k0_max_abs=None if stored0 is None else float(np.abs(nn0-stored0).max()))
    ew = lambda s: MF.causal_ewma(s, c0, f0, alpha=.5, window=5)
    res = dict(unit=u, identity=ident, A2_K0=ew(nn0)[:, HEAD_Q], A2_K10=ew(nn1)[:, HEAD_Q], y=y0[:, HEAD_Q], frame=f0)
    for j, q in enumerate(HEAD_Q):
        sq = s0[:, q]                                        # [n,8,8,16]
        dist = zone_distance(sq)
        variants = {'IN': sq, 'OUT': ~sq}
        for d in RINGS:
            variants[f'RING{d}'] = sq | np.broadcast_to((dist <= d)[..., None], sq.shape)
        for name, mask in variants.items():
            m = np.broadcast_to(mask[:, None], x0.shape)
            xh = np.where(m, x1, x0)
            res.setdefault(name, np.zeros((len(x0), len(HEAD_Q))))[:, j] = ew(score(nets, xh, s0))[:, q]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'artifacts.local/work/cnh-a2-leak-20260929')
    ap.add_argument('--units', type=int, nargs='*')
    a = ap.parse_args()
    se = setup()
    import torch
    import cnh_learned_readout as L
    nets = []
    for m in MODELS:
        net = L.Readout().to(L.DEV)
        net.load_state_dict(torch.load(m, map_location=L.DEV, weights_only=True))
        net.eval()
        nets.append(net)
    bias = np.load(CLUTTER/'K10'/'predictions'/'bias.npy')
    results = json.loads((CLUTTER/'results.json').read_text())
    thr = results['doses']['10']['HEAD']['nominal']['A2']['2.0']['threshold']
    audit = a.units or [int(u) for u in results['split_units']['audit']]
    (a.out/'raw').mkdir(parents=True, exist_ok=True)
    for u in audit:
        f = a.out/'raw'/f'unit{u:03d}.npz'
        if f.exists():
            continue
        r = unit_job(u, nets, bias, se)
        np.savez_compressed(f, **{k: (json.dumps(v) if k == 'identity' else v) for k, v in r.items()})
        print(u, r['identity'], flush=True)
    summarize(a.out, thr)


def summarize(out, thr):
    keys = ('A2_K0', 'A2_K10', 'IN', 'OUT') + tuple(f'RING{d}' for d in RINGS)
    rows = {k: [] for k in keys + ('y', 'frame')}
    ident = []
    for f in sorted((out/'raw').glob('unit*.npz')):
        z = np.load(f)
        ident.append(json.loads(str(z['identity'])))
        for k in keys:
            rows[k].append(z[k])
        rows['y'].append(z['y'])
        rows['frame'].append(np.repeat(z['frame'][:, None], len(HEAD_Q), 1))
    R = {k: np.concatenate(v).ravel() for k, v in rows.items()}
    neg = (R['y'] == 0) & (R['frame'] >= 3) & (R['frame'] <= 11)
    al = {k: R[k] >= thr for k in keys}
    added = neg & al['A2_K10'] & ~al['A2_K0']
    removed = neg & ~al['A2_K10'] & al['A2_K0']
    rep = dict(threshold=thr, units=len(ident), negative_query_frames=int(neg.sum()),
               identity_max_abs=dict(k10=max(i['k10_max_abs'] for i in ident),
                                     k0=max((i['k0_max_abs'] or 0) for i in ident)),
               added=int(added.sum()), removed=int(removed.sum()))
    rep['added_still_alarm'] = {k: int((added & al[k]).sum()) for k in keys[2:]}
    rep['removed_still_quiet'] = {k: int((removed & ~al[k]).sum()) for k in keys[2:]}
    delta = R['A2_K10']-R['A2_K0']
    rep['mean_delta_all_negatives'] = {k: float((R[k]-R['A2_K0'])[neg].mean()) for k in ('A2_K10',)+keys[2:]}
    rep['mean_delta_added'] = {k: float((R[k]-R['A2_K0'])[added].mean()) for k in ('A2_K10',)+keys[2:]}
    rep['share_of_mean_delta_all_negatives'] = {k: float((R[k]-R['A2_K0'])[neg].mean()/delta[neg].mean()) for k in keys[2:]}
    (out/'report.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
