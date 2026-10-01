"""Is the ~10 cm near-range transition plateau caused by noisy-pose transport?

Re-voxelise the near-range evaluation units with the true sensor poses instead of the
noisy ones (everything else identical), score with the frozen NEAR 5-seed ensemble
(trained on noisy-pose voxels), and refit the psychometric bands with the same rule
(10% alarm on clear rows of the same set).
"""
import argparse
import time

import numpy as np

import cnh_structure_space as SS
import cnh_readout_fix as RF
import cnh_near_range as NR

OUT = NR.OUT
SPLIT = 'evaluation'


def materialize(chunk):
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    torch.set_num_threads(2)
    k, n = map(int, chunk.split('/')); units = NR.SPLITS[SPLIT][k::n]; frames = SS.EVAL_FRAMES
    folder = OUT/'data/evaluation_exactpose'; folder.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(folder/f'features_c{k}.npy', mode='w+', dtype=np.float16, shape=(len(units)*40*len(frames), 3, *SHAPE))
    meta = {key: [] for key in ('unit', 'config', 'frame')}; o = 0; projector = BatchedProjector(); t0 = time.time()
    for u in units:
        d = SS.read(OUT/'features'/SPLIT/f'unit{u}.npz')
        for config in range(40):
            ids = np.flatnonzero(d['scene'] == config)
            sensor, travel, _ = motion_metadata(u, config)
            for f in frames:  # true poses replace the noisy ones in the transport
                feats[o] = projector.sequence(d['z1'][ids[max(0, f-7):f+1]], relative_transforms(sensor, travel, sensor, int(f))).cpu().numpy().astype(np.float16); o += 1
            meta['unit'] += [u]*len(frames); meta['config'] += [config]*len(frames); meta['frame'] += frames.tolist()
        print('exactpose', u, round(time.time()-t0, 1), flush=True)
    feats.flush(); del feats
    np.savez_compressed(folder/f'metadata_c{k}.npz', **{key: np.asarray(v) for key, v in meta.items()})


def infer():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    parts = sorted((OUT/'data/evaluation_exactpose').glob('features_c*.npy'))
    xs = [np.load(p, mmap_mode='r') for p in parts]
    m = [SS.read(p.with_name(p.name.replace('features_', 'metadata_').replace('.npy', '.npz'))) for p in parts]
    m = {key: np.concatenate([z[key] for z in m]) for key in m[0]}
    nets = []
    for s in range(5):
        net = CVR().cuda(); net.load_state_dict(torch.load(OUT/'models/NEAR'/f'model_seed{s}.pt', map_location='cuda', weights_only=True)); nets.append(net.eval())
    raw = []
    with torch.no_grad():
        for x in xs:
            for i in range(0, len(x), 128): raw.append(torch.stack([n_(SS.prep(torch, x[i:i+128], masks)) for n_ in nets]).mean(0).cpu().numpy())
    raw = np.concatenate(raw); out = {}
    for u in NR.SPLITS[SPLIT]:
        sel = np.flatnonzero(m['unit'] == u); sc = []
        for c in range(40):
            loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
            assert np.array_equal(m['frame'][loc], SS.EVAL_FRAMES)
            sc.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None]).sum(0)/SS.WEIGHTS.sum())
        out[str(u)] = np.asarray(sc)
    np.savez_compressed(OUT/'scores_NEAR_EXACTPOSE.npz', **out)


def evaluate():
    from sklearn.linear_model import LogisticRegression
    sc = {a: SS.read(OUT/f'scores_{a}.npz') for a in ('NEAR', 'NEAR_EXACTPOSE')}; rows = []
    for u in NR.SPLITS[SPLIT]:
        d = SS.read(OUT/'features'/SPLIT/f'unit{u}.npz'); scenes = NR.scenes_for(u); lab = d['labels'].reshape(40, 16, 6)[:, -1, 2:4]
        for c in range(40):
            tq = int(d['group'][c])
            for q in (0, 1):
                rows.append(dict(unit=u, g=q, cond=str(d['family'][c]), rng=scenes[c]['meta']['range'], lab=int(lab[c, q]),
                                 off=-float(d['margin'][c]) if q == tq else None, s={a: float(sc[a][str(u)][c, q]) for a in sc}))
    thr = {(a, q): float(np.quantile([r['s'][a] for r in rows if r['g'] == q and r['off'] is None and r['lab'] == 0], .9)) for a in sc for q in (0, 1)}
    fits = {}
    for cond in ('none', 'corner'):
        for lo, hi in NR.RANGE_BINS:
            rr = [r for r in rows if r['off'] is not None and r['cond'] == cond and lo <= r['rng'] < hi]
            o = np.array([r['off'] for r in rr]); uid = np.array([r['unit'] for r in rr])
            for a in sc:
                y = np.array([r['s'][a] >= thr[(a, r['g'])] for r in rr], int)

                def fit(idx):
                    lr = LogisticRegression(C=1e6, max_iter=2000).fit(o[idx, None], y[idx]); return 1/lr.coef_[0, 0]
                g = np.random.default_rng(9); boot = []
                for _ in range(200):
                    idx = np.concatenate([np.flatnonzero(uid == p) for p in g.choice(NR.SPLITS[SPLIT], 96)])
                    boot.append(fit(idx))
                s = fit(np.arange(len(o))); fits[f'{a}|{cond}|{lo}-{hi}'] = dict(n=len(o), band_cm=2*s*np.log(9)*100,
                    band_ci_cm=[float(np.percentile(boot, q))*2*np.log(9)*100 for q in (2.5, 97.5)])
    SS.save(OUT/'results_exactpose.json', dict(scope='same evaluation units re-voxelised with true poses; frozen NEAR ensemble', fits=fits, script_sha256=SS.sha(__file__)))
    for k, v in fits.items(): print(f"{k:34s} n={v['n']} band={v['band_cm']:.1f} [{v['band_ci_cm'][0]:.1f},{v['band_ci_cm'][1]:.1f}]")


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', required=True); p.add_argument('--chunk'); a = p.parse_args()
    tag = a.stage+(f"_c{a.chunk.replace('/', 'of')}" if a.chunk else '')
    try:
        materialize(a.chunk) if a.stage == 'materialize' else {'infer': infer, 'evaluate': evaluate}[a.stage]()
        SS.save(OUT/f'terminal_exactpose_{tag}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_exactpose_{tag}.json', dict(status='failed', error=repr(e))); raise
