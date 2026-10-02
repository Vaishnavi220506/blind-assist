"""Graded (range-gated) alert vs single threshold on approach sequences, no new rendering.

Reuses the near-range evaluation units (94000-94095): each 16-frame sequence is an
approach (range at frame t = final range + 0.16*(15-t)). Frames 3-10 are voxelised here
(11-15 exist). The frozen NEAR 5-seed ensemble scores every frame; deployed smoothing
(weights 1,2,4,8,16 over the last <=5 frames). An observation-only range estimate is the
depth of the strongest accumulated in-corridor evidence. Odd units calibrate thresholds,
even units evaluate. Policies at <=1 false stop per walking minute on clear rows:
  A single: stop when score >= T.
  B graded: caution when score >= Tc; stop when score >= Ts and estimated range <= 1.5 m.
"""
import argparse
import json
import time

import numpy as np

import cnh_structure_space as SS
import cnh_near_range as NR

OUT = NR.OUT
EARLY = np.arange(3, 11)
ALL = np.arange(3, 16)
STEP = .16           # metres walked per frame (0.8 m/s, 0.2 s)
SAFE = .9            # a stop counts as timely when issued at range >= 0.9 m
CONFIRM = 1.5        # policy B issues stops only when the estimated range is within this
BUDGET = 1.0         # false stops per walking minute on clear rows
FRAME_S = .2


def materialize(chunk):
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    torch.set_num_threads(2)
    k, n = map(int, chunk.split('/')); units = NR.SPLITS['evaluation'][k::n]
    folder = OUT/'data/evaluation_early'; folder.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(folder/f'features_c{k}.npy', mode='w+', dtype=np.float16, shape=(len(units)*40*len(EARLY), 3, *SHAPE))
    meta = {key: [] for key in ('unit', 'config', 'frame')}; o = 0; projector = BatchedProjector(); t0 = time.time()
    for u in units:
        d = SS.read(OUT/'features/evaluation'/f'unit{u}.npz')
        for c in range(40):
            ids = np.flatnonzero(d['scene'] == c); sensor, travel, noisy = motion_metadata(u, c)
            for f in EARLY:
                feats[o] = projector.sequence(d['z1'][ids[max(0, f-7):f+1]], relative_transforms(sensor, travel, noisy, int(f))).cpu().numpy().astype(np.float16); o += 1
            meta['unit'] += [u]*len(EARLY); meta['config'] += [c]*len(EARLY); meta['frame'] += EARLY.tolist()
        print('early', u, round(time.time()-t0, 1), flush=True)
    feats.flush(); del feats
    np.savez_compressed(folder/f'metadata_c{k}.npz', **{key: np.asarray(v) for key, v in meta.items()})


def _parts(name):
    parts = sorted((OUT/'data'/name).glob('features_c*.npy'))
    ms = [SS.read(p.with_name(p.name.replace('features_', 'metadata_').replace('.npy', '.npz'))) for p in parts]
    return [np.load(p, mmap_mode='r') for p in parts], {key: np.concatenate([m[key] for m in ms]) for key in ('unit', 'config', 'frame')}


def infer():
    """Per-frame 5-seed mean logits and observation-only range estimates, array [unit, config, frame(3..15), query]."""
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    nets = []
    for s in range(5):
        net = CVR().cuda(); net.load_state_dict(torch.load(OUT/'models/NEAR'/f'model_seed{s}.pt', map_location='cuda', weights_only=True)); nets.append(net.eval())
    zc = (torch.arange(masks.shape[-1], device='cuda', dtype=torch.float32)+.5)*.1  # voxel depth centres (m)
    U = {u: i for i, u in enumerate(NR.SPLITS['evaluation'])}; F = {int(f): i for i, f in enumerate(ALL)}
    logit = np.full((96, 40, len(ALL), 2), np.nan, np.float32); rng_est = np.full_like(logit, np.nan)
    for name in ('evaluation_early', 'evaluation'):
        xs, m = _parts(name); o = 0
        with torch.no_grad():
            for x in xs:
                for i in range(0, len(x), 128):
                    b = SS.prep(torch, x[i:i+128], masks)
                    lg = torch.stack([n_(b) for n_ in nets]).mean(0)
                    ev = b[:, 0:1]*b[:, 3:5]                                        # accumulated evidence inside each query mask
                    prof = ev.amax(dim=(2, 3))                                      # [N,2,Z]
                    r = zc[prof.argmax(-1)]
                    sl = slice(o+i, o+i+len(b))
                    for j, (u, c, f) in enumerate(zip(m['unit'][sl], m['config'][sl], m['frame'][sl])):
                        logit[U[int(u)], int(c), F[int(f)]] = lg[j].cpu().numpy(); rng_est[U[int(u)], int(c), F[int(f)]] = r[j].cpu().numpy()
                o += len(x)
        print('infer', name, flush=True)
    assert np.isfinite(logit).all()
    np.savez_compressed(OUT/'graded_frame_scores.npz', logit=logit, range_est=rng_est)


def analyze():
    z = SS.read(OUT/'graded_frame_scores.npz'); logit, rest = z['logit'], z['range_est']
    w = np.array([1, 2, 4, 8, 16], float)
    score = np.zeros_like(logit)
    for t in range(len(ALL)):
        ww = w[-min(5, t+1):]
        score[:, :, t] = np.tensordot(logit[:, :, t+1-len(ww):t+1], ww/ww.sum(), axes=([2], [0]))
    eps = []
    for ui, u in enumerate(NR.SPLITS['evaluation']):
        d = SS.read(OUT/'features/evaluation'/f'unit{u}.npz'); scenes = NR.scenes_for(u); lab = d['labels'].reshape(40, 16, 6)[:, :, 2:4]
        for c in range(40):
            tq = int(d['group'][c]); rf = scenes[c]['meta']['range']
            for q in (0, 1):
                off = -float(d['margin'][c]) if q == tq else None
                kind = 'clear' if off is None else ('contact' if off > 0 else 'pass')
                eps.append(dict(unit=u, half='calib' if u % 2 else 'eval', kind=kind, off=off, rfinal=rf, cond=str(d['family'][c]),
                                ranges=rf+STEP*(15-ALL), s=score[ui, c, :, q], r_est=rest[ui, c, :, q], lab_final=int(lab[c, -1, q])))
    minutes = lambda E: len(E)*len(ALL)*FRAME_S/60

    def first(mask):
        idx = np.flatnonzero(mask); return idx[0] if len(idx) else None

    def stops(e, pol, T):
        if pol == 'A': return first(e['s'] >= T)
        return first((e['s'] >= T) & (e['r_est'] <= CONFIRM))

    def fsr(E, pol, T):  # false stops per minute on clear episodes
        return sum(stops(e, pol, T) is not None for e in E)/minutes(E)

    def calibrate(pol):
        E = [e for e in eps if e['half'] == 'calib' and e['kind'] == 'clear']
        grid = np.quantile(np.concatenate([e['s'] for e in E]), np.linspace(.5, .9999, 400))
        ok = [T for T in grid if fsr(E, pol, T) <= BUDGET]
        return float(min(ok))
    TA, TB = calibrate('A'), calibrate('B')
    Ec = [e for e in eps if e['half'] == 'calib' and e['kind'] == 'clear']
    gridc = np.quantile(np.concatenate([e['s'] for e in Ec]), np.linspace(.5, .9999, 400))
    TC = float(min(T for T in gridc if sum((e['s'] >= T).any() for e in Ec)/minutes(Ec) <= 6.0))  # caution budget 6/min
    ev = [e for e in eps if e['half'] == 'eval']
    approach = [e for e in ev if e['rfinal'] <= 1.0]  # sequences that reach the critical range

    def summary(pol, T):
        out = {}
        for name, sel in (('contact0-2cm', lambda e: e['kind'] == 'contact' and e['off'] <= .02), ('contact2-5cm', lambda e: e['kind'] == 'contact' and .02 < e['off'] <= .05),
                          ('contact>5cm', lambda e: e['kind'] == 'contact' and e['off'] > .05), ('pass0-5cm', lambda e: e['kind'] == 'pass' and e['off'] > -.05),
                          ('pass5-10cm', lambda e: e['kind'] == 'pass' and e['off'] <= -.05)):
            E = [e for e in approach if sel(e)]; ts = [stops(e, pol, T) for e in E]
            timely = [t is not None and e['ranges'][t] >= SAFE for e, t in zip(E, ts)]
            lead = [(e['ranges'][t]-.5)/.8 for e, t in zip(E, ts) if t is not None]
            out[name] = dict(n=len(E), stop_any=float(np.mean([t is not None for t in ts])), stop_timely=float(np.mean(timely)),
                             median_lead_s=float(np.median(lead)) if lead else None)
        out['false_stops_per_min'] = fsr([e for e in ev if e['kind'] == 'clear'], pol, T)
        return out
    res = dict(thresholds=dict(A=TA, B=TB, caution=TC), A=summary('A', TA), B=summary('B', TB))
    cont = [e for e in approach if e['kind'] == 'contact']
    clead = [(e['ranges'][t]-.5)/.8 for e in cont for t in [first(e['s'] >= TC)] if t is not None]
    res['B']['caution'] = dict(contact_caution_rate=float(np.mean([(e['s'] >= TC).any() for e in cont])), contact_median_caution_lead_s=float(np.median(clead)),
                               clear_cautions_per_min=float(sum((e['s'] >= TC).any() for e in ev if e['kind'] == 'clear')/minutes([e for e in ev if e['kind'] == 'clear'])),
                               pass_caution_rate=float(np.mean([(e['s'] >= TC).any() for e in approach if e['kind'] == 'pass'])))
    # range-estimate sanity on contact episodes at frames where the target is inside the query depth
    errs = [e['r_est'][t]-e['ranges'][t] for e in cont for t in range(len(ALL)) if e['ranges'][t] <= 2.5 and e['s'][t] >= TB]
    res['range_estimate_error_m'] = dict(n=len(errs), median=float(np.median(errs)) if errs else None, p90_abs=float(np.quantile(np.abs(errs), .9)) if errs else None)
    res['denominators'] = dict(eval_clear_minutes=minutes([e for e in ev if e['kind'] == 'clear']), approach_episodes=len(approach))
    SS.save(OUT/'graded_alert_results.json', res); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', required=True); p.add_argument('--chunk'); a = p.parse_args()
    tag = a.stage+(f"_c{a.chunk.replace('/', 'of')}" if a.chunk else '')
    try:
        materialize(a.chunk) if a.stage == 'materialize' else {'infer': infer, 'analyze': analyze}[a.stage]()
        SS.save(OUT/f'terminal_graded_{tag}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_graded_{tag}.json', dict(status='failed', error=repr(e))); raise
