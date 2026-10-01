"""Detectability envelope: P(alarm) vs signed boundary offset and range, deployed readout.

New Development scenes with a continuous signed lateral offset of the target
(-0.15 m outside ... +0.10 m inside... see sign below) and continuous range, with no
context or a near same-side panel. The frozen 5-seed BASE CVR ensemble and PARTIAL are
scored at the deployed 10% calibration thresholds of the readout-fix benchmark. A
logistic psychometric fit per (condition, range bin) gives the effective boundary
location (mu) and blur width (s); compared with the sensor zone width (0.1036*range).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_readout_fix as RF

OUT = SS.WORK/'cnh-detectability-envelope-20261001'
UNITS = list(range(92000, 92096))
RANGE_BINS = ((1.15, 1.6), (1.6, 2.1), (2.1, 2.6))
ZONE_TAN = 2*np.tan(np.pi/8)/8  # lateral zone width per metre of range (8 zones over +/-22.5 deg)
_ORIGINAL = {}


def known(unit):
    import cnh_proposal_attribution_scenes as S
    return _ORIGINAL.get('make_scenes', S.make_scenes)(unit)


def scenes_for(unit):
    """40 scenes: 20 without context, 20 with a near same-side panel; margin>0 means outside."""
    import cnh_proposal_attribution_scenes as S
    ref = known(unit)[0]; rng = np.random.default_rng([2026100103, int(unit)]); out = []
    for s in range(40):
        cond = 'none' if s < 20 else 'corner'; group = (unit//6+s) % 2
        margin = float(rng.uniform(-.15, .10))
        t = SS.draw_target(rng, margin, group)
        z = float(rng.uniform(1.15, 2.6)); dz = z-t['z']
        t['box']['lo'][2] += dz; t['box']['hi'][2] += dz; t['z'] = z
        if cond == 'corner':
            sh = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.08, .15)))
            sc = SS.panel_scene(unit, s, ref, t, sh, SS.EDGES[0]+float(rng.uniform())*(SS.EDGES[5]-SS.EDGES[0]))
        else:
            boxes = [t['box'], SS.FLOOR, SS.BACK]
            sc = dict(unit=int(unit), config=s, family='none', margin=t['margin'], group=t['group'], boxes=boxes,
                      poses=ref['poses'].copy(), travel=ref['travel'].copy(), head=ref['head'].copy(),
                      labels=S.labels_for(boxes, ref['travel']), mode=ref['mode'], dt=.2, speed=.8, meta={})
        sc['family'] = cond; sc['meta'] = dict(sc.get('meta', {}), cond=cond, range=z)
        out.append(sc)
    return out


def generate(chunk):
    import cnh_proposal_attribution_scenes as S
    import cnh_corridor_diagnostic as D
    shared_save = D.old.save
    tag = f"evaluation_c{chunk.replace('/', 'of')}"

    def split_save(path, obj):
        path = Path(path)
        if path.name in ('generation_progress.json', 'generation_terminal.json'):
            return SS.save(path.with_name(path.stem+f'_{tag}.json'), obj)
        return shared_save(path, obj)
    D.old.save = split_save
    _ORIGINAL['make_scenes'] = S.make_scenes; S.make_scenes = scenes_for
    k, n = map(int, chunk.split('/'))
    D.OUT = OUT; D.SPLITS = {'evaluation': UNITS[k::n]}
    D.generate()


def materialize():
    RF.OUT, RF.SPLITS = OUT, {'evaluation': UNITS}
    RF.materialize('evaluation')


def infer():
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    nets = []
    for s in range(5):
        net = CVR().cuda(); net.load_state_dict(torch.load(RF.model_path('BASE', s), map_location='cuda', weights_only=True)); nets.append(net.eval())
    x = np.load(OUT/'data/evaluation/features.npy', mmap_mode='r'); m = SS.read(OUT/'data/evaluation/metadata.npz'); raw = []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            raw.append(torch.stack([n(SS.prep(torch, x[i:i+128], masks)) for n in nets]).mean(0).cpu().numpy())
    raw = np.concatenate(raw); out = {}
    for u in UNITS:
        sel = np.flatnonzero(m['unit'] == u); sc = []
        for c in range(40):
            loc = sel[m['config'][sel] == c]; loc = loc[np.argsort(m['frame'][loc])]
            assert np.array_equal(m['frame'][loc], SS.EVAL_FRAMES)
            sc.append((raw[loc].astype(np.float64)*SS.WEIGHTS[:, None]).sum(0)/SS.WEIGHTS.sum())
        out[str(u)] = np.asarray(sc)
    np.savez_compressed(OUT/'base_ensemble.npz', **out)


def evaluate():
    from cnh_corridor_late_fusion import ranks, threshold
    from sklearn.linear_model import LogisticRegression
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rf = SS.read(RF.OUT/'predictions/BASE.npz')
    # Deployed operating point: 10% budget on the readout-fix calibration negatives (BASE 5-seed mean; PARTIAL).
    thr = {}
    for q, g in enumerate(SS.GROUPS):
        neg_b, neg_p = [], []
        for u in RF.SPLITS['calib']:
            d = SS.read(RF.OUT/'features/calib'/f'unit{u}.npz'); lab = d['labels'].reshape(40, 16, 6)[:, -1, [2, 3]]
            for c in range(40):
                if lab[c, q] == 0: neg_b.append(rf[f'calib|{u}'][c, q].mean()); neg_p.append(d['motion'][c, 2, q])
        for name, v in (('BASE', neg_b), ('PARTIAL', neg_p)):
            ref = np.sort(v); thr[(name, g)] = (ref, threshold(ranks(ref, ref), len(ref)))
    pred = SS.read(OUT/'base_ensemble.npz'); rows = []
    for u in UNITS:
        d = SS.read(OUT/'features/evaluation'/f'unit{u}.npz'); scenes = scenes_for(u)
        for c in range(40):
            q = int(d['group'][c]); g = SS.GROUPS[q]; lab = int(d['labels'].reshape(40, 16, 6)[c, -1, 2+q])
            sc = {'BASE': float(pred[str(u)][c, q]), 'PARTIAL': float(d['motion'][c, 2, q])}
            alarm = {a: int(np.searchsorted(thr[(a, g)][0], sc[a], side='right') >= thr[(a, g)][1]) for a in sc}
            rows.append(dict(unit=u, cond=str(d['family'][c]), group=g, offset_inside=-float(d['margin'][c]),
                             range=scenes[c]['meta']['range'], label=lab, alarm=alarm))
    fits = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ai, cond in enumerate(('none', 'corner')):
        for bi, (lo, hi) in enumerate(RANGE_BINS):
            rr = [r for r in rows if r['cond'] == cond and lo <= r['range'] < hi]
            o = np.array([r['offset_inside'] for r in rr]); uid = np.array([r['unit'] for r in rr])
            for a in ('BASE', 'PARTIAL'):
                yv = np.array([r['alarm'][a] for r in rr])

                def fit(idx):
                    lr = LogisticRegression(C=1e6, max_iter=2000).fit(o[idx, None], yv[idx])
                    b1, b0 = lr.coef_[0, 0], lr.intercept_[0]; mu = -b0/b1; s = 1/b1
                    return mu, s, mu+s*np.log(9), mu-s*np.log(9)
                mu, s, o90, o10 = fit(np.arange(len(o)))
                rng = np.random.default_rng(7); boot = []
                for _ in range(300):
                    pick = rng.choice(UNITS, len(UNITS)); idx = np.concatenate([np.flatnonzero(uid == p) for p in pick])
                    try: boot.append(fit(idx))
                    except Exception: pass
                boot = np.array(boot); ci = lambda j: [float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]
                fits[f'{cond}|{lo}-{hi}|{a}'] = dict(n=len(o), positives=int(sum(r['label'] for r in rr)), mu_m=float(mu), mu_ci=ci(0), s_m=float(s), s_ci=ci(1),
                                                    depth_for_90pct_alarm_m=float(o90), d90_ci=ci(2), outside_offset_10pct_alarm_m=float(o10),
                                                    zone_width_m=float(ZONE_TAN*(lo+hi)/2),
                                                    empirical={f'{e:+.2f}': float(yv[(o >= e) & (o < e+.025)].mean()) for e in np.arange(-.10, .15, .025) if ((o >= e) & (o < e+.025)).any()})
                if a == 'BASE' or bi == 1:
                    xs = np.linspace(-.10, .15, 200)
                    axes[ai].plot(xs*100, 1/(1+np.exp(-(xs-mu)/s)), ls='-' if a == 'BASE' else '--', color=f'C{bi}' if a == 'BASE' else 'gray',
                                  label=f"{'learned' if a == 'BASE' else 'geometry'} {lo}-{hi} m" if a == 'BASE' or bi == 1 else None)
        axes[ai].axvline(0, color='k', lw=.6); axes[ai].set_title({'none': 'no context', 'corner': 'near same-side panel'}[cond])
        axes[ai].set_xlabel('target intrusion into corridor (cm; <0 = outside)'); axes[ai].legend(fontsize=8); axes[ai].grid(alpha=.3)
    axes[0].set_ylabel('P(alarm) at deployed 10% budget')
    fig.tight_layout(); fig.savefig(OUT/'detectability_envelope.png', dpi=150)
    SS.save(OUT/'results.json', dict(scope='Development characterisation, new units 92000-92095; deployed BASE 5-seed and PARTIAL thresholds from readout-fix calib',
                                     sign='offset_inside > 0 means the target surface intrudes into the corridor', fits=fits, script_sha256=SS.sha(__file__)))
    for k, v in fits.items():
        print(f"{k:28s} n={v['n']} mu={v['mu_m']*100:+.1f}cm s={v['s_m']*100:.1f}cm d90={v['depth_for_90pct_alarm_m']*100:+.1f}cm{np.round(np.array(v['d90_ci'])*100, 1).tolist()} out10={v['outside_offset_10pct_alarm_m']*100:+.1f}cm zone={v['zone_width_m']*100:.0f}cm")


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['generate', 'materialize', 'infer', 'evaluate'], required=True); p.add_argument('--chunk')
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    tag = a.stage+(f"_c{a.chunk.replace('/', 'of')}" if a.chunk else '')
    try:
        generate(a.chunk) if a.stage == 'generate' else {'materialize': materialize, 'infer': infer, 'evaluate': evaluate}[a.stage]()
        SS.save(OUT/f'terminal_{tag}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_{tag}.json', dict(status='failed', error=repr(e))); raise
