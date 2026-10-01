"""Where along the CVR pipeline does inside/outside information shrink near large walls?

Matched pairs (target just inside vs outside, identical wall) are rendered K
times each with independent noise and noisy poses. For each representation
(raw 8-frame histogram, z1 evidence, CVR voxel input at frame 15, FULL CVR
logit) the noisy samples are projected on the noise-free difference direction
(known-signal matched filter; covariance handled along that direction) and a
1-D d' is computed. Privileged diagnostic, no training.
Plan: artifacts.local/work/cnh-representation-separability-20260930/REPR_PLAN.md
"""
import json
import time

import numpy as np

import cnh_structure_space as SS
from cnh_expected_separability import target_at

OUT = SS.WORK/'cnh-representation-separability-20260930'
PLAN = OUT/'REPR_PLAN.md'
UNITS = list(range(86000, 86024))
MARGINS = (.045, .12)
CONDITIONS = ('no_panel', 'B0', 'B2', 'B4')
K = 12
REPS = ('raw_hist', 'z1', 'voxel', 'logit')


def main():
    assert PLAN.exists()
    import torch
    import cnh_proposal_attribution as old
    import cnh_proposal_attribution_scenes as S
    from cnh_corridor_labels import labels_for_all
    from cnh_cvr_pilot import CVR, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import query_masks
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    _, _, F, _, _, noisy_fn, _, _ = old.setup()
    bias = np.load(old.DATA/'readouts-gpu/primary-mount-10-snr6/bias.npy')
    projector = BatchedProjector(); masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    nets = []
    for s in range(3):
        net = CVR().cuda(); net.load_state_dict(torch.load(SS.OUT/'models/FULL'/f'model_seed{s}.pt', map_location='cuda', weights_only=True)); nets.append(net.eval())

    def represent(h, a, scene, noisy, group):
        tq = np.linalg.inv(scene['travel'])@scene['poses']
        _, z1, _ = F.sequence_features(h, a, bias, tq, noisy)
        vox = projector.sequence(z1[8:16], relative_transforms(scene['poses'], scene['travel'], noisy, 15)).cpu().numpy()[None]
        x = SS.prep(torch, vox, masks)
        with torch.no_grad(): logit = float(torch.stack([n(x) for n in nets]).mean(0)[0, group])
        v = vox[0].astype(np.float64).copy(); v[0] = np.sign(v[0])*np.log1p(np.abs(v[0])); v[2] = np.sign(v[2])*np.log1p(np.abs(v[2])); v[1] /= 8
        return dict(raw_hist=h[-8:].ravel(), z1=np.asarray(z1[-8:], np.float64).ravel(), voxel=v.ravel(), logit=np.array([logit]))

    rows = []; start = time.time()
    for ci, cond in enumerate(CONDITIONS):
        for mi, m in enumerate(MARGINS):
            for u in UNITS:
                rng = np.random.default_rng([2026093003, ci, mi, u])
                ref = S.make_scenes(u)[0]; group = int(rng.integers(2))
                t = SS.draw_target(rng, -m, group); ctx = []
                if cond != 'no_panel':
                    b = int(cond[1]); sh = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.08, .15)))
                    la = SS.EDGES[b]+float(rng.uniform())*(SS.EDGES[b+1]-SS.EDGES[b])
                    ctx = SS.panel_scene(u, 0, ref, t, sh, la)['boxes'][1:2]
                scenes = []
                for sign in (-1, 1):
                    boxes = [target_at(t, sign*m)]+ctx+[SS.FLOOR, SS.BACK]
                    lab = labels_for_all(boxes, ref['travel'])[:, [2, 3]]
                    scenes.append(dict(boxes=boxes, poses=ref['poses'], travel=ref['travel'], label=int(lab[-1, group])))
                assert scenes[0]['label'] == 1 and scenes[1]['label'] == 0
                seed = 2026093003000+ci*100000+mi*1000+(u-86000)*50
                expect, samples = [], []
                for sc in scenes:
                    o = S.render(sc, seed, noise_scale=0)
                    expect.append(represent(o['hist'], o['ambient'], sc, sc['poses'], group))
                    ss = []
                    for k in range(K):
                        o = S.render(sc, seed+1+k)
                        ss.append(represent(o['hist'], o['ambient'], sc, noisy_fn(sc['poses'], seed+1+k+7, dt=.2), group))
                    samples.append(ss)
                row = dict(condition=cond, margin=m, unit=u, group=group)
                for r in REPS:
                    w = expect[0][r]-expect[1][r]
                    if r == 'logit': w = np.sign(w) if w[0] != 0 else np.ones(1)
                    p = [np.array([s[r]@w for s in samples[j]]) for j in (0, 1)]
                    sd = np.sqrt((p[0].var(ddof=1)+p[1].var(ddof=1))/2)
                    row[r] = float((p[0].mean()-p[1].mean())/sd) if sd > 0 else float('nan')
                    if r == 'logit': row['logit_mean_in'], row['logit_mean_out'] = float(p[0].mean()), float(p[1].mean())
                rows.append(row)
            print(cond, m, round(time.time()-start, 1), {r: round(float(np.median([x[r] for x in rows if x['condition'] == cond and x['margin'] == m])), 2) for r in REPS}, flush=True)
    summary = {}
    for cond in CONDITIONS:
        for m in MARGINS:
            rr = [x for x in rows if x['condition'] == cond and x['margin'] == m]
            summary[f'{cond}|{m}'] = {r: dict(median=float(np.median([x[r] for x in rr])), q25=float(np.percentile([x[r] for x in rr], 25)),
                                              q75=float(np.percentile([x[r] for x in rr], 75))) for r in REPS}
    ratio = {f'{r}|{m}': summary[f'B4|{m}'][r]['median']/summary[f'no_panel|{m}'][r]['median'] for r in REPS for m in MARGINS}
    reading = {}
    for m in MARGINS:
        rv, rl, rh = ratio[f'voxel|{m}'], ratio[f'logit|{m}'], ratio[f'raw_hist|{m}']
        if rh >= .9 and rv <= .7: reading[m] = 'REPRESENTATION_LOSES_INFORMATION'
        elif rv >= .9 and rl <= .7: reading[m] = 'READOUT_DOES_NOT_USE_INFORMATION'
        elif min(rh, rv, rl) >= .9: reading[m] = 'ALL_PRESERVED'
        else: reading[m] = 'MIXED'
    SS.save(OUT/'results.json', dict(status='PRIVILEGED_DIAGNOSTIC', reading=reading, b4_over_no_panel_ratio=ratio, summary=summary, rows=rows,
                                     plan_sha256=SS.sha(PLAN), script_sha256=SS.sha(__file__)))
    print('reading', reading); print('ratios', {k: round(v, 3) for k, v in ratio.items()})


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        main(); SS.save(OUT/'terminal.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/'terminal.json', dict(status='failed', error=repr(e))); raise
