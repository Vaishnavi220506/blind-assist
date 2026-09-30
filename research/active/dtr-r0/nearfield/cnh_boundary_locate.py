"""Locate where shallow-intrusion information is lost: voxel -> conv features -> pooled -> logits.

Matched inside/outside pairs (same wall), K noisy renders each. Representations:
voxel input (frame 15), conv features before pooling (5 seeds concatenated),
pooled head input for the target's query (5 seeds), single-seed frame-15 logit,
and the deployed 5-seed ensemble score (frames 11-15 weighted). Known-signal
matched-filter d' per pair. Privileged diagnostic, no training.
Plan: artifacts.local/work/cnh-boundary-locate-20261001/LOCATE_PLAN.md
"""
import argparse
import json
import time

import numpy as np

import cnh_structure_space as SS
from cnh_expected_separability import target_at

OUT = SS.WORK/'cnh-boundary-locate-20261001'
PLAN = OUT/'LOCATE_PLAN.md'
UNITS = list(range(89000, 89024))
MARGINS = (.015, .045, .12)
CONDITIONS = ('no_panel', 'B0', 'B2', 'B4')
CELLS = [(c, m) for c in CONDITIONS for m in MARGINS]
K = 12
REPS = ('voxel', 'conv', 'pooled', 'logit_single', 'logit_ensemble')
FIX = SS.WORK/'cnh-readout-fix-20261001'


def run(part):
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
    paths = [SS.OUT/'models/FULL'/f'model_seed{s}.pt' for s in range(3)]+[FIX/'models/BASE'/f'model_seed{s}.pt' for s in (3, 4)]
    nets = []
    for p in paths:
        net = CVR().cuda(); net.load_state_dict(torch.load(p, map_location='cuda', weights_only=True)); nets.append(net.eval())
    frames = SS.EVAL_FRAMES; w = SS.WEIGHTS/SS.WEIGHTS.sum()

    def forward(net, x):
        f = net.body(x)
        mask = torch.nn.functional.adaptive_avg_pool3d(x[:, 3:5], f.shape[2:])[:, :, None]; ff = f[:, None]
        mx = ff.masked_fill(mask == 0, -1e4).flatten(3).max(-1).values
        mean = (ff*mask).flatten(3).sum(-1)/mask.flatten(3).sum(-1).clamp_min(1e-8)
        emb = net.emb.weight[None].expand(len(x), -1, -1)
        return f, torch.cat([mx, mean], -1), net.head(torch.cat([mx, mean, emb], -1)).squeeze(-1)

    def represent(h, a, scene, noisy, group):
        tq = np.linalg.inv(scene['travel'])@scene['poses']
        _, z1, _ = F.sequence_features(h, a, bias, tq, noisy)
        vox = np.stack([projector.sequence(z1[max(0, f-7):f+1], relative_transforms(scene['poses'], scene['travel'], noisy, int(f))).cpu().numpy() for f in frames])
        x = SS.prep(torch, vox, masks)
        with torch.no_grad():
            outs = [forward(n, x) for n in nets]
        conv = np.concatenate([o[0][-1].flatten().cpu().numpy() for o in outs]).astype(np.float64)
        pooled = np.concatenate([o[1][-1, group].cpu().numpy() for o in outs]).astype(np.float64)
        logits = np.stack([o[2][:, group].cpu().numpy() for o in outs]).astype(np.float64)  # [seeds, frames]
        v = x[-1, :3].cpu().numpy().astype(np.float64).ravel()
        return dict(voxel=v, conv=conv, pooled=pooled, logit_single=logits[0, -1:], logit_ensemble=np.array([(logits.mean(0)*w).sum()]))

    k, n = map(int, part.split('/')); rows = []; start = time.time()
    for cond, m in CELLS[k::n]:
        ci, mi = CONDITIONS.index(cond), MARGINS.index(m)
        for u in UNITS:
            rng = np.random.default_rng([2026100102, ci, mi, u])
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
            seed = 2026100102000+ci*100000+mi*1000+(u-89000)*50
            expect, samples = [], []
            for sc in scenes:
                o = S.render(sc, seed, noise_scale=0); expect.append(represent(o['hist'], o['ambient'], sc, sc['poses'], group))
                ss = []
                for j in range(K):
                    o = S.render(sc, seed+1+j)
                    ss.append(represent(o['hist'], o['ambient'], sc, noisy_fn(sc['poses'], seed+1+j+7, dt=.2), group))
                samples.append(ss)
            row = dict(condition=cond, margin=m, unit=u, group=group)
            for r in REPS:
                d = expect[0][r]-expect[1][r]
                if d.size == 1: d = np.sign(d) if d[0] != 0 else np.ones(1)
                p = [np.array([s[r]@d for s in samples[j]]) for j in (0, 1)]
                sd = np.sqrt((p[0].var(ddof=1)+p[1].var(ddof=1))/2)
                row[r] = float((p[0].mean()-p[1].mean())/sd) if sd > 0 else float('nan')
            rows.append(row)
        print(cond, m, round(time.time()-start, 1), {r: round(float(np.median([x[r] for x in rows if x['condition'] == cond and x['margin'] == m])), 2) for r in REPS}, flush=True)
    SS.save(OUT/f'part_{k}of{n}.json', rows)


def combine():
    rows = sum([json.loads(p.read_text()) for p in sorted(OUT.glob('part_*of*.json'))], [])
    assert len(rows) == len(CELLS)*len(UNITS), len(rows)
    med = {f'{c}|{m}': {r: float(np.median([x[r] for x in rows if x['condition'] == c and x['margin'] == m])) for r in REPS} for c, m in CELLS}
    retention = {k: {r: v[r]/v['voxel'] for r in REPS if r != 'voxel'} for k, v in med.items()}
    reading = {}
    for m in (.015, .045):
        keys = [f'no_panel|{m}', f'B4|{m}']
        ens = min(retention[k]['logit_ensemble'] for k in keys); conv = min(retention[k]['conv'] for k in keys)
        pool = min(retention[k]['pooled'] for k in keys)
        if ens >= .85: reading[m] = 'SMALL_HEADROOM'
        elif conv >= .85 and pool <= .7: reading[m] = 'POOLING_LOSS'
        elif conv <= .7: reading[m] = 'CONV_LOSS'
        else: reading[m] = 'MIXED'
    SS.save(OUT/'results.json', dict(status='PRIVILEGED_DIAGNOSTIC', reading=reading, median_dprime=med, retention_vs_voxel=retention,
                                     rows=rows, plan_sha256=SS.sha(PLAN), script_sha256=SS.sha(__file__)))
    print('reading', reading)
    for k, v in med.items():
        print(f'{k:14s} ' + ' '.join(f'{r}={v[r]:.2f}' for r in REPS) + '  retention ' + ' '.join(f'{r}={retention[k][r]:.2f}' for r in retention[k]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--part'); p.add_argument('--combine', action='store_true'); a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = 'combine' if a.combine else f"part_{a.part.replace('/', 'of')}"
    try:
        combine() if a.combine else run(a.part)
        SS.save(OUT/f'terminal_{tag}.json', dict(status='complete'))
    except BaseException as e:
        SS.save(OUT/f'terminal_{tag}.json', dict(status='failed', error=repr(e))); raise
