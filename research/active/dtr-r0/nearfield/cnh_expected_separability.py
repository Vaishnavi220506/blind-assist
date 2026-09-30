"""Expected-response and noise-scale diagnostic for matched inside/outside targets.

Not an ideal-observer accuracy: diagonal-Gaussian d' on the coarse 8x8x16
histogram over the last 8 frames, with the wall and both candidate target
positions known (privileged simulation control). No training.
Plan: artifacts.local/work/cnh-coverage-sweep-20260930/SWEEP_PLAN.md
"""
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS

OUT = SS.WORK/'cnh-coverage-sweep-20260930'/'separability'
PLAN = SS.WORK/'cnh-coverage-sweep-20260930'/'SWEEP_PLAN.md'
UNITS = list(range(85000, 85024))
MARGINS = (.015, .045, .12)
CONDITIONS = ('no_panel', 'B0', 'B1', 'B2', 'B3', 'B4')
NOISY = 8


def target_at(t, margin):
    lo, hi = list(t['box']['lo']), list(t['box']['hi']); width = hi[0]-lo[0]
    inside = .30+margin
    lo[0], hi[0] = (inside, inside+width) if t['side'] == 1 else (-inside-width, -inside)
    return dict(lo=lo, hi=hi, rho=t['box']['rho'])


def main():
    assert PLAN.exists()
    import cnh_proposal_attribution_scenes as S
    from cnh_corridor_labels import labels_for_all
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []; start = time.time(); mean_err = []
    for ci, cond in enumerate(CONDITIONS):
        for mi, m in enumerate(MARGINS):
            for u in UNITS:
                rng = np.random.default_rng([2026093002, ci, mi, u])
                ref = S.make_scenes(u)[0]; group = int(rng.integers(2))
                t = SS.draw_target(rng, -m, group)
                boxes_ctx = []
                if cond != 'no_panel':
                    b = int(cond[1]); sh = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.08, .15)))
                    la = SS.EDGES[b]+float(rng.uniform())*(SS.EDGES[b+1]-SS.EDGES[b])
                    boxes_ctx = SS.panel_scene(u, 0, ref, t, sh, la)['boxes'][1:2]
                scenes = []
                for sign in (-1, 1):
                    boxes = [target_at(t, sign*m)]+boxes_ctx+[SS.FLOOR, SS.BACK]
                    lab = labels_for_all(boxes, ref['travel'][-1:])[0, [2, 3]]
                    scenes.append(dict(boxes=boxes, poses=ref['poses'], label=int(lab[group])))
                assert scenes[0]['label'] == 1 and scenes[1]['label'] == 0, (cond, m, u)
                seed = 2026093002000+ci*100000+mi*1000+(u-85000)
                mu = [S.render(sc, seed, noise_scale=0)['hist'][-8:] for sc in scenes]
                noisy = [np.stack([S.render(sc, seed+17*(k+1))['hist'][-8:] for k in range(NOISY)]) for sc in scenes]
                # Poisson-consistent diagonal noise model var = a*mu + c, fitted over all bins of the pair
                # (signed = mu + counts - ambient_estimate - mu, so var grows with mu plus a background term).
                var_emp = np.concatenate([n-n.mean(0) for n in noisy]).var(0)*(2*NOISY)/(2*NOISY-2)
                mbar = np.maximum((mu[0]+mu[1])/2, 0)
                (a, c), *_ = np.linalg.lstsq(np.stack([mbar.ravel(), np.ones(mbar.size)], 1), var_emp.ravel(), rcond=None)
                assert c > 0 and a >= 0, (a, c)
                var = a*mbar+c
                diff = mu[0]-mu[1]
                dprime = float(np.sqrt((diff**2/var).sum()))
                for k in (0, 1):  # standardized mean check: E[(mean-mu)^2/(var/NOISY)] ~ 1
                    mean_err.append(float(np.mean((noisy[k].mean(0)-mu[k])**2/(var/NOISY))))
                rows.append(dict(condition=cond, margin=m, unit=u, group=group, dprime=dprime,
                                 exact_zero_difference=bool(np.all(diff == 0)), max_abs_diff=float(np.abs(diff).max())))
            print(cond, m, round(time.time()-start, 1), flush=True)
    summary = {}
    for cond in CONDITIONS:
        for m in MARGINS:
            d = np.array([r['dprime'] for r in rows if r['condition'] == cond and r['margin'] == m])
            summary[f'{cond}|{m}'] = dict(n=len(d), median=float(np.median(d)), q25=float(np.percentile(d, 25)), q75=float(np.percentile(d, 75)),
                                         frac_below_1=float((d < 1).mean()), frac_below_2=float((d < 2).mean()),
                                         exact_zero=int(sum(r['exact_zero_difference'] for r in rows if r['condition'] == cond and r['margin'] == m)))
    SS.save(OUT/'results.json', dict(status='DIAGNOSTIC_NOT_IN_READING', note='diagonal-Gaussian dprime, known wall and positions; not ideal-observer accuracy',
                                     noisy_mean_standardized_sq_error=dict(median=float(np.median(mean_err)), max=float(np.max(mean_err))),
                                     summary=summary, rows=rows, script_sha256=SS.sha(__file__)))
    print(json.dumps(summary, indent=0))
    print('standardized mean check (expect ~1): median', np.median(mean_err), 'max', np.max(mean_err))


if __name__ == '__main__':
    try:
        main(); SS.save(OUT/'terminal.json', dict(status='complete'))
    except BaseException as e:
        OUT.mkdir(parents=True, exist_ok=True); SS.save(OUT/'terminal.json', dict(status='failed', error=repr(e))); raise
