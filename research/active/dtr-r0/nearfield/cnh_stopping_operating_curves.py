"""Existing-output-only stopping diagnostic; never trains or generates scenes."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts.local/work/cnh-corridor-stopping-20260929'
OUT = ROOT / 'artifacts.local/work/cnh-stopping-curves-20260929'
POLICIES = ('head', 'travel', 'old_head', 'old_travel')
BUDGETS = (.5, 1., 2., 5.)
DT = .2
SPEED = .8
REACTION = .6


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def episodes(scores, threshold):
    alarm = scores >= threshold
    starts = alarm & ~np.pad(alarm[:, :-1], ((0, 0), (1, 0)))
    return int(starts.sum())


def calibrate(scores, budget):
    """Stop on FIRST budget violation descending through unique scores."""
    assert scores.ndim == 2 and len(scores) and np.isfinite(scores).all()
    flat = scores.ravel()
    order = np.argsort(flat)[::-1]
    active = np.zeros_like(scores, bool)
    minutes = scores.size * DT / 60
    threshold = float(np.nextafter(float(flat.max()), np.inf))
    count = 0
    i = 0
    while i < len(order):
        value = flat[order[i]]
        j = i
        while j < len(order) and flat[order[j]] == value:
            row, col = np.unravel_index(order[j], scores.shape)
            count += 1 - (int(active[row, col-1]) if col else 0) - (int(active[row, col+1]) if col+1 < scores.shape[1] else 0)
            active[row, col] = True
            j += 1
        if count / minutes > budget:
            break
        threshold = float(value)
        i = j
    return threshold


def outcomes(scores, threshold, distance):
    alarm = scores >= threshold
    hit = alarm.any(1)
    first = np.where(hit, (alarm.argmax(1) + 7) * DT, np.inf)
    hazard = np.isfinite(distance)
    collision = hazard & (SPEED * first + SPEED * REACTION + SPEED**2 / 3 > distance)
    unnecessary = ~hazard & hit
    return collision, unnecessary, hit, first


def full_curve(scores, distance):
    """Exact all-threshold counts from per-clip maxima; no O(thresholds*frames)."""
    hazard = np.isfinite(distance)
    times = (np.arange(scores.shape[1]) + 7) * DT
    timely = SPEED * times[None] + SPEED * REACTION + SPEED**2 / 3 <= distance[:, None]
    timely_max = np.where(timely, scores, -np.inf).max(1)[hazard]
    safe_max = scores[~hazard].max(1)
    thresholds = np.r_[np.nextafter(float(scores.max()), np.inf), np.unique(scores)[::-1]]
    saved = len(timely_max) - np.searchsorted(np.sort(timely_max), thresholds, side='left')
    unnecessary = len(safe_max) - np.searchsorted(np.sort(safe_max), thresholds, side='left')
    return thresholds, len(timely_max) - saved, unnecessary


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = ('Existing predictions only; no scene generation, rendering, inference or training.\n'
            'Head self-calibration negatives: whole clips with no positive center HEAD/BODY own-query label in frames 7..40.\n'
            'Travel and both old models: original physical-clear whole clips. Budgets .5/1/2/5 episodes/minute.\n'
            'Retain first-violation threshold rule, 34*.2s sample-duration convention, first-stop physics, .6s reaction.\n'
            'Bootstrap travel minus head(self), 48 paired units, 5000 draws, seed 20260929, both outcomes at 2/min.\n'
            'Evaluation all-threshold curve is post-hoc description, not selection. Frozen advance=false unchanged.\n'
            'First collider depth is inward edge displacement -stored margin, not intersection width. Source geometry excludes all other objects from collision.\n')
    (OUT / 'PLAN.md').write_text(plan, encoding='utf8')
    paths = [Path(__file__), BASE / 'results.json', BASE / 'source/cnh_proposal_attribution_scenes.py', BASE / 'source/cnh_corridor_stopping.py']
    sets = {'calib': list(range(7100, 7124)), 'evaluation': list(range(8100, 8148))}
    data = {}
    for split, ids in sets.items():
        parts = []
        for unit in ids:
            path = BASE / 'predictions' / f'unit{unit}.npz'
            paths.append(path)
            d = read(path)
            d['unit'] = np.full(len(d['distance']), unit)
            if split == 'calib':
                featurepath = BASE / 'features/calib' / f'unit{unit}.npz'
                paths.append(featurepath)
                # Load labels only, not large feature tensors.
                with np.load(featurepath) as z:
                    own = z['labels_head'].reshape(22, 41, 6)[:, 7:, 2:4]
                d['own_negative'] = ~own.astype(bool).any((1, 2))
            parts.append(d)
        data[split] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    save(OUT / 'request.json', {'plan_sha256': hashlib.sha256((OUT / 'PLAN.md').read_bytes()).hexdigest(),
         'sources_and_inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
         'backend': 'CPU: TASK_NOT_GPU_SUITABLE; sorting and count aggregation'})
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    cal, ev = data['calib'], data['evaluation']
    hazard = np.isfinite(ev['distance'])
    assert int(hazard.sum()) == 528 and int((~hazard).sum()) == 528
    rows, comparison = [], {}
    for policy in POLICIES:
        mask = cal['own_negative'] if policy == 'head' else np.isinf(cal['distance'])
        cs = cal[policy][mask, 7:]
        for budget in BUDGETS:
            threshold = calibrate(cs, budget)
            collision, unnecessary, hit, first = outcomes(ev[policy][:, 7:], threshold, ev['distance'])
            actual = episodes(cs, threshold)
            assert actual / (cs.size * DT / 60) <= budget + 1e-12
            row = dict(policy=policy, calibration_truth='own_query' if policy == 'head' else 'physical_body', budget=budget,
                       threshold=threshold, calib_negative_clips=int(mask.sum()), calib_minutes=cs.size*DT/60,
                       calib_episodes=actual, calib_rate=actual/(cs.size*DT/60), collisions=int(collision.sum()),
                       hazard_n=528, unnecessary=int(unnecessary.sum()), clear_n=528,
                       eval_clear_episodes=episodes(ev[policy][~hazard, 7:], threshold), eval_clear_minutes=528*34*DT/60)
            rows.append(row)
            if budget == 2 and policy in ('head', 'travel'):
                comparison[policy] = (collision, unnecessary)
    rng = np.random.default_rng(20260929)
    draws = rng.integers(48, size=(5000, 48))
    boot = {}
    for index, (name, denominator) in enumerate([('collision', hazard), ('unnecessary', ~hazard)]):
        delta = (comparison['travel'][index].astype(int) - comparison['head'][index].astype(int)).reshape(48, 22).sum(1)
        counts = denominator.reshape(48, 22).sum(1)
        samples = delta[draws].sum(1) / counts[draws].sum(1)
        boot[name] = dict(point=float(delta.sum()/counts.sum()), ci95=np.quantile(samples, [.025, .975]).tolist(), units=48, draws=5000, seed=20260929)
    frozen = json.loads((BASE / 'results.json').read_text())
    assert frozen['comparison']['advance'] is False
    threshold = frozen['thresholds']['travel']
    collision, unnecessary, hit, first = outcomes(ev['travel'][:, 7:], threshold, ev['distance'])
    assert collision.sum() == 122 and unnecessary.sum() == 90
    decomposition = []
    ledger = []
    for i in np.flatnonzero(collision):
        depth = -float(ev['margin'][i])
        assert depth > 0
        band = 'le5cm' if depth <= .05 else '5to15cm' if depth <= .15 else 'gt15cm'
        ledger.append(dict(unit=int(ev['unit'][i]), config=int(i % 22), family=str(ev['family'][i]),
                           depth_m=depth, depth_band=band, cause='late' if hit[i] else 'never',
                           first_alarm_s=float(first[i]) if hit[i] else None,
                           deadline_s=float(ev['distance'][i]/SPEED-REACTION-SPEED/3)))
    for family in ('boundary', 'mixed_surface', 'sidewall', 'general'):
        for band in ('le5cm', '5to15cm', 'gt15cm'):
            rr = [r for r in ledger if r['family'] == family and r['depth_band'] == band]
            decomposition.append(dict(family=family, depth_band=band, never=sum(r['cause']=='never' for r in rr), late=sum(r['cause']=='late' for r in rr), total=len(rr)))
    curves = {}
    for policy in POLICIES:
        th, c, n = full_curve(ev[policy][:, 7:], ev['distance'])
        # Independent direct frame-wise cross-check over endpoints and spaced thresholds.
        for j in np.unique(np.linspace(0, len(th)-1, 31).astype(int)):
            cc, nn, _, _ = outcomes(ev[policy][:, 7:], th[j], ev['distance'])
            assert (int(cc.sum()), int(nn.sum())) == (int(c[j]), int(n[j]))
        curves[policy] = (th, c, n)
    np.savez_compressed(OUT / 'full_threshold_curves.npz', **{f'{p}_{name}': a for p, arrays in curves.items() for name, a in zip(('threshold', 'collisions', 'unnecessary'), arrays)})
    with (OUT / 'operating_points.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / 'collision_decomposition.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(decomposition[0])); w.writeheader(); w.writerows(decomposition)
    (OUT / 'collision_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger), encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(9.5, 6.5), layout='constrained')
    names = {'head': '头随动（独立重训）', 'travel': '行进走廊（独立重训）', 'old_head': '旧模型＋头随动', 'old_travel': '旧模型＋行进走廊'}
    colors = dict(zip(POLICIES, ('#b55a30', '#127d85', '#a48d76', '#769fae')))
    for p, (th, c, n) in curves.items():
        ax.plot(n/528*100, c/528*100, color=colors[p], label=names[p], linewidth=2, alpha=.95)
        pr = [r for r in rows if r['policy']==p]
        ax.scatter([r['unnecessary']/528*100 for r in pr], [r['collisions']/528*100 for r in pr], color=colors[p], s=28)
        r = next(r for r in pr if r['budget']==2)
        ax.scatter(r['unnecessary']/528*100, r['collisions']/528*100, color=colors[p], marker='*', s=160, edgecolors='white', zorder=5)
    ax.scatter([0, 100], [100, 0], marker='s', color='#333333', zorder=6)
    ax.annotate('从不报警 (0, 528)', (0, 100), xytext=(8, -18), textcoords='offset points')
    ax.annotate('始终报警 (528, 0)', (100, 0), xytext=(-138, 12), textcoords='offset points')
    ax.set(xlabel='无必要首停 / 528 个物理安全片段 (%)', ylabel='碰撞 / 528 个危险片段 (%)', xlim=(-2, 102), ylim=(-2, 102),
           title='停步操作曲线：0.6 s 反应，48 个评估单位\n事后描述、不据此选点')
    ax.grid(alpha=.2); ax.legend(loc='upper right')
    fig.text(.13, -.035, '圆点：calib 预定 0.5 / 1 / 2 / 5 段/分钟；星号：2 段/分钟。头随动按自身查询真值校准，其余按身体真值。', fontsize=9)
    fig.savefig(OUT / 'operating_curves.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'operating_curves.svg', bbox_inches='tight')
    plt.close(fig)
    result = dict(scope='Consumed Development; post-hoc descriptive diagnostic, not confirmation', rows=rows, bootstrap_travel_minus_head_self=boot,
                  frozen_advance=False, decomposition=decomposition,
                  head_own_negative_clips=int(cal['own_negative'].sum()), physical_safe_with_head_positives=int((np.isinf(cal['distance']) & ~cal['own_negative']).sum()),
                  runtime_s=time.monotonic()-started, deviations=[], calibration_convention=plan)
    save(OUT / 'results.json', result)
    save(OUT / 'acceptance.json', dict(status='pass', original_travel_counts=[122,90], checked_curve_points=124,
         operating_points=16, collision_ledger_count=len(ledger), no_training=True, no_generation=True))
    save(OUT / 'terminal.json', dict(status='complete', runtime_s=result['runtime_s']))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
