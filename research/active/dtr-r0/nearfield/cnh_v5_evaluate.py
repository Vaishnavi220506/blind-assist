"""Frozen v5 evaluator: no training, no audit threshold selection.

Use --stage predict after readout gates; --stage evaluate after oracle diagnostics.
All existing inference/feature/history code imports from the frozen source snapshot.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

GROUPS = (('HEAD', (0, 2, 4)), ('BODY', (1, 3, 5)))
BUDGETS = (.01, .05, .10, .20)
MODEL_HASHES = (
    'b7ffb180ddde81a6c28e1e7c3922d9653e948a9023ba88fea9d1b8d7a9ee0209',
    '7bafc3201a395f60790dbdfc95b834e5cea43580acd15a61611c972656e33c8d',
    '9e8296de5a0af23ecb752299c4ea1f9f729237b560c3549c6bd472be34e13083')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def threshold(peaks, budget, strict=False):
    peaks = np.sort(np.asarray(peaks, dtype=np.float64))[::-1]
    if not len(peaks) or not np.isfinite(peaks).all():
        raise ValueError('invalid calibration peaks')
    k = int(np.floor(budget * len(peaks)))
    value = float(peaks[min(k, len(peaks)-1)])
    return value if strict else float(np.nextafter(value, np.inf))


def guard_ranking(a2, guard):
    return np.where(guard, 1., .5 + np.arctan(a2) / np.pi)


def pack(units, keys, group_indices):
    arrays = {k: [] for k in ('y', 'w', 'strata', 'A0', 'A1', 'A2')}
    unit_ids, configs = [], []
    for u in keys:
        d = units[u]
        for c in np.unique(d['config']):
            idx = np.flatnonzero(d['config'] == c)
            idx = idx[np.argsort(d['frame'][idx])]
            assert np.array_equal(d['frame'][idx], np.arange(12)), (u, c)
            idx = idx[3:]
            for k in arrays:
                arrays[k].append(d[k][idx][:, group_indices])
            unit_ids.append(u)
            configs.append(int(c))
    out = {k: np.stack(v) for k, v in arrays.items()}
    out['unit'], out['config'] = np.array(unit_ids), np.array(configs)
    return out


def empty_peaks(packed, arm):
    empty = ~(packed['y'] == 1).any(axis=1)
    return packed[arm].max(axis=1)[empty]


def alert_stats(p, alarms):
    """Events follow v4; group-empty minutes count each physical sequence once."""
    pos = p['y'] == 1
    events = pos.any(axis=1)
    empty = ~events
    hits = pos & alarms
    detected = hits.any(axis=1)
    first = hits.argmax(axis=1)
    lead = np.take_along_axis(p['w'], first[:, None, :], axis=1)[:, 0, :]
    near = events & (np.min(np.where(pos, p['w'], np.inf), axis=1) <= 1.)
    timely = detected & (lead >= 1.)
    # Match existing modal stratum over positive frames; sorted argmax would
    # change tie semantics, so retain first-observed Counter behavior.
    from collections import Counter
    tiny = np.zeros_like(events)
    for i, q in zip(*np.nonzero(events)):
        tiny[i, q] = Counter(p['strata'][i, :, q][pos[i, :, q]].tolist()).most_common(1)[0][0] == 'tiny'
    false_pairs = (alarms.any(axis=1) & empty)
    group_empty = empty.all(axis=1)
    ga = alarms[group_empty].any(axis=2)
    starts = ga & ~np.concatenate((np.zeros((len(ga), 1), bool), ga[:, :-1]), axis=1)
    empty_frames = int(ga.size)
    minutes = empty_frames / 5. / 60.
    # Query-minute companion includes separately empty queries; not real time.
    qa = alarms.transpose(0, 2, 1)[empty]
    qstarts = qa & ~np.concatenate((np.zeros((len(qa), 1), bool), qa[:, :-1]), axis=1)
    result = dict(empty_pairs=int(empty.sum()), false_pairs=int(false_pairs.sum()),
        false_pair_rate=float(false_pairs.sum()/empty.sum()) if empty.any() else None,
        empty_group_sequences=int(group_empty.sum()), empty_group_frames=empty_frames,
        frame_rate_hz=5, empty_group_minutes=minutes, false_episodes=int(starts.sum()),
        false_episodes_per_simulated_empty_minute=float(starts.sum()/minutes) if minutes else None,
        false_alarm_frames=int(ga.sum()), empty_query_frames=int(qa.size),
        query_false_episodes=int(qstarts.sum()),
        false_episodes_per_empty_query_minute=float(qstarts.sum()/(qa.size/5/60)) if qa.size else None)
    for name, mask in (('all', events), ('tiny', events & tiny)):
        nn = near & mask
        n = int(nn.sum())
        result[name] = dict(events=int(mask.sum()), near=n,
            timely_count=int((timely & nn).sum()), late_count=int((detected & ~timely & nn).sum()),
            never_count=int((~detected & nn).sum()),
            timely=float((timely & nn).sum()/n) if n else None,
            late=float((detected & ~timely & nn).sum()/n) if n else None,
            never=float((~detected & nn).sum()/n) if n else None)
    return result


def paired(delta):
    d = np.asarray(delta)
    draws = np.random.default_rng(20260928).integers(len(d), size=(10000, len(d)))
    ci = np.percentile(d[draws].mean(axis=1), [2.5, 97.5])
    return dict(delta=float(d.mean()), ci95=ci.tolist(), units=len(d),
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))


def predict(a):
    import torch
    import cnh_learned_features as F
    import cnh_learned_readout as L
    import cnh_track_a_scale_evaluate as se
    F.FAMILY = a.family
    se.sensor_module.FAMILY = a.family
    models = []
    for i, m in enumerate(a.models):
        assert digest(m) == MODEL_HASHES[i], ('model_identity', m)
        net = L.Readout().to(L.DEV)
        net.load_state_dict(torch.load(m, map_location=L.DEV, weights_only=True))
        models.append(net)
    rdir = a.data/'readouts-gpu/primary-mount-10-snr6'
    fdir, pdir = a.data/'features', a.data/'predictions'
    fdir.mkdir(exist_ok=True)
    pdir.mkdir(exist_ok=True)
    bias = np.load(rdir/'bias.npy')
    files = sorted(rdir.glob('unit*.npz'))
    started = time.time()
    for i, f in enumerate(files):
        u = int(f.stem[4:])
        feature = fdir/f'unit{u:03d}.npz'
        if not feature.exists():
            np.savez_compressed(feature, **F.unit_features(a.data/'geometry', a.data/'sensor', u, bias, a.family))
        # L.load uses a directory; temporarily assemble the exact one-unit
        # dictionary with its frozen encoding without copying feature files.
        with np.load(feature) as z:
            d = dict(x=np.stack([L.squash(z['z4'].astype(np.float32)), L.squash(z['z1'].astype(np.float32))], 1),
                     sup=np.unpackbits(z['sup'], axis=-1)[..., :16].astype(bool), y=z['labels'].astype(np.float32),
                     main=z['main'], w=z['witness'], strata=z['strata'], config=z['config'], frame=z['frame'], split=str(z['split']))
        with np.load(f) as r:
            for x, y in (('labels', 'y'), ('main', 'main'), ('config', 'config'), ('frame', 'frame'), ('strata', 'strata')):
                assert np.array_equal(r[x], d[y]), (u, x)
            assert str(r['split']) == d['split'] and np.allclose(r['witness'], d['w'], equal_nan=True)
            d['S2'] = r['S2__noisy@0.75']
        predictions = []
        for net in models:
            L.predict(net, {u: d}, [u])
            predictions.append(d['NN'])
        d['NN'] = np.mean(predictions, axis=0)
        np.savez_compressed(pdir/f'unit{u:03d}.npz', **{k: d[k] for k in ('NN', 'S2', 'y', 'main', 'w', 'strata', 'config', 'frame')}, split=np.array(d['split']))
        save(pdir/f'unit{u:03d}.json', dict(feature_sha256=digest(feature), readout_sha256=digest(f), models=MODEL_HASHES))
        save(a.data/'prediction_progress.json', dict(completed=i+1, total=len(files), elapsed_s=time.time()-started))
    save(a.data/'prediction_terminal.json', dict(status='complete', units=len(files), elapsed_s=time.time()-started,
         backend=str(L.DEV), device=torch.cuda.get_device_name() if L.DEV.type == 'cuda' else 'CPU'))


def evaluate(a):
    from cnh_learned_memory_fusion import causal_ewma
    units = {}
    for f in sorted((a.data/'predictions').glob('unit*.npz')):
        u = int(f.stem[4:])
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        # Smooth float32 logits with the byte-identical frozen implementation.
        smooth = causal_ewma(d['NN'], d['config'], d['frame'], alpha=.5, window=5)
        d['A0'], d['A1'], d['A2'] = d['S2'].astype(np.float64), d['NN'].astype(np.float64), smooth.astype(np.float64)
        units[u] = d
    splits = {s: sorted(u for u, d in units.items() if str(d['split']) == s) for s in ('calib', 'audit')}
    assert set(splits['calib']).issubset(range(32)) and set(splits['audit']).issubset(range(32, 96))
    assert len(splits['calib']) >= 31 and len(splits['audit']) >= 61
    packed = {g: {s: pack(units, keys, ix) for s, keys in splits.items()} for g, ix in GROUPS}
    guards = {g: threshold(empty_peaks(packed[g]['calib'], 'A0'), .01, strict=True) for g, _ in GROUPS}
    for d in units.values():
        d['A3'] = np.empty_like(d['A2'])
        for g, ix in GROUPS:
            d['A3'][:, ix] = guard_ranking(d['A2'][:, ix], d['A0'][:, ix] > guards[g])
    ap, pairs = {}, {}
    for g, ix in GROUPS:
        per_unit = {}
        for u in splits['audit']:
            d = units[u]
            y = d['y'][d['main']][:, ix].ravel()
            if 0 < y.sum() < len(y):
                per_unit[u] = {arm: float(average_precision_score(y, d[arm][d['main']][:, ix].ravel())) for arm in ('A0', 'A1', 'A2', 'A3')}
        ap[g] = dict(units=len(per_unit), macro={arm: float(np.mean([r[arm] for r in per_unit.values()])) for arm in ('A0', 'A1', 'A2', 'A3')}, per_unit=per_unit)
        pairs[g] = {f'{x}-{y}': paired([r[x]-r[y] for r in per_unit.values()]) for x, y in (('A2','A0'), ('A1','A0'), ('A2','A1'), ('A3','A2'))}
    report = dict(scope='Frozen v5 fresh-family controlled-generator replication; EXPLORE per user; no real-world claims',
        family=a.family, split_units=splits, AP=ap, comparisons=pairs,
        primary_pass=bool(all(pairs[g]['A2-A0']['ci95'][0] > 0 for g, _ in GROUPS)),
        guard_thresholds=guards, curves={}, working_points={}, guard_only={})
    grid = np.linspace(0, .30, 121)
    for g, _ in GROUPS:
        cal, aud = packed[g]['calib'], packed[g]['audit']
        for arm in ('A0', 'A1', 'A2', 'A3'):
            base = 'A2' if arm == 'A3' else arm
            peaks = empty_peaks(cal, base)
            curve = []
            for budget in grid:
                thr = threshold(peaks, float(budget))
                alarms = aud[base] >= thr
                if arm == 'A3':
                    alarms |= aud['A0'] > guards[g]
                curve.append(dict(calib_budget=float(budget), threshold=thr, **alert_stats(aud, alarms)))
            report['curves'][f'{g}|{arm}'] = curve
            for budget in BUDGETS:
                thr = threshold(peaks, budget)
                out = {}
                for name, p in (('calib', cal), ('audit', aud)):
                    alarms = p[base] >= thr
                    if arm == 'A3':
                        alarms |= p['A0'] > guards[g]
                    out[name] = alert_stats(p, alarms)
                report['working_points'][f'{g}|{arm}|{budget:.2f}'] = dict(threshold=thr, **out)
        report['guard_only'][g] = {name: alert_stats(p, p['A0'] > guards[g]) for name, p in (('calib', cal), ('audit', aud))}
    strong = []
    for u in splits['audit']:
        path = a.data/'analysis/ceiling'/f'unit{u}.json'
        rows = json.loads(path.read_text(encoding='utf-8'))
        d = units[u]
        lookup = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(d['config'], d['frame']))}
        for r in rows:
            if r['box'] not in (1, 3, 5) or r['z_mf4'] < 5:
                continue
            i, q = lookup[(r['config'], r['frame'])], r['box']
            assert d['main'][i] and d['y'][i, q] == 1
            assert np.isclose(d['A0'][i, q], r['s2'], rtol=1e-6, atol=1e-6)
            strong.append(dict(**r, A0=float(d['A0'][i,q]), A1=float(d['A1'][i,q]), A2=float(d['A2'][i,q])))
    report['strong_BODY'] = {}
    for stratum in ('all', 'tiny'):
        for lo, hi in ((0., math.inf), (0., 1.), (1., 2.), (2., math.inf)):
            rs = [r for r in strong if (stratum == 'all' or r['stratum'] == stratum) and lo <= r['range_m'] < hi]
            key = f'{stratum}|{lo}-{hi}'
            report['strong_BODY'][key] = dict(n=len(rs), units=len({r['unit'] for r in rs}), working_points={})
            for budget in BUDGETS:
                hits = {arm: np.array([r[arm] >= report['working_points'][f'BODY|{arm}|{budget:.2f}']['threshold'] for r in rs], bool) for arm in ('A0','A1','A2')}
                guard = np.array([r['A0'] > guards['BODY'] for r in rs], bool)
                hits['A3'] = hits['A2'] | guard
                report['strong_BODY'][key]['working_points'][f'{budget:.2f}'] = dict(
                    detected={arm: int(v.sum()) for arm, v in hits.items()},
                    rescued_A2_miss=int((guard & ~hits['A2']).sum()),
                    guard_hits_A1_miss=int((guard & ~hits['A1']).sum()))
    rescued = report['strong_BODY']['all|0.0-inf']['working_points']['0.10']['rescued_A2_miss']
    guard_ok = all(report['guard_only'][g]['audit']['false_pair_rate'] <= .01 for g, _ in GROUPS)
    report['demo_guard_selection'] = dict(rescued_strong_BODY=rescued, guard_audit_at_most_one_percent_both_groups=guard_ok,
        eligible_in_simulation=bool(rescued > 0 and guard_ok), real_S2_mapping_required=True)
    save(a.data/'analysis/v5_results.json', report)
    save(a.data/'analysis/demo_thresholds.json', dict(
        A2_thresholds={g: report['working_points'][f'{g}|A2|0.10']['threshold'] for g, _ in GROUPS},
        S2_guard_thresholds=guards, recommended_demo_arm='A3' if rescued > 0 and guard_ok else 'A2',
        fps=5, source='v5 calib only', scope='simulation-derived; real-sensor unvalidated', real_S2_mapping_required=True))
    render(a.data/'analysis', report)
    print(json.dumps(dict(primary_pass=report['primary_pass'], AP={g: ap[g]['macro'] for g,_ in GROUPS}, comparisons=pairs,
                         demo_guard_selection=report['demo_guard_selection']), indent=2))


def render(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(A0='#777777', A1='#2876ba', A2='#009c79', A3='#d27c27')
    labels = dict(A0='A0  S2', A1='A1  Frozen NN', A2='A2  NN + causal smoothing', A3='A3  A2 OR high S2')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
    for col, (g, _) in enumerate(GROUPS):
        for row, subset in enumerate(('all','tiny')):
            ax = axes[row, col]
            for arm in ('A0','A1','A2','A3'):
                points = report['curves'][f'{g}|{arm}']
                ax.plot([p['false_episodes_per_simulated_empty_minute'] for p in points], [100*p[subset]['timely'] for p in points], color=colors[arm], lw=1.7, label=labels[arm])
                wp = report['working_points'][f'{g}|{arm}|0.10']['audit']
                ax.scatter(wp['false_episodes_per_simulated_empty_minute'], 100*wp[subset]['timely'], color=colors[arm], s=22)
            den = report['working_points'][f'{g}|A0|0.10']['audit']
            ax.set_title(f"{g} / {subset} near events (n={den[subset]['near']})")
            ax.set_xlabel('False alert episodes / simulated empty minute')
            ax.set_ylabel('Timely alert rate (%)')
            ax.set_ylim(0, 100)
            ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Frozen v5: thresholds from calibration, points from audit', fontsize=14)
    fig.text(.5,.012,'5 Hz; 9 evaluated frames/sequence. Empty group sequences only; overlapping queries merged.\nDots: 10% calibration empty-pair budget. Short-clip normalized rates, not real-world reminders/minute.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.96])
    fig.savefig(out/'timely_vs_false_episodes.png', dpi=180)
    fig.savefig(out/'timely_vs_false_episodes.svg')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--models', type=Path, nargs=3, required=True)
    p.add_argument('--family', default='cnh-track-a-scale-v5-20260928')
    p.add_argument('--stage', choices=('predict','evaluate','all'), default='all')
    a = p.parse_args()
    a.data = a.data.resolve()
    request = json.loads((a.data/'request.json').read_text())
    assert request['family'] == a.family
    assert json.loads((a.data/'readout_terminal.json').read_text())['status'] == 'READOUTS_READY'
    sys.path.insert(0, str(a.data/'source'))
    manifest = json.loads((a.data/'source_manifest.json').read_text())
    for name, h in manifest['sha256'].items():
        assert digest(a.data/'source'/name) == h, ('source_identity', name)
    start_hash = digest(__file__)
    contract = json.loads((a.data/'evaluation_contract.json').read_text())
    assert contract['evaluator_sha256'] == start_hash, 'evaluator differs from pre-generation contract'
    assert [digest(m) for m in a.models] == list(MODEL_HASHES)
    save(a.data/f'{a.stage}_evaluator_identity.json', dict(source_sha256=start_hash, source_file=str(Path(__file__).resolve()), models=[digest(m) for m in a.models]))
    try:
        if a.stage in ('predict','all'):
            predict(a)
        if a.stage in ('evaluate','all'):
            evaluate(a)
        assert digest(__file__) == start_hash
        save(a.data/f'{a.stage}_terminal.json', dict(status='complete', evaluator_sha256=start_hash))
    except BaseException as error:
        save(a.data/f'{a.stage}_terminal.json', dict(status='STOP', error=repr(error), evaluator_sha256=start_hash))
        raise


if __name__ == '__main__':
    main()
