"""Frozen v5 robustness sweep (EXPLORE fast lane): SNR tier and believed-mount pitch error.

No training and no audit threshold selection. Reuses the frozen v5 geometry/sensor files,
the frozen source snapshot and the three frozen checkpoints. SNR conditions read the
stored SNR3/12 observations. Pitch conditions keep the sensor data at the true -10 deg
mount and perturb only the query geometry both readouts believe: tq @ rx(delta), i.e. a
mount-calibration error of delta degrees. Thresholds come from calib units only.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = {'snr6': (1, 0.), 'snr3': (0, 0.), 'snr12': (2, 0.), 'pitch-5': (1, -5.), 'pitch+5': (1, 5.)}
MODEL_HASHES = (
    'b7ffb180ddde81a6c28e1e7c3922d9653e948a9023ba88fea9d1b8d7a9ee0209',
    '7bafc3201a395f60790dbdfc95b834e5cea43580acd15a61611c972656e33c8d',
    '9e8296de5a0af23ecb752299c4ea1f9f729237b560c3549c6bd472be34e13083')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(data):
    source = str(Path(data)/'source')
    if source not in sys.path:
        sys.path.insert(0, source)
    if str(HERE) not in sys.path:
        sys.path.append(str(HERE))


def unit_job(job):
    data, out, u, cond, family, models = job
    data, out = Path(data), Path(out)
    target = out/f'unit{u:03d}.npz'
    if target.exists():
        return u
    setup(data)
    import torch
    import cnh_track_a_scale_evaluate as se
    import cnh_track_a_gpu_readout2 as g2
    import cnh_learned_features as F
    import cnh_learned_readout as L
    from cnh_track_a_readout import noisy_poses
    from cnh_track_a_fov import rx
    se.sensor_module.FAMILY = family
    snr_index, delta = CONDITIONS[cond]
    bias = np.load(out/'bias.npy')
    split, records, _ = se.unit_records(data/'geometry', data/'sensor', u, -10, snr_index)
    believed = np.eye(4)
    believed[:3, :3] = rx(delta)
    rows = {}
    for rec in records:
        tq = np.asarray(rec['tq']) @ believed
        noisy = noisy_poses(rec['poses'], rec['ego_seed'], dt=.2)
        s2 = g2.sequence_readouts(rec['hist'], rec['ambient'], bias, rec['poses'], tq, noisy, with_r4=False)['S2/noisy|0.75']
        z4, z1, sup = F.sequence_features(rec['hist'], rec['ambient'], bias, tq, noisy)
        # Same float16 storage round trip as the frozen v5 feature files.
        x = np.stack([L.squash(z4.astype(np.float16).astype(np.float32)),
                      L.squash(z1.astype(np.float16).astype(np.float32))], 1)
        item = dict(x=x, sup=sup.astype(bool), y=rec['labels'].astype(np.float32), main=rec['main'],
                    w=rec['witness'], S2=s2, config=np.full(len(x), rec['config']), frame=np.arange(len(x)))
        for k, v in item.items():
            rows.setdefault(k, []).append(v)
    d = {k: np.concatenate(v) for k, v in rows.items()}
    d['strata'], d['split'] = se.strata_of(records), str(split)
    predictions = []
    for m in models:
        net = L.Readout().to(L.DEV)
        net.load_state_dict(torch.load(m, map_location=L.DEV, weights_only=True))
        L.predict(net, {u: d}, [u])
        predictions.append(d['NN'])
    d['NN'] = np.mean(predictions, axis=0)
    tmp = target.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **{k: d[k] for k in ('NN', 'S2', 'y', 'main', 'w', 'strata', 'config', 'frame')},
                        split=np.array(d['split']))
    tmp.replace(target)
    return u


def generate(a, cond):
    setup(a.data)
    import cnh_track_a_scale_evaluate as se
    family = json.loads((a.data/'request.json').read_text())['family']
    se.sensor_module.FAMILY = family
    for i, m in enumerate(a.models):
        assert digest(m) == MODEL_HASHES[i], ('model_identity', m)
    out = a.out/cond
    out.mkdir(parents=True, exist_ok=True)
    snr_index, _ = CONDITIONS[cond]
    if not (out/'bias.npy').exists():
        if snr_index == 1:
            bias = np.load(a.data/'readouts-gpu/primary-mount-10-snr6/bias.npy')
        else:
            calib = [r['hist'] for u in range(32) if (a.data/'geometry'/f'unit{u:02d}'/f'unit{u:02d}.json').exists()
                     for r in se.unit_records(a.data/'geometry', a.data/'sensor', u, -10, snr_index)[1]]
            bias = np.median(np.concatenate(calib), axis=0)
        np.save(out/'bias.npy', bias)
    units = a.units or [u for u in range(96) if (a.data/'geometry'/f'unit{u:02d}'/f'unit{u:02d}.json').exists()]
    jobs = [(str(a.data), str(out), u, cond, family, [str(m) for m in a.models]) for u in units]
    started = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for i, u in enumerate(pool.map(unit_job, jobs), 1):
            print(cond, u, f'{i}/{len(jobs)}', f'{time.time()-started:.0f}s', flush=True)


def load(folder):
    from cnh_learned_memory_fusion import causal_ewma
    units = {}
    for f in sorted(Path(folder).glob('unit*.npz')):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        d['A0'], d['A1'] = d['S2'].astype(np.float64), d['NN'].astype(np.float64)
        d['A2'] = causal_ewma(d['NN'], d['config'], d['frame'], alpha=.5, window=5).astype(np.float64)
        units[int(f.stem[4:])] = d
    return units


def evaluate(a):
    setup(a.data)
    import cnh_v5_evaluate as E
    from sklearn.metrics import average_precision_score
    conds = [c for c in CONDITIONS if (a.out/c).exists() and len(list((a.out/c).glob('unit*.npz'))) >= 92]
    if 'snr6' not in conds:
        conds.insert(0, 'v5-stored')
    loaded = {c: load(a.out/c) for c in conds if c != 'v5-stored'}
    if 'v5-stored' in conds:
        loaded['v5-stored'] = load(a.data/'predictions')
    ref = 'snr6' if 'snr6' in loaded else 'v5-stored'
    report = dict(scope='EXPLORE fast-lane robustness of frozen v5 arms on the v5 audit units; controlled generator only; '
                        'calib-only thresholds; pitch = believed-mount error with true mount -10 deg', conditions={})
    ref_thr = {}
    for cond in [ref]+[c for c in conds if c != ref]:
        units = loaded[cond]
        splits = {s: sorted(u for u, d in units.items() if str(d['split']) == s) for s in ('calib', 'audit')}
        res = dict(units={s: len(v) for s, v in splits.items()}, AP={}, paired={}, working_point_10={})
        for g, ix in E.GROUPS:
            per_unit = {}
            for u in splits['audit']:
                d = units[u]
                y = d['y'][d['main']][:, ix].ravel()
                if 0 < y.sum() < len(y):
                    per_unit[u] = {arm: float(average_precision_score(y, d[arm][d['main']][:, ix].ravel()))
                                   for arm in ('A0', 'A1', 'A2')}
            res['AP'][g] = {arm: float(np.mean([r[arm] for r in per_unit.values()])) for arm in ('A0', 'A1', 'A2')}
            res['AP'][g]['units'] = len(per_unit)
            res['paired'][g] = E.paired([r['A2']-r['A0'] for r in per_unit.values()])
            cal, aud = (E.pack(units, splits[s], ix) for s in ('calib', 'audit'))
            wp = {}
            for arm in ('A0', 'A2'):
                thr = E.threshold(E.empty_peaks(cal, arm), .10)
                if cond == ref:
                    ref_thr[(g, arm)] = thr
                entry = dict(own_calib=dict(threshold=thr, **summary(E.alert_stats(aud, aud[arm] >= thr))))
                if cond != ref:
                    entry['reference_calib'] = dict(threshold=ref_thr[(g, arm)],
                                                    **summary(E.alert_stats(aud, aud[arm] >= ref_thr[(g, arm)])))
                wp[arm] = entry
            res['working_point_10'][g] = wp
        report['conditions'][cond] = res
    (a.out/'robustness_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for cond, res in report['conditions'].items():
        line = [cond]
        for g in ('HEAD', 'BODY'):
            ap, pr = res['AP'][g], res['paired'][g]
            wp = res['working_point_10'][g]
            line.append(f"{g} AP S2 {ap['A0']:.3f} A2 {ap['A2']:.3f} d {pr['delta']:+.3f} "
                        f"[{pr['ci95'][0]:+.3f},{pr['ci95'][1]:+.3f}] w{pr['wins']}/{pr['units']} | "
                        + ' '.join(f"{arm} tiny {v['own_calib']['tiny_timely']} fp {v['own_calib']['false_pair_rate']:.3f}"
                                   for arm, v in wp.items()))
        print('\n  '.join(line))


def summary(s):
    return dict(tiny_timely=f"{s['tiny']['timely_count']}/{s['tiny']['near']}", all_timely=f"{s['all']['timely_count']}/{s['all']['near']}",
                false_pair_rate=s['false_pair_rate'], false_pairs=f"{s['false_pairs']}/{s['empty_pairs']}",
                false_episodes_per_min=s['false_episodes_per_simulated_empty_minute'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--models', type=Path, nargs=3)
    p.add_argument('--conditions', nargs='*', default=[], choices=list(CONDITIONS))
    p.add_argument('--units', type=int, nargs='*')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--evaluate', action='store_true')
    a = p.parse_args()
    for cond in a.conditions:
        generate(a, cond)
    if a.evaluate:
        evaluate(a)


if __name__ == '__main__':
    main()
