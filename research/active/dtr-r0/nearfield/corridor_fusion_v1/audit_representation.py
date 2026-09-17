"""Recompute counts, event onsets and saved-model scores without any fitting."""
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

from run_representation import ARMS, OUT, DATA, read, write, sha, raw_indices


def counts(y, p):
    tp = sum(bool(a) and bool(b) for a, b in zip(y, p))
    fp = sum(not bool(a) and bool(b) for a, b in zip(y, p))
    fn = sum(bool(a) and not bool(b) for a, b in zip(y, p))
    tn = len(y)-tp-fp-fn
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, frames=len(y))


def events(cases, flags, core):
    grouped = {}
    for i, c in enumerate(cases):
        grouped.setdefault(c['episode_id'], []).append(i)
    out = []
    for episode, ii in grouped.items():
        segment = []
        for i in ii + [None]:
            positive = i is not None and cases[i]['truth'] and (not core or cases[i]['stratum'] != 'boundary')
            if positive:
                segment.append(i)
            elif segment:
                found = [j for j in segment if flags[j]]
                out.append((episode, cases[segment[0]]['time_s'],
                    cases[found[0]]['time_s']-cases[segment[0]]['time_s'] if found else None))
                segment = []
    return out


def main():
    summary = read(OUT/'summary.json')
    model_seal = read(OUT/'model-seal.json')
    cases = read(OUT/'cases.json')
    meta = read(DATA/'metadata.json')
    train = np.load(OUT/'train-features.npz')
    report = np.load(OUT/'report-features.npz')
    oof = np.load(OUT/'oof-scores.npz')
    folds = read(OUT/'freeze.json')['folds']
    selections = read(OUT/'selection.json')
    y = np.array([c['truth'] for c in cases], bool)
    clear = np.array([c['stratum'] != 'boundary' for c in cases])
    native = np.array([c['sampled_witness'] for c in cases])
    zero = np.array([c['usable_tof_returns'] == 0 for c in cases])
    masks = dict(clear=clear, strict=np.ones(288, bool), boundary=~clear,
        native_supported=native, zero_return=zero, clear_native_supported=clear & native,
        clear_zero_return=clear & zero)
    names = read(OUT/'feature-names.json')['full']
    for z in (train, report):
        np.testing.assert_array_equal(z['raw'], z['multi'][:, raw_indices(names)])
        np.testing.assert_array_equal(z['single'][:, :2217], z['multi'][:, :2217])
    extra, flags = {}, {}
    for arm in ARMS:
        cfg = model_seal['models'][arm]
        assert sha(OUT/cfg['path']) == cfg['sha256']
        model = pickle.loads((OUT/cfg['path']).read_bytes())
        ss = model.predict_proba(report[arm])[:, 1]
        np.testing.assert_array_equal(ss, [c['ablation']['scores'][arm] for c in cases])
        p = ss >= cfg['threshold']; flags[arm] = p
        np.testing.assert_array_equal(p, [c['ablation']['flags'][arm] for c in cases])
        extra[arm] = {k: counts(y[v], p[v]) for k, v in masks.items()}
        for key, vv in extra[arm].items():
            if key in summary['methods'][arm]['metrics']:
                for k, v in vv.items():
                    assert summary['methods'][arm]['metrics'][key][k] == v
        for key, core in [('core', True), ('strict_events', False)]:
            ee = events(cases, p, core)
            saved = summary['methods'][arm][key]['events']
            assert ee == [(e['episode'], e['start_s'], e['first_in_core_delay_s']) for e in saved]
            assert sum(e[2] is not None for e in ee) == summary['methods'][arm][key]['core_events_detected']
        recovered = np.full(1344, np.nan)
        for f in folds:
            path = OUT/f"{arm}-fold{f['fold']}.pkl"
            assert sha(path) == read(path.with_suffix('.json'))['sha256']
            m = pickle.loads(path.read_bytes())
            recovered[f['report']] = m.predict_proba(train[arm][f['report']])[:, 1]
        np.testing.assert_array_equal(recovered[oof['indices']], oof[arm])
        # Independent exact rational F1 comparison avoids rounding tie ambiguity.
        yy = np.array([meta[i]['truth'] for i in oof['indices']], bool)
        cc = np.array([meta[i]['stratum'] != 'boundary' for i in oof['indices']])
        best = None
        for tau in np.r_[np.nextafter(oof[arm].max(), np.inf), np.unique(oof[arm][cc])]:
            c = counts(yy[cc], oof[arm][cc] >= tau)
            from fractions import Fraction
            key = (Fraction(2*c['TP'], 2*c['TP']+c['FP']+c['FN']), -c['FP'], float(tau))
            if best is None or key > best:
                best = key
        assert best[2] == cfg['threshold'] == selections[arm]['chosen']['threshold']
    for arm in ARMS:
        extra[arm]['clear_native_supported_changes'] = dict(
            lost_TP=int(sum(clear & native & y & flags['multi'] & ~flags[arm])),
            rescued_FN=int(sum(clear & native & y & ~flags['multi'] & flags[arm])))
    write(OUT/'audit.json', dict(status='PASS', independently_recounted_frames=288,
        final_score_replay=3, oof_model_replay=18, independent_threshold_reselection=3,
        sensor_parity_rows=1632, summaries=extra,
        summary_sha256=sha(OUT/'summary.json'), limitation='Reuses saved native labels, sklearn inference and source feature mask; no new training'))
    print('PASS: 3 final / 18 OOF replays; exact thresholds, counts, events, sensor parity')


if __name__ == '__main__':
    with threadpool_limits(4):
        main()
