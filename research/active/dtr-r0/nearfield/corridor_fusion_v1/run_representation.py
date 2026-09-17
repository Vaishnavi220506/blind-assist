"""One fixed three-arm A* representation ablation; separately resumable stages."""
import argparse
import copy
import json
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(HERE.parent), str(ROOT/'tools')]
from representation_features import extract, raw_indices
from mz143_corridor_features import extract as original_extract
from prepare_single_a import read, write, sha
from prepare_positive_v2 import CAPS
from train_single_a import PARAMS
from tolerance_eval import metric, temporal
from ccrl import pairing, local_pairs
from research_backend import BackendCandidate, DeviceObservation, select_backend

WORK = ROOT/'artifacts.local/work'
SOURCE = WORK/'corridor-public-single-20260917'
DATA = SOURCE/'a-control'
OUT = WORK/'corridor-representation-20260918'
ARMS = ('raw', 'single', 'multi')


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines()]


def bind(bindings, path, expected=None):
    actual = sha(path)
    assert expected is None or actual == expected, str(path)
    bindings[str(path)] = actual


def verify(bindings):
    for path, expected in bindings.items():
        assert sha(path) == expected, path


def features(cap, wanted, bindings, expected=None):
    """Only public rows/RGB reach the extractor; evaluator is never opened here."""
    receipt = read(cap/'receipt.json')
    bind(bindings, cap/'receipt.json')
    bind(bindings, cap/'raw.jsonl', receipt['hashes']['raw.jsonl'])
    rows = lines(cap/'raw.jsonl')
    yaw, episode = 0., None
    result, audit, timings = {}, [], []
    for row in rows:
        if row['id'] not in wanted:
            continue
        if row['episode_id'] != episode:
            yaw = 0.
        if row['imu_valid']:
            yaw += row['delta_yaw']
        episode = row['episode_id']
        bind(bindings, cap/row['rgb_path'], receipt['hashes'][row['rgb_path']])
        image = cv2.imread(str(cap/row['rgb_path']))
        start = time.perf_counter()
        multi = extract(row, image, yaw, 'multi')
        single = extract(row, image, yaw, 'single')
        timings.append(time.perf_counter()-start)
        vm = np.r_[multi['sensor'], multi['geometry']]
        vs = np.r_[single['sensor'], single['geometry']]
        if expected is not None:
            np.testing.assert_array_equal(vm, expected[row['id']])
        else:
            original = original_extract(row, image, yaw)
            np.testing.assert_array_equal(vm, np.r_[original['sensor'], original['geometry']])
        np.testing.assert_array_equal(multi['sensor'], single['sensor'])
        names = multi['sensor_names'] + multi['geometry_names']
        indices = raw_indices(names)
        assert len(indices) == 2224 and len(vm) == len(vs) == 2485
        nonmerged = int(multi['geometry'][5])
        assert single['audit']['hypothesis_count'] <= min(nonmerged, multi['audit']['hypothesis_count'])
        assert np.array_equal(vm[indices], vs[indices])
        # The first real frame per source exercises the public allowlist.
        if not result:
            changed = copy.deepcopy(row)
            changed.update(truth=not bool(row.get('truth')), native_bounds=[{'secret': 1}],
                           id='DO_NOT_USE_ID', episode_id='DO_NOT_USE_EPISODE')
            altered = extract(changed, image, yaw, 'single')
            np.testing.assert_array_equal(vs, np.r_[altered['sensor'], altered['geometry']])
        result[row['id']] = (vm[indices], vs, vm)
        audit.append(dict(id=row['id'], multi=multi['audit']['hypothesis_count'],
            single=single['audit']['hypothesis_count'], usable_nonmerged=nonmerged,
            merged=int(multi['geometry'][6]), outside_or_partial=int(multi['geometry'][7])))
        if len(result) % 48 == 0:
            print(f'features {cap.parents[2].name}: {len(result)}/{len(wanted)}', flush=True)
    assert set(result) == set(wanted)
    return result, audit, timings, names


def prepare():
    assert not OUT.exists(), 'Preserve existing run'
    assert (ROOT/'artifacts.local').resolve() == Path('F:/ba-data/blindassist-artifacts-20260805').resolve()
    OUT.mkdir(parents=True)
    bindings = {}
    ds = read(DATA/'data-seal.json')
    bind(bindings, DATA/'data-seal.json')
    for path, h in ds['bindings'].items():
        bind(bindings, Path(path), h)
    for name, h in ds['outputs'].items():
        bind(bindings, DATA/name, h)
    for path in [Path(__file__), HERE/'representation_features.py', HERE/'REPRESENTATION_PROTOCOL_20260918.md',
                 HERE/'train_single_a.py', HERE/'ccrl.py', HERE/'tolerance_eval.py',
                 HERE.parent/'mz143_corridor_features.py', HERE.parent/'mz136_boundary_geometry.py',
                 HERE.parent/'mz125_observable_correction.py', HERE.parent/'mz115_spatial_allocation.py']:
        bind(bindings, path)
    bind(bindings, DATA/'freeze.json')
    bind(bindings, DATA/'selection.json')
    bind(bindings, DATA/'oof-scores.npz', read(DATA/'selection.json')['oof_sha256'])
    bind(bindings, DATA/'retrainedA.pkl', read(DATA/'threshold.json')['model_sha256'])
    meta = read(DATA/'metadata.json')
    z = np.load(DATA/'features.npz')
    ids = [m['id'] for m in meta]
    assert list(z['ids']) == ids and z['base'].shape == (1344, 2485)
    folds = read(DATA/'freeze.json')['folds']
    assert read(DATA/'freeze.json')['parameters'] == PARAMS
    write(OUT/'freeze.json', dict(arms=ARMS, params=PARAMS, folds=folds, max_fits=21,
        fit_ids=ids, inputs=bindings, protocol_sha256=sha(HERE/'REPRESENTATION_PROTOCOL_20260918.md'),
        backend_reason='GPU_BACKEND_UNAVAILABLE', report_authority='CONSUMED_DEVELOPMENT',
        threshold='ALL1152_GROUP_OOF_CLEAR_F1_FEWER_FP_HIGHER_THRESHOLD'))
    expected = dict(zip(ids, z['base']))
    values, audits, timings = {}, [], []
    start = time.perf_counter()
    for cohort, stem in CAPS.items():
        wanted = {m['id'] for m in meta if m['cohort'] == cohort}
        cap = WORK/stem/'source/returned-v1/capture-v1'
        v, a, t, names = features(cap, wanted, bindings, expected)
        values.update(v); audits.extend(a); timings.extend(t)
    arrays = {arm: np.stack([values[i][k] for i in ids]).astype(z['base'].dtype)
              for k, arm in enumerate(ARMS)}
    np.testing.assert_array_equal(arrays['multi'], z['base'])
    np.savez_compressed(OUT/'train-features.npz', ids=ids, **arrays)
    write(OUT/'feature-names.json', dict(full=names, raw=[names[i] for i in raw_indices(names)]))
    write(OUT/'feature-audit.json', dict(rows=audits, original_multi_parity_rows=1344,
        sensor_parity_rows=1344, original_dtype=str(z['base'].dtype),
        feature_seconds=sum(timings), wall_seconds=time.perf_counter()-start))
    verify(bindings)
    write(OUT/'feature-seal.json', dict(inputs=bindings, outputs={n: sha(OUT/n)
        for n in ('train-features.npz', 'feature-names.json', 'feature-audit.json')}))


def select(scores, y, clear):
    curve = []
    for threshold in np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores[clear])]:
        mm = metric(y[clear], scores[clear] >= threshold)
        curve.append(dict(threshold=float(threshold), **mm))
    return max(curve, key=lambda v: (v['f1'], -v['FP'], v['threshold'])), curve


def fit():
    assert not (OUT/'model-seal.json').exists()
    freeze = read(OUT/'freeze.json')
    fs = read(OUT/'feature-seal.json')
    verify(fs['inputs'])
    for n, h in fs['outputs'].items():
        assert sha(OUT/n) == h
    meta = read(DATA/'metadata.json')
    z = np.load(OUT/'train-features.npz')
    y = np.array([m['truth'] for m in meta], bool)
    report = np.array([i for i, m in enumerate(meta) if m['cohort'] != 'anchor'])
    clear = np.array([meta[i]['stratum'] != 'boundary' for i in report])
    select_backend('model-inference', cpu=BackendCandidate('sklearn-hgb', 'cpu',
        lambda: pickle.loads((DATA/'retrainedA.pkl').read_bytes()).predict_proba(z['multi'][:1]),
        lambda _: DeviceObservation('cpu', 'host CPU', 'sklearn '+sklearn.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE', record_path=OUT/'backend.json')
    selections, checkpoints, score_arrays = {}, {}, {}
    for arm in ARMS:
        start = time.perf_counter()
        x = z[arm]
        scores = np.full(len(y), np.nan)
        for f in freeze['folds']:
            fi, ri = f['fit'], f['report']
            assert not {meta[i]['group'] for i in fi} & {meta[i]['group'] for i in ri}
            checkpoint = OUT/f"{arm}-fold{f['fold']}.pkl"
            receipt = checkpoint.with_suffix('.json')
            if receipt.exists():
                rec = read(receipt)
                assert rec['sha256'] == sha(checkpoint) and rec['feature_sha256'] == sha(OUT/'train-features.npz')
                model = pickle.loads(checkpoint.read_bytes())
            else:
                assert not checkpoint.exists(), 'Incomplete fit: preserve and diagnose'
                tick = time.perf_counter()
                model = HistGradientBoostingClassifier(**PARAMS).fit(x[fi], y[fi])
                checkpoint.write_bytes(pickle.dumps(model))
                write(receipt, dict(sha256=sha(checkpoint), seconds=time.perf_counter()-tick,
                    feature_sha256=sha(OUT/'train-features.npz'), fit=fi, report=ri))
            scores[ri] = model.predict_proba(x[ri])[:, 1]
            print(f"fit {arm} fold{f['fold']} complete", flush=True)
        assert np.isfinite(scores[report]).all()
        chosen, curve = select(scores[report], y[report], clear)
        if arm == 'multi':
            saved = np.load(DATA/'oof-scores.npz')
            np.testing.assert_array_equal(saved['indices'], report)
            np.testing.assert_array_equal(saved['scores'], scores[report])
            assert chosen['threshold'] == read(DATA/'threshold.json')['threshold']
        path = OUT/f'{arm}-final.pkl'
        receipt = path.with_suffix('.json')
        if receipt.exists():
            assert read(receipt)['sha256'] == sha(path)
        else:
            assert not path.exists()
            model = HistGradientBoostingClassifier(**PARAMS).fit(x, y)
            path.write_bytes(pickle.dumps(model))
            write(receipt, dict(sha256=sha(path), rows=len(y), features=x.shape[1]))
        checkpoints[arm] = dict(path=path.name, sha256=sha(path), threshold=chosen['threshold'],
            features=x.shape[1], total_fit_seconds=time.perf_counter()-start)
        selections[arm] = dict(chosen=chosen, curve=curve)
        score_arrays[arm] = scores[report]
    np.savez_compressed(OUT/'oof-scores.npz', indices=report, ids=z['ids'][report], **score_arrays)
    write(OUT/'selection.json', selections)
    write(OUT/'model-seal.json', dict(models=checkpoints, selection_sha256=sha(OUT/'selection.json'),
        oof_sha256=sha(OUT/'oof-scores.npz'), feature_seal_sha256=sha(OUT/'feature-seal.json'),
        freeze_sha256=sha(OUT/'freeze.json'), multi_oof_exact_parity=True))


def predict():
    assert not (OUT/'prediction-seal.json').exists()
    seal = read(OUT/'model-seal.json')
    verify(read(OUT/'freeze.json')['inputs'])
    bindings = {}
    cap = SOURCE/'source/returned-v1/capture-v1'
    raw = lines(cap/'raw.jsonl')
    ids = [r['id'] for r in raw]
    assert len(ids) == 288 and not set(ids) & set(read(OUT/'freeze.json')['fit_ids'])
    values, audits, timings, names = features(cap, set(ids), bindings)
    arrays = {arm: np.stack([values[i][k] for i in ids]) for k, arm in enumerate(ARMS)}
    scores = {}
    for arm in ARMS:
        cfg = seal['models'][arm]
        bind(bindings, OUT/cfg['path'], cfg['sha256'])
        model = pickle.loads((OUT/cfg['path']).read_bytes())
        scores[arm] = model.predict_proba(arrays[arm])[:, 1]
    np.savez_compressed(OUT/'report-features.npz', ids=ids, **arrays)
    predictions = [dict(id=identifier, scores={a: float(scores[a][i]) for a in ARMS},
        flags={a: bool(scores[a][i] >= seal['models'][a]['threshold']) for a in ARMS}) for i, identifier in enumerate(ids)]
    write(OUT/'predictions.json', predictions)
    write(OUT/'report-feature-audit.json', dict(rows=audits, original_multi_parity_rows=288,
        sensor_parity_rows=288, feature_seconds=sum(timings)))
    verify(bindings)
    write(OUT/'prediction-seal.json', dict(model_seal_sha256=sha(OUT/'model-seal.json'),
        feature_sha256=sha(OUT/'report-features.npz'), predictions_sha256=sha(OUT/'predictions.json'),
        inputs=bindings, labels_joined=False, authority='CONSUMED_DEVELOPMENT'))


def summarize(rows, scores, flags, pairs):
    y = np.array([m['truth'] for m in rows], bool)
    states = np.array([m['stratum'] for m in rows])
    clear = states != 'boundary'
    masks = dict(clear=clear, strict=np.ones(len(y), bool), boundary=~clear)
    masks.update({f: clear & np.array([m['family'] == f for m in rows]) for f in sorted({m['family'] for m in rows})})
    if 'sampled_witness' in rows[0]:
        masks['native_supported'] = np.array([m['sampled_witness'] for m in rows])
        masks['zero_return'] = np.array([m['usable_tof_returns'] == 0 for m in rows])
    methods = {}
    reference = flags['multi']
    for arm, p in flags.items():
        methods[arm] = dict(metrics={k: metric(y[v], p[v]) for k, v in masks.items()},
            coverage=float(clear.mean()), core=temporal(rows, states, p),
            strict_events=temporal(rows, np.where(y, 'positive', 'negative'), p),
            changes={k: dict(rescued_FN=int(sum(v & y & ~reference & p)),
                added_FP=int(sum(v & ~y & ~reference & p)), lost_TP=int(sum(v & y & reference & ~p)),
                removed_FP=int(sum(v & ~y & reference & ~p))) for k, v in masks.items()},
            pair_count=len(pairs), pair_order_accuracy=float(np.mean(scores[arm][pairs[:, 0]] > scores[arm][pairs[:, 1]])),
            pair_probability_margin_mean=float(np.mean(scores[arm][pairs[:, 0]]-scores[arm][pairs[:, 1]])))
        for label, mask in masks.items():
            methods[arm].setdefault('lost_TP_ids', {})[label] = [rows[i]['id'] for i in np.flatnonzero(mask & y & reference & ~p)]
    for arm, entry in methods.items():
        for kind in ('core', 'strict_events'):
            old = {(e['episode'], e['start_s']): e for e in methods['multi'][kind]['events']}
            entry[kind+'_changes'] = [dict(episode=e['episode'], start_s=e['start_s'],
                baseline=old[(e['episode'], e['start_s'])]['first_in_core_delay_s'], current=e['first_in_core_delay_s'])
                for e in entry[kind]['events'] if old[(e['episode'], e['start_s'])]['first_in_core_delay_s'] != e['first_in_core_delay_s']]
    return methods


def frontiers(rows, scores, reference):
    """Descriptive report-only curves; no model or saved threshold is changed."""
    y = np.array([m['truth'] for m in rows], bool)
    clear = np.array([m['stratum'] != 'boundary' for m in rows])
    groups = {}
    for i, m in enumerate(rows):
        groups.setdefault(m['episode_id'], []).append(i)
    event_indices = {}
    for name, positive in [('core', clear & y), ('strict', y)]:
        segments = []
        for ii in groups.values():
            current = []
            for i in ii:
                if positive[i]:
                    current.append(i)
                elif current:
                    segments.append(current); current = []
            if current:
                segments.append(current)
        event_indices[name] = segments
    result = {}
    for arm, ss in scores.items():
        maxima = {k: np.array([max(ss[ii]) for ii in vv]) for k, vv in event_indices.items()}
        curve = []
        for tau in np.r_[np.nextafter(ss.max(), np.inf), np.unique(ss)]:
            p = ss >= tau
            curve.append(dict(threshold=float(tau), **metric(y[clear], p[clear]),
                core_events=int(sum(maxima['core'] >= tau)), strict_events=int(sum(maxima['strict'] >= tau))))
        fp_budget = reference['metrics']['clear']['FP']
        at_fp = max((c for c in curve if c['FP'] <= fp_budget), key=lambda c: (c['TP'], -c['FP'], c['threshold']))
        target_core = reference['core']['core_events_detected']
        target_strict = reference['strict_events']['core_events_detected']
        at_events = min((c for c in curve if c['core_events'] >= target_core and c['strict_events'] >= target_strict),
            key=lambda c: (c['FP'], -c['TP'], -c['threshold']))
        result[arm] = dict(at_most_baseline_clear_FP=at_fp,
            at_least_baseline_core_and_strict_events=at_events, curve=curve)
    return result


def evaluate():
    assert not (OUT/'completion.json').exists()
    ps = read(OUT/'prediction-seal.json')
    assert sha(OUT/'predictions.json') == ps['predictions_sha256']
    assert sha(OUT/'report-features.npz') == ps['feature_sha256']
    verify(ps['inputs'])
    cases_path = SOURCE/'confirmation/cases.json'
    cases = read(cases_path)
    predictions = read(OUT/'predictions.json')
    prior = read(SOURCE/'confirmation/predictions.json')
    assert sha(SOURCE/'confirmation/predictions.json') == read(SOURCE/'confirmation/prediction-seal.json')['predictions_sha256']
    assert [r['id'] for r in cases] == [r['id'] for r in predictions] == [r['id'] for r in prior]
    scores = {a: np.array([p['scores'][a] for p in predictions]) for a in ARMS}
    flags = {a: np.array([p['flags'][a] for p in predictions]) for a in ARMS}
    np.testing.assert_array_equal(scores['multi'], [p['control_score'] for p in prior])
    cap = SOURCE/'source/returned-v1/capture-v1'
    receipt = read(cap/'receipt.json')
    assert sha(cap/'spec.json') == receipt['spec_sha256']
    assert sha(cap/'evaluator.jsonl') == receipt['hashes']['evaluator.jsonl']
    from tolerance_eval import geometry, classify
    from mz171_return_labels import make_witness_labels
    for row, native, case in zip(lines(cap/'raw.jsonl'), lines(cap/'evaluator.jsonl'), cases):
        assert row['id'] == native['id'] == case['id']
        g = geometry(native)
        assert g['strict'] == case['truth'] and classify(g, .05) == case['stratum']
        witness = make_witness_labels(row, native)
        assert bool(((witness['target'][:128] > 0) & witness['known'][:128]).any()) == case['sampled_witness']
    frames = {f['id']: f for f in read(cap/'spec.json')['frames']}
    pairs, _, rejected = pairing(cases, frames)
    methods = summarize(cases, scores, flags, pairs)
    meta = read(DATA/'metadata.json')
    train_frames = {}
    for cohort, stem in CAPS.items():
        p = WORK/stem/'source/returned-v1/capture-v1/spec.json'
        assert sha(p) == read(p.with_name('receipt.json'))['spec_sha256']
        train_frames.update({f['id']: f for f in read(p)['frames'] if cohort != 'anchor' or f['split'] == 'train'})
    tp, _, _ = pairing(meta, train_frames)
    oof = np.load(OUT/'oof-scores.npz')
    ii = oof['indices']
    thresholds = read(OUT/'model-seal.json')['models']
    oof_report = summarize([meta[i] for i in ii], {a: oof[a] for a in ARMS},
        {a: oof[a] >= thresholds[a]['threshold'] for a in ARMS}, local_pairs(tp, ii))
    curves = frontiers(cases, scores, methods['multi'])
    write(OUT/'descriptive-frontiers.json', dict(authority='POSTHOC_REPORT_DIAGNOSTIC_NOT_CALIBRATION', methods=curves))
    write(OUT/'cases.json', [dict(**m, ablation=p) for m, p in zip(cases, predictions)])
    write(OUT/'summary.json', dict(authority='CONSUMED_CONTROLLED_DEVELOPMENT', methods=methods,
        oof_selection=oof_report, report_pair_count=len(pairs), rejected_pairs=rejected,
        thresholds={a: thresholds[a]['threshold'] for a in ARMS},
        model_fits=21, original_A_star_unchanged=True, multi_exact_score_parity=True,
        matched_diagnostics={a: {k: v for k, v in curves[a].items() if k != 'curve'} for a in ARMS},
        cases_source_sha256=sha(cases_path), evaluator_sha256=sha(cap/'evaluator.jsonl')))
    verify(read(OUT/'feature-seal.json')['inputs'])
    write(OUT/'completion.json', dict(status='PASS', summary_sha256=sha(OUT/'summary.json'),
        prediction_seal_sha256=sha(OUT/'prediction-seal.json'),
        feature_multi_exact_parity_rows=1632, native_sensor_parity_rows=1632,
        resources='No persistent services; process exits and releases all model memory'))
    print(json.dumps({a: m['metrics']['clear'] for a, m in methods.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'fit', 'predict', 'evaluate'])
    args = parser.parse_args()
    with threadpool_limits(4):
        globals()[args.stage]()
