"""One fixed geometry-benefit selector on consumed Development caches.

Fit only on 24 old calibration units. Score the 96 old evaluation units
after persisting the final selector. No new-cohort access, expert training,
parameter search, or outcome-dependent cutoff. Geometry is selected only on
expert disagreement and predicted positive balanced-error benefit.
Fixed learned-score Platt calibration is a working-point control.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from cnh_corridor_late_fusion import ranks, threshold
from cnh_observable_shift_dev import family_metrics, paired_macro_delta, auc

ROOT = Path(__file__).resolve().parents[4]
PILOT = ROOT / 'artifacts.local/work/cnh-structure-space-20260929'
OBS = ROOT / 'artifacts.local/work/cnh-observable-shift-dev-20260930'
DEFAULT_OUT = ROOT / 'artifacts.local/work/cnh-geometry-benefit-dev-20260930'
GROUPS = ('HEAD', 'BODY')
PARAMS = dict(loss='squared_error', learning_rate=.05, max_iter=80,
              max_depth=2, max_leaf_nodes=4, min_samples_leaf=20,
              l2_regularization=5., early_stopping=False, random_state=2026093008)
FEATURES = [f'voxel_summary_{i}' for i in range(36)] + [
    'learned_logit', 'geometry_score', 'learned_calibration_margin',
    'geometry_calibration_margin', 'seed_logit_std',
    'group_body', 'learned_decision', 'geometry_decision']


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pooled(split):
    with np.load(OBS / f'pooled_{split}.npz', allow_pickle=False) as z:
        # Observation-only arrays; no labels, units-as-features, or scene truth.
        return {tuple(map(int, key)): dict(summary=z['summary'][i].copy(),
                    spread=z['seed_logits'][i].std(0))
                for i, key in enumerate(z['keys'])}


def calibration_rows():
    pooled = load_pooled('calib')
    manifest = json.loads((PILOT / 'scene_manifest.json').read_text())
    support = {(r['unit'], r['config']): r['bin'] <= 2
               for r in manifest if r['split'] == 'calib'}
    out = []
    with np.load(PILOT / 'predictions/EXTRAP.npz', allow_pickle=False) as predictions:
        for unit in range(81000, 81024):
            with np.load(PILOT / f'features/calib/unit{unit}.npz', allow_pickle=False) as z:
                k = len(z['family'])
                labels = z['labels'].reshape(k, 16, 6)[:, -1, [2, 3]]
                geometry = z['motion'][:, 2, :]
            for config in range(k):
                for q, group in enumerate(GROUPS):
                    out.append(dict(unit=unit, config=config, group=group,
                        label=int(labels[config, q]),
                        scores=dict(EXTRAP=float(predictions[f'calib|{unit}'][config, q]),
                                    PARTIAL=float(geometry[config, q])),
                        in_support=support[unit, config],
                        summary=pooled[unit, config]['summary'],
                        spread=float(pooled[unit, config]['spread'][q])))
    return out


def expert_calibration(rows):
    refs, details = {}, {}
    for group in GROUPS:
        for expert in ('EXTRAP', 'PARTIAL'):
            values = np.array([r['scores'][expert] for r in rows
                if r['group'] == group and r['label'] == 0
                and (expert == 'PARTIAL' or r['in_support'])])
            rank_cut = threshold(ranks(values, values), len(values))
            refs[group, expert] = (values, rank_cut)
            details[f'{group}|{expert}'] = dict(n_negative=len(values),
                rank_threshold=rank_cut, false_positive=int((ranks(values, values) >= rank_cut).sum()))
    return refs, details


def attach_predictions(rows, refs, verify_existing=False):
    for r in rows:
        prediction, margin = {}, {}
        for expert in ('EXTRAP', 'PARTIAL'):
            values, cut = refs[r['group'], expert]
            rank = int(ranks(values, [r['scores'][expert]])[0])
            prediction[expert] = int(rank >= cut)
            margin[expert] = (rank - cut) / len(values)
        if verify_existing and prediction != {k: r['prediction'][k] for k in prediction}:
            raise AssertionError('expert working-point reproduction mismatch')
        r['prediction'] = prediction
        r['calibration_margin'] = margin


def features(row):
    """Explicit observable allowlist; evaluator fields cannot affect output."""
    return np.r_[row['summary'], row['scores']['EXTRAP'], row['scores']['PARTIAL'],
                 row['calibration_margin']['EXTRAP'], row['calibration_margin']['PARTIAL'],
                 row['spread'], float(row['group'] == 'BODY'),
                 row['prediction']['EXTRAP'], row['prediction']['PARTIAL']]


def training_target(rows):
    """Positive target means geometry reduces expected balanced error."""
    weights = {}
    for group in GROUPS:
        rr = [r for r in rows if r['group'] == group]
        p = sum(r['label'] for r in rr)
        weights[group] = {1: len(rr) / (2 * p), 0: len(rr) / (2 * (len(rr) - p))}
    return np.array([weights[r['group']][r['label']] * (
        int(r['prediction']['EXTRAP'] != r['label'])
        - int(r['prediction']['PARTIAL'] != r['label'])) for r in rows]), weights


def fit(rows):
    target, weights = training_target(rows)
    disagreement = np.array([r['prediction']['EXTRAP'] != r['prediction']['PARTIAL'] for r in rows])
    if disagreement.sum() < 40:
        raise ValueError('too few calibration disagreements for the fixed shallow model')
    x = np.stack([features(r) for r in rows])
    model = HistGradientBoostingRegressor(**PARAMS).fit(x[disagreement], target[disagreement])
    controls, priors = {}, {}
    for group in GROUPS:
        rr = [r for r in rows if r['group'] == group]
        xx = np.array([[r['scores']['EXTRAP']] for r in rr])
        y = np.array([r['label'] for r in rr])
        controls[group] = LogisticRegression(C=1., solver='lbfgs', max_iter=1000,
                        random_state=2026093008).fit(xx, y)
        priors[group] = float(y.mean())
    receipt = dict(units=sorted({r['unit'] for r in rows}), rows=len(rows),
                   disagreements=int(disagreement.sum()), class_costs=weights,
                   benefit_positive=int((target[disagreement] > 0).sum()),
                   benefit_negative=int((target[disagreement] < 0).sum()))
    return dict(selector=model, controls=controls, priors=priors, receipt=receipt,
                feature_names=FEATURES, parameters=PARAMS)


def predict(model, rows):
    benefits = model['selector'].predict(np.stack([features(r) for r in rows]))
    for r, benefit in zip(rows, benefits):
        r['predicted_benefit'] = float(benefit)
        disagree = r['prediction']['EXTRAP'] != r['prediction']['PARTIAL']
        r['selected_geometry'] = bool(disagree and benefit > 0.)
        r['benefit_prediction'] = r['prediction']['PARTIAL'] if r['selected_geometry'] else r['prediction']['EXTRAP']
        probability = float(model['controls'][r['group']].predict_proba([[r['scores']['EXTRAP']]])[0, 1])
        # For balanced error, calibrated P(Y=1|score) is compared with P(Y=1).
        r['recalibrated_prediction'] = int(probability > model['priors'][r['group']])


def summary(rows):
    models = dict(learned=lambda r: r['prediction']['EXTRAP'],
                  geometry=lambda r: r['prediction']['PARTIAL'],
                  benefit_selector=lambda r: r['benefit_prediction'],
                  learned_recalibration=lambda r: r['recalibrated_prediction'])
    result = dict(rows=len(rows), units=len({r['unit'] for r in rows}),
                  positive=sum(r['label'] for r in rows), negative=sum(not r['label'] for r in rows))
    for name, prediction in models.items():
        result[name] = family_metrics(rows, prediction)
        if name not in ('learned', 'geometry'):
            result[name]['delta_vs_learned'] = paired_macro_delta(rows, prediction)
    result['selector_vs_recalibration'] = paired_macro_delta(
        [dict(r, prediction=dict(r['prediction'], EXTRAP=r['recalibrated_prediction'])) for r in rows],
        lambda r: r['benefit_prediction'])
    eligible = [r for r in rows if r['prediction']['EXTRAP'] != r['prediction']['PARTIAL']]
    chosen = [r for r in rows if r['selected_geometry']]
    result['routing'] = dict(disagreements=len(eligible), selected=len(chosen),
        corrected_fn=sum(r['label'] == 1 and r['prediction']['EXTRAP'] == 0 and r['benefit_prediction'] == 1 for r in chosen),
        corrected_fp=sum(r['label'] == 0 and r['prediction']['EXTRAP'] == 1 and r['benefit_prediction'] == 0 for r in chosen),
        introduced_fn=sum(r['label'] == 1 and r['prediction']['EXTRAP'] == 1 and r['benefit_prediction'] == 0 for r in chosen),
        introduced_fp=sum(r['label'] == 0 and r['prediction']['EXTRAP'] == 0 and r['benefit_prediction'] == 1 for r in chosen),
        useful_disagreements=sum(r['prediction']['PARTIAL'] == r['label'] for r in eligible),
        correct_selection=sum(r['benefit_prediction'] == r['label'] for r in chosen))
    oracle = lambda r: r['label'] if r['prediction']['EXTRAP'] != r['prediction']['PARTIAL'] else r['prediction']['EXTRAP']
    result['oracle_evaluator_only'] = family_metrics(rows, oracle)
    result['benefit_ordering_auc'] = {g: auc(
        [r['prediction']['PARTIAL'] == r['label'] for r in eligible if r['group'] == g],
        [r['predicted_benefit'] for r in eligible if r['group'] == g]) for g in GROUPS}
    return result


def main(out):
    if out.exists():
        raise FileExistsError('fresh output directory required; preserve all prior results')
    out.mkdir(parents=True)
    start = time.monotonic()
    request = dict(scope='consumed Development diagnostic; no formal or device claim',
        pid=os.getpid(), script_sha256=sha(Path(__file__)), backend='CPU/scikit-learn',
        backend_reason='TASK_NOT_GPU_SUITABLE: small cached feature table and shallow trees',
        selector_parameters=PARAMS, observable_features=FEATURES,
        cost='per-group BER inverse-class-frequency weights estimated on fit units',
        selection='expert disagreement and predicted benefit > 0, no cutoff search',
        control='per-group C=1 Platt logistic calibration; fixed probability cutoff = fit positive prevalence',
        fits='four whole-unit OOF folds, then one final fit on all 24 old calib units',
        fold='(unit-81000)//6; each held fold has six consecutive units',
        expert_operating_points='original EXTRAP supported-negative and PARTIAL all-negative calibration',
        evaluation='old 82000..82095 only, after final model persistence',
        interpretation='report whole-unit bootstrap and actual FP/FN; no evaluation-based revisions',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in [OBS/'pooled_calib.npz',
            OBS/'pooled_evaluation.npz', PILOT/'sample_ledger.jsonl', PILOT/'scene_manifest.json']})
    save(out / 'request.json', request)
    rows = calibration_rows()
    refs, expert_details = expert_calibration(rows)
    attach_predictions(rows, refs)
    folds = []
    for fold in range(4):
        train = [r for r in rows if (r['unit'] - 81000) // 6 != fold]
        held = [r for r in rows if (r['unit'] - 81000) // 6 == fold]
        assert not {r['unit'] for r in train} & {r['unit'] for r in held}
        fitted = fit(train)
        predict(fitted, held)
        folds.append(dict(fold=fold, fit=fitted['receipt'], held_units=sorted({r['unit'] for r in held})))
    crossfit = summary(rows)
    final = fit(rows)
    joblib.dump(dict(model=final, expert_references=refs), out / 'selector.joblib')
    save(out / 'fit_receipt.json', dict(final=final['receipt'], folds=folds,
                                      expert_calibration=expert_details))
    save(out / 'progress.json', dict(stage='selector_persisted_before_evaluation', elapsed_s=time.monotonic() - start))

    # Only now score the old evaluation cohort. No evaluation target enters fit.
    pooled = load_pooled('evaluation')
    evaluation = [json.loads(s) for s in (PILOT / 'sample_ledger.jsonl').read_text().splitlines()]
    manifest = {(r['unit'], r['config']): r for r in
                json.loads((PILOT / 'scene_manifest.json').read_text()) if r['split'] == 'evaluation'}
    for r in evaluation:
        q = GROUPS.index(r['group'])
        obs = pooled[r['unit'], r['config']]
        r['summary'] = obs['summary']
        r['spread'] = float(obs['spread'][q])
    attach_predictions(evaluation, refs, verify_existing=True)
    predict(final, evaluation)
    for r in evaluation:
        m = manifest[r['unit'], r['config']]
        r['outside'] = m['bin'] > 2
        r['corner'] = m['same_side'] > 0 and m['gap'] < .15
    subsets = dict(all=evaluation, out_of_support=[r for r in evaluation if r['outside']],
                   near_same_side=[r for r in evaluation if r['corner']],
                   near_same_side_out=[r for r in evaluation if r['corner'] and r['outside']])
    result = dict(scope=request['scope'], crossfit=crossfit,
                  calibration=expert_details, subsets={k: summary(v) for k, v in subsets.items()},
                  elapsed_s=time.monotonic() - start)
    save(out / 'results.json', result)
    for name, rr in [('calibration_oof', rows), ('evaluation', evaluation)]:
        serial = []
        for r in rr:
            serial.append(json.dumps({k: v for k, v in r.items() if k != 'summary'}, ensure_ascii=False))
        (out / f'{name}_ledger.jsonl').write_text('\n'.join(serial) + '\n', encoding='utf-8')
    save(out / 'terminal.json', dict(status='complete', elapsed_s=result['elapsed_s'],
                                    new_scenes=0, expert_models_trained=0, selector_recipe_count=1))
    print(json.dumps({k: {name: round(v[name]['macro_ber'], 6)
        for name in ('learned', 'geometry', 'benefit_selector', 'learned_recalibration')}
        for k, v in result['subsets'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            main(args.output)
    except BaseException as error:
        if args.output.exists() and not (args.output / 'terminal.json').exists():
            save(args.output / 'terminal.json', dict(status='failed', error=repr(error)))
        raise
