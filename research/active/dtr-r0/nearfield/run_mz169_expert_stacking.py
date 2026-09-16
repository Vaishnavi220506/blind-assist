"""Frozen experts, disjoint meta training, matched mean, staged consumed transfer."""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import time

import cv2
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from mz169_expert_stacking import METHOD, fit_model, predict_score, mean_score
from mz145_causal_confirmation import fit_onset, predict, previous_scores
from run_mz143_corridor_evidence import (
    ROOT, CODE, CAP, read, sha as path_sha, write, truth, selected_jsonl,
    public_observations, augmented_score, native_account, operating_threshold,
    local_dependencies, pairs_for,
)
from evaluate_mz136_corridor_pair import retention, pair_metrics
from run_mz148_background_residual import features as temporal_features
from research_backend import BackendCandidate, DeviceObservation, select_backend

WORK = ROOT/'artifacts.local/work/mz169-expert-stacking-20260916'
STATIC = ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
ONSET = ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'
TEMPORAL = ROOT/'artifacts.local/work/mz148-background-residual-20260916/run-v1'
REFLECTION = ROOT/'artifacts.local/work/mz159-reflection-invariance-20260916/run-v1'
META = ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'
TRANSFER = ROOT/'artifacts.local/work/mz158-crossview-agreement-20260916'
EXPERTS = ('mz145', 'unregistered', 'compensated', 'reflection')
ARMS = ('stack', 'mean')


def sha(path):
    return path_sha(Path(path))


class Inputs:
    def __init__(self):
        self.bindings = {}
        self.comparisons = 0

    def check(self, path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
            self.comparisons += 1
        if str(path) in self.bindings:
            assert digest == self.bindings[str(path)]
        self.bindings[str(path)] = digest
        return digest

    def json(self, path, expected=None):
        self.check(path, expected)
        return read(path)

    def model_seal(self, folder, expected=None):
        seal = self.json(folder/'model-seal.json', expected)
        freeze = self.json(folder/'freeze.json', seal['freeze_sha256'])
        for path, digest in freeze.get('inputs', {}).items():
            self.check(path, digest)
        for arm, digest in seal['models'].items():
            self.check(folder/(arm+'.pkl'), digest)
        if 'oof_sha256' in seal:
            self.check(folder/'oof.npz', seal['oof_sha256'])
        if 'train_summary_sha256' in seal:
            self.check(folder/'train-summary.json', seal['train_summary_sha256'])
        return seal

    def saved(self, folder, label):
        seal = self.json(folder/(label+'-prediction-seal.json'))
        self.model_seal(folder, seal['model_seal_sha256'])
        feature = self.json(folder/(label+'-feature-seal.json'), seal['feature_seal_sha256'])
        self.check(folder/(label+'-features.npz'), feature['feature_sha256'])
        if 'audit_sha256' in feature:
            self.check(folder/(label+'-feature-audit.json'), feature['audit_sha256'])
        if 'input_seal_sha256' in seal:
            linked = self.json(folder/(label+'-input-seal.json'), seal['input_seal_sha256'])
            for path, digest in linked['inputs'].items():
                self.check(path, digest)
        return self.json(folder/(label+'-predictions.json'), seal['predictions_sha256'])


def capture_panel(name, inputs):
    capture = CAP if name == 'dev' else (META if name == 'mz146' else TRANSFER)/'source/returned-v1/capture-v1'
    receipt = inputs.json(capture/'receipt.json')
    assert receipt['status'] == 'PASS'
    spec = inputs.json(capture/'spec.json', receipt['spec_sha256'])
    for filename in ('raw.jsonl', 'evaluator.jsonl'):
        inputs.check(capture/filename, receipt['hashes'][filename])
    split = 'dev' if name == 'dev' else 'confirmation'
    ids = {f['id'] for f in spec['frames'] if f['split'] == split}
    rows = public_observations(selected_jsonl(capture/'raw.jsonl', ids))
    assert len(rows) == (48 if name == 'dev' else 288)
    assert len(ids) == len(rows) and {r['id'] for r in rows} == ids
    return dict(name=name, capture=capture, spec=spec, split=split, rows=rows, ids=ids)


def existing_scores(panel, inputs, out, started):
    name = panel['name']; rows = panel['rows']
    reference = inputs.saved(REFLECTION, name)
    assert [p['id'] for p in reference] == [r['id'] for r in rows]
    if name != 'mz158':
        temporal = inputs.saved(TEMPORAL, 'dev' if name == 'dev' else 'mz146-consumed')
        if name == 'mz146':
            linked = inputs.json(TEMPORAL/'second-input-seal.json')
            for path, digest in linked['inputs'].items():
                inputs.check(path, digest)
        assert [p['id'] for p in temporal] == [r['id'] for r in rows]
        temporal_scores = {a:np.array([p['scores'][a] for p in temporal]) for a in ('unregistered', 'compensated')}
    else:
        assert time.perf_counter()-started < 900
        # New output sealing alone does not authenticate the old compensated path.
        for path, digest in read(TEMPORAL/'freeze.json')['sources'].items():
            inputs.check(path, digest)
        folder = TRANSFER/'learned-v1'
        seal = inputs.json(folder/'prediction-seal.json')
        linked = inputs.json(folder/'input-seal.json', seal['input_seal_sha256'])
        for path, digest in linked.get('inputs', {}).items():
            inputs.check(path, digest)
        saved = inputs.json(folder/'predictions.json', seal['predictions_sha256'])
        inputs.check(folder/'features.npz', seal['context']['features_sha256'])
        assert [p['id'] for p in saved] == [r['id'] for r in rows]
        static = np.load(folder/'features.npz')['static']
        assert static.shape == (288, 2485)
        # Its per-frame check expires at min(feature_start+600, run_start+900).
        feature_clock = min(time.perf_counter(), started+300.)
        temporal_values = temporal_features(rows, panel['capture'], out, 'mz158-new-temporal', feature_clock)
        temporal_scores = {}
        for arm in ('unregistered', 'compensated'):
            model = pickle.loads((TEMPORAL/(arm+'.pkl')).read_bytes())
            temporal_scores[arm] = model.predict_proba(np.c_[static, temporal_values[arm]])[:, 1]
        np.testing.assert_array_equal(temporal_scores['unregistered'], [p['scores']['raw_change'] for p in saved])
        np.testing.assert_array_equal([p['scores']['mz145'] for p in reference], [p['scores']['static'] for p in saved])
        write(out/'mz158-adapter-parity.json', dict(unregistered_scores_bitwise_equal=True,
            static_scores_bitwise_equal=True, frames=288,
            feature_seal_sha256=sha(out/'mz158-new-temporal-feature-seal.json'),
            backend='OpenCV '+cv2.__version__+' / sklearn '+sklearn.__version__+' CPU',
            cpu_reason='GPU_BACKEND_UNAVAILABLE', seconds_since_start=time.perf_counter()-started))
    score_columns = dict(mz145=np.array([p['scores']['mz145'] for p in reference]),
        reflection=np.array([p['scores']['reflection'] for p in reference]), **temporal_scores)
    x = np.column_stack([score_columns[a] for a in EXPERTS])
    old = inputs.json(ONSET/'onset-seal.json')
    cuts = dict(mz145={k:old[k] for k in ('low', 'high')},
        reflection=read(REFLECTION/'model-seal.json')['cutoffs']['reflection'],
        **read(TEMPORAL/'model-seal.json')['cutoffs'])
    flags = {a:predict(rows, score_columns[a], **cuts[a]) for a in EXPERTS}
    for a in ('mz145', 'reflection'):
        np.testing.assert_array_equal(flags[a], [p['flags'][a] for p in reference])
    if name != 'mz158':
        for a in ('unregistered', 'compensated'):
            np.testing.assert_array_equal(flags[a], [p['flags'][a] for p in temporal])
    flags['baseline'] = np.array([p['flags']['baseline'] for p in reference], bool)
    panel.update(x=x, flags=flags)
    return panel


def labels(panel):
    es = selected_jsonl(panel['capture']/'evaluator.jsonl', panel['ids'])
    assert [e['id'] for e in es] == [r['id'] for r in panel['rows']]
    return es, np.array([truth(e) for e in es], bool)


def reports_for(panel, es, y, flags):
    return {a:augmented_score(panel['rows'], es, y, p.astype(float), .5,
        panel['flags']['baseline'], panel['spec'], panel['split']) for a,p in flags.items()}


def compare(rows, y, flags, reports, reference):
    rr = retention(reports['stack'], reports[reference])
    lost = [r['id'] for i,r in enumerate(rows) if y[i] and flags[reference][i] and not flags['stack'][i]]
    return dict(lost_true_frames=lost, retention=rr,
        no_true_loss=not lost,
        no_event_delay=all(v['relative_delay_s'] is not None and v['relative_delay_s'] <= 1e-9 for v in rr['per_event_delta']))


def evaluate(panel, inputs, out, model, cuts):
    name = panel['name']; rows = panel['rows']; x = panel['x']
    scores = dict(stack=predict_score(model, x), mean=mean_score(x))
    flags = dict(panel['flags'])
    flags.update({a:predict(rows, scores[a], **cuts[a]) for a in ARMS})
    write(out/(name+'-predictions.json'), [dict(id=r['id'], episode_id=r['episode_id'], time_s=r['time_s'],
        expert_scores=x[i].tolist(), scores={a:float(v[i]) for a,v in scores.items()},
        flags={a:bool(v[i]) for a,v in flags.items()}) for i,r in enumerate(rows)])
    write(out/(name+'-prediction-seal.json'), dict(predictions_sha256=sha(out/(name+'-predictions.json')),
        model_seal_sha256=sha(out/'model-seal.json'), inputs=dict(inputs.bindings),
        authority='CONSUMED_PREDICTIONS_SEALED_BEFORE_SELECTED_EVALUATOR_PARSE'))
    es, y = labels(panel)
    reports = reports_for(panel, es, y, flags)
    for a, s in scores.items():
        reports[a]['score_pair_ordering'] = pair_metrics(y, s, flags[a], pairs_for(rows, panel['spec'], panel['split']))
    native = {a:native_account(rows, es, flags[a], flags['baseline']) for a in ARMS}
    for n in native.values(): n['radar_native_lineage'] = 'NOT_EVALUABLE'
    write(out/(name+'-native-contributors.json'), native)
    comparisons = {a:compare(rows, y, flags, reports, a) for a in ('baseline', 'mz145', 'mean')}
    candidate = reports['stack']; checks = {}
    for arm, comparison in comparisons.items():
        checks[arm+'_true_frames'] = comparison['no_true_loss']
        checks[arm+'_event_onsets'] = comparison['no_event_delay']
        checks[arm+'_false_segments'] = candidate['events']['false_alert_segments'] <= reports[arm]['events']['false_alert_segments']
    checks['mean_fp_noninferior'] = candidate['metrics']['FP'] <= reports['mean']['metrics']['FP']
    checks['baseline_family_fp'] = all(candidate['families'][f]['FP'] <= reports['baseline']['families'][f]['FP'] for f in candidate['families'])
    checks['native_retention'] = not native['stack']['nonalert_with_native_corridor_contributors']
    if name == 'dev':
        checks['dev_counts'] = candidate['metrics']['TP'] == 24 and candidate['metrics']['FP'] <= 3
        checks['dev_segments'] = candidate['events']['false_alert_segments'] <= 2
    else:
        checks['baseline_fp_reduction'] = candidate['metrics']['FP'] <= .7*reports['baseline']['metrics']['FP']
        checks['mz145_fp_reduction'] = candidate['metrics']['FP'] < reports['mz145']['metrics']['FP']
    result = dict(reports=reports, comparisons=comparisons, checks=checks, passed=all(checks.values()),
        strict_mean_fp_gain=candidate['metrics']['FP'] < reports['mean']['metrics']['FP'],
        authority='CONSUMED_DEVELOPMENT_NOT_FRESH')
    write(out/(name+'-summary.json'), result)
    print(json.dumps(dict(stage='evaluated', panel=name, passed=result['passed'],
        metrics={a:v['metrics'] for a,v in reports.items()}, checks=checks)), flush=True)
    return result


def run(out):
    out = out.resolve(); assert out.is_relative_to(WORK.resolve()) and not out.exists()
    out.mkdir(parents=True); started = time.perf_counter(); inputs = Inputs()
    sources = local_dependencies(__file__)
    sources.update(local_dependencies(CODE/'mz169_expert_stacking.py'))
    sources[str(CODE/'MZ169_PROTOCOL_20260916.md')] = sha(CODE/'MZ169_PROTOCOL_20260916.md')
    for p in sources:
        dest = out/'source-snapshot'/Path(p).relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    write(out/'recipe-freeze.json', dict(method=METHOD, sources=sources, experts=EXPERTS,
        meta_training='MZ146288', budget_seconds=900, optional_temporal_budget_seconds=600,
        original_test_predictions_raw_evaluator_decoded=False,
        authority='FIXED_RECIPE_BEFORE_META_SCORES_LABELS_AND_FIT'))
    select_backend('batch-tensor', cpu=BackendCandidate('sklearn-cpu', 'cpu',
        lambda:mean_score(np.full((288, 4), .5)),
        lambda _:DeviceObservation('cpu', 'host CPU', 'sklearn '+sklearn.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE', record_path=out/'backend.json',
        capabilities={'reason':'288x4 fixed logistic fit; no tensor image training; optional OpenCV expert is CPU-only'})
    inputs.model_seal(TEMPORAL)
    meta = existing_scores(capture_panel('mz146', inputs), inputs, out, started)
    es, y = labels(meta); rows = meta['rows']; x = meta['x']
    by_id = {f['id']:f for f in meta['spec']['frames']}
    groups = sorted({by_id[r['id']]['scene_group'] for r in rows})
    assert len(groups) == 24
    mapping = {g:i % 4 for i,g in enumerate(groups)}
    fold = np.array([mapping[by_id[r['id']]['scene_group']] for r in rows])
    assert all(np.sum(fold == f) == 72 for f in range(4))
    for episode in {r['episode_id'] for r in rows}:
        assert len({int(fold[i]) for i,r in enumerate(rows) if r['episode_id'] == episode}) == 1
    write(out/'meta-folds.json', [dict(id=r['id'], scene_group=by_id[r['id']]['scene_group'],
        fold=int(fold[i])) for i,r in enumerate(rows)])
    write(out/'fit-freeze.json', dict(recipe_sha256=sha(out/'recipe-freeze.json'), inputs=dict(inputs.bindings),
        folds_sha256=sha(out/'meta-folds.json'), meta_ids=[r['id'] for r in rows],
        required_mask='truth AND (saved MZ129 OR saved MZ145)', authority='BEFORE_FIRST_META_FIT'))
    tick = time.perf_counter(); oof = np.empty(len(rows)); fold_details = []
    for f in range(4):
        assert time.perf_counter()-started < 900
        model = fit_model(x[fold != f], y[fold != f])
        oof[fold == f] = predict_score(model, x[fold == f])
        (out/('fold'+str(f)+'.pkl')).write_bytes(pickle.dumps(model))
        fold_details.append(dict(fold=f, fit_rows=int(np.sum(fold != f)), held_rows=int(np.sum(fold == f))))
    scores = dict(stack=oof, mean=mean_score(x)); reference = meta['flags']['baseline'] | meta['flags']['mz145']
    cuts = {}; supports = {}; flags = dict(meta['flags'])
    for a, s in scores.items():
        low = operating_threshold(s, y, reference); high = fit_onset(rows, s, y, reference, low)
        cuts[a] = dict(low=low, high=high); flags[a] = predict(rows, s, low, high)
        required = y & reference; strong = required & (previous_scores(rows, s) < low)
        supports[a] = dict(low=[rows[i]['id'] for i in np.where(required & (s == low))[0]],
            high=[rows[i]['id'] for i in np.where(strong & (s == high))[0]], required_frames=int(required.sum()))
    model = fit_model(x, y); (out/'stack.pkl').write_bytes(pickle.dumps(model))
    np.savez_compressed(out/'meta-scores.npz', experts=x, truth=y, fold=fold, **scores)
    meta_reports = reports_for(meta, es, y, flags)
    for a, s in scores.items():
        meta_reports[a]['score_pair_ordering'] = pair_metrics(y, s, flags[a], pairs_for(rows, meta['spec'], meta['split']))
    write(out/'meta-training-summary.json', dict(reports=meta_reports, cutoffs=cuts, support=supports,
        fold_details=fold_details, fitting_seconds=time.perf_counter()-tick,
        authority='META_TRAINING_CALIBRATION_DIAGNOSTIC_NOT_UNBIASED_VALIDATION'))
    write(out/'meta-predictions.json', [dict(id=r['id'], episode_id=r['episode_id'], time_s=r['time_s'],
        truth=bool(y[i]), expert_scores=x[i].tolist(), scores={a:float(s[i]) for a,s in scores.items()},
        flags={a:bool(p[i]) for a,p in flags.items()}) for i,r in enumerate(rows)])
    write(out/'model-seal.json', dict(model_sha256=sha(out/'stack.pkl'),
        fold_models={str(f):sha(out/('fold'+str(f)+'.pkl')) for f in range(4)}, cutoffs=cuts,
        fit_freeze_sha256=sha(out/'fit-freeze.json'), scores_sha256=sha(out/'meta-scores.npz'),
        training_summary_sha256=sha(out/'meta-training-summary.json'),
        training_predictions_sha256=sha(out/'meta-predictions.json'),
        authority='SEALED_BEFORE_NEW_DEV_PREDICTION_AND_SELECTED_DEV_LABELS'))
    print(json.dumps(dict(stage='fit', cutoffs=cuts, supports=supports,
        metrics={a:meta_reports[a]['metrics'] for a in ARMS})), flush=True)
    panels = {}
    assert time.perf_counter()-started < 900
    dev = existing_scores(capture_panel('dev', inputs), inputs, out, started)
    panels['dev'] = evaluate(dev, inputs, out, model, cuts)
    if panels['dev']['passed']:
        assert time.perf_counter()-started < 900
        transfer = existing_scores(capture_panel('mz158', inputs), inputs, out, started)
        panels['mz158'] = evaluate(transfer, inputs, out, model, cuts)
    gain = len(panels) == 2 and all(p['passed'] for p in panels.values()) and any(p['strict_mean_fp_gain'] for p in panels.values())
    summary = dict(decision='MZ169_LEARNED_COMBINATION_DEVELOPMENT_GAIN' if gain else 'MZ169_FIXED_STACKING_NOT_ADMITTED',
        panels=panels, development_gain=gain, stage2_scored='mz158' in panels,
        seconds=time.perf_counter()-started, hash_comparisons=inputs.comparisons,
        original_test_predictions_raw_evaluator_decoded=False, authority='CONSUMED_DEVELOPMENT_NOT_FRESH')
    write(out/'summary.json', summary)
    write(out/'all-inputs.json', inputs.bindings)
    assert all(sha(p) == h for p,h in sources.items())
    assert all(sha(p) == h for p,h in inputs.bindings.items())
    assert time.perf_counter()-started < 900
    write(out/'completion.json', dict(status='PASS', summary_sha256=sha(out/'summary.json'),
        model_seal_sha256=sha(out/'model-seal.json'), inputs_sha256=sha(out/'all-inputs.json'),
        sources_inputs_unchanged=True, resources='Process-local CPU models; no worker or paid allocations'))
    print(json.dumps(dict(decision=summary['decision'], stage2_scored=summary['stage2_scored'], seconds=summary['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=4):
        run(parser.parse_args().output)
