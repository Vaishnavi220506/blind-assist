"""One initial-camera-relative query replay; frozen data and no new observations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import active_view as av
import initial_relative_inference as model
import run_observation_deepening as scoring
from run_shared_bias_pilot import cause

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'artifacts.local/work/ba-bias-drift-20260922/run-v1'
GAUGE = ROOT/'artifacts.local/work/ba-boundary-observability-20260922/run-v1/counterexamples.json'
PARENT = ROOT/'artifacts.local/work/ba-initial-relative-20260922'
DOCS = ROOT/'research/active/dtr-r0/nearfield'
STRATA = ('all', 'general', 'boundary', 'boundary_left', 'boundary_right', 'boundary_far')
FROZEN = ('shared_bias_inference.py', 'active_view.py', 'active_view_positive.py',
          'continuous_boundary_witness.py', 'path_constraint_inference.py',
          'candidate_diagnostics.py', 'bent_path_inference.py', 'run_observation_deepening.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def truth(row, initial_camera):
    x, z = initial_camera
    return any(b['x']+b['width']/2 >= x-.3-av.EPS and b['x']-b['width']/2 <= x+.3+av.EPS
               and b['z']+b['thickness']/2 >= z+.3-av.EPS and b['z']-b['thickness']/2 <= z+3.+av.EPS
               for b in row['boxes'])


def selected(row, stratum):
    return stratum == 'all' or row['stratum'] == stratum or (stratum == 'boundary' and row['stratum'].startswith('boundary_'))


def correct(label, decision):
    return decision == ('INTERSECTS' if label else 'NONINTERSECTING_HYPOTHESES')


def evaluate(output, predictions):
    source = {r['id']: r for r in read(OLD/'source.json')}
    observations = {(r['id'], r['condition']): r['views'] for r in read(OLD/'observations.json')}
    selections = read(OLD/'selections.json')
    mapping = {(r['id'], r['condition']): r['query_id'] for r in selections}
    global_results = {r['query_id']: r['results']['range_pose'] for r in read(OLD/'predictions.json')}
    answers = {r['query_id']: r['result'] for r in predictions}
    conditions = sorted({r['condition'] for r in selections})
    rows, wrong = [], []
    for sel in selections:
        ident, condition, q = sel['id'], sel['condition'], sel['query_id']
        initial = observations[ident, condition][0]['actual_camera']
        nominal_q = mapping[ident, 'nominal']
        nominal_camera = observations[ident, 'nominal'][0]['actual_camera']
        row = dict(id=ident, stratum=source[ident]['stratum'], condition=condition, query_id=q,
            initial_camera=initial, truth=truth(source[ident], initial), global_truth=source[ident]['truth'],
            relative=scoring.MAP[answers[q]['decision']], global_decision=scoring.MAP[global_results[q]['decision']],
            nominal_relative=scoring.MAP[answers[nominal_q]['decision']],
            nominal_global=scoring.MAP[global_results[nominal_q]['decision']],
            nominal_truth=truth(source[ident], nominal_camera),
            unknown_cause=cause(answers[q]), in_declared_model=(condition == 'nominal' or '_constant_' in condition))
        if condition == 'nominal':
            assert row['truth'] == row['global_truth']
        if row['relative'] != 'UNKNOWN' and not correct(row['truth'], row['relative']):
            wrong.append(dict(**row, result=answers[q], source=source[ident]))
        rows.append(row)
    metrics, label_changes, causes, retention = {}, {}, {}, {}
    nominal_contrasts = {}
    for stratum in STRATA:
        metrics[stratum], label_changes[stratum], causes[stratum], retention[stratum] = {}, {}, {}, {}
        for condition in conditions:
            group = [r for r in rows if r['condition'] == condition and selected(r, stratum)]
            oldtruth = [dict(r, truth=r['global_truth']) for r in group]
            metrics[stratum][condition] = dict(relative=scoring.counts(group, 'relative'),
                global_own_truth=scoring.counts(oldtruth, 'global_decision'),
                global_on_relative_truth_SEMANTIC_MISMATCH=scoring.counts(group, 'global_decision'))
            label_changes[stratum][condition] = dict(
                global_IN_to_relative_OUT=[r['id'] for r in group if r['global_truth'] and not r['truth']],
                global_OUT_to_relative_IN=[r['id'] for r in group if not r['global_truth'] and r['truth']])
            causes[stratum][condition] = dict(Counter(r['unknown_cause'] for r in group))
            originally_correct = [r for r in group if correct(r['nominal_truth'], r['nominal_relative'])]
            unchanged = [r for r in originally_correct if r['truth'] == r['nominal_truth']]
            changed = [r for r in originally_correct if r['truth'] != r['nominal_truth']]
            retention[stratum][condition] = dict(nominal_correct=len(originally_correct),
                same_truth_cases=len(unchanged), retained_correct=sum(correct(r['truth'], r['relative']) for r in unchanged),
                same_truth_to_UNKNOWN=[r['id'] for r in unchanged if r['relative'] == 'UNKNOWN'],
                same_truth_to_wrong=[r['id'] for r in unchanged if r['relative'] != 'UNKNOWN' and not correct(r['truth'], r['relative'])],
                changed_truth_cases=[dict(id=r['id'], truth=r['truth'], relative=r['relative'],
                                           now_correct=correct(r['truth'], r['relative'])) for r in changed])
        nominal = [r for r in rows if r['condition'] == 'nominal' and selected(r, stratum)]
        nominal_contrasts[stratum] = scoring.contrast(nominal, 'relative', 'global_decision')
    gauge = []
    for witness in read(GAUGE):
        initial = [round(witness['bias']['pose_x_m'], 12), round(witness['bias']['pose_z_m'], 12)]
        relative = truth(witness['alternative'], initial)
        gauge.append(dict(id=witness['id'], original_truth=source[witness['id']]['truth'],
                          old_global_alternative_truth=witness['alternative_truth'], relative_alternative_truth=relative,
                          same_relative_label=relative == source[witness['id']]['truth']))
    assert len(gauge) == 40 and all(r['same_relative_label'] for r in gauge)
    summary = dict(kind='CONSUMED_INITIAL_CAMERA_RELATIVE_QUERY_PILOT', scenes=180, histories=len(rows),
        public_queries=len(predictions), solver_calls=2*len(predictions), new_observations=0,
        query_reference='ACTUAL_INITIAL_CAMERA_FIXED_FOR_ALL13VIEWS', metrics=metrics,
        label_changes=label_changes, unknown_causes=causes, truth_aware_nominal_retention=retention,
        nominal_paired_contrasts=nominal_contrasts, gauge_relative_labels_preserved=40,
        wrong_case_conditions=len(wrong), in_scope_wrong_case_conditions=sum(r['in_declared_model'] for r in wrong))
    av.write_json(output/'evaluation.json', rows)
    av.write_json(output/'wrong-cases.json', wrong)
    av.write_json(output/'gauge-label-check.json', gauge)
    av.write_json(output/'summary.json', summary)
    return summary


def run(output):
    if output.exists():
        raise FileExistsError('Refusing overwrite')
    output.mkdir(parents=True)
    started, stages = time.perf_counter(), []

    def stage(name):
        stages.append(dict(name=name, seconds=time.perf_counter()-started))
        print(json.dumps(stages[-1]), flush=True)

    try:
        av.verify_seal(OLD/'completion-seal.json')
        oldhashes = {str(p.resolve()): av.file_hash(p) for p in OLD.iterdir() if p.is_file()}
        oldhashes[str(GAUGE.resolve())] = av.file_hash(GAUGE)
        av.write_json(output/'old-input-hashes.json', oldhashes)
        frozen = {}
        for name in FROZEN:
            p = Path(__file__).with_name(name)
            assert av.file_hash(p) == av.file_hash(OLD/name), 'Frozen dependency changed: '+name
            frozen[str(p.resolve())] = av.file_hash(p)
        av.write_json(output/'frozen-dependency-hashes.json', frozen)
        files = [Path(__file__), Path(model.__file__), Path(__file__).with_name('test_initial_relative_inference.py'),
                 Path(__file__).with_name('test_shared_bias_inference.py'),
                 Path(__file__).with_name('run_shared_bias_pilot.py')]
        files += [Path(p) for p in frozen]
        files += [DOCS/n for n in ('INITIAL_RELATIVE_PROTOCOL_20260922.md', 'INITIAL_RELATIVE_INFERENCE_BRIEF_20260922.md')]
        for p in files:
            shutil.copyfile(p, output/p.name)
        public = read(OLD/'public-queries.json')
        assert len(public) == 375
        av.write_json(output/'public-queries.json', public)
        av.seal(output/'pre-readout-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        stage('code_protocol_and_public_inputs_sealed')
        predictions = []
        for i, q in enumerate(public):
            result = model.infer(q['observations'])
            assert result['solver_calls'] == 2 and result['decision'] in scoring.MAP
            answer = dict(**q, result=result)
            av.write_json(output/(q['query_id']+'.json'), answer)
            predictions.append(answer)
            if i % 100 == 0 or i == len(public)-1:
                print(json.dumps(dict(completed=i+1, total=len(public), seconds=time.perf_counter()-started)), flush=True)
        av.write_json(output/'predictions.json', predictions)
        av.seal(output/'prediction-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        stage('all_predictions_sealed_before_evaluator_join')
        av.verify_seal(output/'prediction-seal.json')
        summary = evaluate(output, predictions)
        assert all(av.file_hash(Path(p)) == h for p, h in oldhashes.items())
        assert all(av.file_hash(Path(p)) == h for p, h in frozen.items())
        stage('evaluation_complete')
        av.write_json(output/'stage-order.json', stages)
        av.seal(output/'completion-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(status='COMPLETE', solver_calls=750,
                              in_scope_wrong=summary['in_scope_wrong_case_conditions'],
                              nominal=summary['metrics']['all']['nominal']['relative'], seconds=time.perf_counter()-started)), flush=True)
    except BaseException as exc:
        av.write_json(output/'failure.json', dict(type=type(exc).__name__, message=str(exc), stages=stages))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if PARENT.resolve() not in args.output.resolve().parents:
        parser.error('Output must be a new child of the canonical initial-relative artifact root')
    run(args.output)
