"""One fixed new-instance drift stress of unchanged shared constant-bias models."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import time

import active_view as av
import bias_drift_observation as capture
import shared_bias_inference as model
import run_observation_deepening as scoring
from run_shared_bias_pilot import cause

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT/'artifacts.local/work/ba-bias-drift-20260922'
OLD = ROOT/'artifacts.local/work/ba-shared-bias-20260922/run-v1'
SOURCE = ROOT/'artifacts.local/work/ba-observation-mechanisms-20260922/run-v1/source.json'
DOCS = ROOT/'research/active/dtr-r0/nearfield'
MODES = ('range_only', 'range_pose')
STRATA = ('all', 'general', 'boundary', 'boundary_left', 'boundary_right', 'boundary_far')
FROZEN = ('shared_bias_inference.py', 'active_view.py', 'active_view_positive.py',
          'continuous_boundary_witness.py', 'path_constraint_inference.py',
          'candidate_diagnostics.py', 'bent_path_inference.py', 'run_observation_deepening.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def selected(row, stratum):
    return stratum == 'all' or row['stratum'] == stratum or (stratum == 'boundary' and row['stratum'].startswith('boundary_'))


def bin_changes(a, b):
    counts = [sum(x != y for x, y in zip(va['bins'], vb['bins'], strict=True))
              for va, vb in zip(a, b, strict=True)]
    return dict(changed_history=any(counts), changed_views=sum(n > 0 for n in counts), changed_bins=sum(counts))


def in_scope(condition, mode):
    if condition == 'nominal':
        return True
    return '_constant_' in condition and (mode == 'range_pose' or condition.startswith('range_'))


def evaluate(output, predictions):
    # Only invoked after the prediction seal is verified.
    source = read(output/'source.json')
    observed = read(output/'observations.json')
    selections = read(output/'selections.json')
    byid = {r['id']: r for r in source}
    mapping = {(r['id'], r['condition']): r['query_id'] for r in selections}
    answers = {r['query_id']: r['results'] for r in predictions}
    views = {(r['id'], r['condition']): r['views'] for r in observed}
    rows, wrong = [], []
    for selection in selections:
        ident, condition = selection['id'], selection['condition']
        truth = byid[ident]['truth']
        result = answers[selection['query_id']]
        nominal = answers[mapping[ident, 'nominal']]
        row = dict(id=ident, stratum=byid[ident]['stratum'], condition=condition, truth=truth,
                   query_id=selection['query_id'],
                   **{m: scoring.MAP[result[m]['decision']] for m in MODES},
                   **{'nominal_'+m: scoring.MAP[nominal[m]['decision']] for m in MODES})
        row['unknown_causes'] = {m: cause(result[m]) for m in MODES}
        row['in_declared_model'] = {m: in_scope(condition, m) for m in MODES}
        refs = ['nominal']
        if '_drift_' in condition:
            family = condition.split('_')[0]
            refs += [family+'_constant_plus', family+'_constant_minus']
        row['bin_changes'] = {ref: bin_changes(views[ident, condition], views[ident, ref]) for ref in refs}
        for ref in refs[1:]:
            for m in MODES:
                row[ref+'_'+m] = scoring.MAP[answers[mapping[ident, ref]][m]['decision']]
        for m in MODES:
            iswrong = (truth and row[m] == 'NONINTERSECTING_HYPOTHESES') or (not truth and row[m] == 'INTERSECTS')
            if iswrong:
                wrong.append(dict(id=ident, stratum=row['stratum'], condition=condition, mode=m,
                                  truth=truth, decision=row[m], query_id=selection['query_id'],
                                  in_declared_model=in_scope(condition, m), source=byid[ident],
                                  nominal_decision=row['nominal_'+m], bin_changes=row['bin_changes'],
                                  solver_metadata=result[m]['solver_metadata'], witnesses=result[m]['witnesses']))
        rows.append(row)
    metrics, contrasts, reasons, changes, transitions = {}, {}, {}, {}, {}
    for stratum in STRATA:
        metrics[stratum], contrasts[stratum], reasons[stratum], changes[stratum], transitions[stratum] = {}, {}, {}, {}, {}
        for condition in capture.CONDITIONS:
            group = [r for r in rows if r['condition'] == condition and selected(r, stratum)]
            metrics[stratum][condition] = {m: scoring.counts(group, m) for m in MODES}
            refs = list(group[0]['bin_changes'])
            contrasts[stratum][condition] = {m: {ref: scoring.contrast(group, m, ('nominal_' + m) if ref == 'nominal' else ref+'_'+m)
                                                for ref in refs} for m in MODES}
            reasons[stratum][condition] = {m: dict(Counter(r['unknown_causes'][m] for r in group)) for m in MODES}
            changes[stratum][condition] = {ref: {k: sum(r['bin_changes'][ref][k] for r in group)
                                               for k in ('changed_history', 'changed_views', 'changed_bins')} for ref in refs}
            transitions[stratum][condition] = {}
            for m in MODES:
                correct = [r for r in group if (r['truth'] and r['nominal_'+m] == 'INTERSECTS') or
                           (not r['truth'] and r['nominal_'+m] == 'NONINTERSECTING_HYPOTHESES')]
                transitions[stratum][condition][m] = dict(nominal_correct=len(correct),
                    retained=sum(r[m] == r['nominal_'+m] for r in correct),
                    to_UNKNOWN=[r['id'] for r in correct if r[m] == 'UNKNOWN'],
                    to_wrong=[r['id'] for r in correct if r[m] not in ('UNKNOWN', r['nominal_'+m])])
    lengths = {c: sum(math.dist(a['actual_camera'], b['actual_camera'])
                      for a, b in zip(views[source[0]['id'], c], views[source[0]['id'], c][1:])) for c in capture.CONDITIONS}
    summary = dict(kind='NEW_SYNTHETIC_INSTANCE_CONSTANT_BIAS_MODEL_DRIFT_STRESS', seed=capture.SEED,
                   scenes=len(source), histories=len(rows), public_queries=len(predictions),
                   solver_calls=4*len(predictions), new_simulated_views=len(rows)*13,
                   new_simulated_ray_returns=len(rows)*13*8, presets=list(MODES), primary='range_pose',
                   metrics=metrics, contrasts=contrasts, unknown_causes=reasons, bin_changes=changes,
                   nominal_correct_transitions=transitions, actual_travel_m=lengths,
                   assumption_matrix={c: {m: in_scope(c, m) for m in MODES} for c in capture.CONDITIONS},
                   in_scope_wrong_count=sum(r['in_declared_model'] for r in wrong),
                   wrong_rows=len(wrong), any_drift_wrong=any('_drift_' in r['condition'] for r in wrong))
    av.write_json(output/'evaluation.json', rows)
    av.write_json(output/'wrong-cases.json', wrong)
    av.write_json(output/'summary.json', summary)
    return summary


def run(output):
    if output.exists():
        raise FileExistsError('Refusing overwrite of any run directory')
    output.mkdir(parents=True)
    started, stages = time.perf_counter(), []

    def stage(name):
        stages.append(dict(name=name, seconds=time.perf_counter()-started))
        print(json.dumps(stages[-1]), flush=True)

    try:
        av.verify_seal(OLD/'completion-seal.json')
        hashes = {str(p.resolve()): av.file_hash(p) for p in OLD.iterdir() if p.is_file()}
        hashes[str(SOURCE.resolve())] = av.file_hash(SOURCE)
        frozen = {}
        for name in FROZEN:
            p = Path(__file__).with_name(name)
            assert av.file_hash(p) == av.file_hash(OLD/name), 'Frozen inference dependency changed: '+name
            frozen[str(p.resolve())] = av.file_hash(p)
        av.write_json(output/'old-input-hashes.json', hashes)
        av.write_json(output/'frozen-model-hashes.json', frozen)
        files = [Path(__file__), Path(capture.__file__), Path(__file__).with_name('test_bias_drift_observation.py'),
                 Path(__file__).with_name('run_shared_bias_pilot.py'), Path(__file__).with_name('bent_path_observation.py')]
        files += [Path(p) for p in frozen]
        files += [DOCS/'BIAS_DRIFT_PROTOCOL_20260922.md']
        for p in files:
            shutil.copyfile(p, output/p.name)
        av.seal(output/'design-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        stage('design_and_frozen_code_sealed_before_generation')
        excluded = [capture.geometry_key(r['scene']) for r in read(SOURCE)]
        stats = {}
        source = capture.generate(excluded, stats)
        av.write_json(output/'source.json', source)
        av.write_json(output/'generation.json', stats)
        observations, selections, unique = [], [], {}
        for row in source:
            value = capture.scene(row)
            for condition in capture.CONDITIONS:
                views = capture.observe(value, condition)
                public = capture.public(views)
                key = av.canonical(public).decode()
                if key not in unique:
                    unique[key] = dict(query_id=f'query_{len(unique):04d}', observations=public)
                selections.append(dict(id=row['id'], condition=condition, query_id=unique[key]['query_id']))
                observations.append(dict(id=row['id'], condition=condition, views=views))
        av.write_json(output/'observations.json', observations)
        av.write_json(output/'selections.json', selections)
        av.write_json(output/'public-queries.json', list(unique.values()))
        av.seal(output/'input-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        stage('all_generated_inputs_sealed')
        # Clear evaluator variables before calling the public-only inference loop.
        del source, observations, selections, value, views, row, excluded
        predictions = []
        for i, q in enumerate(unique.values()):
            results = {m: model.infer(q['observations'], mode=m) for m in MODES}
            assert all(r['solver_calls'] == 2 and r['decision'] in scoring.MAP for r in results.values())
            answer = dict(**q, results=results)
            av.write_json(output/(q['query_id']+'.json'), answer)
            predictions.append(answer)
            if i % 100 == 0 or i == len(unique)-1:
                print(json.dumps(dict(completed=i+1, total=len(unique), seconds=time.perf_counter()-started)), flush=True)
        assert 4*len(predictions) <= 9360
        av.write_json(output/'predictions.json', predictions)
        av.seal(output/'prediction-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        stage('all_predictions_sealed_before_evaluator_join')
        av.verify_seal(output/'prediction-seal.json')
        summary = evaluate(output, predictions)
        assert all(av.file_hash(Path(p)) == h for p, h in hashes.items())
        assert all(av.file_hash(Path(p)) == h for p, h in frozen.items())
        stage('evaluation_complete')
        av.write_json(output/'stage-order.json', stages)
        av.seal(output/'completion-seal.json', {p.name: p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(status='COMPLETE', queries=len(predictions), solver_calls=4*len(predictions),
                              in_scope_wrong=summary['in_scope_wrong_count'], any_drift_wrong=summary['any_drift_wrong'],
                              seconds=time.perf_counter()-started)), flush=True)
    except BaseException as exc:
        av.write_json(output/'failure.json', dict(type=type(exc).__name__, message=str(exc), stages=stages))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if PARENT.resolve() not in args.output.resolve().parents:
        parser.error('Output must be a new child of the canonical bias-drift artifact root')
    run(args.output)
