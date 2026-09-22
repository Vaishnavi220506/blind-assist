"""One frozen additional-precision comparison on saved analytic histories."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import active_view as av
import inherit_precision as candidate
import run_observation_deepening as scoring
from test_shared_bias_inference import residual

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import research_backend as backend

CONDITIONS = ('nominal', 'range_constant_minus', 'combined_constant_plus')
STRATA = ('all', 'general', 'boundary', 'boundary_left', 'boundary_right', 'boundary_far')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def correct(truth, decision):
    return decision == ('IN_MODEL_CONDITIONAL' if truth else 'OUT_MODEL_CONDITIONAL')


def counts(rows, key):
    p = sum(r['truth'] for r in rows)
    tp = sum(r['truth'] and r[key] == 'IN_MODEL_CONDITIONAL' for r in rows)
    fp = sum(not r['truth'] and r[key] == 'IN_MODEL_CONDITIONAL' for r in rows)
    false_out = sum(r['truth'] and r[key] == 'OUT_MODEL_CONDITIONAL' for r in rows)
    out = sum(not r['truth'] and r[key] == 'OUT_MODEL_CONDITIONAL' for r in rows)
    return dict(n=len(rows), positives=p, negatives=len(rows)-p, tp=tp, fp=fp,
        fn=p-tp, false_out=false_out, correct_out=out,
        unknown=sum(r[key] == 'UNKNOWN' for r in rows), correct=tp+out,
        recall=tp/p if p else None, precision=tp/(tp+fp) if tp+fp else None,
        fpr=fp/(len(rows)-p) if len(rows)>p else None)


def run(args):
    output = args.result.parent
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Only a new empty governed output directory is allowed')
    start = time.perf_counter()
    stages = []
    def stage(name):
        stages.append(dict(name=name, seconds=time.perf_counter()-start))
        print(json.dumps(stages[-1]), flush=True)
    try:
        input_paths = [args.drift/'observations.json', args.drift/'source.json',
            args.drift/'selections.json', args.parent/'predictions.json']
        oldhashes = {str(p.resolve()): av.file_hash(p) for p in input_paths}
        av.write_json(output/'input-hashes.json', oldhashes)
        sources = [Path(__file__), Path(candidate.__file__),
            Path(__file__).with_name('test_inherit_precision.py'),
            Path(__file__).with_name('test_shared_bias_inference.py'),
            ROOT/'research/active/dtr-r0/nearfield/INHERIT_PRECISION_PROTOCOL_20260922.md']
        sources += [Path(m.__file__) for m in (candidate.original, candidate.original.shared,
            candidate.original.shared.base, candidate.original.shared.paths,
            candidate.original.shared.geometry, candidate.original.shared.diagnostic, av)]
        for path in sources:
            shutil.copyfile(path, output/path.name)
        backend.select_backend('model-inference', cpu=backend.BackendCandidate(
            name='CPU SciPy HiGHS', expected_device_type='cpu', run_probe=lambda: 1,
            observe=lambda _: backend.DeviceObservation('cpu', platform.processor(), 'SciPy/HiGHS')),
            cpu_reason='TASK_NOT_GPU_SUITABLE', record_path=output/'backend.json')

        # Simulation adapter only: analog range is NOT admitted into inference.
        private = [r for r in read(args.drift/'observations.json') if r['condition'] in CONDITIONS]
        assert len(private) == 540 and all(len(r['views']) == 13 for r in private)
        queries, mapping, query_ids = [], [], {}
        coarse_parity = 0
        for row in private:
            coarse = candidate.public_views(row['views'], .1)
            assert coarse == [dict(camera=v['camera'], bins=v['bins']) for v in row['views']]
            coarse_parity += 1
            public = candidate.public_views(row['views'], .001)
            signature = hashlib.sha256(av.canonical(public)).hexdigest()
            if signature not in query_ids:
                query_ids[signature] = 'fine_' + str(len(queries)).zfill(4)
                queries.append(dict(query_id=query_ids[signature], observations=public))
            mapping.append(dict(id=row['id'], condition=row['condition'], query_id=query_ids[signature]))
        av.write_json(output/'public-queries.json', queries)
        av.write_json(output/'mapping.json', mapping)
        av.seal(output/'pre-inference-seal.json', {p.name:p for p in output.iterdir() if p.is_file()})
        stage('public_queries_and_code_sealed')
        model = candidate.model_for(.001)
        predictions = []
        for i, row in enumerate(queries):
            answer = model.infer(row['observations'])
            assert answer['solver_calls'] == 2
            predictions.append(dict(query_id=row['query_id'], result=answer))
            av.write_json(output/(row['query_id']+'.json'), predictions[-1])
            if i%50 == 0 or i == len(queries)-1:
                print(json.dumps(dict(complete=i+1, total=len(queries), seconds=time.perf_counter()-start)), flush=True)
        av.write_json(output/'predictions.json', predictions)
        av.seal(output/'prediction-seal.json', {p.name:p for p in output.iterdir() if p.is_file()})
        stage('all_predictions_sealed_before_truth_and_old_decisions')

        # Evaluator begins here. No future access or hidden fields in model inputs.
        truth = {r['id']:r for r in read(args.drift/'source.json')}
        oldmap = {(r['id'],r['condition']):r['query_id'] for r in read(args.drift/'selections.json')}
        old = {r['query_id']:r['result'] for r in read(args.parent/'predictions.json')}
        answers = {r['query_id']:r['result'] for r in predictions}
        observation_map = {(r['id'],r['condition']):r for r in private}
        public_map = {r['query_id']:r['observations'] for r in queries}
        rows, containment = [], []
        for item in mapping:
            ident, cond = item['id'],item['condition']
            record = observation_map[ident,cond]
            s = truth[ident]
            w = dict(scene={k:s[k] for k in ('boxes','wall_z')}, bias=record['views'][0]['bias'])
            label = model.relative_label(w)
            obs = public_map[item['query_id']]
            valid = model.validate_witness(w, obs, label)
            error = residual(model._constraints(obs, label), w)
            fine = answers[item['query_id']]
            coarse = old[oldmap[ident,cond]]
            opposite_exclusion = next(r for r in fine['solver_metadata'] if r['requested_label']==label)['exclusion_supported']
            containment.append(dict(id=ident, condition=cond, forward_valid=valid,
                max_positive_residual=max(0.,error), true_class_excluded=opposite_exclusion))
            rows.append(dict(**item, stratum=s['stratum'], truth=label=='IN',
                coarse=coarse['decision'], fine=fine['decision'], fine_reason=fine['reason']))
        metrics, changes = {}, {}
        for cond in CONDITIONS:
            metrics[cond], changes[cond] = {}, {}
            for stratum in STRATA:
                group = [r for r in rows if r['condition']==cond and (stratum=='all' or
                    r['stratum']==stratum or (stratum=='boundary' and r['stratum'].startswith('boundary_')))]
                metrics[cond][stratum] = {k:counts(group,k) for k in ('coarse','fine')}
                changes[cond][stratum] = dict(
                    gained_correct=[r['id'] for r in group if correct(r['truth'],r['fine']) and not correct(r['truth'],r['coarse'])],
                    lost_correct=[r['id'] for r in group if correct(r['truth'],r['coarse']) and not correct(r['truth'],r['fine'])],
                    fine_wrong=[r['id'] for r in group if r['fine']!='UNKNOWN' and not correct(r['truth'],r['fine'])])
        valid = all(r['forward_valid'] and r['max_positive_residual']<=1e-9 and not r['true_class_excluded'] for r in containment)
        nominal=metrics['nominal']['all']; ch=changes['nominal']['all']
        retain = (valid and nominal['fine']['correct']-nominal['coarse']['correct']>=10
            and len(ch['lost_correct'])<=2 and all(not changes[c]['all']['fine_wrong'] for c in CONDITIONS)
            and all(metrics[c]['all']['fine']['correct']>metrics[c]['all']['coarse']['correct'] for c in CONDITIONS[1:]))
        summary = dict(kind='CONSUMED_ANALYTIC_PRECISION_WITH_SHARED_BIAS', conditions=CONDITIONS,
            source_scenes=180, histories=540, unique_queries=len(queries), solver_calls=2*len(queries),
            new_physical_observations=0, synthetic_fine_range_values=540*13*8,
            quantization_m=dict(coarse=.1,fine=.001), views=13,path_distance_m=.12,
            model='Single axis-aligned rectangle and anchored wall; actual initial-camera query',
            metrics=metrics, changes=changes, validity_pass=valid, component_gate=retain,
            unknown_reasons={c:dict(Counter(r['fine_reason'] for r in rows if r['condition']==c)) for c in CONDITIONS},
            coarse_requantization_parity=coarse_parity,
            max_containment_residual=max(r['max_positive_residual'] for r in containment),
            measured_elapsed_seconds=time.perf_counter()-start,
            evidence_boundary='Consumed analytic Development, hypothetical higher resolution, no real hardware or formal proof')
        av.write_json(output/'evaluation.json', rows)
        av.write_json(output/'containment-audit.json', containment)
        av.write_json(output/'summary.json', summary)
        if not all(av.file_hash(Path(p))==h for p,h in oldhashes.items()):
            raise RuntimeError('Old source or prediction bytes changed')
        stage('evaluation_and_true_assignment_containment_complete')
        av.write_json(output/'stage-order.json', stages)
        av.seal(output/'completion-seal.json', {p.name:p for p in output.iterdir() if p.is_file()})
        av.write_json(args.result, dict(status='COMPLETE' if valid else 'INVALID_CONTAINMENT',
            summary='summary.json', component_gate=retain, histories=540, solver_calls=2*len(queries)))
        print(json.dumps(summary['metrics']), flush=True)
    except BaseException as exc:
        av.write_json(output/'failure.json', dict(type=type(exc).__name__, message=str(exc),stages=stages))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drift',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    run(parser.parse_args())
