"""Two frozen shared-bias models on already sealed bent-path histories."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import active_view as av
import shared_bias_inference as model
import run_observation_deepening as evaluation

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts.local/work/ba-bent-path-20260922/run-v1'
DOCS=ROOT/'research/active/dtr-r0/nearfield'
MODES=('range_only','range_pose')
PATHS=('straight_x','x_then_z','z_then_x')
CONDITIONS=('nominal','range_plus2mm','range_minus2mm','pose_plus1mm','pose_minus1mm')

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def cause(result):
    if result['decision']!='UNKNOWN':
        return 'conditional'
    if all(result['witnesses'].values()):
        return 'opposing'
    if all(r.get('solver_status')==2 and r.get('exclusion_supported') for r in result['solver_metadata']):
        return 'both_classes_infeasible'
    return 'incomplete_or_numerical'

def run(output):
    if output.exists():
        raise FileExistsError('Refusing overwrite')
    output.mkdir(parents=True)
    started=time.perf_counter()
    stages=[]
    def stage(name):
        stages.append(dict(name=name,seconds=time.perf_counter()-started))
        print(json.dumps(stages[-1]),flush=True)
    try:
        av.verify_seal(OLD/'completion-seal.json')
        hashes={str(p.resolve()):av.file_hash(p) for p in OLD.iterdir() if p.is_file()}
        av.write_json(output/'old-input-hashes.json',hashes)
        public=read(OLD/'public-queries.json')
        assert len(public)==749
        av.write_json(output/'public-queries.json',public)
        files=[Path(__file__),Path(model.__file__),Path(av.__file__),Path(evaluation.__file__)]
        files += [Path(__file__).with_name(n) for n in ('test_shared_bias_inference.py','active_view_positive.py',
            'continuous_boundary_witness.py','path_constraint_inference.py','candidate_diagnostics.py','bent_path_inference.py')]
        files += [DOCS/n for n in ('SHARED_BIAS_PROTOCOL_20260922.md','SHARED_BIAS_INFERENCE_BRIEF_20260922.md')]
        for p in files:
            shutil.copyfile(p,output/p.name)
        av.seal(output/'pre-readout-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        stage('protocol_code_and_public_inputs_sealed')
        predictions=[]
        for i,q in enumerate(public):
            results={name:model.infer(q['observations'],mode=name) for name in MODES}
            assert all(r['decision'] in evaluation.MAP for r in results.values())
            assert all(r['solver_calls']==2 for r in results.values())
            answer=dict(**q,results=results)
            av.write_json(output/(q['query_id']+'.json'),answer)
            predictions.append(answer)
            if i%100==0 or i==len(public)-1:
                print(json.dumps(dict(completed=i+1,total=len(public),seconds=time.perf_counter()-started)),flush=True)
        av.write_json(output/'predictions.json',predictions)
        av.seal(output/'prediction-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        stage('all_predictions_sealed_before_evaluator_read')
        av.verify_seal(output/'prediction-seal.json')
        selected=read(OLD/'selected-histories.json')
        frozen={(r['id'],r['condition']):r for r in read(OLD/'evaluation.json')}
        oldpred={r['query_id']:r['result'] for r in read(OLD/'predictions.json')}
        answers={r['query_id']:r['results'] for r in predictions}
        rows=[]
        metrics={}; contrasts={}; reasons={}; stability={}
        for condition in CONDITIONS:
            metrics[condition]={}; contrasts[condition]={}; reasons[condition]={}; stability[condition]={}
            for path in PATHS:
                group=[]
                counts={name:Counter() for name in ('exact',*MODES)}
                for r in selected:
                    q=r['queries'][condition][path]
                    old=frozen[r['id'],condition]
                    row=dict(id=r['id'],condition=condition,path=path,truth=old['truth'],exact=old[path],
                        exact_nominal=frozen[r['id'],'nominal'][path],
                        **{name:evaluation.MAP[result['decision']] for name,result in answers[q].items()})
                    rows.append(row); group.append(row)
                    counts['exact'][cause(oldpred[q])]+=1
                    for name,result in answers[q].items():
                        counts[name][cause(result)]+=1
                metrics[condition][path]={name:evaluation.counts(group,name) for name in ('exact',*MODES)}
                contrasts[condition][path]={name:evaluation.contrast(group,name,'exact') for name in MODES}
                contrasts[condition][path]['range_pose_vs_range_only']=evaluation.contrast(group,'range_pose','range_only')
                stability[condition][path]={name:evaluation.contrast(group,name,'exact_nominal') for name in MODES}
                reasons[condition][path]={k:dict(v) for k,v in counts.items()}
        capability={}
        stability_gate={}
        for name in MODES:
            capability[name]=all(metrics[c]['x_then_z'][name]['correct_decisive']>0 and
                metrics[c]['x_then_z'][name]['wrong_decisive']==0 for c in ('range_minus2mm','pose_plus1mm'))
            stability_gate[name]={}
            for path in PATHS:
                stability_gate[name][path]=all(not stability[c][path][name][k][side]
                    for c in CONDITIONS for k,side in (('TP','lost'),('correct_negative','lost'),('FP','gained'),('false_OUT','gained')))
        summary=dict(kind='CONSUMED_SHARED_BIAS_MODEL_DEVELOPMENT',public_queries=len(public),
            presets=list(MODES),solver_calls=4*len(public),cases_per_cell=180,selected_histories=len(rows),new_observations=0,
            metrics=metrics,contrasts=contrasts,unknown_causes=reasons,versus_exact_nominal=stability,
            primary_collapse_recovery=capability,strong_nominal_retention_across_conditions=stability_gate,
            elapsed_seconds=time.perf_counter()-started)
        av.write_json(output/'evaluation.json',rows)
        av.write_json(output/'summary.json',summary)
        assert all(av.file_hash(Path(p))==h for p,h in hashes.items())
        stage('evaluation_complete')
        av.write_json(output/'stage-order.json',stages)
        av.seal(output/'completion-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(status='COMPLETE',queries=len(public),solver_calls=4*len(public),
            capability=capability,stability=stability_gate,seconds=time.perf_counter()-started)),flush=True)
    except BaseException as exc:
        av.write_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),stages=stages))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    parent=(ROOT/'artifacts.local/work/ba-shared-bias-20260922').resolve()
    if parent not in args.output.resolve().parents:
        parser.error('Output must be new child of canonical shared-bias artifact root')
    run(args.output)
