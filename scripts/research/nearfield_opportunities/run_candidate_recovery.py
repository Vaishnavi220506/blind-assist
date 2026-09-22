"""One consumed numerical diagnosis, geometric projection and initial-witness cache."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import time

import active_view as av
import active_view_positive as pos
import run_observation_deepening as prior
import path_constraint_inference as model
import candidate_cache as cache
import candidate_diagnostics as diagnostic

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts.local/work/ba-observation-deepening-20260922/run-v1'
SOURCE=ROOT/'artifacts.local/work/ba-observation-mechanisms-20260922/run-v1'
DOCS=ROOT/'research/active/dtr-r0/nearfield'
MODES=('endpoint','fixed_path','guided_path','feedback','openloop')

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def semantics(result):
    return {k:result[k] for k in ('decision','reason','witnesses')}

def projected_result(baseline,attempts):
    """Only validated existence changes; exclusion authority never changes."""
    result=copy.deepcopy(baseline)
    added=[]
    for attempt in attempts:
        label=attempt['label']
        if result['witnesses'][label] is None and attempt.get('projected_valid'):
            result['witnesses'][label]=attempt['projected_candidate']
            added.append(label)
    by_label={r['requested_label']:r for r in result['solver_metadata']}
    result['decision']='UNKNOWN'
    conflicts=[k for k,v in result['witnesses'].items() if v is not None and by_label[k]['exclusion_supported']]
    if not conflicts:
        for label,opposite in (('IN','OUT'),('OUT','IN')):
            if (result['witnesses'][label] is not None and by_label[label].get('solver_status')==0
                    and by_label[opposite]['exclusion_supported']):
                result['decision']=label+'_MODEL_CONDITIONAL'
    result['reason']=('CONSTRUCTIVE_WITNESS_EXCLUSION_CONFLICT' if conflicts else
        'VALIDATED_WITNESS_AND_OPPOSITE_OUTER_RELAXATION_REPORTED_INFEASIBLE' if result['decision']!='UNKNOWN' else
        'BOTH_LABELS_HAVE_VALIDATED_WITNESSES' if all(result['witnesses'].values()) else
        'UNKNOWN_INCOMPLETE_OR_NUMERICAL_SEARCH')
    result['recovery_conflicts']=conflicts
    result['projection_recovery']=dict(added_labels=added,original_solver_receipts_preserved=True,extra_solver_calls=0)
    if added:
        result['opposing_witness_analysis']=None
    result['valid_witness_count']=sum(v is not None for v in result['witnesses'].values())
    result['status']='UNKNOWN' if result['decision']=='UNKNOWN' else 'MODEL_CONDITIONAL'
    return result

def run(output):
    if output.exists():
        raise FileExistsError('Refusing overwrite')
    output.mkdir(parents=True)
    start=time.perf_counter()
    stages=[]
    def stage(name):
        stages.append(dict(name=name,seconds=time.perf_counter()-start))
        print(json.dumps(stages[-1]),flush=True)
    try:
        for folder in (OLD,SOURCE):
            av.verify_seal(folder/'completion-seal.json')
        original_hashes={str(p.resolve()):av.file_hash(p) for folder in (OLD,SOURCE) for p in folder.iterdir() if p.is_file()}
        av.write_json(output/'consumed-inputs.json',original_hashes)
        queries=read(OLD/'continuous-public-queries.json')
        assert len(queries)==258
        pool=cache.build_initial_pool(SOURCE)
        av.write_json(output/'initial-witness-pool.json',pool)
        av.write_json(output/'public-queries.json',queries)
        modules=[Path(__file__),Path(diagnostic.__file__),Path(cache.__file__),Path(model.__file__),
            Path(pos.__file__),Path(av.__file__),Path(prior.__file__),Path(model.base.__file__)]
        for name in ('test_candidate_diagnostics.py','test_candidate_cache.py','test_candidate_recovery.py'):
            modules.append(Path(__file__).with_name(name))
        for name in ('CANDIDATE_RECOVERY_PROTOCOL_20260922.md','CANDIDATE_DIAGNOSTICS_BRIEF_20260922.md'):
            modules.append(DOCS/name)
        for p in modules:
            shutil.copyfile(p,output/p.name)
        av.seal(output/'pre-readout-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        stage('inputs_pool_code_sealed')
        old={r['query_id']:r['result'] for r in read(OLD/'continuous-predictions.json')}
        predictions=[]
        for i,q in enumerate(queries):
            collected=diagnostic.collect(q['observations'])
            baseline=collected['baseline']
            projection=projected_result(baseline,collected['attempts'])
            cached=cache.recover_with_cache(q['observations'],pool,baseline)
            answer=dict(**q,diagnostic=collected,projection=projection,cache=cached,
                baseline_semantics_reproduced=semantics(baseline)==semantics(old[q['query_id']]))
            predictions.append(answer)
            av.write_json(output/(q['query_id']+'.json'),answer)
            if i%50==0 or i==len(queries)-1:
                print(json.dumps(dict(completed=i+1,total=len(queries))),flush=True)
        av.write_json(output/'predictions.json',predictions)
        av.seal(output/'prediction-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        stage('all_predictions_sealed_before_truth')
        av.verify_seal(output/'prediction-seal.json')
        selected=read(OLD/'selected-traces.json')
        truth={r['id']:av.intersects_query(pos.source_scene(r)) for r in read(SOURCE/'source.json')}
        answers={r['query_id']:r for r in predictions}
        frozen={r['id']:r for r in read(OLD/'evaluation.json')}
        rows=[]
        causes={}
        for mode in MODES:
            for r in selected:
                a=answers[r['queries'][mode]]
                rows.append(dict(id=r['id'],mode=mode,truth=truth[r['id']],frozen=frozen[r['id']][mode],
                    replay=prior.MAP[a['diagnostic']['baseline']['decision']],projection=prior.MAP[a['projection']['decision']],
                    cache=prior.MAP[a['cache']['result']['decision']]))
            causes[mode]={}
            for name in ('replay','projection','cache'):
                counts=Counter()
                for r in selected:
                    a=answers[r['queries'][mode]]
                    result=a['diagnostic']['baseline'] if name=='replay' else a['projection'] if name=='projection' else a['cache']['result']
                    counts['conditional' if result['decision']!='UNKNOWN' else 'opposing' if all(result['witnesses'].values()) else 'incomplete']+=1
                causes[mode][name]=dict(counts)
        metrics={mode:{key:prior.counts([r for r in rows if r['mode']==mode],key) for key in ('frozen','replay','projection','cache')} for mode in MODES}
        contrasts={mode:{key:prior.contrast([r for r in rows if r['mode']==mode],key,'frozen') for key in ('replay','projection','cache')} for mode in MODES}
        attempts=[v for r in predictions for v in r['diagnostic']['attempts']]
        summary=dict(kind='CONSUMED_NUMERICAL_DIAGNOSIS_NO_NEW_OBSERVATIONS',queries=len(queries),solver_calls=len(attempts),
            reproduced_queries=sum(r['baseline_semantics_reproduced'] for r in predictions),pool_size=len(pool),
            metrics=metrics,contrasts=contrasts,causes=causes,
            projected_valid=sum(bool(v.get('projected_valid')) for v in attempts),
            cache_validation_checks=sum(r['cache']['matches']['checks']['total'] for r in predictions),
            cache_validation_seconds=sum(r['cache']['matches']['elapsed_seconds'] for r in predictions),
            elapsed_seconds=time.perf_counter()-start)
        av.write_json(output/'evaluation.json',rows)
        av.write_json(output/'summary.json',summary)
        stage('evaluation_complete')
        assert all(av.file_hash(Path(p))==h for p,h in original_hashes.items())
        av.write_json(output/'stage-order.json',stages)
        av.seal(output/'completion-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        av.write_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),stages=stages))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    parent=(ROOT/'artifacts.local/work/ba-candidate-recovery-20260922').resolve()
    if parent not in args.output.resolve().parents:
        parser.error('Output must be new child of canonical recovery artifact root')
    run(args.output)
