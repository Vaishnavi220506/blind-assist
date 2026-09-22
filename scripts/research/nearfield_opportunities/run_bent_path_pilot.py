"""Frozen equal-cost path geometry and deliberately misspecified sensor stress."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

import active_view as av
import active_view_positive as pos
import bent_path_observation as observation
import bent_path_inference as inference
import run_observation_deepening as evaluation

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'artifacts.local/work/ba-observation-mechanisms-20260922/run-v1'
RECOVERY=ROOT/'artifacts.local/work/ba-candidate-recovery-20260922/run-v1'
DEEP=ROOT/'artifacts.local/work/ba-observation-deepening-20260922/run-v1'
DOCS=ROOT/'research/active/dtr-r0/nearfield'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def strong_gain(rows,candidate,baseline):
    contrast=evaluation.contrast(rows,candidate,baseline)
    return bool(contrast['TP']['gained'] and not contrast['TP']['lost']
        and not contrast['FP']['gained'] and not contrast['false_OUT']['gained']
        and evaluation.counts(rows,candidate)['correct_decisive']>=evaluation.counts(rows,baseline)['correct_decisive'])

def run(output):
    if output.exists():
        raise FileExistsError('Refusing overwrite')
    output.mkdir(parents=True)
    stages=[]
    started=time.perf_counter()
    def stage(name):
        stages.append(dict(name=name,seconds=time.perf_counter()-started))
        print(json.dumps(stages[-1]),flush=True)
    try:
        old_hashes={}
        for folder in (SOURCE,RECOVERY,DEEP):
            av.verify_seal(folder/'completion-seal.json')
            old_hashes.update({str(p.resolve()):av.file_hash(p) for p in folder.iterdir() if p.is_file()})
        paths=observation.paths()
        assert paths==inference.PATHS, 'Simulation and inference path whitelists disagree'
        conditions=observation.conditions()
        poses=sorted({p for path in paths.values() for p in path})
        av.write_json(output/'design.json',dict(paths=paths,conditions=conditions,primary='x_then_z',
            baseline='straight_x',order_control='z_then_x',poses=poses,
            inference_inputs=['camera','bins'],nominal_authority='NUMERICAL_SINGLE_RECTANGLE_MODEL_ONLY',
            stress_authority='MIS_SPECIFIED_SENSOR_STRESS'))
        av.write_json(output/'old-input-hashes.json',old_hashes)
        files=[Path(__file__),Path(observation.__file__),Path(inference.__file__),Path(evaluation.__file__),Path(av.__file__),Path(pos.__file__)]
        files += [Path(__file__).with_name(n) for n in ('path_constraint_inference.py','continuous_boundary_witness.py',
            'candidate_diagnostics.py','test_bent_path_observation.py','test_bent_path_inference.py')]
        files += [DOCS/n for n in ('BENT_PATH_PROTOCOL_20260922.md','BENT_PATH_INFERENCE_BRIEF_20260922.md')]
        for p in files:
            shutil.copyfile(p,output/p.name)
        av.seal(output/'design-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        stage('design_and_code_sealed_before_generation')
        source=read(SOURCE/'source.json')
        old_views={r['id']:{tuple(v['camera']):v for v in r['views']} for r in read(SOURCE/'all-pose-observations.json')}
        captures=[]
        acquisition=Counter()
        for r in source:
            scene=pos.source_scene(r)
            views=[]
            for condition,spec in conditions.items():
                for pose in poses:
                    if condition=='nominal' and pose in old_views[r['id']]:
                        old=old_views[r['id']][pose]
                        view=dict(camera=list(pose),actual_camera=list(pose),bins=old['bins'],
                            raw_ranges=old['raw_radial_m'],biased_ranges=old['raw_radial_m'],condition=condition,
                            range_bias_m=0.,pose_bias_m=[0.,0.],acquisition='REPLAY_SAVED_NOMINAL')
                    else:
                        view=observation.observe(scene,pose,condition)
                        view['acquisition']='NEW_SIMULATED_CONDITION'
                    acquisition[view['acquisition']]+=1
                    views.append(view)
            captures.append(dict(id=r['id'],views=views))
        av.write_json(output/'observations.json',captures)
        av.seal(output/'observation-seal.json',dict(observations=output/'observations.json',design=output/'design-seal.json'))
        stage('all_measurements_sealed_before_inference')
        queries={}
        selections=[]
        for r in captures:
            table={(v['condition'],tuple(v['camera'])):v for v in r['views']}
            mapping={}
            for condition in conditions:
                mapping[condition]={}
                for path,route in paths.items():
                    public=[dict(camera=list(p),bins=table[condition,p]['bins']) for p in route]
                    key=av.canonical(public).decode()
                    if key not in queries:
                        queries[key]=dict(query_id=f'query_{len(queries):04d}',observations=public)
                    mapping[condition][path]=queries[key]['query_id']
            selections.append(dict(id=r['id'],queries=mapping))
        av.write_json(output/'public-queries.json',list(queries.values()))
        av.write_json(output/'selected-histories.json',selections)
        av.seal(output/'solver-input-seal.json',dict(public=output/'public-queries.json',mapping=output/'selected-histories.json',
            observation_seal=output/'observation-seal.json'))
        answers=[]
        for i,q in enumerate(queries.values()):
            answer=dict(**q,result=inference.infer(q['observations']))
            assert answer['result']['decision'] in evaluation.MAP
            av.write_json(output/(q['query_id']+'.json'),answer)
            answers.append(answer)
            if i%100==0 or i==len(queries)-1:
                print(json.dumps(dict(completed=i+1,total=len(queries),seconds=time.perf_counter()-started)),flush=True)
        av.write_json(output/'predictions.json',answers)
        av.seal(output/'prediction-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        stage('all_predictions_sealed_before_truth_computation')
        av.verify_seal(output/'prediction-seal.json')
        answers={r['query_id']:r for r in answers}
        sources={r['id']:r for r in source}
        rows=[]
        by_condition={}
        causes={}
        for condition in conditions:
            group=[]
            reasons={name:Counter() for name in paths}
            for r in selections:
                geometry=pos.source_scene(sources[r['id']])
                b=geometry.boxes[0]
                row=dict(id=r['id'],condition=condition,truth=av.intersects_query(geometry),
                    side='left' if b.x<0 else 'right',width=b.width,depth=b.z)
                for path,q in r['queries'][condition].items():
                    result=answers[q]['result']
                    row[path]=evaluation.MAP[result['decision']]
                    reasons[path]['conditional' if result['decision']!='UNKNOWN' else
                        'opposing' if all(result['witnesses'].values()) else 'incomplete']+=1
                rows.append(row)
                group.append(row)
            by_condition[condition]=group
            causes[condition]={k:dict(v) for k,v in reasons.items()}
        metrics={c:{p:evaluation.counts(g,p) for p in paths} for c,g in by_condition.items()}
        contrasts={c:{p:evaluation.contrast(g,p,'straight_x') for p in paths if p!='straight_x'} for c,g in by_condition.items()}
        stress_changes={}
        nominal={r['id']:r for r in by_condition['nominal']}
        for c,group in by_condition.items():
            if c=='nominal':
                continue
            stress_changes[c]={}
            for path in paths:
                paired=[dict(id=r['id'],truth=r['truth'],stressed=r[path],nominal=nominal[r['id']][path]) for r in group]
                stress_changes[c][path]=evaluation.contrast(paired,'stressed','nominal')
        old_eval={r['id']:r for r in read(DEEP/'evaluation.json')}
        baseline_diffs=[r['id'] for r in nominal.values() if r['straight_x']!=old_eval[r['id']]['fixed_path']]
        # Prior98 opposing cases are an evaluator slice, never an inference route.
        recovery={r['query_id']:r['projection'] for r in read(RECOVERY/'predictions.json')}
        deep_selection=read(DEEP/'selected-traces.json')
        prior_opposing={r['id'] for r in deep_selection if all(recovery[r['queries']['fixed_path']]['witnesses'].values())}
        subgroup={}
        for c,group in by_condition.items():
            subgroup[c]={}
            for field in ('side','width','depth'):
                subgroup[c][field]={str(v):{p:evaluation.counts([r for r in group if r[field]==v],p) for p in paths}
                    for v in sorted({r[field] for r in group})}
        summary=dict(kind='CONSUMED_SCENES_NEW_PATH_VIEWS_AND_MISSPECIFIED_SENSOR_STRESS',cases_per_mode=len(source),
            conditions=len(conditions),paths=len(paths),unique_poses=len(poses),acquisition=dict(acquisition),
            selected_histories=len(source)*len(conditions)*len(paths),unique_queries=len(queries),solver_calls=2*len(queries),
            metrics=metrics,contrasts=contrasts,stress_changes=stress_changes,unknown_causes=causes,
            baseline_different_ids=baseline_diffs,prior_opposing_cases=len(prior_opposing),
            nominal_prior_opposing_metrics={p:evaluation.counts([r for r in by_condition['nominal'] if r['id'] in prior_opposing],p) for p in paths},
            primary_strong_gain={c:strong_gain(g,'x_then_z','straight_x') for c,g in by_condition.items()},
            elapsed_seconds=time.perf_counter()-started)
        av.write_json(output/'evaluation.json',rows)
        av.write_json(output/'subgroups.json',subgroup)
        av.write_json(output/'summary.json',summary)
        assert all(av.file_hash(Path(p))==h for p,h in old_hashes.items())
        stage('evaluation_complete')
        av.write_json(output/'stage-order.json',stages)
        av.seal(output/'completion-seal.json',{p.name:p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(status='COMPLETE',queries=len(queries),acquisition=dict(acquisition),
            metrics=metrics,strong_gain=summary['primary_strong_gain'],seconds=time.perf_counter()-started)),flush=True)
    except BaseException as exc:
        av.write_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),stages=stages))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    parent=(ROOT/'artifacts.local/work/ba-bent-path-20260922').resolve()
    if parent not in args.output.resolve().parents:
        parser.error('Output must be new child of canonical bent-path artifact root')
    run(args.output)
