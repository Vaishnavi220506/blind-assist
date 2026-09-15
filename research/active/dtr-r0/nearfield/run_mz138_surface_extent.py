"""Complete requested oracle2 with full native cuboid faces, not sample spans.

Posthoc scope correction after the sealed sampled-only diagnostic. No tuning,
new samples, sensor changes or observer-side surface completion is implemented.
"""
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from mz138_support_ceiling import aggregate, possible, certain, METHOD
from run_mz138_support_ceiling import DEFAULT as INITIAL, evaluate
from run_mz137_edge_corridor import WORK, selected_jsonl, read
from run_mz107_four_sensor import sha, write

OUTPUT=INITIAL.parent/'full-surface-extent-v1'
COMPLETION_METHOD=dict(authority='EVALUATOR_ONLY_FULL_NATIVE_CUBOID_FACE_ORACLE',
    scope_correction='REQUESTED_TRUE_SURFACE_EXTENT_WAS_NOT_IMPLEMENTED_BY_INITIAL_SAMPLE_ENVELOPES',
    known_initial_results='angular22/3/2 sampled-native21/2/3; three misses have no returned corridor points',
    geometry='UNION_OF_VERIFIED_COMPLETE_NATIVE_CUBOID_FACES_ATTACHED_TO_RETURNED_SLOTS',
    clipping='NONE_TO_ZONE_RGB_OR_SAMPLED_POINTS',certainty='ALL_FACES_PER_ORIGINAL_RETURN',
    fallback='INCUMBENT_RETURN_IF_ANY_FACE_UNKNOWN_OR_AMBIGUOUS',
    parameters_tuned=0,native_point_audit_tolerance_m=1e-5)


def native_face_pieces(record,evaluation):
    keys=sorted(set(tuple(k) for k in record['surface_keys']))
    if not keys:return None
    bounds={o['name']:o for o in evaluation['native_bounds']};result=[]
    for actor,face in keys:
        if face.startswith(('UNKNOWN','AMBIGUOUS')):return None
        obj=bounds.get(actor.rsplit('/',1)[-1])
        if obj is None:return None
        axis,side=face.split(':');axis=int(axis);assert side in ('MIN','MAX')
        low=np.array(obj['center_m'])-obj['extent_m']-evaluation['body_origin_m']
        high=np.array(obj['center_m'])+obj['extent_m']-evaluation['body_origin_m']
        fixed=(low if side=='MIN' else high)[axis];low[axis]=high[axis]=fixed
        result.append(np.stack([low,high],axis=1).tolist())
    return result


def complete_frames(rows,es,initial,radars):
    records=[];coverage=dict(returns=0,completed=0,fallback=0,contributors=0,corridor_contributors=0,
        native_points_outside_completed_faces=0,corridor_points_outside_completed_faces=0)
    for row,e,f,radar in zip(rows,es,initial,radars):
        bits=[];changed=0;pieces_by_key={};returns=[]
        for record in f['returns']:
            coverage['returns']+=1;pieces=native_face_pieces(record,e)
            if pieces is None:
                value=record['original'];coverage['fallback']+=1
            else:
                value=dict(possible=any(possible(p) for p in pieces),certain=all(certain(p) for p in pieces));coverage['completed']+=1
            key=(record['zone'],record['slot']);pieces_by_key[key]=pieces
            bits.append(dict(zone=key[0],slot=key[1],**value));changed+=value!=record['original']
            returns.append(dict(zone=key[0],slot=key[1],pieces=pieces,**value))
        prediction=aggregate(row,bits,radar);prediction['changed_returns']=changed
        for point in f['contributors']:
            coverage['contributors']+=1;coverage['corridor_contributors']+=point['corridor']
            pieces=pieces_by_key[(point['zone'],point['slot'])]
            retained=pieces is None or any(all(lo-1e-5<=v<=hi+1e-5 for (lo,hi),v in zip(p,point['body_point_m'])) for p in pieces)
            coverage['native_points_outside_completed_faces']+=not retained
            coverage['corridor_points_outside_completed_faces']+=point['corridor'] and not retained
        records.append(dict(id=row['id'],prediction=prediction,returns=returns))
    return records,coverage


def run():
    out=OUTPUT.resolve();assert not out.exists()
    done=read(INITIAL/'completion.json');assert done['status']=='PASS'
    initialseal=read(INITIAL/'oracle-output-seal.json')
    assert sha(INITIAL/'oracle-output-seal.json')==done['oracle_output_seal_sha256']
    assert sha(INITIAL/'oracle-outputs.json')==initialseal['sha256']
    frozen=read(INITIAL/'freeze.json')
    for p,digest in frozen['inputs'].items():assert sha(Path(p))==digest
    for p,digest in frozen['sources'].items():assert sha(Path(p))==digest
    source_files=[Path(__file__),Path(__file__).with_name('mz138_support_ceiling.py'),
        Path(__file__).with_name('mz115_zonal_capture.py'),Path(__file__).with_name('mz113_dynamic_sensors.py')]
    out.mkdir(parents=True)
    for p in source_files:shutil.copyfile(p,out/p.name)
    write(out/'freeze.json',dict(method=COMPLETION_METHOD,initial_oracle_sha256=initialseal['sha256'],
        inputs=frozen['inputs'],sources={str(p):sha(p) for p in source_files},
        initial_summary_sha256=sha(INITIAL/'summary.json'),initial_completion_sha256=sha(INITIAL/'completion.json')))
    rows=selected_jsonl(WORK/'incumbent/fresh-v1/nominal/raw.jsonl',set(frozen['ids']))
    cap=WORK/'source/returned-v1/capture-v1';es=selected_jsonl(cap/'evaluator.jsonl',set(frozen['ids']))
    spec=read(cap/'spec.json');cache=read(WORK/'incumbent/fresh-v1/nominal/predictions.json')
    mapping={p['id']:i for i,p in enumerate(cache['predictions'])};radars=[cache['radar'][mapping[r['id']]] for r in rows]
    initial=read(INITIAL/'oracle-outputs.json')
    assert [f['id'] for f in initial]==[e['id'] for e in es]==[r['id'] for r in rows]
    start=time.perf_counter();outputs,coverage=complete_frames(rows,es,initial,radars)
    write(out/'oracle-outputs.json',outputs)
    write(out/'oracle-output-seal.json',dict(sha256=sha(out/'oracle-outputs.json'),seconds=time.perf_counter()-start,
        authority='EVALUATOR_ONLY_POSTHOC_SCOPE_COMPLETION_NOT_FRESH_CONFIRMATION'))
    # Reuse exact scoring functions; preserve initial predictions verbatim in
    # a separate file and add a new arm to an independently computed report.
    from evaluate_mz136_corridor_pair import score,retention,pair_metrics
    from run_mz107_four_sensor import truth,metrics
    gt=np.array([truth(e) for e in es],bool);flags=np.array([p['prediction']['candidate'] for p in outputs],bool)
    base=np.array([f['arms']['mz129']['candidate'] for f in initial],bool);ii=list(range(48))
    result=score(rows,es,gt,flags,ii,base);baseline=read(INITIAL/'summary.json')['arms']['mz129']
    result['retention']=retention(result,baseline);result['joint_target_met']=bool(result['retention']['pass_retention'] and result['metrics']['FP']<=10)
    result['tof_only']=score(rows,es,gt,np.array([p['prediction']['tof_candidate'] for p in outputs],bool),ii)
    m=result['metrics'];result.update(precision=m['TP']/max(1,m['TP']+m['FP']),recall=m['TP']/max(1,m['TP']+m['FN']))
    episodes={}
    for i,r in enumerate(rows):episodes.setdefault(r['episode_id'],[]).append(i)
    pairs=[dict(a=i,b=j) for p in spec['pairs'] if p['split']=='dev' for i,j in zip(*(episodes[e] for e in p['episodes']))]
    result['paired_binary']=pair_metrics(gt,flags.astype(float),flags,pairs)
    oldcases=read(INITIAL/'cases.json');strata={}
    for i,c in enumerate(oldcases):
        for s in (c['stratum'],'shallow_family' if c['family']=='shallow_boundary_stress' else 'other_families'):strata.setdefault(s,[]).append(i)
    result['strata']={s:dict(frames=len(ix),**metrics(gt[ix],flags[ix])) for s,ix in strata.items()}
    result['changed_from_sampled_native']=[dict(id=r['id'],truth=bool(gt[i]),sampled=initial[i]['arms']['full_native']['candidate'],full_surface=bool(flags[i]))
        for i,r in enumerate(rows) if initial[i]['arms']['full_native']['candidate']!=flags[i]]
    summary=dict(authority=COMPLETION_METHOD['authority'],scope_correction=COMPLETION_METHOD['scope_correction'],
        result=result,coverage=coverage,decision='TRUE_SURFACE_EXTENT_HEADROOM_NOT_OWNERSHIP_GROUPING_GAIN' if result['joint_target_met'] else 'FULL_SURFACE_ORACLE_JOINT_TARGET_NOT_MET')
    write(out/'summary.json',summary)
    assert (outputs,coverage)==complete_frames(rows,es,initial,radars)
    for p,digest in frozen['inputs'].items():assert sha(Path(p))==digest
    for p in source_files:assert sha(p)==read(out/'freeze.json')['sources'][str(p)]
    write(out/'completion.json',dict(status='PASS',exact_replay=True,inputs_unchanged=True,summary_sha256=sha(out/'summary.json'),
        output_seal_sha256=sha(out/'oracle-output-seal.json'),resources='No persistent workers; same scalar CPU geometry'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
