"""One frozen-A existing-return surface intervention and reachability audit."""
from collections import Counter
import json
import pickle
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_e1 import ROOT,HERE,BASE,read,write,sha,dataset,labels,report,native_account
from run_tolerance import NEW,EV
from tolerance_eval import geometry,classify,metric,temporal
from surface_oracle import intervene,intersects,SUPPORT_NAMES
import mz138_support_ceiling
import run_mz138_surface_extent
from research_backend import BackendCandidate,DeviceObservation,select_backend

WORK=ROOT/'artifacts.local/work/corridor-surface-oracle-20260917'
FEATURE=ROOT/'artifacts.local/work/corridor-intrusion-20260917/diagnosis/new-base-features.npz'
AP=ROOT/'artifacts.local/work/corridor-depth-e1-20260917/run-v2/A-model.pkl'
AT=.3917890013717321

def native_return_diagnosis(row,e,record):
    """Post-seal only. Undetected private rays are never oracle feature inputs."""
    origin=np.array(e['body_origin_m']);objs={o['name']:o for o in e['native_bounds']}
    eligible={}
    for name,o in objs.items():
        p=np.stack([np.array(o['center_m'])-o['extent_m']-origin,np.array(o['center_m'])+o['extent_m']-origin],-1)
        if intersects(p):eligible[name]=p
    returned={a for s in record['slots'] if s['usable'] for a in s.get('actors',[])}
    completed={a for s in record['slots'] if s['completed'] for a in s.get('actors',[])}
    private={h['actor_id'].rsplit('/',1)[-1] for z in e['zonal_tof_native'] for h in z['private_rays'] if 'actor_id' in h}
    linked=sorted(returned&eligible.keys());priv=sorted(private&eligible.keys())
    if record['full_face_reachable']:
        category='FULL_FACE_PRESENT_SAMPLED_POINT_PRESENT' if record['sampled_point_reachable'] else 'UNSAMPLED_FULL_FACE_COMPLETION_OPPORTUNITY'
    elif linked:category='RETURNED_CORRIDOR_OBJECT_BUT_NO_RESOLVED_INTERSECTING_FACE'
    elif not row['tof_packet_received']:category='TOF_PACKET_MISSING'
    elif priv:category='PRIVATE_OBJECT_HIT_WITHOUT_USABLE_RETURN'
    else:category='NO_NATIVE_RAY_HIT_ON_CORRIDOR_OBJECT'
    return dict(category=category,corridor_objects=sorted(eligible),returned_corridor_objects=linked,
        completed_corridor_objects=sorted(completed&eligible.keys()),private_hit_corridor_objects=priv,
        zero_usable_returns=record['usable_slots']==0,
        note='PRIVATE_UNRETURNED_RAYS_DIAGNOSIS_ONLY_NOT_ORACLE_INPUT')

def main():
    start=time.perf_counter();assert not WORK.exists();WORK.mkdir(parents=True)
    assert sha(AP)=='d1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef'
    assert sha(FEATURE)==read(FEATURE.with_name('new-base-feature-seal.json'))['feature_sha256']
    done=read(EV/'completion.json');assert done['status']=='PASS'
    seal=read(EV/'prediction-seal.json');assert sha(EV/'predictions.json')==seal['predictions_sha256']
    saved=read(EV/'predictions.json');data=dataset(NEW,'confirmation');ids=[r['id'] for r in data['rows']]
    z=np.load(FEATURE);base=z['base'];assert list(z['ids'])==ids==[r['id'] for r in saved]
    names0=read(BASE/'feature-names.json');names=names0['sensor_names']+names0['geometry_names'];assert len(names)==base.shape[1]==2485
    model=pickle.loads(AP.read_bytes());score0=model.predict_proba(base)[:,1];assert np.array_equal(score0,[r['A_score'] for r in saved])
    sources=[Path(__file__),HERE/'surface_oracle.py',HERE/'SURFACE_ORACLE_PROTOCOL_20260917.md',HERE/'tolerance_eval.py',
        Path(mz138_support_ceiling.__file__),Path(run_mz138_surface_extent.__file__),HERE.parent/'mz143_corridor_features.py',HERE.parent/'mz115_spatial_allocation.py']
    inputs=[AP,FEATURE,FEATURE.with_name('new-base-feature-seal.json'),BASE/'feature-names.json',EV/'predictions.json',EV/'prediction-seal.json']+[NEW/n for n in ['raw.jsonl','evaluator.jsonl','receipt.json','spec.json']]
    bindings={str(p):sha(p) for p in sources+inputs};write(WORK/'freeze.json',dict(bindings=bindings,threshold=AT,
        authority='EVALUATOR_ONLY_CONSUMED_DOMAIN_DIAGNOSTIC',ids=ids,feature_changes='usable_slot_six_support_endpoints_only',
        zero_fit=True,zero_capture=True,arms=['A','A_sampled_support','A_full_surface_support']))
    # Native geometry legitimately enters this explicitly privileged oracle;
    # evaluation labels do not select a support, actor, threshold or model.
    es,y=labels(data);values=[];full=[];records=[]
    for r,e,b in zip(data['rows'],es,base):
        s,f,a=intervene(r,e,b,names);values.append(s);full.append(f);records.append(a)
    sampled=np.stack(values);full=np.stack(full);scores=dict(A=score0,A_sampled_support=model.predict_proba(sampled)[:,1],A_full_surface_support=model.predict_proba(full)[:,1])
    flags={a:s>=AT for a,s in scores.items()}
    allowed=np.array([n.rsplit('.',1)[-1] in SUPPORT_NAMES and n.startswith('zone') for n in names])
    assert np.array_equal(base[:,~allowed],sampled[:,~allowed]) and np.array_equal(base[:,~allowed],full[:,~allowed])
    np.savez_compressed(WORK/'oracle-features.npz',ids=z['ids'],sampled=sampled,full=full)
    np.savez(WORK/'scores.npz',ids=z['ids'],**scores)
    write(WORK/'oracle-returns.json',records)
    write(WORK/'prediction-seal.json',dict(scores_sha256=sha(WORK/'scores.npz'),features_sha256=sha(WORK/'oracle-features.npz'),
        returns_sha256=sha(WORK/'oracle-returns.json'),freeze_sha256=sha(WORK/'freeze.json'),
        authority='NATIVE_GEOMETRY_USED_PRIVILEGED_ORACLE_NOT_OBSERVATION_MODEL'))
    # Fixed5cm strata and explanatory errors decoded after oracle outputs seal.
    states=np.array([classify(geometry(e),.05) for e in es]);clear=states!='boundary';boundary=~clear
    assert clear.sum()==216 and boundary.sum()==72
    family=np.array([e['family'] for e in es]);summary={}
    def count_transitions(p,ix):
        a=flags['A'];return dict(FP_removed=int((ix&~y&a&~p).sum()),FP_added=int((ix&~y&~a&p).sum()),
            FN_rescued=int((ix&y&~a&p).sum()),TP_lost=int((ix&y&a&~p).sum()))
    for arm,p in flags.items():
        summary[arm]=dict(clear=metric(y[clear],p[clear]),coverage=float(clear.mean()),boundary_strict=metric(y[boundary],p[boundary]),
            strict=metric(y,p),clear_changes=count_transitions(p,clear),
            families={f:dict(clear=metric(y[clear&(family==f)],p[clear&(family==f)]),
                changes=count_transitions(p,clear&(family==f)),strict=metric(y[family==f],p[family==f])) for f in sorted(set(family))},
            temporal=temporal(data['rows'],states,p),
            feature_changed_frames=int(np.any((base if arm=='A' else sampled if arm=='A_sampled_support' else full)!=base,axis=1).sum()),
            score_changed_frames=int((scores[arm]!=score0).sum()),alert_changed_frames=int((p!=flags['A']).sum()),
            max_probability_change=float(np.abs(scores[arm]-score0).max()))
    native={a:native_account(data['rows'],es,p,flags['A']) for a,p in flags.items()};write(WORK/'native.json',native)
    cases=[];support_by_family={};sreach=np.array([r['sampled_point_reachable'] for r in records]);freach=np.array([r['full_face_reachable'] for r in records])
    assert not np.any(freach&~y),'Native face should be subset of native object truth'
    for i,(r,e,rec) in enumerate(zip(data['rows'],es,records)):
        diag=native_return_diagnosis(r,e,rec)
        cases.append(dict(id=r['id'],episode=r['episode_id'],time_s=r['time_s'],family=e['family'],truth=bool(y[i]),state=states[i],
            A_error='FN' if y[i] and not flags['A'][i] else 'FP' if not y[i] and flags['A'][i] else None,
            scores={a:float(s[i]) for a,s in scores.items()},alerts={a:bool(p[i]) for a,p in flags.items()},
            sampled_point_reachable=bool(sreach[i]),full_face_reachable=bool(freach[i]),diagnosis=diag,
            usable_returns=rec['usable_slots'],completed_returns=rec['completed_slots'],fallback_returns=rec['fallback_slots']))
    for fam in sorted(set(family)):
        ix=clear&(family==fam);pos=ix&y;fn=pos&~flags['A'];fp=ix&~y&flags['A']
        support_by_family[fam]=dict(clear_positive=int(pos.sum()),A_FN=int(fn.sum()),A_FP=int(fp.sum()),
            positive_sampled_reachable=int((pos&sreach).sum()),positive_full_reachable=int((pos&freach).sum()),
            A_FN_sampled_reachable=int((fn&sreach).sum()),A_FN_full_reachable=int((fn&freach).sum()),
            A_FN_full_only=int((fn&freach&~sreach).sum()),A_FP_full_reachable=int((fp&freach).sum()),
            A_FN_categories=dict(Counter(cases[i]['diagnosis']['category'] for i in np.flatnonzero(fn))))
    write(WORK/'cases.json',cases)
    write(WORK/'clear-errors.json',[c for c in cases if c['state']!='boundary' and c['A_error']])
    split_usage=Counter(names[int(n['feature_idx'])] for rr in model._predictors for pred in rr for n in pred.nodes if not n['is_leaf'])
    support_splits=[]
    for ti,rr in enumerate(model._predictors):
        for pred in rr:
            for ni,n in enumerate(pred.nodes):
                if not n['is_leaf'] and allowed[int(n['feature_idx'])]:
                    support_splits.append(dict(tree=ti,node=ni,feature=names[int(n['feature_idx'])],threshold=float(n['num_threshold']),
                        left=int(n['left']),right=int(n['right'])))
    slots=[s for r in records for s in r['slots']];complete=[s for s in slots if s['completed']]
    coverage=dict(public_returns=len(slots),usable_returns=sum(s['usable'] for s in slots),completed_returns=len(complete),
        fallback_returns=sum(s['usable'] and not s['completed'] for s in slots),zero_usable_frames=sum(r['usable_slots']==0 for r in records),
        fallback_reasons=dict(Counter(s['fallback_reason'] for s in slots if s['usable'] and not s['completed'])),
        multi_face_returns=sum(len(s['faces'])>1 for s in complete),multi_actor_returns=sum(len(s['actors'])>1 for s in complete),
        hull_false_bridges=sum(r['hull_bridges'] for r in records),contributors=sum(s['contributors'] for s in complete),
        corridor_contributors=sum(s['corridor_contributors'] for s in complete),contributors_excluded=sum(r['contributors_excluded'] for r in records),
        corridor_contributors_excluded=sum(r['corridor_contributors_excluded'] for r in records))
    assert coverage['contributors_excluded']==coverage['corridor_contributors_excluded']==0
    result=dict(methods=summary,support_reachability=support_by_family,coverage=coverage,
        A_structure=dict(total_splits=sum(split_usage.values()),features_used=len(split_usage),support_splits=support_splits,
            support_columns=768,warning='Frozen_A_almost_ignores_this_support_seam; weak_result_is_not_zero_information_ceiling'),
        authority='NATIVE_GEOMETRY_PRIVILEGED_CONTROLLED_DEVELOPMENT',seconds=time.perf_counter()-start,
        zero_training=True,zero_capture=True,original_test_access=False,decision='INSPECT_FIXED_HEAD_RESPONSE_AND_RETURNED_SURFACE_REACHABILITY_SEPARATELY')
    write(WORK/'summary.json',result)
    select_backend('scalar-scoring',cpu=BackendCandidate('oracle-geometry-cpu','cpu',lambda:len(complete),
        lambda _:DeviceObservation('cpu','host CPU','Python NumPy and frozen sklearn HGB')),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=WORK/'backend.json')
    assert all(sha(p)==h for p,h in bindings.items())
    write(WORK/'completion.json',dict(status='PASS',summary_sha256=sha(WORK/'summary.json'),prediction_seal_sha256=sha(WORK/'prediction-seal.json'),
        baseline_bitwise_parity=True,non_support_features_bitwise_unchanged=True,resources='No worker; process-local CPU only'))
    print(json.dumps(dict(methods={a:dict(clear=v['clear'],changes=v['clear_changes'],score_changes=v['score_changed_frames']) for a,v in summary.items()},
        reachability=support_by_family,coverage=coverage),indent=2))

if __name__=='__main__':
    with threadpool_limits(4):main()
