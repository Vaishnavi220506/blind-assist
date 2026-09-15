"""One consumed-dev48 EVALUATOR-ONLY support-ceiling diagnostic, no successor."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from mz138_support_ceiling import ARMS, METHOD, frame_oracles
from mz136_incumbent import public_observations
from run_mz137_edge_corridor import selected_jsonl, WORK, read
from run_mz107_four_sensor import ROOT, sha, write, truth, metrics
from evaluate_mz136_corridor_pair import score, retention, pair_metrics
from research_backend import BackendCandidate, DeviceObservation, select_backend

DEFAULT=ROOT/'artifacts.local/work/mz138-support-ceiling-20260915/ceiling-v1'


def evaluate(rows,es,frames,spec,radars,out):
    gt=np.array([truth(e) for e in es],bool);indices=list(range(len(rows)))
    names=('mz129',*ARMS)
    flags={a:np.array([f['arms'][a]['candidate'] for f in frames],bool) for a in names}
    results={a:score(rows,es,gt,p,indices,flags['mz129']) for a,p in flags.items()}
    assert results['mz129']['metrics']==dict(TP=24,FP=13,FN=0,TN=11,UNKNOWN=11)
    episode_indices={}
    for i,r in enumerate(rows):episode_indices.setdefault(r['episode_id'],[]).append(i)
    pairs=[dict(a=i,b=j) for pair in spec['pairs'] if pair['split']=='dev'
           for i,j in zip(*(episode_indices[e] for e in pair['episodes']))]
    strata={};cases=[]
    for i,(row,e,f,radar) in enumerate(zip(rows,es,frames,radars)):
        target=next(o for o in e['native_bounds'] if o['name']=='shape0')
        low=np.array(target['center_m'])-target['extent_m']-e['body_origin_m']
        high=np.array(target['center_m'])+target['extent_m']-e['body_origin_m']
        overlap=float(min(high[1],.3)-max(low[1],-.3))
        label='LT_1CM_ABS_GAP_OR_OVERLAP' if abs(overlap)<.01 else 'GE_1CM_ABS_GAP_OR_OVERLAP'
        for s in (label,'shallow_family' if e['family']=='shallow_boundary_stress' else 'other_families'):
            strata.setdefault(s,[]).append(i)
        cases.append(dict(id=row['id'],family=e['family'],truth=bool(gt[i]),native_target_overlap_m=overlap,
            stratum=label,arms=f['arms'],native_corridor_contributors=sum(c['corridor'] for c in f['contributors']),
            frozen_radar=dict(candidate=radar['candidate'],raw=radar['raw_center_support'],
                corrected_current=radar['corrected_current_radar'],carry=radar['inherited_carry_support'],
                guards=radar['guard_events'])))
    for a,result in results.items():
        m=result['metrics'];tof=np.array([f['arms'][a]['tof_candidate'] for f in frames],bool)
        result.update(precision=m['TP']/max(1,m['TP']+m['FP']),recall=m['TP']/max(1,m['TP']+m['FN']),
            retention=retention(result,results['mz129']),tof_only=score(rows,es,gt,tof,indices),
            paired_binary=pair_metrics(gt,flags[a].astype(float),flags[a],pairs),
            strata={s:dict(frames=len(ix),**metrics(gt[ix],flags[a][ix])) for s,ix in strata.items()})
        result['joint_target_met']=bool(result['retention']['pass_retention'] and m['FP']<=10)
    contributors=[c for f in frames for c in f['contributors']]
    returns=[r for f in frames for r in f['returns']]
    audit=dict(return_count=len(returns),contributor_count=len(contributors),
        corridor_contributors=sum(c['corridor'] for c in contributors),
        statuses=dict(Counter(r['status'] for r in returns)),
        fallback_returns=sum(r['oracles']['angular']['fallback'] for r in returns),
        multi_actor_returns=sum(r['actors']>1 for r in returns),multi_surface_returns=sum(r['surface_groups']>1 for r in returns),
        unresolved_face_contributors=sum(c['surface'][1].startswith('UNKNOWN') for c in contributors),
        ambiguous_face_contributors=sum(c['surface'][1].startswith('AMBIGUOUS') for c in contributors),arms={})
    for a in ARMS:
        audit['arms'][a]=dict(excluded=sum(not c[a+'_contains'] for c in contributors),
            corridor_excluded=sum(c['corridor'] and not c[a+'_contains'] for c in contributors),
            previously_enclosed_corridor_excluded=sum(c['corridor'] and c['incumbent_contains'] and not c[a+'_contains'] for c in contributors),
            corridor_without_possible=sum(c['corridor'] and not c[a+'_possible'] for c in contributors),
            changed_returns=sum(r['original']!={k:r['oracles'][a][k] for k in ('possible','certain')} for r in returns),
            changed_final_frames=int((flags[a]!=flags['mz129']).sum()),
            changed_zone_sets=sum(f['arms'][a]['active_zones']!=f['arms']['mz129']['active_zones'] for f in frames))
    radflags=np.array([r['candidate'] for r in radars],bool)
    floor=score(rows,es,gt,radflags,indices)
    remaining=[dict(id=c['id'],family=c['family'],arms={a:c['arms'][a] for a in names},frozen_radar=c['frozen_radar'])
               for c in cases if not c['truth'] and c['arms']['full_native']['candidate']]
    grouping=dict(changed_return_possible=sum(r['oracles']['pooled_native']['possible']!=r['oracles']['ownership_extent']['possible'] for r in returns),
        changed_final_frames=int((flags['pooled_native']!=flags['ownership_extent']).sum()),
        fp_reduction=results['pooled_native']['metrics']['FP']-results['ownership_extent']['metrics']['FP'],
        retention=retention(results['ownership_extent'],results['pooled_native']))
    if results['angular']['joint_target_met']:decision='ANGULAR_HEADROOM_EXISTS_ORACLE_ONLY'
    elif results['ownership_extent']['joint_target_met']:
        decision='OWNERSHIP_GROUPING_HEADROOM_ORACLE_ONLY' if grouping['fp_reduction']>0 else 'NATIVE_RANGE_EXTENT_HEADROOM_WITHOUT_GROUPING_GAIN'
    elif results['full_native']['joint_target_met']:decision='EXACT_NATIVE_SUPPORT_HEADROOM_BEYOND_ENVELOPES'
    elif floor['metrics']['FP']>10:decision='FIXED_RADAR_FLOOR_BLOCKS_TOF_ONLY_FP_TARGET_ON_THIS_PANEL'
    else:decision='NO_JOINT_CEILING_GAIN_INSPECT_NATIVE_COVERAGE_OR_FALLBACK'
    summary=dict(authority='EVALUATOR_ONLY_CONSUMED_DEV48_DIAGNOSTIC_NOT_ALGORITHM',frames=48,arms=results,
        coverage_and_retention=audit,fixed_radar_floor=floor,grouping_attribution=grouping,
        remaining_full_native_fp=remaining,decision=decision,
        limits='Sampled returned contributors only; angular oracle keeps noisy range and orientation uncertainty and Cartesian envelope. Native groups are not continuous surfaces. No successor executed.')
    write(out/'cases.json',cases);write(out/'summary.json',summary)
    return summary


def run(out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    cap=WORK/'source/returned-v1/capture-v1';prep=WORK/'incumbent/fresh-v1'
    spec=read(cap/'spec.json');receipt=read(cap/'receipt.json');assert receipt['status']=='PASS'
    assert sha(cap/'spec.json')==receipt['spec_sha256']
    for n in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/n)==receipt['hashes'][n]
    ids={f['id'] for f in spec['frames'] if f['split']=='dev'};assert len(ids)==48
    rows=public_observations(selected_jsonl(cap/'raw.jsonl',ids));assert len(rows)==48
    complete=read(prep/'completion.json');seal=read(prep/'prediction-seal.json')
    assert complete['status']=='PASS' and sha(prep/'prediction-seal.json')==complete['prediction_seal_sha256']
    assert sha(prep/'nominal/predictions.json')==seal['predictions_sha256']['nominal']
    assert sha(prep/'observation-seal.json')==seal['observation_seal_sha256']
    assert sha(prep/'nominal/raw.jsonl')==read(prep/'observation-seal.json')['hashes']['nominal']
    assert rows==selected_jsonl(prep/'nominal/raw.jsonl',ids)
    for p,digest in seal['source_hashes'].items():assert sha(Path(p))==digest
    cache=read(prep/'nominal/predictions.json');mapping={p['id']:i for i,p in enumerate(cache['predictions'])}
    base=[cache['predictions'][mapping[r['id']]] for r in rows]
    corrected=[cache['corrected'][mapping[r['id']]] for r in rows]
    radars=[cache['radar'][mapping[r['id']]] for r in rows]
    paths=[cap/n for n in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json')]+[prep/n for n in
        ('completion.json','prediction-seal.json','observation-seal.json','nominal/raw.jsonl','nominal/predictions.json')]
    inputs={str(p):sha(p) for p in paths}
    sources={str(Path(m.__file__).resolve()):sha(Path(m.__file__)) for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().is_relative_to(Path(__file__).parent.resolve())}
    sources[str(Path(__file__).resolve())]=sha(Path(__file__))
    out.mkdir(parents=True);(out/'source-snapshot').mkdir()
    for p in sources:shutil.copyfile(p,out/'source-snapshot'/Path(p).name)
    shutil.copyfile(Path(__file__).with_name('MZ138_SUPPORT_CEILING_20260915.md'),out/'protocol-before-outcomes.md')
    write(out/'freeze.json',dict(method=METHOD,ids=[r['id'] for r in rows],inputs=inputs,sources=sources,
        protocol_sha256=sha(out/'protocol-before-outcomes.md'),python=sys.executable,numpy=np.__version__))
    select_backend('scalar-scoring',cpu=BackendCandidate('native-support-cpu','cpu',
        lambda:sum(float(e['range_m']) for c in corrected for e in c['spatial_evidence']),
        lambda _:DeviceObservation('cpu','host CPU','Python / NumPy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=out/'backend.json')
    # Evaluator truth intentionally enters the oracle here, after method freeze.
    es=selected_jsonl(cap/'evaluator.jsonl',ids);assert [e['id'] for e in es]==[r['id'] for r in rows]
    start=time.perf_counter();frames=[frame_oracles(*v) for v in zip(rows,corrected,base,radars,es)]
    seconds=time.perf_counter()-start
    write(out/'oracle-outputs.json',frames)
    write(out/'oracle-output-seal.json',dict(sha256=sha(out/'oracle-outputs.json'),freeze_sha256=sha(out/'freeze.json'),seconds=seconds,
        authority='EVALUATOR_TRUTH_USED_ORACLE_OUTPUTS_SEALED_BEFORE_FRAME_SCORE_NOT_OBSERVABLE'))
    summary=evaluate(rows,es,frames,spec,radars,out)
    replay=[frame_oracles(*v) for v in zip(rows,corrected,base,radars,es)]
    assert json.loads(json.dumps(replay))==read(out/'oracle-outputs.json')
    assert inputs=={p:sha(Path(p)) for p in inputs};assert sources=={p:sha(Path(p)) for p in sources}
    write(out/'completion.json',dict(status='PASS',exact_replay=True,baseline_reproduced_frames=48,inputs_unchanged=True,
        summary_sha256=sha(out/'summary.json'),oracle_output_seal_sha256=sha(out/'oracle-output-seal.json'),
        resources='No persistent worker/process; durable oracle evidence retained'))
    print(json.dumps(dict(decision=summary['decision'],arms={a:r['metrics'] for a,r in summary['arms'].items()},
        native=summary['coverage_and_retention'],grouping=summary['grouping_attribution'],fixed_radar=summary['fixed_radar_floor']['metrics']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    run(parser.parse_args().output)
