"""One sealed public replay, then native audit on consumed MZ155 evidence."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from mz156_stationary_witness import METHOD,predict
from run_mz155_active_sampling import ROOT,CODE,WORK as PRIOR,observations,read,write,sha
from run_mz139_surface_fit import local_dependencies
from run_mz143_corridor_evidence import augmented_score,native_account,truth
from mz155_active_sampling import check_source
from mz155_sampling_audit import audit_sampling

WORK=ROOT/'artifacts.local/work/mz156-stationary-witness-20260916'


def prior_predictions(capture,rows):
    inputs={str(capture/name):sha(capture/name) for name in ('raw.jsonl','receipt.json','spec.json')}
    inputs.update({str((capture/r['rgb_path']).resolve()):sha(capture/r['rgb_path']) for r in rows})
    predictions={};bound={}
    for name in ('baseline','learned'):
        folder=PRIOR/(name+'-v1');seal=read(folder/'prediction-seal.json');done=read(folder/'completion.json')
        assert done['status']=='PASS' and sha(folder/'prediction-seal.json')==done['prediction_seal_sha256']
        assert sha(folder/'predictions.json')==seal['predictions_sha256']
        assert sha(folder/'input-seal.json')==seal['input_seal_sha256']
        assert read(folder/'input-seal.json')['inputs']==inputs
        assert sha(folder/'predictor-binding.json')==seal['predictor_binding_sha256']
        predictions[name]=read(folder/'predictions.json')
        assert [p['id'] for p in predictions[name]]==[r['id'] for r in rows]
        assert all(isinstance(p['candidate'],bool) for p in predictions[name])
        for filename in ('completion.json','prediction-seal.json','predictions.json','input-seal.json','predictor-binding.json'):
            bound[str(folder/filename)]=sha(folder/filename)
    return predictions,inputs|bound


def witness_audit(rows,es,predictions):
    """Evaluate each returned contributor without feeding native identity to inference."""
    rowmap={r['id']:r for r in rows};emap={e['id']:e for e in es}
    pmap={p['id']:p for p in predictions};details=[];violations=[];prefix_errors=[]
    contributor_count=0;prefix_count=0
    for p in predictions:
        r=rowmap[p['id']];e=emap[p['id']]
        zones={z['zone_id']:z for z in r['tof_zones']}
        natives={z['zone_id']:z for z in e['zonal_tof_native']}
        for w in p['current_supports']:
            z=zones[w['zone_id']];n=natives[w['zone_id']]
            assert z['targets'][w['target_slot']]['status']=='SIM_VALID'
            rays={v['subray']:v for v in n['private_rays']}
            hits=[rays[h] for line in n['returned_lineage'] if line['target_index']==w['target_slot'] for h in line['hit_indices']]
            record=dict(id=p['id'],zone_id=w['zone_id'],slot=w['target_slot'],contributors=len(hits),violations=[])
            if not hits:record['violations'].append(dict(reason='NO_NATIVE_LINEAGE'))
            for ray in hits:
                point=[float(v-o) for v,o in zip(ray['hit_point_m'],e['body_origin_m'])]
                enclosed=all(lo-1e-9<=v<=hi+1e-9 for v,(lo,hi) in zip(point,w['xyz_m']))
                corridor=all(lo-1e-9<=v<=hi+1e-9 for v,(lo,hi) in zip(point,METHOD['corridor_m']))
                if not enclosed or not corridor:
                    record['violations'].append(dict(point_body_m=point,enclosed=enclosed,in_corridor=corridor))
                contributor_count+=1
            details.append(record)
            if record['violations']:violations.append(record)
        for w in p['prefix_supports']:
            prefix_count+=1;source=pmap.get(w['source_id']);age=p['time_s']-w['source_time_s']
            original=dict(w,evidence_age_s=0.)
            valid=(source is not None and source['episode_id']==p['episode_id'] and p['active']
                and 0<=age<=1.25+1e-9 and abs(age-w['evidence_age_s'])<1e-9
                and original in source['current_supports'])
            if not valid:prefix_errors.append(dict(id=p['id'],witness=w))
    return dict(current_witnesses=len(details),native_contributor_samples=contributor_count,
        prefix_references=prefix_count,violations=violations,prefix_errors=prefix_errors,cases=details,
        passed=not violations and not prefix_errors,
        scope='RETURNED_SAMPLES_NOT_FULL_OBJECT_SURFACE; NOMINAL_BOUNDS_NOT_SIMULTANEOUS_GUARANTEE')


def observation_compare(rows,gt,flags,spec,reports):
    byid={r['id']:(bool(y),bool(p)) for r,y,p in zip(rows,gt,flags)};cases=[]
    for pair in spec['observation_pairs']:
        pp=[f for f in spec['frames'] if f['episode']==pair['passive']]
        ss=[f for f in spec['frames'] if f['episode']==pair['scan']]
        yy=[byid[f['id']][0] for f in pp]
        assert yy==[byid[f['id']][0] for f in ss] and len(set(yy))==1
        a=[byid[f['id']][1] for f in pp];b=[byid[f['id']][1] for f in ss];y=yy[0]
        firsta=next((f['time_s'] for f,p in zip(pp,a) if p),None)
        firstb=next((f['time_s'] for f,p in zip(ss,b) if p),None)
        cases.append(dict(matched_case=pair['matched_case'],family=pp[0]['family'],truth=y,
            passive=a,scan=b,passive_first_s=firsta,scan_first_s=firstb,
            lost_true_frames=sum(x and not z for x,z in zip(a,b)) if y else 0,
            new_true_frames=sum(z and not x for x,z in zip(a,b)) if y else 0,
            removed_false_frames=sum(x and not z for x,z in zip(a,b)) if not y else 0,
            new_false_frames=sum(z and not x for x,z in zip(a,b)) if not y else 0,
            event_lost=bool(y and any(a) and not any(b)),
            excessive_delay=bool(y and firsta is not None and (firstb is None or firstb>firsta+.25))))
    a=reports['passive'];b=reports['scan'];gain=(a['metrics']['FP']-b['metrics']['FP'])/max(1,a['metrics']['FP'])
    checks=dict(fp_reduction_at_least20pct=gain>=.2,no_true_frame_lost=not any(c['lost_true_frames'] for c in cases),
        no_event_lost=not any(c['event_lost'] for c in cases),no_excessive_delay=not any(c['excessive_delay'] for c in cases),
        no_family_fp_increase=all(b['families'][f]['FP']<=a['families'][f]['FP'] for f in a['families']))
    return dict(relative_fp_reduction=gain,checks=checks,passed=all(checks.values()),cases=cases)


def run(out):
    started=time.perf_counter();out=out.resolve();capture=(PRIOR/'source/returned-v1/capture-v1').resolve()
    assert out.is_relative_to(WORK.resolve()) and not out.exists()
    rows,receipt=observations(capture);prior,inputs=prior_predictions(capture,rows)
    source_freeze=PRIOR/'source/design-v1/freeze.json'
    for path,h in read(source_freeze)['source_inputs'].items():assert sha(path)==h,path
    inputs.update({str(p):sha(p) for p in [source_freeze,capture/'evaluator.jsonl']})
    sources=local_dependencies(__file__)
    sources.update(local_dependencies(CODE/'test_mz156_stationary_witness.py'))
    sources[str(CODE/'MZ156_PROTOCOL_20260916.md')]=sha(CODE/'MZ156_PROTOCOL_20260916.md')
    out.mkdir(parents=True)
    for path in sources:
        source=Path(path);dest=out/'source-snapshot'/source.relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    write(out/'freeze.json',dict(method=METHOD,sources=sources,inputs=inputs,frames=len(rows),
        authority='CONSUMED_MZ155_DEVELOPMENT_EXPLICIT_STATIC_BODY_POSITION_CAMERA_POSITION_WORLD',
        backend='CPU_TASK_NOT_GPU_SUITABLE',python=sys.executable,compute_cap_s=120,
        original_test_access=False,fit_or_threshold_selection=False,native_decode_before_prediction=False))
    baseline=[p['candidate'] for p in prior['baseline']]
    tick=time.perf_counter();predictions=predict(rows,baseline,stationary_world=True);predict_seconds=time.perf_counter()-tick
    write(out/'predictions.json',predictions)
    assert all(sha(p)==h for p,h in (sources|inputs).items())
    write(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.json'),
        freeze_sha256=sha(out/'freeze.json'),predict_seconds=predict_seconds,
        authority='PUBLIC_ONLY_PREDICTIONS_SEALED_BEFORE_THIS_NATIVE_REPLAY'))
    print(json.dumps(dict(stage='SEALED',frames=len(predictions),seconds=predict_seconds)),flush=True)
    assert time.perf_counter()-started<120
    write(out/'evaluation-start.json',dict(prediction_seal_sha256=sha(out/'prediction-seal.json'),
        native_sha256=sha(capture/'evaluator.jsonl')))
    spec=read(capture/'spec.json');check_source(spec)
    es=[json.loads(s) for s in (capture/'evaluator.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==[e['id'] for e in es]
    sampling=audit_sampling(spec,rows,es);assert sampling['native_admission']['status']=='PASS'
    write(out/'native-admission.json',sampling['native_admission'])
    audit=witness_audit(rows,es,predictions);write(out/'witness-audit.json',audit)
    gt=np.array([truth(e) for e in es],bool);base=np.array(baseline,bool)
    flagmap=dict(baseline=base,retained_mz145=np.array([p['candidate'] for p in prior['learned']],bool))
    for key in ('current_candidate','prefix_candidate','current_witness','prefix_witness'):
        flagmap[key]=np.array([p[key] for p in predictions],bool)
    metadata={f['id']:f for f in spec['frames']};reports={};changes={};compare={}
    for name,flags in flagmap.items():
        reports[name]={}
        for arm in ('passive','scan'):
            ix=[i for i,r in enumerate(rows) if metadata[r['id']]['observation_arm']==arm]
            rr=[rows[i] for i in ix];ee=[es[i] for i in ix];pairs=[p for p in spec['pairs'] if p['pair_id'].endswith('_'+arm)]
            report=augmented_score(rr,ee,gt[ix],flags[ix].astype(float),.5,base[ix],dict(pairs=pairs),'development')
            report['native']=native_account(rr,ee,flags[ix],base[ix]);report['unknown_frames']=int((~flags[ix]).sum())
            reports[name][arm]=report
        compare[name]=observation_compare(rows,gt,flags,spec,reports[name])
    for name,old,new in [('current_over_baseline','baseline','current_candidate'),
                         ('prefix_over_current','current_candidate','prefix_candidate'),
                         ('prefix_over_baseline','baseline','prefix_candidate')]:
        changes[name]={}
        for arm in ('passive','scan'):
            added=[]
            for i,(row,p) in enumerate(zip(rows,predictions)):
                if metadata[row['id']]['observation_arm']!=arm:continue
                assert not flagmap[old][i] or flagmap[new][i]
                if flagmap[new][i] and not flagmap[old][i]:
                    added.append(dict(id=row['id'],truth=bool(gt[i]),family=es[i]['family'],time_s=row['time_s'],
                        witnesses=p['prefix_supports'] if new=='prefix_candidate' else p['current_supports']))
            changes[name][arm]=dict(added_true_frames=sum(c['truth'] for c in added),
                added_false_frames=sum(not c['truth'] for c in added),cases=added)
    memory=changes['prefix_over_current'];total=changes['prefix_over_baseline']
    checks=dict(adds_true_frame_over_current=any(c['added_true_frames'] for c in memory.values()),
        no_added_false_frame=all(c['added_false_frames']==0 for c in total.values()),native_containment_and_prefix_valid=audit['passed'])
    passed=all(checks.values())
    summary=dict(reports=reports,changes=changes,paired_observation_effect=compare,
        component_gate=dict(checks=checks,passed=passed),witness_audit_sha256=sha(out/'witness-audit.json'),
        decision='RETAIN_CONSUMED_STATIC_CONDITION_COMPONENT_ONLY' if passed else 'STOP_FIXED_STATIC_MEMORY_RECIPE',
        authority='CONSUMED_STATIC_SIMULATION_NOT_FRESH_WALKING_DEVICE_OR_DEFAULT',
        fp_effect='Within-arm OR cannot reduce FP; scan-vs-passive changes include inherited MZ155 effects')
    write(out/'summary.json',summary)
    assert all(sha(p)==h for p,h in (sources|inputs).items())
    seconds=time.perf_counter()-started;assert seconds<120
    write(out/'completion.json',dict(status='PASS',frames=len(rows),seconds=seconds,predict_seconds=predict_seconds,
        summary_sha256=sha(out/'summary.json'),prediction_seal_sha256=sha(out/'prediction-seal.json'),
        witness_audit_sha256=sha(out/'witness-audit.json'),inputs_sources_unchanged=True,resources='CPU process exits'))
    print(json.dumps(dict(metrics={n:{a:v['metrics'] for a,v in arms.items()} for n,arms in reports.items()},
        component_gate=summary['component_gate'],combined_gate=compare['prefix_candidate']['passed'],
        witnesses=audit['current_witnesses'],native_violations=len(audit['violations']),seconds=seconds),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'run-v1')
    args=parser.parse_args();run(args.output)
