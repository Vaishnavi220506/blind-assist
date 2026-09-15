"""Frozen observation-only predictors, followed by separately sealed evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work/mz155-active-sampling-20260916'
CODE = Path(__file__).resolve().parent
MODEL = ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
ONSET = ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sources():
    return {str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'
        and Path(m.__file__).resolve().is_relative_to(CODE)}


def observations(capture):
    from mz136_incumbent import public_observations
    receipt=read(capture/'receipt.json')
    assert receipt['status']=='PASS' and receipt['frames']==288
    assert sha(capture/'spec.json')==receipt['spec_sha256']==read(WORK/'source/design-v1/freeze.json')['files']['spec.json']
    for name,h in receipt['hashes'].items():assert sha(capture/name)==h,name
    rows=public_observations([json.loads(s) for s in (capture/'raw.jsonl').read_text().splitlines()])
    assert len(rows)==len({r['id'] for r in rows})==288
    assert all(r['id'].startswith('mz155_') for r in rows)
    return rows,receipt


def infer(capture, out, arm):
    import cv2
    import numpy as np
    from mz136_incumbent import predict_incumbent
    assert out.is_relative_to(WORK.resolve()) and not out.exists()
    rows,receipt=observations(capture);out.mkdir(parents=True);start=time.perf_counter()
    inputs={str(capture/name):sha(capture/name) for name in ('raw.jsonl','receipt.json','spec.json')}
    for row in rows:
        image=(capture/row['rgb_path']).resolve();assert image.is_relative_to(capture)
        inputs[str(image)]=sha(image)
    write(out/'input-seal.json',dict(inputs=inputs,authority='PUBLIC_OBSERVATIONS_ONLY_NO_EVALUATOR_DECODE'))
    def loader(row):
        image=cv2.imread(str(capture/row['rgb_path']));assert image is not None and image.shape==(360,640,3)
        return image
    if arm=='baseline':
        assert cv2.__version__=='5.0.0'
        prior=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916/incumbent-v1/prediction-seal.json'
        prior_sources=read(prior)['source_hashes']
        bound_sources=prior_sources
        for path,h in bound_sources.items():assert sha(path)==h,path
        write(out/'predictor-binding.json',dict(authenticated_sources=bound_sources,
            prior_seal_sha256=sha(prior),runner_sha256=sha(__file__)))
        result=predict_incumbent(rows,loader)
        write(out/'predictions.json',result['predictions'])
        write(out/'incumbent-details.json',result)
        context=dict(model='UNCHANGED_MZ129',audit=result['audit'],prior_source_seal_sha256=sha(prior),
            prior_sources_verified=len(prior_sources))
    else:
        import pickle
        from threadpoolctl import threadpool_limits
        from mz143_corridor_features import extract
        from mz145_causal_confirmation import predict
        model_seal=read(MODEL/'model-seal.json');onset=read(ONSET/'onset-seal.json')
        source_seal=read(MODEL/'freeze.json');design=read(WORK/'source/design-v1/freeze.json')
        assert sha(MODEL/'fused_hgb.pkl')==model_seal['models']['fused_hgb']==design['model_sha256']
        assert sha(ONSET/'onset-seal.json')==design['onset_sha256']
        bound_sources=dict(source_seal['sources'])
        bound_sources[str(CODE/'mz145_causal_confirmation.py')]=onset['sources'][str(CODE/'mz145_causal_confirmation.py')]
        for path,h in bound_sources.items():assert sha(path)==h,path
        assert onset['low']==.020014435971556582 and onset['high']==.2957935335969224
        assert cv2.__version__=='4.10.0'
        names=read(MODEL/'feature-names.json')
        write(out/'predictor-binding.json',dict(authenticated_sources=bound_sources,
            feature_source_seal_sha256=sha(MODEL/'freeze.json'),onset_seal_sha256=sha(ONSET/'onset-seal.json'),
            runner_sha256=sha(__file__)))
        with (MODEL/'fused_hgb.pkl').open('rb') as f:model=pickle.load(f)
        features=[];audit=[];yaw=0.;episode=None
        with threadpool_limits(limits=4):
            for i,row in enumerate(rows):
                if episode!=row['episode_id']:yaw=0.
                if row['imu_valid']:yaw+=row['delta_yaw']
                episode=row['episode_id'];value=extract(row,loader(row),yaw)
                assert all(value[k]==names[k] for k in ('sensor_names','geometry_names'))
                features.append(np.r_[value['sensor'],value['geometry']]);audit.append(dict(id=row['id'],**value['audit']))
                if (i+1)%48==0:print(json.dumps(dict(stage='features',frames=i+1)),flush=True)
            features=np.stack(features);assert np.isfinite(features).all()
            scores=model.predict_proba(features)[:,1]
            flags=predict(rows,scores,onset['low'],onset['high'])
        np.savez_compressed(out/'features.npz',features=features)
        write(out/'feature-audit.json',audit)
        write(out/'predictions.json',[dict(id=r['id'],score=float(s),candidate=bool(p),
            candidate_state='ALERT' if p else 'UNKNOWN') for r,s,p in zip(rows,scores,flags)])
        context=dict(model='UNCHANGED_MZ143_MZ145',model_sha256=sha(MODEL/'fused_hgb.pkl'),
            temporal_seal_sha256=sha(ONSET/'onset-seal.json'),low=onset['low'],high=onset['high'],
            features_sha256=sha(out/'features.npz'),feature_audit_sha256=sha(out/'feature-audit.json'))
    for path,h in bound_sources.items():assert sha(path)==h,path
    assert sha(__file__)==read(out/'predictor-binding.json')['runner_sha256']
    assert all(sha(p)==h for p,h in inputs.items())
    write(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.json'),
        input_seal_sha256=sha(out/'input-seal.json'),sources=sources(),context=context,
        predictor_binding_sha256=sha(out/'predictor-binding.json'),bound_sources_and_inputs_unchanged=True,
        backend='CPU_GPU_BACKEND_UNAVAILABLE_FROZEN_OPENCV_SKLEARN_IMPLEMENTATION',
        python=sys.executable,opencv=cv2.__version__,numpy=np.__version__,seconds=time.perf_counter()-start,
        authority='SEALED_PUBLIC_ONLY_PREDICTIONS_BEFORE_NATIVE_EVALUATION'))
    write(out/'completion.json',dict(status='PASS',frames=len(rows),
        prediction_seal_sha256=sha(out/'prediction-seal.json'),resources='Process exits'))
    print(json.dumps(dict(status='PASS',arm=arm,seconds=time.perf_counter()-start)),flush=True)


def evaluate(capture,out):
    import numpy as np
    from mz155_active_sampling import check_source
    from mz155_sampling_audit import audit_sampling
    from run_mz143_corridor_evidence import augmented_score,native_account,truth
    assert out.is_relative_to(WORK.resolve()) and not out.exists()
    source_inputs=read(WORK/'source/design-v1/freeze.json')['source_inputs']
    assert all(sha(p)==h for p,h in source_inputs.items()),'Prospective source/protocol changed'
    rows,receipt=observations(capture);predictions={};seals={}
    expected_inputs={str(capture/name):sha(capture/name) for name in ('raw.jsonl','receipt.json','spec.json')}
    expected_inputs.update({str((capture/r['rgb_path']).resolve()):sha(capture/r['rgb_path']) for r in rows})
    for name in ('baseline','learned'):
        folder=WORK/(name+'-v1');seal=read(folder/'prediction-seal.json');done=read(folder/'completion.json')
        assert done['status']=='PASS' and sha(folder/'prediction-seal.json')==done['prediction_seal_sha256']
        assert sha(folder/'predictions.json')==seal['predictions_sha256']
        assert sha(folder/'input-seal.json')==seal['input_seal_sha256']
        assert read(folder/'input-seal.json')['inputs']==expected_inputs
        assert sha(folder/'predictor-binding.json')==seal['predictor_binding_sha256']
        predictions[name]=read(folder/'predictions.json');seals[name]=sha(folder/'prediction-seal.json')
        assert [p['id'] for p in predictions[name]]==[r['id'] for r in rows]
    out.mkdir(parents=True)
    write(out/'evaluation-start.json',dict(prediction_seals=seals,source_sha256=sha(__file__),
        evaluator_sources=sources(),native_file_sha256=sha(capture/'evaluator.jsonl')))
    spec=read(capture/'spec.json');check_source(spec)
    es=[json.loads(s) for s in (capture/'evaluator.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==[e['id'] for e in es]
    sampling=audit_sampling(spec,rows,es);write(out/'sampling-audit.json',sampling)
    assert sampling['native_admission']['status']=='PASS'
    gt=np.array([truth(e) for e in es],bool)
    frame_meta={f['id']:f for f in spec['frames']};reports={};compare={}
    flagmap={n:np.array([p['candidate'] for p in predictions[n]],bool) for n in predictions}
    for name,flags in flagmap.items():
        reports[name]={}
        for arm in ('passive','scan'):
            ix=[i for i,r in enumerate(rows) if frame_meta[r['id']]['observation_arm']==arm]
            rr=[rows[i] for i in ix];ee=[es[i] for i in ix]
            pairs=[p for p in spec['pairs'] if p['pair_id'].endswith('_'+arm)]
            report=augmented_score(rr,ee,gt[ix],flags[ix].astype(float),.5,flagmap['baseline'][ix],dict(pairs=pairs),'development')
            report['native']=native_account(rr,ee,flags[ix],flagmap['baseline'][ix])
            reports[name][arm]=report
        pm={r['id']:bool(p) for r,p in zip(rows,flags)}
        cases=[]
        for pair in spec['observation_pairs']:
            pp=[f for f in spec['frames'] if f['episode']==pair['passive']]
            ss=[f for f in spec['frames'] if f['episode']==pair['scan']]
            label=pp[0]['pair_member']=='in';pflags=[pm[f['id']] for f in pp];sflags=[pm[f['id']] for f in ss]
            firstp=next((f['time_s'] for f in pp if pm[f['id']]),None)
            firsts=next((f['time_s'] for f in ss if pm[f['id']]),None)
            cases.append(dict(matched_case=pair['matched_case'],family=pp[0]['family'],truth=label,
                passive=pflags,scan=sflags,passive_first_s=firstp,scan_first_s=firsts,
                lost_true_frames=sum(p and not s for p,s in zip(pflags,sflags)) if label else 0,
                new_true_frames=sum(s and not p for p,s in zip(pflags,sflags)) if label else 0,
                removed_false_frames=sum(p and not s for p,s in zip(pflags,sflags)) if not label else 0,
                new_false_frames=sum(s and not p for p,s in zip(pflags,sflags)) if not label else 0,
                event_lost=bool(label and any(pflags) and not any(sflags)),
                excessive_delay=bool(label and firstp is not None and (firsts is None or firsts>firstp+.25))))
        base=reports[name]['passive'];scan=reports[name]['scan']
        fp_gain=(base['metrics']['FP']-scan['metrics']['FP'])/max(1,base['metrics']['FP'])
        checks=dict(fp_reduction_at_least20pct=fp_gain>=.2,
            no_true_frame_lost=not any(c['lost_true_frames'] for c in cases),
            no_event_lost=not any(c['event_lost'] for c in cases),
            no_excessive_delay=not any(c['excessive_delay'] for c in cases),
            no_family_fp_increase=all(scan['families'][f]['FP']<=base['families'][f]['FP'] for f in base['families']))
        compare[name]=dict(relative_fp_reduction=fp_gain,checks=checks,passed=all(checks.values()),cases=cases)
    summary=dict(reports=reports,paired_observation_effect=compare,
        sampling_gate=sampling['component_gate'],sampling_audit_sha256=sha(out/'sampling-audit.json'),
        authority='FROZEN_PREDICTORS_ON_NEW_CONTROLLED_STATIONARY_WORLD_SAME_SIMULATOR',
        decision='FIXED_ANGULAR_SAMPLING_COMPARISON_COMPLETE_NO_AUTOMATIC_PROMOTION',
        original_data_or_protected_test_decoded=False,model_fit_or_threshold_change=False)
    write(out/'summary.json',summary)
    assert all(sha(p)==h for p,h in source_inputs.items()),'Prospective source/protocol changed during evaluation'
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        prediction_seals=seals,source_inputs_unchanged=True))
    print(json.dumps(dict(alert_metrics={n:{a:v['metrics'] for a,v in arms.items()} for n,arms in reports.items()},
        sampling_gate=sampling['component_gate'],alert_gates={n:v['passed'] for n,v in compare.items()}),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('baseline','learned','evaluate'),required=True)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cap=args.capture.resolve();out=args.output.resolve()
    if args.mode=='evaluate':evaluate(cap,out)
    else:infer(cap,out,args.mode)
