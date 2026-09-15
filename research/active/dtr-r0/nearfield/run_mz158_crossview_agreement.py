"""Frozen public inference, then separate evaluation of one fresh MZ158 cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
CODE=Path(__file__).resolve().parent
WORK=ROOT/'artifacts.local/work/mz158-crossview-agreement-20260916'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sources():
    return {str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'
        and Path(m.__file__).resolve().is_relative_to(CODE)}


def observations(capture):
    from mz136_incumbent import public_observations
    recipe=read(WORK/'recipe-freeze.json');design=read(WORK/'source/design-v1/freeze.json')
    assert sha(WORK/'recipe-freeze.json')==design['recipe_freeze']['sha256']==design['files']['recipe-freeze.json']
    receipt=read(capture/'receipt.json');assert receipt['status']=='PASS' and receipt['frames']==288
    assert sha(capture/'spec.json')==receipt['spec_sha256']==sha(WORK/'source/design-v1/spec.json')==design['files']['spec.json']
    for p,h in recipe['sources'].items():assert sha(p)==h,p
    for p,h in recipe['inputs'].items():assert sha(p)==h,p
    for name,h in receipt['hashes'].items():assert sha(capture/name)==h,name
    rows=public_observations([json.loads(s) for s in (capture/'raw.jsonl').read_text().splitlines()])
    assert len(rows)==len({r['id'] for r in rows})==288
    assert all(r['id'].startswith('mz158_') for r in rows)
    return rows,receipt,recipe


def infer(capture,out,mode):
    import cv2
    import numpy as np
    assert out.is_relative_to(WORK.resolve()) and not out.exists()
    rows,receipt,recipe=observations(capture);out.mkdir(parents=True);start=time.perf_counter()
    inputs={str(capture/n):sha(capture/n) for n in ('raw.jsonl','receipt.json','spec.json')}
    for row in rows:
        path=(capture/row['rgb_path']).resolve();assert path.is_relative_to(capture)
        inputs[str(path)]=sha(path)
    runner_sha256=sha(__file__)
    write(out/'input-seal.json',dict(inputs=inputs,recipe_freeze_sha256=sha(WORK/'recipe-freeze.json'),runner_sha256=runner_sha256,
        source_freeze_sha256=sha(WORK/'source/design-v1/freeze.json'),
        authority='OBSERVATIONS_ONLY_BEFORE_FRESH_EVALUATOR_DECODE'))
    def loader(row):
        image=cv2.imread(str(capture/row['rgb_path']));assert image.shape==(360,640,3)
        return image
    if mode=='baseline':
        from mz136_incumbent import predict_incumbent
        assert cv2.__version__=='5.0.0'
        prior=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916/incumbent-v1/prediction-seal.json'
        bound=read(prior)['source_hashes']
        for p,h in bound.items():assert sha(p)==h,p
        result=predict_incumbent(rows,loader)
        predictions=result['predictions'];write(out/'incumbent-details.json',result)
        context=dict(mode='UNCHANGED_MZ129',prior_source_seal_sha256=sha(prior),audit=result['audit'])
    else:
        import sklearn
        from threadpoolctl import threadpool_limits
        from mz158_crossview_agreement import RawResidual,load_models,decisions,extract
        from research_backend import BackendCandidate,DeviceObservation,select_backend
        assert cv2.__version__=='4.10.0'
        bound=recipe['sources'];models,cuts=load_models();assert cuts==recipe['cutoffs']
        select_backend('batch-tensor',cpu=BackendCandidate('opencv-sklearn-cpu','cpu',
            lambda:extract(rows[0],loader(rows[0]),rows[0]['delta_yaw'] if rows[0]['imu_valid'] else 0.),
            lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__+' sklearn '+sklearn.__version__)),
            cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=out/'backend.json',
            capabilities={'reason':'Inherited NumPy/OpenCV features and sklearn HGB have no implemented GPU backend'})
        raw=RawResidual();static=[];temporal=[];audit=[];episode=None;yaw=0.;tick=time.perf_counter()
        with threadpool_limits(limits=4):
            for i,row in enumerate(rows):
                if episode!=row['episode_id']:yaw=0.
                if row['imu_valid']:yaw+=row['delta_yaw']
                episode=row['episode_id'];image=loader(row);value=extract(row,image,yaw)
                static.append(np.r_[value['sensor'],value['geometry']]);temporal.append(raw.update(row,image))
                audit.append(dict(id=row['id'],**value['audit']))
                assert time.perf_counter()-start<600
                if (i+1)%48==0:print(json.dumps(dict(stage='fresh_features',frames=i+1)),flush=True)
            static=np.stack(static);temporal=np.stack(temporal);feature_seconds=time.perf_counter()-tick
            tick=time.perf_counter();scores,flags=decisions(rows,static,temporal,models,cuts)
            readout_seconds=time.perf_counter()-tick
        np.savez_compressed(out/'features.npz',static=static,raw_change=temporal)
        write(out/'feature-audit.json',audit)
        predictions=[dict(id=r['id'],scores={k:float(v[i]) for k,v in scores.items()},
            flags={k:bool(v[i]) for k,v in flags.items()},candidate_state='ALERT' if flags['agreement'][i] else 'UNKNOWN')
            for i,r in enumerate(rows)]
        context=dict(mode='FIXED_MZ145_AND_MZ148_UNREGISTERED',cutoffs=cuts,
            features_sha256=sha(out/'features.npz'),feature_audit_sha256=sha(out/'feature-audit.json'),
            feature_seconds=feature_seconds,readout_seconds=readout_seconds,
            python=sys.executable,numpy=np.__version__,opencv=cv2.__version__,sklearn=sklearn.__version__)
    write(out/'predictions.json',predictions)
    for p,h in bound.items():assert sha(p)==h,p
    assert all(sha(p)==h for p,h in inputs.items())
    assert sha(__file__)==runner_sha256
    write(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.json'),
        input_seal_sha256=sha(out/'input-seal.json'),sources=sources(),context=context,
        authority='SEALED_FRESH_PUBLIC_PREDICTIONS_BEFORE_EVALUATOR_PARSE'))
    write(out/'completion.json',dict(status='PASS',frames=len(rows),seconds=time.perf_counter()-start,
        prediction_seal_sha256=sha(out/'prediction-seal.json'),resources='Process-local CPU models; process exits'))
    print(json.dumps(dict(stage='sealed',mode=mode,frames=len(rows),seconds=time.perf_counter()-start)),flush=True)


def evaluate(capture,out):
    import numpy as np
    from run_mz143_corridor_evidence import augmented_score,native_account,truth
    from evaluate_mz136_corridor_pair import retention
    from mz136_paired_source import intersects
    assert out.is_relative_to(WORK.resolve()) and not out.exists()
    rows,receipt,recipe=observations(capture);out.mkdir(parents=True);start=time.perf_counter()
    bindings={str(WORK/'recipe-freeze.json'):sha(WORK/'recipe-freeze.json'),str(Path(__file__)):sha(__file__)}
    predictions={};seals={}
    for name in ('baseline','learned'):
        folder=WORK/(name+'-v1');seal=read(folder/'prediction-seal.json');done=read(folder/'completion.json')
        assert done['status']=='PASS' and sha(folder/'prediction-seal.json')==done['prediction_seal_sha256']
        assert sha(folder/'predictions.json')==seal['predictions_sha256']
        assert sha(folder/'input-seal.json')==seal['input_seal_sha256']
        input_seal=read(folder/'input-seal.json')
        assert input_seal['recipe_freeze_sha256']==sha(WORK/'recipe-freeze.json')
        assert input_seal['source_freeze_sha256']==sha(WORK/'source/design-v1/freeze.json')
        assert input_seal['runner_sha256']==sha(__file__)
        for p,h in input_seal['inputs'].items():assert sha(p)==h,p
        assert input_seal['inputs'][str(capture/'raw.jsonl')]==sha(capture/'raw.jsonl')
        if name=='learned':
            assert sha(folder/'features.npz')==seal['context']['features_sha256']
            assert sha(folder/'feature-audit.json')==seal['context']['feature_audit_sha256']
        for p,h in seal['sources'].items():assert sha(p)==h,p
        pp=read(folder/'predictions.json');assert [p['id'] for p in pp]==[r['id'] for r in rows]
        predictions[name]=pp;seals[name]=sha(folder/'prediction-seal.json')
    write(out/'evaluation-start.json',dict(bindings=bindings,prediction_seals=seals,
        authority='BOTH_PREDICTION_ARMS_SEALED_BEFORE_NATIVE_DECODE'))
    spec=read(capture/'spec.json');es=[json.loads(s) for s in (capture/'evaluator.jsonl').read_text().splitlines()]
    assert [e['id'] for e in es]==[r['id'] for r in rows]
    frames={f['id']:f for f in spec['frames']};errors=[];mismatches=[]
    for e in es:
        frame=frames[e['id']];native={o['name']:o for o in e['native_bounds']}
        assert set(native)=={o['name'] for o in frame['objects']}
        errors.append(float(np.max(np.abs(np.array(e['body_origin_m'])-frame['body_origin_m']))))
        for obj in frame['objects']:
            actual=native[obj['name']]
            errors.extend([float(np.max(np.abs(np.array(actual['center_m'])-obj['center_m']))),
                           float(np.max(np.abs(2*np.array(actual['extent_m'])-obj['size_m'])))])
        if any(intersects(frame,obj) for obj in frame['objects'])!=bool(truth(e)):mismatches.append(e['id'])
    admission=dict(status='PASS' if max(errors)<=1e-5 and not mismatches else 'FAIL',
        frames=len(rows),max_geometry_error_m=max(errors),label_mismatches=mismatches)
    write(out/'native-admission.json',admission);assert admission['status']=='PASS'
    y=np.array([truth(e) for e in es],bool)
    flags={'baseline':np.array([p['candidate'] for p in predictions['baseline']],bool)}
    for arm in ('static','raw_change','agreement'):
        flags[arm]=np.array([p['flags'][arm] for p in predictions['learned']],bool)
    assert np.array_equal(flags['agreement'],flags['static']&flags['raw_change'])
    reports={name:augmented_score(rows,es,y,v.astype(float),.5,flags['baseline'],spec,'confirmation') for name,v in flags.items()}
    native={name:native_account(rows,es,v,flags['baseline']) for name,v in flags.items()}
    for value in native.values():value['radar_native_lineage']='NOT_EVALUABLE: inherited source has no per-return Radar lineage'
    write(out/'native-contributors.json',native)
    candidate=reports['agreement'];checks={};comparisons={}
    for name,required_gain in (('baseline',.30),('static',.10)):
        base=reports[name];rr=retention(candidate,base)
        lost=[r['id'] for r,gt,b,c in zip(rows,y,flags[name],flags['agreement']) if gt and b and not c]
        gain=1-candidate['metrics']['FP']/max(1,base['metrics']['FP'])
        conditions=dict(required_fp_reduction=gain>=required_gain,
            precision_improved=candidate['precision']>base['precision'],no_true_frame_lost=not lost,
            no_lost_or_delayed_event=all(v['relative_delay_s'] is not None and v['relative_delay_s']<=1e-9 for v in rr['per_event_delta']),
            all_family_fp_noninferior=all(candidate['families'][f]['FP']<=base['families'][f]['FP'] for f in base['families']),
            false_segments_noninferior=candidate['events']['false_alert_segments']<=base['events']['false_alert_segments'])
        checks.update({name+'_'+k:bool(v) for k,v in conditions.items()})
        comparisons[name]=dict(relative_fp_reduction=gain,lost_true_frames=lost,retention=rr,checks=conditions)
    checks['native_returned_corridor_retention']=not native['agreement']['nonalert_with_native_corridor_contributors']
    cases=[dict(id=r['id'],episode_id=r['episode_id'],time_s=r['time_s'],family=e['family'],truth=bool(gt),
        flags={k:bool(v[i]) for k,v in flags.items()},scores=predictions['learned'][i]['scores'])
        for i,(r,e,gt) in enumerate(zip(rows,es,y))]
    write(out/'cases.json',cases)
    passed=all(checks.values());result=dict(reports=reports,comparisons=comparisons,checks=checks,passed=passed,
        decision='FRESH_CONTROLLED_AGREEMENT_GAIN' if passed else 'FRESH_CONTROLLED_AGREEMENT_NOT_MET',
        authority='NEW_CONFIGURATIONS_SAME_GENERATOR_SIMULATION_ONLY',frames=len(rows),
        no_model_or_cutoff_changes=True,original_test_access=False,seconds=time.perf_counter()-start)
    write(out/'summary.json',result)
    assert all(sha(p)==h for p,h in bindings.items());assert time.perf_counter()-start<600
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        prediction_seals=seals,cases_sha256=sha(out/'cases.json'),sources_inputs_unchanged=True))
    print(json.dumps(dict(decision=result['decision'],metrics={k:v['metrics'] for k,v in reports.items()},checks=checks),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('baseline','learned','evaluate'),required=True)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='evaluate':evaluate(args.capture.resolve(),args.output.resolve())
    else:infer(args.capture.resolve(),args.output.resolve(),args.mode)
