"""No-training fresh controlled transfer of frozen MZ143 + MZ145 pipeline."""
import argparse
import json
from pathlib import Path
import pickle
import time
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from mz143_corridor_features import extract
from mz145_causal_confirmation import predict
from run_mz143_corridor_evidence import (ROOT,read,sha,write,truth,public_observations,
    augmented_score,native_account)
from evaluate_mz136_corridor_pair import retention

MODEL=ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
TEMPORAL=ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'
DESIGN=ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916/source/design-v1'


def run(capture,incumbent,out):
    capture=capture.resolve();incumbent=incumbent.resolve();out=out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter()
    source_freeze=read(DESIGN/'freeze.json')
    spec=read(capture/'spec.json');receipt=read(capture/'receipt.json')
    assert receipt['status']=='PASS' and receipt['frames']==288
    assert sha(capture/'spec.json')==receipt['spec_sha256']
    assert sha(capture/'spec.json')==source_freeze['files']['spec.json']==sha(DESIGN/'spec.json')
    addendum=DESIGN.parent/'acceptance-addendum.json'
    assert addendum.is_file(),'Pre-outcome family acceptance addendum must be retained'
    for name,digest in receipt['hashes'].items():assert sha(capture/name)==digest,name
    model_seal=read(MODEL/'model-seal.json');onset=read(TEMPORAL/'onset-seal.json')
    selected=model_seal['selected'];assert selected=='fused_hgb' and onset['source_arm']==selected
    assert sha(MODEL/(selected+'.pkl'))==model_seal['models'][selected]
    assert sha(MODEL/(selected+'.pkl'))==source_freeze['candidate']['model_sha256']
    assert onset['low']==source_freeze['candidate']['low'] and onset['high']==source_freeze['candidate']['high']
    assert sha(MODEL/'model-seal.json')==onset['model_seal_sha256']
    frozen=read(MODEL/'freeze.json')
    for code in ('mz143_corridor_features.py','mz125_observable_correction.py','mz136_boundary_geometry.py'):
        path=Path(__file__).with_name(code).resolve();assert sha(path)==frozen['sources'][str(path)]
    causal=Path(__file__).with_name('mz145_causal_confirmation.py').resolve()
    assert sha(causal)==onset['sources'][str(causal)]
    names=read(MODEL/'feature-names.json')
    rows=public_observations([json.loads(s) for s in (capture/'raw.jsonl').read_text(encoding='utf-8').splitlines()])
    assert len(rows)==len({r['id'] for r in rows})==288
    assert all(r['id'].startswith('mz146_') for r in rows)
    split=spec['pairs'][0]['split'];assert all(p['split']==split for p in spec['pairs'])
    with (MODEL/(selected+'.pkl')).open('rb') as stream:model=pickle.load(stream)
    write(out/'freeze.json',dict(source_spec_sha256=sha(capture/'spec.json'),
        model_sha256=sha(MODEL/(selected+'.pkl')),model_seal_sha256=sha(MODEL/'model-seal.json'),
        temporal_seal_sha256=sha(TEMPORAL/'onset-seal.json'),low=onset['low'],high=onset['high'],
        source_binding=source_freeze,source_freeze_sha256=sha(DESIGN/'freeze.json'),
        acceptance_addendum_sha256=sha(addendum),model_refit=False,threshold_selection=False,
        authority='UNCHANGED_PIPELINE_FRESH_SCENE_GROUPS_SAME_GENERATOR_CONTROLLED_CONFIRMATION'))
    features=[];audit=[];yaw=0.;episode=None;tick=time.perf_counter()
    for i,row in enumerate(rows):
        if row['episode_id']!=episode:yaw=0.
        if row['imu_valid']:yaw+=row['delta_yaw']
        episode=row['episode_id'];image=cv2.imread(str(capture/row['rgb_path']))
        value=extract(row,image,yaw)
        assert all(value[key]==names[key] for key in ('sensor_names','geometry_names'))
        features.append(np.r_[value['sensor'],value['geometry']]);audit.append(dict(id=row['id'],**value['audit']))
        if (i+1)%48==0:print(json.dumps(dict(stage='fresh_features',frames=i+1,seconds=time.perf_counter()-tick)),flush=True)
    features=np.stack(features);np.savez_compressed(out/'features.npz',features=features)
    write(out/'feature-audit.json',audit);feature_seconds=time.perf_counter()-tick
    tick=time.perf_counter();scores=model.predict_proba(features)[:,1]
    flags=predict(rows,scores,onset['low'],onset['high']);model_seconds=time.perf_counter()-tick
    write(out/'predictions.json',[dict(id=r['id'],score=float(s),candidate=bool(p),
        candidate_state='ALERT' if p else 'UNKNOWN') for r,s,p in zip(rows,scores,flags)])
    write(out/'prediction-seal.json',dict(predictions_sha256=sha(out/'predictions.json'),
        features_sha256=sha(out/'features.npz'),feature_audit_sha256=sha(out/'feature-audit.json'),
        freeze_sha256=sha(out/'freeze.json'),
        authority='FRESH_PREDICTIONS_SEALED_BEFORE_EVALUATOR_PARSE',feature_seconds=feature_seconds,readout_seconds=model_seconds))
    inc_seal=read(incumbent/'prediction-seal.json');inc_done=read(incumbent/'completion.json')
    assert inc_done['status']=='PASS' and sha(incumbent/'prediction-seal.json')==inc_done['prediction_seal_sha256']
    assert sha(incumbent/'predictions.json')==inc_seal['predictions_sha256']
    inc=read(incumbent/'predictions.json')['predictions'];assert [r['id'] for r in rows]==[p['id'] for p in inc]
    baseline=np.array([p['candidate'] for p in inc],bool)
    es=[json.loads(s) for s in (capture/'evaluator.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['id'] for r in rows]==[e['id'] for e in es]
    from mz136_paired_source import intersects
    frames={f['id']:f for f in spec['frames']};max_error=0.;mismatches=[]
    for e in es:
        frame=frames[e['id']];native={o['name']:o for o in e['native_bounds']}
        assert set(native)=={o['name'] for o in frame['objects']}
        errors=[float(np.max(np.abs(np.asarray(e['body_origin_m'])-frame['body_origin_m'])))]
        for obj in frame['objects']:
            actual=native[obj['name']]
            errors.extend([float(np.max(np.abs(np.asarray(actual['center_m'])-obj['center_m']))),
                float(np.max(np.abs(2*np.asarray(actual['extent_m'])-obj['size_m'])))])
        max_error=max(max_error,*errors)
        designed=any(intersects(frame,obj) for obj in frame['objects'])
        if designed!=bool(truth(e)):mismatches.append(e['id'])
    admission=dict(status='PASS' if max_error<=1e-5 and not mismatches else 'FAIL',
        frames=len(es),max_source_native_geometry_error_m=max_error,label_mismatches=mismatches,
        authority='SOURCE_NATIVE_ADMISSION_AFTER_PREDICTION_SEAL_NOT_PREDICTOR_INPUT')
    write(out/'source-admission.json',admission)
    assert admission['status']=='PASS','Source-native admission failed; do not claim model failure'
    gt=np.array([truth(e) for e in es],bool)
    br=augmented_score(rows,es,gt,baseline.astype(float),.5,baseline,spec,split)
    sr=augmented_score(rows,es,gt,scores,onset['low'],baseline,spec,split)
    cr=augmented_score(rows,es,gt,flags.astype(float),.5,baseline,spec,split)
    cr['pairs']['ordering_semantics']='BINARY_CAUSAL_DECISIONS_WITH_TIES'
    cr['retention']=retention(cr,br)
    family_ok=True
    for f in br['families']:
        a,b=cr['families'][f],br['families'][f]
        family_ok &= a['FP']<=b['FP'] and a['TP']/max(1,a['TP']+a['FN'])>=b['TP']/max(1,b['TP']+b['FN'])-.02-1e-9
    fp_gain=1-cr['metrics']['FP']/max(1,br['metrics']['FP'])
    passed=bool(cr['retention']['pass_retention'] and fp_gain>=.3 and cr['precision']>br['precision']
        and cr['events']['false_alert_segments']<=br['events']['false_alert_segments'] and family_ok)
    contributor_audit=native_account(rows,es,flags,baseline)
    contributor_audit['tof_authority']='RETURNED_NATIVE_TOF_RAY_CONTRIBUTORS_ONLY'
    contributor_audit['radar']=dict(native_lineage_status='NOT_EVALUABLE',
        reason='Capture evaluator emits native AABBs and ToF ray lineage, no per-return Radar contributor mapping',
        public_slots_per_frame=4,all_public_slots_encoded=True,
        valid_public_returns=sum(sum(bool(v) for v in row['radar_valid']) for row in rows),
        native_bounds_do_not_establish_return_ownership=True)
    write(out/'native-contributors.json',contributor_audit)
    result=dict(baseline=br,single_frame=sr,candidate=cr,family_noninferior=bool(family_ok),
        relative_fp_reduction=fp_gain,joint_confirmation=passed,
        decision='FRESH_CONTROLLED_CONFIRMATION_GAIN' if passed else 'FRESH_CONTROLLED_CONFIRMATION_NOT_MET',
        source_frames=len(rows),source_scene_groups=len(spec['scene_groups']),
        seconds=dict(features=feature_seconds,readout=model_seconds,total=time.perf_counter()-start),
        model_or_threshold_changes=False,authority='NEW_SCENE_GROUPS_SAME_PROCEDURAL_GENERATOR_SIMULATION_ONLY')
    write(out/'summary.json',result)
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        prediction_seal_sha256=sha(out/'prediction-seal.json'),resource_state='CPU inference exits; no allocations'))
    print(json.dumps(dict(decision=result['decision'],baseline=br['metrics'],candidate=cr['metrics'],
        events=cr['events'],relative_fp_reduction=fp_gain,family_noninferior=bool(family_ok)),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('capture','incumbent','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.capture,args.incumbent,args.output)
