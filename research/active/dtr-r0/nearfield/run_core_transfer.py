"""One immutable controlled transfer; freeze/capture/materialize/predict/evaluate.

No optimizer, fit, threshold selection, crop selection or alternate checkpoint.
Native depth is accessible only in source construction and post-seal evaluation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from evaluate_ba_camera_corridor import read, sha, require, write_new, primitive_truth, validate_readiness
from core_transfer_spec import specification, bounds, classify, PROFILE
from ba_camera_corridor import sample_native, rays
from tof_fov45_core import simulate, boxes45
from tof_corridor_calibration import score_frame, decide
from tof_lateral_core import tiny_head, local_input, eligible, learned_decision, frame_decision, CLASSES

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/ba-core-transfer-20260920'
OLD = ROOT/'artifacts.local/work/ba-tof-lateral-attribution-20260920'
CAL = ROOT/'artifacts.local/work/ba-tof-corridor-calibration-20260920'
IDENTITY = ('id','clip_id','frame_in_clip','time_s')
CODE = ('core_transfer_spec.py','core_transfer_capture.py','launch_core_transfer.py','run_core_transfer.py',
        'core_transfer_metrics.py','test_core_transfer_metrics.py','test_core_transfer_spec.py',
        'ba_camera_corridor.py','ba_camera_corridor_metrics.py','tof_fov45_core.py',
        'tof_corridor_calibration.py','tof_lateral_core.py','evaluate_ba_camera_corridor.py','ue_capture_readiness.py')


def freeze(out):
    require(not out.exists(), 'No overwrite or scientific retry')
    spec = specification()
    require(len(spec['cases']) == 432 and len(spec['clips']) == 36, 'Fixed budget')
    seals = read(OLD/'model-seal.json')
    for name, digest in seals['checkpoints'].items():
        require(sha(OLD/name) == digest, 'Frozen checkpoint differs')
    for name in ('tof_lateral_core.py','tof_corridor_calibration.py','ba_camera_corridor.py','tof_fov45_core.py'):
        # Git object comparison protects the original recipe without editing it.
        old = subprocess.check_output(['git','show','d844c347:research/active/dtr-r0/nearfield/'+name], cwd=ROOT)
        require(old.replace(b'\r\n',b'\n') == Path(__file__).with_name(name).read_bytes().replace(b'\r\n',b'\n'), 'Recipe differs from d844c347: '+name)
    out.mkdir(parents=True)
    write_new(out/'spec.json', spec)
    report = Path(__file__).with_name('CORE_TRANSFER_PROTOCOL_20260920.md')
    (out/'protocol-before-run.md').write_bytes(report.read_bytes())
    protocol = dict(id='ba-core-transfer-20260920', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='FROZEN_NEW_CONTROLLED_ARRANGEMENT_TRANSFER_NOT_NATURAL_OR_HARDWARE',
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frames=432, clips=36, frames_per_clip=12, dt_s=.2, training_updates=0, profile=PROFILE,
        threshold=read(CAL/'operating-point.json')['threshold'], outside_cutoff=.95,
        weights=seals['checkpoints'], spec_sha256=sha(out/'spec.json'),
        protocol_text_sha256=sha(out/'protocol-before-run.md'),
        code_hashes={n:sha(Path(__file__).with_name(n)) for n in CODE},
        input_hashes={str(p.relative_to(ROOT).as_posix()):sha(p) for p in
                      (OLD/'C.pt',OLD/'C_no_rgb.pt',OLD/'model-seal.json',CAL/'operating-point.json')},
        stop='One complete432 evaluation; no fit, new checkpoint, crop, cutoff or structure changes; source failure is NOT_EVALUABLE; mechanical repair only with unchanged hypothesis and receipts')
    write_new(out/'protocol.json', protocol)
    print(json.dumps(dict(status='FROZEN', protocol_sha256=sha(out/'protocol.json'))))


def verify(out):
    p = read(out/'protocol.json')
    require(p['spec_sha256'] == sha(out/'spec.json'), 'Spec changed')
    require(p['protocol_text_sha256'] == sha(out/'protocol-before-run.md'), 'Protocol text changed')
    for name, digest in p['code_hashes'].items():
        require(sha(Path(__file__).with_name(name)) == digest, 'Code changed: '+name)
    for name, digest in p['input_hashes'].items():
        require(sha(ROOT/name) == digest, 'Input changed: '+name)
    return p


def check_capture(out):
    p = verify(out); cap=out/'capture'
    receipt, launch = read(cap/'receipt.json'), read(cap/'launch-receipt.json')
    require(receipt['status'] == 'PASS' and receipt['frame_count'] == 432, 'Incomplete source')
    require(receipt['source_unchanged'] and receipt['task_actors_released'] and read(cap/'process-release.json')['released'], 'Source/resource failure')
    require(receipt['protocol_sha256'] == launch['protocol_sha256'] == sha(out/'protocol.json'), 'Capture protocol mismatch')
    require(receipt['spec_sha256'] == launch['spec_sha256'] == p['spec_sha256'], 'Capture spec mismatch')
    require(receipt['script_sha256'] == launch['capture_script_sha256'] == p['code_hashes']['core_transfer_capture.py'], 'Capture code mismatch')
    require(launch['launcher_sha256'] == p['code_hashes']['launch_core_transfer.py'], 'Launcher mismatch')
    require(receipt['readiness_helper_sha256'] == launch['readiness_helper_sha256'] == p['code_hashes']['ue_capture_readiness.py'], 'Readiness mismatch')
    plugin=Path(launch['plugin_path'])
    require(sha(plugin)==launch['plugin_sha256'] and sha(plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll')==launch['plugin_binary_sha256'], 'Plugin changed')
    require(len(receipt['view_readiness']) == 432, 'Readiness coverage')
    for i,r in enumerate(receipt['view_readiness']):
        require(r['sample_index'] == i, 'Readiness order')
        validate_readiness(r)
    spec=read(out/'spec.json')
    require(receipt['map_sha256_before']==receipt['map_sha256_after']==spec['expected_map_sha256'], 'Map mismatch')
    for name,digest in launch['assets'].items():
        require(sha(name)==digest, 'Capture asset changed')
    return spec,read(cap/'observations/manifest.json'),read(cap/'evaluator/geometry.json')


def masks(native, case, geo):
    obj=next(o for o in geo['objects'] if o['name']=='target')
    center,extent=np.array(obj['render_bounds_center_m']),np.array(obj['render_bounds_extent_m'])
    yy,xx=np.mgrid[:360,:640];f=640/(2*np.tan(np.deg2rad(50)))
    a,b=(xx+.5-320)/f,(yy+.5-180)/f;p=case['camera']
    world=np.stack([native+p['x'],a*native+p['y'],p['z']-b*native],-1)
    known=np.isfinite(native)&(native>0)
    target=known&np.all((world>=center-extent-.02)&(world<=center+extent+.02),axis=-1)
    inside=known&(native>=.3)&(native<=3)&(abs(a*native)<=.3)&(b*native>=-.2)&(b*native<=.9)
    return target,inside


def source_check(case,geo,row,index):
    require(case['name']==geo['id']==row['id'] and geo['sample_index']==row['sample_index']==index,'Source identity')
    require(case['clip_id']==geo['clip_id']==row['clip_id'] and case['frame_in_clip']==geo['frame_in_clip']==row['frame_in_clip'],'Source sequence')
    require(case['time_s']==geo['nominal_time_s']==row['time_s'],'Source timestamps')
    require(len(case['objects'])==len(geo['objects'])==2,'Object inventory')
    require(geo['declared_camera']==case['camera'],'Camera declaration')
    require(max(abs(geo['actual_camera_location_m'][i]-case['camera'][k]) for i,k in enumerate(('x','y','z')))<=.002,'Camera position')
    require(max(abs(v) for v in geo['actual_camera_rotation'])<=1e-6,'Camera rotation')
    for declared,actual in zip(case['objects'],geo['objects']):
        require(declared['name']==actual['name'] and actual['mesh_path']=='/Engine/BasicShapes/Cube.Cube','Mesh identity')
        require(actual['material_path']==declared['material']+'.'+declared['material'].split('/')[-1],'Material identity')
        require(max(abs(a-b) for a,b in zip(actual['render_bounds_center_m'],declared['center_m']))<=.002,'Object position')
        require(max(abs(2*a-b) for a,b in zip(actual['render_bounds_extent_m'],declared['size_m']))<=.002,'Object size')
    target=next(o for o in geo['objects'] if o['name']=='target')
    require(target['trace']['blocking'] and target['trace']['hit_expected_actor'] and target['trace']['hit_actor_path']==target['actor_path'],'Native target trace')
    warm=32 if case['frame_in_clip']==0 else 16
    require(geo['unchanged_warmup_ticks']==warm and geo['post_ready_rgb_render_calls']==warm+1,'Settling changed')
    validate_readiness(geo['readiness'])


def materialize(out):
    spec,manifest,geometry=check_capture(out);cap=out/'capture'
    require(not (out/'observations').exists(),'Materialization already attempted')
    (out/'observations').mkdir()
    require(len(spec['cases'])==len(manifest['frames'])==len(geometry)==432,'Source count')
    observed,private,checks=[],[],[]
    for i,(case,row,geo) in enumerate(zip(spec['cases'],manifest['frames'],geometry)):
        source_check(case,geo,row,i)
        native_path=cap/'evaluator'/geo['native_path'];rgb_path=cap/'observations'/row['rgb_path']
        require(sha(native_path)==geo['native_sha256'] and sha(rgb_path)==row['rgb_sha256']==geo['rgb_sha256'],'Native/RGB hashes')
        native=np.load(native_path,allow_pickle=False);require(native.shape==(360,640),'Native shape')
        target,inside=masks(native,case,geo)
        checks.append(dict(id=row['id'],visible_target_pixels=int(target.sum()),competing_corridor_pixels=int((inside&~target).sum())))
        identity='core-transfer-v1/'+case['name']
        values,traces=simulate(sample_native(native),identity,boxes45())
        path=out/'observations'/f'f{i:04d}.npz'
        np.savez_compressed(path,boxes=boxes45(),values=values)
        observed.append(dict(id=f'f{i:04d}',clip_id=row['clip_id'],frame_in_clip=row['frame_in_clip'],time_s=row['time_s'],
            path=path.relative_to(out).as_posix(),sha256=sha(path),rgb_path=rgb_path.relative_to(out).as_posix(),rgb_sha256=sha(rgb_path)))
        private.append(dict(id=f'f{i:04d}',seed=identity,native_sha256=sha(native_path),
            traces=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in t.items()} for t in traces]))
    passed=all(r['visible_target_pixels']>0 and r['competing_corridor_pixels']<4 for r in checks)
    write_new(out/'source-admission.json',dict(status='PASS' if passed else 'NOT_EVALUABLE',checks=checks))
    write_new(out/'observations.json',observed);write_new(out/'private-lineage.json',private)
    require(passed,'Source admission failed; no exclusions or inference')
    write_new(out/'observation-seal.json',dict(status='COMPLETE',frames=432,protocol_sha256=sha(out/'protocol.json'),
        hashes={n:sha(out/n) for n in ('observations.json','private-lineage.json','source-admission.json','capture/evaluator/geometry.json','capture/observations/manifest.json')}))
    print(json.dumps(dict(status='OBSERVATIONS_SEALED',frames=432)))


def check_seal(out,name):
    seal=read(out/name)
    require(seal['status']=='COMPLETE' and seal['frames']==432 and seal['protocol_sha256']==sha(out/'protocol.json'),'Stage identity')
    for n,digest in seal['hashes'].items():require(sha(out/n)==digest,'Sealed data changed: '+n)


def predict(out):
    p=verify(out);check_seal(out,'observation-seal.json')
    require(not (out/'inference-start.json').exists(),'One prediction pass only')
    import torch
    require(torch.cuda.is_available(),'Frozen GPU unavailable')
    models={}
    for arm,filename in (('rgb','C.pt'),('no_rgb','C_no_rgb.pt')):
        checkpoint=torch.load(OLD/filename,map_location='cpu',weights_only=True)
        require(tuple(checkpoint['classes'])==CLASSES,'Class order')
        model=tiny_head();model.load_state_dict(checkpoint['state_dict']);model.eval().to('cuda')
        require(sum(q.numel() for q in model.parameters())==5915,'Model architecture')
        models[arm]=model
    write_new(out/'inference-start.json',dict(protocol_sha256=sha(out/'protocol.json'),device=torch.cuda.get_device_name(0),
        torch=torch.__version__,python=sys.executable,training_updates=0,weights=p['weights'],
        placement='CUDA GPU-first frozen small tensor inference; CPU OpenCV crops and scalar geometry',
        placement_reference_sha256=sha(OLD/'backend.json')))
    predictions,samples=[],[];began=time.perf_counter();gpu_seconds=0.;calls=0
    for obs in read(out/'observations.json'):
        require(sha(out/obs['path'])==obs['sha256'] and sha(out/obs['rgb_path'])==obs['rgb_sha256'],'Observation changed')
        with np.load(out/obs['path'],allow_pickle=False) as data:boxes,values=data['boxes'],data['values']
        score=score_frame(boxes,values);cal=decide(score,p['threshold'])
        rgb=cv2.cvtColor(cv2.imread(str(out/obs['rgb_path'])),cv2.COLOR_BGR2RGB)
        require(rgb.shape==(360,640,3),'RGB shape')
        inputs,zones=[],[]
        for anchor in score['anchors']:
            z=anchor['zone']
            if eligible(boxes[z],values[z],anchor):
                zones.append(z);inputs.append(local_input(rgb,boxes[z],float(values[z]),anchor['interval_m']))
        outputs={arm:[] for arm in models}
        if inputs:
            x=torch.from_numpy(np.stack(inputs)).to('cuda')
            for arm,model in models.items():
                inp=x.clone()
                if arm=='no_rgb':inp[:,:3]=0
                torch.cuda.synchronize();t=time.perf_counter()
                with torch.inference_mode():probs=model(inp).softmax(-1)
                torch.cuda.synchronize();gpu_seconds+=time.perf_counter()-t;calls+=1
                require(bool(torch.isfinite(probs).all()),'Nonfinite probabilities')
                outputs[arm]=[dict(probabilities=v,**learned_decision(v)) for v in probs.cpu().tolist()]
        suppressed={arm:[z for z,d in zip(zones,outputs[arm]) if d['suppress']] for arm in models}
        arms=dict(raw=score['baseline'],calibrated=cal)
        for arm in models:arms[arm]=frame_decision(cal,score['anchors'],score['zone_scores'],p['threshold'],set(suppressed[arm]))
        for j,z in enumerate(zones):samples.append(dict(sample_id=obs['id']+':z'+str(z),frame_id=obs['id'],zone=z,outputs={arm:outputs[arm][j] for arm in models}))
        predictions.append({**{k:obs[k] for k in IDENTITY},'observation_sha256':obs['sha256'],
            'predictions':arms,'suppressed_zones':suppressed,'anchors':score['anchors'],'zone_scores':score['zone_scores']})
    write_new(out/'predictions.json',predictions);write_new(out/'zone-decisions.json',samples)
    write_new(out/'prediction-seal.json',dict(status='COMPLETE',frames=432,protocol_sha256=sha(out/'protocol.json'),
        hashes={n:sha(out/n) for n in ('predictions.json','zone-decisions.json','observation-seal.json','inference-start.json')},
        elapsed_s=time.perf_counter()-began,gpu_model_seconds=gpu_seconds,model_calls=calls,training_updates=0))
    del models;torch.cuda.empty_cache()
    print(json.dumps(dict(status='PREDICTIONS_SEALED',frames=432,eligible_samples=len(samples))))


def evaluate(out):
    from core_transfer_metrics import evaluate as metrics
    p=verify(out);check_seal(out,'prediction-seal.json');check_seal(out,'observation-seal.json')
    require(not (out/'results.json').exists(),'One evaluation only')
    spec,manifest,geometry=check_capture(out)
    observations,predictions,lineages=read(out/'observations.json'),read(out/'predictions.json'),read(out/'private-lineage.json')
    samples=read(out/'zone-decisions.json');by_frame={}
    for s in samples:by_frame.setdefault(s['frame_id'],[]).append(s)
    rows,audits=[],[]
    a,b=rays()
    require(len(observations)==len(predictions)==len(lineages)==432,'Evaluation count')
    for i,(case,geo,obs,pred,lin) in enumerate(zip(spec['cases'],geometry,observations,predictions,lineages)):
        source_check(case,geo,manifest['frames'][i],i)
        require(obs['id']==pred['id']==lin['id']==f'f{i:04d}','Evaluation identity')
        require(all(obs[k]==pred[k] for k in IDENTITY),'Prediction timeline')
        require(all(obs[k]==case[k] for k in ('clip_id','frame_in_clip','time_s')),'Source-to-prediction timeline')
        require(sha(out/obs['path'])==obs['sha256']==pred['observation_sha256'] and sha(out/obs['rgb_path'])==obs['rgb_sha256'],'Input binding')
        native_path=out/'capture/evaluator'/geo['native_path']
        require(sha(native_path)==geo['native_sha256']==lin['native_sha256'],'Native binding')
        native=np.load(native_path,allow_pickle=False);depth=sample_native(native)
        target,inside=masks(native,case,geo)
        require(target.any() and (inside&~target).sum()<4,'Source admission recheck')
        native_target=sample_native(target);native_inside=sample_native(inside)
        values,traces=simulate(depth,lin['seed'],boxes45())
        with np.load(out/obs['path'],allow_pickle=False) as data:
            require(np.array_equal(values,data['values'],equal_nan=True) and np.array_equal(boxes45(),data['boxes']),'Sensor reproduction')
        require([{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in t.items()} for t in traces]==lin['traces'],'Lineage reproduction')
        truth=primitive_truth(dict(case=case,geometry=geo),PROFILE)
        expected=classify(*bounds(case))
        require(truth['truth']==expected['truth'] and truth['boundary']==expected['boundary'],'Geometry reproduction')
        score=score_frame(boxes45(),values);cal=decide(score,p['threshold'])
        require(pred['anchors']==score['anchors'] and pred['zone_scores']==score['zone_scores'],'Raw support immutability')
        require(pred['predictions']['raw']==score['baseline'] and pred['predictions']['calibrated']==cal,'Baseline parity')
        for arm in ('rgb','no_rgb'):
            suppressed={s['zone'] for s in by_frame.get(obs['id'],[]) if s['outputs'][arm]['suppress']}
            require(sorted(suppressed)==pred['suppressed_zones'][arm],'Suppression binding')
            require(pred['predictions'][arm]==frame_decision(cal,score['anchors'],score['zone_scores'],p['threshold'],suppressed),'Final decision parity')
        for s in by_frame.get(obs['id'],[]):
            z=s['zone'];t=traces[z];pixels=t['pixel_indices'];owned=native_target.ravel()[pixels]
            anchor=next(x for x in score['anchors'] if x['zone']==z)
            require(eligible(boxes45()[z],values[z],anchor),'Ineligible suppression')
            low,high=expected['target_camera_bounds_m']['lower'][0],expected['target_camera_bounds_m']['upper'][0]
            relation=('OUTSIDE' if high<-.3-1e-9 or low>.3+1e-9 else 'INSIDE' if low>-.3+1e-9 and high<.3-1e-9 else 'CROSSING') if len(owned) and owned.all() else 'UNKNOWN'
            for arm in ('rgb','no_rgb'):
                d=s['outputs'][arm];require(all(d[k]==v for k,v in learned_decision(d['probabilities']).items()),'Cutoff changed')
            audits.append(dict(sample_id=s['sample_id'],frame_id=obs['id'],truth_relation=relation,
                rgb_relation=s['outputs']['rgb']['relation'],no_rgb_relation=s['outputs']['no_rgb']['relation'],
                rgb_suppress=s['outputs']['rgb']['suppress'],no_rgb_suppress=s['outputs']['no_rgb']['suppress'],
                native_corridor_contributors=int(native_inside.ravel()[pixels].sum()),
                contributor_count=len(pixels),target_contributor_count=int(owned.sum())))
        rows.append({**{k:obs[k] for k in IDENTITY},**expected,
            **{k:case[k] for k in ('layout_relation','layer','type_id','background','arrangement_id')},'predictions':pred['predictions']})
    result=metrics(rows,audits)
    result['scope']=spec['independence'];result['protocol_sha256']=sha(out/'protocol.json')
    result['training_updates']=0;result['sampled_timing_not_latency']=True
    write_new(out/'frame-results.json',rows);write_new(out/'native-zone-audit.json',audits);write_new(out/'results.json',result)
    write_new(out/'evaluation-seal.json',dict(status='COMPLETE',frames=432,protocol_sha256=sha(out/'protocol.json'),
        hashes={n:sha(out/n) for n in ('frame-results.json','native-zone-audit.json','results.json','prediction-seal.json')}))
    print(json.dumps(dict(status='EVALUATED',primary_gate=result['primary_gate'],calibration_gate=result['calibration_gate'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('freeze','materialize','predict','evaluate'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    require(args.out.resolve().is_relative_to((ROOT/'artifacts.local').resolve()),'Canonical artifact route required')
    globals()[args.phase](args.out)
