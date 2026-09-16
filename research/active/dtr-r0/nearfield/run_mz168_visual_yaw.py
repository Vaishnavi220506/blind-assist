"""Seal observable visual yaw, then compare saved first-hit range projections."""
import argparse
from collections import Counter
import math
from pathlib import Path
import time

import cv2
import numpy as np

from run_mz161_dense_task import ROOT,CODE,CAP,read,write,sha
from run_mz139_surface_fit import selected_jsonl,local_dependencies
from mz136_incumbent import public_observations
from mz161_dense_task import ray_query
from mz168_visual_yaw import CausalVisualYaw,METHOD
from evaluate_mz136_corridor_pair import score,pair_metrics
from evaluate_mz161_dense_task import _rates,_comparison

WORK=ROOT/'artifacts.local/work/mz168-visual-yaw-20260916'
PARENT=ROOT/'artifacts.local/work/mz167-range-readout-20260916/run-v1'
LIMIT=600.


def spatial(target,pred,known,valid):
    counts=dict(TP=int((known&target&pred).sum()),FP=int((known&~target&pred).sum()),
        FN=int((known&target&~pred).sum()),TN=int((known&~target&~pred).sum()),
        known_pixels=int(known.sum()),unknown_pixels=int((~known).sum()),
        unknown_positive_pixels=int((~known&pred).sum()),public_valid_pixels=int(valid.sum()))
    columns=target.sum(0);detected=(target&pred).sum(0)
    counts.update(native_corridor_columns=int((columns>0).sum()),
        native_corridor_half_detected_columns=int(((columns>0)&(2*detected>=columns)).sum()))
    return counts


def aggregate_spatial(cases,arm,indices):
    counts=Counter()
    for i in indices:counts.update(cases[i]['spatial'][arm])
    value=dict(counts)
    ratio=lambda a,b:float(a/b) if b else None
    value.update(precision=ratio(counts['TP'],counts['TP']+counts['FP']),
        recall=ratio(counts['TP'],counts['TP']+counts['FN']),
        native_corridor_half_column_recall=ratio(counts['native_corridor_half_detected_columns'],counts['native_corridor_columns']))
    return value


def pose_metrics(predictions,audits,indices):
    result=dict(frames=len(indices))
    for arm,key in (('imu','imu_yaw_deg'),('visual','yaw_deg')):
        errors=np.array([((predictions[i][key]-audits[i]['reference_yaw_deg']+180)%360)-180 for i in indices])
        result[arm]=dict(mae_deg=float(np.abs(errors).mean()) if len(errors) else None,
            p95_abs_deg=float(np.quantile(np.abs(errors),.95)) if len(errors) else None,
            max_abs_deg=float(np.abs(errors).max()) if len(errors) else None,
            signed_mean_deg=float(errors.mean()) if len(errors) else None)
    return result


def run(output):
    out=Path(output).resolve();assert out.is_relative_to(WORK.resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter();cv2.setNumThreads(4)
    def within():
        if time.perf_counter()-start>=LIMIT:raise TimeoutError('MZ168 fixed600s exhausted')
    pc=read(PARENT/'completion.json');ps=read(PARENT/'output-seal.json')
    assert pc['status']=='PASS' and sha(PARENT/'output-seal.json')==pc['output_seal_sha256']
    assert sha(PARENT/'freeze.json')==ps['freeze_sha256']
    pf=read(PARENT/'freeze.json');ids=pf['ids'];metadata=pf['metadata'];assert len(ids)==144
    inputs={str(PARENT/n):sha(PARENT/n) for n in ('completion.json','output-seal.json','freeze.json')}
    for name,expected in {**ps['outputs'],**pc['outputs']}.items():
        assert sha(PARENT/name)==expected,name;inputs[str(PARENT/name)]=expected
    for path,expected in pf['inputs'].items():assert sha(path)==expected,path
    receipt=read(CAP/'receipt.json');assert receipt['status']=='PASS'
    assert sha(CAP/'raw.jsonl')==receipt['hashes']['raw.jsonl']
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',set(ids)))
    assert [r['id'] for r in rows]==ids
    for p in (CAP/'receipt.json',CAP/'raw.jsonl',WORK/'tests.log',WORK/'preflight.json'):
        inputs[str(p)]=sha(p)
    for row in rows:
        p=(CAP/row['rgb_path']).resolve();assert p.is_relative_to(CAP.resolve())
        assert sha(p)==receipt['hashes'][row['rgb_path']];inputs[str(p)]=sha(p)
    sources=local_dependencies(__file__)
    for name in ('MZ168_PROTOCOL_20260916.md','test_mz168_visual_yaw.py'):sources[str(CODE/name)]=sha(CODE/name)
    write(out/'freeze.json',dict(method=METHOD,ids=ids,metadata=metadata,inputs=inputs,sources=sources,
        budget_seconds=LIMIT,backend='CPU_OPENCV_NUMPY',backend_reason='GPU_BACKEND_UNAVAILABLE_FOR_TRACKING_AND_HOMOGRAPHY',
        observation_access='FIT144 public raw and RGB only; saved evaluator arrays remain unopened until pose seal'))
    print('MZ168 FROZEN',flush=True);within()
    model=CausalVisualYaw();predictions=[]
    for i,row in enumerate(rows):
        image=cv2.imread(str(CAP/row['rgb_path']));assert image is not None
        value=model.update(row,image);predictions.append(dict(id=row['id'],**value));within()
        if (i+1)%24==0:print(f'POSE {i+1}/144 seconds={time.perf_counter()-start:.2f}',flush=True)
    write(out/'poses.json',predictions)
    write(out/'pose-seal.json',dict(authority='PUBLIC_RGB_IMU_ONLY_BEFORE_SAVED_REFERENCE_DECODE',
        poses_sha256=sha(out/'poses.json'),freeze_sha256=sha(out/'freeze.json')))
    del model
    poses=read(out/'poses.json');assert sha(out/'poses.json')==read(out/'pose-seal.json')['poses_sha256']
    # Evaluator-only saved data becomes available after all observable pose output is sealed.
    oldcases=read(PARENT/'cases.json');audits=read(PARENT/'readout-audit.json')
    assert [c['id'] for c in oldcases]==ids==[a['id'] for a in audits]
    masks=np.load(PARENT/'masks.npy',mmap_mode='r',allow_pickle=False)
    known=np.load(PARENT/'known.npy',mmap_mode='r',allow_pickle=False)
    slant=np.load(PARENT/'reference-slant.npy',mmap_mode='r',allow_pickle=False)
    assert masks.shape==(4,144,360,640) and known.shape==slant.shape==(144,360,640)
    candidate=np.lib.format.open_memmap(out/'candidate-masks.npy',mode='w+',dtype=bool,shape=known.shape)
    cases=[];flags={a:[] for a in ('baseline','native','imu_exact','visual_exact')}
    for i,(row,p,old,a) in enumerate(zip(rows,poses,oldcases,audits)):
        assert math.isfinite(p['yaw_deg']) and abs(p['imu_yaw_deg']-a['public_yaw_deg'])<1e-12
        qi=ray_query(row,p['imu_yaw_deg'])[1];qv=ray_query(row,p['yaw_deg'])[1]
        original=known[i]&(qi[2]>0)&(slant[i]>=qi[0]*4)&(slant[i]<=qi[1]*4)
        assert np.array_equal(original,masks[1,i]),row['id']
        candidate[i]=known[i]&(qv[2]>0)&(slant[i]>=qv[0]*4)&(slant[i]<=qv[1]*4)
        arm_masks=dict(native=masks[0,i],imu_exact=original,visual_exact=candidate[i])
        c=dict(id=row['id'],episode_id=row['episode_id'],time_s=row['time_s'],**metadata[row['id']],
            truth=old['truth'],baseline=old['baseline'],visual_accepted=p['accepted'],flags={},spatial={},
            saved_native_counts=old['native']['baseline'])
        flags['baseline'].append(old['baseline'])
        for arm,mask in arm_masks.items():
            c['flags'][arm]=bool(mask.any());flags[arm].append(c['flags'][arm])
            c['spatial'][arm]=spatial(masks[0,i],mask,known[i],qv[2]>0 if arm=='visual_exact' else qi[2]>0)
        c['changed_pixels_vs_imu']=int((original!=candidate[i]).sum())
        cases.append(c);within()
    candidate.flush()
    write(out/'cases.json',cases)
    write(out/'output-seal.json',dict(authority='EVALUATOR_RANGE_ORACLES_WITH_OBSERVABLE_YAW_NOT_FINAL_PREDICTORS',
        pose_seal_sha256=sha(out/'pose-seal.json'),outputs={n:sha(out/n) for n in ('candidate-masks.npy','cases.json')}))
    flags={a:np.asarray(v,bool) for a,v in flags.items()};gt=np.array([c['truth'] for c in cases],bool)
    families=sorted({c['family'] for c in cases});indices=list(range(144));pairs=[]
    groups={}
    for i,c in enumerate(cases):groups.setdefault((c['scene_group'],c['time_s']),[]).append(i)
    for value in groups.values():
        assert len(value)==2;pairs.append(dict(a=value[0],b=value[1]))
    assert len(pairs)==72
    reports={};spatial_reports={}
    for arm,flag in flags.items():
        r=_rates(score(rows,cases,gt,flag,indices,flags['baseline']))
        r['families']={f:_rates(dict(metrics=m)) for f,m in r['families'].items()}
        r['strata']={name:_rates(score(rows,cases,gt,flag,[i for i,c in enumerate(cases) if (c['family']=='shallow_boundary_stress')==pressure],flags['baseline']))
            for name,pressure in (('ordinary',False),('boundary_pressure',True))}
        r['pairs']=pair_metrics(gt,flag.astype(float),flag,pairs)
        r['native']=dict(returned_contributor_samples=sum(c['saved_native_counts'].get('returned_contributor_samples',0) for c in cases),
            corridor_contributor_samples=sum(c['saved_native_counts'].get('corridor_contributor_samples',0) for c in cases),
            nonalert_corridor_samples=sum(c['saved_native_counts'].get('corridor_contributor_samples',0) for i,c in enumerate(cases) if not flag[i]),
            supported_nonalerts=[dict(id=c['id'],corridor_samples=c['saved_native_counts'].get('corridor_contributor_samples',0)) for i,c in enumerate(cases)
                if not flag[i] and c['saved_native_counts'].get('corridor_contributor_samples',0)>0],radar_lineage='NOT_EVALUABLE')
        reports[arm]=r
        if arm!='baseline':spatial_reports[arm]=dict(all=aggregate_spatial(cases,arm,indices),
            families={f:aggregate_spatial(cases,arm,[i for i,c in enumerate(cases) if c['family']==f]) for f in families})
    for reference in ('baseline','native','imu_exact'):
        reports['visual_exact']['vs_'+reference]=_comparison(rows,gt,flags['visual_exact'],flags[reference],indices,reports['visual_exact'],reports[reference])
    accepted=[i for i,p in enumerate(poses) if p['accepted']]
    boundary=[i for i,c in enumerate(cases) if c['family']=='shallow_boundary_stress']
    pose=dict(all=pose_metrics(poses,audits,indices),accepted=pose_metrics(poses,audits,accepted),
        boundary=pose_metrics(poses,audits,boundary),families={f:pose_metrics(poses,audits,[i for i,c in enumerate(cases) if c['family']==f]) for f in families},
        accepted_frames=len(accepted),eligible_noninitial_frames=sum(r['time_s']>0 for r in rows),
        reasons=dict(Counter(p['audit']['reason'] for p in poses)))
    r=reports['visual_exact'];native=r['vs_native'];baseline=r['vs_baseline']
    gates=dict(any_visual_accepted=bool(accepted),all_frame_yaw_mae_improved=pose['all']['visual']['mae_deg']<pose['all']['imu']['mae_deg'],
        boundary_yaw_mae_improved=pose['boundary']['visual']['mae_deg']<pose['boundary']['imu']['mae_deg'],
        all_native_true_frames_retained=native['all_reference_true_frames_retained'],no_false_frames=r['metrics']['FP']==0,
        native_events_onsets_retained=native['all_reference_events_without_extra_delay'],
        baseline_events_onsets_retained=baseline['all_reference_events_without_extra_delay'],
        family_pixel_precision_recall_90=all(v[k] is not None and v[k]>=.9 for v in spatial_reports['visual_exact']['families'].values() for k in ('precision','recall')),
        native_supported_nonalerts_zero=r['native']['nonalert_corridor_samples']==0)
    passed=all(gates.values());gates['overall_pass']=passed
    summary=dict(authority='CONSUMED_FIT144_OBSERVABLE_YAW_WITH_EVALUATOR_RANGE_NOT_ALERT_GAIN',frames=144,pose=pose,
        arms=reports,spatial=spatial_reports,gates=gates,decision='MZ168_VISUAL_POSE_INTERFACE_ELIGIBLE' if passed else 'MZ168_FIXED_VISUAL_YAW_NOT_ADMITTED',
        limits=['No learned range, real-system alert integration, new capture or fresh confirmation.',
            'Static dominant plane and public initial alignment are assumptions, not certified by match count.',
            'Saved native counts reused; original evaluator JSONL and HELD/dev/test remain undecoded.'])
    write(out/'summary.json',summary)
    for p,h in sources.items():assert sha(p)==h,p
    for p,h in inputs.items():assert sha(p)==h,p
    within();write(out/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,decision=summary['decision'],
        output_seal_sha256=sha(out/'output-seal.json'),summary_sha256=sha(out/'summary.json'),
        model_invocations=0,resources='CPU process only; no persistent allocation'))
    print(summary['decision'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'run-v1');args=parser.parse_args()
    try:run(args.output)
    except Exception as exc:
        if args.output.exists():write(args.output/'failure.json',dict(error=type(exc).__name__,message=str(exc),resume='NO_AUTOMATIC_RESTART'))
        raise
