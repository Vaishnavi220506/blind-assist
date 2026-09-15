"""Seal standalone relative-prior/public-calibration predictions, then evaluate."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import torch
from mz136_incumbent import public_observations
from mz157_relative_calibration import METHOD,load_prior,relative_prediction,calibrate
from run_mz139_surface_fit import selected_jsonl,local_dependencies
from run_mz155_active_sampling import ROOT,CODE,read,write,sha
from evaluate_mz140_depthor import evaluate_frame

WORK=ROOT/'artifacts.local/work/mz157-relative-calibration-20260916'
SOURCE=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'
OLD=ROOT/'artifacts.local/work/mz140-depthor-20260915'
ARMS=('median','regional_modes')


def summarize(cases,audits):
    result={}
    for arm in ARMS:
        groups={}
        for family in sorted({c['family'] for c in cases}):
            rr=[c['arms'][arm] for c in cases if c['family']==family]
            cols=sum(r['target']['transverse_columns']['visible'] for r in rr)
            good=sum(r['target']['transverse_columns']['with_at_least_half_pixels_agreeing'] for r in rr)
            px=sum(r['target']['visible_pixels'] for r in rr);comparable=sum(r['target']['comparable_pixels'] for r in rr)
            true=sum(r['target']['range_agreeing_pixels'] for r in rr)
            false=sum(r['silhouette_neighborhood']['target_depth_band_spurious_pixels'] for r in rr)
            ring=sum(r['silhouette_neighborhood']['referenced_pixels'] for r in rr)
            over=sum(r['silhouette_neighborhood']['closer_or_target_like_error_pixels'] for r in rr)
            error_sum=sum((r['target']['absolute_error_m']['mean'] or 0)*r['target']['absolute_error_m']['count'] for r in rr)
            native={}
            for part in ('all','corridor'):
                total=sum(r['native_returned_contributors']['summary'][part]['inside_rgb'] for r in rr)
                agree=sum(r['native_returned_contributors']['summary'][part]['diagnostic_012m_agreement'] for r in rr)
                native[part]=dict(in_rgb=total,agreement=agree,fraction=agree/total if total else None)
            groups[family]=dict(frames=len(rr),target_columns=cols,recovered_columns=good,column_coverage=good/cols if cols else None,
                target_pixels=px,comparable_pixels=comparable,prediction_coverage=comparable/px if px else None,
                target_mae_m=error_sum/comparable if comparable else None,agreeing_pixels=true,pixel_coverage=true/px if px else None,
                spurious_ring_pixels=false,local_precision=true/(true+false) if true+false else None,
                referenced_ring_pixels=ring,ring_closer_errors=over,ring_closer_rate=over/ring if ring else None,native=native)
        true=sum(g['agreeing_pixels'] for g in groups.values());false=sum(g['spurious_ring_pixels'] for g in groups.values())
        valid=sum(a['arms'][arm]['fit']['valid'] for a in audits)
        hold=[a['arms'][arm]['withheld_zone_check']['median_abs_range_residual_m'] for a in audits
            if a['arms'][arm]['withheld_zone_check']['median_abs_range_residual_m'] is not None]
        result[arm]=dict(families=groups,calibrated_frames=valid,unknown_frames=len(audits)-valid,
            local_precision=true/(true+false) if true+false else None,
            withheld_comparable_frames=len(hold),withheld_median_of_frame_medians_m=float(np.median(hold)) if hold else None)
    a=result['median'];b=result['regional_modes']
    checks=dict(each_family_column_recovery_at_least_half=all(g['column_coverage'] is not None and g['column_coverage']>=.5 for g in b['families'].values()),
        rod_boundary_gain_at_least_5pp=all(b['families'][f]['column_coverage']>=a['families'][f]['column_coverage']+.05 for f in ('near_rod_farwall','shallow_boundary_stress')),
        body_head_noninferior=all(b['families'][f]['column_coverage']>=a['families'][f]['column_coverage'] for f in ('substantial_body','suspended_head')),
        local_precision_noninferior=b['local_precision'] is not None and a['local_precision'] is not None and b['local_precision']>=a['local_precision'],
        no_family_ring_closer_increase=all(b['families'][f]['ring_closer_rate']<=a['families'][f]['ring_closer_rate']+1e-12 for f in b['families']))
    return dict(frames=len(cases),arms=result,gate=dict(checks=checks,passed=all(checks.values())),
        decision='RETAIN_CONSUMED_RELATIVE_CALIBRATION_COMPONENT' if all(checks.values()) else 'STOP_FIXED_RELATIVE_CALIBRATION_GEOMETRY_NOT_ADMITTED',
        dev_accessed=False,test_accessed=False,alerts_evaluated=False,fit_training=False,
        scope='CONSUMED_TRAIN_DENSE_GEOMETRY_POINT_ESTIMATES_NOT_ALERTS_OR_CERTIFIED_SURFACES')


def run(out):
    start=time.perf_counter();out=out.resolve();cap=SOURCE.resolve();upstream=(OLD/'upstream').resolve();checkpoint=OLD/'depthor-zju-small.pt'
    assert out.is_relative_to(WORK.resolve()) and not out.exists() and torch.cuda.is_available()
    spec=read(cap/'spec.json');receipt=read(cap/'receipt.json')
    assert receipt['status']=='PASS' and sha(cap/'spec.json')==receipt['spec_sha256']
    ids={f['id'] for f in spec['frames'] if f['split']=='train'};assert len(ids)==192
    rows=public_observations(selected_jsonl(cap/'raw.jsonl',ids));assert len(rows)==192
    expected_checkpoint=[h for p,h in read(OLD/'train-v2/freeze.json')['inputs'].items() if Path(p).name==checkpoint.name]
    assert len(expected_checkpoint)==1 and sha(checkpoint)==expected_checkpoint[0]
    files=[cap/n for n in ('spec.json','receipt.json','raw.jsonl','evaluator.jsonl')]+[checkpoint,OLD/'train-v2/freeze.json']
    for name in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/name)==receipt['hashes'][name]
    for row in rows:
        path=(cap/row['rgb_path']).resolve();assert path.is_relative_to(cap) and sha(path)==receipt['hashes'][row['rgb_path']]
        files.append(path)
    inputs={str(p):sha(p) for p in files};sources=local_dependencies(__file__)
    sources.update(local_dependencies(CODE/'test_mz157_relative_calibration.py'))
    sources[str(CODE/'MZ157_PROTOCOL_20260916.md')]=sha(CODE/'MZ157_PROTOCOL_20260916.md')
    for path in (upstream/'src').rglob('*.py'):sources[str(path)]=sha(path)
    out.mkdir(parents=True);(out/'predictions').mkdir()
    for filename in sources:
        path=Path(filename)
        relative=Path('upstream')/path.relative_to(upstream) if path.is_relative_to(upstream) else path.relative_to(ROOT.resolve())
        dest=out/'source-snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    write(out/'freeze.json',dict(method=METHOD,sources=sources,inputs=inputs,ids=[r['id'] for r in rows],
        authority='CONSUMED_TRAIN192_ONLY',device=torch.cuda.get_device_name(0),torch=torch.__version__,
        inference_backend='CUDA_FP32',calibration_backend='CPU_SCIPY_TASK_NOT_GPU_SUITABLE',compute_cap_s=1200,
        weights_sha256=sha(checkpoint),no_native_decode_before_prediction=True))
    model=None;audits=[];manifest={};gpu_seconds=0.;calibration_seconds=0.;torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    try:
        model,info=load_prior(upstream,checkpoint);write(out/'model-binding.json',info)
        for index,row in enumerate(rows):
            image=cv2.imread(str(cap/row['rgb_path']));assert image.shape==(360,640,3)
            torch.cuda.synchronize();tick=time.perf_counter();relative,network=relative_prediction(model,image)
            torch.cuda.synchronize();gpu_seconds+=time.perf_counter()-tick
            tick=time.perf_counter();predictions={};detail={}
            for arm in ARMS:predictions[arm],detail[arm]=calibrate(row,relative,arm)
            calibration_seconds+=time.perf_counter()-tick
            dest=out/'predictions'/(row['id']+'.npz');np.savez_compressed(dest,relative=relative,**predictions)
            manifest[str(dest.relative_to(out))]=sha(dest)
            audits.append(dict(id=row['id'],network=network,arms=detail))
            assert time.perf_counter()-start<1200,'Declared compute cap reached'
            if (index+1)%24==0:print(json.dumps(dict(stage='public_prediction',frames=index+1,seconds=time.perf_counter()-start)),flush=True)
        write(out/'calibration-audit.json',audits)
        assert all(sha(p)==h for p,h in (sources|inputs).items())
        write(out/'prediction-seal.json',dict(freeze_sha256=sha(out/'freeze.json'),model_binding_sha256=sha(out/'model-binding.json'),
            files=manifest,audit_sha256=sha(out/'calibration-audit.json'),gpu_seconds=gpu_seconds,calibration_seconds=calibration_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),authority='SEALED_BEFORE_THIS_NATIVE_PARSE'))
        print(json.dumps(dict(stage='SEALED',frames=len(rows),gpu_seconds=gpu_seconds,calibration_seconds=calibration_seconds)),flush=True)
        write(out/'evaluation-start.json',dict(prediction_seal_sha256=sha(out/'prediction-seal.json')))
        es=selected_jsonl(cap/'evaluator.jsonl',ids);assert [e['id'] for e in es]==[r['id'] for r in rows]
        cases=[];episode=None;yaw=0.
        for i,(row,e) in enumerate(zip(rows,es)):
            if row['episode_id']!=episode:yaw=0.
            episode=row['episode_id']
            if row['imu_valid']:yaw+=row['delta_yaw']
            with np.load(out/'predictions'/(row['id']+'.npz')) as saved:
                cases.append(dict(id=row['id'],family=e['family'],arms={arm:evaluate_frame(row,e,saved[arm],dict(integrated_yaw_deg=yaw)) for arm in ARMS}))
            assert time.perf_counter()-start<1200,'Declared compute cap reached'
            if (i+1)%48==0:print(json.dumps(dict(stage='native_evaluation',frames=i+1)),flush=True)
        write(out/'geometry-cases.json',cases);summary=summarize(cases,audits);write(out/'summary.json',summary)
        assert all(sha(p)==h for p,h in (sources|inputs).items())
        write(out/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,
            summary_sha256=sha(out/'summary.json'),cases_sha256=sha(out/'geometry-cases.json'),
            prediction_seal_sha256=sha(out/'prediction-seal.json'),sources_inputs_unchanged=True,
            resources='Process-local CUDA model released in finally; no remote/persistent job'))
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),predictions_sealed=(out/'prediction-seal.json').exists()))
        raise
    finally:
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'run-v1')
    run(parser.parse_args().output)
