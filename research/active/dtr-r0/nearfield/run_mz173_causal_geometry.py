"""Frozen public-only causal geometry; native evaluation follows prediction seal."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import zipfile

import cv2
import numpy as np
import torch

from mz136_incumbent import public_observations
from mz173_da3 import load_model, predict, prefix_indices, MODEL_SHA256, SOURCE_REVISION, MODEL_REVISION
from mz173_scale import calibrate
from run_mz139_surface_fit import selected_jsonl, local_dependencies
from run_mz161_dense_task import ROOT, CODE, CAP, read, write, sha
from run_mz164_metric_prior import alert

WORK = ROOT/'artifacts.local/work/mz173-causal-geometry-20260916'
ASSETS = ROOT/'artifacts.local/work/geometry-foundation-screen-20260916'
OLD = ROOT/'artifacts.local/work/mz164-camera-metric-prior-20260916/run-v1'
ARMS = ('repeated_current', 'temporal')
LIMIT = 1800.


def summarize(rows, es, spec, cases, predictions, baseline):
    from run_mz164_metric_prior import geometry_summary
    from evaluate_mz136_corridor_pair import score, pair_metrics
    from run_mz143_corridor_evidence import native_account
    from run_mz107_four_sensor import truth
    gt = np.array([truth(e) for e in es], bool)
    flags = dict(baseline=baseline)
    for arm in ARMS:
        flags[arm] = np.array([p['arms'][arm]['candidate'] for p in predictions], bool)
    episodes = {}
    for i,row in enumerate(rows): episodes.setdefault(row['episode_id'], []).append(i)
    pairs = []
    for pair in spec['pairs']:
        a,b = pair['episodes']
        if a in episodes and b in episodes:
            pairs += [dict(a=i,b=j) for i,j in zip(episodes[a],episodes[b])]
    reports = {}
    for arm,ff in flags.items():
        r = score(rows,es,gt,ff,list(range(192)),baseline)
        m = r['metrics']
        r.update(precision=m['TP']/(m['TP']+m['FP']) if m['TP']+m['FP'] else None,
                 recall=m['TP']/(m['TP']+m['FN']),
                 pairs=pair_metrics(gt,ff.astype(float),ff,pairs),
                 native=native_account(rows,es,ff,baseline), strata={})
        for label,indices in dict(ordinary=[i for i,e in enumerate(es) if e['family']!='shallow_boundary_stress'],
                                  pressure=[i for i,e in enumerate(es) if e['family']=='shallow_boundary_stress']).items():
            r['strata'][label] = score(rows,es,gt,ff,indices,baseline)
        reports[arm] = r
    geom = {arm:geometry_summary(cases,arm) for arm in ARMS}
    families = sorted({c['family'] for c in cases})
    current, temporal = geom['repeated_current'], geom['temporal']
    geometry_checks = dict(
        each_family_half_column_recovery_at_least_half=all(temporal[f]['all']['column_coverage'] >= .5 for f in families),
        rod_boundary_gain_at_least_5pp=all(temporal[f]['all']['column_coverage'] >= current[f]['all']['column_coverage']+.05
            for f in ('near_rod_farwall','shallow_boundary_stress')),
        body_head_noninferior=all(temporal[f]['all']['column_coverage'] >= current[f]['all']['column_coverage']
            for f in ('substantial_body','suspended_head')),
        aggregate_local_precision_noninferior=temporal['ALL']['all']['local_precision'] is not None and
            current['ALL']['all']['local_precision'] is not None and
            temporal['ALL']['all']['local_precision'] >= current['ALL']['all']['local_precision'],
        no_family_closer_ring_increase=all(temporal[f]['all']['too_near_ring_rate'] <= current[f]['all']['too_near_ring_rate']+1e-12 for f in families))
    base, candidate = reports['baseline'], reports['temporal']
    event_changes = []
    for a,b in zip(candidate['event_details'],base['event_details']):
        assert (a['episode'],a['onset_s']) == (b['episode'],b['onset_s'])
        event_changes.append(dict(episode=a['episode'],onset_s=a['onset_s'],baseline=b['first_alert_s'],
            candidate=a['first_alert_s'],delta_s=None if a['first_alert_s'] is None or b['first_alert_s'] is None
            else a['first_alert_s']-b['first_alert_s']))
    before = {x['id'] for x in base['native']['nonalert_with_native_corridor_contributors']}
    after = {x['id'] for x in candidate['native']['nonalert_with_native_corridor_contributors']}
    alert_checks = dict(
        fewer_fp_than_control_and_baseline=candidate['metrics']['FP'] < min(base['metrics']['FP'],reports['repeated_current']['metrics']['FP']),
        old_true_frames_retained=not candidate['lost_baseline_tp'],
        old_event_onsets_retained=all(e['candidate'] is not None and e['candidate'] <= e['baseline']
            for e in event_changes if e['baseline'] is not None),
        no_new_native_supported_nonalert=not(after-before),
        no_family_fp_increase=all(candidate['families'][f]['FP'] <= base['families'][f]['FP'] for f in families))
    gp, ap = all(geometry_checks.values()), all(alert_checks.values())
    return dict(authority='CONSUMED_ORIGINAL_TRAIN192_CAUSAL_GEOMETRY',
        arms=reports,geometry=geom,geometry_checks=geometry_checks,alert_checks=alert_checks,
        geometry_passed=gp,alert_passed=ap,joint_passed=gp and ap,event_changes=event_changes,
        calibration={arm:dict(valid_frames=sum(p['arms'][arm]['finite_positive_pixels']>0 for p in predictions),
            unknown_frames=sum(p['arms'][arm]['finite_positive_pixels']==0 for p in predictions)) for arm in ARMS},
        decision='MZ173_RETAIN_CONSUMED_CAUSAL_GEOMETRY_AND_ALERT_COMPONENT' if gp and ap else
                 'MZ173_CAUSAL_GEOMETRY_JOINT_GAIN_NOT_MET',
        original_dev_test_access=False,training=False,default_changed=False,
        raw_sensor_inputs_unchanged=True,radar_native_lineage='NOT_EVALUABLE',
        candidate_sensor_roles=dict(rgb='JOINT_RELATIVE_GEOMETRY',tof='ELIGIBLE_REGIONAL_SCALE_ONLY',
            imu='PUBLIC_CORRIDOR_PROJECTION',radar='RETAINED_IN_BASELINE_NOT_CANDIDATE_DEPTH'))


def run(output):
    start = time.perf_counter()
    output = output.resolve()
    assert not output.exists() and output.is_relative_to(WORK.resolve()) and torch.cuda.is_available()
    output.mkdir(parents=True); (output/'predictions').mkdir()
    torch.set_num_threads(4); cv2.setNumThreads(4)
    torch.manual_seed(173016); torch.cuda.manual_seed_all(173016)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    def within():
        if time.perf_counter()-start > LIMIT: raise TimeoutError('Frozen1800s computation cap reached')
    model = None
    try:
        spec,receipt = read(CAP/'spec.json'),read(CAP/'receipt.json')
        assert receipt['status']=='PASS' and sha(CAP/'spec.json')==receipt['spec_sha256']
        ids = {f['id'] for f in spec['frames'] if f['split']=='train'}
        rows = public_observations(selected_jsonl(CAP/'raw.jsonl',ids))
        assert len(ids)==len(rows)==192 and set(Counter(r['episode_id'] for r in rows).values())=={6}
        for n in ('raw.jsonl','evaluator.jsonl'): assert sha(CAP/n)==receipt['hashes'][n]
        old_done,old_seal=read(OLD/'completion.json'),read(OLD/'prediction-seal.json')
        assert old_done['status']=='PASS'
        assert sha(OLD/'prediction-seal.json')==old_done['prediction_seal_sha256']
        assert sha(OLD/'freeze.json')==old_seal['freeze_sha256']
        assert sha(OLD/'predictions.json')==old_seal['predictions_sha256']
        old_inputs=read(OLD/'freeze.json')['inputs']
        for n in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json'):
            same=[h for p,h in old_inputs.items() if Path(p).resolve()==(CAP/n).resolve()]
            assert same==[sha(CAP/n)]
        prior = read(OLD/'predictions.json')
        assert [p['id'] for p in prior]==[r['id'] for r in rows]
        baseline = np.array([p['baseline'] for p in prior],bool)
        assert sha(ASSETS/'da3-small/model.safetensors')==MODEL_SHA256
        preparation=read(ASSETS/'da3-preparation.json')
        assert preparation['source_revision']==SOURCE_REVISION and preparation['model_revision']==MODEL_REVISION
        for download in preparation['downloads']: assert sha(download['path'])==download['sha256']
        source_bindings=0
        with zipfile.ZipFile(ASSETS/'da3-source.zip') as archive:
            prefix=archive.namelist()[0].split('/')[0]+'/'
            for p in (ASSETS/'da3-source/src').rglob('*.py'):
                assert p.read_bytes()==archive.read(prefix+p.relative_to(ASSETS/'da3-source').as_posix())
                source_bindings+=1
        assert read(WORK/'adapter-preflight.json')['status']=='PASS'
        write(output/'source-admission.json',dict(unchanged_official_source_files=source_bindings,
            archive_sha256=sha(ASSETS/'da3-source.zip'),complete_baseline_seal_chain=True))
        paths = [CAP/n for n in ('spec.json','receipt.json','raw.jsonl','evaluator.jsonl')]
        paths += [OLD/n for n in ('predictions.json','prediction-seal.json','completion.json','freeze.json')]
        paths += [ASSETS/n for n in ('da3-small/model.safetensors','da3-small/config.json','da3-preparation.json','da3-smoke.json')]
        paths += [WORK/'adapter-preflight.json',output/'source-admission.json']
        for row in rows:
            path = (CAP/row['rgb_path']).resolve()
            assert path.is_relative_to(CAP.resolve()) and sha(path)==receipt['hashes'][row['rgb_path']]
            paths.append(path)
        sources = local_dependencies(__file__)
        for name in ('MZ173_PROTOCOL_20260916.md','test_mz173_scale.py'):
            p=CODE/name; sources[str(p)]=sha(p)
        for p in (ASSETS/'da3-source/src').rglob('*.py'): sources[str(p)]=sha(p)
        inputs = {str(p):sha(p) for p in paths}
        for source in sources:
            p=Path(source)
            rel = Path('upstream')/p.relative_to(ASSETS/'da3-source') if p.is_relative_to(ASSETS/'da3-source') else p.relative_to(ROOT)
            dest=output/'source-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        write(output/'freeze.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),sources=sources,inputs=inputs,
            ids=[r['id'] for r in rows],arms=ARMS,compute_cap_s=LIMIT,model_frozen=True,
            authority='CONSUMED_TRAIN192_ONLY_NATIVE_DECODE_AFTER_PREDICTIONS',
            original_dev_test_access=False,baseline_cache_scope='AUTHENTICATED_MZ164_TRAIN192_ONLY'))
        model,processor,binding = load_model(ASSETS)
        write(output/'model-binding.json',binding)
        images = [cv2.cvtColor(cv2.imread(str(CAP/r['rgb_path'])),cv2.COLOR_BGR2RGB) for r in rows]
        assert all(im.shape==(360,640,3) for im in images)
        tensors,_,_ = processor(images,process_res=504,process_res_method='upper_bound_resize',sequential=True)
        assert tuple(tensors.shape)==(192,3,280,504)
        del images
        torch.cuda.reset_peak_memory_stats()
        predictions=[]; prefixes=[]; audits=[]; manifest={}; gpu_s={arm:0. for arm in ARMS}
        yaw=0.;previous=None; replay_reference=None
        for i,row in enumerate(rows):
            within(); idx=prefix_indices(rows,i)
            if row['episode_id']!=previous: yaw=0.
            if row['imu_valid']: yaw+=row['delta_yaw']
            previous=row['episode_id']
            prefixes.append(dict(id=row['id'],indices=idx,ids=[rows[j]['id'] for j in idx],
                times_s=[rows[j]['time_s'] for j in idx]))
            payload={}; details=dict(id=row['id'],arms={}); report=dict(id=row['id'],yaw_deg=yaw,arms={})
            for arm in ARMS:
                selected=idx if arm=='temporal' else [i]*len(idx)
                if arm=='temporal' and len(idx)==1:
                    rel,raw=previous_relative,previous_raw
                else:
                    torch.cuda.synchronize(); tick=time.perf_counter()
                    rel,raw=predict(model,tensors[selected])
                    torch.cuda.synchronize(); gpu_s[arm]+=time.perf_counter()-tick
                if arm=='repeated_current': previous_relative,previous_raw=rel,raw
                scale_rows=[rows[j] for j in idx] if arm=='temporal' else [row]
                scale_maps=rel if arm=='temporal' else rel[-1:]
                depth,audit=calibrate(scale_rows,scale_maps)
                assert depth.shape==(360,640) and depth.dtype==np.float32
                payload[arm+'_relative_z']=scale_maps
                payload[arm+'_depth']=depth
                for key,value in raw.items(): payload[arm+'_'+key+'_raw']=value
                details['arms'][arm]=audit
                report['arms'][arm]=dict(**alert(row,depth,yaw),finite_positive_pixels=int((np.isfinite(depth)&(depth>0)).sum()))
                if i==4 and arm=='temporal': replay_reference=rel.copy()
            dest=output/'predictions'/(row['id']+'.npz');np.savez_compressed(dest,**payload)
            manifest[str(dest.relative_to(output))]=sha(dest)
            audits.append(details);predictions.append(report)
            if (i+1)%24==0: print(json.dumps(dict(stage='public_prediction',frames=i+1,seconds=time.perf_counter()-start)),flush=True)
        replay,_=predict(model,tensors[prefix_indices(rows,4)])
        replay_error=float(np.max(np.abs(replay-replay_reference)))
        assert replay_error<=1e-6,('Cross-request replay mismatch',replay_error)
        write(output/'predictions.json',predictions);write(output/'prefixes.json',prefixes);write(output/'calibration-audit.json',audits)
        assert all(sha(p)==h for p,h in (sources|inputs).items())
        write(output/'prediction-seal.json',dict(freeze_sha256=sha(output/'freeze.json'),maps=manifest,
            predictions_sha256=sha(output/'predictions.json'),prefixes_sha256=sha(output/'prefixes.json'),
            calibration_sha256=sha(output/'calibration-audit.json'),model_binding_sha256=sha(output/'model-binding.json'),
            gpu_seconds=gpu_s,peak_allocated_bytes=torch.cuda.max_memory_allocated(),replay_max_abs_error=replay_error,
            device=torch.cuda.get_device_name(),torch=torch.__version__,authority='SEALED_BEFORE_NATIVE_PARSE'))
        del model;model=None;torch.cuda.empty_cache()
        print(json.dumps(dict(stage='SEALED',seconds=time.perf_counter()-start,gpu_seconds=gpu_s)),flush=True)
        from evaluate_mz140_depthor import evaluate_frame
        from run_mz164_metric_prior import target_stratum
        within()
        write(output/'evaluation-start.json',dict(prediction_seal_sha256=sha(output/'prediction-seal.json')))
        es=selected_jsonl(CAP/'evaluator.jsonl',ids)
        assert [e['id'] for e in es]==[r['id'] for r in rows]
        cases=[]
        for row,e,p in zip(rows,es,predictions):
            with np.load(output/'predictions'/(row['id']+'.npz'),allow_pickle=False) as maps:
                cases.append(dict(id=row['id'],family=e['family'],tof_stratum=target_stratum(row,e),arms={
                    arm:dict(geometry=evaluate_frame(row,e,maps[arm+'_depth'],dict(integrated_yaw_deg=p['yaw_deg']))) for arm in ARMS}))
            within()
        write(output/'geometry-cases.json',cases)
        summary=summarize(rows,es,spec,cases,predictions,baseline)
        summary['seconds']=time.perf_counter()-start;summary['gpu_seconds']=gpu_s
        write(output/'summary.json',summary)
        assert all(sha(p)==h for p,h in (sources|inputs).items());within()
        write(output/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,decision=summary['decision'],
            summary_sha256=sha(output/'summary.json'),cases_sha256=sha(output/'geometry-cases.json'),
            prediction_seal_sha256=sha(output/'prediction-seal.json'),sources_inputs_unchanged=True,
            resources='Process-local CUDA model released; no worker or persistent allocation'))
        print(json.dumps(dict(decision=summary['decision'],geometry_checks=summary['geometry_checks'],
            alert_checks=summary['alert_checks'],alerts={a:r['metrics'] for a,r in summary['arms'].items()},seconds=summary['seconds'])),flush=True)
    except Exception as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.perf_counter()-start,
            predictions_sealed=(output/'prediction-seal.json').exists()))
        raise
    finally:
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'run-v1')
    run(parser.parse_args().output)
