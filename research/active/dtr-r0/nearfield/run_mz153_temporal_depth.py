"""Frozen TRAIN geometry check; seal predictions before loading native labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from mz136_incumbent import public_observations
from mz153_temporal_depth import METHOD, make_input, full_rgb_depth, prefix_indices
from run_mz139_surface_fit import selected_jsonl, local_dependencies
from run_mz107_four_sensor import ROOT, sha, write

CODE=ROOT/'research/active/dtr-r0/nearfield'
WORK=ROOT/'artifacts.local/work/mz153-temporal-depth-20260916'
CAP=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'
INC=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/incumbent/fresh-v1'
ARMS=('temporal','repeated_current')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reduce_cases(cases):
    result={}
    for arm in ARMS:
        families={}
        for family in sorted({c['family'] for c in cases}):
            group=[c['arms'][arm] for c in cases if c['family']==family]
            col=sum(g['target']['transverse_columns']['visible'] for g in group)
            recovered=sum(g['target']['transverse_columns']['with_at_least_half_pixels_agreeing'] for g in group)
            true=sum(g['target']['range_agreeing_pixels'] for g in group)
            false=sum(g['silhouette_neighborhood']['target_depth_band_spurious_pixels'] for g in group)
            target=sum(g['target']['visible_pixels'] for g in group)
            visible=sum(g['common_fov']['visible_target_pixels'] for g in group)
            contributors=[g['native_returned_contributors']['summary']['corridor'] for g in group]
            families[family]=dict(frames=len(group),target_columns=col,recovered_columns=recovered,
                column_coverage=recovered/col if col else None, target_pixels=target,
                agreeing_pixels=true,pixel_coverage=true/target if target else None,
                common_fov_target_pixels=visible,common_fov_fraction=visible/target if target else None,
                conditional_common_fov_pixel_coverage=true/visible if visible else None,
                spurious_ring_pixels=false,local_precision=true/(true+false) if true+false else None,
                corridor_returned_samples=sum(g['total'] for g in contributors),
                corridor_samples_in_rgb=sum(g['inside_rgb'] for g in contributors),
                corridor_samples_with_dense_prediction=sum(g['finite_prediction'] for g in contributors),
                corridor_samples_agreeing_012m=sum(g['diagnostic_012m_agreement'] for g in contributors))
        true=sum(g['agreeing_pixels'] for g in families.values())
        false=sum(g['spurious_ring_pixels'] for g in families.values())
        result[arm]=dict(families=families,agreeing_pixels=true,spurious_ring_pixels=false,
                        local_precision=true/(true+false) if true+false else None)
    return result


def summarize(cases):
    arms=reduce_cases(cases)
    temporal,repeated=(arms[a] for a in ARMS)
    changes={f:g['column_coverage']-repeated['families'][f]['column_coverage']
             for f,g in temporal['families'].items()}
    difficult=[f for f in changes if 'rod' in f.lower() or 'boundary' in f.lower()]
    assert len(difficult)==2
    gate=dict(each_family_half_column_coverage_at_least_50pct=all(
        g['column_coverage'] is not None and g['column_coverage']>=.5 for g in temporal['families'].values()),
        difficult_families_improve_5pp=all(changes[f]>=.05 for f in difficult),
        local_precision_not_worse=temporal['local_precision'] is not None and
            repeated['local_precision'] is not None and temporal['local_precision']>=repeated['local_precision'])
    gate['passed']=all(gate.values())
    return dict(frames=len(cases),split='ORIGINAL_TRAIN192_CONSUMED_DEVELOPMENT',arms=arms,
        column_coverage_changes=changes,gate=gate,
        history_strata={label:reduce_cases([c for c in cases if c['genuine_history']==flag])
                        for label,flag in [('first_frames',False),('has_past_frames',True)]},
        decision='TEMPORAL_GEOMETRY_COMPONENT_ADMITTED_ALERTS_NOT_TESTED' if gate['passed'] else
                 'STOP_FIXED_PRETRAINED_TEMPORAL_TRANSFER_KEEP_MZ129',
        non_train_raw_or_evaluator_decoded=False,non_train_model_scored=False,
        old_baseline_cache_scope='ALL288_PARSED_AFTER_SEAL_ONLY_TRAIN192_USED',
        training=False,alerts_changed=False)


def run(out):
    from mz153_dvsr_runtime import load_model,infer_latest
    out=out.resolve();assert not out.exists() and out.is_relative_to(WORK.resolve())
    assert torch.cuda.is_available(),'CUDA required for this experiment'
    spec,receipt=read(CAP/'spec.json'),read(CAP/'receipt.json')
    ids={f['id'] for f in spec['frames'] if f['split']=='train'}
    assert len(ids)==192
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',ids))
    assert len(rows)==192 and set(Counter(r['episode_id'] for r in rows).values())=={6}
    assert sha(CAP/'spec.json')==receipt['spec_sha256']
    for n in ('raw.jsonl','evaluator.jsonl'):
        assert sha(CAP/n)==receipt['hashes'][n]
    paths=[CAP/n for n in ('spec.json','receipt.json','raw.jsonl','evaluator.jsonl')]
    rgb,depth,audits=[],[],[]
    for row in rows:
        path=CAP/row['rgb_path'];assert path.resolve().is_relative_to(CAP.resolve())
        assert sha(path)==receipt['hashes'][row['rgb_path']]
        a,b,c=make_input(row,cv2.imread(str(path)))
        rgb.append(a);depth.append(b);audits.append(c);paths.append(path)
    rgb,depth=np.stack(rgb),np.stack(depth)
    # All runtime files and upstream source are immutable evidence for this run.
    runtime=WORK/'runtime'
    paths.extend(p for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend([INC/'nominal/predictions.json',INC/'prediction-seal.json',INC/'completion.json'])
    sources=local_dependencies(__file__)
    for name in ('mz153_dvsr_runtime.py','evaluate_mz140_depthor.py','test_mz153_temporal_depth.py'):
        p=CODE/name;sources[str(p.resolve())]=sha(p)
    inputs={str(p.resolve()):sha(p) for p in paths}
    out.mkdir(parents=True)
    for src in sources:
        p=Path(src);dest=out/'source-snapshot'/p.relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    shutil.copyfile(CODE/'MZ153_PROTOCOL_20260916.md',out/'protocol-before-inference.md')
    write(out/'freeze.json',dict(method=METHOD,ids=[r['id'] for r in rows],inputs=inputs,
        sources=sources,protocol_sha256=sha(out/'protocol-before-inference.md'),
        device='cuda',authority='OBSERVABLE_INPUTS_ONLY_NO_NATIVE_LABEL_DECODE'))
    np.savez_compressed(out/'observable-inputs.npz',rgb=rgb,depth=depth)
    write(out/'input-audits.json',audits)
    torch.set_num_threads(4)
    model,info=load_model(runtime,'cuda')
    outputs={};timings=[];prefixes=[]
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        for i,row in enumerate(rows):
            idx=prefix_indices(rows,i)
            assert all(rows[k]['episode_id']==row['episode_id'] and rows[k]['time_s']<=row['time_s'] for k in idx)
            prefixes.append(dict(id=row['id'],indices=idx,ids=[rows[k]['id'] for k in idx],
                                 genuine_history=len(set(idx))>1))
            pair={}
            for arm in ARMS:
                selected=idx if arm=='temporal' else [i]*len(idx)
                torch.cuda.synchronize();tick=time.perf_counter()
                value=np.asarray(infer_latest(model,rgb[selected],depth[selected]),np.float32)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                assert value.shape==(128,128) and np.isfinite(value).all()
                raw_path=out/f'{i:03d}-{arm}-raw.npy';np.save(raw_path,value,allow_pickle=False)
                original,shared=full_rgb_depth(row,value)
                p=out/f'{i:03d}-{arm}.npy';np.save(p,original,allow_pickle=False)
                for path in (p,raw_path):outputs[str(path)]=sha(path)
                pair[arm]=value
                timings.append(dict(id=row['id'],arm=arm,seconds=elapsed,
                    peak_allocated_bytes=model.last_inference_runtime['peak_allocated_bytes'],
                    raw_min=float(value.min()),raw_max=float(value.max()),
                    positive_pixels=int((value>0).sum()),shared_rgb_pixels=int(shared.sum())))
            if len(set(idx))==1:
                assert np.allclose(pair['temporal'],pair['repeated_current'],atol=1e-6,rtol=0)
            print(json.dumps(dict(completed=i+1,total=len(rows),id=row['id'],
                history=len(set(idx))-1,temporal_difference_m=float(np.abs(pair['temporal']-pair['repeated_current']).mean()*4))),flush=True)
            if time.perf_counter()-started>1200:raise TimeoutError('Frozen compute cap exceeded')
        # Repeat a real full prefix after all other requests; no state leakage.
        chosen=5;idx=prefix_indices(rows,chosen)
        replay=np.asarray(infer_latest(model,rgb[idx],depth[idx]),np.float32)
        before=np.load(out/f'{chosen:03d}-temporal-raw.npy')
        error=float(np.abs(replay-before).max())
        assert error<=1e-6, f'Stateful runtime: {error}'
        write(out/'runtime.json',dict(model=info,device=torch.cuda.get_device_name(),torch=torch.__version__,
            peak_allocated_bytes=max(t['peak_allocated_bytes'] for t in timings),prefix_replay_index=chosen,
            prefix_replay_max_abs_error=error,inference_seconds=time.perf_counter()-started))
        write(out/'prefixes.json',prefixes);write(out/'timings.json',timings)
        write(out/'prediction-seal.json',dict(outputs=outputs,freeze_sha256=sha(out/'freeze.json'),
            runtime_sha256=sha(out/'runtime.json'),prefixes_sha256=sha(out/'prefixes.json'),
            input_audits_sha256=sha(out/'input-audits.json'),input_arrays_sha256=sha(out/'observable-inputs.npz'),
            authority='BOTH_ARMS_SEALED_BEFORE_NATIVE_TRAIN_LABEL_PARSE'))
        from evaluate_mz140_depthor import evaluate_frame,rasterize_native_bounds
        old=read(INC/'nominal/predictions.json');seal=read(INC/'prediction-seal.json')
        assert read(INC/'completion.json')['status']=='PASS'
        assert sha(INC/'nominal/predictions.json')==seal['predictions_sha256']['nominal']
        cache={p['id']:c for p,c in zip(old['predictions'],old['corrected'])}
        es=selected_jsonl(CAP/'evaluator.jsonl',ids)
        cases=[]
        for i,(row,e) in enumerate(zip(rows,es)):
            assert row['id']==e['id']
            reference=rasterize_native_bounds(row,e,cache[row['id']])
            _,shared=full_rgb_depth(row,np.ones((128,128),np.float32))
            arms={}
            for arm in ARMS:
                value=np.load(out/f'{i:03d}-{arm}.npy')
                r=evaluate_frame(row,e,value,cache[row['id']])
                r['common_fov']=dict(rgb_pixels=int(shared.sum()),
                    visible_target_pixels=int((shared&reference['target_visible']).sum()),
                    unknown_prediction_pixels=int((shared&~np.isfinite(value)).sum()))
                arms[arm]=r
            cases.append(dict(id=row['id'],family=e['family'],genuine_history=prefixes[i]['genuine_history'],arms=arms))
        write(out/'geometry-cases.json',cases)
        summary=summarize(cases);write(out/'summary.json',summary)
        assert inputs=={p:sha(Path(p)) for p in inputs}
        assert sources=={p:sha(Path(p)) for p in sources}
        write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
            geometry_cases_sha256=sha(out/'geometry-cases.json'),prediction_seal_sha256=sha(out/'prediction-seal.json'),
            freeze_sha256=sha(out/'freeze.json'),inputs_sources_unchanged=True,
            total_seconds=time.perf_counter()-started,resources='Process-local GPU model released on exit'))
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),
            elapsed_seconds=time.perf_counter()-started,predictions_sealed=(out/'prediction-seal.json').exists()))
        raise
    finally:
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
