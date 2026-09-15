"""One frozen direct-RGB detail contrast at256, with unchanged8x8 ToF."""
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
from mz153_temporal_depth import prefix_indices
from run_mz153_temporal_depth import reduce_cases
from run_mz139_surface_fit import selected_jsonl,local_dependencies
from run_mz107_four_sensor import ROOT,sha,write

CODE=ROOT/'research/active/dtr-r0/nearfield'
WORK=ROOT/'artifacts.local/work/mz154-rgb-resolution-20260916'
PRIOR=ROOT/'artifacts.local/work/mz153-temporal-depth-20260916'
CAP=ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'
ARMS=('direct256','upsampled128')
ALL_ARMS=(*ARMS,'prior128')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def aggregate(cases,arm):
    # Reuse the existing pure reduction without modifying its module constants.
    mapped=[dict(family=c['family'],arms={a:c['arms'][arm] for a in ('temporal','repeated_current')}) for c in cases]
    return reduce_cases(mapped)['temporal']


def summarize(cases,absent_ids):
    groups={a:aggregate(cases,a) for a in ALL_ARMS}
    candidate=groups['direct256'];family='near_rod_farwall'
    absent=[c for c in cases if c['id'] in absent_ids]
    assert len(absent)==24
    col=sum(c['arms']['direct256']['target']['transverse_columns']['visible'] for c in absent)
    assert col==175
    recovered={a:sum(c['arms'][a]['target']['transverse_columns']['with_at_least_half_pixels_agreeing'] for c in absent) for a in ALL_ARMS}
    changes={a:{f:candidate['families'][f]['column_coverage']-g['column_coverage']
                for f,g in groups[a]['families'].items()} for a in ('upsampled128','prior128')}
    gate=dict(rod_coverage_at_least_half=candidate['families'][family]['column_coverage']>=.5,
        rod_gain_5pp_over_both=all(changes[a][family]>=.05 for a in changes),
        missing_current_return_quarter_columns_and_beats_control=(recovered['direct256']/col>=.25 and
                                                               recovered['direct256']>recovered['upsampled128']),
        other_families_noninferior=all(changes[a][f]>=0 for a in changes for f in changes[a] if f!=family),
        aggregate_precision_noninferior=all(candidate['local_precision'] is not None and groups[a]['local_precision'] is not None
            and candidate['local_precision']>=groups[a]['local_precision'] for a in ('upsampled128','prior128')))
    gate['passed']=all(gate.values())
    return dict(frames=len(cases),authority='CONSUMED_TRAIN192_GEOMETRY_COMPONENT_ONLY',arms=groups,
        column_coverage_changes=changes,gate=gate,
        current_rod_return_absent=dict(frames=len(absent),target_columns=col,recovered_columns=recovered,
                                      column_coverage={a:n/col for a,n in recovered.items()}),
        history_strata={label:{a:aggregate([c for c in cases if c['genuine_history']==flag],a) for a in ALL_ARMS}
                        for label,flag in [('first_frames',False),('has_past_frames',True)]},
        decision='RGB_DETAIL_GEOMETRY_COMPONENT_ADMITTED_ALERTS_NOT_TESTED' if gate['passed'] else
                 'STOP_FIXED_PRETRAINED_RGB_RESOLUTION_FOLLOWUP_KEEP_MZ129',
        non_train_raw_or_evaluator_decoded=False,non_train_model_scored=False,
        mixed_split_baseline_cache_parsed=False,training=False,alerts_changed=False)


def run(out):
    from mz154_rgb_resolution import load_model,infer_latest,make_inputs,full_rgb_depth,state_dict_digest
    out=out.resolve();assert not out.exists() and out.is_relative_to(WORK.resolve())
    assert torch.cuda.is_available()
    previous=PRIOR/'run-v1';freeze=read(previous/'freeze.json');completion=read(previous/'completion.json')
    assert completion['status']=='PASS' and sha(previous/'freeze.json')==completion['freeze_sha256']
    previous_seal=read(previous/'prediction-seal.json')
    assert sha(previous/'prediction-seal.json')==completion['prediction_seal_sha256']
    assert sha(previous/'observable-inputs.npz')==previous_seal['input_arrays_sha256']
    ids=set(freeze['ids']);assert len(ids)==192
    receipt=read(CAP/'receipt.json')
    for name in ('raw.jsonl','evaluator.jsonl'):
        assert sha(CAP/name)==receipt['hashes'][name]
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',ids))
    assert [r['id'] for r in rows]==freeze['ids']
    assert set(Counter(r['episode_id'] for r in rows).values())=={6}
    old_inputs=np.load(previous/'observable-inputs.npz',allow_pickle=False)
    old_audits=read(previous/'input-audits.json')
    data={a:[] for a in ARMS};depth=[];audits=[]
    paths=[CAP/n for n in ('receipt.json','raw.jsonl','evaluator.jsonl')]
    paths += [previous/n for n in ('freeze.json','prediction-seal.json','completion.json','observable-inputs.npz',
                                   'input-audits.json','geometry-cases.json','summary-scope-corrected.json')]
    paths += [PRIOR/'rod-input-diagnostic.json']
    for i,row in enumerate(rows):
        p=CAP/row['rgb_path'];assert p.resolve().is_relative_to(CAP.resolve())
        assert sha(p)==receipt['hashes'][row['rgb_path']]
        direct,control,d,audit=make_inputs(row,cv2.imread(str(p)))
        assert np.array_equal(d,old_inputs['depth'][i])
        # Audit must retain actual original slot choices and missingness.
        assert audit['mz153']['used']==old_audits[i]['used'] and audit['mz153']['omitted']==old_audits[i]['omitted']
        # Compare supplied control to independently resized sealed old guidance.
        expected=cv2.resize(old_inputs['rgb'][i].transpose(1,2,0),(256,256),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
        assert np.array_equal(control,expected)
        data['direct256'].append(direct);data['upsampled128'].append(control)
        depth.append(d);audits.append(audit);paths.append(p)
    data={a:np.stack(values) for a,values in data.items()};depth=np.stack(depth)
    del old_inputs
    for directory in (PRIOR/'runtime',WORK/'engineering'):
        paths += [p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    sources=local_dependencies(__file__)
    for name in ('mz154_rgb_resolution.py','test_mz154_rgb_resolution.py','evaluate_mz140_depthor.py'):
        p=CODE/name;sources[str(p.resolve())]=sha(p)
    inputs={str(p.resolve()):sha(p) for p in paths}
    out.mkdir(parents=True)
    for src in sources:
        p=Path(src);dest=out/'source-snapshot'/p.relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    shutil.copyfile(CODE/'MZ154_PROTOCOL_20260916.md',out/'protocol-before-inference.md')
    write(out/'freeze.json',dict(ids=[r['id'] for r in rows],inputs=inputs,sources=sources,
        protocol_sha256=sha(out/'protocol-before-inference.md'),model_scale=32,rgb_size=256,tof_size=8,
        arms=list(ARMS),old_tof_and_control_arrays_bit_identical=True,device='cuda'))
    np.savez_compressed(out/'observable-inputs.npz',depth=depth,**data)
    write(out/'input-audits.json',audits)
    torch.set_num_threads(4);model,info=load_model(PRIOR/'runtime','cuda')
    started=time.perf_counter();outputs={};prefixes=[];timings=[]
    try:
        for i,row in enumerate(rows):
            idx=prefix_indices(rows,i)
            assert all(rows[k]['episode_id']==row['episode_id'] and rows[k]['time_s']<=row['time_s'] for k in idx)
            prefixes.append(dict(id=row['id'],indices=idx,ids=[rows[k]['id'] for k in idx],genuine_history=len(set(idx))>1))
            for arm in ARMS:
                torch.cuda.synchronize();tick=time.perf_counter()
                raw=np.asarray(infer_latest(model,data[arm][idx],depth[idx]),np.float32)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                assert raw.shape==(256,256) and np.isfinite(raw).all()
                full,shared=full_rgb_depth(row,raw)
                for suffix,value in (('-raw',raw),('',full)):
                    p=out/f'{i:03d}-{arm}{suffix}.npy';np.save(p,value,allow_pickle=False);outputs[str(p)]=sha(p)
                timings.append(dict(id=row['id'],arm=arm,seconds=elapsed,
                    peak_allocated_bytes=model.last_inference_runtime['peak_allocated_bytes'],
                    shared_pixels=int(shared.sum()),positive_raw_pixels=int((raw>0).sum())))
            if (i+1)%12==0:print(json.dumps(dict(completed=i+1,total=len(rows),seconds=time.perf_counter()-started)),flush=True)
            if time.perf_counter()-started>1200:raise TimeoutError('Frozen inference budget exceeded')
        idx=prefix_indices(rows,5)
        replay=np.asarray(infer_latest(model,data['direct256'][idx],depth[idx]),np.float32)
        replay_error=float(np.abs(replay-np.load(out/'005-direct256-raw.npy')).max())
        assert replay_error<=1e-6
        final_weights=state_dict_digest(model)
        assert final_weights==info['state_dict_sha256']
        write(out/'runtime.json',dict(model=info,device=torch.cuda.get_device_name(),torch=torch.__version__,
            peak_allocated_bytes=max(t['peak_allocated_bytes'] for t in timings),
            inference_seconds=time.perf_counter()-started,prefix_replay_max_abs_error=replay_error,
            state_dict_sha256_after_inference=final_weights,weights_unchanged=True))
        write(out/'prefixes.json',prefixes);write(out/'timings.json',timings)
        write(out/'prediction-seal.json',dict(outputs=outputs,freeze_sha256=sha(out/'freeze.json'),
            runtime_sha256=sha(out/'runtime.json'),prefixes_sha256=sha(out/'prefixes.json'),
            input_arrays_sha256=sha(out/'observable-inputs.npz'),input_audits_sha256=sha(out/'input-audits.json'),
            authority='BOTH_ARMS_SEALED_BEFORE_NATIVE_TRAIN_LABEL_PARSE'))
        # Evaluation-only reference and yaw. The mixed-split cache is not read.
        from evaluate_mz140_depthor import evaluate_frame,rasterize_native_bounds
        assert sha(previous/'geometry-cases.json')==completion['geometry_cases_sha256']
        prior_cases=read(previous/'geometry-cases.json')
        es=selected_jsonl(CAP/'evaluator.jsonl',ids);cases=[]
        for i,(row,e,old) in enumerate(zip(rows,es,prior_cases)):
            assert row['id']==e['id']==old['id']
            cached=dict(integrated_yaw_deg=old['arms']['temporal']['pose_diagnostic']['public_yaw_deg'])
            reference=rasterize_native_bounds(row,e,cached)
            _,shared=full_rgb_depth(row,np.ones((256,256),np.float32))
            arms={'prior128':old['arms']['temporal']}
            for arm in ARMS:
                value=np.load(out/f'{i:03d}-{arm}.npy')
                r=evaluate_frame(row,e,value,cached)
                r['common_fov']=dict(rgb_pixels=int(shared.sum()),
                    visible_target_pixels=int((shared&reference['target_visible']).sum()),
                    unknown_prediction_pixels=int((shared&~np.isfinite(value)).sum()))
                assert r['common_fov']['visible_target_pixels']==arms['prior128']['common_fov']['visible_target_pixels']
                arms[arm]=r
            cases.append(dict(id=row['id'],family=e['family'],genuine_history=prefixes[i]['genuine_history'],arms=arms))
        absent={r['id'] for r in read(PRIOR/'rod-input-diagnostic.json')['records'] if not r['selected_target_return']}
        write(out/'geometry-cases.json',cases);summary=summarize(cases,absent);write(out/'summary.json',summary)
        assert inputs=={p:sha(Path(p)) for p in inputs}
        assert sources=={p:sha(Path(p)) for p in sources}
        write(out/'completion.json',dict(status='PASS',freeze_sha256=sha(out/'freeze.json'),
            prediction_seal_sha256=sha(out/'prediction-seal.json'),geometry_cases_sha256=sha(out/'geometry-cases.json'),
            summary_sha256=sha(out/'summary.json'),inputs_sources_unchanged=True,
            total_seconds=time.perf_counter()-started,resources='Process-local model only; released on process exit'))
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
