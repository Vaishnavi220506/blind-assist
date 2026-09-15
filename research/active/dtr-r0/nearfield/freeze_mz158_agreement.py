"""Authenticate existing models and raw-feature parity before fresh capture."""
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from mz158_crossview_agreement import ROOT,STATIC,ONSET,TEMPORAL,METHOD,RawResidual,load_models,decisions
from run_mz143_corridor_evidence import CAP,CODE,read,sha,write,selected_jsonl,public_observations,local_dependencies

WORK=ROOT/'artifacts.local/work/mz158-crossview-agreement-20260916'


def run():
    out=WORK/'recipe-freeze.json';assert not out.exists();start=time.perf_counter()
    ss=read(STATIC/'model-seal.json');ts=read(TEMPORAL/'model-seal.json');onset=read(ONSET/'onset-seal.json')
    assert ss['selected']=='fused_hgb' and onset['source_arm']=='fused_hgb'
    assert sha(STATIC/'fused_hgb.pkl')==ss['models']['fused_hgb']
    assert sha(TEMPORAL/'unregistered.pkl')==ts['models']['unregistered']
    assert sha(STATIC/'model-seal.json')==onset['model_seal_sha256']
    old=read(STATIC/'freeze.json');temp=read(TEMPORAL/'freeze.json')
    for name in ('mz143_corridor_features.py','mz125_observable_correction.py','mz136_boundary_geometry.py','mz115_spatial_allocation.py'):
        path=CODE/name;assert sha(path)==old['sources'][str(path)]
    for name in ('mz148_background_residual.py','mz145_causal_confirmation.py'):
        path=CODE/name;assert sha(path)==temp['sources'][str(path)]
    receipt=read(CAP/'receipt.json');assert sha(CAP/'raw.jsonl')==receipt['hashes']['raw.jsonl']
    models,cuts=load_models();parity={};inputs={}
    for split in ('train','dev'):
        ids=old['selected_frame_ids'][split]
        rows=public_observations(selected_jsonl(CAP/'raw.jsonl',set(ids)));assert [r['id'] for r in rows]==ids
        saved=TEMPORAL/(split+'-features.npz');seal=TEMPORAL/(split+'-feature-seal.json')
        assert sha(saved)==read(seal)['feature_sha256']
        assert read(seal)['ids']==ids
        raw=RawResidual();values=[];rgb={}
        for row in rows:
            path=(CAP/row['rgb_path']).resolve();assert path.is_relative_to(CAP.resolve())
            rgb[str(path)]=sha(path);assert rgb[str(path)]==receipt['hashes'][row['rgb_path']]
            values.append(raw.update(row,cv2.imread(str(path))))
        values=np.stack(values);reference=np.load(saved)['unregistered']
        np.testing.assert_array_equal(values,reference)
        parity[split]=dict(frames=len(rows),values=int(values.size),bitwise_raw_parity=True)
        inputs.update(rgb);inputs[str(saved)]=sha(saved);inputs[str(seal)]=sha(seal)
        if split=='dev':
            fused=STATIC/'dev-features.npz';fs=STATIC/'dev-feature-seal.json'
            assert sha(fused)==read(fs)['feature_sha256']
            data=np.load(fused);x=np.c_[data['sensor'],data['geometry']]
            scores,flags=decisions(rows,x,values,models,cuts)
            pp=TEMPORAL/'dev-predictions.json';ps=TEMPORAL/'dev-prediction-seal.json'
            sp=STATIC/'dev-predictions.json';sps=STATIC/'prediction-seal.json'
            assert sha(pp)==read(ps)['predictions_sha256']
            assert sha(sp)==read(sps)['predictions_sha256']
            p=read(pp);s=read(sp);assert [v['id'] for v in p]==[v['id'] for v in s]==ids
            np.testing.assert_array_equal(scores['raw_change'],[v['scores']['unregistered'] for v in p])
            np.testing.assert_array_equal(flags['raw_change'],[v['flags']['unregistered'] for v in p])
            np.testing.assert_array_equal(scores['static'],[v['scores']['fused_hgb'] for v in s])
            write(WORK/'consumed-replay.json',[dict(id=r['id'],scores={k:float(v[i]) for k,v in scores.items()},
                flags={k:bool(v[i]) for k,v in flags.items()}) for i,r in enumerate(rows)])
            parity[split]['frozen_model_scores_exact']=True
            for path in (fused,fs,pp,ps,sp,sps):inputs[str(path)]=sha(path)
        print(json.dumps(dict(stage='recipe_parity',split=split,**parity[split])),flush=True)
    files=[STATIC/'fused_hgb.pkl',STATIC/'model-seal.json',STATIC/'freeze.json',ONSET/'onset-seal.json',
        TEMPORAL/'unregistered.pkl',TEMPORAL/'model-seal.json',TEMPORAL/'freeze.json',CAP/'raw.jsonl',CAP/'receipt.json',
        ROOT/'artifacts.local/work/mz158-decision-complementarity-20260916/review/complementarity.json']
    inputs.update({str(p):sha(p) for p in files})
    sources=local_dependencies(__file__);sources.update(local_dependencies(CODE/'test_mz158_crossview_agreement.py'))
    sources[str(CODE/'MZ158_PROTOCOL_20260916.md')]=sha(CODE/'MZ158_PROTOCOL_20260916.md')
    for path in sources:
        dest=WORK/'recipe-snapshot'/Path(path).relative_to(ROOT.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    assert all(sha(Path(p))==h for p,h in inputs.items())
    write(out,dict(method=METHOD,cutoffs=cuts,models=dict(static=ss['models']['fused_hgb'],raw_change=ts['models']['unregistered']),
        inputs=inputs,sources=sources,parity=parity,seconds=time.perf_counter()-start,
        authority='FIXED_POSTHOC_SELECTED_RECIPE_SEALED_BEFORE_FRESH_SOURCE_OR_CAPTURE',
        original_test_access=False,new_fit=False,new_threshold_selection=False,
        consumed_replay_sha256=sha(WORK/'consumed-replay.json')))
    print(json.dumps(dict(recipe_freeze=str(out),sha256=sha(out),seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
