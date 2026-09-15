"""TRAIN-only frozen pretrained transfer check before any dev alert experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import torch
from mz140_depthor import METHOD,load_model,make_input,box_only_rgb,predict
from mz136_incumbent import public_observations
from run_mz139_surface_fit import read,selected_jsonl,local_dependencies,WORK,CODE
from run_mz107_four_sensor import ROOT,sha,write


def summarize(cases):
    result={}
    for arm in ('full_rgb','box_rgb'):
        groups={}
        for family in sorted({c['family'] for c in cases}):
            records=[c['arms'][arm] for c in cases if c['family']==family]
            cols=sum(r['target']['transverse_columns']['visible'] for r in records)
            good=sum(r['target']['transverse_columns']['with_at_least_half_pixels_agreeing'] for r in records)
            px=sum(r['target']['visible_pixels'] for r in records)
            true=sum(r['target']['range_agreeing_pixels'] for r in records)
            false=sum(r['silhouette_neighborhood']['target_depth_band_spurious_pixels'] for r in records)
            groups[family]=dict(frames=len(records),target_columns=cols,recovered_columns=good,column_coverage=good/cols if cols else None,
                target_pixels=px,agreeing_pixels=true,pixel_coverage=true/px if px else None,
                spurious_ring_pixels=false,local_precision=true/(true+false) if true+false else None)
        true=sum(g['agreeing_pixels'] for g in groups.values());false=sum(g['spurious_ring_pixels'] for g in groups.values())
        result[arm]=dict(families=groups,agreeing_pixels=true,spurious_ring_pixels=false,local_precision=true/(true+false) if true+false else None)
    full,box=result['full_rgb'],result['box_rgb']
    coverage=all(g['column_coverage'] is not None and g['column_coverage']>=.5 for g in full['families'].values())
    improved=[f for f,g in full['families'].items() if g['column_coverage'] is not None and box['families'][f]['column_coverage'] is not None and g['column_coverage']>box['families'][f]['column_coverage']]
    precision=full['local_precision'] is not None and box['local_precision'] is not None and full['local_precision']>=box['local_precision']
    passed=coverage and len(improved)>=3 and precision
    return dict(frames=len(cases),split='TRAIN_CONSUMED_IMPLEMENTATION_CHECK',arms=result,
        gate=dict(each_family_half_column_coverage_at_least_50pct=coverage,improved_families=improved,local_precision_not_worse=precision,passed=passed),
        decision='TRAIN_MECHANISM_ADMITTED_DEV_NOT_YET_RUN' if passed else 'STOP_PRETRAINED_TRANSFER_TRAIN_GEOMETRY_NOT_ADMITTED_KEEP_MZ129',
        dev_scored=False,original_test_scored=False,limits='Ablation sensitivity is not geometric correctness; native raster is evaluation-only collision geometry, not a UE depth image.')


def run(out):
    out=out.resolve();assert not out.exists() and out.is_relative_to((ROOT/'artifacts.local').resolve())
    base=ROOT/'artifacts.local/work/mz140-depthor-20260915';cap=WORK/'source/returned-v1/capture-v1'
    spec,receipt=read(cap/'spec.json'),read(cap/'receipt.json')
    counts=Counter();ids=set()
    for frame in spec['frames']:
        if frame['split']=='train' and counts[frame['family']]<12:
            ids.add(frame['id']);counts[frame['family']]+=1
    assert len(ids)==48 and set(counts.values())=={12}
    rows=public_observations(selected_jsonl(cap/'raw.jsonl',ids))
    prep=WORK/'incumbent/fresh-v1';cache=read(prep/'nominal/predictions.json')
    seal=read(prep/'prediction-seal.json');completed=read(prep/'completion.json')
    assert completed['status']=='PASS' and sha(prep/'prediction-seal.json')==completed['prediction_seal_sha256']
    assert sha(prep/'nominal/predictions.json')==seal['predictions_sha256']['nominal']
    assert rows==selected_jsonl(prep/'nominal/raw.jsonl',ids)
    assert sha(cap/'spec.json')==receipt['spec_sha256']
    for name in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/name)==receipt['hashes'][name]
    mapping={p['id']:i for i,p in enumerate(cache['predictions'])}
    cached=[cache['corrected'][mapping[r['id']]] for r in rows]
    files=[cap/n for n in ('spec.json','receipt.json','raw.jsonl','evaluator.jsonl')]
    files += [prep/'nominal/predictions.json',prep/'prediction-seal.json',base/'depthor-zju-small.pt',base/'backend-selection.json']
    images=[]
    for row in rows:
        path=cap/row['rgb_path'];assert path.resolve().is_relative_to(cap.resolve())
        assert sha(path)==receipt['hashes'][row['rgb_path']]
        im=cv2.imread(str(path));assert im.shape==(360,640,3);images.append(im);files.append(path)
    sources=local_dependencies(__file__)
    for path in (base/'upstream').rglob('*.py'):sources[str(path.resolve())]=sha(path)
    for name in ('evaluate_mz140_depthor.py','test_mz140_depthor.py'):
        path=CODE/name;assert path.exists();sources[str(path.resolve())]=sha(path)
    inputs={str(p):sha(p) for p in files}
    out.mkdir(parents=True)
    for src in sources:
        path=Path(src)
        relative=path.relative_to(ROOT.resolve()) if path.is_relative_to(ROOT.resolve()) else Path('artifact-dependencies')/path.relative_to(base.resolve())
        dest=out/'source-snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    shutil.copyfile(CODE/'MZ140_DEPTHOR_20260915.md',out/'protocol-before-fit.md')
    write(out/'freeze.json',dict(method=METHOD,split='train',ids=[r['id'] for r in rows],sources=sources,inputs=inputs,
        protocol_sha256=sha(out/'protocol-before-fit.md'),train_admission='target column coverage>=.5 each family; improve >=3 families; local precision not worse',
        device='cuda',no_training=True,no_native_inputs=True))
    torch.set_num_threads(4);model,info=load_model(base/'upstream',base/'depthor-zju-small.pt','cuda')
    outputs={};input_audits=[];timings=[]
    try:
        for i,(row,bgr,old) in enumerate(zip(rows,images,cached)):
            pair={};ablated,fraction=box_only_rgb(bgr,old)
            for arm,im in [('full_rgb',bgr),('box_rgb',ablated)]:
                data,audit=make_input(row,im,'cuda')
                torch.cuda.synchronize();start=time.perf_counter()
                depth=predict(model,data);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                value=depth.cpu().numpy();assert np.isfinite(value).all()
                path=out/f'{i:03d}-{arm}.npy';np.save(path,value,allow_pickle=False)
                outputs[str(path)]=sha(path);pair[arm]=value
                timings.append(dict(id=row['id'],arm=arm,seconds=elapsed))
                if arm=='full_rgb':input_audits.append(dict(id=row['id'],box_fraction=fraction,**audit))
            print(json.dumps(dict(completed=i+1,total=48,id=row['id'],used=len(audit['used']),rgb_effect_mean_m=float(np.abs(pair['full_rgb']-pair['box_rgb']).mean()))),flush=True)
        write(out/'input-audits.json',input_audits);write(out/'timings.json',timings)
        write(out/'prediction-seal.json',dict(outputs=outputs,freeze_sha256=sha(out/'freeze.json'),
            authority='OBSERVATION_PREDICTIONS_SEALED_BEFORE_TRAIN_NATIVE_PARSE',model_load=info))
        from evaluate_mz140_depthor import evaluate_frame
        es=selected_jsonl(cap/'evaluator.jsonl',ids)
        cases=[]
        for i,(row,e,old) in enumerate(zip(rows,es,cached)):
            assert row['id']==e['id']
            cases.append(dict(id=row['id'],family=e['family'],arms={arm:evaluate_frame(row,e,np.load(out/f'{i:03d}-{arm}.npy'),old) for arm in ('full_rgb','box_rgb')}))
        write(out/'geometry-cases.json',cases)
        summary=summarize(cases);write(out/'summary.json',summary)
        assert inputs=={p:sha(Path(p)) for p in inputs}
        assert sources=={p:sha(Path(p)) for p in sources}
        write(out/'completion.json',dict(status='PASS',freeze_sha256=sha(out/'freeze.json'),prediction_seal_sha256=sha(out/'prediction-seal.json'),
            geometry_cases_sha256=sha(out/'geometry-cases.json'),summary_sha256=sha(out/'summary.json'),inputs_and_sources_unchanged=True,dev_accessed=False,training=False,
            resources='Process-local model only; no persistent worker or paid allocation'))
        print(json.dumps(summary,indent=2))
    except Exception as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),predictions_sealed=(out/'prediction-seal.json').exists()))
        raise
    finally:
        del model;torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
