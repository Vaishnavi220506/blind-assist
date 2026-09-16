"""Evaluator-only first-hit/public-pose/grid feasibility. No model invocation."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from run_mz161_dense_task import ROOT,CODE,CAP,BASE,FOLDS,read,write,sha
from run_mz139_surface_fit import selected_jsonl,local_dependencies
from mz136_incumbent import public_observations
from mz161_dense_task import ray_query
from run_mz107_four_sensor import truth
from run_mz143_corridor_evidence import native_account
from evaluate_mz136_corridor_pair import score,pair_metrics
from evaluate_mz161_dense_task import _rates,_comparison,_spatial_counts,_spatial_report
from mz167_range_readout import reference_readouts

WORK=ROOT/'artifacts.local/work/mz167-range-readout-20260916'
OLD=ROOT/'artifacts.local/work/mz161-dense-task-20260916/run-v1'
ARMS=('native','public_exact','bin_midpoint','bin_mass')
LIMIT=300.


def run(out):
    out=Path(out).resolve();assert out.is_relative_to(WORK.resolve()) and not out.exists()
    out.mkdir(parents=True);start=time.perf_counter();cv2.setNumThreads(4)
    def within():
        if time.perf_counter()-start>=LIMIT:raise TimeoutError('MZ167 fixed300s exhausted')
    spec,receipt=read(CAP/'spec.json'),read(CAP/'receipt.json')
    assert receipt['status']=='PASS' and sha(CAP/'spec.json')==receipt['spec_sha256']
    for n in ('raw.jsonl','evaluator.jsonl'):assert sha(CAP/n)==receipt['hashes'][n]
    frames=[f for f in spec['frames'] if f['split']=='train'];assert len(frames)==192
    metadata={f['id']:dict(family=f['family'],scene_group=f['scene_group'],partition='fit')
        for f in frames if not f['scene_group'].endswith('_scene3')}
    assert len(metadata)==144
    rows=public_observations(selected_jsonl(CAP/'raw.jsonl',set(metadata)));ids=[r['id'] for r in rows]
    assert set(ids)==set(metadata) and len(ids)==144
    freeze=read(OLD/'freeze.json');inputs0=read(OLD/'input-seal.json');labels0=read(OLD/'fit-label-seal.json')
    complete0=read(OLD/'completion.json');pred0=read(OLD/'prediction-seal.json');model0=read(OLD/'model-seal.json')
    assert complete0['status']=='PASS' and sha(OLD/'prediction-seal.json')==complete0['prediction_seal_sha256']
    assert sha(OLD/'model-seal.json')==pred0['model_seal_sha256']
    assert sha(OLD/'input-seal.json')==pred0['input_seal_sha256']
    assert sha(OLD/'fit-label-seal.json')==model0['fit_label_seal_sha256']
    assert sha(OLD/'freeze.json')==inputs0['freeze_sha256']
    for p,h in freeze['inputs'].items():assert sha(p)==h,p
    for name,key in (('public-features.npy','features_sha256'),('public-tokens.npz','tokens_sha256')):
        assert sha(OLD/name)==inputs0[key]
    for name,key in (('fit-targets.npz','targets_sha256'),('fit-label-audit.json','audit_sha256')):
        assert sha(OLD/name)==labels0[key]
    allids=[r['id'] for r in read(FOLDS)]
    assert allids==[r['id'] for r in read(OLD/'input-audit.json')]
    ix=np.array([i for i,id in enumerate(allids) if id in metadata]);assert [allids[i] for i in ix]==ids
    assert sorted(labels0['fit_ids'])==sorted(ids)
    for id in ids:assert metadata[id]==freeze['metadata'][id]
    assert [r['id'] for r in read(BASE/'folds.json')[:192]]==allids
    assert sha(BASE/'oof.npz')==read(BASE/'model-seal.json')['oof_sha256']
    with np.load(BASE/'oof.npz',allow_pickle=False) as c:
        assert c['baseline'].shape==(480,);baseline=c['baseline'][ix].astype(bool)
    with np.load(OLD/'public-tokens.npz',allow_pickle=False) as c:yaws=c['yaws'][ix]
    with np.load(OLD/'fit-targets.npz',allow_pickle=False) as c:
        assert np.array_equal(c['indices'],ix);parent_targets=c['target'];parent_frames=c['frame'].astype(bool)
    sources=local_dependencies(__file__)
    for name in ('MZ167_PROTOCOL_20260916.md','test_mz167_range_readout.py'):sources[str(CODE/name)]=sha(CODE/name)
    inputs=dict(freeze['inputs'])
    for n in ('completion.json','prediction-seal.json','model-seal.json','freeze.json','input-seal.json','input-audit.json',
              'public-features.npy','public-tokens.npz','fit-targets.npz','fit-label-seal.json','fit-label-audit.json'):
        inputs[str(OLD/n)]=sha(OLD/n)
    for p in [FOLDS,*[BASE/n for n in ('folds.json','oof.npz','model-seal.json')],WORK/'preflight.json',WORK/'tests.log']:
        inputs[str(p)]=sha(p)
    write(out/'freeze.json',dict(authority='EVALUATOR_ONLY_ORACLE_NOT_PREDICTOR',sources=sources,inputs=inputs,
        ids=ids,metadata=metadata,grid=dict(lower=0.,upper=8.,bins=128,width_m=.0625,phase_m=0.,overflow_representative_m=8.03125),
        mass_threshold=.5,budget_seconds=LIMIT,backend='NumPy CPU',backend_reason='TASK_NOT_GPU_SUITABLE',
        access='FIT144_NATIVE_ONLY; FULL_SPEC_AND480TRAIN_BASELINE_MEMBER; NO_HELD_DEV_TEST_NATIVE_OR_RGB_DECODE'))
    print('ORACLE FREEZE SEALED',flush=True);within()
    es=selected_jsonl(CAP/'evaluator.jsonl',set(ids));assert [e['id'] for e in es]==ids
    n=len(rows);shape=(n,360,640)
    masks=np.lib.format.open_memmap(out/'masks.npy',mode='w+',dtype=bool,shape=(len(ARMS),*shape))
    known=np.lib.format.open_memmap(out/'known.npy',mode='w+',dtype=bool,shape=shape)
    slant=np.lib.format.open_memmap(out/'reference-slant.npy',mode='w+',dtype=np.float64,shape=shape)
    mass=np.lib.format.open_memmap(out/'bin-mass.npy',mode='w+',dtype=np.float32,shape=shape)
    cases=[];audits=[];gt=np.array([truth(e) for e in es],bool);assert np.array_equal(gt,parent_frames)
    for i,(row,e,yaw) in enumerate(zip(rows,es,yaws)):
        result=reference_readouts(row,e,float(yaw));label=result['label']
        assert np.array_equal(result['masks']['native'],parent_targets[i])
        known[i]=result['known'];slant[i]=result['reference_slant'];mass[i]=result['scores']['bin_mass']
        public=ray_query(row,float(yaw))[1][2]>0
        case=dict(id=row['id'],episode_id=row['episode_id'],time_s=row['time_s'],**metadata[row['id']],
            truth=bool(gt[i]),baseline=bool(baseline[i]),flags={},spatial={})
        for ai,arm in enumerate(ARMS):
            masks[ai,i]=result['masks'][arm];assert not masks[ai,i][~known[i]].any()
            case['flags'][arm]=bool(masks[ai,i].any())
            case['spatial'][arm]=_spatial_counts(label,np.where(masks[ai,i],1.,-1.),public)
        case['native_volume_vs_visible_mismatch']=bool(gt[i]!=case['flags']['native'])
        cases.append(case);audits.append(dict(id=row['id'],**result['audit']));within()
        if (i+1)%24==0:print(f'ORACLE {i+1}/144 seconds={time.perf_counter()-start:.2f}',flush=True)
    for a in (masks,known,slant,mass):a.flush()
    write(out/'readout-audit.json',audits)
    outputs=('masks.npy','known.npy','reference-slant.npy','bin-mass.npy','readout-audit.json')
    write(out/'output-seal.json',dict(authority='ALL_ARMS_EVALUATOR_ONLY_INCLUDING_RANGE_AND_KNOWN_MASK',
        outputs={name:sha(out/name) for name in outputs},freeze_sha256=sha(out/'freeze.json'),arm_order=list(ARMS)))
    flags={'baseline':baseline,**{arm:np.array([c['flags'][arm] for c in cases],bool) for arm in ARMS}}
    episodes={}
    for i,r in enumerate(rows):episodes.setdefault(r['episode_id'],[]).append(i)
    pairs=[]
    for pair in spec['pairs']:
        a,b=pair['episodes']
        if a not in episodes and b not in episodes:continue
        assert a in episodes and b in episodes and len(episodes[a])==len(episodes[b])
        for i,j in zip(episodes[a],episodes[b]):
            assert rows[i]['time_s']==rows[j]['time_s'];pairs.append(dict(a=i,b=j))
    reports={};spatial={};indices=list(range(n));families=sorted({m['family'] for m in metadata.values()})
    for arm,flag in flags.items():
        r=_rates(score(rows,es,gt,flag,indices,baseline));r['pairs']=pair_metrics(gt,flag.astype(float),flag,pairs)
        r['families']={f:_rates(dict(metrics=m)) for f,m in r['families'].items()}
        r['strata']={name:_rates(score(rows,es,gt,flag,[i for i,e in enumerate(es) if (e['family']=='shallow_boundary_stress')==pressure],baseline))
            for name,pressure in (('ordinary',False),('boundary_pressure',True))}
        native=native_account(rows,es,flag,baseline);r['native']={k:v for k,v in native.items() if k!='cases'}
        r['native']['radar_native_lineage']='NOT_EVALUABLE'
        for c,v in zip(cases,native['cases']):c.setdefault('native',{})[arm]=v
        reports[arm]=r
        if arm!='baseline':
            r['vs_baseline']=_comparison(rows,gt,flag,baseline,indices,r,reports['baseline'])
            spatial[arm]=dict(all=_spatial_report(cases,arm,indices),families={f:_spatial_report(cases,arm,[i for i,e in enumerate(es) if e['family']==f]) for f in families})
    for arm in ARMS:
        for reference in ('native','public_exact'):
            reports[arm]['vs_'+reference]=_comparison(rows,gt,flags[arm],flags[reference],indices,reports[arm],reports[reference])
    e=reports['public_exact'];exact=dict(native_true_frames_retained=e['vs_native']['all_reference_true_frames_retained'],
        baseline_true_frames_retained=e['vs_baseline']['all_reference_true_frames_retained'],
        no_new_false_frames=not e['vs_native']['new_fp'],baseline_events_onsets_retained=e['vs_baseline']['all_reference_events_without_extra_delay'])
    quant={}
    for arm in ('bin_midpoint','bin_mass'):
        c=reports[arm]['vs_public_exact'];s=spatial[arm]['families']
        quant[arm]=dict(exact_true_frames_retained=c['all_reference_true_frames_retained'],no_new_false_frames=not c['new_fp'],
            exact_events_onsets_retained=c['all_reference_events_without_extra_delay'],
            all_family_pixel_precision_recall_90=all(s[f][k] is not None and s[f][k]>=.9 for f in families for k in ('precision','recall')))
    passed=all(exact.values()) and all(all(c.values()) for c in quant.values())
    decision='MZ167_FIRST_HIT_GRID_ELIGIBLE_FOR_LEARNING' if passed else 'MZ167_FIXED_FIRST_HIT_GRID_NOT_ADMITTED'
    summary=dict(authority='EVALUATOR_ONLY_FIT144_ORACLE_NOT_LEARNED_OR_DEPLOYABLE',frames=n,arms=reports,spatial=spatial,
        gates=dict(exact_public_checks=exact,quantized_checks=quant,overall_pass=passed),decision=decision,
        native_volume_vs_visible_mismatches=sum(c['native_volume_vs_visible_mismatch'] for c in cases),
        limits=['All masks use native reference ranges and known masks; none is a public predictor.',
            'UNKNOWN is masked for evaluation, not a model-derived availability gate or clear-space claim.',
            'Grid mass assumes uniform within-bin density; overflow point is explicitly capped.',
            'No original HELD/dev/test native or RGB decode; no learned distribution, fit or alert integration.'])
    write(out/'cases.json',cases);write(out/'summary.json',summary)
    for p,h in sources.items():assert sha(p)==h,p
    for p,h in inputs.items():assert sha(p)==h,p
    within();write(out/'completion.json',dict(status='PASS',seconds=time.perf_counter()-start,decision=decision,
        output_seal_sha256=sha(out/'output-seal.json'),outputs={n:sha(out/n) for n in ('summary.json','cases.json')},
        learned_model_invocations=0,resources='Process-local CPU; no persistent allocation'))
    print(decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WORK/'run-v1');args=parser.parse_args()
    try:run(args.output)
    except Exception as exc:
        if args.output.exists():write(args.output/'failure.json',dict(error=type(exc).__name__,message=str(exc),resume='NO_AUTOMATIC_RESTART'))
        raise
