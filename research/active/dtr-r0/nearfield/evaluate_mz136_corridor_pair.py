"""Select on new development groups, seal, evaluate unseen groups unchanged.

Candidate outputs never receive MZ129 alarm inputs. No test threshold curves.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
from mz136_corridor_pair import CorridorNet
from mz120_occupancy import encode
from run_mz136_corridor_pair import load,pair_indices,predict,fit_scores
from run_mz107_four_sensor import readrows,sha,write,truth,metrics
from run_mz111_spatial_temporal import event_metrics
from mz136_incumbent import public_observations


def flags_at_threshold(scores,threshold):
    assert np.isfinite(scores).all() and 0<=threshold<=1
    if threshold==0:return np.ones(scores.shape,bool)
    if threshold==1:return np.zeros(scores.shape,bool)
    return scores>=np.log(threshold/(1-threshold))


def pair_metrics(gt,scores,flags,pairs):
    changed=[p for p in pairs if gt[p['a']]!=gt[p['b']]]
    correct=sum((scores[p['a']]-scores[p['b']])*(int(gt[p['a']])-int(gt[p['b']]))>0 for p in changed)
    both=sum(flags[p['a']]==gt[p['a']] and flags[p['b']]==gt[p['b']] for p in pairs)
    return dict(changed_pairs=len(changed),order_correct=int(correct),pairs=len(pairs),
                both_correct=int(both),both_rate=both/max(1,len(pairs)))


def event_details(rows,gt,pred):
    events=[];active=None;previous=None
    for row,y,p in zip(rows,gt,pred):
        if row['episode_id']!=previous or not y: active=None
        previous=row['episode_id']
        if y and active is None:
            active=dict(episode=previous,onset_s=row['time_s'],first_alert_s=None)
            events.append(active)
        if y and p and active['first_alert_s'] is None:active['first_alert_s']=row['time_s']
    return events


def score(rows,es,gt,flags,indices,baseline=None):
    ix=np.array(indices,dtype=int);rr=[rows[i] for i in ix]
    result=dict(metrics=metrics(gt[ix],flags[ix]),events=event_metrics(rr,gt[ix],flags[ix]),
                event_details=event_details(rr,gt[ix],flags[ix]),families={},frames=len(ix),
                unknown_semantics='No-alert only; not calibrated abstention or clear-space evidence')
    for family in sorted({es[i]['family'] for i in ix}):
        fi=np.array([i for i in ix if es[i]['family']==family])
        result['families'][family]=metrics(gt[fi],flags[fi])
    if baseline is not None:
        result['lost_baseline_tp']=[rows[i]['id'] for i in ix if gt[i] and baseline[i] and not flags[i]]
        result['removed_baseline_fp']=[rows[i]['id'] for i in ix if not gt[i] and baseline[i] and not flags[i]]
        result['new_fp']=[rows[i]['id'] for i in ix if not gt[i] and not baseline[i] and flags[i]]
    return result


def retention(candidate,baseline):
    cm,bm=candidate['metrics'],baseline['metrics']
    cr=cm['TP']/max(1,cm['TP']+cm['FN']);br=bm['TP']/max(1,bm['TP']+bm['FN'])
    events_ok=True;deltas=[]
    assert len(candidate['event_details'])==len(baseline['event_details'])
    for c,b in zip(candidate['event_details'],baseline['event_details']):
        assert (c['episode'],c['onset_s'])==(b['episode'],b['onset_s'])
        if b['first_alert_s'] is not None:
            delta=None if c['first_alert_s'] is None else c['first_alert_s']-b['first_alert_s']
            events_ok &= delta is not None and delta<=.25+1e-9
            deltas.append(dict(episode=c['episode'],onset_s=c['onset_s'],relative_delay_s=delta))
    return dict(recall_within_2pp=cr>=br-.02-1e-9,incumbent_events_and_timing_retained=bool(events_ok),
                per_event_delta=deltas,pass_retention=bool(cr>=br-.02-1e-9 and events_ok))


def choose(rows,es,gt,scores,dev,baseline):
    base=score(rows,es,gt,baseline,dev)
    curve=[]
    for t in np.linspace(0,1,101):
        candidate=score(rows,es,gt,flags_at_threshold(scores,float(t)),dev,baseline)
        candidate.update(threshold=float(t),retention=retention(candidate,base))
        curve.append(candidate)
    eligible=[c for c in curve if c['retention']['pass_retention']]
    chosen=min(eligible,key=lambda c:(c['metrics']['FP'],-c['metrics']['TP'],
        c['events']['max_detected_delay_s'] or 0,-c['threshold'])) if eligible else None
    return chosen,curve


def infer(model_path,data):
    model=CorridorNet().cuda();model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))
    t=time.perf_counter();s=predict(model,data);torch.cuda.synchronize();seconds=time.perf_counter()-t
    del model;return s,seconds


def encode_rows(rows,original_data):
    values=[];yaw=0.;episode=None
    for r in rows:
        if episode!=r['episode_id']:yaw=0.
        if r['imu_valid']:yaw+=r['delta_yaw']
        episode=r['episode_id'];values.append(encode(r,yaw))
    data={k:torch.from_numpy(np.stack([v[k] for v in values])) for k in values[0]}
    data['rgb']=original_data['rgb']
    return data


def run(args):
    out=args.output.resolve();assert not out.exists() and out.is_relative_to((ROOT/'artifacts.local').resolve());out.mkdir(parents=True)
    torch.set_num_threads(4);assert torch.cuda.is_available()
    rows,es,spec,data=load(args.capture,args.capture)
    gt=np.array([truth(e) for e in es],bool);pairs=pair_indices(rows,spec)
    partitions={s:sorted({i for p in pairs if p['split']==s for i in (p['a'],p['b'])}) for s in ('train','dev','test')}
    assert [len(partitions[s]) for s in ('train','dev','test')]==[192,48,48]
    prep=json.loads((args.incumbent/'completion.json').read_text());assert prep['status']=='PASS'
    seal=json.loads((args.incumbent/'prediction-seal.json').read_text())
    assert sha(args.incumbent/'prediction-seal.json')==prep['prediction_seal_sha256']
    for path,digest in seal['inputs'].items():assert sha(Path(path))==digest,path
    for path,digest in seal['source_hashes'].items():assert sha(Path(path))==digest,path
    for arm,digest in seal['predictions_sha256'].items():assert sha(args.incumbent/arm/'predictions.json')==digest
    assert sha(args.incumbent/'observation-seal.json')==seal['observation_seal_sha256']
    observations=json.loads((args.incumbent/'observation-seal.json').read_text())
    for arm,digest in observations['hashes'].items():assert sha(args.incumbent/arm/'raw.jsonl')==digest
    nominal_rows=readrows(args.incumbent/'nominal/raw.jsonl')
    assert nominal_rows==public_observations(rows), 'Incumbent source differs from learner capture'
    nominal=json.loads((args.incumbent/'nominal/predictions.json').read_text())['predictions']
    assert [p['id'] for p in nominal]==[r['id'] for r in rows]
    baseline=np.array([p['candidate'] for p in nominal],bool)
    frozen=json.loads((args.fits/'freeze.json').read_text())
    fit_receipt=json.loads((args.fits/'completion.json').read_text())
    assert fit_receipt['status']=='PASS' and sha(args.fits/'freeze.json')==fit_receipt['freeze_sha256']
    fit_summary=json.loads((args.fits/'summary.json').read_text())
    assert sha(args.fits/'summary.json')==fit_receipt['summary_sha256']
    for name,digest in frozen['code'].items():
        assert sha(Path(__file__).with_name(name))==digest,name
    for arm,value in fit_summary['arms'].items():
        assert sha(args.fits/(arm+'-model.pt'))==value['model_sha256']
        assert sha(args.fits/(arm+'-scores.npy'))==fit_receipt['scores_sha256'][arm]
    for name in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json'):
        assert sha(args.capture/name)==frozen['source'][name]
    fits={a:np.load(args.fits/(a+'-scores.npy')) for a in ('bce','pair')}
    assert all(s.shape==(288,) for s in fits.values())
    selected={};curves={}
    for a in fits:selected[a],curves[a]=choose(rows,es,gt,fits[a],partitions['dev'],baseline)
    for a,c in curves.items():write(out/(a+'-development-curve.json'),c)
    thresholds={a:None if c is None else c['threshold'] for a,c in selected.items()}
    write(out/'operating-point-seal.json',dict(thresholds=thresholds,
        models={a:sha(args.fits/(a+'-model.pt')) for a in fits},scores={a:sha(args.fits/(a+'-scores.npy')) for a in fits},
        dev_frame_ids=[rows[i]['id'] for i in partitions['dev']],test_frame_ids=[rows[i]['id'] for i in partitions['test']],
        policy='DEV_ONLY_MIN_FP_UNDER_RECALL_AND_PER_EVENT_TIMING_RETENTION',
        source=frozen['source'],evaluator_code_sha256=sha(Path(__file__)),
        test_outcomes_used_for_selection=False))
    result=dict(thresholds=thresholds,partitions={},perturbation={},legacy_recheck={})
    for part,idx in partitions.items():
        arms={'mz129':score(rows,es,gt,baseline,idx)}
        for a,s in fits.items():
            if thresholds[a] is None:arms[a]=None;continue
            pred=flags_at_threshold(s,thresholds[a])
            arms[a]=score(rows,es,gt,pred,idx,baseline)
            arms[a]['retention']=retention(arms[a],arms['mz129'])
            selected_pairs=[p for p in pairs if p['split']==part]
            arms[a]['paired']=pair_metrics(gt,s,pred,selected_pairs)
        result['partitions'][part]=arms
    shifted=readrows(args.incumbent/'shifted/raw.jsonl')
    sb=json.loads((args.incumbent/'shifted/predictions.json').read_text())['predictions']
    assert [r['id'] for r in shifted]==[r['id'] for r in rows]==[p['id'] for p in sb]
    for original,new in zip(nominal_rows,shifted):
        assert {k:v for k,v in original.items() if k!='tof_zones'}=={k:v for k,v in new.items() if k!='tof_zones'}
    shifted_base=np.array([p['candidate'] for p in sb],bool)
    sd=encode_rows(shifted,data);shifted_scores={};timings={}
    for a in fits:
        shifted_scores[a],timings[a]=infer(args.fits/(a+'-model.pt'),sd)
        np.save(out/(a+'-shifted-scores.npy'),shifted_scores[a])
    write(out/'perturbation-prediction-seal.json',dict(operating_point_sha256=sha(out/'operating-point-seal.json'),
        raw_sha256=sha(args.incumbent/'shifted/raw.jsonl'),incumbent_sha256=sha(args.incumbent/'shifted/predictions.json'),
        scores={a:sha(out/(a+'-shifted-scores.npy')) for a in fits}))
    idx=partitions['test'];result['perturbation']['mz129']=score(rows,es,gt,shifted_base,idx)
    for a,s in shifted_scores.items():
        if thresholds[a] is None:result['perturbation'][a]=None;continue
        pred=flags_at_threshold(s,thresholds[a])
        r=score(rows,es,gt,pred,idx,shifted_base)
        r['retention']=retention(r,result['perturbation']['mz129']);result['perturbation'][a]=r
    # The historical panel stays a consumed Development recheck, never test selection.
    old=ROOT/'artifacts.local/work/mz123-frozen-early-20260913/returned-v1/capture-v1'
    rgb=ROOT/'artifacts.local/work/mz125-observable-correction-20260913/rgb-v1'
    lr,le,ls,ld=load(old,rgb);ly=np.array([truth(e) for e in le],bool)
    historical=ROOT/'artifacts.local/work/mz129-extent-correction-20260914/replay-v1r2/predictions.json'
    hp=json.loads(historical.read_text())['radar'];assert [p['id'] for p in hp]==[r['id'] for r in lr]
    lb=np.array([p['candidate'] for p in hp],bool);ii=list(range(len(lr)))
    result['legacy_recheck']['mz129']=score(lr,le,ly,lb,ii)
    for a in fits:
        s,seconds=infer(args.fits/(a+'-model.pt'),ld);np.save(out/(a+'-legacy-scores.npy'),s)
        if thresholds[a] is None:result['legacy_recheck'][a]=None;continue
        p=flags_at_threshold(s,thresholds[a])
        r=score(lr,le,ly,p,ii,lb);r['retention']=retention(r,result['legacy_recheck']['mz129'])
        result['legacy_recheck'][a]=r
    tests=result['partitions']['test'];flags={}
    for a in fits:
        flags[a]={}
        for n,rr in [('nominal',tests),('shifted',result['perturbation'])]:
            v=rr[a];bm=rr['mz129']['metrics']
            flags[a][n]=bool(v and v['retention']['pass_retention'] and bm['FP']>0 and v['metrics']['FP']<=.8*bm['FP'])
    pair_increment=bool(tests['pair'] and tests['bce'] and result['perturbation']['pair'] and result['perturbation']['bce']
        and retention(tests['pair'],tests['bce'])['pass_retention']
        and retention(result['perturbation']['pair'],result['perturbation']['bce'])['pass_retention']
        and tests['pair']['metrics']['FP']<tests['bce']['metrics']['FP']
        and result['perturbation']['pair']['metrics']['FP']<result['perturbation']['bce']['metrics']['FP'])
    result.update(target_flags=flags,pair_increment_over_bce=pair_increment,
        decision=('RETAIN_PAIRED_DEVELOPMENT_CHALLENGER' if all(flags['pair'].values()) and pair_increment else
                  'PREFER_SIMPLER_BCE_DEVELOPMENT_CHALLENGER' if all(flags['bce'].values()) else
                  'KEEP_MZ129_NO_JOINT_ALERT_GAIN'),
        inference_timings_seconds=timings,limits='Single seed, small procedural groups in same renderer; no calibrated probabilities, natural-use, hardware or safety claims')
    write(out/'summary.json',result);torch.cuda.empty_cache()
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},owned_resources='foreground process only'))
    print(json.dumps(dict(decision=result['decision'],thresholds=thresholds,flags=flags,pair_increment=pair_increment,
        test={a:None if r is None else r['metrics'] for a,r in tests.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('capture','fits','incumbent','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
