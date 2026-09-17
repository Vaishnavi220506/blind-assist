"""Independent nested split/checkpoint/threshold audit; no training imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
HOME=ROOT/'artifacts.local/work/corridor-public-positive-v2-20260917'
PREP=HOME/'preparation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,p):return dict(frames=len(y),TP=int(sum(y&p)),FP=int(sum(~y&p)),FN=int(sum(y&~p)),TN=int(sum(~y&~p)))

def audit(arm):
    out=HOME/arm;done=read(out/'completion.json');assert done['status']=='PASS'
    assert sha(out/'summary.json')==done['summary_sha256']
    assert sha(out/'prediction-seal.json')==done['prediction_seal_sha256']
    seal=read(out/'prediction-seal.json')
    for n,h in seal['outputs'].items():assert sha(out/n)==h,n
    freeze=read(out/'freeze.json');assert freeze['arm']==arm and freeze['peak_lambda']==int(arm=='peak')
    for p,h in freeze['sources'].items():assert sha(p)==h,p
    assert sha(PREP/'data-seal.json')==freeze['data_seal_sha256']
    ds=read(PREP/'data-seal.json')
    for n,h in ds['outputs'].items():assert sha(PREP/n)==h,n
    for p,h in ds['bindings'].items():assert sha(p)==h,p
    meta=read(PREP/'metadata.json');data=np.load(PREP/'public-tokens.npz');labels=np.load(PREP/'offline-targets.npz')
    v=data['valid'][:,:128];y=labels['target'][:,:128];known=labels['known'][:,:128]
    assert len(meta)==1344 and not (known&~v).any() and not labels['known'][:,128:].any()
    token=data['tokens'][:,:128];units=np.array([4,2,3],np.float32)
    center=token[:,:,5:8]*units;bounds=token[:,:,8:14].reshape(1344,128,3,2)*units[:,None]
    low=np.array([.2,-.3,.4],np.float32);high=np.array([3.6,.3,2.05],np.float32)
    features=np.concatenate([token,np.minimum(center-low,high-center),np.minimum(bounds[:,:,:,1]-low,high-bounds[:,:,:,0]),
        np.minimum(bounds[:,:,:,0]-low,high-bounds[:,:,:,1])],-1)
    preds=np.load(out/'predictions.npz');ri_all=[i for i,m in enumerate(meta) if m['cohort']!='anchor']
    assert list(preds['indices'])==ri_all;positions={i:j for j,i in enumerate(ri_all)}
    device=freeze['device'];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    seen=Counter();checks=[];checkpoint_count=0;paired_fit_count=0
    def checkpoint(tag,fit,seed,infer_indices):
        nonlocal checkpoint_count,paired_fit_count
        record=read(out/(tag+'-fit.json'));assert record['indices']==fit and record['seed']==seed
        other=HOME/('bce' if arm=='peak' else 'peak')/(tag+'-fit.json')
        if other.exists():
            partner=read(other)
            for key in ['indices','seed','orders','peak_eligible_indices']:assert record[key]==partner[key],(tag,key)
            paired_fit_count+=1
        assert len(record['orders'])==len(record['history'])==120
        rng=np.random.default_rng(seed)
        for order in record['orders']:assert order==rng.permutation(np.array(fit)).tolist()
        eligible=[i for i in fit if meta[i]['A'] is False and not meta[i]['truth'] and meta[i]['stratum']=='negative' and v[i].any()]
        assert record['peak_eligible_indices']==sorted(eligible)
        assert all(meta[i]['cohort']!='anchor' for i in eligible)
        weights=torch.load(out/(tag+'.pt'),map_location=device,weights_only=False)
        assert weights['seed']==seed and weights['epoch']==120
        s=weights['state_dict'];vals=features[fit][v[fit]]
        np.testing.assert_array_equal(s['mean'].cpu().numpy(),vals.mean(0))
        np.testing.assert_array_equal(s['scale'].cpu().numpy(),vals.std(0).clip(.01))
        chunks=[]
        with torch.inference_mode():
            for offset in range(0,len(infer_indices),64):
                ix=infer_indices[offset:offset+64];valid=torch.as_tensor(v[ix],device=device)
                f=torch.as_tensor(features[ix],device=device)
                f=(torch.where(valid[...,None],f,s['mean'])-s['mean'])/s['scale']
                for layer in [0,2,4]:
                    f=torch.nn.functional.linear(f,s[f'net.{layer}.weight'],s[f'net.{layer}.bias'])
                    if layer!=4:f=torch.relu(f)
                chunks.append(f.squeeze(-1).masked_fill(~valid,-30.).cpu().numpy())
        checkpoint_count+=1
        return np.concatenate(chunks)
    for fold in freeze['folds']:
        k=fold['fold'];fit=[i for i,m in enumerate(meta) if m['cohort']=='anchor' or m['scene_index']!=k]
        report=[i for i,m in enumerate(meta) if m['cohort']!='anchor' and m['scene_index']==k]
        assert fold['fit']==fit and fold['report']==report
        assert not {meta[i]['group'] for i in fit}&{meta[i]['group'] for i in report}
        dev=[i for i in range(6) if i!=k];cal_by_index={};cal_seen=Counter()
        for j,inner in enumerate(fold['inner']):
            held=dev[j::3];ii=[i for i in fit if meta[i]['cohort']=='anchor' or meta[i]['scene_index'] not in held]
            ci=[i for i in fit if meta[i]['cohort']!='anchor' and meta[i]['scene_index'] in held]
            assert inner==dict(inner=j,fit=ii,calibration=ci,scene_indices=held)
            assert not {meta[i]['group'] for i in ii}&{meta[i]['group'] for i in ci}
            logits=checkpoint(f'outer{k}-inner{j}',ii,184017+10*k+j,ci)
            for i,z in zip(ci,logits):cal_by_index[i]=z;cal_seen[i]+=1
        ci=[i for i in fit if meta[i]['cohort']!='anchor'];assert cal_seen==Counter({i:1 for i in ci})
        saved=np.load(out/f'outer{k}-calibration.npz');assert list(saved['indices'])==ci
        np.testing.assert_array_equal(np.stack([cal_by_index[i] for i in ci]),saved['logits'])
        score=np.where(v[ci],saved['logits'],-30.).max(1).astype(np.float64);eligible=v[ci].any(1)
        a=np.array([meta[i]['A'] for i in ci]);truth=np.array([meta[i]['truth'] for i in ci]);clear=np.array([meta[i]['stratum']!='boundary' for i in ci])
        choices=np.r_[np.nextafter(float(score.max()),np.inf),np.unique(score[eligible&clear])];curve=[]
        assert choices[0]>score.max()
        for threshold in choices:
            p=(a|(eligible&(score>=threshold)))[clear];m=metric(truth[clear],p)
            curve.append(dict(threshold=float(threshold),TP=m['TP'],FP=m['FP'],FN=m['FN'],f1=2*m['TP']/max(1,2*m['TP']+m['FP']+m['FN'])))
        chosen=max(curve,key=lambda m:(m['f1'],-m['FP'],m['threshold']));sel=read(out/f'outer{k}-selection.json')
        assert sel['curve']==curve and sel['chosen']==chosen
        logits=checkpoint(f'outer{k}-final',fit,184017+10*k+3,report)
        pp=[positions[i] for i in report];np.testing.assert_array_equal(logits,preds['logits'][pp])
        assert (preds['thresholds'][pp]==chosen['threshold']).all() and (preds['folds'][pp]==k).all()
        witness=((y[ci]>0)&known[ci]).any(1)
        assert sel['clear_A_FN']==int(sum(clear&truth&~a))
        assert sel['clear_supported_A_FN']==int(sum(clear&truth&~a&witness))
        added=v[ci].any(1)&(score>=chosen['threshold'])&~a
        seen.update(report);checks.append(dict(fold=k,calibration_frames=len(ci),report_frames=len(report),threshold=chosen['threshold'],
            calibration_clear_A_FN=sel['clear_A_FN'],calibration_clear_supported_A_FN=sel['clear_supported_A_FN'],
            inactive_candidate_selected=chosen['threshold']==float(choices[0]),inactive_candidate=curve[0],chosen=chosen,
            calibration_added_clear=[dict(id=meta[i]['id'],truth=bool(truth[n]),sampled_witness=bool(witness[n]),logit=float(score[n]))
                for n,i in enumerate(ci) if clear[n] and added[n]]))
    assert seen==Counter({i:1 for i in ri_all})
    mm=[meta[i] for i in ri_all];a=np.array([m['A'] for m in mm]);truth=np.array([m['truth'] for m in mm]);state=np.array([m['stratum'] for m in mm]);cohort=np.array([m['cohort'] for m in mm])
    vv=v[ri_all];tt=y[ri_all];kk=known[ri_all];witness=((tt>0)&kk).any(1)
    score=np.where(vv,preds['logits'],-30.).max(1).astype(np.float64)
    flags={'A':a,'calibrated':a|(vv.any(1)&(score>=preds['thresholds'])),'fixed_zero':a|(vv.any(1)&(score>=0.)),'oracle_reference':a|witness}
    summary=read(out/'summary.json');results={};selected_witness={}
    for name,p in flags.items():assert not (a&~p).any() and np.array_equal(p[~vv.any(1)],a[~vv.any(1)])
    for c in sorted(set(cohort)):
        results[c]={};ix=cohort==c;clear=ix&(state!='boundary')
        for name,p in flags.items():
            got=summary['cohorts'][c][name]
            for tag,mask in [('strict',ix),('clear',clear),('boundary',ix&~clear)]:
                m=metric(truth[mask],p[mask]);results[c][name+'_'+tag]=m
                for key,value in m.items():assert got[tag][key]==value
            change=dict(FN_rescued=int(sum(clear&truth&~a&p)),FP_added=int(sum(clear&~truth&~a&p)),TP_lost=0,FP_removed=0)
            assert got['clear_changes']==change
            changes=[]
            for event in got['temporal']['events']:
                ii=[i for i,m in enumerate(mm) if m['cohort']==c and m['episode_id']==event['episode'] and state[i]=='positive' and event['start_s']<=m['time_s']<=event['end_last_sample_s']]
                hits=[i for i in ii if p[i]];old=[i for i in ii if a[i]]
                after=mm[hits[0]]['time_s']-event['start_s'] if hits else None
                before=mm[old[0]]['time_s']-event['start_s'] if old else None
                assert event['detected_in_core']==bool(hits) and event['first_in_core_delay_s']==after
                if old:assert hits and after<=before
                if after!=before:changes.append(dict(episode=event['episode'],A_first_s=before,branch_first_s=after))
            assert changes==got['first_alert_changes']
            added=ix&p&~a&truth
            assert got['added_true_frames_with_sampled_witness']==int(sum(added&witness))
            assert got['added_true_frames_without_sampled_witness']==int(sum(added&~witness))
        selected_witness[c]={}
        masked=np.where(vv,preds['logits'],-30.)
        for name in ['calibrated','fixed_zero']:
            added=ix&flags[name]&~a&truth
            records=[]
            for i in np.flatnonzero(added):
                j=int(masked[i].argmax());ties=np.flatnonzero(vv[i]&(masked[i]==masked[i,j]))
                records.append(dict(id=mm[i]['id'],stratum=state[i],max_slot=j,zone=j//2,return_slot=j%2,
                    logit=float(masked[i,j]),max_return_known=bool(kk[i,j]),max_return_sampled_witness=bool(kk[i,j] and tt[i,j]>0),
                    any_tied_max_sampled_witness=bool(((tt[i,ties]>0)&kk[i,ties]).any()),
                    frame_has_sampled_witness=bool(witness[i])))
            selected_witness[c][name]=dict(added_true_frames=len(records),max_return_witness=sum(r['max_return_sampled_witness'] for r in records),records=records)
    result=dict(status='PASS',arm=arm,checkpoints_independently_replayed=checkpoint_count,logit_parity='BITWISE',
        outer_inner_group_exclusion=True,fit_only_normalization=True,training_orders_exact=True,peak_eligible_fit_indices_verified=True,
        anchor_excluded_from_frame_loss=True,calibration_is_inner_OOF_report_is_distinct_final_refit=True,
        thresholds_exact=True,calibration_inactive_candidate_not_global_disable=True,each_report_frame_once=True,
        all_A_alerts_and_zero_return_fallback_retained=True,core_first_alert_verified=True,
        source_review='No native geometry/known mask/target/group enters forward inference; known labels enter return loss and fitting frame metadata determines eligible negative peak loss only.',
        paired_seed_order_fit_checks=paired_fit_count,selected_max_return_witness=selected_witness,
        folds=checks,metrics=results,prediction_seal_sha256=sha(out/'prediction-seal.json'))
    (out/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',arm=arm,checkpoints=checkpoint_count,clear={c:{n:results[c][n+'_clear'] for n in flags} for c in results}),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=['bce','peak']);audit(parser.parse_args().arm)
