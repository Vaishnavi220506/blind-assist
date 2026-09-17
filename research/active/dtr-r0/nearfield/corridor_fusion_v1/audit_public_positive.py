"""Independent frozen checkpoint/calibration/group audit; no model helpers."""
import json
import hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
HOME=ROOT/'artifacts.local/work/corridor-public-positive-20260917'
PREP=HOME/'preparation';OUT=HOME/'run-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,p):
    return dict(frames=len(y),TP=int(sum(y&p)),FP=int(sum(~y&p)),FN=int(sum(y&~p)),TN=int(sum(~y&~p)))

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    done=read(OUT/'completion.json');assert done['status']=='PASS'
    assert sha(OUT/'summary.json')==done['summary_sha256']
    assert sha(OUT/'prediction-seal.json')==done['prediction_seal_sha256']
    seal=read(OUT/'prediction-seal.json');freeze=read(OUT/'freeze.json');ds=read(PREP/'data-seal.json')
    assert sha(OUT/'freeze.json')==seal['freeze_sha256']
    assert sha(PREP/'data-seal.json')==freeze['data_seal_sha256']
    for p,h in ds['bindings'].items():assert sha(p)==h,p
    for p,h in freeze['sources'].items():assert sha(p)==h,p
    for name,h in ds['outputs'].items():assert sha(PREP/name)==h,name
    for name,h in (seal['models']|seal['outputs']).items():assert sha(OUT/name)==h,name
    data=np.load(PREP/'public-tokens.npz');lab=np.load(PREP/'offline-targets.npz')
    meta=read(PREP/'metadata.json');folds=read(PREP/'folds.json');summary=read(OUT/'summary.json');cases=read(OUT/'cases.json')
    assert len(meta)==768 and len({m['id'] for m in meta})==768
    assert Counter(m['cohort'] for m in meta)=={'anchor':192,'old':288,'new':288}
    assert list(data['ids'])==[m['id'] for m in meta]
    valid=data['valid'][:,:128];target=lab['target'][:,:128];known=lab['known'][:,:128]
    assert not lab['known'][:,128:].any() and not (known&~valid).any()
    # Independently reconstruct 30 public features in the inherited units.
    token=data['tokens'][:,:128];sc=np.array([4,2,3],np.float32)
    center=token[:,:,5:8]*sc;box=token[:,:,8:14].reshape(768,128,3,2)*sc[:,None]
    lo=np.array([.2,-.3,.4],np.float32);hi=np.array([3.6,.3,2.05],np.float32)
    x=np.concatenate([token,np.minimum(center-lo,hi-center),np.minimum(box[:,:,:,1]-lo,hi-box[:,:,:,0]),
                      np.minimum(box[:,:,:,0]-lo,hi-box[:,:,:,1])],-1)
    oof=np.load(OUT/'oof-predictions.npz');expected_report=[i for i,m in enumerate(meta) if m['cohort']!='anchor']
    assert list(oof['indices'])==expected_report
    report_seen=Counter();maxdiff=0.;calibration=[]
    device=freeze['device']
    for fold in folds:
        k=fold['fold'];fi,ci,ri=[np.array(fold[n]) for n in ['fit','calibration','report']]
        expected={
            'report':[i for i,m in enumerate(meta) if m['cohort']!='anchor' and m['scene_index']==k],
            'calibration':[i for i,m in enumerate(meta) if m['cohort']!='anchor' and m['scene_index']==(k+1)%6],
            'fit':[i for i,m in enumerate(meta) if m['cohort']=='anchor' or m['scene_index'] not in (k,(k+1)%6)]}
        for n,ii in [('fit',fi),('calibration',ci),('report',ri)]:assert list(ii)==expected[n]
        groups=[{meta[i]['group'] for i in ii} for ii in [fi,ci,ri]]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        orders=read(OUT/f'fold{k}-orders.json');assert len(orders)==120
        for order in orders:assert sorted(order)==sorted(fi.tolist())
        checkpoint=torch.load(OUT/f'fold{k}-model.pt',map_location=device,weights_only=False)
        state=checkpoint['state_dict'];assert checkpoint['epoch']==120 and checkpoint['seed']==183017+k
        fitvalues=x[fi][valid[fi]]
        np.testing.assert_array_equal(state['mean'].cpu().numpy(),fitvalues.mean(0))
        np.testing.assert_array_equal(state['scale'].cpu().numpy(),fitvalues.std(0).clip(.01))
        def infer(ii):
            chunks=[]
            with torch.inference_mode():
                for off in range(0,len(ii),64):
                    jj=ii[off:off+64];v=torch.as_tensor(valid[jj],device=device)
                    t=torch.as_tensor(x[jj],device=device)
                    t=(torch.where(v[...,None],t,state['mean'])-state['mean'])/state['scale']
                    for layer in [0,2,4]:
                        t=torch.nn.functional.linear(t,state[f'net.{layer}.weight'],state[f'net.{layer}.bias'])
                        if layer!=4:t=torch.relu(t)
                    chunks.append(t.squeeze(-1).masked_fill(~v,-30.).cpu().numpy())
            return np.concatenate(chunks)
        cal=np.load(OUT/f'fold{k}-calibration.npz');rp=np.load(OUT/f'fold{k}-report.npz');sel=read(OUT/f'fold{k}-selection.json')
        assert list(cal['indices'])==list(ci) and list(rp['indices'])==list(ri)
        for ii,z in [(ci,cal),(ri,rp)]:
            recomputed=infer(ii);maxdiff=max(maxdiff,float(abs(recomputed-z['logits']).max()))
            np.testing.assert_array_equal(recomputed,z['logits'])
        assert sel['fit_indices']==fi.tolist() and sel['calibration_indices']==ci.tolist() and sel['report_indices']==ri.tolist()
        assert sel['known_positive']==int((known[fi]&(target[fi]>0)).sum())
        assert sel['known_negative']==int((known[fi]&(target[fi]==0)).sum())
        assert sel['model_sha256']==sha(OUT/f'fold{k}-model.pt') and sel['calibration_sha256']==sha(OUT/f'fold{k}-calibration.npz')
        ca=np.array([meta[i]['A'] for i in ci]);cy=np.array([meta[i]['truth'] for i in ci]);clear=np.array([meta[i]['stratum']!='boundary' for i in ci])
        score=np.max(np.where(valid[ci],cal['logits'],-30.),axis=1).astype(np.float64);eligible=valid[ci].any(1)
        candidates=np.r_[np.nextafter(float(score.max()),np.inf),np.unique(score[eligible&clear])]
        assert candidates[0]>score.max() and not (score>=candidates[0]).any()
        curve=[]
        for th in candidates:
            p=(ca|(eligible&(score>=th)))[clear];m=metric(cy[clear],p)
            curve.append(dict(threshold=float(th),TP=m['TP'],FP=m['FP'],FN=m['FN'],f1=2*m['TP']/max(1,2*m['TP']+m['FP']+m['FN'])))
        chosen=max(curve,key=lambda r:(r['f1'],-r['FP'],r['threshold']))
        assert curve==sel['curve'] and chosen==sel['chosen']
        for j,i in enumerate(ri):
            q=expected_report.index(int(i));assert oof['folds'][q]==k
            assert oof['thresholds'][q]==chosen['threshold'];np.testing.assert_array_equal(oof['logits'][q],rp['logits'][j])
            report_seen[int(i)]+=1
        cal_witness=((target[ci]>0)&known[ci]).any(1)
        report_score=np.where(valid[ri],rp['logits'],-30.).max(1).astype(np.float64)
        calibration.append(dict(fold=k,threshold=chosen['threshold'],fit=576,calibration=96,report=96,
            selected_calibration_inactive_candidate=chosen['threshold']==float(candidates[0]),
            calibration_clear_A_FN=int(sum(clear&cy&~ca)),calibration_clear_supported_A_FN=int(sum(clear&cy&~ca&cal_witness)),
            report_branch_active_frames=int(sum(valid[ri].any(1)&(report_score>=chosen['threshold'])))))
    assert report_seen==Counter({i:1 for i in expected_report})
    mm=[meta[i] for i in expected_report];vv=valid[expected_report];yy=target[expected_report];kk=known[expected_report]
    y=np.array([m['truth'] for m in mm]);a=np.array([m['A'] for m in mm]);strata=np.array([m['stratum'] for m in mm]);cohort=np.array([m['cohort'] for m in mm])
    logits=oof['logits'];threshold=oof['thresholds'];scores=np.where(vv,logits,-30.).max(1).astype(np.float64)
    branch=vv.any(1)&(scores>=threshold);p=a|branch;witness=((yy>0)&kk).any(1);oracle=a|witness
    assert not (a&~p).any() and np.array_equal(p[~vv.any(1)],a[~vv.any(1)])
    for i,c in enumerate(cases):
        assert c['id']==mm[i]['id'] and c['public_OR']==bool(p[i]) and c['evidence_alert']==bool(branch[i])
        assert c['oracle_sampled_OR']==bool(oracle[i]) and c['sampled_witness']==bool(witness[i])
    results={}
    for c in ['old','new']:
        ix=cohort==c;clear=ix&(strata!='boundary');results[c]={}
        for name,flags in [('A',a),('A_public_OR',p),('A_oracle_sampled_OR',oracle)]:
            got=summary['cohorts'][c][name]
            for tag,mask in [('strict',ix),('clear',clear),('boundary',ix&~clear)]:
                m=metric(y[mask],flags[mask]);results[c][name+'_'+tag]=m
                for key,value in m.items():assert got[tag][key]==value
            changes=dict(FN_rescued=int(sum(clear&y&~a&flags)),FP_added=int(sum(clear&~y&~a&flags)),TP_lost=int(sum(clear&y&a&~flags)),FP_removed=int(sum(clear&~y&a&~flags)))
            assert changes==got['clear_changes']
            for event in got['temporal']['events']:
                ii=[i for i,m in enumerate(mm) if m['cohort']==c and m['episode_id']==event['episode'] and strata[i]=='positive' and event['start_s']<=m['time_s']<=event['end_last_sample_s']]
                hits=[i for i in ii if flags[i]]
                delay=mm[hits[0]]['time_s']-event['start_s'] if hits else None
                assert event['detected_in_core']==bool(hits) and event['first_in_core_delay_s']==delay
                if name!='A':
                    oldhits=[i for i in ii if a[i]]
                    if oldhits:assert hits and mm[hits[0]]['time_s']<=mm[oldhits[0]]['time_s']
        added=ix&p&~a;selected=vv&(logits>=threshold[:,None]);selected_native=(selected&kk&(yy>0)).any(1)
        results[c]['added_warning_support']=dict(added=int(sum(added)),with_any_native_sampled_witness=int(sum(added&witness)),
            with_selected_positive_slot_native_witness=int(sum(added&selected_native)),without_native_witness=int(sum(added&~witness)),
            clear_added=int(sum(added&clear)),clear_with_native_witness=int(sum(added&clear&witness)))
        assert summary['cohorts'][c]['zero_tof_frames']==int(sum(ix&~vv.any(1)))
        assert summary['cohorts'][c]['zero_tof_changed_alerts']==0
    zero=a|(vv.any(1)&(scores>=0.));zero_diagnostic={}
    for c in ['old','new']:
        ix=cohort==c;cc=ix&(strata!='boundary');added=ix&zero&~a
        selected=vv&(logits>=0.);selected_native=(selected&kk&(yy>0)).any(1)
        zero_diagnostic[c]={tag:metric(y[mask],zero[mask]) for tag,mask in [('strict',ix),('clear',cc),('boundary',ix&~cc)]}
        zero_diagnostic[c]['clear_changes']=dict(rescued_FN=int(sum(cc&y&~a&zero)),new_FP=int(sum(cc&~y&~a&zero)))
        zero_diagnostic[c]['added_native_support']=dict(added=int(sum(added)),any_sampled_witness=int(sum(added&witness)),
            selected_return_witness=int(sum(added&selected_native)))
    assert zero_diagnostic['old']['clear']==dict(frames=216,TP=108,FP=23,FN=0,TN=85)
    assert zero_diagnostic['new']['clear']==dict(frames=216,TP=103,FP=33,FN=5,TN=75)
    result=dict(status='PASS',fit_cal_report_group_disjoint=True,report_each_once=True,normalization_fit_only_bitwise=True,
        checkpoint_logits_bitwise_equal=True,max_logit_difference=maxdiff,calibration_final_OR_F1_exact=True,
        float64_calibration_only_inactive_threshold_verified=True,training_orders_fit_only=True,no_native_or_label_predictor_input=True,
        inactive_candidate_scope='Finite max(calibration score)+nextafter is inactive on calibration only; report may exceed it. Not a universal branch-disable flag. Sealed finite-threshold selection is reproduced exactly.',
        source_review='Predictor accepts only public features and validity; native target/known used in loss and scoring only; group controls partitions only.',
        A_retained=True,zero_return_fallback_verified=True,core_first_alert_verified=True,folds=calibration,results=results,
        posthoc_fixed_zero_logit_diagnostic=zero_diagnostic,
        zero_diagnostic_authority='One fixed standard logit0 replay after seeing outcomes; not selected or promoted, no sweep, not the preregistered calibrated result.',
        prediction_seal_sha256=sha(OUT/'prediction-seal.json'))
    (OUT/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
