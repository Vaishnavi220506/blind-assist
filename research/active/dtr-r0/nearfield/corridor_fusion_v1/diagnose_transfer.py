"""Authenticated descriptive A transfer diagnosis; no fitting or selection."""
import csv
import json
import pickle
from pathlib import Path
import time
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
from run_e1 import ROOT,BASE,read,write,sha,dataset,labels
from mz143_corridor_features import extract

WORK=ROOT/'artifacts.local/work/corridor-intrusion-20260917/diagnosis'
OLD=ROOT/'artifacts.local/work/mz170-mean-confirmation-20260916/source/returned-v1/capture-v1'
NEW=ROOT/'artifacts.local/work/corridor-depth-confirmation-recovery-20260917/source/returned-v1/capture-v1'
E1=ROOT/'artifacts.local/work/corridor-depth-e1-20260917/run-v2'
S1=ROOT/'artifacts.local/work/corridor-depth-specialist-20260917'
CONF=ROOT/'artifacts.local/work/corridor-depth-confirmation-20260917/evaluation-v1'
THRESHOLD=.3917890013717321


def metrics(rr):
    tp=sum(r['truth'] and r['A_alert'] for r in rr);fp=sum(not r['truth'] and r['A_alert'] for r in rr)
    fn=sum(r['truth'] and not r['A_alert'] for r in rr)
    return dict(frames=len(rr),positive=sum(r['truth'] for r in rr),TP=tp,FP=fp,FN=fn,
        precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))


def run():
    started=time.perf_counter();WORK.mkdir(parents=True,exist_ok=True)
    inputs={}
    def check(p,h=None):
        actual=sha(p);assert h is None or actual==h,str(p);inputs[str(p)]=actual
    check(E1/'report-features.npz',read(E1/'feature-seal.json')['report'])
    check(S1/'report-predictions.npz',read(S1/'prediction-seal.json')['sha256'])
    done=read(CONF/'completion.json');assert done['status']=='PASS'
    check(CONF/'prediction-seal.json',done['prediction_seal_sha256'])
    seal=read(CONF/'prediction-seal.json');check(CONF/'predictions.json',seal['predictions_sha256'])
    check(CONF/'input-seal.json',seal['input_seal_sha256'])
    for p,h in read(CONF/'input-seal.json')['inputs'].items():check(p,h)
    oldbase=np.load(E1/'report-features.npz');oldpred=np.load(S1/'report-predictions.npz');newpred=read(CONF/'predictions.json')
    modelpath=E1/'A-model.pkl';check(modelpath,read(E1/'selection-seal.json')['models']['A']);model=pickle.loads(modelpath.read_bytes())
    names=read(BASE/'feature-names.json');names=names['sensor_names']+names['geometry_names']
    chosen=[n for n in ('nearest.count','nearest.range_m.q100','nearest.side_overlap_m.q50',
        'nearest.height_overlap_m.q50','nearest.width_m.q50','forward_height.side_overlap_m.q50') if n in names]
    cohorts={k:dataset(p,'confirmation') for k,p in [('old',OLD),('new',NEW)]};bases={'old':oldbase['base']}
    assert list(oldbase['ids'])==[r['id'] for r in cohorts['old']['rows']]
    yaw=0.;episode=None;values=[]
    cached=WORK/'new-base-features.npz'
    if cached.exists():
        check(cached,read(WORK/'new-base-feature-seal.json')['feature_sha256'])
        assert list(np.load(cached)['ids'])==[r['id'] for r in cohorts['new']['rows']]
        values=list(np.load(cached)['base'])
    for r in ([] if values else cohorts['new']['rows']):
        if r['episode_id']!=episode:yaw=0.
        if r['imu_valid']:yaw+=r['delta_yaw']
        episode=r['episode_id'];path=NEW/r['rgb_path'];check(path,cohorts['new']['receipt']['hashes'][r['rgb_path']])
        x=extract(r,cv2.imread(str(path)),yaw);values.append(np.r_[x['sensor'],x['geometry']])
    bases['new']=np.stack(values)
    for k,d in cohorts.items():
        for n in ('spec.json','receipt.json','raw.jsonl','evaluator.jsonl'):check(d['cap']/n)
        score=model.predict_proba(bases[k])[:,1]
        previous=oldpred['A'] if k=='old' else np.array([p['A_score'] for p in newpred])
        np.testing.assert_array_equal(score,previous)
    np.savez_compressed(WORK/'new-base-features.npz',base=bases['new'],ids=[r['id'] for r in cohorts['new']['rows']])
    write(WORK/'new-base-feature-seal.json',dict(feature_sha256=sha(WORK/'new-base-features.npz'),
        capture=str(NEW),frames=288,features=2485,score_parity='BITWISE_EQUAL_FROZEN_CONFIRMATION_A',
        source_sha256=sha(Path(__file__)),base_extract_sha256=sha(Path(__file__).parent.parent/'mz143_corridor_features.py')))
    rows=[]
    for cohort,d in cohorts.items():
        es,y=labels(d);frames={f['id']:f for f in d['spec']['frames']};groups={g['scene_group']:g for g in d['spec']['scene_groups']}
        scores=model.predict_proba(bases[cohort])[:,1]
        for i,(r,e) in enumerate(zip(d['rows'],es)):
            f=frames[r['id']];group=groups[f['scene_group']];target=f['objects'][0]
            targets=[t for z in r['tof_zones'] for t in z['targets'] if r['tof_packet_received'] and t['status'] in ('SIM_VALID','SIM_MERGED')]
            def median(key):return float(np.median([t[key] for t in targets])) if targets else None
            native=np.array([o['center_m'] for o in e['native_bounds']])-np.array([o['extent_m'] for o in e['native_bounds']])-np.array(e['body_origin_m'])
            rec=dict(cohort=cohort,id=r['id'],family=e['family'],truth=bool(y[i]),A_score=float(scores[i]),A_alert=bool(scores[i]>=THRESHOLD),
                wall_distance_m=f.get('wall_distance_m',None),background_style=f.get('background_style','legacy_far_background'),
                foreground_width_m=target['size_m'][1],band='HEAD' if e['family']=='suspended_head' else 'BODY',
                target_count=len(targets),dual_return_zones=sum(len(z['targets'])==2 for z in r['tof_zones']) if r['tof_packet_received'] else 0,
                signal=median('signal_strength_proxy'),sigma=median('range_noise_sigma_m'),
                packet_received=bool(r['tof_packet_received']))
            rec['width_bin']='<=.08' if rec['foreground_width_m']<=.08 else '.08-.5' if rec['foreground_width_m']<=.5 else '>.5'
            for n in chosen:rec[n]=float(bases[cohort][i,names.index(n)])
            for n in chosen:
                if not n.endswith('.count'):
                    count_name=n.split('.')[0]+'.count'
                    if count_name in names and bases[cohort][i,names.index(count_name)]<=0:rec[n]=None
            rec['usable_returns_bin']='0' if not targets else '1-16' if len(targets)<=16 else '>16'
            rec['dual_returns_bin']='0' if rec['dual_return_zones']==0 else '>=1'
            rec['side_overlap_bin']='missing' if rec.get('nearest.side_overlap_m.q50') is None else 'positive' if rec['nearest.side_overlap_m.q50']>0 else 'nonpositive'
            rows.append(rec)
    with (WORK/'frame-diagnosis.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    strata={};distributions={}
    numeric=['A_score','foreground_width_m','target_count','dual_return_zones','signal','sigma']+chosen
    for cohort in cohorts:
        rr=[r for r in rows if r['cohort']==cohort];strata[cohort]={'all':metrics(rr)}
        for field in ('family','band','wall_distance_m','background_style','width_bin','usable_returns_bin','dual_returns_bin','side_overlap_bin'):
            strata[cohort][field]={str(v):metrics([r for r in rr if r[field]==v]) for v in sorted({r[field] for r in rr},key=str)}
        distributions[cohort]={}
        for n in numeric:
            valid=[r[n] for r in rr if r[n] is not None and np.isfinite(r[n])]
            distributions[cohort][n]=dict(total=len(rr),available=len(valid),missing=len(rr)-len(valid),
                quantiles=dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(valid,[0,.25,.5,.75,1]))))) if valid else dict(total=len(rr),available=0,missing=len(rr))
    write(WORK/'strata.json',strata);write(WORK/'distributions.json',distributions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);colors={'old':'#246b9e','new':'#cf673a'}
    for k in cohorts:
        for gt,ls in [(True,'-'),(False,'--')]:
            values=sorted(r['A_score'] for r in rows if r['cohort']==k and r['truth']==gt)
            axes[0,0].plot(values,np.arange(1,len(values)+1)/len(values),ls,color=colors[k],label=f'{k} '+('positive' if gt else 'negative'))
    axes[0,0].axvline(THRESHOLD,color='black',alpha=.4);axes[0,0].set(title='Frozen A score ECDF (144 per class/cohort)',xlabel='A probability',ylabel='Cumulative fraction');axes[0,0].legend(fontsize=8)
    for ax,field,title in [(axes[0,1],'nearest.range_m.q100','Observable nearest range q100'),(axes[1,0],'target_count','Public usable ToF return count')]:
        for k in cohorts:
            val=[r[field] for r in rows if r['cohort']==k and r[field] is not None];ax.hist(val,bins=20,histtype='step',linewidth=2,color=colors[k],label=f'{k} n={len(val)}')
        ax.set(title=title,ylabel='Frames (N=288/cohort)');ax.legend()
    families=list(strata['old']['family']);x=np.arange(len(families))
    for j,k in enumerate(cohorts):axes[1,1].bar(x+(j-.5)*.35,[strata[k]['family'][f]['f1'] for f in families],.35,label=k,color=colors[k])
    axes[1,1].set(title='A family F1 (72 frames/family/cohort)',ylim=(0,1),xticks=x,xticklabels=['rod','boundary','body','head'] if families==['near_rod_farwall','shallow_boundary_stress','substantial_body','suspended_head'] else families);axes[1,1].tick_params(axis='x',labelsize=8);axes[1,1].legend()
    fig.suptitle('A transfer: consumed controlled cohorts; descriptive association, not causal attribution',fontsize=13)
    fig.savefig(WORK/'transfer-shift.png',dpi=180);fig.savefig(WORK/'transfer-shift.svg');plt.close(fig)
    write(WORK/'input-seal.json',inputs)
    write(WORK/'completion.json',dict(status='PASS',frames=576,no_fit=True,no_threshold_changes=True,
        old_new_score_parity=True,seconds=time.perf_counter()-started,cache_sha256=sha(WORK/'new-base-features.npz')))
    print(json.dumps(dict(metrics={k:v['all'] for k,v in strata.items()},distributions=distributions),indent=2))


if __name__=='__main__':
    cv2.setNumThreads(4)
    with threadpool_limits(4):run()
