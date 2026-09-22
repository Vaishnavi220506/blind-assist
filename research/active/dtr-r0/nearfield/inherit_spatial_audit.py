"""Saved-output audit without fitting, threshold changes, or runner metrics.

Independent arithmetic reconstructs every declared cutoff, readout, stratum,
event and decision from sealed probabilities and the declared labels. This is
an implementation audit, not independent scientific evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def count(p, y, v, metas):
    tp = sum(bool(a and b and c) for a,b,c in zip(p,y,v))
    fp = sum(bool(a and not b and c) for a,b,c in zip(p,y,v))
    fn = sum(bool(not a and b and c) for a,b,c in zip(p,y,v))
    previous, segments = {}, 0
    for a,b,c,meta in zip(p,y,v,metas):
        active = bool(a and not b and c)
        segments += active and not previous.get(meta['clip_id'],False)
        previous[meta['clip_id']] = active
    return dict(TP=tp,FP=fp,FN=fn,false_segments=segments)


def events(p, y, v, metas):
    answer=[]
    for clip in sorted({r['clip_id'] for r in metas}):
        indices=sorted([i for i,m in enumerate(metas) if m['clip_id']==clip],key=lambda i:metas[i]['frame_in_clip'])
        j=0
        while j<len(indices):
            i=indices[j]
            if not(v[i] and y[i]):
                j+=1
                continue
            begin=j
            while j<len(indices) and v[indices[j]] and y[indices[j]]:
                j+=1
            active=indices[begin:j]
            first=next((metas[k]['time_s'] for k in active if p[k]),None)
            answer.append(dict(clip_id=clip,start_frame=metas[active[0]]['frame_in_clip'],
                               end_frame=metas[active[-1]]['frame_in_clip'],
                               detected=first is not None,first_alert_time_s=first))
    return answer


def run(args):
    root=args.run
    freeze=read(root/'freeze.json')
    features_seal=read(root/'feature-seal.json')
    model_seal=read(root/'model-seal.json')
    prediction_seal=read(root/'prediction-seal.json')
    result=read(root/'result.json')
    metrics=read(root/'metrics.json')
    rows=read(root/'frame-results.json')
    selections=read(root/'selection.json')
    checks=0
    def check(condition,description):
        nonlocal checks
        if not condition:
            raise AssertionError(description)
        checks+=1
    for name,value in freeze['sources'].items():
        check(digest(root/'source-snapshot'/name)==value,'snapshot '+name)
        check(digest(args.repo/name)==value,'current source '+name)
    check(digest(root/'freeze.json')==features_seal['freeze_sha256'],'freeze hash')
    check(digest(root/'features.npz')==features_seal['sha256'],'feature hash')
    check(digest(root/'feature-seal.json')==model_seal['features_sha256'],'feature seal')
    check(digest(root/'model-seal.json')==prediction_seal['models_sha256'],'model seal')
    check(digest(root/'selection.json')==model_seal['selection_sha256']==prediction_seal['selection_sha256'],'selection')
    for name,key in [('metrics.json','metrics_sha256'),('frame-results.json','frame_results_sha256'),
                     ('prediction-seal.json','prediction_seal_sha256'),('costs.json','costs_sha256')]:
        check(digest(root/name)==result[key],name)
    for arm in ('raw','local'):
        check(digest(root/(arm+'.pkl'))==model_seal['models'][arm],arm+' model')
        check(digest(root/(arm+'-predictions.npz'))==prediction_seal['predictions'][arm],arm+' prediction')
    check(not freeze['evaluation_labels_opened'] and not model_seal['evaluation_labels_opened']
          and not prediction_seal['evaluation_labels_opened'],'prediction-before-label contract')
    features=dict(np.load(root/'features.npz',allow_pickle=False))
    check(features['raw'].shape==features['local'].shape==(1728,6,961),'matched feature shape')
    check(np.array_equal(features['raw'][:,:,:910],features['local'][:,:,:910]),'identical raw carrier')
    check(not np.any(features['raw'][:,:,910:]),'raw padding')
    check(np.isfinite(features['raw']).all() and np.isfinite(features['local']).all(),'finite features')
    ids=read(args.identities)
    check(digest(args.identities)==freeze['input_hashes']['observations/identities.json'],'public metadata identity')
    by_split={s:[m for m in ids if m['split']==s] for s in ('train','dev','evaluation')}
    groups={s:{m['base_group_id'] for m in rows} for s,rows in by_split.items()}
    check([len(groups[s]) for s in groups]==[24,8,16],'group counts')
    check(not any(groups[a]&groups[b] for a,b in [('train','dev'),('train','evaluation'),('dev','evaluation')]),'group separation')
    dev=dict(np.load(args.dev,allow_pickle=False))
    evaluation=dict(np.load(args.evaluation,allow_pickle=False))
    check(digest(args.dev)==freeze['input_hashes']['labels/dev.npz'],'dev labels')
    check(digest(args.evaluation)==freeze['expected_evaluation_labels_sha256'],'evaluation labels')
    check(np.array_equal(dev['indices'],features_seal['indices']['dev']),'dev order')
    check(np.array_equal(evaluation['indices'],features_seal['indices']['evaluation']),'eval order')
    yd=(dev['classes'][:,[1,4]]<6).any(1); vd=dev['valid'][:,[1,4]].all(1)
    base=np.array([m['baseline']['alert'] for m in by_split['dev']],bool)
    b=count(base,yd,vd,by_split['dev'])
    for arm in ('raw','local'):
        scores=np.load(root/(arm+'-development-scores.npz'),allow_pickle=False)['dev'][:,[1,4]].max(1)
        expected=[math.nextafter(float(max(scores)),math.inf)]+sorted(set(scores.tolist()),reverse=True)
        curve=selections[arm]['curve']
        check(len(curve)==len(expected),'curve length')
        winners=[]
        for cut,record in zip(expected,curve):
            c=count(base|(scores>=cut),yd,vd,by_split['dev'])
            check(record['threshold']==cut,'atomic cutoff')
            for key,value in c.items():
                check(record[key]==value,'dev '+key)
            admissible=c['FP']-b['FP']<=1 and c['false_segments']-b['false_segments']<=1
            check(admissible==record['admissible'],'admission')
            if admissible:
                winners.append((c['TP']-b['TP'],-c['false_segments'],-c['FP'],cut))
        check(max(winners)[-1]==selections[arm]['selection']['threshold'],'selected cutoff')
    y=(evaluation['classes'][:,[1,4]]<6).any(1); v=evaluation['valid'][:,[1,4]].all(1)
    metas=by_split['evaluation']
    check(len(rows)==len(metas)==576,'all eval frames')
    flags={'A_current':np.array([m['baseline']['alert'] for m in metas],bool)}
    last={}; flags['A_hold']=[]
    for meta in metas:
        flags['A_hold'].append(meta['baseline']['alert'] or last.get(meta['clip_id'],False))
        last[meta['clip_id']]=meta['baseline']['alert']
    flags['A_hold']=np.array(flags['A_hold'],bool)
    for arm in ('raw','local'):
        pred=dict(np.load(root/(arm+'-predictions.npz'),allow_pickle=False))
        check(np.array_equal(pred['indices'],evaluation['indices']),'prediction indices')
        check(pred['probability'].shape==(576,6) and np.isfinite(pred['probability']).all(),'probability shape')
        flags[arm+'_standalone']=pred['probability'][:,[1,4]].max(1)>=selections[arm]['selection']['threshold']
        flags[arm]=flags['A_current']|flags[arm+'_standalone']
        check(not np.any(flags['A_current']&~flags[arm]),'A alert retention')
    for i,(r,m) in enumerate(zip(rows,metas)):
        check(r['id']==m['id'] and r['truth']==(bool(y[i]) if v[i] else None),'frame join')
        for arm, p in flags.items():
            check(r['predictions'][arm]['alert']==bool(p[i]),'readout')
            check(r['predictions'][arm]['unknown']==m['baseline']['unknown'],'UNKNOWN preserved')
    for arm,p in flags.items():
        c=count(p,y,v,metas)
        saved=metrics['metrics']['arms'][arm]
        for key in ('TP','FP','FN'):
            check(c[key]==saved['frames']['all_known'][key],'eval '+key)
        check(c['false_segments']==saved['false_alert_segment_count'],'false segment count')
        independent_events=events(p,y,v,metas)
        check(len(independent_events)==saved['event_count'],'event denominator')
        for a,b in zip(independent_events,saved['events']):
            for key in a:
                check(a[key]==b[key],'event '+key)
        for key in ('base_group_id','type_id','layer','layout_relation'):
            for value in {m[key] for m in metas}:
                subset=np.array([i for i,m in enumerate(metas) if m[key]==value])
                actual=count(p[subset],y[subset],v[subset],[metas[i] for i in subset])
                stored=metrics['strata'][key][str(value)]['arms'][arm]
                for k in ('TP','FP','FN'):
                    check(actual[k]==stored['frames']['all_known'][k],'stratum '+key+'/'+k)
                check(actual['false_segments']==stored['false_alert_segment_count'],'stratum segments')
    summaries={arm:count(p,y,v,metas) for arm,p in flags.items()}
    diagnostics={}
    for arm,reference in [('raw','A_current'),('local','A_current'),('local','raw')]:
        a,b=summaries[arm],summaries[reference]
        c=metrics['comparisons'][arm+'_vs_'+reference]
        recall=(a['TP']-b['TP'])/int((y&v).sum())
        check(math.isclose(recall,c['recall_delta'],abs_tol=1e-12),'recall delta')
        check(a['FP']-b['FP']==c['FP_delta'],'FP delta')
        check(a['false_segments']-b['false_segments']==c['false_segment_delta'],'segment delta')
        improvement_groups=0
        totals=[]
        for group in sorted(groups['evaluation']):
            ix=np.array([i for i,m in enumerate(metas) if m['base_group_id']==group])
            ca,cb=(int((flags[k][ix]&y[ix]&v[ix]).sum()) for k in (arm,reference))
            improvement_groups+=ca>cb
            totals.append([int((y[ix]&v[ix]).sum()),ca,cb])
        check(improvement_groups==c['improved_layouts'],'improved groups')
        rng=np.random.default_rng(202609224); totals=np.array(totals); boot=[]
        for _ in range(1000):
            t=totals[rng.integers(0,16,16)].sum(0); boot.append((t[1]-t[2])/max(1,t[0]))
        ci=np.quantile(boot,[.025,.975]).tolist()
        check(np.array_equal(ci,c['bootstrap']['recall_delta_95']),'layout bootstrap')
        cost=a['FP']-b['FP']<=2 and a['false_segments']-b['false_segments']<=1
        if reference=='A_current':
            passed=recall>=.10-1e-12 and cost and not np.any(flags[reference]&y&v&~flags[arm])
        else:
            passed=recall>=.05-1e-12 and cost and improvement_groups>=4 and ci[0]>0
            for layer in {m['layer'] for m in metas}:
                ix=np.array([i for i,m in enumerate(metas) if m['layer']==layer])
                difference=int((flags[arm][ix]&y[ix]&v[ix]).sum())-int((flags[reference][ix]&y[ix]&v[ix]).sum())
                passed=passed and difference/max(1,int((y[ix]&v[ix]).sum()))>=-.05-1e-12
        check(bool(passed)==c['gate_met'],'terminal gate')
        diagnostics[arm+'_vs_'+reference]=dict(gate_met=bool(passed),recall_delta=recall,
            FP_delta=a['FP']-b['FP'],segment_delta=a['false_segments']-b['false_segments'],
            improved_layouts=improvement_groups,bootstrap95=ci)
    answer=dict(status='PASS',checks=checks,frames=576,queries=3456,summary=summaries,decisions=diagnostics,
                audit='Independent saved-output arithmetic; no fit, feature/threshold change or new scientific sample',
                input_prediction_seal_sha256=digest(root/'prediction-seal.json'),
                scoped_limits='Does not independently reconstruct every RGB statistic or rendering; synthetic feature/geometry tests cover implementation boundaries')
    args.result.parent.mkdir(parents=True,exist_ok=True)
    with args.result.open('x',encoding='utf-8') as stream:
        json.dump(answer,stream,indent=2,allow_nan=False)
    print(json.dumps(answer),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','repo','identities','dev','evaluation','result'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
