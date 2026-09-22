"""Inference-only diagnostics of frozen selected checkpoints on train/dev."""
from __future__ import annotations
import argparse
import itertools
import os
from pathlib import Path
import sys
import time

import numpy as np

from query_occupancy_data import read, write, sha, stage_path, new_stage_directory, CENTRE, QUERIES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO/'artifacts.local/evidence/ba-query-occupancy-20260922'
DEST = REPO/'artifacts.local/evidence/ba-query-occupancy-diagnostic-20260922'


def rank_metrics(score, positive, negative):
    """Weighted ROC AUC and step AP, with entire equal-score blocks atomic.

    A pooled pixel can contain both foreground and background area. Weights
    represent native area, not binary labels assigned to a mixed cell.
    """
    s,p,n = [np.asarray(v,np.float64).ravel() for v in (score,positive,negative)]
    assert s.shape==p.shape==n.shape
    assert np.isfinite(s).all() and np.isfinite(p).all() and np.isfinite(n).all()
    assert (p>=0).all() and (n>=0).all()
    total_p,total_n = p.sum(),n.sum()
    prevalence = float(total_p/(total_p+total_n)) if total_p+total_n else None
    if total_p==0 or total_n==0:
        return dict(auc=None,ap=None,prevalence=prevalence)
    observed=(p+n)>0
    s,p,n=s[observed],p[observed],n[observed]
    order=np.argsort(-s,kind='stable'); s,p,n=s[order],p[order],n[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    pg,ng=np.add.reduceat(p,starts),np.add.reduceat(n,starts)
    cp,cn=np.cumsum(pg),np.cumsum(ng)
    auc=np.sum(pg*(total_n-cn+.5*ng))/(total_p*total_n)
    ap=np.sum(pg/total_p*cp/(cp+cn))
    return dict(auc=float(auc),ap=float(ap),prevalence=prevalence)


def summarize(values):
    a=np.asarray(values,np.float64)
    a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
                p50=float(np.median(a)) if len(a) else None,
                p05=float(np.quantile(a,.05)) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None)


def pair_summary(margins):
    a=np.asarray(margins,np.float64)
    return dict(**summarize(a), wins=int((a>0).sum()),ties=int((a==0).sum()),losses=int((a<0).sum()),
                win_rate=float(((a>0)+.5*(a==0)).mean()) if len(a) else None)


def query_pairs(score,classes,valid):
    rows=[]
    for i in range(len(score)):
        for a,b in itertools.combinations(range(6),2):
            if not(valid[i,a] and valid[i,b]) or (classes[i,a]<6)==(classes[i,b]<6):
                continue
            pos,neg=(a,b) if classes[i,a]<6 else (b,a)
            rows.append(dict(row=i,positive_query=pos,negative_query=neg,
                kind='lateral' if a//3==b//3 else 'height',margin=float(score[i,pos])-float(score[i,neg])))
    return rows


def split_metrics(pred,labels,identities,mode,threshold):
    p=np.asarray(pred['probability'],np.float64); classes=labels['classes']; valid=labels['valid']
    y=classes<6; fy=y[:,CENTRE].any(1); fv=valid[:,CENTRE].all(1); fs=p[:,CENTRE].max(1)
    alert=fs>=float(threshold)
    tp,fp,fn=[int(a.sum()) for a in (alert&fy&fv,alert&~fy&fv,~alert&fy&fv)]
    clipped=np.clip(p,1e-12,1-1e-12)
    bce=-(y*np.log(clipped)+(~y)*np.log(1-clipped))
    pairs=query_pairs(p,classes,valid)
    relation_pairs=[]
    lookup={(m['base_group_id'],m['frame_in_clip'],m['layout_relation']):i for i,m in enumerate(identities)}
    for i,m in enumerate(identities):
        if m['layout_relation']=='OUTSIDE' or not(fv[i] and fy[i]): continue
        j=lookup[(m['base_group_id'],m['frame_in_clip'],'OUTSIDE')]
        if fv[j] and not fy[j]:
            relation_pairs.append(dict(row=i,outside_row=j,relation=m['layout_relation'],margin=float(fs[i]-fs[j])))
    result=dict(frames=len(p),known_frames=int(fv.sum()),positive_frames=int((fy&fv).sum()),
        negative_frames=int((~fy&fv).sum()),unknown_sensor_frames=sum(bool(m['baseline']['unknown']) for m in identities),
        fixed_cutoff=dict(threshold=threshold,TP=tp,FP=fp,FN=fn,recall=tp/(tp+fn) if tp+fn else None,
            fpr=fp/int((~fy&fv).sum()) if (~fy&fv).any() else None),
        frame_rank=rank_metrics(fs[fv],fy[fv],~fy[fv]),
        query_rank=rank_metrics(p[valid],y[valid],~y[valid]),query_bce=float(bce[valid].mean()),
        positive_probability=summarize(p[valid&y]),negative_probability=summarize(p[valid&~y]),
        across_query_probability_spread=summarize(np.ptp(p,axis=1)),
        lateral_probability_spread=summarize(np.ptp(p.reshape(-1,2,3),axis=-1).ravel()),
        same_image_query_order={k:pair_summary([r['margin'] for r in pairs if k=='all' or r['kind']==k]) for k in ('all','lateral','height')},
        paired_corridor_order={k:pair_summary([r['margin'] for r in relation_pairs if r['relation']==k]) for k in ('INSIDE','BOUNDARY')})
    spatial_rows=[]
    if mode=='occupancy':
        d=pred['distribution']; cls=d.argmax(-1); pos=valid&y
        ce=-np.log(np.clip(np.take_along_axis(d,classes[...,None],axis=-1)[...,0],1e-12,1))
        result['bin']=dict(queries=int(valid.sum()),positive_queries=int(pos.sum()),
            accuracy=float((cls[valid]==classes[valid]).mean()),positive_accuracy=float((cls[pos]==classes[pos]).mean()),
            always_none_accuracy=float((~y[valid]).mean()),positive_argmax_none=int((cls[pos]==6).sum()),
            occupied_conditional_bin_accuracy=float((d[...,:6].argmax(-1)[pos]==classes[pos]).mean()))
        masks=pred['mask']; target=labels['mask']; cover=labels['coverage']
        cov=np.broadcast_to(cover[:,None],target.shape).astype(np.float64)
        mp=np.clip(masks.astype(np.float64),1e-12,1-1e-12)
        soft_target=np.divide(target,cov,out=np.zeros_like(mp),where=cov>0)
        weights=cov*valid[:,:,None,None]
        mask_bce=-(soft_target*np.log(mp)+(1-soft_target)*np.log(1-mp))
        dice=1-(2*(mp*target).sum((-1,-2))+1)/((mp*cov).sum((-1,-2))+target.sum((-1,-2))+1)
        result['eval_loss_components']=dict(ce=float(ce[valid].mean()),
            mask_bce=float((mask_bce*weights).sum()/weights.sum()),dice=float(dice[valid].mean()))
        result['eval_loss_components']['total']=sum(result['eval_loss_components'].values())
        result['geometric_positive_without_visible_pixels']=int((pos&(target.sum((-1,-2))==0)).sum())
        result['mask_global']=dict(max_probability=float(masks.max()),pixels_above_original_cutoff=int((masks>=.5).sum()),
            foreground_area_fraction=float(target[valid].sum()/np.broadcast_to(cover[:,None],target.shape)[valid].sum()))
        map_changes=[]
        for i,a,b in ((i,a,b) for i in range(len(p)) for a,b in itertools.combinations(range(6),2)):
            map_changes.append(float(np.abs(masks[i,a]-masks[i,b]).mean()))
        result['map_mean_absolute_query_change']=summarize(map_changes)
        for i in range(len(p)):
            for q in range(6):
                if not valid[i,q] or target[i,q].sum()==0: continue
                score=masks[i,q].astype(np.float64); fg=target[i,q].astype(np.float64)
                bg=cover[i].astype(np.float64)-fg
                ranks=rank_metrics(score,fg,bg)
                binary=score>=.5; intersection=(binary*fg).sum(); union=(binary*cover[i]+fg-binary*fg).sum()
                peaks=(score==score[cover[i]>0].max())&(cover[i]>0)
                wrong=[rank_metrics(masks[i,r],fg,bg) for r in range(6) if valid[i,r] and not y[i,r]]
                row=dict(row=i,query=q,**ranks,geometric_positive=bool(y[i,q]),
                    fg_mean=float((score*fg).sum()/fg.sum()),bg_mean=float((score*bg).sum()/bg.sum()) if bg.sum() else None,
                    fixed_iou=float(intersection/union),peak_occupied_fraction=float(fg[peaks].sum()/cover[i][peaks].sum()),
                    soft_iou=float((score*fg).sum()/(score*cover[i]+fg-score*fg).sum()),
                    label_has_majority_foreground_cell=bool((fg>.5*cover[i]).any()),
                    wrong_query_count=len(wrong),wrong_query_mean_ap=float(np.mean([x['ap'] for x in wrong])) if wrong else None)
                spatial_rows.append(row)
        result['visible_positive_ranking']={key:summarize([r[key] for r in spatial_rows if r[key] is not None])
            for key in ('auc','ap','prevalence','fg_mean','bg_mean','fixed_iou','soft_iou','peak_occupied_fraction')}
        result['centre_visible_positive_ranking']={key:summarize([r[key] for r in spatial_rows if r['query'] in CENTRE and r[key] is not None])
            for key in ('auc','ap','prevalence','fixed_iou','soft_iou','peak_occupied_fraction')}
        result['visible_positive_queries_with_majority_foreground_cell']=sum(r['label_has_majority_foreground_cell'] for r in spatial_rows)
        result['correct_minus_wrong_query_ap']=summarize([r['ap']-r['wrong_query_mean_ap'] for r in spatial_rows if r['wrong_query_mean_ap'] is not None])
    return result,dict(query_pairs=pairs,relation_pairs=relation_pairs,spatial_queries=spatial_rows)


def make_spec():
    inputs=[dict(alias='observations',path=str(stage_path(SOURCE,'prepared')/'observations'),role='observation',purpose='train-dev-public-input-only'),
            dict(alias='weights',path=str(stage_path(SOURCE,'fit')),role='configuration',purpose='frozen-selected-checkpoints-and-seals'),
            dict(alias='manifest',path=str(stage_path(SOURCE,'prepared')/'materialization.json'),role='configuration',purpose='verify-input-identity')]
    inputs += [dict(alias=s,path=str(stage_path(SOURCE,'prepared')/f'labels/{s}.npz'),role='evaluator',purpose='consumed-'+s+'-diagnostic') for s in ('train','dev')]
    spec=dict(schema='blindassist-asset-run-v1',id='query-occupancy-diagnostic-20260922',route='ue-query-occupancy',
        question='Did selected checkpoints fit training queries, rank visible foreground, and respond correctly to query changes?',
        evaluator='research/active/dtr-r0/nearfield/diagnose_query_occupancy.py',
        evidence_boundary='Consumed train/dev diagnostic; no evaluation arrays, training, calibration or new operating point',
        reuse=dict(mode='development',query='Existing query occupancy train dev frozen checkpoint diagnostic'),
        inputs=inputs,outputs=[dict(alias='result',path=str(DEST/'result.json'),role='result',required=True)],result_output='result',
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'run'])
    path=DEST.with_name(DEST.name+'-run.json');write(path,spec);print(path)


def run():
    import torch
    from query_occupancy_learning import configure,load_inputs,predict,choose_backend
    from query_occupancy_model import QueryOccupancyNet
    journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state']=='running', 'Use governed research-ue runner'
    new_stage_directory(DEST); configure(); started=time.perf_counter()
    fit=stage_path(SOURCE,'fit'); prepared=stage_path(SOURCE,'prepared')
    source=read(fit/'learning-source-seal.json'); selection=read(fit/'selection.json')
    input_hashes={str(fit/'selection.json'):sha(fit/'selection.json'),str(fit/'learning-source-seal.json'):sha(fit/'learning-source-seal.json')}
    for name in ('query_occupancy_data.py','query_occupancy_learning.py','query_occupancy_model.py','../../../../tools/research_backend.py'):
        assert sha(HERE/name)==source['code'][name]
    for name,digest in source['used_inputs'].items():
        assert sha(prepared/name)==digest
        input_hashes[str(prepared/name)]=digest
    code={name:sha(HERE/name) for name in ('diagnose_query_occupancy.py','QUERY_OCCUPANCY_DIAGNOSTIC_20260922.md')}
    write(DEST/'input-seal.json',dict(code=code,input_hashes=input_hashes,source_code=source['code'],held_arrays_accessed=False))
    rgb,tof,ids=load_inputs(SOURCE)
    indices={s:np.array([i for i,r in enumerate(ids) if r['split']==s]) for s in ('train','dev')}
    pred_seal={}
    for mode in ('classifier','occupancy'):
        path=fit/f'{mode}.pt';assert sha(path)==selection[mode]['checkpoint_sha256']
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=QueryOccupancyNet(None,mode=mode,n_bins=6);model.load_state_dict(checkpoint['state_dict'])
        device=choose_backend(model,rgb,tof,indices['train'],torch.as_tensor(QUERIES),DEST/f'{mode}-backend.json')
        model.to(device).eval(); queries=torch.as_tensor(QUERIES,device=device)
        for s in ('train','dev'):
            pred=predict(model,rgb,tof,indices[s],queries,device,mode,masks=True)
            p=DEST/f'{mode}-{s}.npz';np.savez_compressed(p,indices=indices[s],**pred)
            pred_seal[mode+'-'+s]=dict(sha256=sha(p),checkpoint_sha256=sha(path),device=device,elapsed_s=pred['elapsed_s'])
            print('PREDICT',mode,s,len(indices[s]),flush=True)
        del model
    write(DEST/'prediction-seal.json',dict(predictions=pred_seal,labels_loaded=False,held_arrays_accessed=False))
    report={};detail={}
    for s in ('train','dev'):
        labels=dict(np.load(prepared/f'labels/{s}.npz',allow_pickle=False));assert np.array_equal(labels['indices'],indices[s])
        meta=[ids[int(i)] for i in indices[s]]
        report[s]={};detail[s]={}
        for mode in ('classifier','occupancy'):
            pred=dict(np.load(DEST/f'{mode}-{s}.npz',allow_pickle=False))
            report[s][mode],detail[s][mode]=split_metrics(pred,labels,meta,mode,selection[mode]['selection']['threshold'])
            if s=='dev':
                for k in ('TP','FP','FN'):
                    assert report[s][mode]['fixed_cutoff'][k]==selection[mode]['selection'][k], 'Dev fixed-cutoff replay mismatch'
            print('METRICS',s,mode,flush=True)
    for path,digest in input_hashes.items(): assert sha(path)==digest
    for mode in ('classifier','occupancy'): assert sha(fit/f'{mode}.pt')==selection[mode]['checkpoint_sha256']
    result=dict(status='PASS',splits=report,held_arrays_accessed=False,training_performed=False,
        inputs_unchanged=True,selected_epochs={a:v['epoch'] for a,v in selection.items()},elapsed_s=time.perf_counter()-started,
        limits=['Only selected states, not last epoch', 'Consumed train/dev from one generator',
                'No causal attribution to loss imbalance, capacity or normalization without a controlled contrast',
                'Query pairs and pixels correlated; descriptive counts, not independent trials'])
    write(DEST/'details.json',detail);write(DEST/'result.json',result);print('PASS',DEST)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=['spec','run'])
    args=parser.parse_args();make_spec() if args.command=='spec' else run()
