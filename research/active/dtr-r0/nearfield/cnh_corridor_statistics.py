"""Paired cluster bootstrap of pooled balanced error, retaining one-class units."""
import numpy as np


def paired_cluster_ber(y,pred_base,pred_method,unitids,draws=5000,seed=20260929):
    """Bootstrap complete units, pool confusion counts, then subtract BERs.

    Units with only negatives/positives remain in every resampling population.
    A resample with an entirely absent class has undefined BER: it is counted
    and omitted from the percentile calculation, never redrawn or set to zero.
    point is method minus base on all supplied observations, not a mean of
    per-unit BERs. Call with a predeclared common subset for both predictions.
    """
    yy,bb,mm,uu=map(np.asarray,(y,pred_base,pred_method,unitids))
    if yy.ndim!=1 or any(a.shape!=yy.shape for a in (bb,mm,uu)):
        raise ValueError('all arrays must have the same one-dimensional shape')
    if any(not np.isin(a,[0,1]).all() for a in (yy,bb,mm)):
        raise ValueError('binary labels and predictions required')
    if not isinstance(draws,(int,np.integer)) or draws<1:raise ValueError('positive integer draws required')
    ids,index=np.unique(uu,return_inverse=True)
    n=len(ids)
    yy,bb,mm=yy.astype(bool),bb.astype(bool),mm.astype(bool)
    counts=np.zeros((n,6),dtype=np.int64)
    # P, N, FN_base, FP_base, FN_method, FP_method.
    for j,v in enumerate((yy,~yy,yy&~bb,~yy&bb,yy&~mm,~yy&mm)):
        np.add.at(counts[:,j],index,v.astype(np.int64))
    total=counts.sum(0)
    result=dict(point=None,ci=None,units=n,positive=int(total[0]),negative=int(total[1]),
        total_draws=int(draws),valid_draws=0,excluded_draws=int(draws),base_ber=None,method_ber=None)
    if not n or not total[0] or not total[1]:return result
    def ber(c,fn,fp):return .5*(c[...,fn]/c[...,0]+c[...,fp]/c[...,1])
    result['base_ber']=float(ber(total,2,3));result['method_ber']=float(ber(total,4,5))
    result['point']=result['method_ber']-result['base_ber']
    sampled=np.random.default_rng(seed).integers(0,n,size=(draws,n))
    pooled=counts[sampled].sum(1)
    valid=(pooled[:,0]>0)&(pooled[:,1]>0)
    result['valid_draws']=int(valid.sum());result['excluded_draws']=int((~valid).sum())
    if valid.any():
        c=pooled[valid]
        delta=ber(c,4,5)-ber(c,2,3)
        result['ci']=np.quantile(delta,[.025,.975]).tolist()
    return result
