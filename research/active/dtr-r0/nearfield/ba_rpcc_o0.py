"""Frozen rank-preserving cardinality oracle curve; no inference or training."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from ba_depth_probe import sha, write, rect, mixed_mask
from ba_zmr_probe import metric, derived

CS = (.90,.95,1.,1.025,1.05,1.075,1.10,1.15,1.20,1.25,1.30)

def select(depth, k):
    out=np.zeros(depth.size,bool)
    order=np.argsort(np.where(np.isfinite(depth),depth,np.inf),kind='stable')
    out[order[:max(0,min(int(k),depth.size))]]=True
    return out

def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text()); seal=json.loads((source/'prediction-seal.json').read_text())
    assert sha(source/'protocol.json')==seal['protocol_sha256']
    write(out/'protocol.json',dict(id='ba-rpcc-o0-20260918',source_seal=sha(source/'prediction-seal.json'),runner=sha(__file__),
        multipliers=list(CS),support='same reference-valid mixed/raw-known domain as 2x2 oracle, per nonoverlapping zone',
        ranking='frozen DEPTHOR continuous depth ascending, stable raster ties; only cardinality changes',
        k='ceil(c * GT near count) clipped to zone valid support; no model outputs or ToF values changed',
        recall_constraint='global baseline recall >= 95.6887% on primary domain; select highest IoU among curve points meeting it',
        zone_oracle='per-zone unconstrained IoU and per-zone recall-preserving IoU using baseline zone recall as minimum; descriptive ceiling, not trainable output',
        thresholds='2m primary; 1.5m and 3m secondary curves',
        closed='SCDE existence-only and mono-guided ZPA; no training'))
    by_c={c:[] for c in CS}; zone_rows=[]
    baseline_recall=224725/(224725+10125)
    for i,e in enumerate(protocol['selected']):
        path=source/'inputs'/e['filename']; assert sha(path)==e['sha256']
        with h5py.File(path) as f: gt,fr=np.array(f['depth']),np.array(f['fr'])
        fn=f'{i:03d}-depthor.npz'; assert sha(source/'predictions'/fn)==seal['files'][fn]
        pred=np.load(source/'predictions'/fn)['depth']
        valid=np.isfinite(gt)&(gt>.001)&(gt<10); mixed,_=mixed_mask(gt,fr,valid); domain=mixed&np.isfinite(pred)
        occupied=np.zeros(gt.shape,bool); zones=[]
        for j,box in enumerate(fr):
            s=rect(box,gt.shape); m=domain[s]
            if not m.any(): continue
            assert not occupied[s].any(); occupied[s]=True
            p=pred[s][m]; g=gt[s][m]
            y=g<2; order=np.argsort(p,kind='stable'); k0=int(y.sum()); current=int((p<2).sum())
            zone_cur=derived(metric(p<2,y,np.ones(y.shape,bool)))
            zone_candidates=[]
            for c in CS:
                k=int(np.ceil(c*k0)); q=np.zeros(len(p),bool);q[order[:min(k,len(q))]]=True
                z=derived(metric(q,y,np.ones(y.shape,bool)))
                zone_candidates.append((c,z))
                by_c[c].append(dict(filename=e['filename'],scene=e['filename'].split('/')[0],zone=j,**z))
            feasible=[z for c,z in zone_candidates if z['recall'] is not None and z['recall']>=zone_cur['recall']]
            best=max(zone_candidates,key=lambda cz: (cz[1]['iou'] if cz[1]['iou'] is not None else -1))[1]
            best_r=max(feasible,key=lambda z:(z['iou'] if z['iou'] is not None else -1)) if feasible else None
            zone_rows.append(dict(filename=e['filename'],scene=e['filename'].split('/')[0],zone=j,pixels=len(p),gt_k=k0,pred_k=current,
                baseline=zone_cur,unconstrained_best=best,recall_preserving_best=best_r,
                count_error=current-k0))
        assert not (domain&~occupied).any()
    curve=[]
    for c,rows in by_c.items():
        sums={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','tn')}
        curve.append(dict(c=c,**derived(sums),zones=len(rows)))
    def agg_zone(key):
        vals=[r[key] for r in zone_rows if r[key] is not None]; sums={k:sum(v[k] for v in vals) for k in ('tp','fp','fn','tn')}; return dict(**derived(sums),zones=len(vals))
    oracle_un=agg_zone('unconstrained_best'); oracle_r=agg_zone('recall_preserving_best')
    eligible=[r for r in curve if r['recall']>=baseline_recall]
    best=max(eligible,key=lambda r:r['iou']) if eligible else None
    secondary={}
    # Curves at 1.5m/3m are intentionally computed from saved depth once more.
    for threshold in (1.5,3.):
        vals=[]
        for c in CS:
            ss={'tp':0,'fp':0,'fn':0,'tn':0}
            for i,e in enumerate(protocol['selected']):
                pth=source/'inputs'/e['filename']
                with h5py.File(pth) as f:gt,fr=np.array(f['depth']),np.array(f['fr'])
                pred=np.load(source/'predictions'/f'{i:03d}-depthor.npz')['depth']; valid=np.isfinite(gt)&(gt>.001)&(gt<10);mixed,_=mixed_mask(gt,fr,valid);domain=mixed&np.isfinite(pred);occ=np.zeros(gt.shape,bool)
                for box in fr:
                    s=rect(box,gt.shape);m=domain[s]
                    if not m.any():continue
                    occ[s]=True;p=pred[s][m];g=gt[s][m];y=g<threshold;k=int(y.sum());q=select(p,int(np.ceil(c*k)));z=metric(q,y,np.ones(y.shape,bool))
                    for k2 in ss:ss[k2]+=z[k2]
            vals.append(dict(c=c,**derived(ss)))
        secondary[str(threshold)]=vals
    result=dict(curve=curve,baseline_recall=baseline_recall,best_at_recall=best,oracle_unconstrained=oracle_un,oracle_recall_preserving=oracle_r,secondary=secondary,
        decision='CARDINALITY_HEAD_WORTH_FURTHER_FROZEN_RECIPE_ONLY' if best and best['iou']-curve[2]['iou']>=.05 else 'CLOSE_RPCC_TRAINING_TRIGGER',
        caveats=['GT count uses reference-valid pixels and is privileged','curve uses one global multiplier, not learned per-zone features','oracle is descriptive ceiling, not a confirmation result','count and placement errors interact; gains are not additive physical causes'])
    write(out/'results.json',result);write(out/'curve-per-zone.json',by_c);write(out/'zone-oracles.json',zone_rows)
    print(json.dumps({k:result[k] for k in ('curve','best_at_recall','oracle_unconstrained','oracle_recall_preserving','decision')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
