"""Posthoc diagnostic of sealed BA-depth predictions; no inference or training."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from ba_depth_probe import sha, write, rect, mixed_mask, boundary_mask


def transitions(gt, raw, pred, domain, threshold=2.):
    near = domain & (gt < threshold)
    return dict(new_fn=near & (raw < threshold) & (pred >= threshold),
                rescued_fn=near & (raw >= threshold) & (pred < threshold),
                shared_fn=near & (raw >= threshold) & (pred >= threshold))


def main(source, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text())
    seal=json.loads((source/'prediction-seal.json').read_text())
    write(out/'protocol.json',dict(question='Does loss of zone-level measured near existence explain new FN?',
        type='POSTHOC_DIAGNOSTIC_CONSUMED_160',source_seal_sha256=sha(source/'prediction-seal.json'),
        runner_sha256=sha(__file__),threshold=2,epsilons=[.05,.1,.2],primary_epsilon=.1,
        tests='paired new/rescued/shared FN; valid-neighbor boundary bands 5/10px; GT>=1.9 and prediction<=2.1 separately',
        support='each valid near zone covering new FN: any prediction<2, any abs(pred-d)<=epsilon, Q10<=d+epsilon; full projected visible zone',
        caution='zone existence does not localize the echo; mask validity is not physical truth; Q10 one-sided is not range compatibility',
        training_condition='majority of new FN lose zone-level compatible support at epsilon .1, and not just boundary/threshold overlap; otherwise no automatic training',
        backend='CPU TASK_NOT_GPU_SUITABLE: saved-array diagnostic; no model'))
    rows=[];zones=[]
    for i,e in enumerate(protocol['selected']):
        p=source/'inputs'/e['filename'];assert sha(p)==e['sha256']
        with h5py.File(p) as f:gt,fr,hist,mask=[np.array(f[k]) for k in ('depth','fr','hist_data','mask')]
        arms={}
        for k in ('raw','depthor'):
            name=f'{i:03d}-{k}.npz';assert sha(source/'predictions'/name)==seal['files'][name]
            arms[k]=np.load(source/'predictions'/name)['depth']
        raw,pred=arms['raw'],arms['depthor']
        valid=np.isfinite(gt)&(gt>.001)&(gt<10)
        mixed,_=mixed_mask(gt,fr,valid)
        domain=mixed&np.isfinite(raw)
        masks=transitions(gt,raw,pred,domain)
        new=masks['new_fn'];band5=boundary_mask(gt,valid,5);band10=boundary_mask(gt,valid,10)
        threshold_band=(gt>=1.9)&(pred<=2.1)
        # Overlap-safe union: a pixel retains support if ANY covering valid near zone retains it.
        any_near=np.zeros(gt.shape,bool);compatible={ep:np.zeros(gt.shape,bool) for ep in (.05,.1,.2)}
        qpass=np.zeros(gt.shape,bool);covered=np.zeros(gt.shape,bool)
        for j,(box,hm,ok) in enumerate(zip(fr,hist,mask)):
            d=float(hm[0])
            if not ok or not np.isfinite(d) or not .001<d<2:continue
            s=rect(box,gt.shape);n=int(new[s].sum())
            if not n:continue
            pv=pred[s];vals=pv[np.isfinite(pv)&(pv>.001)]
            assert vals.size
            exists=bool((vals<2).any());quantile=float(np.quantile(vals,.1));qp=quantile<=d+.1
            covered[s]=True
            if exists:any_near[s]=True
            if qp:qpass[s]=True
            fractions={str(ep):float((np.abs(vals-d)<=ep).mean()) for ep in compatible}
            for ep in compatible:
                if fractions[str(ep)]>0:compatible[ep][s]=True
            zones.append(dict(filename=e['filename'],zone=j,range_m=d,new_fn_pixels=n,
                prediction_min=float(vals.min()),q10=quantile,q10_pass=qp,near_exists=exists,
                compatible_fraction=fractions,visible_rectangle=list(map(int,box))))
        assert not (new&~covered).any(), 'New raw-TP losses must have a near covering return'
        lost=new&~compatible[.1]
        row=dict(filename=e['filename'],scene=e['filename'].split('/')[0],
            **{k:int(v.sum()) for k,v in masks.items()},
            raw_fn=int((domain&(gt<2)&(raw>=2)).sum()),depthor_fn=int((domain&(gt<2)&(pred>=2)).sum()),
            boundary5=int((new&band5).sum()),boundary10=int((new&band10).sum()),
            threshold_both_10cm=int((new&threshold_band).sum()),
            gt_within_10cm_of_2m=int((new&(gt>=1.9)).sum()),
            pred_within_10cm_of_2m=int((new&(pred<=2.1)).sum()),
            zone_near_existence_lost=int((new&~any_near).sum()),
            q10_constraint_already_satisfied=int((new&qpass).sum()),
            compatible_support_lost={str(ep):int((new&~m).sum()) for ep,m in compatible.items()},
            lost_compatible_off_boundary_and_threshold=int((lost&~band10&~threshold_band).sum()))
        rows.append(row)
    totals={k:sum(r[k] for r in rows) for k,v in rows[0].items() if isinstance(v,int)}
    totals['compatible_support_lost']={str(ep):sum(r['compatible_support_lost'][str(ep)] for r in rows) for ep in (.05,.1,.2)}
    assert totals['new_fn']-totals['rescued_fn']==totals['depthor_fn']-totals['raw_fn']==3841
    assert totals['raw_fn']==6284 and totals['depthor_fn']==10125
    # The stated condition requires evidence loss, not merely the tautological raw-positive label.
    majority=totals['compatible_support_lost']['0.1']>totals['new_fn']/2
    robust=totals['lost_compatible_off_boundary_and_threshold']>totals['new_fn']/2
    summary=dict(totals=totals,frames_with_new=sum(r['new_fn']>0 for r in rows),
        zones_with_new=len(zones),zones_near_existence_lost=sum(not z['near_exists'] for z in zones),
        zones_q10_already_satisfied=sum(z['q10_pass'] for z in zones),
        majority_compatibility_loss=majority,majority_robust_loss=robust,
        decision='TRAINING_CONDITION_MET_REQUIRES_SEPARATE_FROZEN_RECIPE' if majority and robust else 'NO_AUTOMATIC_SCDE_TRAINING',
        limitations=['A/B/C are overlapping evidence axes, not exclusive causes',
          'All new FN already raw-near by definition; class B cannot occur in that subset',
          'A positive zone need not localize its return at the missing pixel',
          'No certified trusted-return label, exact body pose or contributor provenance',
          'Existence is weaker than the proposed 10-percent area constraint',
          'No claim of causal ToF benefit from distinct mono and fusion architectures'])
    write(out/'per-frame.json',rows);write(out/'zones.json',zones);write(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
