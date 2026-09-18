"""Common-support 2x2 diagnostic and joint threshold bands; no model execution."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from ba_depth_probe import sha,write,rect,mixed_mask
from ba_zmr_probe import metric,derived


def arms_1d(pred,gt):
    y=gt<2;k=int((pred<2).sum());g=int(y.sum())
    order=np.argsort(pred,kind='stable')
    mass=np.zeros(len(y),bool);mass[order[:g]]=True
    perfect=np.zeros(len(y),bool)
    preferred=np.concatenate([np.flatnonzero(y),np.flatnonzero(~y)])
    perfect[preferred[:k]]=True
    assert perfect.sum()==k and mass.sum()==g
    return dict(current=pred<2,mass_only=mass,placement_only=perfect,both=y)


def bands(gt,pred,mask):
    # Nonoverlapping absolute-distance bins on BOTH axes, no threshold retuning.
    cuts=np.array([.05,.10,.20])+1e-6  # Stored float32 decimal endpoints.
    g=np.searchsorted(cuts,np.abs(gt[mask].astype(np.float64)-2),side='left')
    p=np.searchsorted(cuts,np.abs(pred[mask].astype(np.float64)-2),side='left')
    return np.bincount(4*g+p,minlength=16).reshape(4,4)


def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());seal=json.loads((source/'prediction-seal.json').read_text())
    assert sha(source/'protocol.json')==seal['protocol_sha256']
    write(out/'protocol.json',dict(source_seal=sha(source/'prediction-seal.json'),runner=sha(__file__),
        support='same reference-valid mixed/raw-known pixels, independently within each nonoverlapping projected zone; all arms privileged common mask',
        mass='k=current predicted near count in that exact support; g=GT near count there',
        placement='current ranking ascending depth; oracle maximizes overlap at exact k',
        bands=['0-.05 inclusive','>.05-.10 inclusive','>.10-.20 inclusive','>.20'],band_roundoff_tolerance_m=1e-6,
        error_sets='all current FN/FP plus raw-correct to current-wrong FN/FP; joint GT x prediction bins',
        interpretation='posthoc ceiling, not trainable gain or disjoint physical error budget; no new model or training',
        closed='existence-only SCDE and mono-guided ZPA',backend='CPU TASK_NOT_GPU_SUITABLE saved-array diagnostic'))
    rows=[];zone_rows=[];band_rows=[]
    totals={k:np.zeros((4,4),np.int64) for k in ('all_fn','all_fp','new_fn','new_fp')}
    for i,e in enumerate(protocol['selected']):
        path=source/'inputs'/e['filename'];assert sha(path)==e['sha256']
        with h5py.File(path) as f:gt,fr=np.array(f['depth']),np.array(f['fr'])
        a={}
        for name in ('raw','depthor'):
            file=f'{i:03d}-{name}.npz';assert sha(source/'predictions'/file)==seal['files'][file]
            a[name]=np.load(source/'predictions'/file)['depth']
        pred=a['depthor'];raw=a['raw'];valid=np.isfinite(gt)&(gt>.001)&(gt<10)
        mixed,_=mixed_mask(gt,fr,valid);domain=mixed&np.isfinite(raw)
        occupancy=np.zeros(gt.shape,bool)
        outputs={k:np.zeros(gt.shape,bool) for k in ('current','mass_only','placement_only','both')}
        for j,box in enumerate(fr):
            s=rect(box,gt.shape);m=domain[s]
            if not m.any():continue
            assert not occupancy[s].any();occupancy[s]=True
            p=pred[s][m];g=gt[s][m];aa=arms_1d(p,g)
            for k,b in aa.items():outputs[k][s][m]=b
            zone_rows.append(dict(filename=e['filename'],zone=j,pixels=int(m.sum()),
                                  pred_mass=int((p<2).sum()),gt_mass=int((g<2).sum())))
        assert not (domain&~occupancy).any()
        for arm,p in outputs.items():
            rows.append(dict(filename=e['filename'],scene=e['filename'].split('/')[0],arm=arm,
                             **metric(p,gt<2,domain)))
        masks=dict(all_fn=domain&(gt<2)&(pred>=2),all_fp=domain&(gt>=2)&(pred<2),
                   new_fn=domain&(gt<2)&(raw<2)&(pred>=2),new_fp=domain&(gt>=2)&(raw>=2)&(pred<2))
        for name,m in masks.items():
            b=bands(gt,pred,m);totals[name]+=b
            band_rows.append(dict(filename=e['filename'],error=name,joint_counts=b.tolist()))
    def agg(keys):
        groups={}
        for r in rows:
            k=tuple(r[x] for x in keys);v=groups.setdefault(k,dict(tp=0,fp=0,fn=0,tn=0))
            for f in v:v[f]+=r[f]
        return [dict(zip(keys,k),**derived(v)) for k,v in groups.items()]
    primary=agg(['arm']);by={r['arm']:r for r in primary}
    assert (by['current']['tp'],by['current']['fp'],by['current']['fn'])==(224725,63181,10125)
    assert by['both']['iou']==1
    summary={}
    for k,b in totals.items():
        n=int(b.sum());summary[k]=dict(count=n,joint_gt_rows_prediction_columns=b.tolist(),
            both_within_5cm=int(b[:1,:1].sum()),both_within_10cm=int(b[:2,:2].sum()),
            both_within_20cm=int(b[:3,:3].sum()),both_beyond_20cm=int(b[3,3]),
            gt_beyond_20cm=int(b[3,:].sum()),prediction_beyond_20cm=int(b[:,3].sum()))
    c=by['current']['iou'];m=by['mass_only']['iou'];p=by['placement_only']['iou']
    write(out/'results.json',dict(primary=primary,scenes=agg(['scene','arm']),threshold=summary,
        iou_gain_mass=m-c,iou_gain_placement=p-c,interaction=1-m-p+c,
        attribution_warning='interaction is metric factorial contrast, not a physical error cause; threshold overlaps both factors',
        sum_zone_absolute_count_error=sum(abs(z['pred_mass']-z['gt_mass']) for z in zone_rows),
        decision='DIAGNOSTIC_ONLY_NO_TRAINING'))
    write(out/'per-frame.json',rows);write(out/'zone-mass.json',zone_rows);write(out/'threshold-per-frame.json',band_rows)
    print((out/'results.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
