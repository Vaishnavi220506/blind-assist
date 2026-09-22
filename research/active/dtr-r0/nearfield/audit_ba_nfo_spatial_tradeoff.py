"""Read-only joint-to-late paired pixel/zone changes, no inference or tuning."""
import json
from collections import defaultdict

import numpy as np

import ba_nfo_matched as m
import ba_nfo_hardfit as h
import ba_nfo_jointfit as j
import ba_nfo_latefusion as late

OUT = m.ROOT/'artifacts.local/work/ba-nfo-spatial-tradeoff-audit-20260919'


def transitions(old, new, truth, mask):
    pos, neg = mask & truth, mask & ~truth
    return dict(near=int(pos.sum()), far=int(neg.sum()),
        retained_tp=int((old&new&pos).sum()), lost_tp=int((old&~new&pos).sum()),
        rescued_fn=int((~old&new&pos).sum()), retained_fn=int((~old&~new&pos).sum()),
        removed_fp=int((old&~new&neg).sum()), added_fp=int((~old&new&neg).sum()),
        retained_fp=int((old&new&neg).sum()), retained_tn=int((~old&~new&neg).sum()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.json').exists(), 'Preserve completed diagnostic'
    selection=json.loads((late.OUT/'selection.json').read_text())
    assert len(selection)==32 and all(r['row']['split']=='train' for r in selection)
    files=[late.OUT/'selection.json', j.OUT/'fit-scores.npz', late.OUT/'fit-scores.npz',
           j.OUT/'results.json',late.OUT/'results.json',late.OUT/'batch-plan.json']
    protocol=dict(scope='READ_ONLY_32_CONSUMED_TRAIN_SAVED_PREDICTIONS',
        question='Where do the net122 additional small-support FN occur, and what old support is lost versus rescued?',
        cutoff=.081,near_threshold=2.,
        strata='Selected vs other zones; <=1.8m vs (1.8,2)m near depths; near count1-4/5-15/16-64/65+; observed ToF<2/2-2.2/2.2+/missing; original depth-separated criterion',
        limits='Posthoc descriptive audit; no causal supervision claim, retuning, relabeling or removal of previous failures',
        stop='One saved-output audit; no training, forward passes, validation, test or successor',
        backend='CPU TASK_NOT_GPU_SUITABLE: small array boolean reductions and file reads',
        inputs={str(p):m.sha(p) for p in files},code_sha256=m.sha(__file__))
    m.write(OUT/'protocol.json',protocol)
    with np.load(j.OUT/'fit-scores.npz') as a: old_scores=a['scores'][:,2].copy()
    with np.load(late.OUT/'fit-scores.npz') as a: new_scores=a['scores'][:,2].copy()
    arrays=h.load_selection(); totals=defaultdict(lambda:defaultdict(int)); zones=[]; frames=[]
    known_counts=[];target_counts=[]
    for i,(a,row) in enumerate(zip(arrays,selection)):
        old,new=old_scores[i]>=.081,new_scores[i]>=.081
        known,truth,mixed,small=m.masks(a['depth'],a['boxes'],2.)
        y0,x0,y1,x1=row['box']; selected=np.zeros_like(known);selected[y0:y1,x0:x1]=True
        known_counts.append(int(known.sum()));target_counts.append(int((known&selected).sum()))
        domains=dict(full=known,small=small,selected_small=small&selected,other_small=small&~selected,
            small_near_le1p8=small&truth&(a['depth']<=1.8),small_near_gt1p8=small&truth&(a['depth']>1.8))
        frame=dict(id=row['row']['id'],frame_kind=row['kind'],metrics={})
        for name,mask in domains.items():
            c=transitions(old,new,truth,mask)
            for k,v in c.items():totals[name][k]+=v
            frame['metrics'][name]=c
        for zi,(yy0,xx0,yy1,xx1) in enumerate(a['boxes']):
            mask=np.zeros_like(known);mask[yy0:yy1,xx0:xx1]=True;mask &= known
            near=int((mask&truth).sum());far=int((mask&~truth).sum())
            if not near or not far or near/(near+far)>.2:continue
            c=transitions(old,new,truth,mask)
            q90=float(np.quantile(a['depth'][mask&truth],.9));q10=float(np.quantile(a['depth'][mask&~truth],.1))
            observed=float(a['values'][zi]);qgap=q10-q90
            # Exact original case-selection separation conditions, without reselection.
            separated=q90<=1.8 and q10>=2.2 and qgap>=.5 and np.isfinite(observed) and observed>=2.2
            size='1-4' if near<=4 else '5-15' if near<=15 else '16-64' if near<=64 else '65+'
            tof='missing' if not np.isfinite(observed) else '<2' if observed<2 else '2-2.2' if observed<2.2 else '>=2.2'
            entry=dict(id=row['row']['id'],frame_index=i,zone=zi,selected=zi==row['zone'],frame_kind=row['kind'],
                box=[int(v) for v in [yy0,xx0,yy1,xx1]],size=size,tof=tof,depth_separated=bool(separated),
                q90near=q90,q10far=q10,observed_tof=observed if np.isfinite(observed) else None,
                old=m.metrics(m.counts(old,truth,mask)),new=m.metrics(m.counts(new,truth,mask)),
                changes=c,net_extra_fn=c['lost_tp']-c['rescued_fn'])
            zones.append(entry)
            for name in [f'size:{size}',f'tof:{tof}',f'separated:{separated}',f'frame_kind:{row["kind"]}']:
                for k,v in c.items():totals[name][k]+=v
        frames.append(frame)
    for key in ['full','small']:
        for source,pred in [(j.OUT,'old'),(late.OUT,'new')]:
            result=json.loads((source/'results.json').read_text())
            expected=result['joint_fit' if pred=='old' else 'late_fit']['metrics']['2.0'][key]
            c=totals[key]
            got=dict(tp=c['retained_tp']+c['lost_tp' if pred=='old' else 'rescued_fn'],
                fn=c['retained_fn']+c['rescued_fn' if pred=='old' else 'lost_tp'],
                fp=c['retained_fp']+c['removed_fp' if pred=='old' else 'added_fp'],
                tn=c['retained_tn']+c['added_fp' if pred=='old' else 'removed_fp'])
            assert all(got[k]==expected[k] for k in got)
    for key in totals['small']:
        assert totals['selected_small'][key]+totals['other_small'][key]==totals['small'][key]
        assert sum(z['changes'][key] for z in zones)==totals['small'][key]
    ratios=[]
    for batch in json.loads((late.OUT/'batch-plan.json').read_text()):
        ratios.append(1+sum(known_counts[i] for i in batch)/sum(target_counts[i] for i in batch))
    # Equal half means are not equal per-pixel weights. Applies identically to both models.
    loss_support=dict(full_known=sum(known_counts),selected_known=sum(target_counts),
        per_pixel_selected_to_unselected_coefficient=dict(min=float(min(ratios)),median=float(np.median(ratios)),max=float(max(ratios))),
        note='BCE and ordinal local mean coefficients only; identical recipe in both models, not causal attribution of this architecture difference')
    result=dict(status='COMPLETE',frames=32,small_zones=len(zones),totals=dict(totals),loss_support=loss_support,
        zone_effect_counts=dict(net_hurt=sum(z['net_extra_fn']>0 for z in zones),net_helped=sum(z['net_extra_fn']<0 for z in zones),net_equal=sum(z['net_extra_fn']==0 for z in zones)),
        zones=zones,frame_details=frames,validation_frames=0,test_frames=0,forward_calls=0)
    m.write(OUT/'results.json',result)
    m.write(OUT/'top-net-loss-zones.json',sorted(zones,key=lambda z:(-z['net_extra_fn'],z['id'],z['zone']))[:10])
    print(json.dumps(dict(small_zones=len(zones),totals=dict(totals),loss_support=loss_support,zone_effect_counts=result['zone_effect_counts']),indent=2))


if __name__=='__main__':main()
