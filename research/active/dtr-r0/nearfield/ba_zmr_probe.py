"""Zero-training within-zone mass reassignment of sealed predictions."""
import argparse
import heapq
import json
from pathlib import Path
import h5py
import numpy as np
from ba_depth_probe import sha, write, rect, mixed_mask, boundary_mask


def rank_mask(depth, k):
    flat=np.where(np.isfinite(depth)&(depth>0),depth,np.inf).ravel()
    ids=np.argsort(flat,kind='stable')[:k]
    out=np.zeros(flat.size,bool);out[ids]=True
    return out.reshape(depth.shape)


def connected_mask(depth,rgb,k):
    """4-connected minimax region growth; depth percentile + RGB edge/4."""
    h,w=depth.shape
    if not k:return np.zeros((h,w),bool)
    order=np.argsort(np.where(np.isfinite(depth)&(depth>0),depth,np.inf).ravel(),kind='stable')
    rank=np.empty(h*w,float);rank[order]=np.arange(h*w)/max(1,h*w-1);rank=rank.reshape(h,w)
    seed=int(order[0]);queue=[(0.,seed)];seen=np.zeros((h,w),bool);selected=seen.copy()
    seen.flat[seed]=True;n=0
    while n<k:
        cost,idx=heapq.heappop(queue);y,x=divmod(idx,w);selected[y,x]=True;n+=1
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<h and 0<=xx<w and not seen[yy,xx]:
                edge=float(np.abs(rgb[yy,xx].astype(float)-rgb[y,x]).mean()/255)
                seen[yy,xx]=True
                heapq.heappush(queue,(max(cost,float(rank[yy,xx])+.25*edge),yy*w+xx))
    return selected


def metric(p,y,m):
    return dict(tp=int((p&y&m).sum()),fp=int((p&~y&m).sum()),fn=int((~p&y&m).sum()),tn=int((~p&~y&m).sum()))


def derived(r):
    def d(a,b):return a/b if b else None
    return dict(**r,precision=d(r['tp'],r['tp']+r['fp']),recall=d(r['tp'],r['tp']+r['fn']),iou=d(r['tp'],r['tp']+r['fp']+r['fn']))


def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());seal=json.loads((source/'prediction-seal.json').read_text())
    assert sha(source/'protocol.json')==seal['protocol_sha256']
    write(out/'protocol.json',dict(id='ba-zmr-20260918',source_seal=sha(source/'prediction-seal.json'),runner=sha(__file__),
        selection='unchanged 160 frames; placement on ALL fr rectangles without GT-based mixed selection',
        threshold=2,rank='existing UniDepth metric prediction ascending; stable raster tie',
        connected='4-neighbor minimax growth from minimum depth; percentile rank + .25 RGB adjacent mean abs diff/255; no confidence tensor available',
        mass='exact DEPTHOR near count over each full clipped rectangle; reject overlapping rectangles before evaluation',
        oracle='GT near count among reference-valid pixels only; ranking among reference-valid pixels only; privileged diagnostic, not guaranteed ceiling',
        score='same reference-valid mixed/raw-known domain; boundary5/10 and complement; per-scene and gross transitions',
        go='rank or connected improves primary precision/recall/IoU and majority scene IoU; no automatic training this turn',
        limits='invalid GT masks can alter evaluated mass despite exact full-rectangle conservation; report both',
        SCDE='existence-only proposed Q10 route closed by explicit user decision',
        backend='CPU TASK_NOT_GPU_SUITABLE saved arrays + small heaps, no inference or training'))
    rows=[];audits=[]
    for i,e in enumerate(protocol['selected']):
        path=source/'inputs'/e['filename'];assert sha(path)==e['sha256']
        with h5py.File(path) as f:gt,fr,rgb=[np.array(f[k]) for k in ('depth','fr','rgb')]
        arrays={}
        for arm in ('raw','mono','depthor'):
            name=f'{i:03d}-{arm}.npz';assert sha(source/'predictions'/name)==seal['files'][name]
            arrays[arm]=np.load(source/'predictions'/name)['depth']
        valid=np.isfinite(gt)&(gt>.001)&(gt<10);y=gt<2;c=arrays['depthor']<2
        outputs={'depthor':c,'rank':c.copy(),'connected':c.copy(),'oracle_mass_rank':c.copy()}
        occupied=np.zeros(gt.shape,bool);mass=[]
        for box in fr:
            s=rect(box,gt.shape)
            if occupied[s].size==0:continue
            assert not occupied[s].any(),'Overlapping rectangles: cannot claim independent exact mass'
            occupied[s]=True
            k=int(c[s].sum());md=arrays['mono'][s]
            outputs['rank'][s]=rank_mask(md,k)
            outputs['connected'][s]=connected_mask(md,rgb[s],k)
            v=valid[s];kg=int((y[s]&v).sum())
            oracle=np.zeros(v.shape,bool)
            ids=np.flatnonzero(v);order=np.argsort(md.ravel()[ids],kind='stable');oracle.flat[ids[order[:kg]]]=True
            outputs['oracle_mass_rank'][s]=oracle
            assert outputs['rank'][s].sum()==outputs['connected'][s].sum()==k
            mass.append(dict(k=k,gt_valid_mass=kg,pixels=int(v.size)))
        mixed,_=mixed_mask(gt,fr,valid);domain=mixed&np.isfinite(arrays['raw'])
        b5=boundary_mask(gt,valid,5);b10=boundary_mask(gt,valid,10)
        old_new=domain&y&(arrays['raw']<2)&~c
        for arm,p in outputs.items():
            for name,m in dict(mixed_common=domain,boundary5=domain&b5,nonboundary5=domain&~b5,
                                boundary10=domain&b10,nonboundary10=domain&~b10).items():
                rows.append(dict(filename=e['filename'],scene=e['filename'].split('/')[0],arm=arm,stratum=name,
                    **metric(p,y,m),recovered_previous_new_fn=int((p&old_new&m).sum()),
                    rescued_c_fn=int((p&~c&y&m).sum()),lost_c_tp=int((~p&c&y&m).sum()),
                    added_fp=int((p&~c&~y&m).sum()),removed_fp=int((~p&c&~y&m).sum()),
                    predicted_near=int((p&m).sum())))
        audits.append(dict(filename=e['filename'],rectangles=mass,union_pixels=int(occupied.sum()),
            union_mass={a:int((p&occupied).sum()) for a,p in outputs.items()},
            scored_mass={a:int((p&domain).sum()) for a,p in outputs.items()}))
        np.savez_compressed(out/f'assignment-{i:03d}.npz',**outputs)
        if i%20==0:print('frame',i,flush=True)
    fields=['tp','fp','fn','tn','recovered_previous_new_fn','rescued_c_fn','lost_c_tp','added_fp','removed_fp','predicted_near']
    def aggregate(by):
        groups={}
        for r in rows:
            key=tuple(r[k] for k in by);g=groups.setdefault(key,{k:0 for k in fields})
            for k in fields:g[k]+=r[k]
        return [dict(zip(by,k),**derived(v)) for k,v in groups.items()]
    totals=aggregate(['arm','stratum']);scenes=aggregate(['scene','arm','stratum'])
    primary=[r for r in totals if r['stratum']=='mixed_common']
    base=next(r for r in primary if r['arm']=='depthor')
    assert (base['tp'],base['fp'],base['fn'])==(224725,63181,10125)
    write(out/'results.json',dict(primary=primary,strata=totals,scenes=scenes,
        exact_full_rectangle_mass=True,frames=len(audits),oracle_ceiling=False,
        note='GT mass/validity oracle not deployable; output masks never substitute for raw evidence'))
    write(out/'per-frame.json',rows);write(out/'mass-audit.json',audits)
    print(json.dumps(primary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
