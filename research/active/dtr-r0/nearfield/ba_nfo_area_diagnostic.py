"""Fixed-model area and observed-return decomposition; no threshold fitting."""
import json
import time
from collections import defaultdict
import numpy as np
import torch
import ba_nfo_matched as m

OUT=m.ROOT/'artifacts.local/work/ba-nfo-area-20260919'
BINS=['0-5%','5-10%','10-20%','20-50%','50-100%']

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device='cuda'
    models=m.load_models(device)
    rows=[r for r in json.loads((m.OUT/'manifest.json').read_text()) if r['split']=='test']
    cuts={a:v['cutoff'] for a,v in json.loads((m.OUT/'calibration.json').read_text()).items()}
    sums=defaultdict(lambda:np.zeros(4,np.int64))
    zones=defaultdict(int)
    support=defaultdict(lambda:np.zeros(5,np.int64))
    start=time.perf_counter()
    m.write(OUT/'protocol.json',dict(scope='Consumed Development diagnostic, no tuning',
        thresholds=m.THRESHOLDS.tolist(),area_bins=BINS,boundary='(0,5],(5,10],(10,20],(20,50],(50,100); exclude pure near/far',
        cuts=cuts,return_states=['near','far','missing'],
        compatibility='diagnostic only abs(GT axial depth - public return)<=3*(0.01+0.02*return)',
        model_hashes={a:m.sha(m.OUT/f'trained-{a}.pt') for a in models},code_sha256=m.sha(__file__)))
    for offset in range(0,len(rows),24):
        batch=rows[offset:offset+24]
        arrays=[]
        for r in batch:
            with np.load(m.OLD/r['prepared']) as a:
                arrays.append({k:a[k].copy() for k in ['rgb','depth','values','boxes']})
        rgb=torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).to(device)
        z=torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).to(device)
        pred={arm:(m.probabilities(model(rgb,z),arm).cpu().numpy()>=cuts[arm]) for arm,model in models.items()}
        for j,a in enumerate(arrays):
            for ti,t in enumerate(m.THRESHOLDS):
                for zi,(y0,x0,y1,x1) in enumerate(a['boxes']):
                    d=a['depth'][y0:y1,x0:x1];known=np.isfinite(d);near=known&(d<t)
                    n=int(near.sum());total=int(known.sum())
                    if not 0<n<total:continue
                    binid=int(np.searchsorted([.05,.1,.2,.5],n/total,side='left'))
                    b=BINS[binid];value=a['values'][zi]
                    state='missing' if not np.isfinite(value) else ('near' if value<t else 'far')
                    for group in ['all',state]:
                        zones[str(float(t)),b,group]+=1
                        for arm in models:
                            sums[str(float(t)),b,group,arm]+=m.counts(pred[arm][j,ti,y0:y1,x0:x1],near,known)
                    dp=pred['depth'][j,ti,y0:y1,x0:x1];npred=pred['nfo'][j,ti,y0:y1,x0:x1]
                    lost=near&dp&~npred;rescued=near&~dp&npred
                    compatible=near& (np.abs(d-value)<=3*(.01+.02*value)) if np.isfinite(value) else np.zeros_like(near)
                    support[str(float(t)),b,state]+=np.array([n,lost.sum(),rescued.sum(),(lost&compatible).sum(),(near&~npred&compatible).sum()],np.int64)
        if offset%120==0:print('DIAG',offset,flush=True)
    records=[dict(threshold=t,area=b,return_state=g,arm=a,zones=zones[t,b,g],**m.metrics(c)) for (t,b,g,a),c in sums.items()]
    paired=[dict(threshold=t,area=b,return_state=s,near_pixels=int(c[0]),depth_tp_lost_by_nfo=int(c[1]),depth_fn_rescued_by_nfo=int(c[2]),lost_compatible_with_return=int(c[3]),nfo_fn_compatible_with_return=int(c[4])) for (t,b,s),c in support.items()]
    # Reproduce original totals before interpreting any subdivision.
    previous=json.loads((m.OUT/'results.json').read_text())
    for t in map(lambda x:str(float(x)),m.THRESHOLDS):
        for arm in models:
            c=sum((v for (tt,b,g,a),v in sums.items() if tt==t and g=='all' and a==arm),np.zeros(4,np.int64))
            expected=[previous['metrics'][arm][t]['mixed'][k] for k in ['tp','fp','fn','tn']]
            assert c.tolist()==expected,(t,arm,c.tolist(),expected)
    m.write(OUT/'results.json',dict(records=records,paired=paired,original_counts_match=True,
        device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start))
    for b in BINS:
        print(b,{a:m.metrics(sums['2.0',b,'all',a]) for a in models},flush=True)
    print('RETURN_DECOMPOSITION',[r for r in paired if r['threshold']=='2.0' and r['area'] in BINS[:3]],flush=True)

if __name__=='__main__':main()
