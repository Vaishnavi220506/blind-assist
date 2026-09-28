"""Travel-frame attribution pilot, equal-exposure spatial controls, no training."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data'
OUT=ROOT/'artifacts.local/work/cnh-attribution-diagnostic-20260929/attribution'
ARMS=('SINGLE1','S2K4','A2_HEAD','A2','MOTION8','EXACT8','WRONG8','STATIC8','A2_STATIC8','BOUNDARY_REF8')
CALIB=list(range(6)); AUDIT=list(range(6,18))


def save(path,obj):
    p=Path(path); tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)


def setup():
    sys.path.insert(0,str(DATA/'source'))
    import torch
    import cnh_track_a_gpu_readout as g
    import cnh_learned_features as F
    import cnh_learned_readout as L
    from cnh_learned_memory_fusion import causal_ewma
    from cnh_track_a_readout import noisy_poses
    import cnh_track_a_gpu_readout2 as g2
    torch.set_num_threads(2)
    models=[]
    for seed in range(3):
        path=ROOT/f'artifacts.local/work/cnh-learned-readout-20260928/seeds/model_seed{seed}.pt'
        net=L.Readout().to(L.DEV)
        net.load_state_dict(torch.load(path,map_location=L.DEV,weights_only=True));models.append(net.eval())
    return torch,g,F,L,causal_ewma,noisy_poses,g2,models


def network(h,a,poses,travel,head,bias,seed,mod,with_s2=True):
    torch,g,F,L,smooth,noisy_fn,g2,models=mod
    noisy=noisy_fn(poses,seed,dt=.2)
    tq=np.linalg.inv(travel)@poses
    hq=np.linalg.inv(head)@poses
    z4,z1,sup=F.sequence_features(h,a,bias,tq,noisy)
    x=np.stack([L.squash(z4.astype(np.float16).astype(np.float32)),L.squash(z1.astype(np.float16).astype(np.float32))],1)
    sup_head=(g.query_weights(torch.as_tensor(hq,dtype=g.D64,device=g.DEV))>=.75).reshape(len(h),6,8,8,16).cpu().numpy()
    xb=torch.as_tensor(np.concatenate([x,x]),device=L.DEV)
    sb=torch.as_tensor(np.concatenate([sup,sup_head]),device=L.DEV)
    with torch.no_grad():
        pred=np.mean([net(xb,sb).cpu().numpy() for net in models],axis=0)
    result={}
    for k,scores in zip(('A2','A2_HEAD'),np.split(pred,2)):
        result[k]=smooth(scores,np.zeros(len(h),int),np.arange(len(h)),alpha=.5,window=5)[:,[2,3]]
    single=g.scan(g.T(h[-1:]-bias),g.T(16*a[-1:,...,None]+np.maximum(bias,0)),torch.as_tensor(sup[-1:],device=g.DEV))
    result['SINGLE1']=single.cpu().numpy()[0,[2,3]]
    if with_s2:
        r=g2.sequence_readouts(h[-4:],a[-4:],bias,poses[-4:],tq[-4:],noisy[-4:],with_r4=False)
        result['S2K4']=r['S2/noisy|0.75'][-1,[2,3]]
    return result,noisy


def point_score(hist,ambient,bias,poses,query_from_sensor):
    """No scene or labels: sum evidence for identical 3D candidate points over 8 frames.

    Reference and coarse sensor use this identical method; only angle bins differ.
    Every candidate must be geometrically in-view for all eight exposures.
    """
    hist=hist[-8:];ambient=ambient[-8:];poses=poses[-8:]
    side=hist.shape[1]
    residual=hist-bias
    variance=16*ambient[...,None]+np.maximum(bias,0)
    out=[]; coverage=[]
    for yl,yh in [(-.2,.42),(.42,.9)]:
        # Include public query boundary planes; cell centres alone leave a
        # deterministic 25mm blind strip in the explicit boundary diagnostic.
        xs=np.unique(np.r_[np.arange(-.275,.3,.05),-.299,.299])
        ys=np.unique(np.r_[np.linspace(yl+.025,yh-.025,5),yl+.001,yh-.001])
        zs=np.unique(np.r_[np.arange(.35,3,.1),.301,2.999])
        q=np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),-1).reshape(-1,3)
        local=(q-query_from_sensor[:3,3])@query_from_sensor[:3,:3]
        world=local@poses[-1,:3,:3].T+poses[-1,:3,3]
        total=np.zeros(len(q));var=np.zeros(len(q));seen=np.zeros(len(q),int)
        now=np.linalg.norm(local,axis=1)
        for h,a,r,p in zip(residual,variance,hist,poses):
            v=(world-p[:3,3])@p[:3,:3]
            radius=np.linalg.norm(v,axis=1)
            xy=v[:,:2]/np.maximum(v[:,2,None],1e-12)
            colrow=np.floor((xy+np.tan(np.pi/8))/(2*np.tan(np.pi/8))*side).astype(int)
            bins=np.floor(radius/(8*.0375348)).astype(int)
            valid=(v[:,2]>0)&(colrow>=0).all(1)&(colrow<side).all(1)&(bins>=0)&(bins<16)
            ii=np.flatnonzero(valid);cc=colrow[ii,0];rr=colrow[ii,1];bb=bins[ii]
            gain=(radius[ii]/now[ii])**2
            total[ii]+=gain*h[rr,cc,bb];var[ii]+=gain**2*a[rr,cc,bb];seen[ii]+=1
        good=seen==8
        score=total/np.sqrt(np.maximum(var,1e-9))
        out.append(float(score[good].max()) if good.any() else -50.)
        coverage.append(float(good.mean()))
    return np.array(out),np.array(coverage)


def run(limit=None):
    mod=setup();torch,g,*_=mod
    from cnh_proposal_attribution_scenes import make_scenes,render
    assert torch.cuda.is_available()
    bias=np.load(DATA/'readouts-gpu/primary-mount-10-snr6/bias.npy')
    finebias=np.repeat(np.repeat(bias/4,2,axis=0),2,axis=1)
    paths=[Path(__file__),HERE/'cnh_proposal_attribution_scenes.py',OUT/'DECISIONS.md']
    identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if (OUT/'request.json').exists():
        old=json.loads((OUT/'request.json').read_text())
        assert old['identity']==identity,'source/criteria changed; retain outputs and inspect before resume'
    else:
        save(OUT/'request.json',dict(identity=identity,calib=CALIB,audit=AUDIT,arms=ARMS,device=torch.cuda.get_device_name(),backend='CUDA neural/transport; NumPy small-point inference and renderer electronics'))
        for p in paths:
            if p.name!='DECISIONS.md':(OUT/p.name).write_bytes(p.read_bytes())
    units=(CALIB+AUDIT)[:limit] if limit else CALIB+AUDIT
    started=time.time()
    try:
        for u in units:
            path=OUT/f'unit{u:02d}.npz'
            if path.exists():continue
            rows=[];obs=[]
            for scene in make_scenes(u):
                c=scene['config'];seed=2026092900+u*1000+c*10
                data=render(scene,seed)
                h,a=data['hist'],data['ambient']
                # Exact conservation, not approximate equal photon budget.
                reconstructed=data['finehist'].reshape(16,8,2,8,2,16).sum((2,4))
                assert np.allclose(h,reconstructed,atol=1e-5,rtol=1e-6)
                assert np.allclose(a,data['fineambient'].reshape(16,8,2,8,2).sum((2,4)))
                scores,noisy=network(h,a,scene['poses'],scene['travel'],scene['head'],bias,seed+7,mod)
                tq=(np.linalg.inv(scene['travel'])@scene['poses'])[-1]
                scores['MOTION8'],coverage=point_score(h,a,bias,noisy,tq)
                scores['EXACT8'],_=point_score(h,a,bias,scene['poses'],tq)
                scores['WRONG8'],_=point_score(h,a,bias,np.repeat(noisy[-1:],16,axis=0),tq)
                scores['BOUNDARY_REF8'],_=point_score(data['finehist'],data['fineambient'],finebias,noisy,tq)
                static=render(scene,seed+5,static=True)
                p=np.repeat(scene['poses'][-1:],8,axis=0);q=np.repeat(scene['travel'][-1:],8,axis=0);head=np.repeat(scene['head'][-1:],8,axis=0)
                ss,_=network(static['hist'],static['ambient'],p,q,head,bias,seed+8,mod,with_s2=False)
                scores['STATIC8'],_=point_score(static['hist'],static['ambient'],bias,p,tq)
                scores['A2_STATIC8']=ss['A2'][-1]
                final=np.array([scores[k][-1] if k in ('A2','A2_HEAD') else scores[k] for k in ARMS])
                rows.append(dict(config=c,family=scene['family'],margin=scene['margin'],group=scene['group'],
                        y=scene['labels'],final=final,head=scores['A2_HEAD'],aligned=scores['A2'],coverage=coverage))
                obs.append(data)
            np.savez_compressed(path,unit=u,config=[r['config'] for r in rows],family=[r['family'] for r in rows],
                margin=[r['margin'] for r in rows],group=[r['group'] for r in rows],labels=np.array([r['y'] for r in rows]),
                final=np.array([r['final'] for r in rows]),head=np.array([r['head'] for r in rows]),aligned=np.array([r['aligned'] for r in rows]),
                coverage=np.array([r['coverage'] for r in rows]))
            np.savez_compressed(OUT/f'unit{u:02d}_observations.npz',**{k:np.stack([o[k] for o in obs]) for k in ('hist','ambient','finehist','fineambient')})
            save(OUT/'progress.json',dict(unit_completed=u,total=18,elapsed_s=time.time()-started))
            print(f'unit {u} complete, {time.time()-started:.1f}s',flush=True)
        save(OUT/'terminal.json',dict(status='pilot_complete' if limit else 'complete',elapsed_s=time.time()-started))
    except BaseException as e:
        save(OUT/'terminal.json',dict(status='failed',error=repr(e),elapsed_s=time.time()-started));raise


def threshold(values,budget=.1):
    values=np.sort(values)[::-1]
    return float(np.nextafter(values[int(np.floor(budget*len(values)))],np.inf))


def binary_metrics(y,alarm):
    y=np.asarray(y,bool);alarm=np.asarray(alarm,bool)
    tp=int((y&alarm).sum());fp=int((~y&alarm).sum());p=int(y.sum());n=int((~y).sum())
    return dict(positive=p,negative=n,tp=tp,fp=fp,fn=p-tp,tn=n-fp,tpr=tp/p if p else None,
                fpr=fp/n if n else None,errors=fp+p-tp,ber=.5*(fp/n+(p-tp)/p) if p and n else None)


def episodes(y,score,thr):
    empty=~y.any(axis=1);alarms=score>=thr
    neg=alarms[empty]
    starts=neg&~np.concatenate([np.zeros((len(neg),1),bool),neg[:,:-1]],1)
    minutes=neg.size/5/60
    return dict(empty_sequences=int(empty.sum()),false_sequences=int(neg.any(axis=1).sum()),
                false_episodes=int(starts.sum()),empty_minutes=minutes,episodes_per_min=float(starts.sum()/minutes) if minutes else None,
                positive_sequences=int((~empty).sum()),detected_positive_sequences=int((alarms[~empty]&y[~empty]).any(axis=1).sum()))


def episode_threshold(y,score):
    empty=~y.any(axis=1)
    vals=np.unique(score[empty]);best=float(np.nextafter(vals[-1],np.inf))
    for v in vals[::-1]:
        if episodes(y,score,v)['episodes_per_min']<=2:best=float(v)
        else:break
    return best


def evaluate():
    from sklearn.metrics import average_precision_score
    units={}
    for u in CALIB+AUDIT:
        with np.load(OUT/f'unit{u:02d}.npz') as z:units[u]={k:z[k] for k in z.files}
    cat=lambda keys,key:np.concatenate([units[u][key] for u in keys])
    cy=cat(CALIB,'labels')[:,-1]; ay=cat(AUDIT,'labels')[:,-1]
    cs=cat(CALIB,'final');ass=cat(AUDIT,'final')
    families=cat(AUDIT,'family');margins=cat(AUDIT,'margin');groups=cat(AUDIT,'group')
    unitids=np.repeat(AUDIT,22)
    thresholds={}; rows=[]; ledger=[]
    for qi,group in enumerate(('HEAD','BODY')):
        for j,arm in enumerate(ARMS):
            thr=threshold(cs[:,j,qi][~cy[:,qi].astype(bool)])
            thresholds[group+'|'+arm]=thr
            pred=ass[:,j,qi]>=thr
            for subset,mask in [('all',np.ones(len(ay),bool)),('boundary5cm',(abs(margins)<=.05)&(groups==qi)),
                                ('general',families=='general')]+[(f,families==f) for f in ('boundary','mixed_surface','sidewall')]:
                yy=ay[mask,qi].astype(bool);ss=ass[mask,j,qi]
                row=dict(group=group,arm=arm,subset=subset,threshold=thr,**binary_metrics(yy,pred[mask]))
                row['ap']=float(average_precision_score(yy,ss)) if yy.any() else None
                rows.append(row)
        predictions={arm:ass[:,j,qi]>=thresholds[group+'|'+arm] for j,arm in enumerate(ARMS)}
        for i in range(len(ay)):
            y=bool(ay[i,qi]);baseline=bool(predictions['A2'][i]);fixed=[arm for arm,p in predictions.items() if bool(p[i])==y and baseline!=y]
            ledger.append(dict(unit=int(unitids[i]),config=i%22,group=group,family=str(families[i]),margin=float(margins[i]),label=int(y),
                A2_error=baseline!=y,correct_arms=[a for a,p in predictions.items() if bool(p[i])==y],repaired_arms=fixed,
                unresolved=bool(all(bool(p[i])!=y for p in predictions.values())),predictions={a:int(p[i]) for a,p in predictions.items()}))
    interface={}
    for qi,group in enumerate(('HEAD','BODY')):
        yy=cat(CALIB,'labels')[:,7:,qi].astype(bool);aa=cat(AUDIT,'labels')[:,7:,qi].astype(bool)
        hcal=cat(CALIB,'head')[:,7:,qi];acal=cat(CALIB,'aligned')[:,7:,qi]
        h=cat(AUDIT,'head')[:,7:,qi];a=cat(AUDIT,'aligned')[:,7:,qi]
        th=episode_threshold(yy,hcal);ta=episode_threshold(yy,acal)
        old=episodes(aa,h,th);same=episodes(aa,a,th);adapt=episodes(aa,a,ta)
        reduction=(old['false_sequences']-same['false_sequences'])/old['false_sequences'] if old['false_sequences'] else None
        loss=(old['detected_positive_sequences']-same['detected_positive_sequences'])/old['positive_sequences'] if old['positive_sequences'] else None
        interface[group]=dict(head_threshold=th,aligned_threshold=ta,head=old,aligned_fixed=same,aligned_recalibrated=adapt,
            fraction_false_stops_removed=reduction,event_detection_loss=loss,interface_dominant=bool(reduction is not None and reduction>=.5 and loss<=.03))
    lookup={(r['group'],r['arm'],r['subset']):r for r in rows}
    comparisons={}
    for qi,group in enumerate(('HEAD','BODY')):
        base=lookup[group,'A2','all'];motion=lookup[group,'MOTION8','all'];ref=lookup[group,'BOUNDARY_REF8','all']
        diffs=[]
        for u in AUDIT:
            d=units[u];y=d['labels'][:,-1,qi]
            b=binary_metrics(y,d['final'][:,ARMS.index('A2'),qi]>=thresholds[group+'|A2'])
            m=binary_metrics(y,d['final'][:,ARMS.index('MOTION8'),qi]>=thresholds[group+'|MOTION8'])
            if b['ber'] is not None:diffs.append(m['ber']-b['ber'])
        diffs=np.array(diffs);rng=np.random.default_rng(20260929);ci=np.quantile(diffs[rng.integers(0,len(diffs),(2000,len(diffs)))].mean(1),[.025,.975]).tolist()
        comparisons[group]=dict(motion_ber_delta=motion['ber']-base['ber'],paired_unit_ber_ci95=ci,units=len(diffs),
            advance_motion=bool(motion['ber']<=base['ber']-.05 and ci[1]<0 and motion['tpr']>=base['tpr']-.02
             and all(motion['ber']<lookup[group,x,'all']['ber'] for x in ['STATIC8','WRONG8'])),
            reference_ber_delta=ref['ber']-base['ber'],spatial_reference_useful=bool(ref['ber']<=base['ber']-.10 and ref['fp']<=base['fp'] and ref['fn']<=base['fn']))
    overlap={g:dict(A2_errors=sum(r['A2_error'] for r in ledger if r['group']==g),
                   motion_repairs=sum('MOTION8' in r['repaired_arms'] for r in ledger if r['group']==g),
                   reference_repairs=sum('BOUNDARY_REF8' in r['repaired_arms'] for r in ledger if r['group']==g),
                   both_repairs=sum('MOTION8' in r['repaired_arms'] and 'BOUNDARY_REF8' in r['repaired_arms'] for r in ledger if r['group']==g),
                   current_methods_unresolved=sum(r['unresolved'] for r in ledger if r['group']==g)) for g in ['HEAD','BODY']}
    result=dict(status='complete',calib_units=CALIB,audit_units=AUDIT,audit_sequences=len(ay),rows=rows,interface=interface,comparisons=comparisons,overlap=overlap,
                candidate_geometric_coverage=cat(AUDIT,'coverage').mean(0).tolist(),criteria_sha256=hashlib.sha256((OUT/'DECISIONS.md').read_bytes()).hexdigest())
    save(OUT/'results.json',result)
    with (OUT/'sample_ledger.jsonl').open('w',encoding='utf-8') as f:
        for row in ledger:f.write(json.dumps(row)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in rows:
        if r['subset']=='all':print(r)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','evaluate'],required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    run(a.limit) if a.stage=='run' else evaluate()
