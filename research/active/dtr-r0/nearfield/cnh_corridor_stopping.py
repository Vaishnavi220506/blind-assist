"""Fresh, independently trained head/travel queries; open-loop first-stop evaluation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import cnh_proposal_attribution_scenes as S
import cnh_corridor_retrain as T
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-corridor-stopping-20260929'
DATA=ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data'
SPLITS={'train':list(range(6100,6148)),'calib':list(range(7100,7124)),'evaluation':list(range(8100,8148))}
FRAMES=41;SPEED=.8;DT=.2;START=-3.6;EPS=1e-9
REACTIONS=(.3,.6,1.,1.5)
BOXES=np.array([([x-.3,y0,.3],[x+.3,y1,3.]) for x in (-.3,0.,.3) for y0,y1 in ((-.2,.42),(.42,.9))])

def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def query_labels(boxes,poses):
    """Exact solid-volume OBB/AABB overlap for yaw-only query frames, no target identity."""
    low=np.array([b['lo'] for b in boxes]);high=np.array([b['hi'] for b in boxes]);bc=(low+high)/2;bh=(high-low)/2
    qc=BOXES.mean(1);qh=(BOXES[:,1]-BOXES[:,0])/2
    labels=[]
    for pose in poses:
        R=pose[:3,:3];center=qc@R.T+pose[:3,3];delta=bc[None]-center[:,None]
        good=abs(delta[:,:,1]) < (bh[None,:,1]+qh[:,None,1]-EPS)
        for axis in (np.array([1.,0,0]),np.array([0.,0,1]),R[:,0],R[:,2]):
            gap=abs(delta@axis);rb=bh@abs(axis);rq=qh@abs(R.T@axis)
            good &= gap<rq[:,None]+rb[None,:]-EPS
        labels.append(good.any(1))
    return np.array(labels,np.int8)

def first_collision(boxes):
    """Continuous straight swept body/AABB collision, independent of sensor query labels."""
    best=np.inf;ids=[]
    for i,b in enumerate(boxes):
        lo,hi=np.array(b['lo']),np.array(b['hi'])
        if min(hi[0],.3)-max(lo[0],-.3)<=EPS or min(hi[1],1.65)-max(lo[1],-.2)<=EPS:continue
        if hi[2]<=START-.15+EPS:continue
        distance=max(0.,lo[2]-.15-START)
        if distance>SPEED*(FRAMES-1)*DT+EPS:continue
        if distance<best-EPS:best=distance;ids=[i]
        elif abs(distance-best)<=EPS:ids.append(i)
    return best,ids

def scenes(unit):
    base=S.make_scenes(unit)
    timegrid=np.arange(FRAMES)*DT;position=np.column_stack([timegrid*0,timegrid*0,START+SPEED*timegrid]);mode=unit%3
    yaw=np.zeros(FRAMES) if mode==0 else np.full(FRAMES,15.) if mode==1 else 20*np.sin(np.linspace(-np.pi/2,3*np.pi/2,FRAMES))
    travel=np.array([S._pose(np.eye(3),p) for p in position]);head=np.array([S._pose(S.ry(y),p) for y,p in zip(yaw,position)])
    poses=np.array([S._pose(q[:3,:3]@S.rx(-10),p) for q,p in zip(head,position)])
    for b in base:
        b['boxes'][-1]['lo'][2]=8.;b['boxes'][-1]['hi'][2]=8.2
        b.update(travel=travel.copy(),head=head.copy(),poses=poses.copy(),mode=mode)
    return base

def collider_visibility(scene,colliders):
    if not colliders:return np.zeros(FRAMES,bool)
    rays,_=S.ray_grid();visible=[]
    for p in scene['poses']:
        hit=S.raycast_boxes(p[:3,3],rays@p[:3,:3].T,scene['boxes'])
        visible.append(bool((np.isin(hit['object_id'],colliders)&(hit['distance']<128*.0375348)).any()))
    return np.array(visible)

def setup():
    import torch
    torch.set_num_threads(2)
    import cnh_learned_features as F,cnh_track_a_gpu_readout as g
    from cnh_track_a_readout import noisy_poses
    from cnh_learned_memory_fusion import causal_ewma
    return torch,F,g,noisy_poses,causal_ewma

def generate(limit=None):
    torch,F,g,noisy_fn,smooth=setup();bias=np.load(DATA/'readouts-gpu/primary-mount-10-snr6/bias.npy');started=time.monotonic();done=0
    for split,units in SPLITS.items():
        folder=OUT/'features'/split;folder.mkdir(parents=True,exist_ok=True)
        for unit in units:
            path=folder/f'unit{unit}.npz'
            if path.exists():continue
            rows=[]
            for scene in scenes(unit):
                seed=2026092900+unit*1000+scene['config']*10
                obs=S.render(scene,seed);h,a=obs['hist'],obs['ambient'];noisy=noisy_fn(scene['poses'],seed+7,dt=DT)
                tq=np.linalg.inv(scene['travel'])@scene['poses'];hq=np.linalg.inv(scene['head'])@scene['poses']
                z4,z1,sup=F.sequence_features(h,a,bias,tq,noisy)
                headsup=(g.query_weights(torch.as_tensor(hq,dtype=g.D64,device=g.DEV))>=.75).reshape(FRAMES,6,8,8,16).cpu().numpy()
                distance,colliders=first_collision(scene['boxes']);visible=collider_visibility(scene,colliders) if split!='train' else np.zeros(FRAMES,bool)
                rows.append(dict(z4=z4.astype(np.float16),z1=z1.astype(np.float16),support_travel=sup,support_head=headsup,labels_travel=query_labels(scene['boxes'],scene['travel']),labels_head=query_labels(scene['boxes'],scene['head']),scene=np.full(FRAMES,scene['config']),frame=np.arange(FRAMES),distance=distance,visible=visible,family=scene['family'],margin=scene['margin'],collider_ids=colliders))
            payload={k:np.concatenate([r[k] for r in rows]) for k in ['z4','z1','support_travel','support_head','labels_travel','labels_head','scene','frame']}
            for k in ['distance','visible','family','margin']:payload[k]=np.array([r[k] for r in rows])
            payload.update(unit=unit,split=split);np.savez_compressed(path,**payload)
            done+=1;save(OUT/'generation_progress.json',dict(split=split,unit=unit,elapsed_s=time.monotonic()-started));print('generated',split,unit,round(time.monotonic()-started,1),flush=True)
            if limit and done>=limit:return
    save(OUT/'generation_complete.json',dict(status='complete',elapsed_s=time.monotonic()-started))

def train(kind):
    torch,F,g,noisy_fn,smooth=setup();L=T.frozen_module();dev=torch.device('cuda');out=OUT/'models'/kind;out.mkdir(parents=True,exist_ok=True)
    arrays={k:[] for k in ('x','sup','y')};inputs=[]
    for u in SPLITS['train']:
        p=OUT/'features/train'/f'unit{u}.npz'
        with np.load(p) as z:
            assert str(z['split'])=='train' and int(z['unit'])==u
            m=z['frame']>=3;a=z['z4'][m].astype(np.float32);b=z['z1'][m].astype(np.float32)
            arrays['x'].append(np.stack([L.squash(a),L.squash(b)],1));arrays['sup'].append(z[f'support_{kind}'][m]);arrays['y'].append(z[f'labels_{kind}'][m].astype(np.float32))
        inputs.append(dict(unit=u,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    x,sup,y=(np.concatenate(arrays[k]) for k in ('x','sup','y'));del arrays
    request=dict(kind=kind,recipe=T.RECIPE,inputs=inputs,train_samples=len(x),query_frame=kind,calib_access=False,eval_access=False)
    save(out/'request.json',request);start=time.monotonic()
    for seed in (0,1,2):
        target=out/f'model_seed{seed}.pt'
        if target.exists():continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);model=L.Readout().to(dev)
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20);history=[]
        resume=out/f'resume_seed{seed}.pt';start_epoch=0
        if resume.exists():
            state=torch.load(resume,map_location=dev,weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);rng.bit_generator.state=state['rng'];history=state['history'];start_epoch=len(history)
        for ep in range(start_epoch,20):
            total=0.;n=0;model.train();epstart=time.monotonic()
            for batch in T.training_batches(rng.permutation(len(x)),256,64):
                opt.zero_grad(set_to_none=True)
                for ids,weight in batch:
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.as_tensor(x[ids],device=dev),torch.as_tensor(sup[ids],device=dev)),torch.as_tensor(y[ids],device=dev))
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                    (loss*weight).backward();total+=float(loss.detach())*len(ids);n+=len(ids)
                opt.step()
            sched.step();history.append(dict(epoch=ep+1,loss=total/n,seconds=time.monotonic()-epstart));save(out/f'history_seed{seed}.json',history)
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng.bit_generator.state,history=history),resume)
            print('training',kind,seed,ep+1,round(total/n,5),round(history[-1]['seconds'],1),flush=True)
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},target);del model,opt,sched;torch.cuda.empty_cache()
    save(out/'complete.json',dict(status='complete',elapsed_s=time.monotonic()-start))

def infer():
    torch,F,g,noisy_fn,smooth=setup()
    sets={k:T.load_models([OUT/'models'/k/f'model_seed{i}.pt' for i in range(3)]) for k in ['head','travel']}
    old=T.load_models([ROOT/f'artifacts.local/work/cnh-learned-readout-20260928/seeds/model_seed{i}.pt' for i in range(3)])
    folder=OUT/'predictions';folder.mkdir(exist_ok=True)
    for split in ['calib','evaluation']:
        for u in SPLITS[split]:
            path=folder/f'unit{u}.npz'
            if path.exists():continue
            with np.load(OUT/'features'/split/f'unit{u}.npz') as z:d={k:z[k] for k in z.files}
            result={}
            for k in ['head','travel']:
                for prefix,models in [('',sets[k]),('old_',old)]:
                    raw=T.predict(models,d['z4'],d['z1'],d[f'support_{k}']);scores=smooth(raw,d['scene'],d['frame'],alpha=.5,window=5).reshape(22,FRAMES,6)
                    result[prefix+k]=scores[:,:,[2,3]].max(-1)
            result.update({k:d[k] for k in ['distance','visible','family','margin']});np.savez_compressed(path,**result);print('inferred',u,flush=True)

def episodes(scores,threshold):
    alarm=scores>=threshold;starts=alarm&~np.concatenate([np.zeros((len(alarm),1),bool),alarm[:,:-1]],1)
    return int(starts.sum()),alarm

def calibrate(scores):
    """High-to-low first budget violation, never exploit nonmonotone episode counts."""
    values=np.unique(scores);threshold=float(np.nextafter(values[-1],np.inf));minutes=scores.size*DT/60
    # Efficient exact nested activation: segments change by 1-neighbor_count.
    order=np.argsort(scores.ravel())[::-1];active=np.zeros_like(scores,bool);count=0;flat=scores.ravel();i=0
    while i<len(order):
        value=flat[order[i]];j=i
        while j<len(order) and flat[order[j]]==value:
            row,col=np.unravel_index(order[j],scores.shape);count+=1-(int(active[row,col-1]) if col>0 else 0)-(int(active[row,col+1]) if col+1<scores.shape[1] else 0);active[row,col]=True;j+=1
        if count/minutes>2:break
        threshold=float(value);i=j
    return threshold

def task_metrics(scores,threshold,distance,reaction):
    count,alarm=episodes(scores,threshold);hits=alarm.any(1);first=np.where(hits,alarm.argmax(1)*DT+7*DT,np.inf)
    hazard=np.isfinite(distance);stop=SPEED*first+SPEED*reaction+SPEED**2/3
    collide=hazard&(stop>distance);clear=~hazard
    return dict(collisions=int(collide.sum()),hazard_n=int(hazard.sum()),unnecessary=int((clear&hits).sum()),clear_n=int(clear.sum()),commands=int(hits.sum()),total=len(distance),episodes=count,minutes=scores.size*DT/60,clear_episodes=episodes(scores[clear],threshold)[0],clear_minutes=scores[clear].size*DT/60,alarm_frames=int(alarm.sum()),frames=alarm.size),collide,clear&hits

def evaluate():
    units={}
    for split in ['calib','evaluation']:
        for u in SPLITS[split]:
            with np.load(OUT/'predictions'/f'unit{u}.npz') as z:units[u]={k:z[k] for k in z.files}
    cat=lambda us,k:np.concatenate([units[u][k] for u in us]);cal=SPLITS['calib'];ev=SPLITS['evaluation'];distance=cat(ev,'distance');clearcal=np.isinf(cat(cal,'distance'));rows=[];thresholds={};diagnostics={};alarms={}
    for policy in ['head','travel','old_head','old_travel','always','never']:
        if policy in ('always','never'):
            score=np.ones((len(distance),FRAMES-7)) if policy=='always' else np.zeros((len(distance),FRAMES-7));thr=.5
        else:
            thr=calibrate(cat(cal,policy)[clearcal,7:]);score=cat(ev,policy)[:,7:];thresholds[policy]=thr
        for reaction in REACTIONS:
            m,collision,stop=task_metrics(score,thr,distance,reaction);rows.append(dict(policy=policy,reaction=reaction,**m))
            if reaction==.6:alarms[policy]=(collision,stop)
    vis=cat(ev,'visible');hazard=np.isfinite(distance);firstvis=np.where(vis.any(1),vis.argmax(1)*DT,np.inf)
    for reaction in REACTIONS:
        latest=distance/SPEED-reaction-SPEED/3
        diagnostics[str(reaction)]=dict(too_late_at_first_allowed_alert=int((hazard&(latest<1.4)).sum()),first_collider_geometrically_visible_before_deadline=int((hazard&(firstvis<=latest)).sum()),hazard_n=int(hazard.sum()))
    h=alarms['head'];t=alarms['travel'];rng=np.random.default_rng(20260929);draw=rng.integers(48,size=(2000,48));clear=(~hazard).reshape(48,22);delta=(t[1].astype(int)-h[1].astype(int)).reshape(48,22)
    numerator=delta.sum(1)[draw].sum(1);denominator=clear.sum(1)[draw].sum(1);ci=np.quantile(numerator/denominator,[.025,.975]).tolist()
    base=next(r for r in rows if r['policy']=='head' and r['reaction']==.6);new=next(r for r in rows if r['policy']=='travel' and r['reaction']==.6)
    stop_delta=new['unnecessary']/new['clear_n']-base['unnecessary']/base['clear_n'];collision_delta=new['collisions']/new['hazard_n']-base['collisions']/base['hazard_n']
    result=dict(scope='Development open-loop first-stop counterfactual, not closed-loop',splits=SPLITS,thresholds=thresholds,rows=rows,geometry=dict(initial_overlap=int((distance==0).sum()),hazard_n=int(hazard.sum()),clear_n=int((~hazard).sum()),clip_span_s=8.,diagnostics=diagnostics),comparison=dict(stop_delta=stop_delta,collision_delta=collision_delta,stop_delta_ci95=ci,paired_units=48,advance=bool(stop_delta<=-.05 and collision_delta<=.02 and ci[1]<0)),completion_and_time='NOT_EVALUABLE')
    save(OUT/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['generate','train','infer','evaluate','all']);p.add_argument('--kind',choices=['head','travel']);p.add_argument('--limit',type=int);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='all':
        generate();train('head');train('travel');infer();evaluate()
    else:{'generate':lambda:generate(a.limit),'train':lambda:train(a.kind),'infer':infer,'evaluate':evaluate}[a.stage]()
