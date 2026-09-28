"""Frozen A2, finite-horizon first-stop proxy. See pre-result DECISIONS.md."""
import argparse, hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np

REACTIONS=(.3,.6,1.,1.5)
LENGTHS=(1.,1.2,1.4)
FREQUENCIES=(.5,1.,2.)
MISSES=(0.,.2,.5)
PHASES=(0.,np.pi/2,np.pi,3*np.pi/2)
TIMES=np.arange(81)/50

def contact(origins,directions,triangles,length):
    origins=np.asarray(origins,float); directions=np.asarray(directions,float); triangles=np.asarray(triangles,float)
    # Vectorized Moller-Trumbore with independently moving origins.
    a=triangles[:,0]; e1=triangles[:,1]-a; e2=triangles[:,2]-a
    p=np.cross(directions[:,None,:],e2[None,:,:]); det=np.sum(p*e1,axis=2)
    inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-12)
    s=origins[:,None,:]-a; u=np.sum(p*s,axis=2)*inv
    q=np.cross(s,e1); v=np.sum(directions[:,None,:]*q,axis=2)*inv
    t=np.sum(e2*q,axis=2)*inv
    return ((np.abs(det)>1e-12)&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(t>1e-8)&(t<=length+1e-8)).any(axis=1)

def collision_distance(objects,height,z0,speed):
    best=np.inf
    for obj in objects:
        pts=np.asarray(obj['triangles_world']).reshape(-1,3); lo=pts.min(0); hi=pts.max(0)
        if hi[0]<-.3 or lo[0]>.3 or hi[1]<-height or lo[1]>=0: continue
        if hi[2]<z0-.15: continue
        d=max(0.,lo[2]-.15-z0)
        if d<=speed*TIMES[-1]+1e-10: best=min(best,d)
    return best

def summarize(first,distance,speed,reaction):
    eligible=distance!=0; hazard=np.isfinite(distance)&eligible; clear=np.isinf(distance)
    commands=np.isfinite(first)&eligible
    stop_distance=speed*first+speed*reaction+speed**2/3.
    residual=hazard&(stop_distance>distance)
    return dict(collisions=int(residual.sum()),hazard_n=int(hazard.sum()),unnecessary_stops=int((commands&clear).sum()),clear_n=int(clear.sum()),commands=int(commands.sum()),eligible_n=int(eligible.sum()))

def run(data,out):
    sys.path.insert(0,str(data/'source'))
    from cnh_learned_memory_fusion import causal_ewma
    thresholds=json.loads((data/'analysis/demo_thresholds.json').read_text())['A2_thresholds']
    threshold=np.array([thresholds['HEAD'],thresholds['BODY']]*3)
    records=[]; perception=dict(positive_frames=0,detected_positive_frames=0,query_frames=0,near_events=0,timely=0,late=0,never=0,alert_episodes=0)
    started=time.monotonic()
    for unit in range(32,96):
        with np.load(data/f'predictions/unit{unit:03d}.npz') as z: d={k:z[k] for k in z.files}
        scores=causal_ewma(d['NN'],d['config'],d['frame'],alpha=.5,window=5)
        geom=json.loads((data/f'geometry/unit{unit:02d}/unit{unit:02d}.json').read_text())
        for c in geom['configs']:
            idx=np.flatnonzero(d['config']==c['config']); idx=idx[np.argsort(d['frame'][idx])][3:]
            assert np.array_equal(d['frame'][idx],np.arange(3,12))
            assert np.array_equal(d['y'][idx],np.array(c['labels'])[3:])
            pos=d['y'][idx]==1; alarms=scores[idx]>=threshold; hits=pos&alarms
            perception['positive_frames']+=int(pos.sum()); perception['detected_positive_frames']+=int(hits.sum()); perception['query_frames']+=int(pos.size)
            near=pos.any(0)&(np.min(np.where(pos,d['w'][idx],np.inf),axis=0)<=1)
            detected=hits.any(0); firsthit=hits.argmax(0); lead=d['w'][idx][firsthit,np.arange(6)]
            perception['near_events']+=int(near.sum()); perception['timely']+=int((near&detected&(lead>=1)).sum()); perception['late']+=int((near&detected&(lead<1)).sum()); perception['never']+=int((near&~detected).sum())
            merged=alarms.any(1); perception['alert_episodes']+=int((merged&~np.r_[False,merged[:-1]]).sum())
            first=float(np.flatnonzero(merged)[0]/5) if merged.any() else np.inf
            poses=np.array(c['world_from_Q']); speed=c['speed']; z0=poses[3,2,3]
            assert np.allclose(poses[:,0,3],0) and np.allclose(np.diff(poses[:,2,3]),speed/5)
            dist=collision_distance(c['objects'],c['height'],z0,speed)
            tri=np.concatenate([np.array(o['triangles_world']) for o in c['objects'] if np.array(o['triangles_world'])[:,:,1].min()<0])
            origins=np.column_stack([TIMES*0,np.full(len(TIMES),-.8),z0+speed*TIMES])
            uniforms=np.random.default_rng(20260929+unit*32+c['config']).random((4,len(TIMES)))
            cane=np.full((3,3,3,4),np.inf)
            for li,length in enumerate(LENGTHS):
                reach=np.sqrt(length**2-.8**2)
                for fi,freq in enumerate(FREQUENCIES):
                    for pi,phase in enumerate(PHASES):
                        angle=np.deg2rad(35)*np.sin(2*np.pi*freq*TIMES+phase)
                        dirs=np.column_stack([reach*np.sin(angle),np.full(len(TIMES),.8),reach*np.cos(angle)])/length
                        contacts=contact(origins,dirs,tri,length)
                        for mi,miss in enumerate(MISSES):
                            good=contacts&(uniforms[pi]>=miss)
                            if good.any(): cane[li,fi,mi,pi]=TIMES[np.flatnonzero(good)[0]]
            records.append(dict(unit=unit,config=c['config'],distance=dist,speed=speed,a2=first,cane=cane))
        if unit%8==7: print(f'Completed {unit-31}/64 units',flush=True)
    distance=np.array([r['distance'] for r in records]); speed=np.array([r['speed'] for r in records]); a2=np.array([r['a2'] for r in records]); cane=np.stack([r['cane'] for r in records])
    rows=[]
    for reaction,li,fi,mi in itertools.product(REACTIONS,range(3),range(3),range(3)):
        ca=cane[:,li,fi,mi].reshape(-1); dd=np.repeat(distance,4); ss=np.repeat(speed,4); aa=np.repeat(a2,4)
        for name,first in [('never',np.full_like(aa,np.inf)),('always',np.zeros_like(aa)),('cane',ca),('A2',aa),('A2+cane',np.minimum(aa,ca))]:
            rows.append(dict(reaction=reaction,length=LENGTHS[li],frequency=FREQUENCIES[fi],miss=MISSES[mi],policy=name,**summarize(first,dd,ss,reaction)))
    report=dict(status='COMPLETE',backend='CPU TASK_NOT_GPU_SUITABLE',units=64,configurations=len(records),phase_replicates=4,initial_overlap=int((distance==0).sum()),collision_free=int(np.isinf(distance).sum()),hazard=int(((distance>0)&np.isfinite(distance)).sum()),elapsed_s=time.monotonic()-started,thresholds=thresholds,perception=perception,scan=rows,completion_and_travel_time='NOT_EVALUABLE',decisions_sha256=hashlib.sha256((out/'DECISIONS.md').read_bytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    np.savez_compressed(out/'episode_traces.npz',distance=distance,speed=speed,a2=a2,cane=cane)
    print(json.dumps({k:v for k,v in report.items() if k!='scan'},indent=2))
    print(json.dumps([r for r in rows if r['reaction']==.6 and r['length']==1.2 and r['frequency']==1 and r['miss']==.2],indent=2))

def decompose(data,out):
    """Post-result mechanism accounting; no new scenarios or parameter choice."""
    with np.load(out/'episode_traces.npz') as z:
        dist=np.repeat(z['distance'],4); v=np.repeat(z['speed'],4)
        a=np.repeat(z['a2'],4); c=z['cane'][:,1,1,1,:].reshape(-1)
    hazard=np.isfinite(dist)&(dist>0); impact=dist/v; required=.6+v/3
    avoid=hazard&(impact>=required); touching=hazard&np.isfinite(c)
    result=dict(hazard_n=int(hazard.sum()),avoidable_n=int(avoid.sum()),too_late_at_start_n=int((hazard&~avoid).sum()),a2_residual_avoidable=int((avoid&(a+required>impact)).sum()),cane_residual_avoidable=int((avoid&(c+required>impact)).sum()),hazard_cane_contact_n=int(touching.sum()),hazard_contact_before_impact_n=int((touching&(c<impact)).sum()),hazard_contact_timely_n=int((touching&(c+required<=impact)).sum()),cane_contact_lead_quantiles_s=np.quantile(impact[touching]-c[touching],[0,.25,.5,.75,1]).tolist(),required_lead_quantiles_s=np.quantile(required[hazard],[0,.25,.5,.75,1]).tolist(),cane_forward_reach_m=float(np.sqrt(1.2**2-.8**2)))
    above=0; count=0
    for u in range(32,96):
        geom=json.loads((data/f'geometry/unit{u:02d}/unit{u:02d}.json').read_text())
        for config in geom['configs']:
            z0=config['world_from_Q'][3][2][3]
            ds=[collision_distance([o],config['height'],z0,config['speed']) for o in config['objects']]
            m=min(ds)
            if not np.isfinite(m) or m<=0: continue
            count+=1
            above+=int(all(np.array(o['triangles_world'])[:,:,1].max()<-.8 for o,d in zip(config['objects'],ds) if abs(d-m)<1e-8))
    result.update(collision_configs=count,first_collision_above_hand_configs=above)
    sys.path.insert(0,str(data/'source'))
    from cnh_learned_memory_fusion import causal_ewma
    thresholds=json.loads((data/'analysis/demo_thresholds.json').read_text())['A2_thresholds']
    central=[]
    for u in range(32,96):
        with np.load(data/f'predictions/unit{u:03d}.npz') as z: pred={k:z[k] for k in z.files}
        scores=causal_ewma(pred['NN'],pred['config'],pred['frame'],alpha=.5,window=5)
        for ci in sorted(np.unique(pred['config'])):
            idx=np.flatnonzero(pred['config']==ci); idx=idx[np.argsort(pred['frame'][idx])][3:]
            alarm=(scores[idx][:,[2,3]]>=np.array([thresholds['HEAD'],thresholds['BODY']])).any(1)
            central.append(float(np.flatnonzero(alarm)[0]/5) if alarm.any() else np.inf)
    ca=np.repeat(central,4)
    result['post_result_central_queries_only']=summarize(ca,dist,v,.6)
    result['post_result_central_plus_cane']=summarize(np.minimum(ca,c),dist,v,.6)
    result['all6_commands_absent_in_central']=int((np.isfinite(a)&~np.isfinite(ca)).sum())
    (out/'mechanism_decomposition.json').write_text(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); run(a.data,a.out); decompose(a.data,a.out)
