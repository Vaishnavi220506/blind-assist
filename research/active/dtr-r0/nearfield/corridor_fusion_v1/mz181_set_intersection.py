"""Public-only bounded angular subdivision; unresolved support stays possible."""
import argparse
import json
import math
import time
from pathlib import Path
from mz178_current_frame import ART, sha, write, support_readout, finite
from mz115_spatial_allocation import slant_envelope, zone_box

CORRIDOR = ((.2,3.6),(-.3,.3),(.4,2.05))
EPS = 1e-10
MAX_DEPTH = 6


def overlapping(bounds):
    return all(lo <= b+EPS and hi >= a-EPS for (lo,hi),(a,b) in zip(bounds,CORRIDOR))


def direction(theta, phi, pitch, yaw):
    a,b = math.tan(math.radians(theta)), math.tan(math.radians(phi))
    p,y = math.radians(pitch), math.radians(yaw)
    f,u = math.cos(p)-b*math.sin(p), math.sin(p)+b*math.cos(p)
    n = math.sqrt(1+a*a+b*b)
    return ((f*math.cos(y)-a*math.sin(y))/n,
            (f*math.sin(y)+a*math.cos(y))/n,u/n)


def ray_witness(theta, phi, ranges, row, yaw):
    """One ray suffices for acceptance, never for rejection of the zone."""
    d=direction(theta,phi,row['camera_pitch_deg'],yaw)
    lo,hi=ranges
    for v,o,(a,b) in zip(d,row['camera_in_body_m'],CORRIDOR):
        if v==0:
            if not a <= o <= b: return None
        else:
            near,far=sorted(((a-o)/v,(b-o)/v))
            lo,hi=max(lo,near),min(hi,far)
    if lo>hi: return None
    r=(lo+hi)/2
    point=[o+r*v for o,v in zip(row['camera_in_body_m'],d)]
    if not all(a <= v <= b for v,(a,b) in zip(point,CORRIDOR)): return None
    return dict(theta_deg=theta,phi_deg=phi,range_m=r,point=point)


def refine(row, zone, target, yaw):
    ranges=(max(.02,target['distance_m']-3*target['range_noise_sigma_m']),
            target['distance_m']+3*target['range_noise_sigma_m'])
    visits=[0]*(MAX_DEPTH+1)
    hits=[False]*(MAX_DEPTH+1)
    unresolved=0
    witness=None

    def walk(theta,phi,depth,parent=None):
        nonlocal unresolved,witness
        visits[depth]+=1
        z=dict(zone,theta_bounds_deg=theta,phi_bounds_deg=phi)
        bounds=slant_envelope(zone_box(z,row['rgb_intrinsics']),ranges,row['rgb_intrinsics'],
                              (row['camera_pitch_deg'],)*2,(yaw,)*2,0.)
        bounds=[[a+o,b+o] for (a,b),o in zip(bounds,row['camera_in_body_m'])]
        if parent is not None:
            bounds=[[max(a,c),min(b,d)] for (a,b),(c,d) in zip(bounds,parent)]
        if not overlapping(bounds): return
        hits[depth]=True
        tm,pm=sum(theta)/2,sum(phi)/2
        w=ray_witness(tm,pm,ranges,row,yaw)
        if w is not None:
            witness=w
            for j in range(depth,MAX_DEPTH+1): hits[j]=True
            return
        if depth==MAX_DEPTH:
            unresolved+=1
            return
        for t in ((theta[0],tm),(tm,theta[1])):
            for p in ((phi[0],pm),(pm,phi[1])):
                walk(t,p,depth+1,bounds)
                if witness is not None: return

    walk(zone['theta_bounds_deg'],zone['phi_bounds_deg'],0)
    assert all(not hits[i+1] or hits[i] for i in range(MAX_DEPTH))
    return dict(possible_by_depth=hits,visits=visits,unresolved_leaves=unresolved,
                witness=witness,state='WITNESS' if witness else 'UNRESOLVED_POSSIBLE' if hits[-1] else 'DISJOINT')


def public_frame(row, yaw):
    baseline=support_readout(row,yaw)
    clean=[c for c in baseline['contributors'] if c.get('status')!='SIM_MERGED']
    zones={z['zone_id']:z for z in row['tof_zones']}
    details=[]
    for c in clean:
        if c['sensor']=='tof':
            z=zones[c['zone_id']]
            details.append(dict(zone_id=c['zone_id'],target_index=c['target_index'],
                                **refine(row,z,z['targets'][c['target_index']],yaw)))
    return clean, details


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert out.is_relative_to(ART) and not out.exists()
    root=ART/'work/continuous-approach-20260918'
    raw=root/'capture-complete/raw.jsonl';base=root/'baselines-v1/predictions.jsonl'
    reference=ART/'work/mz180-public-check-20260918'
    seal=json.loads((reference/'public-seal.json').read_text())
    assert sha(reference/'public-predictions.jsonl')==seal['predictions_sha256']
    receipt=json.loads((root/'capture-complete/receipt.json').read_text())
    assert receipt['status']=='PASS' and sha(raw)==receipt['hashes']['raw.jsonl']
    completed=json.loads((root/'baselines-v1/completion.json').read_text())
    assert completed['status']=='PASS' and not completed['evaluator_read']
    for name,h in completed['outputs'].items(): assert sha(base.parent/name)==h
    here=Path(__file__).parent
    inputs={str(p):sha(p) for p in (raw,base,reference/'public-predictions.jsonl',Path(__file__),
        here/'MZ181_PROTOCOL_20260918.md',here/'mz178_current_frame.py',
        here.parent/'mz115_spatial_allocation.py',here.parent/'mz109_interval_extent.py')}
    rows=[json.loads(l) for l in raw.read_text().splitlines()]
    bs=[json.loads(l) for l in base.read_text().splitlines()]
    refs=[json.loads(l) for l in (reference/'public-predictions.jsonl').read_text().splitlines()]
    assert len(rows)==len(bs)==len(refs)==1920
    results=[];ep=None;yaw=0.
    for row,b,ref in zip(rows,bs,refs):
        assert all(row[k]==b[k]==ref[k] for k in ('id','episode_id','time_s'))
        if ep!=row['episode_id']: ep=row['episode_id'];yaw=0.
        if row['imu_valid']: yaw+=row['delta_yaw']
        assert finite(yaw) and abs(yaw-b['yaw'])<1e-10
        start=time.perf_counter();clean,details=public_frame(row,yaw);elapsed=time.perf_counter()-start
        assert clean==ref['contributors']
        radar=[c for c in clean if c['sensor']=='radar']
        alerts=[bool(b['astar_alert'] or radar or any(d['possible_by_depth'][i] for d in details)) for i in range(MAX_DEPTH+1)]
        assert alerts[0]==ref['alert']
        results.append(dict(id=row['id'],episode_id=ep,time_s=row['time_s'],yaw=yaw,
            astar_alert=b['astar_alert'],unknown_alert=ref['alert'],alerts=alerts,radar=radar,
            tof=details,readout_s=elapsed,geometry_state='POSSIBLE_OCCUPANCY' if radar or any(d['possible_by_depth'][-1] for d in details) else 'UNKNOWN'))
    assert all(sha(p)==h for p,h in inputs.items())
    out.mkdir(parents=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in results),encoding='utf-8')
    write(out/'input-seal.json',dict(inputs=inputs,evaluator_read=False,backend='CPU',backend_reason='TASK_NOT_GPU_SUITABLE',max_depth=MAX_DEPTH))
    write(out/'completion.json',dict(status='PASS',frames=len(results),evaluator_read=False,
          outputs={n:sha(out/n) for n in ('predictions.jsonl','input-seal.json')}))
    print(json.dumps(dict(status='PASS',frames=len(results),readout_s=sum(r['readout_s'] for r in results))))


if __name__=='__main__': main()
