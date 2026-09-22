"""Oracle object-ID association ceiling; no native geometry enters readout."""
import json
import math
from pathlib import Path
import hashlib
from collections import Counter


def union(intervals):
    out=[]
    for lo,hi in sorted(intervals):
        if out and lo<=out[-1][1]:out[-1][1]=max(hi,out[-1][1])
        else:out.append([lo,hi])
    return out


def intersect(a,b):
    return union([[max(x,u),min(y,v)] for x,y in a for u,v in b if max(x,u)<=min(y,v)])


def clip(poly,a,b,c):
    result=[]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        vp=a*p[0]+b*p[1]-c;vq=a*q[0]+b*q[1]-c
        if vp>=0:result.append(p)
        if (vp>=0)!=(vq>=0):
            t=vp/(vp-vq);result.append([p[i]+t*(q[i]-p[i]) for i in (0,1)])
    return result


def feasible(angles,radius):
    for lo,hi in angles:
        poly=[[.2,-.3],[3.6,-.3],[3.6,.3],[.2,.3]]
        lo,hi=math.radians(lo),math.radians(hi)
        poly=clip(poly,-math.sin(lo),math.cos(lo),0)
        if not poly:continue
        poly=clip(poly,math.sin(hi),-math.cos(hi),0)
        if not poly:continue
        ds=[math.hypot(*p) for p in poly]
        for p,q in zip(poly,poly[1:]+poly[:1]):
            v=[q[i]-p[i] for i in (0,1)];norm=sum(x*x for x in v)
            t=max(0,min(1,-sum(p[i]*v[i] for i in (0,1))/norm)) if norm else 0
            ds.append(math.hypot(*(p[i]+t*v[i] for i in (0,1))))
        if min(ds)<=radius[1]+1e-10 and max(ds)>=radius[0]-1e-10:return True
    return False


def slots(public,identity_only):
    angles=[];radii=[];selected=[]
    for z in public['tof_zones']:
        for i,t in enumerate(z['targets']):
            if t['status']!='SIM_VALID' or (z['zone_id'],i) not in identity_only:continue
            angles.append([a+public['yaw_deg'] for a in z['theta_bounds_deg']])
            # Conservative horizontal-radius enclosure of the full zone; no native range.
            phi=max(abs(a) for a in z['phi_bounds_deg'])
            low=max(.02,t['distance_m']-3*t['range_noise_sigma_m'])*math.cos(math.radians(phi))
            high=t['distance_m']+3*t['range_noise_sigma_m']
            radii.append([low,high]);selected.append([z['zone_id'],i])
    return dict(angles=union(angles),radius=[min(r[0] for r in radii),max(r[1] for r in radii)] if radii else None,slots=selected)


def readout(phases):
    baseline=bool(phases[0]['angles'] and feasible(phases[0]['angles'],phases[0]['radius']))
    if any(not p['angles'] for p in phases):return dict(state='UNKNOWN_MISSING',alert=baseline,angles=[])
    angles=phases[0]['angles']
    for p in phases[1:]:angles=intersect(angles,p['angles'])
    if not angles:return dict(state='UNKNOWN_EMPTY',alert=baseline,angles=[])
    # Same object need not return the same surface point/range. Preserve all phase ranges.
    radius=[min(p['radius'][0] for p in phases),max(p['radius'][1] for p in phases)]
    return dict(state='POSSIBLE' if feasible(angles,radius) else 'OUTSIDE_HORIZONTAL',
                alert=feasible(angles,radius),angles=angles,radius=radius)


def main():
    root=Path('artifacts.local/work/tof-dither-20260918')
    cap=root/'capture';receipt=json.loads((cap/'receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['frames']==144
    for name,h in receipt['hashes'].items():assert hashlib.sha256((cap/name).read_bytes()).hexdigest()==h
    public=[json.loads(l) for l in (cap/'raw.jsonl').read_text().splitlines()]
    native=[json.loads(l) for l in (cap/'evaluator.jsonl').read_text().splitlines()]
    assert len(public)==len(native)==144
    entries={};association=[]
    for p,e in zip(public,native):
        assert p['id']==e['id']
        ids=set()
        for z in e['private']:
            for lin in z['returned_lineage']:
                actors={z['private_rays'][i].get('actor_id') for i in lin['hit_indices']}
                if actors=={p['scene_id']}:ids.add((z['zone_id'],lin['target_index']))
        association.append(dict(id=p['id'],slots=sorted(ids)))
        entries[p['id']]=slots(p,ids)
    # Identity-only oracle predictions sealed before source truth classification.
    predictions=[]
    for scene in sorted({p['scene_id'] for p in public}):
        d=[entries[f'{scene}_dither_{i}'] for i in range(3)]
        r=[entries[f'{scene}_repeat_{i}'] for i in range(3)]
        assert d[0]==r[0]
        predictions.append(dict(scene=scene,single=readout([d[0]]),repeat=readout(r),dither=readout(d),phases=dict(dither=d,repeat=r)))
    (root/'oracle-predictions.json').write_text(json.dumps(predictions,indent=2))
    (root/'association.json').write_text(json.dumps(association))
    summary={a:Counter() for a in ('single','repeat','dither')};cases=[]
    for result in predictions:
        e=next(e for e in native if e['scene_id']==result['scene'])
        c=e['native']['center'];ext=e['native']['extent']
        truth=c[1]-ext[1]<=.3 and c[1]+ext[1]>=-.3 and c[0]-ext[0]<=3.6 and c[0]+ext[0]>=.2
        corners=[math.degrees(math.atan2(y,x)) for y in (c[1]-ext[1],c[1]+ext[1]) for x in (c[0]-ext[0],c[0]+ext[0])]
        actual=[min(corners),max(corners)]
        for arm in summary:
            v=result[arm];a=v['alert'];s=summary[arm]
            s['TP' if truth and a else 'FN' if truth else 'FP' if a else 'TN']+=1
            s['UNKNOWN']+=v['state'].startswith('UNKNOWN')
            if result['single']['alert']:
                s['baseline_tp_retained']+=truth and a
                s['baseline_fp_removed']+=not truth and not a
            if v['angles']:
                s['native_bearing_overlap']+=bool(intersect(v['angles'],[actual]))
                s['native_extent_not_fully_retained']+=sum(b-a for a,b in intersect(v['angles'],[actual]))<actual[1]-actual[0]-1e-8
        cases.append(dict(**result,truth=truth,native_bearing=actual))
    s=summary['single'];d=summary['dither']
    gate=bool(s['TP'] and s['FP'] and d['baseline_tp_retained']/s['TP']>=.95 and d['baseline_fp_removed']/s['FP']>=.5)
    report=dict(status='PASS',decision='CEILING_GATE_MET_CONDITIONAL' if gate else 'STOP_FIXED_DITHER_CEILING_GATE_NOT_MET',
        summary=summary,gate=gate,cases=cases,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(summary=summary,gate=gate),indent=2))


if __name__=='__main__':main()
