"""Reconstructed pixel visibility -> ownership-only sealed inputs -> range readout."""
import json
import math
from pathlib import Path
import numpy as np
from mz178_current_frame import ART,sha,write
from mz115_spatial_allocation import slant_envelope,zone_box,possible
from pixel_query_labels import camera_basis,scene_boxes


def visible_mask(row,e,frame,spec):
    scene_boxes(e,dict(spec,frames=[frame]))  # source/native bounds authentication
    intr=row['rgb_intrinsics'];h,w=intr['height'],intr['width']
    origin=np.array([e['camera'][k] for k in ('x','y','z')]);basis=camera_basis(e['camera'])
    depth=np.full((h,w),np.inf);mask=np.zeros((h,w),np.uint8)
    objects=[(o,i+1) for i,o in enumerate(frame['objects'])]
    objects += [(spec[k],0) for k in ('background','floor') if spec.get(k)]
    boxes=[]
    for obj,owner in objects:
        c=np.array(obj['center_m']);s=np.array(obj['size_m']);boxes.append((c,s,owner))
        if obj.get('texture',True) and owner:
            ny,nz=obj.get('texture_grid',[8,8])
            for iy in range(ny):
                for iz in range(nz):
                    boxes.append((np.array([c[0]-s[0]/2-.001,c[1]+s[1]*((iy+.5)/ny-.5),c[2]+s[2]*((iz+.5)/nz-.5)]),
                                  np.array([.001,s[1]/ny*.98,s[2]/nz*.98]),owner))
    for c,s,owner in boxes:
        lo,hi=c-s/2,c+s/2
        corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        cam=(corners-origin)@basis.T
        if np.all(cam[:,0]<=0):continue
        if np.any(cam[:,0]<=0):x0,y0,x1,y1=0,0,w,h
        else:
            uv=np.stack([intr['cx']+intr['fx']*cam[:,1]/cam[:,0],intr['cy']-intr['fy']*cam[:,2]/cam[:,0]],1)
            x0,y0=np.maximum(0,np.floor(uv.min(0)-1)).astype(int);x1,y1=np.minimum([w,h],np.ceil(uv.max(0)+1)).astype(int)
        if x1<=x0 or y1<=y0:continue
        u,v=np.meshgrid(np.arange(x0,x1)+.5,np.arange(y0,y1)+.5)
        local=np.stack([np.ones_like(u),(u-intr['cx'])/intr['fx'],(intr['cy']-v)/intr['fy']],-1)
        d=local@basis;parallel=np.abs(d)<1e-12;den=np.where(parallel,1,d)
        aa,bb=(lo-origin)/den,(hi-origin)/den
        near=np.where(parallel,-np.inf,np.minimum(aa,bb)).max(-1)
        far=np.where(parallel,np.inf,np.maximum(aa,bb)).min(-1)
        outside=(parallel&((origin<lo)|(origin>hi))).any(-1)
        hit=np.where(near>0,near,far)
        valid=~outside&(far>=np.maximum(near,0))&(far>0)
        sub=depth[y0:y1,x0:x1];take=valid&(hit<sub)
        sub[take]=hit[take];mask[y0:y1,x0:x1][take]=owner
    return mask


def rects(binary):
    """Lossless union of pixel cells compressed into vertical run rectangles."""
    active={};out=[]
    for y,row in enumerate(binary):
        edge=np.diff(np.r_[False,row,False].astype(np.int8));runs=list(zip(np.where(edge==1)[0],np.where(edge==-1)[0]))
        current=set(runs)
        for key in list(active):
            if key not in current:out.append([key[0],active.pop(key),key[1],y])
        for key in current:active.setdefault(key,y)
    out.extend([[k[0],v,k[1],len(binary)] for k,v in active.items()])
    return out


def supported(row,yaw,zone,t,mask,owners):
    box=zone_box(zone,row['rgb_intrinsics']);w=row['rgb_intrinsics']['width'];h=row['rgb_intrinsics']['height']
    ranges=(max(.02,t['distance_m']-3*t['range_noise_sigma_m']),t['distance_m']+3*t['range_noise_sigma_m'])
    def hit(b):
        bounds=slant_envelope(b,ranges,row['rgb_intrinsics'],(row['camera_pitch_deg'],)*2,(yaw,)*2,0.)
        return possible([[a+o,b+o] for (a,b),o in zip(bounds,row['camera_in_body_m'])])
    # Independent evidence outside RGB coverage must survive.
    strips=[[box[0],box[1],min(0,box[2]),box[3]],[max(w,box[0]),box[1],box[2],box[3]],
            [max(0,box[0]),box[1],min(w,box[2]),min(0,box[3])],
            [max(0,box[0]),max(h,box[1]),min(w,box[2]),box[3]]]
    if any(b[2]>b[0] and b[3]>b[1] and hit(b) for b in strips):return True,'OUTSIDE_RGB_RETAINED'
    x0,y0=max(0,int(math.floor(box[0]))),max(0,int(math.floor(box[1])))
    x1,y1=min(w,int(math.ceil(box[2]))),min(h,int(math.ceil(box[3])))
    if not owners or x1<=x0 or y1<=y0:return True,'UNKNOWN_ID_OR_COVERAGE'
    binary=np.isin(mask[y0:y1,x0:x1],owners)
    if not binary.any():return True,'UNKNOWN_EMPTY_MASK'
    for a,b,c,d in rects(binary):
        bb=[max(box[0],a+x0),max(box[1],b+y0),min(box[2],c+x0),min(box[3],d+y0)]
        if bb[2]>bb[0] and bb[3]>bb[1] and hit(bb):return True,'MASK_POSSIBLE'
    return False,'MASK_DISJOINT'


def main():
    out=ART/'work/mz183-visible-ownership-20260918';maskdir=out/'masks';assert not maskdir.exists();maskdir.mkdir()
    cap=ART/'work/continuous-approach-20260918/capture-complete'
    spec=json.loads((cap/'spec.json').read_text());frames={f['id']:f for f in spec['frames']}
    rows=[json.loads(l) for l in (cap/'raw.jsonl').read_text().splitlines()]
    basepath=ART/'work/mz181-set-intersection-20260918/predictions-v1'
    completion=json.loads((basepath/'completion.json').read_text())
    for n,h in completion['outputs'].items():assert sha(basepath/n)==h
    base=[json.loads(l) for l in (basepath/'predictions.jsonl').read_text().splitlines()]
    receipt=json.loads((cap/'receipt.json').read_text())
    for n in ('raw.jsonl','evaluator.jsonl'):assert sha(cap/n)==receipt['hashes'][n]
    associations=[];hashes={}
    with (cap/'evaluator.jsonl').open() as ef:
        for index,(r,line) in enumerate(zip(rows,ef)):
            e=json.loads(line);assert r['id']==e['id']
            f=frames[r['id']];mask=visible_mask(r,e,f,spec)
            path=maskdir/f'{index:04d}.npz';np.savez_compressed(path,mask=mask);hashes[path.name]=sha(path)
            actor_ids={r['episode_id']+'/'+o['name']:i+1 for i,o in enumerate(f['objects'])}
            slots={}
            for z in e['zonal_tof_native']:
                for lin in z['returned_lineage']:
                    actors={z['private_rays'][i].get('actor_id') for i in lin['hit_indices']}
                    slots[f"{z['zone_id']}:{lin['target_index']}"]=sorted(actor_ids[a] for a in actors if a in actor_ids) if None not in actors else []
            associations.append(dict(id=r['id'],slots=slots))
            if index%240==0:print('mask',index,flush=True)
    write(out/'association.json',associations)
    write(out/'mask-seal.json',dict(authority='RECONSTRUCTED_RENDER_MESH_PIXEL_VISIBILITY_NOT_CAPTURED_MASK',hashes=hashes,
        association_sha256=sha(out/'association.json'),inputs={str(p):sha(p) for p in (cap/'raw.jsonl',cap/'evaluator.jsonl',cap/'spec.json',Path(__file__),Path(__file__).with_name('MZ183_PROTOCOL_20260918.md'))}))
    preds=[]
    for i,(r,b,a) in enumerate(zip(rows,base,associations)):
        assert r['id']==b['id']==a['id'];mask=np.load(maskdir/f'{i:04d}.npz')['mask'];zones={z['zone_id']:z for z in r['tof_zones']};details=[]
        for d in b['tof']:
            if not d['possible_by_depth'][-1]:continue
            zid,slot=d['zone_id'],d['target_index'];z=zones[zid]
            keep,state=supported(r,b['yaw'],z,z['targets'][slot],mask,a['slots'].get(f'{zid}:{slot}',[]))
            details.append(dict(zone_id=zid,target_index=slot,keep=keep,state=state))
        preds.append(dict(id=r['id'],episode_id=r['episode_id'],time_s=r['time_s'],astar_alert=b['astar_alert'],
            old_alert=b['alerts'][-1],alert=bool(b['astar_alert'] or b['radar'] or any(d['keep'] for d in details)),radar=b['radar'],tof=details))
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in preds))
    write(out/'completion.json',dict(status='PASS',frames=len(preds),task_truth_read=False,oracle_mask_and_identity=True,
         outputs={n:sha(out/n) for n in ('mask-seal.json','association.json','predictions.jsonl')}))


if __name__=='__main__':main()
