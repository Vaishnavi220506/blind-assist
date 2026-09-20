"""Bounded existing-source feasibility, never a classifier fit or evaluation."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from reuse_corridor_geometry import classify_cube

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts.local/work/ba-existing-corridor-data-audit-20260921'
SOURCES={'body5k':ROOT/'artifacts.local/work/body-query-5000-20260909/dataset-v1',
         'body10k':ROOT/'artifacts.local/work/body-query-10000-20260909/final-dataset-v2'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def stats(values):
    return dict(n=len(values),minimum=float(min(values)),median=float(np.median(values)),maximum=float(max(values))) if values else dict(n=0)


def load_sources():
    allrows=[];cache={};inputs={};failures=[]
    for dataset,folder in SOURCES.items():
        manifest=read(folder/'manifest.json')
        assert manifest['status']=='PASS'
        for name in ('index','summary'):
            assert sha(folder/(name+'.json'))==manifest[name+'_sha256']
        for name in ('index.json','manifest.json','summary.json'):inputs[str(folder/name)]=sha(folder/name)
        for row in read(folder/'index.json')['frames']:
            row=dict(row,dataset=dataset,uid=dataset+'/'+str(row['frame_id']))
            cap=Path(row['capture'])
            if str(cap) not in cache:
                spec=read(cap/'source/spec.json');receipt=read(cap/'receipt.json')
                world=read(cap/'world-verification.json')
                assert receipt['status']==world['status']=='PASS' and receipt['source_unchanged']
                assert sha(cap/'source/spec.json')==receipt['spec_sha256']==world['source_spec_sha256']
                assert sha(cap/'receipt.json')==world['receipt_sha256']
                objects=defaultdict(dict)
                for o in receipt['controlled_objects']:
                    assert o['name'] not in objects[o['case']]
                    objects[o['case']][o['name']]=o
                cache[str(cap)]=(spec,objects,world)
                for name in ('source/spec.json','receipt.json','world-verification.json'):inputs[str(cap/name)]=sha(cap/name)
            spec,objects,world=cache[str(cap)]
            case=spec['cases'][row['sample_index']]
            assert case['name']==row['name']==world['rows'][row['sample_index']]['name']
            assert case['camera']==row['camera']
            row['_case']=case;row['_objects']=objects.get(case['name'],{})
            missing=[key for key in ('rgb_file','native_file','label_file') if not Path(row[key]).is_file()]
            row['_missing']=missing
            if missing:failures.append(dict(uid=row['uid'],missing=missing))
            allrows.append(row)
    return allrows,inputs,failures


def geometry(row):
    case=row['_case'];rendered=row['_objects'];cubes=[];errors=[]
    if {o['name'] for o in case['objects']}!=set(rendered):errors.append('DECLARED_RENDERED_NAME_MISMATCH')
    for obj in case['objects']:
        try:
            actual=rendered[obj['name']]
            assert max(abs(a-b) for a,b in zip(obj['center_m'],actual['actor_origin_m']))<.002
            assert max(abs(a-b) for a,b in zip(obj['size_m'],actual['scale']))<.002
            assert obj.get('rotation_deg',{})==actual['rotation_deg']
            c=classify_cube(case['camera'],actual)
            cubes.append(dict(name=obj['name'],target=bool(obj.get('target_part',False)),**c))
        except (AssertionError,ValueError,KeyError) as e:
            errors.append(obj['name']+':'+str(e))
    targets=[c for c in cubes if c['target']]
    near_targets=[c for c in targets if c['near_lateral_outside']]
    target_hit=any(c['intersects'] for c in targets)
    fixture_hit=any(c['intersects'] for c in cubes)
    target_out=bool(targets) and not target_hit and bool(near_targets)
    record={k:row[k] for k in ('uid','dataset','group_id','region_id','site_id','source_role','family','condition','declared_range','original_fixture_group_id')}
    record.update(geometry_valid=not errors,errors=errors,target_components=len(targets),
        target_intersects=target_hit,any_fixture_intersects=fixture_hit,near_target_lateral_outside=target_out,
        whole_fixture_negative_candidate=target_out and not fixture_hit and not errors,
        lateral_clearance_m=min((c['lateral_separation_m'] for c in near_targets),default=None),
        near_target_z_min=min((c['aabb_lo'][2] for c in near_targets),default=None),
        missing_payloads=row['_missing'])
    row['_cubes']=cubes
    return record


def select_pairs(rows,geo):
    by_group=defaultdict(list);byid={r['uid']:r for r in rows};gbyid={r['uid']:r for r in geo}
    for r in geo:by_group[(r['dataset'],r['group_id'])].append(r)
    pairs=[]
    for (dataset,group),members in sorted(by_group.items()):
        negs=[r for r in members if r['whole_fixture_negative_candidate'] and not r['missing_payloads']]
        pos=[r for r in members if r['geometry_valid'] and r['target_intersects'] and not r['missing_payloads']]
        if not negs or not pos:continue
        n=min(negs,key=lambda r:(r['condition']!='LATERAL_OUT',r['condition'],r['uid']))
        p=min(pos,key=lambda r:({'BODY_ONLY':0,'HEAD_ONLY':1,'BOTH':2}.get(r['condition'],3),r['uid']))
        assert byid[n['uid']]['camera']==byid[p['uid']]['camera']
        pairs.append(dict(dataset=dataset,family=n['family'],region=n['region_id'],site=n['site_id'],group=group,negative=n['uid'],positive=p['uid']))
    selected=[]
    strata=defaultdict(list)
    for p in pairs:strata[(p['dataset'],p['family'])].append(p)
    for key,values in sorted(strata.items()):
        regions=defaultdict(list)
        for p in sorted(values,key=lambda p:(p['region'],p['site'],p['group'])):regions[p['region']].append(p)
        count=0
        for offset in range(max(map(len,regions.values()))):
            for region in sorted(regions):
                if offset<len(regions[region]) and count<5:selected.append(regions[region][offset]);count+=1
            if count==5:break
    assert len(selected)<=40
    return pairs,selected


def inspect_payloads(rows,selected):
    import torch
    from PIL import Image
    from ba_camera_corridor import sample_native
    from tof_fov45_core import boxes45,simulate
    assert torch.cuda.is_available(),'Dense native audit requires GPU; preserve metadata if unavailable'
    torch.set_num_threads(4);device='cuda'
    yy,xx=torch.meshgrid(torch.arange(360,device=device),torch.arange(640,device=device),indexing='ij')
    f=640/(2*np.tan(np.deg2rad(50)))
    rx=(xx+.5-320)/f;ry=(yy+.5-180)/f
    limit=torch.minimum(torch.minimum(.3/rx.abs().clamp_min(1e-12),
        torch.where(ry>0,.9/ry.clamp_min(1e-12),-.2/ry.clamp_max(-1e-12))),torch.full_like(rx,3.))
    corridor_rays=limit>=.3
    byid={r['uid']:r for r in rows};answer=[]
    for uid in dict.fromkeys(p[k] for p in selected for k in ('negative','positive')):
        r=byid[uid];checks={}
        for key,digest_key in (('rgb_file','rgb_sha256'),('native_file','native_sha256'),('label_file',None)):
            expected=r[digest_key] if digest_key else r['labels']['sha256']
            checks[key]=sha(r[key])==expected
        with Image.open(r['rgb_file']) as im:
            size=im.size;im.verify()
        native=np.load(r['native_file'],allow_pickle=False)
        assert size==(640,360) and native.shape==(360,640) and native.dtype==np.float32
        z=torch.from_numpy(native).to(device);known=torch.isfinite(z)&(z>0)&(z<100)
        pts=torch.stack((rx*z,ry*z,z),-1)
        inside=known&(z>=.3)&(z<=3)&(pts[...,0].abs()<=.3)&(pts[...,1]>=-.2)&(pts[...,1]<=.9)
        target=torch.zeros_like(known)
        for cube in r['_cubes']:
            if not cube['target']:continue
            obb=cube['camera_obb']
            local=(pts-torch.tensor(obb['center'],device=device,dtype=torch.float32))@torch.tensor(obb['axes'],device=device,dtype=torch.float32)
            target|=known&(local.abs()<=torch.tensor(obb['half'],device=device)+.02).all(-1)
        values,_=simulate(sample_native(native),'existing-corridor-audit/'+uid,boxes45())
        answer.append(dict(uid=uid,hash_checks=checks,rgb_size=list(size),native_shape=list(native.shape),
            invalid_native_pixels=int((~known).sum().item()),
            corridor_intersecting_invalid_rays=int((~known&corridor_rays).sum().item()),
            current_visible_corridor_pixels=int(inside.sum().item()),
            target_obb_matched_visible_pixels=int(target.sum().item()),
            target_corridor_visible_pixels=int((target&inside).sum().item()),
            valid_single_return_zones=int((np.isfinite(values)&(values>=.1)&(values<8)).sum()),
            full_scene_negative_certified=False))
    return answer,dict(backend='CUDA',device=torch.cuda.get_device_name(),scope='Dense native geometry only; zero model inference')


def run():
    assert not OUT.exists(),'One new audit output only'
    OUT.mkdir(parents=True);started=time.perf_counter()
    (OUT/'protocol-before-run.md').write_bytes((HERE/'EXISTING_CORRIDOR_DATA_PROTOCOL_20260921.md').read_bytes())
    write(OUT/'protocol.json',dict(id='ba-existing-corridor-data-audit-20260921',time_utc=datetime.now(timezone.utc).isoformat(),
        codes={n:sha(HERE/n) for n in ('audit_existing_corridor_data.py','reuse_corridor_geometry.py','tof_fov45_core.py','ba_camera_corridor.py')},
        source_indexes={k:str(v/'index.json') for k,v in SOURCES.items()},max_pairs=40,max_payload_frames=80))
    rows,inputs,missing=load_sources();assert len(rows)==15000
    print('SOURCE_METADATA_BOUND',len(rows),flush=True)
    geo=[]
    for i,row in enumerate(rows):
        geo.append(geometry(row))
        if i%3000==0:print('GEOMETRY',i,flush=True)
    write(OUT/'input-seal.json',inputs);write(OUT/'geometry-records.json',geo);write(OUT/'missing-payloads.json',missing)
    pairs,selected=select_pairs(rows,geo)
    write(OUT/'all-candidate-pairs.json',pairs);write(OUT/'selected-pairs-before-payload-check.json',selected)
    payloads,backend=inspect_payloads(rows,selected)
    write(OUT/'payload-checks.json',payloads)
    summaries={}
    for dataset in SOURCES:
        subset=[r for r in geo if r['dataset']==dataset]
        candidates=[r for r in subset if r['whole_fixture_negative_candidate']]
        summaries[dataset]=dict(frames=len(subset),valid_geometry=sum(r['geometry_valid'] for r in subset),
            conditions=dict(Counter(r['condition'] for r in subset)),
            target_intersection=sum(r['target_intersects'] for r in subset),
            near_target_outside=sum(r['near_target_lateral_outside'] for r in subset),
            whole_fixture_negative_candidates=len(candidates),
            candidates_by_old_condition=dict(Counter(r['condition'] for r in candidates)),
            candidates_by_family=dict(Counter(r['family'] for r in candidates)),
            candidate_regions=len({r['region_id'] for r in candidates}),candidate_sites=len({r['site_id'] for r in candidates}),
            candidate_original_fixtures=len({r['original_fixture_group_id'] for r in candidates}),
            clearance_m=stats([r['lateral_clearance_m'] for r in candidates]),
            near_z_min_m=stats([r['near_target_z_min'] for r in candidates]),
            candidate_pairs=sum(p['dataset']==dataset for p in pairs))
    pd={r['uid']:r for r in payloads}
    pair_checks=[dict(**p,negative_visible=pd[p['negative']]['target_obb_matched_visible_pixels']>=3,
        negative_has_corridor_surface=pd[p['negative']]['current_visible_corridor_pixels']>0,
        positive_target_visible_in_corridor=pd[p['positive']]['target_corridor_visible_pixels']>=3,
        all_hashes_match=all(all(pd[p[k]]['hash_checks'].values()) for k in ('negative','positive'))) for p in selected]
    write(OUT/'pair-checks.json',pair_checks)
    result=dict(status='AUDIT_COMPLETE_NO_TRAINING',sources=summaries,indexed_missing_payload_frames=len(missing),
        selected_pairs=len(selected),payload_frames=len(payloads),all_sampled_hashes_match=all(all(p['hash_checks'].values()) for p in payloads),
        sampled_visible_target_pairs=sum(p['negative_visible'] and p['positive_target_visible_in_corridor'] for p in pair_checks),
        sampled_negatives_with_corridor_surface=sum(p['negative_has_corridor_surface'] for p in pair_checks),
        full_scene_negative_certified=False,backend=backend,training_steps=0,model_inference_frames=0,capture_frames=0,
        elapsed_s=time.perf_counter()-started,
        interpretation='Target/controlled-fixture geometry is distinct from full-scene absence; preserve unseen/background UNKNOWN')
    write(OUT/'result.json',result)
    write(OUT/'output-seal.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='output-seal.json'})
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()
