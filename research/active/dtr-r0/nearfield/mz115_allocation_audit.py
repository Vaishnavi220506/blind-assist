"""Evaluator-only private-hit audit of sealed finite-footprint allocation.

No predictor imports. Real contributor IDs come exclusively from native ray-hit
lineage. Projected boxes provide secondary correspondence checks, never identity.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

CORRIDOR = ((.2,3.6),(-.3,.3),(.4,2.05))


def possible(xyz): return all(hi>=a and lo<=b for (lo,hi),(a,b) in zip(xyz,CORRIDOR))
def certain(xyz): return all(lo>=a and hi<=b for (lo,hi),(a,b) in zip(xyz,CORRIDOR))
def native_bounds(obj,origin): return [(c-e-b,c+e-b) for c,e,b in zip(obj['center_m'],obj['extent_m'],origin)]
def native_truth(e): return any(possible(native_bounds(o,e['body_origin_m'])) for o in e['native_bounds'])
def contains(box,pixel): return box[0]-1e-7<=pixel[0]<=box[2]+1e-7 and box[1]-1e-7<=pixel[1]<=box[3]+1e-7


def camera_basis(camera):
    p,y,r=(math.radians(camera.get(k,0.)) for k in ('pitch','yaw','roll'))
    cp,sp,cy,sy,cr,sr=math.cos(p),math.sin(p),math.cos(y),math.sin(y),math.cos(r),math.sin(r)
    return ((cp*cy,cp*sy,sp),(sr*sp*cy-cr*sy,sr*sp*sy+cr*cy,-sr*cp),(-cr*sp*cy-sr*sy,-cr*sp*sy+sr*cy,cr*cp))


def project_point(point,camera,intr):
    delta=[point[k]-camera[name] for k,name in enumerate(('x','y','z'))]
    f,right,up=[sum(a*b for a,b in zip(axis,delta)) for axis in camera_basis(camera)]
    if f<=0 or not all(math.isfinite(x) for x in (f,right,up)):return None
    return [intr['cx']+intr['fx']*right/f,intr['cy']-intr['fy']*up/f]


def projected_box(obj,camera,intr):
    pixels=[project_point([c+s*e for c,s,e in zip(obj['center_m'],sign,obj['extent_m'])],camera,intr)
            for sign in itertools.product((-1,1),repeat=3)]
    if any(p is None for p in pixels):return None
    result=[max(0.,min(p[0] for p in pixels)),max(0.,min(p[1] for p in pixels)),
            min(intr['width'],max(p[0] for p in pixels)),min(intr['height'],max(p[1] for p in pixels))]
    return result if result[2]>result[0] and result[3]>result[1] else None


def iou(a,b):
    if a is None or b is None:return None
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def audit_target(row,e,allocation,item):
    zone_id=item['zone_id'];slot=item['target_slot']
    zone=next(z for z in row['tof_zones'] if z['zone_id']==zone_id)
    target=zone['targets'][slot]
    private=next(z for z in e['zonal_tof_native'] if z['zone_id']==zone_id)
    lineages=[x for x in private['returned_lineage'] if x['target_index']==slot]
    if len(lineages)!=1:raise ValueError('Returned target requires exactly one native lineage')
    lineage=lineages[0];indices=lineage['hit_indices']
    if len(set(indices))!=len(indices):raise ValueError('Duplicate contributor index in target lineage')
    if item['status']!=target['status'] or not math.isclose(item['range_m'],target['distance_m'],abs_tol=1e-9):
        raise ValueError('Allocation record differs from public target packet')
    objects={row['episode_id']+'/'+o['name']:o for o in e['native_bounds']}
    proposals=allocation['proposals'];j=item.get('proposal');proposal=proposals[j] if j is not None else None
    zone_box=item['zone_box'];roi=item['roi'];narrowed=any(abs(a-b)>1e-8 for a,b in zip(zone_box,roi))
    violations=[]
    if narrowed and target['status']!='SIM_VALID':violations.append('NON_VALID_TARGET_NARROWED')
    if narrowed and j is None:violations.append('NARROWED_WITHOUT_PROPOSAL')
    if not contains(zone_box,[roi[0],roi[1]]) or not contains(zone_box,[roi[2],roi[3]]):violations.append('ROI_OUTSIDE_ZONE')
    if target['status']=='SIM_MERGED' and 'range_bounds' in item and list(item['range_bounds'])!=[.02,4.]:violations.append('MERGED_FULL_RANGE_NOT_PRESERVED')
    coarse=possible(item['coarse_xyz']);preserved=certain(item['coarse_xyz']);local=preserved or possible(item['localized_xyz'])
    contributors=[]
    for hit_index in indices:
        if not isinstance(hit_index,int) or not 0<=hit_index<len(private['private_rays']):raise ValueError('Invalid native hit index')
        hit=private['private_rays'][hit_index]
        if hit.get('range_m') is None or 'hit_point_m' not in hit:raise ValueError('Selected return contributor has no native hit')
        actor=hit.get('actor_id');obj=objects.get(actor);pixel=project_point(hit['hit_point_m'],e['camera'],row['rgb_intrinsics'])
        inside=None if pixel is None else contains(roi,pixel)
        native_hit=[x-b for x,b in zip(hit['hit_point_m'],e['body_origin_m'])]
        hit_hazard=possible([(x,x) for x in native_hit])
        object_hazard=bool(obj is not None and possible(native_bounds(obj,e['body_origin_m'])))
        native_box=projected_box(obj,e['camera'],row['rgb_intrinsics']) if obj is not None else None
        contributors.append(dict(hit_index=hit_index,subray=hit.get('subray'),actor_id=actor,
            actor_status='KNOWN_NATIVE_ACTOR' if obj is not None else 'ENVIRONMENT_OR_UNMATCHED',
            native_range_m=hit['range_m'],native_hit_point_m=hit['hit_point_m'],projected_hit_pixel=pixel,
            hit_inside_roi=inside,native_hit_inside_corridor=bool(hit_hazard),native_actor_intersects_corridor=object_hazard,
            projected_actor_box=native_box,proposal_actor_iou=iou(proposal,native_box),
            roi_actor_iou=iou(roi,native_box),reflectance_proxy=hit.get('reflectance_proxy')))
    actors=sorted({c['actor_id'] for c in contributors if c['actor_status']=='KNOWN_NATIVE_ACTOR'})
    overlapping=[actor for actor,obj in objects.items() if (iou(proposal,projected_box(obj,e['camera'],row['rgb_intrinsics'])) or 0)>=.5]
    dropped=[c for c in contributors if c['hit_inside_roi'] is False]
    return dict(zone_id=zone_id,target_slot=slot,status=target['status'],group=item.get('group'),proposal=j,proposal_box=proposal,
        zone_box=zone_box,roi=roi,narrowed=narrowed,state=item['state'],coarse_support=bool(coarse),certain_coarse_preserved=bool(preserved),
        allocation_support=bool(local),target_support_suppressed=bool(coarse and not local),
        coarse_xyz=item['coarse_xyz'],localized_xyz=item['localized_xyz'],contract_violations=violations,
        contributor_actor_ids=actors,proposal_overlap_actor_ids_iou50=overlapping,
        single_contributor_actor_uniquely_box_supported=bool(len(actors)==1 and overlapping==actors and all(c['actor_status']=='KNOWN_NATIVE_ACTOR' for c in contributors)),
        contributor_count=len(contributors),dropped_contributor_count=len(dropped),
        dropped_hazard_actor_contributor_count=sum(c['native_actor_intersects_corridor'] for c in dropped),
        dropped_hazard_point_contributor_count=sum(c['native_hit_inside_corridor'] for c in dropped),
        unknown_projection_count=sum(c['hit_inside_roi'] is None for c in contributors),contributors=contributors)


def audit(rows,es,preds):
    """Primary 3.6m audit: preds has nominal/allocation frame arrays."""
    if '3.6' in preds:preds=preds['3.6']
    nominal=preds['nominal'];allocation=preds['allocation']
    if not len(rows)==len(es)==len(nominal)==len(allocation):raise ValueError('Mismatched primary frame counts')
    records=[];all_targets=[];totals=Counter();frame_changes=Counter()
    for index,(row,e,n,a) in enumerate(zip(rows,es,nominal,allocation)):
        if row['id']!=e['id']:raise ValueError('Raw/evaluator frame order mismatch')
        gt=native_truth(e);items=[audit_target(row,e,a,item) for item in a['spatial_evidence']]
        if len({(i['zone_id'],i['target_slot']) for i in items})!=len(items):raise ValueError('Duplicate allocation target pair')
        coarse=any(i['coarse_support'] for i in items);local=any(i['allocation_support'] for i in items)
        if bool(n['candidate'])!=bool(a['common_radar'] or coarse) or bool(a['candidate'])!=bool(a['common_radar'] or local):
            raise ValueError('Primary flags do not match frozen possible/certain readout and shared Radar')
        changed=bool(n['candidate'])!=bool(a['candidate']);change='UNCHANGED'
        if changed:
            change=('GAINED_TP' if gt else 'ADDED_FP') if a['candidate'] else ('LOST_TP' if gt else 'REMOVED_FP')
            frame_changes[change]+=1
        totals.update(frames=1,positive_frames=int(gt),negative_frames=int(not gt),returned_targets=len(items))
        for item in items:
            all_targets.append((index,item));totals['narrowed_targets']+=item['narrowed']
            totals['suppressed_targets']+=item['target_support_suppressed']
            totals['contract_violations']+=len(item['contract_violations'])
            if item['status']=='SIM_MERGED':totals['merged_targets']+=1
        interesting=[i for i in items if i['narrowed'] or i['target_support_suppressed'] or i['contract_violations']]
        if interesting or changed:
            records.append(dict(index=index,id=row['id'],episode_id=row['episode_id'],family=e.get('family'),native_truth=gt,
                nominal_alert=bool(n['candidate']),allocation_alert=bool(a['candidate']),common_radar=bool(a['common_radar']),
                frame_change=change,native_bounds=e['native_bounds'],targets=interesting))
    narrowed=[(index,item) for index,item in all_targets if item['narrowed']]
    suppressed=[(index,item) for index,item in all_targets if item['target_support_suppressed']]
    def contribution_summary(subset):
        return dict(target_pairs=len(subset),frame_count=len({index for index,_ in subset}),
            total_contributors=sum(i['contributor_count'] for _,i in subset),dropped_contributors=sum(i['dropped_contributor_count'] for _,i in subset),
            dropped_true_hazard_actor_contributors=sum(i['dropped_hazard_actor_contributor_count'] for _,i in subset),
            dropped_true_hazard_point_contributors=sum(i['dropped_hazard_point_contributor_count'] for _,i in subset),
            unknown_hit_projection=sum(i['unknown_projection_count'] for _,i in subset),
            targets_with_any_dropped_hazard_actor=sum(i['dropped_hazard_actor_contributor_count']>0 for _,i in subset),
            targets_with_single_real_actor_unique_box_support=sum(i['single_contributor_actor_uniquely_box_supported'] for _,i in subset),
            targets_with_environment_contributors=sum(any(c['actor_status']=='ENVIRONMENT_OR_UNMATCHED' for c in i['contributors']) for _,i in subset))
    return dict(status='EVALUATOR_ONLY_ALLOCATION_AUDIT_COMPLETE',primary_distance_m=3.6,counts=dict(totals),frame_changes=dict(frame_changes),
        narrowed=contribution_summary(narrowed),suppressed=contribution_summary(suppressed),records=records,
        limits=['Contributor identities are native private-ray lineage, never assigned from proposal overlap.',
            'Projected AABB IoU >= 0.5 is secondary correspondence support; it does not prove segmentation or continuous surface fill.',
            'Dropped contributing samples are exact private quadrature hits; they do not enumerate unsampled target surface or all physical photons.',
            'A true-hazard actor and a hit point inside the corridor are different labels, reported separately.',
            'Target-level suppression and frame-level missed alerts differ because common Radar and other ToF targets can preserve an alert.',
            'Environment hits with unknown actor ID retain their actual point geometry and are not silently relabeled as known objects.'])


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def readrows(path):return [json.loads(line) for line in path.read_text().splitlines()]
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(analysis,output):
    root=Path(__file__).resolve().parents[4];analysis=analysis.resolve();output=output.resolve()
    if not output.is_relative_to((root/'artifacts.local').resolve()) or output.exists():raise ValueError('Fresh canonical audit output required')
    seal_path=analysis/'prediction-seal.json';seal=json.loads(seal_path.read_text())
    if seal.get('status') not in ('ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE','ALL_PREDICTIONS_SEALED_BEFORE_EVALUATOR_PARSE',
                                'ALL_ARMS_AND_DISTANCE_CURVES_SEALED_BEFORE_EVALUATOR_PARSE'):
        raise ValueError('Completed prediction seal required before evaluator access')
    for name,digest in seal.get('code_sha256',{}).items():
        if sha(analysis/name)!=digest:raise ValueError('Sealed prediction code hash mismatch: '+name)
    packets={}
    # MZ115 seals one capture and the entire distance-curve dictionary at the
    # analysis root. Earlier studies use the same hash fields per named panel.
    panels=seal['panels'] if 'panels' in seal else {'mz115':seal}
    for panel,item in panels.items():
        capture=Path(item['capture'])
        predpath=analysis/panel/'predictions.json' if 'panels' in seal else analysis/'predictions.json'
        assert sha(capture/'raw.jsonl')==item['raw_sha256'] and sha(predpath)==item['predictions_sha256']
        assert sha(capture/'receipt.json')==item['receipt_sha256']
        receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS'
        for name,digest in receipt['hashes'].items():assert sha(capture/name)==digest,name
        packets[panel]=(capture,readrows(capture/'raw.jsonl'),json.loads(predpath.read_text()))
    output.mkdir(parents=True);shutil.copyfile(Path(__file__),output/Path(__file__).name);summaries={}
    for panel,(capture,rows,preds) in packets.items():
        result=audit(rows,readrows(capture/'evaluator.jsonl'),preds);write(output/(panel+'-audit.json'),result)
        summaries[panel]={k:v for k,v in result.items() if k!='records'}
    write(output/'summary.json',dict(status='EVALUATOR_ONLY_ALLOCATION_AUDIT_COMPLETE',panels=summaries))
    write(output/'completion.json',dict(status='PASS',source_seal_sha256=sha(seal_path),output_hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(summaries,indent=2))


def self_test():
    row=dict(id='synthetic',episode_id='synthetic',rgb_intrinsics=dict(width=640,height=360,fx=320.,fy=320.,cx=320.,cy=180.),
        tof_zones=[dict(zone_id=0,targets=[dict(status='SIM_VALID',distance_m=2.)])])
    obj=dict(name='shape0',center_m=[2.,0.,1.7],extent_m=[.05,.1,.1])
    e=dict(id='synthetic',camera=dict(x=0.,y=0.,z=1.7,pitch=0.,yaw=0.,roll=0.),body_origin_m=[0.,0.,0.],native_bounds=[obj],
        zonal_tof_native=[dict(zone_id=0,returned_lineage=[dict(target_index=0,hit_indices=[0])],
            private_rays=[dict(subray=0,range_m=2.,actor_id='synthetic/shape0',hit_point_m=[2.,0.,1.7],reflectance_proxy=.5)])])
    item=dict(zone_id=0,target_slot=0,status='SIM_VALID',group=0,proposal=0,zone_box=[280.,140.,360.,220.],roi=[300.,160.,340.,200.],
        range_m=2.,state='RGB_SIGNAL_ASSOCIATION_PROXY',coarse_xyz=[(1.9,2.1),(-.4,.4),(1.5,1.9)],localized_xyz=[(1.9,2.1),(-.1,.1),(1.6,1.8)])
    a=dict(proposals=[[300.,160.,340.,200.]],common_radar=False,candidate=True,spatial_evidence=[item])
    good=audit([row],[e],dict(nominal=[dict(candidate=True)],allocation=[a]))
    assert good['narrowed']['dropped_contributors']==0 and good['narrowed']['targets_with_single_real_actor_unique_box_support']==1
    wrong=dict(item,roi=[340.,160.,350.,200.],localized_xyz=[(1.9,2.1),(.4,.5),(1.6,1.8)])
    failed=audit([row],[e],dict(nominal=[dict(candidate=True)],allocation=[dict(a,candidate=False,spatial_evidence=[wrong])]))
    assert failed['frame_changes']==dict(LOST_TP=1) and failed['suppressed']['dropped_true_hazard_point_contributors']==1
    assert failed['suppressed']['dropped_true_hazard_actor_contributors']==1
    merged_row=dict(row,tof_zones=[dict(zone_id=0,targets=[dict(status='SIM_MERGED',distance_m=2.)])])
    merged=audit_target(merged_row,e,a,dict(item,status='SIM_MERGED',range_bounds=[.02,4.]))
    assert 'NON_VALID_TARGET_NARROWED' in merged['contract_violations']
    environment=dict(e,zonal_tof_native=[dict(zone_id=0,returned_lineage=[dict(target_index=0,hit_indices=[0])],private_rays=[dict(range_m=2.,actor_id=None,hit_point_m=[2.,0.,1.7])])])
    assert not audit_target(row,environment,a,item)['single_contributor_actor_uniquely_box_supported']
    return dict(status='PASS',checks=['native projection and real lineage','dropped true hazard hit and frame loss','merged narrowing violation','environment not assigned by box overlap'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    elif args.analysis and args.output:run(args.analysis,args.output)
    else:p.error('Use --self-test or --analysis and --output')
