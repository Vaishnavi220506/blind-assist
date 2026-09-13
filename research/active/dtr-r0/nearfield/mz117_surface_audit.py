"""Evaluator-only containment audit of a conditional ToF surface hypothesis.

No predictor imports. Native returned-hit lineage establishes identity; projected
RGB overlap never supplies an actor identity or a clearance certificate.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

from mz115_allocation_audit import contains, native_bounds, native_truth, possible, certain, project_point


def inside_interval(value, bounds):
    return bounds[0]-1e-7<=value<=bounds[1]+1e-7


def support(item):
    return certain(item['coarse_xyz']) or possible(item['localized_xyz'])


def is_refined(item):
    return item['status']=='SIM_MERGED' and (
        item.get('range_bounds',[.02,4.])!=[.02,4.] or item.get('roi')!=item.get('zone_box')
        or item.get('localized_xyz')!=item.get('coarse_xyz') or bool(item.get('surface_refined'))
        or item.get('surface_state')=='CONDITIONAL_SINGLE_SURFACE_PROXY')


def audit_component(row,e,frame,item):
    """Canonical record: roi pixels, range_bounds slant metres, localized_xyz body metres."""
    zone=next(z for z in row['tof_zones'] if z['zone_id']==item['zone_id'])
    target=zone['targets'][item['target_slot']]
    assert target['status']==item['status']=='SIM_MERGED'
    assert math.isclose(target['distance_m'],item['range_m'],abs_tol=1e-9)
    private=next(z for z in e['zonal_tof_native'] if z['zone_id']==item['zone_id'])
    lineages=[x for x in private['returned_lineage'] if x['target_index']==item['target_slot']]
    assert len(lineages)==1
    indices=lineages[0]['hit_indices'];assert len(indices)==len(set(indices)) and indices
    ranges=item['range_bounds'];xyz=item['localized_xyz'];roi=item['roi']
    assert len(ranges)==2 and 0<ranges[0]<=ranges[1] and len(xyz)==3
    assert all(len(v)==2 and all(math.isfinite(x) for x in v) and v[0]<=v[1] for v in [ranges,*xyz])
    objects={row['episode_id']+'/'+o['name']:o for o in e['native_bounds']}
    contributors=[]
    for index in indices:
        hit=private['private_rays'][index];actor=hit.get('actor_id');obj=objects.get(actor)
        point=hit['hit_point_m'];body=[x-b for x,b in zip(point,e['body_origin_m'])]
        pixel=project_point(point,e['camera'],row['rgb_intrinsics'])
        spatial=all(inside_interval(x,b) for x,b in zip(body,xyz))
        slant=inside_interval(hit['range_m'],ranges)
        in_roi=None if pixel is None else contains(roi,pixel)
        hazard_hit=possible([(v,v) for v in body])
        hazard_actor=bool(obj is not None and possible(native_bounds(obj,e['body_origin_m'])))
        excluded=[]
        if in_roi is False:excluded.append('ROI')
        if not slant:excluded.append('SLANT_RANGE')
        if not spatial:excluded.append('BODY_XYZ')
        contributors.append(dict(hit_index=index,actor_id=actor,actor_status='REAL_NATIVE_ACTOR' if obj else 'ENVIRONMENT_OR_UNMATCHED',
            native_range_m=hit['range_m'],native_point_world_m=point,native_point_body_m=body,projected_pixel=pixel,
            inside_roi=in_roi,inside_slant_range=slant,inside_xyz=spatial,excluded_by=excluded,
            retained=in_roi is True and slant and spatial,native_hit_inside_corridor=hazard_hit,native_actor_intersects_corridor=hazard_actor))
    real={c['actor_id'] for c in contributors if c['actor_status']=='REAL_NATIVE_ACTOR'}
    environmental=any(c['actor_status']!='REAL_NATIVE_ACTOR' for c in contributors)
    identity='SINGLE_REAL_ACTOR' if len(real)==1 and not environmental else 'MULTIPLE_REAL_ACTORS' if len(real)>1 and not environmental else 'REAL_PLUS_ENVIRONMENT' if real else 'ENVIRONMENT_ONLY'
    dropped=[c for c in contributors if c['excluded_by']]
    anchors=[]
    for key in item.get('plane_model',{}).get('anchor_keys',[]):
        zone_id,slot=key
        anchor_public=next(z for z in row['tof_zones'] if z['zone_id']==zone_id)
        anchor_private=next(z for z in e['zonal_tof_native'] if z['zone_id']==zone_id)
        matches=[l for l in anchor_private['returned_lineage'] if l['target_index']==slot]
        assert len(matches)==1
        actor_ids={anchor_private['private_rays'][j].get('actor_id') for j in matches[0]['hit_indices']}
        anchors.append(dict(zone_id=zone_id,target_slot=slot,status=anchor_public['targets'][slot]['status'],
            public_zone_target_count=len(anchor_public['targets']),native_actor_ids=sorted(x for x in actor_ids if x is not None),
            has_environment_or_unmatched=any(x not in objects for x in actor_ids)))
    anchor_actors={x for a in anchors for x in a['native_actor_ids'] if x in objects}
    return dict(zone_id=item['zone_id'],target_slot=item['target_slot'],proposal=item.get('plane_proposal',item.get('proposal')),
        state=item.get('state'),identity_class=identity,contributor_actor_ids=sorted(real),
        roi=roi,range_bounds=ranges,coarse_xyz=item['coarse_xyz'],localized_xyz=xyz,
        original_range_bounds=item.get('original_range_bounds'),original_localized_xyz=item.get('original_localized_xyz'),
        coarse_support=possible(item['coarse_xyz']),refined_support=support(item),
        component_support_suppressed=possible(item['coarse_xyz']) and not support(item),
        contributor_count=len(contributors),dropped_contributor_count=len(dropped),
        dropped_hazard_hit_count=sum(c['native_hit_inside_corridor'] for c in dropped),
        dropped_hazard_actor_count=sum(c['native_actor_intersects_corridor'] for c in dropped),
        unknown_projection_count=sum(c['inside_roi'] is None for c in contributors),
        exclusion_counts_nonexclusive=Counter(reason for c in contributors for reason in c['excluded_by']),
        contributors=contributors,anchors=anchors,anchor_native_actor_ids=sorted(anchor_actors),
        contributor_native_actor_ids_absent_from_anchors=sorted(real-anchor_actors) if anchors else None,
        conditional_plane_diagnostics={k:v for k,v in item.items() if 'plane' in k or 'surface' in k})


def audit(rows,es,preds,baseline='resolution_guard',candidate='surface'):
    if '3.6' in preds:preds=preds['3.6']
    previous=preds[baseline];current=preds[candidate]
    assert len(rows)==len(es)==len(previous)==len(current)
    counts=Counter();changes=Counter();records=[]
    for index,(row,e,b,a) in enumerate(zip(rows,es,previous,current)):
        assert row['id']==e['id']
        gt=native_truth(e);items=[audit_component(row,e,a,t) for t in a['spatial_evidence'] if is_refined(t)]
        assert len({(t['zone_id'],t['target_slot']) for t in items})==len(items)
        change='UNCHANGED' if a['candidate']==b['candidate'] else ('GAINED_TP' if gt else 'ADDED_FP') if a['candidate'] else ('LOST_TP' if gt else 'REMOVED_FP')
        changes[change]+=1;counts['frames']+=1
        if not items and change=='UNCHANGED':continue
        counts['frames_with_refined_merged']+=bool(items);counts['refined_merged_targets']+=len(items)
        loss=any(t['dropped_hazard_hit_count'] for t in items)
        counts['frames_with_dropped_hazard_hits']+=loss
        counts['frames_with_dropped_hazard_hits_but_alert_retained']+=bool(loss and a['candidate'])
        counts['frames_with_dropped_hazard_hits_and_common_radar']+=bool(loss and a.get('common_radar'))
        for t in items:
            counts['identity_'+t['identity_class']]+=1
            for name in ('contributor_count','dropped_contributor_count','dropped_hazard_hit_count','dropped_hazard_actor_count','unknown_projection_count'):
                counts[name]+=t[name]
            counts['suppressed_components']+=t['component_support_suppressed']
            counts['suppressed_components_with_dropped_hazard_hits']+=bool(t['component_support_suppressed'] and t['dropped_hazard_hit_count'])
            for reason,n in t['exclusion_counts_nonexclusive'].items():counts['excluded_'+reason]+=n
        records.append(dict(index=index,id=row['id'],episode_id=row['episode_id'],family=e.get('family'),native_truth=gt,
            baseline_alert=b['candidate'],candidate_alert=a['candidate'],common_radar=a.get('common_radar'),frame_change=change,targets=items))
    return dict(status='EVALUATOR_ONLY_CONDITIONAL_SURFACE_AUDIT_COMPLETE',baseline_arm=baseline,candidate_arm=candidate,
        counts=counts,frame_changes=changes,records=records,
        limits=['The fitted plane is a conditional proxy, not a certified empty-space or contributor-containment guarantee.',
            'ROI, native slant range, and initial-body-frame XYZ containment are audited separately and jointly.',
            'Returned-hit lineage establishes native identity; no actor identity is assigned using a projected RGB box.',
            'A hazardous actor and a contributing point inside the corridor are distinct, separately counted facts.',
            'Native finite quadrature cannot enumerate unsampled surfaces or all physical photons.',
            'A frame alert retained by Radar or another target does not excuse discarded hazardous contributors.'])


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(analysis,output,baseline='resolution_guard',candidate='surface'):
    root=Path(__file__).resolve().parents[4];analysis=analysis.resolve();output=output.resolve()
    if not output.is_relative_to((root/'artifacts.local').resolve()) or output.exists():raise ValueError('Fresh canonical audit output required')
    sealpath=analysis/'prediction-seal.json';seal=json.loads(sealpath.read_text())
    if seal.get('status') not in ('ALL_ARMS_AND_DISTANCE_CURVES_SEALED_BEFORE_EVALUATOR_PARSE',
            'ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE','ALL_PREDICTIONS_SEALED_BEFORE_EVALUATOR_PARSE'):
        raise ValueError('Completed prediction seal required before evaluator access')
    for name,digest in seal.get('code_sha256',{}).items():assert sha(analysis/name)==digest
    panels=seal.get('panels',{'source':seal});packets={}
    for panel,item in panels.items():
        capture=Path(item['capture']);predpath=analysis/panel/'predictions.json' if 'panels' in seal else analysis/'predictions.json'
        assert sha(capture/'raw.jsonl')==item['raw_sha256'] and sha(capture/'receipt.json')==item['receipt_sha256']
        assert sha(predpath)==item['predictions_sha256']
        receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS'
        for name,digest in receipt['hashes'].items():assert sha(capture/name)==digest
        packets[panel]=(capture,readrows(capture/'raw.jsonl'),json.loads(predpath.read_text()))
    output.mkdir(parents=True);shutil.copyfile(Path(__file__),output/Path(__file__).name);summary={}
    for panel,(capture,rows,preds) in packets.items():
        result=audit(rows,readrows(capture/'evaluator.jsonl'),preds,baseline,candidate)
        write(output/(panel+'-audit.json'),result);summary[panel]={k:v for k,v in result.items() if k!='records'}
    write(output/'summary.json',dict(status='EVALUATOR_ONLY_CONDITIONAL_SURFACE_AUDIT_COMPLETE',panels=summary))
    write(output/'completion.json',dict(status='PASS',source_seal_sha256=sha(sealpath),output_hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(summary,indent=2))


def self_test():
    row=dict(id='synthetic',episode_id='synthetic',rgb_intrinsics=dict(width=640,height=360,fx=320.,fy=320.,cx=320.,cy=180.),
        tof_zones=[dict(zone_id=0,targets=[dict(status='SIM_MERGED',distance_m=3.1)])])
    e=dict(id='synthetic',camera=dict(x=0.,y=0.,z=1.7,pitch=0.,yaw=0.,roll=0.),body_origin_m=[0.,0.,0.],
        native_bounds=[dict(name='near',center_m=[2.5,0.,1.7],extent_m=[.05,.05,.05]),dict(name='far',center_m=[3.2,.6,1.7],extent_m=[.05,.05,.05])],
        zonal_tof_native=[dict(zone_id=0,returned_lineage=[dict(target_index=0,hit_indices=[0,1])],private_rays=[
            dict(range_m=2.5,actor_id='synthetic/near',hit_point_m=[2.5,0.,1.7]),
            dict(range_m=math.hypot(3.2,.6),actor_id='synthetic/far',hit_point_m=[3.2,.6,1.7])])])
    item=dict(zone_id=0,target_slot=0,status='SIM_MERGED',range_m=3.1,range_bounds=[3.,3.4],roi=[300,160,400,200],zone_box=[280,140,420,220],
        coarse_xyz=[[.02,4.],[-.7,.7],[.2,2.5]],localized_xyz=[[3.,3.4],[.5,.7],[1.5,1.9]],proposal=0,state='CONDITIONAL_SURFACE_PROXY')
    current=dict(candidate=True,common_radar=True,spatial_evidence=[item]);preds=dict(resolution_guard=[dict(candidate=True)],surface=[current])
    result=audit([row],[e],preds);t=result['records'][0]['targets'][0]
    assert t['identity_class']=='MULTIPLE_REAL_ACTORS' and t['dropped_hazard_hit_count']==1
    assert t['exclusion_counts_nonexclusive']==dict(SLANT_RANGE=1,BODY_XYZ=1)
    assert result['counts']['frames_with_dropped_hazard_hits_but_alert_retained']==1
    assert result['counts']['frames_with_dropped_hazard_hits_and_common_radar']==1
    assert result['frame_changes']==dict(UNCHANGED=1)
    loss=audit([row],[e],dict(resolution_guard=[dict(candidate=True)],surface=[dict(current,candidate=False,common_radar=False)]))
    assert loss['frame_changes']==dict(LOST_TP=1)
    full=dict(item,range_bounds=[.02,4.],roi=item['zone_box'],localized_xyz=item['coarse_xyz'])
    assert not is_refined(full)
    roi_loss=audit_component(row,e,current,dict(item,roi=[370,160,400,200],range_bounds=[.02,4.],localized_xyz=item['coarse_xyz']))
    assert roi_loss['exclusion_counts_nonexclusive']==dict(ROI=1)
    anchor_row=dict(row,tof_zones=row['tof_zones']+[dict(zone_id=1,targets=[dict(status='SIM_VALID',distance_m=3.2)])])
    anchor_e=dict(e,zonal_tof_native=e['zonal_tof_native']+[dict(zone_id=1,returned_lineage=[dict(target_index=0,hit_indices=[0])],
        private_rays=[e['zonal_tof_native'][0]['private_rays'][1]])])
    anchored=audit_component(anchor_row,anchor_e,current,dict(item,plane_model=dict(anchor_keys=[[1,0]])))
    assert anchored['anchor_native_actor_ids']==['synthetic/far']
    assert anchored['contributor_native_actor_ids_absent_from_anchors']==['synthetic/near']
    return dict(status='PASS',checks=['mixed native actor lineage','range/XYZ exclusion with ROI retained',
        'hazard contributor loss hidden by common Radar','frame-level TP loss','unrefined merged fallback','ROI-only exclusion',
        'near contributor absent from actual anchor identities'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--baseline',default='resolution_guard');p.add_argument('--candidate',default='surface');p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    elif args.analysis and args.output:run(args.analysis,args.output,args.baseline,args.candidate)
    else:p.error('Use --self-test or --analysis and --output')
