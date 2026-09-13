"""Post-seal native-pose and ray/AABB audit; never imported by inference.

Translation uses current CV axes (right, down, forward). Point intervals are
slant ranges along unit current-camera rays. AABB first hits are a geometric
reference, not pixel segmentation truth or a physical sensor guarantee.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

MAX_PAIR_AGE_S=.75
EPS=1e-7
CORRIDOR=((.2,3.6),(-.3,.3),(.4,2.05))


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(v):return math.sqrt(dot(v,v))
def finite_vector(v,n=3):return isinstance(v,(list,tuple)) and len(v)==n and all(isinstance(x,(int,float)) and math.isfinite(x) for x in v)
def interval_contains(value,b):return b[0]-EPS<=value<=b[1]+EPS
def valid_interval(b):return finite_vector(b,2) and b[0]<=b[1]
def point_hazard(point,origin):return all(lo<=x-o<=hi for x,o,(lo,hi) in zip(point,origin,CORRIDOR))
def object_hazard(obj,origin):return all(c+e-o>=lo and c-e-o<=hi for c,e,o,(lo,hi) in zip(obj['center_m'],obj['extent_m'],origin,CORRIDOR))


def cv_basis(camera):
    """World-coordinate unit axes, returned in CV right/down/forward order."""
    pitch,yaw,roll=[math.radians(camera.get(k,0.)) for k in ('pitch','yaw','roll')]
    cp,sp,cy,sy,cr,sr=math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw),math.cos(roll),math.sin(roll)
    forward=(cp*cy,cp*sy,sp)
    right=(sr*sp*cy-cr*sy,sr*sp*sy+cr*cy,-sr*cp)
    up=(-cr*sp*cy-sr*sy,-cr*sp*sy+sr*cy,cr*cp)
    return right,tuple(-x for x in up),forward


def native_translation(reference,current):
    displacement=[reference[k]-current[k] for k in ('x','y','z')]
    return [dot(axis,displacement) for axis in cv_basis(current)]


def pixel_ray(pixel,intr,camera):
    q=[(pixel[0]-intr['cx'])/intr['fx'],(pixel[1]-intr['cy'])/intr['fy'],1.]
    length=norm(q);q=[x/length for x in q];basis=cv_basis(camera)
    return [sum(q[j]*basis[j][k] for j in range(3)) for k in range(3)]


def ray_aabb(origin,direction,obj):
    low=-math.inf;high=math.inf
    for o,d,c,e in zip(origin,direction,obj['center_m'],obj['extent_m']):
        lo,hi=c-e,c+e
        if abs(d)<1e-14:
            if o<lo-EPS or o>hi+EPS:return None
            continue
        t0,t1=sorted(((lo-o)/d,(hi-o)/d));low=max(low,t0);high=min(high,t1)
        if low>high+EPS:return None
    if high<=EPS:return None
    distance=low if low>EPS else high
    return distance if math.isfinite(distance) else None


def context_objects(spec,receipt):
    result=[]
    for key in ('background','floor'):
        obj=spec.get(key,receipt.get(key))
        if obj is None:raise ValueError('Missing frozen scene context: '+key)
        if key in spec and key in receipt:
            for name in ('center_m','size_m'):assert spec[key][name]==receipt[key][name]
        result.append(dict(actor_id='CONTEXT/'+key,kind='CONTEXT',source_role=key,
            center_m=obj['center_m'],extent_m=[v/2 for v in obj['size_m']]))
    return result


def first_hit(row,e,pixel,contexts,roles=None):
    if not finite_vector(pixel,2):return dict(status='INVALID_PIXEL')
    intr=row['rgb_intrinsics']
    if not(0<=pixel[0]<intr['width'] and 0<=pixel[1]<intr['height']):return dict(status='PIXEL_OUTSIDE_IMAGE')
    camera=e['camera'];origin=[camera[k] for k in ('x','y','z')];direction=pixel_ray(pixel,intr,camera)
    objects=[dict(o,actor_id=row['episode_id']+'/'+o['name'],kind='NATIVE_ACTOR',
        source_role=(roles or {}).get(row['episode_id']+'/'+o['name'])) for o in e['native_bounds']]+contexts
    hits=[]
    for obj in objects:
        distance=ray_aabb(origin,direction,obj)
        if distance is not None:hits.append((distance,obj))
    if not hits:return dict(status='NO_KNOWN_AABB_HIT',world_ray=direction)
    hits.sort(key=lambda x:(x[0],x[1]['actor_id']));distance=hits[0][0]
    nearest=[o for d,o in hits if abs(d-distance)<=EPS]
    point=[o+distance*d for o,d in zip(origin,direction)]
    return dict(status='FIRST_AABB_HIT' if len(nearest)==1 else 'TIED_FIRST_AABB_HIT',range_m=distance,
        actor_ids=[o['actor_id'] for o in nearest],kinds=[o['kind'] for o in nearest],source_roles=[o['source_role'] for o in nearest],
        world_ray=direction,point_world_m=point,point_inside_corridor=point_hazard(point,e['body_origin_m']),
        foreground_native_actor=any(o['kind']=='NATIVE_ACTOR' for o in nearest),
        actor_intersects_corridor=any(o['kind']=='NATIVE_ACTOR' and object_hazard(o,e['body_origin_m']) for o in nearest))


def reference_issues(rows,current_index,reference_index):
    if not isinstance(reference_index,int) or isinstance(reference_index,bool) or not 0<=reference_index<len(rows):return ['INVALID_REFERENCE_INDEX']
    issues=[]
    if reference_index>=current_index:issues.append('NONCAUSAL_REFERENCE')
    current,reference=rows[current_index],rows[reference_index]
    if current['episode_id']!=reference['episode_id']:issues.append('CROSS_EPISODE_REFERENCE')
    dt=current['time_s']-reference['time_s']
    if not math.isfinite(dt) or not 0<dt<=MAX_PAIR_AGE_S+EPS:issues.append('INVALID_OR_STALE_PAIR_AGE')
    return issues


def audit(rows,es,preds,spec,receipt,baseline='resolution_guard',candidate='parallax',geometry=None):
    if '3.6' in preds:preds=preds['3.6']
    before,after=preds[baseline],preds[candidate]
    assert len(rows)==len(es)==len(before)==len(after)
    if geometry is not None:assert len(geometry)==len(rows)
    assert [r['id'] for r in rows]==[e['id'] for e in es]
    contexts=context_objects(spec,receipt)
    cohorts={a['episode']:a['camera_motion_mode'] for a in spec.get('source_audit',[])}
    roles={f['episode']+'/'+o['name']:o.get('source_role') for f in spec.get('frames',[]) for o in f['objects']}
    counts=Counter();changes=Counter();by_cohort={};by_family={};records=[];pose_errors=[]
    for index,(row,e,b,p) in enumerate(zip(rows,es,before,after)):
        gt=any(object_hazard(o,e['body_origin_m']) for o in e['native_bounds'])
        change='UNCHANGED' if b['candidate']==p['candidate'] else ('GAINED_TP' if gt else 'ADDED_FP') if p['candidate'] else ('LOST_TP' if gt else 'REMOVED_FP')
        changes[change]+=1;cohort=cohorts.get(row['episode_id'],'UNSPECIFIED');cc=by_cohort.setdefault(cohort,Counter())
        fc=by_family.setdefault(e.get('family','UNSPECIFIED'),Counter());fc['frames']+=1
        counts['frames']+=1;cc['frames']+=1;diagnostic=geometry[index] if geometry is not None else p.get('parallax',{})
        poses=[];points=[];bad_pose=False;bad_depth=False
        for pose in diagnostic.get('poses',[]):
            reference=pose['reference_index'];issues=reference_issues(rows,index,reference)
            record=dict(reference_index=reference,contract_issues=issues,inferred_translation_m=pose['translation_m'],
                translation_bounds=pose.get('translation_bounds'),zero_translation_feasible=pose.get('zero_translation_feasible'),
                anchor_zones=pose.get('anchor_zones'),anchor_features=pose.get('anchor_features'))
            counts['pose_pairs']+=1;cc['pose_pairs']+=1
            counts['pose_pairs_zero_translation_feasible']+=pose.get('zero_translation_feasible') is True
            counts['pose_pairs_exclude_zero_translation']+=pose.get('zero_translation_feasible') is False
            cc['pose_pairs_exclude_zero_translation']+=pose.get('zero_translation_feasible') is False
            counts['pose_reference_violations']+=bool(issues)
            if 'INVALID_REFERENCE_INDEX' not in issues:
                true=native_translation(es[reference]['camera'],e['camera']);estimate=pose['translation_m']
                assert finite_vector(estimate)
                error=[x-y for x,y in zip(estimate,true)];bounds=pose.get('translation_bounds')
                enclosed=None
                if bounds is not None:
                    assert len(bounds)==3 and all(valid_interval(v) for v in bounds)
                    enclosed=all(interval_contains(v,bd) for v,bd in zip(true,bounds))
                magnitude=norm(error);pose_errors.append(magnitude)
                record.update(native_translation_m=true,error_m=error,error_norm_m=magnitude,native_translation_norm_m=norm(true),
                    native_translation_inside_bounds=enclosed,pair_age_s=row['time_s']-rows[reference]['time_s'])
                counts['pose_bounds_missing']+=enclosed is None;counts['pose_bounds_exclude_native_translation']+=enclosed is False
                cc['pose_bounds_exclude_native_translation']+=enclosed is False
                bad_pose|=enclosed is False or bool(issues)
            poses.append(record)
        for point_index,point in enumerate(diagnostic.get('points',[])):
            ranges=point['range_bounds_m'];assert valid_interval(ranges) and ranges[0]>0
            references=point.get('source_references',[])
            issues=[dict(reference_index=ref,issues=reference_issues(rows,index,ref)) for ref in references if reference_issues(rows,index,ref)]
            if len(set(references))<2:issues.append(dict(issues=['FEWER_THAN_TWO_DISTINCT_REFERENCES']))
            hit=first_hit(row,e,point['pixel'],contexts,roles)
            enclosed=interval_contains(hit['range_m'],ranges) if 'range_m' in hit else None
            reference_hits=[]
            for rp in point.get('reference_pixels',[]):
                ref=rp['index']
                if reference_issues(rows,index,ref):continue
                old_hit=first_hit(rows[ref],es[ref],rp['pixel'],contexts,roles)
                same=None if 'actor_ids' not in old_hit or 'actor_ids' not in hit else bool(set(old_hit['actor_ids']).intersection(hit['actor_ids']))
                reference_hits.append(dict(reference_index=ref,pixel=rp['pixel'],first_hit=old_hit,current_first_actor_correspondence=same))
            reused=[pose['reference_index'] for pose in diagnostic.get('poses',[]) if pose['reference_index'] in references and
                any(anchor.get('feature_id')==point.get('feature_id') for anchor in pose.get('anchor_features',[]))]
            record=dict(point_index=point_index,pixel=point['pixel'],inferred_range_bounds_m=ranges,
                source_references=references,reference_issues=issues,first_hit=hit,native_depth_contained=enclosed,
                reference_pixel_first_hits=reference_hits,query_reused_as_pose_anchor_in_references=reused,
                observable_diagnostics={k:v for k,v in point.items() if k not in ('pixel','range_bounds_m','source_references')})
            points.append(record);counts['points']+=1;cc['points']+=1
            fc['points']+=1;fc['foreground_native_actor_points']+=hit.get('foreground_native_actor',False)
            fc['depth_intervals_exclude_native_first_hit']+=enclosed is False
            counts['points_reusing_own_tof_pose_anchor']+=bool(reused)
            counts['points_with_reference_current_actor_mismatch']+=any(h['current_first_actor_correspondence'] is False for h in reference_hits)
            counts['point_reference_violations']+=bool(issues)
            counts['points_without_reference_depth']+=enclosed is None
            counts['depth_intervals_contain_native_first_hit']+=enclosed is True
            counts['depth_intervals_exclude_native_first_hit']+=enclosed is False
            cc['depth_intervals_exclude_native_first_hit']+=enclosed is False
            counts['foreground_native_actor_points']+=hit.get('foreground_native_actor',False)
            if enclosed is False:
                counts['excluded_native_foreground_depth']+=hit['foreground_native_actor']
                counts['excluded_corridor_hazard_depth']+=hit['point_inside_corridor']
                counts['depth_interval_entirely_behind_native_hit']+=ranges[0]>hit['range_m']
                counts['depth_interval_entirely_before_native_hit']+=ranges[1]<hit['range_m']
            bad_depth|=enclosed is False or bool(issues)
        counts['frames_with_poses']+=bool(poses);counts['frames_with_points']+=bool(points)
        counts['frames_with_false_translation_geometry']+=bad_pose;counts['frames_with_false_depth_geometry']+=bad_depth
        counts['frames_false_depth_geometry_baseline_and_candidate_alert']+=bool(bad_depth and b['candidate'] and p['candidate'])
        counts['frames_false_translation_geometry_baseline_and_candidate_alert']+=bool(bad_pose and b['candidate'] and p['candidate'])
        cc['frames_with_points']+=bool(points);cc['frames_with_false_depth_geometry']+=bad_depth
        if poses or points or change!='UNCHANGED':records.append(dict(index=index,id=row['id'],episode_id=row['episode_id'],
            family=e.get('family'),camera_motion_cohort=cohort,native_truth=gt,baseline_alert=b['candidate'],candidate_alert=p['candidate'],
            frame_change=change,poses=poses,points=points))
    return dict(status='EVALUATOR_ONLY_PARALLAX_GEOMETRY_AUDIT_COMPLETE',counts=counts,by_camera_motion_cohort=by_cohort,by_family=by_family,
        frame_changes=changes,translation_error_norm_m=dict(n=len(pose_errors),maximum=max(pose_errors,default=None),
            mean=sum(pose_errors)/len(pose_errors) if pose_errors else None),records=records,
        limits=['Pose and scene geometry are evaluator-only and never supplied to inference.',
            'True translation is B_current.T @ (C_reference-C_current), in current CV right/down/forward axes.',
            'Range intervals are audited as slant distance along the normalized current pixel ray, not forward optical depth.',
            'First intersections use native actor AABBs plus frozen background/floor boxes. They are geometric proxies, not segmentation truth; texture overlays and unmodeled rendering details are not reconstructed.',
            'No known AABB hit is UNAVAILABLE reference depth, not a proven empty ray.',
            'Repeated feature tracks/frames are not independent obstacles. Cohorts come only from frozen source metadata.',
            'Incorrect geometry is counted even when independent baseline evidence preserves the alert.'])


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(analysis,output,baseline='resolution_guard',candidate='parallax'):
    root=Path(__file__).resolve().parents[4];analysis=analysis.resolve();output=output.resolve()
    if not output.is_relative_to((root/'artifacts.local').resolve()) or output.exists():raise ValueError('Fresh canonical audit output required')
    sealpath=analysis/'prediction-seal.json';seal=json.loads(sealpath.read_text())
    if seal.get('status') not in ('ALL_ARMS_AND_DISTANCE_CURVES_SEALED_BEFORE_EVALUATOR_PARSE','ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE',
                                'ALL_PREDICTIONS_SEALED_BEFORE_EVALUATOR_PARSE'):raise ValueError('Completed prediction seal required')
    for name,digest in seal.get('code_sha256',{}).items():assert sha(analysis/name)==digest
    packets={};panels=seal.get('panels',{'source':seal})
    for panel,item in panels.items():
        capture=Path(item['capture']);predpath=analysis/panel/'predictions.json' if 'panels' in seal else analysis/'predictions.json'
        assert sha(capture/'raw.jsonl')==item['raw_sha256'] and sha(capture/'receipt.json')==item['receipt_sha256']
        assert sha(predpath)==item['predictions_sha256']
        receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS'
        for name,digest in receipt['hashes'].items():assert sha(capture/name)==digest
        assert sha(capture/'spec.json')==receipt['spec_sha256']
        geometry=None
        if 'geometry_sha256' in item:
            geompath=predpath.with_name('geometry.json');assert sha(geompath)==item['geometry_sha256']
            geometry=json.loads(geompath.read_text())
        packets[panel]=(capture,readrows(capture/'raw.jsonl'),json.loads(predpath.read_text()),json.loads((capture/'spec.json').read_text()),receipt,geometry)
    output.mkdir(parents=True);shutil.copyfile(Path(__file__),output/Path(__file__).name);summary={}
    for panel,(capture,rows,preds,spec,receipt,geometry) in packets.items():
        result=audit(rows,readrows(capture/'evaluator.jsonl'),preds,spec,receipt,baseline,candidate,geometry)
        write(output/(panel+'-audit.json'),result);summary[panel]={k:v for k,v in result.items() if k!='records'}
    write(output/'summary.json',dict(status='EVALUATOR_ONLY_PARALLAX_GEOMETRY_AUDIT_COMPLETE',panels=summary))
    write(output/'completion.json',dict(status='PASS',source_seal_sha256=sha(sealpath),output_hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(summary,indent=2))


def self_test():
    camera=dict(x=0.,y=0.,z=1.7,yaw=0.,pitch=0.,roll=0.)
    moved=dict(camera,y=.1);assert native_translation(camera,moved)==[-.1,0.,0.]
    turned=dict(moved,yaw=90.)
    assert max(abs(a-b) for a,b in zip(native_translation(camera,turned),[0.,0.,-.1]))<1e-12
    intr=dict(width=640,height=360,cx=320.,cy=180.,fx=320.,fy=320.)
    row=dict(id='pair1',episode_id='pair',time_s=.25,rgb_intrinsics=intr)
    near=dict(name='near',center_m=[2.,0.,1.7],extent_m=[.1,.1,.1])
    far=dict(name='far',center_m=[3.,0.,1.7],extent_m=[.1,.5,.5])
    e=dict(id='pair1',camera=camera,body_origin_m=[0.,0.,0.],native_bounds=[far,near])
    context=dict(background=dict(center_m=[12.,0.,2.],size_m=[.1,20.,20.]),floor=dict(center_m=[4.,0.,-.05],size_m=[24.,20.,.1]))
    hit=first_hit(row,e,[320.,180.],context_objects(context,{}));assert hit['actor_ids']==['pair/near'] and abs(hit['range_m']-1.9)<1e-12
    floor=first_hit(row,dict(e,native_bounds=[]),[320.,359.],context_objects(context,{}));assert floor['actor_ids']==['CONTEXT/floor']
    previous=dict(row,id='pair0',time_s=0.);previous_e=dict(e,id='pair0')
    p=dict(candidate=True,parallax=dict(poses=[dict(reference_index=0,translation_m=[.2,0.,0.],translation_bounds=[[.19,.21],[-.01,.01],[-.01,.01]])],
        points=[dict(pixel=[320.,180.],range_bounds_m=[2.8,3.2],source_references=[0])]))
    result=audit([previous,row],[previous_e,e],dict(resolution_guard=[dict(candidate=True)]*2,parallax=[dict(candidate=True),p]),context,{})
    assert result['counts']['depth_intervals_exclude_native_first_hit']==1
    assert result['counts']['excluded_corridor_hazard_depth']==1
    assert result['counts']['pose_bounds_exclude_native_translation']==1
    assert result['counts']['frames_false_depth_geometry_baseline_and_candidate_alert']==1
    separate=audit([previous,row],[previous_e,e],dict(resolution_guard=[dict(candidate=True)]*2,parallax=[dict(candidate=True)]*2),
        context,{},geometry=[{},p['parallax']])
    assert separate['counts']==result['counts']
    assert 'NONCAUSAL_REFERENCE' in reference_issues([previous,row],1,1)
    assert ray_aabb([0,2,0],[1,0,0],dict(center_m=[2,0,0],extent_m=[.1,.1,.1])) is None
    return dict(status='PASS',checks=['CV translation sign and rotated basis','near-first AABB occlusion','frozen floor context',
        'false depth/translation geometry despite retained alert','causal reference rejection','parallel-ray miss','separate sealed geometry schema'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--baseline',default='resolution_guard');parser.add_argument('--candidate',default='parallax');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    elif args.analysis and args.output:run(args.analysis,args.output,args.baseline,args.candidate)
    else:parser.error('Use --self-test or --analysis and --output')
