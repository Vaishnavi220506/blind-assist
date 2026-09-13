"""Positive pixel/echo correspondence; missing corroboration never clears support."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import mz115_spatial_allocation as tof
from mz107_rgb_association import pixel_ray
from mz124_measurement_geometry import slant_box, metrics
from mz129_extent_correction import ROOT, SOURCE, CORRECTION, events
from mz128_zone_weighting import column_weights, FOUR, THRESHOLD
from mz126_central_tof import read, write, sha
from research_backend import BackendCandidate, DeviceObservation, select_backend

MZ129 = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'
BRIEF = Path(__file__).with_name('MZ130_LOCAL_SUPPORT_BRIEF_20260914.md')


def clipped(tiles, box):
    return [q for tile in tiles if (q := tof.intersection(tile, box)) is not None]


def slant(tile, evidence, row, yaw):
    p = row['camera_pitch_deg']; dy = .5+.2*row['time_s']
    return slant_box(tile, evidence['range_bounds'], row['rgb_intrinsics'],
                     (p-.5,p+.5), (yaw-dy,yaw+dy), row['camera_in_body_m'][2])


def horizontal_bounds(xyz):
    def sq(bounds):
        a,b=bounds
        return (0. if a<=0<=b else min(a*a,b*b),max(a*a,b*b))
    x,y=sq(xyz[0]),sq(xyz[1])
    return (math.sqrt(x[0]+y[0]),math.sqrt(x[1]+y[1]))


def local_tof(row, cache, components):
    evidence = cache['spatial_evidence']; yaw=cache['integrated_yaw_deg']
    groups=[]
    for i in sorted((i for i,e in enumerate(evidence) if e['status']=='SIM_VALID'),
                    key=lambda i:(evidence[i]['forward_depth'],evidence[i]['zone_id'],evidence[i]['target_slot'])):
        if not groups or evidence[i]['forward_depth']-evidence[groups[-1][0]]['forward_depth']>tof.DEPTH_SPAN_M:
            groups.append([])
        groups[-1].append(i)
    chosen={}; group_records=[]
    for indices in groups:
        items=[evidence[i] for i in indices]
        record=dict(indices=indices,accepted=False,reason='INSUFFICIENT_OR_COMPETING_ZONES',candidates=[])
        group_records.append(record)
        if len(items)<tof.MIN_ZONES or len({e['zone_id'] for e in items})!=len(items):continue
        weights=np.array([e['signal']*e['range_m']**2 for e in items],float)
        if weights.sum()<=0:record['reason']='NO_POSITIVE_SIGNAL';continue
        candidates=[]
        for component in components:
            if component['status']!='MASK_COMPONENT':continue
            fractions=np.array([sum(tof.area(t) for t in clipped(component['tiles'],e['zone_box']))/tof.area(e['zone_box']) for e in items])
            coverage=float(weights[fractions>0].sum()/weights.sum())
            norm=float(np.linalg.norm(fractions)*np.linalg.norm(weights))
            cosine=float(np.dot(fractions,weights)/norm) if norm else 0.
            if coverage>=tof.MIN_COVERAGE and np.count_nonzero(fractions)>1:
                candidates.append((cosine,component['proposal'],coverage))
        candidates.sort(key=lambda x:(-x[0],x[1]))
        record['candidates']=[dict(cosine=c,proposal=j,coverage=v) for c,j,v in candidates]
        margin=candidates[0][0]-(candidates[1][0] if len(candidates)>1 else 0.) if candidates else 0.
        if not candidates or candidates[0][0]<tof.MIN_COSINE or margin<tof.MIN_MARGIN:
            record['reason']='MASK_SIGNAL_MATCH_UNRESOLVED';continue
        j=candidates[0][1]
        if any(e['proposal'] is not None and e['proposal']!=j for e in items):
            record['reason']='CONFLICT_WITH_EXISTING_ASSOCIATION';continue
        eligible={i:clipped(components[j]['tiles'],evidence[i]['roi']) for i in indices}
        if sum(bool(v) for v in eligible.values())<tof.MIN_ZONES:
            record['reason']='INSUFFICIENT_NONEMPTY_LOCAL_ZONES';continue
        record.update(accepted=True,reason='POSITIVE_MASK_SIGNAL_COHORT',proposal=j,margin=margin)
        for i,tiles in eligible.items():
            if tiles:chosen[i]=(j,tiles)
    records=[]
    for i,e in enumerate(evidence):
        baseline=tof.possible(e['localized_xyz'])
        rec=dict(zone_id=e['zone_id'],slot=e['target_slot'],status=e['status'],range_bounds=e['range_bounds'],
                 old_proposal=e['proposal'],proposal=e['proposal'],localized=False,tiles=[e['roi']],
                 old_possible=baseline,possible=baseline,reason='INHERITED_UNRESOLVED_OR_NO_MATCH')
        if i in chosen:
            j,tiles=chosen[i]; boxes=[slant(t,e,row,yaw) for t in tiles]
            support=any(tof.possible(xyz) for xyz in boxes)
            assert not support or baseline
            ranges=[horizontal_bounds(xyz) for xyz in boxes]
            rec.update(proposal=j,localized=True,tiles=tiles,possible=support,
                       horizontal_range_bounds=[min(x[0] for x in ranges),max(x[1] for x in ranges)],
                       reason='POSITIVE_MASK_SIGNAL_COHORT')
        if e['status']=='SIM_MERGED':assert not rec['localized'] and rec['range_bounds']==[.02,4.]
        records.append(rec)
    weights=column_weights(row,FOUR)
    active={r['zone_id'] for r in records if r['possible']}
    score=sum(weights[z] for z in active)
    return dict(returns=records,groups=group_records,score=score,
                certain_coarse=any(tof.certain(e['coarse_xyz']) for e in evidence))


def fixed_plane_boxes(tiles, anchor_box, distance, row, yaw):
    """Same original plane for every tile; no tile-center range reanchoring."""
    intr=row['rgb_intrinsics']; pitch=row['camera_pitch_deg']
    center=pixel_ray((anchor_box[0]+anchor_box[2])/2,(anchor_box[1]+anchor_box[3])/2,intr,pitch,yaw)
    anchor=center*distance/math.hypot(center[0],center[1])
    normal=pixel_ray(intr['cx'],intr['cy'],intr,pitch,yaw)
    numerator=float(np.dot(normal,anchor)); result=[]
    for box in tiles:
        points=[]
        for u in (box[0],box[2]):
            for v in (box[1],box[3]):
                ray=pixel_ray(u,v,intr,pitch,yaw); denominator=float(np.dot(normal,ray))
                if denominator<=0:return None
                points.append(ray*numerator/denominator+[0.,0.,row['camera_in_body_m'][2]])
        pts=np.array(points); result.append(list(zip(pts.min(axis=0).tolist(),pts.max(axis=0).tolist())))
    return result


def local_radar(row, cache, old, local):
    records=[]; yaw=old['integrated_yaw_deg']; intr=row['rgb_intrinsics']
    for ret in old['corrected_current_evidence']:
        j=ret['proposal']
        r=dict(slot=ret['slot'],proposal=j,old_possible=bool(ret['support']),possible=bool(ret['support']),
               localized=False,tiles=[],matched_zones=[],reason='NO_UNIQUE_EXISTING_ASSOCIATION')
        records.append(r)
        if j is None:continue
        measured=row['radar_range_m'][ret['slot']]; angle=row['radar_angle'][ret['slot']]
        matches=[e for e in local['returns'] if e['localized'] and e['proposal']==j
                 and e['horizontal_range_bounds'][1]>=max(.02,measured-.15)
                 and e['horizontal_range_bounds'][0]<=measured+.15]
        r['reason']='INSUFFICIENT_RANGE_COMPATIBLE_TOF_ZONES'
        if len({e['zone_id'] for e in matches})<2:continue
        # Same image-angle working convention as MZ111 association, not a hardware CI.
        if not -89.<angle-12<angle+12<89.:r['reason']='UNBOUNDED_IMAGE_ANGLE';continue
        beam=[intr['cx']+intr['fx']*math.tan(math.radians(angle-12)),0.,
              intr['cx']+intr['fx']*math.tan(math.radians(angle+12)),float(intr['height'])]
        tiles=clipped([t for e in matches for t in e['tiles']],beam)
        tiles=clipped(tiles,ret['box'])
        if not tiles:r['reason']='EMPTY_INTERSECTION_FALLBACK';continue
        boxes=fixed_plane_boxes(tiles,ret['box'],ret['range_m'],row,yaw)
        if boxes is None:r['reason']='INVALID_PLANE_FALLBACK';continue
        support=any(tof.possible(xyz) for xyz in boxes)
        assert not support or ret['support']
        r.update(localized=True,possible=support,tiles=tiles,
                 matched_zones=sorted({e['zone_id'] for e in matches}),reason='POSITIVE_TOF_RGB_RADAR_LOCAL_SUPPORT')
    current=any(r['possible'] for r in records)
    common=bool(current or old['raw_center_support'] or old['inherited_carry_support'])
    flag=bool(common or old['guard_events'])
    assert not flag or old['candidate']
    return dict(returns=records,current=current,common_radar=common,guard_events=old['guard_events'],candidate=flag)


def run(mask_dir, output):
    mask_dir,output=mask_dir.absolute(),output.absolute()
    assert not output.exists() and output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    raw=SOURCE/'capture-v1/raw.jsonl'; cp=CORRECTION/'predictions.json'
    baselinepath=MZ129/'replay-v1r2/predictions.json'; radarpath=MZ129/'radar-v1/predictions.json'
    # Input hashes authenticate prior outputs; no evaluator fields enter prediction.
    ms=read(mask_dir/'prediction-seal.json')
    assert sha(mask_dir/'components.json')==ms['components_sha256']
    assert sha(raw)==ms['raw_sha256'] and sha(cp)==ms['correction_sha256']
    for key in ('input_hashes','source_hashes'):
        for name,digest in ms[key].items():assert sha(Path(name))==digest,name
    for directory in (MZ129/'replay-v1r2',MZ129/'radar-v1'):
        seal=read(directory/'prediction-seal.json')
        assert sha(directory/'predictions.json')==seal['predictions_sha256']
        for key in ('input_hashes','dependencies','source_hashes'):
            for name,digest in seal.get(key,{}).items():assert sha(ROOT/name)==digest,name
    rows=[json.loads(x) for x in raw.read_text().splitlines()]
    masks=read(mask_dir/'components.json'); cache=read(cp)
    baseline=read(baselinepath)['radar']; oldradar=read(radarpath)
    assert len(rows)==len(cache)==len(masks)==len(baseline)==len(oldradar)==288
    assert [r['id'] for r in rows]==[m['id'] for m in masks]==[p['id'] for p in baseline]==[p['id'] for p in oldradar]
    output.mkdir(parents=True)
    select_backend('scalar-scoring',cpu=BackendCandidate('local-support-intervals','cpu',
        lambda:horizontal_bounds([[1.,2.],[-.2,.2],[0.,1.]]),
        lambda _:DeviceObservation('cpu','host CPU','Python / NumPy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=output/'backend.json')
    started=time.perf_counter(); details=[]
    arms=dict(baseline=baseline,tof=[],radar=[],combined=[])
    for row,cached,mask,old,rb in zip(rows,cache,masks,baseline,oldradar):
        assert bool(old['score']>=THRESHOLD or old['certain_coarse'] or rb['candidate'])==old['candidate']
        local=local_tof(row,cached,mask['components'])
        radar=local_radar(row,cached,rb,local)
        assert local['certain_coarse']==old['certain_coarse']
        for name,score_value,rad in [('tof',local['score'],rb),('radar',old['score'],radar),('combined',local['score'],radar)]:
            flag=bool(score_value>=THRESHOLD or old['certain_coarse'] or rad['candidate'])
            assert not flag or old['candidate']
            arms[name].append(dict(id=row['id'],candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN',
                 score=score_value,certain_coarse=old['certain_coarse'],common_radar=rad['common_radar'],guard_events=rad['guard_events']))
        details.append(dict(id=row['id'],tof=local,radar=radar))
    seconds=time.perf_counter()-started
    write(output/'predictions.json',arms);write(output/'local-support.json',details)
    paths=[raw,cp,baselinepath,radarpath,mask_dir/'components.json',mask_dir/'prediction-seal.json']
    deps=[Path(__file__),BRIEF,Path(__file__).with_name('mz130_local_masks.py'),Path(tof.__file__)]
    write(output/'prediction-seal.json',dict(predictions_sha256=sha(output/'predictions.json'),
        local_support_sha256=sha(output/'local-support.json'),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps},seconds=seconds,
        authority='CONSUMED_OBSERVABLE_PREDICTIONS_SAVED_BEFORE_SCORING',cpu_reason='TASK_NOT_GPU_SUITABLE'))
    score(output,rows,arms,details,cache,oldradar)


def score(output,rows,arms,details,cache,oldradar):
    # Evaluator schema/source code was inspected during development; outcomes
    # for these new hypotheses are computed only after the prediction seal.
    from mz115_allocation_audit import native_truth, project_point, contains, possible, native_bounds
    report=read(SOURCE/'analysis-v1/frame-report.json')
    evaluator=[json.loads(s) for s in (SOURCE/'capture-v1/evaluator.jsonl').read_text().splitlines()]
    provenance=[json.loads(s) for s in (SOURCE/'capture-v1/provenance.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==[e['id'] for e in evaluator]==[r['id'] for r in report]==[p['id'] for p in provenance]
    truth=[native_truth(e) for e in evaluator];assert truth==[r['truth'] for r in report]
    native=[]
    for row,ev,pr,local,cached,rb in zip(rows,evaluator,provenance,details,cache,oldradar):
        actors={row['episode_id']+'/'+o['name']:o for o in ev['native_bounds']}
        for record,old in zip(local['tof']['returns'],cached['spatial_evidence']):
            if not record['localized']:continue
            zone=next(z for z in ev['zonal_tof_native'] if z['zone_id']==record['zone_id'])
            lineage=next(l for l in zone['returned_lineage'] if l['target_index']==record['slot'])
            for k in lineage['hit_indices']:
                hit=zone['private_rays'][k]
                native.append(native_item(row,ev,actors,record,old['roi'],hit['hit_point_m'],hit.get('actor_id'),'TOF',k,project_point,contains,possible,native_bounds))
        for record,old in zip(local['radar']['returns'],rb['corrected_current_evidence']):
            if not record['localized']:continue
            origin=pr['radar_slots'][record['slot']]
            actor=actors.get(origin.get('actor_id')) if origin else None
            if origin and origin['kind']=='real_actor' and actor:
                camera=[ev['camera'][k] for k in ('x','y','z')]
                delta=[c-o for c,o in zip(actor['center_m'],camera)]
                scale=origin['pre_noise_range_m']/math.hypot(delta[0],delta[1])
                point=[c+scale*d for c,d in zip(camera,delta)]
                native.append(native_item(row,ev,actors,record,old['box'],point,origin['actor_id'],'RADAR',None,project_point,contains,possible,native_bounds))
            else:native.append(dict(id=row['id'],sensor='RADAR',slot=record['slot'],evaluable=False,reason='NO_REAL_NATIVE_SURFACE',source_kind=origin.get('kind') if origin else None))
    write(output/'native-local-retention.json',native)
    stats={}
    for sensor in ('TOF','RADAR'):
        records=[r for r in native if r['sensor']==sensor]
        valid=[r for r in records if r['evaluable']]
        dropped=[r for r in valid if r['previously_contained'] and not r['locally_contained']]
        stats[sensor]=dict(audited_samples=len(records),not_evaluable=sum(not r['evaluable'] for r in records),
            newly_dropped_samples=len(dropped),newly_dropped_corridor_samples=sum(r['corridor_point'] for r in dropped),
            newly_dropped_hazard_actor_samples=sum(r['hazard_actor'] for r in dropped))
    oldflags=[p['candidate'] for p in arms['baseline']];oldevents=events(rows,truth,oldflags)
    result=dict(frames=288,arms={},native_retention=stats,
        local_tof_returns=sum(e['localized'] for p in details for e in p['tof']['returns']),
        new_tof_associations=sum(e['localized'] and e['old_proposal'] is None for p in details for e in p['tof']['returns']),
        local_radar_returns=sum(e['localized'] for p in details for e in p['radar']['returns']),
        tof_fallbacks=dict(Counter(e['reason'] for p in details for e in p['tof']['groups'])),
        radar_fallbacks=dict(Counter(e['reason'] for p in details for e in p['radar']['returns'])))
    for name,predictions in arms.items():
        flags=[p['candidate'] for p in predictions];m=metrics(rows,truth,flags);m['events']=events(rows,truth,flags)
        m['lost_TP_ids']=[r['id'] for r,t,a,b in zip(rows,truth,oldflags,flags) if t and a and not b]
        m['changed']=[dict(id=r['id'],family=l['family'],truth=t,before=a,after=b) for r,l,t,a,b in zip(rows,report,truth,oldflags,flags) if a!=b]
        m['event_times_identical']=m['events']==oldevents
        m['families']={}
        for family in sorted({r['family'] for r in report}):
            ix=[i for i,r in enumerate(report) if r['family']==family]
            m['families'][family]=metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
        affected=[] if name=='baseline' else ['TOF'] if name=='tof' else ['RADAR'] if name=='radar' else ['TOF','RADAR']
        m['native_retention_pass']=all(not stats[s]['not_evaluable'] and not stats[s]['newly_dropped_corridor_samples'] for s in affected)
        m['candidate_criterion']=m['FP']<93 and not m['lost_TP_ids'] and m['event_times_identical'] and m['native_retention_pass']
        result['arms'][name]=m
    write(output/'summary.json',result)
    seal=read(output/'prediction-seal.json')
    assert sha(output/'predictions.json')==seal['predictions_sha256']
    for path,digest in seal['input_hashes'].items():assert sha(ROOT/path)==digest,path
    write(output/'completion.json',dict(status='PASS',summary_sha256=sha(output/'summary.json'),
        evaluator_sha256=sha(SOURCE/'capture-v1/evaluator.jsonl'),provenance_sha256=sha(SOURCE/'capture-v1/provenance.jsonl'),
        resources='No persistent process or allocation; source inputs unchanged'))
    print(json.dumps({name:{k:m[k] for k in ('TP','FP','FN','false_segments','max_detected_delay_s','native_retention_pass','candidate_criterion')} for name,m in result['arms'].items()},indent=2))
    print(json.dumps({k:result[k] for k in ('local_tof_returns','new_tof_associations','local_radar_returns','native_retention')},indent=2))


def native_item(row,ev,actors,record,oldbox,point,actor_id,sensor,sample,project,contains,possible,bounds):
    pixel=project(point,ev['camera'],row['rgb_intrinsics'])
    if pixel is None:return dict(id=row['id'],sensor=sensor,slot=record['slot'],evaluable=False,reason='PROJECTION_UNKNOWN')
    body=[p-o for p,o in zip(point,ev['body_origin_m'])]
    actor=actors.get(actor_id)
    return dict(id=row['id'],sensor=sensor,zone_id=record.get('zone_id'),slot=record['slot'],sample=sample,
        evaluable=True,actor_id=actor_id,pixel=pixel,native_point=point,
        previously_contained=contains(oldbox,pixel),locally_contained=any(contains(t,pixel) for t in record['tiles']),
        corridor_point=possible([(x,x) for x in body]),hazard_actor=bool(actor and possible(bounds(actor,ev['body_origin_m']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--masks',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.masks,args.output)
