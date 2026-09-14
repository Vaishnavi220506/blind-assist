"""Five-frame conditional image-flow transport of coarse ToF support.

Flow is observable image motion, never metric camera translation. Association
and locally translational surface motion are testable assumptions, not bounds.
"""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from mz129_extent_correction import ROOT, SOURCE, CORRECTION, events
from mz126_central_tof import read, write, sha, serial, tof
from mz128_zone_weighting import FOUR, THRESHOLD, column_weights, readout
from mz124_measurement_geometry import metrics, slant_box
from mz132_contour_adapter import outside_image
from research_backend import BackendCandidate, DeviceObservation, select_backend

WINDOW = 5
MAX_FEATURES = 256
FB_ERROR_PX = .5
FLOW_PAD_PX = 2.
RANGE_SPEED_M_S = 2.
MZ129 = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'
RGB = ROOT/'artifacts.local/work/mz125-observable-correction-20260913/rgb-v1'
BRIEF = Path(__file__).with_name('MZ135_TEMPORAL_BRIEF_20260914.md')


def basis(pitch,yaw):
    p,y=map(math.radians,(pitch,yaw));cp,sp,cy,sy=math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[-sy,sp*cy,cp*cy],[cy,sp*sy,cp*sy],[0,-cp,sp]])


def ray(pixel,intr):
    q=np.array([(pixel[0]-intr['cx'])/intr['fx'],(pixel[1]-intr['cy'])/intr['fy'],1.])
    return q/np.linalg.norm(q)


def inside(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def causal_tracks(rows, cached, image_loader):
    """Only one adjacent-image LK pass; retained IDs carry through the window."""
    history = []; previous = None; tracks = {}; next_id = 0; episode = None
    for row, cache in zip(rows, cached):
        if row['episode_id'] != episode:
            history = []; previous = None; tracks = {}; next_id = 0
        episode = row['episode_id']; gray = image_loader(row)
        if gray is None:
            raise ValueError('Missing authenticated source image')
        current = {}
        if previous is not None and tracks:
            ids = list(tracks); old = np.float32([tracks[k] for k in ids]).reshape(-1,1,2)
            new, ok, _ = cv2.calcOpticalFlowPyrLK(previous, gray, old, None,
                winSize=(21,21), maxLevel=3)
            if new is not None:
                back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, previous, new, None,
                    winSize=(21,21), maxLevel=3)
                if back is not None:
                    for k,p,q,b,a,c in zip(ids,new[:,0],old[:,0],back[:,0],ok[:,0],back_ok[:,0]):
                        if a and c and np.isfinite(p).all() and np.linalg.norm(q-b)<=FB_ERROR_PX and 2<=p[0]<gray.shape[1]-2 and 2<=p[1]<gray.shape[0]-2:
                            current[k] = p.tolist()
        mask = np.full(gray.shape,255,np.uint8)
        for p in current.values(): cv2.circle(mask,tuple(np.round(p).astype(int)),7,0,-1)
        if len(current) < MAX_FEATURES:
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=MAX_FEATURES-len(current),
                qualityLevel=.01, minDistance=7, blockSize=7, mask=mask)
            if corners is not None:
                for p in corners[:,0]: current[next_id]=p.tolist(); next_id+=1
        entry = dict(row=row, cache=cache, tracks=current)
        yield entry, history[-(WINDOW-1):]
        history.append(entry); history = history[-(WINDOW-1):]
        previous = gray; tracks = current


def conditional_tiles(roi, per_reference, intr):
    """Intersection only under the declared shared/local-flow model.

The full outside-image part survives independently. An inconsistent model is
unresolved rather than interpreted as a measurement proving empty space.
"""
    if len(per_reference) < 2:
        return [list(roi)], 'INSUFFICIENT_TEMPORAL_REFERENCES'
    tile = list(roi)
    for boxes in per_reference:
        tile = tof.intersection(tile, union_box(boxes))
        if tile is None: return [list(roi)], 'CONFLICT_FALLBACK'
    outside = outside_image(roi,intr['width'],intr['height'])
    if tile == list(roi): return [list(roi)], 'NO_ENVELOPE_CHANGE'
    return [tile]+outside, 'CONDITIONAL_TEMPORAL_ASSOCIATION'


def predict_frame(entry, history, radar):
    row, cached, tracks = entry['row'], entry['cache'], entry['tracks']
    intr = row['rgb_intrinsics']; weights = column_weights(row,FOUR)
    current_yaw = cached['integrated_yaw_deg']; records=[]; pair_stats=[]
    pair_data=[]
    for prior in history:
        ref=prior['row']; dt=row['time_s']-ref['time_s']
        assert ref['episode_id']==row['episode_id'] and 0 < dt <= 1.000001
        shared=sorted(set(tracks)&set(prior['tracks']))
        residual=[]; cross_zone=0
        R=basis(row['camera_pitch_deg'],current_yaw).T@basis(ref['camera_pitch_deg'],prior['cache']['integrated_yaw_deg'])
        for k in shared:
            p=prior['tracks'][k]; q=tracks[k]; v=R@ray(p,intr)
            if v[2] > 0:
                projected=[intr['cx']+intr['fx']*v[0]/v[2],intr['cy']+intr['fy']*v[1]/v[2]]
                residual.append(float(np.linalg.norm(np.array(q)-projected)))
            def zone_at(point):
                return next((z['zone_id'] for z in row['tof_zones'] if inside(tof.zone_box(z,intr),point)),None)
            cross_zone += zone_at(p) != zone_at(q)
        pair_stats.append(dict(reference_id=ref['id'],age_s=dt,shared_tracks=len(shared),
            cross_zone_tracks=cross_zone,median_imu_compensated_flow_px=float(np.median(residual)) if residual else None,
            delta_yaw_deg=current_yaw-prior['cache']['integrated_yaw_deg'],metric_translation_estimated=False))
        pair_data.append((prior,shared,dt))
    for e in cached['spatial_evidence']:
        references=[]; links=[]
        if e['status']=='SIM_VALID' and row['imu_valid']:
            for prior,shared,dt in pair_data:
                if not prior['row']['imu_valid']: continue
                boxes=[]; local=[]
                for k in shared:
                    now=tracks[k]; old=prior['tracks'][k]
                    if not inside(e['zone_box'],now): continue
                    for prev in prior['cache']['spatial_evidence']:
                        if prev['status']!='SIM_VALID' or not inside(prev['zone_box'],old): continue
                        # A deliberately broad working relative-range model; no missing Doppler=0.
                        slack=RANGE_SPEED_M_S*dt
                        if prev['range_bounds'][0]-slack > e['range_bounds'][1] or prev['range_bounds'][1]+slack < e['range_bounds'][0]: continue
                        dx,dy=now[0]-old[0],now[1]-old[1]; box=prev['zone_box']
                        warped=[box[0]+dx-FLOW_PAD_PX,box[1]+dy-FLOW_PAD_PX,
                                box[2]+dx+FLOW_PAD_PX,box[3]+dy+FLOW_PAD_PX]
                        boxes.append(warped)
                        local.append(dict(reference_id=prior['row']['id'],zone_id=prev['zone_id'],
                            slot=prev['target_slot'],track_id=k,previous_pixel=old,current_pixel=now,
                            displacement_px=[dx,dy],warped_box=warped))
                if boxes: references.append(boxes); links.extend(local)
        tiles,reason=conditional_tiles(e['roi'],references,intr)
        if e['status']=='SIM_MERGED': reason='MERGED_UNRESOLVED'
        pitch=row['camera_pitch_deg'];dy=.5+.2*row['time_s']
        possible=any(tof.possible(slant_box(t,e['range_bounds'],intr,(pitch-.5,pitch+.5),
            (current_yaw-dy,current_yaw+dy),row['camera_in_body_m'][2])) for t in tiles)
        before=tof.possible(e['localized_xyz']); assert not possible or before
        records.append(dict(zone_id=e['zone_id'],slot=e['target_slot'],status=e['status'],
            range_bounds=e['range_bounds'],weight=weights[e['zone_id']],old_roi=e['roi'],tiles=tiles,
            old_possible=before,possible=possible,reason=reason,reference_count=len(references),links=links,
            narrowed=tiles!=[e['roi']],raw_return_retained=True,
            attribution_score=len(references),score_semantics='observed reference count, not probability'))
    active={r['zone_id'] for r in records if r['possible']}
    score=sum(weights[z] for z in active)
    certain=any(tof.certain(e['coarse_xyz']) for e in cached['spatial_evidence'])
    only=bool(score>=THRESHOLD or certain); full=bool(only or radar['candidate'])
    base=dict(id=row['id'],score=score,certain_coarse=certain)
    return (dict(base,candidate=only,candidate_state='ALERT' if only else 'UNKNOWN'),
        dict(base,candidate=full,candidate_state='ALERT' if full else 'UNKNOWN'),
        dict(id=row['id'],window_frames=len(history)+1,track_count=len(tracks),pairs=pair_stats,returns=records))


def falsifiers():
    intr=dict(width=640,height=360);roi=[100.,100.,200.,200.]
    still,state=conditional_tiles(roi,[[roi],[roi]],intr);assert still==[roi]
    # Both moving object and moving camera produce p_relative(t)=p_world(t)-c(t).
    t=np.arange(5)*.25; stationary=np.column_stack([np.full(5,3.),np.full(5,.5),np.ones(5)])
    camera=np.column_stack([np.zeros(5),.1*t,np.zeros(5)])
    moving_object=stationary-camera
    assert np.allclose(stationary-camera,moving_object)
    # Image feature is attached to background but shares both current/prior beam.
    # Wrong linkage can trim a real contributor even with exact pixel roundtrip.
    candidate,_=conditional_tiles(roi,[[[100.,100.,160.,200.]],[[100.,100.,150.,200.]]],intr)
    hit=[180.,150.]; assert inside(roi,hit) and not any(inside(b,hit) for b in candidate)
    return dict(stationary=dict(status='PASS_NO_NEW_BOUND',tiles=still),
        coherent_motion=dict(status='COUNTEREXAMPLE_SAME_RELATIVE_OBSERVATIONS',
            stationary_world=stationary.tolist(),moving_camera=camera.tolist(),
            moving_world_fixed_camera=moving_object.tolist(),
            limit='Constructed relative-coordinate ambiguity, not rendered multisensor packets'),
        wrong_association=dict(status='COUNTEREXAMPLE_CONDITIONAL_COVERAGE_LOSS',
            old_roi=roi,candidate_tiles=candidate,untracked_native_contributor=hit,
            limit='Perfect image roundtrip cannot certify feature-to-ToF return identity'))


def evaluate(output,rows,arms,details):
    from mz115_allocation_audit import native_truth, project_point, contains
    ep=SOURCE/'capture-v1/evaluator.jsonl';lp=SOURCE/'analysis-v1/frame-report.json'
    evs=[json.loads(s) for s in ep.read_text().splitlines()];labels=read(lp)
    assert [r['id'] for r in rows]==[e['id'] for e in evs]==[e['id'] for e in labels]
    truth=[native_truth(e) for e in evs];assert truth==[l['truth'] for l in labels]
    result=dict(frames=len(rows),arms={});native=[];edge_audit=[]
    evaluator={e['id']:e for e in evs}
    def native_return(fid,zid,slot):
        z=next(z for z in evaluator[fid]['zonal_tof_native'] if z['zone_id']==zid)
        l=next(l for l in z['returned_lineage'] if l['target_index']==slot)
        return [z['private_rays'][i] for i in l['hit_indices']]
    for row,ev,detail in zip(rows,evs,details):
        for ret in detail['returns']:
            hits=native_return(row['id'],ret['zone_id'],ret['slot'])
            actor_ids={h['actor_id'] for h in hits if h.get('actor_id') is not None}
            for index,h in enumerate(hits):
                point=h['hit_point_m'];pixel=project_point(point,ev['camera'],row['rgb_intrinsics'])
                assert pixel is not None
                body=[v-o for v,o in zip(point,ev['body_origin_m'])]
                before=contains(ret['old_roi'],pixel);after=any(contains(t,pixel) for t in ret['tiles'])
                native.append(dict(id=row['id'],zone_id=ret['zone_id'],slot=ret['slot'],sample=index,
                    before=before,after=after,newly_dropped=before and not after,
                    corridor=tof.possible([(v,v) for v in body]),
                    outside_rgb=not(0<=pixel[0]<=640 and 0<=pixel[1]<=360)))
            for link in ret['links']:
                prior_hits=native_return(link['reference_id'],link['zone_id'],link['slot'])
                prior_ids={h['actor_id'] for h in prior_hits if h.get('actor_id') is not None}
                edge_audit.append(dict(id=row['id'],zone_id=ret['zone_id'],slot=ret['slot'],
                    reference_id=link['reference_id'],prior_zone=link['zone_id'],prior_slot=link['slot'],
                    actor_sets_overlap=bool(actor_ids&prior_ids),
                    known_actor_sets=bool(actor_ids and prior_ids),
                    same_actor_is_not_same_surface_certificate=True))
    for name,preds in arms.items():
        flags=[p['candidate'] for p in preds];base=arms['mz129_tof' if name.endswith('_tof') else 'mz129_full']
        m=metrics(rows,truth,flags);m['events']=events(rows,truth,flags)
        m['lost_TP_ids']=[r['id'] for r,t,b,a in zip(rows,truth,base,flags) if t and b['candidate'] and not a]
        m['added_FP_ids']=[r['id'] for r,t,b,a in zip(rows,truth,base,flags) if not t and not b['candidate'] and a]
        m['event_times_identical']=m['events']==events(rows,truth,[p['candidate'] for p in base])
        m['families']={}
        for family in sorted({l['family'] for l in labels}):
            ix=[i for i,l in enumerate(labels) if l['family']==family]
            m['families'][family]=metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
        result['arms'][name]=m
    returns=[r for d in details for r in d['returns']];pairs=[p for d in details for p in d['pairs']]
    result['temporal']=dict(window_histogram=dict(Counter(d['window_frames'] for d in details)),
        retained_returns=len(returns),return_reasons=dict(Counter(r['reason'] for r in returns)),
        narrowed_returns=sum(r['narrowed'] for r in returns),
        removed_possible_bits=sum(r['old_possible'] and not r['possible'] for r in returns),
        image_pair_records=len(pairs),track_pair_records=sum(p['shared_tracks'] for p in pairs),
        cross_zone_track_pairs=sum(p['cross_zone_tracks'] for p in pairs),
        pairs_with_motion_residual_over_2px=sum((p['median_imu_compensated_flow_px'] or 0)>2 for p in pairs))
    result['native']=dict(samples=len(native),previously_retained=sum(s['before'] for s in native),
        newly_dropped=sum(s['newly_dropped'] for s in native),
        newly_dropped_corridor=sum(s['newly_dropped'] and s['corridor'] for s in native),
        newly_dropped_outside_rgb=sum(s['newly_dropped'] and s['outside_rgb'] for s in native),
        source_link_records=len(edge_audit),source_actor_disjoint_links=sum(e['known_actor_sets'] and not e['actor_sets_overlap'] for e in edge_audit),
        unknown_environment_links=sum(not e['known_actor_sets'] for e in edge_audit))
    impact=[]
    for row,label,t,base,pred,detail in zip(rows,labels,truth,arms['mz129_full'],arms['temporal_full'],details):
        if t or not base['candidate']: continue
        rr=[r for r in detail['returns'] if r['weight'] and r['old_possible']]
        impact.append(dict(id=row['id'],family=label['family'],narrowed_triggering=sum(r['narrowed'] for r in rr),
            removed_triggering_bits=sum(not r['possible'] for r in rr),alert_removed=not pred['candidate']))
    result['fp_impact_families']={f:dict(frames=sum(i['family']==f for i in impact),
        reached=sum(i['family']==f and i['narrowed_triggering']>0 for i in impact),
        bit_changed=sum(i['family']==f and i['removed_triggering_bits']>0 for i in impact),
        removed=sum(i['family']==f and i['alert_removed'] for i in impact)) for f in sorted({i['family'] for i in impact})}
    m=result['arms']['temporal_full']
    result['retained']=m['FP']<result['arms']['mz129_full']['FP'] and not m['lost_TP_ids'] and m['event_times_identical'] and not result['native']['newly_dropped_corridor']
    result['terminal']='MZ135_CONSUMED_TEMPORAL_CONDITIONAL_GAIN' if result['retained'] else 'MZ135_TEMPORAL_FLOW_CONSTRAINT_NO_RETAINED_GAIN'
    result['intended_inheritance']='COMPONENT_OR_CHALLENGER' if result['retained'] else 'NEGATIVE_CONTROL'
    write(output/'native-retention.json',native);write(output/'native-link-audit.json',edge_audit)
    write(output/'fp-impact.json',impact);write(output/'summary.json',result)
    seal=read(output/'prediction-seal.json')
    for path,digest in seal['input_hashes'].items():assert sha(Path(path))==digest,path
    for path,digest in seal['dependencies'].items():assert sha(ROOT/path)==digest,path
    assert sha(output/'predictions.json')==seal['predictions_sha256']
    assert sha(output/'temporal-support.json')==seal['support_sha256']
    write(output/'completion.json',dict(status='PASS',summary_sha256=sha(output/'summary.json'),
        evaluator_sha256=sha(ep),labels_sha256=sha(lp),resources='Synchronous CPU run ended; no persistent allocation'))
    print(json.dumps(dict(arms={n:{k:v[k] for k in ('TP','FP','FN','UNKNOWN')} for n,v in result['arms'].items()},
        temporal=result['temporal'],native=result['native'],retained=result['retained']),indent=2))


def run(output):
    assert not output.exists() and output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    raw=SOURCE/'capture-v1/raw.jsonl';cp=CORRECTION/'predictions.json'
    bp=MZ129/'replay-v1r2/predictions.json';rp=MZ129/'radar-v1/predictions.json'
    cs=read(CORRECTION/'prediction-seal.json');bs=read(MZ129/'replay-v1r2/prediction-seal.json')
    assert sha(raw)==cs['raw_sha256'] and sha(cp)==cs['predictions_sha256'] and sha(bp)==bs['predictions_sha256']
    rs=read(MZ129/'radar-v1/prediction-seal.json')
    assert sha(rp)==rs['predictions_sha256']
    for seal in (bs,rs):
        for key in ('input_hashes','dependencies','source_hashes'):
            for path,digest in seal.get(key,{}).items():assert sha(ROOT/path)==digest,path
    rows=[json.loads(s) for s in raw.read_text().splitlines()];cached=read(cp);base=read(bp)['radar'];radar=read(rp)
    assert len(rows)==len(cached)==len(base)==len(radar)==288
    assert [r['id'] for r in rows]==[p['id'] for p in base]==[p['id'] for p in radar]
    for r,c,b,rb in zip(rows,cached,base,radar):
        assert serial(tof.allocate(r,c))==c['spatial_evidence']
        d=readout(c,column_weights(r,FOUR));assert d['score']==b['score']
        assert bool(d['score']>=THRESHOLD or d['certain_coarse'] or rb['candidate'])==b['candidate']
    images=[RGB/r['rgb_path'] for r in rows]
    for r,p in zip(rows,images):assert sha(p)==cs['rgb_sha256'][r['id']]
    output.mkdir(parents=True);cv2.setNumThreads(1)
    probe0=cv2.imread(str(images[0]),cv2.IMREAD_GRAYSCALE);probe1=cv2.imread(str(images[1]),cv2.IMREAD_GRAYSCALE)
    points=cv2.goodFeaturesToTrack(probe0,maxCorners=MAX_FEATURES,qualityLevel=.01,minDistance=7,blockSize=7)
    assert points is not None
    select_backend('batch-tensor',cpu=BackendCandidate('opencv-lk-temporal','cpu',
        lambda:cv2.calcOpticalFlowPyrLK(probe0,probe1,points,None,winSize=(21,21),maxLevel=3),
        lambda _:DeviceObservation('cpu','host CPU',f'OpenCV {cv2.__version__} LK + NumPy')),
        cpu_reason='GPU_BACKEND_UNAVAILABLE',record_path=output/'backend.json',
        capabilities=dict(opencv_cuda_devices=cv2.cuda.getCudaEnabledDeviceCount(),
            cuda_sparse_lk_exposed=hasattr(cv2,'cuda_SparsePyrLKOpticalFlow'),
            probe='actual first adjacent RGB pair LK, not end-to-end latency'))
    write(output/'model-falsifiers.json',falsifiers())
    arms=dict(mz129_full=base,mz129_tof=[],temporal_full=[],temporal_tof=[]);details=[]
    started=time.perf_counter()
    for i,(entry,history) in enumerate(causal_tracks(rows,cached,lambda r:cv2.imread(str(RGB/r['rgb_path']),cv2.IMREAD_GRAYSCALE))):
        b=base[i];flag=bool(b['score']>=THRESHOLD or b['certain_coarse'])
        arms['mz129_tof'].append(dict(id=rows[i]['id'],candidate=flag,candidate_state='ALERT' if flag else 'UNKNOWN'))
        only,full,detail=predict_frame(entry,history,radar[i])
        arms['temporal_tof'].append(only);arms['temporal_full'].append(full);details.append(detail)
    seconds=time.perf_counter()-started
    write(output/'predictions.json',arms);write(output/'temporal-support.json',details)
    deps=[Path(__file__),BRIEF]+[Path(__file__).with_name(n) for n in (
        'mz115_spatial_allocation.py','mz128_zone_weighting.py','mz124_measurement_geometry.py',
        'mz129_extent_correction.py','mz126_central_tof.py','mz132_contour_adapter.py','mz119_parallax.py')]
    write(output/'prediction-seal.json',dict(predictions_sha256=sha(output/'predictions.json'),
        support_sha256=sha(output/'temporal-support.json'),seconds=seconds,
        input_hashes={str(p):sha(p) for p in [raw,cp,bp,rp]+images},
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps},
        authority='CONSUMED_OBSERVABLE_CAUSAL_PREDICTIONS_SAVED_BEFORE_SCORING',
        cv2_version=cv2.__version__,cuda_device_count=cv2.cuda.getCudaEnabledDeviceCount()))
    evaluate(output,rows,arms,details)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
