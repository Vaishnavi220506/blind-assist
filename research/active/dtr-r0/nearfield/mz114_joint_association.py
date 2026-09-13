"""Frozen observable global Radar/RGB assignments with optional short flow prior.

Null assignments remain explicit. Only unanimous non-null choices across every
assignment within best+.5 resolve a pair. Missing history adds zero cost; unknown
Doppler explicitly holds the last measured range. Neither correspondence nor
assignment cost proves identity or rejects a ghost. No evaluator/source access.
"""
import math

from mz111_spatial_evidence import plane_inside
from mz113_flow_persistence import flow_box, gray_image, overlap, valid_box


RADAR_MARGIN_DEG = 12.
BEARING_SCALE_DEG = 6.
UNMATCHED_COST = 2.
RANGE_SCALE_M = .20
MAX_TEMPORAL_COST = 9.
ASSIGNMENT_MARGIN = .5
MAX_AGE_S = .5
MIN_FLOW_IOU = .2


def assignments(pair_costs):
    """Exact injective enumeration over at most four returns, including null."""
    if len(pair_costs)>4:raise ValueError('Expected at most four current Radar returns')
    for costs in pair_costs:
        if any(j is None or not math.isfinite(c) or c<0 for j,c in costs.items()):
            raise ValueError('Pair costs must be finite nonnegative values for proposal indices')
    best=UNMATCHED_COST*len(pair_costs)
    retained=[]

    def visit(index,used,chosen,cost):
        nonlocal best,retained
        # An optimistic remaining cost is valid even if its independent choices
        # collide. This prunes work without dropping a globally eligible answer.
        lower=sum(min([UNMATCHED_COST]+[c for j,c in costs.items() if j not in used])
                  for costs in pair_costs[index:])
        if cost+lower>best+ASSIGNMENT_MARGIN+1e-12:return
        if index==len(pair_costs):
            if cost<best:
                best=cost
                retained=[a for a in retained if a['cost']<=best+ASSIGNMENT_MARGIN+1e-12]
            if cost<=best+ASSIGNMENT_MARGIN+1e-12:
                retained.append(dict(proposals=list(chosen),cost=float(cost)))
            return
        choices=[(j,c) for j,c in pair_costs[index].items() if j not in used]+[(None,UNMATCHED_COST)]
        choices.sort(key=lambda p:(p[1],-1 if p[0] is None else p[0]))
        for j,extra in choices:
            visit(index+1,used if j is None else used|{j},chosen+[j],cost+extra)

    visit(0,set(),[],0.)
    retained.sort(key=lambda a:(a['cost'],tuple(-1 if j is None else j for j in a['proposals'])))
    resolved=[]
    for k in range(len(pair_costs)):
        choices={a['proposals'][k] for a in retained}
        resolved.append(next(iter(choices)) if len(choices)==1 and None not in choices else None)
    return dict(best_cost=float(best),retained=retained,resolved=resolved)


def current_returns(row,nominal,boxes,yaw):
    if not row['radar_packet_received']:return []
    result=[];intr=row['rgb_intrinsics']
    values=list(zip(row['radar_range_m'],row['radar_angle'],row['radar_valid']))
    if len(values)>4:raise ValueError('Expected at most four Radar slots')
    for slot,(distance,bearing,valid) in enumerate(values):
        if not valid or distance is None or bearing is None or not math.isfinite(distance+bearing) or distance<=0:continue
        point_support=False
        if yaw is not None:
            angle=math.radians(bearing+yaw)
            point_support=.2<=distance*math.cos(angle)<=3.6 and abs(distance*math.sin(angle))<=.3
        costs=[]
        for j,box in enumerate(boxes):
            if not valid_box(box):continue
            edges=[math.degrees(math.atan((box[k]-intr['cx'])/intr['fx'])) for k in (0,2)]
            low,high=min(edges),max(edges)
            if not low-RADAR_MARGIN_DEG<=bearing<=high+RADAR_MARGIN_DEG:continue
            outside=max(low-bearing,0.,bearing-high)
            costs.append(dict(proposal=j,bearing_interval_deg=[low,high],bearing_outside_deg=outside,
                              bearing_cost=(outside/BEARING_SCALE_DEG)**2))
        velocities=row.get('radar_velocity',[])
        velocity=velocities[slot] if slot<len(velocities) else None
        if velocity is not None and not math.isfinite(velocity):velocity=None
        result.append(dict(slot=slot,range_m=distance,bearing_deg=bearing,velocity_mps=velocity,
                           velocity_state='UNKNOWN' if velocity is None else 'MEASURED',
                           baseline=bool(point_support),candidates=[p['proposal'] for p in costs],
                           pair_costs=costs))
    return result


def predict(rows,nominal,image_loader,use_temporal=True):
    if len(rows)!=len(nominal):raise ValueError('Rows and nominal cache must have equal length')
    output=[];tracks=[];previous_gray=None;episode=None;previous_time=None;imu_available=True
    for index,(row,pred) in enumerate(zip(rows,nominal)):
        if 'id' in pred and pred['id']!=row['id']:raise ValueError('Observation/cache identity mismatch')
        resets=[]
        if row['episode_id']!=episode:
            tracks=[];previous_gray=None;previous_time=None;imu_available=True
            resets.append('EPISODE_RESET')
        episode=row['episode_id'];now=row.get('time_s');raw_yaw=pred.get('integrated_yaw_deg')
        time_finite=isinstance(now,(int,float)) and math.isfinite(now)
        time_valid=time_finite and (previous_time is None or now>previous_time)
        if not time_valid:resets.append('INVALID_OR_NONMONOTONIC_TIME')
        previous_time=now if time_finite else None
        yaw_finite=isinstance(raw_yaw,(int,float)) and math.isfinite(raw_yaw)
        imu_available=imu_available and bool(row['imu_valid']) and yaw_finite
        yaw=raw_yaw if imu_available else None
        image=gray_image(image_loader(row))
        image_valid=image is not None
        if not image_valid:resets.append('RGB_UNAVAILABLE')
        if not imu_available:resets.append('IMU_UNAVAILABLE_UNTIL_EPISODE_RESET')
        association_valid=image_valid and imu_available and time_valid
        boxes=pred['proposals'] if association_valid else []
        returns=current_returns(row,pred,boxes,yaw)
        diagnostic=dict(mode='GLOBAL_WITH_FLOW_RANGE_PRIOR' if use_temporal else 'CURRENT_GLOBAL_ONLY',
                        resets=resets,seeded=0,visual_only_carried=0,expired=0,prior_matches=[],visual_failures=[])
        priors={};live=[]
        if not association_valid:
            tracks=[]
        elif use_temporal:
            live=[t for t in tracks if 0<now-t['range_time_s']<=MAX_AGE_S]
            diagnostic['expired']=len(tracks)-len(live)
            flows=[flow_box(previous_gray,image,t['box']) for t in live]
            matches={j:[k for k,f in enumerate(flows) if valid_box(box) and f['state']=='FLOW_MATCHABLE'
                         and overlap(f['predicted_box'],box)>=MIN_FLOW_IOU] for j,box in enumerate(boxes)}
            counts=[sum(k in candidates for candidates in matches.values()) for k in range(len(live))]
            diagnostic['visual_failures']=[dict(track=k,**f) for k,f in enumerate(flows) if f['state']!='FLOW_MATCHABLE']
            for j,matched in matches.items():
                if len(matched)!=1 or counts[matched[0]]!=1:continue
                k=matched[0];track=live[k];age=now-track['range_time_s'];velocity=track['velocity_mps']
                distance=track['range_m'] if velocity is None else track['range_m']+velocity*age
                if not math.isfinite(distance) or distance<=0:continue
                prior=dict(proposal=j,track=k,age_s=age,predicted_range_m=distance,
                           measured_range_m=track['range_m'],range_time_s=track['range_time_s'],
                           velocity_mps=velocity,velocity_state='UNKNOWN' if velocity is None else 'MEASURED',
                           range_assumption='HOLD_RANGE_VELOCITY_UNKNOWN' if velocity is None else 'CONSTANT_MEASURED_RADIAL_VELOCITY',
                           measurement_index=track['measurement_index'],measurement_slot=track['measurement_slot'],
                           visual=dict(flows[k],match_iou=overlap(flows[k]['predicted_box'],boxes[j])))
                priors[j]=prior;diagnostic['prior_matches'].append(prior)
        pair_costs=[]
        for ret in returns:
            costs={}
            for pair in ret['pair_costs']:
                prior=priors.get(pair['proposal'])
                temporal_cost=0. if prior is None else min(MAX_TEMPORAL_COST,((ret['range_m']-prior['predicted_range_m'])/RANGE_SCALE_M)**2)
                pair.update(temporal_cost=temporal_cost,total_cost=pair['bearing_cost']+temporal_cost,
                            temporal_state='NO_SUPPORTED_PRIOR_ZERO_COST' if prior is None else prior['range_assumption'],
                            prior=prior)
                costs[pair['proposal']]=pair['total_cost']
            ret['null_explanation']=dict(proposal=None,cost=UNMATCHED_COST,
                                         state='EXPLICIT_UNMATCHED_NOT_CLEARANCE')
            pair_costs.append(costs)
        solved=assignments(pair_costs)
        diagnostic.update(best_assignment_cost=solved['best_cost'],retained_assignments=solved['retained'],
                          return_slots=[r['slot'] for r in returns],retained_assignment_count=len(solved['retained']))
        support=bool(pred['tof_support']);new_tracks=[];seeded_proposals=set()
        for k,ret in enumerate(returns):
            j=solved['resolved'][k]
            choices={a['proposals'][k] for a in solved['retained']}
            ret.update(proposal=j,retained_choices=sorted(choices,key=lambda q:-1 if q is None else q),
                       evidence_age_s=0.)
            if j is not None:
                inside=plane_inside(boxes[j],ret['range_m'],row,yaw)
                ret.update(box=list(boxes[j]),state='GLOBAL_CONSENSUS_ASSOCIATED_PROXY',height_state='RGB_RANGE_PROXY',
                           support=bool(inside),sources=['RGB','RADAR','IMU'])
                if use_temporal:
                    new_tracks.append(dict(box=list(boxes[j]),range_m=ret['range_m'],velocity_mps=ret['velocity_mps'],
                                           range_time_s=now,measurement_index=index,measurement_slot=ret['slot']))
                    seeded_proposals.add(j);diagnostic['seeded']+=1
            else:
                ret.update(state='GLOBAL_UNRESOLVED' if ret['candidates'] else 'UNASSOCIATED',
                           height_state='HEIGHT_UNKNOWN',support=ret['baseline'],sources=['RADAR','IMU'])
            support|=ret['support']
        if not association_valid:
            # The supplied existing baseline is the authority for this fallback;
            # invalid IMU does not synthesize a zero yaw or native-pose substitute.
            support|=bool(pred['baseline'])
            diagnostic['baseline_fallback']=True
        if association_valid and use_temporal:
            for j,prior in priors.items():
                if j in seeded_proposals:continue
                # Only the optical footprint advances. Range, Doppler, source
                # timestamp and measurement identity remain exactly unchanged.
                new_tracks.append(dict(live[prior['track']],box=list(boxes[j])))
                diagnostic['visual_only_carried']+=1
        tracks=new_tracks;previous_gray=image
        output.append(dict(id=row['id'],candidate=bool(support),candidate_state='ALERT' if support else 'UNKNOWN',
                           baseline=bool(pred['baseline']),tof_support=bool(pred['tof_support']),
                           spatial_evidence=returns,diagnostics=diagnostic))
    return output
