"""Conditional common-plane bounds from finite-zone positive range anchors.

No image-center depth is asserted as truth. Ray and range uncertainty are
propagated through a contraction bound; surface identity remains an assumption.
"""
import copy
import math
import numpy as np
import mz115_spatial_allocation as zonal
import mz116_four_sensor as incumbent
from mz109_interval_extent import add,mul,div,square

MIN_ANCHORS=6
MAX_DESIGN_CONDITION=100.
BOX_ERROR_PX=2.
RANGE_ERROR_M=.13  # existing3sigma=.12 plus half of .02m quantization


def unit_ray_intervals(box,intr):
    a=((box[0]-intr['cx'])/intr['fx'],(box[2]-intr['cx'])/intr['fx'])
    b=((intr['cy']-box[3])/intr['fy'],(intr['cy']-box[1])/intr['fy'])
    norm=tuple(math.sqrt(v) for v in add((1.,1.),add(square(a),square(b))))
    return [div((1.,1.),norm),div(a,norm),div(b,norm)]


def center_ray(box,intr):
    q=np.array([1.,((box[0]+box[2])/2-intr['cx'])/intr['fx'],(intr['cy']-(box[1]+box[3])/2)/intr['fy']])
    return q/np.linalg.norm(q)


def fit_plane(anchors,intr):
    if len(anchors)<MIN_ANCHORS:return None
    if len({a['zone_id']//8 for a in anchors})<2 or len({a['zone_id']%8 for a in anchors})<2:return None
    U=np.array([center_ray(a['zone_box'],intr) for a in anchors]);r=np.array([a['range_m'] for a in anchors])
    if np.any(r<=RANGE_ERROR_M):return None
    norms=np.linalg.norm(U,axis=0)
    if np.any(norms<1e-10):return None
    condition=float(np.linalg.cond(U/norms))
    if not math.isfinite(condition) or condition>MAX_DESIGN_CONDITION:return None
    M=np.linalg.pinv(U);y=1/r;beta=M@y;denom=U@beta
    if np.any(denom<=0):return None
    residual=float(np.max(np.abs(1/denom-r)))
    if residual>RANGE_ERROR_M:return None
    delta=np.array([[max(abs(lo-u),abs(hi-u)) for (lo,hi),u in zip(unit_ray_intervals(a['zone_box'],intr),u)] for a,u in zip(anchors,U)])
    measurement=np.maximum(1/(r-RANGE_ERROR_M)-y,y-1/(r+RANGE_ERROR_M))
    A=np.abs(M)@delta;radius=float(max(abs(np.linalg.eigvals(A))))
    if not math.isfinite(radius):return None
    error=None
    if radius<1-1e-6:
        error=np.linalg.solve(np.eye(3)-A,np.abs(M)@(measurement+delta@np.abs(beta)))
        if not np.isfinite(error).all() or np.any(error<0):return None
    return dict(beta=beta.tolist(),coefficient_error=None if error is None else error.tolist(),A_radius=radius,normalized_design_condition=condition,
                residual_m=residual,anchor_keys=[[a['zone_id'],a['target_slot']] for a in anchors],
                authority='CONDITIONAL_COMMON_PLANE_AND_WORKING_NOISE_BOUNDS_NOT_OBJECT_IDENTITY')


def range_interval(model,box,intr,padding=0.):
    if model['coefficient_error'] is None:return None
    denominator=(0.,0.)
    for b,e,u in zip(model['beta'],model['coefficient_error'],unit_ray_intervals(box,intr)):
        denominator=add(denominator,mul((b-e,b+e),u))
    if denominator[0]<=0:return None
    interval=(1/denominator[1]-padding,1/denominator[0]+padding)
    return (max(.02,interval[0]),min(4.,interval[1])) if interval[0]<=4. and interval[1]>=.02 else None


def refine(row,prediction,use_interval=True):
    result=copy.deepcopy(prediction);items=result['spatial_evidence'];boxes=result['proposals'];intr=row['rgb_intrinsics']
    models=[];counts={}
    for item in items:counts[item['zone_id']]=counts.get(item['zone_id'],0)+1
    for j,b in enumerate(boxes):
        inner=[b[0]+BOX_ERROR_PX,b[1]+BOX_ERROR_PX,b[2]-BOX_ERROR_PX,b[3]-BOX_ERROR_PX]
        anchors=[]
        for item in items:
            z=item['zone_box'];overlapping=[k for k,box in enumerate(boxes) if zonal.intersection(z,box)]
            if (item['status']=='SIM_VALID' and counts[item['zone_id']]==1 and overlapping==[j] and
                inner[0]<=z[0] and inner[1]<=z[1] and inner[2]>=z[2] and inner[3]>=z[3]):
                anchors.append(item)
        model=fit_plane(anchors,intr)
        if model:models.append(dict(model,proposal=j))
    result['surface_models']=models
    for item in items:
        if item['status']!='SIM_MERGED':continue
        z=item['zone_box'];overlapping=[k for k,b in enumerate(boxes) if zonal.intersection(z,b)]
        eligible=[m for m in models if overlapping==[m['proposal']]]
        if len(eligible)!=1:continue
        model=eligible[0];j=model['proposal'];b=boxes[j]
        roi=zonal.intersection(z,[b[0]-2,b[1]-2,b[2]+2,b[3]+2])
        bounds=range_interval(model if use_interval else dict(model,coefficient_error=[0.,0.,0.]),roi,intr,0. if use_interval else RANGE_ERROR_M)
        if bounds is None or bounds[1]<item['range_m']-RANGE_ERROR_M or bounds[0]>item['range_m']+RANGE_ERROR_M:continue
        yaw=result['integrated_yaw_deg'];dy=.5+.2*row['time_s'];pitch=row['camera_pitch_deg']
        xyz=zonal.slant_envelope(roi,bounds,intr,(pitch-.5,pitch+.5),(yaw-dy,yaw+dy),row['camera_in_body_m'][2])
        item.update(original_range_bounds=item['range_bounds'],original_localized_xyz=item['localized_xyz'],original_roi=item['roi'],
                    range_bounds=bounds,localized_xyz=xyz,roi=roi,proposal=j,plane_proposal=j,plane_model=model,
                    plane_geometry='COEFFICIENT_INTERVAL' if use_interval else 'FIXED_COEFFICIENT_RANGE_PAD',
                    surface_state='CONDITIONAL_SINGLE_SURFACE_PROXY',sources=['RGB','TOF','IMU'])
    return result


def predict(rows,image_loader):
    curves=incumbent.predict(rows,image_loader)
    evidence=[refine(r,p) for r,p in zip(rows,curves['3.6']['resolution_guard'])]
    proxy=[refine(r,p,False) for r,p in zip(rows,curves['3.6']['resolution_guard'])]
    for distance,arms in curves.items():
        arms['surface']=copy.deepcopy(evidence)
        arms['surface_nominal_proxy']=copy.deepcopy(proxy)
    # Reuse observable Radar traces to separate its independent contribution.
    import mz111_spatial_evidence as radar
    import mz113_flow_persistence as flow
    nominal=[dict(proposals=p['proposals'],integrated_yaw_deg=p['integrated_yaw_deg'],tof_support=False,
        candidate=zonal.raw_radar(r,p['integrated_yaw_deg'],3.6),baseline=zonal.raw_radar(r,p['integrated_yaw_deg'],3.6))
        for r,p in zip(rows,evidence)]
    empty=[zonal.legacy_empty_tof(r) for r in rows];current=radar.predict(empty,nominal,surface='plane',filter_range=True)
    persisted=flow.predict(empty,nominal,image_loader)
    for distance,arms in curves.items():
        d=float(distance)
        for row,n,cur,old,guard,out,original in zip(rows,nominal,current,persisted,arms['resolution_guard'],arms['surface'],curves['3.6']['resolution_guard']):
            yaw=n['integrated_yaw_deg'];base=zonal.raw_radar(row,yaw,d);common=False
            for ret in cur['spatial_evidence']:
                if ret['proposal'] is not None:common|=zonal.possible(zonal.radar_xyz(ret['box'],ret['range_m'],row,yaw),d)
                else:
                    r=ret['range_m'];a=math.radians(row['radar_angle'][ret['slot']]+yaw)
                    common|=.2<=r*math.cos(a)<=d and abs(r*math.sin(a))<=.3
            common|=not base and any(zonal.possible(zonal.radar_xyz(n['proposals'][p['proposal']],p['range_m'],row,yaw,False),d) for p in old['diagnostics']['propagated'])
            common|=bool(guard['guard_events'])
            original_tof=any(zonal.certain(item['coarse_xyz'],d) or zonal.possible(item['localized_xyz'],d) for item in original['spatial_evidence'])
            assert bool(common or original_tof)==guard['candidate']
            tof=any(zonal.certain(item['coarse_xyz'],d) or zonal.possible(item['localized_xyz'],d) for item in out['spatial_evidence'])
            out.update(candidate=bool(common or tof),common_radar=bool(common),guard_events=guard['guard_events'],guard_added=guard['guard_added'],candidate_state='ALERT' if common or tof else 'UNKNOWN')
            if d==3.6:assert common==bool(guard['common_radar'] or guard['guard_events'])
        for guard,out,proxy_out in zip(arms['resolution_guard'],arms['surface'],arms['surface_nominal_proxy']):
            common=out['common_radar'];tof=any(zonal.certain(item['coarse_xyz'],d) or zonal.possible(item['localized_xyz'],d) for item in proxy_out['spatial_evidence'])
            proxy_out.update(candidate=bool(common or tof),common_radar=common,guard_events=guard['guard_events'],guard_added=guard['guard_added'],candidate_state='ALERT' if common or tof else 'UNKNOWN')
    return curves
