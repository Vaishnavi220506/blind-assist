"""Observable-only finite-zone range allocation; all identities remain proxies.

Positive signal patterns may select a visual box; missing zones are never free
space. Unresolved and mixed components retain their entire measured footprint.
ToF is slant range; Radar's older horizontal-range readout remains separate.
"""
import math
import numpy as np
import mz108_competitive_association as visual
import mz111_spatial_evidence as radar
import mz113_flow_persistence as flow
from mz107_rgb_association import pixel_ray
from mz109_interval_extent import add, sub, mul, div, square, trig

DEPTH_SPAN_M = .25
MIN_ZONES = 2
MIN_COVERAGE = .8
MIN_COSINE = .9
MIN_MARGIN = .1
BOX_PAD_PX = 2.
DISTANCES = (1.,1.4,1.8,2.2,2.6,3.,3.4,3.6,3.8,4.)


def zone_box(zone, intr):
    a,b=zone['theta_bounds_deg'];c,d=zone['phi_bounds_deg']
    return [intr['cx']+intr['fx']*math.tan(math.radians(a)),
            intr['cy']-intr['fy']*math.tan(math.radians(d)),
            intr['cx']+intr['fx']*math.tan(math.radians(b)),
            intr['cy']-intr['fy']*math.tan(math.radians(c))]


def intersection(a,b):
    out=[max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
    return out if out[2]>out[0] and out[3]>out[1] else None


def area(box):
    return 0. if box is None else (box[2]-box[0])*(box[3]-box[1])


def slant_envelope(box, ranges, intr, pitches, yaws, height):
    """Interval enclosure over all pixels, ranges and bounded pitch/yaw."""
    a=((box[0]-intr['cx'])/intr['fx'],(box[2]-intr['cx'])/intr['fx'])
    b=((intr['cy']-box[3])/intr['fy'],(intr['cy']-box[1])/intr['fy'])
    norm=tuple(math.sqrt(v) for v in add((1.,1.),add(square(a),square(b))))
    p=tuple(map(math.radians,pitches));y=tuple(map(math.radians,yaws))
    cp,sp,cy,sy=trig(p,True),trig(p),trig(y,True),trig(y)
    f=sub(cp,mul(b,sp));up=add(sp,mul(b,cp))
    x=mul(ranges,div(sub(mul(f,cy),mul(a,sy)),norm))
    side=mul(ranges,div(add(mul(f,sy),mul(a,cy)),norm))
    z=add((height,height),mul(ranges,div(up,norm)))
    return [x,side,z]


def possible(xyz, distance=3.6):
    return all(hi>=lo2 and lo<=hi2 for (lo,hi),(lo2,hi2) in
               zip(xyz,((.2,distance),(-.3,.3),(.4,2.05))))


def certain(xyz, distance=3.6):
    return all(lo>=lo2 and hi<=hi2 for (lo,hi),(lo2,hi2) in
               zip(xyz,((.2,distance),(-.3,.3),(.4,2.05))))


def assign_groups(evidence, boxes):
    """Complete-link forward-depth cohorts, positive signal only, no zero tests."""
    valid=sorted((e for e in evidence if e['status']=='SIM_VALID'),key=lambda e:(e['forward_depth'],e['zone_id'],e['target_slot']))
    groups=[]
    for item in valid:
        if not groups or item['forward_depth']-groups[-1][0]['forward_depth']>DEPTH_SPAN_M:
            groups.append([])
        groups[-1].append(item)
    for g,items in enumerate(groups):
        for item in items:item['group']=g
        # Two components in one zone may represent competing surfaces: unresolved.
        if len({e['zone_id'] for e in items})!=len(items) or len(items)<MIN_ZONES:continue
        weights=np.array([e['signal']*e['range_m']**2 for e in items],float)
        candidates=[]
        for j,box in enumerate(boxes):
            shape=np.array([area(intersection(e['zone_box'],box))/area(e['zone_box']) for e in items])
            coverage=float(weights[shape>0].sum()/weights.sum())
            norm=float(np.linalg.norm(shape)*np.linalg.norm(weights))
            cosine=float(np.dot(shape,weights)/norm) if norm else 0.
            if coverage>=MIN_COVERAGE:candidates.append((cosine,j,coverage))
        candidates.sort(key=lambda t:(-t[0],t[1]))
        winner=candidates[0] if candidates else None
        margin=winner[0]-(candidates[1][0] if len(candidates)>1 else 0.) if winner else 0.
        for item in items:item['association_candidates']=[dict(proposal=j,cosine=s,coverage=c) for s,j,c in candidates]
        if winner and winner[0]>=MIN_COSINE and margin>=MIN_MARGIN:
            for item in items:
                b=boxes[winner[1]];roi=intersection(item['zone_box'],[b[0]-BOX_PAD_PX,b[1]-BOX_PAD_PX,b[2]+BOX_PAD_PX,b[3]+BOX_PAD_PX])
                if roi:
                    item.update(proposal=winner[1],roi=roi,state='RGB_SIGNAL_ASSOCIATION_PROXY',association_margin=margin)


def allocate(row, pred):
    out=[];intr=row['rgb_intrinsics'];yaw=pred['integrated_yaw_deg']
    dy=.5+.2*row['time_s'];p=row['camera_pitch_deg'];height=row['camera_in_body_m'][2]
    if not row['tof_packet_received']:return out
    for zone in row['tof_zones']:
        box=zone_box(zone,intr)
        for slot,target in enumerate(zone['targets']):
            r=target['distance_m'];sigma=target['range_noise_sigma_m']
            if target['status'] not in ('SIM_VALID','SIM_MERGED') or not math.isfinite(r+sigma) or r<=0 or sigma<0:continue
            # A merged mean has unknown component depths: whole sensing range,
            # not the Gaussian noise of its reported average, bounds the evidence.
            bounds=(.02,4.) if target['status']=='SIM_MERGED' else (max(.02,r-3*sigma),r+3*sigma)
            a=math.radians(sum(zone['theta_bounds_deg'])/2);e=math.radians(sum(zone['phi_bounds_deg'])/2)
            out.append(dict(zone_id=zone['zone_id'],target_slot=slot,status=target['status'],group=None,proposal=None,
                zone_box=box,roi=box,range_m=r,range_bounds=bounds,signal=target['signal_strength_proxy'],
                forward_depth=r/math.sqrt(1+math.tan(a)**2+math.tan(e)**2),state='WHOLE_ZONE_UNRESOLVED',
                height_state='TOF_FOOTPRINT_INTERVAL',sources=['TOF','IMU'],evidence_age_s=0.))
    assign_groups(out,pred['proposals'])
    for item in out:
        args=(item['range_bounds'],intr,(p-.5,p+.5),(yaw-dy,yaw+dy),height)
        item['coarse_xyz']=slant_envelope(item['zone_box'],*args)
        item['localized_xyz']=slant_envelope(item['roi'],*args)
        if item['proposal'] is not None:item['sources']=['RGB','TOF','IMU']
    return out


def legacy_empty_tof(row):
    return dict(row,tof_packet_received=False,tof64_range_m=[],tof64_status=[],tof64_theta_deg=[],tof64_phi_deg=[])


def radar_xyz(box, distance, row, yaw, plane=True):
    intr=row['rgb_intrinsics'];pitch=row['camera_pitch_deg'];height=row['camera_in_body_m'][2]
    normal=pixel_ray(intr['cx'],intr['cy'],intr,pitch,yaw)
    center=pixel_ray((box[0]+box[2])/2,(box[1]+box[3])/2,intr,pitch,yaw)
    anchor=center*distance/math.hypot(center[0],center[1]);points=[]
    for u in (box[0],box[2]):
        for v in (box[1],box[3]):
            ray=pixel_ray(u,v,intr,pitch,yaw)
            scale=float(np.dot(normal,anchor))/float(np.dot(normal,ray)) if plane else distance/math.hypot(ray[0],ray[1])
            points.append(ray*scale+[0,0,height])
    pts=np.array(points)
    return list(zip(pts.min(axis=0).tolist(),pts.max(axis=0).tolist()))


def raw_radar(row,yaw,distance):
    if not row['radar_packet_received']:return False
    return any(valid and r is not None and a is not None and math.isfinite(r+a) and
        .2<=r*math.cos(math.radians(a+yaw))<=distance and abs(r*math.sin(math.radians(a+yaw)))<=.3
        for r,a,valid in zip(row['radar_range_m'],row['radar_angle'],row['radar_valid']))


def predict(rows, image_loader):
    empty=[legacy_empty_tof(r) for r in rows]
    nominal=visual.predict(empty,image_loader)
    current=radar.predict(empty,nominal,surface='plane',filter_range=True)
    persisted=flow.predict(empty,nominal,image_loader)
    maps=[allocate(r,n) for r,n in zip(rows,nominal)]
    values={}
    for distance in DISTANCES:
        arms={k:[] for k in ('baseline','nominal','allocation','tof_coarse','tof_allocation','center_diagnostic')}
        for row,n,cur,old,evidence in zip(rows,nominal,current,persisted,maps):
            yaw=n['integrated_yaw_deg'];base=raw_radar(row,yaw,distance);common=False
            for ret in cur['spatial_evidence']:
                if ret['proposal'] is not None:common|=possible(radar_xyz(ret['box'],ret['range_m'],row,yaw),distance)
                else:
                    a=math.radians(row['radar_angle'][ret['slot']]+yaw);r=ret['range_m']
                    common|=.2<=r*math.cos(a)<=distance and abs(r*math.sin(a))<=.3
            additions=any(possible(radar_xyz(n['proposals'][v['proposal']],v['range_m'],row,yaw,False),distance)
                          for v in old['diagnostics']['propagated'])
            common|=additions and not base
            coarse=any(possible(e['coarse_xyz'],distance) for e in evidence)
            local=any(certain(e['coarse_xyz'],distance) or possible(e['localized_xyz'],distance) for e in evidence)
            point=False
            for e in evidence:
                if e['status']!='SIM_VALID':continue
                b=e['zone_box'];d=pixel_ray((b[0]+b[2])/2,(b[1]+b[3])/2,row['rgb_intrinsics'],row['camera_pitch_deg'],yaw)
                xyz=d*e['range_m']+[0,0,row['camera_in_body_m'][2]]
                point|=possible([(float(x),float(x)) for x in xyz],distance)
            flags=dict(baseline=base or coarse,nominal=common or coarse,allocation=common or local,
                       tof_coarse=coarse,tof_allocation=local,center_diagnostic=common or point)
            for arm,flag in flags.items():
                arms[arm].append(dict(candidate=bool(flag),candidate_state='ALERT' if flag else 'UNKNOWN'))
            if distance==3.6:
                arms['allocation'][-1].update(spatial_evidence=evidence,common_radar=bool(common),
                    proposals=n['proposals'],integrated_yaw_deg=yaw,unseen_zones=[z['zone_id'] for z in row['tof_zones'] if not z['targets']])
        values[str(distance)]=arms
    # Shared Radar readout reproduces its frozen 3.6m formula exactly.
    for a,c,f,n in zip(values['3.6']['allocation'],current,persisted,nominal):
        assert a['common_radar']==bool(c['candidate'] or (f['candidate'] and not n['candidate']))
    return values
