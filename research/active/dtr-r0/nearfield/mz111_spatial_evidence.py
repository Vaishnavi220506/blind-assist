"""Observable sparse spatial hypotheses with explicit surface approximations."""
import math
import numpy as np
from mz107_rgb_association import pixel_ray, extent_inside


def angular_box(box, row, yaw):
    intr = row['rgb_intrinsics']
    return [math.degrees(math.atan((box[0]-intr['cx'])/intr['fx']))+yaw,
            math.degrees(math.atan((intr['cy']-box[3])/intr['fy'])),
            math.degrees(math.atan((box[2]-intr['cx'])/intr['fx']))+yaw,
            math.degrees(math.atan((intr['cy']-box[1])/intr['fy']))]


def overlap(a, b):
    area = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0.


def plane_inside(box, distance, row, yaw):
    """Camera-frontoparallel surface through center range, not known 3-D shape."""
    intr = row['rgb_intrinsics']; pitch = row['camera_pitch_deg']
    center = pixel_ray((box[0]+box[2])/2, (box[1]+box[3])/2, intr, pitch, yaw)
    anchor = center*distance/math.hypot(center[0],center[1])
    normal = pixel_ray(intr['cx'], intr['cy'], intr, pitch, yaw)
    points = []
    for u in (box[0],box[2]):
        for v in (box[1],box[3]):
            ray = pixel_ray(u,v,intr,pitch,yaw)
            denom = float(np.dot(normal,ray))
            if denom <= 0:
                return False
            points.append(ray*float(np.dot(normal,anchor))/denom+[0,0,row['camera_in_body_m'][2]])
    points = np.array(points); lo = points.min(axis=0); hi = points.max(axis=0)
    return bool(hi[0]>=.2 and lo[0]<=3.6 and hi[1]>=-.3 and lo[1]<=.3 and hi[2]>=.4 and lo[2]<=2.05)


def current_returns(row, pred):
    if not row['radar_packet_received']:
        return []
    returns = []; yaw = pred['integrated_yaw_deg']
    for k,(r,a,valid) in enumerate(zip(row['radar_range_m'],row['radar_angle'],row['radar_valid'])):
        if not valid or r is None or a is None or not math.isfinite(r+a) or r <= 0:
            continue
        angle = math.radians(a+yaw)
        baseline = .2 <= r*math.cos(angle) <= 3.6 and abs(r*math.sin(angle)) <= .3
        eligible = []
        for j,box in enumerate(pred['proposals']):
            bounds = angular_box(box,row,0.)
            if bounds[0]-12 <= a <= bounds[2]+12:
                eligible.append(j)
        velocity = row['radar_velocity'][k]
        if velocity is not None and not math.isfinite(velocity):
            velocity = None
        returns.append(dict(slot=k, range_m=r, velocity_mps=velocity, baseline=baseline, candidates=eligible))
    counts = {j: sum(j in ret['candidates'] for ret in returns) for j in range(len(pred['proposals']))}
    for ret in returns:
        ret['proposal'] = ret['candidates'][0] if len(ret['candidates']) == 1 and counts[ret['candidates'][0]] == 1 else None
    return returns


def predict(rows, nominal, surface='shell', filter_range=False):
    if surface not in ('shell','plane'):
        raise ValueError('Unknown surface representation')
    assert len(rows) == len(nominal)
    output = []; previous = []; episode = None; previous_time = None
    for row,pred in zip(rows,nominal):
        now = row['time_s']
        if row['episode_id'] != episode or previous_time is None or now <= previous_time or not row['imu_valid']:
            previous = []
        episode = row['episode_id']; previous_time = now
        if not row['imu_valid']:
            raise ValueError('No true-pose fallback for invalid IMU')
        yaw = pred['integrated_yaw_deg']; boxes = pred['proposals']; returns = current_returns(row,pred)
        previous = [t for t in previous if 0 < now-t['time_s'] <= .5]
        current = [r for r in returns if r['proposal'] is not None]
        matches = {r['slot']: [i for i,t in enumerate(previous)
                   if overlap(angular_box(boxes[r['proposal']],row,yaw),t['angular_box']) >= .2] for r in current}
        new_tracks = []; support = bool(pred['tof_support'])
        for ret in returns:
            j = ret['proposal']; distance = ret['range_m']; variance = .08**2; filtered = False
            if j is not None:
                eligible = matches[ret['slot']]
                if filter_range and len(eligible) == 1 and sum(eligible[0] in m for m in matches.values()) == 1:
                    old = previous[eligible[0]]; dt = now-old['time_s']
                    if old['velocity_mps'] is not None:
                        prior = old['range_m']+old['velocity_mps']*dt
                        prior_var = old['variance']+(.15*dt)**2
                        if prior > 0 and abs(distance-prior) <= .35:
                            gain = prior_var/(prior_var+variance)
                            distance = prior+gain*(distance-prior); variance *= gain; filtered = True
                box = boxes[j]
                refined = (extent_inside(box,distance,row['rgb_intrinsics'],row['camera_pitch_deg'],yaw,row['camera_in_body_m'][2])
                           if surface == 'shell' else plane_inside(box,distance,row,yaw))
                new_tracks.append(dict(time_s=now, angular_box=angular_box(box,row,yaw), range_m=distance,
                                       variance=variance, velocity_mps=ret['velocity_mps']))
                ret.update(box=box, range_m=distance, range_variance=variance, range_filtered=filtered,
                           state='ASSOCIATED_PROXY', height_state='RGB_RANGE_PROXY', support=refined,
                           evidence_age_s=0., sources=['RGB','RADAR','IMU'])
            else:
                ret.update(state='AMBIGUOUS' if ret['candidates'] else 'UNASSOCIATED',
                           height_state='HEIGHT_UNKNOWN', support=ret['baseline'], evidence_age_s=0., sources=['RADAR','IMU'])
            support |= ret['support']
        previous = new_tracks
        output.append(dict(candidate=bool(support), baseline=pred['baseline'], tof_support=pred['tof_support'],
                           candidate_state='ALERT' if support else 'UNKNOWN', spatial_evidence=returns))
    return output
