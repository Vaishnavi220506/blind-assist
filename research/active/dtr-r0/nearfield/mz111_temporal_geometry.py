"""Causal short range transport through current, uniquely corresponding RGB.

Frozen mechanism-only contrast: retain nominal alerts and add only transported
past range, never a direct current no-ToF spatial alert. No confidence/hit count
is accumulated. Angular correspondence and constant radial velocity are working
assumptions, not identity proof; persistent ghosts may seed tracks too.
"""
import copy
import math

from mz107_rgb_association import extent_inside, pixel_ray

MAX_AGE_S = .5
RADAR_MARGIN_DEG = 12.
MIN_ANGULAR_IOU = .2
MAX_CENTER_SHIFT_DEG = 8.
MIN_AREA_RATIO = .25
MAX_AREA_RATIO = 4.


def angular_box(box, row, yaw):
    if len(box) != 4 or not all(math.isfinite(x) for x in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    points = [pixel_ray(u, v, row['rgb_intrinsics'], row['camera_pitch_deg'], yaw)
              for u in (box[0], box[2]) for v in (box[1], box[3])]
    az = [yaw + (math.degrees(math.atan2(p[1], p[0])) - yaw + 180) % 360 - 180
          for p in points]
    el = [math.degrees(math.atan2(p[2], math.hypot(p[0], p[1]))) for p in points]
    return min(az), min(el), max(az), max(el)


def corresponds(a, b):
    if a is None or b is None:
        return False
    aa = (a[2]-a[0]) * (a[3]-a[1]); ab = (b[2]-b[0]) * (b[3]-b[1])
    if min(aa, ab) <= 0 or not MIN_AREA_RATIO <= ab/aa <= MAX_AREA_RATIO:
        return False
    overlap = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    shift = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
    return overlap/(aa+ab-overlap) >= MIN_ANGULAR_IOU and shift <= MAX_CENTER_SHIFT_DEG


def radar_matches(row, boxes):
    """All valid current returns compete, including those without velocity."""
    matches = [[] for _ in boxes]
    returns = {}
    if not row['radar_packet_received']:
        return matches, returns
    intr = row['rgb_intrinsics']
    for k, (r, angle, valid) in enumerate(zip(row['radar_range_m'], row['radar_angle'], row['radar_valid'])):
        if not valid or r is None or angle is None or not math.isfinite(r+angle) or r <= 0:
            continue
        eligible = []
        for j, box in enumerate(boxes):
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            angles = [math.degrees(math.atan((u-intr['cx'])/intr['fx'])) for u in (box[0], box[2])]
            if min(angles)-RADAR_MARGIN_DEG <= angle <= max(angles)+RADAR_MARGIN_DEG:
                eligible.append(j)
                matches[j].append(k)
        velocities = row.get('radar_velocity', [])
        velocity = velocities[k] if k < len(velocities) else None
        returns[k] = dict(range_m=r, velocity_mps=velocity, eligible=eligible)
    return matches, returns


def predict(rows, nominal_predictions):
    if len(rows) != len(nominal_predictions):
        raise ValueError('Rows and authenticated nominal cache must have equal length')
    output = []; tracks = []; previous_episode = None; previous_time = None; imu_available = True
    for index, (row, nominal) in enumerate(zip(rows, nominal_predictions)):
        if 'id' in nominal and row['id'] != nominal['id']:
            raise ValueError('Observation/cache identity mismatch')
        now = row['time_s']; yaw = nominal['integrated_yaw_deg']; reasons = []
        if row['episode_id'] != previous_episode:
            tracks = []; previous_time = None; imu_available = True
            reasons.append('EPISODE_RESET')
        previous_episode = row['episode_id']
        time_valid = math.isfinite(now)
        if not time_valid or (previous_time is not None and now <= previous_time):
            tracks = []; reasons.append('INVALID_OR_NONMONOTONIC_TIME')
            time_valid = False
        previous_time = now if math.isfinite(now) else None
        imu_available = imu_available and bool(row['imu_valid']) and math.isfinite(yaw)
        result = copy.deepcopy(nominal)
        # Nominal already preserves unassociated Radar; explicit ToF OR also
        # protects the independent-support invariant for supplied caches.
        support = bool(nominal['candidate'] or nominal['tof_support'])
        diagnostic = dict(propagated=[], seeded=0, expired=0, ambiguous=0, resets=reasons)
        if not imu_available or not time_valid:
            tracks = []
            if not imu_available:
                reasons.append('IMU_UNAVAILABLE_UNTIL_EPISODE_RESET')
        else:
            live = [t for t in tracks if 0 < now-t['range_time_s'] <= MAX_AGE_S]
            diagnostic['expired'] = len(tracks)-len(live)
            boxes = nominal['proposals']
            angles = [angular_box(box, row, yaw) for box in boxes]
            current_matches, returns = radar_matches(row, boxes)
            previous_matches = [[k for k, t in enumerate(live) if corresponds(t['angular_box'], angle)] for angle in angles]
            current_counts = [sum(k in matches for matches in previous_matches) for k in range(len(live))]
            next_tracks = []
            for j, box in enumerate(boxes):
                current = current_matches[j]
                if current:
                    # Any current ambiguous range/identity prevents stale rescue.
                    if len(current) == 1 and len(returns[current[0]]['eligible']) == 1:
                        k = current[0]; measured = returns[k]; v = measured['velocity_mps']
                        if v is not None and math.isfinite(v) and angles[j] is not None:
                            next_tracks.append(dict(angular_box=angles[j], range_m=measured['range_m'], velocity_mps=v,
                                range_time_s=now, measurement_index=index, measurement_slot=k))
                            diagnostic['seeded'] += 1
                    else:
                        diagnostic['ambiguous'] += 1
                    continue
                previous = previous_matches[j]
                if len(previous) != 1 or current_counts[previous[0]] != 1:
                    diagnostic['ambiguous'] += bool(previous)
                    continue
                track = live[previous[0]]
                age = now-track['range_time_s']
                distance = track['range_m'] + track['velocity_mps'] * age
                if not math.isfinite(distance) or distance <= 0:
                    continue
                refined = extent_inside(box, distance, row['rgb_intrinsics'], row['camera_pitch_deg'], yaw, row['camera_in_body_m'][2])
                support |= refined
                diagnostic['propagated'].append(dict(proposal=j, age_s=age, range_m=distance,
                    velocity_mps=track['velocity_mps'], measurement_index=track['measurement_index'],
                    measurement_slot=track['measurement_slot'], refined_support=bool(refined)))
                # Updating the visual footprint does not refresh range lifetime.
                next_tracks.append(dict(track, angular_box=angles[j]))
            tracks = next_tracks
        diagnostic['added_support'] = bool(support and not nominal['candidate'])
        result.update(candidate=bool(support), candidate_state='ALERT' if support else 'UNKNOWN', diagnostics=diagnostic)
        output.append(result)
    return output
