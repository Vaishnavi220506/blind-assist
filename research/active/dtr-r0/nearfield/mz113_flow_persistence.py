"""Causal RGB optical-flow correspondence for short lived measured range.

This additive Development candidate uses pixels, not semantic/source identity.
Flow supplies 2-D correspondence only. A missing Doppler measurement explicitly
uses a hold-range assumption; it is not a measured zero velocity. Neither flow
nor repeated appearance establishes real identity or rejects Radar ghosts.
"""
import copy
import math

import cv2
import numpy as np

from mz107_rgb_association import extent_inside
from mz111_temporal_geometry import angular_box, corresponds, radar_matches

MAX_AGE_S = .5
MIN_MATCH_IOU = .2
MAX_FEATURES = 40
FEATURE_QUALITY = .01
FEATURE_DISTANCE_PX = 3.
LK_WINDOW_PX = 21
LK_LEVELS = 3
LK_ITERATIONS = 20
LK_EPSILON = .03
MAX_FB_ERROR_PX = 1.5
MIN_FLOW_POINTS = 3


def overlap(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def gray_image(image):
    if image is None or not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        return None
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return None


def valid_box(box):
    return len(box) == 4 and all(math.isfinite(x) for x in box) and box[2] > box[0] and box[3] > box[1]


def flow_box(previous_gray, current_gray, box):
    """Predict the old region by median forward/backward checked LK motion."""
    info = dict(state='FLOW_UNAVAILABLE', valid_points=0, feature_points=0, max_fb_error_px=None)
    if previous_gray is None or current_gray is None or previous_gray.shape != current_gray.shape:
        return dict(info, reason='missing_or_changed_image_shape')
    if not valid_box(box):
        return dict(info, reason='invalid_previous_box')
    height, width = previous_gray.shape
    x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x2, y2 = min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))
    if x2 <= x1 or y2 <= y1:
        return dict(info, reason='previous_box_outside_image')
    mask = np.zeros_like(previous_gray); mask[y1:y2, x1:x2] = 255
    try:
        points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=MAX_FEATURES, qualityLevel=FEATURE_QUALITY,
            minDistance=FEATURE_DISTANCE_PX, mask=mask, blockSize=3)
        info['feature_points'] = 0 if points is None else len(points)
        if points is None or len(points) < MIN_FLOW_POINTS:
            return dict(info, reason='insufficient_texture')
        parameters = dict(winSize=(LK_WINDOW_PX, LK_WINDOW_PX), maxLevel=LK_LEVELS,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, LK_ITERATIONS, LK_EPSILON))
        forward, valid_forward, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, **parameters)
        if forward is None:
            return dict(info, reason='forward_flow_missing')
        backward, valid_backward, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, forward, None, **parameters)
        if backward is None:
            return dict(info, reason='backward_flow_missing')
    except cv2.error:
        return dict(info, reason='opencv_flow_failure')
    old = points.reshape(-1, 2); new = forward.reshape(-1, 2); back = backward.reshape(-1, 2)
    error = np.linalg.norm(back-old, axis=1)
    finite = np.isfinite(new).all(axis=1) & np.isfinite(back).all(axis=1) & np.isfinite(error)
    in_image = (new[:,0] >= 0) & (new[:,0] < width) & (new[:,1] >= 0) & (new[:,1] < height)
    accepted = valid_forward.ravel().astype(bool) & valid_backward.ravel().astype(bool) & finite & in_image & (error <= MAX_FB_ERROR_PX)
    info['valid_points'] = int(accepted.sum())
    if info['valid_points'] < MIN_FLOW_POINTS:
        return dict(info, reason='insufficient_forward_backward_consistency')
    dx, dy = np.median(new[accepted]-old[accepted], axis=0)
    predicted = [float(box[0]+dx), float(box[1]+dy), float(box[2]+dx), float(box[3]+dy)]
    return dict(info, state='FLOW_MATCHABLE', predicted_box=predicted,
        displacement_px=[float(dx), float(dy)], max_fb_error_px=float(error[accepted].max()))


def predict(rows, nominal_predictions, image_loader, use_flow=True):
    if len(rows) != len(nominal_predictions):
        raise ValueError('Rows and authenticated nominal cache must have equal length')
    output = []; tracks = []; previous_episode = None; previous_time = None; previous_gray = None; imu_available = True
    for index, (row, nominal) in enumerate(zip(rows, nominal_predictions)):
        if 'id' in nominal and row['id'] != nominal['id']:
            raise ValueError('Observation/cache identity mismatch')
        now = row.get('time_s'); yaw = nominal['integrated_yaw_deg']; resets = []
        if row['episode_id'] != previous_episode:
            tracks = []; previous_gray = None; previous_time = None; imu_available = True
            resets.append('EPISODE_RESET')
        previous_episode = row['episode_id']
        time_finite = isinstance(now, (int, float)) and math.isfinite(now)
        time_valid = time_finite and (previous_time is None or now > previous_time)
        if not time_valid:
            resets.append('INVALID_OR_NONMONOTONIC_TIME')
        previous_time = now if time_finite else None
        imu_available = imu_available and bool(row['imu_valid']) and math.isfinite(yaw)
        current_gray = gray_image(image_loader(row))
        result = copy.deepcopy(nominal); support = bool(nominal['candidate'] or nominal['tof_support'])
        diagnostic = dict(mode='OPTICAL_FLOW' if use_flow else 'ANGULAR_CONTROL', propagated=[],
            seeded=0, expired=0, ambiguous=0, visual_failures=[], resets=resets)
        if not imu_available or not time_valid or current_gray is None:
            tracks = []
            if not imu_available:
                resets.append('IMU_UNAVAILABLE_UNTIL_EPISODE_RESET')
            if current_gray is None:
                resets.append('RGB_UNAVAILABLE')
        else:
            live = [t for t in tracks if 0 < now-t['range_time_s'] <= MAX_AGE_S]
            diagnostic['expired'] = len(tracks)-len(live)
            boxes = nominal['proposals']; angles = [angular_box(b, row, yaw) for b in boxes]
            flows = [flow_box(previous_gray, current_gray, t['box']) for t in live] if use_flow else []
            if use_flow:
                diagnostic['visual_failures'] = [dict(track=k, **flow) for k, flow in enumerate(flows) if flow['state'] != 'FLOW_MATCHABLE']
                previous_matches = [[k for k, flow in enumerate(flows) if valid_box(box) and flow['state'] == 'FLOW_MATCHABLE'
                                     and overlap(flow['predicted_box'], box) >= MIN_MATCH_IOU] for box in boxes]
            else:
                previous_matches = [[k for k, track in enumerate(live) if corresponds(track['angular_box'], angle)] for angle in angles]
            current_counts = [sum(k in matches for matches in previous_matches) for k in range(len(live))]
            current_matches, returns = radar_matches(row, boxes)
            next_tracks = []
            for j, box in enumerate(boxes):
                current = current_matches[j]
                if current:
                    if len(current) == 1 and len(returns[current[0]]['eligible']) == 1 and angles[j] is not None:
                        slot = current[0]; measured = returns[slot]; velocity = measured['velocity_mps']
                        velocity = velocity if velocity is not None and math.isfinite(velocity) else None
                        next_tracks.append(dict(box=box, angular_box=angles[j], range_m=measured['range_m'],
                            velocity_mps=velocity, range_time_s=now, measurement_index=index, measurement_slot=slot))
                        diagnostic['seeded'] += 1
                    else:
                        diagnostic['ambiguous'] += 1
                    continue
                previous = previous_matches[j]
                if len(previous) != 1 or current_counts[previous[0]] != 1:
                    diagnostic['ambiguous'] += bool(previous)
                    continue
                k = previous[0]; track = live[k]; age = now-track['range_time_s']; velocity = track['velocity_mps']
                distance = track['range_m'] if velocity is None else track['range_m']+velocity*age
                if not math.isfinite(distance) or distance <= 0:
                    continue
                refined = extent_inside(box, distance, row['rgb_intrinsics'], row['camera_pitch_deg'], yaw, row['camera_in_body_m'][2])
                support |= refined
                visual = dict(flows[k], match_iou=overlap(flows[k]['predicted_box'], box)) if use_flow else dict(state='ANGULAR_MATCH', match_iou=overlap(track['angular_box'], angles[j]))
                diagnostic['propagated'].append(dict(proposal=j, age_s=age, range_m=distance,
                    velocity_mps=velocity, velocity_state='UNKNOWN' if velocity is None else 'MEASURED',
                    range_assumption='HOLD_RANGE_VELOCITY_UNKNOWN' if velocity is None else 'CONSTANT_MEASURED_RADIAL_VELOCITY',
                    measurement_index=track['measurement_index'], measurement_slot=track['measurement_slot'],
                    refined_support=bool(refined), visual=visual))
                # Visual updates never refresh the age of the range measurement.
                next_tracks.append(dict(track, box=box, angular_box=angles[j]))
            tracks = next_tracks
        previous_gray = current_gray
        diagnostic['added_support'] = bool(support and not nominal['candidate'])
        result.update(candidate=bool(support), candidate_state='ALERT' if support else 'UNKNOWN', diagnostics=diagnostic)
        output.append(result)
    return output
