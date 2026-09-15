"""Observation-only native-resolution corridor features; no learned decisions.

Every public ToF slot survives in ``sensor``. RGB boxes and region-center depths
generate multiple *plane hypotheses*, never point-hit identities or complete
surface enclosures. Missing/merged/outside-image support is not cleared by RGB.
CPU: GPU_BACKEND_UNAVAILABLE for the inherited NumPy/OpenCV frontend.
"""
import math

import cv2
import numpy as np

from mz115_spatial_allocation import area, intersection, slant_envelope, zone_box
from mz125_observable_correction import foreground
from mz136_boundary_geometry import camera_to_body, proposals


METHOD = dict(
    schema='MZ143_PUBLIC_CORRIDOR_FEATURES_V1', native_rgb_resolution=True,
    tof_slots=128, radar_slots=4, no_trained_parameters=True,
    geometry='ALL_OVERLAPPING_RGB_PROPOSAL_AND_SINGLE_VALID_RETURN_PLANE_HYPOTHESES',
    merged='RAW_FEATURES_AND_FULL_0.02_TO_4M_SUPPORT_NO_POINT_DEPTH_HYPOTHESIS',
    support='INTERVAL_ENCLOSURE_AT_PUBLIC_IMU_ORIENTATION_AND_RANGE_3SIGMA',
    no_return='FINITE_ZERO_SENTINEL_WITH_EXPLICIT_MASK_NOT_CLEAR_SPACE',
    pooling='ALL_NEAREST_RANGE_AND_CORRIDOR_HEIGHT_FORWARD_HYPOTHESES_QUANTILES',
    unknown_imu='PUBLIC_ORIENTATION_VALUE_WITH_INVALID_MASK_NOT_CERTIFIED_GEOMETRY',
    backend_reason='GPU_BACKEND_UNAVAILABLE',
)
SLOT_NAMES = ('present', 'usable', 'merged', 'range_m', 'sigma_m', 'signal_r2',
              'center_x', 'center_y', 'center_z',
              'support_x_lo', 'support_x_hi', 'support_y_lo', 'support_y_hi',
              'support_z_lo', 'support_z_hi')
HYPOTHESIS_NAMES = ('range_m', 'signal_r2', 'shared_zone_fraction',
                    'center_x', 'center_y', 'center_z', 'forward_overlap_m',
                    'side_overlap_m', 'height_overlap_m', 'side_q10_m',
                    'side_q90_m', 'width_m', 'mask_fraction',
                    'left_positive_edge', 'right_negative_edge',
                    'boundary_mask_fraction', 'boundary_gradient')
QUANTILES = (0., .25, .5, .75, 1.)


def _finite(value, default=0.):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _public(row):
    """Create a fresh allowlisted object; IDs and evaluator metadata cannot flow."""
    result = {k: row[k] for k in ('camera_in_body_m', 'camera_pitch_deg',
        'tof_packet_received', 'imu_valid', 'radar_packet_received',
        'radar_range_m', 'radar_angle', 'radar_velocity', 'radar_valid')}
    result['rgb_intrinsics'] = {k: row['rgb_intrinsics'][k]
        for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy')}
    result['tof_zones'] = [dict(
        zone_id=int(z['zone_id']), theta_bounds_deg=list(z['theta_bounds_deg']),
        phi_bounds_deg=list(z['phi_bounds_deg']),
        targets=[{k: t[k] for k in ('distance_m', 'range_noise_sigma_m',
                                   'signal_strength_proxy', 'status')}
                 for t in z['targets']]) for z in row['tof_zones']]
    if sorted(z['zone_id'] for z in result['tof_zones']) != list(range(64)):
        raise ValueError('Exactly the 64 unique public zones are required')
    if any(len(z['targets']) > 2 for z in result['tof_zones']):
        raise ValueError('The fixed public sensor contract has at most two slots')
    for key in ('radar_range_m', 'radar_angle', 'radar_velocity', 'radar_valid'):
        if len(result[key]) != 4:
            raise ValueError('The fixed public radar contract has four slots')
    return result


def _project_pixels(uv, depth, intr, rotation, origin):
    uv = np.asarray(uv, float)
    rays = np.stack([np.ones(len(uv)), (uv[:, 0]-intr['cx'])/intr['fx'],
                     (intr['cy']-uv[:, 1])/intr['fy']], -1)
    return depth*rays@rotation.T+origin


def _overlap(low, high, a, b):
    return min(high, b)-max(low, a)


def _box_descriptor(box, mask, gradient):
    h, w = mask.shape
    l, t, r, b = box
    l, t = max(0, int(math.floor(l))), max(0, int(math.floor(t)))
    r, b = min(w, int(math.ceil(r))), min(h, int(math.ceil(b)))
    if r <= l or b <= t:
        return None
    roi = mask[t:b, l:r]
    yy, xx = np.nonzero(roi)
    if len(xx):
        qx = np.quantile(xx+l, [.1, .9])
        ym = float(np.median(yy+t))
    else:
        qx = np.array([l, r-1], float)
        ym = (t+b-1)/2
    left = gradient[t:b, max(0, l-2):min(w, l+3)]
    right = gradient[t:b, max(0, r-3):min(w, r+2)]
    return dict(box=[l, t, r, b], qx=qx, ym=ym, mask_fraction=float(roi.mean()),
        left_positive_edge=float(np.maximum(left, 0).mean()),
        right_negative_edge=float(np.maximum(-right, 0).mean()))


def _boundary_samples(descriptor, depth, intr, rotation, origin, mask, gradient):
    # For each image row solve the exact lateral plane intersection at y=+/-.3.
    _, top, _, bottom = descriptor['box']
    vv = np.arange(top, bottom, dtype=np.float32)
    if not len(vv) or abs(rotation[1, 1]) < 1e-6:
        return 0., 0.
    rays_up = (intr['cy']-vv)/intr['fy']
    values, edges = [], []
    h, w = mask.shape
    for side in (-.3, .3):
        u = intr['cx']+intr['fx']*((side-origin[1])/depth-rotation[1, 0]
                                  -rotation[1, 2]*rays_up)/rotation[1, 1]
        for offset in (-1., 0., 1.):
            x = u+offset
            valid = (x >= 0) & (x <= w-1) & (vv >= 0) & (vv <= h-1)
            if not valid.any():
                continue
            mx, my = x[valid].astype(np.float32)[None], vv[valid][None]
            values.extend(cv2.remap(mask, mx, my, cv2.INTER_LINEAR)[0].tolist())
            edges.extend(np.abs(cv2.remap(gradient, mx, my, cv2.INTER_LINEAR)[0]).tolist())
    return (float(np.mean(values)), float(np.mean(edges))) if values else (0., 0.)


def extract(row, image, yaw):
    """Return finite fixed float32 vectors, names and observation-only audit.

    ``yaw`` is a causal public IMU integration in degrees supplied by the runner.
    The function does not read episode IDs/time or integrate motion itself.
    ``image`` is unchanged native-size BGR uint8. No model, label or alarm input.
    """
    public = _public(row)
    intr = public['rgb_intrinsics']
    if image.dtype != np.uint8 or image.shape != (intr['height'], intr['width'], 3):
        raise ValueError('Unresized BGR uint8 matching public intrinsics required')
    if not math.isfinite(float(yaw)):
        raise ValueError('Public integrated yaw must be finite')
    rotation = camera_to_body(public, float(yaw))
    origin = np.asarray(public['camera_in_body_m'], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Finite public camera extrinsics required')
    sensor, sensor_names = [], []

    def add(prefix, names, values):
        sensor_names.extend(prefix+n for n in names)
        sensor.extend(map(float, values))

    add('global.', ('tof_packet', 'radar_packet', 'imu_valid', 'yaw_sin', 'yaw_cos',
                   'pitch_deg', 'camera_x', 'camera_y', 'camera_z'),
        (public['tof_packet_received'], public['radar_packet_received'], public['imu_valid'],
         math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), public['camera_pitch_deg'], *origin))
    audit_slots, observations = [], []
    for zone in sorted(public['tof_zones'], key=lambda z: z['zone_id']):
        zid = zone['zone_id']; box = zone_box(zone, intr)
        add(f'zone{zid:02d}.', ('theta_lo', 'theta_hi', 'phi_lo', 'phi_hi'),
            (*zone['theta_bounds_deg'], *zone['phi_bounds_deg']))
        theta = math.radians(sum(zone['theta_bounds_deg'])/2)
        phi = math.radians(sum(zone['phi_bounds_deg'])/2)
        center_ray = np.array([1., math.tan(theta), math.tan(phi)])
        norm = np.linalg.norm(center_ray)
        full = 0 <= box[0] <= box[2] <= intr['width']-1 and 0 <= box[1] <= box[3] <= intr['height']-1
        for slot in range(2):
            target = zone['targets'][slot] if slot < len(zone['targets']) else None
            usable = bool(target and public['tof_packet_received'] and target['status'] in ('SIM_VALID', 'SIM_MERGED')
                and _finite(target['distance_m'], -1) > 0 and _finite(target['range_noise_sigma_m'], -1) >= 0
                and _finite(target['signal_strength_proxy'], -1) >= 0)
            merged = bool(target and target['status'] == 'SIM_MERGED')
            distance = _finite(target['distance_m']) if target else 0.
            sigma = _finite(target['range_noise_sigma_m']) if target else 0.
            strength = _finite(target['signal_strength_proxy']) if target else 0.
            center = rotation@(distance*center_ray/norm)+origin if usable else np.zeros(3)
            support = np.zeros((3, 2))
            if usable:
                ranges = (.02, 4.) if merged else (max(.02, distance-3*sigma), distance+3*sigma)
                support = np.asarray(slant_envelope(box, ranges, intr,
                    (public['camera_pitch_deg'],)*2, (float(yaw),)*2, 0.))+origin[:, None]
            add(f'zone{zid:02d}.slot{slot}.', SLOT_NAMES,
                (target is not None, usable, merged, distance, sigma, strength*distance**2,
                 *center, *support.flatten()))
            if target:
                audit_slots.append(dict(zone=zid, slot=slot, usable=usable, status=target['status'],
                    outside_or_partial_rgb=not full, regional_center_is_hypothesis=True))
            if usable and not merged:
                observations.append(dict(zone=zid, slot=slot, box=box, range=distance,
                    depth=distance/norm, signal_r2=strength*distance**2))
    for slot in range(4):
        r, a, v = [public[k][slot] for k in ('radar_range_m', 'radar_angle', 'radar_velocity')]
        valid = bool(public['radar_packet_received'] and public['radar_valid'][slot]
                     and r is not None and a is not None and _finite(r, -1) > 0)
        angle = math.radians(_finite(a)+yaw)
        add(f'radar{slot}.', ('valid', 'range_m', 'angle_sin', 'angle_cos', 'velocity',
                             'velocity_valid', 'body_x', 'body_y'),
            (valid, _finite(r), math.sin(angle) if valid else 0., math.cos(angle) if valid else 0.,
             _finite(v), valid and v is not None, _finite(r)*math.cos(angle) if valid else 0.,
             _finite(r)*math.sin(angle) if valid else 0.))

    boxes, mask_u8 = foreground(image)
    seeds, _ = proposals(public, image, float(yaw))
    # Keep every distinct proposal. Neither proposal rank nor ToF strength picks
    # a winning object/return ownership; the learner sees pooled alternatives.
    all_boxes = sorted({tuple(map(float, b)) for b in boxes} |
                       {tuple(map(float, p['box'])) for p in seeds})
    mask = (mask_u8 > 0).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    gradient = cv2.Scharr(gray, cv2.CV_32F, 1, 0)/32.
    descriptors = [d for b in all_boxes if (d := _box_descriptor(b, mask, gradient)) is not None]
    hypotheses = []
    for obs in observations:
        for desc in descriptors:
            overlap = intersection(obs['box'], desc['box'])
            if overlap is None:
                continue
            l, t, r, b = desc['box']
            points = _project_pixels([(l, t), (r, t), (r, b), (l, b)], obs['depth'], intr, rotation, origin)
            lo, hi = points.min(0), points.max(0)
            center = (lo+hi)/2
            quantile_points = _project_pixels([(desc['qx'][0], desc['ym']), (desc['qx'][1], desc['ym'])],
                                              obs['depth'], intr, rotation, origin)
            boundary_mask, boundary_gradient = _boundary_samples(desc, obs['depth'], intr, rotation, origin, mask, gradient)
            hypotheses.append([obs['range'], obs['signal_r2'], area(overlap)/area(obs['box']),
                *center, _overlap(lo[0], hi[0], .2, 3.6), _overlap(lo[1], hi[1], -.3, .3),
                _overlap(lo[2], hi[2], .4, 2.05), *sorted(quantile_points[:, 1]), hi[1]-lo[1],
                desc['mask_fraction'], desc['left_positive_edge'], desc['right_negative_edge'],
                boundary_mask, boundary_gradient])
    values = np.asarray(hypotheses, float).reshape(-1, len(HYPOTHESIS_NAMES))
    geometry_names = ['rgb.mask_fraction', 'rgb.foreground_boxes', 'rgb.narrow_seeds',
        'rgb.distinct_boxes', 'hypotheses.count', 'slots.usable_valid', 'slots.merged',
        'slots.outside_or_partial_rgb', 'rgb.positive_gradient_mean', 'rgb.negative_gradient_mean']
    geometry = [float(mask.mean()), len(boxes), len(seeds), len(descriptors), len(values), len(observations),
        sum(a['status'] == 'SIM_MERGED' for a in audit_slots), sum(a['outside_or_partial_rgb'] for a in audit_slots),
        float(np.maximum(gradient, 0).mean()), float(np.maximum(-gradient, 0).mean())]
    groups = dict(all=values,
        nearest=values[values[:, 0] <= values[:, 0].min()+.25] if len(values) else values,
        forward_height=values[(values[:, 6] >= 0) & (values[:, 8] >= 0)] if len(values) else values)
    for group_name, group in groups.items():
        geometry_names.append(group_name+'.count'); geometry.append(len(group))
        pooled = np.quantile(group, QUANTILES, axis=0).T if len(group) else np.zeros((len(HYPOTHESIS_NAMES), len(QUANTILES)))
        for name, quantiles in zip(HYPOTHESIS_NAMES, pooled):
            for q, value in zip(QUANTILES, quantiles):
                geometry_names.append(f'{group_name}.{name}.q{int(100*q):02d}')
                geometry.append(float(value))
    sensor, geometry = np.asarray(sensor, np.float32), np.asarray(geometry, np.float32)
    assert len(sensor) == len(sensor_names) and len(geometry) == len(geometry_names)
    assert len(sensor)+len(geometry) < 2500
    if not np.isfinite(sensor).all() or not np.isfinite(geometry).all():
        raise ValueError('Nonfinite feature from public sensor contract')
    audit = dict(schema=METHOD['schema'], public_slots=len(audit_slots), encoded_slots=128,
        all_public_slots_encoded=True, slots=audit_slots, hypothesis_count=len(hypotheses),
        distinct_rgb_proposals=len(descriptors), no_rgb_support_is_not_clear=True,
        no_native_evaluator_inputs=True, original_rgb_shape=list(image.shape),
        sensor_features=len(sensor), geometry_features=len(geometry))
    return dict(sensor=sensor, geometry=geometry, audit=audit,
                sensor_names=sensor_names, geometry_names=geometry_names)
