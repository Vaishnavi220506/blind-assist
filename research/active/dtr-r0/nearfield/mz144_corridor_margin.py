"""Direct signed corridor margin from observable regional-depth hypotheses.

RGB proposal ownership, uniform reflectance and body-forward planes are explicit
hypotheses. No private rays, native surfaces, labels or incumbent decisions enter.
All public slots remain in audit; this module never vetoes another sensor branch.
"""
import math

import numpy as np

from mz115_spatial_allocation import intersection, slant_envelope, zone_box
from mz125_observable_correction import foreground
from mz136_boundary_geometry import camera_to_body, proposals
from mz136_rectified_edges import refine, transform
from mz143_corridor_features import _public


METHOD = dict(schema='MZ144_PUBLIC_DEPTH_CONSENSUS_CORRIDOR_MARGIN_V1',
    proposal='ALL_MZ125_BOXES_ALL_MZ136_SEEDS_FIRST_FINE_REPLACES_OWN_SEED',
    eligible_returns='EVERY_SIM_VALID_SLOT_DISTINCT_ZONE_VOTING',
    range_interval='PROPOSAL_ZONE_INTERSECTION_RADIAL_3SIGMA_BODY_FORWARD_ENCLOSURE',
    depth='REGIONAL_MEAN_RANGE_TIMES_MEAN_QX_SQUARED_OVER_MEAN_QX',
    quadrature='NATIVE_PIXEL_CENTERS_INSIDE_ROI_FOREGROUND_RECTANGLE_IF_EMPTY',
    assumptions='UNIFORM_REFLECTANCE_VISIBLE_PROPOSAL_OWNERSHIP_BODY_FORWARD_PLANE',
    consensus_minimum_zones=2, selection='MOST_DISTINCT_ZONES_THEN_NEAREST_DEPTH',
    competitors='RETAIN_ALL_MODES_AND_SIGN_DISAGREEMENT_IN_AUDIT',
    score='MAX_PROPOSAL_MIN_FORWARD_MARGIN_LATERAL_OVERLAP_HEIGHT_OVERLAP_METERS',
    no_candidate='SCORE_MINUS_10_UNKNOWN_NOT_CLEAR', no_alarms=True,
    backend_reason='GPU_BACKEND_UNAVAILABLE')


def _range_factor(row, roi, mask, yaw):
    """Uniform pixel quadrature, not actual sensor subray coordinates/lineage."""
    intr = row['rgb_intrinsics']; h, w = mask.shape
    l, t, r, b = roi
    l, t = max(0, int(math.ceil(l))), max(0, int(math.ceil(t)))
    r, b = min(w, int(math.floor(r))+1), min(h, int(math.floor(b))+1)
    if l >= r or t >= b:
        return None
    yy, xx = np.mgrid[t:b, l:r]
    foreground_pixels = mask[t:b, l:r] > 0
    use_mask = bool(foreground_pixels.any())
    choose = foreground_pixels if use_mask else np.ones(xx.shape, bool)
    xx, yy = xx[choose], yy[choose]
    rays = np.stack([np.ones(len(xx)), (xx-intr['cx'])/intr['fx'],
                     (intr['cy']-yy)/intr['fy']], -1)
    directions = (rays/np.linalg.norm(rays, axis=1)[:, None])@camera_to_body(row, yaw).T
    qx = directions[:, 0]
    if not len(qx) or not np.isfinite(qx).all() or (qx <= 0).any():
        return None
    return dict(factor=float(np.mean(qx*qx)/np.mean(qx)), pixels=len(qx),
                mask_used=use_mask, qx_min=float(qx.min()), qx_max=float(qx.max()))


def _depth_modes(observations):
    """Endpoint sweep of interval support with one vote per physical zone.

    Two returns from one zone are alternatives, never two independent witnesses.
    Distinct supporter sets remain separate even when their intervals overlap.
    """
    if not observations:
        return []
    edges = sorted({x for o in observations for x in o['interval']})
    probes = sorted(set(edges+[(a+b)/2 for a, b in zip(edges, edges[1:])]))
    modes = {}
    for point in probes:
        per_zone = {}
        for obs in observations:
            if obs['interval'][0]-1e-10 <= point <= obs['interval'][1]+1e-10:
                per_zone.setdefault(obs['zone'], []).append(obs)
        if len(per_zone) < METHOD['consensus_minimum_zones']:
            continue
        chosen = [min(items, key=lambda o: (abs(o['depth']-point), o['depth'], o['slot']))
                  for _, items in sorted(per_zone.items())]
        key = tuple((o['zone'], o['slot']) for o in chosen)
        if key in modes:
            continue
        lo = max(o['interval'][0] for o in chosen)
        hi = min(o['interval'][1] for o in chosen)
        if lo > hi+1e-9:
            continue
        depth = float(np.clip(np.median([o['depth'] for o in chosen]), lo, hi))
        modes[key] = dict(depth_m=depth, interval_m=[float(lo), float(max(lo, hi))],
            coarse_depth_m=float(np.median([o['coarse_depth'] for o in chosen])),
            distinct_zones=len(chosen), returns=[list(k) for k in key],
            signal_r2_sum=float(sum(o['signal_r2'] for o in chosen)),
            correction_spread_m=float(np.ptp([o['depth'] for o in chosen])),
            rectangle_fallback_returns=sum(not o['quadrature']['mask_used'] for o in chosen))
    return sorted(modes.values(), key=lambda m: (-m['distinct_zones'], m['depth_m'], m['returns']))


def _surface(row, pixels, depth, yaw):
    """Visible boundary at a body-forward plane; not unseen full object extent."""
    intr = row['rgb_intrinsics']; uv = np.asarray(pixels, float)
    rays = np.stack([np.ones(len(uv)), (uv[:, 0]-intr['cx'])/intr['fx'],
                     (intr['cy']-uv[:, 1])/intr['fy']], -1)@camera_to_body(row, yaw).T
    if not np.isfinite(rays).all() or (rays[:, 0] <= 1e-9).any() or depth <= 0:
        return None
    points = depth*rays/rays[:, 0, None]+np.asarray(row['camera_in_body_m'])
    lo, hi = points.min(0), points.max(0)
    # A zero-thickness plane has forward occupancy but zero overlap LENGTH.
    # Its signed forward clearance is distance to the nearest query endpoint.
    forward = min(lo[0]-.2, 3.6-hi[0])
    lateral = min(hi[1], .3)-max(lo[1], -.3)
    height = min(hi[2], 2.05)-max(lo[2], .4)
    return dict(score=float(min(forward, lateral, height)),
        forward_margin_m=float(forward), lateral_overlap_m=float(lateral), height_overlap_m=float(height),
        xyz_bounds=np.stack([lo, hi], -1).tolist(), vertices=points.tolist())


def _proposals(row, image, yaw, mask, boxes):
    seeds, _ = proposals(row, image, yaw)
    # refine uses its ID only for output bookkeeping; supply a fixed nonidentity.
    edge = refine(dict(row, id='PUBLIC_OBSERVATION'), image, yaw)
    fine_seed = tuple(edge['seed']['box']) if edge.get('candidate') else None
    result = {}
    for kind, batch in (('MZ125_BOX', boxes), ('MZ136_SEED', [s['box'] for s in seeds])):
        for box in batch:
            key = tuple(map(float, box))
            l, t, r, b = key
            pixels = [[l, t], [r, t], [r, b], [l, b]]
            if key == fine_seed:
                continue
            if key not in result:
                result[key] = dict(kind=kind, box=list(key), pixels=pixels)
    if fine_seed is not None:
        # Gradient fitting trims the seed's vertical span. Refine only lateral
        # edges; retain the full observed seed height rather than shrinking it.
        l, t, r, b = fine_seed
        matrix = np.asarray(edge['homography'], float)
        rectified = transform([[l, t], [r, t], [r, b], [l, b]], matrix)
        top, bottom = float(rectified[:, 1].min()), float(rectified[:, 1].max())
        left, right = edge['candidate']['rectified_edges_px']
        pixels = transform([[left, top], [right, top], [right, bottom], [left, bottom]],
                           np.linalg.inv(matrix))
        lo, hi = pixels.min(0), pixels.max(0)
        key = ('FINE', *fine_seed)
        result[key] = dict(kind='MZ136_FINE_REPLACES_SEED', box=[*lo.tolist(), *hi.tolist()],
            pixels=pixels.tolist(), coarse_box=list(fine_seed))
    return list(result.values()), edge['state']


def extract_margin(row, image, yaw):
    """One deterministic public RGB/ToF/IMU extraction; no cutoff or alarm.

    Ambiguous depth modes remain observable hypotheses. ``unknown`` means no
    credible two-zone proposal exists, not proven clear space. Sign disagreement
    is reported separately for the runner's coverage/ambiguity accounting.
    """
    public = _public(row); intr = public['rgb_intrinsics']
    if image.dtype != np.uint8 or image.shape != (intr['height'], intr['width'], 3):
        raise ValueError('Native unresized BGR uint8 required')
    if not math.isfinite(float(yaw)):
        raise ValueError('Finite public integrated yaw required')
    raw_slots = []
    zones = sorted(public['tof_zones'], key=lambda z: z['zone_id'])
    for z in zones:
        box = zone_box(z, intr)
        outside = not (0 <= box[0] <= box[2] <= intr['width']-1 and 0 <= box[1] <= box[3] <= intr['height']-1)
        for slot, target in enumerate(z['targets']):
            raw_slots.append(dict(zone=z['zone_id'], slot=slot, target=dict(target),
                theta_bounds_deg=z['theta_bounds_deg'], phi_bounds_deg=z['phi_bounds_deg'],
                outside_or_partial_rgb=outside, retained_for_independent_sensor_audit=True))
    audit = dict(schema=METHOD['schema'], raw_slots=raw_slots, public_slots=len(raw_slots),
        all_public_slots_retained=True, no_evaluator_or_identity_inputs=True,
        plane_and_ownership_are_hypotheses=True, no_rgb_veto_to_other_branches=True,
        tof_packet_received=bool(public['tof_packet_received']), imu_valid=bool(public['imu_valid']))
    if not public['tof_packet_received'] or not public['imu_valid']:
        audit.update(state='MISSING_PACKET_OR_IMU', proposal_count=0, credible_proposals=0, ambiguous_proposals=0)
        return dict(score=-10., unknown=True, candidates=[], audit=audit)
    boxes, mask = foreground(image)
    visual, edge_state = _proposals(public, image, yaw, mask, boxes)
    candidates = []
    for proposal in visual:
        observations = []
        for zone in zones:
            roi = intersection(zone_box(zone, intr), proposal['box'])
            if roi is None:
                continue
            quadrature = _range_factor(public, roi, mask, yaw)
            if quadrature is None:
                continue
            a = math.radians(sum(zone['theta_bounds_deg'])/2)
            b = math.radians(sum(zone['phi_bounds_deg'])/2)
            ray = np.array([1., math.tan(a), math.tan(b)])
            center_qx = float((camera_to_body(public, yaw)@(ray/np.linalg.norm(ray)))[0])
            for slot, target in enumerate(zone['targets']):
                if target['status'] != 'SIM_VALID':
                    continue
                distance, sigma, signal = [float(target[k]) for k in
                    ('distance_m', 'range_noise_sigma_m', 'signal_strength_proxy')]
                if not all(map(math.isfinite, (distance, sigma, signal))) or distance <= 0 or sigma < 0 or signal < 0:
                    continue
                bounds = slant_envelope(roi, (max(.02, distance-3*sigma), distance+3*sigma), intr,
                    (public['camera_pitch_deg'],)*2, (float(yaw),)*2, 0.)[0]
                if bounds[1] <= 0:
                    continue
                observations.append(dict(zone=zone['zone_id'], slot=slot,
                    interval=[max(.001, float(bounds[0])), float(bounds[1])],
                    depth=distance*quadrature['factor'], coarse_depth=distance*center_qx,
                    signal_r2=signal*distance*distance, quadrature=quadrature))
        modes = _depth_modes(observations)
        record = dict(**proposal, observation_count=len(observations), modes=modes,
                      credible=False, unknown=True)
        if modes:
            for mode in modes:
                mode['surface'] = _surface(public, proposal['pixels'], mode['depth_m'], yaw)
                mode['coarse_surface'] = _surface(public, proposal['pixels'], mode['coarse_depth_m'], yaw)
                mode['endpoint_scores'] = [surface['score'] if surface else None for surface in
                    (_surface(public, proposal['pixels'], d, yaw) for d in mode['interval_m'])]
            eligible = [m for m in modes if m['surface'] is not None]
            if eligible:
                chosen = eligible[0]
                signs = {m['surface']['score'] >= 0 for m in eligible}
                record.update(credible=True, unknown=False, score=chosen['surface']['score'],
                    coarse_score=chosen['coarse_surface']['score'] if chosen['coarse_surface'] else -10.,
                    selected_depth_m=chosen['depth_m'], selected_returns=chosen['returns'],
                    mode_sign_disagreement=len(signs) > 1, selected_surface=chosen['surface'])
        candidates.append(record)
    credible = [c for c in candidates if c['credible']]
    score = max((c['score'] for c in credible), default=-10.)
    audit.update(state='PUBLIC_PLANE_HYPOTHESES' if credible else 'NO_TWO_ZONE_PROPOSAL',
        proposal_count=len(candidates), credible_proposals=len(credible), edge_state=edge_state,
        ambiguous_proposals=sum(c.get('mode_sign_disagreement', False) for c in credible),
        hypothesis_score_range=[min((c['score'] for c in credible), default=-10.), float(score)])
    return dict(score=float(score), unknown=not bool(credible), candidates=candidates, audit=audit)
