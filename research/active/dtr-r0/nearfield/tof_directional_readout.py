"""Camera-relative angular support, additive to the frozen collision alert.

Only observable ToF zones enter this engineering readout. Connected regions are
not object identities. Missing returns remain UNKNOWN; merged distances are not
used for a metric estimate. Fixed quality weights are illustrative, not fitted.
"""
import math

QUALITY = {'SIM_VALID': 1.0, 'SIM_MERGED': .5}
CENTER_HALF_ANGLE_DEG = 5.625


def zone_weights(zone):
    theta = sum(zone['theta_bounds_deg']) / 2
    phi = sum(zone['phi_bounds_deg']) / 2
    h = 'LEFT' if theta < -CENTER_HALF_ANGLE_DEG else 'RIGHT' if theta > CENTER_HALF_ANGLE_DEG else 'CENTER'
    v = 'UPPER' if phi > CENTER_HALF_ANGLE_DEG else 'LOWER' if phi < -CENTER_HALF_ANGLE_DEG else 'MIDDLE'
    return dict(theta_deg=theta, phi_deg=phi, horizontal=h, vertical=v,
                horizontal_position_weight=theta / 2.8125,
                vertical_position_weight=phi / 2.8125)


def readout(row, *, equal_weights=False):
    if not row['tof_packet_received']:
        return dict(state='UNKNOWN_PACKET_MISSING', regions=[], zone_evidence=[])
    zones = row['tof_zones']
    if len(zones) != 64 or {z['zone_id'] for z in zones} != set(range(64)):
        raise ValueError('Exactly 64 uniquely identified zones required')
    evidence = {}
    for z in zones:
        valid = [(i, t) for i, t in enumerate(z['targets'])
                 if t['status'] in QUALITY and math.isfinite(t['distance_m']) and t['distance_m'] > 0]
        if not valid:
            continue
        # One zone contributes once, irrespective of its number of target slots.
        quality = max(1. if equal_weights else QUALITY[t['status']] for _, t in valid)
        evidence[z['zone_id']] = dict(zone_id=z['zone_id'], **zone_weights(z),
            quality_weight=quality, targets=[dict(slot=i, **t) for i, t in valid])
    remaining = set(evidence)
    groups = []
    while remaining:
        seed = min(remaining); remaining.remove(seed); group = [seed]; stack = [seed]
        while stack:
            idx = stack.pop(); r, c = divmod(idx, 8)
            neighbors = [rr*8+cc for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                         if 0 <= rr < 8 and 0 <= cc < 8]
            for j in neighbors:
                if j in remaining:
                    remaining.remove(j); group.append(j); stack.append(j)
        groups.append(sorted(group))
    regions = []
    for group in groups:
        entries = [evidence[i] for i in group]; total = sum(e['quality_weight'] for e in entries)
        theta = sum(e['theta_deg']*e['quality_weight'] for e in entries)/total
        phi = sum(e['phi_deg']*e['quality_weight'] for e in entries)/total
        horizontal = 'LEFT' if theta < -CENTER_HALF_ANGLE_DEG else 'RIGHT' if theta > CENTER_HALF_ANGLE_DEG else 'CENTER'
        vertical = 'UPPER' if phi > CENTER_HALF_ANGLE_DEG else 'LOWER' if phi < -CENTER_HALF_ANGLE_DEG else 'MIDDLE'
        # Metric summary from valid returns only; preserve multiple-return ambiguity.
        distances = [min(t['distance_m'] for t in e['targets'] if t['status']=='SIM_VALID')
                     for e in entries if any(t['status']=='SIM_VALID' for t in e['targets'])]
        distances.sort(); n = len(distances)
        median = None if not n else (distances[n//2] + distances[(n-1)//2])/2
        regions.append(dict(zone_ids=group, quality_mass=total, bearing_deg=theta,
            elevation_deg=phi, horizontal=horizontal, vertical=vertical,
            horizontal_mass={k:sum(e['quality_weight'] for e in entries if e['horizontal']==k) for k in ('LEFT','CENTER','RIGHT')},
            valid_zone_count=n, merged_zone_count=sum(any(t['status']=='SIM_MERGED' for t in e['targets']) for e in entries),
            valid_slant_median_m=median, valid_slant_span_m=[min(distances),max(distances)] if n else None,
            distance_state='VALID_RETURN_SUMMARY_NOT_OBJECT_DISTANCE' if n else 'UNKNOWN_MERGED_ONLY',
            suggested_message='FORWARD_POSITION_NOTICE' if horizontal=='CENTER' else 'SIDE_POSITION_NOTICE',
            collision_state='NOT_INFERRED_FROM_BEARING'))
    return dict(state='ANGULAR_SUPPORT' if regions else 'UNKNOWN_NO_RETURNS', regions=regions,
        zone_evidence=[evidence[k] for k in sorted(evidence)],
        coordinate_frame='CAMERA_RELATIVE_NOT_BODY_HEIGHT',
        policy='ADDITIVE_DIRECTION_ONLY_INCUMBENT_ALERT_UNCHANGED')
