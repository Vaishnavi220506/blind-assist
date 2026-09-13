"""Zone-level depth-gated connectivity; no within-zone surface decomposition.

The frozen equal-weight readout supplies observations and region summaries.
SIM_MERGED never supplies a known depth or a certificate of far-only space.
"""
import copy
import math

from tof_directional_readout import readout


def known_depth(entry):
    depths = [t['distance_m'] for t in entry['targets'] if t['status'] == 'SIM_VALID']
    return min(depths) if depths else None


def neighbors(a, b):
    ar, ac = divmod(a, 8)
    br, bc = divmod(b, 8)
    return abs(ar-br) + abs(ac-bc) == 1


def depth_edge(a, b, abs_gap_m, rel_gap):
    return (a is not None and b is not None and
            abs(a-b) < abs_gap_m + rel_gap * min(a, b))


def depth_readout(row, *, abs_gap_m, rel_gap=0.):
    if not all(math.isfinite(v) and v >= 0 for v in (abs_gap_m, rel_gap)):
        raise ValueError('Finite nonnegative depth gates required')
    base = readout(row, equal_weights=True)
    entries = {e['zone_id']: e for e in base['zone_evidence']}
    depths = {i: known_depth(e) for i, e in entries.items()}
    links = {i: set() for i in entries}
    accepted, rejected = [], []
    for i in sorted(entries):
        for j in sorted(entries):
            if j <= i or not neighbors(i, j):
                continue
            if depth_edge(depths[i], depths[j], abs_gap_m, rel_gap):
                links[i].add(j); links[j].add(i); accepted.append([i, j])
            else:
                rejected.append(dict(zones=[i, j], reason='UNKNOWN_DEPTH' if
                    depths[i] is None or depths[j] is None else 'DEPTH_DIFFERENCE'))
    remaining = set(entries)
    groups = []
    while remaining:
        first = min(remaining); remaining.remove(first)
        group, stack = [first], [first]
        while stack:
            for other in sorted(links[stack.pop()] & remaining):
                remaining.remove(other); group.append(other); stack.append(other)
        groups.append(sorted(group))
    regions = []
    for group in groups:
        # Reuse the exact incumbent summary on this component; nothing is fitted.
        masked = copy.deepcopy(row)
        for zone in masked['tof_zones']:
            if zone['zone_id'] not in group:
                zone['targets'] = []
                zone['target_count'] = 0
        summary = readout(masked, equal_weights=True)['regions']
        assert len(summary) == 1 and summary[0]['zone_ids'] == group
        regions.extend(summary)
    return dict(base, regions=regions, graph=dict(
        node_depths_m=depths, accepted_edges=accepted, rejected_edges=rejected,
        abs_gap_m=abs_gap_m, rel_gap=rel_gap,
        depth_rule='MINIMUM_VALID_SLOT_NOT_SURFACE_IDENTITY',
        unknown_rule='PRESERVE_ISOLATED_ANGULAR_SUPPORT_NO_DEPTH_EDGE'))


def near_view(result, near_m):
    """Common post-component view; preserve unresolved support and raw far context."""
    if not math.isfinite(near_m) or near_m <= 0:
        raise ValueError('Positive finite near range required')
    selected, deferred = [], []
    for region in result['regions']:
        span = region['valid_slant_span_m']
        if region['merged_zone_count'] or span is None or span[0] <= near_m:
            selected.append(region)
        else:
            deferred.append(region)
    state = result['state'] if selected or not result['regions'] else 'NO_NEAR_VALID_SUPPORT'
    return dict(result, state=state, regions=selected, deferred_far_regions=deferred,
                near_view_policy='VALID_NEAR_OR_UNRESOLVED_NOT_A_CLEARANCE_DECISION', near_m=near_m)
