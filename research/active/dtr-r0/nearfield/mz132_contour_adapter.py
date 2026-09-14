"""Anonymous 2D visible candidates with fixed MZ129 association parameters.

No evaluator, actor identity, truth range or semantic category is accepted here.
"""
import copy
import math

import cv2
import numpy as np

import mz115_spatial_allocation as tof
import mz111_spatial_evidence as radar
from mz116_radar_resolution_guard import protect
from mz128_zone_weighting import column_weights, FOUR, THRESHOLD
from mz130_local_masks import run_tiles
from mz130_local_support import fixed_plane_boxes


def anonymous_components(label_image):
    """Integer raster IDs are discarded; all visible nonzero instances survive.

    Label zero is no rendered surface (sky). Sort only by image geometry. Keep
    disconnected visible pieces of the same rendered instance as one mask.
    """
    assert label_image.ndim == 2 and np.issubdtype(label_image.dtype, np.integer)
    result = []
    for value in np.unique(label_image):
        if value == 0:
            continue
        binary = (label_image == value).astype(np.uint8)
        y, x = np.nonzero(binary)
        box = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]
        padded = cv2.dilate(binary, np.ones((5, 5), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
        result.append(dict(box=box, tiles=run_tiles(binary),
                           padded_tiles=run_tiles(padded), pixels=int(binary.sum())))
    result.sort(key=lambda c: (-tof.area(c['box']), c['box'], c['tiles']))
    return result


def outside_image(box, width, height):
    """Disjoint rectangle partition of a native zone outside RGB coverage."""
    inside = tof.intersection(box, [0., 0., float(width), float(height)])
    if inside is None:
        return [list(box)]
    x0, y0, x1, y1 = box
    a, b, c, d = inside
    candidates = [[x0,y0,a,y1], [c,y0,x1,y1], [a,y0,c,b], [a,d,c,y1]]
    return [r for r in candidates if r[2] > r[0] and r[3] > r[1]]


def clipped(tiles, box):
    return [r for tile in tiles if (r := tof.intersection(tile, box)) is not None]


def mask_assign(evidence, components):
    """MZ115 positive complete-link matcher with mask-area shape fractions."""
    valid = sorted((e for e in evidence if e['status'] == 'SIM_VALID'),
                   key=lambda e: (e['forward_depth'], e['zone_id'], e['target_slot']))
    groups = []
    records = []
    for item in valid:
        if not groups or item['forward_depth']-groups[-1][0]['forward_depth'] > tof.DEPTH_SPAN_M:
            groups.append([])
        groups[-1].append(item)
    for group, items in enumerate(groups):
        record = dict(group=group, accepted=False, candidates=[], reason='INSUFFICIENT_OR_COMPETING_ZONES')
        records.append(record)
        for item in items:
            item['group'] = group
        if len({e['zone_id'] for e in items}) != len(items) or len(items) < tof.MIN_ZONES:
            continue
        weights = np.array([e['signal']*e['range_m']**2 for e in items], float)
        candidates = []
        if weights.sum() <= 0:
            record['reason'] = 'NO_POSITIVE_SIGNAL'
            continue
        for j, component in enumerate(components):
            shape = np.array([sum(tof.area(t) for t in clipped(component['tiles'], e['zone_box']))/
                              tof.area(e['zone_box']) for e in items])
            coverage = float(weights[shape > 0].sum()/weights.sum())
            norm = float(np.linalg.norm(shape)*np.linalg.norm(weights))
            cosine = float(np.dot(shape, weights)/norm) if norm else 0.
            if coverage >= tof.MIN_COVERAGE:
                candidates.append((cosine, j, coverage))
        candidates.sort(key=lambda t: (-t[0], t[1]))
        winner = candidates[0] if candidates else None
        margin = winner[0]-(candidates[1][0] if len(candidates)>1 else 0.) if winner else 0.
        record['candidates'] = [dict(cosine=s, proposal=j, coverage=c) for s,j,c in candidates]
        record['reason'] = 'UNRESOLVED_SIGNAL_MATCH'
        if winner and winner[0] >= tof.MIN_COSINE and margin >= tof.MIN_MARGIN:
            record.update(accepted=True, proposal=winner[1], margin=margin, reason='MASK_SIGNAL_ASSOCIATION_PROXY')
            for item in items:
                tiles = clipped(components[winner[1]]['padded_tiles'], item['zone_box'])
                if tiles:
                    item.update(proposal=winner[1], tiles=tiles)
    return records


def tof_frame(row, old, components, use_masks):
    boxes = [c['box'] for c in components]
    current = tof.allocate(row, dict(old, proposals=[] if use_masks else boxes))
    groups = mask_assign(current, components) if use_masks else []
    assert len(current) == len(old['spatial_evidence'])
    records = []
    intr = row['rgb_intrinsics']
    yaw = old['integrated_yaw_deg']; pitch = row['camera_pitch_deg']; dy = .5+.2*row['time_s']
    weights = column_weights(row, FOUR)
    for new, previous in zip(current, old['spatial_evidence']):
        assert (new['zone_id'],new['target_slot']) == (previous['zone_id'],previous['target_slot'])
        assert list(new['range_bounds']) == previous['range_bounds']
        matched = new['proposal'] is not None
        tiles = new.get('tiles', [new['roi']]) if matched else [previous['roi']]
        external = outside_image(new['zone_box'], intr['width'], intr['height']) if matched else []
        tiles = tiles+external
        xyz = [tof.slant_envelope(t, new['range_bounds'], intr, (pitch-.5,pitch+.5),
                                 (yaw-dy,yaw+dy), row['camera_in_body_m'][2]) for t in tiles]
        possible = any(tof.possible(x) for x in xyz)
        records.append(dict(zone_id=new['zone_id'], slot=new['target_slot'], status=new['status'],
            range_bounds=new['range_bounds'], old_proposal=previous['proposal'], proposal=new['proposal'],
            association_namespace='anonymous_current' if matched else 'inherited_mz125_fallback',
            new_association=matched and previous['proposal'] is None,
            matched=matched, tiles=tiles, external_tiles=external,
            old_roi=previous['roi'], old_possible=tof.possible(previous['localized_xyz']),
            possible=possible, weight=weights[new['zone_id']], group=new['group'],
            association_candidates=new.get('association_candidates', [])))
        if new['status'] == 'SIM_MERGED':
            assert not matched and tiles == [previous['roi']]
    active = {r['zone_id'] for r in records if r['possible']}
    return dict(returns=records, groups=groups, score=sum(weights[z] for z in active),
        certain_coarse=any(tof.certain(e['coarse_xyz']) for e in current))


def radar_frames(rows, old_cache, old_radar, components, use_masks):
    """Frozen box association/filter, optional same-anchor mask plane extent."""
    nominal = [dict(proposals=[c['box'] for c in cs], integrated_yaw_deg=old['integrated_yaw_deg'],
                    tof_support=False, candidate=False, baseline=False) for old,cs in zip(old_cache,components)]
    current = radar.predict([tof.legacy_empty_tof(r) for r in rows], nominal,
                            surface='plane', filter_range=True)
    output = []
    for row, old, cs, cur, nom in zip(rows, old_radar, components, current, nominal):
        old_returns = {r['slot']:r for r in old['corrected_current_evidence']}
        records = []
        for ret in cur['spatial_evidence']:
            j = ret['proposal']; rec = copy.deepcopy(ret)
            before = old_returns[ret['slot']]
            rec['old_support'] = bool(before['support'])
            rec['new_association'] = j is not None and before['proposal'] is None
            rec['mask_applied'] = False
            rec['angular_tiles'] = [ret['box']] if j is not None else ([before['box']] if before['proposal'] is not None else [])
            if j is None:
                rec['support'] = bool(ret['support'] or before['support'])
                rec['fallback'] = 'INHERITED_MZ129_RETURN_SUPPORT'
            elif use_masks:
                planes = fixed_plane_boxes(cs[j]['tiles'], ret['box'], ret['range_m'],
                                           row, nom['integrated_yaw_deg'])
                if planes is not None:
                    rec['support'] = any(tof.possible(x) for x in planes)
                    rec['mask_applied'] = True
                    rec['angular_tiles'] = cs[j]['tiles']
            records.append(rec)
        new_guards = protect(row, {'candidate':False}, cur, nom['integrated_yaw_deg'])['guard_events']
        guards = copy.deepcopy(old['guard_events']) + [dict(g, proposal_namespace='anonymous_current') for g in new_guards]
        common = bool(any(r['support'] for r in records) or old['raw_center_support'] or old['inherited_carry_support'])
        output.append(dict(returns=records, common_radar=common, guard_events=guards,
            raw_center_support=old['raw_center_support'], inherited_carry_support=old['inherited_carry_support'],
            candidate=bool(common or guards)))
    return output


def predict(rows, cache, old_radar, component_frames, use_masks):
    assert len(rows) == len(cache) == len(old_radar) == len(component_frames)
    radars = radar_frames(rows, cache, old_radar, component_frames, use_masks)
    predictions, details = [], []
    for row, old, cs, rb in zip(rows, cache, component_frames, radars):
        local = tof_frame(row, old, cs, use_masks)
        flag = bool(local['score'] >= THRESHOLD or local['certain_coarse'] or rb['candidate'])
        predictions.append(dict(id=row['id'], candidate=flag, candidate_state='ALERT' if flag else 'UNKNOWN',
            score=local['score'], certain_coarse=local['certain_coarse'], radar=rb['candidate']))
        details.append(dict(id=row['id'], tof=local, radar=rb))
    return predictions, details
