"""Observable-only fixed spatial graph; normalized-score sums are not probabilities."""
import math


def adjacent(i, j):
    return abs(i // 8 - j // 8) + abs(i % 8 - j % 8) == 1


def depth_overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def axis_limit(slope, lower, upper):
    return upper / slope if slope > 0 else lower / slope if slope < 0 else math.inf


def shared_corridor_depth(i, j, boxes, intervals):
    """Shared face has positive 1D width inside corridor over this depth span.

    All corridor slabs include zero; the closest variable slope gives the first
    intersection. Positive z span excludes point contact at the limit.
    """
    if not adjacent(i, j) or not depth_overlap(intervals[i], intervals[j]):
        return None
    y0, x0, y1, x1 = boxes[i]
    v0, u0, v1, u1 = boxes[j]
    focal = 640 / (2 * math.tan(math.radians(50)))
    aa = lambda x: (x * 640 / 256 - 320) / focal
    bb = lambda y: (y * 360 / 192 - 180) / focal
    closest = lambda lo, hi: 0. if lo <= 0 <= hi else min((lo, hi), key=abs)
    if (x1 == u0 or u1 == x0) and max(y0, v0) < min(y1, v1):
        fixed = aa(x1 if x1 == u0 else x0)
        variable = closest(bb(max(y0, v0)), bb(min(y1, v1)))
        limit = min(axis_limit(fixed, -.3, .3), axis_limit(variable, -.2, .9))
    elif (y1 == v0 or v1 == y0) and max(x0, u0) < min(x1, u1):
        fixed = bb(y1 if y1 == v0 else y0)
        variable = closest(aa(max(x0, u0)), aa(min(x1, u1)))
        limit = min(axis_limit(variable, -.3, .3), axis_limit(fixed, -.2, .9))
    else:
        return None
    low = max(.3, intervals[i][0], intervals[j][0])
    high = min(3., intervals[i][1], intervals[j][1], limit)
    return [low, high] if low < high else None


def features(boxes, anchors, zone_scores):
    assert len(boxes) == 64
    intervals = {a['zone']: a['interval_m'] for a in anchors}
    possible = {a['zone'] for a in anchors if a['possible']}
    scores = {s['zone']: s['joint'] for s in zone_scores if s['zone'] in possible and s['joint'] > 0}
    assert all(math.isfinite(s) and 0 < s <= 1 for s in scores.values())
    ranked = sorted(scores, key=lambda z: (-scores[z], z))
    graph = {z: set() for z in scores}
    edges = []
    for i in sorted(scores):
        for j in sorted(scores):
            if j <= i:
                continue
            span = shared_corridor_depth(i, j, boxes, intervals)
            if span is not None:
                edges.append(dict(zones=[i, j], shared_corridor_depth=span))
                graph[i].add(j); graph[j].add(i)
    components, remaining = [], set(scores)
    while remaining:
        seed = min(remaining); todo = [seed]; members = set()
        while todo:
            z = todo.pop()
            if z in members:
                continue
            members.add(z); todo.extend(graph[z] - members)
        remaining -= members
        components.append(dict(zones=sorted(members), mass=math.fsum(scores[z] for z in sorted(members))))
    components.sort(key=lambda c: (-c['mass'], c['zones']))
    top = ranked[:2]
    pair = dict(zones=top, scores=[scores[z] for z in top],
                sum=math.fsum(scores[z] for z in top), adjacent=False,
                depth_consistent=False, corridor_continuous=False, same_component=False)
    if len(top) == 2:
        i, j = top
        pair.update(adjacent=adjacent(i, j), depth_consistent=depth_overlap(intervals[i], intervals[j]),
                    corridor_continuous=shared_corridor_depth(i, j, boxes, intervals) is not None,
                    same_component=any(i in c['zones'] and j in c['zones'] for c in components))
    return dict(top2=pair, s1=scores[top[0]] if top else 0., s2=scores[top[1]] if len(top) == 2 else 0.,
                S_pair=pair['sum'], S_component=components[0]['mass'] if components else 0.,
                S_all=math.fsum(scores.values()), components=components, edges=edges,
                positive_zone_count=len(scores))
