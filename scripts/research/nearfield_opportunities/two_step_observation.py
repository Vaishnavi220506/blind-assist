"""Public finite-prior two-step lookahead and separate evaluator ceilings."""
from collections import defaultdict

import active_view as av

STEPS = ((.06, 0.), (-.06, 0.), (0., .06), (0., -.06))


def add(a, b):
    return tuple(round(x+y, 8) for x,y in zip(a,b,strict=True))


def tree_poses():
    return tuple(sorted({av.ORIGIN, *STEPS, *(add(a,b) for a in STEPS for b in STEPS)}))


def public_forecasts(hypotheses):
    return {pose: tuple(av.observe(s, pose) for s in hypotheses) for pose in tree_poses()}


def mixed_cost(candidates, forecast, labels):
    groups = defaultdict(list)
    for i in candidates:
        groups[forecast[i]].append(i)
    positive = total = 0
    for group in groups.values():
        p = sum(labels[i] for i in group)
        if 0 < p < len(group):
            positive += p
            total += len(group)
    return positive, total


def second_choice(candidates, pose, forecasts, labels):
    costs = [(*mixed_cost(candidates, forecasts[add(pose, step)], labels), i) for i,step in enumerate(STEPS)]
    return dict(action=min(costs)[2], costs=costs)


def first_choice(initial, forecasts, labels):
    candidates = tuple(i for i,v in enumerate(forecasts[av.ORIGIN]) if v == tuple(initial))
    costs = []
    for action,pose in enumerate(STEPS):
        groups = defaultdict(list)
        for i in candidates:
            groups[forecasts[pose][i]].append(i)
        pos = total = 0
        for group in groups.values():
            choice = second_choice(group, pose, forecasts, labels)
            cost = choice["costs"][choice["action"]]
            pos += cost[0]
            total += cost[1]
        costs.append((pos,total,action))
    return dict(action=min(costs)[2], costs=costs, candidates=list(candidates))


def update(candidates, bins, pose, forecasts):
    return tuple(i for i in candidates if forecasts[pose][i] == tuple(bins))


def decision(candidates, labels):
    values = {labels[i] for i in candidates}
    return "INTERSECTS" if values == {True} else "NONINTERSECTING_HYPOTHESES" if values == {False} else "UNKNOWN"


def pure_count(ids, poses, observations, labels):
    buckets = defaultdict(list)
    for ident in ids:
        buckets[tuple(tuple(observations[ident][p]) for p in poses)].append(ident)
    return sum(len(group) for group in buckets.values() if len({labels[i] for i in group}) == 1)


def evaluator_ceiling(observations, labels):
    """Label-optimized cohort ceilings; never used by first/second_choice."""
    groups = defaultdict(list)
    for ident,views in observations.items():
        groups[tuple(views[av.ORIGIN])].append(ident)
    classes = []
    for initial,ids in sorted(groups.items()):
        adaptive, nonadaptive = [], []
        for a,mid in enumerate(STEPS):
            mid_groups = defaultdict(list)
            for ident in ids:
                mid_groups[tuple(observations[ident][mid])].append(ident)
            children = []
            for signature,members in sorted(mid_groups.items()):
                counts = [pure_count(members, [add(mid,b)], observations, labels) for b in STEPS]
                chosen = max(range(4), key=lambda i:(counts[i],-i))
                children.append(dict(mid_bins=signature, ids=members, counts=counts, second_action=chosen))
            adaptive.append(dict(first_action=a, resolved=sum(c["counts"][c["second_action"]] for c in children), children=children))
            nonadaptive.extend(dict(first_action=a, second_action=b,
                resolved=pure_count(ids,[mid,add(mid,step)],observations,labels)) for b,step in enumerate(STEPS))
        best_a = max(adaptive,key=lambda r:(r["resolved"],-r["first_action"]))
        best_n = max(nonadaptive,key=lambda r:(r["resolved"],-r["first_action"],-r["second_action"]))
        classes.append(dict(initial_bins=initial, ids=ids, adaptive=best_a, nonadaptive=best_n,
                            fixed_resolved=pure_count(ids,[STEPS[0],(.12,0.)],observations,labels)))
    return dict(cases=len(labels), classes=classes,
        adaptive_resolved=sum(c["adaptive"]["resolved"] for c in classes),
        nonadaptive_resolved=sum(c["nonadaptive"]["resolved"] for c in classes),
        fixed_resolved=sum(c["fixed_resolved"] for c in classes))
