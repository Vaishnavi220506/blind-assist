"""One frozen public-prior comparator: both path actions chosen before moving."""
from collections import defaultdict

import two_step_observation as two


def choose(initial, forecasts, labels):
    """Only current bins and declared prior; never actual midpoint or source ID.

    Cost is the same mixed-positive, mixed-total objective as two-step feedback,
    but the second action must be shared by every predicted midpoint outcome.
    Ties follow the existing +X/-X/+Z/-Z action ordering at each step.
    """
    candidates = [i for i, bins in enumerate(forecasts[(0., 0.)]) if bins == tuple(initial)]
    costs = []
    for first, mid in enumerate(two.STEPS):
        for second, step in enumerate(two.STEPS):
            final = two.add(mid, step)
            groups = defaultdict(list)
            for i in candidates:
                groups[(forecasts[mid][i], forecasts[final][i])].append(i)
            positive = total = 0
            for group in groups.values():
                p = sum(labels[i] for i in group)
                if 0 < p < len(group):
                    positive += p
                    total += len(group)
            costs.append((positive, total, first, second))
    best = min(costs)
    return dict(first_action=best[2], second_action=best[3], costs=costs, candidates=candidates)
