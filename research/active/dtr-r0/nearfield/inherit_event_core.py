"""One fixed observable-support continuation, without evaluator information."""
from __future__ import annotations

from dataclasses import dataclass
import math

THRESHOLD = 0.4071309640537889
TTL_NS = 200_000_000
ARMS = ('A_current', 'A_hold', 'support_inherit')


@dataclass(frozen=True)
class Support:
    zone: int
    range_m: float
    low_m: float
    high_m: float
    possible: bool
    definite: bool
    joint: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in
                   (self.range_m, self.low_m, self.high_m, self.joint)):
            raise ValueError('Nonfinite support')
        if not (0 <= self.zone < 64 and 0 < self.low_m < self.high_m):
            raise ValueError('Invalid original support')

    @property
    def decisive(self):
        return self.definite or (self.possible and self.joint >= THRESHOLD)


class SupportInheritance:
    """An inherited alert is never itself a seed; identity is not inferred."""

    def __init__(self):
        self.previous = None

    def step(self, frame_id, clip_id, index, captured_at_ns, current, supports):
        if not isinstance(index, int) or index < 0 or captured_at_ns < 0:
            raise ValueError('Invalid capture identity')
        if len({s.zone for s in supports}) != len(supports):
            raise ValueError('Duplicate observed zone')
        decisive = [s for s in supports if s.decisive]
        if bool(current) != bool(decisive):
            raise ValueError('Birth decision and decisive support disagree')
        previous = self.previous
        same_clip = previous is not None and previous['clip_id'] == clip_id
        adjacent = bool(same_clip and index == previous['index'] + 1)
        age = captured_at_ns - previous['captured_at_ns'] if same_clip else None
        if same_clip and (index <= previous['index'] or age <= 0):
            raise ValueError('Nonmonotonic capture within clip')
        old_hold = bool(current or (adjacent and previous['current']))
        eligible = bool(adjacent and previous['current'] and 0 < age <= TTL_NS)
        checks, matching = [], []
        if eligible:
            by_zone = {s.zone: s for s in supports}
            for seed in previous['decisive']:
                now = by_zone.get(seed.zone)
                if now is None:
                    reason = 'CURRENT_RETURN_MISSING'
                elif not now.possible:
                    reason = 'CURRENT_SUPPORT_OUTSIDE_QUERY'
                elif not .3 <= now.range_m <= 3.:
                    reason = 'CURRENT_RANGE_CENTRE_OUTSIDE_DEPTH_SLAB'
                elif max(seed.low_m, now.low_m) >= min(seed.high_m, now.high_m):
                    reason = 'ORIGINAL_INTERVAL_DISJOINT'
                else:
                    reason = 'OBSERVABLE_SUPPORT_COMPATIBLE'
                    matching.append(seed.zone)
                checks.append(dict(zone=seed.zone, reason=reason))
        inherited = bool(not current and eligible and matching)
        if current:
            reason = 'CURRENT_TRIGGER'
        elif inherited:
            reason = 'ORIGINAL_SUPPORT_INHERITED_ONCE'
        elif not same_clip:
            reason = 'NO_CLIP_SEED'
        elif not adjacent:
            reason = 'NONADJACENT_SAMPLE'
        elif not previous['current']:
            reason = 'PREVIOUS_WAS_NOT_CURRENT_TRIGGER'
        elif not 0 < age <= TTL_NS:
            reason = 'ORIGINAL_CAPTURE_EXPIRED'
        else:
            reason = 'NO_COMPATIBLE_CURRENT_SUPPORT'
        result = dict(flags=dict(A_current=bool(current), A_hold=old_hold,
                                 support_inherit=bool(current or inherited)),
                      seed_id=previous['id'] if eligible else None,
                      seed_age_ns=age if eligible else None,
                      decisive_zones=[s.zone for s in decisive],
                      inherited=inherited, matched_zones=matching,
                      support_checks=checks, reason=reason)
        self.previous = dict(id=frame_id, clip_id=clip_id, index=index,
                             captured_at_ns=captured_at_ns, current=bool(current),
                             decisive=decisive)
        return result
