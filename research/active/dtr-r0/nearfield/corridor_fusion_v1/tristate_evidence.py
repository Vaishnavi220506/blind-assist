"""Privileged returned-support probe; never import into a public predictor."""

CORRIDOR = ((.2, 3.6), (-.3, .3), (.4, 2.05))


def intersects(piece):
    return all(hi >= start and lo <= end
               for (lo, hi), (start, end) in zip(piece, CORRIDOR))


def classify_returns(record, packet_received, representation):
    if representation not in ('sampled_points', 'full_faces'):
        raise ValueError(representation)
    usable = [s for s in record['slots'] if s['usable']]
    assert len(usable) == record['usable_slots']
    if not packet_received:
        assert not usable, 'Missing packet cannot supply a usable return'
        return 'UNKNOWN', 'PACKET_MISSING'
    if not usable:
        return 'UNKNOWN', 'PACKET_PRESENT_NO_USABLE_RETURN'
    completed = [s for s in usable if s['completed']]
    for slot in completed:
        pieces = (slot['faces'] if representation == 'full_faces' else
                  [[[v, v] for v in p] for p in slot['points']])
        if any(intersects(p) for p in pieces):
            return 'POSITIVE', 'RETURNED_SUPPORT_INTERSECTS'
    if len(completed) != len(usable):
        return 'UNKNOWN', 'UNRESOLVED_RETURN_SUPPORT'
    return 'OUTSIDE_ONLY', 'ALL_RESOLVED_RETURNED_SUPPORTS_OUTSIDE'


def decisions(a, state):
    if state not in ('POSITIVE', 'OUTSIDE_ONLY', 'UNKNOWN'):
        raise ValueError(state)
    p, n = state == 'POSITIVE', state == 'OUTSIDE_ONLY'
    return {'A': bool(a), 'A_OR_P': bool(a or p),
            'A_VETO_N': bool(a and not n),
            'A_POSITIVE_AND_VETO': bool((a or p) and not n)}
