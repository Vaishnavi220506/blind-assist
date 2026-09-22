"""Fixed geometric score on unchanged ToF supports, not a probability model.

Uniform measure in (ray slope a, ray slope b, axial Z) is a ranking convention.
It is not a physical return distribution, hardware confidence or collision risk.
Only the scalar operating threshold is calibrated on disclosed Development data.
"""
import math

import numpy as np

from ba_camera_corridor import WIDTH, HEIGHT, LOW_W, LOW_H, HFOV, tof_readout


def _overlap_coeff(lo, hi, vmin, vmax, z):
    """Overlap width = A + B/z between fixed slopes and a volume slab."""
    upper = (hi, 0.) if hi <= vmax / z else (0., vmax)
    lower = (lo, 0.) if lo >= vmin / z else (0., vmin)
    a, b = upper[0] - lower[0], upper[1] - lower[1]
    return (a, b) if a + b / z > 0 else (0., 0.)


def support_score(aa, bb, interval):
    """Exact piecewise integral over the complete original support interval.

    At each Z, integrate the fraction of the full zone's ray-slope rectangle
    inside X=[-.3,.3], Y=[-.2,.9]. Normalize by full interval length, including
    out-of-corridor depth. No centre ray, interval shrinking or pixel sampling.
    """
    low, high = map(float, interval)
    if not (0 < low < high and aa[0] < aa[1] and bb[0] < bb[1]):
        raise ValueError('Invalid support')
    start, stop = max(low, .3), min(high, 3.)
    if start >= stop:
        return dict(joint=0., depth=0., angular_given_depth=None)
    cuts = {start, stop}
    for slopes, faces in ((aa, (-.3, .3)), (bb, (-.2, .9))):
        for slope in slopes:
            if slope:
                cuts.update(face / slope for face in faces if start < face / slope < stop)
    cuts = sorted(cuts)
    integral = 0.
    for left, right in zip(cuts, cuts[1:]):
        mid = (left + right) / 2
        ax, bx = _overlap_coeff(*aa, -.3, .3, mid)
        ay, by = _overlap_coeff(*bb, -.2, .9, mid)
        integral += (ax * ay * (right - left)
                     + (ax * by + ay * bx) * math.log(right / left)
                     + bx * by * (1 / left - 1 / right))
    area = (aa[1] - aa[0]) * (bb[1] - bb[0])
    depth = (stop - start) / (high - low)
    joint = float(np.clip(integral / (area * (high - low)), 0., depth))
    return dict(joint=joint, depth=depth, angular_given_depth=joint / depth)


def score_frame(boxes, values):
    baseline, anchors = tof_readout(boxes, values)
    focal = WIDTH / (2 * np.tan(np.deg2rad(HFOV / 2)))
    scores = []
    for anchor in anchors:
        y0, x0, y1, x1 = boxes[anchor['zone']]
        aa = ((x0 * WIDTH / LOW_W - WIDTH / 2) / focal,
              (x1 * WIDTH / LOW_W - WIDTH / 2) / focal)
        bb = ((y0 * HEIGHT / LOW_H - HEIGHT / 2) / focal,
              (y1 * HEIGHT / LOW_H - HEIGHT / 2) / focal)
        value = support_score(aa, bb, anchor['interval_m'])
        scores.append(dict(zone=anchor['zone'], **value))
    possible = {a['zone'] for a in anchors if a['possible']}
    score = max((s['joint'] for s in scores if s['zone'] in possible), default=0.)
    return dict(baseline=baseline, anchors=anchors, zone_scores=scores, score=score)


def decide(scored, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold outside [0,1]')
    base = scored['baseline']
    # Independently definite ToF support always alerts. Ambiguous supports and
    # every original observation remain available even when no alert is emitted.
    supported = base['definite_zones'] > 0
    alert = bool(supported or (base['alert'] and scored['score'] >= threshold))
    return dict(alert=alert, ambiguous=bool(alert and not supported), unknown=not supported,
                state='ALERT_SUPPORTED' if supported else 'ALERT_AMBIGUOUS' if alert
                else 'UNKNOWN_WITHHELD' if base['alert'] else 'NO_SUPPORTED_HIT',
                possible_zones=base['possible_zones'], definite_zones=base['definite_zones'],
                valid_zones=base['valid_zones'], geometry_ambiguous=base['ambiguous'],
                score=scored['score'], threshold=threshold,
                meaning='Calibrated Development decision; all supports retained; silence is not clear space')


def calibrate(scored, truth):
    """Largest inclusive threshold preserving every baseline positive alert.

    No candidate model/weight/threshold grid. Monotonicity makes the minimum
    required ambiguous score the maximum admissible threshold and minimizes FP
    among thresholds with exact baseline-TP retention. Ties cannot be split.
    Definite-support bypass is mandatory. This is in-sample selection, not testing.
    """
    if len(scored) != len(truth) or any(t is not True and t is not False for t in truth):
        raise ValueError('Need matched known Development labels')
    required = [s for s, t in zip(scored, truth) if t and s['baseline']['alert']]
    constrained = [s['score'] for s in required if not s['baseline']['definite_zones']]
    threshold = min(constrained, default=1.)
    return dict(threshold=threshold, required_tp=len(required), ambiguous_required_tp=len(constrained),
                binding_ids=[s['id'] for s in required
                             if not s['baseline']['definite_zones'] and s['score'] == threshold],
                rule='maximal inclusive scalar threshold preserving all baseline TP; definite bypass',
                scope='ALL96_CONSUMED_DEVELOPMENT_IN_SAMPLE_NO_HOLDOUT')
