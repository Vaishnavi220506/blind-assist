"""One local RGB lateral-attribution recipe; no RGB-created metric range.

All conclusions are hypotheses about observed return ownership. Original sensor
supports remain immutable; only an ambiguous zone's alert vote can be suppressed.
"""
import cv2
import numpy as np

from ba_camera_corridor import WIDTH, HEIGHT, LOW_W, LOW_H, HFOV

FOCAL = WIDTH / (2 * np.tan(np.deg2rad(HFOV / 2)))
SIDE = 48
CLASSES = ('INSIDE', 'OUTSIDE', 'CROSSING')
OUTSIDE_CUTOFF = .95


def slopes(box):
    y0, x0, y1, x1 = box
    return ((x0 * WIDTH / LOW_W - WIDTH / 2) / FOCAL,
            (x1 * WIDTH / LOW_W - WIDTH / 2) / FOCAL)


def lateral_relation(aa, interval):
    xs = [a * z for a in aa for z in interval]
    low, high = min(xs), max(xs)
    if high < -.3 - 1e-9 or low > .3 + 1e-9:
        return 'OUTSIDE'
    if low > -.3 + 1e-9 and high < .3 - 1e-9:
        return 'INSIDE'
    return 'CROSSING'


def eligible(box, value, anchor):
    return bool(np.isfinite(value) and .1 <= value < 3 and anchor['possible']
                and not anchor['definite'] and lateral_relation(slopes(box), anchor['interval_m']) == 'CROSSING')


def crop_geometry(box):
    y0, x0, y1, x1 = box
    xn0, xn1 = x0 * WIDTH / LOW_W, x1 * WIDTH / LOW_W
    yn0, yn1 = y0 * HEIGHT / LOW_H, y1 * HEIGHT / LOW_H
    dx, dy = xn1 - xn0, yn1 - yn0
    return [max(0, int(np.floor(xn0 - dx))), max(0, int(np.floor(yn0 - dy))),
            min(int(WIDTH), int(np.ceil(xn1 + dx))), min(int(HEIGHT), int(np.ceil(yn1 + dy)))]


def local_input(rgb, box, value, interval):
    """3x3-zone context, RGB + public zone/return/corridor geometry, 8x48x48."""
    left, top, right, bottom = crop_geometry(box)
    patch = rgb[top:bottom, left:right]
    image = cv2.resize(patch, (SIDE, SIDE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    xx, yy = np.meshgrid(left + (np.arange(SIDE) + .5) * (right - left) / SIDE,
                         top + (np.arange(SIDE) + .5) * (bottom - top) / SIDE)
    a, b = (xx - WIDTH / 2) / FOCAL, (yy - HEIGHT / 2) / FOCAL
    y0, x0, y1, x1 = box
    mask = ((xx >= x0 * WIDTH / LOW_W) & (xx < x1 * WIDTH / LOW_W)
            & (yy >= y0 * HEIGHT / LOW_H) & (yy < y1 * HEIGHT / LOW_H))
    geometry = np.stack([mask, a * value / .3, a * interval[0] / .3,
                         a * interval[1] / .3, b * value / .9], axis=-1)
    return np.ascontiguousarray(np.concatenate([image, np.clip(geometry, -4, 4)], axis=-1).transpose(2, 0, 1), dtype=np.float32)


def simple_geometry(rgb, box, interval):
    """Fixed Otsu minority region, both polarities, full-zone fallback.

    Use all minority pixels within the zone, not a truth-selected component.
    Their horizontal bounding support includes an explicit one-native-pixel pad.
    Foreground minority and contrast are appearance assumptions, not ownership
    proof. Missing/low-contrast/multiple texture evidence is never free space.
    """
    left, top, right, bottom = crop_geometry(box)
    patch = rgb[top:bottom, left:right]
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray <= threshold
    if not dark.any() or dark.all():
        return dict(relation='UNKNOWN', suppress=False, reason='NO_TWO_INTENSITY_GROUPS')
    foreground = dark if dark.sum() <= (~dark).sum() else ~dark
    contrast = abs(float(gray[dark].mean()) - float(gray[~dark].mean()))
    if contrast < 12:
        return dict(relation='UNKNOWN', suppress=False, reason='LOW_CONTRAST', contrast=contrast)
    yy, xx = np.mgrid[top:bottom, left:right]
    y0, x0, y1, x1 = box
    in_zone = ((xx + .5 >= x0 * WIDTH / LOW_W) & (xx + .5 < x1 * WIDTH / LOW_W)
               & (yy + .5 >= y0 * HEIGHT / LOW_H) & (yy + .5 < y1 * HEIGHT / LOW_H))
    selected = foreground & in_zone
    if selected.sum() < 2:
        return dict(relation='UNKNOWN', suppress=False, reason='NO_VISIBLE_REGION_IN_ZONE', contrast=contrast)
    # Pixel footprints plus one pixel of fixed image discretization padding.
    amin = max(slopes(box)[0], (float(xx[selected].min()) - 1 - WIDTH / 2) / FOCAL)
    amax = min(slopes(box)[1], (float(xx[selected].max()) + 2 - WIDTH / 2) / FOCAL)
    relation = lateral_relation((amin, amax), interval)
    return dict(relation=relation, suppress=relation == 'OUTSIDE', reason='OTSU_MINORITY_REGION_HYPOTHESIS',
                contrast=contrast, threshold=float(threshold), selected_pixels=int(selected.sum()),
                proposed_x_slopes=[amin, amax])


def tiny_head():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(6),
                         nn.Flatten(), nn.Linear(256, 16), nn.ReLU(), nn.Linear(16, 3))


def learned_decision(probabilities):
    p = np.asarray(probabilities)
    index = int(p.argmax())
    return dict(relation=CLASSES[index], outside_score=float(p[1]),
                suppress=bool(p[1] >= OUTSIDE_CUTOFF),
                meaning='Softmax classifier score, not calibrated physical probability')


def frame_decision(baseline, anchors, zone_scores, threshold, suppressed):
    lookup = {row['zone']: row['joint'] for row in zone_scores}
    votes = [a for a in anchors if a['definite'] or (a['possible'] and lookup[a['zone']] >= threshold)]
    require_keep = [a for a in votes if a['definite'] or a['zone'] not in suppressed]
    alert = bool(baseline['alert'] and require_keep)
    supported = baseline['definite_zones'] > 0
    return {**baseline, 'alert': alert, 'ambiguous': bool(alert and not supported),
            'unknown': baseline['unknown'], 'state': baseline['state'] if alert else 'UNKNOWN_WITHHELD' if baseline['alert'] else baseline['state'],
            'suppressed_zone_votes': len(suppressed), 'geometry_supports_unchanged': True}
