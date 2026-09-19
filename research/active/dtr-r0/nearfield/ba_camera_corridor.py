"""Fixed observation readouts for the camera-forward corridor experiment.

Depth bins and zone returns remain intervals/supports, never point surfaces.
ALERT_AMBIGUOUS is a conservative alert and stays UNKNOWN spatial evidence.
NO_SUPPORTED_HIT is a model readout, never a certificate of free space.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/ba-camera-corridor-20260919'
WIDTH, HEIGHT, HFOV = 640, 360, 100.
LOW_W, LOW_H = 256, 192
PROFILE = dict(z_min=.30, z_max=3., x_min=-.30, x_max=.30, y_min=-.20, y_max=.90)
CUT = .081
MIN_PIXELS = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sample_indices():
    # Sensor and model depth use the identical native point-ray samples.
    x = np.floor((np.arange(LOW_W)+.5)*WIDTH/LOW_W).astype(int)
    y = np.floor((np.arange(LOW_H)+.5)*HEIGHT/LOW_H).astype(int)
    return y, x


def sample_native(depth):
    assert depth.shape == (HEIGHT, WIDTH)
    y, x = sample_indices()
    return depth[y[:, None], x[None, :]].copy()


def rays():
    y, x = sample_indices()
    f = WIDTH/(2*np.tan(np.deg2rad(HFOV/2)))
    a, b = np.meshgrid((x+.5-WIDTH/2)/f, (y+.5-HEIGHT/2)/f)
    return a, b


def ray_limit(a, b):
    # This frozen corridor includes the optical axis in both lateral axes.
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.where(a != 0, .3/np.abs(a), np.inf)
        y = np.where(b > 0, .9/b, np.where(b < 0, -.2/b, np.inf))
    return np.minimum(np.minimum(x, y), 3.)


def largest(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def decision(possible, definite, missing=False):
    p, d = largest(possible), largest(definite)
    alert = p >= MIN_PIXELS
    ambiguous = alert and d < MIN_PIXELS
    return dict(alert=alert, ambiguous=ambiguous, unknown=bool(missing or ambiguous),
                state='ALERT_AMBIGUOUS' if ambiguous else 'ALERT_SUPPORTED' if alert else 'NO_SUPPORTED_HIT',
                possible_pixels=int(possible.sum()), definite_pixels=int(definite.sum()),
                largest_possible=p, largest_definite=d,
                meaning='Prediction/support readout, not a free-space or confidence certificate')


def nfo_readout(probabilities):
    assert probabilities.shape == (4, LOW_H, LOW_W) and np.isfinite(probabilities).all()
    assert (np.diff(probabilities, axis=0) >= 0).all()
    near = probabilities >= CUT
    first = np.where(near.any(axis=0), near.argmax(axis=0), 4)
    bounds = np.array([0., 1., 1.5, 2., 3., np.inf])
    lower, upper = bounds[first], bounds[first+1]
    limit = ray_limit(*rays())
    # Positive-length overlap. A singleton at the exact3m threshold of the
    # open-ended far bin is unresolved contact, not an all-image alert.
    possible = np.maximum(lower, .3) < np.minimum(upper, limit)
    definite = (lower >= .3) & (upper <= limit)
    return decision(possible, definite), possible, definite


def depth_readout(depth):
    assert depth.shape == (LOW_H, LOW_W) and np.isfinite(depth).all() and (depth > 0).all()
    a, b = rays(); x, y = a*depth, b*depth
    inside = (depth >= .3) & (depth <= 3.) & (np.abs(x) <= .3) & (y >= -.2) & (y <= .9)
    # Same declared2cm geometric boundary stratum; this is not a calibrated
    # learned-depth confidence interval.
    margin = np.minimum.reduce([depth-.3, 3.-depth, .3-np.abs(x), y+.2, .9-y])
    return decision(inside, inside & (margin > .02)), inside, inside & (margin > .02)


def tof_readout(boxes, values):
    possible, definite, anchors = [], [], []
    for zone, (box, value) in enumerate(zip(boxes, values)):
        if not np.isfinite(value) or value <= 0:
            continue
        y0, x0, y1, x1 = box
        # Full zone footprint, not a centre ray or only sampled pixel centres.
        f = WIDTH/(2*np.tan(np.deg2rad(HFOV/2)))
        aa = ((x0*WIDTH/LOW_W-WIDTH/2)/f, (x1*WIDTH/LOW_W-WIDTH/2)/f)
        bb = ((y0*HEIGHT/LOW_H-HEIGHT/2)/f, (y1*HEIGHT/LOW_H-HEIGHT/2)/f)
        closest_a = 0. if aa[0] <= 0 <= aa[1] else min(aa, key=abs)
        closest_b = 0. if bb[0] <= 0 <= bb[1] else min(bb, key=abs)
        best = float(ray_limit(np.array(closest_a), np.array(closest_b)))
        worst = min(float(ray_limit(np.array(x), np.array(y))) for x in aa for y in bb)
        radius = .1+3*(.01+.02*float(value))
        lower, upper = max(.1, float(value)-radius), float(value)+radius
        p = max(lower, .3) <= min(upper, best)
        d = lower >= .3 and upper <= worst
        possible.append(bool(p)); definite.append(bool(d))
        anchors.append(dict(zone=zone, possible=bool(p), definite=bool(d), interval_m=[lower, upper]))
    alert, supported = any(possible), any(definite)
    ambiguous = alert and not supported
    result = dict(alert=alert, ambiguous=ambiguous, unknown=bool(not anchors or ambiguous or not alert),
                  state='ALERT_AMBIGUOUS' if ambiguous else 'ALERT_SUPPORTED' if alert else 'NO_SUPPORTED_HIT',
                  possible_zones=sum(possible), definite_zones=sum(definite), valid_zones=len(anchors),
                  meaning='Outside-only or missing returns do not establish free space')
    return result, anchors


def global_scale(depth, boxes, values):
    """Exact previously frozen global recipe, with shared region/peak pairing."""
    from ba_nfo_depthpro_echo import predict
    result, receipt = predict(depth, boxes, values)
    return result['global_depth'], receipt
