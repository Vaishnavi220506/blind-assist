"""Public-input HGB representations; interval-conditioned RGB is not ownership.

Both arms retain the same raw regional descriptors and query coordinates. The
local arm fills otherwise-zero slots with RGB statistics grouped by a query's
relation to each complete regional range interval. This adds a representation,
not a depth observation, object mask, or independently measured surface.
"""
from __future__ import annotations

import numpy as np

SEED = 202609224
PARAMS = dict(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
              min_samples_leaf=20, l2_regularization=1., max_bins=64,
              early_stopping=False, random_state=SEED)
QUERIES = np.array([[x-.3, x+.3, lo, hi]
                    for lo, hi in ((.42, .9), (-.2, .42))
                    for x in (-.3, 0., .3)], dtype=np.float64)
CENTRE = [1, 4]
RGB_HW = (180, 320)
FOCAL = 320 / (2 * np.tan(np.deg2rad(50)))
LOCAL_SIZE = 51
RAW_SIZE = 64 * 14 + 10 + 4
FEATURE_SIZE = RAW_SIZE + LOCAL_SIZE
ARMS = ('raw', 'local')


def canonical_tof(tof):
    t = np.array(tof, dtype=np.float64, copy=True)
    if t.shape != (64, 6) or not np.isin(t[:, 1], [0, 1]).all():
        raise ValueError('Need 64 public range/valid/box rows')
    b = t[:, 2:]
    if not (np.isfinite(b).all() and (b >= 0).all() and (b <= 1).all()
            and (b[:, 2:] > b[:, :2]).all()):
        raise ValueError('Malformed public boxes')
    z = t[:, 0] * 8
    valid = (t[:, 1] == 1) & np.isfinite(z) & (z >= .1) & (z < 8)
    z = np.where(valid, z, 0.)
    radius = .1 + 3 * (.01 + .02 * z)
    low = np.where(valid, np.maximum(.1, z-radius), 0.)
    high = np.where(valid, z+radius, 0.)
    return z, valid, b, low, high


def query_bands(ax, ay, lower, upper, query):
    """Disjoint definite/possible-only/outside memberships of original intervals.

    These are hypotheses about regional supports. They never shrink the original
    interval or claim that a pixel generated the regional return.
    """
    xl, xh, yl, yh = map(float, query)
    if not (xl < xh and yl < yh):
        raise ValueError('Query bounds must be ordered')
    entry = np.full_like(ax, .3, dtype=np.float64)
    leave = np.full_like(ax, 3., dtype=np.float64)
    for slope, lo, hi in ((ax, xl, xh), (ay, yl, yh)):
        nonzero = slope != 0
        first = np.divide(lo, slope, out=np.zeros_like(slope), where=nonzero)
        last = np.divide(hi, slope, out=np.zeros_like(slope), where=nonzero)
        lower_slab = np.where(nonzero, np.minimum(first, last),
                              -np.inf if lo <= 0 <= hi else np.inf)
        upper_slab = np.where(nonzero, np.maximum(first, last),
                              np.inf if lo <= 0 <= hi else -np.inf)
        entry = np.maximum(entry, lower_slab)
        leave = np.minimum(leave, upper_slab)
    # Closed geometric boundaries; protect decimal equalities from binary64 ulps.
    possible = np.maximum(entry, lower) <= np.minimum(leave, upper) + 1e-12
    definite = possible & (lower >= entry-1e-12) & (upper <= leave+1e-12)
    return definite, possible & ~definite, ~possible


def visual_stats(rgb, gx, gy, gray):
    if len(rgb) == 0:
        return np.zeros(10, np.float64)
    return np.r_[rgb.mean(0), rgb.std(0), np.mean(gx), np.mean(gy),
                 np.quantile(gray, [.1, .9])]


def extract(rgb, tof, queries=QUERIES):
    """Return [query, arm, feature]; no identities/labels/native depth accepted."""
    im = np.asarray(rgb)
    if im.dtype != np.uint8 or im.shape != (3, *RGB_HW):
        raise ValueError('Need unchanged public uint8 RGB [3,180,320]')
    q = np.asarray(queries, np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError('Need finite query bounds')
    z, valid, boxes, low, high = canonical_tof(tof)
    pixels = im.transpose(1, 2, 0).astype(np.float64) / 255
    gray = pixels @ np.array([.299, .587, .114])
    gy, gx = (np.abs(v) for v in np.gradient(gray))
    pooled = []
    valid_y, valid_x, valid_zone = [], [], []
    coverage = np.zeros(RGB_HW, bool)
    for zone, box in enumerate(boxes):
        y0, x0, y1, x1 = box * [180, 320, 180, 320]
        ys = np.arange(max(0, int(np.ceil(y0-.5))), min(180, int(np.ceil(y1-.5))))
        xs = np.arange(max(0, int(np.ceil(x0-.5))), min(320, int(np.ceil(x1-.5))))
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        yy, xx = yy.ravel(), xx.ravel()
        if coverage[yy, xx].any():
            raise ValueError('Overlapping public zone footprints')
        coverage[yy, xx] = True
        stat = visual_stats(pixels[yy, xx], gx[yy, xx], gy[yy, xx], gray[yy, xx])
        pooled.extend(np.r_[float(valid[zone]), z[zone]/8, low[zone]/8, high[zone]/8, stat])
        if valid[zone]:
            valid_y.extend(yy)
            valid_x.extend(xx)
            valid_zone.extend([zone] * len(yy))
    whole = visual_stats(pixels.reshape(-1, 3), gx.ravel(), gy.ravel(), gray.ravel())
    yy, xx, zones = (np.asarray(v, np.int64) for v in (valid_y, valid_x, valid_zone))
    ax, ay = (xx+.5-160)/FOCAL, (yy+.5-90)/FOCAL
    output = np.zeros((len(q), 2, FEATURE_SIZE), np.float32)
    for qi, query in enumerate(q):
        common = np.r_[pooled, whole, query]
        output[qi, :, :RAW_SIZE] = common
        local, means = [], []
        for mask in query_bands(ax, ay, low[zones], high[zones], query):
            iy, ix, iz = yy[mask], xx[mask], zones[mask]
            stats = visual_stats(pixels[iy, ix], gx[iy, ix], gy[iy, ix], gray[iy, ix])
            if len(iz):
                geometry = [low[iz].mean()/8, high[iz].mean()/8,
                            z[iz].mean()/8, z[iz].std()/8, len(iz)/max(1, coverage.sum())]
            else:
                geometry = [0.] * 5
            local.extend(np.r_[stats, geometry])
            means.append(stats[:3])
        local.extend(means[0]-means[2])
        local.extend(means[1]-means[2])
        output[qi, 1, RAW_SIZE:] = local
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite representation')
    return output
