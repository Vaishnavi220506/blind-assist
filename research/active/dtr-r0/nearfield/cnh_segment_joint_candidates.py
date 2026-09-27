"""Privileged segmentation for repaired-v2 Development; never a camera claim.

Only object identity/category and angular first-hit support enter this module.
Coverage is the fraction of a zone's solid angle, using angular_rays(16), not
its unweighted pixel count. DILATE1/ERODE1 mean ONE SUBRAY pixel (3x3 square),
not the one-zone dilation named DILATE1 in the older candidate-quality sweep.
FA1 reuses its calibration-only zone footprint pool and exact placement RNG;
occupied zones are rasterized fully, a deliberately coarse false-mask model.
IDs survive perturbation only for downstream evaluation, never range scoring.
"""
from functools import lru_cache
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from cnh_candidate_quality_readout import SEED, fake_footprints, to_pixels

CONDITIONS = ('IDEAL', 'DILATE1', 'ERODE1', 'SHIFT05', 'DROP20', 'FA1')


@lru_cache(maxsize=1)
def angular_geometry():
    from cnh_route_sensor import angular_rays
    from cnh_track_a_readout import EDGE
    rays, weights = angular_rays(16)
    xy = (rays[..., :2]/rays[..., 2:]+EDGE)/(2*EDGE)*8
    return xy, weights/weights.sum(-1, keepdims=True)


def mask_coverage(mask):
    """128x128 row/column raster -> solid-angle fractions, float32 [8,8]."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (128, 128):
        raise ValueError('Expected 128x128 subray mask')
    _, weights = angular_geometry()
    hits = mask.reshape(8, 16, 8, 16).transpose(0, 2, 1, 3).reshape(8, 8, 256)
    return (hits*weights).sum(-1).astype(np.float32)


def morph(mask, dilate):
    """One-subray Chebyshev dilation/erosion; outside the FOV is empty."""
    op = maximum_filter if dilate else minimum_filter
    return op(mask, size=3, mode='constant', cval=0).astype(bool)


def _candidate(ident, mask):
    return dict(id=int(ident), coverage=mask_coverage(mask), mask=mask)


def candidate_frames(oid, objects, unit, config, pool):
    """Return condition -> frames -> candidates, independent of query/labels.

    oid: [T,8,8,256] first-hit object IDs; objects need only id/category.
    unit/config only seed perturbations using the prior quality experiment's
    SeedSequence. pool is its calib-only list of relative occupied-zone rc pairs.
    All visible non-BACKGROUND objects enter, whether near, far, positive or not.
    Empty eroded/shifted masks disappear; DROP20 drops whole frame candidates.
    """
    oid = np.asarray(oid)
    if oid.ndim != 4 or oid.shape[1:] != (8, 8, 256):
        raise ValueError('Expected oracle object_id [T,8,8,256]')
    if not 96 <= int(unit) <= 191 or int(unit) == 143:
        raise ValueError('Only repaired-v2 calib/audit units, excluding 143')
    ids = [int(o['id']) for o in objects if o['category'] != 'BACKGROUND']
    if len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError('Real candidate IDs must be distinct and nonnegative')
    pool = [np.asarray(p, dtype=int) for p in pool]
    if not pool or any(p.ndim != 2 or p.shape[1] != 2 or not len(p)
                       or np.any(p < 0) or np.any(p >= 8) for p in pool):
        raise ValueError('Need nonempty calibration zone-footprint pool')
    xy, _ = angular_geometry()
    out = {name: [] for name in CONDITIONS}
    for t, frame in enumerate(oid):
        current = {name: [] for name in CONDITIONS}
        for ident in ids:
            hit = frame == ident
            if not hit.any():
                continue
            rng = np.random.default_rng(np.random.SeedSequence([SEED, unit, config, t, ident]))
            angle = rng.uniform(0, 2*np.pi)
            direction, drop = np.array([np.cos(angle), np.sin(angle)]), rng.random()
            ideal = to_pixels(xy[hit], direction, 0.)
            variants = dict(IDEAL=ideal, DILATE1=morph(ideal, True),
                            ERODE1=morph(ideal, False),
                            SHIFT05=to_pixels(xy[hit], direction, .5),
                            DROP20=ideal if drop >= .2 else np.zeros_like(ideal), FA1=ideal)
            for name, mask in variants.items():
                if mask.any():
                    current[name].append(_candidate(ident, mask))
        rng = np.random.default_rng(np.random.SeedSequence([SEED, unit, config, t, 99999]))
        fake, _ = fake_footprints(pool, rng, 1)
        current['FA1'].append(_candidate(-1, np.repeat(np.repeat(fake[0], 16, 0), 16, 1)))
        for name in CONDITIONS:
            out[name].append(current[name])
    return out
