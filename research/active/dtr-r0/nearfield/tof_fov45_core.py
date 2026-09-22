"""One nominal 45x45-degree layout on the unchanged camera-corridor lattice.

This is the same uncalibrated single-return sensor proxy with a different fixed
zone footprint. It is not a physical VL53 model. No target labels enter this API.
"""
import hashlib

import numpy as np

NATIVE_W, NATIVE_H = 640, 360
LOW_W, LOW_H = 256, 192
CAMERA_HFOV_DEG = 100.
NOMINAL_FOV_DEG = 45.


def _projected_boundaries():
    focal = NATIVE_W / (2 * np.tan(np.deg2rad(CAMERA_HFOV_DEG / 2)))
    limit = np.tan(np.deg2rad(NOMINAL_FOV_DEG / 2))
    slopes = np.linspace(-limit, limit, 9)
    x = (NATIVE_W / 2 + focal * slopes) * LOW_W / NATIVE_W
    y = (NATIVE_H / 2 + focal * slopes) * LOW_H / NATIVE_H
    return slopes, x, y


def boxes45():
    """64 row-major y0,x0,y1,x1 half-open boxes, nearest lattice boundaries.

    Rounding is floor(value+0.5). The intervals are uniform on the image plane,
    not uniform in angle. The 192x256 sampled depth array is never resized.
    """
    _, ideal_x, ideal_y = _projected_boundaries()
    xs = np.floor(ideal_x + .5).astype(np.int64)
    ys = np.floor(ideal_y + .5).astype(np.int64)
    return np.array([[ys[y], xs[x], ys[y + 1], xs[x + 1]]
                     for y in range(8) for x in range(8)], dtype=np.int64)


def geometry_description():
    """JSON-ready nominal, quantized-edge, and actual sampled-ray geometry."""
    slopes, ideal_x, ideal_y = _projected_boundaries()
    boxes = boxes45()
    xs, ys = np.unique(boxes[:, [1, 3]]), np.unique(boxes[:, [0, 2]])
    focal = NATIVE_W / (2 * np.tan(np.deg2rad(CAMERA_HFOV_DEG / 2)))
    edge_angles, sample_angles = {}, {}
    for axis, boundaries, native_size, low_size in (
            ('x', xs, NATIVE_W, LOW_W), ('y', ys, NATIVE_H, LOW_H)):
        edges = boundaries * native_size / low_size
        angles = np.rad2deg(np.arctan((edges - native_size / 2) / focal))
        # Exactly ba_camera_corridor.sample_indices(), then native pixel centers.
        first = np.floor((boundaries[0] + .5) * native_size / low_size) + .5
        last = np.floor((boundaries[-1] - .5) * native_size / low_size) + .5
        centers = np.rad2deg(np.arctan((np.array([first, last]) - native_size / 2) / focal))
        edge_angles[axis] = dict(boundaries_deg=angles.tolist(), total_fov_deg=float(angles[-1] - angles[0]))
        sample_angles[axis] = dict(first_last_deg=centers.tolist(), span_deg=float(centers[-1] - centers[0]))
    return dict(nominal_fov_deg=dict(horizontal=45., vertical=45.), grid=[8, 8],
        source=dict(native_shape=[360, 640], native_hfov_deg=100., sampled_shape=[192, 256],
                    sample_rule='native_index=floor((low_index+0.5)*native_size/low_size); use native pixel center'),
        interval_rule='9 equally spaced image-plane ray slopes from -tan(22.5deg) to +tan(22.5deg)',
        quantization='project to native image edges, scale to lowres boundaries, floor(boundary+0.5)',
        ideal_ray_slopes=slopes.tolist(), ideal_lowres_boundaries=dict(x=ideal_x.tolist(), y=ideal_y.tolist()),
        lowres_boundaries=dict(x=xs.tolist(), y=ys.tolist()), boxes=boxes.tolist(),
        quantized_edge_angles=edge_angles, actual_sample_center_angles=sample_angles,
        footprint_sampled_pixels=int((xs[-1] - xs[0]) * (ys[-1] - ys[0])),
        unchanged='Same native capture and point-sampled depth lattice; no interpolation, resize, or new rays.',
        rng_semantics='Same SHA256 identity seed and conditional draw algorithm. Layout-dependent eligible-hit counts and normal draws can change later RNG correspondence.',
        limitation='One nominal simulated layout; not calibrated hardware FoV or physical weak-return evidence.')


def simulate(depth, identity, boxes):
    """Return (64 float32 distances, 64 observed winning-bin lineage records).

    Exact original sensor selection: >=4 eligible samples in [0.1,8)m, conditional
    5% dropout, dominant inverse-square-energy 10cm bin, mean plus Gaussian noise
    sigma=0.01+0.02*mean, and output-range check [0.1,8)m. No hidden dropped bins
    become observations. Caller retains these traces outside model observations.
    """
    depth, boxes = np.asarray(depth), np.asarray(boxes)
    if depth.ndim != 2 or depth.dtype.kind != 'f':
        raise ValueError('Depth must be a floating two-dimensional sampled array')
    if boxes.shape != (64, 4) or boxes.dtype.kind not in 'iu':
        raise ValueError('Expected 64 integer y0,x0,y1,x1 boxes')
    h, w = depth.shape
    coverage = np.zeros(depth.shape, bool)
    for y0, x0, y1, x1 in boxes:
        if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            raise ValueError('Box outside sampled lattice or empty')
        if coverage[y0:y1, x0:x1].any():
            raise ValueError('Overlapping zones')
        coverage[y0:y1, x0:x1] = True
    rng = np.random.default_rng(int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16))
    values, traces = [], []
    for zone_id, (y0, x0, y1, x1) in enumerate(boxes):
        patch = depth[y0:y1, x0:x1]
        valid = np.isfinite(patch) & (patch > .001)
        v = patch[valid]
        hits = v[(v >= .1) & (v < 8)]
        value, winner = np.nan, None
        indices, contributor_weights = np.array([], np.int64), np.array([], float)
        reason = 'INSUFFICIENT_HITS'
        if hits.size >= 4:
            if rng.random() < .05:
                reason = 'SIMULATED_DROPOUT'
            else:
                bins = np.minimum((hits / .1).astype(int), 79)
                weights = np.bincount(bins, weights=1 / np.maximum(hits, .3)**2, minlength=80)
                selected = int(np.argmax(weights))
                center = np.mean(hits[bins == selected])
                noisy = center + rng.normal(0, .01 + .02 * center)
                reason = 'NOISY_RANGE_OUTSIDE_LIMIT'
                if .1 <= noisy < 8:
                    value, winner, reason = noisy, selected, 'OBSERVED'
                    eligible = valid & (patch >= .1) & (patch < 8)
                    py, px = np.nonzero(eligible)
                    selected_samples = bins == selected
                    indices = ((py[selected_samples] + y0) * w + px[selected_samples] + x0).astype(np.int64)
                    contributor_weights = (1 / np.maximum(hits[selected_samples], .3)**2).astype(float)
        values.append(value)
        traces.append(dict(zone_id=zone_id, observed=bool(np.isfinite(value)),
            distance_m=float(np.float32(value)) if np.isfinite(value) else None,
            winner_bin=winner, pixel_indices=indices, weights=contributor_weights, reason=reason))
    return np.asarray(values, np.float32), traces
