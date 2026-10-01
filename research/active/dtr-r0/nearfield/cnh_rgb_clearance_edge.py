"""Truth-blind local depth-edge clearance diagnostic.

Inputs are predicted optical-Z depth, public calibration factors and an assumed
coarse angular cell.  The 8 by 8, 45-degree field of view is a public geometric
proxy, not a measured ToF cell or a recorded ToF trigger.  No truth, instance or
semantic label is accepted by this module.  The fixed 30 cm half-width is a
diagnostic query assumption, not a calibrated wearer body boundary.
"""

import json

import numpy as np


EDGE_LOG_THRESHOLD = float(np.log(1.02))
HALF_BODY_WIDTH_M = 0.30
VERTICAL_WINDOW = 15
MIN_VALID_VERTICAL_PAIRS = 8


def zone_map(camera_matrix, shape=(768, 1024)):
    """Assign full-matrix negative-Z camera rays to 64 angular proxy cells.

    Pixel centers use the native Hypersim UV convention. Horizontal and
    vertical bounds are [-22.5, 22.5) degrees; increasing row is increasing
    downward angle. Forward-invalid or out-of-field rays receive -1.
    """
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("shape must contain positive height and width")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be a finite 3 by 3 matrix")
    yy, xx = np.indices((height, width), dtype=np.float64)
    uv = np.stack(((xx + 0.5) * 2 / width - 1,
                   1 - (yy + 0.5) * 2 / height, np.ones_like(xx)), axis=-1)
    rays = uv @ matrix.T
    horizontal = np.arctan2(rays[..., 0], -rays[..., 2])
    vertical = np.arctan2(-rays[..., 1], -rays[..., 2])
    half_fov = np.deg2rad(22.5)
    valid = ((rays[..., 2] < 0) & (horizontal >= -half_fov)
             & (horizontal < half_fov) & (vertical >= -half_fov)
             & (vertical < half_fov))
    result = np.full((height, width), -1, dtype=np.int32)
    horizontal_cell = np.floor((horizontal[valid] + half_fov) / (2 * half_fov) * 8).astype(int)
    vertical_cell = np.floor((vertical[valid] + half_fov) / (2 * half_fov) * 8).astype(int)
    result[valid] = vertical_cell * 8 + horizontal_cell
    return result


def _vertical_mean(values, valid):
    """15-row arithmetic mean of valid pairs, truncated at image borders."""
    radius = VERTICAL_WINDOW // 2
    total = np.vstack((np.zeros((1, values.shape[1])),
                       np.cumsum(np.where(valid, values, 0), axis=0)))
    count = np.vstack((np.zeros((1, values.shape[1]), dtype=np.int64),
                       np.cumsum(valid, axis=0)))
    rows = np.arange(values.shape[0])
    lo, hi = np.maximum(0, rows - radius), np.minimum(len(rows), rows + radius + 1)
    sums, counts = total[hi] - total[lo], count[hi] - count[lo]
    means = np.zeros_like(values)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means, counts


def extract_boundary(depth, lateral_factor, zones, zone_id):
    """Extract strongest directed jump in the supplied proxy cell only.

    Right side selects foreground to the right; left side selects foreground
    to the left. Vertical smoothing ignores invalid pairs and pairs outside
    the cell, but the selected center pair must itself be valid and at least
    8 pairs must support its 15-row window. Foreground Z is the valid median
    in y +/- 3 and columns x+2..x+5 (right) or x-1..x-4 (left), the integer
    convention for offsets 2..5 from the half-pixel edge toward the object.
    Invalid samples are excluded and counted;
    an empty foreground patch never produces a clearance.

    Pixel coordinates are [row, column + .5] in pixel-center index coordinates.
    Undefined quantities are JSON null, never NaN. Ties use raster order.
    """
    depth = np.asarray(depth, dtype=np.float64)
    lateral_factor = np.asarray(lateral_factor, dtype=np.float64)
    zones = np.asarray(zones)
    if depth.ndim != 2 or min(depth.shape) < 1:
        raise ValueError("depth must be a nonempty 2D optical-Z array")
    if lateral_factor.shape != depth.shape or zones.shape != depth.shape:
        raise ValueError("depth, lateral_factor and zones shapes must match")
    if not isinstance(zone_id, (int, np.integer)) or not 0 <= int(zone_id) < 64:
        raise ValueError("zone_id must be an integer in 0..63")
    if not np.issubdtype(zones.dtype, np.integer):
        raise ValueError("zones must be an integer array")
    result = dict(status="EMPTY_ZONE", side=None, edge_pixel=None,
                  foreground_depth_m=None, lateral_factor=None, clearance_m=None,
                  strength=0.0, foreground_valid_samples=0,
                  foreground_total_samples=0, smoothing_valid_pairs=0)
    cell = zones == zone_id
    factors = lateral_factor[cell & np.isfinite(lateral_factor)]
    if not len(factors):
        return result
    side = int(np.sign(np.median(factors)))
    result["side"] = side
    if side == 0:
        result["status"] = "AMBIGUOUS_SIDE"
        return result
    valid_z = np.isfinite(depth) & (depth > 0)
    logz = np.zeros_like(depth)
    np.log(depth, out=logz, where=valid_z)
    valid_pair = (cell[:, :-1] & cell[:, 1:] & valid_z[:, :-1]
                  & valid_z[:, 1:] & np.isfinite(lateral_factor[:, :-1])
                  & np.isfinite(lateral_factor[:, 1:]))
    if not valid_pair.any():
        result["status"] = "NO_VALID_EDGE_PAIR"
        return result
    jump = side * (logz[:, :-1] - logz[:, 1:])
    strength, counts = _vertical_mean(jump, valid_pair)
    supported = valid_pair & (counts >= MIN_VALID_VERTICAL_PAIRS)
    if not supported.any():
        result["status"] = "INSUFFICIENT_VERTICAL_SUPPORT"
        return result
    candidate_strength = np.where(supported, strength, -np.inf)
    y, x = np.unravel_index(np.argmax(candidate_strength), candidate_strength.shape)
    best = float(candidate_strength[y, x])
    result.update(strength=best, smoothing_valid_pairs=int(counts[y, x]))
    if best < EDGE_LOG_THRESHOLD:
        result["status"] = "NO_EDGE"
        return result
    edge_factor = float((lateral_factor[y, x] + lateral_factor[y, x + 1]) / 2)
    result.update(edge_pixel=[int(y), float(x + 0.5)], lateral_factor=edge_factor)
    columns = x + np.arange(2, 6) if side > 0 else x - np.arange(1, 5)
    columns = columns[(columns >= 0) & (columns < depth.shape[1])]
    patch = depth[max(0, y - 3):min(depth.shape[0], y + 4), columns]
    valid_foreground = np.isfinite(patch) & (patch > 0)
    result.update(foreground_valid_samples=int(valid_foreground.sum()),
                  foreground_total_samples=int(patch.size))
    if not valid_foreground.any():
        result["status"] = "NO_FOREGROUND_DEPTH"
        return result
    foreground_depth = float(np.median(patch[valid_foreground]))
    clearance = side * edge_factor * foreground_depth - HALF_BODY_WIDTH_M
    if not np.isfinite(clearance):
        result["status"] = "NONFINITE_CLEARANCE"
        return result
    result.update(status="OK", foreground_depth_m=foreground_depth,
                  clearance_m=float(clearance))
    return result


def self_check():
    """Pure mathematical contract checks; no data, predictions or labels read."""
    shape = (31, 64)
    cell = np.zeros(shape, dtype=np.int32)
    factor = np.broadcast_to(np.linspace(.10, .60, shape[1]), shape).copy()
    depth = np.full(shape, 4.0)
    depth[:, 32:] = 2.0
    right = extract_boundary(depth, factor, cell, 0)
    assert right["status"] == "OK" and right["edge_pixel"][1] == 31.5
    assert right["foreground_depth_m"] == 2.0
    assert np.isclose(right["strength"], np.log(2))
    assert np.isclose(right["clearance_m"], .4)
    left = extract_boundary(depth[:, ::-1], -factor[:, ::-1], cell, 0)
    assert left["status"] == "OK" and left["side"] == -1
    assert left["edge_pixel"][1] == 31.5
    assert np.isclose(left["clearance_m"], right["clearance_m"])
    flat = extract_boundary(np.ones(shape), factor, cell, 0)
    assert flat["status"] == "NO_EDGE" and flat["clearance_m"] is None
    invalid = depth.copy()
    invalid[:, 33:38] = np.nan
    missing = extract_boundary(invalid, factor, cell, 0)
    assert missing["status"] == "NO_FOREGROUND_DEPTH"
    assert missing["clearance_m"] is None and missing["foreground_valid_samples"] == 0
    partial = depth.copy()
    partial[0, 33] = np.nan
    counted = extract_boundary(partial, factor, cell, 0)
    assert counted["status"] == "OK"
    assert counted["foreground_valid_samples"] < counted["foreground_total_samples"]
    assert counted["foreground_depth_m"] == 2.0
    excluded = cell.copy()
    excluded[:, 32:] = -1
    assert extract_boundary(depth, factor, excluded, 0)["status"] == "NO_EDGE"
    sparse = np.full(shape, np.nan)
    sparse[15] = depth[15]
    assert extract_boundary(sparse, factor, cell, 0)["status"] == "INSUFFICIENT_VERTICAL_SUPPORT"
    matrix = np.array([[.8, .02, .03], [.04, .6, -.07], [.02, .01, -1.]])
    mapped = zone_map(matrix, shape)
    for y, x in ((0, 0), (15, 31), (30, 63), (10, 20)):
        ray = matrix @ np.array([(x + .5) * 2 / shape[1] - 1,
                                 1 - (y + .5) * 2 / shape[0], 1.])
        angles = np.array([np.arctan2(ray[0], -ray[2]),
                           np.arctan2(-ray[1], -ray[2])])
        half = np.deg2rad(22.5)
        expected = -1
        if ray[2] < 0 and np.all((angles >= -half) & (angles < half)):
            h, v = np.floor((angles + half) / (2 * half) * 8).astype(int)
            expected = int(v * 8 + h)
        assert mapped[y, x] == expected
    for result in (right, left, flat, missing, counted):
        json.dumps(result, allow_nan=False)
    return dict(status="PASS", checks=["exact_step", "flat_no_edge", "left_right_symmetry",
                "missing_foreground_explicit", "partial_invalid_counted", "zone_pair_restriction",
                "minimum_valid_vertical_support", "full_matrix_angular_cells", "strict_json_finite"])


if __name__ == "__main__":
    print(json.dumps(self_check(), allow_nan=False))
