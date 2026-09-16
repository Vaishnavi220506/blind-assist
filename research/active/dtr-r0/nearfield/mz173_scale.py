"""One public regional slant-range scale for a supplied causal depth sequence.

This is an arithmetic pixel-mean proxy, not a physical return-ownership model.
No raw sensor support is altered. The caller owns causal prefix selection and
passes only the final repeated-current row/map for that control. Duplicate IDs
are defensively collapsed to their last occurrence, never counted as new data.
"""
from collections import Counter
import math
import numpy as np
from mz115_spatial_allocation import zone_box

METHOD = dict(schema='MZ173_PUBLIC_REGIONAL_SINGLE_SCALE_V1',
    input_shape=[360, 640], maximum_sequence_frames=5,
    scale='MEDIAN_RETURNED_SLANT_OVER_MEAN_PREDICTED_SLANT',
    region_proxy='ARITHMETIC_MEAN_OF_ALL_CEIL_FLOOR_ZONE_PIXELS',
    anchors='EXACTLY_ONE_USABLE_SIM_VALID_AND_NO_SIM_MERGED_TARGET',
    duplicates='LAST_SUPPLIED_MAP_PER_FRAME_ID;ONE_ANCHOR_PER_FRAME_ZONE',
    unknown='NO_ANCHOR_OR_INVALID_CURRENT_PIXEL_IS_NOT_CLEAR',
    authority='PUBLIC_INPUT_ONLY;NO_RETURN_OWNERSHIP_OR_NATIVE_SUPPORT_CLAIM')


def _finite(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value)))


def _intrinsics(row):
    k = row['rgb_intrinsics']
    if (k['height'], k['width']) != (360, 640):
        raise ValueError('Expected native640x360 intrinsics')
    if not all(_finite(k[n]) for n in ('fx', 'fy', 'cx', 'cy')) or min(k['fx'], k['fy']) <= 0:
        raise ValueError('Finite positive focal lengths and finite principal point required')
    return k


def _median(values):
    # Positive ratios may be large; avoid overflow in an even-count midpoint.
    values = sorted(values);n = len(values)
    return float(values[n//2] if n % 2 else values[n//2-1]/2 + values[n//2]/2)


def calibrate(rows, maps):
    """Return (scaled current-last axis-z float32[360,640], JSON-safe audit).

    ``rows`` and ``maps`` are corresponding already-observed views, at most5.
    Maps are relative camera-axis depth, never inverse depth. A regional anchor
    requires *every* included pixel to have finite positive predicted depth.
    Scale uses all distinct eligible frame/zone pairs; it has no intercept,
    minimum-zone count beyond one, residual gate or fitted outcome threshold.
    Invalid current pixels, including scaled float32 overflow, remain NaN.
    ``audit['valid']`` means a usable scale exists, not all pixels are valid.
    """
    if not 1 <= len(rows) <= 5 or len(rows) != len(maps):
        raise ValueError('Expected1..5 corresponding supplied rows/maps')
    last = {}
    for i, row in enumerate(rows):
        if not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('Nonempty public frame ID required for anchor deduplication')
        _intrinsics(row)
        if np.asarray(maps[i]).shape != (360, 640):
            raise ValueError('Expected native relative-axis-depth map360x640')
        last[row['id']] = i
    indices = sorted(last.values())
    anchors = [];excluded = [];yy, xx = np.mgrid[:360, :640]

    def exclude(row, zid, reason):
        excluded.append(dict(frame_id=row['id'], zone_id=zid, reason=reason))

    for i in indices:
        row = rows[i];k = _intrinsics(row);zmap = np.asarray(maps[i], dtype=np.float64)
        norm = np.sqrt(1 + ((xx-k['cx'])/k['fx'])**2 + ((k['cy']-yy)/k['fy'])**2)
        zones = row['tof_zones'];zids = [z['zone_id'] for z in zones]
        if (any(not isinstance(zid, (int, np.integer)) or isinstance(zid, (bool, np.bool_))
                or not 0 <= zid < 64 for zid in zids) or len(set(zids)) != len(zids)):
            raise ValueError('Public zone IDs must be unique integers0..63')
        for zone in sorted(zones, key=lambda z: z['zone_id']):
            zid = int(zone['zone_id'])
            if not row['tof_packet_received']:
                exclude(row, zid, 'MISSING_TOF_PACKET');continue
            bounds = [zone.get(key) for key in ('theta_bounds_deg', 'phi_bounds_deg')]
            if any(not isinstance(b, (list, tuple, np.ndarray)) or len(b) != 2
                   or not all(_finite(v) for v in b) or not -90 < b[0] < b[1] < 90 for b in bounds):
                exclude(row, zid, 'INVALID_PUBLIC_ZONE_GEOMETRY');continue
            box = np.asarray(zone_box(zone, k), dtype=np.float64)
            if not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= 639 and 0 <= box[1] < box[3] <= 359):
                exclude(row, zid, 'INCOMPLETE_RGB_ZONE');continue
            targets = zone.get('targets')
            if not isinstance(targets, list) or len(targets) > 2 or any(not isinstance(t, dict) for t in targets):
                exclude(row, zid, 'MALFORMED_TARGETS');continue
            if any(t.get('status') == 'SIM_MERGED' for t in targets):
                exclude(row, zid, 'MERGED_TARGET_PRESENT');continue
            usable = [(j, t) for j, t in enumerate(targets)
                      if t.get('status') == 'SIM_VALID'
                      and all(_finite(t.get(n)) for n in ('distance_m', 'range_noise_sigma_m', 'signal_strength_proxy'))
                      and t['distance_m'] > 0 and min(t['range_noise_sigma_m'], t['signal_strength_proxy']) >= 0]
            if len(usable) != 1:
                exclude(row, zid, 'NO_USABLE_VALID_TARGET' if not usable else 'MULTIPLE_USABLE_VALID_TARGETS');continue
            l, top = math.ceil(box[0]), math.ceil(box[1]);r, bottom = math.floor(box[2]), math.floor(box[3])
            if l > r or top > bottom:
                exclude(row, zid, 'EMPTY_ZONE_RASTER');continue
            patch = zmap[top:bottom+1, l:r+1]
            if not np.isfinite(patch).all() or not (patch > 0).all():
                exclude(row, zid, 'INVALID_RELATIVE_ZONE_PIXELS');continue
            with np.errstate(over='ignore', invalid='ignore'):
                mean_slant = float(np.mean(patch*norm[top:bottom+1, l:r+1], dtype=np.float64))
            slot, target = usable[0]
            if not math.isfinite(mean_slant) or mean_slant <= 0:
                exclude(row, zid, 'INVALID_PREDICTED_MEAN_SLANT');continue
            ratio = float(target['distance_m'])/mean_slant
            if not math.isfinite(ratio) or ratio <= 0:
                exclude(row, zid, 'INVALID_SCALE_RATIO');continue
            anchors.append(dict(frame_id=row['id'], zone_id=zid, slot=slot,
                range_m=float(target['distance_m']), predicted_mean_slant=mean_slant,
                ratio=ratio, pixel_count=int(patch.size), pixel_bounds_inclusive=[l, top, r, bottom]))

    ratios = [a['ratio'] for a in anchors]
    scale = _median(ratios) if ratios else None
    ratio_mad = _median([abs(v-scale) for v in ratios]) if ratios else None
    for anchor in anchors:
        predicted = scale*anchor['predicted_mean_slant']
        anchor['scaled_predicted_mean_slant_m'] = predicted if math.isfinite(predicted) else None
        residual = abs(predicted-anchor['range_m'])
        anchor['abs_residual_m'] = residual if math.isfinite(residual) else None
    current = np.asarray(maps[-1], dtype=np.float64)
    good = np.isfinite(current) & (current > 0)
    result = np.full((360, 640), np.nan, dtype=np.float32);overflow = 0
    if scale is not None:
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = current[good]*scale
        representable = np.isfinite(scaled) & (scaled > 0) & (scaled <= np.finfo(np.float32).max)
        converted = np.full(scaled.shape, np.nan, dtype=np.float32)
        converted[representable] = scaled[representable].astype(np.float32)
        # Float32 underflow is also UNKNOWN rather than a zero/free-space value.
        converted[~(converted > 0)] = np.nan
        result[good] = converted;overflow = int((~np.isfinite(converted)).sum())
    scaled_count = int(np.isfinite(result).sum())
    status = ('UNKNOWN_NO_ANCHORS' if scale is None else
              'SCALED' if scaled_count else 'UNKNOWN_NO_VALID_CURRENT_PIXELS')
    audit = dict(method=METHOD, status=status, valid=scale is not None,
        scale=scale, ratio_mad=ratio_mad, input_frames=len(rows), unique_frames=len(indices),
        duplicates_removed=len(rows)-len(indices), supplied_frame_ids=[r['id'] for r in rows],
        used_frame_ids=[rows[i]['id'] for i in indices], current_frame_id=rows[-1]['id'],
        anchor_count=len(anchors), excluded_count=len(excluded), exclusion_counts=dict(Counter(e['reason'] for e in excluded)),
        anchors=anchors, excluded=excluded, current_finite_positive_pixels=int(good.sum()),
        current_invalid_pixels=int((~good).sum()), scaled_valid_pixels=scaled_count,
        scaled_unrepresentable_pixels=overflow, raw_sensor_inputs_changed=False,
        measurement_authority='UNIFORM_REGIONAL_SLANT_PROXY_NOT_RETURN_OWNERSHIP',
        causal_authority='ONLY_SUPPLIED_ROWS;PREFIX_SELECTION_REMAINS_CALLER_RESPONSIBILITY')
    return result, audit
