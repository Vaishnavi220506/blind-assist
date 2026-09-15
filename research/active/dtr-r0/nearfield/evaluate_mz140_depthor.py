"""Evaluator-only dense geometry diagnostics for the frozen MZ140 baseline.

This module accepts native evaluation data; it must never be called by the
predictor or supply an algorithm input. Reference rendering covers saved native
AABBs under native pose when available. Unsaved scene geometry remains unknown.
"""
import math

import cv2
import numpy as np

from mz136_boundary_geometry import camera_to_body

AGREEMENT_M = .12
NUMERICAL_M = 1e-6
SILHOUETTE_RING_PX = 8.
METHOD = dict(
    authority='EVALUATOR_ONLY_NATIVE_AABB_REFERENCE_NATIVE_CAMERA_WHEN_AVAILABLE',
    reference='FIRST_HIT_OF_ALL_SAVED_NATIVE_BOUNDS_WITH_TARGET_OCCLUSION',
    depth='CAMERA_FORWARD_AXIS_METERS_NOT_SLANT_RANGE',
    pixel_coordinates='INTEGER_RGB_PIXEL_CENTERS_WITH_PUBLIC_INTRINSICS',
    range_agreement_m=AGREEMENT_M,
    contributor_numerical_tolerance_m=NUMERICAL_M,
    contributor_sampling='BILINEAR_AT_REFERENCE_CAMERA_PROJECTED_NATIVE_HIT',
    silhouette_ring_px=SILHOUETTE_RING_PX,
    tolerances='DIAGNOSTIC_ONLY_NOT_ALERT_THRESHOLDS_OR_DILATION',
    unknown_reference='NO_SAVED_AABB_HIT_IS_UNKNOWN_NOT_FREE_SPACE',
)


def _corridor(point):
    return bool(.2 <= point[0] <= 3.6 and -.3 <= point[1] <= .3 and .4 <= point[2] <= 2.05)


def _reference_camera(row, evaluation, cached_corrected):
    camera = evaluation.get('camera', {})
    if all(key in camera for key in ('x', 'y', 'z', 'yaw', 'pitch')):
        p, y, r = [math.radians(camera.get(key, 0.)) for key in ('pitch', 'yaw', 'roll')]
        cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
        forward = [cp*cy, cp*sy, sp]
        right = [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, -sr*cp]
        up = [-cr*sp*cy-sr*sy, -cr*sp*sy+sr*cy, cr*cp]
        return np.asarray([forward, right, up]).T, np.asarray([camera[k] for k in ('x', 'y', 'z')]), 'NATIVE_EVALUATOR_CAMERA'
    rotation = camera_to_body(row, float(cached_corrected['integrated_yaw_deg']))
    origin = np.asarray(evaluation['body_origin_m'], float)+np.asarray(row['camera_in_body_m'], float)
    return rotation, origin, 'PUBLIC_POSE_FALLBACK_NATIVE_CAMERA_UNAVAILABLE'


def rasterize_native_bounds(row, evaluation, cached_corrected):
    """Return evaluator arrays; parameter along an unnormalized ray is z-depth.

    Body forward is the first axis in this project. The camera ray [1,a,b]
    deliberately remains unnormalized: its intersection parameter is camera
    forward depth rather than Euclidean distance. Saved object AABBs are in
    world coordinates; subtract body origin and camera offset exactly once.
    """
    intr = row['rgb_intrinsics']
    h, w = int(intr['height']), int(intr['width'])
    rotation, origin, pose_source = _reference_camera(row, evaluation, cached_corrected)
    yy, xx = np.mgrid[:h, :w]
    camera_rays = np.stack((np.ones((h, w)), (xx-intr['cx'])/intr['fx'],
                            (intr['cy']-yy)/intr['fy']), axis=-1)
    directions = camera_rays @ rotation.T
    depth = np.full((h, w), np.inf, dtype=np.float64)
    owner = np.full((h, w), -1, dtype=np.int32)
    objects = evaluation['native_bounds']
    names = [obj['name'] for obj in objects]
    if len(set(names)) != len(names):
        raise ValueError('Native object names are not unique')
    for index, obj in enumerate(objects):
        low = np.asarray(obj['center_m'], float)-np.asarray(obj['extent_m'], float)
        high = np.asarray(obj['center_m'], float)+np.asarray(obj['extent_m'], float)
        if not np.isfinite(np.r_[low, high]).all() or np.any(low > high):
            raise ValueError('Invalid native bounds')
        enter = np.full((h, w), -np.inf)
        leave = np.full((h, w), np.inf)
        for axis in range(3):
            direction = directions[..., axis]
            parallel = np.abs(direction) < 1e-14
            safe = np.where(parallel, 1., direction)
            first = (low[axis]-origin[axis])/safe
            last = (high[axis]-origin[axis])/safe
            slab_lo, slab_hi = np.minimum(first, last), np.maximum(first, last)
            origin_inside = low[axis] <= origin[axis] <= high[axis]
            slab_lo = np.where(parallel, -np.inf if origin_inside else np.inf, slab_lo)
            slab_hi = np.where(parallel, np.inf if origin_inside else -np.inf, slab_hi)
            enter = np.maximum(enter, slab_lo)
            leave = np.minimum(leave, slab_hi)
        first_positive = np.where(enter > 0, enter, leave)
        hit = (leave >= enter) & (leave > 0) & (first_positive > 0) & np.isfinite(first_positive)
        nearer = hit & (first_positive < depth)
        depth[nearer] = first_positive[nearer]
        owner[nearer] = index
    target_indices = [i for i, name in enumerate(names) if name == 'shape0']
    target = owner == target_indices[0] if target_indices else np.zeros((h, w), bool)
    depth[owner < 0] = np.nan
    return dict(depth=depth, owner=owner, target_visible=target, names=names,
                camera_rotation=rotation, camera_origin_world_m=origin, pose_source=pose_source)


def _finite_stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, median=None, p90=None, maximum=None)
    return dict(count=int(len(values)), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()))


def _bilinear(depth, u, v):
    h, w = depth.shape
    if not (0 <= u <= w-1 and 0 <= v <= h-1):
        return None
    left, top = int(math.floor(u)), int(math.floor(v))
    right, bottom = min(left+1, w-1), min(top+1, h-1)
    du, dv = u-left, v-top
    weighted = [(top, left, (1-du)*(1-dv)), (top, right, du*(1-dv)),
                (bottom, left, (1-du)*dv), (bottom, right, du*dv)]
    total = 0.
    for y, x, weight in weighted:
        if weight <= 0:
            continue
        value = float(depth[y, x])
        if not math.isfinite(value) or value <= 0:
            return None
        total += weight*value
    return total


def contributor_diagnostics(row, evaluation, depth, rotation, camera_origin=None):
    """Read returned lineage only; unreturned native subrays are not observations."""
    intr = row['rgb_intrinsics']
    if camera_origin is None:
        camera_origin = np.asarray(evaluation['body_origin_m'], float)+np.asarray(row['camera_in_body_m'], float)
    records = []
    unresolved = 0
    for zone in evaluation.get('zonal_tof_native', []):
        rays = {ray['subray']: ray for ray in zone['private_rays']}
        for lineage in zone['returned_lineage']:
            for index in lineage['hit_indices']:
                hit = rays.get(index)
                if hit is None or 'hit_point_m' not in hit or not np.isfinite(hit['hit_point_m']).all():
                    unresolved += 1
                    continue
                world = np.asarray(hit['hit_point_m'], float)
                body = world-np.asarray(evaluation['body_origin_m'], float)
                camera = (world-camera_origin) @ rotation
                forward = float(camera[0])
                uv = None
                sampled = None
                in_rgb = False
                if forward > 0:
                    uv = [float(intr['cx']+intr['fx']*camera[1]/forward),
                          float(intr['cy']-intr['fy']*camera[2]/forward)]
                    in_rgb = bool(0 <= uv[0] <= intr['width']-1 and 0 <= uv[1] <= intr['height']-1)
                    sampled = _bilinear(depth, *uv) if in_rgb else None
                error = abs(sampled-forward) if sampled is not None else None
                records.append(dict(zone=zone['zone_id'], slot=lineage['target_index'], subray=index,
                    actor=hit.get('actor_id'), point_body_m=body.tolist(), corridor=_corridor(body),
                    in_rgb=in_rgb, pixel_uv=uv, native_axis_depth_m=forward,
                    predicted_axis_depth_m=sampled, absolute_error_m=error,
                    numerical_agreement=bool(error is not None and error <= NUMERICAL_M),
                    diagnostic_012m_agreement=bool(error is not None and error <= AGREEMENT_M)))
    summary = dict(returned_contributor_records=len(records), unresolved_lineage_hits=unresolved,
                   numerical_tolerance_m=NUMERICAL_M, diagnostic_tolerance_m=AGREEMENT_M,
                   sampling=METHOD['contributor_sampling'])
    for name, group in [('all', records), ('corridor', [r for r in records if r['corridor']])]:
        visible = [r for r in group if r['in_rgb']]
        summary[name] = dict(total=len(group), inside_rgb=len(visible), outside_rgb=len(group)-len(visible),
            finite_prediction=sum(r['predicted_axis_depth_m'] is not None for r in visible),
            numerical_agreement=sum(r['numerical_agreement'] for r in visible),
            diagnostic_012m_agreement=sum(r['diagnostic_012m_agreement'] for r in visible),
            diagnostic_012m_fraction=sum(r['diagnostic_012m_agreement'] for r in visible)/len(visible) if visible else None,
            absolute_error_m=_finite_stats([r['absolute_error_m'] for r in visible if r['absolute_error_m'] is not None]))
    return dict(summary=summary, records=records)


def evaluate_frame(row, evaluation, depth, cached_corrected):
    """JSONable per-frame diagnostics; no output here is an oracle alarm arm."""
    if 'id' in evaluation and evaluation['id'] != row['id']:
        raise ValueError('Frame/evaluator ID mismatch')
    depth = np.asarray(depth, float)
    intr = row['rgb_intrinsics']
    if depth.shape != (intr['height'], intr['width']):
        raise ValueError('Predicted depth must match original RGB resolution/intrinsics')
    reference = rasterize_native_bounds(row, evaluation, cached_corrected)
    gt = reference['depth']
    target = reference['target_visible']
    pred_valid = np.isfinite(depth) & (depth > 0)
    ref_valid = np.isfinite(gt) & (gt > 0)
    comparable = pred_valid & ref_valid
    error = np.full(depth.shape, np.nan)
    error[comparable] = np.abs(depth[comparable]-gt[comparable])
    agreeing = comparable & (error <= AGREEMENT_M)
    target_total = int(target.sum())
    retained = agreeing & target
    target_columns = np.flatnonzero(target.any(axis=0))
    column_records = [dict(column=int(x), visible_target_pixels=int(target[:, x].sum()),
                           agreeing_target_pixels=int(retained[:, x].sum())) for x in target_columns]
    any_columns = sum(c['agreeing_target_pixels'] > 0 for c in column_records)
    half_columns = sum(c['agreeing_target_pixels'] >= .5*c['visible_target_pixels'] for c in column_records)
    ring_summary = dict(radius_px=SILHOUETTE_RING_PX, pixels=0, referenced_pixels=0,
        unknown_reference_pixels=0, invalid_prediction_pixels=0,
        target_depth_band_spurious_pixels=0, closer_or_target_like_error_pixels=0)
    if target_total:
        distances, labels = cv2.distanceTransformWithLabels((~target).astype(np.uint8),
            cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        nearest_depth = np.full(int(labels.max())+1, np.nan)
        nearest_depth[labels[target]] = gt[target]
        nearby_target_depth = nearest_depth[labels]
        ring = (~target) & (distances <= SILHOUETTE_RING_PX)
        ring_ref = ring & ref_valid
        wrong = ring_ref & pred_valid & ~agreeing
        target_like = np.abs(depth-nearby_target_depth) <= AGREEMENT_M
        closer_or_target_like = depth <= nearby_target_depth+AGREEMENT_M
        ring_summary.update(pixels=int(ring.sum()), referenced_pixels=int(ring_ref.sum()),
            unknown_reference_pixels=int((ring & ~ref_valid).sum()),
            invalid_prediction_pixels=int((ring_ref & ~pred_valid).sum()),
            target_depth_band_spurious_pixels=int((wrong & target_like).sum()),
            closer_or_target_like_error_pixels=int((wrong & closer_or_target_like).sum()))
    ring_summary['semantics'] = ('Non-target pixels within8px of visible target. Spurious requires a saved '
        'native first-hit reference, disagreement>.12m with that reference, and predicted depth within '
        '+/-.12m of the nearest visible target depth. Unknown scene reference never counts as spurious.')
    contributor = contributor_diagnostics(row, evaluation, depth, reference['camera_rotation'], reference['camera_origin_world_m'])
    public_yaw = float(cached_corrected['integrated_yaw_deg'])
    native_camera = evaluation.get('camera', {})
    public_origin = np.asarray(evaluation['body_origin_m'], float)+np.asarray(row['camera_in_body_m'], float)
    pose_delta = dict(reference_pose_source=reference['pose_source'],
        public_yaw_deg=public_yaw, public_pitch_deg=float(row['camera_pitch_deg']),
        native_minus_public_yaw_deg=float(native_camera['yaw'])-public_yaw if 'yaw' in native_camera else None,
        native_minus_public_pitch_deg=float(native_camera['pitch'])-float(row['camera_pitch_deg']) if 'pitch' in native_camera else None,
        native_minus_public_origin_m=([float(native_camera[k])-public_origin[i] for i, k in enumerate(('x', 'y', 'z'))]
            if all(k in native_camera for k in ('x', 'y', 'z')) else None))
    true_surface = int(retained.sum())
    spurious_surface = ring_summary['target_depth_band_spurious_pixels']
    local_precision = true_surface/(true_surface+spurious_surface) if true_surface+spurious_surface else None
    return dict(id=row['id'], family=evaluation.get('family'), method=METHOD, pose_diagnostic=pose_delta,
        pixels=int(depth.size), valid_prediction_pixels=int(pred_valid.sum()),
        native_bounds_reference_pixels=int(ref_valid.sum()), unknown_reference_pixels=int((~ref_valid).sum()),
        full_reference_absolute_error_m=_finite_stats(error[comparable]),
        full_reference_agreeing_pixels=int(agreeing.sum()),
        target=dict(present_in_saved_bounds='shape0' in reference['names'],
            visible_pixels=target_total, comparable_pixels=int((target & pred_valid).sum()),
            range_agreeing_pixels=int(retained.sum()),
            retained_fraction=int(retained.sum())/target_total if target_total else None,
            absolute_error_m=_finite_stats(error[target & pred_valid]),
            transverse_columns=dict(visible=len(column_records), with_any_agreement=any_columns,
                any_agreement_fraction=any_columns/len(column_records) if column_records else None,
                with_at_least_half_pixels_agreeing=half_columns,
                half_pixel_agreement_fraction=half_columns/len(column_records) if column_records else None,
                per_column=column_records)),
        silhouette_neighborhood=ring_summary,
        local_target_surface=dict(true_agreeing_pixels=true_surface, spurious_pixels=spurious_surface,
            precision=local_precision, semantics='TARGET_AGREEING_PIXELS_DIVIDED_BY_TARGET_AGREEING_PLUS_REFERENCED_RING_TARGET_LIKE_ERRORS'),
        native_returned_contributors=contributor,
        assumptions=[
            'Saved native bounds are axis-aligned collision cuboids; unrecorded floor/background and RGB-only texture tiles are excluded.',
            'Occlusion is resolved among all saved native bounds. This is not a native UE depth image.',
            'Native evaluator camera registers the image reference when available; public-pose fallback is reported explicitly.',
            'Public-versus-native pose discrepancy is separate from image depth error and remains relevant to algorithm-side body unprojection.',
            'Range agreement and column coverage describe sampled visible geometry, not obstacle recall or certified empty space.',
            '.12m is a fixed3-sigma-scale diagnostic only; no evaluator geometry or metric controls prediction.'
        ])
