"""Offline TRAIN labels from saved native AABBs and source-commanded pose.

This module is evaluator authority, never a predictor input. Analytic first hits
are not rendered segmentation/depth. A missing saved AABB hit remains UNKNOWN.
"""
import math

import numpy as np

from evaluate_mz140_depthor import rasterize_native_bounds


CORRIDOR_LO = np.array([.2, -.3, .4], dtype=np.float64)
CORRIDOR_HI = np.array([3.6, .3, 2.05], dtype=np.float64)
METHOD = dict(
    schema='MZ161_MASKED_VISIBLE_FIRST_HIT_CORRIDOR_LABELS_V1',
    authority='OFFLINE_TRAIN_NATIVE_AABB_LABELS_NOT_PREDICTOR_INPUT',
    pose_authority='SOURCE_COMMANDED_REFERENCE',
    geometry='FIRST_POSITIVE_RAY_HIT_OF_ALL_SAVED_NATIVE_ACTOR_AABBS',
    unknown='NO_SAVED_AABB_HIT_IS_UNKNOWN_NOT_FREE_SPACE',
    pixels='ORIGINAL_RGB_INTEGER_PIXEL_CENTERS',
    weights='POSITIVE_AND_KNOWN_NEGATIVE_CLASS_MASS_ONE_HALF_EACH_WHEN_BOTH_PRESENT',
    rendered_labels=False,
)


def _column_counts(visible, corridor):
    return [dict(column=int(x), visible_pixels=int(visible[:, x].sum()),
                 corridor_pixels=int(corridor[:, x].sum()))
            for x in np.flatnonzero(visible.any(axis=0))]


def make_labels(row, evaluation, yaw):
    """Return boolean HxW masks, float32 class-balanced weights and JSON audit.

    ``target`` covers every saved actor's visible first hit in the body corridor.
    ``target_visible`` and ``target_corridor`` identify the named shape0 actor
    only for diagnostics. Native AABB volume truth can differ from visible risky
    pixels through occlusion, field of view or pixel sampling; neither is changed.
    ``yaw`` is public integrated yaw; the complete commanded reference pose owns
    the analytic label projection. Callers own TRAIN-only record selection.
    """
    if not math.isfinite(float(yaw)):
        raise ValueError('Finite public yaw required')
    if 'id' in row and 'id' in evaluation and row['id'] != evaluation['id']:
        raise ValueError('Frame/evaluator ID mismatch')
    camera = evaluation.get('camera', {})
    if not all(key in camera and math.isfinite(float(camera[key]))
               for key in ('x', 'y', 'z', 'yaw', 'pitch')):
        raise ValueError('Complete finite source-commanded reference camera required')
    if not math.isfinite(float(camera.get('roll', 0.))):
        raise ValueError('Finite reference roll required')
    body_origin = np.asarray(evaluation['body_origin_m'], dtype=np.float64)
    if body_origin.shape != (3,) or not np.isfinite(body_origin).all():
        raise ValueError('Finite three-dimensional body origin required')
    intr = row['rgb_intrinsics']
    if (int(intr['width']) <= 0 or int(intr['height']) <= 0
            or not all(math.isfinite(float(intr[k])) for k in ('fx', 'fy', 'cx', 'cy'))
            or intr['fx'] <= 0 or intr['fy'] <= 0):
        raise ValueError('Valid original RGB intrinsics required')
    reference = rasterize_native_bounds(row, evaluation, {'integrated_yaw_deg':float(yaw)})
    depth = reference['depth']
    known = (reference['owner'] >= 0) & np.isfinite(depth) & (depth > 0)
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    rays = np.stack([np.ones(depth.shape), (xx-intr['cx'])/intr['fx'],
                     (intr['cy']-yy)/intr['fy']], axis=-1)
    # The raster's parameter is camera-forward depth, so rays stay unnormalized.
    directions = rays @ reference['camera_rotation'].T
    body_points = (reference['camera_origin_world_m']-body_origin
                   + np.where(known, depth, 0.)[..., None]*directions)
    target = known & np.all((body_points >= CORRIDOR_LO) & (body_points <= CORRIDOR_HI), axis=-1)
    target_visible = reference['target_visible'] & known
    target_corridor = target_visible & target
    negative = known & ~target
    positive_count, negative_count = int(target.sum()), int(negative.sum())
    weights = np.zeros(depth.shape, dtype=np.float32)
    classes = int(positive_count > 0)+int(negative_count > 0)
    if positive_count:
        weights[target] = 1./(classes*positive_count)
    if negative_count:
        weights[negative] = 1./(classes*negative_count)
    actor_audit = []
    native_volume_truth = False
    for index, actor in enumerate(evaluation['native_bounds']):
        lo = np.asarray(actor['center_m'])-np.asarray(actor['extent_m'])-body_origin
        hi = np.asarray(actor['center_m'])+np.asarray(actor['extent_m'])-body_origin
        intersects = bool(np.all((hi >= CORRIDOR_LO) & (lo <= CORRIDOR_HI)))
        native_volume_truth |= intersects
        visible = known & (reference['owner'] == index)
        risky = visible & target
        actor_audit.append(dict(name=actor['name'], native_aabb_corridor_truth=intersects,
            visible_pixels=int(visible.sum()), visible_corridor_pixels=int(risky.sum()),
            visible_columns=int(visible.any(axis=0).sum()),
            visible_corridor_columns=int(risky.any(axis=0).sum()), columns=_column_counts(visible, risky)))
    visible_truth = bool(target.any())
    audit = dict(**METHOD, shape=list(depth.shape), known_pixels=int(known.sum()),
        unknown_pixels=int((~known).sum()), positive_pixels=positive_count,
        known_negative_pixels=negative_count, positive_weight_mass=float(weights[target].sum(dtype=np.float64)),
        negative_weight_mass=float(weights[negative].sum(dtype=np.float64)),
        total_weight_mass=float(weights.sum(dtype=np.float64)),
        native_aabb_corridor_truth=native_volume_truth, any_visible_first_hit_corridor=visible_truth,
        native_volume_truth_vs_visible_mismatch=bool(native_volume_truth != visible_truth),
        frame_truth_authority='ALL_SAVED_NATIVE_AABB_VOLUME_CORRIDOR_OVERLAP_NOT_SOURCE_DESIGN_LABEL',
        target_name='shape0', target_visible_pixels=int(target_visible.sum()),
        target_corridor_pixels=int(target_corridor.sum()),
        target_visible_columns=int(target_visible.any(axis=0).sum()),
        target_corridor_columns=int(target_corridor.any(axis=0).sum()),
        target_columns=_column_counts(target_visible, target_corridor), actors=actor_audit,
        public_yaw_deg=float(yaw), reference_yaw_deg=float(camera['yaw']),
        reference_minus_public_yaw_deg=float(camera['yaw'])-float(yaw),
        limits=['Saved native AABBs and source-commanded pose define an analytic reference, not rendered labels.',
                'Unknown scene geometry, occluded surfaces and between-pixel surfaces do not become negative evidence.',
                'Dense visible labels may miss a native volume intrusion; retain the separate frame label.',
                'No source-spec geometry or source-designed frame labels are accepted by this API.'])
    return dict(target=target, known=known, target_visible=target_visible,
                target_corridor=target_corridor, weights=weights, audit=audit)
