"""Evaluator-only fixed categorical slant-range readout feasibility.

Native AABB first hits and commanded reference pose are oracle labels, never
predictor inputs. This module performs no fitting, inference, I/O or alerts.
UNKNOWN pixels keep a separate mask; their zero scores do not mean clear space.
"""
import math

import numpy as np

from evaluate_mz140_depthor import rasterize_native_bounds
from mz161_dense_labels import make_labels
from mz161_dense_task import ray_query


ARMS = ('native', 'public_exact', 'bin_midpoint', 'bin_mass')
BIN_WIDTH_M = .0625
FINITE_BINS = 128
OVERFLOW_START_M = BIN_WIDTH_M * FINITE_BINS
OVERFLOW_MIDPOINT_M = OVERFLOW_START_M + BIN_WIDTH_M / 2
RISK_THRESHOLD = .5
METHOD = dict(
    schema='MZ167_FIXED_FIRST_HIT_SLANT_CATEGORY_READOUT_V1',
    authority='EVALUATOR_ONLY_NATIVE_AABB_ORACLE_NOT_PREDICTOR_INPUT',
    pose_authority='SOURCE_COMMANDED_REFERENCE',
    rendered_labels=False,
    bin_width_m=BIN_WIDTH_M, finite_bins=FINITE_BINS, grid_phase_m=0.,
    finite_range_m=[0., OVERFLOW_START_M], overflow_start_m=OVERFLOW_START_M,
    overflow_midpoint_m=OVERFLOW_MIDPOINT_M,
    bin_mass='ONE_HOT_REFERENCE_CATEGORY_UNIFORM_WITHIN_FINITE_BIN',
    risk_threshold=RISK_THRESHOLD, threshold_ties='POSITIVE',
    unknown='NO_SAVED_AABB_HIT_IS_UNKNOWN_NOT_CLEAR',
)


def _validated_arrays(slant, enter, leave, valid, known):
    s, lo, hi = (np.asarray(v, dtype=np.float64) for v in (slant, enter, leave))
    v, k = np.asarray(valid, dtype=bool), np.asarray(known, dtype=bool)
    if any(a.shape != s.shape for a in (lo, hi, v, k)):
        raise ValueError('Slant, interval and masks must have identical shapes')
    if np.any(k & (~np.isfinite(s) | (s <= 0))):
        raise ValueError('Known reference slant must be finite and positive')
    if np.any(v & (~np.isfinite(lo) | ~np.isfinite(hi) | (lo < 0) | (hi < lo))):
        raise ValueError('Valid intervals must be finite, nonnegative and ordered')
    # All valid rays, including rays with UNKNOWN first hit, obey this premise.
    # A closed interval ending exactly at 8 m could contain an overflow point.
    if np.any(v & (hi >= OVERFLOW_START_M)):
        raise ValueError('Every valid public leave must be strictly below 8 m')
    return s, lo, hi, v, k


def uniform_bin_mass(slant, enter, leave, valid, known):
    """Return float32 uniform-bin overlap fractions with the input shape.

    The fixed categories are [j/16,(j+1)/16) for j=0..127 and [8,infinity).
    The latter has zero mass in a valid public interval because leave < 8 m.
    Zero-width intervals have zero uniform mass, even at a closed endpoint.
    Reversed or negative intervals marked valid are rejected. Invalid entries
    may contain sentinels and always return zero; UNKNOWN remains caller-owned.
    """
    s, lo, hi, v, k = _validated_arrays(slant, enter, leave, valid, known)
    result = np.zeros(s.shape, dtype=np.float32)
    use = k & v & (s < OVERFLOW_START_M)
    left = np.floor(s[use] / BIN_WIDTH_M) * BIN_WIDTH_M
    overlap = np.maximum(0., np.minimum(hi[use], left + BIN_WIDTH_M)
                         - np.maximum(lo[use], left))
    result[use] = np.clip(overlap / BIN_WIDTH_M, 0., 1.).astype(np.float32)
    return result


def reference_readouts(row, evaluation, yaw):
    """Compare four HxW masks/scores against the unchanged MZ161 reference.

    Returns ``masks`` (bool) and ``scores`` (float32), indexed by ``ARMS``,
    ``reference_slant`` (float64, NaN on UNKNOWN), ``known`` (bool), original
    evaluator-only ``label`` (the make_labels result), and a JSON audit. The
    returned float64 slant map is exactly the array used for all readouts, so
    saved-output replay adds no rounding at category or interval boundaries.
    Public intervals retain ray_query's float32 precision, multiplied by 4
    exactly as in the existing interface. No named actor selects a readout.
    """
    if not math.isfinite(float(yaw)):
        raise ValueError('Finite public yaw required')
    # This validates complete commanded reference pose and original intrinsics.
    labels = make_labels(row, evaluation, yaw)
    reference = rasterize_native_bounds(row, evaluation,
                                         {'integrated_yaw_deg': float(yaw)})
    depth = reference['depth']
    known = (reference['owner'] >= 0) & np.isfinite(depth) & (depth > 0)
    if not np.array_equal(known, labels['known']):
        raise ValueError('Native first-hit reference disagrees with MZ161 known mask')
    intr = row['rgb_intrinsics']
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    a, b = (xx-intr['cx'])/intr['fx'], (intr['cy']-yy)/intr['fy']
    slant = np.full(depth.shape, np.nan, dtype=np.float64)
    slant[known] = depth[known] * np.sqrt(1. + a[known]**2 + b[known]**2)
    _, query = ray_query(row, float(yaw))
    enter, leave, valid = query[0]*4., query[1]*4., query[2] > 0
    _, _, _, valid, _ = _validated_arrays(slant, enter, leave, valid, known)
    exact = known & valid & (slant >= enter) & (slant <= leave)
    midpoint = np.full(depth.shape, np.nan, dtype=np.float64)
    finite = known & (slant < OVERFLOW_START_M)
    overflow = known & ~finite
    midpoint[finite] = (np.floor(slant[finite]/BIN_WIDTH_M)+.5)*BIN_WIDTH_M
    midpoint[overflow] = OVERFLOW_MIDPOINT_M
    mid_mask = known & valid & (midpoint >= enter) & (midpoint <= leave)
    mass = uniform_bin_mass(slant, enter, leave, valid, known)
    masks = dict(native=labels['target'].copy(), public_exact=exact,
                 bin_midpoint=mid_mask, bin_mass=known & valid & (mass >= RISK_THRESHOLD))
    scores = {arm: mask.astype(np.float32) for arm, mask in masks.items()}
    scores['bin_mass'] = mass
    visible_truth = bool(masks['native'].any())
    known_values = slant[known]
    audit = dict(**METHOD, shape=list(depth.shape), reference_known_pixels=int(known.sum()),
        reference_unknown_pixels=int((~known).sum()), reference_overflow_pixels=int(overflow.sum()),
        reference_slant_min_m=float(known_values.min()) if known_values.size else None,
        reference_slant_max_m=float(known_values.max()) if known_values.size else None,
        reference_slant_storage='FLOAT64_NAN_UNKNOWN; IDENTICAL_TO_READOUT_COMPUTATION',
        public_interval_precision='INHERITED_RAY_QUERY_FLOAT32_MULTIPLIED_BY_4',
        public_valid_pixels=int(valid.sum()),
        public_enter_min_m=float(enter[valid].min()) if valid.any() else None,
        public_leave_max_m=float(leave[valid].max()) if valid.any() else None,
        overflow_mass_zero_premise='ALL_VALID_PUBLIC_LEAVE_STRICTLY_BELOW_8M',
        native_volume_truth=labels['audit']['native_aabb_corridor_truth'],
        native_visible_truth=visible_truth,
        native_volume_truth_vs_visible_mismatch=labels['audit']['native_volume_truth_vs_visible_mismatch'],
        exact_public_changed_pixels=int((exact != masks['native']).sum()),
        exact_public_removed_native_pixels=int((masks['native'] & ~exact).sum()),
        exact_public_added_native_negative_pixels=int((known & ~masks['native'] & exact).sum()),
        positive_pixels={arm: int(mask.sum()) for arm, mask in masks.items()},
        any_positive={arm: bool(mask.any()) for arm, mask in masks.items()},
        public_yaw_deg=float(yaw), reference_yaw_deg=float(evaluation['camera']['yaw']),
        reference_minus_public_yaw_deg=float(evaluation['camera']['yaw'])-float(yaw),
        public_pitch_deg=float(row['camera_pitch_deg']),
        reference_pitch_deg=float(evaluation['camera']['pitch']),
        reference_roll_deg=float(evaluation['camera'].get('roll',0.)),
        public_camera_in_body_m=list(row['camera_in_body_m']),
        reference_camera_in_body_m=(reference['camera_origin_world_m']-np.asarray(evaluation['body_origin_m'])).tolist(),
        inherited_raster_pose_source=reference['pose_source'],
        limits=['All saved named native AABBs participate; these are analytic source-commanded reference hits, not rendered depth.',
                'Public exact uses public pitch/yaw/offset and can differ from reference-pose body-point labels.',
                'Finite categories use phase zero and half-open ownership; interval membership is closed and bin mass uses overlap length.',
                'Native volume truth is audit-only and is not supplied to any predictor.',
                'Unknown and unsaved geometry do not become clear-space evidence; visible readout is not whole-system coverage.'])
    return dict(masks=masks, scores=scores, reference_slant=slant,
                known=known.copy(), label=labels, audit=audit)
