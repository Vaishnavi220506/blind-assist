"""Pure geometry metrics and fixed selection for one paired MZ141 training round.

No file/model access. References are training labels or evaluator-only data,
never inference inputs. All depth errors use camera-axis meters.
"""
import math

import numpy as np

ARMS = ('frozen', 'pixel', 'surface')
THIN = ('near_rod_farwall', 'shallow_boundary_stress')
LARGE = ('substantial_body', 'suspended_head')
FAMILIES = THIN + LARGE
CRITERIA = dict(
    thin_mae_ratio_max=.5, thin_mae_m_max=.5, thin_over_1m_ratio_max=.5,
    thin_half_column_coverage_min=.25, ring_closer_error_fraction_allowance=.02,
    non_target_over_near_fraction_allowance=.02, large_mae_m_allowance=.12,
    large_half_column_coverage_allowance=.10,
    surface_extra_thin_mean_mae_ratio_max=.90,
    selection='PIXEL_IF_ELIGIBLE_UNLESS_SURFACE_HAS_FIXED_EXTRA_GAIN_WITHOUT_SPILL_REGRESSION',
    missing_metrics='NOT_EVALUABLE_FAILS_ADMISSION',
    invalid_predictions='ANY_INVALID_PREDICTION_FAILS_ADMISSION',
)


def extra_metrics(depth, reference):
    """Supplement evaluate_mz140_depthor.evaluate_frame; preserve UNKNOWN.

    Over-near counts known non-target pixels with gt-pred > .12m, including
    errors outside the local silhouette ring. Invalid outputs are counted
    separately by the original evaluator, never treated as correct geometry.
    """
    depth = np.asarray(depth, dtype=float)
    gt = np.asarray(reference['depth'], dtype=float)
    target = np.asarray(reference['target_visible'], dtype=bool)
    if depth.shape != gt.shape or target.shape != gt.shape:
        raise ValueError('Prediction/reference/target shapes differ')
    known = np.isfinite(gt) & (gt > 0)
    valid = np.isfinite(depth) & (depth > 0)
    target_known = target & known
    target_comparable = target_known & valid
    non_target = (~target) & known
    non_target_comparable = non_target & valid
    target_errors = np.abs(depth[target_comparable]-gt[target_comparable])
    non_target_errors = gt[non_target_comparable]-depth[non_target_comparable]
    return dict(
        target_gt_median=float(np.median(gt[target_known])) if target_known.any() else None,
        target_pred_median=float(np.median(depth[target_comparable])) if target_comparable.any() else None,
        target_over_1m_pixels=int((target_errors > 1.).sum()),
        non_target_known_pixels=int(non_target.sum()),
        non_target_comparable_pixels=int(non_target_comparable.sum()),
        non_target_over_near_pixels=int((non_target_errors > .12).sum()),
        non_target_mae=float(np.abs(non_target_errors).mean()) if len(non_target_errors) else None,
    )


def _ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def _mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return sum(values)/len(values) if values else None


def _pooled(records):
    targets = [r['target'] for r in records]
    rings = [r['silhouette_neighborhood'] for r in records]
    target_pixels = sum(t['visible_pixels'] for t in targets)
    target_comparable = sum(t['comparable_pixels'] for t in targets)
    agreeing = sum(t['range_agreeing_pixels'] for t in targets)
    error_count = sum(t['absolute_error_m']['count'] for t in targets)
    error_sum = sum(t['absolute_error_m']['count']*t['absolute_error_m']['mean']
                    for t in targets if t['absolute_error_m']['count'])
    columns = sum(t['transverse_columns']['visible'] for t in targets)
    recovered_columns = sum(t['transverse_columns']['with_at_least_half_pixels_agreeing'] for t in targets)
    ring_known = sum(r['referenced_pixels'] for r in rings)
    ring_band = sum(r['target_depth_band_spurious_pixels'] for r in rings)
    ring_closer = sum(r['closer_or_target_like_error_pixels'] for r in rings)
    non_target_known = sum(r['non_target_known_pixels'] for r in records)
    # The additional comparable count avoids weighting a finite-only MAE by
    # unknown/invalid predictions. Older hand-built records may omit it.
    non_target_comparable = sum(r.get('non_target_comparable_pixels', r['non_target_known_pixels']) for r in records)
    non_target_error_sum = sum(r.get('non_target_comparable_pixels', r['non_target_known_pixels'])*r['non_target_mae']
                               for r in records if r['non_target_mae'] is not None)
    non_target_over_near = sum(r['non_target_over_near_pixels'] for r in records)
    over_1m = sum(r['target_over_1m_pixels'] for r in records)
    pixels = sum(r['pixels'] for r in records)
    valid_pixels = sum(r['valid_prediction_pixels'] for r in records)
    return dict(
        frames=len(records), target_pixels=target_pixels, target_comparable_pixels=target_comparable,
        target_invalid_prediction_pixels=target_pixels-target_comparable,
        target_agreeing_pixels=agreeing, target_pixel_coverage=_ratio(agreeing, target_pixels),
        target_absolute_error_sum_m=error_sum, target_absolute_error_count=error_count,
        target_mae_m=_ratio(error_sum, error_count),
        target_per_frame_p90_mean_m=_mean([t['absolute_error_m']['p90'] for t in targets]),
        target_per_frame_error_median_mean_m=_mean([t['absolute_error_m']['median'] for t in targets]),
        target_per_frame_gt_median_mean_m=_mean([r['target_gt_median'] for r in records]),
        target_per_frame_pred_median_mean_m=_mean([r['target_pred_median'] for r in records]),
        target_over_1m_pixels=over_1m, target_over_1m_fraction=_ratio(over_1m, target_pixels),
        target_columns=columns, recovered_half_columns=recovered_columns,
        column_coverage=_ratio(recovered_columns, columns),
        ring_pixels=sum(r['pixels'] for r in rings), ring_known_pixels=ring_known,
        ring_unknown_pixels=sum(r['unknown_reference_pixels'] for r in rings),
        ring_invalid_prediction_pixels=sum(r['invalid_prediction_pixels'] for r in rings),
        ring_target_band_spurious_pixels=ring_band, ring_target_band_fraction=_ratio(ring_band, ring_known),
        ring_closer_error_pixels=ring_closer, ring_closer_error_fraction=_ratio(ring_closer, ring_known),
        local_target_surface_precision=_ratio(agreeing, agreeing+ring_band),
        known_non_target_pixels=non_target_known, non_target_comparable_pixels=non_target_comparable,
        non_target_invalid_prediction_pixels=non_target_known-non_target_comparable,
        non_target_absolute_error_sum_m=non_target_error_sum,
        non_target_mae_m=_ratio(non_target_error_sum, non_target_comparable),
        known_non_target_over_near_pixels=non_target_over_near,
        known_non_target_overnear_fraction=_ratio(non_target_over_near, non_target_known),
        prediction_pixels=pixels, invalid_prediction_pixels=pixels-valid_pixels,
        native_reference_pixels=sum(r['native_bounds_reference_pixels'] for r in records),
        unknown_reference_pixels=sum(r['unknown_reference_pixels'] for r in records),
    )


def _le(value, bound):
    return value is not None and bound is not None and math.isfinite(value) and math.isfinite(bound) and value <= bound+1e-12


def _ge(value, bound):
    return value is not None and bound is not None and math.isfinite(value) and math.isfinite(bound) and value+1e-12 >= bound


def _offset(value, offset):
    return None if value is None else value+offset


def _scaled(value, scale):
    return None if value is None else value*scale


def _admission(candidate, baseline):
    complete = set(candidate) == set(FAMILIES) == set(baseline)
    checks = {}
    for family in FAMILIES:
        if family not in candidate or family not in baseline:
            checks[family] = dict(available=False, passed=False)
            continue
        c, b = candidate[family], baseline[family]
        gate = dict(available=True,
            all_predictions_valid=c['invalid_prediction_pixels'] == 0,
            ring_closer_noninferior=_le(c['ring_closer_error_fraction'], _offset(b['ring_closer_error_fraction'], .02)),
            whole_non_target_overnear_noninferior=_le(c['known_non_target_overnear_fraction'], _offset(b['known_non_target_overnear_fraction'], .02)))
        if family in THIN:
            gate.update(
                target_mae_halved=_le(c['target_mae_m'], _scaled(b['target_mae_m'], .5)),
                target_mae_at_most_half_meter=_le(c['target_mae_m'], .5),
                over_1m_fraction_halved=_le(c['target_over_1m_fraction'], _scaled(b['target_over_1m_fraction'], .5)),
                half_column_coverage_at_least_quarter=_ge(c['column_coverage'], .25))
        else:
            gate.update(target_mae_noninferior=_le(c['target_mae_m'], _offset(b['target_mae_m'], .12)),
                        half_column_coverage_noninferior=_ge(c['column_coverage'], _offset(b['column_coverage'], -.10)))
        gate['passed'] = all(gate.values())
        checks[family] = gate
    return dict(complete_families=complete, families=checks,
                eligible=complete and all(g['passed'] for g in checks.values()))


def _surface_extra_gain(surface, pixel):
    complete = set(surface) == set(pixel) == set(FAMILIES)
    if not complete:
        return dict(complete_families=False, passed=False)
    surface_thin = [surface[f]['target_mae_m'] for f in THIN]
    pixel_thin = [pixel[f]['target_mae_m'] for f in THIN]
    surface_mean = sum(surface_thin)/2 if all(v is not None for v in surface_thin) else None
    pixel_mean = sum(pixel_thin)/2 if all(v is not None for v in pixel_thin) else None
    families = {}
    for family in FAMILIES:
        c, b = surface[family], pixel[family]
        checks = dict(
            ring_closer_noninferior=_le(c['ring_closer_error_fraction'], _offset(b['ring_closer_error_fraction'], .02)),
            whole_non_target_overnear_noninferior=_le(c['known_non_target_overnear_fraction'], _offset(b['known_non_target_overnear_fraction'], .02)))
        if family in LARGE:
            checks['half_column_coverage_noninferior'] = _ge(c['column_coverage'], _offset(b['column_coverage'], -.10))
        checks['passed'] = all(checks.values())
        families[family] = checks
    improved = _le(surface_mean, _scaled(pixel_mean, .90))
    return dict(complete_families=True, surface_thin_mean_mae_m=surface_mean,
                pixel_thin_mean_mae_m=pixel_mean, thin_mae_extra_ten_percent=improved,
                families=families, passed=improved and all(c['passed'] for c in families.values()))


def summarize(cases):
    """Aggregate paired records and select using only fixed heldout criteria.

    The caller retains full frame records separately; summaries preserve raw
    counts/sums and explicit denominators. p90 and median summaries are means
    of frame statistics, not reconstructed pooled quantiles.
    """
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    for case in cases:
        if case['partition'] not in ('fit', 'heldout') or set(case['arms']) != set(ARMS):
            raise ValueError('Expected fit/heldout cases with all three paired arms')
    partitions = {}
    for partition in ('fit', 'heldout'):
        subset = [c for c in cases if c['partition'] == partition]
        arms = {}
        for arm in ARMS:
            families = {f: _pooled([c['arms'][arm] for c in subset if c['family'] == f])
                        for f in sorted({c['family'] for c in subset})}
            scenes = {s: dict(families=sorted({c['family'] for c in subset if c['scene_group'] == s}),
                              **_pooled([c['arms'][arm] for c in subset if c['scene_group'] == s]))
                      for s in sorted({c['scene_group'] for c in subset})}
            arms[arm] = dict(families=families, scenes=scenes, overall=_pooled([c['arms'][arm] for c in subset]))
        partitions[partition] = dict(frames=len(subset), ids=[c['id'] for c in subset], arms=arms)
    heldout = partitions['heldout']['arms']
    gates = {a: _admission(heldout[a]['families'], heldout['frozen']['families']) for a in ('pixel', 'surface')}
    extra = _surface_extra_gain(heldout['surface']['families'], heldout['pixel']['families'])
    selected = None
    if gates['pixel']['eligible']:
        selected = 'surface' if gates['surface']['eligible'] and extra['passed'] else 'pixel'
    elif gates['surface']['eligible']:
        selected = 'surface'
    return dict(
        frames=len(cases), partitions=partitions, criteria=CRITERIA, gates=gates,
        surface_extra_gain=extra, selected_arm=selected,
        geometry_admitted=selected is not None,
        decision=('HELDOUT_GEOMETRY_ADMITTED_'+selected.upper()+'_ALERT_COMPARISON_NOT_YET_RUN'
                  if selected else 'STOP_PAIRED_TRAINING_RECIPE_NO_HELDOUT_GEOMETRY_ADMISSION_KEEP_MZ129'),
        authority='CONSUMED_TRAIN_SCENE_HELDOUT_DEVELOPMENT_NOT_FRESH_CONFIRMATION',
        dev_scored=False, original_test_scored=False,
        quantile_semantics='TARGET_P90_AND_MEDIANS_ARE_UNWEIGHTED_MEANS_OF_FRAME_STATISTICS_NOT_POOLED_QUANTILES',
        limits='Known collision-AABB reference only; unknown is not clear. Geometry admission is not an alert gain. Full per-frame records are retained by the caller.',
    )
