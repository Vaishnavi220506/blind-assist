"""Fixed global median-range anchoring and whole-range visible query readout.

This reuses a scalar scale anchor, not a new fusion algorithm. The predictor
accepts only dense predicted optical Z, public ray geometry, and 64 object-blind
sensor-interface median radial returns. Evaluator reference pixels never enter
the anchor. Whole-range truth remains visible-point camera-relative truth.
"""
import argparse
import json

import numpy as np

import cnh_rgb_visible_query as V

MIN_VALID_ZONES = 8
MIN_SUPPORT = V.MIN_SUPPORT
MIN_COVERAGE = V.MIN_COVERAGE


def anchor_depth(predicted_optical_z, geometry, sensor_median_radial):
    """Apply one unbounded, untuned global scale from median log zone ratios.

    At least eight finite positive predicted/sensor zone pairs are necessary.
    Otherwise use exactly scale=1 and available=False, retaining the input
    prediction and explicit unavailable diagnostics. No clipped or fallback fit.
    """
    depth, _ = V._depth(predicted_optical_z, geometry)
    sensor = np.asarray(sensor_median_radial, dtype=float)
    if sensor.shape not in ((64,), (8, 8)):
        raise ValueError('sensor interface must provide exactly 64 median radial returns')
    sensor = sensor.reshape(8, 8)
    predicted = V.coarse_returns(depth*geometry['radial_factor'], geometry, .5, weight_power=2)
    median = predicted['zone_return_radial']
    valid = np.isfinite(sensor) & (sensor > 0) & np.isfinite(median) & (median > 0)
    log_ratio = np.full((8, 8), np.nan)
    log_ratio[valid] = np.log(sensor[valid])-np.log(median[valid])
    available = int(valid.sum()) >= MIN_VALID_ZONES
    log_scale = float(np.median(log_ratio[valid])) if available else 0.
    scale = float(np.exp(log_scale))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('unclipped fitted global scale is not representable as positive finite float')
    residual = log_ratio-log_scale
    ratio = np.full((8, 8), np.nan)
    with np.errstate(over='ignore', under='ignore'):
        ratio[valid] = np.exp(log_ratio[valid])
    rr = residual[valid]
    return dict(scaled_depth=depth*scale, scale=scale, log_scale=log_scale,
                available=available, status='AVAILABLE' if available else 'UNAVAILABLE_FEWER_THAN_8_VALID_ZONES',
                minimum_valid_zones=MIN_VALID_ZONES, valid_zone_count=int(valid.sum()),
                valid_zone_mask=valid, zone_log_ratio=log_ratio, zone_ratio=ratio,
                sensor_zone_median_radial=sensor.copy(), predicted_zone_median_radial=median.copy(),
                predicted_zone_valid_count=predicted['zone_valid_count'],
                predicted_zone_missing_count=predicted['zone_missing_count'],
                zone_log_residual_after=residual,
                residual_median=float(np.median(rr)) if len(rr) else None,
                residual_median_abs=float(np.median(np.abs(rr))) if len(rr) else None,
                residual_rms=float(np.sqrt(np.mean(rr*rr))) if len(rr) else None,
                estimator='exp(median(log(sensor zone r^-2 median / predicted zone r^-2 median))); equal weight per valid zone; no clipping/search')


def whole_geometry(geometry):
    """Union only public ray masks, not observations or near/far truth classes."""
    if len(geometry['queries']) == 2:
        assert all(q['z_near'] == .6 and q['z_far'] == 2.1 for q in geometry['queries'])
        return geometry
    assert len(geometry['queries']) == 4
    pairs = []
    for group in ('HEAD', 'BODY'):
        ids = [i for i, q in enumerate(geometry['queries']) if q['group'] == group]
        assert len(ids) == 2
        ids.sort(key=lambda i: geometry['queries'][i]['z_near'])
        near, far = [geometry['queries'][i] for i in ids]
        assert (near['z_near'], near['z_far'], far['z_near'], far['z_far']) == (.6, 1.2, 1.2, 2.1)
        assert near['y_low'] == far['y_low'] and near['y_high'] == far['y_high']
        pairs.append(ids)
    output = dict(geometry)
    output['queries'] = [dict(geometry['queries'][ids[0]], name=f'{group}_0.6-2.1', z_far=2.1)
                         for group, ids in zip(('HEAD', 'BODY'), pairs)]
    output['possible_query_rays'] = np.stack([np.logical_or(*geometry['possible_query_rays'][ids]) for ids in pairs])
    output['possible_query_entry_z'] = np.stack([np.fmin(*geometry['possible_query_entry_z'][ids]) for ids in pairs])
    for name in ('query_fov_clipped', 'query_native_clipped', 'query_nominal_fov_clipped'):
        output[name] = np.array([np.any(geometry[name][ids]) for ids in pairs])
    bounds = geometry['expanded_query_angle_bounds_deg']
    output['expanded_query_angle_bounds_deg'] = np.array([[bounds[ids, 0].min(), bounds[ids, 1].max(),
                                                         bounds[ids, 2].min(), bounds[ids, 3].max()] for ids in pairs])
    # Keep original per-range native-corner arrays explicitly labelled, rather
    # than leaving four-query arrays ambiguously attached to a two-query object.
    for name in ('query_native_corner_uv', 'query_native_corner_homogeneous_scale'):
        if name in output:
            output['component_'+name] = output.pop(name)
    output['component_query_indices'] = pairs
    return output


def query_scores(depth, geometry):
    """Compute two whole-range D16 scores from the complete observed point cloud."""
    geometry = whole_geometry(geometry)
    depth, valid = V._depth(depth, geometry)
    fov_valid = valid & geometry['fov_mask']
    xabs, y = np.abs(depth*geometry['fx']), depth*geometry['fy']
    scores, order16, maxima, counts, contact, expanded = [], [], [], [], [], []
    closest, farthest, contact_z16 = [], [], []
    for q in geometry['queries']:
        take = fov_valid & (depth >= .6) & (depth < 2.1) & (y >= q['y_low']) & (y <= q['y_high'])
        xx = xabs[take]
        zz = depth[take]
        cc = xx < .3
        counts.append(len(xx)); contact.append(int(cc.sum())); expanded.append(int(np.sum(xx <= .4)))
        d16 = float(np.partition(xx, MIN_SUPPORT-1)[MIN_SUPPORT-1]) if len(xx) >= MIN_SUPPORT else np.inf
        order16.append(d16); scores.append(.3-d16)
        maxima.append(float(.3-xx.min()) if len(xx) else -np.inf)
        cz = zz[cc]
        closest.append(float(cz.min()) if len(cz) else np.nan)
        farthest.append(float(cz.max()) if len(cz) else np.nan)
        contact_z16.append(float(np.partition(cz, MIN_SUPPORT-1)[MIN_SUPPORT-1]) if len(cz) >= MIN_SUPPORT else np.nan)
    possible = geometry['possible_query_rays']
    pc = possible.sum(axis=(1, 2)); vc = (possible & valid[None]).sum(axis=(1, 2))
    coverage = np.divide(vc, pc, out=np.zeros(2, dtype=float), where=pc > 0)
    abstain = (coverage < MIN_COVERAGE) | (pc == 0)
    raw_score = np.asarray(scores)
    return dict(query_names=[q['name'] for q in geometry['queries']], score=np.where(abstain, -np.inf, raw_score),
                score_without_coverage=raw_score, abstain=abstain, abs_x_order16=np.asarray(order16),
                max_single_point_intrusion=np.asarray(maxima), return_count=np.asarray(counts),
                contact_count=np.asarray(contact), expanded_count=np.asarray(expanded),
                possible_ray_count=pc, valid_possible_count=vc, missing_possible_count=pc-vc, coverage=coverage,
                fov_pixel_count=int(geometry['fov_mask'].sum()), valid_fov_pixel_count=int(fov_valid.sum()),
                invalid_fov_pixel_count=int((geometry['fov_mask'] & ~valid).sum()), minimum_pixel_support=MIN_SUPPORT,
                closest_contact_optical_z=np.asarray(closest), farthest_contact_optical_z=np.asarray(farthest),
                contact_optical_z_order16=np.asarray(contact_z16),
                longitudinal_diagnostics='min/max contact-point optical Z and 16th-smallest contact Z; descriptive only, no sample selection')


def visible_truth(reference_optical_z, geometry):
    """Whole-range visible truth; D16 acts on the union of all points in [.6,2.1)."""
    geometry = whole_geometry(geometry)
    out = query_scores(reference_optical_z, geometry)
    depth, valid = V._depth(reference_optical_z, geometry)
    front = geometry['possible_query_rays'] & valid[None] & (depth[None] < geometry['possible_query_entry_z'])
    occluded = front.sum(axis=(1, 2))
    fraction = np.divide(occluded, out['valid_possible_count'], out=np.zeros(2), where=out['valid_possible_count'] > 0)
    c, e = out['contact_count'], out['expanded_count']
    unknown = out['abstain'] | ((c > 0) & (c < MIN_SUPPORT)) | ((e > 0) & (e < MIN_SUPPORT))
    contact = ~unknown & (c >= MIN_SUPPORT)
    passing = ~unknown & (c == 0) & (e >= MIN_SUPPORT)
    clear = ~unknown & (e == 0)
    assert np.all(contact.astype(int)+passing+clear+unknown == 1)
    category = np.full(2, 'UNKNOWN', dtype='<U8')
    category[contact] = 'contact'; category[passing] = 'pass'; category[clear] = 'clear'
    bins = np.full(2, '', dtype='<U12')
    p = out['score']
    bins[contact & (p > 0) & (p <= .02)] = 'contact0-2'
    bins[contact & (p > .02) & (p <= .05)] = 'contact2-5'
    bins[contact & (p > .05)] = 'contact>5'
    assert np.all(bins[contact] != '')
    out.update(category=category, contact=contact, **{'pass': passing}, clear=clear, unknown=unknown,
               contact_bin=bins, minimum_coverage=MIN_COVERAGE,
               occluded_possible_count=occluded, foreground_occlusion_fraction=fraction,
               occlusion_fraction_denominator=out['valid_possible_count'].copy(),
               query_fov_clipped=geometry['query_fov_clipped'].copy(),
               query_native_clipped=geometry['query_native_clipped'].copy(),
               query_nominal_fov_clipped=geometry['query_nominal_fov_clipped'].copy(),
               expanded_query_angle_bounds_deg=geometry['expanded_query_angle_bounds_deg'].copy(),
               native_covers_nominal_fov=geometry['native_covers_nominal_fov'],
               truth_scope='FOV-contained visible points in whole [.6,2.1) camera-relative query; D16/16-pixel support; no object identity or hidden/FOV-cropped clearance claim')
    return out


def fixtures():
    geometry = V.ray_geometry((32, 32), np.diag([.4, .4, -1.]))
    reference = np.full((32, 32), 1.)
    sensor = V.coarse_returns(reference*geometry['radial_factor'], geometry, .5)['zone_return_radial']
    anchored = anchor_depth(reference*2.5, geometry, sensor)
    assert anchored['available'] and anchored['valid_zone_count'] == 64
    np.testing.assert_allclose(anchored['scale'], .4, rtol=0, atol=1e-14)
    np.testing.assert_allclose(anchored['scaled_depth'], reference, rtol=0, atol=1e-14)
    np.testing.assert_allclose(anchored['zone_log_residual_after'], 0., rtol=0, atol=1e-14)
    sparse_sensor = sensor.copy(); sparse_sensor.flat[7:] = np.nan
    sparse = anchor_depth(reference*2.5, geometry, sparse_sensor)
    assert not sparse['available'] and sparse['scale'] == 1. and sparse['valid_zone_count'] == 7
    np.testing.assert_array_equal(sparse['scaled_depth'], reference*2.5)
    # Exactly eight contact pixels in each range: neither component is known
    # contact, but the complete 16-pixel whole-range cloud is supported contact.
    toy = V.ray_geometry((1, 32), np.diag([.05, .05, -1.]))
    depth = np.full((1, 32), 3.)
    depth[0, :8] = .8; depth[0, 8:16] = 1.6
    components = V.visible_truth(depth, toy)
    assert components['unknown'][0] and components['unknown'][2]
    full = visible_truth(depth, toy)
    assert full['contact'][0] and full['contact_count'][0] == 16
    assert full['closest_contact_optical_z'][0] == .8 and full['contact_optical_z_order16'][0] == 1.6
    old_specs = [dict(q) for q in toy['queries']]
    whole = whole_geometry(toy)
    assert toy['queries'] == old_specs and len(toy['queries']) == 4 and len(whole['queries']) == 2
    missing = depth.copy(); missing[0, :4] = np.nan
    rejected = visible_truth(missing, toy)
    assert rejected['unknown'][0] and rejected['abstain'][0] and np.isneginf(rejected['score'][0])
    too_few = depth.copy(); too_few[0, 15] = 3.
    assert visible_truth(too_few, toy)['unknown'][0] and not query_scores(too_few, toy)['abstain'][0]
    return dict(global_scale_recovery=True, minimum_eight_zones=True, whole_cloud_not_class_or=True,
                missing_abstention=True, sparse_contact_unknown=True, no_old_geometry_mutation=True,
                scientific_data_accessed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['fixtures'], required=True)
    parser.parse_args()
    print(json.dumps(fixtures()))
