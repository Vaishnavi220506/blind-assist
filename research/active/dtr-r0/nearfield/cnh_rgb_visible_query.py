"""Observation-only camera-relative visible-surface query readouts.

Inputs are an optical-Z image and its public camera ray matrix. No pose, GT
object identity, floor identity or target selection enters the predictor. The
coarse simulator interface consumes radial reference depth only to produce
object-blind zone returns; downstream methods receive its expanded optical-Z.
Truth describes visible points with a 16-pixel support requirement. It is not
complete physical-surface, solid-occupancy, body-frame or wearer collision truth.
"""
import argparse
import json

import numpy as np

HALF_FOV_RAD = np.deg2rad(22.5)
MIN_SUPPORT = 16
MIN_COVERAGE = .95
QUERY_SPECS = tuple(dict(name=f'{group}_{near}-{far}', group=group,
                        y_low=yl, y_high=yh, z_near=near, z_far=far)
                    for near, far in ((.6, 1.2), (1.2, 2.1))
                    for group, yl, yh in (('HEAD', -.2, .42), ('BODY', .42, .9)))


def _possible_rays(fx, fy, fov, query):
    """Existence of Z in a fixed y/z query and |X|<=.4, without reading depth."""
    near, far = query['z_near'], query['z_far']
    lower = np.full(fx.shape, near)
    upper = np.full(fx.shape, far)
    nonzero_y = fy != 0
    a = np.divide(query['y_low'], fy, out=np.full(fy.shape, -np.inf), where=nonzero_y)
    b = np.divide(query['y_high'], fy, out=np.full(fy.shape, np.inf), where=nonzero_y)
    lower = np.maximum(lower, np.minimum(a, b))
    upper = np.minimum(upper, np.maximum(a, b))
    lateral_limit = np.divide(.4, np.abs(fx), out=np.full(fx.shape, np.inf), where=fx != 0)
    upper = np.minimum(upper, lateral_limit)
    y_possible = nonzero_y | (query['y_low'] <= 0 <= query['y_high'])
    # The far z boundary is open; lateral and y boundaries are closed.
    possible = fov & y_possible & (lower <= upper) & (lower < far) & (upper >= near)
    return possible, np.where(possible, lower, np.nan)


def ray_geometry(shape, M):
    """Construct pixel-centre rays, an equiangular 8x8 grid and query masks.

    Query order is HEAD near, BODY near, HEAD far, BODY far. Native image shape
    (768,1024) is used as given, with no resize; other shapes support fixtures.
    The public matrix convention has forward camera -Z and upward camera +Y.
    """
    if len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape):
        raise ValueError('shape must contain positive integer height and width')
    h, w = map(int, shape)
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3) or not np.isfinite(M).all():
        raise ValueError('finite 3x3 M_cam_from_uv required')
    inverse = np.linalg.inv(M)

    def native_projection(camera_points):
        homogeneous = np.asarray(camera_points, float)@inverse.T
        forward = np.isfinite(homogeneous).all(axis=-1) & (homogeneous[..., 2] > 0)
        coordinates = np.divide(homogeneous[..., :2], homogeneous[..., 2, None],
                                out=np.full(homogeneous.shape[:-1]+(2,), np.nan),
                                where=forward[..., None])
        inside = forward & np.all(np.abs(coordinates) <= 1., axis=-1)
        return coordinates, homogeneous[..., 2], inside
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    uv = np.stack((2*(xx+.5)/w-1, 1-2*(yy+.5)/h, np.ones((h, w))), axis=-1)
    rays = uv@M.T
    denominator = -rays[..., 2]
    ray_valid = np.isfinite(rays).all(-1) & (denominator > 0)
    fx = np.divide(rays[..., 0], denominator, out=np.full((h, w), np.nan), where=ray_valid)
    fy = np.divide(-rays[..., 1], denominator, out=np.full((h, w), np.nan), where=ray_valid)
    radial_factor = np.sqrt(1+fx*fx+fy*fy)
    ax, ay = np.arctan(fx), np.arctan(fy)
    fov = ray_valid & (np.abs(ax) <= HALF_FOV_RAD) & (np.abs(ay) <= HALF_FOV_RAD)
    zone_id = np.full((h, w), -1, dtype=np.int16)
    zx = np.clip(np.floor((ax[fov]+HALF_FOV_RAD)/(2*HALF_FOV_RAD)*8), 0, 7).astype(int)
    zy = np.clip(np.floor((ay[fov]+HALF_FOV_RAD)/(2*HALF_FOV_RAD)*8), 0, 7).astype(int)
    zone_id[fov] = 8*zy+zx
    queries = [dict(q) for q in QUERY_SPECS]
    ray_intervals = [_possible_rays(fx, fy, fov, q) for q in queries]
    possible = np.stack([pair[0] for pair in ray_intervals])
    entry = np.stack([pair[1] for pair in ray_intervals])
    extents, native_corners = [], []
    for q in queries:
        xa = [np.arctan(x/z) for x in (-.4, .4) for z in (q['z_near'], q['z_far'])]
        ya = [np.arctan(y/z) for y in (q['y_low'], q['y_high']) for z in (q['z_near'], q['z_far'])]
        extents.append([min(xa), max(xa), min(ya), max(ya)])
        native_corners.append([[x, -y, -z] for x in (-.4, .4)
                               for y in (q['y_low'], q['y_high'])
                               for z in (q['z_near'], q['z_far'])])
    extents = np.asarray(extents)
    native_uv, native_scale, native_inside = native_projection(native_corners)
    native_clipped = ~np.all(native_inside, axis=1)
    nominal_rays = [[np.tan(ax), -np.tan(ay), -1.]
                    for ax in (-HALF_FOV_RAD, HALF_FOV_RAD)
                    for ay in (-HALF_FOV_RAD, HALF_FOV_RAD)]
    nominal_uv, nominal_scale, nominal_inside = native_projection(nominal_rays)
    ideal_clipped = np.any(np.abs(extents) > HALF_FOV_RAD, axis=1)
    return dict(shape=(h, w), M_cam_from_uv=M.copy(), fx=fx, fy=fy,
                radial_factor=radial_factor, fov_mask=fov, zone_id=zone_id,
                possible_query_rays=possible, possible_query_entry_z=entry, queries=queries,
                query_fov_clipped=ideal_clipped | native_clipped,
                query_nominal_fov_clipped=ideal_clipped, query_native_clipped=native_clipped,
                query_native_corner_uv=native_uv, query_native_corner_homogeneous_scale=native_scale,
                native_covers_nominal_fov=bool(np.all(nominal_inside)),
                nominal_fov_corner_native_uv=nominal_uv,
                nominal_fov_corner_homogeneous_scale=nominal_scale,
                expanded_query_angle_bounds_deg=np.rad2deg(extents),
                angle_bound_order=('x_min', 'x_max', 'y_min', 'y_max'),
                zone_ray_count=np.bincount(zone_id[fov], minlength=64).reshape(8, 8),
                coordinate_frame='fixed camera-relative x right, y down, optical Z forward',
                native_shape=(768, 1024), resampling='none')


def _depth(depth, geometry):
    depth = np.asarray(depth, dtype=float)
    if depth.shape != geometry['shape']:
        raise ValueError('depth shape must equal ray geometry shape')
    return depth, np.isfinite(depth) & (depth > 0)


def query_scores(depth, geometry):
    """Four scores .3 minus the 16th-smallest absolute X in query y/z.

    Every valid observed pixel in the 45x45 degree FOV and query y/z contributes,
    including points outside the lateral corridor. Fewer than 16 returns gives
    -infinity. Coverage below95% or zero possible rays explicitly abstains with
    score=-infinity. Sufficient coverage with fewer than16 query returns is a
    legitimate absence of a supported nearby surface, not missing-data abstention.
    """
    depth, valid = _depth(depth, geometry)
    fov_valid = valid & geometry['fov_mask']
    xabs = np.abs(depth*geometry['fx'])
    y = depth*geometry['fy']
    scores, rank16, maxima, counts, contact, expanded = [], [], [], [], [], []
    for q in geometry['queries']:
        take = fov_valid & (depth >= q['z_near']) & (depth < q['z_far']) & (y >= q['y_low']) & (y <= q['y_high'])
        xx = xabs[take]
        counts.append(len(xx))
        contact.append(int(np.sum(xx < .3)))
        expanded.append(int(np.sum(xx <= .4)))
        order16 = float(np.partition(xx, MIN_SUPPORT-1)[MIN_SUPPORT-1]) if len(xx) >= MIN_SUPPORT else np.inf
        rank16.append(order16)
        scores.append(.3-order16)
        maxima.append(float(.3-xx.min()) if len(xx) else -np.inf)
    possible = geometry['possible_query_rays']
    pcount = possible.sum(axis=(1, 2))
    vcount = (possible & valid[None]).sum(axis=(1, 2))
    coverage = np.divide(vcount, pcount, out=np.zeros(4, dtype=float), where=pcount > 0)
    abstain = (coverage < MIN_COVERAGE) | (pcount == 0)
    raw_score = np.asarray(scores)
    score = np.where(abstain, -np.inf, raw_score)
    return dict(query_names=[q['name'] for q in geometry['queries']], score=score,
                score_without_coverage=raw_score, abstain=abstain,
                abs_x_order16=np.asarray(rank16), return_count=np.asarray(counts),
                max_single_point_intrusion=np.asarray(maxima),
                contact_count=np.asarray(contact), expanded_count=np.asarray(expanded),
                possible_ray_count=pcount, valid_possible_count=vcount,
                missing_possible_count=pcount-vcount, coverage=coverage,
                fov_pixel_count=int(geometry['fov_mask'].sum()),
                valid_fov_pixel_count=int(fov_valid.sum()),
                invalid_fov_pixel_count=int((geometry['fov_mask'] & ~valid).sum()),
                minimum_pixel_support=MIN_SUPPORT)


def coarse_returns(radial, geometry, quantile, weight_power=2):
    """Simulate one object-blind weighted radial quantile per angular zone.

    All valid radial pixels in a zone contribute, without target/query filtering.
    The weighted empirical inverse CDF chooses an observed radial distance.
    A zone return is expanded over its whole angular support as Z=r/factor,
    including pixels whose reference depth was missing. Downstream predictors
    receive only expanded_optical_z, not the original per-pixel radial input.
    """
    if not np.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError('quantile must be in (0,1]')
    if not np.isfinite(weight_power) or weight_power < 0:
        raise ValueError('weight_power must be finite and nonnegative')
    radial, valid = _depth(radial, geometry)
    zone = geometry['zone_id']
    values = np.full(64, np.nan)
    counts = np.zeros(64, dtype=int)
    weight_sums = np.zeros(64)
    weight_scales = np.full(64, np.nan)
    for k in range(64):
        rr = np.sort(radial[(zone == k) & valid])
        counts[k] = len(rr)
        if not len(rr):
            continue
        # Relative weights preserve r^-power quantiles without overflow at tiny r.
        with np.errstate(over='ignore', under='ignore'):
            weights = (rr/rr[0])**(-float(weight_power))
        cumulative = np.cumsum(weights)
        index = min(int(np.searchsorted(cumulative, float(quantile)*cumulative[-1], side='left')), len(rr)-1)
        values[k] = rr[index]
        weight_sums[k] = cumulative[-1]
        weight_scales[k] = rr[0]
    expanded = np.full(geometry['shape'], np.nan)
    fov = geometry['fov_mask']
    expanded[fov] = values[zone[fov]]/geometry['radial_factor'][fov]
    possible = geometry['zone_ray_count']
    return dict(quantile=float(quantile), weight_power=float(weight_power),
                zone_return_radial=values.reshape(8, 8), zone_valid_count=counts.reshape(8, 8),
                zone_ray_count=possible.copy(), zone_missing_count=possible-counts.reshape(8, 8),
                zone_weight_sum_scaled=weight_sums.reshape(8, 8),
                zone_weight_scale_radial=weight_scales.reshape(8, 8),
                expanded_optical_z=expanded, return_zone_count=int(np.isfinite(values).sum()),
                empty_zone_count=int(np.sum(~np.isfinite(values))),
                expansion='constant-range-on-zone reconstruction prior; entire angular zone Z=zone radial return / pixel radial factor',
                support_interpretation='expanded pixel counts measure unified reconstruction support, not independent observed depth returns')


def visible_truth(reference_optical_z, geometry):
    """Visible-point truth with support and missing-depth abstention.

    UNKNOWN: coverage below95%, no possible rays, or 1..15 points in contact or
    expanded query. Contact requires >=16 |X|<.3 points; otherwise pass requires
    >=16 |X|<=.4 points, and clear requires no expanded points. This is visible
    point support, not physical-area weighting, complete solids or wearer truth.
    """
    out = query_scores(reference_optical_z, geometry)
    depth, valid = _depth(reference_optical_z, geometry)
    in_front = (geometry['possible_query_rays'] & valid[None] &
                (depth[None] < geometry['possible_query_entry_z']))
    occluded_count = in_front.sum(axis=(1, 2))
    occluded_fraction = np.divide(occluded_count, out['valid_possible_count'],
                                 out=np.zeros(4, dtype=float), where=out['valid_possible_count'] > 0)
    c, e = out['contact_count'], out['expanded_count']
    unknown = ((out['coverage'] < MIN_COVERAGE) | (out['possible_ray_count'] == 0) |
               ((c > 0) & (c < MIN_SUPPORT)) | ((e > 0) & (e < MIN_SUPPORT)))
    contact = ~unknown & (c >= MIN_SUPPORT)
    passing = ~unknown & (c == 0) & (e >= MIN_SUPPORT)
    clear = ~unknown & (e == 0)
    assert np.all(contact.astype(int)+passing+clear+unknown == 1)
    category = np.full(4, 'UNKNOWN', dtype='<U8')
    category[contact] = 'contact'; category[passing] = 'pass'; category[clear] = 'clear'
    bins = np.full(4, '', dtype='<U12')
    penetration = out['score']
    bins[contact & (penetration > 0) & (penetration <= .02)] = 'contact0-2'
    bins[contact & (penetration > .02) & (penetration <= .05)] = 'contact2-5'
    bins[contact & (penetration > .05)] = 'contact>5'
    assert np.all(bins[contact] != '')
    out.update(category=category, contact=contact, **{'pass': passing}, clear=clear, unknown=unknown,
               contact_bin=bins, minimum_coverage=MIN_COVERAGE,
               occluded_possible_count=occluded_count, foreground_occlusion_fraction=occluded_fraction,
               occlusion_fraction_denominator=out['valid_possible_count'].copy(),
               query_fov_clipped=geometry['query_fov_clipped'].copy(),
               query_native_clipped=geometry['query_native_clipped'].copy(),
               query_nominal_fov_clipped=geometry['query_nominal_fov_clipped'].copy(),
               native_covers_nominal_fov=geometry['native_covers_nominal_fov'],
               expanded_query_angle_bounds_deg=geometry['expanded_query_angle_bounds_deg'].copy(),
               truth_scope='FOV-contained visible depth points only in fixed camera-relative query; 16-pixel support, no object identity; not full surfaces/solids or wearer collision; visible clear does not certify hidden or FOV-cropped regions clear')
    return out


def fixtures():
    # Downward optical geometry and near/far boundary membership.
    g = ray_geometry((32, 32), np.diag([.4, .4, -1.]))
    assert g['fx'][0, 0] < 0 and g['fy'][0, 0] < 0 and g['fy'][-1, 0] > 0
    assert g['fov_mask'].all() and g['zone_ray_count'].sum() == 1024
    d = np.full((32, 32), .8)
    scores = query_scores(d, g)
    assert scores['return_count'][0] > 16 and scores['return_count'][1:].sum() == 0
    assert visible_truth(d, g)['contact'][0]
    far = query_scores(np.full((32, 32), 1.2), g)
    assert far['return_count'][:2].sum() == 0 and far['return_count'][2:].sum() > 0
    # Coverage counts only public rays capable of intersecting the expanded query.
    missing = d.copy(); missing[g['possible_query_rays'][0]] = np.nan
    assert visible_truth(missing, g)['unknown'][0]
    assert query_scores(missing, g)['missing_possible_count'][0] == g['possible_query_rays'][0].sum()
    assert query_scores(missing, g)['abstain'][0] and np.isneginf(query_scores(missing, g)['score'][0])
    occlusion = visible_truth(np.full((32, 32), .1), g)
    assert occlusion['foreground_occlusion_fraction'][0] == 1.
    assert occlusion['clear'][0] and not occlusion['abstain'][0]
    assert g['query_fov_clipped'].shape == (4,) and g['query_fov_clipped'][0]
    # A native 22.6-degree-wide camera cannot observe the nominal 45-degree FOV.
    narrow = ray_geometry((4, 4), np.diag([.2, .2, -1.]))
    assert not narrow['native_covers_nominal_fov'] and narrow['query_native_clipped'].all()
    assert np.all(narrow['nominal_fov_corner_homogeneous_scale'] > 0)
    assert np.max(np.abs(narrow['nominal_fov_corner_native_uv'])) > 2.
    # A single zone with two radial populations shows r^-2 median selection.
    radial = np.full((32, 32), np.nan)
    k = int(g['zone_id'][16, 16]); ids = np.flatnonzero(g['zone_id'] == k)
    assert len(ids) >= 4
    radial.flat[ids[:2]] = 1.; radial.flat[ids[2:4]] = 2.
    coarse = coarse_returns(radial, g, .5)
    assert coarse['zone_return_radial'].flat[k] == 1.
    assert coarse['zone_valid_count'].flat[k] == 4
    expanded = coarse['expanded_optical_z']
    assert np.isfinite(expanded[g['zone_id'] == k]).all()
    np.testing.assert_allclose(expanded[g['zone_id'] == k]*g['radial_factor'][g['zone_id'] == k], 1.)
    # Build a small explicit ray geometry to isolate 15/16 point semantics.
    fake = ray_geometry((1, 32), np.diag([.4, .4, -1.]))
    fake['fx'][:] = .35/.8; fake['fy'][:] = 0
    fake['possible_query_rays'][:] = True
    dd = np.full((1, 32), .8)
    fake['fx'][0, :15] = .29/.8
    assert visible_truth(dd, fake)['unknown'][0]
    fake['fx'][0, 15] = .29/.8
    t = visible_truth(dd, fake)
    assert t['contact'][0] and t['contact_bin'][0] == 'contact0-2'
    fake['fx'][:] = .4/.8
    assert visible_truth(dd, fake)['pass'][0]
    fake['fx'][:] = .401/.8
    assert visible_truth(dd, fake)['clear'][0]
    return dict(pixel_centres_and_axes=True, halfopen_range=True, possible_ray_coverage=True,
                weighted_zone_return=True, entire_zone_expansion=True, support15_vs16=True,
                contact_pass_clear_boundaries=True, missing_abstention=True,
                expanded_query_entry_occlusion=True, fov_clipping_recorded=True,
                native_projection_footprint=True, scientific_data_accessed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['fixtures'], required=True)
    parser.parse_args()
    print(json.dumps(fixtures()))
