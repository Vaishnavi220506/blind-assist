"""Public lateral installation offsets; visible-point truth, not new viewpoints.

Existing visible categories are preserved at offset0. A separate conservative
full-volume category prevents cropped/occluded visible clear from being called
clear for the complete body query. No reference depth enters geometry.
"""
import argparse
import numpy as np
import cnh_rgb_range_anchor as A

V = A.V
OFFSETS = (-.2, -.1, 0., .1, .2)


def possible_rays(fx, fy, fov, query, offset):
    """Intersect camera rays with closed x/y slabs and half-open z interval."""
    lower = np.full(fx.shape, query['z_near'], dtype=float)
    upper = np.full(fx.shape, query['z_far'], dtype=float)
    feasible = np.asarray(fov, bool).copy()
    for factor, lo, hi in ((fx, offset-.4, offset+.4),
                           (fy, query['y_low'], query['y_high'])):
        nonzero = factor != 0
        a = np.divide(lo, factor, out=np.full(factor.shape, -np.inf), where=nonzero)
        b = np.divide(hi, factor, out=np.full(factor.shape, np.inf), where=nonzero)
        lower = np.maximum(lower, np.minimum(a, b))
        upper = np.minimum(upper, np.maximum(a, b))
        feasible &= np.isfinite(factor) & (nonzero | (lo <= 0 <= hi))
    feasible &= (lower <= upper) & (lower < query['z_far']) & (upper >= query['z_near'])
    return feasible, np.where(feasible, lower, np.nan), np.where(feasible, upper, np.nan)


def geometry(base, offset):
    """Return two whole-range public queries; base is V.ray_geometry output."""
    offset = float(offset)
    if offset not in OFFSETS:
        raise ValueError('Only the five frozen lateral offsets are supported')
    out = dict(A.whole_geometry(base))
    out.update(body_center_offset_m=offset, body_x_definition='X_body=X_camera-offset',
               installation_scope='Same camera and visible image; translated query only, no viewpoint change')
    if offset == 0:
        # Exact legacy masks/entry values, including near/far union convention.
        upper = [possible_rays(base['fx'], base['fy'], base['fov_mask'], q, 0)[2]
                 for q in out['queries']]
        out['possible_query_exit_z'] = np.stack(upper)
        points = np.array([[[x, -y, -z] for x in (-.4, .4)
                     for y in (q['y_low'], q['y_high']) for z in (q['z_near'], q['z_far'])]
                     for q in out['queries']])
        homogeneous = points@np.linalg.inv(base['M_cam_from_uv']).T
        out['query_native_corner_homogeneous_scale'] = homogeneous[..., 2]
        out['query_native_corner_uv'] = np.divide(homogeneous[..., :2], homogeneous[..., 2, None],
            out=np.full(homogeneous.shape[:-1]+(2,), np.nan), where=homogeneous[..., 2, None] != 0)
        return out
    intervals = [possible_rays(base['fx'], base['fy'], base['fov_mask'], q, offset)
                 for q in out['queries']]
    out['possible_query_rays'] = np.stack([x[0] for x in intervals])
    out['possible_query_entry_z'] = np.stack([x[1] for x in intervals])
    out['possible_query_exit_z'] = np.stack([x[2] for x in intervals])
    inverse = np.linalg.inv(base['M_cam_from_uv'])
    def corners(queries):
        coords = np.array([[[x, -y, -z] for x in (offset-.4, offset+.4)
                    for y in (q['y_low'], q['y_high'])
                    for z in (q['z_near'], q['z_far'])] for q in queries])
        homogeneous = coords@inverse.T
        scale = homogeneous[..., 2]
        uv = np.divide(homogeneous[..., :2], scale[..., None],
                       out=np.full(homogeneous.shape[:-1]+(2,), np.nan), where=scale[..., None] != 0)
        inside = np.isfinite(homogeneous).all(-1) & (scale > 0) & (np.abs(uv) <= 1).all(-1)
        return uv, scale, ~inside.all(1)
    uv, scale, native = corners(out['queries'])
    bounds = []
    for q in out['queries']:
        xa = [np.arctan(x/z) for x in (offset-.4, offset+.4) for z in (q['z_near'], q['z_far'])]
        ya = [np.arctan(y/z) for y in (q['y_low'], q['y_high']) for z in (q['z_near'], q['z_far'])]
        bounds.append((min(xa), max(xa), min(ya), max(ya)))
    bounds = np.asarray(bounds)
    nominal = (np.abs(bounds) > V.HALF_FOV_RAD).any(1)
    out.update(query_native_corner_uv=uv, query_native_corner_homogeneous_scale=scale,
               query_native_clipped=native, query_nominal_fov_clipped=nominal,
               query_fov_clipped=native | nominal, expanded_query_angle_bounds_deg=np.rad2deg(bounds))
    if len(base['queries']) == 4:
        comp_uv, comp_scale, _ = corners(base['queries'])
        out.update(component_query_native_corner_uv=comp_uv,
                   component_query_native_corner_homogeneous_scale=comp_scale)
    return out


offset_geometry = geometry


def query_scores(predicted_optical_z, geo):
    """Whole D16 from an observed/predicted cloud; no GT or object identity."""
    offset = geo['body_center_offset_m']
    if offset == 0:
        return A.query_scores(predicted_optical_z, geo)
    depth, valid = V._depth(predicted_optical_z, geo)
    fov_valid = valid & geo['fov_mask']
    xabs, y = np.abs(depth*geo['fx']-offset), depth*geo['fy']
    scores, order16, maxima, counts, contact, expanded = [], [], [], [], [], []
    closest, farthest, contact_z16 = [], [], []
    for q in geo['queries']:
        take = fov_valid & (depth >= .6) & (depth < 2.1) & (y >= q['y_low']) & (y <= q['y_high'])
        xx, zz = xabs[take], depth[take]
        cc = xx < .3
        counts.append(len(xx)); contact.append(int(cc.sum())); expanded.append(int((xx <= .4).sum()))
        d16 = float(np.partition(xx, V.MIN_SUPPORT-1)[V.MIN_SUPPORT-1]) if len(xx) >= V.MIN_SUPPORT else np.inf
        order16.append(d16); scores.append(.3-d16)
        maxima.append(float(.3-xx.min()) if len(xx) else -np.inf)
        cz = zz[cc]
        closest.append(float(cz.min()) if len(cz) else np.nan)
        farthest.append(float(cz.max()) if len(cz) else np.nan)
        contact_z16.append(float(np.partition(cz, V.MIN_SUPPORT-1)[V.MIN_SUPPORT-1]) if len(cz) >= V.MIN_SUPPORT else np.nan)
    possible = geo['possible_query_rays']
    pc = possible.sum(axis=(1, 2)); vc = (possible & valid[None]).sum(axis=(1, 2))
    coverage = np.divide(vc, pc, out=np.zeros(2), where=pc > 0)
    abstain = (coverage < V.MIN_COVERAGE) | (pc == 0)
    raw = np.asarray(scores)
    return dict(query_names=[q['name'] for q in geo['queries']], score=np.where(abstain, -np.inf, raw),
        score_without_coverage=raw, abstain=abstain, abs_x_order16=np.asarray(order16),
        max_single_point_intrusion=np.asarray(maxima), return_count=np.asarray(counts),
        contact_count=np.asarray(contact), expanded_count=np.asarray(expanded),
        possible_ray_count=pc, valid_possible_count=vc, missing_possible_count=pc-vc, coverage=coverage,
        fov_pixel_count=int(geo['fov_mask'].sum()), valid_fov_pixel_count=int(fov_valid.sum()),
        invalid_fov_pixel_count=int((geo['fov_mask'] & ~valid).sum()), minimum_pixel_support=V.MIN_SUPPORT,
        closest_contact_optical_z=np.asarray(closest), farthest_contact_optical_z=np.asarray(farthest),
        contact_optical_z_order16=np.asarray(contact_z16),
        longitudinal_diagnostics='min/max contact-point optical Z and 16th-smallest contact Z; descriptive only, no sample selection')


def visible_truth(reference_optical_z, geo):
    """Visible labels plus conservative full-volume limits, never unseen clear."""
    if geo['body_center_offset_m'] == 0:
        out = A.visible_truth(reference_optical_z, geo)
    else:
        out = query_scores(reference_optical_z, geo)
        depth, valid = V._depth(reference_optical_z, geo)
        front = geo['possible_query_rays'] & valid[None] & (depth[None] < geo['possible_query_entry_z'])
        occluded = front.sum(axis=(1, 2))
        fraction = np.divide(occluded, out['valid_possible_count'], out=np.zeros(2), where=out['valid_possible_count'] > 0)
        c, e = out['contact_count'], out['expanded_count']
        unknown = out['abstain'] | ((c > 0) & (c < V.MIN_SUPPORT)) | ((e > 0) & (e < V.MIN_SUPPORT))
        contact = ~unknown & (c >= V.MIN_SUPPORT)
        passing = ~unknown & (c == 0) & (e >= V.MIN_SUPPORT)
        clear = ~unknown & (e == 0)
        category = np.full(2, 'UNKNOWN', dtype='<U8')
        category[contact] = 'contact'; category[passing] = 'pass'; category[clear] = 'clear'
        bins = np.full(2, '', dtype='<U12'); score = out['score']
        bins[contact & (score > 0) & (score <= .02)] = 'contact0-2'
        bins[contact & (score > .02) & (score <= .05)] = 'contact2-5'
        bins[contact & (score > .05)] = 'contact>5'
        assert np.all(contact.astype(int)+passing+clear+unknown == 1) and np.all(bins[contact] != '')
        out.update(category=category, contact=contact, **{'pass':passing}, clear=clear, unknown=unknown,
            contact_bin=bins, minimum_coverage=V.MIN_COVERAGE,
            occluded_possible_count=occluded, foreground_occlusion_fraction=fraction,
            occlusion_fraction_denominator=out['valid_possible_count'].copy(),
            query_fov_clipped=geo['query_fov_clipped'].copy(), query_native_clipped=geo['query_native_clipped'].copy(),
            query_nominal_fov_clipped=geo['query_nominal_fov_clipped'].copy(),
            expanded_query_angle_bounds_deg=geo['expanded_query_angle_bounds_deg'].copy(),
            native_covers_nominal_fov=geo['native_covers_nominal_fov'],
            truth_scope='FOV-contained visible points in whole [.6,2.1) laterally translated query; D16/16-pixel support; no hidden/FOV-cropped clearance claim')
    depth, valid = V._depth(reference_optical_z, geo)
    # Legacy occlusion counts only foreground before entry. For a conservative
    # whole-volume label, a first surface anywhere before ray exit may also
    # hide another surface inside the query; even one missing ray is unknown.
    hidden_tail = (geo['possible_query_rays'] & valid[None]
                   & (depth[None] < geo['possible_query_exit_z'])).sum(axis=(1, 2))
    limited = (out['query_fov_clipped'] | (hidden_tail > 0)
               | (out['missing_possible_count'] > 0))
    full = out['category'].copy()
    full[limited & ~out['contact']] = 'UNKNOWN'
    out.update(body_center_offset_m=geo['body_center_offset_m'], visible_category=out['category'].copy(),
               full_volume_category=full, full_volume_unknown=(full == 'UNKNOWN'),
               full_volume_hidden_tail_ray_count=hidden_tail,
               full_volume_limits='Observed supported contact remains positive; cropped volume, any missing ray, or first surface before ray exit makes noncontact UNKNOWN. This conservative diagnostic is not physical occupancy certification; visible categories remain legacy-compatible.')
    return out


def selftest():
    base = V.ray_geometry((32, 64), np.diag([.6, .6, -1.]))
    depths = [np.full((32, 64), z) for z in (.3, .8, 1.5, 3.)]
    missing = depths[1].copy(); missing[:, :20] = np.nan; depths.append(missing)
    zero = geometry(base, 0)
    for depth in depths:
        old, new = A.visible_truth(depth, A.whole_geometry(base)), visible_truth(depth, zero)
        for key, value in old.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, new[key])
            else:
                assert value == new[key], key
    # Horizontal reflection with symmetric rays swaps positive/negative offset.
    for offset in OFFSETS:
        geo = geometry(base, offset); mirror = geometry(base, -offset)
        np.testing.assert_array_equal(geo['possible_query_rays'][:, :, ::-1], mirror['possible_query_rays'])
        a = query_scores(depths[1], geo); b = query_scores(depths[1][:, ::-1], mirror)
        np.testing.assert_allclose(a['score'], b['score'], atol=1e-14)
    # Explicit ray slopes cover both signs and zero; independent dense-z witness.
    fx = np.array([[-1., -.5, 0., .2, .5, 1.]])
    fy = np.array([[0., -.2, .6, 0., .4, 1.]])
    q = dict(z_near=.6, z_far=2.1, y_low=-.2, y_high=.9)
    zz = np.linspace(.6, 2.1, 60001, endpoint=False)
    for offset in OFFSETS:
        ok, lo, hi = possible_rays(fx, fy, np.ones_like(fx, bool), q, offset)
        brute = ((fx[..., None]*zz >= offset-.4) & (fx[..., None]*zz <= offset+.4)
                 & (fy[..., None]*zz >= q['y_low']) & (fy[..., None]*zz <= q['y_high'])).any(-1)
        np.testing.assert_array_equal(ok, brute)
        witness = np.where(lo < hi, (lo+hi)/2, lo)
        assert np.all(np.abs((fx*witness-offset)[ok]) <= .4+1e-14)
    # Closed lateral boundary at the near plane; open far boundary excludes
    # a ray that could enter the x-slab only at z_far.
    q2 = dict(z_near=.6, z_far=2.1, y_low=-1., y_high=1.)
    ok, lo, _ = possible_rays(np.array([[1.]]), np.array([[0.]]), np.array([[True]]), q2, .2)
    assert ok[0,0] and lo[0,0] == .6
    q3 = dict(z_near=.6, z_far=2.1, y_low=2.1, y_high=3.)
    ok, _, _ = possible_rays(np.array([[0.]]), np.array([[1.]]), np.array([[True]]), q3, 0.)
    assert not ok.any()
    clear = visible_truth(depths[3], geometry(base, .2))
    assert clear['clear'].all() and clear['query_fov_clipped'].all() and clear['full_volume_unknown'].all()
    print('PASS_OFFSET_PUBLIC_SLAB_MIRROR_ZERO_LEGACY_UNKNOWN_SYNTHETIC')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['selftest']); p.parse_args()
    selftest()
