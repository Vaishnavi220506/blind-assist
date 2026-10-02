"""Fixed whole-zone q10 inverse-Z plane fit with one untouched validation zone.

Prediction sees RGB region labels, public rays and 64 sensor returns only.
One RGB mask need not be one plane. Rank of public zone-mean rays is only a
support proxy; it is not proof of nonlinear quantile identifiability.
"""
import argparse
import json

import numpy as np
from scipy.optimize import LinearConstraint, minimize

import cnh_rgb_region_range as C

V = C.V
MIN_INVERSE_Z = 1e-6


def _support(labels, geometry, returns):
    base = C.anchor_prediction(labels, geometry, returns)
    labels = np.asarray(labels); zone = geometry['zone_id']; fov = geometry['fov_mask']
    eligible = {r['region']: r['anchors'] for r in base['regions']}
    items = []
    for region in np.unique(labels[labels >= 0]):
        anchors = sorted(eligible.get(int(region), []), key=lambda a: a['zone'])
        zones = [a['zone'] for a in anchors]
        train = zones[:-1] if len(zones) else []
        heldout = zones[-1] if zones else None
        matrix = np.array([[float(np.mean(geometry['fx'][zone == k])),
                            float(np.mean(geometry['fy'][zone == k])), 1.] for k in train], dtype=np.float64).reshape(-1,3)
        singular = np.linalg.svd(matrix, compute_uv=False) if len(matrix) else np.array([])
        rank = int(np.sum(singular > 1e-10))
        reasons = []
        if len(anchors) < 4:
            reasons.append('fewer_than_four_complete_zone_anchors')
        if rank < 3:
            reasons.append('training_public_mean_ray_rank_below_three')
        take = fov & (labels == region)
        if not take.any():
            reasons.append('region_has_no_FOV_pixels')
        items.append(dict(region=int(region), eligible_zones=zones, training_zones=train,
            heldout_zone=heldout, eligible_anchor_count=len(zones), public_mean_ray_rank=rank,
            singular_values=singular.tolist(), condition_number=float(singular[0]/singular[-1]) if rank == 3 else None,
            support_status='supported' if not reasons else 'unsupported', support_reasons=reasons))
    return items, base


def support(labels, geometry, returns):
    """Eligibility/rank only; no fitting or held-out prediction/error is computed."""
    items = _support(labels, geometry, returns)[0]
    return [dict(item, status='ELIGIBLE' if item['support_status'] == 'supported'
                 else item['support_reasons'][0]) for item in items]


def _rays(geometry, zones):
    zone = geometry['zone_id']
    return [(np.asarray(geometry['radial_factor'][zone == k], np.float64),
             np.asarray(geometry['fx'][zone == k], np.float64),
             np.asarray(geometry['fy'][zone == k], np.float64)) for k in zones]


def _forward(parameters, rays):
    """Exact empirical r^-2 q10, using all native public rays of each zone."""
    a,b,c = np.asarray(parameters, np.float64)
    answer = []
    for factor, fx, fy in rays:
        inverse = a*fx+b*fy+c
        if not np.isfinite(inverse).all() or np.any(inverse <= 0):
            return np.full(len(rays), np.nan)
        rr = np.sort(factor/inverse)
        if not len(rr) or not np.isfinite(rr).all() or np.any(rr <= 0):
            return np.full(len(rays), np.nan)
        weights = (rr/rr[0])**(-2.)
        cumulative = np.cumsum(weights)
        index = min(int(np.searchsorted(cumulative,.1*cumulative[-1],side='left')),len(rr)-1)
        answer.append(float(rr[index]))
    return np.asarray(answer, np.float64)


def forward_returns(parameters, geometry, zones):
    """Public forward model, also useful for synthetic checks; no sensor values."""
    return _forward(parameters, _rays(geometry, zones))


def predict(labels, geometry, returns):
    items, base = _support(labels, geometry, returns)
    values = np.asarray(returns, np.float64).reshape(64)
    labels = np.asarray(labels); fov = geometry['fov_mask']
    predicted = base['coarse_q10'].copy()
    region_map = np.full(labels.shape,-1,np.int32)
    diagnostics = []
    for item in items:
        info = dict(item, status=item['support_status'], reasons=list(item['support_reasons']),
                    initial_parameters=None, parameters=None, training_rms_m=None,
                    heldout_absolute_error_m=None, optimizer_success=None)
        diagnostics.append(info)
        if item['support_status'] != 'supported':
            continue
        take = fov & (labels == item['region'])
        xmin,xmax = float(geometry['fx'][take].min()),float(geometry['fx'][take].max())
        ymin,ymax = float(geometry['fy'][take].min()),float(geometry['fy'][take].max())
        corners = np.array([[x,y,1.] for x in (xmin,xmax) for y in (ymin,ymax)],np.float64)
        training = item['training_zones']; heldout = item['heldout_zone']
        # This initial condition never reads the held-out return.
        train_z = base['zone_constant_z'][training]
        initial = np.array([0.,0.,1./np.median(train_z)],np.float64)
        info.update(initial_parameters=initial.tolist(), bbox_corners=corners.tolist(),
                    training_actual_returns=values[training].tolist(),
                    heldout_actual_return=float(values[heldout]))
        train_rays = _rays(geometry,training)
        def objective(parameters):
            modeled = _forward(parameters,train_rays)
            return float(np.mean((modeled-values[training])**2)) if np.isfinite(modeled).all() else 1e30
        try:
            fit = minimize(objective,initial,method='SLSQP',
                           constraints=[LinearConstraint(corners,MIN_INVERSE_Z,np.inf)],
                           options=dict(maxiter=100,ftol=1e-12))
            parameters = np.asarray(fit.x,np.float64)
            info.update(parameters=parameters.tolist(),optimizer_success=bool(fit.success),
                        optimizer_status=int(fit.status),optimizer_message=str(fit.message),iterations=int(fit.nit))
            if not fit.success:
                info['reasons'].append('optimizer_success_false')
            if not np.isfinite(parameters).all():
                info['reasons'].append('nonfinite_parameters')
            inverse_corners = corners@parameters
            info['bbox_inverse_z'] = inverse_corners.tolist()
            if not np.isfinite(inverse_corners).all() or np.any(inverse_corners < MIN_INVERSE_Z):
                info['reasons'].append('bbox_inverse_z_constraint_failed')
            fitted = _forward(parameters,train_rays)
            held = forward_returns(parameters,geometry,[heldout])[0]
            info['training_predicted_returns'] = fitted.tolist()
            info['heldout_predicted_return'] = float(held)
            if not np.isfinite(fitted).all() or not np.isfinite(held):
                info['reasons'].append('nonfinite_forward_returns')
            else:
                rms = float(np.sqrt(np.mean((fitted-values[training])**2)))
                error = float(abs(held-values[heldout]))
                info.update(training_rms_m=rms,heldout_absolute_error_m=error)
                if rms > .05:
                    info['reasons'].append('training_rms_above_0.05m')
                if error > .05:
                    info['reasons'].append('heldout_absolute_error_above_0.05m')
            if info['reasons']:
                numeric = any(r in info['reasons'] for r in ('optimizer_success_false','nonfinite_parameters',
                              'bbox_inverse_z_constraint_failed','nonfinite_forward_returns'))
                info['status'] = 'numeric_failure' if numeric else 'rejected'
                continue
            # Keep the train-only fit. There is deliberately no all-anchor refit.
            z = 1./(parameters[0]*geometry['fx'][take]+parameters[1]*geometry['fy'][take]+parameters[2])
            if not np.isfinite(z).all() or np.any(z <= 0):
                info['status']='numeric_failure';info['reasons'].append('nonfinite_or_nonpositive_propagation')
                continue
            predicted[take] = z;region_map[take] = item['region'];info['status']='accepted'
        except Exception as exc:
            info['status']='numeric_failure';info['reasons'].append('optimizer_or_forward_exception')
            info['exception']=repr(exc)
    return dict(predicted_z=predicted,region_map=region_map,diagnostics=diagnostics)


def selftest():
    g = V.ray_geometry((32,32),np.diag([.4,.4,-1.]))
    labels = np.zeros(g['shape'],np.int32); fov = g['fov_mask']; zones = list(range(64))
    truth_parameters = np.array([.18,-.12,.65])
    inverse = truth_parameters[0]*g['fx']+truth_parameters[1]*g['fy']+truth_parameters[2]
    z = 1/inverse
    expected = V.coarse_returns(g['radial_factor']/inverse,g,.1)['zone_return_radial'].reshape(64)
    np.testing.assert_array_equal(forward_returns(truth_parameters,g,zones),expected)
    fitted = predict(labels,g,expected);d=fitted['diagnostics'][0]
    assert d['status']=='accepted',d
    np.testing.assert_allclose(fitted['predicted_z'][fov],z[fov],atol=1e-4)
    flat = V.coarse_returns(1.5*g['radial_factor'],g,.1)['zone_return_radial'].reshape(64)
    const = predict(labels,g,flat)
    assert const['diagnostics'][0]['status']=='accepted'
    np.testing.assert_allclose(const['predicted_z'][fov],1.5,atol=1e-5)
    badheld = expected.copy();badheld[63] += .5
    bad = predict(labels,g,badheld);bd=bad['diagnostics'][0]
    assert bd['status']=='rejected' and 'heldout_absolute_error_above_0.05m' in bd['reasons']
    np.testing.assert_array_equal(bd['parameters'],d['parameters'])
    assert np.all(bad['region_map']==-1)
    scarce = np.full(64,np.nan);scarce[[0,8,16]]=expected[[0,8,16]]
    assert support(labels,g,scarce)[0]['support_status']=='unsupported'
    collinear = np.full(64,np.nan);collinear[[0,1,2,3]]=expected[[0,1,2,3]]
    assert support(labels,g,collinear)[0]['public_mean_ray_rank']<3
    missing = predict(labels,g,np.full(64,np.nan))
    assert missing['diagnostics'][0]['status']=='unsupported' and np.isnan(missing['predicted_z']).all()
    uncovered = np.full(g['shape'],-1,np.int32);fallback = predict(uncovered,g,expected)
    np.testing.assert_array_equal(fallback['predicted_z'],C.anchor_prediction(uncovered,g,expected)['coarse_q10'])
    assert fallback['diagnostics']==[] and np.all(fallback['region_map']==-1)
    print('PASS exact tilted forward, heldout plane/constantZ recovery, train independence, scarce/collinear/missing/fallback')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selftest',action='store_true',required=True)
    parser.parse_args();selftest()
