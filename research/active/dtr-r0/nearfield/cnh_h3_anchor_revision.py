"""Existing-recording H3 candidate constraints; no simulator edits or task rerun.

Run with the hardware-bringup NumPy Python, --source <session> --output <new-dir>.
The output directory must not exist. All grids/diagnostic tolerances below are
exploratory engineering choices, not confidence intervals or calibration gates.
The existing synthesize_response is the forward model. Its exact Poisson moments
are used instead of random Monte Carlo fits. CPU: tiny arrays/metadata, no GPU job.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from cnh_route_sensor import (RAW_BIN_M, SensorParameters, angular_rays,
                              axial_to_radial, synthesize_response)
from cnh_h3_recording_diagnostics import load_recording, temporal_diagnostics

SEGMENTS = ('01-background-attempt1', '02-object-attempt2',
            '03-movement-attempt3', '04-restored-attempt4')
CENTRE = [27, 28, 35, 36]
TARGET = [1, 2, 3, 4]
FAR = [8, 9, 10, 11, 12, 13]
DISTANCES = [.60, .635, .67]  # assumed search bracket, NOT ruler uncertainty
RHOS = [.3, .5, .7, .9]      # assumed diffuse NIR reflectance bracket
ZERO = [-.12, -.09, -.06, -.03, 0., .03, .06]
SIGMA = [1., 2., 3., 4.]
TAIL = [0., .1, .25]
DECAY = [4., 8.]
GROUPS = (TARGET, FAR)


def forward_planes(distances, params=SensorParameters(), rho=1., samples=8):
    """Axial fronto-parallel planes, correct radial ray distances and incidence.

    Unit signal_counts and output_gain, deterministic expected output only.
    No sensor state is mutated. Returns [plane, 64, 16] aggregate expectations.
    """
    rays, weights = angular_rays(samples)
    d = np.asarray(distances)[:, None, None, None]
    radial = axial_to_radial(d, rays)
    p = replace(params, signal_counts=1., output_gain=1., noise_scale=0.)
    raw = synthesize_response(radial, rho, rays[..., 2], weights, params=p,
                              seed=0)['histogram']
    return raw.reshape(len(distances), 64, 16, 8).sum(-1)


def reference_snr(g, ambient, params=SensorParameters()):
    """Exactly the 2m, rho=.5, zone(3,3), 7 raw-bin definition in
    cnh_track_a_v13_sensor.reference_parameters, without importing geometry/scipy.
    c cancels in this model SNR. The original fixed reference window is retained
    even for candidate pulse/zero parameters; no H3-bin-to-raw-bin reconstruction.
    """
    d = np.full((8, 8, 1), np.inf)
    d[3, 3, 0] = 2.
    p = replace(params, signal_counts=1., output_gain=1., noise_scale=0.)
    a = synthesize_response(d, .5, 1., 1., params=p, seed=0)['histogram'][3, 3]
    b = synthesize_response(np.full_like(d, np.inf), .5, 1., 1.,
                            params=p, seed=0)['histogram'][3, 3]
    k = int(np.floor(2./RAW_BIN_M))
    w = np.arange(k-3, k+4)
    return float(g*(a-b)[w].sum()/np.sqrt(g*a[w].sum()+2*ambient*len(w)))


def nnls2(x, y):
    """Exact nonnegative two-column least squares, including boundary optima."""
    candidates = [np.zeros(2)]
    v = np.linalg.lstsq(x, y, rcond=None)[0]
    if np.all(v >= 0):
        candidates.append(v)
    for j in range(2):
        a = np.zeros(2)
        a[j] = max(0., float(x[:, j]@y / max(x[:, j]@x[:, j], 1e-300)))
        candidates.append(a)
    return min(candidates, key=lambda a: np.sum((x@a-y)**2))


def fit_moments(unit, mean, var):
    """Joint weighted LS of mean and variance, equal weights for four blocks:
    mean/variance x target/far bins, across four central zones (40 cells).
    M=c*g*u = p*u; V=c^2*(g*u+16*A) = q*u+r. Independent Poisson
    raw bins imply diagonal H3 covariance. Optimize p>0,q>=0,r>=0 exactly.
    c=q/p, g=p^2/q, A=r/(16*c^2); zero q is non-identifiable, not clipped.
    Mean scale=max(abs(observed mean), 5% max target mean); variance scale=
    observed variance (positive). These are deterministic relative weights,
    not sampling standard errors or an inferential likelihood.
    """
    floor = max(float(np.max(np.abs(mean[:, TARGET])))*.05, 1e-12)
    xm, ym, xv, yv = [], [], [], []
    for bins in GROUPS:
        u, m, v = unit[:, bins].ravel(), mean[:, bins].ravel(), var[:, bins].ravel()
        if np.any(v <= 0):
            raise ValueError('Zero variance in moment fit; cannot identify relative weights')
        wm = 1/np.maximum(np.abs(m), floor)/np.sqrt(len(m))
        wv = 1/v/np.sqrt(len(v))
        xm.extend(u*wm); ym.extend(m*wm)
        xv.extend(np.stack((u*wv, wv), axis=1)); yv.extend(v*wv)
    xm, ym, xv, yv = map(np.asarray, (xm, ym, xv, yv))
    p = max(0., float(xm@ym/max(xm@xm, 1e-300)))
    q, r = nnls2(xv, yv)
    if p <= 0 or q <= 0 or r <= 0:
        return dict(identified=False, p=float(p), q=float(q), r=float(r))
    c, g = q/p, p*p/q
    a = r/(16*c*c)
    return dict(identified=True, signal_counts=float(g), ambient_counts=float(a),
                unit_scale_c=float(c), p=float(p), q=float(q), r=float(r),
                loss=float((np.mean((xm*p-ym)**2)*len(xm) +
                            np.mean((xv@np.array([q, r])-yv)**2)*len(yv))/4))


def moment_errors(unit, mean, var, fit):
    pred_m = fit['p']*unit
    pred_v = fit['q']*unit+fit['r']
    result = {}
    for label, bins in zip(('target', 'far'), GROUPS):
        m, v = mean[:, bins], var[:, bins]
        # L2 relative error for means avoids dividing by near-zero signed bins.
        result[label+'_mean_relative_l2'] = float(np.linalg.norm(pred_m[:, bins]-m)/
                                                  max(np.linalg.norm(m), 1e-12))
        result[label+'_variance_relative_rmse'] = float(np.sqrt(np.mean((pred_v[:, bins]/v-1)**2)))
    result['xtalk_bin0_mean_ratio_pred_over_obs'] = float(pred_m[:, 0].mean()/mean[:, 0].mean())
    return result


def stable_windows(data):
    """Firmware SINGLE RETURN stability candidates, never firmware 'plane' truth.
    Nonoverlapping eight-frame blocks on original acquisition ordering; all eight
    status5, nb_target1, positive distance, <=20mm range, seq contiguous,
    each dt 100..300ms; per-zone central candidates, median 100..550mm.
    No selection by CNH residual. All eligible windows retained, none handpicked.
    """
    rows = []
    for name in SEGMENTS[1:3]:
        d = data[name]
        for i in range(0, len(d['H'])-7, 8):
            sl = slice(i, i+8)
            dt = np.diff(d['ms'][sl])
            if not (np.all(np.diff(d['seq'][sl]) == 1) and np.all((dt >= 100) & (dt <= 300))):
                continue
            for z in CENTRE:
                distance = d['D'][sl, z]
                if not (np.all(d['S'][sl, z] == 5) and np.all(d['N'][sl, z] == 1)
                        and np.ptp(distance) <= 20 and 100 < np.median(distance) < 550):
                    continue
                rows.append(dict(segment=name, start_index=i, zone=z, frames=8,
                                 seq_first=int(d['seq'][i]), seq_last=int(d['seq'][i+7]),
                                 distance_reference_m=float(np.median(distance)/1000),
                                 span_mm=float(np.ptp(distance)),
                                 mean=d['H'][sl, z].mean(0).tolist()))
    return rows


def shape_error(pred, observed, bins):
    """Free positive amplitude per window; normalized shape L2 error only."""
    u, y = pred[bins], observed[bins]
    amp = max(0., float(u@y/max(u@u, 1e-300)))
    return float(np.linalg.norm(amp*u-y)/max(np.linalg.norm(y), 1e-12))


def covariance_check(d):
    rows = []
    for z in CENTRE:
        h = d['H'][:, z, TARGET]
        cov = np.cov(h, rowvar=False, ddof=1)
        rows.append(dict(zone=z, covariance=cov.tolist(),
                         variance_sum_over_sum_variances=float(cov.sum()/np.trace(cov))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    data = {s: load_recording(args.source/s/'tof'/'raw.bin') for s in SEGMENTS}
    train, hold = data[SEGMENTS[0]], data[SEGMENTS[3]]
    m, v = train['H'][:, CENTRE].mean(0), train['H'][:, CENTRE].var(0, ddof=1)
    hm, hv = hold['H'][:, CENTRE].mean(0), hold['H'][:, CENTRE].var(0, ddof=1)
    results = dict(scope='Existing consumed data; candidate constraints, no calibration or v4 rerun',
                   backend=dict(name='numpy', device='cpu', reason='TASK_NOT_GPU_SUITABLE: small analytic moments and grids'),
                   settings=dict(distances=DISTANCES, rhos=RHOS, centre_zones=CENTRE,
                                 target_bins=TARGET, far_bins=FAR, quadrature_per_axis=8,
                                 zero=ZERO, sigma=SIGMA, tail=TAIL, decay=DECAY,
                                 candidate_tolerance='relative shape L2<=0.20 at every reference; exploratory, not CI',
                                 moment_tolerance='target mean L2<=0.20 and both variance relative RMSE<=0.25; far mean diagnostic only'),
                   quality={s: data[s]['quality'] for s in SEGMENTS},
                   temporal={s: temporal_diagnostics(data[s]) for s in (SEGMENTS[0], SEGMENTS[3])},
                   covariance={s: covariance_check(data[s]) for s in (SEGMENTS[0], SEGMENTS[3])})
    fits = []
    for rho in RHOS:
        units = forward_planes(DISTANCES, rho=rho)
        for distance, unit in zip(DISTANCES, units):
            fit = fit_moments(unit[CENTRE], m, v)
            row = dict(distance_m=distance, rho=rho, **fit)
            if fit['identified']:
                row['fit_errors'] = moment_errors(unit[CENTRE], m, v, fit)
                row['holdout_errors'] = moment_errors(unit[CENTRE], hm, hv, fit)
                row['reference_snr'] = reference_snr(fit['signal_counts'], fit['ambient_counts'])
                row['moment_compatible'] = all(e['target_mean_relative_l2'] <= .20
                    and e['target_variance_relative_rmse'] <= .25
                    and e['far_variance_relative_rmse'] <= .25
                    for e in (row['fit_errors'], row['holdout_errors']))
            fits.append(row)
    results['moment_fits_fixed_pulse'] = fits
    print('Moment fits completed', flush=True)
    windows = stable_windows(data)
    results['stable_windows'] = windows
    # Firmware gives radial distance. Convert to approximate axial plane at zone
    # centre (fronto-parallel assumption); no physical plane/registration claim.
    rays, weights = angular_rays(8)
    cosz = (rays[..., 2]*weights).sum(-1)/weights.sum(-1)
    axial = [w['distance_reference_m']*float(cosz.reshape(64)[w['zone']]) for w in windows]
    # Candidate shape comparison uses bins1..5: bin0 excluded as contaminated.
    # Existing crosstalk remains in forward model, never subtracted as truth.
    # Near 0.3m this excludes much of the echo: record that identification limit.
    shape_bins = [1, 2, 3, 4, 5]
    family = []
    for zero, sigma, tail, decay in itertools.product(ZERO, SIGMA, TAIL, DECAY):
        if tail == 0 and decay != DECAY[0]:
            continue  # identical zero-tail model; avoid duplicate candidates
        p = replace(SensorParameters(), range_zero_m=zero, pulse_sigma_bins=sigma,
                    tail_mass=tail, tail_decay_bins=decay)
        rendered = forward_planes(DISTANCES+axial, p, rho=.5)
        static_errors = [shape_error(u[CENTRE].mean(0), m.mean(0), shape_bins)
                         for u in rendered[:len(DISTANCES)]]
        static_index = int(np.argmin(static_errors))
        object_errors = [shape_error(u[w['zone']], np.array(w['mean']), shape_bins)
                         for u, w in zip(rendered[len(DISTANCES):], windows)]
        worst = max([static_errors[static_index]]+object_errors)
        family.append(dict(range_zero_m=zero, apparent_shift_sign_note='positive zero shifts histogram left',
                           pulse_sigma_bins=sigma, tail_mass=tail, tail_decay_bins=decay,
                           cabinet_candidate_m=DISTANCES[static_index],
                           cabinet_shape_error=static_errors[static_index],
                           object_shape_errors=object_errors, max_shape_error=worst,
                           compatible=bool(worst <= .20)))
    results['shape_family'] = family
    accepted = [r for r in family if r['compatible']]
    results['shape_compatible_count'] = len(accepted)
    # Refit moments for every shape-compatible family at all cabinet/rho choices.
    # If none compatible, keep the least-discrepant family as explicitly rejected
    # diagnostic, not an accepted posterior or a replacement simulator.
    best = min(family, key=lambda r: r['max_shape_error'])
    selected = accepted or [best]
    expanded = []
    for row in selected:
        p = replace(SensorParameters(), **{k: row[k] for k in
                    ('range_zero_m', 'pulse_sigma_bins', 'tail_mass', 'tail_decay_bins')})
        for rho in RHOS:
            for distance, unit in zip(DISTANCES, forward_planes(DISTANCES, p, rho)):
                fit = fit_moments(unit[CENTRE], m, v)
                record = dict(shape={k: row[k] for k in ('range_zero_m', 'pulse_sigma_bins', 'tail_mass', 'tail_decay_bins')},
                              shape_compatible=row['compatible'], distance_m=distance, rho=rho, **fit)
                if fit['identified']:
                    record['fit_errors'] = moment_errors(unit[CENTRE], m, v, fit)
                    record['holdout_errors'] = moment_errors(unit[CENTRE], hm, hv, fit)
                    record['reference_snr'] = reference_snr(fit['signal_counts'], fit['ambient_counts'], p)
                    record['moment_compatible'] = all(e['target_mean_relative_l2'] <= .20
                        and e['target_variance_relative_rmse'] <= .25
                        and e['far_variance_relative_rmse'] <= .25
                        for e in (record['fit_errors'], record['holdout_errors']))
                expanded.append(record)
    results['moment_fits_candidate_pulses'] = expanded
    results['best_shape_diagnostic'] = best
    near_best = [r for r in family if r['max_shape_error'] <= best['max_shape_error']+.05]
    results['near_best_shape_envelope'] = dict(
        rule='max shape error <= minimum + 0.05; exploratory relative-ranking envelope, NOT supported confidence set',
        count=len(near_best), ranges={k: [min(r[k] for r in near_best), max(r[k] for r in near_best)]
            for k in ('range_zero_m', 'pulse_sigma_bins', 'tail_mass', 'tail_decay_bins', 'cabinet_candidate_m')})
    results['status_coverage'] = {s: dict(
        total_region_readings=int(d['S'].size),
        strict=int(((d['S'] == 5) & (d['N'] > 0) & (d['D'] > 0)).sum()),
        lenient=int((np.isin(d['S'], [5, 6, 9]) & (d['N'] > 0) & (d['D'] > 0)).sum()))
        for s, d in data.items()}
    results['mean_ambient_centre'] = float(train['A'][:, CENTRE].mean())
    results['identifiability_notes'] = [
        'Mean/variance matching uses Poisson diagonal covariance; empirical covariance is separately reported, not repaired by c.',
        'signal_counts and rho are confounded; search distance/rho brackets are assumptions, not measured uncertainty.',
        'Reported reference SNR is a model implication, not a seven-raw-bin observation recovered from H3.',
        'No firmware plane label exists. Stable single-return windows and camera context are only conditional plane references.',
        'Firmwares radial distances are not independent truth; joint fits cannot prove an absolute zero or compensation.',
        'No near bin0 crosstalk amplitude fit; shape family holds existing residual crosstalk fixed, a possible source of misfit.',
        'Same-scene restored segment is a holdout consistency check, not independent generalization evidence.',
        'All grids and tolerances are exploratory; no formal acceptance or confidence interval.'
    ]
    results['runtime_seconds'] = time.perf_counter()-t0
    results['source_sha256'] = {s: hashlib.sha256((args.source/s/'tof'/'raw.bin').read_bytes()).hexdigest() for s in SEGMENTS}
    results['code_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__), Path(__file__).with_name('cnh_h3_recording_diagnostics.py'),
         Path(__file__).with_name('cnh_route_sensor.py'), Path(__file__).with_name('cnh_track_a_v13_sensor.py'))}
    (args.output/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), runtime=results['runtime_seconds'],
                         stable_windows=len(windows), shape_compatible=len(accepted),
                         best_shape_error=best['max_shape_error']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
