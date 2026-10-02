"""Sub-bin pulse fitting on existing H3 synthetic observations.

This tests information discarded by the strongest-bin-center readout. The pulse
model and bin zero are granted simulator parameters, not hardware calibration.
The estimator receives only an observed histogram, ambient and pulse dictionary;
target truth and RGB range never enter fitting. Multiple surfaces remain mixed.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from cnh_route_sensor import RAW_BIN_M, RAW_BINS, H3, SensorParameters, _pulse_matrix
from cnh_rgb_clearance_probe import summarize

ROOT = Path(__file__).resolve().parents[4]
PRIOR = ROOT / 'artifacts.local/work/cnh-rgb-zone-association-20261001-v2'
DEFAULT_OUT = ROOT / 'artifacts.local/work/cnh-tof-subbin-20261002'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dictionary(parameters):
    if parameters.range_zero_m != 0:
        raise ValueError('This diagnostic supports the declared zero-offset proxy only')
    return _pulse_matrix(parameters).reshape(RAW_BINS, H3.bins, H3.sub_sample).sum(-1)


def fit_peak(histogram, templates, ambient=4., noise_scale=1.):
    """Fit a single pulse plus nonnegative flat nuisance in five peak bins.

    Candidate raw centers lie inside the observed peak aggregate or its immediate
    neighbors. Fixed inverse-variance weights use observed signed counts only.
    Each two-column NNLS is solved exactly by comparing interior and boundaries.
    The local fit refines the strongest measured response, not target identity.
    """
    y = np.asarray(histogram, float)
    assert y.shape == (16,) and np.isfinite(y).all()
    peak = int(y.argmax())
    if y[peak] <= 0:
        return dict(range_m=None, status='NO_POSITIVE_SIGNAL')
    use = np.arange(max(0, peak-2), min(16, peak+3))
    ids = np.arange(max(0, (peak-1)*8), min(128, (peak+2)*8))
    w = 1 / np.sqrt(np.maximum(np.maximum(y[use], 0)+2*ambient*8, 1.)) if noise_scale else np.ones(len(use))
    obs = y[use]*w
    best = None
    for index in ids:
        matrix = np.column_stack((templates[index, use], np.ones(len(use))))*w[:, None]
        gram = matrix.T@matrix
        rhs = matrix.T@obs
        solutions = [np.zeros(2), np.array([max(0., rhs[0]/max(gram[0,0], 1e-30)), 0.]),
                     np.array([0., max(0., rhs[1]/gram[1,1])])]
        interior = np.linalg.lstsq(matrix, obs, rcond=None)[0]
        if np.all(interior >= 0):
            solutions.append(interior)
        for coefficient in solutions:
            error = float(np.sum((matrix@coefficient-obs)**2))
            candidate = (error, int(index), coefficient)
            if best is None or error < best[0]:
                best = candidate
    error, index, coefficient = best
    return dict(range_m=float((index+.5)*RAW_BIN_M) if coefficient[0]>0 else None,
                status='OK' if coefficient[0]>0 else 'FLAT_ONLY', raw_bin=index,
                observed_peak_bin=peak, amplitude=float(coefficient[0]),
                nuisance=float(coefficient[1]), weighted_sse=error,
                relative_local_residual=float(np.sqrt(error/max(float(obs@obs), 1e-30))))


def self_check():
    p = replace(SensorParameters(), noise_scale=0, tail_mass=0, crosstalk_fraction=0, neighbour_leak=0)
    templates = dictionary(p)
    # Interior-bin and boundary-bin cases with a flat nuisance verify actual
    # interpolation information, rather than mirroring the fit implementation.
    for k in (17, 20, 24, 31, 39, 50, 63, 77):
        result = fit_peak(templates[k]*800+3, templates, noise_scale=0)
        assert result['raw_bin'] == k, (k, result)
        assert abs(result['nuisance']-3)<1e-5
    assert fit_peak(np.zeros(16), templates)['range_m'] is None
    try:
        dictionary(replace(p, range_zero_m=.1))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonzero offsets must not silently return zero-offset ranges')
    return dict(status='PASS', checks=['eight single echoes recover raw bin and nuisance', 'empty signal stays missing',
                                     'unsupported nonzero range offset rejected'])


def noise_assay():
    """Synthetic point returns: identifiability only, no obstacle claim."""
    p = replace(SensorParameters(), tail_mass=0, crosstalk_fraction=0, neighbour_leak=0)
    templates = dictionary(p)
    rng = np.random.default_rng(2026100207)
    rows = []
    for k in range(16, 80):
        distance = (k+.5)*RAW_BIN_M
        # Uniform rho=.5; return law from the existing uncalibrated proxy.
        expectation = templates[k]*(2000/distance**2)
        for draw in range(20):
            y = rng.poisson(expectation+32)-rng.poisson(np.full(16, 32))
            center = (int(y.argmax())+.5)*8*RAW_BIN_M
            fit = fit_peak(y, templates)
            rows.append(dict(raw_bin=k, draw=draw, truth_m=distance, center_m=float(center), fit_m=fit['range_m']))
    errors = {a:np.asarray([abs(r[a]-r['truth_m']) for r in rows if r[a] is not None]) for a in ('center_m','fit_m')}
    return dict(n=len(rows), raw_distances=64, draws=20, role='matched-pulse isolated point-return synthetic sanity assay',
                metrics={a:dict(valid=len(e), median_cm=float(np.median(e)*100), p95_cm=float(np.quantile(e,.95)*100),
                                within5cm=int((e<=.05).sum())) for a,e in errors.items()}), rows


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PLAN.json').exists():
        raise FileExistsError('Use a new output directory; preserve earlier results')
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in (PRIOR/'case-ledger.json', Path(__file__),
        Path(__file__).with_name('cnh_route_sensor.py'), Path(__file__).with_name('cnh_rgb_clearance_probe.py'))}
    save(out/'PLAN.json', dict(question='Does fitting the known pulse recover useful precision discarded by H3 bin centers?',
        scope='same consumed 15 Hypersim edges, no acquisition/training; existing clean and stress H3 histograms',
        estimator='single local pulse plus nonnegative flat nuisance; no RGB prior/identity/target truth',
        assumptions='known proxy pulse shape and bin zero; single dominant component locally; not ST firmware or hardware calibration',
        controls='same original bin-center readout; known-pulse clean/stress; stress with clean pulse exposes shape mismatch',
        input_sha256=inputs, backend='numpy CPU: TASK_NOT_GPU_SUITABLE, tiny 5x2 least squares'))
    started = time.perf_counter()
    save(out/'self-check.json', self_check())
    old = read(PRIOR/'case-ledger.json')
    ledger = []
    for event in old:
        estimates = {a:event['estimates'][a] for a in ('depthpro', 'zone_q10', 'h3_clean', 'h3_stress')}
        fits = {}
        for name in ('clean', 'stress', 'stress_mismatch'):
            source = event['electronics']['stress' if name.startswith('stress') else 'clean']
            params = SensorParameters(**(event['electronics']['clean']['params'] if name=='stress_mismatch' else source['params']))
            zone = event['zone_id']
            # Retain the prior observable SNR validity; do not rescue invalids.
            fitted = fit_peak(source['histogram'][zone], dictionary(params), source['ambient'][zone], source['params']['noise_scale'])
            if not source['valid'][zone]:
                fitted.update(range_m=None, status='PRIOR_SNR_INVALID')
            fits[name] = fitted
            estimates['fit_'+name] = event['radial_edge_factor']*fitted['range_m']-.30 if fitted['range_m'] is not None else None
        ledger.append(dict(id=event['id'], scene=event['scene'], gt_clearance_m=event['gt_clearance_m'],
                           estimates=estimates, fits=fits,
                           errors_m={a:v-event['gt_clearance_m'] if v is not None else None for a,v in estimates.items()}))
    noise, noise_rows = noise_assay()
    save(out/'noise-ledger.json', noise_rows)
    result = dict(status='COMPLETE', n=len(ledger), scenes=len({e['scene'] for e in ledger}),
        arms={a:summarize(ledger,a) for a in ledger[0]['estimates']}, isolated_return_assay=noise,
        elapsed_seconds=time.perf_counter()-started, python=sys.executable,
        limits=['same15 selected clean edges/8scenes; not contact-recall benchmark',
                'known uncalibrated pulse and zero; fit does not solve target association',
                'stress is one retained simulated draw; no physical precision claim',
                'raw grid37.5mm is assumed simulator quantization; not fitted sub-raw precision'])
    save(out/'case-ledger.json', ledger)
    save(out/'result.json', result)
    save(out/'terminal.json', dict(status='COMPLETE', elapsed_seconds=result['elapsed_seconds']))
    (out/'source').mkdir(exist_ok=True)
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    main(args.out)
