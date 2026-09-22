"""Frozen two-template CNH compatibility test, never a metric-range decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from capture import validate_frame
from manual_orientation_review import load_streams
from orientation_review import camera_samples

SECOND = 1_000_000_000
PHASES = ('background', 'foreground', 'mixture', 'return')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_two(y, background, foreground):
    """Exact 2-column NNLS: interior solution or either nonnegative boundary."""
    y = np.asarray(y, dtype=float)
    matrix = np.column_stack((background, foreground))
    norm = np.linalg.norm(y)
    if not np.isfinite(matrix).all() or not np.isfinite(y).all() or norm == 0:
        raise ValueError('nonfinite/zero waveform')
    norms = np.sum(matrix*matrix, axis=0)
    if np.any(norms == 0):
        raise ValueError('zero reference')
    single = np.maximum(0, matrix.T@y / norms)
    candidates = [np.zeros(2), np.array([single[0], 0]), np.array([0, single[1]])]
    interior = np.linalg.lstsq(matrix, y, rcond=None)[0]
    if np.all(interior >= 0):
        candidates.append(interior)
    errors = [float(np.linalg.norm(y-matrix@c)/norm) for c in candidates]
    index = int(np.argmin(errors))
    return {'a': float(candidates[index][0]), 'b': float(candidates[index][1]),
            'error': errors[index], 'best_single_error': min(errors[1:3]),
            'gain': min(errors[1:3])-errors[index]}


def classify(fit, a_floor, b_floor, protocol):
    return (fit['a'] > a_floor and fit['b'] > b_floor and
            fit['error'] <= protocol['maximum_relative_rmse'] and
            fit['gain'] >= protocol['minimum_single_template_error_gain'])


def marker_windows(markers):
    if markers.get('schema') != 'hardware-bringup.cnh-components.v1':
        raise ValueError('Not a CNH four-stage capture')
    segments = markers['segments']
    if [s['phase'] for s in segments] != list(PHASES):
        raise ValueError('Missing/reordered four-stage markers')
    windows, previous = {}, markers['start_host_monotonic_ns']
    for s in segments:
        start, end = s['start_host_monotonic_ns'], s['end_host_monotonic_ns']
        if type(start) is not int or type(end) is not int or end-start != 6*SECOND or start < previous:
            raise ValueError('Invalid/overlapping marker window')
        windows[s['phase']] = (start+SECOND, end-SECOND)
        previous = end
    return windows


def load(run, protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    marker_path = run.parent/'.dashboard-control'/f'{run.name}.manual.json'
    markers = json.loads(marker_path.read_text(encoding='utf-8'))
    if markers.get('protocol_sha256') != sha(protocol_path):
        raise ValueError('Protocol hash does not match capture-time sealed protocol')
    windows = marker_windows(markers)
    provenance = {'sources': {}, 'exclusions': [], 'insufficient_reasons': []}
    frames, cameras = load_streams(run, provenance)
    if provenance['exclusions'] or provenance['insufficient_reasons']:
        raise ValueError(f'Input validation failed: {provenance}')
    phases = {}
    for name, (start, end) in windows.items():
        records = []
        for r in frames:
            stamp = r['host_received_monotonic_ns']
            if start <= stamp < end:
                sensor = r['sensor']
                derived = validate_frame(sensor)
                records.append({'sensor': sensor, 'block': (stamp-start)//SECOND, 'seq': sensor['seq'],
                    'h': np.asarray(derived['hist_normalized']), 'valid': derived['range_valid']})
        phases[name] = records
    samples = {name: camera_samples(cameras, start, end, markers['start_host_monotonic_ns'])
               for name, (start, end) in windows.items()}
    provenance.update(protocol_sha256=sha(protocol_path), markers_sha256=sha(marker_path),
        camera_serial_sha256=sha(run/'camera/serial.bin'), tof_raw_sha256=sha(run/'tof/raw.bin'))
    return protocol, phases, samples, provenance


def zone_result(zone, phases, p):
    data = {'zone': zone, 'eligible': False, 'reasons': [], 'scalar': {}, 'verdict': 'NOT_EVALUABLE'}
    for name, rows in phases.items():
        counts = [sum(r['block'] == i for r in rows) for i in range(4)]
        known = [r['sensor']['distance_mm'][zone] for r in rows if r['valid'][zone] and r['sensor']['distance_mm'][zone] > 3]
        data['scalar'][name] = {'frames': len(rows), 'block_counts': counts, 'known_above3mm': len(known),
            'unknown': sum(not r['valid'][zone] for r in rows), 'median_known_mm': float(np.median(known)) if known else None}
        if min(counts) < p['minimum_frames_per_second']:
            data['reasons'].append(f'{name}: insufficient frame coverage')
    if data['reasons']:
        return data
    for name in PHASES[:2]:
        s = data['scalar'][name]
        if s['known_above3mm']/s['frames'] < p['minimum_reference_valid_fraction']:
            data['reasons'].append(f'{name}: insufficient scalar-valid pure reference')
    if data['reasons']:
        return data
    if data['scalar']['foreground']['median_known_mm'] >= data['scalar']['background']['median_known_mm']:
        data['reasons'].append('foreground reference is not nearer in scalar ordering')
    templates = [np.median(np.stack([r['h'][zone] for r in phases[name] if r['block'] < 2]), axis=0) for name in PHASES[:2]]
    b, f = templates
    denominator = np.linalg.norm(b)*np.linalg.norm(f)
    if denominator == 0 or not np.isfinite(denominator):
        data['reasons'].append('zero/nonfinite template')
        return data
    cosine = float(b@f/denominator)
    data.update(background_template=b.tolist(), foreground_template=f.tolist(), template_cosine=cosine,
                background_peak=int(np.argmax(b)), foreground_peak=int(np.argmax(f)))
    if cosine > p['maximum_template_cosine']:
        data['reasons'].append('reference shapes too similar')
    if int(np.argmax(b))-int(np.argmax(f)) < p['minimum_peak_separation_bins']:
        data['reasons'].append('reference dominant peaks insufficiently separated')
    if data['reasons']:
        return data
    data['eligible'] = True
    try:
        hold = {name: [fit_two(r['h'][zone], b, f) for r in phases[name] if r['block'] >= 2] for name in PHASES[:2]}
        a_floor = max(p['minimum_component_coefficient'], max(v['a'] for v in hold['foreground']))
        b_floor = max(p['minimum_component_coefficient'], max(v['b'] for v in hold['background']))
        data.update(a_floor=a_floor, b_floor=b_floor, pure_holdout=hold)
        for name in PHASES[2:]:
            scored = []
            for r in phases[name]:
                fit = fit_two(r['h'][zone], b, f)
                fit.update(seq=r['seq'], block=r['block'], scalar_known=bool(r['valid'][zone]))
                fit['dual'] = classify(fit, a_floor, b_floor, p)
                low, high = p['return_background_coefficient_range']
                fit['background_compatible'] = low <= fit['a'] <= high and fit['b'] <= b_floor and fit['error'] <= p['maximum_relative_rmse']
                scored.append(fit)
            data[name] = {'frames': scored, 'dual_fraction': sum(v['dual'] for v in scored)/len(scored),
                'background_fraction': sum(v['background_compatible'] for v in scored)/len(scored),
                'block_dual_fraction': [float(np.mean([v['dual'] for v in scored if v['block'] == i])) for i in range(4)]}
    except ValueError as exc:
        data['reasons'].append(str(exc))
        return data
    m, r = data['mixture'], data['return']
    passed = (m['dual_fraction'] >= p['minimum_mixture_dual_fraction'] and
        min(m['block_dual_fraction']) >= p['minimum_block_dual_fraction'] and
        r['dual_fraction'] <= p['maximum_return_dual_fraction'] and
        r['background_fraction'] >= p['minimum_return_background_fraction'])
    data['verdict'] = 'DUAL_TEMPLATE_COMPATIBILITY_SUPPORTED' if passed else 'NOT_SUPPORTED'
    return data


def review_sheet(run, output, samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    for i, (name, row) in enumerate(samples.items()):
        for ax, sample in zip(axes[i], row):
            ax.axis('off')
            if sample['path']:
                with Image.open(run/sample['path']) as image:
                    ax.imshow(image)
            ax.set_title(f"{name} | {sample['status']}\n{sample.get('receipt_delta_ms')} ms", fontsize=9)
    fig.suptitle('Physical execution review: fixed rig / full foreground / partial foreground / clear return\nHost-receipt pairing only; not ToF pixel projection or exact distance verification')
    fig.savefig(output/'rgb-review.png', dpi=135)
    plt.close(fig)


def result_plot(output, phases, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
    for ax, result in zip(axes.flat, results):
        z = result['zone']
        for name, color in zip(PHASES, ('#317d91', '#cb6c31', '#756298', '#579448')):
            if phases[name]:
                ax.plot(np.median(np.stack([r['h'][z] for r in phases[name]]), axis=0), color=color, label=name)
        ax.set_yscale('symlog', linthresh=1)
        ax.set_title(f"Z{z}: {result['verdict']}", fontsize=8)
        ax.set_xlabel('raw bin', fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('All 16 zones: signed CNH medians (symlog)\nFixed controls; compatibility is not physical two-target or distance proof')
    fig.savefig(output/'components.png', dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rgb-review', type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    p, phases, samples, provenance = load(run, args.protocol.resolve())
    report = {'schema': p['schema'], 'run': str(run), 'provenance': provenance, 'protocol': p,
              'script_sha256': sha(__file__), 'camera_samples': samples,
              'compute': 'CPU, small 24x2 least-squares problems; no model training'}
    review_sheet(run, output, samples)
    if args.rgb_review:
        review = json.loads(args.rgb_review.read_text(encoding='utf-8'))
        required = ['fixed_rig_background', 'foreground_full_image', 'mixture_partial_image', 'approximately_same_foreground_distance', 'return_clear']
        good = (review.get('run_id') == run.name and all(review.get(k) is True for k in required)
                and all(s['path'] for row in samples.values() for s in row))
        report['rgb_review'] = review
        report['rgb_review_sha256'] = sha(args.rgb_review)
        report['zones'] = [zone_result(z, phases, p) for z in range(16)]
        eligible = [z for z in report['zones'] if z['eligible'] and z['verdict'] != 'NOT_EVALUABLE']
        passed = [z['zone'] for z in eligible if z['verdict'] == 'DUAL_TEMPLATE_COMPATIBILITY_SUPPORTED']
        report.update(eligible_zones=[z['zone'] for z in eligible], passing_zones=passed,
            verdict='NOT_EVALUABLE' if not good or not eligible else 'DUAL_TEMPLATE_COMPATIBILITY_SUPPORTED' if passed else 'NOT_SUPPORTED')
        result_plot(output, phases, report['zones'])
    else:
        report['verdict'] = 'AWAITING_RGB_EXECUTION_REVIEW'
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('verdict', 'eligible_zones', 'passing_zones') if k in report}))


if __name__ == '__main__':
    main()
