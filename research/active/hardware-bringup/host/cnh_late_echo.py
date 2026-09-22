"""Consumed-data local late-echo contrast against empirical foreground tails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
from cnh_components import load, sha, PHASES


def execution_review_good(old, run_name):
    review = old.get('rgb_review', {})
    keys = ('fixed_rig_background', 'foreground_full_image', 'mixture_partial_image',
            'approximately_same_foreground_distance', 'return_clear')
    samples = old.get('camera_samples', {})
    return (review.get('run_id') == run_name and all(review.get(k) is True for k in keys)
            and all(len(samples.get(n, [])) == 3 and all(s.get('path') for s in samples[n]) for n in PHASES))


def shift_zero(wave, offset):
    result = np.zeros_like(wave, dtype=float)
    if offset == 0:
        result[:] = wave
    elif offset > 0:
        result[offset:] = wave[:-offset]
    else:
        result[:offset] = wave[-offset:]
    return result


def near(wave, p):
    return float(np.maximum(np.asarray(wave)[p['near_bins']], 0).sum())


def contrast(wave, peak, p):
    center = [peak+i for i in p['center_offsets']]
    flanks = [peak+i for i in p['flank_offsets']]
    return float(np.mean(np.asarray(wave)[center])-np.mean(np.asarray(wave)[flanks]))


def cutoff(values, p):
    values = np.asarray(values, dtype=float)
    mad = float(np.median(np.abs(values-np.median(values))))
    return max(p['numerical_floor'], max(0., float(values.max()))+
               p['mad_multiplier']*p['mad_normalization']*mad)


def fit_reference(background, foreground, fg_calibration, bg_calibration, peak, p):
    """Only pure training/calibration records may enter this function."""
    window = [peak+i for i in p['center_offsets']+p['flank_offsets']]
    if min(window) < 0 or max(window) >= len(foreground[0]) or set(window) & set(p['near_bins']):
        raise ValueError('late window missing or overlaps near bins')
    candidates = []
    for wave in foreground:
        for offset in p['template_shifts']:
            shifted = shift_zero(wave, offset)
            n = near(shifted, p)
            if n <= p['numerical_floor']:
                raise ValueError('degenerate shifted foreground near normalization')
            candidates.append(contrast(shifted, peak, p)/n)
    reference = {'peak': peak, 'tail_ratio_upper': max(0., max(candidates)),
                 'foreground_near_median': float(np.median([near(v,p) for v in foreground]))}
    if reference['foreground_near_median'] <= p['numerical_floor']:
        raise ValueError('degenerate foreground near normalization')
    reference['late_cutoff'] = cutoff([score(v, reference, p)['score'] for v in fg_calibration], p)
    reference['near_cutoff'] = cutoff([near(v,p) for v in background+bg_calibration], p)
    amplitudes = [near(v,p) for v in foreground+fg_calibration]
    reference['foreground_near_range'] = [min(amplitudes), max(amplitudes)]
    reference['tail_candidate_count'] = len(candidates)
    return reference


def score(wave, ref, p):
    n, c = near(wave, p), contrast(wave, ref['peak'], p)
    denominator = max(1., n/ref['foreground_near_median'])
    result = {'near': n, 'contrast': c, 'tail_upper': n*ref['tail_ratio_upper'],
              'noise_scale': denominator, 'score': (c-n*ref['tail_ratio_upper'])/denominator}
    if 'late_cutoff' in ref:
        result['late'] = result['score'] > ref['late_cutoff'] and c > 0
        result['near_present'] = n > ref['near_cutoff']
        result['dual'] = result['late'] and result['near_present']
        lo, hi = ref['foreground_near_range']
        result['amplitude_extrapolation'] = not lo <= n <= hi
    return result


def evaluate_zone(old, phases, p):
    z = old['zone']
    result = {'zone': z, 'eligible': False, 'reasons': list(old['reasons']), 'verdict': 'NOT_EVALUABLE'}
    if not old['eligible'] or old['verdict'] == 'NOT_EVALUABLE':
        return result
    train = {name: [r['h'][z] for r in phases[name] if r['block'] in p['training_blocks']] for name in PHASES[:2]}
    cal = {name: [r['h'][z] for r in phases[name] if r['block'] == p['calibration_block']] for name in PHASES[:2]}
    peak = int(np.argmax(np.median(train['background'], axis=0)))
    try:
        ref = fit_reference(train['background'], train['foreground'], cal['foreground'], cal['background'], peak, p)
    except ValueError as exc:
        result['reasons'].append(str(exc))
        return result
    result.update(eligible=True, reference=ref)
    scored = {}
    for name in PHASES:
        rows = phases[name] if name in PHASES[2:] else [r for r in phases[name] if r['block'] == p['negative_check_block']]
        values = []
        for row in rows:
            v = score(row['h'][z], ref, p)
            v.update(seq=row['seq'], block=row['block'], scalar_known=bool(row['valid'][z]))
            values.append(v)
        scored[name] = {'frames': values, 'count': len(values), 'unknown': sum(not v['scalar_known'] for v in values)}
        for key in ('late', 'near_present', 'dual', 'amplitude_extrapolation'):
            scored[name][key+'_count'] = sum(v[key] for v in values)
            scored[name][key+'_fraction'] = scored[name][key+'_count']/len(values)
    m, ret, fg, bg = [scored[n] for n in ('mixture','return','foreground','background')]
    blocks = [float(np.mean([v['dual'] for v in m['frames'] if v['block'] == b])) for b in range(4)]
    gates = {'mixture_fraction': m['dual_fraction'] >= p['minimum_mixture_fraction'],
             'mixture_blocks': min(blocks) >= p['minimum_block_fraction'],
             'foreground_negative': fg['late_count'] <= p['maximum_foreground_check_late_count'],
             'foreground_near': fg['near_present_fraction'] >= p['minimum_foreground_near_fraction'],
             'background_negative': bg['dual_count'] <= p['maximum_background_check_dual_count'],
             'return_late': ret['late_fraction'] >= p['minimum_return_late_fraction'],
             'return_dual': ret['dual_fraction'] <= p['maximum_return_dual_fraction']}
    result.update(phases=scored, gates=gates, mixture_block_dual_fraction=blocks,
                  verdict='EXPLORATORY_TAIL_EXCESS' if all(gates.values()) else 'NOT_SUPPORTED')
    return result


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,4,figsize=(15,11),constrained_layout=True)
    colors = ['#318297','#d67435','#88549d','#57914b']
    for ax, result in zip(axes.flat, report['zones']):
        z = result['zone']
        ax.set_title(f"Z{z}: {result['verdict']}", fontsize=8)
        if not result['eligible']:
            ax.text(.5,.5,'Reference not evaluable',ha='center',transform=ax.transAxes,fontsize=8)
            continue
        for i,name in enumerate(PHASES):
            vals = result['phases'][name]['frames']
            ax.scatter(i+np.linspace(-.14,.14,len(vals)), [v['score'] for v in vals], s=10,color=colors[i])
        ax.axhline(result['reference']['late_cutoff'],color='black',linestyle='--',linewidth=1)
        ax.set_xticks(range(4),['B check','F check','Mixed','Return'],fontsize=7)
        ax.set_ylabel('Local residual / noise scale',fontsize=7)
    fig.suptitle('Consumed Development: background-aligned late contrast vs shifted foreground-tail envelope\nBlack = foreground-calibrated cutoff; a late score alone is not dual-target evidence')
    fig.savefig(output/'late-echo.png',dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run','capture-protocol','protocol','previous-report','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    old = json.loads(args.previous_report.read_text(encoding='utf-8'))
    _, phases, _, provenance = load(args.run.resolve(),args.capture_protocol.resolve())
    if args.run.name != p['source_run'] or Path(old['run']).name != args.run.name:
        raise ValueError('source run identity mismatch')
    for field, expected in [('tof_raw_sha256',p['source_tof_sha256']),('camera_serial_sha256',p['source_camera_sha256'])]:
        if provenance[field] != expected or old['provenance'][field] != expected:
            raise ValueError('raw input identity mismatch')
    if not all(len([r for r in rows if r['block']==b])>=3 for rows in phases.values() for b in range(4)):
        raise ValueError('insufficient frame coverage')
    if [z['zone'] for z in old['zones']] != list(range(16)):
        raise ValueError('incomplete previous zone coverage')
    report = {'schema':p['schema'],'protocol':p,'protocol_sha256':sha(args.protocol),
              'script_sha256':sha(__file__),'loader_sha256':sha(Path(__file__).with_name('cnh_components.py')),
              'previous_report_sha256':sha(args.previous_report),'provenance':provenance,
              'run':str(args.run.resolve()),'rgb_review':old['rgb_review'],
              'phase_counts':{k:len(v) for k,v in phases.items()},
              'zones':[evaluate_zone(z,phases,p) for z in old['zones']]}
    report['eligible_zones'] = [z['zone'] for z in report['zones'] if z['eligible']]
    report['passing_zones'] = [z['zone'] for z in report['zones'] if z['verdict']=='EXPLORATORY_TAIL_EXCESS']
    report['execution_review_good'] = execution_review_good(old, args.run.name)
    report['verdict'] = ('NOT_EVALUABLE' if not report['execution_review_good'] else
                         'EXPLORATORY_TAIL_EXCESS' if report['passing_zones'] else
                         'NOT_SUPPORTED' if report['eligible_zones'] else 'NOT_EVALUABLE')
    report['compute'] = {'backend':'CPU','reason':'TASK_NOT_GPU_SUITABLE: 24-bin scalar contrasts, no training',
                         'seconds':time.perf_counter()-start}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plot(report,args.output)
    print(json.dumps({k:report[k] for k in ('verdict','eligible_zones','passing_zones','compute')}))


if __name__ == '__main__':
    main()
