"""Three-level stopping on existing approach sequences; no render/materialization.

One threshold per arm/budget is calibrated on odd-unit nominal clear episodes,
then held fixed in both heading versions. Episode truth is evaluated at the
fixed 0.9 m stop deadline, independent of an arm's first alarm. The heading
version is counterfactual label sensitivity, not a physically replayed path.
Run only after the margin-label result and route log have been pushed.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import norm

import cnh_graded_alert as GA
import cnh_near_range as NR
import cnh_structure_space as SS

OUT = SS.WORK/'cnh-three-level-sequence-20261002'
MARGIN = SS.WORK/'cnh-margin-labels-20261002'
BUDGETS = (.5, 1., 2.)
DRAWS = 400
HEADING_SEED = 2026100201
BOOT_SEED = 2026100203
BOOTSTRAPS = 1000
REFERENCE_M = .9
OFFSET_SUPPORT = (-.10, .15)
RULE = ('Primary: at 1 clear false stop/min, expansion minus NEAR timely stop rate '
        'for contact 0-2 cm has paired unit-bootstrap 95% CI lower > 0 and '
        'contact >5 cm timely rate decline <= 2 percentage points. Report each '
        'heading version separately; no choosing sigma or expansion after results. '
        'A >5% truncated bounded-bin cell is not interpreted. If both upstream '
        'expansions are THRESHOLD_SUFFICES, run NEAR only and make no expansion claim.')


def git(*args):
    return subprocess.check_output(['git', '-C', str(SS.ROOT), *args], text=True, encoding='utf8').strip()


def upstream(commit):
    """Require completed evidence and its already-pushed route-log receipt."""
    full = git('rev-parse', '--verify', commit+'^{commit}')
    subprocess.run(['git', '-C', str(SS.ROOT), 'merge-base', '--is-ancestor', full, 'origin/master'], check=True)
    changed = git('diff-tree', '--no-commit-id', '--name-only', '-r', full).splitlines()
    route = 'research/active/dtr-r0/RUNS.md'
    if route not in changed:
        raise ValueError('margin commit must contain the completed RUNS.md receipt')
    log = git('show', full+':'+route)
    receipt_rows = [line for line in log.splitlines()
                    if 'cnh_margin_labels.py' in line and 'cnh-margin-labels-20261002' in line
                    and ('THRESHOLD_SUFFICES' in line or 'HELPS' in line) and 'NOT_RUN' not in line]
    if not receipt_rows:
        raise ValueError('pushed route log lacks margin result identity/verdict')
    margin_source = 'research/active/dtr-r0/nearfield/cnh_margin_labels.py'
    committed_blob = git('rev-parse', full+':'+margin_source)
    if committed_blob != git('hash-object', str(SS.ROOT/margin_source)):
        raise ValueError('working margin script differs from the pushed result commit')
    result = json.loads((MARGIN/'result.json').read_text(encoding='utf8'))
    if result.get('status') != 'COMPLETE':
        raise ValueError('margin result is not COMPLETE')
    verdicts = {a: result['primary'][a]['verdict'] for a in ('M3', 'M8')}
    if any(v not in ('HELPS', 'THRESHOLD_SUFFICES') for v in verdicts.values()):
        raise ValueError('unknown upstream verdict')
    arms = ['NEAR'] if all(v == 'THRESHOLD_SUFFICES' for v in verdicts.values()) else ['NEAR', 'M3', 'M8']
    return arms, dict(commit=full, origin_master=git('rev-parse', 'origin/master'), verdicts=verdicts,
                      committed_script_blob=committed_blob, completed_run_rows=receipt_rows,
                      result_sha256=SS.sha(MARGIN/'result.json'))


def smooth(logit):
    score = np.empty_like(logit, dtype=np.float64)
    for t in range(logit.shape[2]):
        w = SS.WEIGHTS[-min(5, t+1):]
        score[:, :, t] = np.tensordot(logit[:, :, t+1-len(w):t+1], w/w.sum(), axes=([2], [0]))
    return score


def frame_scores(arm):
    """Reuse NEAR cache; other arms score only the already-voxelized frames."""
    path = NR.OUT/'graded_frame_scores.npz' if arm == 'NEAR' else OUT/f'frame_scores_{arm}.npz'
    if path.exists():
        with np.load(path) as cache:
            logit = cache['logit']
    else:
        import torch
        from cnh_cvr_pilot import CVR
        from cnh_cvr_projection import query_masks
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
        nets = []
        try:
            for seed in range(5):
                net = CVR().cuda()
                net.load_state_dict(torch.load(MARGIN/'models'/arm/f'model_seed{seed}.pt', map_location='cuda', weights_only=True))
                nets.append(net.eval())
            logit = np.full((96, 40, len(GA.ALL), 2), np.nan, np.float32)
            unit_pos = {u: i for i, u in enumerate(NR.SPLITS['evaluation'])}
            frame_pos = {int(f): i for i, f in enumerate(GA.ALL)}
            for name in ('evaluation_early', 'evaluation'):
                xs, meta = GA._parts(name)
                offset = 0
                with torch.no_grad():
                    for x in xs:
                        for start in range(0, len(x), 128):
                            b = SS.prep(torch, x[start:start+128], masks)
                            pred = torch.stack([n(b) for n in nets]).mean(0).cpu().numpy()
                            sl = slice(offset+start, offset+start+len(pred))
                            for j, (u, c, f) in enumerate(zip(meta['unit'][sl], meta['config'][sl], meta['frame'][sl])):
                                logit[unit_pos[int(u)], int(c), frame_pos[int(f)]] = pred[j]
                        offset += len(x)
                print('inference', arm, name, flush=True)
            np.savez_compressed(path, logit=logit)
        finally:
            nets.clear()
            torch.cuda.empty_cache()
    if logit.shape != (96, 40, 13, 2) or not np.isfinite(logit).all():
        raise ValueError('incomplete/incorrect frame cache: '+str(path))
    score = smooth(logit)
    final_path = NR.OUT/'scores_NEAR.npz' if arm == 'NEAR' else MARGIN/f'scores_{arm}.npz'
    with np.load(final_path) as frozen:
        discrepancy = max(float(np.abs(score[i, :, -1]-frozen[str(u) if arm == 'NEAR' else f'near|{u}']).max())
                          for i, u in enumerate(NR.SPLITS['evaluation']))
    if discrepancy >= 1e-4:
        raise ValueError(f'{arm} final-frame smoothing fails frozen score parity: {discrepancy}')
    return score.transpose(0, 1, 3, 2).reshape(-1, 13), dict(path=str(path), sha256=SS.sha(path), final_score_max_abs_error=discrepancy)


def metadata():
    units, offsets, final_ranges = [], [], []
    for u in NR.SPLITS['evaluation']:
        d = SS.read(NR.OUT/'features/evaluation'/f'unit{u}.npz')
        scenes = NR.scenes_for(u)
        labs = d['labels'].reshape(40, 16, 6)[:, -1, 2:4]
        for c in range(40):
            tq = int(d['group'][c])
            if labs[c, 1-tq] != 0:
                raise ValueError('other-height query is not clear in stored truth')
            for q in (0, 1):
                units.append(u)
                offsets.append(-float(d['margin'][c]) if q == tq else np.nan)
                final_ranges.append(float(scenes[c]['meta']['range']))
    units, offsets, final_ranges = map(np.asarray, (units, offsets, final_ranges))
    return units, offsets, final_ranges, final_ranges[:, None]+GA.STEP*(15-GA.ALL)


def truth_weights(offsets, z, sigma):
    """Integrate paired heading draws; no arm/score/stop time is an input."""
    actual = offsets[:, None]+REFERENCE_M*np.tan(np.deg2rad(sigma)*z)
    other = np.isnan(offsets)
    return {
        'contact0-2cm': ((actual > 0) & (actual <= .02)).mean(1),
        'contact2-5cm': ((actual > .02) & (actual <= .05)).mean(1),
        'contact>5cm': (actual > .05).mean(1),
        'pass0-10cm': ((actual >= -.10) & (actual <= 0)).mean(1),
        'clear': other.astype(float)+(actual < -.10).mean(1),
    }


def calibrate(score, clear, budget):
    """Lowest empirical operating threshold within budget, handling ties exactly."""
    maxima = score[clear].max(1).astype(np.float64)
    if not len(maxima):
        raise ValueError('no calibration clear episodes')
    minutes = len(maxima)*len(GA.ALL)*GA.FRAME_S/60
    allowed = int(np.floor(budget*minutes+1e-12))
    ordered = np.sort(maxima)[::-1]
    threshold = float(np.nextafter(ordered[allowed], np.inf)) if allowed < len(maxima) else float(ordered[-1])
    count = int((maxima >= threshold).sum())
    assert count/minutes <= budget+1e-12
    return threshold, dict(clear_episodes=len(maxima), clear_minutes=minutes, stops=count, stops_per_min=count/minutes)


def first_stops(score, threshold, ranges):
    alarm = score >= threshold
    stopped = alarm.any(1)
    first = alarm.argmax(1)
    stop_range = ranges[np.arange(len(ranges)), first]
    timely = stopped & (stop_range >= GA.SAFE)
    lead = (stop_range-.5)/.8
    return stopped, timely, lead


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]


def unit_totals(values, unit_index, n_units):
    return np.bincount(unit_index, weights=values, minlength=n_units)


def rates(num, den, boot):
    with np.errstate(divide='ignore', invalid='ignore'):
        sampled = (boot@num)/(boot@den)
    return dict(value=float(num.sum()/den.sum()) if den.sum() else None, ci95=interval(sampled)), sampled


def weighted_median(values, weights):
    keep = weights > 0
    values, weights = np.asarray(values)[keep], np.asarray(weights)[keep]
    if not len(weights):
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    half = weights.sum()/2
    index = int(np.searchsorted(cumulative, half))
    if index+1 < len(order) and np.isclose(cumulative[index], half, rtol=1e-12, atol=1e-12):
        return float((values[order[index]]+values[order[index+1]])/2)
    return float(values[order[index]])


def summarize(weights, approach, stopped, timely, lead, ui, boot):
    metrics, sampled_rates = {}, {}
    for category, probability in weights.items():
        p = probability if category == 'clear' else probability*approach
        den = unit_totals(p, ui, boot.shape[1])
        numerator = unit_totals(p*stopped, ui, boot.shape[1])
        common = dict(expected_episodes=float(p.sum()), contributing_episodes=int((p > 0).sum()),
                      contributing_units=int((den > 0).sum()), episode_draw_pairs=int(round(p.sum()*DRAWS)))
        if category == 'clear':
            minutes = den*len(GA.ALL)*GA.FRAME_S/60
            rate, _ = rates(numerator, minutes, boot)
            metrics[category] = dict(**common, expected_stops=float(numerator.sum()), clear_minutes=float(minutes.sum()), false_stops_per_min=rate)
            continue
        stop_rate, _ = rates(numerator, den, boot)
        timely_num = unit_totals(p*timely, ui, boot.shape[1])
        timely_rate, sampled_rates[category] = rates(timely_num, den, boot)
        keep = (p > 0) & stopped
        median = weighted_median(lead[keep], p[keep])
        median_boot = [weighted_median(lead[keep], p[keep]*row[ui[keep]]) for row in boot]
        metrics[category] = dict(**common, expected_stops=float(numerator.sum()), expected_timely_stops=float(timely_num.sum()),
                                stop_rate=stop_rate, timely_rate=timely_rate,
                                median_lead_to_0p5m_s=dict(value=median, ci95=interval([x for x in median_boot if x is not None])))
    return metrics, sampled_rates


def truncated_mass(sigma, lo, hi):
    scale = REFERENCE_M*np.tan(np.deg2rad(sigma))
    centre = (lo+hi)/2
    return 0. if scale == 0 else float(norm.cdf((OFFSET_SUPPORT[0]-centre)/scale)+norm.sf((OFFSET_SUPPORT[1]-centre)/scale))


def run(commit):
    arms, receipt = upstream(commit)
    if (OUT/'result.json').exists():
        raise FileExistsError('completed result already exists; do not silently replace evidence')
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(rule=RULE, upstream=receipt, arms=arms, budgets=BUDGETS, sigmas_deg=[0., 1.], truth_reference_m=REFERENCE_M,
                truth_formula='nominal intrusion + 0.9*tan(delta); one constant delta per episode/draw',
                thresholds='one pooled HEAD/BODY threshold per arm/budget; odd-unit sigma=0 clear calibration; frozen across sigma',
                approach='final nominal range <=1.0 m, same as graded alert; timely alarm at nominal range >=0.9 m',
                heading_draws=DRAWS, heading_seed=HEADING_SEED, bootstrap_seed=BOOT_SEED, unit_bootstraps=BOOTSTRAPS,
                bootstrap='48 even evaluation units, paired across arms; conditional on fixed odd-unit calibration thresholds',
                lead='(nominal range at first alarm - 0.5 m)/0.8 m/s, conditional on stopping; not human response time',
                clear_count='one first stop maximum per sequence/query, full 13-frame exposure even after stopping',
                script_sha256=SS.sha(__file__), limits=['consumed Development; short simulated sequence/query exposure, not real walking burden',
                    'heading changes truth only, not sensor inputs or geometry; 1 degree is assumed, not measured',
                    'nominal offset support [-10,+15] cm; no nominal same-height farther-clear episodes',
                    '>5cm category is conditional on the finite observed support; no unbounded population claim',
                    '95% percentile unit-bootstrap intervals do not include retraining or threshold-calibration uncertainty'])
    SS.save(OUT/'PLAN.json', plan)
    units, offsets, final_ranges, ranges = metadata()
    z = np.zeros((len(units), DRAWS))
    target = np.isfinite(offsets)
    z[target] = np.random.default_rng(HEADING_SEED).standard_normal((int(target.sum()), DRAWS))
    ev = units % 2 == 0
    cal = ~ev
    eval_units, ui = np.unique(units[ev], return_inverse=True)
    boot_rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(boot_rng.integers(len(eval_units), size=len(eval_units)), minlength=len(eval_units)) for _ in range(BOOTSTRAPS)])
    truths = {sig: truth_weights(offsets[ev], z[ev], sig) for sig in (0., 1.)}
    cells, samples, provenance = {}, {}, {}
    for arm in arms:
        score, provenance[arm] = frame_scores(arm)
        for budget in BUDGETS:
            threshold, calibration = calibrate(score, cal & np.isnan(offsets), budget)
            stopped, timely, lead = first_stops(score[ev], threshold, ranges[ev])
            for sigma, weights in truths.items():
                metrics, sample = summarize(weights, final_ranges[ev] <= 1., stopped, timely, lead, ui, boot)
                if sigma == 0:
                    for metric in metrics.values():
                        metric['n'] = int(round(metric['expected_episodes']))
                        metric['stops'] = int(round(metric['expected_stops']))
                        if 'expected_timely_stops' in metric:
                            metric['timely_stops'] = int(round(metric['expected_timely_stops']))
                key = f'{arm}|{budget}|{sigma}'
                cells[key] = dict(threshold=threshold, calibration=calibration, metrics=metrics)
                samples[key] = sample
            print('analyzed', arm, budget, flush=True)
    truncation = {str(s): {name: dict(kernel_mass_outside_support=truncated_mass(s, lo, hi),
                                     interpretable=truncated_mass(s, lo, hi) <= .05)
                          for name, lo, hi in [('contact0-2cm', 0., .02), ('contact2-5cm', .02, .05)]} for s in truths}
    comparisons = {}
    for sigma in truths:
        for arm in arms[1:]:
            k, base = f'{arm}|1.0|{sigma}', f'NEAR|1.0|{sigma}'
            m, b = cells[k]['metrics'], cells[base]['metrics']
            delta = m['contact0-2cm']['timely_rate']['value']-b['contact0-2cm']['timely_rate']['value']
            ci = interval(samples[k]['contact0-2cm']-samples[base]['contact0-2cm'])
            d5 = m['contact>5cm']['timely_rate']['value']-b['contact>5cm']['timely_rate']['value']
            valid = truncation[str(sigma)]['contact0-2cm']['interpretable']
            verdict = ('NOT_INTERPRETABLE_TRUNCATED' if not valid else
                       'SEQUENCE_BENEFIT' if ci[0] is not None and ci[0] > 0 and d5 >= -.02 else 'SEQUENCE_BENEFIT_NOT_ESTABLISHED')
            comparisons[f'{arm}|{sigma}'] = dict(delta_contact02=delta, paired_unit_ci95=ci, delta_contact_gt5=d5, verdict=verdict)
    result = dict(status='COMPLETE', rule=RULE, plan=plan, frame_scores=provenance, cells=cells, comparisons=comparisons,
                  bounded_bin_truncation=truncation, evaluation_units=eval_units.tolist(),
                  primary_status='NEAR_ONLY_UPSTREAM_THRESHOLD_SUFFICES' if len(arms) == 1 else 'SEE_PER_SIGMA_COMPARISONS')
    SS.save(OUT/'result.json', result)
    print(json.dumps(dict(primary_status=result['primary_status'], comparisons=comparisons), indent=2), flush=True)


def check():
    """Focused invariants for the decision-changing truth and threshold logic."""
    offsets = np.array([.01, .03, .07, 0., -.05, -.1, -.101, np.nan])
    z = np.ones((len(offsets), DRAWS))
    nominal = truth_weights(offsets, z, 0.)
    assert np.allclose(sum(nominal.values()), 1.)
    assert np.array_equal(nominal['clear'], [0, 0, 0, 0, 0, 0, 1, 1])
    shifted = truth_weights(offsets, z, 1.)
    assert shifted['contact2-5cm'][0] == 1
    assert np.array_equal(shifted['clear'], nominal['clear']*np.array([1, 1, 1, 1, 1, 1, 0, 1]))
    x = np.array([[0., 2.], [1., 2.], [0., 3.]])
    t, c = calibrate(x, np.ones(3, dtype=bool), 1/(3*len(GA.ALL)*GA.FRAME_S/60))
    assert 2 < t <= 3 and c['stops'] == 1
    stopped, timely, _ = first_stops(x, t, np.array([[1., .8], [1., .8], [1., .9]]))
    assert np.array_equal(stopped, [False, False, True]) and timely[-1]
    assert np.array_equal(unit_totals(np.array([2., 4., 3.]), np.array([0, 0, 1]), 2), [6., 3.])
    assert weighted_median(np.array([1., 2., 3., 4.]), np.ones(4)) == np.median([1., 2., 3., 4.])
    assert weighted_median(np.array([1., 2., 3.]), np.array([1., 0., 1.])) == 2.
    logits = np.arange(6., dtype=float).reshape(1, 1, 6, 1)
    assert smooth(logits)[0, 0, 0, 0] == 0
    assert np.isclose(smooth(logits)[0, 0, -1, 0], np.dot(np.arange(1., 6.), SS.WEIGHTS)/SS.WEIGHTS.sum())
    print('focused truth, threshold ties, smoothing, deadline and cluster-total checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('check', 'run'), required=True)
    parser.add_argument('--margin-commit')
    args = parser.parse_args()
    if args.stage == 'check':
        check()
    else:
        if not args.margin_commit:
            parser.error('--margin-commit is required after upstream result is pushed')
        run(args.margin_commit)
