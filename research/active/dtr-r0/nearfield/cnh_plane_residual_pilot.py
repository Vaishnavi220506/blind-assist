"""Paired target-preservation/plane-mismatch pilot; no training or gate claims.

Reuses consumed Development scene/pose units 86000..86011, generating declared
counterfactual observations. Evaluator-only signal is full minus absent-target
expected response, including occlusion. Fitter receives CNH+ambient only.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'tools'))
import cnh_proposal_attribution_scenes as S
from cnh_route_sensor import synthesize_response
from cnh_plane_residual import PlaneFitter
import cnh_structure_space as SS
from cnh_expected_separability import target_at

STRESSES = ('nominal', 'pulse', 'range_offset', 'crosstalk', 'reflectance', 'nonplanar')
CONTEXTS = ('no_panel', 'sidewall', 'large_panel')


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def render(boxes, pose, params, seed):
    rays, weights = S.ray_grid()
    hits = S.raycast_boxes(pose[:3, 3], rays@pose[:3, :3].T, boxes)
    response = synthesize_response(hits['distance'], hits['rho'], hits['cos'], weights, params=params, seed=seed)
    return response['histogram'].reshape(8, 8, 16, 8).sum(-1), response['ambient']


def background(unit, context, stress, target, ref):
    rng = np.random.default_rng([2026100101, unit])
    if context == 'no_panel':
        wall = []
    elif context == 'sidewall':
        sign = target['side']
        xx = (.44, 1.) if sign == 1 else (-1., -.44)
        wall = [dict(lo=[xx[0], -.3, .65], hi=[xx[1], 1.5, 3.8], rho=.60)]
    else:
        shape = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.08, .15)))
        area = SS.EDGES[4]+rng.uniform()*(SS.EDGES[5]-SS.EDGES[4])
        wall = SS.panel_scene(unit, 0, ref, target, shape, area)['boxes'][1:2]
    if stress in ('reflectance', 'nonplanar') and wall:
        b = wall[0]
        pieces = []
        for j in range(4):
            lo, hi = np.array(b['lo'], float), np.array(b['hi'], float)
            lo[2] = b['lo'][2]+j*(b['hi'][2]-b['lo'][2])/4
            hi[2] = b['lo'][2]+(j+1)*(b['hi'][2]-b['lo'][2])/4
            if stress == 'nonplanar':
                shift = target['side']*[0, .08, -.05, .12][j]
                lo[0] += shift; hi[0] += shift
            rho = [.30, .65, .40, .55][j] if stress == 'reflectance' else b['rho']
            pieces.append(dict(lo=lo.tolist(), hi=hi.tolist(), rho=rho))
        wall = pieces
    return wall+[SS.FLOOR, SS.BACK]


def parameters(nominal, stress):
    changes = dict(pulse=dict(pulse_sigma_bins=3.2, tail_mass=.2),
        range_offset=dict(range_zero_m=.075), crosstalk=dict(neighbour_leak=.08, crosstalk_fraction=.05))
    return replace(nominal, **changes.get(stress, {}))


def summarize(rows):
    out = {}
    for context in CONTEXTS:
        for stress in STRESSES:
            rr = [r for r in rows if r['context'] == context and r['stress'] == stress]
            if not rr:
                continue
            visible = [r for r in rr if r['signed_retention'] is not None]
            background_rows = {r['unit']: r for r in rr}.values()
            background_values = [r['background_removed'] for r in background_rows]
            out[f'{context}|{stress}'] = dict(n=len(rr),
                visible=len(visible), invisible=len(rr)-len(visible),
                background_removed=dict(n_units=len(background_values), median=float(np.median(background_values)), min=float(min(background_values))),
                **{key: dict(median=float(np.median([r[key] for r in visible])) if visible else None,
                            min=float(min(r[key] for r in visible)) if visible else None) for key in
                    ('signed_retention', 'positive_retention', 'noisy_signed_retention')},
                signed_below_80pct=sum(r['signed_retention'] < .8 for r in visible),
                noisy_signed_below_80pct=sum(r['noisy_signed_retention'] < .8 for r in visible))
    return out


def boundary_summary(rows, arrays):
    """Post-hoc known-signal inside/outside separation, not detection accuracy."""
    grouped = {}
    for i, row in enumerate(rows):
        key = (row['unit'], row['context'], row['stress'])
        grouped.setdefault(key, {})[row['inside']] = i
    out = {}
    for (_, context, stress), ids in grouped.items():
        i, j = ids[True], ids[False]
        delta = arrays[i, 0].astype(float)-arrays[j, 0].astype(float)
        norm = np.square(delta).sum()
        key = f'{context}|{stress}'
        entry = out.setdefault(key, dict(n_pairs=0, zero_response_difference=0, retentions=[]))
        entry['n_pairs'] += 1
        if norm < 1e-12:
            entry['zero_response_difference'] += 1
        else:
            entry['retentions'].append(float((delta*(arrays[i, 2]-arrays[j, 2])).sum()/norm))
    for entry in out.values():
        values = entry.pop('retentions')
        entry.update(evaluable_pairs=len(values), median=float(np.median(values)) if values else None,
                     min=float(min(values)) if values else None)
    return out


def main(output, units, stresses):
    output.mkdir(parents=True, exist_ok=False)
    nominal, anchor = S.nominal_parameters()
    source_hash = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), HERE/'cnh_plane_residual.py', S.SOURCE/'cnh_route_sensor.py', HERE/'cnh_proposal_attribution_scenes.py']}
    save(output/'request.json', dict(role='consumed Development counterfactual mechanism diagnostic',
        units=units, contexts=CONTEXTS, stresses=stresses, margin_m=.045, target_states=['absent', 'inside', 'outside'],
        exposures_per_state=1, repeats=['expected', 'one noisy exposure'],
        fitting='one infinite plane, nominal fixed electronics/IRF, free reflectance, each frame independently',
        criterion='descriptive retention below .8 marks a lossy case; no classification or promotion gate',
        nominal_params=asdict(nominal), anchor=anchor, source_sha256=source_hash))
    fitter = PlaneFitter(nominal)
    ref = S.make_scenes(units[0])[0]
    probe, ambient = render(ref['boxes'], ref['poses'][-1], nominal, 1)
    backend = fitter.select_backend(probe, ambient)
    backend['local_refinement'] = 'scipy/cpu: TASK_NOT_GPU_SUITABLE (small sequential bounded nonlinear fits)'
    save(output/'backend.json', backend)
    rows = []
    started = time.perf_counter()
    retained = []
    for ui, unit in enumerate(units):
        ref = S.make_scenes(unit)[0]
        rng = np.random.default_rng([2026100102, unit])
        target = SS.draw_target(rng, -.045, unit % 2)
        pose = ref['poses'][-1]
        for context in CONTEXTS:
            for stress in stresses:
                boxes = background(unit, context, stress, target, ref)
                p = parameters(nominal, stress)
                seed = 2026100102000+1000*ui+100*CONTEXTS.index(context)+10*STRESSES.index(stress)
                bg, ambient = render(boxes, pose, replace(p, noise_scale=0), seed)
                noisy_bg, _ = render(boxes, pose, p, seed+1)
                fit_bg = fitter.fit(bg, ambient)
                fit_noisy_bg = fitter.fit(noisy_bg, ambient)
                baseline = bg-fitter.electronics
                removed = 1-np.linalg.norm(fit_bg['residual'])/max(np.linalg.norm(baseline), 1e-12)
                for sign in (-1, 1):
                    full_boxes = [target_at(target, sign*.045)]+boxes
                    full, _ = render(full_boxes, pose, replace(p, noise_scale=0), seed)
                    noisy_full, _ = render(full_boxes, pose, p, seed+2+(sign+1)//2)
                    fit = fitter.fit(full, ambient)
                    fit_noisy = fitter.fit(noisy_full, ambient)
                    delta = full-bg
                    norm = float(np.square(delta).sum())
                    remaining = fit['residual']-fit_bg['residual']
                    noisy_remaining = delta-(fit_noisy['plane']-fit_noisy_bg['plane'])
                    positive_delta = np.maximum(full-fitter.electronics-fit['plane'], 0)-np.maximum(bg-fitter.electronics-fit_bg['plane'], 0)
                    labels = S.labels_for(full_boxes, ref['travel'][-1:])[0]
                    assert int(labels[target['group']]) == int(sign < 0)
                    np.testing.assert_allclose(fit['residual']+fit['plane']+fit['electronics'], full, rtol=1e-12, atol=1e-12)
                    row = dict(unit=unit, context=context, stress=stress, inside=bool(sign < 0), group=target['group'],
                        mode=unit % 3, delta_l2=float(np.sqrt(norm)), background_removed=float(removed),
                        signed_retention=float((delta*remaining).sum()/norm) if norm > 1e-12 else None,
                        positive_retention=float((delta*positive_delta).sum()/norm) if norm > 1e-12 else None,
                        noisy_signed_retention=float((delta*noisy_remaining).sum()/norm) if norm > 1e-12 else None,
                        negative_delta_fraction=float(np.square(np.minimum(delta, 0)).sum()/norm) if norm > 1e-12 else None,
                        residual_direction_cosine=float((delta*remaining).sum()/(np.sqrt(norm)*max(np.linalg.norm(remaining), 1e-12))) if norm > 1e-12 else None,
                        fitted_normal=fit['normal'].tolist(), fitted_offset_m=fit['offset_m'], fitted_amplitude=fit['amplitude'],
                        fit_elapsed_s=fit['elapsed_s'], noisy_fit_elapsed_s=fit_noisy['elapsed_s'],
                        optimizer_success=fit['optimizer_success'], noisy_optimizer_success=fit_noisy['optimizer_success'])
                    rows.append(row)
                    retained.append(np.stack([full, fit['plane'], fit['residual'], bg, delta]))
        save(output/'progress.json', dict(completed_units=ui+1, total_units=len(units), rows=len(rows), elapsed_s=time.perf_counter()-started))
        print('unit', unit, 'rows', len(rows), 'elapsed_s', round(time.perf_counter()-started, 2), flush=True)
    # One-wall controls: a genuine planar hazard can be fully explained by the wall model.
    controls = []
    for x in (.22, .55):
        wall = [dict(lo=[x, -.3, .35], hi=[x+.3, 1.5, 3.8], rho=.6)]
        pose = np.eye(4)
        h, a = render(wall, pose, replace(nominal, noise_scale=0), 17)
        fit = fitter.fit(h, a)
        labels = S.labels_for(wall, np.eye(4)[None])[0]
        controls.append(dict(wall_near_x=x, query_labels=labels.tolist(),
            explained_l2_fraction=float(1-np.linalg.norm(fit['residual'])/np.linalg.norm(h-fitter.electronics)),
            fitted_normal=fit['normal'].tolist(), fitted_offset_m=fit['offset_m']))
    arrays = np.stack(retained).astype(np.float32)
    np.save(output/'decompositions.npy', arrays)
    save(output/'results.json', dict(role='Development mechanism diagnostic, not detector accuracy', rows=rows,
        summary=summarize(rows), controls=controls, boundary_summary=boundary_summary(rows, arrays), elapsed_s=time.perf_counter()-started,
        formula='retention=<full-bg,(full-plane_full)-(bg-plane_bg)>/||full-bg||^2; electronics cancels',
        noisy_formula='use expected full-bg, subtract difference of independently noisy fitted planes; not SNR improvement',
        limits=['one final frame; no temporal/pose test', 'one noisy draw per state, no confidence intervals',
            'infinite plane is an approximation to finite opaque boxes', 'no training, LOFO BER or PARTIAL comparison',
            'raw+plane+signed residual reconstructs input; preservation does not prove task gain']))
    print(json.dumps(summarize(rows), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--units', type=int, default=12)
    parser.add_argument('--stresses', nargs='+', choices=STRESSES, default=STRESSES)
    args = parser.parse_args()
    if args.units < 1 or args.units > 24:
        parser.error('This small diagnostic supports 1..24 reused units')
    if args.output.exists():
        parser.error('Output already exists; choose a new run directory to preserve prior results')
    with threadpool_limits(limits=2):
        try:
            main(args.output, list(range(86000, 86000+args.units)), args.stresses)
            save(args.output/'terminal.json', dict(status='complete'))
        except BaseException as e:
            if args.output.exists() and not (args.output/'terminal.json').exists():
                save(args.output/'terminal.json', dict(status='failed', error=repr(e)))
            raise
