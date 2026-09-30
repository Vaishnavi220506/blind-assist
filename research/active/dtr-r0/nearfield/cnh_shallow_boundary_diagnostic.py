"""Paired 15-mm boundary diagnostic on frozen FULL CVR; no training.

Each target/trajectory is reused across no wall, B0 and B4 walls. Scalar
readouts retain their occupancy direction. Vector probes are privileged
noiseless-reference matched directions, not information ceilings.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts.local/work/cnh-shallow-boundary-20261001'
UNITS = list(range(87000, 87024))
SMOKE_UNIT = 87999
CONDITIONS = ('no_panel', 'B0', 'B4')
K = 12
MARGIN = .015
LIMIT_SECONDS = 1200
REPS = ('raw8', 'z1_8', 'voxel15', 'logit15', 'raw12', 'logit5')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temp.replace(path)


def prepare():
    # Renderer establishes the retained v5 source before importing its features.
    import cnh_structure_space as SS
    import cnh_proposal_attribution_scenes as S
    from cnh_expected_separability import target_at
    from cnh_corridor_labels import labels_for_all
    import torch
    import cnh_track_a_gpu_readout as g
    import cnh_learned_features as F
    from cnh_track_a_readout import noisy_poses
    from cnh_cvr_pilot import CVR, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.30)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bias_path = ROOT / 'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'
    bias = np.load(bias_path)
    projector = BatchedProjector()
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    nets = []
    models = []
    for seed in range(3):
        path = SS.OUT / 'models/FULL' / f'model_seed{seed}.pt'
        net = CVR().cuda()
        net.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
        nets.append(net.eval())
        models.append(path)
    dependency_paths = [Path(__file__), bias_path, *models]
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('cnh_') and path and Path(path).suffix == '.py':
            dependency_paths.append(Path(path))
    sources = {str(p.resolve()): sha(p) for p in sorted(set(dependency_paths), key=str)}
    return dict(SS=SS, S=S, target_at=target_at, labels=labels_for_all, torch=torch,
                g=g, F=F, noisy=noisy_poses, relative=relative_transforms, bias=bias,
                projector=projector, masks=masks, nets=nets, sources=sources)


def scenes(ctx, unit):
    rng = np.random.default_rng([2026100101, unit])
    ref = ctx['S'].make_scenes(unit)[0]
    group = (unit - UNITS[0]) % 2
    target = ctx['SS'].draw_target(rng, -MARGIN, group)
    shape = dict(ctx['SS'].draw_shape(rng), same_side=1, gap=float(rng.uniform(.08, .15)))
    quantile = float(rng.uniform())
    result = {}
    for cond in CONDITIONS:
        context = []
        if cond != 'no_panel':
            b = int(cond[1:])
            area = ctx['SS'].EDGES[b] + quantile * (ctx['SS'].EDGES[b + 1] - ctx['SS'].EDGES[b])
            context = ctx['SS'].panel_scene(unit, 0, ref, target, shape, area)['boxes'][1:2]
        pair = []
        for sign in (-1, 1):
            boxes = [ctx['target_at'](target, sign * MARGIN)] + context + [ctx['SS'].FLOOR, ctx['SS'].BACK]
            lab = ctx['labels'](boxes, ref['travel'])[-1, [2, 3]]
            assert lab[group] == int(sign == -1) and lab[1 - group] == 0
            pair.append(dict(boxes=boxes, poses=ref['poses'], travel=ref['travel']))
        result[cond] = pair
    for sign in range(2):
        assert all(result[cond][sign]['boxes'][0] == result['no_panel'][sign]['boxes'][0] for cond in CONDITIONS)
    return result, group


def represent(ctx, scene, obs, noisy, group, check=False):
    torch, g = ctx['torch'], ctx['g']
    h, a = obs['hist'], obs['ambient']
    residual = g.T(h) - g.T(ctx['bias'])
    variance = 16 * g.T(a)[..., None] + g.T(ctx['bias']).clamp_min(0)
    z1_full = (residual / variance.clamp_min(1e-9).sqrt()).cpu().numpy()
    if check:
        tq = np.linalg.inv(scene['travel']) @ scene['poses']
        _, reference, _ = ctx['F'].sequence_features(h, a, ctx['bias'], tq, noisy)
        assert np.array_equal(z1_full, reference), 'fast z1 differs from retained implementation'
    z1 = z1_full.astype(np.float16)
    voxels = []
    for f in (11, 12, 13, 14, 15):
        voxel = ctx['projector'].sequence(z1[f - 7:f + 1],
                                         ctx['relative'](scene['poses'], scene['travel'], noisy, f))
        # Reproduce both float16 cache round trips in the actual scan pipeline.
        voxels.append(voxel.cpu().numpy().astype(np.float16))
    x = ctx['SS'].prep(torch, np.stack(voxels), ctx['masks'])
    with torch.no_grad():
        logits = torch.stack([net(x) for net in ctx['nets']]).mean(0).cpu().numpy().astype(np.float64)
    weighted = (logits * ctx['SS'].WEIGHTS[:, None]).sum(0) / ctx['SS'].WEIGHTS.sum()
    reps = dict(raw8=h[-8:].ravel().astype(np.float64),
                z1_8=z1[-8:].ravel().astype(np.float64),
                voxel15=x[-1, :3].cpu().numpy().ravel().astype(np.float64),
                logit15=np.array([logits[-1, group]]),
                raw12=h[-12:].ravel().astype(np.float64),
                logit5=np.array([weighted[group]]))
    assert all(np.isfinite(value).all() for value in reps.values())
    return reps, weighted


def threshold_reference():
    base = ROOT / 'artifacts.local/work/cnh-coverage-sweep-20260930'
    manifest_path = base / 'scene_manifest.json'
    pred_path = base / 'predictions/UB4.npz'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    with np.load(pred_path, allow_pickle=False) as z:
        pred = {key: z[key] for key in z.files}
    refs = []
    for group in range(2):
        reference = np.sort([float(pred[f"calib|{r['unit']}"][r['config'], group])
                             for r in manifest if r['split'] == 'calib' and not r['label_final'][group]])
        ranks = np.searchsorted(reference, reference, side='right')
        k = next(k for k in range(len(reference) + 2) if 10 * int((ranks >= k).sum()) <= len(reference))
        refs.append((reference, k))
    return refs, {str(p.resolve()): sha(p) for p in (manifest_path, pred_path)}


def one_pair(ctx, unit, cond, pair, group, reps_count, check=False):
    # Common random seeds across wall conditions; independent inside/outside seeds.
    base = 202610010000 + (unit - UNITS[0]) * 1000
    clean = []
    for sign, scene in enumerate(pair):
        clean.append(represent(ctx, scene, ctx['S'].render(scene, base + sign * 100, noise_scale=0),
                               scene['poses'], group, check=check)[0])
    directions = {}
    for rep in REPS:
        if rep.startswith('logit'):
            directions[rep] = np.ones(1)  # Fixed positive=occupied orientation, never sign-flipped.
        else:
            difference = clean[0][rep] - clean[1][rep]
            norm = np.linalg.norm(difference)
            directions[rep] = difference / norm if norm > 0 else np.zeros_like(difference)
    projected = {rep: [[], []] for rep in REPS}
    scores = [[], []]
    for sign, scene in enumerate(pair):
        for k in range(reps_count):
            seed = base + sign * 100 + k + 1
            obs = ctx['S'].render(scene, seed)
            values, weighted = represent(ctx, scene, obs, ctx['noisy'](scene['poses'], seed + 31, dt=.2), group)
            for rep in REPS:
                projected[rep][sign].append(float(values[rep] @ directions[rep]))
            scores[sign].append(weighted.tolist())
    diagnostics = {}
    for rep in REPS:
        p, q = (np.asarray(values) for values in projected[rep])
        sd = float(np.sqrt((p.var(ddof=1) + q.var(ddof=1)) / 2))
        diagnostics[rep] = dict(mean_in=float(p.mean()), mean_out=float(q.mean()), pooled_sd=sd,
                                signed_dprime=float((p.mean() - q.mean()) / sd) if sd > 0 else None,
                                clean_difference_zero=bool(np.array_equal(clean[0][rep], clean[1][rep])))
    return dict(unit=unit, group=group, mode=unit % 3, condition=cond, margin=MARGIN,
                K=reps_count, probes=diagnostics, projections=projected, weighted_logits=scores)


def summarize(rows, refs):
    summaries = {}
    for cond in CONDITIONS:
        rr = [r for r in rows if r['condition'] == cond]
        summaries[cond] = dict(units=len(rr), probes={})
        for rep in REPS:
            d = [r['probes'][rep]['signed_dprime'] for r in rr]
            assert all(value is not None for value in d), 'zero empirical variance'
            summaries[cond]['probes'][rep] = dict(median=float(np.median(d)),
                q25=float(np.percentile(d, 25)), q75=float(np.percentile(d, 75)),
                nonpositive_units=sum(value <= 0 for value in d))
        counts = dict(P=len(rr) * K, N_target_outside=len(rr) * K, N_other_height=len(rr) * K * 2,
                      fn=0, fp_target_outside=0, fp_other_height=0)
        for r in rr:
            group = r['group']
            for sign in range(2):
                values = np.asarray(r['weighted_logits'][sign])
                for g in range(2):
                    reference, k = refs[g]
                    pred = np.searchsorted(reference, values[:, g], side='right') >= k
                    if g != group:
                        counts['fp_other_height'] += int(pred.sum())
                    elif sign == 0:
                        counts['fn'] += int((~pred).sum())
                    else:
                        counts['fp_target_outside'] += int(pred.sum())
        summaries[cond]['frozen_workpoint'] = counts
    # Unit-paired comparisons; repeat noise draws are never bootstrap units.
    comparisons = {}
    rng = np.random.default_rng(2026100102)
    units = sorted({r['unit'] for r in rows})
    index = {(r['unit'], r['condition']): r for r in rows}
    samples = rng.integers(0, len(units), size=(2000, len(units)))
    for cond in ('B0', 'B4'):
        comparisons[cond] = {}
        for rep in REPS:
            delta = np.array([index[u, cond]['probes'][rep]['signed_dprime'] -
                              index[u, 'no_panel']['probes'][rep]['signed_dprime'] for u in units])
            boot = np.median(delta[samples], axis=1)
            comparisons[cond][rep] = dict(paired_median_delta=float(np.median(delta)),
                exploratory_ci95=np.percentile(boot, [2.5, 97.5]).tolist())
    return summaries, comparisons


def run(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ctx = prepare()
    if smoke:
        pair_map, group = scenes(ctx, SMOKE_UNIT)
        row = one_pair(ctx, SMOKE_UNIT, 'B4', pair_map['B4'], group, 2, check=True)
        save(OUT / 'smoke.json', dict(status='PASS', excluded_unit=SMOKE_UNIT,
             elapsed_s=time.monotonic() - started, estimated_pair_seconds=time.monotonic() - started,
             finite_projections=all(np.isfinite(v).all() for r in row['projections'].values() for v in r),
             z1_exact_match=True, backend='numpy CPU frozen renderer; CUDA projector and FULL inference'))
        print('smoke PASS', round(time.monotonic() - started, 1), 'seconds', flush=True)
        return
    lock = json.loads((OUT / 'freeze.json').read_text(encoding='utf8'))
    assert lock['plan_sha256'] == sha(OUT / 'PLAN.md')
    refs, calibration_hashes = threshold_reference()
    sources = dict(ctx['sources'], **calibration_hashes)
    assert sources == lock['sources'], 'input/source drift after freeze'
    assert not (OUT / 'terminal.json').exists() and not (OUT / 'pairs.jsonl').exists(), 'one-shot already started'
    save(OUT / 'request.json', dict(pid=os.getpid(), started_unix=time.time(), total_pairs=72,
         K=K, runtime_limit_seconds=LIMIT_SECONDS, sources=sources,
         device=ctx['torch'].cuda.get_device_name(), backend='numpy CPU frozen renderer; CUDA projector and FULL inference'))
    rows = []
    for unit in UNITS:
        pair_map, group = scenes(ctx, unit)
        for cond in CONDITIONS:
            if time.monotonic() - started > LIMIT_SECONDS:
                save(OUT / 'terminal.json', dict(status='BUDGET_STOP', completed_pairs=len(rows),
                     elapsed_s=time.monotonic() - started, pid=os.getpid(), remaining_pairs=72-len(rows)))
                return
            row = one_pair(ctx, unit, cond, pair_map[cond], group, K)
            rows.append(row)
            with (OUT / 'pairs.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            save(OUT / 'progress.json', dict(completed_pairs=len(rows), total_pairs=72, unit=unit,
                 condition=cond, elapsed_s=time.monotonic() - started, pid=os.getpid()))
            print('pair', len(rows), '/72', unit, cond, round(time.monotonic()-started, 1), flush=True)
    assert len(rows) == 72
    assert all(sha(path) == digest for path, digest in sources.items()), 'source changed during run'
    summary, comparisons = summarize(rows, refs)
    save(OUT / 'results.json', dict(scope='privileged shallow-boundary Development witness; not an information ceiling',
         plan_sha256=lock['plan_sha256'], sources=sources, summary=summary, comparisons=comparisons,
         completed_pairs=len(rows), independent_target_units=len(UNITS), K=K,
         elapsed_s=time.monotonic()-started))
    save(OUT / 'terminal.json', dict(status='complete', completed_pairs=72, pid=os.getpid(),
         elapsed_s=time.monotonic()-started))
    print('complete', round(time.monotonic()-started, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        run(args.smoke)
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        save(OUT / ('smoke_failure.json' if args.smoke else 'terminal.json'),
             dict(status='failed', error=repr(exc), pid=os.getpid()))
        raise
