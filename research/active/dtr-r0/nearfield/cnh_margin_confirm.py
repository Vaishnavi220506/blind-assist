"""New-unit, frozen-model margin-label comparison; EXPLORE, not blind/final.

Same near-range generator, with nominal intrusion support widened to [-.20,.15].
No training. Generation is chunked and voxel files have separate chunk owners.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_near_range as NR
import cnh_readout_fix as RF

OUT = SS.WORK/'cnh-margin-confirm-20261002'
SPLITS = {'calib': list(range(95000, 95048)), 'evaluation': list(range(96000, 96096))}
ARMS = ('NEAR', 'M3', 'M8')
MARGIN = SS.WORK/'cnh-margin-labels-20261002'
RULE = ('At per-arm/per-query calibration merged-clear 90th-percentile thresholds, compare '
        'M3 minus NEAR on evaluation contact 0-2 cm, pooled conditions and final range [0.6,2.1) m. '
        '1000 paired evaluation-unit bootstraps: 95% lower >0, contact >5 cm decrease <=2 pp, '
        'and full-range evaluation merged-clear FPR increase <=2 pp => M3_CONFIRMED; otherwise NOT_CONFIRMED. '
        'M8 is reported separately by the same rule, not selected. Heading 1 degree is secondary only. '
        'If M3_CONFIRMED, describe sequence stopping on these same units; no training or retuning.')


def model_paths(arm):
    return [(NR.OUT if arm == 'NEAR' else MARGIN)/'models'/arm/f'model_seed{s}.pt' for s in range(5)]


def scenes_for(unit, outside=.20):
    import cnh_proposal_attribution_scenes as S
    ref = NR.known(unit)[0]
    rng = np.random.default_rng([2026100105, int(unit)])
    out = []
    for c in range(40):
        cond = 'none' if c < 20 else 'corner'
        t = NR._target(rng, float(rng.uniform(-.15, outside)), (unit//6+c) % 2)
        if cond == 'none':
            scene = NR._none_scene(S, unit, c, ref, t)
        else:
            shape = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.08, .15)))
            scene = SS.panel_scene(unit, c, ref, t, shape, SS.EDGES[0]+float(rng.uniform())*(SS.EDGES[5]-SS.EDGES[0]))
        scene['family'] = cond
        scene['meta'] = dict(scene.get('meta', {}), cond=cond, range=t['z'])
        out.append(scene)
    return out


def plan():
    import torch
    assert len(set(sum(SPLITS.values(), []))) == 144
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('PLAN already exists; inspect/resume the existing run')
    OUT.mkdir(parents=True, exist_ok=True)
    models = {a: {str(p.relative_to(SS.ROOT)): SS.sha(p) for p in model_paths(a)} for a in ARMS}
    sources = [Path(__file__), Path(__file__).with_name('cnh_margin_confirm_evaluate.py'),
               Path(NR.__file__), Path(SS.__file__), Path(__file__).with_name('cnh_proposal_attribution_scenes.py')]
    SS.save(OUT/'PLAN.json', dict(rule=RULE, splits=SPLITS, arms=ARMS, model_sha256=models,
        source_sha256={p.name: SS.sha(p) for p in sources}, role='EXPLORE: new simulated units in existing generator family',
        primary_range=[.6, 2.1], clear_budget_range=[.6, 2.6], margins=[-.15, .20], configurations_per_unit=40,
        deployment_frames=SS.EVAL_FRAMES.tolist(), weights=SS.WEIGHTS.tolist(), threshold_quantile=.9,
        unit_bootstraps=1000, bootstrap_seed=2026100205, heading_seed=2026100201, heading_draws=400,
        fresh_id_audit=dict(script_used_splits='all nearfield cnh scripts checked, existing reservations through 94095',
            cn_h_work_roots=92, feature_files=1584, unique_existing_ids=1392, max_existing_id=94095,
            top_level_plan_result_files_searched=772, candidate_id_matches=0),
        backend=dict(device=torch.cuda.get_device_name(), rendering='same NumPy electronic renderer + CUDA feature extraction',
                     voxelization='CUDA BatchedProjector; 2 evaluation chunks, 1 calibration chunk',
                     inference='CUDA frozen 5-seed ensembles; no training'),
        limits=['new random units, same synthetic generator family; not independent real-world or protected confirmation',
                'clear mixture composition follows this generator; not deployment prevalence',
                'heading changes labels only, not observations or full swept-body geometry']))
    snap = OUT/'source'; snap.mkdir(exist_ok=True)
    for p in sources:
        (snap/p.name).write_bytes(p.read_bytes())


def generate(chunk, one_unit=None):
    import cnh_proposal_attribution as old
    import cnh_proposal_attribution_scenes as S
    from cnh_corridor_labels import labels_for_all
    torch, _, features, _, _, noisy_fn, _, _ = old.setup()
    bias = np.load(old.DATA/'readouts-gpu/primary-mount-10-snr6/bias.npy')
    k, n = map(int, chunk.split('/'))
    pairs = [(split, u) for split, units in SPLITS.items() for u in units]
    selected = [p for p in pairs if p[1] == one_unit] if one_unit is not None else pairs[k::n]
    started = time.monotonic()
    for split, u in selected:
        dest = OUT/'features'/split/f'unit{u}.npz'
        manifest = OUT/'manifest_units'/split/f'unit{u}.json'
        if dest.exists() and manifest.exists():
            with np.load(dest) as cached:
                assert cached['z1'].shape[0] == 640 and int(cached['unit']) == u
            continue
        rows, descriptions = [], []
        for scene in scenes_for(u):
            seed = 2026092900+u*1000+scene['config']*10
            obs = S.render(scene, seed)
            noisy = noisy_fn(scene['poses'], seed+7, dt=.2)
            tq = np.linalg.inv(scene['travel'])@scene['poses']
            z4, z1, support = features.sequence_features(obs['hist'], obs['ambient'], bias, tq, noisy)
            labels = labels_for_all(scene['boxes'], scene['travel'])
            if not np.array_equal(labels[:, [2, 3]], scene['labels']):
                closed = labels_for_all(scene['boxes'], scene['travel'], boundary='closed')
                assert np.array_equal(closed[:, [2, 3]], scene['labels']) and (labels <= closed).all()
            assert np.array_equal(labels[-1, [2, 3]], scene['labels'][-1])
            rows.append(dict(z4=z4.astype(np.float16), z1=z1.astype(np.float16), support=support,
                             labels=labels, scene=np.full(16, scene['config']), frame=np.arange(16)))
            descriptions.append(dict(split=split, unit=u, config=scene['config'], group=int(scene['group']),
                off=-float(scene['margin']), range=float(scene['meta']['range']), cond=scene['family'],
                labels=labels[-1, [2, 3]].tolist(), boxes=scene['boxes']))
        payload = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
        payload.update(unit=u, split=split, query_frame='travel', train_mask=np.zeros(640, bool),
            family=np.array([r['cond'] for r in descriptions]), margin=np.array([-r['off'] for r in descriptions]),
            group=np.array([r['group'] for r in descriptions]))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.partial.npz'); np.savez_compressed(tmp, **payload); tmp.replace(dest)
        SS.save(manifest, descriptions)
        SS.save(OUT/f'progress_generate_c{k}.json', dict(unit=u, split=split, elapsed_s=time.monotonic()-started))
        print('generate', split, u, round(time.monotonic()-started, 1), flush=True)
    torch.cuda.empty_cache()


def materialize(split, chunk, early=False):
    import torch
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import SHAPE
    torch.set_num_threads(2)
    k, n = map(int, chunk.split('/')); units = SPLITS[split][k::n]
    frames = np.arange(3, 11) if early else SS.EVAL_FRAMES
    name = split+('_early' if early else '')
    folder = OUT/'data'/name; folder.mkdir(parents=True, exist_ok=True)
    final = folder/f'features_c{k}.npy'; metadata = folder/f'metadata_c{k}.npz'
    if final.exists() and metadata.exists():
        with np.load(metadata) as m:
            assert len(m['unit']) == len(units)*40*len(frames)
        return
    temp = folder/f'features_c{k}.partial.npy'
    feats = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float16, shape=(len(units)*40*len(frames), 3, *SHAPE))
    meta = {key: [] for key in ('unit', 'config', 'frame')}
    projector = BatchedProjector(); offset = 0; started = time.monotonic()
    for u in units:
        d = RF.wait_complete(OUT/'features'/split/f'unit{u}.npz')
        for c in range(40):
            ids = np.flatnonzero(d['scene'] == c)
            assert np.array_equal(d['frame'][ids], np.arange(16))
            sensor, travel, noisy = motion_metadata(u, c)
            for f in frames:
                vox = projector.sequence(d['z1'][ids[max(0, f-7):f+1]], relative_transforms(sensor, travel, noisy, int(f)))
                feats[offset] = vox.cpu().numpy().astype(np.float16); offset += 1
            meta['unit'] += [u]*len(frames); meta['config'] += [c]*len(frames); meta['frame'] += frames.tolist()
        SS.save(OUT/f'progress_materialize_{name}_c{k}.json', dict(unit=u, elapsed_s=time.monotonic()-started))
        print('materialize', name, u, round(time.monotonic()-started, 1), flush=True)
    feats.flush(); del feats; temp.replace(final)
    np.savez_compressed(metadata, **{key: np.asarray(value) for key, value in meta.items()})


def assemble():
    rows = []
    for split, units in SPLITS.items():
        for u in units:
            data = json.loads((OUT/'manifest_units'/split/f'unit{u}.json').read_text(encoding='utf8'))
            assert len(data) == 40 and [r['config'] for r in data] == list(range(40))
            rows.extend(data)
    assert len(rows) == 144*40
    SS.save(OUT/'scene_manifest.json', rows)


def infer(early=False):
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
    for arm in ARMS:
        cache = OUT/f'frame_scores_{arm}{"_early" if early else ""}.npz'
        if cache.exists():
            continue
        nets = []
        try:
            for path in model_paths(arm):
                net = CVR().cuda(); net.load_state_dict(torch.load(path, weights_only=True, map_location='cuda')); nets.append(net.eval())
            raw_by_unit, scores = {}, {}
            for split, units in SPLITS.items():
                frames = np.arange(3, 11) if early else SS.EVAL_FRAMES
                values = np.full((len(units), 40, len(frames), 2), np.nan, np.float32)
                ui = {u: i for i, u in enumerate(units)}; fi = {int(f): i for i, f in enumerate(frames)}
                folder = OUT/'data'/(split+('_early' if early else ''))
                for part in sorted(folder.glob('features_c*.npy')):
                    assert '.partial' not in part.name
                    x = np.load(part, mmap_mode='r'); m = SS.read(part.with_name(part.name.replace('features_', 'metadata_').replace('.npy', '.npz')))
                    with torch.no_grad():
                        for start in range(0, len(x), 128):
                            b = SS.prep(torch, x[start:start+128], masks)
                            pred = torch.stack([net_(b) for net_ in nets]).mean(0).cpu().numpy()
                            for j in range(len(pred)):
                                loc = start+j
                                values[ui[int(m['unit'][loc])], int(m['config'][loc]), fi[int(m['frame'][loc])]] = pred[j]
                assert np.isfinite(values).all()
                for i, u in enumerate(units):
                    raw_by_unit[str(u)] = values[i]
                    if not early:
                        scores[str(u)] = (values[i].astype(np.float64)*SS.WEIGHTS[None, :, None]).sum(1)/SS.WEIGHTS.sum()
                print('infer', arm, split, flush=True)
            np.savez_compressed(cache, **raw_by_unit)
            if not early:
                np.savez_compressed(OUT/f'scores_{arm}.npz', **scores)
        finally:
            nets.clear(); torch.cuda.empty_cache()


def check():
    # Same seed and old support must exactly replay the original generator.
    old, replay = NR.scenes_for(95000), scenes_for(95000, outside=.10)
    for a, b in zip(old, replay):
        for key in ('boxes', 'margin', 'group', 'family', 'meta'):
            assert a[key] == b[key], key
        for key in ('poses', 'travel', 'labels'):
            assert np.array_equal(a[key], b[key]), key
    print('40/40 old-support scene identities match near-range; new IDs disjoint', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['plan', 'check', 'generate', 'materialize', 'assemble', 'infer'])
    parser.add_argument('--split', choices=list(SPLITS)); parser.add_argument('--chunk', default='0/1')
    parser.add_argument('--unit', type=int); parser.add_argument('--early', action='store_true')
    args = parser.parse_args()
    tag = args.stage+(f'_{args.split}' if args.split else '')+f'_c{args.chunk.replace("/", "of")}' + ('_early' if args.early else '')
    if args.unit is not None:
        tag += f'_u{args.unit}'
    start = time.monotonic()
    try:
        if args.stage not in ('plan', 'check'):
            assert (OUT/'PLAN.json').exists(), 'write the plan and RUNS decision before generating or evaluating'
        if args.stage == 'plan': plan()
        elif args.stage == 'check': check()
        elif args.stage == 'generate': generate(args.chunk, args.unit)
        elif args.stage == 'materialize': materialize(args.split, args.chunk, args.early)
        elif args.stage == 'assemble': assemble()
        else: infer(args.early)
        if args.stage != 'check': SS.save(OUT/f'terminal_{tag}.json', dict(status='COMPLETE', elapsed_s=time.monotonic()-start, source_sha256=SS.sha(__file__)))
    except BaseException as error:
        if OUT.exists(): SS.save(OUT/f'terminal_{tag}.json', dict(status='FAILED', error=repr(error), elapsed_s=time.monotonic()-start))
        raise
