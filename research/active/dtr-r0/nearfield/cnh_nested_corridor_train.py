"""One fixed nested-corridor auxiliary training experiment on retained NR data.

Labels use all surfaces and true travel poses, only train units 93000..93095.
CONT and NEST start at the same five M3 models. Deployment keeps original CVR
keys and main forward; auxiliary supervision is never a runtime alarm input.
"""
import argparse
import gc
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

import cnh_structure_space as SS
import cnh_near_range as NR
import cnh_margin_labels as ML
import cnh_proposal_attribution_scenes as S

HERE = Path(__file__).resolve().parent
OUT = SS.WORK / 'cnh-nested-corridor-20261002'
RUN_ID = 'CNH_NESTED_CORRIDOR_20261002'
UNITS = list(range(93000, 93096))
WIDTHS = np.asarray([.25, .28, .30, .35, .40, .45])
ALL_WIDTHS = np.sort(np.append(WIDTHS, .33))
FRAMES = np.arange(3, 16)
ARMS = ('CONT', 'NEST')
EPOCHS, BATCH, LR, WD, AUX_WEIGHT = 5, 256, 2e-4, 1e-4, .5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def create_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plan_sha():
    plan = OUT / 'PLAN.json'
    if not plan.is_file() or RUN_ID not in (HERE.parent / 'RUNS.md').read_text(encoding='utf8'):
        raise RuntimeError('Parent PLAN and RUNS pre-registration required before labels/training')
    value = read(plan)
    fixed = dict(run=RUN_ID, train_units=UNITS, training_rows=27456,
                 decision_frames=FRAMES.tolist(), train_arms=list(ARMS), seeds=list(range(5)),
                 main_half_width=.33, aux_half_widths=WIDTHS.tolist(), aux_weight=AUX_WEIGHT)
    if any(value.get(key) != expected for key, expected in fixed.items()):
        raise ValueError('Frozen PLAN does not match the implemented training definition')
    fixed_training = dict(epochs=EPOCHS, optimizer='AdamW', learning_rate=LR,
                          weight_decay=WD, batch_size=BATCH, scheduler='CosineAnnealingLR T_max=5')
    if any(value['training'].get(key) != expected for key, expected in fixed_training.items()):
        raise ValueError('Frozen optimizer/epoch recipe differs')
    retained = {Path(p).name: checksum for p, checksum in value['prior_sha256'].items()}
    for name in ['train_labels.npz'] + [f'model_seed{s}.pt' for s in range(5)]:
        source = ML.OUT / name if name == 'train_labels.npz' else ML.OUT / 'models/M3' / name
        if retained.get(name) != sha(source):
            raise ValueError('Frozen parent M3 input changed: '+name)
    return sha(plan)


def metadata():
    paths = sorted((NR.OUT / 'data/train').glob('metadata_c*.npz'))
    if not paths:
        raise FileNotFoundError('Retained NR train metadata missing')
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            parts.append({key: data[key] for key in ('unit', 'config', 'frame', 'labels')})
    meta = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    expected = {(u, c, f) for u in UNITS for c in range(22) for f in FRAMES}
    observed = list(zip(meta['unit'].tolist(), meta['config'].tolist(), meta['frame'].tolist()))
    if len(observed) != len(expected) or set(observed) != expected or meta['labels'].shape != (27456, 2):
        raise ValueError('NR training identities/order/shape mismatch')
    return paths, meta


def label_identity():
    meta_paths, _ = metadata()
    sources = [Path(__file__), Path(NR.__file__), Path(ML.__file__), Path(SS.__file__), Path(S.__file__),
               HERE / 'cnh_readout_fix.py', HERE / 'cnh_coverage_sweep.py']
    sources += [S.SOURCE / name for name in ('cnh_track_a_geometry.py', 'cnh_track_a_fov.py',
                                            'cnh_track_a_v13_sensor.py', 'cnh_route_sensor.py')]
    return dict(plan_sha256=plan_sha(), source_sha256={str(p): sha(p) for p in sources},
        metadata_sha256={str(p): sha(p) for p in meta_paths}, m3_labels_sha256=sha(ML.OUT / 'train_labels.npz'),
        units=UNITS, frames=FRAMES.tolist(), widths=WIDTHS.tolist(), main_half_width=.33,
        geometry='All box surfaces, existing closed intersection convention, true per-frame travel pose; no nominal target offset')


def nested_labels(boxes, poses):
    """Shared triangle transforms, unchanged exact clip helper for every width."""
    triangles = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in boxes])
    bounds = []
    for half in ALL_WIDTHS:
        lo, hi = S.QUERY_LOW.copy(), S.QUERY_HIGH.copy()
        lo[:, 0], hi[:, 0] = -half, half
        bounds.append((lo, hi))
    result = np.empty((len(poses), 2, len(ALL_WIDTHS)), np.int8)
    for frame, pose in enumerate(poses):
        local = (triangles-pose[:3, 3]) @ pose[:3, :3]
        for j, (lo, hi) in enumerate(bounds):
            result[frame, :, j] = [int(len(S.clip_triangles(local, a, b)) > 0) for a, b in zip(lo, hi)]
    if np.any(np.diff(result, axis=-1) < 0):
        raise ValueError('Nested geometric labels must be nondecreasing in corridor width')
    return result


def label_unit(unit, identity, meta, old_main):
    folder = OUT / 'labels/units'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'unit{unit}.npz'
    receipt = path.with_suffix('.json')
    digest = value_sha(identity)
    if receipt.exists():
        old = read(receipt)
        if old['status'] != 'COMPLETE' or old['identity_sha256'] != digest or old['sha256'] != sha(path):
            raise ValueError('Existing unit labels identity/hash mismatch')
        return old
    if path.exists():
        raise RuntimeError('Partial label unit requires inspection: '+str(path))
    start = time.monotonic()
    scenes = NR.scenes_for(unit)
    if [int(s['config']) for s in scenes] != list(range(22)):
        raise ValueError('Expected 22 unchanged NR train scenes')
    values = np.stack([nested_labels(s['boxes'], s['travel'][FRAMES]) for s in scenes])
    pos = np.flatnonzero(meta['unit'] == unit)
    rows = values[meta['config'][pos], meta['frame'][pos]-3]
    near_i, main_i = int(np.flatnonzero(ALL_WIDTHS == .30)[0]), int(np.flatnonzero(ALL_WIDTHS == .33)[0])
    if not np.array_equal(rows[:, :, near_i], meta['labels'][pos]):
        raise ValueError(f'Half-width .30 does not reproduce NR labels: unit {unit}')
    if not np.array_equal(rows[:, :, main_i], old_main[pos]):
        raise ValueError(f'Half-width .33 does not reproduce stored M3 labels: unit {unit}')
    with path.open('xb') as f:
        np.savez_compressed(f, labels=values, unit=np.asarray(unit), widths=ALL_WIDTHS, frames=FRAMES)
    record = dict(status='COMPLETE', unit=unit, identity_sha256=digest, sha256=sha(path),
        near_exact=True, m3_exact=True, rows=len(pos), elapsed_s=time.monotonic()-start)
    create_json(receipt, record)
    return record


def labels(chunk):
    identity = label_identity()
    _, meta = metadata()
    with np.load(ML.OUT / 'train_labels.npz', allow_pickle=False) as z:
        old_main = z['M3']
    if old_main.shape != (27456, 2):
        raise ValueError('Stored M3 training labels have wrong shape')
    k, n = map(int, chunk.split('/'))
    if not 1 <= n <= len(UNITS) or not 0 <= k < n:
        raise ValueError('Invalid label chunk')
    begin = time.monotonic()
    for i, unit in enumerate(UNITS[k::n]):
        record = label_unit(unit, identity, meta, old_main)
        print('labels', unit, round(record['elapsed_s'], 3), 's',
              'elapsed', round(time.monotonic()-begin, 1), 'done', i+1, '/', len(UNITS[k::n]), flush=True)
    if label_identity() != identity:
        raise ValueError('Label sources changed during chunk')
    print('COMPLETE labels chunk', chunk, flush=True)


def assemble_labels():
    identity = label_identity()
    digest = value_sha(identity)
    target, receipt = OUT / 'train_labels.npz', OUT / 'labels_receipt.json'
    if receipt.exists():
        old = read(receipt)
        if old['status'] != 'COMPLETE' or old['identity'] != identity or old['output_sha256'] != sha(target):
            raise ValueError('Completed labels changed')
        print('Existing COMPLETE labels verified', flush=True)
        return old
    _, meta = metadata()
    all_labels = np.empty((27456, 2, len(ALL_WIDTHS)), np.int8)
    unit_hashes = {}
    for unit in UNITS:
        path = OUT / 'labels/units' / f'unit{unit}.npz'
        old = read(path.with_suffix('.json'))
        if old['status'] != 'COMPLETE' or old['identity_sha256'] != digest or old['sha256'] != sha(path):
            raise ValueError('Unit label input mismatch')
        with np.load(path, allow_pickle=False) as z:
            if int(z['unit']) != unit or not np.array_equal(z['frames'], FRAMES) or not np.array_equal(z['widths'], ALL_WIDTHS):
                raise ValueError('Unit label metadata mismatch')
            values = z['labels']
        if values.shape != (22, 13, 2, 7):
            raise ValueError('Unit label shape mismatch')
        pos = np.flatnonzero(meta['unit'] == unit)
        all_labels[pos] = values[meta['config'][pos], meta['frame'][pos]-3]
        unit_hashes[str(unit)] = old['sha256']
    near = all_labels[:, :, ALL_WIDTHS == .30].squeeze(-1)
    main = all_labels[:, :, ALL_WIDTHS == .33].squeeze(-1)
    aux = all_labels[:, :, ALL_WIDTHS != .33]
    with np.load(ML.OUT / 'train_labels.npz', allow_pickle=False) as z:
        if not np.array_equal(near, meta['labels']) or not np.array_equal(main, z['M3']):
            raise ValueError('Full label reproduction failed')
    if np.any(np.diff(all_labels, axis=-1) < 0) or set(np.unique(all_labels)) - {0, 1}:
        raise ValueError('Invalid nested labels')
    with target.open('xb') as f:
        np.savez_compressed(f, main=main, aux=aux, widths=WIDTHS,
                            **{key: meta[key] for key in ('unit', 'config', 'frame')})
    record = dict(status='COMPLETE', identity=identity, output_sha256=sha(target), unit_sha256=unit_hashes,
        rows=27456, query_rows=54912, near_exact=True, m3_exact=True, nested_exact=True,
        main_positive_by_group=main.sum(0).tolist(), auxiliary_positive_by_group_width=aux.sum(0).tolist())
    create_json(receipt, record)
    print('COMPLETE labels: .30 NR and .33 M3 exact across 54912 query rows', flush=True)
    return record


def model_class():
    import torch
    from cnh_cvr_pilot import CVR

    class NestedCVR(CVR):
        def __init__(self, arm):
            super().__init__()
            if arm == 'NEST':
                self.aux = torch.nn.Linear(72, 6)

        def forward(self, x):
            f = self.body(x)
            mask = torch.nn.functional.adaptive_avg_pool3d(x[:, 3:5], f.shape[2:])[:, :, None]
            ff = f[:, None]
            mx = ff.masked_fill(mask == 0, -1e4).flatten(3).max(-1).values
            mean = (ff*mask).flatten(3).sum(-1)/mask.flatten(3).sum(-1).clamp_min(1e-8)
            emb = self.emb.weight[None].expand(len(x), -1, -1)
            features = torch.cat([mx, mean, emb], -1)
            main = self.head(features).squeeze(-1)
            return main, self.aux(features) if hasattr(self, 'aux') else None
    return NestedCVR


def deployment_state(net):
    return {key: value.detach().cpu() for key, value in net.state_dict().items() if not key.startswith('aux.')}


def recipe():
    return dict(arms=list(ARMS), seeds=list(range(5)), epochs=EPOCHS, batch=BATCH, optimizer='AdamW',
        lr=LR, weight_decay=WD, cosine_t_max=EPOCHS, auxiliary_weight=AUX_WEIGHT,
        auxiliary='Linear(72,6), shared group parameters; gradient into body; deployment discarded',
        widths=WIDTHS.tolist(), main='unchanged M3 BCE', shuffle='numpy default_rng(seed), permutation each epoch, sorted indices within batch')


def training_request(runtime):
    labels_record = assemble_labels()
    parts = sorted((NR.OUT / 'data/train').glob('features_c*.npy'))
    meta_paths, _ = metadata()
    if len(parts) != len(meta_paths):
        raise ValueError('Voxel/metadata part count mismatch')
    for p, m in zip(parts, meta_paths):
        if p.stem.replace('features_', 'metadata_') != m.stem:
            raise ValueError('Voxel/metadata order mismatch')
    sources = [HERE / name for name in ('cnh_cvr_pilot.py', 'cnh_cvr_projection.py', 'cnh_corridor_late_fusion.py')]
    value = dict(plan_sha256=plan_sha(), label_identity=labels_record['identity'],
        labels_receipt_sha256=sha(OUT / 'labels_receipt.json'), labels_sha256=sha(OUT / 'train_labels.npz'),
        source_sha256={str(p): sha(p) for p in sources},
        voxel_sha256={str(p): sha(p) for p in parts},
        initial_models_sha256={str(ML.OUT / 'models/M3' / f'model_seed{s}.pt'):
                              sha(ML.OUT / 'models/M3' / f'model_seed{s}.pt') for s in range(5)},
        recipe=recipe(), runtime=runtime)
    path = OUT / 'training_request.json'
    if path.exists():
        if read(path) != value:
            raise ValueError('Training request changed; do not resume checkpoints')
    else:
        create_json(path, value)
    return value, sha(path), parts


def load_epoch(torch, folder, digest):
    latest = None
    for epoch in range(1, EPOCHS+1):
        path = folder / f'epoch{epoch}.pt'
        receipt = path.with_suffix('.json')
        if not receipt.exists():
            if path.exists():
                raise RuntimeError('Partial epoch checkpoint requires inspection: '+str(path))
            if any((folder / f'epoch{j}.json').exists() or (folder / f'epoch{j}.pt').exists()
                   for j in range(epoch+1, EPOCHS+1)):
                raise ValueError('Epoch checkpoint gap')
            break
        record = read(receipt)
        if record['status'] != 'COMPLETE' or record['request_sha256'] != digest or record['sha256'] != sha(path) or record['epoch'] != epoch:
            raise ValueError('Epoch checkpoint changed')
        latest = torch.load(path, map_location='cpu', weights_only=True)
        if latest['epoch'] != epoch or latest['request_sha256'] != digest:
            raise ValueError('Internal epoch identity mismatch')
    return latest


def train():
    plan_sha()
    import torch
    from cnh_cvr_projection import query_masks
    from cnh_cvr_pilot import CVR
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required for training')
    runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                   threads=2, tf32=False, cudnn_benchmark=torch.backends.cudnn.benchmark)
    frozen, digest, parts = training_request(runtime)
    done = OUT / 'training_receipt.json'
    if done.exists():
        old = read(done)
        if old['status'] != 'COMPLETE' or old['request_sha256'] != digest:
            raise ValueError('Completed training receipt mismatch')
        for path, expected in old['models_sha256'].items():
            if sha(OUT / path) != expected:
                raise ValueError('Completed deployment model changed')
        print('Existing COMPLETE training verified', flush=True)
        return old
    X = Y = A = masks = net = opt = scheduler = None
    histories, hashes = {}, {}
    started = time.monotonic()
    try:
        _, meta = metadata()
        with np.load(OUT / 'train_labels.npz', allow_pickle=False) as z:
            if any(not np.array_equal(meta[key], z[key]) for key in ('unit', 'config', 'frame')):
                raise ValueError('Label/voxel row alignment mismatch')
            Y = torch.as_tensor(z['main'].astype(np.float32), device='cuda')
            A = torch.as_tensor(z['aux'].astype(np.float32), device='cuda')
        count, offset = 27456, 0
        masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
        for path in parts:
            x = np.load(path, mmap_mode='r')
            if x.dtype != np.float16 or x.shape[1:] != (3, 24, 17, 33):
                raise ValueError('Retained voxel shape/dtype changed')
            if X is None:
                X = torch.empty((count, *x.shape[1:]), dtype=torch.float16, device='cuda')
            for row in range(0, len(x), 1024):
                block = np.array(x[row:row+1024])
                if not np.isfinite(block).all():
                    raise ValueError('Nonfinite retained train voxels')
                X[offset+row:offset+row+len(block)] = torch.as_tensor(block, device='cuda')
            offset += len(x)
            del x
        if offset != count:
            raise ValueError('Training row count mismatch')

        def prep(xb):
            x = xb.float()
            x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p()
            x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p()
            x[:, 1] /= 8
            return torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)

        Model = model_class()
        for arm in ARMS:
            for seed in range(5):
                folder = OUT / 'checkpoints' / arm / f'seed{seed}'
                folder.mkdir(parents=True, exist_ok=True)
                last = load_epoch(torch, folder, digest)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                rng = np.random.default_rng(seed)
                net = Model(arm).cuda()
                initial = torch.load(ML.OUT / 'models/M3' / f'model_seed{seed}.pt', map_location='cuda', weights_only=True)
                incompatible = net.load_state_dict(initial, strict=False)
                expected_missing = ['aux.weight', 'aux.bias'] if arm == 'NEST' else []
                if incompatible.missing_keys != expected_missing or incompatible.unexpected_keys:
                    raise ValueError('M3 initialization key mismatch')
                opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
                history, start_epoch = [], 0
                if last is not None:
                    if last['arm'] != arm or last['seed'] != seed:
                        raise ValueError('Epoch checkpoint arm/seed mismatch')
                    net.load_state_dict(last['model'])
                    opt.load_state_dict(last['optimizer'])
                    scheduler.load_state_dict(last['scheduler'])
                    rng.bit_generator.state = last['numpy_rng']
                    torch.set_rng_state(last['torch_rng'].cpu())
                    torch.cuda.set_rng_state_all([x.cpu() for x in last['cuda_rng']])
                    history, start_epoch = last['history'], last['epoch']
                for epoch in range(start_epoch, EPOCHS):
                    epoch_start = time.monotonic()
                    net.train()
                    order = rng.permutation(np.arange(count))
                    losses = np.zeros(3, float)
                    for row in range(0, count, BATCH):
                        pos = torch.as_tensor(np.sort(order[row:row+BATCH]), device='cuda')
                        opt.zero_grad(set_to_none=True)
                        main, aux = net(prep(X[pos]))
                        main_loss = torch.nn.functional.binary_cross_entropy_with_logits(main, Y[pos])
                        aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux, A[pos]) if arm == 'NEST' else main_loss.new_zeros(())
                        loss = main_loss + AUX_WEIGHT*aux_loss
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite training loss')
                        loss.backward()
                        opt.step()
                        losses += np.asarray([float(loss.detach()), float(main_loss.detach()), float(aux_loss.detach())])*len(pos)
                    scheduler.step()
                    entry = dict(epoch=epoch+1, loss=float(losses[0]/count), main_loss=float(losses[1]/count),
                        aux_loss=float(losses[2]/count), permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                        elapsed_s=time.monotonic()-epoch_start)
                    history.append(entry)
                    if any(not bool(torch.isfinite(p).all()) for p in net.parameters()):
                        raise FloatingPointError('Nonfinite trained parameters')
                    payload = dict(epoch=epoch+1, arm=arm, seed=seed, request_sha256=digest,
                        model=net.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), history=history)
                    path = folder / f'epoch{epoch+1}.pt'
                    with path.open('xb') as f:
                        torch.save(payload, f)
                    create_json(path.with_suffix('.json'), dict(status='COMPLETE', epoch=epoch+1, arm=arm, seed=seed,
                        request_sha256=digest, sha256=sha(path), runtime=runtime))
                    print('train', arm, seed, entry, flush=True)
                relative = f'models/{arm}/model_seed{seed}.pt'
                final = OUT / relative
                final.parent.mkdir(parents=True, exist_ok=True)
                deployed = deployment_state(net)
                reference = CVR()
                reference.load_state_dict(deployed, strict=True)
                del reference
                if final.exists():
                    existing = torch.load(final, map_location='cpu', weights_only=True)
                    if existing.keys() != deployed.keys() or any(not torch.equal(existing[k], deployed[k]) for k in deployed):
                        raise ValueError('Existing deployment does not match completed epoch')
                else:
                    with final.open('xb') as f:
                        torch.save(deployed, f)
                hashes[relative] = sha(final)
                histories[f'{arm}|{seed}'] = history
                net = opt = scheduler = last = initial = deployed = payload = None
                gc.collect()
                torch.cuda.empty_cache()
        for seed in range(5):
            a = [x['permutation_sha256'] for x in histories[f'CONT|{seed}']]
            b = [x['permutation_sha256'] for x in histories[f'NEST|{seed}']]
            if a != b or len(a) != EPOCHS:
                raise ValueError('CONT/NEST epoch shuffles did not match')
        final_request, final_digest, _ = training_request(runtime)
        if final_request != frozen or final_digest != digest:
            raise ValueError('Training inputs changed during execution')
        result = dict(status='COMPLETE', plan_sha256=frozen['plan_sha256'], request_sha256=digest,
            labels_receipt_sha256=frozen['labels_receipt_sha256'], models_sha256=hashes,
            source_sha256={**frozen['label_identity']['source_sha256'], **frozen['source_sha256']},
            runtime=runtime, recipe=recipe(), history=histories, paired_shuffles_exact=True,
            elapsed_s=time.monotonic()-started, deployment='original CVR keys only; auxiliary heads discarded')
        create_json(done, result)
        print('COMPLETE paired CONT/NEST five-seed training', flush=True)
        return result
    finally:
        X = Y = A = masks = net = opt = scheduler = None
        gc.collect()
        torch.cuda.empty_cache()


def check():
    """CPU-only numerical checks, synthetic geometry and optimizer resume fixture."""
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    assert UNITS == NR.SPLITS['train'] and np.array_equal(FRAMES, SS.TRAIN_FRAMES)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = S.ry(15.)
    boxes = [dict(lo=[.315, -.1, 1.], hi=[.395, .2, 1.2]),
             dict(lo=[-.43, .5, 1.], hi=[-.38, .8, 1.2])]
    before = time.perf_counter()
    actual = nested_labels(boxes, poses)
    geometry_seconds = time.perf_counter()-before
    for j, half in enumerate(ALL_WIDTHS):
        np.testing.assert_array_equal(actual[:, :, j], ML.labels_for(boxes, poses, half))
    assert not actual[0, 0, ALL_WIDTHS == .30].item() and actual[0, 0, ALL_WIDTHS == .33].item()
    assert actual[0, 1, ALL_WIDTHS == .40].item()  # Background/all-object, independent of a target group.
    torch.manual_seed(17)
    reference = CVR()
    Model = model_class()
    net = Model('NEST')
    net.load_state_dict(reference.state_dict(), strict=False)
    x = torch.randn(2, 5, 24, 17, 33)
    x[:, 3:5] = torch.as_tensor(query_masks())[None]
    main, aux = net(x)
    torch.testing.assert_close(main, reference(x), rtol=0, atol=0)
    assert aux.shape == (2, 2, 6) and sum(p.numel() for p in net.aux.parameters()) == 438
    aux.square().mean().backward()
    assert net.body[0].weight.grad is not None and bool(net.body[0].weight.grad.abs().sum() > 0)
    target = torch.zeros(2, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    def step(model, optimizer, schedule):
        optimizer.zero_grad(set_to_none=True)
        a, b = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a, target) + AUX_WEIGHT*torch.nn.functional.binary_cross_entropy_with_logits(b, torch.zeros_like(b))
        loss.backward()
        optimizer.step()
        schedule.step()

    step(net, opt, scheduler)
    stream = io.BytesIO()
    torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    resumed = Model('NEST')
    resumed.load_state_dict(saved['model'])
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=LR, weight_decay=WD)
    resumed_opt.load_state_dict(saved['optimizer'])
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_opt, EPOCHS)
    resumed_scheduler.load_state_dict(saved['scheduler'])
    step(net, opt, scheduler)
    step(resumed, resumed_opt, resumed_scheduler)
    for key, tensor in net.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
    reference.load_state_dict(deployment_state(net), strict=True)
    torch.testing.assert_close(net(x)[0], reference(x), rtol=0, atol=0)
    rng = np.random.default_rng(2)
    rng.permutation(100)
    saved_rng = rng.bit_generator.state
    other_rng = np.random.default_rng(2)
    other_rng.bit_generator.state = saved_rng
    np.testing.assert_array_equal(rng.permutation(100), other_rng.permutation(100))
    assert not torch.cuda.is_initialized()
    print('PASS CPU: exact CVR main/deploy parity, 438 aux params/body gradients, optimizer+cosine resume, geometric nesting/all-object and original clipping parity')
    print('Synthetic geometry two poses seven widths seconds', geometry_seconds, '(not a real unit timing)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'labels', 'labels-assemble', 'train'))
    parser.add_argument('--chunk', default='0/1')
    args = parser.parse_args()
    if args.stage == 'check':
        check()
    elif args.stage == 'labels':
        labels(args.chunk)
    elif args.stage == 'labels-assemble':
        assemble_labels()
    else:
        train()
