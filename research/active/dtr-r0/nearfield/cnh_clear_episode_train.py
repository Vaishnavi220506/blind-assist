"""Fixed MEAN/MAX clear-episode losses on complete retained training scenes.

Pure original CVR in both arms. Only train93000..93095 geometry-derived labels
are used; no evaluation metadata, deadline, target offset, render or new voxels.
"""
import argparse
import gc
import hashlib
import io
import time
from pathlib import Path

import numpy as np

import cnh_nested_corridor_train as NT

OUT = NT.SS.WORK / 'cnh-clear-episode-20261002'
RUN_ID = 'CNH_CLEAR_EPISODE_20261002'
ARMS = ('MEAN', 'MAX')
UNITS = list(range(93000, 93096))
FRAMES = np.arange(3, 16)
WEIGHTS = np.asarray([1., 2., 4., 8., 16.], np.float32)
EPOCHS, BATCH_SCENES, LR, WD, EXTRA_WEIGHT = 5, 16, 2e-4, 1e-4, .5


def plan_sha():
    path = OUT / 'PLAN.json'
    if not path.is_file() or RUN_ID not in (NT.HERE.parent / 'RUNS.md').read_text(encoding='utf8'):
        raise RuntimeError('Parent PLAN and RUNS pre-registration required')
    value = NT.read(path)
    fixed = dict(run=RUN_ID, train_units=UNITS, decision_frames=FRAMES.tolist(),
                 train_arms=list(ARMS), seeds=list(range(5)), main_half_width=.33,
                 clear_half_width=.40, extra_weight=EXTRA_WEIGHT)
    if any(value.get(key) != expected for key, expected in fixed.items()):
        raise ValueError('Frozen PLAN differs from fixed episode definition')
    training = dict(epochs=EPOCHS, optimizer='AdamW', learning_rate=LR,
                    weight_decay=WD, scheduler='CosineAnnealingLR T_max=5', batch_scenes=BATCH_SCENES)
    if any(value['training'].get(key) != expected for key, expected in training.items()):
        raise ValueError('Frozen PLAN differs from optimizer/batch definition')
    for old_path, expected in value['prior_sha256'].items():
        p = Path(old_path)
        if (p.name in ('train_labels.npz', 'labels_receipt.json') or p.name.startswith('model_seed')) and NT.sha(p) != expected:
            raise ValueError('Frozen training input changed: '+str(p))
    return NT.sha(path)


def scene_index(meta):
    """Map sorted whole scenes to arbitrary concatenated shard row positions."""
    triples = list(zip(meta['unit'].tolist(), meta['config'].tolist(), meta['frame'].tolist()))
    if len(triples) != len(set(triples)):
        raise ValueError('Duplicate unit/config/frame metadata')
    positions = {key: row for row, key in enumerate(triples)}
    frame_sets = {}
    for u, c, f in triples:
        frame_sets.setdefault((int(u), int(c)), set()).add(f)
    scenes = sorted(frame_sets)
    rows = []
    for u, c in scenes:
        if frame_sets[(u, c)] != set(FRAMES):
            raise ValueError('A training scene is missing or has extra decision frames')
        rows.append([positions[(u, c, int(f))] for f in FRAMES])
    return np.asarray(scenes, np.int64), np.asarray(rows, np.int64)


def load_labels():
    meta_paths, meta = NT.metadata()
    receipt_path = NT.OUT / 'labels_receipt.json'
    record = NT.read(receipt_path)
    label_path = NT.OUT / 'train_labels.npz'
    if record['status'] != 'COMPLETE' or not all(record[key] for key in ('near_exact', 'm3_exact', 'nested_exact')):
        raise ValueError('Original all-object label reproduction must be complete')
    if NT.sha(label_path) != record['output_sha256']:
        raise ValueError('Retained nested labels changed')
    for path, expected in record['identity']['metadata_sha256'].items():
        if NT.sha(path) != expected:
            raise ValueError('Original voxel metadata changed since label generation')
    with np.load(label_path, allow_pickle=False) as z:
        if any(not np.array_equal(meta[key], z[key]) for key in ('unit', 'config', 'frame')):
            raise ValueError('Retained labels and voxel metadata row orders disagree')
        slot = np.flatnonzero(z['widths'] == .40)
        if len(slot) != 1:
            raise ValueError('Exactly one all-object half-width .40 label required')
        main, wide = z['main'].copy(), z['aux'][:, :, int(slot[0])].copy()
    if main.shape != (27456, 2) or wide.shape != main.shape or np.any(main > wide):
        raise ValueError('Invalid retained nested main/expanded labels')
    with np.load(NT.ML.OUT / 'train_labels.npz', allow_pickle=False) as z:
        if not np.array_equal(main, z['M3']):
            raise ValueError('Main supervision must stay exactly M3')
    scenes, rows = scene_index(meta)
    if len(scenes) != 2112 or not np.array_equal(scenes, [(u, c) for u in UNITS for c in range(22)]):
        raise ValueError('Only original 96 train units x 22 complete scenes allowed')
    main = main[rows].astype(np.float32)
    clear = np.all(wide[rows] == 0, axis=1)
    if np.any(clear & np.any(main != 0, axis=1)):
        raise ValueError('Whole-clear queries cannot contain any main positive frame')
    support = dict(scenes=len(scenes), queries=int(clear.size), clear_queries=int(clear.sum()),
        clear_by_group=clear.sum(0).tolist(),
        clear_by_unit={str(u): clear[scenes[:, 0] == u].sum(0).tolist() for u in UNITS},
        criterion='Every one of 13 all-object .40 closed-intersection labels is zero; pass/contact excluded',
        denominator='all batch scenes x 2 queries, not only eligible clear queries')
    parent_path = OUT / 'train_support.json'
    parent = NT.read(parent_path)
    if parent['status'] != 'COMPLETE' or parent['overall']['clear_query_episodes'] != support['clear_queries'] or parent['overall']['total_query_episodes'] != support['queries']:
        raise ValueError('Train support differs from frozen independent aggregation')
    if [parent['tables']['query'][g]['clear_query_episodes'] for g in ('HEAD', 'BODY')] != support['clear_by_group']:
        raise ValueError('Train clear support group mismatch')
    for path, expected in parent['input_sha256'].items():
        if NT.sha(path) != expected:
            raise ValueError('Frozen train support input changed')
    support.update(train_support_sha256=NT.sha(parent_path),
                   same_target_query_description=parent['tables']['same_target_query'],
                   nominal_groups_used_for_selection=False)
    return meta_paths, scenes, rows, main, clear, support


def smooth(torch, logits):
    if logits.ndim != 3 or logits.shape[1:] != (13, 2):
        raise ValueError('Expected [scenes,13,2] raw logits')
    weights = logits.new_tensor(WEIGHTS)
    values = []
    for frame in range(13):
        lo = max(0, frame-4)
        w = weights[-(frame-lo+1):]
        values.append((logits[:, lo:frame+1] * w[None, :, None]).sum(1)/w.sum())
    return torch.stack(values, 1)


def losses(torch, arm, logits, labels, clear):
    if arm not in ARMS or labels.shape != logits.shape or clear.shape != (len(logits), 2):
        raise ValueError('Loss inputs/arm mismatch')
    main = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    smoothed = smooth(torch, logits)
    if arm == 'MEAN':
        query_loss = torch.nn.functional.softplus(smoothed).mean(1)
    else:
        query_loss = torch.nn.functional.softplus(smoothed.max(1).values)
    extra = (query_loss*clear.to(query_loss.dtype)).sum()/(len(logits)*2)
    return main + EXTRA_WEIGHT*extra, main, extra


def recipe():
    return dict(arms=list(ARMS), seeds=list(range(5)), epochs=EPOCHS, batch_scenes=BATCH_SCENES,
        frames=FRAMES.tolist(), optimizer='AdamW', lr=LR, weight_decay=WD, cosine_t_max=EPOCHS,
        smoothing_weights=WEIGHTS.tolist(), smoothing='causal last five, available suffix weights normalized, float32 differentiable',
        extra_weight=EXTRA_WEIGHT, main='mean M3 BCE across all scene/frame/query entries',
        extra_MEAN='mean_t softplus(smoothed logit)', extra_MAX='softplus(max_t smoothed logit)',
        extra_normalizer='batch_scenes * 2, regardless of clear count',
        clear='all 13 retained all-object half-width .40 labels zero',
        shuffle='numpy default_rng(seed) scene permutation each epoch; both arms identical',
        deployment='original CVR only; no auxiliary parameters')


def request(runtime, support):
    metadata_paths, _ = NT.metadata()
    paths = [p.with_name(p.name.replace('metadata_', 'features_').replace('.npz', '.npy')) for p in metadata_paths]
    sources = [Path(__file__), Path(NT.__file__), NT.HERE / 'cnh_cvr_pilot.py',
               NT.HERE / 'cnh_cvr_projection.py', NT.HERE / 'cnh_structure_space.py',
               NT.HERE / 'cnh_near_range.py']
    value = dict(plan_sha256=plan_sha(), source_sha256={str(p): NT.sha(p) for p in sources},
        labels_receipt_sha256=NT.sha(NT.OUT / 'labels_receipt.json'), labels_sha256=NT.sha(NT.OUT / 'train_labels.npz'),
        metadata_sha256={str(p): NT.sha(p) for p in metadata_paths}, voxel_sha256={str(p): NT.sha(p) for p in paths},
        initial_models_sha256={str(NT.ML.OUT / 'models/M3' / f'model_seed{s}.pt'):
                              NT.sha(NT.ML.OUT / 'models/M3' / f'model_seed{s}.pt') for s in range(5)},
        clear_support=support, recipe=recipe(), runtime=runtime)
    path = OUT / 'training_request.json'
    if path.exists():
        if NT.read(path) != value:
            raise ValueError('Training identity changed; cannot resume')
    else:
        NT.create_json(path, value)
    return value, NT.sha(path), paths, metadata_paths


def train():
    plan_sha()
    _, scenes, rows, main_labels, clear, support = load_labels()
    import torch
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required for training')
    runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                   threads=2, tf32=False, cudnn_benchmark=torch.backends.cudnn.benchmark,
                   voxel_loading='1024-row memmap blocks fill one shared CUDA float16 X once; scene batches gather on GPU')
    frozen, digest, paths, metadata_paths = request(runtime, support)
    completed = OUT / 'training_receipt.json'
    if completed.exists():
        old = NT.read(completed)
        if old['status'] != 'COMPLETE' or old['request_sha256'] != digest:
            raise ValueError('Completed training request mismatch')
        for relative, checksum in old['models_sha256'].items():
            if NT.sha(OUT / relative) != checksum:
                raise ValueError('Completed deployment changed')
        print('Existing COMPLETE episode training verified', flush=True)
        return old
    X = net = opt = scheduler = masks = Y = C = xb = raw = total = main = extra = None
    histories, hashes = {}, {}
    started = time.monotonic()
    try:
        X = torch.empty((27456, 3, 24, 17, 33), dtype=torch.float16, device='cuda')
        offset = 0
        for path, metadata_path in zip(paths, metadata_paths):
            x = np.load(path, mmap_mode='r')
            try:
                with np.load(metadata_path, allow_pickle=False) as meta:
                    if len(x) != len(meta['unit']):
                        raise ValueError('Voxel shard row count does not match metadata')
                if x.dtype != np.float16 or x.shape[1:] != (3, 24, 17, 33):
                    raise ValueError('Unexpected retained voxel dtype/shape')
                for start in range(0, len(x), 1024):
                    block = np.array(x[start:start+1024])
                    if not np.isfinite(block).all():
                        raise ValueError('Nonfinite cached voxel values')
                    X[offset+start:offset+start+len(block)] = torch.as_tensor(block, device='cuda')
                offset += len(x)
            finally:
                x._mmap.close()
        if offset != 27456:
            raise ValueError('Voxel total does not match scene metadata')
        masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
        Y = torch.as_tensor(main_labels, dtype=torch.float32, device='cuda')
        C = torch.as_tensor(clear, device='cuda')

        def prep(voxels):
            value = voxels.float()
            value[:, 0] = value[:, 0].sign()*value[:, 0].abs().log1p()
            value[:, 2] = value[:, 2].sign()*value[:, 2].abs().log1p()
            value[:, 1] /= 8
            return torch.cat((value, masks[None].expand(len(value), -1, -1, -1, -1)), 1)

        for arm in ARMS:
            for seed in range(5):
                folder = OUT / 'checkpoints' / arm / f'seed{seed}'
                folder.mkdir(parents=True, exist_ok=True)
                last = NT.load_epoch(torch, folder, digest)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                rng = np.random.default_rng(seed)
                net = CVR().cuda()
                net.load_state_dict(torch.load(NT.ML.OUT / 'models/M3' / f'model_seed{seed}.pt',
                                               map_location='cuda', weights_only=True), strict=True)
                opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
                history, start_epoch = [], 0
                if last is not None:
                    if last['arm'] != arm or last['seed'] != seed:
                        raise ValueError('Epoch arm/seed mismatch')
                    net.load_state_dict(last['model'], strict=True)
                    opt.load_state_dict(last['optimizer'])
                    scheduler.load_state_dict(last['scheduler'])
                    rng.bit_generator.state = last['numpy_rng']
                    torch.set_rng_state(last['torch_rng'].cpu())
                    torch.cuda.set_rng_state_all([v.cpu() for v in last['cuda_rng']])
                    history, start_epoch = last['history'], last['epoch']
                for epoch in range(start_epoch, EPOCHS):
                    begin = time.monotonic()
                    net.train()
                    order = rng.permutation(len(scenes))
                    sums = np.zeros(3)
                    for offset in range(0, len(order), BATCH_SCENES):
                        selected = order[offset:offset+BATCH_SCENES]
                        pos = torch.as_tensor(selected, device='cuda')
                        indices = torch.as_tensor(rows[selected].reshape(-1), device='cuda')
                        xb = prep(X[indices])
                        opt.zero_grad(set_to_none=True)
                        raw = net(xb).reshape(len(selected), 13, 2)
                        total, main, extra = losses(torch, arm, raw, Y[pos], C[pos])
                        if not bool(torch.isfinite(total)):
                            raise FloatingPointError('Nonfinite episode training loss')
                        total.backward()
                        opt.step()
                        sums += np.asarray([float(total.detach()), float(main.detach()), float(extra.detach())])*len(selected)
                    scheduler.step()
                    if any(not bool(torch.isfinite(p).all()) for p in net.parameters()):
                        raise FloatingPointError('Nonfinite trained model')
                    entry = dict(epoch=epoch+1, loss=float(sums[0]/len(scenes)), main_loss=float(sums[1]/len(scenes)),
                        clear_loss=float(sums[2]/len(scenes)), permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                        elapsed_s=time.monotonic()-begin)
                    history.append(entry)
                    payload = dict(epoch=epoch+1, arm=arm, seed=seed, request_sha256=digest,
                        model=net.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), history=history)
                    path = folder / f'epoch{epoch+1}.pt'
                    with path.open('xb') as f:
                        torch.save(payload, f)
                    NT.create_json(path.with_suffix('.json'), dict(status='COMPLETE', arm=arm, seed=seed,
                        epoch=epoch+1, request_sha256=digest, sha256=NT.sha(path), runtime=runtime))
                    print('train', arm, seed, entry, flush=True)
                deployed = {key: value.detach().cpu() for key, value in net.state_dict().items()}
                reference = CVR()
                reference.load_state_dict(deployed, strict=True)
                del reference
                relative = f'models/{arm}/model_seed{seed}.pt'
                final = OUT / relative
                final.parent.mkdir(parents=True, exist_ok=True)
                if final.exists():
                    prior = torch.load(final, map_location='cpu', weights_only=True)
                    if prior.keys() != deployed.keys() or any(not torch.equal(prior[k], deployed[k]) for k in deployed):
                        raise ValueError('Existing deployment differs from final epoch')
                else:
                    with final.open('xb') as f:
                        torch.save(deployed, f)
                hashes[relative] = NT.sha(final)
                histories[f'{arm}|{seed}'] = history
                net = opt = scheduler = last = deployed = payload = xb = raw = total = main = extra = None
                gc.collect()
                torch.cuda.empty_cache()
        for seed in range(5):
            a = [r['permutation_sha256'] for r in histories[f'MEAN|{seed}']]
            b = [r['permutation_sha256'] for r in histories[f'MAX|{seed}']]
            if a != b or len(a) != EPOCHS:
                raise ValueError('Paired scene shuffles differ')
        if request(runtime, support)[:2] != (frozen, digest):
            raise ValueError('Training identity changed during execution')
        result = dict(status='COMPLETE', plan_sha256=frozen['plan_sha256'], request_sha256=digest,
            models_sha256=hashes, source_sha256=frozen['source_sha256'],
            labels_receipt_sha256=frozen['labels_receipt_sha256'], clear_support=support,
            runtime=runtime, recipe=recipe(), history=histories, paired_shuffles_exact=True,
            elapsed_s=time.monotonic()-started, deployment='ten original CVR state_dicts; no auxiliary parameters')
        NT.create_json(completed, result)
        print('COMPLETE MEAN/MAX full-scene training', flush=True)
        return result
    finally:
        X = net = opt = scheduler = masks = Y = C = xb = raw = total = main = extra = None
        gc.collect()
        torch.cuda.empty_cache()


def check():
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    _, scenes, rows, main_labels, clear, support = load_labels()
    assert len(np.unique(rows)) == 27456 and rows.shape == (2112, 13)
    # Arbitrary metadata order, rather than an assumption of contiguous shards.
    m = dict(unit=np.repeat([93000, 93001], 13), config=np.zeros(26, int), frame=np.tile(FRAMES, 2))
    perm = np.random.default_rng(5).permutation(26)
    shuffled = {key: value[perm] for key, value in m.items()}
    ids, aligned = scene_index(shuffled)
    assert ids.tolist() == [[93000, 0], [93001, 0]]
    np.testing.assert_array_equal(shuffled['frame'][aligned], np.tile(FRAMES, (2, 1)))
    rng = np.random.default_rng(7)
    values = rng.normal(size=(3, 13, 2)).astype(np.float32)
    expected = np.empty_like(values)
    for t in range(13):
        lo = max(0, t-4)
        w = np.asarray([2.**j for j in range(t-lo+1)], np.float64)
        expected[:, t] = np.sum(values[:, lo:t+1].astype(float)*w[None, :, None], axis=1)/w.sum()
    actual = smooth(torch, torch.tensor(values)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    x = torch.tensor(values, requires_grad=True)
    smooth(torch, x)[:, 3].sum().backward()
    assert torch.count_nonzero(x.grad[:, 4:]) == 0
    # One pass/contact frame disqualifies the complete query, including early peaks.
    wide = np.zeros((2, 13, 2), np.int8)
    wide[0, 8, 0] = 1
    wide[1, 12, 1] = 1
    keep = np.all(wide == 0, axis=1)
    np.testing.assert_array_equal(keep, [[False, True], [True, False]])
    for arm in ARMS:
        zero = torch.zeros(1, 13, 2, requires_grad=True)
        total, base, extra = losses(torch, arm, zero, torch.zeros_like(zero), torch.tensor([[True, False]]))
        np.testing.assert_allclose([base.item(), extra.item(), total.item()],
                                   [np.log(2), .5*np.log(2), 1.25*np.log(2)], rtol=1e-6)
        extra.backward()
        assert zero.grad[:, :, 0].abs().sum() > 0 and torch.count_nonzero(zero.grad[:, :, 1]) == 0
        no_clear = losses(torch, arm, zero, torch.zeros_like(zero), torch.zeros(1, 2, dtype=torch.bool))[2]
        assert no_clear.item() == 0
    peaked = torch.full((1, 13, 2), -4.)
    peaked[:, 3, 0] = 5.
    mean = losses(torch, 'MEAN', peaked, torch.zeros_like(peaked), torch.ones(1, 2, dtype=torch.bool))[2]
    maximum = losses(torch, 'MAX', peaked, torch.zeros_like(peaked), torch.ones(1, 2, dtype=torch.bool))[2]
    assert maximum > mean
    # The new differentiable loss and optimizer/scheduler survive a checkpoint.
    torch.manual_seed(3)
    net = torch.nn.Linear(5, 2)
    features = torch.randn(2, 13, 5)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    def step(model, optimizer, schedule):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        losses(torch, 'MAX', logits, torch.zeros_like(logits), torch.tensor(keep))[0].backward()
        optimizer.step()
        schedule.step()

    step(net, opt, scheduler)
    stream = io.BytesIO()
    torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    resumed = torch.nn.Linear(5, 2)
    resumed.load_state_dict(saved['model'])
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=LR, weight_decay=WD)
    opt2.load_state_dict(saved['optimizer'])
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, EPOCHS)
    sch2.load_state_dict(saved['scheduler'])
    step(net, opt, scheduler)
    step(resumed, opt2, sch2)
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    a, b = np.random.default_rng(4), np.random.default_rng(4)
    for epoch in range(EPOCHS):
        np.testing.assert_array_equal(a.permutation(len(scenes)), b.permutation(len(scenes)))
        if epoch == 1:
            state = b.bit_generator.state
            b = np.random.default_rng(999)
            b.bit_generator.state = state
    assert not torch.cuda.is_initialized()
    print('PASS CPU: complete-scene routing, causal suffix smoothing, all-frame clear mask, denominators/gradients, paired scene order and optimizer/RNG resume')
    print('Retained clear support:', support['clear_queries'], '/', support['queries'], support['clear_by_group'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'train'))
    args = parser.parse_args()
    check() if args.stage == 'check' else train()
