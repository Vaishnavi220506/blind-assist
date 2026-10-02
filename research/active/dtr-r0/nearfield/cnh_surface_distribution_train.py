"""Native CNH surface bottleneck: paired single-seed BCEO/SURF prototype.

Only retained training units are used by train/prepare-m3. Inference inputs are
label-free. Surface truth is opened only for SURF training, never by NativeInputs.
"""
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import cnh_cvr_pilot as CP
import cnh_margin_confirm as MC
import cnh_margin_labels as ML
import cnh_near_range as NR
import cnh_nested_corridor_train as NT
import cnh_structure_space as SS

OUT = SS.WORK / 'cnh-surface-distribution-20261002'
RUN_ID = 'CNH_SURFACE_DISTRIBUTION_20261002'
ARMS = ('BCEO', 'SURF')
SPLITS = dict(train=list(range(93000, 93096)), **MC.SPLITS)
FRAMES = np.arange(3, 16)
EPOCHS, BATCH, LR, WD = 10, 64, .001, .0001
sha, read, create_json = NT.sha, NT.read, NT.create_json


def plan_sha():
    path = OUT / 'PLAN.json'
    p = read(path)
    fixed = dict(run=RUN_ID, training_units=SPLITS['train'], calibration_units=SPLITS['calib'],
                 evaluation_units=SPLITS['evaluation'], decision_frames=FRAMES.tolist(),
                 arms=['M3', *ARMS], train_arms=list(ARMS), seeds=[0], epochs=EPOCHS,
                 batch_size=BATCH, learning_rate=LR, weight_decay=WD, loss_surface_weight=1.)
    if any(p.get(k) != v for k, v in fixed.items()):
        raise ValueError('PLAN differs from fixed single-seed surface recipe')
    if RUN_ID not in (Path(__file__).parent.parent / 'RUNS.md').read_text(encoding='utf8'):
        raise ValueError('Missing parent RUNS registration')
    return sha(path)


def radial_shuffle(x):
    """channel=(dy*2+dx)*8+dr; preserve all coarse spatial/range axes."""
    b, c, y, z, r = x.shape
    if c != 32:
        raise ValueError('Expected four subangles times eight radial subbins')
    return x.reshape(b, 2, 2, 8, y, z, r).permute(0, 4, 1, 5, 2, 6, 3).reshape(b, y*2, z*2, r*8)


class SurfaceCVR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(8, 32, 3, padding=1)
        self.pose = nn.Linear(104, 32)
        self.conv2 = nn.Conv3d(32, 32, 3, padding=1)
        self.radial = nn.Conv3d(32, 32, 1)
        self.unknown = nn.Conv2d(32, 4, 1)
        self.query_embedding = nn.Embedding(2, 8)
        self.alarm = nn.Sequential(nn.Linear(777, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.alarm[-1].weight)
        nn.init.zeros_(self.alarm[-1].bias)

    def distribution(self, native_log, pose104):
        h = F.gelu(self.conv1(native_log) + self.pose(pose104)[:, :, None, None, None])
        h = F.gelu(self.conv2(h))
        radial = radial_shuffle(self.radial(h))
        unknown = F.pixel_shuffle(self.unknown(h.mean(-1)), 2).squeeze(1)
        return F.log_softmax(torch.cat((radial, unknown[..., None]), -1), -1)

    def alarm_features(self, logp, weights, baseline):
        p = logp.exp()
        mass = (p[:, None] * weights).sum(-1).flatten(2)
        entropy = -(p * logp).sum(-1).flatten(1) / math.log(129)
        unknown = p[..., 128].flatten(1)
        embedding = self.query_embedding.weight[None].expand(len(p), -1, -1)
        return torch.cat((mass, entropy[:, None].expand(-1, 2, -1),
                          unknown[:, None].expand(-1, 2, -1), embedding, baseline.detach()[..., None]), -1)

    def forward(self, native_log, pose104, query_weights, m3_raw):
        logp = self.distribution(native_log, pose104)
        features = self.alarm_features(logp, query_weights, m3_raw)
        return m3_raw.detach() + self.alarm(features).squeeze(-1), logp


def window_inputs(z_scene, noisy, frame):
    """No future z1/pose is read; slots are oldest first with left zero padding."""
    frame = int(frame)
    if frame not in FRAMES or z_scene.shape != (16, 8, 8, 16) or noisy.shape != (16, 4, 4):
        raise ValueError('Native history shape/frame mismatch')
    start = max(0, frame-7)
    count = frame-start+1
    x = np.zeros((8, 8, 8, 16), np.float32)
    x[-count:] = z_scene[start:frame+1].astype(np.float32)
    x = np.sign(x)*np.log1p(np.abs(x))
    matrices = np.zeros((8, 3, 4), np.float32)
    matrices[-count:] = (np.linalg.inv(noisy[frame]) @ noisy[start:frame+1])[:, :3]
    valid = np.zeros(8, np.float32)
    valid[-count:] = 1
    return x, np.concatenate((matrices.flatten(), valid))


class NativeInputs:
    """Small native observations in RAM, 39 shared public-query weight tables.

    batch(unit[],config[],frame[]) -> native_log, pose104, query_weights numpy.
    This class does not open any surface label/geometry artifact.
    """
    def __init__(self, split, units=None):
        if split not in SPLITS:
            raise ValueError('Unknown split')
        self.split = split
        self.units = list(SPLITS[split] if units is None else units)
        if not self.units or len(set(self.units)) != len(self.units) or set(self.units)-set(SPLITS[split]):
            raise ValueError('Invalid native unit selection')
        self.configs = 22 if split == 'train' else 40
        self.raw, self.poses, self.weights, self.input_sha256 = {}, {}, {}, {}
        from cnh_surface_distribution_data import query_weights
        root = NR.OUT if split == 'train' else MC.OUT
        for unit in self.units:
            path = root / 'features' / split / f'unit{unit}.npz'
            self.input_sha256[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                raw = z['z1']
                expected = [(c, f) for c in range(self.configs) for f in range(16)]
                actual = list(zip(z['scene'].tolist(), z['frame'].tolist()))
                if actual != expected or raw.shape != (self.configs*16, 8, 8, 16) or not np.isfinite(raw).all():
                    raise ValueError(f'Native source axes/identities invalid: {unit}')
                self.raw[unit] = raw.reshape(self.configs, 16, 8, 8, 16).astype(np.float16)
            for config in range(self.configs):
                sensor, travel, noisy = CP.motion_metadata(unit, config)
                self.poses[unit, config] = noisy
                if config == 0:
                    for frame in FRAMES:
                        key = (unit % 3, int(frame))
                        q = np.linalg.inv(travel[frame]) @ sensor[frame]
                        if key not in self.weights:
                            value = np.asarray(query_weights(q), np.float32)
                            if value.shape != (2, 16, 16, 129) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)) or np.any(value[..., 128]):
                                raise ValueError('Invalid public query weights')
                            self.weights[key] = (q, value)
                        elif not np.allclose(self.weights[key][0], q, rtol=0, atol=1e-12):
                            raise ValueError('Public query transform differs within mode/frame')

    def batch(self, units, configs, frames):
        if not (len(units) == len(configs) == len(frames)):
            raise ValueError('Batch identity lengths differ')
        values = [window_inputs(self.raw[int(u)][int(c)], self.poses[int(u), int(c)], int(f))
                  for u, c, f in zip(units, configs, frames)]
        return (np.stack([v[0] for v in values]), np.stack([v[1] for v in values]),
                np.stack([self.weights[int(u) % 3, int(f)][1] for u, f in zip(units, frames)]))

    def close(self):
        self.raw.clear()
        self.poses.clear()
        self.weights.clear()


def normalized_labels(x):
    x = np.asarray(x, np.float32)
    total = x.sum(-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(total <= 0):
        raise ValueError('Invalid surface probability target')
    return x / total


class SurfaceLabels:
    def __init__(self, digest):
        self.maps, self.hashes = {}, {}
        aggregate = read(OUT / 'labels_receipt.json')
        if aggregate['status'] != 'COMPLETE' or aggregate['plan_sha256'] != digest:
            raise ValueError('Surface label aggregate is not COMPLETE for this PLAN')
        for unit in SPLITS['train']:
            folder = OUT / 'labels/train' / f'unit{unit}'
            record = read(folder / 'receipt.json')
            if record['status'] != 'COMPLETE' or record['plan_sha256'] != digest or record['unit'] != unit:
                raise ValueError('Surface label unit receipt mismatch')
            if aggregate['unit_receipt_sha256'][f'train/unit{unit}/receipt.json'] != sha(folder / 'receipt.json'):
                raise ValueError('Surface aggregate/unit receipt identity mismatch')
            for name in ('labels.npy', 'metadata.npz'):
                p = folder / name
                if sha(p) != record['output_sha256'][name]:
                    raise ValueError('Surface label hash mismatch')
                self.hashes[str(p)] = record['output_sha256'][name]
            with np.load(folder / 'metadata.npz', allow_pickle=False) as metadata:
                if not np.array_equal(metadata['configs'], np.arange(22)) or not np.array_equal(metadata['frames'], FRAMES):
                    raise ValueError('Surface label metadata order mismatch')
            z = np.load(folder / 'labels.npy', mmap_mode='r')
            if z.dtype != np.float16 or z.shape != (22, 13, 16, 16, 129):
                raise ValueError('Surface label axes/dtype mismatch')
            self.maps[unit] = z

    def batch(self, units, configs, frames):
        return normalized_labels(np.stack([self.maps[int(u)][int(c), int(f)-3]
                                          for u, c, f in zip(units, configs, frames)]))

    def close(self):
        for x in self.maps.values():
            x._mmap.close()
        self.maps.clear()


def cuda_runtime():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=torch.cuda.get_device_name(), tf32=False)


def verify_receipt(path, digest):
    r = read(path)
    if r['status'] != 'COMPLETE' or r['plan_sha256'] != digest:
        raise ValueError('Receipt plan/status mismatch: '+str(path))
    for relative, expected in r['output_sha256'].items():
        if sha(OUT / relative) != expected:
            raise ValueError('Receipt output changed: '+relative)
    return r


def prepare_m3():
    """One-time retained-voxel baseline inference; no surface arrays are opened."""
    digest = plan_sha()
    receipt = OUT / 'm3_train_receipt.json'
    if receipt.exists():
        return verify_receipt(receipt, digest)
    runtime = cuda_runtime()
    from cnh_cvr_projection import query_masks
    paths, meta = NT.metadata()
    voxpaths = [p.with_name(p.name.replace('metadata_', 'features_')).with_suffix('.npy') for p in paths]
    sources = [Path(__file__), Path(CP.__file__), Path(SS.__file__), *paths, *voxpaths, *MC.model_paths('M3')]
    input_hashes = {str(p): sha(p) for p in sources}
    scores = np.empty((len(meta['unit']), 2), np.float32)
    nets, masks = [], None
    try:
        for path in MC.model_paths('M3'):
            net = CP.CVR().cuda().eval()
            net.load_state_dict(torch.load(path, map_location='cuda', weights_only=True), strict=True)
            nets.append(net)
        masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
        offset = 0
        with torch.inference_mode():
            for path in voxpaths:
                x = np.load(path, mmap_mode='r')
                if x.dtype != np.float16 or x.shape[1:] != (3, 24, 17, 33):
                    raise ValueError('M3 voxel axes/dtype mismatch')
                for row in range(0, len(x), 128):
                    xb = torch.as_tensor(np.array(x[row:row+128]), device='cuda').float()
                    xb[:, 0] = xb[:, 0].sign()*xb[:, 0].abs().log1p()
                    xb[:, 2] = xb[:, 2].sign()*xb[:, 2].abs().log1p()
                    xb[:, 1] /= 8
                    xb = torch.cat((xb, masks[None].expand(len(xb), -1, -1, -1, -1)), 1)
                    scores[offset+row:offset+row+len(xb)] = torch.stack([net(xb) for net in nets]).mean(0).cpu().numpy()
                offset += len(x)
                x._mmap.close()
        if offset != len(scores) or not np.isfinite(scores).all():
            raise ValueError('Baseline row count/nonfinite scores')
        arrays = {}
        for unit in SPLITS['train']:
            ids = np.flatnonzero(meta['unit'] == unit)
            array = np.empty((22, 13, 2), np.float32)
            array[meta['config'][ids], meta['frame'][ids]-3] = scores[ids]
            arrays[str(unit)] = array
        path = OUT / 'm3_train_scores.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, **arrays)
        result = dict(status='COMPLETE', plan_sha256=digest, input_sha256=input_hashes,
                      output_sha256={path.name: sha(path)}, runtime=runtime, frames=FRAMES.tolist(),
                      units=SPLITS['train'], ensemble='five original frozen M3 models, mean raw logits')
        create_json(receipt, result)
        return result
    finally:
        nets.clear()
        net = masks = xb = None
        gc.collect()
        torch.cuda.empty_cache()


def training_rows():
    _, meta = NT.metadata()
    with np.load(ML.OUT / 'train_labels.npz', allow_pickle=False) as z:
        labels = z['M3'].astype(np.float32)
    order = np.lexsort((meta['frame'], meta['config'], meta['unit']))
    if labels.shape != (27456, 2):
        raise ValueError('Original M3 train labels shape mismatch')
    return {k: meta[k][order] for k in ('unit', 'config', 'frame')}, labels[order]


def load_epoch(folder, digest):
    latest = None
    for epoch in range(1, EPOCHS+1):
        path = folder / f'epoch{epoch}.pt'
        receipt = path.with_suffix('.json')
        if not receipt.exists():
            if path.exists() or any((folder / f'epoch{j}.pt').exists() or (folder / f'epoch{j}.json').exists() for j in range(epoch+1, EPOCHS+1)):
                raise ValueError('Partial/gapped checkpoint needs inspection')
            break
        r = read(receipt)
        if r['status'] != 'COMPLETE' or r['request_sha256'] != digest or r['sha256'] != sha(path):
            raise ValueError('Checkpoint hash/identity mismatch')
        latest = torch.load(path, map_location='cpu', weights_only=True)
        if latest['epoch'] != epoch or latest['request_sha256'] != digest:
            raise ValueError('Checkpoint internal identity mismatch')
    return latest


def train():
    digest = plan_sha()
    runtime = cuda_runtime()
    verify_receipt(OUT / 'm3_train_receipt.json', digest)
    data = NativeInputs('train')
    surfaces = net = opt = scheduler = None
    try:
        rows, labels = training_rows()
        with np.load(OUT / 'm3_train_scores.npz', allow_pickle=False) as z:
            baseline = np.stack([z[str(u)][c, f-3] for u, c, f in zip(rows['unit'], rows['config'], rows['frame'])])
        from cnh_surface_distribution_data import __file__ as data_source
        request = dict(plan_sha256=digest, source_sha256={str(p): sha(p) for p in
                       [Path(__file__), Path(data_source), Path(CP.__file__)]},
                       native_sha256=data.input_sha256, m3_receipt_sha256=sha(OUT / 'm3_train_receipt.json'),
                       labels_receipt_sha256=sha(OUT / 'labels_receipt.json'),
                       original_M3_labels_sha256=sha(ML.OUT / 'train_labels.npz'), runtime=runtime,
                       recipe=dict(epochs=EPOCHS, batch_size=BATCH, learning_rate=LR, weight_decay=WD,
                                   optimizer='AdamW', scheduler='CosineAnnealingLR T_max=10', seed=0,
                                   main='original M3 frame BCE', SURF_aux='mean microzone soft CE, weight=1',
                                   BCEO_surface_labels_read=False, head='777->64 GELU->1; final layer zero'))
        req = OUT / 'training_request.json'
        if req.exists():
            if read(req) != request:
                raise ValueError('Frozen training request changed')
        else:
            create_json(req, request)
        rd = sha(req)
        done = OUT / 'training_receipt.json'
        if done.exists():
            result = read(done)
            if result['status'] != 'COMPLETE' or result['request_sha256'] != rd or any(sha(OUT / p) != s for p, s in result['models_sha256'].items()):
                raise ValueError('Completed training changed')
            return result
        histories, models, initial_hashes = {}, {}, {}
        for arm in ARMS:
            if arm == 'SURF':
                surfaces = SurfaceLabels(digest)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            rng = np.random.default_rng(0)
            net = SurfaceCVR().cuda()
            initial_hashes[arm] = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            folder = OUT / 'checkpoints' / arm / 'seed0'
            folder.mkdir(parents=True, exist_ok=True)
            last = load_epoch(folder, rd)
            history, start = [], 0
            if last is not None:
                if last['arm'] != arm:
                    raise ValueError('Checkpoint arm mismatch')
                net.load_state_dict(last['model'], strict=True)
                opt.load_state_dict(last['optimizer'])
                scheduler.load_state_dict(last['scheduler'])
                rng.bit_generator.state = last['numpy_rng']
                torch.set_rng_state(last['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([v.cpu() for v in last['cuda_rng']])
                history, start = last['history'], last['epoch']
            for epoch in range(start, EPOCHS):
                began = time.monotonic()
                order = rng.permutation(len(labels))
                losses = np.zeros(3)
                net.train()
                for offset in range(0, len(order), BATCH):
                    ids = order[offset:offset+BATCH]
                    u, c, f = (rows[k][ids] for k in ('unit', 'config', 'frame'))
                    x, pose, weights = [torch.as_tensor(v, device='cuda') for v in data.batch(u, c, f)]
                    base = torch.as_tensor(baseline[ids], device='cuda')
                    target = torch.as_tensor(labels[ids], device='cuda')
                    opt.zero_grad(set_to_none=True)
                    alarm, logp = net(x, pose, weights, base)
                    main = F.binary_cross_entropy_with_logits(alarm, target)
                    aux = -(torch.as_tensor(surfaces.batch(u, c, f), device='cuda') * logp).sum(-1).mean() if arm == 'SURF' else main.new_zeros(())
                    loss = main + aux
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite surface training loss')
                    loss.backward()
                    opt.step()
                    losses += np.asarray([loss.item(), main.item(), aux.item()])*len(ids)
                scheduler.step()
                entry = dict(epoch=epoch+1, loss=float(losses[0]/len(labels)), main_loss=float(losses[1]/len(labels)),
                             surface_loss=float(losses[2]/len(labels)), permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                             elapsed_s=time.monotonic()-began)
                history.append(entry)
                payload = dict(epoch=epoch+1, arm=arm, request_sha256=rd, model=net.state_dict(), optimizer=opt.state_dict(),
                               scheduler=scheduler.state_dict(), numpy_rng=rng.bit_generator.state,
                               torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), history=history)
                path = folder / f'epoch{epoch+1}.pt'
                with path.open('xb') as stream:
                    torch.save(payload, stream)
                create_json(path.with_suffix('.json'), dict(status='COMPLETE', request_sha256=rd, epoch=epoch+1, sha256=sha(path)))
                print(arm, entry, flush=True)
            final = OUT / 'models' / arm / 'model_seed0.pt'
            final.parent.mkdir(parents=True, exist_ok=True)
            state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            if not all(bool(torch.isfinite(v).all()) for v in state.values()):
                raise FloatingPointError('Nonfinite deployment parameters')
            if final.exists():
                old = torch.load(final, map_location='cpu', weights_only=True)
                if old.keys() != state.keys() or any(not torch.equal(old[k], v) for k, v in state.items()):
                    raise ValueError('Existing deployment differs from last epoch')
            else:
                with final.open('xb') as stream:
                    torch.save(state, stream)
            models[str(final.relative_to(OUT)).replace('\\', '/')] = sha(final)
            histories[arm] = history
            net = opt = scheduler = last = payload = None
            gc.collect()
            torch.cuda.empty_cache()
        if initial_hashes['BCEO'] != initial_hashes['SURF'] or [v['permutation_sha256'] for v in histories['BCEO']] != [v['permutation_sha256'] for v in histories['SURF']]:
            raise ValueError('Paired initialization/order mismatch')
        if plan_sha() != digest or sha(req) != rd or any(sha(Path(p)) != s for p, s in request['source_sha256'].items()):
            raise ValueError('Frozen training inputs changed during run')
        result = dict(status='COMPLETE', plan_sha256=digest, request_sha256=rd, models_sha256=models,
                      labels_receipt_sha256=request['labels_receipt_sha256'], source_sha256=request['source_sha256'],
                      runtime=runtime, recipe=request['recipe'], history=histories, paired_shuffles_exact=True,
                      paired_initialization_exact=True, seeds=[0], baseline='frozen five-seed M3 mean; not trained',
                      unknown_semantics='Geometric no-first-hit/out-of-range bin only; not measured low-SNR or product UNKNOWN',
                      limit='One new-branch seed; no seed stability evidence')
        create_json(done, result)
        return result
    finally:
        data.close()
        if surfaces is not None:
            surfaces.close()
        net = opt = scheduler = None
        gc.collect()
        torch.cuda.empty_cache()


def check(gpu=False):
    """Synthetic mechanics only; never loads cohort observations or labels."""
    torch.set_num_threads(2)
    x = torch.arange(32*8*8*16).reshape(1, 32, 8, 8, 16)
    shuffled = radial_shuffle(x)
    for dy, dx, dr, y, z, r in [(0, 0, 0, 0, 0, 0), (1, 0, 7, 3, 4, 9), (1, 1, 7, 7, 7, 15)]:
        assert shuffled[0, y*2+dy, z*2+dx, r*8+dr] == x[0, (dy*2+dx)*8+dr, y, z, r]
    raw = np.arange(16*8*8*16, dtype=np.float32).reshape(16, 8, 8, 16)
    poses = np.repeat(np.eye(4)[None], 16, axis=0)
    poses[:, 0, 3] = np.arange(16)*.1
    a, p = window_inputs(raw, poses, 3)
    assert not a[:4].any() and np.array_equal(p[-8:], [0, 0, 0, 0, 1, 1, 1, 1])
    assert np.allclose(p[:96].reshape(8, 3, 4)[-1], np.eye(4)[:3])
    raw[4:] = -999
    poses[4:, 0, 3] = -999
    b, q = window_inputs(raw, poses, 3)
    assert np.array_equal(a, b) and np.array_equal(p, q)
    device, batch = ('cuda', 64) if gpu else ('cpu', 2)
    if gpu:
        cuda_runtime()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    net = SurfaceCVR().to(device)
    native = torch.randn(batch, 8, 8, 8, 16, device=device)
    pose = torch.randn(batch, 104, device=device)
    weights = torch.ones(batch, 2, 16, 16, 129, device=device)
    weights[..., 128] = 0
    baseline = torch.randn(batch, 2, device=device, requires_grad=True)
    begin = time.monotonic()
    pred, logp = net(native, pose, weights, baseline)
    torch.testing.assert_close(pred, baseline, rtol=0, atol=0)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones_like(logp[..., 0]), atol=1e-6, rtol=0)
    features = net.alarm_features(logp, weights, baseline)
    assert features.shape == (batch, 2, 777)
    torch.testing.assert_close(features[:, 0, :256], 1-features[:, 0, 512:768], atol=1e-6, rtol=0)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred)).backward()
    assert baseline.grad is None
    assert bool(net.alarm[-1].weight.grad.abs().sum() > 0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    pred, logp = net(native, pose, weights, baseline)
    F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred)).backward()
    assert bool(net.conv1.weight.grad.abs().sum() > 0)
    opt.zero_grad(set_to_none=True)
    logp = net.distribution(native, pose)
    target = torch.zeros_like(logp)
    target[..., 17] = 1
    ce = -(target*logp).sum(-1).mean()
    torch.testing.assert_close(ce, -logp[..., 17].mean())
    ce.backward()
    assert bool(net.conv1.weight.grad.abs().sum() > 0)
    assert np.allclose(normalized_labels(np.ones((2, 129), np.float16)).sum(-1), 1)
    if gpu:
        torch.cuda.synchronize()
    result = dict(status='PASS', synthetic_only=True, device=device, batch=batch,
                  parameters=sum(v.numel() for v in net.parameters()), elapsed_s=time.monotonic()-begin,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else None,
                  checks=['shuffle axes', 'causal left padded history', '104 pose layout', 'normalized129',
                          '777 unknown-aware bottleneck', 'initial exact M3', 'frozen baseline',
                          'BCEO second-step encoder gradient', 'SURF CE encoder gradient'])
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['check', 'prepare-m3', 'train'], required=True)
    parser.add_argument('--gpu', action='store_true', help='Synthetic batch64 resource check only')
    args = parser.parse_args()
    if args.stage == 'check':
        check(args.gpu)
    elif args.stage == 'prepare-m3':
        prepare_m3()
    else:
        train()
