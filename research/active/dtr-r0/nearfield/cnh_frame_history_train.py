"""Paired AGG/HIST adapters on retained CVR observations; no materialization.

Keep original CVR modules and old five-input path. The only new observation
path is a 16->8->2 pointwise adapter followed by a zero-initialized residual
stem. Training uses original M3 frame BCE, not the failed auxiliary losses.
"""
import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

import cnh_nested_corridor_train as NT
from cnh_cvr_pilot import CVR
from cnh_cvr_projection import SHAPE, query_masks

OUT = NT.SS.WORK/'cnh-frame-history-20261002'
RUN_ID = 'CNH_FRAME_HISTORY_20261002'
ARMS = ('AGG', 'HIST')
SPLITS = dict(train=list(range(93000, 93096)), calib=list(range(95000, 95048)),
              evaluation=list(range(96000, 96096)))
FRAMES = np.arange(3, 16)
EPOCHS, BATCH, LR, WD = 5, 128, 2e-4, 1e-4
ADDED_PARAMETERS = 1018


def plan_sha():
    path = OUT/'PLAN.json'
    if not path.is_file() or RUN_ID not in (NT.HERE.parent/'RUNS.md').read_text(encoding='utf8'):
        raise RuntimeError('Frozen PLAN and RUNS pre-registration required')
    plan = NT.read(path)
    expected = dict(run=RUN_ID, train_units=SPLITS['train'], train_arms=list(ARMS),
                    decision_frames=FRAMES.tolist(), seeds=list(range(5)), main_half_width=.33)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen PLAN differs from training definition')
    expected = dict(epochs=EPOCHS, optimizer='AdamW', learning_rate=LR, weight_decay=WD,
                    scheduler='CosineAnnealingLR T_max=5', batch_rows=BATCH)
    if any(plan['training'].get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen optimizer or batch differs')
    for name, checksum in plan['prior_sha256'].items():
        path_ = Path(name)
        if (path_.name in ('train_labels.npz', 'labels_receipt.json') or path_.name.startswith('model_seed')) and NT.sha(path_) != checksum:
            raise ValueError('Frozen training input changed: '+name)
    return NT.sha(path)


class HistoryCVR(CVR):
    """Same old body/emb/head keys, with a parallel residual before first GELU."""
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Sequential(torch.nn.Conv3d(16, 8, 1), torch.nn.GELU(),
                                           torch.nn.Conv3d(8, 2, 1))
        self.residual = torch.nn.Conv3d(2, 16, 3, stride=2, padding=1, bias=False)
        torch.nn.init.zeros_(self.residual.weight)

    def load_m3_state(self, state):
        added = {k for k in self.state_dict() if k.startswith(('adapter.', 'residual.'))}
        if set(state) != set(self.state_dict())-added:
            raise ValueError('Initialization must contain exactly original CVR keys')
        missing, unexpected = self.load_state_dict(state, strict=False)
        if set(missing) != added or unexpected:
            raise ValueError('Original M3 initialization mismatch')

    def forward(self, old_input, history_input):
        if old_input.ndim != 5 or history_input.ndim != 5 or old_input.shape[1] != 5 or history_input.shape[1] != 16:
            raise ValueError('Expected original five channels and separate sixteen history channels')
        if old_input.shape[0] != history_input.shape[0] or old_input.shape[2:] != history_input.shape[2:]:
            raise ValueError('Old/history input alignment mismatch')
        f = self.body[0](old_input)+self.residual(self.adapter(history_input))
        for layer in self.body[1:]:
            f = layer(f)
        mask = torch.nn.functional.adaptive_avg_pool3d(old_input[:, 3:5], f.shape[2:])[:, :, None]
        ff = f[:, None]
        mx = ff.masked_fill(mask == 0, -1e4).flatten(3).max(-1).values
        mean = (ff*mask).flatten(3).sum(-1)/mask.flatten(3).sum(-1).clamp_min(1e-8)
        emb = self.emb.weight[None].expand(len(old_input), -1, -1)
        return self.head(torch.cat([mx, mean, emb], -1)).squeeze(-1)


def prepare_old(old3, masks):
    """Exact original preprocessing; input half cache is never modified."""
    if old3.dtype != torch.float16 or old3.ndim != 5 or old3.shape[1:] != (3, *SHAPE):
        raise ValueError('Expected original half [batch,3,24,17,33] cache')
    x = old3.float()
    x[:, 0] = x[:, 0].sign()*x[:, 0].abs().log1p()
    x[:, 2] = x[:, 2].sign()*x[:, 2].abs().log1p()
    x[:, 1] /= 8
    return torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)


def prepare_history(arm, old3, mass_half=None, coverage_uint8=None):
    """[e0..e7,c0..c7], oldest first/current last, exactly frozen quantization."""
    if old3.dtype != torch.float16 or old3.ndim != 5 or old3.shape[1:] != (3, *SHAPE):
        raise ValueError('Expected original half cache for aligned batch/device')
    if arm == 'AGG':
        if mass_half is not None or coverage_uint8 is not None:
            raise ValueError('AGG must not receive any real frame history or coverage')
        old = old3.float()
        past = ((old[:, 0]-old[:, 2])/7)[:, None].expand(-1, 7, -1, -1, -1)
        e = torch.cat((past, old[:, 2:3]), 1).half().float()
        c = (old[:, 1:2]/8).expand(-1, 8, -1, -1, -1).half().float()
    elif arm == 'HIST':
        if mass_half is None or coverage_uint8 is None:
            raise ValueError('HIST needs real per-exposure mass and midpoint counts')
        mass = torch.as_tensor(mass_half, device=old3.device)
        count = torch.as_tensor(coverage_uint8, device=old3.device)
        expected = (len(old3), 8, *SHAPE)
        if mass.dtype != torch.float16 or count.dtype != torch.uint8 or tuple(mass.shape) != expected or tuple(count.shape) != expected:
            raise ValueError('History mass/count dtype or axes mismatch')
        if bool((count > 27).any()):
            raise ValueError('Coverage is the integer valid-midpoint count 0..27')
        e = mass.float()
        c = (count.double()/27).float().half().float()
    else:
        raise ValueError('Unknown history representation: '+str(arm))
    return torch.cat((e.sign()*e.abs().log1p(), c), 1)


class HistoryStore:
    """Lazy read-only unit memmaps; copy only requested minibatch rows to RAM."""
    def __init__(self, split, units=None, verify_hashes=True):
        if split not in SPLITS:
            raise ValueError('Unknown history split')
        self.split = split
        self.units = list(SPLITS[split] if units is None else units)
        if len(set(self.units)) != len(self.units) or not set(self.units).issubset(SPLITS[split]):
            raise ValueError('Unexpected or duplicate history units')
        self.configs = 22 if split == 'train' else 40
        self.maps, self.receipts, self.input_sha256 = {}, {}, {}
        expected_plan = NT.sha(OUT/'PLAN.json')
        request_path = OUT/'request.json'
        expected_request = NT.sha(request_path)
        self.input_sha256[str(request_path)] = expected_request
        for unit in self.units:
            folder = OUT/'cache'/split/f'unit{unit}'
            path = folder/'receipt.json'
            receipt = NT.read(path)
            if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != expected_plan:
                raise ValueError('History cache incomplete or wrong PLAN: '+str(folder))
            if receipt.get('request_sha256') != expected_request:
                raise ValueError('History cache belongs to a different projection request')
            if receipt.get('split') != split or receipt.get('unit') != unit:
                raise ValueError('History unit identity mismatch')
            expected = [self.configs, 13, 8, *SHAPE]
            if receipt.get('shape') != expected or receipt.get('frames') != FRAMES.tolist():
                raise ValueError('History axes/frame metadata mismatch')
            if receipt.get('configs') != self.configs or receipt.get('slots') != 8:
                raise ValueError('History config count or slot count mismatch')
            if receipt.get('old3_windows_bit_exact') != self.configs*13 or not all(receipt.get(k) is True for k in ('latest_exact', 'coverage_recovery_exact')):
                raise ValueError('Original voxel/current/coverage parity did not pass')
            declared = receipt['output_sha256']
            if set(declared) != {'mass.npy', 'coverage.npy'}:
                raise ValueError('History receipt must name mass/count arrays only')
            self.input_sha256[str(path)] = NT.sha(path)
            for name, checksum in declared.items():
                file = folder/name
                if verify_hashes and NT.sha(file) != checksum:
                    raise ValueError('History cache changed: '+str(file))
                array = np.load(file, mmap_mode='r', allow_pickle=False)
                try:
                    dtype = np.float16 if name == 'mass.npy' else np.uint8
                    if tuple(array.shape) != tuple(expected) or array.dtype != dtype:
                        raise ValueError('History array shape/dtype mismatch')
                finally:
                    array._mmap.close()
                self.input_sha256[str(file)] = checksum
            self.receipts[unit] = receipt

    def batch(self, units, configs, frames):
        units, configs, frames = (np.asarray(x) for x in (units, configs, frames))
        if units.ndim != 1 or configs.shape != units.shape or frames.shape != units.shape:
            raise ValueError('History row identities must be aligned vectors')
        if not np.isin(units, self.units).all() or not np.isin(frames, FRAMES).all() or np.any(configs < 0) or np.any(configs >= self.configs):
            raise ValueError('History request outside admitted unit/config/frame range')
        if any(not np.array_equal(x, x.astype(np.int64)) for x in (units, configs, frames)):
            raise ValueError('History row IDs must be integer valued')
        mass = np.empty((len(units), 8, *SHAPE), dtype=np.float16)
        cover = np.empty(mass.shape, dtype=np.uint8)
        for unit in np.unique(units):
            unit = int(unit)
            if unit not in self.maps:
                folder = OUT/'cache'/self.split/f'unit{unit}'
                self.maps[unit] = tuple(np.load(folder/name, mmap_mode='r', allow_pickle=False)
                                        for name in ('mass.npy', 'coverage.npy'))
            take = np.flatnonzero(units == unit)
            c, f = configs[take].astype(int), frames[take].astype(int)-3
            mass[take] = self.maps[unit][0][c, f]
            cover[take] = self.maps[unit][1][c, f]
        if not np.isfinite(mass).all() or np.any(cover > 27):
            raise ValueError('Nonfinite evidence or invalid midpoint count')
        return mass, cover

    def close(self):
        for arrays in self.maps.values():
            for array in arrays:
                array._mmap.close()
        self.maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_labels():
    metadata_paths, meta = NT.metadata()
    receipt = NT.read(NT.OUT/'labels_receipt.json')
    label_path = NT.OUT/'train_labels.npz'
    if receipt['status'] != 'COMPLETE' or not receipt['m3_exact'] or NT.sha(label_path) != receipt['output_sha256']:
        raise ValueError('Retained exact M3 labels unavailable')
    for path, checksum in receipt['identity']['metadata_sha256'].items():
        if NT.sha(path) != checksum:
            raise ValueError('Retained label metadata changed')
    with np.load(label_path, allow_pickle=False) as labels:
        if any(not np.array_equal(meta[k], labels[k]) for k in ('unit', 'config', 'frame')):
            raise ValueError('Label and old voxel row order mismatch')
        y = labels['main'].astype(np.float32)
    with np.load(NT.ML.OUT/'train_labels.npz', allow_pickle=False) as old:
        if not np.array_equal(y, old['M3']):
            raise ValueError('Supervision must remain original M3 frame labels')
    return metadata_paths, meta, y


def recipe():
    return dict(arms=list(ARMS), seeds=list(range(5)), epochs=EPOCHS, batch_rows=BATCH,
        optimizer='AdamW', lr=LR, weight_decay=WD, cosine_t_max=EPOCHS, added_parameters=ADDED_PARAMETERS,
        supervision='original per-frame M3 mean BCE only', initialization='original corresponding M3 seed; all parameters trainable',
        branch='Conv1x1(16,8),GELU,Conv1x1(8,2); Conv3d(2,16,k3,s2,p1,biasFalse) added before original first GELU',
        zero_initialization='residual only; adapter random', history_layout='e0..e7,c0..c7; oldest to newest/current slot7',
        AGG='old3half->float; past=(total-current)/7 x7,current x1; count/8 x8; pseudo E/C half roundtrip',
        HIST='E half->float; C=(K double/27).float().half().float(); E signedlog1p,C unscaled',
        shuffle='numpy default_rng(seed) complete row permutation; paired arms',
        deployment='five-seed raw logits mean and unchanged causal five-score smoothing')


def training_request(runtime, store):
    metadata_paths, _ = NT.metadata()
    parts = [p.with_name(p.name.replace('metadata_', 'features_').replace('.npz', '.npy')) for p in metadata_paths]
    canary_path = OUT/'model_canary.json'
    canary = NT.read(canary_path)
    if canary.get('status') != 'PASS' or canary.get('source_sha256') != NT.sha(__file__) or canary.get('plan_sha256') != plan_sha():
        raise ValueError('Model/input canary must pass for current source and PLAN')
    for path, checksum in canary['input_sha256'].items():
        if NT.sha(path) != checksum:
            raise ValueError('Canary input changed')
    sources = [Path(__file__), Path(NT.__file__), NT.HERE/'cnh_cvr_pilot.py',
               NT.HERE/'cnh_cvr_projection.py', NT.HERE/'cnh_structure_space.py', NT.HERE/'cnh_near_range.py']
    value = dict(plan_sha256=plan_sha(), source_sha256={str(p): NT.sha(p) for p in sources},
        canary_sha256=NT.sha(canary_path), labels_receipt_sha256=NT.sha(NT.OUT/'labels_receipt.json'),
        labels_sha256=NT.sha(NT.OUT/'train_labels.npz'), metadata_sha256={str(p): NT.sha(p) for p in metadata_paths},
        voxel_sha256={str(p): NT.sha(p) for p in parts}, history_cache_sha256=store.input_sha256,
        initial_models_sha256={str(NT.ML.OUT/'models/M3'/f'model_seed{s}.pt'):
            NT.sha(NT.ML.OUT/'models/M3'/f'model_seed{s}.pt') for s in range(5)},
        recipe=recipe(), runtime=runtime)
    path = OUT/'training_request.json'
    if path.exists():
        if NT.read(path) != value:
            raise ValueError('Training identity changed; cannot resume')
    else:
        NT.create_json(path, value)
    return value, NT.sha(path), parts, metadata_paths


def train():
    plan_sha()
    _, meta, labels = load_labels()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Validated CUDA runtime required for training')
    runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        threads=2, tf32=False, cudnn_benchmark=torch.backends.cudnn.benchmark,
        old_loading='one CUDA half tensor; 1024-row streaming host blocks',
        history_loading='read-only per-unit CPU memmaps; selected minibatch only copied to CUDA')
    with HistoryStore('train') as store:
        frozen, digest, parts, metadata_paths = training_request(runtime, store)
        done = OUT/'training_receipt.json'
        if done.exists():
            prior = NT.read(done)
            if prior['status'] != 'COMPLETE' or prior['request_sha256'] != digest:
                raise ValueError('Completed training identity differs')
            for name, checksum in prior['models_sha256'].items():
                if NT.sha(OUT/name) != checksum:
                    raise ValueError('Completed deployment model changed')
            print('Existing COMPLETE AGG/HIST training verified', flush=True)
            return prior
        started = time.monotonic()
        X = Y = masks = net = opt = scheduler = last = state = deployed = payload = None
        old = h = prediction = loss = mass = coverage = None
        histories, model_hashes = {}, {}
        try:
            X = torch.empty((27456, 3, *SHAPE), dtype=torch.float16, device='cuda')
            offset = 0
            for path, metadata_path in zip(parts, metadata_paths):
                data = np.load(path, mmap_mode='r', allow_pickle=False)
                try:
                    with np.load(metadata_path, allow_pickle=False) as m:
                        if len(data) != len(m['unit']):
                            raise ValueError('Old voxel/metadata shard length mismatch')
                    if data.shape[1:] != (3, *SHAPE) or data.dtype != np.float16:
                        raise ValueError('Old voxel shape/dtype differs')
                    for begin in range(0, len(data), 1024):
                        block = np.array(data[begin:begin+1024])
                        if not np.isfinite(block).all():
                            raise ValueError('Old voxels must be finite')
                        X[offset+begin:offset+begin+len(block)] = torch.as_tensor(block, device='cuda')
                    offset += len(data)
                finally:
                    data._mmap.close()
            if offset != 27456:
                raise ValueError('Old voxel row count mismatch')
            Y = torch.as_tensor(labels, dtype=torch.float32, device='cuda')
            masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')
            for arm in ARMS:
                for seed in range(5):
                    folder = OUT/'checkpoints'/arm/f'seed{seed}'
                    folder.mkdir(parents=True, exist_ok=True)
                    last = NT.load_epoch(torch, folder, digest)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    rng = np.random.default_rng(seed)
                    net = HistoryCVR().cuda()
                    state = torch.load(NT.ML.OUT/'models/M3'/f'model_seed{seed}.pt', weights_only=True, map_location='cuda')
                    net.load_m3_state(state)
                    state = None
                    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
                    history, first_epoch = [], 0
                    if last is not None:
                        if last['arm'] != arm or last['seed'] != seed:
                            raise ValueError('Checkpoint arm/seed mismatch')
                        net.load_state_dict(last['model'], strict=True)
                        opt.load_state_dict(last['optimizer'])
                        scheduler.load_state_dict(last['scheduler'])
                        rng.bit_generator.state = last['numpy_rng']
                        torch.set_rng_state(last['torch_rng'].cpu())
                        torch.cuda.set_rng_state_all([x.cpu() for x in last['cuda_rng']])
                        history, first_epoch = last['history'], last['epoch']
                    for epoch in range(first_epoch, EPOCHS):
                        begin = time.monotonic()
                        net.train()
                        order = rng.permutation(len(X))
                        total = 0.
                        for start in range(0, len(order), BATCH):
                            selected = order[start:start+BATCH]
                            pos = torch.as_tensor(selected, device='cuda')
                            old = X[pos]
                            if arm == 'HIST':
                                mass, coverage = store.batch(meta['unit'][selected], meta['config'][selected], meta['frame'][selected])
                                h = prepare_history(arm, old, mass, coverage)
                            else:
                                h = prepare_history(arm, old)
                            opt.zero_grad(set_to_none=True)
                            prediction = net(prepare_old(old, masks), h)
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, Y[pos])
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError('Nonfinite training loss')
                            loss.backward()
                            opt.step()
                            total += float(loss.detach())*len(selected)
                        scheduler.step()
                        if any(not bool(torch.isfinite(p).all()) for p in net.parameters()):
                            raise FloatingPointError('Nonfinite trained parameters')
                        entry = dict(epoch=epoch+1, loss=total/len(X),
                            permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest(), elapsed_s=time.monotonic()-begin)
                        history.append(entry)
                        payload = dict(epoch=epoch+1, arm=arm, seed=seed, request_sha256=digest,
                            model=net.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(), history=history)
                        path = folder/f'epoch{epoch+1}.pt'
                        with path.open('xb') as file:
                            torch.save(payload, file)
                        NT.create_json(path.with_suffix('.json'), dict(status='COMPLETE', arm=arm, seed=seed,
                            epoch=epoch+1, request_sha256=digest, sha256=NT.sha(path), runtime=runtime))
                        print('train', arm, seed, entry, flush=True)
                    deployed = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                    reference = HistoryCVR()
                    reference.load_state_dict(deployed, strict=True)
                    del reference
                    name = f'models/{arm}/model_seed{seed}.pt'
                    path = OUT/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        before = torch.load(path, weights_only=True, map_location='cpu')
                        if before.keys() != deployed.keys() or any(not torch.equal(before[k], deployed[k]) for k in deployed):
                            raise ValueError('Existing model differs from completed epoch')
                    else:
                        with path.open('xb') as file:
                            torch.save(deployed, file)
                    model_hashes[name] = NT.sha(path)
                    histories[f'{arm}|{seed}'] = history
                    net = opt = scheduler = last = deployed = payload = old = h = prediction = loss = None
                    mass = coverage = None
                    gc.collect()
                    torch.cuda.empty_cache()
            for seed in range(5):
                a = [row['permutation_sha256'] for row in histories[f'AGG|{seed}']]
                b = [row['permutation_sha256'] for row in histories[f'HIST|{seed}']]
                if a != b or len(a) != EPOCHS:
                    raise ValueError('Paired AGG/HIST row permutations differ')
            for path, checksum in store.input_sha256.items():
                if NT.sha(path) != checksum:
                    raise ValueError('History cache changed during training')
            if training_request(runtime, store)[:2] != (frozen, digest):
                raise ValueError('Training identity changed during execution')
            result = dict(status='COMPLETE', plan_sha256=frozen['plan_sha256'], request_sha256=digest,
                models_sha256=model_hashes, labels_receipt_sha256=frozen['labels_receipt_sha256'],
                source_sha256=frozen['source_sha256'], history_cache_sha256=store.input_sha256,
                runtime=runtime, recipe=recipe(), history=histories, paired_shuffles_exact=True,
                added_parameters=ADDED_PARAMETERS, elapsed_s=time.monotonic()-started,
                deployment='HistoryCVR complete state_dict; old CVR keys plus adapter/residual only')
            NT.create_json(done, result)
            print('COMPLETE paired AGG/HIST training', flush=True)
            return result
        finally:
            X = Y = masks = net = opt = scheduler = last = state = deployed = payload = None
            old = h = prediction = loss = mass = coverage = None
            gc.collect()
            torch.cuda.empty_cache()


def synthetic_check():
    """CPU mechanical check only; does not create a production canary receipt."""
    torch.set_num_threads(2)
    torch.manual_seed(2026100224)
    original = CVR().eval()
    net = HistoryCVR().eval()
    net.load_m3_state(original.state_dict())
    extra = sum(p.numel() for p in net.parameters())-sum(p.numel() for p in original.parameters())
    assert extra == ADDED_PARAMETERS
    masks = torch.tensor(query_masks())
    old = torch.randn(1, 3, *SHAPE).half()
    old[:, 1] = old[:, 1].abs().clamp(0, 8)
    saved = old.clone()
    mass = torch.randn(1, 8, *SHAPE).half()
    coverage = torch.randint(0, 28, mass.shape, dtype=torch.uint8)
    x = prepare_old(old, masks)
    # Original SS.prep performs this conversion on CUDA; keep this mechanical
    # check on CPU while independently constructing its exact tensor operations.
    expected_old = torch.from_numpy(np.array(old.numpy(), dtype=np.float32))
    expected_old[:, 0] = expected_old[:, 0].sign()*expected_old[:, 0].abs().log1p()
    expected_old[:, 2] = expected_old[:, 2].sign()*expected_old[:, 2].abs().log1p()
    expected_old[:, 1] /= 8
    expected_old = torch.cat((expected_old, masks[None]), 1)
    torch.testing.assert_close(x, expected_old, rtol=0, atol=0)
    assert torch.equal(old, saved)
    inputs = {arm: prepare_history(arm, old, mass, coverage) if arm == 'HIST' else prepare_history(arm, old) for arm in ARMS}
    with torch.no_grad():
        expected = original(x)
        for arm, h in inputs.items():
            torch.testing.assert_close(net(x, h), expected, rtol=0, atol=0)
    pseudo = np.repeat(((saved[:, 0].float().numpy()-saved[:, 2].float().numpy())/7)[:, None], 7, axis=1)
    pseudo = np.concatenate((pseudo, saved[:, 2:3].float().numpy()), axis=1).astype(np.float16).astype(np.float32)
    expected_e = np.sign(pseudo)*np.log1p(np.abs(pseudo))
    np.testing.assert_allclose(inputs['AGG'][:, :8].numpy(), expected_e, rtol=2e-7, atol=2e-7)
    levels = torch.arange(28, dtype=torch.uint8)
    actual = (levels.double()/27).float().half().float().numpy()
    np.testing.assert_array_equal(actual, (np.arange(28, dtype=np.float64)/27).astype(np.float32).astype(np.float16).astype(np.float32))
    try:
        prepare_history('AGG', old, mass, coverage)
    except ValueError:
        pass
    else:
        raise AssertionError('AGG accepted forbidden perframe information')
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    gradient = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        prediction = net(x, inputs['HIST'])
        torch.nn.functional.binary_cross_entropy_with_logits(prediction, torch.zeros_like(prediction)).backward()
        residual_grad = float(net.residual.weight.grad.abs().sum())
        adapter_grad = sum(float(p.grad.abs().sum()) for p in net.adapter.parameters() if p.grad is not None)
        gradient.append(dict(step=step+1, residual=residual_grad, adapter=adapter_grad))
        assert residual_grad > 0
        assert adapter_grad == 0 if step == 0 else adapter_grad > 0
        opt.step()
    torch.manual_seed(9)
    a = HistoryCVR().state_dict()
    torch.manual_seed(9)
    b = HistoryCVR().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    print('PASS synthetic CPU: 1018 parameters; exact M3/preprocess; AGG information restriction; K/C conversion; live residual and second-step adapter gradients', flush=True)
    return dict(added_parameters=extra, gradient_path=gradient, initial_m3_exact=True,
                old_preprocessing_exact=True, AGG_only_old3=True, coverage_28_levels_exact=True)


def canary():
    digest = plan_sha()
    path = OUT/'model_canary.json'
    if path.exists():
        prior = NT.read(path)
        if prior['status'] != 'PASS' or prior['source_sha256'] != NT.sha(__file__) or prior['plan_sha256'] != digest:
            raise ValueError('Existing canary identity differs')
        for name, checksum in prior['input_sha256'].items():
            if NT.sha(name) != checksum:
                raise ValueError('Canary input changed')
        print('Existing PASS model canary verified', flush=True)
        return prior
    completed = [u for u in SPLITS['train'] if (OUT/'cache/train'/f'unit{u}'/'receipt.json').is_file()]
    if not completed:
        raise RuntimeError('Parent must materialize at least the first complete train unit before actual-input canary')
    unit = completed[0]
    checks = synthetic_check()
    metadata_paths, meta, _ = load_labels()
    identities = [(unit, 0, 3), (unit, 20, 15)]
    triples = list(zip(meta['unit'], meta['config'], meta['frame']))
    selected = np.asarray([triples.index(x) for x in identities])
    old_rows = np.empty((2, 3, *SHAPE), np.float16)
    hashes = {}
    offset = 0
    for metadata_path in metadata_paths:
        feature = metadata_path.with_name(metadata_path.name.replace('metadata_', 'features_').replace('.npz', '.npy'))
        array = np.load(feature, mmap_mode='r', allow_pickle=False)
        try:
            take = np.flatnonzero((selected >= offset) & (selected < offset+len(array)))
            if len(take):
                old_rows[take] = array[selected[take]-offset]
                hashes[str(feature)], hashes[str(metadata_path)] = NT.sha(feature), NT.sha(metadata_path)
            offset += len(array)
        finally:
            array._mmap.close()
    with HistoryStore('train', [unit]) as store:
        mass, coverage = store.batch(*zip(*identities))
        hashes.update(store.input_sha256)
    assert not np.any(mass[0, :4]) and not np.any(coverage[0, :4])
    np.testing.assert_array_equal(mass[:, 7], old_rows[:, 2])
    masks = torch.tensor(query_masks())
    old = torch.from_numpy(old_rows)
    original_path = NT.ML.OUT/'models/M3/model_seed0.pt'
    state = torch.load(original_path, weights_only=True, map_location='cpu')
    original, net = CVR().eval(), HistoryCVR().eval()
    original.load_state_dict(state, strict=True)
    net.load_m3_state(state)
    histories = dict(AGG=prepare_history('AGG', old), HIST=prepare_history('HIST', old, mass, coverage))
    with torch.no_grad():
        x = prepare_old(old, masks)
        expected = original(x)
        for history in histories.values():
            torch.testing.assert_close(net(x, history), expected, rtol=0, atol=0)
    direct = np.concatenate((np.sign(mass.astype(np.float32))*np.log1p(np.abs(mass.astype(np.float32))),
        (coverage.astype(np.float64)/27).astype(np.float32).astype(np.float16).astype(np.float32)), axis=1)
    np.testing.assert_allclose(histories['HIST'].numpy(), direct, rtol=2e-7, atol=2e-7)
    hashes[str(original_path)] = NT.sha(original_path)
    checks.update(actual_input_parity=True, initial_actual_M3_exact=True,
        latest_mass_equals_original_current=True, early_left_padding_exact=True)
    result = dict(status='PASS', plan_sha256=digest, source_sha256=NT.sha(__file__),
        input_sha256=hashes, actual_rows=[list(x) for x in identities], checks=checks,
        scope='CPU model/input mechanism canary only; no cohort inference, training or evaluation')
    NT.create_json(path, result)
    print('PASS actual cached-input model canary', identities, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('canary', 'train'))
    args = parser.parse_args()
    {'canary': canary, 'train': train}[args.stage]()
