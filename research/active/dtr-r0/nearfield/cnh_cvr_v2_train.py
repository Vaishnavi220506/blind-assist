"""Frozen CVR v2 training: train-only fitting, persistent shared wall budget.

Features are materialized by the separately frozen projector; no scene generator
or truth geometry is imported here. Inference consumes only final 3-seed groups.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from cnh_cvr_pilot import CVR
from cnh_cvr_projection import query_masks

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-cvr-v2-20260929'
FAMILIES = ('boundary', 'mixed_surface', 'sidewall', 'general')
JOBS = [('CVR', f) for f in (*FAMILIES, 'allfamily')] + [('DUAL', f) for f in FAMILIES]
BUDGET = 7200.


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf8')
    # A concurrent read/antivirus can briefly hold the destination on Windows.
    # Retry the same atomic replacement without changing permissions or data.
    for attempt in range(7):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.02 * (2 ** attempt))


def checkpoint(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)


class BudgetStop(Exception):
    pass


class Budget:
    """Crash recovery conservatively charges elapsed wall since last heartbeat."""
    def __init__(self, out):
        self.path = out / 'training_budget.json'
        d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.seconds = float(d.get('seconds', 0))
        if d.get('running'):
            self.seconds += max(0., time.time() - d['heartbeat_unix'])
        self.last = time.monotonic(); self.tick(False)

    def tick(self, running=True):
        now = time.monotonic(); self.seconds += now - self.last; self.last = now
        save(self.path, dict(seconds=self.seconds, limit_seconds=BUDGET,
                             running=running, heartbeat_unix=time.time()))
        if running and self.seconds >= BUDGET:
            raise BudgetStop('shared training wall budget exhausted')


def pooled(f, masks):
    mask = masks[:, :, None]; ff = f[:, None]
    mx = ff.masked_fill(mask == 0, -1e4).flatten(3).max(-1).values
    mean = (ff * mask).flatten(3).sum(-1) / mask.flatten(3).sum(-1).clamp_min(1e-8)
    return torch.cat((mx, mean), -1)


class Dual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.voxel = CVR().body
        self.zone = torch.nn.Sequential(torch.nn.Conv3d(2, 16, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv3d(16, 32, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv3d(32, 32, 3, padding=1), torch.nn.GELU())
        self.emb = torch.nn.Embedding(2, 8)
        self.head = torch.nn.Sequential(torch.nn.Linear(136, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))

    def forward(self, x, zone, support):
        f = self.voxel(x)
        masks = torch.nn.functional.adaptive_avg_pool3d(x[:, 3:5], f.shape[2:])
        vf = pooled(f, masks); zf = pooled(self.zone(zone), support)
        emb = self.emb.weight[None].expand(len(x), -1, -1)
        return self.head(torch.cat((vf, zf, emb), -1)).squeeze(-1)


def model_for(arm):
    return CVR() if arm == 'CVR' else Dual()


class Data:
    def __init__(self, out, split, dual=False):
        p = out / 'data' / split
        self.x = np.load(p / 'features.npy', mmap_mode='r')
        with np.load(p / 'metadata.npz', allow_pickle=False) as z:
            self.meta = {k: z[k] for k in z.files}
        n = len(self.x)
        assert self.x.shape == (n, 3, 24, 17, 33) and self.x.dtype == np.float16
        assert self.meta['labels'].shape == (n, 2)
        assert np.isin(self.meta['labels'], [0, 1]).all()
        for k in ('family', 'unit', 'config', 'frame'): assert self.meta[k].shape == (n,)
        assert np.isin(self.meta['family'], FAMILIES).all()
        assert np.isin(self.meta['frame'], np.arange(3, 16) if split == 'train' else np.arange(11, 16)).all()
        self.z = np.load(p / 'zone.npy', mmap_mode='r') if dual else None
        self.s = np.load(p / 'support.npy', mmap_mode='r') if dual else None
        if dual:
            assert self.z.shape == self.s.shape == (n, 2, 8, 8, 16)
            assert self.s.dtype == bool
        self.masks = torch.as_tensor(query_masks(), dtype=torch.float32, device='cuda')

    def batch(self, ids):
        x = torch.as_tensor(np.array(self.x[ids], dtype=np.float32), device='cuda')
        x[:, 0] = x[:, 0].sign() * x[:, 0].abs().log1p()
        x[:, 2] = x[:, 2].sign() * x[:, 2].abs().log1p(); x[:, 1] /= 8
        x = torch.cat((x, self.masks[None].expand(len(ids), -1, -1, -1, -1)), 1)
        assert torch.isfinite(x).all()
        args = [x]
        if self.z is not None:
            z = torch.as_tensor(np.array(self.z[ids], dtype=np.float32), device='cuda')
            z = z.sign() * z.abs().log1p()
            assert torch.isfinite(z).all()
            args += [z, torch.as_tensor(np.array(self.s[ids]), device='cuda')]
        return args, torch.as_tensor(self.meta['labels'][ids], dtype=torch.float32, device='cuda')


def benchmark(out):
    assert not (out / 'benchmark.json').exists(), 'benchmark already frozen'
    budget = Budget(out); rows = []
    try:
        data = Data(out, 'train', True)
        for arm in ('CVR', 'DUAL'):
            torch.manual_seed(0); net = model_for(arm).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
            start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            for step in range(3):
                budget.tick(); args, y = data.batch(np.arange(step * 64, (step + 1) * 64))
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(*args[:1] if arm == 'CVR' else args), y)
                loss.backward(); budget.tick(); opt.step(); torch.cuda.synchronize(); budget.tick()
            rows.append(dict(arm=arm, microbatch=64, parameters=sum(p.numel() for p in net.parameters()),
                seconds=time.monotonic()-start, peak_reserved=torch.cuda.max_memory_reserved()))
            del net, opt, args, y, loss; torch.cuda.empty_cache()
        save(out / 'benchmark.json', dict(status='PASS', microbatch=64, rows=rows))
    finally:
        budget.tick(False)


def train_one(out, data, arm, fold, seed, budget):
    p = out / 'models' / arm / fold; final = p / f'model_seed{seed}.pt'
    if final.exists():
        receipt = json.loads((p / f'seed{seed}_receipt.json').read_text())
        assert receipt['sha256'] == hashlib.sha256(final.read_bytes()).hexdigest()
        return
    ids = np.flatnonzero(np.ones(len(data.x), bool) if fold == 'allfamily' else data.meta['family'] != fold)
    assert len(ids) == 96 * (22 if fold == 'allfamily' else 18 if fold == 'general' else 16) * 13
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
    net = model_for(arm).cuda(); opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20)
    resume = p / f'seed{seed}_resume.pt'; history = []; order = None; cursor = 0; total = 0.; count = 0
    if resume.exists():
        ck = torch.load(resume, map_location='cuda', weights_only=False)
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        history, order, cursor, total, count = [ck[k] for k in ('history', 'order', 'cursor', 'total', 'count')]
        rng.bit_generator.state = ck['rng']; torch.set_rng_state(ck['torch_rng'].cpu()); torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
    def persist():
        checkpoint(resume, dict(model=net.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
            history=history, order=order, cursor=cursor, total=total, count=count, rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
    try:
        if not resume.exists(): persist()
        while len(history) < 20:
            start = time.monotonic(); net.train()
            if order is None: order = rng.permutation(ids); cursor = 0; total = 0.; count = 0
            while cursor < len(order):
                budget.tick(); block = order[cursor:cursor+256]; opt.zero_grad(set_to_none=True)
                block_total = 0.
                for i in range(0, len(block), 64):
                    budget.tick(); sub = block[i:i+64]; args, y = data.batch(sub)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(*args), y)
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                    (loss * (len(sub)/len(block))).backward(); block_total += float(loss.detach()) * len(sub)
                budget.tick(); opt.step(); torch.cuda.synchronize()
                total += block_total; count += len(block); cursor += len(block)
                budget.tick()
            sched.step(); history.append(dict(epoch=len(history)+1, loss=total/count, samples=count,
                elapsed_s=time.monotonic()-start, cumulative_training_s=budget.seconds))
            order = None; cursor = 0; total = 0.; count = 0; persist()
            save(p / f'seed{seed}_history.json', history)
            save(out / 'training_progress.json', dict(arm=arm, fold=fold, seed=seed, **history[-1]))
            print(arm, fold, seed, history[-1], flush=True)
        checkpoint(final, {k: v.detach().cpu() for k, v in net.state_dict().items()})
        save(p / f'seed{seed}_receipt.json', dict(arm=arm, fold=fold, seed=seed, epochs=20,
            parameters=sum(v.numel() for v in net.parameters()), samples=len(ids),
            sha256=hashlib.sha256(final.read_bytes()).hexdigest(), final_loss=history[-1]['loss']))
    finally:
        del net, opt, sched; torch.cuda.empty_cache()


def train(out):
    assert (out / 'data/complete.json').exists(), 'materialization incomplete'
    assert json.loads((out / 'benchmark.json').read_text())['status'] == 'PASS'
    budget = Budget(out); status = 'FAILED'
    try:
        budget.tick()
        for arm in ('CVR', 'DUAL'):
            data = Data(out, 'train', arm == 'DUAL')
            for a, fold in JOBS:
                if a != arm: continue
                for seed in range(3): train_one(out, data, arm, fold, seed, budget)
            del data
        status = 'COMPLETE'
    except BudgetStop:
        status = 'STOPPED_BUDGET'
    finally:
        budget.tick(False)
        receipts = [json.loads(p.read_text()) for p in sorted((out/'models').rglob('*_receipt.json'))]
        save(out/'training_terminal.json', dict(status=status, training_seconds=budget.seconds,
            completed_models=len(receipts), expected_models=27, models=receipts))


@torch.no_grad()
def infer(out):
    terminal = json.loads((out / 'training_terminal.json').read_text())
    assert terminal['status'] in ('COMPLETE', 'STOPPED_BUDGET'), 'finish or budget-stop training before inference'
    for arm, fold in JOBS:
        p = out/'models'/arm/fold
        files = [p/f'model_seed{s}.pt' for s in range(3)]
        if not all(f.exists() and (p/f'seed{s}_receipt.json').exists() for s, f in enumerate(files)): continue
        nets = []
        for f in files:
            net = model_for(arm).cuda(); net.load_state_dict(torch.load(f, map_location='cuda', weights_only=True)); nets.append(net.eval())
        for split in ('calib', 'evaluation'):
            dest = out/'predictions'/arm/fold/f'{split}.npz'
            if dest.exists(): continue
            data = Data(out, split, arm == 'DUAL'); m = data.meta
            keep = np.ones(len(data.x), bool) if fold == 'allfamily' else m['family'] != fold if split == 'calib' else m['family'] == fold
            ids = np.flatnonzero(keep); logits = []
            for i in range(0, len(ids), 64):
                args, _ = data.batch(ids[i:i+64]); logits.append(torch.stack([n(*args) for n in nets]).mean(0).cpu().numpy())
            raw = np.concatenate(logits); keys = sorted(set(zip(m['unit'][ids].tolist(), m['config'][ids].tolist())))
            scores = []; finalids = []
            for unit, config in keys:
                loc = np.flatnonzero((m['unit'][ids] == unit) & (m['config'][ids] == config))
                loc = loc[np.argsort(m['frame'][ids[loc]])]
                assert np.array_equal(m['frame'][ids[loc]], np.arange(11, 16))
                scores.append((raw[loc].astype(np.float64)*np.array([1,2,4,8,16])[:,None]).sum(0)/31)
                finalids.append(ids[loc[-1]])
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, scores=np.asarray(scores), **{k:m[k][finalids] for k in ('unit','config','family','labels')})
            print('infer', arm, fold, split, len(finalids), flush=True)
            del data
        del nets, net; torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('benchmark','train','infer'), required=True)
    p.add_argument('--out', type=Path, default=OUT); a = p.parse_args()
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'GPU required'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    {'benchmark':benchmark, 'train':train, 'infer':infer}[a.stage](a.out)


if __name__ == '__main__': main()
