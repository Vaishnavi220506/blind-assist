"""Three-channel variant of the frozen learned-readout recipe; v2 selection only.

20 epochs, seeds0/1/2, batch256, BCE, AdamW2e-3/1e-4, cosine20 unchanged.
IDEAL selection: v2calib ideal macroAP. AUG selection: mean over IDEAL/MIXED
v2calib macroAP (fixed before training). No audit or v4 files loaded here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
import cnh_learned_readout as L


class Readout(L.Readout):
    def __init__(self):
        super().__init__()
        self.body[0] = nn.Conv3d(3, 16, 3, padding=1)


def load_unit(path):
    with np.load(path) as d:
        return dict(x=np.stack([L.squash(d['z4'].astype(np.float32)), L.squash(d['z1'].astype(np.float32))], 1),
                    sup=np.unpackbits(d['sup'], axis=-1)[..., :16].astype(bool),
                    **{k: d[k].copy() for k in ('labels', 'main', 'witness', 'strata', 'config', 'frame', 'split')})


def tensor_input(x, mask):
    x = torch.as_tensor(x, device=L.DEV)
    m = torch.as_tensor(mask, dtype=x.dtype, device=L.DEV)[:, None, :, :, None].expand(-1, 1, 8, 8, 16)
    return torch.cat((x, m), 1)


@torch.no_grad()
def predict(net, x, sup, mask, batch=256):
    net.eval()
    parts = []
    for i in range(0, len(x), batch):
        z = net(tensor_input(x[i:i+batch], mask[i:i+batch]), torch.as_tensor(sup[i:i+batch], device=L.DEV))
        parts.append(z.cpu().numpy())
    return np.concatenate(parts)


def load_training(features, masks):
    rows = []
    for u in range(96):
        d = load_unit(features/f'unit{u:03d}.npz')
        if str(d['split']) != 'train':
            raise ValueError('Training split mismatch')
        ix = d['main'].astype(bool)
        with np.load(masks/f'unit{u:03d}.npz') as m:
            if not np.array_equal(m['config'], d['config']) or not np.array_equal(m['frame'], d['frame']):
                raise ValueError('Mask/feature misalignment')
            rows.append(dict(x=d['x'][ix], sup=d['sup'][ix], y=d['labels'][ix].astype(np.float32),
                             IDEAL=m['IDEAL'][ix], AUG=m['AUG'][:, ix]))
    result = {k: np.concatenate([r[k] for r in rows], axis=1 if k == 'AUG' else 0) for k in rows[0]}
    return result


def load_calib(features, masks):
    rows = []
    for u in range(96, 128):
        d = load_unit(features/f'unit{u:03d}.npz')
        if str(d['split']) != 'calib':
            raise ValueError('Calib split mismatch')
        with np.load(masks/f'unit{u:03d}.npz') as m:
            if not np.array_equal(m['config'], d['config']) or not np.array_equal(m['frame'], d['frame']):
                raise ValueError('Mask/feature misalignment')
            d.update({k: m[k].copy() for k in ('IDEAL', 'MIXED')})
        rows.append(d)
    return rows


def calib_ap(net, rows, conditions):
    totals = []
    detail = {}
    for condition in conditions:
        aps = {g: [] for g, _ in L.GROUPS}
        for d in rows:
            # Model selection AP uses only main frames, identical to original.
            m = d['main'].astype(bool)
            pred = predict(net, d['x'][m], d['sup'][m], d[condition][m])
            for g, ix in L.GROUPS:
                y = d['labels'][m][:, ix].ravel()
                if 0 < y.sum() < len(y):
                    aps[g].append(average_precision_score(y, pred[:, ix].ravel()))
        detail[condition] = {g: float(np.mean(v)) for g, v in aps.items()}
        totals.extend(detail[condition].values())
    return float(np.mean(totals)), detail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'train-masks', 'calib-masks', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mode', choices=('IDEAL', 'AUG'), required=True)
    p.add_argument('--seed', choices=(0, 1, 2), type=int, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU training')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f'{a.mode}_seed{a.seed}'
    receipt = a.out/f'{stem}.json'
    if receipt.exists() and json.loads(receipt.read_text())['status'] == 'complete':
        print('Already complete', stem); return
    start = time.monotonic()
    train = load_training(a.features, a.train_masks)
    calib = load_calib(a.features, a.calib_masks)
    net = Readout().to(L.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20)
    history, best, best_epoch, best_state, first = [], -float('inf'), -1, None, 0
    resume = a.out/f'{stem}-resume.pt'
    identity = dict(mode=a.mode, seed=a.seed, epochs=20, features=str(a.features),
                    train_masks=str(a.train_masks), calib_masks=str(a.calib_masks),
                    code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume identity changed')
        net.load_state_dict(state['net']); opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched'])
        history, best, best_epoch, best_state, first = [state[k] for k in ('history', 'best', 'best_epoch', 'best_state', 'next_epoch')]
        rng.bit_generator.state = state['rng']
    progress = dict(status='running', **identity, device=torch.cuda.get_device_name(), params=sum(p.numel() for p in net.parameters()))
    try:
        for ep in range(first, 20):
            net.train(); loss_sum = 0.
            mask = train['IDEAL'] if a.mode == 'IDEAL' else train['AUG'][ep]
            for begin in range(0, len(train['x']), 256):
                # Generate one fixed permutation per epoch, identical batching convention.
                if begin == 0:
                    order = rng.permutation(len(train['x']))
                ix = order[begin:begin+256]
                z = net(tensor_input(train['x'][ix], mask[ix]), torch.as_tensor(train['sup'][ix], device=L.DEV))
                loss = nn.functional.binary_cross_entropy_with_logits(z, torch.as_tensor(train['y'][ix], device=L.DEV))
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += float(loss.detach())*len(ix)
            sched.step()
            score, ap = calib_ap(net, calib, ('IDEAL',) if a.mode == 'IDEAL' else ('IDEAL', 'MIXED'))
            if score > best:
                best, best_epoch = score, ep
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            row = dict(epoch=ep, loss_sum=loss_sum, calib=ap, score=score, elapsed_s=time.monotonic()-start)
            history.append(row); print(json.dumps(row), flush=True)
            torch.save(dict(identity=identity, net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            history=history, best=best, best_epoch=best_epoch, best_state=best_state, next_epoch=ep+1,
                            rng=rng.bit_generator.state), resume)
            progress.update(completed_epochs=ep+1, best_epoch=best_epoch, elapsed_s=time.monotonic()-start)
            receipt.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        torch.save(best_state, a.out/f'{stem}.pt')
        progress.update(status='complete', history=history, best_calib=best,
                        model_sha256=hashlib.sha256((a.out/f'{stem}.pt').read_bytes()).hexdigest(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        progress.update(status='failed', error=str(e)); raise
    finally:
        receipt.write_text(json.dumps(progress, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
