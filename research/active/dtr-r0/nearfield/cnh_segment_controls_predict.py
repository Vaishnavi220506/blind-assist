"""Frozen fusion dependence controls and a separately trained mask-only comparator."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import cnh_learned_readout as L
from cnh_segment_learned_train import Readout, load_unit, predict
from cnh_segment_learned_predict import sha, validate_frozen

CONDITIONS = ('IDEAL', 'FA5', 'FA10', 'BG', 'BG_FA5_SHIFT05', 'BG_FA10_SHIFT05')
META = ('labels', 'main', 'witness', 'strata', 'config', 'frame', 'split')


def validate_mask_frozen(root):
    root = Path(root)
    frozen = json.loads((root/'FROZEN.json').read_text())
    paths = [root/f'MASK_seed{s}.pt' for s in range(3)]
    recorded = {str(Path(k).resolve()): v for k, v in frozen['models'].items()}
    if frozen['status'] != 'frozen' or set(recorded) != {str(p.resolve()) for p in paths}:
        raise ValueError('Incomplete mask freeze')
    for path in paths:
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['status'] != 'complete' or receipt['completed_epochs'] != 20:
            raise ValueError('Mask training incomplete')
        if recorded[str(path.resolve())] != sha(path) or receipt['model_sha256'] != sha(path):
            raise ValueError('Mask checkpoints differ from pre-audit freeze')
    for name,digest in frozen['source_sha256'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Mask training source changed after freeze')
    return frozen


class MaskReadout(L.Readout):
    def __init__(self):
        super().__init__()
        self.body[0] = torch.nn.Conv3d(1, 16, 3, padding=1)


@torch.inference_mode()
def mask_predict(net, sup, mask, batch=512):
    out = []
    for i in range(0, len(mask), batch):
        m = torch.as_tensor(np.asarray(mask[i:i+batch], dtype=np.float32), device=L.DEV)
        x = m[:, None, :, :, None].expand(-1, 1, -1, -1, 16)
        s = torch.as_tensor(sup[i:i+batch], device=L.DEV)
        out.append(net(x, s).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('features', 'masks', 'models', 'tof-models', 'mask-models', 'previous', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--dataset', choices=('v2', 'v4'), required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    validate_frozen(a.models, a.tof_models)
    frozen_path = a.mask_models/'FROZEN.json'
    validate_mask_frozen(a.mask_models)
    nets = {}
    for mode in ('IDEAL', 'AUG', 'MASK'):
        nets[mode] = []
        for seed in range(3):
            net = (MaskReadout() if mode == 'MASK' else Readout()).to(L.DEV)
            path = (a.mask_models if mode == 'MASK' else a.models)/f'{mode}_seed{seed}.pt'
            net.load_state_dict(torch.load(path, map_location=L.DEV, weights_only=True))
            nets[mode].append(net.eval())
    identity = dict(dataset=a.dataset, fusion_freeze=sha(a.models/'FROZEN.json'),
                    mask_freeze=sha(frozen_path), code=sha(__file__),
                    inputs={k: str(getattr(a,k).resolve()) for k in ('features','masks','previous')})
    a.out.mkdir(parents=True, exist_ok=True)
    progress_path = a.out/'progress.json'
    if progress_path.exists() and json.loads(progress_path.read_text())['identity'] != identity:
        raise ValueError('Resume identity changed')
    keys = [u for u in range(96,192) if u != 143] if a.dataset == 'v2' else list(range(96))
    progress = dict(identity=identity, status='running', complete=0, total=len(keys))
    start = time.monotonic()
    try:
        for u in keys:
            target = a.out/f'unit{u:03d}.npz'
            if not target.exists():
                d = load_unit(a.features/f'unit{u:03d}.npz')
                with np.load(a.previous/f'unit{u:03d}.npz') as f:
                    for k in META:
                        equal = np.array_equal(d[k], f[k], equal_nan=True) if d[k].dtype.kind not in 'US' else np.array_equal(d[k], f[k])
                        if not equal:
                            raise ValueError(f'Previous metadata changed {u}/{k}')
                    result = {k:d[k] for k in META}
                    result.update(S2=f['S2'], NN=f['NN'])
                    ideal_reference = {mode:f[f'{mode}__IDEAL'] for mode in ('IDEAL','AUG')}
                with np.load(a.masks/f'unit{u:03d}.npz') as f:
                    for k in ('config','frame'):
                        if not np.array_equal(d[k],f[k]):
                            raise ValueError('Mask metadata mismatch')
                    masks = {c:f[c] for c in CONDITIONS}
                zero = np.zeros_like(d['x'])
                for c,m in masks.items():
                    for mode in ('IDEAL','AUG'):
                        for name,x in ((mode,d['x']),('ZERO_'+mode,zero)):
                            result[f'{name}__{c}'] = np.mean([predict(net,x,d['sup'],m,batch=512) for net in nets[mode]],axis=0)
                    result[f'MASK__{c}'] = np.mean([mask_predict(net,d['sup'],m) for net in nets['MASK']],axis=0)
                for mode,reference in ideal_reference.items():
                    if not np.allclose(result[f'{mode}__IDEAL'],reference,rtol=0,atol=2e-5):
                        raise ValueError('Frozen IDEAL reference failed replay check')
                np.savez_compressed(target,**result)
            progress.update(complete=progress['complete']+1,last_unit=u,elapsed_s=time.monotonic()-start)
            progress_path.write_text(json.dumps(progress),encoding='utf-8')
            if progress['complete'] % 10 == 0:
                print(json.dumps(progress),flush=True)
        progress['status']='complete'
        validate_frozen(a.models,a.tof_models)
        validate_mask_frozen(a.mask_models)
    except BaseException as e:
        progress.update(status='failed',error=str(e)); raise
    finally:
        progress_path.write_text(json.dumps(progress,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
