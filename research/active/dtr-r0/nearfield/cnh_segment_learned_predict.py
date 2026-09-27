"""Freeze six v2-selected models, then score both datasets without selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import cnh_learned_readout as L
from cnh_segment_learned_masks import CONDITIONS
from cnh_segment_learned_train import Readout, load_unit, predict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(models, tof_models):
    models, tof_models = Path(models), Path(tof_models)
    files = {}
    for mode in ('IDEAL', 'AUG'):
        for seed in range(3):
            stem = f'{mode}_seed{seed}'
            r = json.loads((models/f'{stem}.json').read_text())
            if r['status'] != 'complete' or r['completed_epochs'] != 20:
                raise ValueError('All six 20epoch jobs must complete before freeze')
            path = models/f'{stem}.pt'
            if sha(path) != r['model_sha256']:
                raise ValueError('Checkpoint hash differs')
            files[str(path)] = sha(path)
    for seed in range(3):
        p = tof_models/f'model_seed{seed}.pt'; files[str(p)] = sha(p)
    code = Path(__file__).parent
    record = dict(scope='Frozen before audit predictions; no v4 model/epoch selection',
                  frozen_unix=time.time(), models=files,
                  code={p.name: sha(p) for p in sorted(code.glob('cnh_segment_learned_*.py'))})
    path = models/'FROZEN.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['models'] != files or old['code'] != record['code']:
            raise ValueError('Existing freeze identity differs')
        return old
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def attach_m1(predictions, m1):
    paths = sorted(Path(predictions).glob('unit*.npz'))
    for p in paths:
        u = int(p.stem[4:])
        if not (Path(m1)/'scores'/f'unit{u:02d}.npz').exists():
            raise ValueError(f'M1 incomplete: unit{u}')
    for p in paths:
        u = int(p.stem[4:])
        with np.load(p) as f:
            d = {k: f[k] for k in f.files}
        with np.load(Path(m1)/'scores'/f'unit{u:02d}.npz') as f:
            for k in ('config', 'frame', 'S2'):
                if not np.array_equal(d[k], f[k]):
                    raise ValueError(f'M1 alignment differs {u}/{k}')
            d.update({f'M1__{c}': f[f'M1__{c}'] for c in CONDITIONS})
        np.savez_compressed(p, **d)


def validate_frozen(models, tof_models):
    models, tof_models = Path(models), Path(tof_models)
    frozen = json.loads((models/'FROZEN.json').read_text())
    expected = [models/f'{mode}_seed{seed}.pt' for mode in ('IDEAL', 'AUG') for seed in range(3)]
    expected += [tof_models/f'model_seed{seed}.pt' for seed in range(3)]
    recorded = {str(Path(path).resolve()): digest for path, digest in frozen['models'].items()}
    if len(recorded) != 9 or set(recorded) != {str(p.resolve()) for p in expected}:
        raise ValueError('Actual nine loaded checkpoints differ from frozen model set')
    for path in expected:
        if sha(path) != recorded[str(path.resolve())]:
            raise ValueError('Frozen model changed')
    for mode in ('IDEAL', 'AUG'):
        for seed in range(3):
            r = json.loads((models/f'{mode}_seed{seed}.json').read_text())
            if r['status'] != 'complete' or r['completed_epochs'] != 20:
                raise ValueError('Incomplete training receipt')
    for name in ('cnh_segment_learned_train.py', 'cnh_segment_learned_masks.py', 'cnh_segment_learned_predict.py'):
        if sha(Path(__file__).with_name(name)) != frozen['code'][name]:
            raise ValueError('Frozen inference implementation changed')
    return frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'masks', 'models', 'tof-models', 'baseline', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--dataset', choices=('v2', 'v4'), required=True)
    p.add_argument('--m1', type=Path)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    validate_frozen(a.models, a.tof_models)
    models = {}
    for mode in ('IDEAL', 'AUG', 'NN'):
        models[mode] = []
        for seed in range(3):
            net = (L.Readout() if mode == 'NN' else Readout()).to(L.DEV)
            checkpoint = a.tof_models/f'model_seed{seed}.pt' if mode == 'NN' else a.models/f'{mode}_seed{seed}.pt'
            net.load_state_dict(torch.load(checkpoint, map_location=L.DEV, weights_only=True))
            net.eval(); models[mode].append(net)
    keys = [u for u in range(96, 192) if u != 143] if a.dataset == 'v2' else list(range(96))
    a.out.mkdir(parents=True, exist_ok=True)
    identity = dict(dataset=a.dataset, freeze_sha256=sha(a.models/'FROZEN.json'),
                    features=str(a.features.resolve()), masks=str(a.masks.resolve()), baseline=str(a.baseline.resolve()),
                    code_sha256=sha(__file__))
    record = a.out/'progress.json'
    if record.exists() and json.loads(record.read_text()).get('identity') != identity:
        raise ValueError('Prediction resume identity mismatch')
    progress = dict(status='running', total=len(keys), complete=0, identity=identity)
    start = time.monotonic()
    try:
        for u in keys:
            target = a.out/f'unit{u:03d}.npz'
            if not target.exists():
                d = load_unit(a.features/f'unit{u:03d}.npz')
                with np.load(a.masks/f'unit{u:03d}.npz') as f:
                    for k in ('config', 'frame'):
                        if not np.array_equal(d[k], f[k]):
                            raise ValueError('Mask order differs')
                    masks = {c: f[c] for c in CONDITIONS}
                bpath = next(p for p in (a.baseline/f'unit{u:02d}.npz', a.baseline/f'unit{u:03d}.npz') if p.exists())
                with np.load(bpath) as f:
                    for k in ('labels', 'config', 'frame'):
                        if not np.array_equal(d[k], f[k]):
                            raise ValueError('S2 metadata differs')
                    result = {k: d[k] for k in ('labels', 'main', 'witness', 'strata', 'config', 'frame', 'split')}
                    result['S2'] = f['G0' if a.dataset == 'v2' else 'S2__noisy@0.75']
                holder = {u: dict(d, y=d['labels'])}
                pred = []
                for net in models['NN']:
                    L.predict(net, holder, [u]); pred.append(holder[u]['NN'])
                result['NN'] = np.mean(pred, axis=0)
                for mode in ('IDEAL', 'AUG'):
                    for c in CONDITIONS:
                        result[f'{mode}__{c}'] = np.mean([predict(net, d['x'], d['sup'], masks[c], batch=512) for net in models[mode]], axis=0)
                np.savez_compressed(target, **result)
            progress.update(complete=progress['complete']+1, last_unit=u, elapsed_s=time.monotonic()-start)
            record.write_text(json.dumps(progress), encoding='utf-8')
            print(json.dumps(progress), flush=True)
        if a.m1:
            attach_m1(a.out, a.m1)
        progress['status'] = 'complete'
    except BaseException as e:
        progress.update(status='failed', error=str(e)); raise
    finally:
        record.write_text(json.dumps(progress, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
