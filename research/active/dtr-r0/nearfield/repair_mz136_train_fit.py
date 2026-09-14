"""Disclosed MZ136 TRAIN-only fit diagnosis. Never select or score heldout rows."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools'))
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from mz120_occupancy import encode
from mz136_corridor_pair import CorridorNet
from mz136_position_readout import PositionCorridorNet
from run_mz136_corridor_pair import pair_indices, digest_state, fit_scores, predict
from run_mz120_occupancy import batch
from run_mz107_four_sensor import readrows, sha, truth, write


def load_train(capture):
    receipt = json.loads((capture/'receipt.json').read_text())
    assert receipt['status'] == 'PASS'
    assert sha(capture/'spec.json') == receipt['spec_sha256']
    for name in ('raw.jsonl', 'evaluator.jsonl'):
        assert sha(capture/name) == receipt['hashes'][name]
    spec = json.loads((capture/'spec.json').read_text())
    all_rows = readrows(capture/'raw.jsonl')
    all_pairs = pair_indices(all_rows, spec)
    selected = [p for p in all_pairs if p['split'] == 'train']
    original_indices = sorted(i for p in selected for i in (p['a'], p['b']))
    remap = {old: new for new, old in enumerate(original_indices)}
    pairs = [dict(p, a=remap[p['a']], b=remap[p['b']]) for p in selected]
    rows = [all_rows[i] for i in original_indices]
    train_ids = {r['id'] for r in rows}
    # Evaluation objects are used only for training supervision and diagnostics.
    es_by_id = {e['id']: e for e in readrows(capture/'evaluator.jsonl') if e['id'] in train_ids}
    es = [es_by_id[r['id']] for r in rows]
    inputs, images, yaw, episode = [], [], 0., None
    for r in rows:
        if r['episode_id'] != episode:
            yaw = 0.
        if r['imu_valid']:
            yaw += r['delta_yaw']
        episode = r['episode_id']
        inputs.append(encode(r, yaw))
        path = (capture/r['rgb_path']).resolve()
        assert path.is_relative_to(capture.resolve())
        assert sha(path) == receipt['hashes'][r['rgb_path']]
        im = cv2.imread(str(path))
        assert im is not None and im.shape == (360, 640, 3)
        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    data = {k: torch.from_numpy(np.stack([v[k] for v in inputs])) for k in inputs[0]}
    data['rgb'] = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2))
    return rows, es, pairs, data, original_indices


def describe(gt, scores, pairs):
    result = fit_scores(gt, scores, pairs)
    result['both_rate'] = float(result['both_rate'])
    # The inherited alert helper names all negative flags UNKNOWN. This is an
    # explicit binary training task, so report its confusion matrix separately.
    result['metrics'].pop('UNKNOWN', None)
    result['bce'] = float(np.mean(np.logaddexp(0, scores)-gt*scores))
    result['families'] = {}
    for family in sorted({p['family'] for p in pairs}):
        idx = sorted(i for p in pairs if p['family'] == family for i in (p['a'], p['b']))
        yy, pp = gt[idx], scores[idx] >= 0
        result['families'][family] = dict(TP=int((yy & pp).sum()), FP=int((~yy & pp).sum()),
                                        FN=int((yy & ~pp).sum()), TN=int((~yy & ~pp).sum()))
    result['fit_pass'] = bool(result['accuracy'] >= .95 and result['both_rate'] >= .9)
    return result


def run(args):
    out = args.output.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(136014)
    assert torch.cuda.is_available()
    rows, es, pairs, data, original_indices = load_train(args.capture.resolve())
    gt = np.array([truth(e) for e in es], bool)
    assert len(rows) == 192 and len(pairs) == 96
    model = (PositionCorridorNet() if args.position else CorridorNet()).cuda()
    state = torch.load(args.checkpoint, map_location='cuda', weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=not args.position)
    assert not unexpected and missing == (['horizontal.weight'] if args.position else [])
    backend_path = ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/full-fits-v1/backend.json'
    assert backend_path.exists()
    snapshots = out/'source-snapshot'
    snapshots.mkdir()
    files = [Path(__file__), Path(__file__).with_name('mz136_corridor_pair.py'),
             Path(__file__).with_name('mz120_occupancy.py')]
    if args.position:
        files.append(Path(__file__).with_name('mz136_position_readout.py'))
    for f in files:
        shutil.copyfile(f, snapshots/f.name)
    freeze = dict(scope='CONSUMED_TRAIN_ONLY_FIT_DIAGNOSIS', frames=len(rows), pairs=len(pairs),
                  original_indices=original_indices, frame_ids=[r['id'] for r in rows],
                  checkpoint_sha256=sha(args.checkpoint), initial_state_digest=digest_state(model.state_dict()),
                  source={n: sha(args.capture/n) for n in ('raw.jsonl', 'evaluator.jsonl', 'spec.json', 'receipt.json')},
                  code={f.name: sha(f) for f in files}, seed=136014, steps=args.steps,
                  augmentation=False, optimizer='AdamW', learning_rate=args.lr,
                  position_residual=args.position, parameters=sum(p.numel() for p in model.parameters()),
                  schedule='cosine to 0.1 x initial LR', weight_decay=.0001,
                  criterion='fixed zero logit; accuracy >= .95 and both_correct_rate >= .90',
                  checkpoint_selection='minimum full TRAIN BCE; no heldout inference',
                  backend=dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                               placement='GPU measured for original; position residual measured separately when enabled',
                               prior_backend_path=str(backend_path), prior_backend_sha256=sha(backend_path)))
    write(out/'freeze.json', freeze)
    if args.position:
        from research_backend import BackendCandidate, DeviceObservation, select_backend
        import copy
        sample = batch(data, np.arange(8))
        cpu_model = copy.deepcopy(model).cpu().eval()
        cpu_sample = {k: v.cpu() for k, v in sample.items()}
        def probe(m, b):
            with torch.no_grad():
                return m(b)
        backend = select_backend('model-inference',
            cpu=BackendCandidate('torch-cpu', 'cpu', lambda: probe(cpu_model, cpu_sample),
                lambda v: DeviceObservation(v.device.type, 'host CPU', f'torch {torch.__version__}')),
            gpu=BackendCandidate('torch-cuda', 'cuda', lambda: probe(model, sample),
                lambda v: DeviceObservation(v.device.type, torch.cuda.get_device_name(), f'torch {torch.__version__}'),
                torch.cuda.synchronize), record_path=out/'backend.json')
        assert backend['selected_device_type'] == 'cuda'
        del cpu_model, cpu_sample, sample
    scores = predict(model, data)
    before = describe(gt, scores, pairs)
    old_scores_path = args.checkpoint.with_name(args.checkpoint.name.replace('-model.pt', '-scores.npy'))
    if old_scores_path.exists():
        assert np.allclose(scores, np.load(old_scores_path)[original_indices], atol=1e-5, rtol=1e-5)
    np.save(out/'before-scores.npy', scores)
    write(out/'before.json', before)
    target = torch.tensor(gt, dtype=torch.float32, device='cuda')
    ix = np.array([[p['a'], p['b']] for p in pairs])
    rng = np.random.default_rng(136014)
    schedule = np.concatenate([ix[rng.permutation(len(ix))].reshape(-1, 8)
                               for _ in range((args.steps+23)//24)])[:args.steps]
    np.save(out/'schedule.npy', schedule)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*.1)
    progress = [dict(step=0, **before)]
    best_loss = before['bce']
    torch.save(model.state_dict(), out/'best-model.pt')
    best_scores = scores.copy()
    tick = time.perf_counter()
    for step, idx in enumerate(schedule, 1):
        assert time.perf_counter()-tick < 600, '10 minute TRAIN-only diagnosis cap'
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(batch(data, idx)), target[idx])
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 240 == 0 or step == args.steps:
            scores = predict(model, data)
            report = describe(gt, scores, pairs)
            progress.append(dict(step=step, learning_rate=scheduler.get_last_lr()[0], **report))
            write(out/'progress.json', progress)
            print(json.dumps(dict(step=step, loss=report['bce'], accuracy=report['accuracy'],
                                  both=report['both_correct'], families=report['families'])), flush=True)
            if report['bce'] < best_loss:
                best_loss = report['bce']
                best_scores = scores.copy()
                torch.save(model.state_dict(), out/'best-model.pt')
    torch.save(model.state_dict(), out/'last-model.pt')
    np.save(out/'after-scores.npy', best_scores)
    after = describe(gt, best_scores, pairs)
    cases = [dict(id=r['id'], episode=r['episode_id'], family=e['family'], target=bool(y),
                  before_logit=float(a), after_logit=float(b), before_correct=bool((a>=0)==y),
                  after_correct=bool((b>=0)==y), rgb_path=r['rgb_path'])
             for r, e, y, a, b in zip(rows, es, gt, np.load(out/'before-scores.npy'), best_scores)]
    write(out/'cases.json', cases)
    summary = dict(before=before, after=after, seconds=time.perf_counter()-tick,
                   best_model_sha256=sha(out/'best-model.pt'), heldout_inference=False,
                   decision='TRAIN_FIT_REPAIRED' if after['fit_pass'] else 'TRAIN_FIT_STILL_FAILS')
    write(out/'summary.json', summary)
    write(out/'completion.json', dict(status='PASS', hashes={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--position', action='store_true')
    run(parser.parse_args())
