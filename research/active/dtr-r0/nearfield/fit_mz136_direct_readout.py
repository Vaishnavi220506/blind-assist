"""One convex TRAIN-only readout fit; original backbone and outcomes stay frozen."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz136_direct_readout import DirectReadoutNet
from repair_mz136_train_fit import ROOT, load_train, describe
from run_mz136_corridor_pair import predict
from run_mz120_occupancy import batch
from run_mz107_four_sensor import truth, sha, write
from research_backend import BackendCandidate, DeviceObservation, select_backend


def run(args):
    out = args.output.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(136014)
    assert torch.cuda.is_available()
    rows, es, pairs, data, original = load_train(args.capture.resolve())
    assert len(rows) == 192 and len(pairs) == 96
    gt = np.array([truth(e) for e in es], bool)
    model = DirectReadoutNet(args.ordered).cuda().eval()
    missing, extra = model.load_state_dict(torch.load(args.checkpoint, map_location='cuda', weights_only=True), strict=False)
    assert not extra and set(missing) == {'feature_mean', 'feature_scale', 'readout.weight', 'readout.bias'}
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('readout.'))
    snapshots = out/'source-snapshot'
    snapshots.mkdir()
    files = [Path(__file__), Path(__file__).with_name('mz136_direct_readout.py'),
             Path(__file__).with_name('repair_mz136_train_fit.py'), Path(__file__).with_name('mz120_occupancy.py')]
    for f in files:
        shutil.copyfile(f, snapshots/f.name)
    freeze = dict(authority='CONSUMED_TRAIN_ONLY_FIT_REPAIR_NO_HELDOUT_INFERENCE',
        frames=192, pairs=96, frame_ids=[r['id'] for r in rows], original_indices=original,
        checkpoint_sha256=sha(args.checkpoint), ordered=args.ordered,
        source={n:sha(args.capture/n) for n in ('raw.jsonl', 'evaluator.jsonl', 'spec.json', 'receipt.json')},
        code={f.name:sha(f) for f in files},
        feature_transform='per-dimension TRAIN mean/std; std floor 0.001',
        optimizer='LBFGS max_iter=200 strong_wolfe full192', regularizer='0.0001 * sum(readout.weight**2)',
        criterion='fixed zero logit; accuracy >= .95 and both_correct_rate >= .90',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        selection='single final optimization result; no threshold search or checkpoint selection')
    write(out/'freeze.json', freeze)
    base, visual = [], []
    tick = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            b, v = model.components(batch(data, np.arange(start, start+8)))
            base.append(b)
            visual.append(v)
    base, visual = torch.cat(base), torch.cat(visual)
    extraction_seconds = time.perf_counter()-tick
    before = base.cpu().numpy()
    np.testing.assert_allclose(before, np.load(args.checkpoint.with_name('bce-scores.npy'))[original], atol=1e-5, rtol=1e-5)
    model.feature_mean.copy_(visual.mean(0))
    model.feature_scale.copy_(visual.std(0, unbiased=False).clamp_min(.001))
    x = (visual-model.feature_mean)/model.feature_scale
    y = torch.tensor(gt, dtype=torch.float32, device='cuda')
    optimizer = torch.optim.LBFGS(model.readout.parameters(), lr=1., max_iter=200,
                                  tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn='strong_wolfe')
    history = []
    tick = time.perf_counter()
    def closure():
        assert time.perf_counter()-tick < 120, 'two minute convex readout fit cap'
        optimizer.zero_grad(set_to_none=True)
        logits = base+model.readout(x).squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(logits, y)
        loss = bce+.0001*model.readout.weight.square().sum()
        loss.backward()
        history.append(dict(evaluation=len(history), bce=float(bce.detach()), objective=float(loss.detach())))
        return loss
    optimizer.step(closure)
    fit_seconds = time.perf_counter()-tick
    torch.save(model.state_dict(), out/'model.pt')
    write(out/'fit-runtime.json', dict(feature_extraction_seconds=extraction_seconds, fit_seconds=fit_seconds))
    write(out/'progress.json', history)
    # Replay via normal observation-only inference, not just the cached matrix.
    after = predict(model, data)
    with torch.no_grad():
        cached = (base+model.readout(x).squeeze(-1)).cpu().numpy()
    np.testing.assert_allclose(after, cached, atol=1e-4, rtol=1e-4)
    np.save(out/'before-scores.npy', before)
    np.save(out/'after-scores.npy', after)
    cases = [dict(id=r['id'], family=e['family'], target=bool(y), before_logit=float(a),
                  after_logit=float(b), before_correct=bool((a>=0)==y), after_correct=bool((b>=0)==y),
                  rgb_path=r['rgb_path']) for r,e,y,a,b in zip(rows, es, gt, before, after)]
    write(out/'cases.json', cases)
    summary = dict(before=describe(gt, before, pairs), after=describe(gt, after, pairs),
                   feature_extraction_seconds=extraction_seconds, fit_seconds=fit_seconds,
                   closure_evaluations=len(history), model_sha256=sha(out/'model.pt'),
                   observation_replay_agrees=True, heldout_inference=False)
    write(out/'summary.json', summary)
    probe = batch(data, np.arange(8))
    cpu_model = copy.deepcopy(model).cpu()
    cpu_probe = {k:v.cpu() for k,v in probe.items()}
    def forward(m, b):
        with torch.no_grad():
            return m(b)
    select_backend('model-inference',
        cpu=BackendCandidate('torch-cpu', 'cpu', lambda:forward(cpu_model, cpu_probe),
            lambda v:DeviceObservation(v.device.type, 'host CPU', 'torch '+torch.__version__)),
        gpu=BackendCandidate('torch-cuda', 'cuda', lambda:forward(model, probe),
            lambda v:DeviceObservation(v.device.type, torch.cuda.get_device_name(), 'torch '+torch.__version__), torch.cuda.synchronize),
        record_path=out/'backend.json')
    write(out/'completion.json', dict(status='PASS', hashes={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ordered', action='store_true')
    run(parser.parse_args())
