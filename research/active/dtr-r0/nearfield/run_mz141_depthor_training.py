"""Bounded paired MZ141 training; all TRAIN predictions seal before holdout truth."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
import shutil
import sys
import time
import cv2
import numpy as np
import torch
from mz140_depthor import make_input, predict, load_model
from mz141_depthor_training import load_trainable, training_prediction, supervision, loss_contribution
from mz136_incumbent import public_observations
from run_mz139_surface_fit import read, selected_jsonl, local_dependencies, WORK, CODE
from run_mz107_four_sensor import ROOT, sha, write

BASE = ROOT/'artifacts.local/work/mz140-depthor-20260915'
CAP = WORK/'source/returned-v1/capture-v1'
SEED, EPOCHS, ACCUM = 141, 8, 4


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def release(model=None):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def module_digest(model, prefix='depth_anything.'):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(prefix):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepared(out):
    spec, receipt = read(CAP/'spec.json'), read(CAP/'receipt.json')
    frames = [f for f in spec['frames'] if f['split'] == 'train']
    assert len(frames) == 192
    ids = {f['id'] for f in frames}
    metadata = {f['id']: dict(family=f['family'], scene_group=f['scene_group'],
        partition='heldout' if f['scene_group'].endswith('_scene3') else 'fit') for f in frames}
    assert Counter(v['partition'] for v in metadata.values()) == {'fit': 144, 'heldout': 48}
    fit_ids = {key for key, value in metadata.items() if value['partition'] == 'fit'}
    rows = public_observations(selected_jsonl(CAP/'raw.jsonl', ids))
    prep = WORK/'incumbent/fresh-v1'
    cache = read(prep/'nominal/predictions.json')
    seal, completed = read(prep/'prediction-seal.json'), read(prep/'completion.json')
    assert completed['status'] == 'PASS' and sha(prep/'prediction-seal.json') == completed['prediction_seal_sha256']
    assert sha(prep/'nominal/predictions.json') == seal['predictions_sha256']['nominal']
    assert rows == selected_jsonl(prep/'nominal/raw.jsonl', ids)
    assert sha(CAP/'spec.json') == receipt['spec_sha256']
    for name in ('raw.jsonl', 'evaluator.jsonl'):
        assert sha(CAP/name) == receipt['hashes'][name]
    mapping = {p['id']: i for i, p in enumerate(cache['predictions'])}
    cached = {r['id']: cache['corrected'][mapping[r['id']]] for r in rows}
    inputs = {str(p): sha(p) for p in [CAP/n for n in ('spec.json', 'receipt.json', 'raw.jsonl', 'evaluator.jsonl')] +
              [prep/'nominal/predictions.json', prep/'prediction-seal.json', BASE/'depthor-zju-small.pt']}
    images = {}
    for row in rows:
        path = CAP/row['rgb_path']
        assert path.resolve().is_relative_to(CAP.resolve())
        assert sha(path) == receipt['hashes'][row['rgb_path']]
        images[row['id']] = cv2.imread(str(path))
        assert images[row['id']].shape == (360, 640, 3)
        inputs[str(path)] = sha(path)
    sources = local_dependencies(__file__)
    for name in ('evaluate_mz141_depthor.py', 'test_mz141_depthor_training.py', 'evaluate_mz140_depthor.py'):
        path = CODE/name
        sources[str(path.resolve())] = sha(path)
    for path in (BASE/'upstream').rglob('*.py'):
        sources[str(path.resolve())] = sha(path)
    for src in sources:
        path = Path(src)
        relative = path.relative_to(ROOT.resolve()) if path.is_relative_to(ROOT.resolve()) else Path('artifact-dependencies')/path.relative_to(BASE.resolve())
        dest = out/'source-snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    shutil.copyfile(CODE/'MZ141_PROTOCOL_20260915.md', out/'protocol-before-fit.md')
    write(out/'freeze.json', dict(seed=SEED, epochs=EPOCHS, effective_batch=ACCUM,
        optimizer=dict(type='Adam', lr=1e-4, betas=[.9, .999], eps=1e-8, weight_decay=0),
        metadata=metadata, sources=sources, inputs=inputs, protocol_sha256=sha(out/'protocol-before-fit.md'),
        scope='CONSUMED_TRAIN_SCENE_HELDOUT_DEVELOPMENT', original_test_access=False,
        training_labels='FIT_NATIVE_AABB_ONLY', inference_inputs='UNCHANGED_PUBLIC_MZ140_INPUTS'))
    from evaluate_mz140_depthor import rasterize_native_bounds
    fit_evaluation = {e['id']: e for e in selected_jsonl(CAP/'evaluator.jsonl', fit_ids)}
    labels = {}
    for row in rows:
        if row['id'] in fit_ids:
            labels[row['id']] = supervision(rasterize_native_bounds(row, fit_evaluation[row['id']], cached[row['id']]))
    write(out/'label-audit.json', {key: value['audit'] for key, value in labels.items()})
    return rows, metadata, images, labels, cached


def gradient_smoke(out, rows, images, labels):
    seed()
    model, info = load_trainable(BASE/'upstream', BASE/'depthor-zju-small.pt')
    old_backbone = module_digest(model)
    row = next(r for r in rows if r['id'] in labels and 'near_rod_farwall' in r['id'])
    data, _ = make_input(row, images[row['id']])
    # eval forward equivalence with the pre-existing port before train-mode backward.
    model.eval()
    a = predict(model, data)
    import src.models.refine as refine
    refine.bpconvlocal = refine.BpConvLocal.apply
    b = predict(model, data)
    difference = float((a-b).abs().max())
    assert difference <= 1e-6
    from mz140_depthor import local_conv
    refine.bpconvlocal = local_conv
    model.train()
    model.depth_anything.eval()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = training_prediction(model, data)
        loss = loss_contribution(prediction, labels[row['id']], 'surface', labels[row['id']]['audit']['known_pixels'], 1)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    modules = {}
    for name in ('img_encoder', 'SpEncoder', 'decoder', 'depth_head', 'conv_out', 'refine', 'up', 'align_mde'):
        params = list(getattr(model, name).parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        grad = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
        assert grad > 0, name
        modules[name] = dict(gradient_l1=grad, tensors_with_gradient=sum(p.grad is not None for p in params))
    assert all(p.grad is None for p in model.depth_anything.parameters())
    assert old_backbone == module_digest(model)
    write(out/'gradient-smoke.json', dict(status='PASS', no_optimizer_updates=True,
        info=info, forward_max_difference_m=difference, module_gradients=modules,
        frozen_backbone_unchanged=True, forward_backward_seconds=times,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), actual_device=torch.cuda.get_device_name(), torch=torch.__version__))
    del model, prediction, loss, data, a, b
    release()


def predict_arm(out, arm, model, rows, images):
    model.eval()
    records, timings, audits = {}, [], []
    for i, row in enumerate(rows):
        data, audit = make_input(row, images[row['id']])
        torch.cuda.synchronize()
        start = time.perf_counter()
        depth = predict(model, data).cpu().numpy()
        torch.cuda.synchronize()
        assert np.isfinite(depth).all()
        timings.append(time.perf_counter()-start)
        path = out/f'predictions/{arm}/{i:03d}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, depth, allow_pickle=False)
        records[row['id']] = dict(path=str(path), sha256=sha(path))
        if arm == 'frozen':
            audits.append(dict(id=row['id'], **audit))
        if (i+1) % 48 == 0:
            print(json.dumps(dict(stage='predict', arm=arm, completed=i+1, total=len(rows))), flush=True)
    write(out/f'{arm}-predictions.json', records)
    write(out/f'{arm}-timings.json', timings)
    if audits:
        write(out/'input-audits.json', audits)
    return records


def train_arm(out, arm, rows, images, labels):
    seed()
    model, info = load_trainable(BASE/'upstream', BASE/'depthor-zju-small.pt')
    backbone = module_digest(model)
    initial = module_digest(model, '')
    optimizer = torch.optim.Adam(model.get_lr_params(), lr=1e-4)
    fit_rows = [r for r in rows if r['id'] in labels]
    schedule = np.random.default_rng(SEED)
    log = []
    start = time.perf_counter()
    step = 0
    for epoch in range(EPOCHS):
        order = schedule.permutation(len(fit_rows)).tolist()
        epoch_losses = []
        for offset in range(0, len(order), ACCUM):
            batch = [fit_rows[j] for j in order[offset:offset+ACCUM]]
            denominator = sum(labels[r['id']]['audit']['known_pixels'] for r in batch)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for row in batch:
                data, _ = make_input(row, images[row['id']])
                prediction = training_prediction(model, data)
                loss = loss_contribution(prediction, labels[row['id']], arm, denominator, len(batch))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                batch_loss += float(loss.detach())
            for parameter in model.get_lr_params():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            step += 1
            epoch_losses.append(batch_loss)
            if step % 12 == 0:
                progress = dict(stage='train', arm=arm, epoch=epoch+1, step=step, total_steps=288,
                    loss=batch_loss, seconds=time.perf_counter()-start)
                write(out/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        record = dict(epoch=epoch+1, steps=step, loss_mean=float(np.mean(epoch_losses)),
            fit_order=[fit_rows[j]['id'] for j in order], seconds=time.perf_counter()-start)
        log.append(record)
        write(out/f'{arm}-training.json', log)
        dest = out/f'checkpoints/{arm}-epoch{epoch+1}.pt'
        dest.parent.mkdir(exist_ok=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
            step=step, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_schedule_rng=schedule.bit_generator.state, freeze_sha256=sha(out/'freeze.json')), dest)
    assert step == 288 and backbone == module_digest(model)
    final = module_digest(model, '')
    assert final != initial
    write(out/f'{arm}-fit-receipt.json', dict(status='PASS', initial_state_sha256=initial,
        final_state_sha256=final, backbone_sha256=backbone, backbone_unchanged=True,
        optimizer_steps=step, info=info, final_checkpoint_sha256=sha(dest), elapsed_seconds=time.perf_counter()-start))
    predictions = predict_arm(out, arm, model, rows, images)
    del optimizer, model, prediction, loss, data
    release()
    return predictions


def evaluate(out, rows, metadata, cached, predictions):
    from evaluate_mz140_depthor import evaluate_frame, rasterize_native_bounds
    from evaluate_mz141_depthor import extra_metrics, summarize
    native = {e['id']: e for e in selected_jsonl(CAP/'evaluator.jsonl', {r['id'] for r in rows})}
    cases = []
    for i, row in enumerate(rows):
        reference = rasterize_native_bounds(row, native[row['id']], cached[row['id']])
        arms = {}
        for arm, records in predictions.items():
            record = records[row['id']]
            assert sha(Path(record['path'])) == record['sha256']
            depth = np.load(record['path'])
            arms[arm] = evaluate_frame(row, native[row['id']], depth, cached[row['id']])
            arms[arm].update(extra_metrics(depth, reference))
        cases.append(dict(id=row['id'], **metadata[row['id']], arms=arms))
        if (i+1) % 48 == 0:
            print(json.dumps(dict(stage='evaluate', completed=i+1, total=192)), flush=True)
    write(out/'geometry-cases.json', cases)
    summary = summarize(cases)
    write(out/'summary.json', summary)
    return summary


def run(out):
    out = out.resolve()
    assert not out.exists() and out.is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    sys.path.insert(0, str(ROOT/'tools'))
    from research_backend import runtime_capabilities
    write(out/'runtime.json', dict(capabilities=runtime_capabilities(), device='cuda',
        placement='GPU_FIRST_TRAINING; SAME_MZ140_FORWARD_BACKEND_REUSED; BACKWARD_SMOKE_MEASURED_SEPARATELY'))
    try:
        rows, metadata, images, labels, cached = prepared(out)
        gradient_smoke(out, rows, images, labels)
        seed()
        model, info = load_model(BASE/'upstream', BASE/'depthor-zju-small.pt')
        # Its process-global operation is already the verified native-autograd forward.
        predictions = {'frozen': predict_arm(out, 'frozen', model, rows, images)}
        del model
        release()
        for arm in ('pixel', 'surface'):
            predictions[arm] = train_arm(out, arm, rows, images, labels)
        assert read(out/'pixel-fit-receipt.json')['initial_state_sha256'] == read(out/'surface-fit-receipt.json')['initial_state_sha256']
        assert [e['fit_order'] for e in read(out/'pixel-training.json')] == [e['fit_order'] for e in read(out/'surface-training.json')]
        write(out/'prediction-seal.json', dict(arms=predictions, freeze_sha256=sha(out/'freeze.json'),
            authority='ALL_TRAIN_PREDICTIONS_SEALED_BEFORE_SCENE3_NATIVE_PARSE; FIT_NATIVE_USED_AS_LABELS'))
        summary = evaluate(out, rows, metadata, cached, predictions)
        frozen = read(out/'freeze.json')
        assert frozen['inputs'] == {p: sha(Path(p)) for p in frozen['inputs']}
        assert frozen['sources'] == {p: sha(Path(p)) for p in frozen['sources']}
        write(out/'completion.json', dict(status='PASS', summary_sha256=sha(out/'summary.json'),
            prediction_seal_sha256=sha(out/'prediction-seal.json'), inputs_sources_unchanged=True,
            dev_accessed=False, original_test_accessed=False, alert_evaluated=False,
            resources='PROCESS_LOCAL_GPU_ONLY_RELEASED_AT_EXIT'))
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        write(out/'failure.json', dict(type=type(exc).__name__, message=str(exc),
            progress=read(out/'progress.json') if (out/'progress.json').exists() else None,
            automatic_resume=False, existing_budget_must_be_preserved=True))
        raise
    finally:
        release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
