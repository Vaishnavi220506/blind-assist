"""Matched single-frame learning; held labels are opened only after prediction seal."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from query_occupancy_data import CENTRE, QUERIES, EDGES, read, write, sha, stage_path, new_stage_directory

SEED = 202609223
EPOCHS = 24
BATCH = 16


def configure():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def supervised_loss(output, labels, mode):
    valid = labels['valid'].float()
    if mode == 'classifier':
        loss = F.binary_cross_entropy_with_logits(output['query_logits'], (labels['classes'] < 6).float(), reduction='none')
        return (loss*valid).sum()/valid.sum().clamp_min(1)
    ce = F.cross_entropy(output['distance_logits'].flatten(0, 1), labels['classes'].flatten(), reduction='none').view_as(valid)
    p = output['mask_logits'].sigmoid()
    cover = labels['coverage'][:, None].expand_as(p)
    mask = labels['mask']
    target = mask/cover.clamp_min(1e-6)
    weight = cover*valid[:, :, None, None]
    bce = (F.binary_cross_entropy_with_logits(output['mask_logits'], target, reduction='none')*weight).sum()/weight.sum().clamp_min(1)
    dice = 1-(2*(p*mask).sum((-1, -2))+1)/((p*cover).sum((-1,-2))+mask.sum((-1,-2))+1)
    return (ce*valid).sum()/valid.sum().clamp_min(1)+bce+(dice*valid).sum()/valid.sum().clamp_min(1)


def scores(output, mode):
    return output['query_logits'].sigmoid() if mode == 'classifier' else output['distance_logits'].softmax(-1)[..., :-1].sum(-1)


def frame_truth(labels):
    return (labels['classes'][:, CENTRE] < 6).any(1), labels['valid'][:, CENTRE].all(1)


def operating_point(score, truth, valid, identities):
    """Atomic ties; no artificial threshold can split identical scores."""
    s = np.asarray(score, np.float64)
    y = np.asarray(truth, bool)
    v = np.asarray(valid, bool)
    if not (y & v).any() or not (~y & v).any():
        raise ValueError('Operating point NOT_EVALUABLE: need known positives and known negatives')
    thresholds = np.r_[np.nextafter(s.max(), np.inf), np.unique(s)[::-1]]
    best = None
    curve = []
    for threshold in thresholds:
        pred = s >= threshold
        tp, fp = int((pred & y & v).sum()), int((pred & ~y & v).sum())
        fn, negatives = int((~pred & y & v).sum()), int((~y & v).sum())
        recall = tp/max(1, int((y & v).sum()))
        fpr = fp/max(1, negatives)
        segments, previous = 0, {}
        for i, row in enumerate(identities):
            active = bool(pred[i] and not y[i] and v[i])
            segments += int(active and not previous.get(row['clip_id'], False))
            previous[row['clip_id']] = active
        record = dict(threshold=float(threshold), TP=tp, FP=fp, FN=fn, recall=recall,
                      FPR=fpr, precision=tp/max(1, tp+fp), false_segments=segments)
        curve.append(record)
        key = (recall, -segments, -fp, float(threshold))
        if fpr <= .05+1e-12 and (best is None or key > best[0]):
            best = (key, record)
    return best[1], curve


def load_inputs(root):
    obs = stage_path(root, 'prepared')/'observations'
    return (np.load(obs/'rgb.npy', mmap_mode='r'), np.load(obs/'tof.npy', mmap_mode='r'),
            read(obs/'identities.json'))


def tensor_batch(rgb, tof, indices, device):
    x = torch.as_tensor(np.array(rgb[indices], copy=True), device=device).float()/255
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return (x-mean)/std, torch.as_tensor(np.array(tof[indices], copy=True), device=device)


def label_tensors(labels, rows, device):
    return {k: torch.as_tensor(labels[k][rows], device=device) for k in ('classes', 'valid', 'mask', 'coverage')}


def predict(model, rgb, tof, indices, queries, device, mode, masks=False):
    model.eval()
    probabilities, distributions, maps = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH):
            x, t = tensor_batch(rgb, tof, indices[start:start+BATCH], device)
            out = model(x, t, queries)
            probabilities.append(scores(out, mode).cpu().numpy())
            if mode == 'occupancy':
                distributions.append(out['distance_logits'].softmax(-1).cpu().numpy())
                if masks:
                    maps.append(out['mask_logits'].sigmoid().cpu().numpy())
    return dict(probability=np.concatenate(probabilities),
                distribution=np.concatenate(distributions) if distributions else np.empty(0),
                mask=np.concatenate(maps) if maps else np.empty(0), elapsed_s=time.perf_counter()-started)


def choose_backend(model, rgb, tof, indices, queries, path, training_labels=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from tools.research_backend import BackendCandidate, Workload, select_backend, torch_observation
    candidates = {}
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        m = copy.deepcopy(model).to(device)
        x, t = tensor_batch(rgb, tof, indices[:BATCH], device)
        q = queries.to(device)
        if training_labels is None:
            m.eval()
            def run(m=m, x=x, t=t, q=q):
                with torch.inference_mode():
                    return m(x, t, q)
        else:
            m.train()
            labels = label_tensors(training_labels, np.arange(min(BATCH,len(indices))), device)
            optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
            def run(m=m, x=x, t=t, q=q, labels=labels, optimizer=optimizer):
                optimizer.zero_grad(set_to_none=True)
                output = m(x,t,q)
                loss = supervised_loss(output,labels,m.mode)
                loss.backward()
                optimizer.step()
                return loss.detach()
        candidates[device] = BackendCandidate(device, device, run,
            lambda out, m=m: torch_observation(model=m),
            torch.cuda.synchronize if device == 'cuda' else lambda: None)
    result = select_backend(Workload.BATCH_TENSOR if training_labels is not None else Workload.MODEL_INFERENCE,
        cpu=candidates['cpu'], gpu=candidates.get('cuda'),
        cpu_reason=None if 'cuda' in candidates else 'ACCELERATOR_UNAVAILABLE',
        record_path=path, warmups=1, repeats=3)
    return result['selected_device_type']


def fit(root):
    from query_occupancy_model import QueryOccupancyNet
    root = Path(root)
    configure()
    out = stage_path(root, 'fit')
    new_stage_directory(out)
    here = Path(__file__).resolve().parent
    source = {p.name:sha(p) for p in here.glob('query_occupancy_*.py')}
    source.update({name:sha(here/name) for name in (
        'run_query_occupancy.py', 'ba_camera_corridor_metrics.py',
        '../../../../tools/research_backend.py')})
    prepared = stage_path(root, 'prepared')
    materialization = read(prepared/'materialization.json')
    used_inputs = {name:materialization['hashes'][name] for name in (
        'observations/rgb.npy', 'observations/tof.npy', 'observations/identities.json',
        'labels/train.npz', 'labels/dev.npz')}
    for name, digest in used_inputs.items():
        assert sha(prepared/name) == digest
    write(out/'learning-source-seal.json', dict(code=source,
        protocol_sha256=sha(root/'plan/learning-protocol.md'), held_labels_opened=False,
        materialization_sha256=sha(prepared/'materialization.json'), used_inputs=used_inputs))
    rgb, tof, identities = load_inputs(root)
    train = dict(np.load(stage_path(root, 'prepared')/'labels/train.npz', allow_pickle=False))
    dev = dict(np.load(stage_path(root, 'prepared')/'labels/dev.npz', allow_pickle=False))
    train_ids, dev_ids = train['indices'], dev['indices']
    assert all(identities[int(i)]['split'] == 'train' for i in train_ids)
    assert all(identities[int(i)]['split'] == 'dev' for i in dev_ids)
    queries = torch.as_tensor(QUERIES)
    results = {}
    # Identical initial parameters and training image order in both arms.
    for mode in ('classifier', 'occupancy'):
        configure()
        model = QueryOccupancyNet(root/'plan/mobilenet_v3_small.pth', mode=mode, n_bins=6)
        device = choose_backend(model, rgb, tof, train_ids, queries, out/f'{mode}-backend.json', train)
        model.to(device)
        q = queries.to(device)
        encoder, heads = [], []
        for name, param in model.named_parameters():
            (encoder if name.startswith('encoder.') else heads).append(param)
        opt = torch.optim.AdamW([dict(params=encoder, lr=3e-5), dict(params=heads, lr=3e-4)], weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        rng = np.random.default_rng(SEED)
        history, best, begun = [], None, time.perf_counter()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, EPOCHS+1):
            model.train()
            loss_total = 0.
            order = rng.permutation(len(train_ids))
            for start in range(0, len(order), BATCH):
                rows = order[start:start+BATCH]
                x, t = tensor_batch(rgb, tof, train_ids[rows], device)
                opt.zero_grad(set_to_none=True)
                output = model(x, t, q)
                loss = supervised_loss(output, label_tensors(train, rows, device), mode)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
                loss_total += float(loss.detach())*len(rows)
            schedule.step()
            record = dict(epoch=epoch, train_loss=loss_total/len(order), elapsed_s=time.perf_counter()-begun)
            if epoch % 4 == 0:
                pred = predict(model, rgb, tof, dev_ids, q, device, mode)
                truth, valid = frame_truth(dev)
                selection, _ = operating_point(pred['probability'][:, CENTRE].max(1), truth, valid,
                                                [identities[int(i)] for i in dev_ids])
                record['dev'] = selection
                key = (selection['recall'], -selection['false_segments'])
                if best is None or key > best[0]:
                    best = (key, dict(epoch=epoch, selection=selection))
                    saved_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    torch.save(dict(state_dict=saved_state, mode=mode, epoch=epoch,
                                    selection=selection, seed=SEED), out/f'{mode}.pt')
            history.append(record)
            print('FIT', mode, record, flush=True)
            # Progress is replaceable operational state, separate from evidence.
            (out/'progress.json').write_text(__import__('json').dumps(dict(mode=mode, epoch=epoch, total_epochs=EPOCHS)))
        results[mode] = dict(**best[1], history=history, checkpoint_sha256=sha(out/f'{mode}.pt'),
            backend=device, train_seconds=time.perf_counter()-begun,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None)
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write(out/'selection.json', results)
    return dict(status='PASS', selected={k:{n:v[n] for n in ('epoch','selection','checkpoint_sha256','train_seconds','parameters','backend')} for k,v in results.items()},
                held_labels_opened=False)


def seal_predictions(root):
    from query_occupancy_model import QueryOccupancyNet
    root = Path(root)
    configure()
    out = stage_path(root, 'predictions')
    new_stage_directory(out)
    rgb, tof, identities = load_inputs(root)
    # Only public split identity; held evaluator files are never loaded here.
    indices = np.array([i for i,r in enumerate(identities) if r['split'] == 'evaluation'])
    selection = read(stage_path(root, 'fit')/'selection.json')
    device = selection['occupancy']['backend']
    q = torch.as_tensor(QUERIES, device=device)
    report = dict(status='PASS', held_labels_opened=False, frames=len(indices), models={},
                  selection_sha256=sha(stage_path(root,'fit')/'selection.json'),
                  source_seal_sha256=sha(stage_path(root,'fit')/'learning-source-seal.json'))
    for mode in ('classifier', 'occupancy'):
        path = stage_path(root, 'fit')/f'{mode}.pt'
        assert sha(path) == selection[mode]['checkpoint_sha256']
        model = QueryOccupancyNet(root/'plan/mobilenet_v3_small.pth', mode=mode, n_bins=6)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        result = predict(model, rgb, tof, indices, q, device, mode, masks=True)
        latency=[]
        model.eval()
        with torch.inference_mode():
            for iteration in range(23):
                if device=='cuda':
                    torch.cuda.synchronize()
                started=time.perf_counter()
                x,t=tensor_batch(rgb,tof,indices[:1],device)
                output=model(x,t,q)
                for value in output.values():
                    value.cpu()
                if device=='cuda':
                    torch.cuda.synchronize()
                if iteration>=3:
                    latency.append((time.perf_counter()-started)*1000)
        np.savez_compressed(out/f'{mode}.npz', indices=indices, **result)
        report['models'][mode] = dict(predictions_sha256=sha(out/f'{mode}.npz'),
                                    checkpoint_sha256=sha(path), elapsed_s=result['elapsed_s'],
                                    single_frame_latency_ms=dict(p50=float(np.median(latency)),p95=float(np.quantile(latency,.95)),
                                        scope='cached uint8 RGB plus ToF to all network CPU outputs; excludes camera and PNG decoding',repeats=len(latency)))
    write(out/'prediction-seal.json', report)
    return report
