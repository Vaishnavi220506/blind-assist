"""Governed, fixed two-arm tiny-train spatial fitting experiment; no dev access."""
from __future__ import annotations

import argparse
import copy
import itertools
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from query_occupancy_data import QUERIES, read, write, sha, stage_path, new_stage_directory
from query_occupancy_learning import configure, load_inputs, tensor_batch, label_tensors, predict
from diagnose_query_occupancy import query_pairs, pair_summary, rank_metrics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO/'artifacts.local/evidence/ba-query-occupancy-20260922'
DEST = REPO/'artifacts.local/evidence/ba-query-spatial-fit-20260922'
SEED, EPOCHS, BATCH = 202609224, 100, 12
FRAMES = (0, 2, 3, 4, 5, 6, 7)


def select_cohort(identities):
    training = [m for m in identities if m['split'] == 'train']
    families = sorted({m['type_id'] for m in training})
    groups = {f: min(m['base_group_id'] for m in training if m['type_id'] == f) for f in families}
    selected = sorted((m for m in training if m['base_group_id'] == groups[m['type_id']]
                       and m['frame_in_clip'] in FRAMES), key=lambda m: m['index'])
    assert len(families) == 4 and len(selected) == 84
    for f in families:
        assert {(m['layout_relation'], m['frame_in_clip']) for m in selected if m['type_id'] == f} == {
            (r, t) for r in ('INSIDE', 'BOUNDARY', 'OUTSIDE') for t in FRAMES}
    return selected, groups


def fit_loss(output, labels):
    """Equal positive/negative area importance per map, not a truth input to forward."""
    valid = labels['valid'].bool()
    ce = F.cross_entropy(output['distance_logits'].flatten(0, 1), labels['classes'].flatten(),
                         reduction='none').view_as(valid)
    logits, fg = output['mask_logits'], labels['mask']
    coverage = labels['coverage'][:, None].expand_as(fg)
    bg = (coverage-fg).clamp_min(0)
    fa, ba = fg.sum((-2, -1)), bg.sum((-2, -1))
    pos = (F.softplus(-logits)*fg).sum((-2, -1))/fa.clamp_min(1e-6)
    neg = (F.softplus(logits)*bg).sum((-2, -1))/ba.clamp_min(1e-6)
    terms = (fa > 0).float()+(ba > 0).float()
    bce = (pos+neg)/terms.clamp_min(1)
    visible = valid & (fa > 0)
    probability = logits.sigmoid()
    dice = 1-(2*(probability*fg).sum((-2, -1))+1)/((probability*coverage).sum((-2, -1))+fa+1)
    dice_loss = dice[visible].mean() if bool(visible.any()) else logits.sum()*0
    return (ce*valid).sum()/valid.sum().clamp_min(1)+(bce*valid).sum()/valid.sum().clamp_min(1)+dice_loss


def cohort_feasibility(labels):
    """Training-only oracle diagnostics; neither labels nor oracle maps enter forward."""
    classes, valid = labels['classes'], labels['valid']
    constraints = query_pairs(np.zeros_like(classes, dtype=np.float32), classes, valid)
    fixed_best = {k: 0. for k in ('all', 'lateral', 'height')}
    for permutation in itertools.permutations(range(6)):
        rank = np.argsort(permutation)
        for kind in fixed_best:
            selected = [p for p in constraints if kind == 'all' or p['kind'] == kind]
            if selected:
                rate = np.mean([rank[p['positive_query']] < rank[p['negative_query']] for p in selected])
                fixed_best[kind] = max(fixed_best[kind], float(rate))
    ceilings = []
    for row, q in np.argwhere(valid & (labels['mask'].sum((-2, -1)) > 0)):
        fg = labels['mask'][row, q].astype(np.float64).ravel()
        coverage = labels['coverage'][row].astype(np.float64).ravel()
        observed = coverage > 0
        fg, coverage = fg[observed], coverage[observed]
        order = np.argsort(-(fg/coverage), kind='stable')
        intersection = np.cumsum(fg[order])
        union = fg.sum()+np.cumsum(coverage[order]-fg[order])
        ceilings.append(float(np.max(intersection/union)))
    mean_ceiling = float(np.mean(ceilings)) if ceilings else 0.
    return dict(fixed_query_order_max_win_rate=fixed_best, visible_positive_queries=len(ceilings),
                binary_grid_oracle_iou_mean=mean_ceiling,
                binary_grid_oracle_iou_min=float(min(ceilings)) if ceilings else None,
                evaluable=fixed_best['all'] < .95 and mean_ceiling >= .5)


def optimizer_for(model):
    return torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if n.startswith('encoder.')], 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': 1e-3},
    ], weight_decay=1e-4)


def choose_device(model, rgb, tof, indices, labels, queries):
    sys.path.insert(0, str(REPO))
    from tools.research_backend import BackendCandidate, Workload, select_backend, torch_observation
    candidates = {}
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        clone = copy.deepcopy(model).to(device).train()
        x, t = tensor_batch(rgb, tof, indices[:BATCH], device)
        y = label_tensors(labels, np.arange(BATCH), device)
        q, optimizer = queries.to(device), optimizer_for(clone)

        def probe(m=clone, x=x, t=t, y=y, q=q, opt=optimizer):
            opt.zero_grad(set_to_none=True)
            loss = fit_loss(m(x, t, q), y)
            loss.backward()
            opt.step()
            return loss.detach()

        candidates[device] = BackendCandidate(device, device, probe,
            lambda output, m=clone: torch_observation(model=m),
            torch.cuda.synchronize if device == 'cuda' else lambda: None)
    result = select_backend(Workload.BATCH_TENSOR, cpu=candidates['cpu'], gpu=candidates.get('cuda'),
        cpu_reason=None if 'cuda' in candidates else 'ACCELERATOR_UNAVAILABLE',
        record_path=DEST/'backend.json', warmups=1, repeats=3)
    return result['selected_device_type']


def binary_counts(probability, truth, valid):
    pred = probability >= .5
    tp, fp, fn, tn = [int(x.sum()) for x in (pred & truth & valid, pred & ~truth & valid,
                                            ~pred & truth & valid, ~pred & ~truth & valid)]
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., FPR=fp/(fp+tn) if fp+tn else 0.)


def metrics(pred, labels, metadata):
    classes, valid = labels['classes'], labels['valid']
    truth, probability = classes < 6, pred['probability']
    fg = labels['mask'].astype(np.float64)
    coverage = np.broadcast_to(labels['coverage'][:, None], fg.shape).astype(np.float64)
    mask = pred['mask'] >= .5
    intersection = (mask*fg).sum((-2, -1))
    union = (mask*coverage+fg-mask*fg).sum((-2, -1))
    visible = valid & (fg.sum((-2, -1)) > 0)
    iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    pairs = query_pairs(probability, classes, valid)
    query = binary_counts(probability, truth, valid)
    mask_iou = float(iou[visible].mean()) if visible.any() else 0.
    pair = {k: pair_summary([p['margin'] for p in pairs if k == 'all' or p['kind'] == k])
            for k in ('all', 'lateral', 'height')}
    positive = valid & truth
    empty = valid & ~truth
    winners = np.argmax(pred['distribution'], axis=-1)
    pixel_tp = float(intersection[valid].sum())
    pixel_fp = float((mask*(coverage-fg))[valid].sum())
    pixel_fn = float(((~mask)*fg)[valid].sum())
    per_family = {}
    for f in sorted({m['type_id'] for m in metadata}):
        rows = np.array([m['type_id'] == f for m in metadata])
        per_family[f] = dict(frames=int(rows.sum()), queries=int(valid[rows].sum()),
            query=binary_counts(probability[rows], truth[rows], valid[rows]),
            visible_positive_queries=int(visible[rows].sum()),
            mask_iou=float(iou[rows][visible[rows]].mean()) if visible[rows].any() else None)
    orderings, counts = np.unique(np.argsort(-probability, axis=1, kind='stable'), axis=0, return_counts=True)
    strict = np.all(np.diff(np.sort(probability, axis=1), axis=1) > 0, axis=1)
    criteria = dict(precision=query['precision'] >= .95, recall=query['recall'] >= .95,
                    mask_iou=mask_iou >= .5, query_pair_order=(pair['all']['win_rate'] or 0) >= .95)
    return dict(frames=len(metadata), known_queries=int(valid.sum()), positive_queries=int(positive.sum()),
        visible_positive_queries=int(visible.sum()), geometric_positive_without_visible_pixels=int((positive & ~visible).sum()),
        query=query, query_rank=rank_metrics(probability[valid], truth[valid], ~truth[valid]),
        mask_iou=mask_iou, mask_pixel=dict(TP_area=pixel_tp, FP_area=pixel_fp, FN_area=pixel_fn,
            precision=pixel_tp/(pixel_tp+pixel_fp) if pixel_tp+pixel_fp else 0.,
            recall=pixel_tp/(pixel_tp+pixel_fn) if pixel_tp+pixel_fn else 0.),
        empty_query_mask=dict(queries=int(empty.sum()), any_predicted_area=int(mask.any((-2, -1))[empty].sum()),
            false_area_fraction=float((mask*coverage)[empty].sum()/coverage[empty].sum())),
        same_image_query_order=pair, distance_bin=dict(accuracy=float((winners[valid] == classes[valid]).mean()),
            positive_accuracy=float((winners[positive] == classes[positive]).mean())),
        query_orderings=[dict(order=o.tolist(), frames=int(n)) for o, n in zip(orderings, counts)],
        strict_order_frames=int(strict.sum()), per_family=per_family,
        criteria=criteria, fit_gate_pass=all(criteria.values()))


def make_spec():
    prepared = stage_path(SOURCE, 'prepared')
    inputs = [dict(alias='observations', path=str(prepared/'observations'), role='observation', purpose='only-selected-training-rows'),
        dict(alias='train', path=str(prepared/'labels/train.npz'), role='evaluator', purpose='consumed-tiny-train-supervision-and-fit-check'),
        dict(alias='manifest', path=str(prepared/'materialization.json'), role='configuration', purpose='input-byte-identity'),
        dict(alias='plan', path=str(SOURCE/'plan'), role='configuration', purpose='local-ImageNet-initialization')]
    spec = dict(schema='blindassist-asset-run-v1', id='query-spatial-fit-20260922', route='ue-query-occupancy',
        question='Can explicit ray-query geometry fit query-dependent localization on 84 consumed training images?',
        evaluator='research/active/dtr-r0/nearfield/run_query_spatial_fit.py',
        evidence_boundary='Tiny training fit only; no dev/held arrays or generalization claim',
        reuse=dict(mode='development', query='Existing query occupancy public training RGB ToF query localization'),
        inputs=inputs, outputs=[dict(alias='result', path=str(DEST/'result.json'), role='result', required=True)],
        result_output='result', command=[sys.executable, '-B', str(Path(__file__).resolve()), 'run'])
    path = DEST.with_name(DEST.name+'-run.json')
    write(path, spec)
    print(path)


def run():
    from query_spatial_fit_model import QuerySpatialFitNet
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state'] == 'running', 'Use governed research-ue runner'
    new_stage_directory(DEST)
    started = time.perf_counter()
    configure()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prepared = stage_path(SOURCE, 'prepared')
    materialization = read(prepared/'materialization.json')
    inputs = {str(prepared/name): materialization['hashes'][name] for name in (
        'observations/identities.json', 'observations/rgb.npy', 'observations/tof.npy', 'labels/train.npz')}
    weights = SOURCE/'plan/mobilenet_v3_small.pth'
    inputs[str(weights)] = sha(weights)
    inputs[str(prepared/'materialization.json')] = sha(prepared/'materialization.json')
    for path, digest in inputs.items():
        assert sha(path) == digest
    rgb, tof, identities = load_inputs(SOURCE)
    metadata, groups = select_cohort(identities)
    indices = np.array([m['index'] for m in metadata], dtype=np.int64)
    write(DEST/'cohort.json', dict(groups=groups, indices=indices.tolist(), metadata=metadata,
          selection_uses_labels=False, dev_or_held_rows_inferred=False))
    full = dict(np.load(prepared/'labels/train.npz', allow_pickle=False))
    lookup = {int(idx): i for i, idx in enumerate(full['indices'])}
    rows = np.array([lookup[int(idx)] for idx in indices])
    labels = {key: value[rows] for key, value in full.items()}
    assert np.array_equal(labels['indices'], indices) and labels['valid'].all()
    assert np.isfinite(labels['mask']).all() and (labels['mask'] >= 0).all()
    assert (labels['mask'] <= labels['coverage'][:, None]+1e-6).all()
    np.savez_compressed(DEST/'training-targets.npz', **labels)
    feasibility = cohort_feasibility(labels)
    write(DEST/'cohort-feasibility.json', feasibility)
    assert feasibility['evaluable'], 'Tiny cohort cannot distinguish fixed ordering or reach mask gate; do not fit or change cohort'
    code_names = ['run_query_spatial_fit.py', 'query_spatial_fit_model.py', 'query_occupancy_model.py',
                  'query_occupancy_data.py', 'query_occupancy_learning.py', 'diagnose_query_occupancy.py',
                  'QUERY_SPATIAL_FIT_PROTOCOL_20260922.md', '../../../../tools/research_backend.py']
    code = {name: sha(HERE/name) for name in code_names}
    write(DEST/'source-input-seal.json', dict(code=code, inputs=inputs, journal=journal,
          seed=SEED, epochs=EPOCHS, batch=BATCH, updates_per_arm=700))
    model = QuerySpatialFitNet(weights, carrier='spatial')
    initial = copy.deepcopy(model.state_dict())
    torch.save(initial, DEST/'initial.pt')
    rng = np.random.default_rng(SEED)
    orders = np.stack([rng.permutation(len(indices)) for _ in range(EPOCHS)])
    np.save(DEST/'batch-orders.npy', orders)
    queries = torch.tensor(QUERIES)
    device = choose_device(model, rgb, tof, indices, labels, queries)
    assert all(torch.equal(value, initial[name]) for name, value in model.state_dict().items())
    queries = queries.to(device)
    x, t = tensor_batch(rgb, tof, indices[:BATCH], device)
    model.to(device).eval()
    with torch.inference_mode():
        spatial_initial = model(x, t, queries)
        model.carrier = 'global'
        global_initial = model(x, t, queries)
        initial_equal = all(torch.equal(spatial_initial[k], global_initial[k]) for k in spatial_initial)
    assert initial_equal
    write(DEST/'pairing.json', dict(initial_sha256=sha(DEST/'initial.pt'),
          schedule_sha256=sha(DEST/'batch-orders.npy'), initial_predictions_exactly_equal=initial_equal,
          parameters=sum(p.numel() for p in model.parameters()), spatial_effective_extra_parameters=768))
    durations, checkpoint_hashes = {}, {}
    del model, x, t
    for carrier in ('global', 'spatial'):
        torch.manual_seed(SEED)
        model = QuerySpatialFitNet(None, carrier=carrier)
        model.load_state_dict(initial, strict=True)
        model.to(device).train()
        optimizer = optimizer_for(model)
        history, arm_started, updates = [], time.perf_counter(), 0
        for epoch, order in enumerate(orders, 1):
            losses = []
            for start in range(0, len(order), BATCH):
                subset = order[start:start+BATCH]
                x, t = tensor_batch(rgb, tof, indices[subset], device)
                y = label_tensors(labels, subset, device)
                optimizer.zero_grad(set_to_none=True)
                loss = fit_loss(model(x, t, queries), y)
                assert bool(torch.isfinite(loss)), 'Nonfinite training loss; preserve this failure'
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                updates += 1
            history.append(dict(epoch=epoch, updates=updates, loss=float(np.mean(losses))))
            if epoch % 10 == 0:
                print('FIT', carrier, epoch, updates, history[-1]['loss'], flush=True)
        assert updates == 700
        durations[carrier] = time.perf_counter()-arm_started
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, DEST/f'{carrier}.pt')
        checkpoint_hashes[carrier] = sha(DEST/f'{carrier}.pt')
        write(DEST/f'{carrier}-history.json', history)
        pred = predict(model, rgb, tof, indices, queries, device, 'occupancy', masks=True)
        np.savez_compressed(DEST/f'{carrier}-predictions.npz', indices=indices, **pred)
        print('SEALED', carrier, flush=True)
        del model, optimizer, loss, x, t, y
        if device == 'cuda':
            torch.cuda.empty_cache()
    write(DEST/'prediction-seal.json', dict(checkpoints=checkpoint_hashes,
          predictions={a: sha(DEST/f'{a}-predictions.npz') for a in ('global', 'spatial')},
          evaluated_state='final_epoch_100', thresholds=dict(query=.5, mask=.5), dev_held_access=False))
    results = {a: metrics(dict(np.load(DEST/f'{a}-predictions.npz', allow_pickle=False)), labels, metadata)
               for a in ('global', 'spatial')}
    g, s = [results[a]['fit_gate_pass'] for a in ('global', 'spatial')]
    decision = ('BOTH_FIT_SPATIAL_NOT_NECESSARY' if g and s else
                'SPATIAL_PACKAGE_FITS_ONLY' if s else
                'GLOBAL_FITS_SPATIAL_BRANCH_REJECTED' if g else 'NEITHER_PACKAGE_FITS')
    for path, digest in inputs.items():
        assert sha(path) == digest
    for name, digest in code.items():
        assert sha(HERE/name) == digest
    result = dict(status='PASS', decision=decision, arms=results, input_hashes_unchanged=True,
        frames=len(indices), groups=groups, feasibility=feasibility,
        unknown_sensor_frames=sum(bool(m['baseline']['unknown']) for m in metadata),
        epochs=EPOCHS, updates_per_arm=700, training_seconds=durations, actual_device=device,
        dev_held_access=False, threshold_selection=False, elapsed_s=time.perf_counter()-started,
        evidence_boundary='Consumed four-group training fit only; no generalization, alert or safety claim')
    write(DEST/'result.json', result)
    print('COMPLETE', decision, DEST, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['spec', 'run'])
    args = parser.parse_args()
    make_spec() if args.command == 'spec' else run()
