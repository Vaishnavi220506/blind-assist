"""One grouped public ToF positive-readout recipe; frozen A is only ORed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

from prepare_public_positive import ROOT, HERE, WORK, OUT as PREP, read, write, sha
from public_positive import PositiveHead, spatial_features, add_positive, select_threshold
from tolerance_eval import metric, temporal
from research_backend import BackendCandidate, DeviceObservation, select_backend

OUT = WORK/'corridor-public-positive-20260917/run-v1'
EPOCHS, BATCH, SEED = 120, 16, 183017


def benchmark(x, valid, target):
    models = {}
    for device in ['cpu', 'cuda']:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        models[device] = (PositiveHead().to(device), torch.tensor(x[:16], device=device),
                          torch.tensor(valid[:16], device=device), torch.tensor(target[:16], device=device))
    def step(device):
        m, xx, vv, yy = models[device]
        m.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(m(xx, vv), yy)
        loss.backward()
        return loss
    def candidate(device):
        return BackendCandidate('positive-head-'+device, device, lambda: step(device),
            lambda v: DeviceObservation(v.device.type, torch.cuda.get_device_name(0) if device == 'cuda' else 'host CPU', torch.__version__),
            torch.cuda.synchronize if device == 'cuda' else lambda: None)
    selected = select_backend('model-inference', cpu=candidate('cpu'),
        gpu=candidate('cuda') if 'cuda' in models else None,
        warmups=2, repeats=5, record_path=OUT/'backend.json')
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return read(OUT/'backend.json')['selected_device_type']


def predict(model, x, valid, indices):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(indices), 64):
            ix = indices[i:i+64]
            chunks.append(model(x[ix], valid[ix]).cpu().numpy())
    return np.concatenate(chunks)


def score_changes(y, a, p):
    return dict(FN_rescued=int(sum(y & ~a & p)), FP_added=int(sum(~y & ~a & p)),
                TP_lost=int(sum(y & a & ~p)), FP_removed=int(sum(~y & a & ~p)))


def evaluate(meta, logits, thresholds, valid, targets, known, fold_indices):
    a = np.array([bool(m['A']) for m in meta])
    p, scores, branch = add_positive(a, logits, valid, thresholds)
    y = np.array([m['truth'] for m in meta])
    state = np.array([m['stratum'] for m in meta])
    cohort = np.array([m['cohort'] for m in meta])
    oracle = a | ((targets > 0) & known).any(1)
    assert not np.any(a & ~p)
    assert not np.any(branch & ~valid.any(1))
    summary = {}
    for c in ['old', 'new']:
        ix = np.flatnonzero(cohort == c)
        rows = [meta[i] for i in ix]
        cy, cs = y[ix], state[ix]
        clear = cs != 'boundary'
        methods = {}
        for name, flags in [('A', a), ('A_public_OR', p), ('A_oracle_sampled_OR', oracle)]:
            cp = flags[ix]
            t = temporal(rows, cs, cp)
            strict_t = temporal(rows, np.where(cy, 'positive', 'negative'), cp)
            methods[name] = dict(clear=metric(cy[clear], cp[clear]), coverage=float(clear.mean()),
                strict=metric(cy, cp), boundary=metric(cy[~clear], cp[~clear]),
                clear_changes=score_changes(cy[clear], a[ix][clear], cp[clear]),
                temporal=t, strict_events=dict(total=strict_t['core_events'], detected=strict_t['core_events_detected']),
                families={f: dict(clear=metric(cy[clear & (np.array([m['family'] for m in rows]) == f)],
                    cp[clear & (np.array([m['family'] for m in rows]) == f)]),
                    changes=score_changes(cy[clear & (np.array([m['family'] for m in rows]) == f)],
                        a[ix][clear & (np.array([m['family'] for m in rows]) == f)],
                        cp[clear & (np.array([m['family'] for m in rows]) == f)]))
                    for f in sorted({m['family'] for m in rows})})
        old_events = {e['episode']:e for e in methods['A']['temporal']['events']}
        for name in ['A_public_OR', 'A_oracle_sampled_OR']:
            events = methods[name]['temporal']['events']
            changes = []
            for e in events:
                old = old_events[e['episode']]['first_in_core_delay_s']
                new = e['first_in_core_delay_s']
                if old is not None:
                    assert new is not None and new <= old
                if old != new:
                    changes.append(dict(episode=e['episode'], A_first_s=old, probe_first_s=new))
            methods[name]['first_alert_changes'] = changes
        slots_p = logits[ix] >= thresholds[ix, None]
        slots_y, slots_k = targets[ix] > 0, known[ix]
        methods['return_diagnostic'] = metric(slots_y[slots_k], slots_p[slots_k])
        methods['zero_tof_frames'] = int(sum(~valid[ix].any(1)))
        methods['zero_tof_changed_alerts'] = int(sum((~valid[ix].any(1)) & (a[ix] != p[ix])))
        methods['folds'] = {str(k): dict(A=metric(y[jj], a[jj]), public_OR=metric(y[jj], p[jj]),
            changes=score_changes(y[jj], a[jj], p[jj])) for k in range(6)
            for jj in [np.flatnonzero((cohort == c) & (fold_indices == k) & (state != 'boundary'))]}
        summary[c] = methods
    cases = [dict(**m, fold=int(fold_indices[i]), threshold=float(thresholds[i]),
        evidence_logit=float(scores[i]), evidence_alert=bool(branch[i]), public_OR=bool(p[i]),
        oracle_sampled_OR=bool(oracle[i]), sampled_witness=bool(((targets[i]>0)&known[i]).any()),
        usable_returns=int(valid[i].sum()), added_alert=bool(p[i] and not a[i]),
        selected_positive_returns=int(sum(valid[i] & (logits[i] >= thresholds[i])))) for i,m in enumerate(meta)]
    return summary, cases


def main():
    assert not OUT.exists(), 'Sealed run outputs must not be overwritten'
    OUT.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_seal = read(PREP/'data-seal.json')
    for path, h in data_seal['bindings'].items():
        assert sha(Path(path)) == h, path
    for name, h in data_seal['outputs'].items():
        assert sha(PREP/name) == h, name
    data = np.load(PREP/'public-tokens.npz')
    labels = np.load(PREP/'offline-targets.npz')
    meta, folds = read(PREP/'metadata.json'), read(PREP/'folds.json')
    features = spatial_features(data['tokens'])
    valid_np = data['valid'][:, :128]
    target_np, known_np = labels['target'][:, :128], labels['known'][:, :128]
    assert len(meta) == 768 and not np.any(known_np & ~valid_np)
    device = benchmark(features, valid_np, target_np)
    sources = {str(p):sha(p) for p in [Path(__file__), HERE/'public_positive.py',
        HERE/'PUBLIC_POSITIVE_PROTOCOL_20260917.md', HERE/'tolerance_eval.py', ROOT/'tools/research_backend.py']}
    write(OUT/'freeze.json', dict(sources=sources, data_seal_sha256=sha(PREP/'data-seal.json'),
        epochs=EPOCHS, batch=BATCH, seed=SEED, parameters=1537, device=device,
        optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001),
        authority='WHOLE_GROUP_FIT_CALIBRATION_REPORT_DISJOINT_CONSUMED_DEVELOPMENT',
        folds=folds, final_checkpoint_only=True, threshold_metric='FINAL_OR_CLEAR_F1_THEN_FEWER_FP_THEN_HIGHER_THRESHOLD'))
    x = torch.tensor(features, device=device)
    valid = torch.tensor(valid_np, device=device)
    target = torch.tensor(target_np, device=device)
    report_indices = np.array([i for i,m in enumerate(meta) if m['cohort'] != 'anchor'])
    oof = np.full((len(meta), 128), np.nan, np.float32)
    thresholds = np.full(len(meta), np.nan)
    fold_id = np.full(len(meta), -1)
    reports = []
    checkpoints = {}
    model = optimizer = None
    try:
        for fold in folds:
            k = fold['fold']
            fi, ci, ri = map(lambda n: np.array(fold[n]), ['fit','calibration','report'])
            torch.manual_seed(SEED+k)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED+k)
            fitting_values = features[fi][valid_np[fi]]
            mean = fitting_values.mean(0)
            scale = fitting_values.std(0).clip(.01)
            model = PositiveHead(mean, scale).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            pos = known_np[fi] & (target_np[fi] > 0)
            neg = known_np[fi] & (target_np[fi] == 0)
            weight = np.zeros_like(target_np)
            weight[fi] = .5*pos/pos.sum() + .5*neg/neg.sum()
            assert not weight[ci].any() and not weight[ri].any()
            w = torch.tensor(weight, device=device)
            rng = np.random.default_rng(SEED+k)
            history, orders = [], []
            tick = time.perf_counter()
            for epoch in range(1, EPOCHS+1):
                model.train()
                order = rng.permutation(fi)
                orders.append(order.tolist())
                loss_sum = 0.
                for j in range(0, len(order), BATCH):
                    ii = order[j:j+BATCH]
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x[ii], valid[ii])
                    loss = (F.binary_cross_entropy_with_logits(output, target[ii], reduction='none') * w[ii]).sum()*len(fi)/len(ii)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite fitting loss')
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach())
                history.append(dict(epoch=epoch, loss=loss_sum/(len(order)/BATCH)))
                if epoch % 30 == 0:
                    print(json.dumps(dict(stage='fit', fold=k, epoch=epoch, loss=history[-1]['loss'], seconds=time.perf_counter()-tick)), flush=True)
            torch.save(dict(state_dict=model.state_dict(), epoch=EPOCHS,
                optimizer=optimizer.state_dict(), seed=SEED+k), OUT/f'fold{k}-model.pt')
            checkpoints[f'fold{k}-model.pt'] = sha(OUT/f'fold{k}-model.pt')
            write(OUT/f'fold{k}-training.json', history)
            write(OUT/f'fold{k}-orders.json', orders)
            calibration = predict(model, x, valid, ci)
            ca = np.array([meta[i]['A'] for i in ci])
            cy = np.array([meta[i]['truth'] for i in ci])
            cc = np.array([meta[i]['stratum'] != 'boundary' for i in ci])
            chosen, curve = select_threshold(ca, calibration, valid_np[ci], cy, cc)
            np.savez_compressed(OUT/f'fold{k}-calibration.npz', indices=ci, logits=calibration)
            write(OUT/f'fold{k}-selection.json', dict(chosen=chosen, curve=curve,
                model_sha256=checkpoints[f'fold{k}-model.pt'], calibration_sha256=sha(OUT/f'fold{k}-calibration.npz'),
                fit_indices=fi.tolist(), calibration_indices=ci.tolist(), report_indices=ri.tolist(),
                known_positive=int(pos.sum()), known_negative=int(neg.sum()),
                normalization_fit_only=True, no_report_selection=True))
            pred = predict(model, x, valid, ri)
            oof[ri] = pred
            thresholds[ri] = chosen['threshold']
            fold_id[ri] = k
            np.savez_compressed(OUT/f'fold{k}-report.npz', indices=ri, logits=pred)
            fit_pred = predict(model, x, valid, fi)
            fit_metrics = metric((target_np[fi] > 0)[known_np[fi]],
                                 (fit_pred >= 0)[known_np[fi]])
            reports.append(dict(fold=k, threshold=chosen['threshold'], calibration=chosen,
                fit_slot_at_zero=fit_metrics, last_loss=history[-1]['loss'], seconds=time.perf_counter()-tick))
            print(json.dumps(dict(stage='fold_sealed', **reports[-1])), flush=True)
            del model, optimizer
            model = optimizer = None
        assert np.isfinite(oof[report_indices]).all() and np.isfinite(thresholds[report_indices]).all()
        np.savez_compressed(OUT/'oof-predictions.npz', indices=report_indices, logits=oof[report_indices],
            thresholds=thresholds[report_indices], folds=fold_id[report_indices])
        write(OUT/'prediction-seal.json', dict(freeze_sha256=sha(OUT/'freeze.json'), models=checkpoints,
            outputs={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.npz','.json') and p.name != 'prediction-seal.json'},
            authority='ALL_OUTER_GROUP_PREDICTIONS_SEALED_BEFORE_OUTER_REDUCTION'))
        s, cases = evaluate([meta[i] for i in report_indices], oof[report_indices], thresholds[report_indices],
            valid_np[report_indices], target_np[report_indices], known_np[report_indices], fold_id[report_indices])
        write(OUT/'cases.json', cases)
        write(OUT/'summary.json', dict(cohorts=s, folds=reports, device=device,
            parameters=1537, seconds=time.perf_counter()-started,
            authority='PUBLIC_INPUT_GROUPED_CONSUMED_DEVELOPMENT_NOT_FRESH_CONFIRMATION',
            zero_capture=True, zero_veto=True, original_test_access=False))
        assert all(sha(Path(p)) == h for p,h in sources.items())
        write(OUT/'completion.json', dict(status='PASS', summary_sha256=sha(OUT/'summary.json'),
            prediction_seal_sha256=sha(OUT/'prediction-seal.json'),
            resources='Process-local model released; no persistent allocation'))
        print(json.dumps({c:{a:s[c][a]['clear'] for a in ['A','A_public_OR','A_oracle_sampled_OR']} for c in s}), flush=True)
    finally:
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            write(OUT/'failure.json', dict(error=type(exc).__name__, message=str(exc)))
        raise
