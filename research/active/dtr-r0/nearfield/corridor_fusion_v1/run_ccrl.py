"""One bounded, cross-fitted CCRL experiment on authenticated consumed inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import json
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(HERE.parent), str(ROOT/'tools')]
from ccrl import CorridorResidual, log_odds, pairing, objective, local_pairs
from prepare_single_a import read, write, sha
from prepare_positive_v2 import CAPS
from train_single_a import PARAMS
from tolerance_eval import metric, temporal
from research_backend import BackendCandidate, DeviceObservation, select_backend
from mz143_corridor_features import extract

WORK = ROOT/'artifacts.local/work'
SOURCE = WORK/'corridor-public-single-20260917'
DATA = SOURCE/'a-control'
OUT = WORK/'corridor-ccrl-20260918'
SEED, STEPS = 187018, 120


def placement(x, y, base, rank, inv):
    objects = {}
    for dev in ('cpu', 'cuda'):
        if dev == 'cuda' and not torch.cuda.is_available():
            continue
        torch.manual_seed(SEED)
        model = CorridorResidual(x.mean(0), np.maximum(x.std(0), .001)).to(dev)
        objects[dev] = (model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01),
                        *[torch.as_tensor(a, device=dev) for a in (x, y, base, rank, inv)])
    def step(dev):
        m, opt, xx, yy, bb, rr, vv = objects[dev]
        opt.zero_grad(set_to_none=True)
        loss = objective(m(xx, bb), yy, rr, vv, True, bool(len(inv)))
        loss.backward()
        opt.step()
        return loss
    def candidate(dev):
        return BackendCandidate('ccrl-'+dev, dev, lambda: step(dev),
            lambda z: DeviceObservation(z.device.type, torch.cuda.get_device_name(0) if dev == 'cuda' else 'host CPU', str(torch.__version__)),
            torch.cuda.synchronize if dev == 'cuda' else lambda: None)
    result = select_backend('batch-tensor', cpu=candidate('cpu'),
        gpu=candidate('cuda') if 'cuda' in objects else None,
        cpu_reason=None if 'cuda' in objects else 'ACCELERATOR_UNAVAILABLE',
        warmups=2, repeats=5, record_path=OUT/'backend.json')
    objects.clear()
    return result['selected_device_type']


def fit_head(x, y, b, pairs, inv, arm, dev, path):
    started = time.perf_counter()
    torch.manual_seed(SEED)
    mean, scale = x.mean(0), np.maximum(x.std(0), .001)
    model = CorridorResidual(mean, scale).to(dev)
    xx, yy, bb, rr, vv = [torch.as_tensor(a, device=dev) for a in (x, y, b, pairs, inv)]
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history = []
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        z = model(xx, bb)
        loss = objective(z, yy, rr, vv, arm != 'bce', arm == 'rank_invariance')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))
    if dev == 'cuda':
        torch.cuda.synchronize()
    model.cpu().eval()
    torch.save(dict(state_dict=model.state_dict(), arm=arm, seed=SEED, steps=STEPS,
                    history=history, seconds=time.perf_counter()-started), path)
    return model


@torch.inference_mode()
def predict(model, x, b):
    return model(torch.as_tensor(x), torch.as_tensor(b)).numpy().astype(np.float64)


def select(scores, y, clear):
    curve = []
    for tau in np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores[clear])]:
        mm = metric(y[clear], scores[clear] >= tau)
        curve.append(dict(threshold=float(tau), **mm))
    return max(curve, key=lambda a: (a['f1'], -a['FP'], a['threshold'])), curve


def report(rows, scores, flags, pairs, reference):
    y = np.array([m['truth'] for m in rows], bool)
    state = np.array([m['stratum'] for m in rows])
    clear = state != 'boundary'
    masks = dict(clear=clear, strict=np.ones(len(rows), bool), boundary=~clear)
    masks.update({f: clear & np.array([m['family'] == f for m in rows]) for f in sorted({m['family'] for m in rows})})
    if 'usable_tof_returns' in rows[0]:
        masks['zero_return'] = np.array([m['usable_tof_returns'] == 0 for m in rows])
        masks['native_supported'] = np.array([m['sampled_witness'] for m in rows])
    methods = {}
    for arm, p in flags.items():
        methods[arm] = dict(metrics={k: metric(y[v], p[v]) for k, v in masks.items()},
            coverage=float(clear.mean()), core=temporal(rows, state, p),
            strict_events=temporal(rows, np.where(y, 'positive', 'negative'), p),
            changes={k: dict(rescued_FN=int(sum(v & y & ~reference & p)),
                added_FP=int(sum(v & ~y & ~reference & p)), lost_TP=int(sum(v & y & reference & ~p)),
                removed_FP=int(sum(v & ~y & reference & ~p))) for k, v in masks.items()},
            pair_order_accuracy=float(np.mean(scores[arm][pairs[:, 0]] > scores[arm][pairs[:, 1]])),
            pair_logit_margin_mean=float(np.mean(scores[arm][pairs[:, 0]]-scores[arm][pairs[:, 1]])))
    base = methods['A_star']
    for arm, entry in methods.items():
        for kind in ('core', 'strict_events'):
            old = {e['episode']: e for e in base[kind]['events']}
            entry[kind+'_changes'] = [dict(episode=e['episode'], baseline=old[e['episode']]['first_in_core_delay_s'],
                current=e['first_in_core_delay_s']) for e in entry[kind]['events']
                if old[e['episode']]['first_in_core_delay_s'] != e['first_in_core_delay_s']]
    return methods


def main():
    started = time.perf_counter()
    assert not OUT.exists(), 'Preserve completed/failed run; never overwrite'
    assert (ROOT/'artifacts.local').resolve() == Path('F:/ba-data/blindassist-artifacts-20260805').resolve()
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    bindings = {}
    def bind(p, expected=None):
        h = sha(p)
        assert expected is None or h == expected, str(p)
        bindings[str(p)] = h
    ds = read(DATA/'data-seal.json')
    bind(DATA/'data-seal.json')
    for p, h in ds['bindings'].items():
        bind(Path(p), h)
    for name, h in ds['outputs'].items():
        bind(DATA/name, h)
    meta = read(DATA/'metadata.json')
    z = np.load(DATA/'features.npz')
    x = z['base'].astype(np.float32)
    assert list(z['ids']) == [m['id'] for m in meta] and x.shape == (1344, 2485)
    frames = {}
    for cohort, stem in CAPS.items():
        path = WORK/stem/'source/returned-v1/capture-v1/spec.json'
        receipt = read(path.with_name('receipt.json'))
        bind(path, receipt['spec_sha256'])
        for f in read(path)['frames']:
            if cohort != 'anchor' or f['split'] == 'train':
                frames[f['id']] = f
    pairs, inv, rejected = pairing(meta, frames)
    write(OUT/'pair-audit.json', dict(rank_pairs=pairs.tolist(), invariant_pairs=inv.tolist(), rejected=rejected,
        rank_count=len(pairs), invariance_count=len(inv),
        invariance_status='ELIGIBLE' if len(inv) else 'NOT_EVALUABLE_NO_EXACT_BACKGROUND_INTERVENTIONS',
        ids=[m['id'] for m in meta], noise='Same random starts, conditional draw alignment not guaranteed'))
    print(json.dumps(dict(rank_pairs=len(pairs), invariant_pairs=len(inv), rejected=len(rejected))), flush=True)
    assert len(pairs) > 0
    arms = ['bce', 'rank'] + (['rank_invariance'] if len(inv) else [])
    y = np.array([m['truth'] for m in meta], np.float32)
    cohort = np.array([m['cohort'] for m in meta])
    scene = np.array([m['scene_index'] for m in meta])
    nonanchor = np.flatnonzero(cohort != 'anchor')
    anchor = np.flatnonzero(cohort == 'anchor')
    saved = np.load(DATA/'oof-scores.npz')
    bind(DATA/'oof-scores.npz', read(DATA/'selection.json')['oof_sha256'])
    assert np.array_equal(saved['indices'], nonanchor)
    base_oof = np.full(1344, np.nan, np.float32)
    base_oof[nonanchor] = log_odds(saved['scores'])
    for p in [Path(__file__), HERE/'ccrl.py', HERE/'CCRL_PROTOCOL_20260918.md', HERE/'tolerance_eval.py']:
        bind(p)
    for k in range(6):
        bind(DATA/f'fold{k}.pkl', read(DATA/f'fold{k}-fit.json')['model_sha256'])
    bind(DATA/'retrainedA.pkl', read(DATA/'threshold.json')['model_sha256'])
    freeze = dict(inputs=bindings, arms=arms, seed=SEED, steps=STEPS, fit_rows=1344,
        residual_rows=1152, architecture='2485->16->1 additive logit residual, zero output initialization',
        parameters=sum(p.numel() for p in CorridorResidual().parameters()),
        protocol_sha256=sha(HERE/'CCRL_PROTOCOL_20260918.md'), source_pairs_sha256=sha(OUT/'pair-audit.json'),
        report_authority='CONSUMED_DEVELOPMENT', threshold='POOLED_OUTER_OOF_CLEAR_F1_FP_TIEBREAK')
    write(OUT/'freeze.json', freeze)
    dev = placement(x[nonanchor], y[nonanchor], base_oof[nonanchor], local_pairs(pairs, nonanchor), local_pairs(inv, nonanchor))
    oof = {arm: np.full(1344, np.nan) for arm in arms}
    fold_receipts = []
    for k in range(6):
        tick = time.perf_counter()
        fi = np.flatnonzero((cohort != 'anchor') & (scene != k))
        ri = np.flatnonzero((cohort != 'anchor') & (scene == k))
        fit_all = np.r_[anchor, fi]
        assert len(fi) == 960 and len(ri) == 192
        assert not {meta[i]['group'] for i in fit_all} & {meta[i]['group'] for i in ri}
        inner = np.full(1344, np.nan, np.float32)
        inner_receipts = []
        for j in range(6):
            if j == k:
                continue
            inner_fit = np.r_[anchor, np.flatnonzero((cohort != 'anchor') & (scene != k) & (scene != j))]
            inner_report = np.flatnonzero((cohort != 'anchor') & (scene == j))
            assert not set(inner_fit) & set(np.r_[ri, inner_report])
            hgb = HistGradientBoostingClassifier(**PARAMS).fit(x[inner_fit], y[inner_fit])
            inner[inner_report] = log_odds(hgb.predict_proba(x[inner_report])[:, 1])
            model_path = OUT/f'outer{k}-inner{j}.pkl'
            model_path.write_bytes(pickle.dumps(hgb))
            inner_receipts.append(dict(fit=inner_fit.tolist(), report=inner_report.tolist(), model_sha256=sha(model_path)))
        assert np.isfinite(inner[fi]).all()
        np.savez_compressed(OUT/f'outer{k}-inner-scores.npz', indices=fi, scores=inner[fi])
        # Saved baseline used float64 cache; use the exact original bytes for parity.
        outer = pickle.loads((DATA/f'fold{k}.pkl').read_bytes())
        exact_score = outer.predict_proba(z['base'][ri])[:, 1]
        np.testing.assert_array_equal(exact_score, saved['scores'][np.isin(nonanchor, ri)])
        for arm in arms:
            model = fit_head(x[fi], y[fi], inner[fi], local_pairs(pairs, fi), local_pairs(inv, fi), arm, dev, OUT/f'{arm}-fold{k}.pt')
            oof[arm][ri] = predict(model, x[ri], log_odds(exact_score))
        fold_receipts.append(dict(fold=k, residual_fit=fi.tolist(), report=ri.tolist(), inner=inner_receipts, seconds=time.perf_counter()-tick))
        write(OUT/'fold-receipts.json', fold_receipts)
        print(f'outer fold {k} complete in {time.perf_counter()-tick:.2f}s', flush=True)
    clear = np.array([meta[i]['stratum'] != 'boundary' for i in nonanchor])
    tau, selections, final = {}, {}, {}
    for arm in arms:
        chosen, curve = select(oof[arm][nonanchor], y[nonanchor].astype(bool), clear)
        tau[arm] = chosen['threshold']
        selections[arm] = dict(chosen=chosen, curve=curve)
        final[arm] = fit_head(x[nonanchor], y[nonanchor], base_oof[nonanchor], local_pairs(pairs, nonanchor),
            local_pairs(inv, nonanchor), arm, dev, OUT/f'{arm}-final.pt')
    write(OUT/'selection.json', selections)
    np.savez_compressed(OUT/'oof-scores.npz', indices=nonanchor, **{a: v[nonanchor] for a, v in oof.items()})
    config = dict(thresholds=tau, baseline_threshold=read(DATA/'threshold.json')['threshold'],
        models={a: sha(OUT/f'{a}-final.pt') for a in arms}, baseline_sha256=sha(DATA/'retrainedA.pkl'),
        selection_sha256=sha(OUT/'selection.json'))
    write(OUT/'model-seal.json', config)
    # Report prediction uses only public rows and images. Labels joined below.
    cap = SOURCE/'source/returned-v1/capture-v1'
    receipt = read(cap/'receipt.json')
    bind(cap/'spec.json', receipt['spec_sha256'])
    bind(cap/'raw.jsonl', receipt['hashes']['raw.jsonl'])
    raw = [json.loads(s) for s in (cap/'raw.jsonl').read_text().splitlines()]
    xx, times = [], []
    yaw, episode = 0., None
    for r in raw:
        if r['episode_id'] != episode:
            yaw = 0.
        if r['imu_valid']:
            yaw += r['delta_yaw']
        episode = r['episode_id']
        bind(cap/r['rgb_path'], receipt['hashes'][r['rgb_path']])
        tick = time.perf_counter()
        f = extract(r, cv2.imread(str(cap/r['rgb_path'])), yaw)
        xx.append(np.r_[f['sensor'], f['geometry']])
        times.append(time.perf_counter()-tick)
    xx = np.stack(xx)
    baseline = pickle.loads((DATA/'retrainedA.pkl').read_bytes())
    prob = baseline.predict_proba(xx)[:, 1]
    b = log_odds(prob)
    report_scores = {'A_star': b.astype(np.float64)}
    flags = {'A_star': prob >= config['baseline_threshold']}
    for arm in arms:
        report_scores[arm] = predict(final[arm], xx.astype(np.float32), b)
        flags[arm] = report_scores[arm] >= tau[arm]
    np.savez_compressed(OUT/'report-features.npz', base=xx, ids=[r['id'] for r in raw])
    predictions = [dict(id=r['id'], scores={a: float(s[i]) for a, s in report_scores.items()},
                        flags={a: bool(p[i]) for a, p in flags.items()}, baseline_probability=float(prob[i])) for i, r in enumerate(raw)]
    write(OUT/'predictions.json', predictions)
    write(OUT/'prediction-seal.json', dict(model_seal_sha256=sha(OUT/'model-seal.json'),
        predictions_sha256=sha(OUT/'predictions.json'), feature_sha256=sha(OUT/'report-features.npz'), inputs=bindings,
        labels_joined=False, authority='CONSUMED_REPORT_PUBLIC_PREDICTIONS_SEALED_BEFORE_LABEL_JOIN'))
    # Native report truth already exists; do not call these fresh unseen outcomes.
    cases = read(SOURCE/'confirmation/cases.json')
    prior = read(SOURCE/'confirmation/predictions.json')
    bind(SOURCE/'confirmation/predictions.json', read(SOURCE/'confirmation/prediction-seal.json')['predictions_sha256'])
    assert [r['id'] for r in raw] == [r['id'] for r in cases] == [r['id'] for r in prior]
    np.testing.assert_array_equal(prob, [r['control_score'] for r in prior])
    report_frames = {f['id']: f for f in read(cap/'spec.json')['frames']}
    rp, rv, rej = pairing(cases, report_frames)
    methods = report(cases, report_scores, flags, rp, flags['A_star'])
    base = methods['A_star']
    decisions = {}
    for arm in arms:
        m = methods[arm]
        no_delay = all(c['baseline'] is None or (c['current'] is not None and c['current'] <= c['baseline'])
                       for kind in ('core_changes', 'strict_events_changes') for c in m[kind])
        passes = (m['metrics']['clear']['f1'] > base['metrics']['clear']['f1'] and
            m['metrics']['clear']['recall'] > base['metrics']['clear']['recall'] and
            m['metrics']['clear']['FP'] <= base['metrics']['clear']['FP'] and
            m['changes']['clear']['lost_TP'] == 0 and no_delay)
        decisions[arm] = dict(keep_challenger=passes, no_lost_or_delayed_baseline_event=no_delay)
    decisions['objective_gain_over_bce'] = bool(decisions['rank']['keep_challenger'] and
        methods['rank']['metrics']['clear']['f1'] > methods['bce']['metrics']['clear']['f1'])
    oof_ref = saved['scores'] >= config['baseline_threshold']
    oof_ss = {'A_star': base_oof[nonanchor].astype(np.float64), **{a: s[nonanchor] for a, s in oof.items()}}
    oof_flags = {'A_star': oof_ref, **{a: s[nonanchor] >= tau[a] for a, s in oof.items()}}
    oof_summary = report([meta[i] for i in nonanchor], oof_ss, oof_flags, local_pairs(pairs, nonanchor), oof_ref)
    summary = dict(authority='CONSUMED_CONTROLLED_DEVELOPMENT_NO_NEW_CONFIRMATION', methods=methods,
        decision=decisions, oof_selection=oof_summary, invariance_pairs=len(inv), report_pair_count=len(rp),
        report_rejected_pairs=rej, rank_pairs=len(pairs), backend=dev,
        feature_extract_seconds=float(sum(times)), total_seconds=time.perf_counter()-started,
        inference_cost='Feature extraction measured; no end-to-end or phone latency claim')
    write(OUT/'summary.json', summary)
    write(OUT/'cases.json', [dict(**m, ccrl=p) for m, p in zip(cases, predictions)])
    assert all(sha(p) == h for p, h in bindings.items())
    write(OUT/'completion.json', dict(status='PASS', summary_sha256=sha(OUT/'summary.json'),
        prediction_seal_sha256=sha(OUT/'prediction-seal.json'), inputs_unchanged=True,
        resources='No persistent services, process exits and releases model memory'))
    print(json.dumps(dict(metrics={a: m['metrics']['clear'] for a, m in methods.items()}, decision=decisions)), flush=True)


if __name__ == '__main__':
    with threadpool_limits(4):
        main()
