"""Train once on disclosed consumed scenes; freeze before transfer evaluation."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from mz120_occupancy import CELLS, CORE, HEAD, OccupancyNet, encode
from run_mz107_four_sensor import ROOT, readrows, sha, write, metrics, truth
from run_mz111_spatial_temporal import event_metrics
from research_backend import BackendCandidate, DeviceObservation, select_backend

SOURCES = {'mz115': 'mz115-zonal-allocation-20260913',
           'mz117': 'mz117-surface-mixtures-20260913',
           'mz119': 'mz119-tof-scaled-parallax-20260913'}
SEED = 120013


def labels(e):
    result = np.zeros(len(CELLS), bool)
    for obj in e['native_bounds']:
        lo = np.array(obj['center_m'])-obj['extent_m']-np.array(e['body_origin_m'])
        hi = np.array(obj['center_m'])+obj['extent_m']-np.array(e['body_origin_m'])
        result |= ((hi >= CELLS[:, :3]) & (lo <= CELLS[:, 3:])).all(1)
    assert bool(result[CORE].any()) == bool(truth(e))
    return result


def read_evaluation(capture):
    es = readrows(capture/'evaluator.jsonl')
    spec = json.loads((capture/'spec.json').read_text())
    roles = {f['id']: {o['name']:o['source_role'] for o in f['objects']} for f in spec['frames']}
    for e in es:
        for obj in e['native_bounds']: obj['evaluation_role'] = roles[e['id']][obj['name']]
    return es


def load_source(name, *, evaluation):
    base = ROOT/'artifacts.local/work'; capture = base/SOURCES[name]/'capture-v1'
    receipt = json.loads((capture/'receipt.json').read_text())
    assert receipt['status'] == 'PASS'
    for path, digest in receipt['hashes'].items():
        assert sha(capture/path) == digest, path
    rows = readrows(capture/'raw.jsonl')
    if name == 'mz115':
        study = base/'mz116-merged-information-20260913/guard-v1'
        predpath = study/'consumed_mz115/predictions.json'
        seal = json.loads((study/'prediction-seal.json').read_text())['panels']['consumed_mz115']
        arm = 'guard'
    else:
        study = base/SOURCES[name]/'analysis-v1'; predpath = study/'predictions.json'
        seal = json.loads((study/'prediction-seal.json').read_text()); arm = 'resolution_guard'
    assert sha(predpath) == seal['predictions_sha256']
    assert sha(capture/'raw.jsonl') == seal['raw_sha256']
    assert sha(capture/'receipt.json') == seal['receipt_sha256']
    baseline = json.loads(predpath.read_text())['3.6'][arm]
    # Integrate only observable IMU deltas, reset at complete episode boundaries.
    inputs = []; images = []; yaw = 0.; episode = None
    for r in rows:
        if r['episode_id'] != episode: yaw = 0.
        if r['imu_valid']: yaw += r['delta_yaw']
        episode = r['episode_id']; inputs.append(encode(r, yaw))
        path = (capture/r['rgb_path']).resolve(); assert path.is_relative_to(capture.resolve())
        im = cv2.imread(str(path)); assert im is not None and im.shape == (360, 640, 3)
        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    arrays = {k: np.stack([v[k] for v in inputs]) for k in inputs[0]}
    arrays['rgb'] = np.stack(images).transpose(0, 3, 1, 2)
    es = read_evaluation(capture) if evaluation else None
    if es is not None: assert [r['id'] for r in rows] == [e['id'] for e in es]
    return dict(name=name, capture=capture, rows=rows, es=es, inputs=arrays,
                baseline=np.array([p['candidate'] for p in baseline], bool),
                hashes=dict(raw=sha(capture/'raw.jsonl'), receipt=sha(capture/'receipt.json'), baseline=sha(predpath)))


def split(sources):
    train = []; dev = []; offset = 0
    for source in sources:
        groups = defaultdict(set)
        for r, e in zip(source['rows'], source['es']): groups[e['family']].add(r['episode_id'])
        held = {sorted(eps)[0 if source['name']=='mz117' and family=='clear_ghost' else -1]
                for family, eps in groups.items() if len(eps) >= 2}
        for i, row in enumerate(source['rows']):
            (dev if row['episode_id'] in held else train).append(offset+i)
        offset += len(source['rows'])
    return np.array(train), np.array(dev)


def batch(data, indices, augment=False):
    value = {k: v[indices].cuda() for k, v in data.items()}
    value['rgb'] = value['rgb'].float()/255
    if augment:
        contrast = .75+.5*torch.rand(len(indices), 1, 1, 1, device='cuda')
        offset = -.1+.2*torch.rand(len(indices), 1, 1, 1, device='cuda')
        value['rgb'] = ((value['rgb']-.5)*contrast+.5+offset).clamp(0, 1)
    return value


def predict(model, data, indices, control=None):
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(indices), 8):
            idx = indices[start:start+8]; b = batch(data, idx)
            if control == 'rgb_off': b['rgb'].zero_()
            if control == 'sensor_shuffle':
                # Fixed cross-frame permutation, including packet flags; RGB/grid unchanged.
                other = batch(data, np.roll(indices, len(indices)//2)[start:start+8])
                for k in ('tof', 'radar'): b[k] = other[k]
            out.append(model(b).sigmoid().cpu().numpy())
    return np.concatenate(out)


def evaluate(rows, es, target, probability, threshold, baseline):
    gt = target[:, CORE].any(1); pred = probability[:, CORE].max(1) >= threshold
    result = dict(metrics=metrics(gt, pred), events=event_metrics(rows, gt, pred),
                  lost_baseline_TP=int((gt & baseline & ~pred).sum()),
                  removed_FP=int((~gt & baseline & ~pred).sum()), added_FP=int((~gt & ~baseline & pred).sum()))
    result['families'] = {}
    for family in sorted({e['family'] for e in es}):
        mask = np.array([e['family'] == family for e in es])
        result['families'][family] = metrics(gt[mask], pred[mask])
    p = probability >= threshold
    result['spatial'] = dict(metrics=metrics(target.ravel(), p.ravel()),
                            iou=float((target&p).sum()/max(1, (target|p).sum())))
    for name, mask in [('HEAD', HEAD), ('near_body', CORE), ('outside_or_far', ~CORE)]:
        result['spatial'][name] = metrics(target[:, mask].ravel(), p[:, mask].ravel())
    rod_target = np.stack([labels(dict(e, native_bounds=[o for o in e['native_bounds']
        if 'rod' in o.get('evaluation_role','') or 'pole' in o.get('evaluation_role','')])) for e in es])
    hits = int((rod_target[:, CORE] & p[:, CORE]).sum()); total = int(rod_target[:, CORE].sum())
    result['spatial']['rod_true_cells'] = dict(TP=hits, FN=total-hits,
        authority='EVALUATOR_ACTOR_ROLE_AND_NATIVE_BOUND_OCCUPIED_CELLS')
    result['event_hits'] = {}; previous = None; active = None; number = 0
    for i, row in enumerate(rows):
        if row['episode_id'] != previous: active = None; number = 0
        previous = row['episode_id']
        if gt[i] and active is None:
            active = row['episode_id']+'/'+str(number); number += 1
            result['event_hits'][active] = False
        if gt[i] and pred[i]: result['event_hits'][active] = True
        if not gt[i]: active = None
    return result


def recall(m): return m['TP']/max(1, m['TP']+m['FN'])


def run(output):
    start = time.perf_counter(); out = output.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    assert torch.cuda.is_available(), 'GPU backend required for this bounded training recipe'
    out.mkdir(parents=True); torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
    sources = [load_source(n, evaluation=True) for n in ('mz115', 'mz117')]
    rows = sum([s['rows'] for s in sources], []); es = sum([s['es'] for s in sources], [])
    target = np.stack([labels(e) for e in es]); baseline = np.concatenate([s['baseline'] for s in sources])
    data = {k: torch.from_numpy(np.concatenate([s['inputs'][k] for s in sources])) for k in sources[0]['inputs']}
    tr, dv = split(sources)
    assert not ({rows[i]['episode_id'] for i in tr} & {rows[i]['episode_id'] for i in dv})
    write(out/'split.json', {n: [rows[i]['id'] for i in idx] for n, idx in [('train', tr), ('dev', dv)]})
    code = ('mz120_occupancy.py', 'run_mz120_occupancy.py', 'MZ120_PROTOCOL_20260913.md')
    for name in code: shutil.copyfile(Path(__file__).with_name(name), out/name)
    write(out/'freeze.json', dict(seed=SEED, steps=800, batch=8, source={s['name']:s['hashes'] for s in sources},
        code={n:sha(out/n) for n in code}, split_sha256=sha(out/'split.json'), cells=CELLS.tolist(),
        authority='CONSUMED_SCENE_DISJOINT_DEVELOPMENT_NO_FRESH_CONFIRMATION'))
    model = OccupancyNet().cuda(); probe = batch(data, tr[:8])
    model.eval()
    cpu_model = copy.deepcopy(model).cpu(); cpu_probe = {k:v.cpu() for k,v in probe.items()}
    def probe_run():
        with torch.no_grad(): return model(probe)
    def cpu_run():
        with torch.no_grad(): return cpu_model(cpu_probe)
    placement = select_backend('model-inference', cpu=BackendCandidate('torch-cpu', 'cpu', cpu_run,
        lambda value: DeviceObservation(value.device.type, 'host CPU', 'torch '+torch.__version__)),
        gpu=BackendCandidate('torch-cuda', 'cuda', probe_run,
        lambda value: DeviceObservation(value.device.type, torch.cuda.get_device_name(), 'torch '+torch.__version__),
        torch.cuda.synchronize), record_path=out/'backend.json')
    assert placement['selected_device_type'] == 'cuda', 'GPU placement not established; revise execution before training'
    del cpu_model, cpu_probe
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    y = torch.from_numpy(target.astype(np.float32)).cuda()
    # One global positive weight avoids per-cell label frequencies as cell-specific priors.
    positive_weight = torch.tensor(float(np.clip((~target[tr]).sum()/max(1,target[tr].sum()), 1, 12)), device='cuda')
    rng = np.random.default_rng(SEED); losses = []; tick = time.perf_counter(); model.train()
    for step in range(800):
        assert time.perf_counter()-tick < 1200, 'Training wall budget exceeded; no partial-model conclusion'
        idx = rng.choice(tr, 8, replace=True); b = batch(data, idx, augment=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(b), y[idx], pos_weight=positive_weight)
        loss.backward(); optimizer.step()
        if (step+1) % 100 == 0:
            torch.cuda.synchronize(); losses.append(dict(step=step+1, loss=float(loss.detach()), seconds=time.perf_counter()-tick))
            write(out/'progress.json', dict(training=losses)); print(json.dumps(losses[-1]), flush=True)
    torch.cuda.synchronize(); training_seconds = time.perf_counter()-tick
    torch.save(model.state_dict(), out/'model.pt')
    prob = predict(model, data, np.arange(len(rows)))
    np.savez_compressed(out/'development_predictions.npz', probabilities=prob, labels=target)
    devrows = [rows[i] for i in dv]; deves = [es[i] for i in dv]
    baseprob = np.repeat(baseline[dv, None], len(CELLS), 1).astype(float)
    baseresult = evaluate(devrows, deves, target[dv], baseprob, .5, baseline[dv])
    # Baseline has no occupancy logits: discard invented cell scores from alert-only helper.
    baseresult.pop('spatial')
    curve = []
    for threshold in np.linspace(0, 1, 101):
        r = evaluate(devrows, deves, target[dv], prob[dv], float(threshold), baseline[dv])
        curve.append(dict(threshold=float(threshold), **r))
    qualified = [r for r in curve if recall(r['metrics']) >= recall(baseresult['metrics'])-.02
                 and all(not hit or r['event_hits'].get(ep, False) for ep, hit in baseresult['event_hits'].items())]
    chosen = min(qualified or curve, key=lambda r: (r['metrics']['FP'] if qualified else -recall(r['metrics']),
                                                   r['metrics']['FP'], -r['threshold']))
    threshold = chosen['threshold']; be = baseresult['events']; ce = chosen['events']
    checks = dict(recall_matched=bool(qualified), fp_reduction_30pct=chosen['metrics']['FP'] <= .7*baseresult['metrics']['FP'],
        false_seconds_reduction_30pct=ce['false_alert_bin_duration_s'] <= .7*be['false_alert_bin_duration_s'],
        event_retention=all(not hit or chosen['event_hits'].get(ep, False) for ep, hit in baseresult['event_hits'].items()),
        delay=(ce['max_detected_delay_s'] is not None and be['max_detected_delay_s'] is not None and
               ce['max_detected_delay_s'] <= be['max_detected_delay_s']+.25),
        head_localization=recall(chosen['spatial']['HEAD']) >= .9,
        rod_localization=recall(chosen['spatial']['rod_true_cells']) >= .9)
    write(out/'threshold-freeze.json', dict(threshold=threshold, model_sha256=sha(out/'model.pt'),
        checks=checks, decision='FRESH_SOURCE_CHECK_REQUIRED' if all(checks.values()) else 'STOP_FIXED_PILOT_RECIPE',
        selection='Development only; final checkpoint; no transfer outcomes accessed'))
    write(out/'development_curve.json', curve)
    controls = {control: evaluate(devrows, deves, target[dv], predict(model, data, dv, control), threshold, baseline[dv])
                for control in ('rgb_off', 'sensor_shuffle')}
    trainresult = evaluate([rows[i] for i in tr], [es[i] for i in tr], target[tr], prob[tr], threshold, baseline[tr])
    # Independent source episodes, but consumed related scene templates; no fresh claim.
    transfer = load_source('mz119', evaluation=False)
    td = {k:torch.from_numpy(v) for k,v in transfer['inputs'].items()}
    tick = time.perf_counter(); tp = predict(model, td, np.arange(len(transfer['rows'])))
    torch.cuda.synchronize(); transfer_seconds = time.perf_counter()-tick
    np.save(out/'transfer_probabilities.npy', tp)
    write(out/'transfer-prediction-seal.json', dict(probabilities_sha256=sha(out/'transfer_probabilities.npy'),
        threshold_sha256=sha(out/'threshold-freeze.json'), source=transfer['hashes'],
        status='PREDICTIONS_SEALED_BEFORE_THIS_EVALUATOR_PARSE_CONSUMED_SOURCE'))
    te = read_evaluation(transfer['capture'])
    assert [r['id'] for r in transfer['rows']] == [e['id'] for e in te]
    ty = np.stack([labels(e) for e in te]); tb = transfer['baseline']
    transferresult = evaluate(transfer['rows'], te, ty, tp, threshold, tb)
    transferbase = evaluate(transfer['rows'], te, ty, np.repeat(tb[:,None], len(CELLS), 1).astype(float), .5, tb)
    transferbase.pop('spatial')
    write(out/'transfer_curve.json', [dict(threshold=float(t), **evaluate(transfer['rows'],te,ty,tp,float(t),tb))
                                    for t in np.linspace(0,1,101)])
    result = dict(status='BOUNDED_PILOT_COMPLETE', decision='FRESH_SOURCE_CHECK_REQUIRED' if all(checks.values()) else 'STOP_FIXED_PILOT_RECIPE',
        training_frames=len(tr), development_frames=len(dv), transfer_frames=len(te), threshold=threshold,
        parameters=sum(p.numel() for p in model.parameters()), training_seconds=training_seconds,
        transfer_inference_seconds=transfer_seconds, training_loss=losses, checks=checks,
        train=trainresult, development=dict(baseline=baseresult, candidate=chosen, controls=controls),
        transfer=dict(baseline=transferbase, candidate=transferresult), elapsed_seconds=time.perf_counter()-start,
        limitation='Consumed constructed related templates, small from-scratch training; no hardware or information-limit claim')
    write(out/'summary.json', result)
    write(out/'completion.json', dict(status='PASS', summary_sha256=sha(out/'summary.json'),
        model_sha256=sha(out/'model.pt'), resources_started=['this foreground Python process; GPU tensors released on exit']))
    print(json.dumps({k:result[k] for k in ('decision','threshold','checks','training_seconds')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
