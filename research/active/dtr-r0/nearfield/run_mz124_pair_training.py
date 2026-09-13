"""Two fixed matched fits on CONSUMED MZ123; no confirmation or promotion.

The only candidate change is a changed-cell paired logit ranking objective.
Pair/family identities schedule training and scoring; encode remains observation-only.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from mz120_occupancy import OccupancyNet, CELLS, CORE, HEAD, encode
from mz121_joint_readout import select_joint, precision, recall, joint_checks
from run_mz120_occupancy import batch, predict, evaluate, labels, read_evaluation
from run_mz107_four_sensor import sha, write, readrows
from run_mz111_spatial_temporal import event_metrics
from research_backend import BackendCandidate, DeviceObservation, select_backend

SEED = 124013
STEPS = 800
RANK_WEIGHT = .25
MARGIN = 1.
AUTHORITY = 'CONSUMED_MZ123_PAIRED_DEVELOPMENT_NO_FRESH_CONFIRMATION'


def pair_partition(rows, spec):
    """Pair-disjoint fixed k0/k1 train, k2 dev; aligned members stay together."""
    index = {}
    for i, row in enumerate(rows): index.setdefault(row['episode_id'], []).append(i)
    train, dev, pairing = [], [], []
    for pair in spec['pairs']:
        a, b = [index[ep] for ep in pair['episodes']]
        assert len(a) == len(b) == 12
        assert [rows[i]['time_s'] for i in a] == [rows[i]['time_s'] for i in b]
        isdev = pair['pair_id'].endswith('pair2')
        assert isdev or pair['pair_id'].endswith(('pair0', 'pair1'))
        (dev if isdev else train).extend(a+b)
        pairing.extend(dict(a=ai, b=bi, pair_id=pair['pair_id'], family=pair['category'],
                            split='dev' if isdev else 'train') for ai, bi in zip(a, b))
    train, dev = np.array(sorted(train)), np.array(sorted(dev))
    assert len(train) == 192 and len(dev) == 96 and set(train).isdisjoint(dev)
    assert set(train) | set(dev) == set(range(len(rows)))
    return train, dev, pairing


def paired_schedule(pairing):
    pairs = np.array([[r['a'], r['b']] for r in pairing if r['split'] == 'train'])
    rng = np.random.default_rng(SEED)
    chosen = rng.integers(0, len(pairs), size=(STEPS, 4))
    indices = pairs[chosen].reshape(STEPS, 8)
    aug_rng = np.random.default_rng(SEED+1)
    # Equal contrast/offset for both members, identical sequence in both arms.
    augmentation = np.repeat(aug_rng.random((STEPS, 2, 4, 1, 1, 1), dtype=np.float32), 2, axis=2)
    return indices, augmentation


def changed_cell_rank(logits, targets):
    """Positive-labelled member must exceed negative by the single fixed margin."""
    a, b = logits[0::2], logits[1::2]
    ya, yb = targets[0::2], targets[1::2]
    changed = ya != yb
    signed_difference = (a-b)*(ya-yb)
    losses = F.relu(MARGIN-signed_difference)
    return (losses*changed).sum()/changed.sum().clamp_min(1)


def tensor_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def score(rows, es, target, probability, threshold, baseline, pairing, indices):
    ix = np.asarray(indices)
    result = dict(threshold=float(threshold), **evaluate([rows[i] for i in ix], [es[i] for i in ix],
        target[ix], probability[ix], float(threshold), baseline[ix]))
    m = result['metrics']; result['frame_precision'] = precision(m); result['frame_recall'] = recall(m)
    result['frame_f1'] = 2*m['TP']/max(1, 2*m['TP']+m['FP']+m['FN'])
    result['no_alert_frames'] = m['TN']+m['FN']
    result['no_alert_rate'] = result['no_alert_frames']/len(ix)
    result['unknown_semantics'] = 'Inherited UNKNOWN equals no-alert; no separate calibrated abstention state; never certified clear'
    head_ix = np.array([i for i in ix if es[i]['family'] == 'suspended_head'])
    gt = target[:, CORE].any(1); pred = probability[:, CORE].max(1) >= threshold
    if len(head_ix):
        result['suspended_HEAD'] = dict(metrics={k: int(v) for k, v in dict(
            TP=(gt[head_ix]&pred[head_ix]).sum(), FP=(~gt[head_ix]&pred[head_ix]).sum(),
            FN=(gt[head_ix]&~pred[head_ix]).sum(), TN=(~gt[head_ix]&~pred[head_ix]).sum()).items()},
            events=event_metrics([rows[i] for i in head_ix], gt[head_ix], pred[head_ix]))
        result['suspended_HEAD']['recall'] = recall(result['suspended_HEAD']['metrics'])
    included = set(ix.tolist()); p = probability >= threshold
    pairs = [r for r in pairing if r['a'] in included and r['b'] in included]
    both = changed_both = changed_total = 0
    for pair in pairs:
        a, b = pair['a'], pair['b']; changed = target[a] != target[b]
        both += int(pred[a] == gt[a] and pred[b] == gt[b])
        changed_total += int(changed.sum())
        changed_both += int((changed & (p[a] == target[a]) & (p[b] == target[b])).sum())
    result['paired'] = dict(frame_pairs=len(pairs), both_alert_answers_correct=both,
        both_alert_answers_correct_rate=both/max(1, len(pairs)), changed_cells=changed_total,
        both_changed_cell_answers_correct=changed_both)
    return result


def run(capture, baseline_dir, initial, output):
    started = time.perf_counter(); out = output.resolve(); cap = capture.resolve()
    art = Path(os.environ['BLINDASSIST_ARTIFACTS']).resolve()
    assert out.is_relative_to(art) and not out.exists()
    assert torch.cuda.is_available(); torch.set_num_threads(4)
    out.mkdir(parents=True)
    receipt = json.loads((cap/'receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['frames'] == 288
    assert sha(cap/'spec.json') == receipt['spec_sha256']
    for name, digest in receipt['hashes'].items(): assert sha(cap/name) == digest, name
    sealed = json.loads((baseline_dir/'prediction-seal.json').read_text())
    assert sealed['raw_sha256'] == sha(cap/'raw.jsonl')
    assert sealed['baseline_sha256'] == sha(baseline_dir/'baseline-predictions.json')
    rows = readrows(cap/'raw.jsonl'); spec = json.loads((cap/'spec.json').read_text())
    es = read_evaluation(cap)
    assert [r['id'] for r in rows] == [e['id'] for e in es]
    assert not any(any(k in r for k in ('native_bounds','truth','depth','actor')) for r in rows)
    tr, dv, pairing = pair_partition(rows, spec)
    y = np.stack([labels(e) for e in es])
    baseline = np.array([p['candidate'] for p in json.loads((baseline_dir/'baseline-predictions.json').read_text())], bool)
    encoded, rgbs, rgb_hashes = [], [], {}; episode = None; yaw = 0.
    for row in rows:
        if row['episode_id'] != episode: yaw = 0.
        if row['imu_valid']: yaw += row['delta_yaw']
        episode = row['episode_id']; encoded.append(encode(row, yaw))
        path = (cap/row['rgb_path']).resolve(); assert path.is_relative_to(cap)
        rgb_hashes[row['id']] = sha(path)
        im = cv2.imread(str(path)); assert im is not None and im.shape == (360, 640, 3)
        rgbs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    arrays = {k: np.stack([r[k] for r in encoded]) for k in encoded[0]}
    arrays['rgb'] = np.stack(rgbs).transpose(0, 3, 1, 2)
    del rgbs, encoded
    data = {k: torch.from_numpy(v) for k,v in arrays.items()}
    schedule, augmentation = paired_schedule(pairing)
    np.savez_compressed(out/'schedule.npz', indices=schedule, augmentation=augmentation)
    write(out/'split.json', dict(train=[rows[i]['id'] for i in tr], dev=[rows[i]['id'] for i in dv],
        pairing=pairing, authority=AUTHORITY, dev_note='Pair2 is untrained in these two fits but already consumed in MZ123 outcome judgment'))
    write(out/'rgb-hashes.json', rgb_hashes)
    torch.manual_seed(SEED)
    initial_state = torch.load(initial, map_location='cpu', weights_only=True)
    initial_digest = tensor_digest(initial_state)
    positive_weight = float(np.clip((~y[tr]).sum()/max(1, y[tr].sum()), 1, 12))
    write(out/'freeze.json', dict(authority=AUTHORITY, seed=SEED, steps=STEPS, batch=8, whole_pairs_per_batch=4,
        initial_sha256=sha(initial), initial_tensor_digest=initial_digest, schedule_sha256=sha(out/'schedule.npz'),
        split_sha256=sha(out/'split.json'), rgb_hashes_sha256=sha(out/'rgb-hashes.json'),
        inputs={n:sha(cap/n) for n in ('raw.jsonl','evaluator.jsonl','spec.json','receipt.json')},
        baseline_sha256=sealed['baseline_sha256'], code_sha256=sha(Path(__file__)),
        optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, positive_weight=positive_weight),
        candidate=dict(ranking_weight=RANK_WEIGHT, changed_cell_logit_margin=MARGIN),
        output_labels='Native occupied object volumes, not sensor echo identity',
        stop='Exactly two 800-step fits; max 1200 training seconds per fit; no sweeps/restarts; no fresh acquisition'))
    target = torch.from_numpy(y.astype(np.float32)).cuda()
    weight = torch.tensor(positive_weight, device='cuda')
    base_dev = evaluate([rows[i] for i in dv], [es[i] for i in dv], y[dv],
        np.repeat(baseline[dv,None], len(CELLS), 1).astype(float), .5, baseline[dv]); base_dev.pop('spatial')
    results = {}
    for arm, coefficient in [('bce', 0.), ('bce_pair_rank', RANK_WEIGHT)]:
        directory = out/arm; directory.mkdir()
        torch.manual_seed(SEED); model = OccupancyNet(); model.load_state_dict(initial_state)
        assert tensor_digest(model.state_dict()) == initial_digest
        model = model.cuda().eval(); probe = batch(data, tr[:8])
        cpu_model = copy.deepcopy(model).cpu(); cpu_probe = {k:v.cpu() for k,v in probe.items()}
        def gpu_run():
            with torch.no_grad(): return model(probe)
        def cpu_run():
            with torch.no_grad(): return cpu_model(cpu_probe)
        backend = select_backend('model-inference', cpu=BackendCandidate('torch-cpu','cpu',cpu_run,
            lambda v: DeviceObservation(v.device.type,'host CPU','torch '+torch.__version__)),
            gpu=BackendCandidate('torch-cuda','cuda',gpu_run,
                lambda v: DeviceObservation(v.device.type,torch.cuda.get_device_name(),'torch '+torch.__version__),torch.cuda.synchronize),
            record_path=directory/'backend.json')
        assert backend['selected_device_type'] == 'cuda'
        del cpu_model, cpu_probe, probe
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        model.train(); progress = []; tick = time.perf_counter()
        for step, idx in enumerate(schedule):
            assert time.perf_counter()-tick < 1200, '20-minute fixed-fit training cap; no partial-model conclusion'
            b = batch(data, idx)
            contrast = torch.from_numpy(.75+.5*augmentation[step,0]).cuda()
            offset = torch.from_numpy(-.1+.2*augmentation[step,1]).cuda()
            b['rgb'] = ((b['rgb']-.5)*contrast+.5+offset).clamp(0,1)
            optimizer.zero_grad(set_to_none=True); logits = model(b)
            bce = F.binary_cross_entropy_with_logits(logits, target[idx], pos_weight=weight)
            ranking = changed_cell_rank(logits, target[idx]); loss = bce+coefficient*ranking
            loss.backward(); optimizer.step()
            if (step+1)%100 == 0:
                torch.cuda.synchronize()
                record = dict(step=step+1, total=STEPS, seconds=time.perf_counter()-tick,
                    bce=float(bce.detach()), rank=float(ranking.detach()), loss=float(loss.detach()))
                progress.append(record); write(directory/'progress.json',progress)
                write(out/'progress.json',dict(arm=arm, **record))
                print(json.dumps(dict(arm=arm, **record)), flush=True)
        torch.cuda.synchronize(); training_seconds = time.perf_counter()-tick
        torch.save(model.state_dict(), directory/'model.pt')
        tick = time.perf_counter(); probabilities = predict(model, data, np.arange(len(rows)))
        torch.cuda.synchronize(); inference_seconds = time.perf_counter()-tick
        np.save(directory/'probabilities.npy', probabilities)
        write(directory/'prediction-seal.json', dict(authority=AUTHORITY, model_sha256=sha(directory/'model.pt'),
            probability_sha256=sha(directory/'probabilities.npy'), freeze_sha256=sha(out/'freeze.json'),
            inference_inputs='Observation encode and RGB only; labels used for training and scoring, never forward input'))
        curves = {}
        for split_name, indices in [('train', tr), ('dev', dv)]:
            curves[split_name] = [score(rows,es,y,probabilities,t,baseline,pairing,indices) for t in np.linspace(0,1,101)]
            write(directory/(split_name+'-curve.json'),curves[split_name])
        selected, feasible = select_joint(curves['dev'], base_dev)
        fixed = {name:score(rows,es,y,probabilities,.58,baseline,pairing,indices) for name,indices in [('train',tr),('dev',dv)]}
        results[arm] = dict(fixed_058=fixed, selected_dev=selected,
            selected_train=None if selected is None else score(rows,es,y,probabilities,selected['threshold'],baseline,pairing,tr),
            feasible_thresholds=[p['threshold'] for p in feasible],
            fixed_dev_joint_checks=joint_checks(fixed['dev'],base_dev), training_seconds=training_seconds,
            inference_seconds=inference_seconds, parameters=sum(p.numel() for p in model.parameters()),
            final_model_sha256=sha(directory/'model.pt'))
        del model, optimizer; torch.cuda.empty_cache()
    np.savez_compressed(out/'evaluation-targets.npz', labels=y, baseline=baseline, train=tr, dev=dv)
    write(out/'summary.json', dict(status='TWO_MATCHED_FIXED_FITS_COMPLETE', authority=AUTHORITY,
        baseline_dev=base_dev, arms=results, frames=dict(train=len(tr),dev=len(dv)),
        elapsed_seconds=time.perf_counter()-started, baseline_policy='MZ116_REMAINS_BASELINE',
        limitations='One seed; related controlled simulation; all 288 frames consumed, including dev pair2; no echo identity, hardware, natural-use or fresh confirmation evidence'))
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        freeze_sha256=sha(out/'freeze.json'), training_runs=2, completed_steps_each=STEPS,
        resources='This owned foreground CUDA process; tensors released on exit; runner must separately verify process and task release'))
    print(json.dumps(dict(status='COMPLETE',elapsed_seconds=time.perf_counter()-started)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('capture','baseline-dir','initial','output'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    try: run(args.capture,args.baseline_dir,args.initial,args.output)
    except BaseException as exc:
        if args.output.is_dir(): write(args.output/'failure.json',dict(status='FAIL',error=repr(exc),no_restart=True))
        raise
