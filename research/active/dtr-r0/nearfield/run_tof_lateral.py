"""One RGB-assisted lateral-attribution mechanism pilot on consumed96.

One simple appearance hypothesis and one tiny head with its matched noRGB
control. The entire96 cohort is consumed; head fitting is explicitly in-sample.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from evaluate_ba_camera_corridor import read, sha, require, write_new
from tof_lateral_core import (CLASSES, OUTSIDE_CUTOFF, eligible, local_input,
    simple_geometry, tiny_head, learned_decision, frame_decision)

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'artifacts.local/work/ba-camera-corridor-20260919'
FOV = ROOT / 'artifacts.local/work/ba-tof-fov45-20260920'
PARENT = ROOT / 'artifacts.local/work/ba-tof-corridor-calibration-20260920'
OUT = ROOT / 'artifacts.local/work/ba-tof-lateral-attribution-20260920'
REPORT = Path(__file__).with_name('TOF_LATERAL_ATTRIBUTION_20260920.md')
CODE = ('tof_lateral_core.py', 'run_tof_lateral.py', 'evaluate_tof_lateral.py',
        'test_tof_lateral_core.py', 'ba_camera_corridor.py', 'ba_camera_corridor_metrics.py')
IDENTITY = ('id', 'clip_id', 'frame_in_clip', 'time_s')
STEPS = 400
SEED = 20260920


def freeze(out):
    require(not out.exists(), 'Output exists; no overwrite or automatic retry')
    inputs = []
    for directory, names in ((PARENT, ('protocol.json', 'scores.json', 'score-seal.json', 'operating-point.json',
        'predictions.json', 'prediction-seal.json', 'frame-results.json', 'results.json', 'evaluator-repair.json')),
        (FOV, ('protocol.json', 'observations.json', 'observation-seal.json', 'predictions.json',
               'prediction-seal.json', 'private-lineage.json', 'frame-results.json')),
        (OLD, ('observations.json', 'observation-seal.json', 'frame-results.json', 'evaluator-source.json'))):
        inputs.extend(directory / n for n in names)
    protocol = dict(id='ba-tof-lateral-attribution-20260920', phase='EXPLORE_CONSUMED96_MECHANISM_FIT',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        question='Can local RGB reject observed-near lateral false votes without losing original TP/onset/native corridor evidence?',
        baseline='Frozen A30TP15FP6FN; original cutoff unchanged',
        eligibility='Original possible, nondefinite,0.1<=value<3, original full-depth X envelope straddles corridor boundary',
        input='3x3-zone native RGB context and5 public geometry channels resampled48x48; no native depth, target identity, frame id or labels as features',
        B='Otsu minority-intensity union in zone; contrast>=12,at least2 pixels;1pixel pad; full original depth interval; only definite lateral OUTSIDE suppresses',
        C=dict(architecture='8->8->16 convolution; fixed6x6 pool to4x4;256->16->3', steps=STEPS, seed=SEED,
               optimizer='AdamW lr0.003 weight_decay0.0001; class-balanced CE; full-batch',
               outside_cutoff=OUTSIDE_CUTOFF, checkpoint='final400 only',
               no_rgb='Same initial weights, same architecture, labels and steps; first3 input channels zero'),
        supervision='Pure target return identified only in supervision; complete authenticated target X extent, exact contact CROSSING; mixed/non-target UNKNOWN',
        split='NONE: consumed in-sample fit only; g1-heldout has zeroOUTSIDE, g2-heldout zeroCROSSING; no random-frame split',
        budget=dict(frames=96, native_captures=0, new_tof=0, geometry_recipes=1, head_architectures=1,
                    fits=2, steps_per_fit=STEPS, large_models=0, cutoff_searches=0),
        goals=['remove4of5 eligible lateralFP as aspirational target', 'retain30TP identities and all event onsets',
               'zero suppressed native corridor contributors including hidden zone losses', 'preserve definite support and UNKNOWN',
               'C RGB increment must beat matched C_no_rgb; fit is not generalization'],
        stop='Stop after B/C/control, independent recount and delivery. No tuning, missing-event rescue, temporal or sensor successor.',
        code_hashes={n: sha(Path(__file__).with_name(n)) for n in CODE},
        input_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        protocol_text_sha256=sha(REPORT), synthetic_tests_passed=6)
    out.mkdir(parents=True)
    (out / 'protocol-before-run.md').write_bytes(REPORT.read_bytes())
    (out / 'current-before-run.md').write_bytes((ROOT / 'research/active/dtr-r0/CURRENT.md').read_bytes())
    write_new(out / 'protocol.json', protocol)
    print(json.dumps(dict(status='FROZEN', protocol_sha256=sha(out / 'protocol.json'))))


def verify(out):
    protocol = read(out / 'protocol.json')
    for n, digest in protocol['code_hashes'].items():
        actual = sha(Path(__file__).with_name(n))
        if actual != digest:
            repair = read(out / 'backend-repair.json')
            record = repair['code_changes'].get(n, {})
            require(n in ('run_tof_lateral.py', 'evaluate_tof_lateral.py')
                    and record.get('original_sha256') == digest
                    and record.get('repaired_sha256') == actual
                    and sha(out / record['before_file']) == digest
                    and repair['protocol_sha256'] == sha(out / 'protocol.json')
                    and repair['public_seal_sha256'] == sha(out / 'public-seal.json')
                    and repair['preparation_seal_sha256'] == sha(out / 'preparation-seal.json')
                    and repair['optimizer_steps_before_repair'] == 0
                    and repair['reason'] == 'BACKEND_SELECTOR_REQUIRES_EQUIVALENT_CPU_PROBE',
                    'Unrecorded frozen code change: ' + n)
    for n, digest in protocol['input_hashes'].items():
        require(sha(ROOT / n) == digest, 'Source differs: ' + n)
    require(sha(out / 'protocol-before-run.md') == protocol['protocol_text_sha256'], 'Protocol text differs')
    return protocol


def prepare(out):
    verify(out)
    require(not (out / 'public-samples.json').exists(), 'Preparation already exists')
    observations, rgbs = read(FOV / 'observations.json'), read(OLD / 'observations.json')
    scores, baseline = read(PARENT / 'scores.json'), read(PARENT / 'predictions.json')
    require(len(observations) == len(rgbs) == len(scores) == len(baseline) == 96, 'Cohort differs')
    samples, tensors, simple, frames = [], [], [], []
    began = time.perf_counter()
    for obs, image, score, base in zip(observations, rgbs, scores, baseline):
        require(all(obs[k] == image[k] == score[k] == base[k] for k in IDENTITY), 'Input alignment differs')
        rgb_path = OLD / image['rgb_path']
        require(sha(rgb_path) == image['rgb_sha256'], 'RGB changed')
        require(sha(FOV / obs['path']) == obs['sha256'] == score['observation_sha256'], 'ToF changed')
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        require(rgb.shape == (360, 640, 3), 'RGB shape differs')
        anchors = {a['zone']: a for a in score['anchors']}
        ids = []
        with np.load(FOV / obs['path'], allow_pickle=False) as data:
            for zone, (box, value) in enumerate(zip(data['boxes'], data['values'])):
                if zone not in anchors or not eligible(box, value, anchors[zone]):
                    continue
                sid = obs['id'] + ':z' + str(zone)
                row = dict(sample_id=sid, frame_id=obs['id'], zone=zone, box=box.tolist(), value=float(value),
                           interval_m=anchors[zone]['interval_m'], rgb_sha256=image['rgb_sha256'],
                           observation_sha256=obs['sha256'])
                samples.append(row)
                tensors.append(local_input(rgb, box, float(value), row['interval_m']))
                simple.append(dict(sample_id=sid, **simple_geometry(rgb, box, row['interval_m'])))
                ids.append(sid)
        frames.append({**{k: obs[k] for k in IDENTITY}, 'sample_ids': ids})
    require(len(samples) == 60 and sum(bool(f['sample_ids']) for f in frames) == 28, 'Pre-audited eligibility differs')
    np.savez_compressed(out / 'public-inputs.npz', x=np.stack(tensors))
    write_new(out / 'public-samples.json', samples)
    write_new(out / 'simple-outputs.json', simple)
    write_new(out / 'frames.json', frames)
    write_new(out / 'public-seal.json', dict(status='COMPLETE', frames=96, samples=len(samples),
        protocol_sha256=sha(out / 'protocol.json'), hashes={n: sha(out / n) for n in
        ('public-inputs.npz', 'public-samples.json', 'simple-outputs.json', 'frames.json')},
        elapsed_s=time.perf_counter() - began,
        rgb_geometry_backend=dict(device='CPU', framework='OpenCV', version=cv2.__version__,
                                  reason='GPU_BACKEND_UNAVAILABLE', scope='Installed CPU crop/Otsu implementation')))
    # Explicit supervised-label materialization AFTER observable inputs/B outputs.
    labels_by_frame = {r['id']: r for r in read(OLD / 'frame-results.json')}
    ownership = {r['id']: r['new_lineage']['zones'] for r in read(FOV / 'frame-results.json')}
    labels = []
    for sample in samples:
        label = labels_by_frame[sample['frame_id']]
        own = ownership[sample['frame_id']][sample['zone']]
        require(own['zone_id'] == sample['zone'], 'Ownership identity differs')
        low, high = label['target_camera_bounds_m']['lower'][0], label['target_camera_bounds_m']['upper'][0]
        relation = ('OUTSIDE' if high < -.3 - 1e-9 or low > .3 + 1e-9 else
                    'INSIDE' if low > -.3 + 1e-9 and high < .3 - 1e-9 else 'CROSSING')
        if own['ownership'] != 'pure':
            relation = 'UNKNOWN'
        labels.append(dict(sample_id=sample['sample_id'], frame_id=sample['frame_id'],
            pair_id=label['pair_id'], relation=relation, label=CLASSES.index(relation) if relation in CLASSES else -1,
            authority='SUPERVISION_ONLY_PURE_TARGET_FULL_EXTENT_NOT_POINT_VETO', full_target_x=[low, high]))
    require(Counter(r['relation'] for r in labels) == dict(INSIDE=35, OUTSIDE=13, CROSSING=12), 'Audited class counts differ')
    write_new(out / 'training-labels.json', labels)
    write_new(out / 'preparation-seal.json', dict(status='COMPLETE', protocol_sha256=sha(out / 'protocol.json'),
        public_seal_sha256=sha(out / 'public-seal.json'), labels_sha256=sha(out / 'training-labels.json'),
        scope='Pure simulated target full-extent supervision; all60 consumed samples, no holdout'))
    print(json.dumps(dict(status='PREPARED', samples=len(samples), frames=28, classes=dict(Counter(r['relation'] for r in labels)))))


def prepared(out):
    verify(out)
    seal = read(out / 'preparation-seal.json')
    require(seal['status'] == 'COMPLETE' and seal['protocol_sha256'] == sha(out / 'protocol.json'), 'Preparation incomplete')
    require(seal['public_seal_sha256'] == sha(out / 'public-seal.json'), 'Public seal differs')
    require(seal['labels_sha256'] == sha(out / 'training-labels.json'), 'Training labels differ')
    for n, digest in read(out / 'public-seal.json')['hashes'].items():
        require(sha(out / n) == digest, 'Public input differs: ' + n)


def train(out):
    prepared(out)
    require(not (out / 'training-start.json').exists(), 'Fit already attempted; no automatic retry')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import torch.nn.functional as functional
    require(torch.cuda.is_available(), 'Declared GPU backend unavailable')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    with np.load(out / 'public-inputs.npz', allow_pickle=False) as data:
        inputs = torch.from_numpy(data['x']).to('cuda')
    labels = torch.tensor([r['label'] for r in read(out / 'training-labels.json')], device='cuda')
    initial = tiny_head().to('cuda')
    initial_state = copy.deepcopy(initial.state_dict())
    parameter_count = sum(p.numel() for p in initial.parameters())
    sys.path.insert(0, str(ROOT / 'tools'))
    from research_backend import BackendCandidate, DeviceObservation, select_backend
    cpu_model = copy.deepcopy(initial).cpu()
    cpu_inputs, cpu_labels = inputs.cpu(), labels.cpu()
    def probe(model, x, y):
        model.zero_grad(set_to_none=True)
        value = functional.cross_entropy(model(x), y)
        value.backward()
        return value
    backend = select_backend('batch-tensor',
        cpu=BackendCandidate('tiny-lateral-cpu', 'cpu', lambda: probe(cpu_model, cpu_inputs, cpu_labels),
            lambda t: DeviceObservation(t.device.type, 'host CPU', 'PyTorch', ('CPU',))),
        gpu=BackendCandidate('tiny-lateral-cuda', 'cuda', lambda: probe(initial, inputs, labels),
        lambda t: DeviceObservation(t.device.type, torch.cuda.get_device_name(0), 'PyTorch', ('CUDA',)),
        torch.cuda.synchronize), capabilities={'scope': 'Equivalent full60x48x48 forward/backward, zero optimizer updates', 'parameters': parameter_count},
        record_path=out / 'backend.json')
    require(backend['selected_device_type'] == 'cuda', 'Declared training placement not supported by measured probe')
    del cpu_model, cpu_inputs, cpu_labels
    torch.save(initial_state, out / 'initialization.pt')
    write_new(out / 'training-start.json', dict(protocol_sha256=sha(out / 'protocol.json'),
        preparation_seal_sha256=sha(out / 'preparation-seal.json'), seed=SEED, steps=STEPS,
        parameters=parameter_count, initialization_sha256=sha(out / 'initialization.pt'),
        device=torch.cuda.get_device_name(0), scope='Two matched in-sample fits; no checkpoint selection'))
    weights = len(labels) / (3 * torch.bincount(labels, minlength=3).float())
    outputs, fits = {}, {}
    for arm in ('C', 'C_no_rgb'):
        model = tiny_head().to('cuda')
        model.load_state_dict(initial_state)
        x = inputs.clone()
        if arm == 'C_no_rgb':
            x[:, :3] = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
        began = time.perf_counter()
        losses = []
        model.train()
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(x), labels, weight=weights)
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                losses.append(dict(step=step + 1, loss=float(loss.detach())))
                print(json.dumps(dict(arm=arm, **losses[-1])), flush=True)
        torch.cuda.synchronize()
        fit_time = time.perf_counter() - began
        model.eval()
        began = time.perf_counter()
        with torch.no_grad():
            probabilities = model(x).softmax(-1)
        torch.cuda.synchronize()
        infer_time = time.perf_counter() - began
        result = probabilities.cpu().numpy()
        require(np.isfinite(result).all() and np.allclose(result.sum(1), 1), 'Invalid classifier scores')
        outputs[arm] = result.tolist()
        checkpoint = out / (arm + '.pt')
        torch.save(dict(state_dict=model.cpu().state_dict(), classes=CLASSES, steps=STEPS,
                        input_shape=[8, 48, 48], seed=SEED, protocol_sha256=sha(out / 'protocol.json')), checkpoint)
        fits[arm] = dict(steps=STEPS, losses=losses, fit_seconds=fit_time,
            final_batch_inference_seconds=infer_time, samples=len(labels), parameters=parameter_count,
            fitting_correct=int((result.argmax(1) == labels.cpu().numpy()).sum()),
            checkpoint_sha256=sha(checkpoint), same_initialization_sha256=sha(out / 'initialization.pt'))
        del model, optimizer, x, probabilities
    write_new(out / 'learned-probabilities.json', outputs)
    write_new(out / 'training-result.json', fits)
    write_new(out / 'model-seal.json', dict(status='COMPLETE', protocol_sha256=sha(out / 'protocol.json'),
        preparation_seal_sha256=sha(out / 'preparation-seal.json'),
        probabilities_sha256=sha(out / 'learned-probabilities.json'), training_result_sha256=sha(out / 'training-result.json'),
        checkpoints={arm + '.pt': sha(out / (arm + '.pt')) for arm in fits}, fits=2, steps_per_fit=STEPS))
    del inputs, initial, initial_state
    torch.cuda.empty_cache()
    print(json.dumps(dict(status='TWO_FIXED_FITS_COMPLETE', fits=fits)))


def predict(out):
    prepared(out)
    seal = read(out / 'model-seal.json')
    require(seal['status'] == 'COMPLETE' and seal['protocol_sha256'] == sha(out / 'protocol.json'), 'Models incomplete')
    require(seal['preparation_seal_sha256'] == sha(out / 'preparation-seal.json'), 'Models use different inputs')
    require(seal['probabilities_sha256'] == sha(out / 'learned-probabilities.json'), 'Model scores differ')
    for n, digest in seal['checkpoints'].items():
        require(sha(out / n) == digest, 'Checkpoint differs')
    samples, simple = read(out / 'public-samples.json'), read(out / 'simple-outputs.json')
    probabilities = read(out / 'learned-probabilities.json')
    decisions = {'B': simple}
    for arm in ('C', 'C_no_rgb'):
        require(len(probabilities[arm]) == len(samples), 'Model output count differs')
        decisions[arm] = [dict(sample_id=s['sample_id'], **learned_decision(p)) for s, p in zip(samples, probabilities[arm])]
    lookup = {s['sample_id']: s for s in samples}
    per_frame = {}
    for arm, outputs in decisions.items():
        for result in outputs:
            sample = lookup[result['sample_id']]
            if result['suppress']:
                per_frame.setdefault(sample['frame_id'], {}).setdefault(arm, []).append(sample['zone'])
    threshold = read(PARENT / 'operating-point.json')['threshold']
    predictions = []
    for parent, scored in zip(read(PARENT / 'predictions.json'), read(PARENT / 'scores.json')):
        require(parent['id'] == scored['id'], 'Parent identity differs')
        suppressed = {a: sorted(per_frame.get(parent['id'], {}).get(a, [])) for a in decisions}
        arms = {'A': parent['candidate']}
        for arm in decisions:
            arms[arm] = frame_decision(arms['A'], scored['anchors'], scored['zone_scores'], threshold, set(suppressed[arm]))
        predictions.append({**{k: parent[k] for k in IDENTITY}, 'arms': arms, 'suppressed_zones': suppressed,
                            'original_anchors': scored['anchors'], 'observation_sha256': parent['observation_sha256']})
    write_new(out / 'zone-decisions.json', decisions)
    write_new(out / 'predictions.json', predictions)
    write_new(out / 'prediction-seal.json', dict(status='COMPLETE', frames=96,
        protocol_sha256=sha(out / 'protocol.json'), predictions_sha256=sha(out / 'predictions.json'),
        model_seal_sha256=sha(out / 'model-seal.json'), zone_decisions_sha256=sha(out / 'zone-decisions.json'),
        public_seal_sha256=sha(out / 'public-seal.json'), threshold=threshold))
    print(json.dumps(dict(status='PREDICTIONS_SEALED', frames=96, original_threshold=threshold)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=('freeze', 'prepare', 'train', 'predict'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.phase](args.out)
