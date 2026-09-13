"""One frozen early checkpoint / threshold on a new complete controlled cohort."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import cv2
import numpy as np
import torch
import mz116_four_sensor as incumbent
from mz120_occupancy import OccupancyNet, CORE, CELLS, encode
from mz121_joint_readout import joint_checks
from run_mz120_occupancy import batch, predict, evaluate, labels, read_evaluation
from run_mz107_four_sensor import ROOT, sha, write, readrows
from research_backend import BackendCandidate, DeviceObservation, select_backend

MODEL_SHA256 = '23ea4d716be3550146db69cfa1ea2bc85e754799df6980cf02e236d98f9042da'
THRESHOLD = .58


def capture(bundle, output):
    """Portable launch only; original MZ115 engine and sensor code stay unchanged."""
    import psutil
    from ue_native_capture import run_owned
    assert not output.exists()
    assert not any((p.info['name'] or '').startswith('UnrealEditor') for p in psutil.process_iter(['name']))
    output.mkdir()
    source = Path(__file__).parent
    for name in ('mz115_zonal_capture.py', 'ue_capture_readiness.py', 'mz115_zonal_sensors.py',
                 'mz115_zonal_tof.py', 'mz113_dynamic_sensors.py', 'mz99_angle_information_capture.py'):
        shutil.copyfile(source/name, output/name)
    shutil.copyfile(bundle/'spec.json', output/'spec.json')
    engine = Path(os.environ['UE_ENGINE_ROOT'])/'Engine/Binaries/Win64/UnrealEditor.exe'
    project = Path(os.environ['BLINDASSIST_SOURCE'])/'artifacts.local/unreal/BlindAssistStreetLab/BlindAssistStreetLab.uproject'
    plugin = Path(os.environ['BLINDASSIST_UE_CAPTURE_PLUGIN'])
    assert engine.is_file() and project.is_file() and plugin.is_file()
    cache = output.parent/'ddc'; cache.mkdir(exist_ok=False)
    with socket.socket() as probe:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    env = dict(os.environ, BA_MZ115_OUT=str(output), BA_MZ115_SPEC=str(output/'spec.json'))
    env['UE-LocalDataCachePath'] = str(cache)
    cmd = [str(engine), str(project), '-RenderOffscreen', '-unattended', '-nosound', '-nop4',
        '-NoSplash', '-ddc=NoShared', '-ini:Engine:[Zen.AutoLaunch]:DesiredPort='+str(port),
        '-PLUGIN='+str(plugin), '-EnablePlugins=PythonScriptPlugin,BlindAssistCapture',
        '-ExecCmds=py '+(output/'mz115_zonal_capture.py').as_posix(), '-abslog='+str(output/'editor.log'),
        '-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=',
        '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
    write(output/'launch.json', dict(command=cmd, spec_sha256=sha(output/'spec.json'),
        project_sha256=sha(project), plugin_binary_sha256=sha(plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'),
        cache=str(cache), port=port, timeout_seconds=1200))
    run_owned(cmd, env, output, 1200)
    assert json.loads((output/'process-release.json').read_text())['released']


def score_group(rows, es, target, probability, baseline):
    candidate = evaluate(rows, es, target, probability, THRESHOLD, baseline)
    base = evaluate(rows, es, target, np.repeat(baseline[:, None], len(CELLS), 1).astype(float), .5, baseline)
    base.pop('spatial')  # MZ116 has no cell logits; never invent spatial predictions.
    return dict(baseline=base, early=candidate)


def analyze(bundle, captured, out):
    assert not out.exists(); out.mkdir()
    receipt = json.loads((captured/'receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['frames'] == 288
    assert sha(captured/'spec.json') == receipt['spec_sha256'] == sha(bundle/'spec.json')
    for name, digest in receipt['hashes'].items():
        assert sha(captured/name) == digest, name
    rows = readrows(captured/'raw.jsonl')
    assert len(rows) == len({r['id'] for r in rows}) == 288
    assert len({r['episode_id'] for r in rows}) == 24
    assert not any(any(s in k for s in ('truth', 'actor', 'native', 'depth')) for r in rows for k in r)
    images = {}; inputs = []; rgb = []; episode = None; yaw = 0.
    for r in rows:
        path = (captured/r['rgb_path']).resolve(); assert path.is_relative_to(captured.resolve())
        im = cv2.imread(str(path)); assert im is not None and im.shape == (360, 640, 3)
        images[r['id']] = im; rgb.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if r['episode_id'] != episode: yaw = 0.
        if r['imu_valid']: yaw += r['delta_yaw']
        episode = r['episode_id']; inputs.append(encode(r, yaw))
    arrays = {k: np.stack([v[k] for v in inputs]) for k in inputs[0]}
    arrays['rgb'] = np.stack(rgb).transpose(0, 3, 1, 2)
    data = {k: torch.from_numpy(v) for k, v in arrays.items()}
    assert sha(bundle/'early-model.pt') == MODEL_SHA256
    assert torch.cuda.is_available(); torch.set_num_threads(4)
    model = OccupancyNet().cuda().eval()
    model.load_state_dict(torch.load(bundle/'early-model.pt', map_location='cuda', weights_only=True))
    probe = batch(data, np.arange(8)); cpu_model = copy.deepcopy(model).cpu()
    cpu_probe = {k: v.cpu() for k, v in probe.items()}
    def gpu_run():
        with torch.no_grad(): return model(probe)
    def cpu_run():
        with torch.no_grad(): return cpu_model(cpu_probe)
    placement = select_backend('model-inference', cpu=BackendCandidate('torch-cpu', 'cpu', cpu_run,
        lambda v: DeviceObservation(v.device.type, 'host CPU', 'torch '+torch.__version__)),
        gpu=BackendCandidate('torch-cuda', 'cuda', gpu_run,
            lambda v: DeviceObservation(v.device.type, torch.cuda.get_device_name(), 'torch '+torch.__version__), torch.cuda.synchronize),
        record_path=out/'early-backend.json')
    assert placement['selected_device_type'] == 'cuda'
    del cpu_model, cpu_probe
    tick = time.perf_counter(); probability = predict(model, data, np.arange(len(rows)))
    torch.cuda.synchronize(); early_seconds = time.perf_counter()-tick
    select_backend('batch-tensor', cpu=BackendCandidate('opencv-scipy-cpu', 'cpu',
        lambda: incumbent.predict(rows[:1], lambda r: images[r['id']]),
        lambda _: DeviceObservation('cpu', 'host CPU', 'OpenCV '+cv2.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE', record_path=out/'baseline-backend.json',
        capabilities={'opencv_cuda_devices': cv2.cuda.getCudaEnabledDeviceCount(),
            'reason': 'Unchanged OpenCV connected-component and geometry implementation has no GPU backend'})
    tick = time.perf_counter()
    base_predictions = incumbent.predict(rows, lambda r: images[r['id']])['3.6']['resolution_guard']
    baseline_seconds = time.perf_counter()-tick
    baseline = np.array([p['candidate'] for p in base_predictions], bool)
    np.save(out/'early-probabilities.npy', probability)
    write(out/'baseline-predictions.json', base_predictions)
    write(out/'prediction-seal.json', dict(status='FROZEN_PREDICTIONS_BEFORE_EVALUATOR_PARSE',
        threshold=THRESHOLD, model_sha256=MODEL_SHA256, freeze_sha256=sha(bundle/'freeze.json'),
        raw_sha256=sha(captured/'raw.jsonl'), receipt_sha256=sha(captured/'receipt.json'),
        early_sha256=sha(out/'early-probabilities.npy'), baseline_sha256=sha(out/'baseline-predictions.json')))
    # No model prediction is changed after this point. Geometry and source roles are evaluator-only.
    es = read_evaluation(captured)
    assert [e['id'] for e in es] == [r['id'] for r in rows]
    spec = json.loads((captured/'spec.json').read_text())
    assert [f['id'] for f in spec['frames']] == [r['id'] for r in rows]
    target = np.stack([labels(e) for e in es])
    design_truth = {a['episode']: a['source_aabb_labels'] for a in spec['source_audit']}
    for episode_id, expected in design_truth.items():
        idx = [i for i, r in enumerate(rows) if r['episode_id'] == episode_id]
        assert target[idx][:, CORE].any(1).tolist() == expected, 'Native/source geometry mismatch: '+episode_id
    overall = score_group(rows, es, target, probability, baseline)
    groups = {}
    for family in sorted({e['family'] for e in es}):
        idx = np.array([i for i, e in enumerate(es) if e['family'] == family])
        groups[family] = score_group([rows[i] for i in idx], [es[i] for i in idx], target[idx], probability[idx], baseline[idx])
    # Group whole episodes only, preserving the original event cadence and denominator.
    stress = np.array(['boundary' in e['family'] for e in es])
    for name, mask in [('nonstress', ~stress), ('boundary_stress', stress)]:
        idx = np.flatnonzero(mask)
        if len(idx): groups[name] = score_group([rows[i] for i in idx], [es[i] for i in idx], target[idx], probability[idx], baseline[idx])
    frame_report = [dict(id=r['id'], episode_id=r['episode_id'], time_s=r['time_s'], family=e['family'],
        truth=bool(y[CORE].any()), baseline=bool(b), early_score=float(p[CORE].max()), early=bool(p[CORE].max() >= THRESHOLD),
        occupied_TP=int((y & (p >= THRESHOLD)).sum()), occupied_FP=int((~y & (p >= THRESHOLD)).sum()),
        occupied_FN=int((y & (p < THRESHOLD)).sum())) for r, e, y, p, b in zip(rows, es, target, probability, baseline)]
    write(out/'frame-report.json', frame_report)
    per_episode = {}
    for episode_id in design_truth:
        idx = np.array([i for i, r in enumerate(rows) if r['episode_id'] == episode_id])
        per_episode[episode_id] = score_group([rows[i] for i in idx], [es[i] for i in idx], target[idx], probability[idx], baseline[idx])
        for arm, pred in [('baseline', baseline[idx]), ('early', probability[idx][:, CORE].max(1) >= THRESHOLD)]:
            longest = run = 0
            for gt, hit in zip(target[idx][:, CORE].any(1), pred):
                run = run+1 if gt and not hit else 0; longest = max(longest, run)
            per_episode[episode_id][arm]['longest_missed_positive_run_frames'] = longest
    write(out/'episode-report.json', per_episode)
    pairs = []
    for pair in spec['pairs']:
        a, b = [np.array([i for i, r in enumerate(rows) if r['episode_id'] == ep]) for ep in pair['episodes']]
        ga = target[a][:, CORE].any(1); gb = target[b][:, CORE].any(1)
        assert np.all(ga != gb)
        pa = probability[a] >= THRESHOLD; pb = probability[b] >= THRESHOLD
        changed = target[a] != target[b]
        pairs.append(dict(pair_id=pair['pair_id'], category=pair['category'], paired_frames=len(a),
            baseline_both_alert_answers_correct=int(((baseline[a] == ga) & (baseline[b] == gb)).sum()),
            early_both_alert_answers_correct=int(((pa[:, CORE].any(1) == ga) & (pb[:, CORE].any(1) == gb)).sum()),
            changed_truth_cells=int(changed.sum()),
            early_both_changed_cell_answers_correct=int((changed & (pa == target[a]) & (pb == target[b])).sum()),
            limitation='Matched geometry/appearance/time; independent sensor noise. Not an isolated sensor replacement control.'))
    write(out/'pair-report.json', pairs)
    checks = joint_checks(overall['early'], overall['baseline'])
    result = dict(status='FROZEN_EARLY_FRESH_CONTROLLED_COMPLETE', frames=len(rows), episodes=24,
        threshold=THRESHOLD, model_sha256=MODEL_SHA256, overall=overall, groups=groups, checks=checks, paired_results=pairs,
        decision='RETAIN_FRESH_CONTROLLED_ALERT_CHALLENGER' if all(checks.values()) else 'FRESH_CONTROLLED_JOINT_REQUIREMENTS_NOT_MET',
        baseline_policy='MZ116_REMAINS_BASELINE', training_runs=0, threshold_searches=0,
        early_inference_seconds=early_seconds, baseline_inference_seconds=baseline_seconds,
        limitation='New constructed scene configurations with inherited renderer and hypothetical sensors; not natural scenes, hardware, or echo identity evidence')
    write(out/'summary.json', result)
    write(out/'completion.json', dict(status='PASS', summary_sha256=sha(out/'summary.json'),
        predictions_sha256=sha(out/'prediction-seal.json'), resources='CUDA tensors released on process exit'))
    print(json.dumps(dict(decision=result['decision'], checks=checks, overall=overall), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    bundle = args.bundle.resolve(); out = args.output.resolve()
    art = Path(os.environ.get('BLINDASSIST_ARTIFACTS', ROOT/'artifacts.local')).resolve()
    assert out.is_relative_to(art) and not out.exists()
    frozen = json.loads((bundle/'freeze.json').read_text())
    for name, digest in frozen['files'].items(): assert sha(bundle/name) == digest, name
    assert frozen['threshold'] == THRESHOLD and frozen['model_sha256'] == MODEL_SHA256
    out.mkdir(); started = time.perf_counter()
    capture(bundle, out/'capture-v1')
    analyze(bundle, out/'capture-v1', out/'analysis-v1')
    write(out/'completion.json', dict(status='PASS', elapsed_seconds=time.perf_counter()-started,
        capture_receipt_sha256=sha(out/'capture-v1/receipt.json'), analysis_summary_sha256=sha(out/'analysis-v1/summary.json'),
        process_release=json.loads((out/'capture-v1/process-release.json').read_text())))


if __name__ == '__main__': main()
