"""Portable frozen MZ129 Radar arm and separately admitted ToF generator.

Inference accepts only public sensor observations and RGB pixels. The analytic
generator is a separate authority: callers must seal/reload its public rows
before calling inference. No labels, scene objects or ray owners are exported.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np

import mz116_four_sensor as mz116
import mz125_observable_correction as mz125
import mz129_radar_box_correction as mz129
from mz128_zone_weighting import FOUR, THRESHOLD, column_weights, readout


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/mz123-frozen-early-20260913/returned-v1'
IMAGE_ROOT = ROOT/'artifacts.local/work/mz125-observable-correction-20260913/rgb-v1'
MZ129_ROOT = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'
DEFAULT_OUTPUT = ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/incumbent'
OBSERVABLE_FIELDS = (
    'id', 'episode_id', 'time_s', 'camera_in_body_m', 'camera_pitch_deg',
    'rgb_path', 'rgb_intrinsics', 'tof_packet_received', 'tof_zones',
    'tof_max_targets', 'tof_target_order', 'tof_model',
    'radar_packet_received', 'radar_range_m', 'radar_angle', 'radar_velocity',
    'radar_valid', 'delta_yaw', 'delta_pitch', 'imu_valid',
    'tof64_range_m', 'tof64_status', 'tof64_theta_deg', 'tof64_phi_deg',
)
ZONE_FIELDS = ('zone_id', 'theta_bounds_deg', 'phi_bounds_deg', 'target_count')
TARGET_FIELDS = ('distance_m', 'range_noise_sigma_m', 'signal_strength_proxy', 'status')
INTRINSIC_FIELDS = ('width', 'height', 'fx', 'fy', 'cx', 'cy')


def public_observations(rows):
    """Copy the sensor contract; discard any accidental evaluator metadata."""
    result = []
    for row in rows:
        public = {k:copy.deepcopy(row[k]) for k in OBSERVABLE_FIELDS if k in row}
        public['rgb_intrinsics'] = {k:public['rgb_intrinsics'][k]
                                    for k in INTRINSIC_FIELDS if k in public['rgb_intrinsics']}
        public['tof_zones'] = [dict({k:copy.deepcopy(z[k]) for k in ZONE_FIELDS},
            targets=[{k:copy.deepcopy(t[k]) for k in TARGET_FIELDS} for t in z['targets']])
            for z in row['tof_zones']]
        result.append(public)
    return result


def serial(value):
    return json.loads(json.dumps(value, allow_nan=False))


def predict_incumbent(rows, image_loader):
    """Return predictions, baseline, corrected, radar and audit for fresh rows.

    This runs original MZ116 at 3.6 m, MZ125 foreground correction, MZ128
    binary-four ToF readout, then original MZ129 Radar correction with complete
    original optical-flow carry and both guards. Final predictions match the
    sealed MZ129 replay's ``radar`` arm, including its diagnostic fields.
    ``image_loader(public_row)`` must return unchanged BGR pixels on every call.
    """
    if cv2.__version__ != '5.0.0':
        raise RuntimeError('Frozen MZ116 MSER requires the verified OpenCV 5.0.0 replay runtime; '
                           'keep it separate from the Torch training environment')
    rows = public_observations(rows)
    started = time.perf_counter()
    baseline = mz116.predict(rows, image_loader)['3.6']['resolution_guard']
    corrected = [mz125.predict_frame(r, p, image_loader(r))[0]
                 for r, p in zip(rows, baseline)]
    radar = mz129.predict(rows, baseline, corrected, image_loader)
    predictions = []
    for row, corrected_frame, radar_frame in zip(rows, corrected, radar['predictions']):
        geom = dict(id=row['id'], **readout(corrected_frame, column_weights(row, FOUR)))
        flag = bool(geom['score'] >= THRESHOLD or geom['certain_coarse'] or radar_frame['candidate'])
        predictions.append(dict(geom, candidate=flag, candidate_state='ALERT' if flag else 'UNKNOWN',
                                common_radar=radar_frame['common_radar'],
                                guard_events=radar_frame['guard_events']))
    return dict(predictions=serial(predictions), baseline=serial(baseline),
        corrected=serial(corrected), radar=serial(radar['predictions']),
        audit=dict(frames=len(rows), seconds=time.perf_counter()-started,
            pipeline='MZ116_ORIGINAL_MZ125_FOREGROUND_MZ128_BINARY_FOUR_MZ129_RADAR',
            authority='OBSERVATION_ONLY_NO_EVALUATOR_INPUT',
            backend='CPU: GPU_BACKEND_UNAVAILABLE for frozen OpenCV flow/MSER and NumPy foreground pipeline',
            python=sys.executable, numpy=np.__version__, opencv=cv2.__version__, radar_audit=radar['audit']))


def shifted_geometry():
    """64 unchanged zones; 9 rays shifted +.1 theta / -.1 phi subcell."""
    from mz133_angular_resolution import geometry
    zones, rays = geometry(8, 3)
    # Angular-positive vertical means elevation, so negative is downward.
    subcell_degrees = 45./8/3
    theta = np.arctan(rays[:, :, 1]) + np.radians(.1*subcell_degrees)
    phi = np.arctan(rays[:, :, 2]) - np.radians(.1*subcell_degrees)
    return zones, np.stack([np.ones(theta.shape), np.tan(theta), np.tan(phi)], axis=-1)


def regenerate_shifted_tof(rows, spec, evaluations):
    """Return (public_rows, admission_and_generation_report), never predictions.

    Failed native admission returns an empty public cohort and a FAIL report.
    All successful rows preserve every existing public non-ToF field exactly.
    The generator consumes identities only for ordered joins/RNG episode state;
    scene identities, owners, seeds and evaluator fields never enter public rows.
    The caller seals these observations before any model/readout invocation.
    """
    from mz133_angular_resolution import admission, trace, reduce_zone, geometry
    rows = public_observations(rows)
    frames = spec['frames']
    if not rows or not len(rows) == len(frames) == len(evaluations):
        raise ValueError('Native admission requires a nonempty, complete matched cohort')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or ids != [f['id'] for f in frames] or ids != [e['id'] for e in evaluations]:
        raise ValueError('Native admission frame IDs/order do not match')
    expected_zones, _ = geometry(8, 3)
    for row, frame, evaluation in zip(rows, frames, evaluations):
        if frame['episode'] != row['episode_id']:
            raise ValueError('Generator episode does not match public episode')
        if len(row['tof_zones']) != 64 or len(evaluation['zonal_tof_native']) != 64:
            raise ValueError('Native admission requires exactly 64 zones')
        for old, native, expected in zip(row['tof_zones'], evaluation['zonal_tof_native'], expected_zones):
            if {k:old[k] for k in expected} != expected or len(native['private_rays']) != 9:
                raise ValueError('Native admission requires original 8x8 bounds and 3x3 rays')
            if old['target_count'] != len(old['targets']):
                raise ValueError('Inconsistent public target count')
    admit = admission(spec, rows, evaluations)
    report = dict(status=admit['status'], admission=admit,
        authority='ADMITTED_ANALYTIC_FIXED_BUDGET_PERTURBATION_NOT_NEW_UE_OR_HARDWARE',
        grid=[8,8], rays_per_zone=9, horizontal_subcell_shift=.1,
        vertical_elevation_subcell_shift=-.1, angular_shift_deg=[.1875,-.1875],
        signal_scale=1., range_noise_sigma_m=.04,
        note='Public rows must be sealed/reloaded before inference; report is generator-side only')
    if admit['status'] != 'PASS':
        return [], report
    zones, rays = shifted_geometry()
    states = {}; generated = []; stats = Counter()
    for row, frame in zip(rows, frames):
        distances, reflectances, _, _, _ = trace(frame, spec, rays)
        rng = states.setdefault(frame['episode'], random.Random(
            frame.get('tof_sensor_seed', frame['sensor_seed']+1150003)))
        newrow = copy.deepcopy(row)
        newrow['tof_zones'] = []
        for z, zone in enumerate(zones):
            targets, _, weak = reduce_zone(distances[z], reflectances[z], rng,
                                           row['tof_packet_received'], 1., .04)
            newrow['tof_zones'].append(dict(copy.deepcopy(zone), target_count=len(targets), targets=targets))
            stats['below_floor_components'] += weak
            stats['empty_zones'] += int(not targets)
            stats.update(t['status'] for t in targets)
        assert {k:v for k,v in newrow.items() if k != 'tof_zones'} == {
            k:v for k,v in row.items() if k != 'tof_zones'}
        stats['changed_frames'] += int(newrow['tof_zones'] != row['tof_zones'])
        generated.append(newrow)
    report.update(frames=len(rows), sample_rays=len(rows)*64*9, statistics=dict(stats),
                  public_non_tof_fields_exact=True)
    return generated, report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare_capture(capture, image_root, output):
    """Seal nominal/shifted public observations and predictions, without scoring.

    Output: nominal/raw.jsonl and shifted/raw.jsonl, each arm's predictions.json
    (full predict_incumbent result), admission.json, input-seal.json,
    observation-seal.json, prediction-seal.json and completion.json.
    Native evaluator data is opened only by generation/admission. It is never
    passed to inference, and there is no frame-label parsing or scoring here.
    """
    capture, image_root, output = map(lambda p:Path(p).resolve(), (capture,image_root,output))
    if output.exists() or not output.is_relative_to(DEFAULT_OUTPUT.resolve()):
        raise ValueError('Use a new output directory inside MZ136 incumbent')
    if cv2.__version__ != '5.0.0':
        raise RuntimeError('Preparation requires the verified OpenCV 5.0.0 runtime')
    paths = {name:capture/name for name in ('raw.jsonl','spec.json','evaluator.jsonl','receipt.json')}
    receipt = read(paths['receipt.json'])
    if receipt['status'] != 'PASS':
        raise ValueError('Capture receipt has not passed')
    for name in ('raw.jsonl','evaluator.jsonl'):
        assert sha(paths[name]) == receipt['hashes'][name], name
    assert sha(paths['spec.json']) == receipt['spec_sha256']
    rows = [json.loads(s) for s in paths['raw.jsonl'].read_text(encoding='utf-8').splitlines()]
    inputs = {str(path):sha(path) for path in paths.values()}
    for row in rows:
        path = (image_root/row['rgb_path']).resolve()
        assert path.is_relative_to(image_root)
        digest = sha(path)
        assert digest == receipt['hashes'][row['rgb_path']], row['id']
        inputs[str(path)] = digest
    output.mkdir(parents=True)
    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n',encoding='utf-8')
    write('input-seal.json',dict(hashes=inputs,opencv=cv2.__version__,numpy=np.__version__,python=sys.executable))
    spec = read(paths['spec.json'])
    evaluations = [json.loads(s) for s in paths['evaluator.jsonl'].read_text(encoding='utf-8').splitlines()]
    shifted, admission = regenerate_shifted_tof(rows,spec,evaluations)
    write('admission.json',admission)
    if admission['status'] != 'PASS':
        write('completion.json',dict(status='SOURCE_NOT_ADMITTED',prediction_count=0,
                                     admission_sha256=sha(output/'admission.json')))
        raise ValueError('Native analytic admission failed; diagnostic preserved, no predictions emitted')
    for arm, observations in (('nominal',public_observations(rows)),('shifted',shifted)):
        (output/arm).mkdir()
        (output/arm/'raw.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n'
                                                for r in observations),encoding='utf-8')
    # Seal before inference and explicitly destroy local generator/evaluator state.
    write('observation-seal.json',dict(authority='OBSERVATIONS_SAVED_BEFORE_INFERENCE',
        admission_sha256=sha(output/'admission.json'),
        hashes={arm:sha(output/arm/'raw.jsonl') for arm in ('nominal','shifted')}))
    del rows, shifted, spec, evaluations
    def loader(row):
        image = cv2.imread(str(image_root/row['rgb_path']))
        assert image is not None
        return image
    for arm in ('nominal','shifted'):
        public = [json.loads(s) for s in (output/arm/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
        assert sha(output/arm/'raw.jsonl') == read(output/'observation-seal.json')['hashes'][arm]
        result = predict_incumbent(public,loader)
        write(arm+'/predictions.json',result)
        print(json.dumps(dict(arm=arm,frames=len(public),seconds=result['audit']['seconds'])),flush=True)
    sources = {str(Path(module.__file__).resolve()):sha(module.__file__)
        for module in list(sys.modules.values()) if getattr(module,'__file__',None)
        and Path(module.__file__).suffix=='.py'
        and Path(module.__file__).resolve().is_relative_to(Path(__file__).parent.resolve())}
    sources[str(Path(__file__).resolve())] = sha(__file__)
    write('prediction-seal.json',dict(authority='OBSERVATION_ONLY_PREDICTIONS_NO_SCORING',
        inputs=inputs,source_hashes=sources,
        observation_seal_sha256=sha(output/'observation-seal.json'),
        predictions_sha256={arm:sha(output/arm/'predictions.json') for arm in ('nominal','shifted')}))
    assert inputs == {path:sha(path) for path in inputs}
    write('completion.json',dict(status='PASS',frames=len(public),inputs_unchanged=True,
        prediction_seal_sha256=sha(output/'prediction-seal.json'),
        resources='No persistent worker or allocation; durable observations, predictions and diagnostics retained'))
    return read(output/'completion.json')


def regression(output=DEFAULT_OUTPUT):
    """Exact sealed old-cohort regression, no labels read or scoring performed."""
    output = Path(output)
    if not output.resolve().is_relative_to(DEFAULT_OUTPUT.resolve()):
        raise ValueError('Regression artifacts must stay inside MZ136 incumbent directory')
    output.mkdir(parents=True, exist_ok=True)
    raw = SOURCE/'capture-v1/raw.jsonl'
    expected = MZ129_ROOT/'replay-v1r2/predictions.json'
    seal = MZ129_ROOT/'replay-v1r2/prediction-seal.json'
    assert sha(expected) == read(seal)['predictions_sha256']
    rows = [json.loads(s) for s in raw.read_text(encoding='utf-8').splitlines()]
    receipt = read(SOURCE/'capture-v1/receipt.json')
    assert sha(raw) == receipt['hashes']['raw.jsonl']
    image_hashes = {}
    for row in rows:
        path = IMAGE_ROOT/row['rgb_path']
        image_hashes[row['rgb_path']] = sha(path)
        assert image_hashes[row['rgb_path']] == receipt['hashes'][row['rgb_path']]
    def loader(row):
        image = cv2.imread(str(IMAGE_ROOT/row['rgb_path']))
        assert image is not None
        return image
    result = predict_incumbent(rows, loader)
    expected_frames = read(expected)['radar']
    mismatches = [row['id'] for row, got, wanted in zip(rows, result['predictions'], expected_frames)
                  if got != wanted]
    base = read(SOURCE/'analysis-v1/baseline-predictions.json')
    correction = read(IMAGE_ROOT.parent/'correction-v1r1/predictions.json')
    radar = read(MZ129_ROOT/'radar-v1/predictions.json')
    checks = dict(frames=len(rows), all_288_frames=len(rows)==len(expected_frames)==288,
        final_prediction_exact=not mismatches, baseline_exact=result['baseline']==base,
        corrected_exact=result['corrected']==correction, radar_exact=result['radar']==radar,
        mismatches=mismatches)
    status = 'PASS' if all(checks[k] for k in ('all_288_frames','final_prediction_exact',
                                               'baseline_exact','corrected_exact','radar_exact')) else 'FAIL'
    for filename, value in [('regression-predictions.json',result), ('regression.json',dict(
            status=status, **checks, raw_sha256=sha(raw), expected_sha256=sha(expected),
            expected_seal_sha256=sha(seal), rgb_sha256=image_hashes, audit=result['audit']))]:
        (output/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    assert status == 'PASS', checks
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--image-root',type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.regression:
        print(json.dumps(regression(args.output), indent=2))
    elif args.capture and args.image_root:
        print(json.dumps(prepare_capture(args.capture,args.image_root,args.output),indent=2))
    else:
        parser.error('Use --regression or both --capture and --image-root')
