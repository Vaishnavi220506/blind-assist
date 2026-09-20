"""Source admission and unchanged single-return observations for one new transfer.

Evaluator geometry is used only to simulate the established sensor law, validate
source admission, and produce isolated labels/lineage. The predictor receives
RGB paths and the 64 observed single returns, never native geometry or labels.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def materialize(out):
    """Materialize once; root runner owns protocol verification and sealing.

    Requires capture receipts and protocol/spec already frozen by the root.
    Writes observations.npz (ranges, boxes), identities.json, baseline.json,
    private-lineage.json, source-admission.json, evaluator/metadata.json,
    evaluator/transfer-labels.json, and observation-seal.json.
    """
    from run_spatial_complement_transfer import verify, seal, read, write, sha, T
    import run_core_transfer as source
    from core_transfer_spec import bounds, classify
    from tof_fov45_core import boxes45, simulate
    from ba_camera_corridor import sample_native
    from tof_corridor_calibration import score_frame, decide

    out = Path(out)
    p = verify(out)
    assert p['frames'] == 1152 and p['clips'] == 48 and p['frames_per_clip'] == 24
    assert p['dt_s'] == .2 and p['strong_threshold'] == T
    files = ['observations.npz', 'identities.json', 'baseline.json', 'private-lineage.json',
             'source-admission.json', 'evaluator/metadata.json', 'evaluator/transfer-labels.json']
    for name in files + ['materialization-start.json', 'observation-seal.json']:
        assert not (out / name).exists(), 'Materialization output already exists: ' + name
    write(out / 'materialization-start.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        backend='TASK_NOT_GPU_SUITABLE', input_law='UNCHANGED_SINGLE_RETURN'))
    cap = out / 'capture'
    receipt, launch = read(cap / 'receipt.json'), read(cap / 'launch-receipt.json')
    assert receipt['status'] == 'PASS' and receipt['frame_count'] == p['frames']
    assert receipt['source_unchanged'] and receipt['task_actors_released']
    assert read(cap / 'process-release.json')['released']
    assert launch['protocol_sha256'] == receipt['protocol_sha256'] == sha(out / 'protocol.json')
    assert receipt['spec_sha256'] == launch['spec_sha256'] == p['spec_sha256']
    assert receipt['script_sha256'] == launch['capture_script_sha256'] == p['code_hashes']['spatial_complement_transfer_capture.py']
    assert launch['launcher_sha256'] == p['code_hashes']['launch_spatial_complement_transfer.py']
    assert receipt['readiness_helper_sha256'] == p['code_hashes']['ue_capture_readiness.py']
    spec = read(out / 'spec.json')
    assert receipt['map_sha256_before'] == receipt['map_sha256_after'] == spec['expected_map_sha256']
    assert len(receipt['view_readiness']) == p['frames']
    manifest = read(cap / 'observations/manifest.json')['frames']
    geometry = read(cap / 'evaluator/geometry.json')
    cases = spec['cases']
    assert len(manifest) == len(geometry) == len(cases) == p['frames']
    assert all(c['split'] == 'transfer' for c in cases), 'Transfer source split only'
    groups = Counter(c['base_group_id'] for c in cases)
    clips = Counter(c['clip_id'] for c in cases)
    assert len(groups) == 16 and set(groups.values()) == {72}
    assert len(clips) == 48 and set(clips.values()) == {24}
    for start in range(0, len(cases), 24):
        clip = cases[start:start + 24]
        assert len({c['clip_id'] for c in clip}) == len({c['base_group_id'] for c in clip}) == 1
        assert [c['frame_in_clip'] for c in clip] == list(range(24))
        assert all(abs(c['time_s'] - i * .2) < 1e-8 for i, c in enumerate(clip))

    values, identities, baseline, admission, private, metadata, labels = [], [], [], [], [], [], []
    boxes = boxes45()
    for i, (case, rgb, geo) in enumerate(zip(cases, manifest, geometry)):
        source.source_check(case, geo, rgb, i)
        source.validate_readiness(receipt['view_readiness'][i])
        native_path = cap / 'evaluator' / geo['native_path']
        rgb_path = cap / 'observations' / rgb['rgb_path']
        assert sha(native_path) == geo['native_sha256']
        assert sha(rgb_path) == rgb['rgb_sha256'] == geo['rgb_sha256']
        native = np.load(native_path, allow_pickle=False)
        assert native.shape == (360, 640), 'Native source dimensions changed'
        target, corridor = source.masks(native, case, geo)
        # Preserve the original proxy and paired group/time random-number identity.
        identity = 'spatial-bce-v1/' + case['sensor_noise_key']
        vector, traces = simulate(sample_native(native), identity, boxes)
        score = score_frame(boxes, vector)
        a = decide(score, T)
        ids = dict(id=f'f{i:04d}', index=i, clip_id=rgb['clip_id'],
            frame_in_clip=rgb['frame_in_clip'], time_s=rgb['time_s'],
            rgb_path=rgb_path.relative_to(out).as_posix(), rgb_sha256=rgb['rgb_sha256'])
        identities.append(ids)
        values.append(vector)
        baseline.append(dict(index=i, current=a['alert'], unknown=a['unknown'], score=score['score'],
            valid_zones=a['valid_zones'], definite_zones=a['definite_zones']))
        # Truth is derived from rendered bounds and never inserted into observations.
        target_geo = next(o for o in geo['objects'] if o['name'] == 'target')
        center = np.asarray(target_geo['render_bounds_center_m'])
        half = np.asarray(target_geo['render_bounds_extent_m'])
        camera = case['camera']
        cc = np.array([center[1] - camera['y'], camera['z'] - center[2], center[0] - camera['x']])
        hh = half[[1, 2, 0]]
        actual = classify(cc - hh, cc + hh)
        declared = classify(*bounds(case))
        assert actual['truth'] == declared['truth']
        meta = {k: case[k] for k in ('base_group_id', 'split', 'clip_id', 'frame_in_clip',
                                   'time_s', 'phase', 'layer', 'background', 'layout_relation', 'type_id')}
        meta.update(index=i, id=ids['id'])
        metadata.append(meta)
        labels.append(dict(index=i, truth=bool(actual['truth'])))
        sampled_target = sample_native(target.astype(np.float32)).astype(bool)
        sampled_corridor = sample_native(corridor.astype(np.float32)).astype(bool)
        native_support = sum(int((sampled_target & sampled_corridor).ravel()[t['pixel_indices']].sum()) for t in traces)
        admission.append(dict(id=ids['id'], visible_target_pixels=int(target.sum()),
            competing_corridor_pixels=int((corridor & ~target).sum()),
            returned_target_corridor_samples=native_support, truth=bool(actual['truth'])))
        private.append(dict(id=ids['id'], identity=identity, native_sha256=geo['native_sha256'],
            traces=[{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in t.items()} for t in traces]))
        if i % 384 == 0:
            print('TRANSFER_MATERIALIZE', i, '/', p['frames'], flush=True)
    passed = all(r['visible_target_pixels'] > 0 and r['competing_corridor_pixels'] < 4 for r in admission)
    write(out / 'source-admission.json', dict(status='PASS' if passed else 'NOT_EVALUABLE', frames=admission))
    assert passed, 'Source failure; no sample exclusions or replacement'
    # Exclusive binary open also protects the payload against accidental overwrite.
    with (out / 'observations.npz').open('xb') as stream:
        np.savez_compressed(stream, ranges=np.stack(values), boxes=boxes)
    write(out / 'identities.json', identities)
    write(out / 'baseline.json', baseline)
    write(out / 'private-lineage.json', private)
    write(out / 'evaluator/metadata.json', metadata)
    write(out / 'evaluator/transfer-labels.json', labels)
    seal(out, 'observation-seal.json', files + [
        'materialization-start.json', 'capture/observations/manifest.json', 'capture/evaluator/geometry.json',
        'capture/receipt.json', 'capture/launch-receipt.json', 'capture/process-release.json'])
    print('TRANSFER_OBSERVATIONS_SEALED', p['frames'], flush=True)
