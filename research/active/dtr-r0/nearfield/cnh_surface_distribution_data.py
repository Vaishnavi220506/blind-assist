"""Training-only first-visible solid-angle labels and public query weights.

No observation synthesis is permitted. Geometry labels are an auxiliary target,
not photon/IRF energy. Query weights use only public rays, bin centers and the
current sensor-to-travel transform; unknown has zero geometric query weight,
which must NOT be interpreted as evidence of clear space.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-surface-distribution-20261002'
LABEL_ROOT = OUT/'labels/train'
OBSERVATIONS = ROOT/'artifacts.local/work/cnh-near-range-20261001/features/train'
UNITS = list(range(93000, 93096))
CONFIGS = np.arange(22)
FRAMES = np.arange(3, 16)
RAW_WIDTH = .0375348
UNKNOWN = 128
LABEL_SHAPE = (22, 13, 16, 16, 129)
QUERY_LOW = np.array([[-.30, -.2, .3], [-.30, .42, .3]])
QUERY_HIGH = np.array([[.30, .42, 3.], [.30, .9, 3.]])
EPS = 1e-8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def create_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    if path.exists():
        raise FileExistsError('Preserve existing evidence: '+str(path))
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.rename(path)


def microzones(array):
    """[zone_y,zone_x,sub_y*16+sub_x,...] -> [micro_y,micro_x,64,...]."""
    array = np.asarray(array)
    if array.shape[:3] != (8, 8, 256):
        raise ValueError('Expected original 16x16 ray quadrature in each 8x8 zone')
    tail = array.shape[3:]
    value = array.reshape(8, 8, 2, 8, 2, 8, *tail)
    return value.transpose(0, 2, 1, 4, 3, 5, *range(6, 6+len(tail))).reshape(16, 16, 64, *tail)


@lru_cache(maxsize=1)
def public_rays():
    # This calls only the frozen public quadrature, never make_scenes/labels.
    import cnh_proposal_attribution_scenes as S
    rays, weights = S.ray_grid()
    rays, weights = microzones(rays), microzones(weights)
    weights = weights/weights.sum(-1, keepdims=True)
    rays.setflags(write=False); weights.setflags(write=False)
    return rays, weights


def distribution_from_radial(radial):
    radial = np.asarray(radial)
    if radial.shape != (16, 16, 64):
        raise ValueError('Expected radial distances for 256 microzones x64 rays')
    _, weights = public_rays()
    valid = np.isfinite(radial) & (radial > 0) & (radial < RAW_WIDTH*128)
    bins = np.full(radial.shape, UNKNOWN, dtype=np.int64)
    bins[valid] = np.floor(radial[valid]/RAW_WIDTH).astype(np.int64)
    row = np.arange(256).reshape(16, 16, 1)
    values = np.bincount((row*129+bins).ravel(), weights=weights.ravel(), minlength=256*129)
    return values.reshape(16, 16, 129)


def surface_distribution(boxes, sensor_pose):
    """Training label only: nearest physical box surface, no reflectance weights."""
    import cnh_proposal_attribution_scenes as S
    rays, _ = public_rays()
    hits = S.raycast_boxes(sensor_pose[:3, 3], rays@sensor_pose[:3, :3].T, boxes)
    return distribution_from_radial(hits['distance'])


def query_weights(sensor_to_travel):
    """Public float32[HEAD/BODY,16,16,129]; unknown column is zero.

    At each native radial-bin center, integrate the fixed interior query test
    over the same64 solid-angle subrays. No scene geometry or labels are inputs.
    This is a bin-center proxy (<=width/2 radial quantization), not exact truth.
    """
    transform = np.asarray(sensor_to_travel, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Expected finite4x4 current sensor-to-travel transform')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10) or not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-9):
        raise ValueError('Query transform must be rigid')
    rays, weights = public_rays()
    directions = rays@transform[:3, :3].T
    radius = (np.arange(128)+.5)*RAW_WIDTH
    result = np.zeros((2, 16, 16, 129), dtype=np.float32)
    # Only one query's boolean mask is live; avoid a giant full XYZ tensor.
    for query in range(2):
        inside = np.ones((16, 16, 64, 128), dtype=bool)
        for axis in range(3):
            coordinate = directions[..., axis, None]*radius+transform[axis, 3]
            inside &= (coordinate >= QUERY_LOW[query, axis]+EPS) & (coordinate <= QUERY_HIGH[query, axis]-EPS)
        result[query, ..., :128] = np.einsum('yxrb,yxr->yxb', inside, weights)
    return result


def specification():
    return dict(units=UNITS, configs=CONFIGS.tolist(), frames=FRAMES.tolist(), shape=list(LABEL_SHAPE),
        dtype='float16', label_row_count_per_unit=22*13*16*16,
        radial_bins=128, width_m=RAW_WIDTH, unknown_index=UNKNOWN,
        microzones='Each original8x8zone splits2x2 contiguous8x8 quadrature rays; axes micro_y,micro_x,radial_class',
        distribution='Solid-angle normalized per microzone; first-visible surface only; retain all radial modes',
        unknown='Class index128 (129th class): geometric ray misses all surfaces or radial range>=4.8044544m; not device UNKNOWN, low-SNR/no-return cause, or clear-space evidence',
        quantization='Read half labels as float and renormalize each129class row before training',
        query='Fixed HEAD/BODY interior boxes with1e-8 inward margin; native radial-bin centers; unknown column0',
        operations=dict(training_geometry_labels_only=True, new_observation=False, synthesize_response=False,
            evaluation_geometry_labels=False, reflectance_weighting=False))


def source_identity():
    import cnh_near_range as NR
    import cnh_proposal_attribution_scenes as S
    import cnh_structure_space as SS
    paths = [Path(__file__), Path(NR.__file__), Path(S.__file__), Path(SS.__file__),
             S.SOURCE/'cnh_route_sensor.py', S.SOURCE/'cnh_track_a_geometry.py',
             S.SOURCE/'cnh_track_a_fov.py', S.SOURCE/'cnh_track_a_v13_sensor.py']
    return {str(p): sha(p) for p in paths}


def unit_metadata(unit, scenes):
    if unit not in UNITS or len(scenes) != 22 or [s['config'] for s in scenes] != CONFIGS.tolist():
        raise ValueError('Only fixed training units/configurations admitted')
    with np.load(OBSERVATIONS/f'unit{unit}.npz', allow_pickle=False) as data:
        # Labels and z1 values are not needed for geometry supervision identity.
        scene, frame = data['scene'], data['frame']
    if len(scene) != 22*16 or frame.shape != scene.shape:
        raise ValueError('Original observation identity shape changed')
    for config in CONFIGS:
        if not np.array_equal(frame[scene == config], np.arange(16)):
            raise ValueError('Original observation scene/frame order changed')
    sensor, travel = np.asarray(scenes[0]['poses']), np.asarray(scenes[0]['travel'])
    if any(not np.array_equal(s['poses'], sensor) or not np.array_equal(s['travel'], travel) for s in scenes):
        raise ValueError('Current-pose sharing across configs no longer holds')
    return dict(unit=np.array(unit), configs=CONFIGS, frames=FRAMES, observation_scene=scene, observation_frame=frame,
        sensor_poses=sensor, travel_poses=travel, sensor_to_travel=(np.linalg.inv(travel)@sensor)[FRAMES])


def canary():
    import cnh_near_range as NR
    import cnh_proposal_attribution_scenes as S
    started = time.perf_counter()
    scenes = NR.scenes_for(93000)
    metadata = unit_metadata(93000, scenes)
    replay_s = time.perf_counter()-started
    rays, weights = public_rays()
    original_rays, _ = S.ray_grid()
    for zy, zx, qy, qx in ((0, 0, 0, 0), (3, 5, 1, 0), (7, 7, 1, 1)):
        expected = original_rays[zy, zx].reshape(16, 16, 3)[qy*8:(qy+1)*8, qx*8:(qx+1)*8].reshape(64, 3)
        assert np.array_equal(rays[zy*2+qy, zx*2+qx], expected)
    # Empty, out-of-window and multimodal rays retain exactly one unit of mass.
    empty = distribution_from_radial(np.full((16, 16, 64), np.inf))
    assert np.allclose(empty[..., 128], 1., atol=1e-14) and np.all(empty[..., :128] == 0)
    multi = np.full((16, 16, 64), 2.)
    multi[..., :32] = 1.
    hist = distribution_from_radial(multi)
    assert np.allclose(hist.sum(-1), 1., atol=1e-14)
    assert np.all(hist[..., int(1/RAW_WIDTH)] > 0) and np.all(hist[..., int(2/RAW_WIDTH)] > 0)
    outside = distribution_from_radial(np.full((16, 16, 64), RAW_WIDTH*128))
    assert np.array_equal(outside, empty)
    # Independent nearest-hit control: a back box cannot alter the front return.
    front = dict(lo=[-4., -4., 1.], hi=[4., 4., 1.1], rho=.1)
    back = dict(lo=[-4., -4., 2.], hi=[4., 4., 2.1], rho=.9)
    identity = np.eye(4)
    assert np.array_equal(surface_distribution([front], identity), surface_distribution([back, front], identity))
    assert np.allclose(weights.sum(-1), 1., atol=1e-14)
    q = query_weights(identity)
    assert np.all(q[..., 128] == 0) and np.all(q[..., :8] == 0)
    assert np.all((q >= 0) & (q <= 1))
    # Independent scalar query integral for representative boxes/bins.
    for group, my, mx, rb in ((0, 7, 7, 16), (1, 13, 8, 50), (0, 0, 0, 127)):
        expected = 0.
        for direction, weight in zip(rays[my, mx], weights[my, mx]):
            point = direction*((rb+.5)*RAW_WIDTH)
            if all(QUERY_LOW[group, a]+EPS <= point[a] <= QUERY_HIGH[group, a]-EPS for a in range(3)):
                expected += weight
        assert abs(float(q[group, my, mx, rb])-expected) < 1e-7
    label_start = time.perf_counter()
    values = surface_distribution(scenes[0]['boxes'], scenes[0]['poses'][3])
    label_s = time.perf_counter()-label_start
    assert np.allclose(values.sum(-1), 1., atol=1e-14)
    stored = values.astype(np.float16)
    half_error = float(np.max(np.abs(stored.astype(float).sum(-1)-1)))
    query_start = time.perf_counter()
    public_query = query_weights(metadata['sensor_to_travel'][0])
    query_s = time.perf_counter()-query_start
    OUT.mkdir(parents=True, exist_ok=True)
    payload = OUT/'label_canary.npz'
    write_start = time.perf_counter()
    with payload.open('xb') as stream:
        np.savez(stream, labels=stored, query_weights=public_query, unit=93000, config=0, frame=3)
        stream.flush(); os.fsync(stream.fileno())
    write_s = time.perf_counter()-write_start
    result = dict(status='PASS', source_sha256=source_identity(), specification=specification(),
        observation_sha256={str(OBSERVATIONS/'unit93000.npz'): sha(OBSERVATIONS/'unit93000.npz')},
        output_sha256={payload.name: sha(payload)}, checked_identity=[93000, 0, 3],
        max_half_row_mass_error=half_error, no_return_or_outside_probability_mean=float(values[..., 128].mean()),
        checks=['microzone ray ordering', 'mass conservation/unknown/out-of-window', 'multimodal preservation',
            'nearest occlusion independent of reflectance', 'identity query centers/interior/unknown'],
        timing=dict(one_unit_scene_metadata_s=replay_s, one_frame_label_s=label_s, one_frame_query_s=query_s,
            canary_write_s=write_s, canary_bytes=payload.stat().st_size),
        estimate=dict(label_payload_bytes=int(np.prod(LABEL_SHAPE))*2*len(UNITS),
            geometry_s=(replay_s+label_s*22*13)*len(UNITS), frames=96*22*13,
            caveat='Single-frame geometry extrapolation; excludes complete data hashing, sustained writes and scene-dependent box counts; not a measured full-run duration'),
        elapsed_s=time.perf_counter()-started, observation_synthesis_calls=0)
    create_json(OUT/'label_canary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def request(prepare=False):
    plan_path, proof_path = OUT/'PLAN.json', OUT/'label_canary.json'
    if not plan_path.exists():
        raise RuntimeError('Parent PLAN required before full-label preparation')
    proof = read(proof_path)
    identity = source_identity()
    if proof['status'] != 'PASS' or proof['source_sha256'] != identity or proof['specification'] != specification():
        raise ValueError('Matching geometric canary required')
    value = dict(plan_sha256=sha(plan_path), canary_sha256=sha(proof_path), source_sha256=identity,
        specification=specification(), observation_sha256={str(OBSERVATIONS/f'unit{u}.npz'): sha(OBSERVATIONS/f'unit{u}.npz') for u in UNITS})
    path = OUT/'labels_request.json'
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen label request changed')
    elif prepare:
        create_json(path, value)
    else:
        raise RuntimeError('Parent must authorize prepare before build')
    return value, sha(path)


def build(chunk):
    import cnh_near_range as NR
    frozen, digest = request()
    k, n = map(int, chunk.split('/'))
    if not 0 <= k < n:
        raise ValueError('Expected zero-based k/n chunk')
    manifest = read(OUT/f'labels_chunk_{k}of{n}.json')
    if manifest['request_sha256'] != digest or manifest['units'] != UNITS[k::n]:
        raise ValueError('Chunk manifest differs from fixed request')
    for unit in UNITS[k::n]:
        folder = LABEL_ROOT/f'unit{unit}'
        receipt_path = folder/'receipt.json'
        if receipt_path.exists():
            old = read(receipt_path)
            if old['status'] != 'COMPLETE' or old['request_sha256'] != digest:
                raise ValueError('Prior label unit belongs to a different request')
            if any(sha(folder/name) != checksum for name, checksum in old['output_sha256'].items()):
                raise ValueError('Completed label cache changed')
            continue
        started = time.perf_counter()
        scenes = NR.scenes_for(unit)
        metadata = unit_metadata(unit, scenes)
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder/'labels.partial.npy'
        if partial.exists() or (folder/'labels.npy').exists():
            raise FileExistsError('Preserve incomplete unit before any restart')
        values = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16, shape=LABEL_SHAPE)
        max_half_error = 0.
        try:
            for config, scene in enumerate(scenes):
                for f_index, frame in enumerate(FRAMES):
                    distribution = surface_distribution(scene['boxes'], scene['poses'][frame])
                    if not np.allclose(distribution.sum(-1), 1., atol=1e-12):
                        raise ValueError('Label probability mass conservation failed')
                    values[config, f_index] = distribution.astype(np.float16)
                    max_half_error = max(max_half_error, float(np.max(np.abs(values[config, f_index].astype(float).sum(-1)-1))))
            values.flush()
        finally:
            values._mmap.close()
        with (folder/'metadata.npz').open('xb') as stream:
            np.savez(stream, **metadata)
        partial.rename(folder/'labels.npy')
        result = dict(status='COMPLETE', unit=unit, split='train', plan_sha256=frozen['plan_sha256'],
            request_sha256=digest, output_sha256={name: sha(folder/name) for name in ('labels.npy', 'metadata.npz')},
            shape=list(LABEL_SHAPE), dtype='float16', configs=CONFIGS.tolist(), frames=FRAMES.tolist(),
            label_row_count=22*13*16*16, max_half_row_mass_error=max_half_error,
            sensor_to_travel_shape=[13, 4, 4], observation_sha256=frozen['observation_sha256'][str(OBSERVATIONS/f'unit{unit}.npz')],
            elapsed_s=time.perf_counter()-started, observation_synthesis_calls=0)
        create_json(receipt_path, result)
        with (OUT/f'labels_progress_{k}of{n}.jsonl').open('a', encoding='utf8') as progress:
            progress.write(json.dumps(dict(unit=unit, status='COMPLETE', receipt_sha256=sha(receipt_path), elapsed_s=result['elapsed_s']))+'\n')
        print('COMPLETE train labels', unit, round(result['elapsed_s'], 2), flush=True)


def prepare(chunk):
    frozen, digest = request(True)
    k, n = map(int, chunk.split('/'))
    if not 0 <= k < n:
        raise ValueError('Expected zero-based k/n chunk')
    path = OUT/f'labels_chunk_{k}of{n}.json'
    value = dict(chunk=chunk, units=UNITS[k::n], request_sha256=digest, plan_sha256=frozen['plan_sha256'])
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing chunk manifest changed')
    else:
        create_json(path, value)
    return value


def finalize():
    frozen, digest = request()
    hashes, byte_count, elapsed = {}, 0, 0.
    for unit in UNITS:
        folder = LABEL_ROOT/f'unit{unit}'
        path = folder/'receipt.json'; receipt = read(path)
        if receipt['status'] != 'COMPLETE' or receipt['request_sha256'] != digest or receipt['unit'] != unit or receipt['shape'] != list(LABEL_SHAPE):
            raise ValueError('Training labels incomplete or wrong request')
        for name, checksum in receipt['output_sha256'].items():
            if sha(folder/name) != checksum:
                raise ValueError('Completed label output changed')
            byte_count += (folder/name).stat().st_size
        elapsed += receipt['elapsed_s']
        hashes[f'train/unit{unit}/receipt.json'] = sha(path)
    result = dict(status='COMPLETE', plan_sha256=frozen['plan_sha256'], request_sha256=digest,
        unit_receipt_sha256=hashes, units=UNITS, specification=specification(),
        label_row_count=96*22*13*16*16, output_bytes=byte_count, sum_unit_elapsed_s=elapsed,
        observations_synthesized=0, train_only=True)
    create_json(OUT/'labels_receipt.json', result)
    print('COMPLETE all training surface labels', byte_count, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('canary', 'prepare', 'build', 'finalize'), required=True)
    parser.add_argument('--chunk', default='0/1')
    args = parser.parse_args()
    {'canary': canary, 'prepare': lambda: prepare(args.chunk), 'build': lambda: build(args.chunk), 'finalize': finalize}[args.stage]()
