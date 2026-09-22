"""Separate observation construction and privileged controlled-object labels."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

EDGES = np.array([.3, .75, 1.25, 1.75, 2.25, 2.75, 3.], np.float32)
QUERIES = np.array([[x-.3, x+.3, lo, hi]
                    for lo, hi in ((.42, .9), (-.2, .42))
                    for x in (-.3, 0., .3)], np.float32)
CENTRE = [1, 4]
FOCAL = 640 / (2 * np.tan(np.deg2rad(50)))


def stage_path(root, stage):
    root = Path(root)
    return root.with_name(root.name+'-'+stage)


def new_stage_directory(path):
    """The governed runner may precreate the empty result-file parent."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Nonempty stage directory: '+str(path))
    path.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def observation_tokens(values, boxes):
    values = np.asarray(values, np.float32)
    valid = np.isfinite(values) & (values >= .1) & (values < 8)
    normalized = np.asarray(boxes, np.float32) / [192, 256, 192, 256]
    return np.concatenate([np.where(valid, values/8, 0)[:, None],
                           valid.astype(np.float32)[:, None], normalized], 1).astype(np.float32)


def camera_bounds(objects, camera):
    """Full rendered AABBs of all declared controlled objects, evaluator only."""
    result = []
    for obj in objects:
        c = np.asarray(obj['render_bounds_center_m'], np.float64)
        h = np.asarray(obj['render_bounds_extent_m'], np.float64)
        center = np.array([c[1]-camera['y'], camera['z']-c[2], c[0]-camera['x']])
        half = h[[1, 2, 0]]
        result.append((center-half, center+half))
    return result


def geometric_labels(bounds, queries=QUERIES):
    distance = np.full(len(queries), np.nan, np.float32)
    classes = np.full(len(queries), len(EDGES)-1, np.int64)
    for qi, (xl, xh, yl, yh) in enumerate(queries):
        near = [max(float(lo[2]), float(EDGES[0])) for lo, hi in bounds
                if hi[0] >= xl and lo[0] <= xh and hi[1] >= yl and lo[1] <= yh
                and hi[2] >= EDGES[0] and lo[2] <= EDGES[-1]]
        if near:
            distance[qi] = min(near)
            classes[qi] = min(len(EDGES)-2, int(np.searchsorted(EDGES[1:], distance[qi], side='left')))
    return classes, distance


def visible_labels(native, bounds, queries=QUERIES):
    """Native visible masks; no-hit is NOT complete geometric free space.

    Output pixels contain native occupied fractions, preserving thin surfaces.
    Invalid source rays are ignored. Geometric labels are separately invalidated
    when any occupied native pixel has no declared-object support.
    """
    if native.shape != (360, 640):
        raise ValueError('Native optical Z must be 360x640')
    yy, xx = np.mgrid[:360, :640]
    z = np.asarray(native, np.float32)
    valid = np.isfinite(z) & (z > 0)
    z = np.where(valid, z, 0)
    x, y = (xx+.5-320)/FOCAL*z, (yy+.5-180)/FOCAL*z
    declared = np.zeros(z.shape, bool)
    for lo, hi in bounds:
        declared |= ((x >= lo[0]-.02) & (x <= hi[0]+.02) &
                     (y >= lo[1]-.02) & (y <= hi[1]+.02) &
                     (z >= lo[2]-.02) & (z <= hi[2]+.02))
    near = valid & (z >= EDGES[0]) & (z <= EDGES[-1])
    masks, counts, unexplained, nearest = [], [], [], []
    for xl, xh, yl, yh in queries:
        mask = near & (x >= xl) & (x <= xh) & (y >= yl) & (y <= yh)
        masks.append(mask.reshape(45, 8, 80, 8).mean((1, 3)))
        counts.append(int(mask.sum()))
        unexplained.append(int((mask & ~declared).sum()))
        nearest.append(float(z[mask].min()) if mask.any() else None)
    coverage = valid.reshape(45, 8, 80, 8).mean((1, 3)).astype(np.float32)
    return dict(mask=np.asarray(masks, np.float32), coverage=coverage,
                count=counts, unexplained=unexplained, visible_nearest=nearest)


def materialize(root):
    from ba_camera_corridor import sample_native
    from tof_fov45_core import boxes45, simulate
    from tof_corridor_calibration import score_frame, decide
    root = Path(root)
    cap = stage_path(root, 'capture')
    spec = read(root/'plan/spec.json')
    receipt = read(cap/'receipt.json')
    launch = read(cap/'launch-receipt.json')
    assert receipt['status'] == 'PASS' and read(cap/'process-release.json')['released']
    assert receipt['source_unchanged'] and receipt['task_actors_released']
    assert receipt['spec_sha256'] == launch['spec_sha256'] == sha(root/'plan/spec.json')
    assert receipt['protocol_sha256'] == launch['protocol_sha256'] == sha(root/'plan/protocol.json')
    manifests = read(cap/'observations/manifest.json')['frames']
    geometry = read(cap/'evaluator/geometry.json')
    cases = spec['cases']
    assert len(cases) == len(manifests) == len(geometry) == spec['frames']
    out = stage_path(root, 'prepared')
    new_stage_directory(out)
    (out/'observations').mkdir()
    (out/'labels').mkdir()
    boxes = boxes45()
    n = len(cases)
    rgb = np.lib.format.open_memmap(out/'observations/rgb.npy', mode='w+', dtype=np.uint8, shape=(n, 3, 180, 320))
    tof = np.lib.format.open_memmap(out/'observations/tof.npy', mode='w+', dtype=np.float32, shape=(n, 64, 6))
    buffers = {s: [] for s in ('train', 'dev', 'evaluation')}
    labels = {s: dict(classes=[], distances=[], mask=[], coverage=[], valid=[]) for s in buffers}
    identities, audits, start = [], [], time.perf_counter()
    for i, (case, row, geo) in enumerate(zip(cases, manifests, geometry)):
        assert case['name'] == row['id'] == geo['id'] and row['sample_index'] == geo['sample_index'] == i
        assert case['camera'] == geo['declared_camera']
        assert max(abs(geo['actual_camera_location_m'][j]-case['camera'][k]) for j, k in enumerate(('x','y','z'))) < .002
        for planned, actual in zip(case['objects'], geo['objects']):
            assert planned['name'] == actual['name']
            assert np.allclose(planned['center_m'], actual['render_bounds_center_m'], atol=.002)
            assert np.allclose(planned['size_m'], 2*np.asarray(actual['render_bounds_extent_m']), atol=.002)
        rp = cap/'observations'/row['rgb_path']
        dp = cap/'evaluator'/geo['native_path']
        assert sha(rp) == row['rgb_sha256'] == geo['rgb_sha256'] and sha(dp) == geo['native_sha256']
        image = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
        assert image.shape == (360, 640, 3)
        rgb[i] = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        depth = np.load(dp, allow_pickle=False)
        values, _ = simulate(sample_native(depth), 'query-occupancy/'+case['sensor_noise_key'], boxes)
        tof[i] = observation_tokens(values, boxes)
        baseline = decide(score_frame(boxes, values), .4071309640537889)
        bounds = camera_bounds(geo['objects'], case['camera'])
        classes, distances = geometric_labels(bounds)
        visible = visible_labels(depth, bounds)
        admissible = np.asarray(visible['unexplained']) == 0
        split = case['split']
        buffers[split].append(i)
        for key, value in dict(classes=classes, distances=distances, mask=visible['mask'],
                               coverage=visible['coverage'], valid=admissible).items():
            labels[split][key].append(value)
        identities.append(dict(index=i, id=case['name'], split=split, clip_id=case['clip_id'],
            frame_in_clip=case['frame_in_clip'], time_s=case['time_s'],
            base_group_id=case['base_group_id'], type_id=case['type_id'], layer=case['layer'],
            layout_relation=case['layout_relation'], rgb_sha256=row['rgb_sha256'], baseline=baseline))
        audits.append(dict(index=i, source_native_sha256=geo['native_sha256'],
            visible_pixels=visible['count'], unexplained_pixels=visible['unexplained'],
            visible_nearest=visible['visible_nearest'], query_label_valid=admissible.tolist()))
        if i % 144 == 0:
            print('MATERIALIZE', i, '/', n, flush=True)
    rgb.flush(); tof.flush()
    del rgb, tof
    for split in labels:
        arrays = {k: np.asarray(v, np.float32 if k in ('mask', 'coverage', 'distances') else np.int64 if k == 'classes' else bool)
                  for k, v in labels[split].items()}
        np.savez_compressed(out/f'labels/{split}.npz', indices=np.asarray(buffers[split]), **arrays)
    write(out/'observations/identities.json', identities)
    write(out/'labels/visibility-audit.json', audits)
    manifest = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    report = dict(status='PASS', frames=n, counts={k:len(v) for k,v in buffers.items()},
                  queries=QUERIES.tolist(), bin_edges_m=EDGES.tolist(),
                  source_spec_sha256=sha(root/'plan/spec.json'), hashes=manifest,
                  invalid_queries=sum(sum(not v for v in a['query_label_valid']) for a in audits),
                  elapsed_s=time.perf_counter()-start, backend='TASK_NOT_GPU_SUITABLE')
    write(out/'materialization.json', report)
    return report
