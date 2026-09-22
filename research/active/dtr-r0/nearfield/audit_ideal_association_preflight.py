"""Check provenance/coverage before a whole-target oracle; never make alerts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ba_camera_corridor import sample_indices, sample_native

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
OUT = ROOT / 'artifacts.local/work/ba-ideal-association-preflight-20260921'
PROTOCOL = HERE / 'IDEAL_ASSOCIATION_PREFLIGHT_20260921.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def mask_adapter(native, geometry):
    """Evaluator-only: return a 2D mask, never depths/bounds/target truth.

    Exactly the source's reconstructed visible mask, not renderer instance IDs.
    """
    obj = next(o for o in geometry['objects'] if o['name'] == 'target')
    centre = np.asarray(obj['render_bounds_center_m'])
    half = np.asarray(obj['render_bounds_extent_m'])
    camera = geometry['declared_camera']
    yy, xx = np.mgrid[:360, :640]
    focal = 640 / (2*np.tan(np.deg2rad(50)))
    world = np.stack([native+camera['x'],
        (xx+.5-320)/focal*native+camera['y'],
        camera['z']-(yy+.5-180)/focal*native], -1)
    return (np.isfinite(native) & (native > 0) &
        np.all((world >= centre-half-.02) & (world <= centre+half+.02), axis=-1))


def coverage(mask, boxes, values, traces):
    """Accept 2D identities plus public ranges only; no target geometry/depth."""
    target = sample_native(mask.astype(np.float32)).astype(bool)
    footprint = np.zeros((192, 256), bool)
    observed = np.zeros_like(footprint)
    contributors = np.zeros_like(footprint)
    owned = []
    for z, (box, value, trace) in enumerate(zip(boxes, values, traces)):
        y0, x0, y1, x1 = map(int, box)
        assert not footprint[y0:y1, x0:x1].any()
        footprint[y0:y1, x0:x1] = True
        assert trace['zone_id'] == z
        assert trace['observed'] == bool(np.isfinite(value))
        indices = np.asarray(trace['pixel_indices'], dtype=np.int64)
        assert len(indices) == len(np.unique(indices))
        assert np.all((indices >= 0) & (indices < 192*256))
        assert np.all((indices//256 >= y0) & (indices//256 < y1))
        assert np.all((indices%256 >= x0) & (indices%256 < x1))
        if np.isfinite(value):
            assert trace['distance_m'] == float(value) and len(indices) > 0
            observed[y0:y1, x0:x1] = True
        else:
            assert trace['distance_m'] is None and len(indices) == 0
        contributors.ravel()[indices] = True
        owned.append(int(target.ravel()[indices].sum()))
    returned = target & contributors
    uncovered = target & ~contributors
    reasons = dict(outside_footprint=int((target & ~footprint).sum()),
        no_observed_zone=int((target & footprint & ~observed).sum()),
        observed_zone_noncontributor=int((target & observed & ~contributors).sum()))
    assert sum(reasons.values()) == int(uncovered.sum())
    sy, sx = sample_indices()
    native_contributors = np.zeros_like(mask)
    yy, xx = np.nonzero(returned)
    native_contributors[sy[yy], sx[xx]] = True
    assert int(native_contributors.sum()) == int(returned.sum())
    return dict(native_target_pixels=int(mask.sum()), sampled_target_pixels=int(target.sum()),
        returned_target_pixels=int(returned.sum()), unreturned_sampled_target_pixels=int(uncovered.sum()),
        unreturned_native_target_pixels=int((mask & ~native_contributors).sum()),
        sampled_target_fully_covered=bool(target.any() and not uncovered.any()),
        native_target_fully_covered=bool(mask.any() and not (mask & ~native_contributors).any()),
        target_return_zones=sum(n > 0 for n in owned), missing_reasons=reasons,
        reconstructed_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())


def summarize(rows):
    fields = ('native_target_pixels', 'sampled_target_pixels', 'returned_target_pixels',
        'unreturned_sampled_target_pixels', 'unreturned_native_target_pixels')
    return dict(frames=len(rows), totals={k:sum(r[k] for r in rows) for k in fields},
        frames_without_target_return=sum(r['returned_target_pixels'] == 0 for r in rows),
        sampled_fully_covered_frames=sum(r['sampled_target_fully_covered'] for r in rows),
        native_fully_covered_frames=sum(r['native_target_fully_covered'] for r in rows),
        missing_reasons={k:sum(r['missing_reasons'][k] for r in rows)
            for k in ('outside_footprint', 'no_observed_zone', 'observed_zone_noncontributor')})


def run():
    assert not OUT.exists(), 'Immutable one-shot output already exists'
    started = time.perf_counter()
    seal = read(SOURCE/'observation-seal.json')
    assert sha(SOURCE/'protocol.json') == seal['protocol_sha256']
    for name, digest in seal['hashes'].items():
        assert sha(SOURCE/name) == digest, name
    OUT.mkdir(parents=True)
    inputs = [SOURCE/n for n in ('observation-seal.json', 'observations.npz', 'private-lineage.json',
        'capture/evaluator/geometry.json', 'source-admission.json')]+[PROTOCOL, Path(__file__)]
    write(OUT/'protocol.json', dict(id=OUT.name, time_utc=datetime.now(timezone.utc).isoformat(),
        scope='CONSUMED_DEVELOPMENT_PREFLIGHT', inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        backend='CPU', reason='TASK_NOT_GPU_SUITABLE', protected_test_access=False))
    geos = read(SOURCE/'capture/evaluator/geometry.json')
    lineage = read(SOURCE/'private-lineage.json')
    admission = read(SOURCE/'source-admission.json')['frames']
    with np.load(SOURCE/'observations.npz', allow_pickle=False) as arrays:
        boxes, ranges = arrays['boxes'], arrays['ranges']
    assert len(geos) == len(lineage) == len(admission) == len(ranges) == 1152
    rows = []
    for i, (geo, private, admitted, values) in enumerate(zip(geos, lineage, admission, ranges)):
        assert private['id'] == admitted['id'] == f'f{i:04d}'
        path = SOURCE/'capture/evaluator'/geo['native_path']
        assert sha(path) == private['native_sha256'] == geo['native_sha256']
        native = np.load(path, allow_pickle=False)
        assert native.shape == (360, 640)
        mask = mask_adapter(native, geo)
        assert int(mask.sum()) == admitted['visible_target_pixels']
        # Neither native values, 3D bounds nor truth are passed to coverage().
        allowed = [{k:t[k] for k in ('zone_id', 'observed', 'distance_m', 'pixel_indices')}
            for t in private['traces']]
        descriptor = coverage(mask, boxes, values, allowed)
        rows.append(dict(id=private['id'], native_sha256=geo['native_sha256'], **descriptor))
        if i % 384 == 0:
            print('COVERAGE', i, '/1152', flush=True)
    write(OUT/'coverage.json', rows)
    write(OUT/'coverage-seal.json', dict(protocol_sha256=sha(OUT/'protocol.json'),
        coverage_sha256=sha(OUT/'coverage.json'), native_hashes_verified=1152,
        source_mask_counts_reproduced=1152, classifier_executed=False))
    # Only after sealing the non-label coverage descriptors: subset evaluation.
    frames = read(SOURCE/'frame-results.json')
    evaluation_seal = read(SOURCE/'evaluation-seal.json')
    assert sha(SOURCE/'frame-results.json') == evaluation_seal['hashes']['frame-results.json']
    axis_root = ROOT/'artifacts.local/work/ba-axis-evidence-20260921'
    assert sha(axis_root/'joined-frame-results.json') == read(axis_root/'analysis-seal.json')['hashes']['joined-frame-results.json']
    axes = {r['source_id']:r for r in read(axis_root/'joined-frame-results.json')}
    subsets = {'all': rows}
    annotated = []
    for row, frame in zip(rows, frames):
        assert row['id'] == frame['id'] and frame['truth'] == axes[row['id']]['truth']
        added = frame['flags']['C_current'] and not frame['flags']['A_current']
        label = ('rescued_'+frame['layout_relation'] if frame['truth'] else axes[row['id']]['axes']['category']) if added else 'not_incremental'
        if added:
            subsets.setdefault(label, []).append(row)
        annotated.append({**row, 'subset':label, 'truth':frame['truth'], 'group':frame['base_group_id']})
    result = dict(status='NOT_EVALUABLE_WHOLE_TARGET_ORACLE',
        reasons=['Visible mask is reconstructed geometry/depth attribution, not independently rendered instance segmentation.',
            'Winning-return intervals do not constrain noncontributing visible or occluded surfaces.',
            'Sampled contributor coverage cannot certify continuous whole-object exclusion without additional assumptions.'],
        subsets={k:summarize(v) for k,v in subsets.items()},
        groups={g:summarize([r for r in annotated if r['group']==g]) for g in sorted({r['group'] for r in annotated})},
        classifier_executed=False, alerts_changed=False, threshold_selected=False,
        protected_test_access=False, elapsed_s=time.perf_counter()-started)
    write(OUT/'evaluated-coverage.json', annotated)
    write(OUT/'result.json', result)
    write(OUT/'evaluation-seal.json', {n:sha(OUT/n) for n in
        ('protocol.json', 'coverage.json', 'coverage-seal.json', 'evaluated-coverage.json', 'result.json')})
    print(json.dumps(result['subsets'], indent=2), flush=True)


if __name__ == '__main__':
    run()
