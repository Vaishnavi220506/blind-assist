"""Pixel residual -> existing positive ToF association, Radar unchanged.

Controlled-contrast Development mechanism, not complete object segmentation.
No labels/identities/range data enter the pixel frontend. No merged depth fit.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import mz115_spatial_allocation as tof
from mz124_measurement_geometry import metrics

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def foreground(image):
    """Per-row robust low-side quadratic handles smooth vignette/horizon.

    Parameters are fixed before the full labelled replay. Missing foreground
    is not used to clear a return; the caller falls back to old proposals.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape; x = np.linspace(-1., 1., w)
    design = np.stack([np.ones(w), x, x*x], 1)
    weights = np.ones_like(gray)
    for _ in range(6):
        lhs = np.einsum('hw,wi,wj->hij', weights, design, design)+np.eye(3)[None]*1e-6
        rhs = np.einsum('hw,hw,wi->hi', weights, gray, design)
        coefficient = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        background = coefficient@design.T; residual = gray-background
        center = np.median(residual, axis=1, keepdims=True)
        scale = np.maximum(1., 1.4826*np.median(np.abs(residual-center), axis=1, keepdims=True))
        weights = np.where(residual > center+2.5*scale, .01, 1.)
    mask = (residual > np.maximum(8., 4.*scale)).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = [[int(a), int(b), int(a+c), int(b+d)] for a, b, c, d, area in stats[1:n]
             if area >= 12 and c < .8*w]
    boxes.sort(key=lambda b: (-((b[2]-b[0])*(b[3]-b[1])), b))
    return boxes, mask


def allocation_alert(cached, allocated):
    return bool(cached['common_radar'] or cached['guard_events'] or
                any(tof.certain(e['coarse_xyz']) or tof.possible(e['localized_xyz']) for e in allocated))


def predict_frame(row, cached, image):
    boxes, mask = foreground(image)
    modified = dict(cached, proposals=boxes if boxes else cached['proposals'])
    allocated = tof.allocate(row, modified)
    return dict(candidate=allocation_alert(cached, allocated),
        candidate_state='ALERT' if allocation_alert(cached, allocated) else 'UNKNOWN',
        proposals=modified['proposals'], recovered_boxes=boxes, fallback=not bool(boxes),
        spatial_evidence=allocated, common_radar=cached['common_radar'],
        guard_events=cached['guard_events'], integrated_yaw_deg=cached['integrated_yaw_deg']), mask


def run(source, image_root, out):
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(parents=True); (out/'masks').mkdir()
    cap, analysis = source/'capture-v1', source/'analysis-v1'
    oldseal = read(analysis/'prediction-seal.json'); receipt = read(cap/'receipt.json')
    assert sha(cap/'raw.jsonl') == oldseal['raw_sha256']
    assert sha(analysis/'baseline-predictions.json') == oldseal['baseline_sha256']
    rows = [json.loads(x) for x in (cap/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    old = read(analysis/'baseline-predictions.json'); manifest = read(image_root/'image-manifest.json')
    assert len(rows) == len(old) == len(manifest['hashes']) == 288
    assert manifest['source_receipt_sha256'] == sha(cap/'receipt.json')
    first = cv2.imread(str(image_root/rows[0]['rgb_path'])); assert first is not None
    select_backend('batch-tensor', cpu=BackendCandidate('numpy-opencv-cpu', 'cpu',
        lambda: foreground(first), lambda _: DeviceObservation('cpu', 'host CPU', 'NumPy '+np.__version__+' OpenCV '+cv2.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE', record_path=out/'backend.json',
        capabilities={'reason':'Existing robust row solve and connected components have no equivalent implemented GPU pipeline'})
    predictions = []; identities = {}; tick = time.perf_counter()
    for row, cached in zip(rows, old):
        path = (image_root/row['rgb_path']).resolve(); assert path.is_relative_to(image_root.resolve())
        assert sha(path) == manifest['hashes'][row['rgb_path']] == receipt['hashes'][row['rgb_path']]
        image = cv2.imread(str(path)); assert image is not None and image.shape == (360, 640, 3)
        original = tof.allocate(row, cached)
        assert json.loads(json.dumps(original)) == cached['spatial_evidence'], 'Inherited allocation not reproduced: '+row['id']
        assert allocation_alert(cached, original) == cached['candidate']
        result, mask = predict_frame(row, cached, image)
        predictions.append(result); identities[row['id']] = sha(path)
        assert cv2.imwrite(str(out/'masks'/(row['id']+'.png')), mask)
    seconds = time.perf_counter()-tick
    write(out/'predictions.json', predictions)
    write(out/'prediction-seal.json', dict(raw_sha256=sha(cap/'raw.jsonl'), baseline_sha256=sha(analysis/'baseline-predictions.json'),
        rgb_sha256=identities, predictions_sha256=sha(out/'predictions.json'), code_sha256=sha(__file__),
        allocation_code_sha256=sha(Path(tof.__file__)), seconds=seconds, frames=288,
        authority='CONSUMED_OBSERVABLE_PREDICTIONS_BEFORE_THIS_TRUTH_PARSE', no_new_training=True))
    # Evaluator labels are never passed to pixel extraction or allocation.
    report = read(analysis/'frame-report.json'); assert [r['id'] for r in report] == [r['id'] for r in rows]
    gt = [r['truth'] for r in report]; base = [p['candidate'] for p in old]; flags = [p['candidate'] for p in predictions]
    details = []
    for r, g, a, b in zip(rows, gt, base, flags):
        if a != b: details.append(dict(id=r['id'], truth=g, baseline=a, candidate=b))
    result = dict(status='OBSERVABLE_CORRECTION_REPLAY_COMPLETE', authority='CONSUMED_CONSTRUCTED_DEVELOPMENT',
        frames=288, baseline=metrics(rows, gt, base), candidate=metrics(rows, gt, flags), changed_frames=details,
        proposals_recovered_frames=sum(bool(p['recovered_boxes']) for p in predictions),
        old_proposal_frames=sum(bool(p['proposals']) for p in old),
        fallback_frames=sum(p['fallback'] for p in predictions), seconds=seconds)
    result['families'] = {}
    for family in sorted({r['family'] for r in report}):
        ix = [i for i, r in enumerate(report) if r['family']==family]
        result['families'][family] = {name:metrics([rows[i] for i in ix], [gt[i] for i in ix], [p[i] for i in ix])
                                     for name, p in [('baseline', base), ('candidate', flags)]}
    write(out/'summary.json', result)
    write(out/'completion.json', dict(status='PASS', summary_sha256=sha(out/'summary.json'),
        resource_state='Foreground CPU process exits; no worker job or accelerator allocation'))
    print(json.dumps({k:result[k] for k in ('baseline','candidate','proposals_recovered_frames','old_proposal_frames','seconds')}, indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','image-root','output'):parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args();run(args.source,args.image_root,args.output)
