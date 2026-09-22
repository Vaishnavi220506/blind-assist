"""One fixed observation-only comparison: uniform vs spatially assigned ToF scale.

The seal command extracts only saved public boxes/values. The infer command
cannot read reference depth, scene labels, evaluation masks or NFO predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/ba-nfo-depthpro-echo-20260919'
OLD = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
SOURCE = ROOT/'artifacts.local/work/ba-nfo-20260919'
RATIO = 1.25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@njit(cache=False)
def merge_regions(z, left, right, order):
    """Ordered union-find; bound the whole region to prevent gradual-depth chains."""
    parent = np.arange(len(z))
    lo, hi = z.copy(), z.copy()
    for edge in order:
        a, b = left[edge], right[edge]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a == b:
            continue
        mn, mx = min(lo[a], lo[b]), max(hi[a], hi[b])
        if mx <= RATIO*mn:
            # Stable tie rule: the smallest row-major root wins.
            if b < a:
                a, b = b, a
            parent[b] = a
            lo[a], hi[a] = mn, mx
    for i in range(len(z)):
        j = i
        while parent[j] != j:
            j = parent[j]
        parent[i] = j
    return parent


def regions(depth):
    assert depth.ndim == 2 and np.isfinite(depth).all() and (depth > 0).all()
    ids = np.arange(depth.size).reshape(depth.shape)
    left = np.r_[ids[:, :-1].ravel(), ids[:-1].ravel()]
    right = np.r_[ids[:, 1:].ravel(), ids[1:].ravel()]
    z = depth.ravel().astype(np.float64)
    order = np.argsort(np.abs(np.log(z[left])-np.log(z[right])), kind='stable')
    roots = merge_regions(z, left, right, order)
    return np.unique(roots, return_inverse=True)[1].reshape(depth.shape).astype(np.int32)


def predict(depth, boxes, values):
    """No GT accepted. Both arms use identical once-only observable echo pairing."""
    labels = regions(depth)
    n = int(labels.max())+1
    logs = [[] for _ in range(n)]
    anchors = []
    for zone, (box, value) in enumerate(zip(boxes, values)):
        if value is None or not np.isfinite(value) or value <= 0:
            continue
        y0, x0, y1, x1 = map(int, box)
        assert 0 <= y0 < y1 <= depth.shape[0] and 0 <= x0 < x1 <= depth.shape[1]
        z, lab = depth[y0:y1, x0:x1].ravel(), labels[y0:y1, x0:x1].ravel()
        eligible = (z >= .1) & (z < 8.)
        if int(eligible.sum()) < 4:
            continue
        # Match the saved sensor proxy's strongest 10cm histogram, not the
        # whole region's energy (a sloping wall can span many weak bins).
        bins = np.minimum((z[eligible]/.1).astype(int), 79)
        weight = 1/np.maximum(z[eligible], .3)**2
        histogram = np.bincount(bins, weights=weight, minlength=80)
        peak = int(np.argmax(histogram))
        in_peak = bins == peak
        energy = np.bincount(lab[eligible][in_peak], weights=weight[in_peak], minlength=n)
        chosen = int(np.argmax(energy))  # Smallest region id wins an energy tie.
        selected = in_peak & (lab[eligible] == chosen)
        representative = float(np.mean(z[eligible][selected]))
        log_scale = float(np.log(float(value)/representative))
        logs[chosen].append(log_scale)
        anchors.append(dict(zone=zone, region=chosen, predicted_m=representative,
                            observed_m=float(value), log_scale=log_scale,
                            predicted_bin=peak,
                            predicted_energy_share=float(energy[chosen]/energy.sum())))
    global_scale = float(np.exp(np.median([a['log_scale'] for a in anchors]))) if anchors else 1.
    scales = np.array([np.exp(np.median(x)) if x else 1. for x in logs], dtype=np.float64)
    support = np.array([len(x) for x in logs], dtype=np.int32)
    result = dict(global_depth=(depth.astype(np.float64)*global_scale).astype(np.float32),
                  layered_depth=(depth.astype(np.float64)*scales[labels]).astype(np.float32),
                  labels=labels, scales=scales, support=support)
    assert all(np.isfinite(result[k]).all() and (result[k] > 0).all()
               for k in ('global_depth', 'layered_depth'))
    assert np.array_equal(result['layered_depth'][support[labels] == 0], depth[support[labels] == 0])
    return result, dict(regions=n, anchored_regions=int((support > 0).sum()),
                        anchored_pixels=int((support[labels] > 0).sum()),
                        global_scale=global_scale, anchors=anchors)


def seal():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'One frozen comparison only'
    rows = read(OLD/'manifest.json')
    original = read(OLD/'prediction-seal.json')
    assert original['status'] == 'COMPLETE' and len(rows) == 500
    observed = []
    for row in rows:
        source = SOURCE/row['prepared']
        assert sha(source) == row['sha256']
        # NPZ lazily loads only these two public arrays, not native depth/raw/mixed.
        with np.load(source, allow_pickle=False) as data:
            boxes, values = data['boxes'], data['values']
        assert boxes.shape == (64, 4) and values.shape == (64,)
        path = OLD/'predictions/native'/f'{row["id"]}.npz'
        digest = original['outputs'][f'predictions/native/{row["id"]}']['sha256']
        assert sha(path) == digest
        observed.append(dict(id=row['id'], depth_path=path.relative_to(ROOT).as_posix(),
                             depth_sha256=digest, boxes=boxes.tolist(),
                             values=[float(v) if np.isfinite(v) else None for v in values]))
    write(OUT/'observations.json', observed)
    write(OUT/'manifest.json', rows)
    protocol = dict(id=OUT.name, phase='EXPLORE_CONSUMED_SYNTHETIC_DEVELOPMENT',
        question='Can existing single-return ranges repair frozen Depth Pro near2m, and does spatial assignment beat uniform scale?',
        hypothesis='Preserve connected depth-coherent regions across zones; correct only the region predicted to dominate each echo.',
        arms=['original_nfo', 'unchanged_native_depthpro', 'global_scale', 'layered_scale'],
        region_rule='192x256 native cache; 4-neighbor edges sorted by abs(log depth difference), stable horizontal-then-vertical row-major order; union iff whole merged max/min<=1.25; smallest root wins; no size pruning, no RGB or GT segmentation.',
        pairing='Per valid public zone require>=4 predicted pixels in[0.1,8)m. Same fixed sensor proxy: select strongest0.1m bin by sum1/max(z,0.3)^2, nearest bin wins tie. Within that bin select region with highest energy, smallest id wins tie; representative is ordinary mean of that region intersection bin intersection zone. No candidate means unused return. Model-based attribution, not observed source identity.',
        scales='Same pairs for both arms. Global exp(median(log(return/representative))) across paired zones. Layered same median within region. Each paired zone one vote. No clipping, fitting or repeat pairing. Unanchored region retains exact original estimate and is separately counted as unsupported by ToF.',
        inputs='Frozen native Depth Pro depth192x256 and saved public boxes/values only; no second return, CNH, truth-derived mask, oracle scale or training.',
        physical_cutoff_m=2., regions_ratio=RATIO, frames=500,
        task_gates=dict(far_small_recall_min=.75, far_small_iou='>= original NFO', mixed_recall_min=.945, pure_far_fp='<= original NFO'),
        layer_contribution='Layered must weakly dominate global on all four task quantities and strictly improve at least one. Also report both against unchanged Depth Pro and NFO; task eligibility requires all four original gates.',
        attribution='Evaluator only after complete prediction seal: near/far composition of selected region; return-compatible true pixels abs(z-return)<=3*(0.01+0.02*return)+0.1. This loose distance-consistency proxy is not source/instance identity. Report near pixels on anchored vs unsupported regions and all changes.',
        metrics='Same11 original domains; fixed2m task, paired changes, all6 families, near-mask boundary tol1 and directed adjacent ratio>1.25. UNKNOWN excluded and counted, never far.',
        g5='NOT_EVALUABLE_NO_FROZEN_TOF: old18-view G5 root has RGB+native evaluator depth only. Keep prior RGB-only diagnostic; do not construct new ToF from truth.',
        budget='One fixed postprocessor over original500; no models, training, tuning, thresholds, fresh test, or automatic successor.',
        decisions='All gates plus layer dominance retain layered candidate for a separately authorized confirmation. Global only success favors simple scale. Neither success closes this exact recipe; inspect unsupported range vs incorrect association without declaring all layering or hardware impossible.',
        backend='CPU TASK_NOT_GPU_SUITABLE: sequential ordered union-find graph, per-zone medians and saved-array scalar audit. No dense learned inference; timing reported including JIT separately.',
        recovery='Mechanical defects only with logged receipts and unchanged scientific rules; completed outputs immutable.',
        source_hashes=dict(inference=sha(__file__), original_manifest=sha(OLD/'manifest.json'),
                           original_prediction_seal=sha(OLD/'prediction-seal.json'), observations=sha(OUT/'observations.json')),
        limits=['Consumed500 synthetic Development, not fresh confirmation or device/safety evidence',
                '1.25 bounds depth-coherent connected regions, not semantic objects; oblique surfaces may split and similar-depth surfaces may merge',
                'Synthetic energy-dominant single optical-z return differs from actual sensor; predicted scale errors can change bin dominance or leave no in-range candidate',
                'A same-depth compatible pixel cannot prove it generated the return; absence of compatibility is diagnostic, not physical impossibility'])
    write(OUT/'protocol.json', protocol)
    print('ECHO_PROTOCOL_SEALED', sha(OUT/'protocol.json'), flush=True)


def infer():
    protocol = read(OUT/'protocol.json')
    assert protocol['source_hashes']['inference'] == sha(__file__)
    assert protocol['source_hashes']['observations'] == sha(OUT/'observations.json')
    assert not (OUT/'prediction-seal.json').exists(), 'Completed run is immutable'
    paths = OUT/'predictions'; paths.mkdir(exist_ok=True)
    # Compile on a synthetic constant, outside cohort timing and without labels.
    began = time.perf_counter(); regions(np.ones((2, 3), dtype=np.float32))
    jit_seconds = time.perf_counter()-began
    receipts, model_seconds = {}, 0.
    started = time.perf_counter()
    for i, observation in enumerate(read(OUT/'observations.json')):
        path = ROOT/observation['depth_path']; assert sha(path) == observation['depth_sha256']
        with np.load(path, allow_pickle=False) as data:
            depth = data['depth'].copy()
        assert depth.shape == (192, 256) and depth.dtype == np.float32
        target = paths/f'{observation["id"]}.npz'
        assert not target.exists(), 'No overwritten scientific predictions'
        tick = time.perf_counter()
        result, receipt = predict(depth, observation['boxes'], observation['values'])
        elapsed = time.perf_counter()-tick; model_seconds += elapsed
        np.savez_compressed(target, **result)
        receipt.update(id=observation['id'], sha256=sha(target), inference_seconds=elapsed,
                       protocol_sha256=sha(OUT/'protocol.json'), depth_sha256=observation['depth_sha256'])
        write(target.with_suffix('.json'), receipt)
        receipts[observation['id']] = receipt
        if (i+1) % 100 == 0:
            print('ECHO_INFERRED', i+1, '/500', flush=True)
    write(OUT/'prediction-seal.json', dict(status='COMPLETE', outputs=receipts,
        protocol_sha256=sha(OUT/'protocol.json'), source_sha256=sha(__file__),
        model_calls=0, training_updates=0, backend=protocol['backend'], jit_seconds=jit_seconds,
        inference_seconds=model_seconds, wall_seconds=time.perf_counter()-started))
    print('ECHO_INFERENCE_COMPLETE', model_seconds, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('seal', 'infer'))
    args = parser.parse_args()
    {'seal': seal, 'infer': infer}[args.command]()
