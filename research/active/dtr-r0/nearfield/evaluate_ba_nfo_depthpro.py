"""Frozen Depth Pro saved-output diagnostic; evaluator truth never enters inference.

Near-mask contours depend on the physical 2m cutoff. Directed ratio edges are
scale invariant, but NFO has no comparable continuous depth output. The two
boundary quantities are deliberately kept separate. No model calls or tuning.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import ba_nfo_matched as m
from ba_nfo_frozen_transfer import evaluation_domains, gates
from audit_ba_nfo_spatial_tradeoff import transitions

OUT = m.ROOT / 'artifacts.local/work/ba-nfo-depthpro-20260919'
BASE = m.ROOT / 'artifacts.local/work/ba-nfo-frozen-transfer500-20260919'
MONO = m.OLD / 'predictions/mono'
ARMS = ('nfo', 'low', 'native')
RATIO = 1.25
BASE_MANIFEST_SHA = '45e67d6c07e4b14ef438ad84a404965079940c4a153015fa62c4765971d6f893'
BASE_PREDICTION_SHA = '8b381bb07f6b02180b6d934ea446a01329b9aaa2c9d9714efc6795611b724b10'
BASE_RESULT_SHA = '9baee43b4379f259c9fac70fb5dc86c4996600639847789cd3ba2c5523c625f5'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def adjacent(array):
    """Two undirected lattice-edge planes; orientation retained during matching."""
    return ((array[:, :-1], array[:, 1:]), (array[:-1], array[1:]))


def ratio_edges(depth, known):
    """Four directed edge planes, excluding any edge touching reference UNKNOWN.

    Strict >1.25 means equal-ratio pairs are not discontinuities. A reversed
    foreground/background order is both a false and a missed directed edge.
    """
    result = []
    for (a, b), (ka, kb) in zip(adjacent(depth), adjacent(known)):
        valid = ka & kb
        result.extend((valid & (a > RATIO*b), valid & (b > RATIO*a)))
    return tuple(result)


def near_edges(near, known):
    return tuple((a != b) & ka & kb
                 for (a, b), (ka, kb) in zip(adjacent(near), adjacent(known)))


def boundary_counts(predicted, reference, tolerance=0):
    """Precision and recall matching counts; tolerance is Chebyshev lattice px.

    Each orientation is matched separately. Reference UNKNOWN pairs have
    already been masked before dilation; unknown pixels create no boundaries.
    """
    counts = np.zeros(4, np.int64)
    for p, r in zip(predicted, reference):
        if tolerance:
            kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
            pd = cv2.dilate(p.astype(np.uint8), kernel).astype(bool)
            rd = cv2.dilate(r.astype(np.uint8), kernel).astype(bool)
        else:
            pd, rd = p, r
        counts += [p.sum(), r.sum(), (p & rd).sum(), (r & pd).sum()]
    return counts


def boundary_metrics(counts):
    predicted, reference, matched_prediction, matched_reference = map(int, counts)
    precision = matched_prediction/predicted if predicted else None
    recall = matched_reference/reference if reference else None
    # A one-sided empty set is a failed contour; two empty sets are unevaluable.
    f1 = (2*precision*recall/(precision+recall)
          if precision is not None and recall is not None and precision+recall
          else (0. if predicted or reference else None))
    return dict(predicted=predicted, reference=reference,
                matched_prediction=matched_prediction, matched_reference=matched_reference,
                precision=precision, recall=recall, f1=f1)


def si_counts(depth, truth_depth, known):
    pred, ref = ratio_edges(depth, known), ratio_edges(truth_depth, known)
    counts = boundary_counts(pred, ref)
    crossing = near_edges(truth_depth < 2., known)
    target = tuple(edge & crossing[i//2] for i, edge in enumerate(ref))
    target_n = sum(int(x.sum()) for x in target)
    target_hits = sum(int((p & r).sum()) for p, r in zip(pred, target))
    return counts, np.array([target_hits, target_n], np.int64)


def near_crossing_metrics(counts):
    hits, reference = map(int, counts)
    return dict(hits=hits, reference=reference, recall=hits/reference if reference else None)


def load_depth(path, require_float32=False):
    with np.load(path, allow_pickle=False) as data:
        depth = data['depth'].copy()
    assert depth.shape == (192, 256), (str(path), depth.shape)
    if require_float32:
        assert depth.dtype == np.float32, (str(path), depth.dtype)
    # Do not silently remove unknown predictions from reference denominators.
    assert np.isfinite(depth).all() and (depth > 0).all(), str(path)
    return depth


def frame_macro(frames, section, arms):
    """Unweighted frame means over nonempty metrics, with frame counts shown."""
    result = {}
    for arm in arms:
        result[arm] = {}
        for key in ('precision', 'recall', 'f1'):
            values = [r[section][arm][key] for r in frames
                      if arm in r.get(section, {}) and r[section][arm][key] is not None]
            result[arm][key] = dict(mean=float(np.mean(values)) if values else None,
                                   evaluable_frames=len(values))
    return result


def evaluate(output=OUT):
    output = Path(output)
    assert not (output/'results.json').exists(), 'Preserve completed frozen evaluation'
    read(output/'protocol.json')  # Require the root-owned pre-inference freeze.
    prediction_seal = read(output/'prediction-seal.json')
    assert prediction_seal['status'] == 'COMPLETE' and len(prediction_seal['outputs']) == 1036
    assert prediction_seal['protocol_sha256'] == m.sha(output/'protocol.json')
    launch = read(output/'launch-seal.json')
    assert launch['protocol_sha256'] == prediction_seal['protocol_sha256']
    for relative, digest in launch['hashes'].items():
        assert m.sha(m.ROOT/relative) == digest, relative
    rows, baseline_rows = read(output/'manifest.json'), read(BASE/'manifest.json')
    assert m.sha(BASE/'manifest.json') == BASE_MANIFEST_SHA
    assert m.sha(BASE/'predictions.npz') == BASE_PREDICTION_SHA
    assert m.sha(BASE/'results.json') == BASE_RESULT_SHA
    assert len(rows) == len(baseline_rows) == 500
    assert [r['id'] for r in rows] == [r['id'] for r in baseline_rows]
    assert all(r['split'] == 'val' for r in rows)
    for row, original in zip(rows, baseline_rows):
        assert all(row[k] == original[k] for k in ('id', 'prepared', 'sha256', 'family', 'scene'))
    expected = read(BASE/'results.json')['metrics']['nfo']
    mono_seal = read(m.OLD/'prediction-seal-mono.json')['files']
    mono_ids = [r['id'] for r in rows if (MONO/(r['id']+'.npz')).exists()]
    assert len(mono_ids) == 70, len(mono_ids)
    with np.load(BASE/'predictions.npz', allow_pickle=False) as archive:
        nfo = np.unpackbits(archive['masks'][:, 0], axis=1, bitorder='little')
    assert nfo.shape == (500, 192*256)
    nfo = nfo.reshape(500, 192, 256).astype(bool)
    total = defaultdict(lambda: np.zeros(4, np.int64))
    family = defaultdict(lambda: np.zeros(4, np.int64))
    paired = defaultdict(lambda: defaultdict(int))
    boundary = defaultdict(lambda: np.zeros(4, np.int64))
    si_total = defaultdict(lambda: np.zeros(4, np.int64))
    si_matched = defaultdict(lambda: np.zeros(4, np.int64))
    crossing_total = defaultdict(lambda: np.zeros(2, np.int64))
    crossing_matched = defaultdict(lambda: np.zeros(2, np.int64))
    frames, unknown = [], 0
    started = time.perf_counter()
    for index, row in enumerate(rows):
        path = m.OLD/row['prepared']
        assert m.sha(path) == row['sha256'], row['id']
        with np.load(path, allow_pickle=False) as source:
            a = {k: source[k].copy() for k in ('depth', 'boxes', 'values')}
        truth, domains = evaluation_domains(a)
        known = domains['full']; unknown += int((~known).sum())
        paths = {arm: output/'predictions'/arm/(row['id']+'.npz') for arm in ARMS[1:]}
        for arm, path in paths.items():
            receipt = prediction_seal['outputs'][f'predictions/{arm}/{row["id"]}']
            assert receipt['sha256'] == m.sha(path)
            assert receipt['protocol_sha256'] == prediction_seal['protocol_sha256']
        depth = {arm: load_depth(p, require_float32=True) for arm, p in paths.items()}
        predictions = dict(nfo=nfo[index], **{arm: d < 2. for arm, d in depth.items()})
        frame = dict(id=row['id'], scene=row['scene'], family=row['family'],
                     unknown_pixels=int((~known).sum()), metrics={}, paired={},
                     near_mask_boundary={}, scale_invariant_boundary={}, near_crossing_discontinuity={},
                     prediction_sha256={arm: m.sha(p) for arm, p in paths.items()})
        reference_boundary = near_edges(truth, known)
        for arm in ARMS:
            frame['metrics'][arm] = {}
            bc = boundary_counts(near_edges(predictions[arm], known), reference_boundary, tolerance=1)
            boundary[arm] += bc
            frame['near_mask_boundary'][arm] = boundary_metrics(bc)
            for domain, mask in domains.items():
                count = m.counts(predictions[arm], truth, mask)
                total[arm, domain] += count; family[row['family'], arm, domain] += count
                frame['metrics'][arm][domain] = m.metrics(count)
        for arm in ARMS[1:]:
            frame['paired'][arm] = {}
            for domain, mask in domains.items():
                changes = transitions(predictions['nfo'], predictions[arm], truth, mask)
                frame['paired'][arm][domain] = changes
                for key, value in changes.items():
                    paired[arm, domain][key] += value
        if row['id'] in mono_ids:
            mono_path = MONO/(row['id']+'.npz')
            assert m.sha(mono_path) == mono_seal[row['id']]
            depth['unidepth_cached'] = load_depth(mono_path)
            frame['prediction_sha256']['unidepth_cached'] = mono_seal[row['id']]
        for arm, d in depth.items():
            sc, nc = si_counts(d, a['depth'], known)
            frame['scale_invariant_boundary'][arm] = boundary_metrics(sc)
            frame['near_crossing_discontinuity'][arm] = near_crossing_metrics(nc)
            if arm in ARMS[1:]:
                si_total[arm] += sc; crossing_total[arm] += nc
            if row['id'] in mono_ids:
                si_matched[arm] += sc; crossing_matched[arm] += nc
        frames.append(frame)
        if (index+1) % 100 == 0:
            print('DEPTHPRO_EVALUATED', index+1, '/500', flush=True)
    assert unknown == 350140, unknown
    metrics = {arm: {domain: m.metrics(c) for (name, domain), c in total.items() if name == arm}
               for arm in ARMS}
    for domain, reference in expected.items():
        for key in ('tp', 'fp', 'fn', 'tn'):
            assert metrics['nfo'][domain][key] == reference[key], (domain, key)
    for (arm, domain), changes in paired.items():
        before, after = metrics['nfo'][domain], metrics[arm][domain]
        assert after['tp']-before['tp'] == changes['rescued_fn']-changes['lost_tp']
        assert after['fp']-before['fp'] == changes['added_fp']-changes['removed_fp']
    near_metrics = {arm: boundary_metrics(c) for arm, c in boundary.items()}
    si_paired_metrics = {arm: boundary_metrics(c) for arm, c in si_matched.items()}
    decisions = {}
    for arm in ARMS[1:]:
        checks = gates(metrics[arm], metrics['nfo'])
        contour = dict(near_mask_boundary_f1_plus_002=near_metrics[arm]['f1'] >= near_metrics['nfo']['f1']+.02,
                       matched70_si_f1_plus_002=si_paired_metrics[arm]['f1'] >= si_paired_metrics['unidepth_cached']['f1']+.02)
        decisions[arm] = dict(task_gates=checks, all_task_gates=all(checks.values()),
                             contour_gates=contour, contour_support=all(contour.values()),
                             joint_support=all(checks.values()) and all(contour.values()),
                             input_authority='same256x192_RGB_information_with_public_camera_rectification' if arm == 'low' else 'additional_native_RGB_detail_not_isolated_representation_effect')
    result = dict(status='COMPLETE', frames=500, families=6, scenes=52,
                  protocol_sha256=m.sha(output/'protocol.json'), manifest_sha256=m.sha(output/'manifest.json'),
                  evaluator_sha256=m.sha(__file__), baseline_manifest_sha256=BASE_MANIFEST_SHA,
                  baseline_prediction_sha256=BASE_PREDICTION_SHA, baseline_result_sha256=BASE_RESULT_SHA,
                  metrics=metrics, decisions=decisions, unknown_pixels=unknown, baseline_counts_exact=True,
                  family_metrics={f: {arm: {d: m.metrics(c) for (ff, aa, d), c in family.items() if ff == f and aa == arm}
                                      for arm in ARMS} for f in sorted({r['family'] for r in rows})},
                  paired={arm: {d: dict(c) for (aa, d), c in paired.items() if aa == arm} for arm in ARMS[1:]},
                  near_mask_boundary=dict(aggregate=near_metrics, frame_macro=frame_macro(frames, 'near_mask_boundary', ARMS),
                      definition='2m near-mask differing adjacent pairs; both endpoints reference-known; separate x/y edge planes; Chebyshev tolerance1px; physical-cutoff-dependent'),
                  scale_invariant_boundary=dict(all500={arm: boundary_metrics(c) for arm, c in si_total.items()},
                      matched70=si_paired_metrics, matched70_ids=mono_ids,
                      all500_frame_macro=frame_macro(frames, 'scale_invariant_boundary', ARMS[1:]),
                      matched70_frame_macro=frame_macro([r for r in frames if r['id'] in mono_ids], 'scale_invariant_boundary', (*ARMS[1:], 'unidepth_cached')),
                      definition='Directed4-neighbor optical-depth ratio strictly>1.25; exact same edge and order; both endpoints reference-known; no GT alignment',
                      nfo='NOT_EVALUABLE: threshold probabilities are not continuous depth; no pseudo-depth conversion'),
                  near_crossing_discontinuity=dict(all500={arm: near_crossing_metrics(c) for arm, c in crossing_total.items()},
                      matched70={arm: near_crossing_metrics(c) for arm, c in crossing_matched.items()},
                      definition='Recall of true directed ratio>1.25 edges crossing2m; prediction uses depth ratio only, independent of predicted metric scale'),
                  model_calls=0, training_updates=0, original_test_frames_read=0,
                  backend='CPU TASK_NOT_GPU_SUITABLE: saved-array scalar counts and morphology',
                  evaluation_seconds=time.perf_counter()-started,
                  limitations=['Consumed synthetic Development, not independent generalization or hardware evidence',
                               'Cached70 mono comparator is UniDepthV2 ViT-S with estimated camera, not DA V2',
                               'No equivalent scale-invariant NFO depth boundary metric',
                               'Public camera rectification changes sampling; low256 preserves original RGB information, not literal input pixels',
                               'Native condition adds input detail; only low256 is the same-information condition',
                               'Pixel/edge metrics do not measure obstacles, final alerts or safety'])
    m.write(output/'frame-results.json', frames)
    m.write(output/'results.json', result)
    print('DEPTHPRO_RESULT', json.dumps(dict(decisions=decisions, seconds=result['evaluation_seconds'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    evaluate(args.output)
