"""Evaluate sealed echo-assignment predictions once; all native depth stays here."""
from collections import Counter, defaultdict
import time

import cv2
import numpy as np

import ba_nfo_depthpro_echo as run
import ba_nfo_matched as m
import evaluate_ba_nfo_depthpro as shared
from ba_nfo_frozen_transfer import evaluation_domains, gates

ARMS = ('nfo', 'native', 'global', 'layered')
PAIRS = (('nfo', 'global'), ('nfo', 'layered'), ('native', 'global'),
         ('native', 'layered'), ('global', 'layered'))


def attribution(reference, boxes, labels, support, receipt, domains, predicted):
    """Distance compatibility is an evaluator proxy, NOT true echo source ID."""
    known = domains['full']; near = known & (reference < 2.)
    supported = support[labels] > 0
    counts = Counter()
    for anchor in receipt['anchors']:
        y0, x0, y1, x1 = boxes[anchor['zone']]
        sl = np.s_[y0:y1, x0:x1]
        selected = labels[sl] == anchor['region']
        value = anchor['observed_m']
        compatible = known[sl] & (np.abs(reference[sl]-value) <= 3*(.01+.02*value)+.1)
        counts['paired_returns'] += 1
        counts['returns_below2m'] += int(value < 2.)
        counts['zone_has_compatible_pixels'] += int(compatible.any())
        counts['selected_has_compatible_pixels'] += int((selected & compatible).any())
        counts['compatible_exists_only_elsewhere'] += int(compatible.any() and not (selected & compatible).any())
        counts['selected_reference_known_pixels'] += int((selected & known[sl]).sum())
        counts['selected_reference_near_pixels'] += int((selected & near[sl]).sum())
        counts['selected_compatible_pixels'] += int((selected & compatible).sum())
        counts['zone_compatible_pixels'] += int(compatible.sum())
        if value < 2.:
            counts['near_returns_selected_no_true_near'] += int(not (selected & near[sl]).any())
    coverage = {}
    for domain in ('full', 'mixed', 'far_small', 'outside'):
        mask = domains[domain]
        positive = mask & near
        coverage[domain] = dict(known_pixels=int(mask.sum()), supported_pixels=int((mask & supported).sum()),
            near_pixels=int(positive.sum()), supported_near_pixels=int((positive & supported).sum()),
            unsupported_near_pixels=int((positive & ~supported).sum()),
            layered_fn_supported=int((positive & supported & ~predicted).sum()),
            layered_fn_unsupported=int((positive & ~supported & ~predicted).sum()))
    return dict(return_consistency=dict(counts), coverage=coverage)


def main():
    output = run.OUT
    assert not (output/'results.json').exists(), 'Preserve completed frozen evaluation'
    protocol = run.read(output/'protocol.json'); seal = run.read(output/'prediction-seal.json')
    assert seal['status'] == 'COMPLETE' and len(seal['outputs']) == 500
    assert seal['protocol_sha256'] == run.sha(output/'protocol.json')
    assert seal['source_sha256'] == protocol['source_hashes']['inference'] == run.sha(run.__file__)
    assert run.sha(output/'manifest.json') == protocol['source_hashes']['original_manifest']
    assert run.sha(output/'observations.json') == protocol['source_hashes']['observations']
    assert run.sha(shared.BASE/'manifest.json') == shared.BASE_MANIFEST_SHA
    assert run.sha(shared.BASE/'predictions.npz') == shared.BASE_PREDICTION_SHA
    old = run.read(run.OLD/'results.json')
    rows = run.read(output/'manifest.json')
    observed = {r['id']: r for r in run.read(output/'observations.json')}
    with np.load(shared.BASE/'predictions.npz', allow_pickle=False) as data:
        nfo = np.unpackbits(data['masks'][:, 0], axis=1, bitorder='little').reshape(500, 192, 256).astype(bool)
    total = defaultdict(lambda: np.zeros(4, np.int64))
    families = defaultdict(lambda: np.zeros(4, np.int64))
    paired = defaultdict(Counter)
    boundary = defaultdict(lambda: np.zeros(4, np.int64))
    si = defaultdict(lambda: np.zeros(4, np.int64))
    consistent = Counter(); cover = defaultdict(Counter)
    support_totals = Counter(); frames = []; unknown = 0
    started = time.perf_counter()
    for i, row in enumerate(rows):
        receipt = seal['outputs'][row['id']]
        path = output/'predictions'/f'{row["id"]}.npz'
        assert run.sha(path) == receipt['sha256']
        assert receipt['protocol_sha256'] == seal['protocol_sha256']
        native_path = run.ROOT/observed[row['id']]['depth_path']
        assert run.sha(native_path) == observed[row['id']]['depth_sha256'] == receipt['depth_sha256']
        source = run.SOURCE/row['prepared']; assert run.sha(source) == row['sha256']
        with np.load(source, allow_pickle=False) as data:
            a = {k: data[k].copy() for k in ('depth', 'boxes', 'values')}
        np.testing.assert_array_equal(a['boxes'], observed[row['id']]['boxes'])
        np.testing.assert_allclose(a['values'], [np.nan if v is None else v for v in observed[row['id']]['values']], equal_nan=True)
        with np.load(native_path, allow_pickle=False) as data:
            depths = dict(native=data['depth'].copy())
        with np.load(path, allow_pickle=False) as data:
            depths.update(global_=data['global_depth'].copy(), layered=data['layered_depth'].copy())
            labels, support = data['labels'].copy(), data['support'].copy()
        depths['global'] = depths.pop('global_')
        assert all(d.shape == (192, 256) and np.isfinite(d).all() and (d > 0).all() for d in depths.values())
        truth, domains = evaluation_domains(a); known = domains['full']
        unknown += int((~known).sum())
        pred = dict(nfo=nfo[i], **{arm: d < 2. for arm, d in depths.items()})
        frame = dict(id=row['id'], scene=row['scene'], family=row['family'], metrics={}, paired={},
                     near_mask_boundary={}, scale_invariant_boundary={})
        for arm in ARMS:
            frame['metrics'][arm] = {}
            for domain, mask in domains.items():
                counts = m.counts(pred[arm], truth, mask)
                total[arm, domain] += counts; families[row['family'], arm, domain] += counts
                frame['metrics'][arm][domain] = m.metrics(counts)
            bc = shared.boundary_counts(shared.near_edges(pred[arm], known), shared.near_edges(truth, known), tolerance=1)
            boundary[arm] += bc; frame['near_mask_boundary'][arm] = shared.boundary_metrics(bc)
            if arm != 'nfo':
                sc, _ = shared.si_counts(depths[arm], a['depth'], known)
                si[arm] += sc; frame['scale_invariant_boundary'][arm] = shared.boundary_metrics(sc)
        for before, after in PAIRS:
            name = before+'->'+after; frame['paired'][name] = {}
            for domain, mask in domains.items():
                changes = shared.transitions(pred[before], pred[after], truth, mask)
                paired[name, domain].update(changes); frame['paired'][name][domain] = changes
        audit = attribution(a['depth'], a['boxes'], labels, support, receipt, domains, pred['layered'])
        consistent.update(audit['return_consistency'])
        for domain, counters in audit['coverage'].items():
            cover[domain].update(counters)
        support_totals.update({k: receipt[k] for k in ('regions', 'anchored_regions', 'anchored_pixels')})
        support_totals['valid_public_returns'] += int(np.isfinite(a['values']).sum())
        frame['attribution'] = audit
        frame['spatial_support'] = {k: receipt[k] for k in ('regions', 'anchored_regions', 'anchored_pixels', 'global_scale')}
        frames.append(frame)
        if (i+1) % 100 == 0:
            print('ECHO_EVALUATED', i+1, '/500', flush=True)
    assert unknown == 350140
    metrics = {arm: {d: m.metrics(c) for (aa, d), c in total.items() if aa == arm} for arm in ARMS}
    for arm in ('nfo', 'native'):
        for domain in metrics[arm]:
            for key in ('tp', 'fp', 'fn', 'tn'):
                assert metrics[arm][domain][key] == old['metrics'][arm][domain][key], (arm, domain, key)
    for (pair, domain), changes in paired.items():
        before, after = pair.split('->'); a, b = metrics[before][domain], metrics[after][domain]
        assert b['tp']-a['tp'] == changes['rescued_fn']-changes['lost_tp']
        assert b['fp']-a['fp'] == changes['added_fp']-changes['removed_fp']
    g = {arm: gates(metrics[arm], metrics['nfo']) for arm in ('global', 'layered')}
    global_, layer = metrics['global'], metrics['layered']
    direction = dict(far_small_recall=layer['far_small']['recall']-global_['far_small']['recall'],
        far_small_iou=layer['far_small']['iou']-global_['far_small']['iou'],
        mixed_recall=layer['mixed']['recall']-global_['mixed']['recall'],
        pure_far_fp_reduction=global_['pure_far']['fp']-layer['pure_far']['fp'])
    dominance = all(x >= 0 for x in direction.values()) and any(x > 0 for x in direction.values())
    result = dict(status='COMPLETE', frames=500, families=6, scenes=52, unknown_pixels=unknown,
        metrics=metrics, gates=g, all_task_gates={arm: all(checks.values()) for arm, checks in g.items()},
        layer_vs_global=dict(oriented_deltas=direction, weakly_dominates_with_one_strict=dominance,
                             joint_eligible=dominance and all(g['layered'].values())),
        paired={pair: {d: dict(c) for (name, d), c in paired.items() if name == pair} for pair in (a+'->'+b for a, b in PAIRS)},
        family_metrics={f: {arm: {d: m.metrics(c) for (ff, aa, d), c in families.items() if ff == f and aa == arm}
                             for arm in ARMS} for f in sorted({r['family'] for r in rows})},
        near_mask_boundary={arm: shared.boundary_metrics(c) for arm, c in boundary.items()},
        scale_invariant_boundary={arm: shared.boundary_metrics(c) for arm, c in si.items()},
        attribution=dict(return_consistency=dict(consistent), coverage={d: dict(c) for d, c in cover.items()},
                         support_totals=dict(support_totals), definition=protocol['attribution']),
        g5=dict(status='NOT_EVALUABLE_NO_FROZEN_TOF', prior_rgb_only_result=str(run.OLD/'diagnostic-results.json'),
                prior_result_sha256=run.sha(run.OLD/'diagnostic-results.json')),
        protocol_sha256=run.sha(output/'protocol.json'), prediction_seal_sha256=run.sha(output/'prediction-seal.json'),
        evaluator_hashes={str(p.relative_to(run.ROOT)): run.sha(p) for p in
                         (run.Path(__file__), run.Path(shared.__file__), run.Path(m.__file__),
                          run.Path(__file__).with_name('ba_nfo_frozen_transfer.py'),
                          run.Path(__file__).with_name('ba_nfo_zcr.py'))},
        baseline_counts_exact=True, training_updates=0, model_calls=0, original_test_frames_read=0,
        backend='CPU TASK_NOT_GPU_SUITABLE: saved-array scalar counts, connected-region attribution and morphology',
        evaluation_seconds=time.perf_counter()-started, limits=protocol['limits'])
    run.write(output/'frame-results.json', frames); run.write(output/'results.json', result)
    print('ECHO_RESULT', {k: result[k] for k in ('gates', 'all_task_gates', 'layer_vs_global', 'evaluation_seconds')}, flush=True)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
