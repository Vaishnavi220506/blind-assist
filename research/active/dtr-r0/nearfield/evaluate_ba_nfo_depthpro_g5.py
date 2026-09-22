"""Separate18-view G5 cache diagnostic at optical z<2m, never old alert scores.

Only this evaluator reads native geometry. The historic1054-pixel bar mask is
reconstructed unchanged on CPU once and used only to summarize saved depths.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

import ba_nfo_matched as m
import evaluate_ba_nfo_depthpro as shared
from near_field import Camera, NearFieldEncoder
from run_surface_support import tensors

OUT = shared.OUT
G5 = m.ROOT/'artifacts.local/nearfield/distinct-views-20260907-v1'
BASELINE_SHA = 'fa29b6ba9e2fecf94caffd0ca05732c579f17fe43ac2ef3b35dfa96057e010ec'
ARMS = ('da_v2', 'low', 'native')


def depth_summary(depth, mask):
    values = depth[mask]
    valid = np.isfinite(values) & (values > 0)
    known = values[valid]
    quantiles = np.quantile(known, [0, .05, .25, .5, .75, .95, 1]) if known.size else [None]*7
    return dict(reference_pixels=int(values.size), known_pixels=int(valid.sum()),
                unknown_pixels=int((~valid).sum()), near2m_pixels=int((known < 2.).sum()),
                near2m_fraction_all_reference=float((known < 2.).sum()/values.size) if values.size else None,
                optical_z_m={key: float(value) if value is not None else None
                             for key, value in zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'), quantiles)})


def bar_mask(native, camera):
    encoder = NearFieldEncoder(Camera(**camera), device='cpu')
    _, forward, _, band, _, _, support = tensors(encoder, native, None)
    mask = ((encoder.direction == 1) & support & (band == 2) & (forward <= 3)).numpy()
    assert int(mask.sum()) == 1054, int(mask.sum())
    return mask


def evaluate(output=OUT):
    output = Path(output)
    assert not (output/'diagnostic-results.json').exists(), 'Preserve completed diagnostic'
    protocol_hash = m.sha(output/'protocol.json')
    seal = shared.read(output/'prediction-seal.json')
    launch = shared.read(output/'launch-seal.json')
    assert seal['status'] == 'COMPLETE' and seal['protocol_sha256'] == protocol_hash
    assert launch['protocol_sha256'] == protocol_hash
    for rel, digest in launch['hashes'].items():
        assert m.sha(m.ROOT/rel) == digest, rel
    assert m.sha(G5/'predictions/result.json') == BASELINE_SHA
    original = shared.read(G5/'predictions/result.json')
    observations = shared.read(output/'diagnostic-observations.json')
    assert len(observations) == len(original['rows']) == 18
    assert [r['sample_index'] for r in original['rows']] == list(range(18))
    totals = defaultdict(lambda: np.zeros(4, np.int64))
    boundary = defaultdict(lambda: np.zeros(4, np.int64))
    scale_invariant = defaultdict(lambda: np.zeros(4, np.int64))
    near_crossing = defaultdict(lambda: np.zeros(2, np.int64))
    paired = defaultdict(lambda: defaultdict(int))
    cases, bar, verified, unknown = [], None, {}, 0
    started = time.perf_counter()
    torch.set_num_threads(1)
    for old, observation in zip(original['rows'], observations):
        index = old['sample_index']; identity = f'g5-{index:04d}'
        assert observation['id'] == identity and observation['case'] == old['case_name']
        assert observation['camera'] == old['camera']
        arrays = {}
        for kind in ('rgb', 'native', 'predicted'):
            path = Path(old[kind+'_path']); digest = old[kind+'_sha256']
            assert m.sha(path) == digest, str(path)
            verified[str(path)] = digest
            if kind != 'rgb':
                arrays[kind] = np.load(path, allow_pickle=False)
        assert m.sha(Path(observation['rgb'])) == observation['rgb_sha256'] == old['rgb_sha256']
        reference = arrays['native']; assert reference.shape == (360, 640)
        known = np.isfinite(reference) & (reference > 0)
        truth = known & (reference < 2.)
        unknown += int((~known).sum())
        depths = dict(da_v2=arrays['predicted'])
        hashes = dict(da_v2=old['predicted_sha256'])
        for arm in ('low', 'native'):
            path = output/'diagnostics'/arm/(identity+'.npz')
            receipt = shared.read(path.with_suffix('.json'))
            assert receipt == seal['outputs'][f'diagnostics/{arm}/{identity}']
            assert receipt['protocol_sha256'] == protocol_hash
            assert receipt['rgb_sha256'] == old['rgb_sha256']
            assert m.sha(path) == receipt['sha256']
            with np.load(path, allow_pickle=False) as data:
                depths[arm] = data['depth'].copy()
            assert depths[arm].dtype == np.float32
            hashes[arm] = receipt['sha256']
        assert all(d.shape == (360, 640) and np.isfinite(d).all() and (d > 0).all() for d in depths.values())
        case = dict(id=identity, case=old['case_name'], reference_known_pixels=int(known.sum()),
                    reference_unknown_pixels=int((~known).sum()), metrics={}, near_mask_boundary={},
                    scale_invariant_boundary={}, near_crossing_discontinuity={}, paired={},
                    source_native_sha256=old['native_sha256'], prediction_sha256=hashes)
        truth_edges = shared.near_edges(truth, known)
        for arm, depth in depths.items():
            count = m.counts(depth < 2., truth, known); totals[arm] += count
            case['metrics'][arm] = m.metrics(count)
            bc = shared.boundary_counts(shared.near_edges(depth < 2., known), truth_edges, tolerance=1)
            boundary[arm] += bc; case['near_mask_boundary'][arm] = shared.boundary_metrics(bc)
            sc, nc = shared.si_counts(depth, reference, known)
            scale_invariant[arm] += sc; near_crossing[arm] += nc
            case['scale_invariant_boundary'][arm] = shared.boundary_metrics(sc)
            case['near_crossing_discontinuity'][arm] = shared.near_crossing_metrics(nc)
        for arm in ('low', 'native'):
            changes = shared.transitions(depths['da_v2'] < 2., depths[arm] < 2., truth, known)
            case['paired'][arm] = changes
            for key, value in changes.items():
                paired[arm][key] += value
        if old['case_name'] == 'bar_near':
            assert bar is None
            mask = bar_mask(reference, old['camera'])
            bar = dict(id=identity, reference_pixels=1054,
                       mask_authority='Historic evaluator-only center/head supported forward<=3m mask, reconstructed unchanged; not a new instance segmentation',
                       reported_quantity='Original optical-axis z<2m, not old heading-forward<=3m alerts',
                       depth={arm: depth_summary(depth, mask) for arm, depth in dict(native_truth=reference, **depths).items()})
        cases.append(case)
    assert bar is not None
    metrics = {arm: m.metrics(count) for arm, count in totals.items()}
    for arm in ('low', 'native'):
        assert metrics[arm]['tp']-metrics['da_v2']['tp'] == paired[arm]['rescued_fn']-paired[arm]['lost_tp']
        assert metrics[arm]['fp']-metrics['da_v2']['fp'] == paired[arm]['added_fp']-paired[arm]['removed_fp']
    result = dict(status='COMPLETE', cohort='Separate18 consumed G5 static views; never pooled with500 Hypersim frames',
                  frames=18, original_test_frames_read=0, model_calls=0, training_updates=0,
                  metrics=metrics, unknown_pixels=unknown, paired={arm: dict(c) for arm, c in paired.items()},
                  near_mask_boundary=dict(aggregate={arm: shared.boundary_metrics(c) for arm, c in boundary.items()},
                      frame_macro=shared.frame_macro(cases, 'near_mask_boundary', ARMS),
                      definition='Optical z<2m, reference-known endpoint pairs, oriented lattice boundaries, tolerance1pixel at640x360; metric-dependent'),
                  scale_invariant_boundary=dict(aggregate={arm: shared.boundary_metrics(c) for arm, c in scale_invariant.items()},
                      frame_macro=shared.frame_macro(cases, 'scale_invariant_boundary', ARMS),
                      definition='Directed adjacent optical-depth ratio strictly>1.25; exact edge/order, no tolerance or GT alignment'),
                  near_crossing_discontinuity={arm: shared.near_crossing_metrics(c) for arm, c in near_crossing.items()},
                  bar=bar, cases=cases, verified_source_inputs=verified,
                  hashes=dict(protocol=protocol_hash, launch_seal=m.sha(output/'launch-seal.json'),
                              prediction_seal=m.sha(output/'prediction-seal.json'), original_receipt=BASELINE_SHA,
                              evaluator=m.sha(__file__), shared_evaluator=m.sha(shared.__file__),
                              near_field=m.sha(Path(__file__).with_name('near_field.py')),
                              surface_support=m.sha(Path(__file__).with_name('run_surface_support.py'))),
                  backend='CPU TASK_NOT_GPU_SUITABLE:18 cached-array counts; one historic230400pixel bar-mask reconstruction; no model or GPU contention',
                  evaluation_seconds=time.perf_counter()-started,
                  limitations=['No old<=3m heading-forward alert score is recomputed or renamed',
                               'Low input256x144 is resized to640x360; native input640x360 has additional detail',
                               'Historic bar mask is evaluator-conditioned geometric support, not a predicted object mask',
                               'Consumed static synthetic diagnostic is not natural, dynamic, device or safety evidence'])
    m.write(output/'diagnostic-results.json', result)
    print('G5_RESULT', json.dumps(dict(metrics=metrics, bar=bar, seconds=result['evaluation_seconds'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    evaluate(args.output)
