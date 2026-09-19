"""Independent saved-array recount and algebraic integrity check; no inference.

Reuse only the frozen domain definitions. Confusion counts use np.bincount,
not the evaluator's counting function; predictions and regions are never rebuilt.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ba_nfo_frozen_transfer import evaluation_domains

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/ba-nfo-depthpro-echo-20260919'
OLD = ROOT/'artifacts.local/work/ba-nfo-depthpro-20260919'
NFO = ROOT/'artifacts.local/work/ba-nfo-frozen-transfer500-20260919'
SOURCE = ROOT/'artifacts.local/work/ba-nfo-20260919'
NFO_PREDICTION_SHA = '8b381bb07f6b02180b6d934ea446a01329b9aaa2c9d9714efc6795611b724b10'
ARMS = ('nfo', 'native', 'global', 'layered')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recount(prediction, truth, domain):
    # Index0=TN,1=FP,2=FN,3=TP; return the recorded TP/FP/FN/TN order.
    codes = truth[domain].astype(np.uint8)*2+prediction[domain].astype(np.uint8)
    bins = np.bincount(codes, minlength=4)
    return bins[[3, 1, 2, 0]]


def verify(output=OUT):
    output = Path(output)
    assert not (output/'verification.json').exists(), 'Preserve completed verifier receipt'
    result = read(output/'results.json')
    protocol, seal = read(output/'protocol.json'), read(output/'prediction-seal.json')
    rows, observations = read(output/'manifest.json'), read(output/'observations.json')
    assert len(rows) == len(observations) == len(seal['outputs']) == 500
    assert rows == read(NFO/'manifest.json') == read(OLD/'manifest.json')
    assert all(r['split'] == 'val' for r in rows)
    assert [r['id'] for r in rows] == [r['id'] for r in observations]
    assert seal['status'] == 'COMPLETE' and seal['protocol_sha256'] == sha(output/'protocol.json')
    assert seal['source_sha256'] == protocol['source_hashes']['inference']
    assert sha(Path(__file__).with_name('ba_nfo_depthpro_echo.py')) == seal['source_sha256']
    assert sha(output/'observations.json') == protocol['source_hashes']['observations']
    assert sha(OLD/'manifest.json') == protocol['source_hashes']['original_manifest']
    assert sha(OLD/'prediction-seal.json') == protocol['source_hashes']['original_prediction_seal']
    assert sha(NFO/'predictions.npz') == NFO_PREDICTION_SHA
    native_seal = read(OLD/'prediction-seal.json')
    assert native_seal['status'] == 'COMPLETE'
    with np.load(NFO/'predictions.npz', allow_pickle=False) as archive:
        nfo = np.unpackbits(archive['masks'][:, 0], axis=1, bitorder='little').reshape(500, 192, 256).astype(bool)
    totals = defaultdict(lambda: np.zeros(4, np.int64))
    unknown = unsupported_pixels = regions_checked = anchors_checked = 0
    frames = []
    for index, (row, observation) in enumerate(zip(rows, observations)):
        identity = row['id']; path = output/'predictions'/f'{identity}.npz'
        receipt = read(path.with_suffix('.json'))
        assert receipt == seal['outputs'][identity]
        assert receipt['id'] == identity and receipt['protocol_sha256'] == seal['protocol_sha256']
        assert receipt['sha256'] == sha(path)
        native_path = OLD/'predictions/native'/f'{identity}.npz'
        native_digest = native_seal['outputs'][f'predictions/native/{identity}']['sha256']
        assert sha(native_path) == receipt['depth_sha256'] == observation['depth_sha256'] == native_digest
        assert (ROOT/observation['depth_path']).resolve() == native_path.resolve()
        with np.load(native_path, allow_pickle=False) as data:
            native = data['depth'].copy()
        with np.load(path, allow_pickle=False) as data:
            global_depth, layered = data['global_depth'].copy(), data['layered_depth'].copy()
            labels, scales, support = data['labels'].copy(), data['scales'].copy(), data['support'].copy()
        assert native.shape == global_depth.shape == layered.shape == labels.shape == (192, 256)
        assert native.dtype == global_depth.dtype == layered.dtype == np.float32
        assert labels.dtype == support.dtype == np.int32 and scales.dtype == np.float64
        assert all(np.isfinite(x).all() and (x > 0).all() for x in (native, global_depth, layered, scales))
        assert labels.min() == 0 and labels.max()+1 == len(scales) == len(support) == receipt['regions']
        assert np.array_equal(np.unique(labels), np.arange(len(scales)))
        assert (support >= 0).all()
        assert np.array_equal(global_depth, (native.astype(np.float64)*receipt['global_scale']).astype(np.float32))
        assert np.array_equal(layered, (native.astype(np.float64)*scales[labels]).astype(np.float32))
        unsupported = support[labels] == 0
        assert np.array_equal(layered[unsupported], native[unsupported])
        assert (scales[support == 0] == 1.).all()
        assert int((support > 0).sum()) == receipt['anchored_regions']
        assert int((~unsupported).sum()) == receipt['anchored_pixels']
        # Verify receipt support counts without recomputing associations.
        anchors = receipt['anchors']
        counted = np.bincount([a['region'] for a in anchors], minlength=len(support))
        assert np.array_equal(counted, support)
        assert len({a['zone'] for a in anchors}) == len(anchors)
        # Check the frozen bounded-depth property of saved regions, not its
        # graph construction or a new segmentation/association run.
        minimum = np.full(len(scales), np.inf); maximum = np.zeros(len(scales))
        np.minimum.at(minimum, labels.ravel(), native.ravel())
        np.maximum.at(maximum, labels.ravel(), native.ravel())
        assert (maximum <= 1.25*minimum).all()
        source_path = SOURCE/row['prepared']; assert sha(source_path) == row['sha256']
        with np.load(source_path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in ('depth', 'boxes', 'values')}
        assert np.array_equal(arrays['boxes'], np.asarray(observation['boxes']))
        observed_values = np.array([np.nan if x is None else x for x in observation['values']])
        assert np.array_equal(arrays['values'], observed_values, equal_nan=True)
        truth, domains = evaluation_domains(arrays)
        assert len(domains) == 11
        unknown += int((~domains['full']).sum())
        predictions = dict(nfo=nfo[index], native=native < 2., global_=global_depth < 2., layered=layered < 2.)
        predictions['global'] = predictions.pop('global_')
        for arm in ARMS:
            for domain, mask in domains.items():
                count = recount(predictions[arm], truth, mask)
                totals[arm, domain] += count
        frames.append(dict(id=identity, unsupported_pixels=int(unsupported.sum()),
                           regions=len(scales), anchors=len(anchors)))
        unsupported_pixels += int(unsupported.sum()); regions_checked += len(scales); anchors_checked += len(anchors)
    assert unknown == result['unknown_pixels'] == 350140
    for (arm, domain), counts in totals.items():
        for key, value in zip(('tp', 'fp', 'fn', 'tn'), counts):
            assert int(value) == result['metrics'][arm][domain][key], (arm, domain, key)
    report = dict(status='PASS', frames=500, arms=list(ARMS), domains=11,
                  frame_confusion_cells_recounted=500*4*11*4,
                  aggregate_confusion_cells_compared=4*11*4,
                  unknown_pixels=unknown, unsupported_pixels_exactly_retained=unsupported_pixels,
                  saved_regions_checked=regions_checked, saved_anchors_checked=anchors_checked,
                  global_float64_multiply_float32_cast_exact=True,
                  per_region_float64_scale_float32_cast_exact=True,
                  regions_within_frozen_ratio=True, source_and_prediction_hashes_verified=True,
                  method='Independent np.bincount TP/FP/FN/TN recount on reused frozen domains; no scientific algorithm or model rerun',
                  model_calls=0, algorithm_reexecutions=0, original_test_frames_read=0,
                  hashes=dict(results=sha(output/'results.json'), protocol=sha(output/'protocol.json'),
                              manifest=sha(output/'manifest.json'), prediction_seal=sha(output/'prediction-seal.json'),
                              verifier=sha(__file__)), frames_checked=frames)
    (output/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('ECHO_VERIFICATION_PASS', json.dumps({k: report[k] for k in ('frames', 'unknown_pixels', 'unsupported_pixels_exactly_retained', 'saved_regions_checked')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    verify(parser.parse_args().output)
