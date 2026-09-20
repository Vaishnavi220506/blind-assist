"""Read-only consumed Core432 audit: existing soft footprint and FP splitting.

Standard library only; no inference, fitting, new scoring rule or truth access
in a predictor. All source artifacts stay immutable. Output is diagnostic.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'artifacts.local/work/ba-core-transfer-20260920'
OUTPUT = ROOT / 'artifacts.local/work/ba-calibration-footprint-audit-20260920'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(rows, predicate):
    result = []
    for row in rows:
        if predicate(row):
            if not result or result[-1][-1]['frame_in_clip'] + 1 != row['frame_in_clip']:
                result.append([])
            result[-1].append(row)
    return result


def audit(source):
    checked = {}
    for name in ('prediction-seal.json', 'observation-seal.json', 'evaluation-seal.json'):
        seal = read(source / name)
        assert seal['status'] == 'COMPLETE' and seal['frames'] == 432
        assert seal['protocol_sha256'] == sha(source / 'protocol.json')
        for filename, digest in seal['hashes'].items():
            assert sha(source / filename) == digest, filename
            checked[filename] = digest
    protocol = read(source / 'protocol.json')
    for name in ('tof_corridor_calibration.py', 'ba_camera_corridor.py'):
        assert sha(Path(__file__).with_name(name)) == protocol['code_hashes'][name]
    observations = read(source / 'observations.json')
    predictions = read(source / 'predictions.json')
    rows = read(source / 'frame-results.json')
    assert len(observations) == len(predictions) == len(rows) == 432
    by_id, clips = {}, defaultdict(list)
    zone_count, max_factorization_error = 0, 0.
    for obs, pred, row in zip(observations, predictions, rows):
        assert obs['id'] == pred['id'] == row['id']
        assert all(obs[k] == pred[k] == row[k] for k in ('clip_id', 'frame_in_clip', 'time_s'))
        assert sha(source / obs['path']) == obs['sha256'] == pred['observation_sha256']
        assert row['predictions'] == pred['predictions']
        assert {s['zone'] for s in pred['zone_scores']} == {a['zone'] for a in pred['anchors']}
        possible = {a['zone'] for a in pred['anchors'] if a['possible']}
        for score in pred['zone_scores']:
            zone_count += 1
            assert 0 <= score['joint'] <= score['depth'] <= 1
            if score['depth']:
                reconstructed = score['depth'] * score['angular_given_depth']
                max_factorization_error = max(max_factorization_error, abs(reconstructed - score['joint']))
                assert math.isclose(reconstructed, score['joint'], abs_tol=1e-15)
            else:
                assert score['joint'] == 0 and score['angular_given_depth'] is None
        maximum = max((s['joint'] for s in pred['zone_scores'] if s['zone'] in possible), default=0.)
        reconstructed_max = max((s['depth'] * (s['angular_given_depth'] or 0.)
                                 for s in pred['zone_scores'] if s['zone'] in possible), default=0.)
        cal, raw = pred['predictions']['calibrated'], pred['predictions']['raw']
        assert cal['score'] == maximum and cal['threshold'] == protocol['threshold']
        for value in (maximum, reconstructed_max):
            assert cal['alert'] == (raw['definite_zones'] > 0 or
                                   (raw['alert'] and value >= protocol['threshold']))
        assert not cal['alert'] or raw['alert']
        by_id[row['id']] = pred
        clips[row['clip_id']].append(row)
    arms = ('raw', 'calibrated', 'no_rgb', 'rgb')
    summary = {}
    for arm in arms:
        tp = sum(r['truth'] and r['predictions'][arm]['alert'] for r in rows)
        fp = sum(not r['truth'] and r['predictions'][arm]['alert'] for r in rows)
        fn = sum(r['truth'] and not r['predictions'][arm]['alert'] for r in rows)
        summary[arm] = dict(TP=tp, FP=fp, FN=fn, precision=tp/(tp+fp),
                            recall=tp/(tp+fn), F1=2*tp/(2*tp+fp+fn),
                            FPR=fp/sum(not r['truth'] for r in rows),
                            outside_layout_FP=sum(r['layout_relation'] == 'OUTSIDE' and
                              not r['truth'] and r['predictions'][arm]['alert'] for r in rows),
                            unknown=sum(r['predictions'][arm]['unknown'] for r in rows))
    topology, gap_details, event_parity = [], [], []
    for clip_id, clip in sorted(clips.items()):
        clip.sort(key=lambda r: r['frame_in_clip'])
        assert [r['frame_in_clip'] for r in clip] == list(range(12))
        for event in runs(clip, lambda r: r['truth']):
            first = {arm: next((r['time_s'] for r in event if r['predictions'][arm]['alert']), None)
                     for arm in arms}
            assert first['raw'] == first['calibrated']
            event_parity.append(dict(clip_id=clip_id, first_in_event=first))
        raw_runs = runs(clip, lambda r: not r['truth'] and r['predictions']['raw']['alert'])
        for segment in raw_runs:
            children = runs(segment, lambda r: r['predictions']['calibrated']['alert'])
            topology.append(dict(clip_id=clip_id, raw_ids=[r['id'] for r in segment],
                                 calibrated_children=[[r['id'] for r in child] for child in children]))
            for left, right in zip(children, children[1:]):
                window = clip[left[-1]['frame_in_clip']:right[0]['frame_in_clip']+1]
                detail = []
                for row in window:
                    pred = by_id[row['id']]
                    possible = {a['zone'] for a in pred['anchors'] if a['possible']}
                    ranked = sorted((s for s in pred['zone_scores'] if s['zone'] in possible),
                                    key=lambda s: (-s['joint'], s['zone']))
                    detail.append(dict(id=row['id'], time_s=row['time_s'],
                        alert=pred['predictions']['calibrated']['alert'],
                        possible_zones=sorted(possible), top_zone=ranked[0] if ranked else None,
                        top_interval=next(a['interval_m'] for a in pred['anchors']
                                          if ranked and a['zone'] == ranked[0]['zone']) if ranked else None))
                gap_details.append(dict(clip_id=clip_id, layout_relation=clip[0]['layout_relation'],
                                        frames=detail))
    multiplicities = Counter(len(t['calibrated_children']) for t in topology)
    assert len(topology) == 37 and sum(k*v for k, v in multiplicities.items()) == 40
    assert sum(r['truth'] and not r['predictions']['calibrated']['alert'] for r in rows) == 0
    return dict(status='PASS_EXISTING_FOOTPRINT_IDENTITY_AUDIT', scope='CONSUMED432_DIAGNOSTIC_NO_NEW_METHOD',
        checked_source_hashes=checked, source_protocol_sha256=sha(source/'protocol.json'),
        frames=432, clips=len(clips), valid_zone_samples=zone_count,
        factorization='joint = depth_fraction * conditional_full_footprint_overlap',
        max_factorization_roundoff=max_factorization_error, reconstructed_alert_parity=432,
        threshold=protocol['threshold'], summary=summary, event_parity=event_parity,
        raw_segment_child_multiplicities=dict(sorted(multiplicities.items())),
        segment_topology=topology, internal_gaps=gap_details,
        limitations=['Fixed nominal zone footprint, not estimated physical angular uncertainty.',
                     'Threshold crossing diagnostics do not identify noise versus scene motion causally.',
                     'No new predictor or squared spatial weighting was evaluated.'],
        backend='TASK_NOT_GPU_SUITABLE: standard-library scalar identity/count/hash audit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    assert not args.out.exists(), 'Do not overwrite an audit'
    result = audit(args.source)
    args.out.mkdir(parents=True)
    result['audit_code_sha256'] = sha(Path(__file__))
    (args.out/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'frames', 'valid_zone_samples',
                                          'max_factorization_roundoff', 'raw_segment_child_multiplicities')}))
