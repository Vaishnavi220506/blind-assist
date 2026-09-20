"""Frozen consumed simulation pilot; no hardware or peak-detectability claim."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from ba_camera_corridor import sample_native
from tof_fov45_core import boxes45, simulate
from tof_corridor_calibration import score_frame, decide
from audit_existing_strata_20260920 import tally, paired_ids
from audit_core_workpoint_20260920 import silence_runs

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'artifacts.local/work/ba-core-workpoint-transfer-20260920'
OUT = ROOT / 'artifacts.local/work/ba-multireturn-pilot-20260920'
T, T0 = .4071309640537889, .007085703945147101
MODES = ('strongest', 'closest_exported', 'two_returns')
ARMS = ('calibration',) + tuple(m + s for m in MODES for s in ('', '_hold'))
IDENTITY = ('id', 'clip_id', 'time_s', 'frame_in_clip')
CODE = ('multireturn_pilot_20260920.py', 'tof_fov45_core.py', 'ba_camera_corridor.py',
        'tof_corridor_calibration.py', 'audit_existing_strata_20260920.py',
        'audit_core_workpoint_20260920.py', 'return_lineage_core.py')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def simulate_two(depth, identity, boxes):
    """Depth-only construction; returns public fields and private lineage."""
    first, traces = simulate(depth, identity, boxes)
    ranges = np.full((64, 2), np.nan, np.float32)
    ranges[:, 0] = first
    sigma = np.full((64, 2), np.nan, np.float32)
    signal = np.zeros((64, 2), np.float64)
    status = np.full((64, 2), 'SIM_NO_RETURN', dtype='<U13')
    private = []
    for z, (y0, x0, y1, x1) in enumerate(boxes):
        indices = [traces[z]['pixel_indices'].tolist(), []]
        bins_out = [traces[z]['winner_bin'], None]
        if np.isfinite(first[z]):
            patch = depth[y0:y1, x0:x1]
            valid = np.isfinite(patch) & (patch >= .1) & (patch < 8)
            hits = patch[valid]
            bins = np.minimum((hits / .1).astype(int), 79)
            weights = np.bincount(bins, weights=1 / np.maximum(hits, .3)**2, minlength=80)
            winner = int(np.argmax(weights))
            center = np.mean(hits[bins == winner])
            sigma[z, 0] = .01 + .02 * center
            signal[z, 0] = weights[winner]
            status[z, 0] = 'SIM_VALID'
            candidates = [int(b) for b in np.flatnonzero(weights)
                          if np.count_nonzero(bins == b) >= 4
                          and weights[b] >= .01 * weights[winner]
                          and abs(float(np.mean(hits[bins == b])) - float(center)) >= .6]
            if candidates:
                b = min(candidates, key=lambda k: (-weights[k], k))
                mean = np.mean(hits[bins == b])
                sd = .01 + .02 * mean
                seed = int(hashlib.sha256(f'{identity}|second|{z}'.encode()).hexdigest()[:16], 16)
                value = mean + np.random.default_rng(seed).normal(0, sd)
                if .1 <= value < 8:
                    ranges[z, 1], sigma[z, 1], signal[z, 1] = value, sd, weights[b]
                    status[z, 1] = 'SIM_VALID'
                    py, px = np.nonzero(valid)
                    use = bins == b
                    indices[1] = ((py[use] + y0) * depth.shape[1] + px[use] + x0).tolist()
                    bins_out[1] = b
        private.append(dict(zone=z, bins=bins_out, indices=indices))
    return dict(ranges=ranges, sigma=sigma, signal=signal, status=status), private


def readouts(boxes, ranges):
    """Observation-only; disjoint supports are evaluated separately."""
    ranges = np.asarray(ranges)
    if ranges.shape != (64, 2):
        raise ValueError('Expected 64 zones with two distinct slots')
    closest = np.min(np.where(np.isfinite(ranges), ranges, np.inf), axis=1)
    closest[~np.isfinite(closest)] = np.nan
    scores = [score_frame(boxes, ranges[:, i]) for i in range(2)]
    flags = [decide(s, T) for s in scores]
    near = decide(score_frame(boxes, closest), T)
    return dict(calibration=decide(scores[0], T0), strongest=flags[0],
                closest_exported=near,
                two_returns=dict(alert=flags[0]['alert'] or flags[1]['alert'],
                                 unknown=flags[0]['unknown'] and flags[1]['unknown'],
                                 score=max(s['score'] for s in scores))), scores


def seal(name, files):
    write(OUT / name, dict(status='COMPLETE', protocol_sha256=sha(OUT / 'protocol.json'),
                          hashes={f: sha(OUT / f) for f in files}))


def verify_seal(root, name):
    s = read(root / name)
    for f, h in s['hashes'].items():
        assert sha(root / f) == h, f
    if root == OUT:
        assert s['status'] == 'COMPLETE' and s['protocol_sha256'] == sha(OUT / 'protocol.json')


def freeze():
    assert not OUT.exists(), 'Do not overwrite a consumed pilot'
    OUT.mkdir(parents=True)
    protocol = Path(__file__).with_name('MULTIRETURN_PILOT_PROTOCOL_20260920.md')
    (OUT / 'protocol-before-run.md').write_bytes(protocol.read_bytes())
    files = ('observations.json', 'observation-seal.json', 'predictions.json', 'prediction-seal.json',
             'frame-results.json', 'evaluation-seal.json', 'private-lineage.json', 'spec.json',
             'capture/evaluator/geometry.json')
    write(OUT / 'protocol.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        inputs={f: sha(SOURCE / f) for f in files},
        code={f: sha(Path(__file__).with_name(f)) for f in CODE},
        protocol_text_sha256=sha(OUT / 'protocol-before-run.md'),
        backend='TASK_NOT_GPU_SUITABLE', scope='CONSUMED_HYPOTHETICAL_MULTI_RETURN', frames=432))
    print('FROZEN one 432-frame pilot')


def verify():
    p = read(OUT / 'protocol.json')
    assert sha(OUT / 'protocol-before-run.md') == p['protocol_text_sha256']
    for f, h in p['inputs'].items():
        assert sha(SOURCE / f) == h, f
    for f, h in p['code'].items():
        assert sha(Path(__file__).with_name(f)) == h, f


def construct():
    verify()
    for name in ('observation-seal.json', 'prediction-seal.json', 'evaluation-seal.json'):
        verify_seal(SOURCE, name)
    observations = read(SOURCE / 'observations.json')
    old = read(SOURCE / 'private-lineage.json')
    geometry = read(SOURCE / 'capture/evaluator/geometry.json')
    arrays, ids, lineage = defaultdict(list), [], []
    assert len(observations) == len(old) == len(geometry) == 432
    for o, previous, g in zip(observations, old, geometry):
        assert o['id'] == previous['id'] and o['clip_id'] == g['clip_id']
        assert o['frame_in_clip'] == g['frame_in_clip']
        path = SOURCE / 'capture/evaluator' / g['native_path']
        assert sha(path) == previous['native_sha256'] == g['native_sha256']
        assert sha(SOURCE / o['path']) == o['sha256']
        with np.load(SOURCE / o['path'], allow_pickle=False) as saved:
            assert np.array_equal(saved['boxes'], boxes45())
            public, traces = simulate_two(sample_native(np.load(path, allow_pickle=False)), previous['seed'], boxes45())
            assert np.array_equal(public['ranges'][:, 0], saved['values'], equal_nan=True)
        for trace, original in zip(traces, previous['traces']):
            assert trace['indices'][0] == original['pixel_indices']
        for k, a in public.items():
            arrays[k].append(a)
        ids.append({k: o[k] for k in IDENTITY})
        lineage.append(dict(id=o['id'], zones=traces))
    np.savez_compressed(OUT / 'observations.npz', boxes=boxes45(), **{k: np.stack(v) for k, v in arrays.items()})
    write(OUT / 'identities.json', ids)
    write(OUT / 'private-lineage.json', lineage)
    seal('observation-seal.json', ('observations.npz', 'identities.json', 'private-lineage.json'))
    print('OBSERVATIONS_SEALED; 432 original slots and lineage reproduced')


def predict():
    verify()
    verify_seal(OUT, 'observation-seal.json')
    ids, original = read(OUT / 'identities.json'), read(SOURCE / 'predictions.json')
    rows, previous = [], {}
    with np.load(OUT / 'observations.npz', allow_pickle=False) as data:
        for i, identity in enumerate(ids):
            decisions, scores = readouts(data['boxes'], data['ranges'][i])
            assert identity['id'] == original[i]['id']
            assert decisions['strongest'] == original[i]['predictions']['candidate']
            assert decisions['calibration'] == original[i]['predictions']['baseline']
            for k in ('score', 'anchors', 'zone_scores'):
                assert scores[0][k] == original[i][k]
            second_factors = {s['zone']: s for s in scores[1]['zone_scores']}
            second_alert_zones = [a['zone'] for a in scores[1]['anchors'] if a['definite']
                or (a['possible'] and second_factors[a['zone']]['joint'] >= T)]
            closest_second_zones = [z for z in second_alert_zones
                if not np.isfinite(data['ranges'][i, z, 0])
                or data['ranges'][i, z, 1] < data['ranges'][i, z, 0]]
            flags = {'calibration': decisions['calibration']['alert']}
            for mode in MODES:
                key = (identity['clip_id'], mode)
                prev = previous.get(key)
                held = bool(prev and abs(identity['time_s'] - prev[0] - .2) < 1e-6 and prev[1])
                flags[mode] = decisions[mode]['alert']
                flags[mode + '_hold'] = flags[mode] or held
                previous[key] = (identity['time_s'], flags[mode])
            rows.append({**identity, 'flags': flags, 'decisions': decisions,
                         'second_alert_zones': second_alert_zones,
                         'closest_second_alert_zones': closest_second_zones,
                         'second_slot_present': int(np.isfinite(data['ranges'][i, :, 1]).sum())})
    write(OUT / 'predictions.json', rows)
    seal('prediction-seal.json', ('predictions.json', 'observation-seal.json'))
    print('PREDICTIONS_SEALED; no truth/ownership used')


def evaluate():
    verify()
    verify_seal(OUT, 'prediction-seal.json')
    verify_seal(OUT, 'observation-seal.json')
    from return_lineage_core import masks
    rows, labels = read(OUT / 'predictions.json'), read(SOURCE / 'frame-results.json')
    lineage, geometry = read(OUT / 'private-lineage.json'), read(SOURCE / 'capture/evaluator/geometry.json')
    spec = read(SOURCE / 'spec.json')['cases']
    for r, label, private, g, case in zip(rows, labels, lineage, geometry, spec):
        assert r['id'] == label['id'] == private['id']
        assert r['clip_id'] == label['clip_id'] == g['clip_id'] == case['clip_id']
        r.update({k: label[k] for k in ('truth', 'boundary', 'layout_relation', 'layer', 'background', 'type_id')})
        path = SOURCE / 'capture/evaluator' / g['native_path']
        assert sha(path) == g['native_sha256']
        target, _, corridor, _ = masks(np.load(path, allow_pickle=False), case, g)
        supported = sample_native(target & corridor).ravel()
        r['native_target_corridor'] = [sum(int(supported[z['indices'][slot]].sum()) for z in private['zones']) for slot in (0, 1)]
        r['native_second_alert_contributors'] = {
            mode: sum(int(supported[z['indices'][1]].sum()) for z in private['zones'] if z['zone'] in r[key])
            for mode, key in (('two_returns', 'second_alert_zones'),
                              ('closest_exported', 'closest_second_alert_zones'))}
    selectors = dict(all432=lambda r: True, core288=lambda r: r['layout_relation'] != 'BOUNDARY',
        boundary144=lambda r: r['layout_relation'] == 'BOUNDARY', outside144=lambda r: r['layout_relation'] == 'OUTSIDE',
        inside_negative=lambda r: r['layout_relation'] == 'INSIDE' and not r['truth'])
    for key in ('layer', 'background'):
        for value in sorted({r[key] for r in rows}):
            selectors[key + ':' + value] = lambda r, k=key, v=value: r['layout_relation'] != 'BOUNDARY' and r[k] == v
    metrics = {a: {n: tally(rows, lambda r, a=a: r['flags'][a], select, .2, 'clip_id')
                   for n, select in selectors.items()} for a in ARMS}
    for a in ARMS:
        for name, select in selectors.items():
            m = metrics[a][name]
            m.pop('zero_return_frames')
            mode = a.removesuffix('_hold')
            m['current_unknown'] = sum(r['decisions'][mode]['unknown'] for r in rows if select(r))
    paired, gate = {}, {}
    for mode in MODES[1:]:
        for suffix in ('', '_hold'):
            a, base = mode + suffix, 'strongest' + suffix
            changes = {n: paired_ids(rows, lambda r: r['flags'][base], lambda r: r['flags'][a], select)
                       for n, select in selectors.items()}
            paired[a] = changes
            c, b = metrics[a]['core288'], metrics[base]['core288']
            recovered = set(changes['core288']['FN_rescued'])
            native = [r['id'] for r in rows if r['id'] in recovered and r['flags'][mode]
                      and r['native_second_alert_contributors'][mode] > 0]
            delays = [dict(episode=x['episode'], baseline=x['first_alert_s'], candidate=y['first_alert_s'])
                      for x, y in zip(b['events'], c['events'])
                      if x['first_alert_s'] is not None and (y['first_alert_s'] is None or y['first_alert_s'] > x['first_alert_s'])]
            gate[a] = dict(native_backed_recovery=native, delayed_events=delays,
                pass_fixed_readout=bool(native and not changes['core288']['TP_lost'] and not delays
                    and c['FP'] <= b['FP'] and c['false_segments'] <= b['false_segments']
                    and c['false_sampled_s'] <= b['false_sampled_s']))
    groups = defaultdict(list)
    for r in rows:
        if r['truth'] and selectors['core288'](r):
            groups[r['clip_id']].append(r)
    coverage = {a: {clip: dict(positive_frames=len(seq), alerted_frames=sum(r['flags'][a] for r in seq),
                        **silence_runs(seq, [r['flags'][a] for r in seq])) for clip, seq in groups.items()} for a in ARMS}
    availability = dict(second_slots=sum(r['second_slot_present'] for r in rows), zone_frames=432*64,
        frames_with_second=sum(r['second_slot_present'] > 0 for r in rows),
        core_positive_new_corridor_frame_ids=[r['id'] for r in rows if selectors['core288'](r) and r['truth']
            and r['native_target_corridor'][0] == 0 and r['native_target_corridor'][1] > 0])
    write(OUT / 'frame-results.json', rows)
    write(OUT / 'results.json', dict(metrics=metrics, paired=paired, gate=gate, availability=availability,
        event_coverage=coverage, selected_onsets=[r for r in rows if r['id'] in ('f0005', 'f0293')],
        scope='CONSUMED_HYPOTHETICAL_INPUT; NOT_PHYSICAL_DETECTABILITY; NO_TRAINING'))
    seal('evaluation-seal.json', ('results.json', 'frame-results.json', 'prediction-seal.json'))
    print(json.dumps(dict(core={a: {k: metrics[a]['core288'][k] for k in ('TP', 'FP', 'FN', 'false_segments')} for a in ARMS},
                          availability=availability, gate=gate)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('freeze', 'construct', 'predict', 'evaluate'))
    args = parser.parse_args()
    start = time.perf_counter()
    globals()[args.stage]()
    print(f'{args.stage}: {time.perf_counter()-start:.3f}s CPU TASK_NOT_GPU_SUITABLE')
