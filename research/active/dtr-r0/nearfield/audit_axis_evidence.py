"""Public ToF axis descriptors, then evaluator-only diagnosis; no new alerts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from tof_corridor_calibration import score_frame
from tof_lateral_core import lateral_relation, slopes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
PRIOR = ROOT / 'artifacts.local/work/ba-current-only-20260921'
OUT = ROOT / 'artifacts.local/work/ba-axis-evidence-20260921'
PROTOCOL = HERE / 'AXIS_EVIDENCE_PROTOCOL_20260921.md'
ROLES = ('selection', 'evaluation')
HEADS = ('original', 'uniform', 'balanced')
HIGH = 7.6612162590026855
STRONG = .4071309640537889
WITNESSES = ('possible_depth', 'contained_depth', 'compatible_contained_depth')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_seal(root, name):
    saved = read(root / name)
    assert saved['protocol_sha256'] == sha(root / 'protocol.json')
    for name, digest in saved['hashes'].items():
        assert sha(root / name) == digest, name


def verify():
    for name, digest in read(OUT / 'protocol.json')['inputs'].items():
        assert sha(ROOT / name) == digest, name
    old = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
    assert not any((old / n).exists() for n in ('test-start.json', 'test-logits.npy', 'test-metrics.json'))


def seal(name, files):
    write(OUT / name, dict(protocol_sha256=sha(OUT / 'protocol.json'),
        hashes={p: sha(OUT / p) for p in files}))


def interval_state(interval):
    lo, hi = interval
    if hi < .3 or lo > 3.:
        return 'DISJOINT'
    return 'CONTAINED' if lo >= .3 and hi <= 3. else 'PARTIAL'


def describe(boxes, values):
    values = np.asarray(values)
    valid = np.isfinite(values) & (values >= .1) & (values < 8.)
    scored = score_frame(boxes, np.where(valid, values, np.nan))
    anchors = {a['zone']: a for a in scored['anchors']}
    factors = {s['zone']: s for s in scored['zone_scores']}
    zones = []
    for i in range(64):
        if not valid[i]:
            zones.append(dict(zone=i, valid=False, range_m=None))
            continue
        a = anchors[i]
        zones.append(dict(zone=i, valid=True, range_m=float(values[i]),
            interval_m=a['interval_m'], depth_state=interval_state(a['interval_m']),
            horizontal_relation=lateral_relation(slopes(boxes[i]), a['interval_m']),
            possible=a['possible'], definite=a['definite'],
            **{k: factors[i][k] for k in ('depth', 'angular_given_depth', 'joint')}))
    observed = [z for z in zones if z['valid']]
    possible = [z for z in observed if z['depth_state'] != 'DISJOINT']
    contained = [z for z in observed if z['depth_state'] == 'CONTAINED']
    compatible = [z for z in contained if z['possible']]
    depth_state = ('MISSING' if not observed else 'OUTSIDE_ONLY' if not possible else
                   'CONTAINED_PRESENT' if contained else 'BOUNDARY_ONLY')
    return dict(zones=zones, valid_zones=len(observed), missing_zones=64-len(observed),
        min_range_m=min((z['range_m'] for z in observed), default=None),
        depth_state=depth_state, depth_counts=dict(Counter(z['depth_state'] for z in observed)),
        depth_overlap_horizontal_counts=dict(Counter(z['horizontal_relation'] for z in possible)),
        contained_horizontal_counts=dict(Counter(z['horizontal_relation'] for z in contained)),
        max_depth_fraction=max((z['depth'] for z in observed), default=0.),
        max_joint_score=scored['score'], possible_corridor_zones=scored['baseline']['possible_zones'],
        definite_corridor_zones=scored['baseline']['definite_zones'],
        unknown=scored['baseline']['unknown'],
        witnesses=dict(possible_depth=bool(possible), contained_depth=bool(contained),
                       compatible_contained_depth=bool(compatible)))


def extract():
    assert not OUT.exists(), 'No overwrite or outcome-conditioned retry'
    check_seal(SOURCE, 'observation-seal.json')
    check_seal(PRIOR, 'evaluation-seal.json')
    inputs = [SOURCE / n for n in ('protocol.json', 'observations.npz', 'identities.json',
        'baseline.json', 'source-admission.json', 'observation-seal.json', 'capture/evaluator/geometry.json')]
    inputs += [PRIOR / n for n in ('protocol.json', 'evaluation-seal.json', 'selection.json')]
    inputs += [PRIOR / (r + '-frame-results.json') for r in ROLES]
    inputs += [PROTOCOL, Path(__file__), HERE / 'tof_corridor_calibration.py',
               HERE / 'tof_lateral_core.py', HERE / 'ba_camera_corridor.py']
    OUT.mkdir(parents=True)
    write(OUT / 'protocol.json', dict(id='ba-axis-evidence-20260921',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), scope='CONSUMED_DEVELOPMENT',
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        inputs={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        backend='CPU', reason='TASK_NOT_GPU_SUITABLE', original_test_activated=False,
        automatic_successor=False, witnesses=WITNESSES))
    (OUT / 'protocol-before-run.md').write_bytes(PROTOCOL.read_bytes())
    started = time.perf_counter()
    ids = read(SOURCE / 'identities.json')
    with np.load(SOURCE / 'observations.npz', allow_pickle=False) as obs:
        ranges, boxes = obs['ranges'], obs['boxes']
    assert ranges.shape == (1152, 64) and boxes.shape == (64, 4) and len(ids) == 1152
    public = [dict(id='transfer:' + row['id'], index=i, **describe(boxes, ranges[i])) for i, row in enumerate(ids)]
    write(OUT / 'public-descriptors.json', public)
    write(OUT / 'extraction-receipt.json', dict(frames=1152, zones=1152*64,
        label_or_geometry_values_read=False, scores_read=False, native_values_read=False,
        hashed_evaluator_sources=True, elapsed_s=time.perf_counter()-started))
    seal('public-seal.json', ['public-descriptors.json', 'extraction-receipt.json'])
    verify()
    print('SEALED_PUBLIC_DESCRIPTORS 1152 frames x 64 zones')


def truth_axes(geo):
    target = next(o for o in geo['objects'] if o['name'] == 'target')
    cam = np.asarray(geo['actual_camera_location_m'])
    c = np.asarray(target['render_bounds_center_m'])
    half = np.asarray(target['render_bounds_extent_m'])[[1, 2, 0]]
    center = np.array([c[1]-cam[1], cam[2]-c[2], c[0]-cam[0]])
    lower, upper = center-half, center+half
    overlap = [bool(upper[i] >= low and lower[i] <= high)
               for i, (low, high) in enumerate(((-.3, .3), (-.2, .9), (.3, 3.)))]
    category = ('DISTANCE_NEGATIVE' if not overlap[2] else 'LATERAL_NEGATIVE' if not overlap[0]
                else 'VERTICAL_NEGATIVE' if not overlap[1] else 'POSITIVE')
    return dict(lower_m=lower.tolist(), upper_m=upper.tolist(), x_overlap=overlap[0],
        y_overlap=overlap[1], z_overlap=overlap[2], category=category, truth=all(overlap))


def summarize(rows):
    descriptors = [r['public'] for r in rows]
    numeric = {}
    for name in ('valid_zones', 'min_range_m', 'max_depth_fraction', 'max_joint_score'):
        values = [d[name] for d in descriptors if d[name] is not None]
        numeric[name] = dict(count=len(values), min=min(values) if values else None,
            median=float(np.median(values)) if values else None, max=max(values) if values else None)
    return dict(frames=len(rows), groups=len({r['base_group_id'] for r in rows}),
        depth_states=dict(Counter(d['depth_state'] for d in descriptors)),
        witnesses={w: sum(d['witnesses'][w] for d in descriptors) for w in WITNESSES},
        no_possible_corridor=sum(d['possible_corridor_zones'] == 0 for d in descriptors),
        with_definite_corridor=sum(d['definite_corridor_zones'] > 0 for d in descriptors),
        baseline_unknown=sum(r['unknown'] for r in rows),
        native_backed=sum(r['native_target_corridor_samples'] > 0 for r in rows), numeric=numeric)


def subsets(rows):
    groups = dict(all_frames=rows, positives=[r for r in rows if r['truth']],
        distance_negatives=[r for r in rows if r['axes']['category'] == 'DISTANCE_NEGATIVE'],
        lateral_negatives=[r for r in rows if r['axes']['category'] == 'LATERAL_NEGATIVE'],
        vertical_negatives=[r for r in rows if r['axes']['category'] == 'VERTICAL_NEGATIVE'])
    extra = [r for r in rows if not r['a'] and r['scores']['original'] >= HIGH]
    groups['high_score_distance_negatives'] = [r for r in extra if r['axes']['category'] == 'DISTANCE_NEGATIVE']
    groups['high_score_lateral_negatives'] = [r for r in extra if r['axes']['category'] == 'LATERAL_NEGATIVE']
    groups['original_high_boundary_rescues'] = [r for r in extra if r['truth'] and r['layout_relation'] == 'BOUNDARY']
    for head in HEADS:
        selected = [r for r in groups['original_high_boundary_rescues'] if not r['flags'][head+'_current']]
        groups[head+'_suppressed_boundary'] = selected
        groups[head+'_suppressed_boundary_native'] = [r for r in selected if r['native_target_corridor_samples'] > 0]
    return groups


def analyze():
    verify()
    check_seal(OUT, 'public-seal.json')
    write(OUT / 'analysis-start.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
        public_seal_sha256=sha(OUT/'public-seal.json')))
    public = {r['id']: r for r in read(OUT/'public-descriptors.json')}
    geos = read(SOURCE/'capture/evaluator/geometry.json')
    base = read(SOURCE/'baseline.json')
    native = {r['id']: r for r in read(SOURCE/'source-admission.json')['frames']}
    reports, gates, joined, outputs = {}, {}, [], []
    for role in ROLES:
        rows = read(PRIOR/(role+'-frame-results.json'))
        assert len(rows) == 576 and len({r['base_group_id'] for r in rows}) == 8
        for row in rows:
            i, d = row['index'], public[row['id']]
            assert d['index'] == i
            b = base[i]
            assert abs(b['score']-d['max_joint_score']) < 1e-12
            assert b['valid_zones'] == d['valid_zones'] and b['definite_zones'] == d['definite_corridor_zones']
            assert b['unknown'] == d['unknown'] == row['unknown']
            assert b['current'] == row['a'] == bool(d['definite_corridor_zones'] or d['max_joint_score'] >= STRONG)
            row['public'] = {k: v for k, v in d.items() if k not in ('zones', 'id', 'index')}
            row['axes'] = truth_axes(geos[i])
            assert row['axes']['truth'] == row['truth'] == native[row['source_id']]['truth']
            assert row['native_target_corridor_samples'] == native[row['source_id']]['returned_target_corridor_samples']
            row['role'] = role
        sets = subsets(rows)
        reports[role] = dict(subsets={k: summarize(v) for k, v in sets.items()},
            subset_ids={k: [r['id'] for r in v] for k, v in sets.items()},
            groups={g: {k: summarize(v) for k, v in subsets([r for r in rows if r['base_group_id']==g]).items()}
                    for g in sorted({r['base_group_id'] for r in rows})})
        negatives = sets['high_score_distance_negatives']
        gates[role] = {}
        for head in HEADS:
            positives = sets[head+'_suppressed_boundary_native']
            gates[role][head] = {}
            for witness in WITNESSES:
                neg_pass = [r for r in negatives if r['public']['witnesses'][witness]]
                pos_pass = [r for r in positives if r['public']['witnesses'][witness]]
                groups = sorted({r['base_group_id'] for r in pos_pass})
                gates[role][head][witness] = dict(negative_total=len(negatives), negative_with_witness=len(neg_pass),
                    negative_without_witness=len(negatives)-len(neg_pass), positive_total=len(positives),
                    positive_with_witness=len(pos_pass), positive_without_witness=len(positives)-len(pos_pass),
                    positive_groups_with_witness=groups,
                    supported=bool(negatives and positives and not neg_pass and len(pos_pass)==len(positives) and len(groups)>=4))
        joined += rows
    assert len({r['id'] for r in joined}) == 1152
    assert not {r['base_group_id'] for r in joined if r['role']=='selection'} & {r['base_group_id'] for r in joined if r['role']=='evaluation'}
    result = dict(status='COMPLETE', scope='CONSUMED_DEVELOPMENT', reports=reports, gates=gates,
        qualified={h: {w: all(gates[r][h][w]['supported'] for r in ROLES) for w in WITNESSES} for h in HEADS},
        original_test_activated=False, automatic_successor=False, alerts_changed=False)
    write(OUT/'joined-frame-results.json', joined)
    write(OUT/'result.json', result)
    focus = {'transfer:f0675','transfer:f0676','transfer:f0677','transfer:f0678','transfer:f0687',
        'transfer:f0688','transfer:f0689','transfer:f0460','transfer:f0461','transfer:f0465',
        'transfer:f0469','transfer:f0472','transfer:f0473'}
    write(OUT/'known-frame-details.json', [dict(**r, zones=public[r['id']]['zones']) for r in joined if r['id'] in focus])
    seal('analysis-seal.json', ['joined-frame-results.json','result.json','known-frame-details.json'])
    verify()
    print(json.dumps(dict(status=result['status'], qualified=result['qualified'],
        subsets={r: result['reports'][r]['subsets'] for r in ROLES}, gates=gates)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('extract', 'analyze'))
    globals()[parser.parse_args().stage]()
