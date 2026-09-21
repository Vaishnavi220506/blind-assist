"""Independent saved axis-audit recount, with the inherited support integrator.

No runner import, fit, new alert policy, input mutation or protected-test access.
The numerical integration formula is reused explicitly; all other descriptors,
geometry, subsets, statistics and diagnostic criteria are reconstructed here.
"""
from collections import Counter
from datetime import datetime
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from tof_corridor_calibration import support_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
PRIOR = ROOT / 'artifacts.local/work/ba-current-only-20260921'
OUT = ROOT / 'artifacts.local/work/ba-axis-evidence-20260921'
HEADS = ('original', 'uniform', 'balanced')
ROLES = ('selection', 'evaluation')
WITNESSES = ('possible_depth', 'contained_depth', 'compatible_contained_depth')
HIGH = 7.6612162590026855
STRONG = .4071309640537889


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same(a, b, path='root'):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + '.' + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + '[' + str(i) + ']')
    elif isinstance(a, float):
        assert b is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


def ray_stop(a, b):
    horizontal = .3 / abs(a) if a else math.inf
    vertical = (.9 if b > 0 else -.2) / b if b else math.inf
    return min(3., horizontal, vertical)


def descriptor(boxes, values):
    focal = 640 / (2 * np.tan(np.deg2rad(50)))
    records = []
    for zone, (box, raw) in enumerate(zip(boxes, values)):
        r = float(raw)
        if not math.isfinite(r) or not .1 <= r < 8.:
            records.append(dict(zone=zone, valid=False, range_m=None))
            continue
        y0, x0, y1, x1 = map(int, box)
        aa = ((x0 * 640 / 256 - 320) / focal, (x1 * 640 / 256 - 320) / focal)
        bb = ((y0 * 360 / 192 - 180) / focal, (y1 * 360 / 192 - 180) / focal)
        radius = .1 + 3 * (.01 + .02 * r)
        lo, hi = max(.1, r - radius), r + radius
        state = ('DISJOINT' if hi < .3 or lo > 3 else
                 'CONTAINED' if lo >= .3 and hi <= 3 else 'PARTIAL')
        xs = [float(a * z) for a in aa for z in (lo, hi)]
        horizontal = ('OUTSIDE' if max(xs) < -.3 - 1e-9 or min(xs) > .3 + 1e-9 else
                      'INSIDE' if min(xs) > -.3 + 1e-9 and max(xs) < .3 - 1e-9 else 'CROSSING')
        nearest_a = 0 if aa[0] <= 0 <= aa[1] else min(aa, key=abs)
        nearest_b = 0 if bb[0] <= 0 <= bb[1] else min(bb, key=abs)
        possible = max(lo, .3) <= min(hi, ray_stop(nearest_a, nearest_b))
        definite = lo >= .3 and hi <= min(ray_stop(a, b) for a in aa for b in bb)
        factor = support_score(aa, bb, (lo, hi))
        independent_depth = max(0., min(hi, 3.) - max(lo, .3)) / (hi - lo)
        same(independent_depth, factor['depth'], 'independent_depth')
        if independent_depth == 0:
            assert factor['angular_given_depth'] is None and factor['joint'] == 0
        else:
            assert 0 <= factor['angular_given_depth'] <= 1 + 1e-12
            same(factor['joint'], factor['depth'] * factor['angular_given_depth'], 'factorization')
        records.append(dict(zone=zone, valid=True, range_m=r, interval_m=[lo, hi],
            depth_state=state, horizontal_relation=horizontal, possible=bool(possible), definite=bool(definite),
            depth=factor['depth'], angular_given_depth=factor['angular_given_depth'], joint=factor['joint']))
    observed = [z for z in records if z['valid']]
    depth = [z for z in observed if z['depth_state'] in ('PARTIAL', 'CONTAINED')]
    contained = [z for z in observed if z['depth_state'] == 'CONTAINED']
    ndefinite = sum(z['definite'] for z in observed)
    return dict(zones=records, valid_zones=len(observed), missing_zones=64 - len(observed),
        min_range_m=min((z['range_m'] for z in observed), default=None),
        depth_state=('MISSING' if not observed else 'OUTSIDE_ONLY' if not depth else
                     'CONTAINED_PRESENT' if contained else 'BOUNDARY_ONLY'),
        depth_counts=dict(Counter(z['depth_state'] for z in observed)),
        depth_overlap_horizontal_counts=dict(Counter(z['horizontal_relation'] for z in depth)),
        contained_horizontal_counts=dict(Counter(z['horizontal_relation'] for z in contained)),
        max_depth_fraction=max((z['depth'] for z in observed), default=0.),
        max_joint_score=max((z['joint'] for z in observed if z['possible']), default=0.),
        possible_corridor_zones=sum(z['possible'] for z in observed), definite_corridor_zones=ndefinite,
        unknown=not bool(ndefinite), witnesses=dict(possible_depth=bool(depth), contained_depth=bool(contained),
        compatible_contained_depth=any(z['possible'] for z in contained)))


def actual_axes(geo):
    assert geo['actual_camera_rotation'] == [0., 0., 0.]
    target = next(o for o in geo['objects'] if o['name'] == 'target')
    c, h = target['render_bounds_center_m'], target['render_bounds_extent_m']
    camera = geo['actual_camera_location_m']
    # Map full rendered world bounds to X-right / Y-down / Z-forward.
    center = [c[1] - camera[1], camera[2] - c[2], c[0] - camera[0]]
    half = [h[1], h[2], h[0]]
    lo = [v - e for v, e in zip(center, half)]
    hi = [v + e for v, e in zip(center, half)]
    x = not (hi[0] < -.3 or lo[0] > .3)
    y = not (hi[1] < -.2 or lo[1] > .9)
    z = not (hi[2] < .3 or lo[2] > 3.)
    category = ('DISTANCE_NEGATIVE' if not z else 'LATERAL_NEGATIVE' if not x else
                'VERTICAL_NEGATIVE' if not y else 'POSITIVE')
    return dict(lower_m=lo, upper_m=hi, x_overlap=x, y_overlap=y, z_overlap=z, category=category, truth=x and y and z)


def selections(rows):
    out = {k: [] for k in ('all_frames', 'positives', 'distance_negatives', 'lateral_negatives',
        'vertical_negatives', 'high_score_distance_negatives', 'high_score_lateral_negatives',
        'original_high_boundary_rescues')}
    for h in HEADS:
        out[h + '_suppressed_boundary'] = []
        out[h + '_suppressed_boundary_native'] = []
    for r in rows:
        out['all_frames'].append(r)
        category = r['axes']['category']
        if r['truth']:
            out['positives'].append(r)
        else:
            key = {'DISTANCE_NEGATIVE': 'distance_negatives', 'LATERAL_NEGATIVE': 'lateral_negatives',
                   'VERTICAL_NEGATIVE': 'vertical_negatives'}[category]
            out[key].append(r)
        high = not r['a'] and r['scores']['original'] >= HIGH
        if high and category in ('DISTANCE_NEGATIVE', 'LATERAL_NEGATIVE'):
            out['high_score_' + category.lower() + 's'].append(r)
        if not (high and r['truth'] and r['layout_relation'] == 'BOUNDARY'):
            continue
        out['original_high_boundary_rescues'].append(r)
        for h in HEADS:
            if not r['flags'][h + '_current']:
                out[h + '_suppressed_boundary'].append(r)
                if r['native_target_corridor_samples'] > 0:
                    out[h + '_suppressed_boundary_native'].append(r)
    return out


def summary(rows):
    data = [r['public'] for r in rows]
    stats = {}
    for key in ('valid_zones', 'min_range_m', 'max_depth_fraction', 'max_joint_score'):
        values = [d[key] for d in data if d[key] is not None]
        stats[key] = dict(count=len(values), min=min(values) if values else None,
                         median=float(statistics.median(values)) if values else None,
                         max=max(values) if values else None)
    return dict(frames=len(rows), groups=len(set(r['base_group_id'] for r in rows)),
        depth_states=dict(Counter(d['depth_state'] for d in data)),
        witnesses={w: sum(d['witnesses'][w] for d in data) for w in WITNESSES},
        no_possible_corridor=sum(d['possible_corridor_zones'] == 0 for d in data),
        with_definite_corridor=sum(d['definite_corridor_zones'] > 0 for d in data),
        baseline_unknown=sum(r['unknown'] for r in rows),
        native_backed=sum(r['native_target_corridor_samples'] > 0 for r in rows), numeric=stats)


def boundary_cost(rows):
    sets = selections(rows)
    positives = sets['positives']
    native = [r for r in positives if r['native_target_corridor_samples'] > 0]
    suppressed = {r['id'] for r in sets['balanced_suppressed_boundary']}
    bounds = {}
    for r in positives:
        previous = bounds.get(r['clip_id'], (r['frame_in_clip'], r['frame_in_clip']))
        bounds[r['clip_id']] = (min(previous[0], r['frame_in_clip']), max(previous[1], r['frame_in_clip']))
    assert len(bounds) == 16
    lost = []
    for r in positives:
        first, last = bounds[r['clip_id']]
        position = ('FIRST_POSITIVE' if r['frame_in_clip'] == first else
                    'LAST_POSITIVE' if r['frame_in_clip'] == last else 'INTERIOR')
        if position != 'INTERIOR':
            assert not r['public']['witnesses']['contained_depth'], ('edge_has_contained', r['id'])
        if not r['public']['witnesses']['contained_depth']:
            lost.append(dict(id=r['id'], group=r['base_group_id'], relation=r['layout_relation'],
                event_position=position, time_s=r['time_s'], native=r['native_target_corridor_samples'],
                balanced_suppressed=r['id'] in suppressed, max_depth=r['public']['max_depth_fraction']))
    return dict(positive_native=dict(total=len(native), contained=sum(r['public']['witnesses']['contained_depth'] for r in native)),
        positive_without_contained=lost, loss_positions=dict(Counter(r['event_position'] for r in lost)),
        by_relation=dict(Counter(r['relation'] for r in lost)),
        balanced_native_losses=[r for r in lost if r['balanced_suppressed'] and r['native'] > 0],
        high_lateral_with_contained=[dict(id=r['id'], group=r['base_group_id'], min_range=r['public']['min_range_m'],
            compatible=r['public']['witnesses']['compatible_contained_depth'])
            for r in sets['high_score_lateral_negatives'] if r['public']['witnesses']['contained_depth']])


def main():
    assert not (OUT / 'independent-audit.json').exists(), 'Do not overwrite completed audit'
    checked, seals = 0, []
    for directory, names in ((SOURCE, ('observation-seal.json', 'prediction-seal.json', 'evaluation-seal.json')),
                             (PRIOR, ('prediction-seal.json', 'evaluation-seal.json')),
                             (OUT, ('public-seal.json', 'analysis-seal.json'))):
        for name in names:
            seal = read(directory / name)
            assert seal['protocol_sha256'] == sha(directory / 'protocol.json')
            for file, digest in seal['hashes'].items():
                assert sha(directory / file) == digest, file
                checked += 1
            seals.append(str((directory / name).relative_to(ROOT)))
    for directory in (PRIOR, OUT):
        for file, digest in read(directory / 'protocol.json')['inputs'].items():
            assert sha(ROOT / file) == digest, file
            checked += 1
    protocol = read(OUT / 'protocol.json')
    assert (OUT / 'protocol-before-run.md').read_bytes() == (HERE / 'AXIS_EVIDENCE_PROTOCOL_20260921.md').read_bytes()
    receipt = read(OUT / 'extraction-receipt.json')
    assert receipt['frames'] == 1152 and receipt['zones'] == 73728
    for key in ('label_or_geometry_values_read', 'scores_read', 'native_values_read'):
        assert receipt[key] is False
    start = read(OUT / 'analysis-start.json')
    assert start['public_seal_sha256'] == sha(OUT / 'public-seal.json')
    assert datetime.fromisoformat(protocol['frozen_at_utc']) < datetime.fromisoformat(start['time_utc'])
    # Static boundary check complements the reviewed extractor and sealed receipt.
    tree = ast.parse((HERE / 'audit_axis_evidence.py').read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'describe')
    assert [a.arg for a in fn.args.args] == ['boxes', 'values']
    assert not {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} & {
        'read', 'SOURCE', 'PRIOR', 'truth_axes', 'labels', 'native', 'scores', 'layout_relation'}
    extraction = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract')
    read_calls = [n for n in ast.walk(extraction) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'read']
    assert len(read_calls) == 1 and ast.unparse(read_calls[0].args[0]) == "SOURCE / 'identities.json'"
    public = read(OUT / 'public-descriptors.json')
    identities = read(SOURCE / 'identities.json')
    baseline = read(SOURCE / 'baseline.json')
    with np.load(SOURCE / 'observations.npz', allow_pickle=False) as obs:
        assert set(obs.files) == {'ranges', 'boxes'}
        ranges, boxes = obs['ranges'], obs['boxes']
    assert ranges.shape == (1152, 64) and boxes.shape == (64, 4)
    assert len(public) == len(identities) == len(baseline) == 1152
    independently_public = {}
    exact_baseline_scores = 0
    valid_count = 0
    for i, (saved, identity, values, b) in enumerate(zip(public, identities, ranges, baseline)):
        assert identity['index'] == i
        value = dict(id='transfer:' + identity['id'], index=i, **descriptor(boxes, values))
        same(value, saved, 'public.' + str(i))
        assert value['max_joint_score'] == b['score'], ('baseline_score_exact', i)
        exact_baseline_scores += 1
        valid_count += value['valid_zones']
        assert b['valid_zones'] == value['valid_zones'] and b['definite_zones'] == value['definite_corridor_zones']
        assert b['unknown'] == value['unknown']
        assert b['current'] == bool(value['definite_corridor_zones'] or value['max_joint_score'] >= STRONG)
        independently_public[value['id']] = value
    assert len(independently_public) == 1152
    geos = read(SOURCE / 'capture/evaluator/geometry.json')
    native = {r['id']: r for r in read(SOURCE / 'source-admission.json')['frames']}
    transfer_labels = {r['index']: r['truth'] for r in read(SOURCE / 'evaluator/transfer-labels.json')}
    joined = read(OUT / 'joined-frame-results.json')
    result = read(OUT / 'result.json')
    expected_join, reports, gates, role_groups = [], {}, {}, {}
    for role in ROLES:
        rows = read(PRIOR / (role + '-frame-results.json'))
        assert len(rows) == 576
        role_groups[role] = {r['base_group_id'] for r in rows}
        assert len(role_groups[role]) == 8
        assert all(g.endswith(('g00', 'g01') if role == 'selection' else ('g02', 'g03')) for g in role_groups[role])
        for r in rows:
            i = r['index']
            d = independently_public[r['id']]
            assert d['index'] == i and identities[i]['id'] == r['source_id']
            assert geos[i]['sample_index'] == i and geos[i]['clip_id'] == r['clip_id']
            assert r['a'] == baseline[i]['current'] and r['unknown'] == baseline[i]['unknown']
            r['public'] = {k: v for k, v in d.items() if k not in ('id', 'index', 'zones')}
            r['axes'] = actual_axes(geos[i])
            assert r['axes']['truth'] == r['truth'] == transfer_labels[i] == native[r['source_id']]['truth']
            assert r['native_target_corridor_samples'] == native[r['source_id']]['returned_target_corridor_samples']
            r['role'] = role
        sets = selections(rows)
        reports[role] = dict(subsets={k: summary(v) for k, v in sets.items()},
            subset_ids={k: [r['id'] for r in v] for k, v in sets.items()},
            groups={g: {k: summary(v) for k, v in selections([r for r in rows if r['base_group_id'] == g]).items()}
                    for g in sorted(role_groups[role])})
        gates[role] = {}
        negatives = sets['high_score_distance_negatives']
        for h in HEADS:
            positive = sets[h + '_suppressed_boundary_native']
            gates[role][h] = {}
            for w in WITNESSES:
                keep = [r for r in positive if r['public']['witnesses'][w]]
                passed_neg = sum(r['public']['witnesses'][w] for r in negatives)
                groups = sorted({r['base_group_id'] for r in keep})
                gates[role][h][w] = dict(negative_total=len(negatives), negative_with_witness=passed_neg,
                    negative_without_witness=len(negatives) - passed_neg, positive_total=len(positive),
                    positive_with_witness=len(keep), positive_without_witness=len(positive) - len(keep),
                    positive_groups_with_witness=groups,
                    supported=bool(len(negatives) and len(positive) and not passed_neg and len(keep) == len(positive) and len(groups) >= 4))
        expected_join.extend(rows)
    assert len(expected_join) == len({r['id'] for r in expected_join}) == 1152
    assert not role_groups['selection'] & role_groups['evaluation']
    same(expected_join, joined, 'joined')
    same(reports, result['reports'], 'reports')
    same(gates, result['gates'], 'gates')
    qualified = {h: {w: all(gates[r][h][w]['supported'] for r in ROLES) for w in WITNESSES} for h in HEADS}
    same(qualified, result['qualified'], 'qualified')
    assert result['status'] == 'COMPLETE' and result['scope'] == 'CONSUMED_DEVELOPMENT'
    assert all(result[k] is False for k in ('original_test_activated', 'automatic_successor', 'alerts_changed'))
    focus = {'transfer:f0675', 'transfer:f0676', 'transfer:f0677', 'transfer:f0678', 'transfer:f0687',
        'transfer:f0688', 'transfer:f0689', 'transfer:f0460', 'transfer:f0461', 'transfer:f0465',
        'transfer:f0469', 'transfer:f0472', 'transfer:f0473'}
    same([dict(**r, zones=independently_public[r['id']]['zones']) for r in expected_join if r['id'] in focus],
         read(OUT / 'known-frame-details.json'), 'focus')
    costs = {role: boundary_cost([r for r in expected_join if r['role'] == role]) for role in ROLES}
    same(costs, read(OUT / 'boundary-cost-detail.json'), 'boundary_cost')
    original = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
    assert not any((original / n).exists() for n in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'))
    key_counts = {r: {k: reports[r]['subsets'][k] for k in ('all_frames', 'positives', 'distance_negatives',
        'lateral_negatives', 'high_score_distance_negatives', 'high_score_lateral_negatives',
        'balanced_suppressed_boundary_native')} for r in ROLES}
    audit = dict(status='PASS', audit_script_sha256=sha(__file__), verified_hash_entries=checked,
        verified_seals=seals, independently_recounted_frames=1152, independently_recounted_zones=73728,
        valid_zones=valid_count, baseline_scores_exact=exact_baseline_scores, role_groups={k: sorted(v) for k, v in role_groups.items()},
        no_new_alerts=True, public_input_boundary_checked=True, original_test_closed=True,
        reports=key_counts, gates=gates, qualified=qualified, boundary_cost=costs,
        boundary_cost_detail_sha256=sha(OUT / 'boundary-cost-detail.json'),
        limits=['The inherited support_score analytic integrator is reused; depth fraction, factorization, geometric bounds, relation flags, summaries and joins are independently checked.',
                'Every max joint score exactly equals its sealed pre-existing baseline score; this is not an independent rederivation of the integral.',
                'Extraction source and receipt support feature-before-evaluator order, but historical process access is not independently observable.',
                'Strong measured distance separation remains diagnostic evidence despite failed lossless gates; public witnesses do not identify target ownership or validate alerts.',
                'Consumed same-simulator Development; no model inference, training, threshold selection or hardware validation.'])
    with (OUT / 'independent-audit.json').open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', verified_hash_entries=checked, frames=1152, zones=73728,
                         baseline_scores_exact=exact_baseline_scores, qualified=qualified, reports=key_counts), indent=2))


if __name__ == '__main__':
    main()
