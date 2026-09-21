"""Read-only recount of saved appearance outputs; never calls model inference.

Accepts a thin metadata export with ranges.npy and all seal-referenced JSON.
Full RGB, feature arrays and native depth remain on the recorded capture host;
their recorded hashes are checked for pairing, not claimed as locally rehashed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    seals = {}
    for name in ('observation', 'prediction', 'evaluation'):
        receipt = read(root / (name + '-seal.json'))
        for relative, digest in receipt['hashes'].items():
            assert sha(root / relative) == digest, relative
        seals[name] = receipt['time_ns']
    assert seals['observation'] < seals['prediction'] < seals['evaluation']
    spec = read(root / 'spec.json')
    protocol = read(root / 'protocol.json')
    assert sha(root / 'spec.json') == protocol['spec_sha256']
    assert sha(root / 'protocol-before-run.md') == protocol['text_sha256']
    assert protocol['cuts'] == {'B': 6.888704776763916, 'N': 7.03014612197876}
    native = read(root / 'capture/evaluator/geometry.json')
    predictions = read(root / 'predictions.json')
    baseline = read(root / 'baseline.json')
    admission = read(root / 'pair-admission.json')
    differences = read(root / 'paired-differences.json')
    scores = read(root / 'metrics.json')
    ranges = np.load(root / 'ranges.npy', allow_pickle=False)
    assert len(spec['cases']) == len(native) == len(predictions) == len(baseline) == 1152
    assert ranges.shape == (1152, 64)
    refs, groups, paired = {}, defaultdict(list), defaultdict(dict)
    native_pairs_equal = True
    metric_groups = 0
    geometry_pair_mismatches = []
    for index, (case, geometry, pred, base) in enumerate(zip(spec['cases'], native, predictions, baseline)):
        assert case['name'] == geometry['id'] == pred['id']
        assert index == geometry['sample_index'] == pred['index']
        assert all(abs(v) < 1e-8 for v in geometry['actual_camera_rotation'])
        camera = geometry['actual_camera_location_m']
        declared = case['camera']
        assert max(abs(camera[i] - declared[a]) for i, a in enumerate(('x', 'y', 'z'))) < .002
        target = next(o for o in geometry['objects'] if o['name'] == 'target')
        center, extent = target['render_bounds_center_m'], target['render_bounds_extent_m']
        # Independent actual-world AABB overlap; do not call source classify().
        cx, cy, cz = center[1] - camera[1], camera[2] - center[2], center[0] - camera[0]
        ex, ey, ez = extent[1], extent[2], extent[0]
        truth = cx + ex >= -.3 - 1e-9 and cx - ex <= .3 + 1e-9
        truth = truth and cy + ey >= -.2 - 1e-9 and cy - ey <= .9 + 1e-9
        truth = bool(truth and cz + ez >= .3 - 1e-9 and cz - ez <= 3. + 1e-9)
        # Tiny UE transform quantization can change an exact boundary contact;
        # record separately rather than relabeling the frozen evaluator.
        target_spec = next(o for o in case['objects'] if o['name'] == 'target')
        d, s = target_spec['center_m'], target_spec['size_m']
        scx, scy, scz = d[1] - declared['y'], declared['z'] - d[2], d[0] - declared['x']
        declared_truth = (scx+s[1]/2 >= -.3-1e-9 and scx-s[1]/2 <= .3+1e-9
                          and scy+s[2]/2 >= -.2-1e-9 and scy-s[2]/2 <= .9+1e-9
                          and scz+s[0]/2 >= .3-1e-9 and scz-s[0]/2 <= 3.+1e-9)
        for name, cutoff in protocol['cuts'].items():
            assert pred['supplements'][name] == (pred['logits'][name] >= cutoff)
            assert pred['flags'][name+'_current'] == (base['current'] or pred['supplements'][name])
        assert pred['flags']['A_current'] == base['current']
        assert pred['unknown'] == base['unknown']
        previous = predictions[index-1] if case['frame_in_clip'] else None
        for name in ('A', 'B', 'N'):
            expected = pred['flags'][name+'_current'] or (previous is not None and previous['flags'][name+'_current'])
            assert pred['flags'][name+'_hold'] == expected
        if case['condition'] == 'reference':
            refs[case['source_id']] = (index, geometry)
        ri, rg = refs[case['source_id']]
        for field in ('actual_camera_location_m', 'actual_camera_rotation'):
            if geometry[field] != rg[field]:
                geometry_pair_mismatches.append(dict(index=index, field=field,
                    maximum_difference=max(abs(a-b) for a,b in zip(geometry[field],rg[field]))))
        for obj, refobj in zip(geometry['objects'], rg['objects']):
            for field in ('name', 'mesh_path', 'actual_location_m', 'actual_scale',
                          'actual_rotation', 'render_bounds_center_m', 'render_bounds_extent_m'):
                if obj[field] != refobj[field]:
                    geometry_pair_mismatches.append(dict(index=index, object=obj['name'], field=field))
            changed_material = obj['material_path'] != refobj['material_path']
            assert changed_material == (case['condition'] == obj['name'])
            prescribed = next(o['material'] for o in case['objects'] if o['name'] == obj['name'])
            assert obj['material_path'] == prescribed + '.' + prescribed.rsplit('/', 1)[-1]
        pair_equal = geometry['native_sha256'] == rg['native_sha256']
        tof_equal = bool(np.array_equal(ranges[index], ranges[ri], equal_nan=True))
        assert admission['rows'][index]['depth_equal'] == pair_equal
        assert admission['rows'][index]['tof_equal'] == tof_equal
        native_pairs_equal &= pair_equal and tof_equal
        item = (case, pred, bool(declared_truth), truth != declared_truth)
        groups[case['condition']].append(item)
        paired[case['source_id']][case['condition']] = item
    assert (admission['status'] == 'PASS') == native_pairs_equal
    for condition, rows in groups.items():
        for stratum in ('overall', 'Core', 'Boundary'):
            selected = [r for r in rows if stratum == 'overall' or
                        (r[0]['layout_relation'] == 'BOUNDARY') == (stratum == 'Boundary')]
            for arm in predictions[0]['flags']:
                count = Counter()
                for case, pred, truth, _ in selected:
                    alert, unknown = pred['flags'][arm], pred['unknown']
                    count['TP' if truth else 'FP'] += int(alert)
                    count['FN'] += int(truth and not alert)
                    count['TN'] += int(not truth and not alert and not unknown)
                    count['current_unknown'] += int(unknown)
                actual = scores[condition][stratum]['arms'][arm]['frames']
                assert all(actual[k] == count[k] for k in ('TP','FP','FN','TN','current_unknown'))
                metric_groups += 1
        for model in ('B', 'N'):
            if condition == 'reference':
                continue
            changes, by_group = Counter(), Counter()
            for source_id, versions in paired.items():
                case, pred, _, _ = versions[condition]
                ref = versions['reference'][1]
                for field, key in (('current', 'current_flips'), ('hold', 'held_flips')):
                    changed = pred['flags'][model+'_'+field] != ref['flags'][model+'_'+field]
                    changes[key] += changed
                    if field == 'current': by_group[case['base_group_id']] += changed
                changes['supplement_flips'] += pred['supplements'][model] != ref['supplements'][model]
            saved = differences[model][condition]
            assert all(saved[k] == v for k, v in changes.items())
            assert all(saved['groups'][g] == by_group[g] for g in spec['groups'])
    any_sensitive = False
    for model in ('B', 'N'):
        repeat = differences[model]['repeat']
        for condition in ('target', 'background'):
            compared = differences[model][condition]
            sensitive = (compared['current_flips'] - repeat['current_flips'] >= 3 and
                         sum(compared['groups'][g] > repeat['groups'][g] for g in spec['groups']) >= 2)
            assert compared['decision_sensitive'] == sensitive
            any_sensitive |= sensitive
    expected_status = ('NOT_EVALUABLE' if not native_pairs_equal else
                       'APPEARANCE_SENSITIVITY_SUPPORTED' if any_sensitive else
                       'NOT_SUPPORTED_FOR_THIS_INTERVENTION')
    assert read(root/'result.json')['status'] == expected_status
    return dict(status='PASS', frames=1152, paired_frames=288,
                metric_groups=metric_groups, source_admitted=native_pairs_equal,
                geometry_pairs_exact=not geometry_pair_mismatches,
                geometry_pair_mismatches=geometry_pair_mismatches,
                actual_geometry_exact_boundary_truth_disagreements=sum(r[3] for v in groups.values() for r in v),
                seals=seals, result_sha256=sha(root/'result.json'),
                limits='Read-only metadata/ranges audit, no new inference; full images/native arrays stay on capture host')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))
