"""Saved-output lateral intervention diagnosis; no model inference or fitting."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from audit_spatial_bce import digest, read, seal_check
from ba_camera_corridor import rays
from tof_fov45_core import boxes45

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'artifacts.local/work/ba-spatial-complement-transfer-20260921'
OUT = ROOT/'artifacts.local/work/ba-lateral-pair-diagnostic-20260921'
HIGH = 7.6612162590026855
REL = ('INSIDE', 'BOUNDARY', 'OUTSIDE')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def camera_bounds(geometry):
    target = next(o for o in geometry['objects'] if o['name'] == 'target')
    c = np.asarray(geometry['actual_camera_location_m'])
    p = np.asarray(target['render_bounds_center_m'])
    half = np.asarray(target['render_bounds_extent_m'])[[1, 2, 0]]
    center = np.array([p[1]-c[1], c[2]-p[2], p[0]-c[0]])
    return center-half, center+half


def ray_box_mask(lower, upper, a, b):
    """Exact slab intersection for forward rays; works for zero direction too."""
    enter = np.full_like(a, max(0., lower[2]))
    leave = np.full_like(a, upper[2])
    for axis, direction in enumerate((a, b)):
        nonzero = direction != 0
        t0 = np.divide(lower[axis], direction, out=np.zeros_like(a), where=nonzero)
        t1 = np.divide(upper[axis], direction, out=np.zeros_like(a), where=nonzero)
        enter = np.maximum(enter, np.where(nonzero, np.minimum(t0, t1), -np.inf))
        leave = np.minimum(leave, np.where(nonzero, np.maximum(t0, t1), np.inf))
        if not lower[axis] <= 0 <= upper[axis]:
            leave = np.where(nonzero, leave, -np.inf)
    return (leave >= enter) & (leave > 0)


def geometry_descriptor(geo, a, b, boxes):
    lower, upper = camera_bounds(geo)
    mask = ray_box_mask(lower, upper, a, b)
    counts = [int(mask[y0:y1, x0:x1].sum()) for y0, x0, y1, x1 in boxes]
    edge_x = lower[0] if lower[0]+upper[0] >= 0 else upper[0]
    focal = 640/(2*np.tan(np.deg2rad(50)))
    u = 320+focal*edge_x/lower[2]
    columns = [j for j, (_, x0, _, x1) in enumerate(boxes[:8]) if x0 <= u*256/640 < x1]
    assert len(columns) <= 1
    return dict(lower=lower.tolist(), upper=upper.tolist(),
        zones=[j for j, n in enumerate(counts) if n], zone_ray_counts=counts,
        inner_front_edge_column=columns[0] if columns else None, inner_front_edge_native_u=float(u))


def verify_pair(cases, geos):
    """Assert the rendered lateral-only intervention, not just source intent."""
    reference, first = cases[0], geos[0]
    largest = 0.
    for case, geo in zip(cases, geos):
        for key in ('camera', 'time_s', 'sensor_noise_key', 'frame_in_clip', 'base_group_id'):
            assert case[key] == reference[key], key
        assert case['objects'][1] == reference['objects'][1]
        for key in ('size_m', 'material'):
            assert case['objects'][0][key] == reference['objects'][0][key]
        for axis in (0, 2):
            assert case['objects'][0]['center_m'][axis] == reference['objects'][0]['center_m'][axis]
        assert all(abs(x) < 1e-10 for x in geo['actual_camera_rotation'])
        diff = np.max(np.abs(np.asarray(geo['actual_camera_location_m'])-first['actual_camera_location_m']))
        largest = max(largest, float(diff))
        for obj, ref_obj in zip(geo['objects'], first['objects']):
            assert obj['name'] == ref_obj['name']
            assert obj['mesh_path'] == ref_obj['mesh_path']
            assert obj['material_path'] == ref_obj['material_path']
            assert all(abs(x) < 1e-10 for x in obj['actual_rotation'])
            for key in ('actual_scale', 'render_bounds_extent_m'):
                largest = max(largest, float(np.max(np.abs(np.asarray(obj[key])-ref_obj[key]))))
            axes = [0, 2] if obj['name'] == 'target' else [0, 1, 2]
            for key in ('actual_location_m', 'render_bounds_center_m'):
                largest = max(largest, float(np.max(np.abs(np.asarray(obj[key])[axes]-np.asarray(ref_obj[key])[axes]))))
    assert largest < 1e-8, largest
    return largest


def pair_record(left, right, descriptors, ranges):
    i, j = left['index'], right['index']
    x, y = descriptors[i], descriptors[j]
    v0, v1 = np.isfinite(ranges[i]), np.isfinite(ranges[j])
    common = v0 & v1
    same = x['zones'] == y['zones']
    footprint = 'empty_endpoint' if not x['zones'] or not y['zones'] else 'same_set' if same else 'changed_set'
    edge = 'outside_fov' if x['inner_front_edge_column'] is None or y['inner_front_edge_column'] is None else (
        'same_column' if x['inner_front_edge_column'] == y['inner_front_edge_column'] else 'changed_column')
    return dict(left_id=left['id'], right_id=right['id'], base_group_id=left['base_group_id'],
        frame_in_clip=left['frame_in_clip'], time_s=left['time_s'], phase=left['phase'],
        left_relation=left['layout_relation'], right_relation=right['layout_relation'],
        left_truth=left['truth'], right_truth=right['truth'],
        left_logit=left['logit'], right_logit=right['logit'], delta=left['logit']-right['logit'],
        left_high=left['logit'] >= HIGH, right_high=right['logit'] >= HIGH,
        outside_added_fp=right['layout_relation'] == 'OUTSIDE' and right['flags']['C_current']
            and not right['flags']['A_current'] and not right['truth'],
        left_native_corridor_samples=left['native_target_corridor_samples'],
        left_extra_tp=left['truth'] and left['flags']['C_current'] and not left['flags']['A_current'],
        footprint=footprint, edge=edge, left_zones=x['zones'], right_zones=y['zones'],
        left_edge_column=x['inner_front_edge_column'], right_edge_column=y['inner_front_edge_column'],
        moved_inner_edge_pixels=abs(x['inner_front_edge_native_u']-y['inner_front_edge_native_u']),
        changed_valid_zones=int((v0 != v1).sum()), common_valid_zones=int(common.sum()),
        common_valid_range_mae_m=float(np.abs(ranges[i][common]-ranges[j][common]).mean()) if common.any() else None)


def summarize(pairs):
    delta = [p['delta'] for p in pairs]
    return dict(pairs=len(pairs), groups=len({p['base_group_id'] for p in pairs}),
        score_decreases=sum(d > 0 for d in delta), ties=sum(d == 0 for d in delta),
        score_increases=sum(d < 0 for d in delta), median_drop=float(np.median(delta)) if delta else None,
        min_drop=min(delta) if delta else None, max_drop=max(delta) if delta else None,
        fixed_pair_separated=sum(p['left_high'] and not p['right_high'] for p in pairs),
        both_high=sum(p['left_high'] and p['right_high'] for p in pairs),
        both_low=sum(not p['left_high'] and not p['right_high'] for p in pairs),
        wrong_way_crossing=sum(not p['left_high'] and p['right_high'] for p in pairs),
        outside_added_fp=sum(p['outside_added_fp'] for p in pairs),
        outside_added_fp_despite_score_decrease=sum(p['outside_added_fp'] and p['delta'] > 0 for p in pairs),
        left_extra_tp=sum(p['left_extra_tp'] for p in pairs),
        left_extra_tp_native_supported=sum(p['left_extra_tp'] and p['left_native_corridor_samples'] > 0 for p in pairs),
        median_changed_valid_zones=float(np.median([p['changed_valid_zones'] for p in pairs])) if pairs else None,
        median_common_range_mae_m=float(np.median([p['common_valid_range_mae_m'] for p in pairs
            if p['common_valid_range_mae_m'] is not None])) if pairs else None)


def charts(rows, triplets, descriptors, identities):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    groups = sorted({r['base_group_id'] for r in rows})
    fig, axes = plt.subplots(4, 4, figsize=(17, 12), sharex=True)
    for group, ax in zip(groups, axes.flat):
        for relation, color in zip(REL, ('#1466ac', '#dd8420', '#b92538')):
            seq = sorted([r for r in rows if r['base_group_id'] == group and r['layout_relation'] == relation], key=lambda r:r['frame_in_clip'])
            ax.plot([r['time_s'] for r in seq], [r['logit'] for r in seq], label=relation, color=color)
            if relation == 'BOUNDARY':
                times = [r['time_s'] for r in seq if r['truth']]
                ax.axvspan(min(times), max(times), alpha=.10, color='green')
        ax.axhline(HIGH, color='black', linestyle='--', linewidth=.8)
        ax.set_title(group.replace('spatial_complement_transfer_', ''), fontsize=9)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Frozen B logits: same layout / same axial depth; green = positive-depth interval')
    fig.supxlabel('Nominal sequence time (s)'); fig.supylabel('Logit; dashed = frozen cutoff')
    fig.tight_layout(); fig.savefig(OUT/'all-group-score-trajectories.png', dpi=130); plt.close(fig)
    chosen = []
    for group in groups:
        mistakes = [r for r in rows if r['base_group_id'] == group and r['layout_relation'] == 'OUTSIDE'
            and r['flags']['C_current'] and not r['flags']['A_current']]
        if mistakes:
            chosen.append(min(mistakes, key=lambda r:r['frame_in_clip']))
    failed = {r['base_group_id'] for r in chosen}
    for group in groups:
        correct = [r for r in rows if r['base_group_id'] == group and r['layout_relation'] == 'BOUNDARY'
            and r['truth'] and r['flags']['C_current']]
        if group not in failed and correct:
            chosen.append(min(correct, key=lambda r:r['frame_in_clip'])); break
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 3.5*len(chosen)))
    selected = []
    for k, choice in enumerate(chosen):
        seq = triplets[(choice['base_group_id'], choice['frame_in_clip'])]
        for n, relation in enumerate(REL):
            row = seq[relation]; i = row['index']; ident = identities[i]
            path = SOURCE/ident['rgb_path']; assert digest(path) == ident['rgb_sha256']
            selected.append(dict(id=row['id'], path=ident['rgb_path'], sha256=ident['rgb_sha256']))
            with Image.open(path) as image:
                axes[k,n].imshow(image)
            axes[k,n].set_title(f"{relation} {row['id']} truth={int(row['truth'])} logit={row['logit']:.2f}\n"
                f"edge column={descriptors[i]['inner_front_edge_column']}; t={row['time_s']:.1f}s", fontsize=9)
            axes[k,n].axis('off')
        axes[k,0].text(0, -.06, choice['base_group_id'].replace('spatial_complement_transfer_', ''),
            transform=axes[k,0].transAxes, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/'selected-paired-images.png', dpi=120); plt.close(fig)
    write(OUT/'visual-selection.json', selected)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'result.json').exists(), 'Write-once diagnosis'
    protocol = HERE/'LATERAL_PAIR_DIAGNOSTIC_PROTOCOL_20260921.md'
    (OUT/'protocol-before-run.md').write_bytes(protocol.read_bytes())
    inputs = ['spec.json', 'frame-results.json', 'identities.json', 'logits.npy', 'predictions.json',
        'observations.npz', 'capture/evaluator/geometry.json', 'observation-seal.json',
        'prediction-seal.json', 'evaluation-seal.json', 'metrics.json', 'private-lineage.json']
    seals = {name:digest(SOURCE/name) for name in inputs}
    write(OUT/'input-seal.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE.relative_to(ROOT)), hashes=seals, code_sha256=digest(Path(__file__)),
        protocol_sha256=digest(protocol), backend='CPU', reason='TASK_NOT_GPU_SUITABLE',
        scope='small saved-array/geometry reductions; no model execution'))
    for name in ('observation-seal.json', 'prediction-seal.json', 'evaluation-seal.json'):
        seal_check(SOURCE, name)
    assert digest(SOURCE/'spec.json') == read(SOURCE/'protocol.json')['spec_sha256']
    rows, spec = read(SOURCE/'frame-results.json'), read(SOURCE/'spec.json')
    geos, predictions = read(SOURCE/'capture/evaluator/geometry.json'), read(SOURCE/'predictions.json')
    identities = read(SOURCE/'identities.json')
    logits = np.load(SOURCE/'logits.npy', allow_pickle=False)
    with np.load(SOURCE/'observations.npz', allow_pickle=False) as data:
        ranges, boxes = data['ranges'], data['boxes']
    np.testing.assert_array_equal(boxes, boxes45())
    assert ranges.shape == (1152, 64) and logits.shape == (1152,)
    assert len(rows) == len(spec['cases']) == len(geos) == len(predictions) == len(identities) == 1152
    assert np.isfinite(logits).all()
    triplets = defaultdict(dict)
    a, b = rays()
    descriptors = []
    for i, (r, pred, case, geo, ident) in enumerate(zip(rows, predictions, spec['cases'], geos, identities)):
        assert i == r['index'] == pred['index'] == ident['index'] == geo['sample_index']
        assert r['id'] == pred['id'] == ident['id']
        assert r['clip_id'] == pred['clip_id'] == ident['clip_id'] == case['clip_id'] == geo['clip_id']
        assert r['frame_in_clip'] == case['frame_in_clip'] == geo['frame_in_clip'] == ident['frame_in_clip']
        assert r['logit'] == pred['logit'] == float(logits[i])
        assert r['flags'] == pred['flags'] and r['current_unknown'] == pred['current_unknown']
        assert r['flags']['C_current'] == (r['flags']['A_current'] or r['logit'] >= HIGH)
        key = r['base_group_id'], r['frame_in_clip']
        assert r['layout_relation'] not in triplets[key]
        triplets[key][r['layout_relation']] = r
        d = geometry_descriptor(geo, a, b, boxes)
        expected = all(d['lower'][k] <= hi+1e-9 and d['upper'][k] >= lo-1e-9
            for k, (lo, hi) in enumerate(((-.3,.3),(-.2,.9),(.3,3.))))
        assert expected == r['truth']
        descriptors.append(d)
    records, max_mismatch = [], 0.
    for key, members in sorted(triplets.items()):
        assert set(members) == set(REL)
        seq = [members[r] for r in REL]; indices = [r['index'] for r in seq]
        max_mismatch = max(max_mismatch, verify_pair([spec['cases'][i] for i in indices], [geos[i] for i in indices]))
        assert seq[0]['truth'] == seq[1]['truth'] and not seq[2]['truth']
        for left, right in ((0, 2), (1, 2), (0, 1)):
            pair = pair_record(seq[left], seq[right], descriptors, ranges)
            pair['primary'] = seq[0]['truth']
            pair['contrast'] = REL[left]+'_vs_'+REL[right]
            records.append(pair)
    assert len(triplets) == 384 and sum(r['primary'] for r in records) == 173*3
    groups = sorted({r['base_group_id'] for r in rows})
    failed = sorted({r['base_group_id'] for r in rows if r['layout_relation'] == 'OUTSIDE'
        and r['flags']['C_current'] and not r['flags']['A_current']})
    summary = {}
    for contrast in ('INSIDE_vs_OUTSIDE', 'BOUNDARY_vs_OUTSIDE', 'INSIDE_vs_BOUNDARY'):
        all_pairs = [p for p in records if p['contrast'] == contrast]
        primary = [p for p in all_pairs if p['primary']]
        summary[contrast] = dict(all_frames=summarize(all_pairs), primary=summarize(primary),
            negative_depth_control=summarize([p for p in all_pairs if not p['primary']]),
            known_failed_groups=summarize([p for p in primary if p['base_group_id'] in failed]),
            other_groups=summarize([p for p in primary if p['base_group_id'] not in failed]),
            by_footprint={s:summarize([p for p in primary if p['footprint'] == s])
                for s in ('same_set', 'changed_set', 'empty_endpoint')},
            by_edge={s:summarize([p for p in primary if p['edge'] == s])
                for s in ('same_column', 'changed_column', 'outside_fov')},
            by_group={g:summarize([p for p in primary if p['base_group_id'] == g]) for g in groups})
    metrics = {}
    for stratum in ('Core', 'Boundary'):
        subset = [r for r in rows if (r['layout_relation'] == 'BOUNDARY') == (stratum == 'Boundary')]
        metrics[stratum] = {}
        for arm in ('A_current', 'C_current', 'A_hold', 'C_hold'):
            metrics[stratum][arm] = dict(TP=sum(r['truth'] and r['flags'][arm] for r in subset),
                FP=sum(not r['truth'] and r['flags'][arm] for r in subset),
                FN=sum(r['truth'] and not r['flags'][arm] for r in subset),
                unknown=sum(r['current_unknown'][arm] for r in subset))
    write(OUT/'pair-records.json', records)
    write(OUT/'geometry-descriptors.json', descriptors)
    charts(rows, triplets, descriptors, identities)
    result = dict(status='COMPLETE_DESCRIPTIVE_DIAGNOSTIC', groups=16, triplets=384,
        primary_triplets=173, negative_depth_triplets=211, failed_groups=failed,
        max_actual_nonlateral_mismatch_m=max_mismatch, cutoff=HIGH,
        summary=summary, unchanged_metrics=metrics,
        inference_runs=0, fits=0, original_test_access=False,
        limits=['Consumed Development; no fresh confirmation', 'Geometric footprint is not observed return ownership',
            'Same-zone strata do not imply equal RGB or ToF', 'Shared seed not identical noise',
            'Frame strata cluster by group; no causal inference or representation-loss proof'])
    write(OUT/'result.json', result)
    assert all(digest(SOURCE/name) == value for name, value in seals.items())
    assert digest(Path(__file__)) == read(OUT/'input-seal.json')['code_sha256']
    write(OUT/'output-seal.json', {p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'output-seal.json'})
    print(json.dumps({k:v for k,v in result.items() if k not in ('summary', 'limits')}))
    for contrast, s in summary.items():
        print(contrast, json.dumps({k:v for k,v in s.items() if k != 'by_group'}))


if __name__ == '__main__':
    main()
