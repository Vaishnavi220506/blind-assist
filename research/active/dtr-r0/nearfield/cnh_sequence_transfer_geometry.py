"""Observed-travel geometry adapter for existing near-range units 94000..94095.

No rendering, inference, training, score loading or calibration. Geometric
predicates and observed-window deadline semantics are inherited unchanged from
cnh_sequence_observed_geometry. This older offset distribution lacks target
same-height clear examples farther than 10 cm outside the body corridor.
"""
import argparse
import ast
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import cnh_sequence_observed_geometry as G
import cnh_near_range as NR
import cnh_corridor_labels as L

OUT = NR.SS.WORK / 'cnh-sequence-transfer-20261002'
UNITS = list(range(94000, 94096))
ORIGINAL_COMMIT = '2f08e2ef'
SOURCE_RELATIVE = 'research/active/dtr-r0/nearfield/cnh_near_range.py'


def geometry_ast(text):
    names = {'known', '_none_scene', '_target', 'scenes_for', 'SPLITS', 'PER', '_ORIGINAL'}
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = ast.dump(node.value, include_attributes=False)
    if set(found) != names:
        raise ValueError('Near-range geometry AST contract missing definitions')
    return found


def provenance():
    """Compare retained generation snapshots and original versioned geometry."""
    request_path = NR.OUT / 'generation_request.json'
    request = json.loads(request_path.read_text(encoding='utf8'))
    recorded = request['identity']
    checked = {}
    for name, expected in recorded.items():
        live, snapshot = G.HERE / name, NR.OUT / 'source' / name
        if G.sha(live) != expected or G.sha(snapshot) != expected:
            raise ValueError(f'Retained near-range generator/source differs: {name}')
        checked[name] = expected
    root = NR.SS.ROOT
    historical = subprocess.check_output(['git', '-C', str(root), 'show',
                                          ORIGINAL_COMMIT + ':' + SOURCE_RELATIVE])
    import hashlib
    historical_sha = hashlib.sha256(historical).hexdigest()
    first = json.loads((NR.OUT / 'results.json').read_text(encoding='utf8'))['script_sha256']
    later = json.loads((NR.OUT / 'results_res.json').read_text(encoding='utf8'))['script_sha256']
    live_sha = G.sha(NR.__file__)
    if historical_sha != first or live_sha != later:
        raise ValueError('Original/current near-range provenance differs from retained result receipts')
    if geometry_ast(historical.decode('utf8')) != geometry_ast(Path(NR.__file__).read_text(encoding='utf8')):
        raise ValueError('Near-range geometry changed since original generation recipe')
    source_geometry_receipt = json.loads((G.OUT / 'geometry_receipt.json').read_text(encoding='utf8'))
    if source_geometry_receipt['status'] != 'COMPLETE' or G.sha(G.__file__) != source_geometry_receipt['source_sha256']:
        raise ValueError('Observed-geometry implementation differs from completed source run')
    dependencies = source_geometry_receipt['inputs']['original_source_sha256']
    for name in ('cnh_structure_space.py', 'cnh_proposal_attribution_scenes.py', 'cnh_near_range.py'):
        if G.sha(G.HERE / name) != dependencies[name]:
            raise ValueError(f'Observed/near-range dependency mismatch: {name}')
    mesh_path = Path(source_geometry_receipt['inputs']['frozen_geometry_path'])
    if G.sha(mesh_path) != source_geometry_receipt['inputs']['frozen_geometry_sha256']:
        raise ValueError('Frozen triangle helper changed')
    terminal_hashes = {}
    for chunk in range(3):
        p = NR.OUT / f'generation_terminal_evaluation_c{chunk}of3.json'
        if json.loads(p.read_text(encoding='utf8'))['status'] != 'complete':
            raise ValueError(f'Original generation chunk incomplete: {chunk}')
        terminal_hashes[p.name] = G.sha(p)
    return dict(generation_request_sha256=G.sha(request_path), generation_identity_sha256=checked,
        generation_terminal_sha256=terminal_hashes, original_near_range_commit=ORIGINAL_COMMIT,
        original_near_range_sha256=historical_sha, current_near_range_sha256=live_sha,
        geometry_ast_unchanged=True, observed_geometry_sha256=G.sha(G.__file__),
        observed_geometry_receipt_sha256=G.sha(G.OUT / 'geometry_receipt.json'),
        dependency_sha256={name: G.sha(G.HERE / name) for name in dependencies},
        frozen_mesh_path=str(mesh_path), frozen_mesh_sha256=G.sha(mesh_path),
        provenance_limit='Original generation_request names base generators, not near_range; original committed near_range recipe is linked by its retained result script hash and unchanged geometry AST')


def validate_metadata(data, unit, scenes):
    if int(data['unit']) != unit or str(data['split']) != 'evaluation' or str(data['query_frame']) != 'travel':
        raise ValueError(f'Original feature identity mismatch: {unit}')
    if len(scenes) != 40 or [s['config'] for s in scenes] != list(range(40)):
        raise ValueError('Expected 40 original scenes per near-range unit')
    if len(data['scene']) != 640 or data['labels'].shape != (640, 6):
        raise ValueError('Expected 40 scenes x 16 frames x six labels')
    for scene in scenes:
        c = scene['config']
        ids = np.flatnonzero(data['scene'] == c)
        if not np.array_equal(data['frame'][ids], np.arange(16)):
            raise ValueError(f'Original frame identity mismatch: {unit}/{c}')
        if str(data['family'][c]) != scene['family'] or float(data['margin'][c]) != scene['margin'] or int(data['group'][c]) != scene['group']:
            raise ValueError(f'Original target metadata mismatch: {unit}/{c}')


def classify_all_frames(mesh, travel):
    """Reuse the shared physical-surface predicate for all six original queries."""
    labels = np.zeros((16, 6), np.int8)
    central = []
    for f, pose in enumerate(travel):
        local = G.transform(mesh, pose)
        centre = []
        for ix, offset in enumerate((-.3, 0., .3)):
            shifted = local.copy()
            shifted[:, :, 0] -= offset
            for q in (0, 1):
                category = G.surface_category(shifted, q)
                labels[f, 2*ix+q] = category.startswith('contact')
                if ix == 1:
                    centre.append(category)
        central.append(centre)
    return labels, np.asarray(central)


def build():
    if not (OUT / 'PLAN.json').is_file():
        raise RuntimeError('Write PLAN.json and RUNS predeclaration before build')
    if any((OUT / name).exists() for name in ('geometry.npz', 'rows.json', 'scene_manifest.json')):
        raise FileExistsError('Existing transfer geometry requires inspection, no silent overwrite')
    started = time.monotonic()
    inputs = provenance()
    values = {key: [] for key in ('unit', 'config', 'query', 'split', 'frame_ranges', 'frame_category',
              'clear_all', 'ref_category', 'covered', 'last_category', 'frame_poses', 'reference_pose',
              'reference_fraction', 'reference_time_s', 'crossing_left_frame', 'crossing_right_frame',
              'interpolation_alpha', 'censor_reason')}
    rows, manifest, feature_hashes = [], [], {}
    nominal_min, nominal_max, step_min, step_max = np.inf, -np.inf, np.inf, -np.inf
    reference_error, closed_difference_cells, closed_difference_scenes = 0., 0, 0
    for unit in UNITS:
        p = NR.OUT / 'features/evaluation' / f'unit{unit}.npz'
        feature_hashes[str(unit)] = G.sha(p)
        with np.load(p, allow_pickle=False) as z:
            data = {key: z[key] for key in ('unit', 'split', 'query_frame', 'scene', 'frame',
                                          'labels', 'family', 'margin', 'group')}
        scenes = NR.scenes_for(unit)
        validate_metadata(data, unit, scenes)
        for scene in scenes:
            c, boxes, travel = scene['config'], scene['boxes'], np.asarray(scene['travel'])
            if travel.shape != (16, 4, 4) or not np.array_equal(travel[-1], np.eye(4)):
                raise ValueError(f'Unexpected final travel pose: {unit}/{c}')
            ids = np.flatnonzero(data['scene'] == c)
            mesh = np.concatenate([G.S.box_mesh(b['lo'], b['hi']) for b in boxes])
            labels, central = classify_all_frames(mesh, travel)
            if not np.array_equal(labels, data['labels'][ids]):
                raise ValueError(f'All-16-frame six-query interior label replay differs: {unit}/{c}')
            difference = labels[:, [2, 3]] != scene['labels']
            if difference.any():
                # This is the unchanged original generator's interior/closed
                # check, not a label correction. Both definitions are retained.
                closed = L.labels_for_all(boxes, travel, boundary='closed')
                if not np.array_equal(closed[:, [2, 3]], scene['labels']) or not np.all(labels <= closed):
                    raise ValueError(f'Closed/interior difference exceeds original boundary convention: {unit}/{c}')
                closed_difference_cells += int(difference.sum()); closed_difference_scenes += 1
            if not np.array_equal(labels[-1, [2, 3]], scene['labels'][-1]):
                raise ValueError('Final central labels differ from original scene helper')
            poses = travel[G.FRAMES]
            vertex = G.corners(boxes[0])
            ranges, reference = G.deadline_reference(vertex, poses)
            final_range = float(scene['meta']['range'])
            if abs(ranges[-1]-final_range) > 1e-12:
                raise ValueError('Final target front differs from stored generator range')
            diff = ranges-(final_range+.16*(15-G.FRAMES))
            nominal_min, nominal_max = min(nominal_min, float(diff.min())), max(nominal_max, float(diff.max()))
            steps = np.diff(ranges)
            step_min, step_max = min(step_min, float(steps.min())), max(step_max, float(steps.max()))
            if np.any(steps > 1e-10):
                raise ValueError(f'Nonmonotone target front requires explicit review: {unit}/{c}')
            local_reference = G.transform(mesh, reference['reference_pose']) if reference['covered'] else None
            if reference['covered']:
                reference_error = max(reference_error, abs(G.target_front(vertex, reference['reference_pose'])-.9))
            off, group = -float(scene['margin']), int(scene['group'])
            manifest.append(dict(split='evaluation', unit=unit, config=c, group=group, off=off,
                range=final_range, cond=scene['family'], boxes=boxes, labels=labels[-1, [2, 3]].tolist()))
            for q in (0, 1):
                cats = central[G.FRAMES, q]
                last = str(cats[-1])
                if last.startswith('contact'):
                    expected = 'contact0-2cm' if 0 < off <= .02 else 'contact2-5cm' if .02 < off <= .05 else 'contact>5cm'
                    if q != group or off <= 0 or last != expected:
                        raise ValueError(f'Final contact bin differs from original target: {unit}/{c}/{q}')
                entry = dict(unit=unit, config=c, query=q, split='evaluation', frame_ranges=ranges,
                    frame_category=cats, clear_all=bool(np.all(cats == 'clear')),
                    ref_category=G.surface_category(local_reference, q) if reference['covered'] else 'censored',
                    last_category=last, frame_poses=poses, **reference)
                for key in values:
                    values[key].append(entry[key])
                rows.append(dict(split='evaluation', unit=unit, config=c, query=q, target_group=group,
                    target_off=off, final_range=final_range, family=scene['family'], row=len(rows)))
        G.save(OUT / 'progress_geometry.json', dict(unit=unit, query_rows=len(rows), elapsed_s=time.monotonic()-started))
        print('transfer geometry', unit, round(time.monotonic()-started, 1), flush=True)
    result = {key: np.asarray(value) for key, value in values.items()}
    result['frames'] = G.FRAMES
    assert len(rows) == 7680 and len(manifest) == 3840 and result['frame_ranges'].shape == (7680, 13)
    assert np.array_equal(result['covered'], result['ref_category'] != 'censored')
    assert provenance() == inputs
    inputs['feature_unit_sha256'] = feature_hashes
    temp = OUT / 'geometry.partial.npz'
    np.savez_compressed(temp, **result)
    temp.replace(OUT / 'geometry.npz')
    G.save(OUT / 'rows.json', rows); G.save(OUT / 'scene_manifest.json', manifest)
    counts = dict(query_rows=7680, scenes=3840, clear_all=int(result['clear_all'].sum()),
        censor_reason={name: int(np.sum(result['censor_reason'] == name)) for name in ('covered', 'left_censored', 'right_censored')},
        ref_category=G.summarize_categories(result['ref_category']),
        last_category=G.summarize_categories(result['last_category']),
        frame_category=G.summarize_categories(result['frame_category']))
    receipt = dict(status='COMPLETE', sigma_heading_deg=0, query_rows=7680, scenes=3840,
        frames=G.FRAMES.tolist(), inputs=inputs, geometry_sha256=G.sha(OUT / 'geometry.npz'),
        rows_sha256=G.sha(OUT / 'rows.json'), scene_manifest_sha256=G.sha(OUT / 'scene_manifest.json'),
        source_sha256=G.sha(__file__), plan_sha256=G.sha(OUT / 'PLAN.json'), counts={'evaluation': counts},
        elapsed_s=time.monotonic()-started, reference_target_front_max_abs_error_m=reference_error,
        observed_minus_old_scalar_progress_min_m=nominal_min, observed_minus_old_scalar_progress_max_m=nominal_max,
        target_front_step_min_m=step_min, target_front_step_max_m=step_max,
        all_frame_ranges_monotone_nonincreasing=True, original_label_cells_verified=96*40*16*6,
        original_closed_interior_difference_cells=closed_difference_cells,
        original_closed_interior_difference_scenes=closed_difference_scenes,
        scores_read=False, rendering=False, limits=['Existing consumed Development units, not fresh confirmation',
            'Nominal target intrusion support [-.10,+.15]m: no same-height target >10cm-outside clear support; not source-distribution equivalence',
            'Whole-scene clear means clear at 13 sampled observed poses, not continuous-time clearance',
            'Target-front .9m deadline and deepest whole-scene reference intrusion; no real stopping or hardware claim'])
    G.save(OUT / 'geometry_receipt.json', receipt)
    print('COMPLETE transfer geometry', json.dumps(counts), flush=True)


def check():
    G.check()
    p = provenance()
    assert p['geometry_ast_unchanged'] and UNITS == NR.SPLITS['evaluation']
    print('PASS original/current geometry AST, retained generator snapshots, observed geometry hash and evaluation IDs; no real scenes or scores processed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('check', 'build'), required=True)
    args = parser.parse_args()
    check() if args.stage == 'check' else build()
