"""Evaluator-only all-object three-level truth; no predictions or rendering.

Contact retains labels_for_all's interior convention. Acceptable pass expands
only x from half-width .30 to .40 m, closed in x and interior in y/z.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import cnh_corridor_labels as L
import cnh_proposal_attribution_scenes as S
import cnh_novel_structure as B
import cnh_novel_structure_scenes as N
import cnh_margin_confirm as MC

HERE = Path(__file__).resolve().parent
OUT = MC.SS.WORK / 'cnh-all-object-truth-20261002'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    MC.SS.save(path, value)


def provenance():
    """Reject changed replay code before reconstructing any existing geometry."""
    manifest_path = B.OUT / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    names = ('cnh_novel_structure_scenes.py', 'cnh_proposal_attribution_scenes.py',
             'cnh_corridor_labels.py')
    hashes = {name: sha(HERE / name) for name in names}
    mismatch = [name for name in names if hashes[name] != manifest[name]]
    if mismatch:
        raise ValueError(f'Context generator changed; use original frozen source: {mismatch}')
    source_plan = json.loads((MC.OUT / 'PLAN.json').read_text(encoding='utf8'))
    source_names = ('cnh_margin_confirm.py', 'cnh_near_range.py', 'cnh_structure_space.py',
                    'cnh_proposal_attribution_scenes.py')
    for name in source_names:
        expected = source_plan['source_sha256'][name]
        if sha(HERE / name) != expected or sha(MC.OUT / 'source' / name) != expected:
            raise ValueError(f'Source generator/snapshot changed: {name}')
    # The mesh/clip helpers are imported by S from retained v5, never live WIP.
    import sys
    geometry_path = Path(sys.modules['cnh_track_a_geometry'].__file__).resolve()
    if geometry_path.parent != S.SOURCE.resolve():
        raise ValueError('Geometry helper must resolve to retained v5 source')
    return dict(context_manifest_sha256=sha(manifest_path), replay_sha256=hashes,
                source_plan_sha256=sha(MC.OUT / 'PLAN.json'),
                source_manifest_sha256=sha(MC.OUT / 'scene_manifest.json'),
                frozen_mesh_path=str(geometry_path), frozen_mesh_sha256=sha(geometry_path))


def classify_boxes(boxes, query, pose):
    pose = np.asarray(pose, float)
    if not np.array_equal(pose, np.eye(4)):
        raise ValueError('Final travel pose must be exactly identity')
    lo, hi = L.QUERY_BOXES[L.CENTRAL_INDICES[query]].copy()
    wide_lo, wide_hi = lo + 1e-8, hi - 1e-8
    wide_lo[0], wide_hi[0] = -.40, .40
    contact_hits, expanded_hits = [], []
    for index, box in enumerate(boxes):
        triangles = S.box_mesh(box['lo'], box['hi'])
        if len(S.clip_triangles(triangles, lo + 1e-8, hi - 1e-8)):
            contact_hits.append(index)
        if len(S.clip_triangles(triangles, wide_lo, wide_hi)):
            expanded_hits.append(index)
    contact, expanded = bool(contact_hits), bool(expanded_hits)
    assert not contact or expanded
    return dict(all_category='contact' if contact else 'pass' if expanded else 'clear',
                contact_bool=contact, expanded_bool=expanded,
                contact_box_indices=contact_hits, expanded_box_indices=expanded_hits,
                background_expanded_bool=any(i > 0 for i in expanded_hits),
                background_box_indices=[i for i in expanded_hits if i > 0])


def record_scene(dataset, split, row, pose):
    group, off = int(row['group']), float(row['off'])
    labels = np.asarray(row['labels'], int)
    if group not in (0, 1) or not np.isfinite(off) or labels.shape != (2,):
        raise ValueError('Invalid target metadata')
    actual = L.labels_for_all(row['boxes'], np.asarray(pose)[None])[0, [2, 3]]
    if not np.array_equal(actual, labels):
        raise ValueError(f"Stored contact labels differ: {dataset}/{row['unit']}/{row['config']}")
    records = []
    for query in (0, 1):
        value = classify_boxes(row['boxes'], query, pose)
        assert value['contact_bool'] == bool(labels[query])
        old = 'contact' if labels[query] else 'pass' if query == group and -.10 <= off <= 0 else 'clear'
        if old == 'pass' and value['all_category'] != 'pass':
            raise ValueError('Original acceptable target pass must remain all-object pass')
        records.append(dict(dataset=dataset, split=split, unit=int(row['unit']), config=int(row['config']),
                            query=query, target_group=group, target_off=off, old_category=old,
                            family=row['family'], range=row.get('range'), **value))
    geometry = dict(dataset=dataset, split=split, unit=int(row['unit']), config=int(row['config']),
                    boxes=row['boxes'], final_pose=np.asarray(pose).tolist(), labels=labels.tolist(),
                    target_group=group, target_off=off, family=row['family'], range=row.get('range'))
    return records, geometry


def build():
    if not (OUT / 'PLAN.json').is_file():
        raise RuntimeError('PLAN.json must be written before geometry build')
    if (OUT / 'ledger.json').exists() or (OUT / 'geometry.json').exists():
        raise FileExistsError('Existing geometry output requires inspection; no silent overwrite')
    started = time.monotonic()
    inputs = provenance()
    source = json.loads((MC.OUT / 'scene_manifest.json').read_text(encoding='utf8'))
    expected = {(split, u, c) for split, units in MC.SPLITS.items() for u in units for c in range(40)}
    if len(source) != len(expected) or {(r['split'], r['unit'], r['config']) for r in source} != expected:
        raise ValueError('Source manifest identities differ from frozen 48/96 units x 40 scenes')
    ledger, geometry = [], []
    final_poses = {}
    for split, units in MC.SPLITS.items():
        for u in units:
            scenes = MC.NR.known(u)
            if not all(np.array_equal(s['travel'][-1], np.eye(4)) for s in scenes):
                raise ValueError(f'Source final pose is not identity: {u}')
            final_poses[u] = scenes[0]['travel'][-1]
    for i, row in enumerate(source):
        row = dict(row, family=row['cond'])
        records, g = record_scene('source', row['split'], row, final_poses[row['unit']])
        ledger.extend(records); geometry.append(g)
        if i % 400 == 0:
            save(OUT / 'progress_geometry.json', dict(dataset='source', scenes=i + 1, elapsed_s=time.monotonic()-started))
    unit_hashes = {}
    for u in range(70000, 70096):
        path = B.OUT / 'features/evaluation' / f'unit{u}.npz'
        unit_hashes[str(u)] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            data = {key: z[key] for key in ('family', 'margin', 'group', 'scene', 'frame', 'labels')}
        scenes = N.make_novel_scenes(u)
        assert len(scenes) == 36 and [s['config'] for s in scenes] == list(range(22, 58))
        for scene in scenes:
            c = scene['config']
            ids = np.flatnonzero(data['scene'] == c)
            assert np.array_equal(data['frame'][ids], np.arange(16))
            assert str(data['family'][c]) == scene['family']
            assert int(data['group'][c]) == scene['group']
            assert float(data['margin'][c]) == scene['margin']
            labels = data['labels'][ids[-1], [2, 3]]
            assert np.array_equal(scene['labels'][-1], labels)
            row = dict(unit=u, config=c, family=scene['family'], group=scene['group'],
                       off=-scene['margin'], labels=labels.tolist(), boxes=scene['boxes'],
                       range=float(scene['boxes'][0]['lo'][2]))
            records, g = record_scene('context', 'evaluation', row, scene['travel'][-1])
            ledger.extend(records); geometry.append(g)
        save(OUT / 'progress_geometry.json', dict(dataset='context', unit=u, elapsed_s=time.monotonic()-started))
        print('geometry context', u, round(time.monotonic()-started, 1), flush=True)
    assert len(ledger) == 18432 and len(geometry) == 9216
    assert len({(r['dataset'], r['unit'], r['config'], r['query']) for r in ledger}) == len(ledger)
    # Recheck immutable inputs before publishing outputs.
    assert provenance() == inputs
    inputs['context_feature_sha256'] = unit_hashes
    save(OUT / 'ledger.json', ledger)
    save(OUT / 'geometry.json', geometry)
    save(OUT / 'geometry_receipt.json', dict(status='COMPLETE', query_rows=len(ledger), scenes=len(geometry),
        ledger_sha256=sha(OUT / 'ledger.json'), geometry_sha256=sha(OUT / 'geometry.json'),
        inputs=inputs, evaluator_sha256=sha(__file__), plan_sha256=sha(OUT / 'PLAN.json'),
        elapsed_s=time.monotonic()-started, predictions_read=False, rendering=False))
    print('COMPLETE geometry', len(ledger), flush=True)


def check():
    def box(x, y=.0, z=1.):
        return dict(lo=[x, y, z], hi=[x+.05, y+.1, z+.1], rho=.5)
    pose = np.eye(4)
    cases = [(box(.299), 'contact'), (box(.30), 'pass'), (box(.40), 'pass'),
             (box(.400001), 'clear'), (box(.35, .5), 'clear'), (box(.35, z=3.1), 'clear'),
             (box(.35, .42), 'clear'), (box(.35, z=3.), 'clear')]
    for item, expected in cases:
        assert classify_boxes([item], 0, pose)['all_category'] == expected
    result = classify_boxes([box(.6), box(.35)], 0, pose)
    assert result['all_category'] == 'pass' and result['background_box_indices'] == [1]
    result = classify_boxes([box(.299), box(.35)], 0, pose)
    assert result['all_category'] == 'contact' and result['background_expanded_bool']
    assert classify_boxes([box(.35, .5)], 1, pose)['all_category'] == 'pass'
    # A solid enclosing the query but without a surface inside remains clear.
    enclosing = dict(lo=[-2., -2., -2.], hi=[4., 4., 4.], rho=.5)
    assert classify_boxes([enclosing], 0, pose)['all_category'] == 'clear'
    print('PASS synthetic surface boundaries, same-height-only expansion, contact precedence and background pass; no scores read')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'build'))
    args = parser.parse_args()
    check() if args.stage == 'check' else build()
