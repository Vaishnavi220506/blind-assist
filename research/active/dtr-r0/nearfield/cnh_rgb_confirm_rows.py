"""CPU-only metadata/score adapter for the completed margin confirmation run.

The original row order, query order and cached scores are preserved. Geometry
is evaluator-only all-box signed surface distance; it is not an RGB estimate.
No voxel reads, rendering, inference, training or writes occur in this module.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import cnh_margin_confirm_evaluate as MC
import cnh_boundary_oracle_geometry as GEOM

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-margin-confirm-20261002'
SUPPORT_M = (-.20, .15)
QUERY_ORDER = ('HEAD', 'BODY')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def input_paths(out=DEFAULT_OUT):
    """Small immutable inputs for the caller's freeze; no dataset arrays read."""
    out = Path(out)
    here = Path(__file__).resolve().parent
    source = ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/source'
    return [Path(__file__), Path(MC.__file__), Path(GEOM.__file__),
            here/'cnh_cvr_pilot.py', here/'cnh_proposal_attribution_scenes.py',
            source/'cnh_track_a_readout.py',
            out/'PLAN.json', out/'result.json', out/'rgb_handoff.json',
            out/'scene_manifest.json',
            *[out/f'scores_{a}.npz' for a in MC.ARMS],
            *[out/f'scores96000_{a}.npz' for a in MC.ARMS]]


def keys(rows):
    return [(str(s), int(u), int(c), int(q)) for s, u, c, q in
            zip(rows['split'], rows['unit'], rows['config'], rows['group'])]


def attach_geometry(rows, manifest, poses):
    """Attach by explicit keys, never by manifest iteration order."""
    lookup = {(str(m['split']), int(m['unit']), int(m['config'])): m for m in manifest}
    assert len(lookup) == len(manifest), 'duplicate scene manifest keys'
    row_keys = keys(rows)
    assert len(set(row_keys)) == len(row_keys), 'duplicate query rows'
    expected = {(*key, q) for key in lookup for q in (0, 1)}
    assert set(row_keys) == expected, 'query rows and manifest keys differ'
    distances = {}
    matches = 0
    for key, m in lookup.items():
        u = key[1]
        # Use the verified returned pose, rather than merely assuming identity.
        scene = dict(boxes=m['boxes'], travel=np.asarray([poses[u]], dtype=float))
        ds = GEOM.scene_distances(scene)
        assert ds.shape == (2, 2) and np.isfinite(ds).all(), f'invalid geometry {key}'
        np.testing.assert_array_equal(ds[0] < 0, np.asarray(m['labels'], dtype=bool),
                                      err_msg=f'full3d/manifest label mismatch {key}')
        distances[key] = ds
        matches += 2
    arrays = dict(rows)
    arrays['full3d'] = np.array([distances[k[:3]][0, k[3]] for k in row_keys])
    arrays['vertical_sides_xz'] = np.array([distances[k[:3]][1, k[3]] for k in row_keys])
    np.testing.assert_array_equal(arrays['full3d'] < 0, rows['label'].astype(bool))
    return arrays, matches


def load_rows(out=DEFAULT_OUT, splits=None, *, expected_hashes=None):
    """Return (original arrays + geometry, units, serialisable receipt).

    expected_hashes maps repository-relative file paths to SHA256 values from
    the caller's run plan. The loader itself never creates a plan or outputs.
    Geometry support is nominal signed intrusion [-.20,+.15] m.
    """
    out = Path(out)
    if expected_hashes is not None:
        for rel, digest in expected_hashes.items():
            assert MC.sha(ROOT/rel) == digest, f'frozen input changed: {rel}'
    plan = read(out/'PLAN.json')
    if splits is None:
        splits = plan['splits']
    assert {s: list(map(int, splits[s])) for s in ('calib', 'evaluation')} == plan['splits']
    assert tuple(plan['arms']) == MC.ARMS == ('NEAR', 'M3', 'M8'), 'model score-column order changed'
    original, units, hashes = MC.load_rows(out, splits)
    handoff = read(out/'rgb_handoff.json')
    result = read(out/'result.json')
    assert result['status'] == handoff['status'] == 'COMPLETE'
    assert result['verdict'] == handoff['primary_verdict'] == 'M3_CONFIRMED'
    assert handoff['evaluation_units'] == units['evaluation']
    assert handoff['calibration_units'] == units['calib']
    assert hashes == result['provenance']['input_sha256'], 'source result provenance differs'
    assert hashes['scene_manifest.json'] == handoff['scene_manifest_sha256']
    assert MC.sha(out/'PLAN.json') == result['provenance']['plan_sha256']
    expected_keys = [(s, u, c, q) for s in ('calib', 'evaluation') for u in units[s]
                     for c in range(40) for q in (0, 1)]
    assert keys(original) == expected_keys, 'original split/unit/config/HEAD-BODY row order changed'
    exports = {}
    for ai, arm in enumerate(MC.ARMS):
        descriptor = handoff['exports'][arm]
        assert descriptor['file'] == f'scores96000_{arm}.npz'
        export = out/descriptor['file']
        assert hashes[f'scores_{arm}.npz'] == descriptor['source_sha256']
        assert MC.sha(export) == descriptor['sha256'], f'{arm} exported file byte hash changed'
        with np.load(export, allow_pickle=False) as z:
            assert set(z.files) == set(map(str, units['evaluation']))
            for u in units['evaluation']:
                exported = z[str(u)]
                assert exported.shape == (40, 2) and np.isfinite(exported).all()
                for c in range(40):
                    loc = np.flatnonzero((original['split'] == 'evaluation') &
                                         (original['unit'] == u) & (original['config'] == c))
                    np.testing.assert_array_equal(original['group'][loc], [0, 1])
                    np.testing.assert_array_equal(original['scores'][loc, ai], exported[c])
        exports[arm] = dict(sha256=descriptor['sha256'], source_sha256=descriptor['source_sha256'],
                            exact_evaluation_values=96*40*2)
    # Import only metadata functions; neither constructs models nor invokes CUDA.
    from cnh_cvr_pilot import motion_metadata
    import cnh_proposal_attribution_scenes as SCENES
    np.testing.assert_array_equal(GEOM.QUERY_LOW, SCENES.QUERY_LOW)
    np.testing.assert_array_equal(GEOM.QUERY_HIGH, SCENES.QUERY_HIGH)
    poses = {}
    errors = []
    for u in units['calib']+units['evaluation']:
        _, travel, _ = motion_metadata(u, 0)
        assert travel.shape == (16, 4, 4) and np.isfinite(travel).all()
        final = travel[-1]
        np.testing.assert_allclose(final, np.eye(4), atol=1e-12, rtol=0,
                                   err_msg=f'unit{u}: nonidentity final travel pose')
        poses[u] = final.copy()
        errors.append(float(np.abs(final-np.eye(4)).max()))
    assert len(poses) == 144
    manifest = read(out/'scene_manifest.json')
    arrays, matches = attach_geometry(original, manifest, poses)
    finite_offsets = arrays['off'][np.isfinite(arrays['off'])]
    assert np.all((finite_offsets >= SUPPORT_M[0]) & (finite_offsets <= SUPPORT_M[1]))
    for name in original:
        np.testing.assert_array_equal(arrays[name], original[name])
    receipt = dict(status='PASS', source=str(out.relative_to(ROOT)), input_sha256=hashes,
        source_file_sha256={str(p.relative_to(ROOT)): MC.sha(p) for p in input_paths(out)},
        query_order=list(QUERY_ORDER), score_column_order=list(MC.ARMS), row_order='split/unit/config/group exact original order',
        rows=len(arrays['unit']), scenes=len(manifest), geometry_label_matches=matches,
        final_travel_identity_units=len(poses), final_travel_identity_max_abs=max(errors),
        motion_config=0, motion_note='travel schedule depends on unit; config affects noisy sensor poses only',
        score_export_checks=exports, nominal_offset_support_m=list(SUPPORT_M),
        geometry='original scene_distances on all manifest box surfaces at verified final travel pose; no visibility or target filtering',
        backend='CPU metadata and cached scalar scores only; no voxel/inference/render/training')
    return arrays, units, receipt


def check():
    """Toy alignment check only; does not access the actual confirmation data."""
    manifest = [dict(split='evaluation', unit=1, config=0, labels=[1, 0],
                     boxes=[dict(lo=[.29, -.10, 1.], hi=[.39, .26, 1.1])])]
    # Reverse query rows to prove the adapter keys rather than zips its inputs.
    original = dict(split=np.array(['evaluation']*2), unit=np.array([1, 1]),
                    config=np.array([0, 0]), group=np.array([1, 0]), label=np.array([0, 1]),
                    off=np.array([np.nan, .01]), scores=np.array([[3., 4., 5.], [6., 7., 8.]]))
    got, matches = attach_geometry(original, manifest, {1: np.eye(4)})
    np.testing.assert_allclose(got['full3d'], [.16, -.01], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got['vertical_sides_xz'], [-.01, -.01], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(got['scores'], original['scores'])
    assert matches == 2
    print('PASS: keyed query alignment, signed full3d labels, XZ no-height semantics, original scores preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['check'])
    parser.parse_args()
    check()
