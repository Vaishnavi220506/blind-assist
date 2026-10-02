"""Evaluator-only all-scene heading truth from physical box surfaces.

The future corridor centre is x=z*tan(delta); y and longitudinal query bounds
stay fixed. Each clipped physical face yields an interval of admissible slopes.
Gaussian heading probabilities integrate the union exactly. Scores never enter
the geometry or membership definitions. This is an assumed straight path, not
observed body motion, future reactions, occlusion or collision severity.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
SOURCE = WORK/'cnh-margin-confirm-20261002'
ROWS = WORK/'cnh-rgb-confirmed-m3-gate-20261002/rows.npz'
OUT = WORK/'cnh-rgb-scene-heading-gate-20261002'
EPS = 1e-8
WIDTHS = (.30, .28, .25, .40)
Y_BOUNDS = ((-.20, .42), (.42, .90))
Z_BOUNDS = (.30, 3.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_keys(data):
    return [(int(u), int(c), int(q)) for u, c, q in
            zip(data['unit'], data['config'], data['group'])]


def clipped_faces(boxes, query):
    """Return x/z corners of six physical faces, clipped only in y and z.

    Two free face dimensions must retain positive extent. A box enclosing the
    query contributes no artificial interior surface: only its original faces
    can survive. Faces exactly at the query y/z boundary are excluded by EPS.
    """
    if query not in (0, 1):
        raise ValueError('query must be HEAD=0 or BODY=1')
    yl, yh = Y_BOUNDS[query]
    output = []
    for box in boxes:
        low, high = np.asarray(box['lo'], float), np.asarray(box['hi'], float)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
            raise ValueError('finite positive-volume box required')
        for axis in range(3):
            for value in (low[axis], high[axis]):
                lo, hi = low.copy(), high.copy()
                lo[axis] = hi[axis] = value
                lo[1:] = np.maximum(lo[1:], [yl+EPS, Z_BOUNDS[0]+EPS])
                hi[1:] = np.minimum(hi[1:], [yh-EPS, Z_BOUNDS[1]-EPS])
                free = [j for j in range(3) if j != axis]
                if np.any(hi < lo) or np.any(hi[free] <= lo[free]):
                    continue
                x = np.array([lo[0], lo[0], hi[0], hi[0]])
                z = np.array([lo[2], hi[2], lo[2], hi[2]])
                assert np.all(z > 0)
                output.append((x, z))
    return output


def merge_intervals(intervals):
    """Union overlapping intervals; preserve touching open-boundary holes."""
    merged = []
    for lo, hi in sorted(intervals):
        assert np.isfinite([lo, hi]).all() and lo < hi
        if merged and lo < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([float(lo), float(hi)])
    return np.asarray(merged, float).reshape(-1, 2)


def face_intervals(faces, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError('positive finite corridor half-width required')
    return merge_intervals([(float(np.min((x-width)/z)), float(np.max((x+width)/z)))
                            for x, z in faces])


def contains(intervals, slope, closed=False):
    if closed:
        return bool(np.any((intervals[:, 0] <= slope) & (slope <= intervals[:, 1])))
    return bool(np.any((intervals[:, 0] < slope) & (slope < intervals[:, 1])))


def interval_probability(intervals, sig, *, closed=False):
    """Integrate all tan branches of delta~N(0,sig degrees); no pseudo-rows."""
    sig = float(sig)
    if not np.isfinite(sig) or sig < 0:
        raise ValueError('heading sigma must be finite and nonnegative')
    if sig == 0:
        return float(contains(intervals, 0., closed=closed))
    if not len(intervals):
        return 0.
    sigma = np.deg2rad(sig)
    angles = np.arctan(intervals)
    # Include all branches intersecting +/-9 sigma; omitted mass < 3e-19.
    branches = int(np.ceil(9*sigma/np.pi))+1
    total = 0.
    for k in range(-branches, branches+1):
        bounds = (angles+k*np.pi)/sigma
        # Upper-tail subtraction avoids cancellation for wholly positive angles.
        mass = np.where(bounds[:, 0] > 0,
                        ndtr(-bounds[:, 0])-ndtr(-bounds[:, 1]),
                        ndtr(bounds[:, 1])-ndtr(bounds[:, 0]))
        total += float(mass.sum())
    assert -.000000000001 <= total <= 1.000000000001
    return float(np.clip(total, 0., 1.))


def build_intervals(data, manifest):
    """Geometry cache reusable across sigmas; contains no scores or outcomes."""
    lookup = {(int(m['unit']), int(m['config'])): m for m in manifest}
    assert len(lookup) == len(manifest), 'duplicate manifest scene'
    keys = row_keys(data)
    assert len(set(keys)) == len(keys)
    assert set(keys) == {(*key, q) for key in lookup for q in (0, 1)}
    for i, (u, c, q) in enumerate(keys):
        m = lookup[(u, c)]
        if 'split' in data:
            assert str(data['split'][i]) == str(m['split'])
        if 'off' in data:
            assert np.isfinite(data['off'][i]) == (q == int(m['group']))
    intervals = {w: [] for w in WIDTHS}
    faces_count = 0
    for u, c, q in keys:
        faces = clipped_faces(lookup[(u, c)]['boxes'], q)
        faces_count += len(faces)
        for w in WIDTHS:
            intervals[w].append(face_intervals(faces, w))
    return dict(keys=keys, intervals=intervals, clipped_physical_faces=faces_count,
                geometry='all six physical faces; fixed y/z interiors; centre x=z*tan(delta)',
                epsilon_yz_m=EPS)


def build_truth(data, manifest, sig, *, interval_cache=None):
    """Return (membership arrays, receipt), using only metadata and boxes.

    clear_same/other partition by original target-query ownership (finite off),
    not by whichever surface causes future contact. Closed width .40 at sigma0
    assigns exact ten-centimetre tangency to acceptable near-pass. Contact and
    contracted widths use strict lateral boundaries.
    """
    cache = build_intervals(data, manifest) if interval_cache is None else interval_cache
    assert cache['keys'] == row_keys(data), 'interval cache row order changed'
    p = {w: np.array([interval_probability(x, sig, closed=(w == .40))
                      for x in cache['intervals'][w]]) for w in WIDTHS}
    for narrow, broad in ((.25, .28), (.28, .30), (.30, .40)):
        assert np.all(p[narrow] <= p[broad]+1e-12), 'nested widths violated'
    difference = lambda broad, narrow: np.maximum(0., p[broad]-p[narrow])
    truth = {'contact0-2': difference(.30, .28), 'contact2-5': difference(.28, .25),
             'contact>5': p[.25], 'contactall': p[.30],
             'pass0-10': difference(.40, .30), 'clear_merged': 1-p[.40]}
    target = np.isfinite(data['off'])
    truth['clear_same'] = truth['clear_merged']*target
    truth['clear_other'] = truth['clear_merged']*(~target)
    np.testing.assert_allclose(truth['contactall']+truth['pass0-10']+truth['clear_merged'], 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(truth['contact0-2']+truth['contact2-5']+truth['contact>5'], truth['contactall'], atol=1e-12, rtol=0)
    receipt = dict(sigma_heading_deg=float(sig), rows=len(target), widths_m=list(WIDTHS),
        integration='Gaussian CDF of union of atan slope intervals, all relevant tan branches; omitted tail <3e-19',
        boundary='y/z shrink1e-8m; sigma0 lateral contact strict and width0.40 closed',
        nominal_clear_ownership='finite original target offset => same, nonfinite => other; not future-surface identity',
        expected_memberships={k: float(v.sum()) for k, v in truth.items()},
        clipped_physical_faces=cache['clipped_physical_faces'],
        limits=['all visible and occluded physical surfaces; no sensor evidence',
                'straight future centreline shear, not rigid camera rotation or measured body trajectory',
                'contact bins are deepest all-scene lateral intrusion bands, not object-specific target-margin bins'])
    return truth, receipt


def mesh_predicate(boxes, q, width, slope, closed=False):
    """Independent triangle mesh clipping, used only for geometry fixtures."""
    import cnh_proposal_attribution_scenes as S
    if not boxes:
        return False
    triangles = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in boxes])
    triangles[:, :, 0] -= triangles[:, :, 2]*slope
    xeps = 0. if closed else EPS
    low = [-width+xeps, Y_BOUNDS[q][0]+EPS, Z_BOUNDS[0]+EPS]
    high = [width-xeps, Y_BOUNDS[q][1]-EPS, Z_BOUNDS[1]-EPS]
    return bool(len(S.clip_triangles(triangles, np.asarray(low), np.asarray(high))))


def fixtures():
    box = lambda lo, hi: dict(lo=lo, hi=hi)
    cases = {
        'head_shallow': [box([.29, -.1, 1.], [.39, .26, 1.1])],
        'body_shallow': [box([-.39, .5, 1.], [-.29, .84, 1.1])],
        'nearpass': [box([.35, -.1, 1.], [.45, .26, 1.2])],
        'contact_tangent': [box([.30, -.1, 1.], [.40, .26, 1.1])],
        'pass_tangent': [box([.40, -.1, 1.], [.50, .26, 1.1])],
        'height_tangent': [box([.1, .90, 1.], [.2, 1., 1.1])],
        'longitudinal_tangent': [box([.1, -.1, 3.], [.2, .26, 3.1])],
        'enclosing_solid': [box([-1., -1., -.5], [1., 2., 4.])],
        'mixed': [box([.45, -.1, .7], [.55, .26, .9]), box([-.5, .5, 2.], [-.4, .84, 2.3])],
        'floor': [box([-8., 1.65, -8.], [8., 1.8, 9.])],
        'empty': [],
    }
    checked = 0
    for name, boxes in cases.items():
        for q in (0, 1):
            faces = clipped_faces(boxes, q)
            for w in WIDTHS:
                intervals = face_intervals(faces, w)
                for degrees in (-12., -5., -1., 0., 1., 5., 12.):
                    slope = float(np.tan(np.deg2rad(degrees)))
                    got = contains(intervals, slope, closed=(w == .40))
                    expected = mesh_predicate(boxes, q, w, slope, closed=(w == .40))
                    assert got == expected, (name, q, w, degrees, got, expected, intervals)
                    checked += 1
    # Open unions must not incorrectly fill a single missing tangent slope.
    intervals = merge_intervals([(-1., 0.), (0., 1.)])
    assert not contains(intervals, 0.) and contains(intervals, 0., closed=True)
    # A scalar-depth side face has a known symmetric closed-form probability.
    intervals = np.array([[-.01, .01]])
    exact = 2*ndtr(np.arctan(.01)/np.deg2rad(1.))-1
    np.testing.assert_allclose(interval_probability(intervals, 1.), exact, atol=1e-14, rtol=0)
    # Closed-set endpoint contact can be a line/point: triangle area vanishes.
    tangent = face_intervals(clipped_faces(cases['head_shallow'], 0), .30)
    endpoint = float(tangent[0, 0])
    assert not contains(tangent, endpoint) and contains(tangent, endpoint, closed=True)
    return dict(mesh_clip_cases=checked, cases=list(cases), analytic_cdf=True,
        open_union_tangent_hole=True,
        boundary_note='closed-set line/point tangencies count acceptable near-pass; independent mesh comparison uses finite-area cases; distinction has zero continuous-heading mass')


def check_dataset():
    """Geometry-only acceptance check; never loads model scores or policies."""
    dest = OUT/'geometry-check.json'
    assert not dest.exists(), 'preserve existing completed check'
    toy = fixtures()
    provenance_path = ROWS.parent/'input-check.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    assert provenance['final_travel_identity_units'] == 144
    assert provenance['final_travel_identity_max_abs'] <= 1e-12
    manifest_path = SOURCE/'scene_manifest.json'
    assert sha(manifest_path) == provenance['input_sha256']['scene_manifest.json']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    with np.load(ROWS, allow_pickle=False) as z:
        data = {k: z[k] for k in ('split', 'unit', 'config', 'group', 'off', 'label')}
    cache = build_intervals(data, manifest)
    truth, receipt = build_truth(data, manifest, 0., interval_cache=cache)
    np.testing.assert_array_equal(truth['contactall'], data['label'])
    finite = np.isfinite(data['off'])
    off = data['off']
    for name, lo, hi in [('contact0-2', 0., .02), ('contact2-5', .02, .05), ('contact>5', .05, np.inf)]:
        original = finite & (off > lo) & (off <= hi)
        np.testing.assert_array_equal(truth[name], original.astype(float), err_msg=name)
    old_clear = (finite & (off < -.10)) | (~finite & (data['label'] == 0))
    reclassified = old_clear & (truth['pass0-10'] == 1)
    assert len(off) == 11520 and old_clear.sum() == 7433 and reclassified.sum() == 384
    assert truth['clear_merged'].sum() == 7049
    assert not np.any(old_clear & (truth['contactall'] > 0))
    report = dict(status='PASS', scope='geometry/truth acceptance only; no policy or score reevaluation',
        fixtures=toy, sigma0=receipt, contact_label_matches=len(off), target_bin_matches=len(off)*3,
        original_clear=int(old_clear.sum()), clear_to_nearpass=int(reclassified.sum()),
        remaining_allscene_clear=int(truth['clear_merged'].sum()),
        split_reclassified={s: int(np.sum(reclassified & (data['split'] == s))) for s in ('calib', 'evaluation')},
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROWS, manifest_path, provenance_path]})
    OUT.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'contact_label_matches', 'original_clear', 'clear_to_nearpass', 'remaining_allscene_clear')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['fixtures', 'check'], required=True)
    args = p.parse_args()
    print(json.dumps(fixtures())) if args.stage == 'fixtures' else check_dataset()
