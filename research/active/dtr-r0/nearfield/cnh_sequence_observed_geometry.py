"""All-object sequence truth on retained observed travel; sigma_heading=0 only.

Reads physical scene metadata and reconstructs the original travel without
rendering or accessing scores. Query surfaces retain their full rigid transform.
The .9 m reference is the first observed-window target-front crossing, with
position/yaw interpolation between its adjacent saved travel poses.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import cnh_margin_confirm as MC
import cnh_proposal_attribution_scenes as S

HERE = Path(__file__).resolve().parent
OUT = MC.SS.WORK / 'cnh-observed-sequence-20261002'
LEDGER_OUT = MC.SS.WORK / 'cnh-all-object-truth-20261002'
FRAMES = np.arange(3, 16)
EPS = 1e-8
DEADLINE = .9
FRAME_S = .2
WIDTHS = (.30, .28, .25, .40)
Y_BOUNDS = ((-.20, .42), (.42, .90))
CATEGORIES = ('contact0-2cm', 'contact2-5cm', 'contact>5cm', 'pass0-10cm', 'clear')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    MC.SS.save(path, value)


def input_provenance():
    plan_path = MC.OUT / 'PLAN.json'
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    source_hashes = {}
    for name in ('cnh_margin_confirm.py', 'cnh_near_range.py', 'cnh_structure_space.py',
                 'cnh_proposal_attribution_scenes.py'):
        expected = plan['source_sha256'][name]
        if sha(HERE / name) != expected or sha(MC.OUT / 'source' / name) != expected:
            raise ValueError(f'Original travel generator or frozen source changed: {name}')
        source_hashes[name] = expected
    geometry_path = Path(sys.modules['cnh_track_a_geometry'].__file__).resolve()
    if geometry_path.parent != S.SOURCE.resolve():
        raise ValueError('Mesh/clip helper must come from retained v5 source')
    receipt_path = LEDGER_OUT / 'geometry_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt['status'] != 'COMPLETE' or sha(LEDGER_OUT / 'ledger.json') != receipt['ledger_sha256']:
        raise ValueError('Original all-object ledger receipt mismatch')
    manifest_sha = sha(MC.OUT / 'scene_manifest.json')
    if receipt['inputs']['source_manifest_sha256'] != manifest_sha:
        raise ValueError('Scene manifest differs from original all-object ledger')
    return dict(scene_manifest_sha256=manifest_sha, original_plan_sha256=sha(plan_path),
                original_source_sha256=source_hashes, ledger_sha256=receipt['ledger_sha256'],
                ledger_receipt_sha256=sha(receipt_path), frozen_geometry_path=str(geometry_path),
                frozen_geometry_sha256=sha(geometry_path))


def corners(box):
    low, high = np.asarray(box['lo'], float), np.asarray(box['hi'], float)
    if low.shape != (3,) or high.shape != (3,) or np.any(high <= low) or not np.isfinite([low, high]).all():
        raise ValueError('Finite positive-volume physical box required')
    return np.array([[low[j] if not (i >> j) & 1 else high[j] for j in range(3)] for i in range(8)])


def transform(vertices, pose):
    return (np.asarray(vertices) - pose[:3, 3]) @ pose[:3, :3]


def target_front(vertices, pose):
    return float(transform(vertices, pose)[..., 2].min())


def yaw_of(pose):
    pose = np.asarray(pose, float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.array_equal(pose[3], [0., 0., 0., 1.]):
        raise ValueError('Finite rigid homogeneous pose required')
    yaw = float(np.arctan2(pose[0, 2], pose[0, 0]))
    if not np.allclose(pose[:3, :3], S.ry(np.rad2deg(yaw)), atol=1e-12, rtol=0):
        raise ValueError('Reference interpolation only supports original yaw-only travel')
    return yaw


def interpolate_pose(left, right, alpha):
    yl, yr = yaw_of(left), yaw_of(right)
    delta = (yr - yl + np.pi) % (2*np.pi) - np.pi
    pose = np.eye(4)
    pose[:3, :3] = S.ry(np.rad2deg(yl + float(alpha)*delta))
    pose[:3, 3] = left[:3, 3] + float(alpha)*(right[:3, 3]-left[:3, 3])
    return pose


def deadline_reference(vertices, poses):
    """Return physical front ranges and model-independent first crossing."""
    poses = np.asarray(poses, float)
    if poses.shape != (13, 4, 4):
        raise ValueError('Expected original saved travel frames 3..15')
    for pose in poses:
        yaw_of(pose)
    ranges = np.asarray([target_front(vertices, p) for p in poses])
    empty = dict(covered=False, reference_pose=np.full((4, 4), np.nan),
                 reference_fraction=np.nan, reference_time_s=np.nan,
                 crossing_left_frame=-1, crossing_right_frame=-1, interpolation_alpha=np.nan)
    if ranges[0] < DEADLINE:
        return ranges, dict(empty, censor_reason='left_censored')
    crossings = np.flatnonzero(ranges <= DEADLINE)
    if not len(crossings):
        return ranges, dict(empty, censor_reason='right_censored')
    right = int(crossings[0])
    if right == 0:
        reference = poses[0].copy()
        alpha, left = 0., 0
    else:
        left = right - 1
        assert ranges[left] > DEADLINE >= ranges[right]
        lo, hi = 0., 1.
        for _ in range(50):
            mid = (lo + hi)/2
            if target_front(vertices, interpolate_pose(poses[left], poses[right], mid)) > DEADLINE:
                lo = mid
            else:
                hi = mid
        alpha = (lo + hi)/2
        reference = interpolate_pose(poses[left], poses[right], alpha)
    residual = abs(target_front(vertices, reference)-DEADLINE)
    if residual > 1e-10:
        raise ValueError(f'Reference bisection residual {residual}')
    fraction = float(FRAMES[left] + alpha) if right else float(FRAMES[0])
    return ranges, dict(covered=True, censor_reason='covered', reference_pose=reference,
                       reference_fraction=fraction, reference_time_s=(fraction-FRAMES[0])*FRAME_S,
                       crossing_left_frame=int(FRAMES[left]), crossing_right_frame=int(FRAMES[right]),
                       interpolation_alpha=alpha)


def clipped_yz_triangles(local, query):
    """Clip transformed physical triangles, not a transformed object's AABB."""
    local = np.asarray(local, float).reshape(-1, 3, 3)
    if query not in (0, 1):
        raise ValueError('query must be HEAD=0 or BODY=1')
    lo = np.array([Y_BOUNDS[query][0]+EPS, .30+EPS])
    hi = np.array([Y_BOUNDS[query][1]-EPS, 3.-EPS])
    if not len(local):
        return local
    mins, maxs = local[:, :, 1:].min(1), local[:, :, 1:].max(1)
    # Strict overlap excludes a physical face tangent only to a y/z boundary.
    keep = ((maxs > lo) & (mins < hi)).all(1)
    local, mins, maxs = local[keep], mins[keep], maxs[keep]
    whole = ((mins >= lo) & (maxs <= hi)).all(1)
    inside, partial = local[whole], local[~whole]
    if len(partial):
        # Finite x planes lie strictly beyond every original vertex, so only
        # the four y/z planes clip. Convex clipping cannot exceed those bounds.
        xmin, xmax = float(partial[:, :, 0].min())-1., float(partial[:, :, 0].max())+1.
        partial = S.clip_triangles(partial, [xmin, *lo], [xmax, *hi])
        inside = np.concatenate([inside, partial])
    if len(inside):
        area2 = np.linalg.norm(np.cross(inside[:, 1]-inside[:, 0], inside[:, 2]-inside[:, 0]), axis=1)
        inside = inside[area2 > 1e-15]
    return inside


def surface_category(local, query):
    tri = clipped_yz_triangles(local, query)
    if not len(tri):
        return 'clear'
    xmin, xmax = tri[:, :, 0].min(1), tri[:, :, 0].max(1)
    hits = {}
    for w in WIDTHS:
        eps = 0. if w == .40 else EPS
        lo, hi = -w+eps, w-eps
        if np.any((xmin >= lo) & (xmax <= hi)):
            hits[w] = True
            continue
        partial = tri[(xmin < hi) & (xmax > lo)]
        if not len(partial):
            hits[w] = False
            continue
        # The broad x interval screen alone cannot distinguish a zero-area
        # tangent edge from a physical surface. Clip remaining partial faces.
        yzlo, yzhi = partial[:, :, 1:].min(axis=(0, 1))-1., partial[:, :, 1:].max(axis=(0, 1))+1.
        hits[w] = bool(len(S.clip_triangles(partial, [lo, *yzlo], [hi, *yzhi])))
    assert not hits[.25] or hits[.28]
    assert not hits[.28] or hits[.30]
    assert not hits[.30] or hits[.40]
    if hits[.25]:
        return 'contact>5cm'
    if hits[.28]:
        return 'contact2-5cm'
    if hits[.30]:
        return 'contact0-2cm'
    return 'pass0-10cm' if hits[.40] else 'clear'


def summarize_categories(values):
    return {c: int(np.count_nonzero(values == c)) for c in (*CATEGORIES, 'censored')}


def build():
    if not (OUT / 'PLAN.json').is_file():
        raise RuntimeError('Write PLAN.json and run predeclaration before build')
    if (OUT / 'geometry.npz').exists() or (OUT / 'rows.json').exists():
        raise FileExistsError('Existing geometry output requires inspection; do not silently overwrite')
    started = time.monotonic()
    inputs = input_provenance()
    manifest = json.loads((MC.OUT / 'scene_manifest.json').read_text(encoding='utf8'))
    scene_lookup = {(r['unit'], r['config']): r for r in manifest}
    ordered = [(split, u, c) for split, units in MC.SPLITS.items() for u in units for c in range(40)]
    if len(scene_lookup) != len(manifest) or set(scene_lookup) != {(u, c) for _, u, c in ordered}:
        raise ValueError('Manifest must contain exactly original 48+96 units x 40 configurations')
    old = json.loads((LEDGER_OUT / 'ledger.json').read_text(encoding='utf8'))
    old = {(r['unit'], r['config'], r['query']): r for r in old if r['dataset'] == 'source'}
    if len(old) != 11520:
        raise ValueError('Original all-object source ledger must contain 11520 rows')
    values = {key: [] for key in ('unit', 'config', 'query', 'split', 'frame_ranges', 'frame_category',
              'clear_all', 'ref_category', 'covered', 'last_category', 'frame_poses', 'reference_pose',
              'reference_fraction', 'reference_time_s', 'crossing_left_frame', 'crossing_right_frame',
              'interpolation_alpha', 'censor_reason')}
    rows, nominal_range_error, reference_residual = [], 0., 0.
    nominal_min, nominal_max, step_min, step_max = np.inf, -np.inf, np.inf, -np.inf
    for split, units in MC.SPLITS.items():
        for u in units:
            originals = MC.NR.known(u)
            travel = np.asarray(originals[0]['travel'], float)
            if travel.shape != (16, 4, 4) or not np.array_equal(travel[-1], np.eye(4)):
                raise ValueError(f'Original travel/final identity changed: {u}')
            if not all(np.array_equal(x['travel'], travel) for x in originals):
                raise ValueError(f'Original configurations disagree on travel: {u}')
            poses = travel[FRAMES]
            for c in range(40):
                scene = scene_lookup[(u, c)]
                assert scene['split'] == split
                boxes = scene['boxes']
                vertex = corners(boxes[0])
                mesh = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in boxes])
                ranges, reference = deadline_reference(vertex, poses)
                assert abs(ranges[-1]-float(scene['range'])) < 1e-12
                nominal = float(scene['range']) + .16*(15-FRAMES)
                difference = ranges-nominal
                nominal_range_error = max(nominal_range_error, float(np.max(abs(difference))))
                nominal_min, nominal_max = min(nominal_min, float(difference.min())), max(nominal_max, float(difference.max()))
                steps = np.diff(ranges)
                step_min, step_max = min(step_min, float(steps.min())), max(step_max, float(steps.max()))
                if np.any(steps > 1e-10):
                    raise ValueError(f'Nonmonotonic target-front ranges require explicit review: {u}/{c}')
                local_frames = [transform(mesh, p) for p in poses]
                local_reference = transform(mesh, reference['reference_pose']) if reference['covered'] else None
                if reference['covered']:
                    reference_residual = max(reference_residual, abs(target_front(vertex, reference['reference_pose'])-.9))
                for q in (0, 1):
                    cats = np.asarray([surface_category(t, q) for t in local_frames])
                    last = str(cats[-1])
                    category3 = 'contact' if last.startswith('contact') else 'pass' if last == 'pass0-10cm' else 'clear'
                    if category3 != old[(u, c, q)]['all_category']:
                        raise ValueError(f'Final all-object category differs: {u}/{c}/{q}')
                    if last.startswith('contact'):
                        off = float(scene['off'])
                        expected = 'contact0-2cm' if 0 < off <= .02 else 'contact2-5cm' if .02 < off <= .05 else 'contact>5cm'
                        if q != int(scene['group']) or off <= 0 or last != expected:
                            raise ValueError(f'Final contact bin differs from original target: {u}/{c}/{q}')
                    ref_category = surface_category(local_reference, q) if reference['covered'] else 'censored'
                    entry = dict(unit=u, config=c, query=q, split=split, frame_ranges=ranges,
                                 frame_category=cats, clear_all=bool(np.all(cats == 'clear')),
                                 ref_category=ref_category, last_category=last, frame_poses=poses, **reference)
                    for key in values:
                        values[key].append(entry[key])
                    rows.append(dict(split=split, unit=u, config=c, query=q, target_group=int(scene['group']),
                                     target_off=float(scene['off']), final_range=float(scene['range']),
                                     family=scene['cond'], row=len(rows)))
            save(OUT / 'progress_geometry.json', dict(split=split, unit=u, query_rows=len(rows),
                elapsed_s=time.monotonic()-started))
            print('observed geometry', split, u, round(time.monotonic()-started, 1), flush=True)
    result = {key: np.asarray(value) for key, value in values.items()}
    result['frames'] = FRAMES
    assert len(rows) == 11520 and result['frame_ranges'].shape == (11520, 13)
    assert np.array_equal(result['covered'], result['ref_category'] != 'censored')
    assert input_provenance() == inputs
    out_path = OUT / 'geometry.npz'
    temp = OUT / 'geometry.partial.npz'
    np.savez_compressed(temp, **result)
    temp.replace(out_path)
    save(OUT / 'rows.json', rows)
    counts = {}
    for split in MC.SPLITS:
        mask = result['split'] == split
        counts[split] = dict(query_rows=int(mask.sum()), scenes=int(mask.sum())//2,
            clear_all=int(np.sum(result['clear_all'] & mask)),
            censor_reason={name: int(np.sum(mask & (result['censor_reason'] == name)))
                           for name in ('covered', 'left_censored', 'right_censored')},
            ref_category=summarize_categories(result['ref_category'][mask]),
            last_category=summarize_categories(result['last_category'][mask]),
            frame_category=summarize_categories(result['frame_category'][mask]))
    receipt = dict(status='COMPLETE', sigma_heading_deg=0, query_rows=len(rows), scenes=len(manifest),
        frames=FRAMES.tolist(), inputs=inputs, geometry_sha256=sha(out_path), rows_sha256=sha(OUT / 'rows.json'),
        output_sha256={'geometry.npz': sha(out_path),
        'rows.json': sha(OUT / 'rows.json')}, source_sha256=sha(__file__), plan_sha256=sha(OUT / 'PLAN.json'),
        elapsed_s=time.monotonic()-started, counts=counts, deadline_m=DEADLINE,
        reference_target_front_max_abs_error_m=reference_residual,
        observed_vs_old_scalar_progress_max_abs_m=nominal_range_error,
        observed_minus_old_scalar_progress_min_m=nominal_min, observed_minus_old_scalar_progress_max_m=nominal_max,
        target_front_step_min_m=step_min, target_front_step_max_m=step_max, all_frame_ranges_monotone_nonincreasing=True,
        contact_definition='Interior lateral .30/.28/.25 widths shrink1e-8; x=.40 closed; y/z shrink1e-8; positive-area physical surfaces only',
        reference_definition='First saved-window target eight-corner min-z crossing .9m; linear position and yaw interpolation plus bisection',
        clear_definition='All physical surfaces clear at all 13 observed travel poses',
        source_final_identity_and_all_object_ledger_matches=11520,
        scores_read=False, rendering=False, limits=['Observed cached simulator travel only; not real walking or stopping',
            'All-scene reference contact bins use deepest surface intrusion; target range defines the common deadline',
            'All-clear means clear at the 13 saved poses, not guaranteed clear between frames',
            'Unreached deadlines are right-censored; windows beginning inside .9m are left-censored',
            'Source scene family is consumed Development; no heading uncertainty or new confirmation'])
    save(OUT / 'geometry_receipt.json', receipt)
    print('COMPLETE observed geometry', json.dumps(counts), flush=True)


def check():
    box = lambda lo, hi: dict(lo=list(lo), hi=list(hi), rho=.5)
    cases = [(box([.29, -.1, 1.], [.39, .26, 1.1]), 'contact0-2cm'),
             (box([.28, -.1, 1.], [.38, .26, 1.1]), 'contact0-2cm'),
             (box([.25, -.1, 1.], [.35, .26, 1.1]), 'contact2-5cm'),
             (box([.249, -.1, 1.], [.35, .26, 1.1]), 'contact>5cm'),
             (box([.30, -.1, 1.], [.40, .26, 1.1]), 'pass0-10cm'),
             (box([.40, -.1, 1.], [.50, .26, 1.1]), 'pass0-10cm'),
             (box([.400001, -.1, 1.], [.50, .26, 1.1]), 'clear'),
             (box([.1, .42, 1.], [.2, .5, 1.1]), 'clear'),
             (box([.1, -.1, 3.], [.2, .26, 3.1]), 'clear'),
             (box([-4., -4., -4.], [4., 4., 4.]), 'clear')]
    for b, expected in cases:
        assert surface_category(S.box_mesh(b['lo'], b['hi']), 0) == expected
    # A rotated enclosing box: true surfaces stay outside; its AABB is not used.
    b = box([-5., -4., -5.], [5., 4., 5.])
    pose = np.eye(4); pose[:3, :3] = S.ry(30.)
    assert surface_category(transform(S.box_mesh(b['lo'], b['hi']), pose), 0) == 'clear'
    # This rotated box passes outside the top-right x/z query corner, while
    # its axis-aligned bounding box invents a face crossing the corridor.
    b = box([1.85, -.1, 1.], [1.90, .26, 5.])
    transformed = transform(S.box_mesh(b['lo'], b['hi']), pose)
    assert surface_category(transformed, 0) == 'clear'
    inflated = S.box_mesh(transformed.min(axis=(0, 1)), transformed.max(axis=(0, 1)))
    assert surface_category(inflated, 0).startswith('contact')
    # A vertex/edge-only lateral tangent has zero area and stays clear.
    tangent = np.array([[[.4, 0., 1.], [.5, .1, 1.], [.5, 0., 1.1]]])
    assert surface_category(tangent, 0) == 'clear'
    # Rotated thin surface: min-z must come from transformed corners, not centre.
    b = box([.2, -.1, .8], [.8, .26, 1.])
    assert abs(target_front(corners(b), pose)-(.2*.5+.8*np.cos(np.deg2rad(30)))) < 1e-12
    # Cross-check yawed mesh classifications against full six-plane clipping.
    for yaw in (-20., -5., 0., 13., 35.):
        pose[:3, :3] = S.ry(yaw)
        local = transform(S.box_mesh(b['lo'], b['hi']), pose)
        for q in (0, 1):
            got = surface_category(local, q)
            hits = {}
            for w in WIDTHS:
                xeps = 0. if w == .40 else EPS
                hits[w] = bool(len(S.clip_triangles(local,
                    [-w+xeps, Y_BOUNDS[q][0]+EPS, .3+EPS],
                    [w-xeps, Y_BOUNDS[q][1]-EPS, 3.-EPS])))
            expected = ('contact>5cm' if hits[.25] else 'contact2-5cm' if hits[.28] else
                        'contact0-2cm' if hits[.30] else 'pass0-10cm' if hits[.40] else 'clear')
            assert got == expected, (yaw, q, got, expected)
    poses = np.repeat(np.eye(4)[None], 13, axis=0)
    poses[:, 2, 3] = np.linspace(-1.92, 0., 13)
    b = box([.2, -.1, .6], [.3, .26, .7])
    ranges, ref = deadline_reference(corners(b), poses)
    assert ref['covered'] and abs(target_front(corners(b), ref['reference_pose'])-.9) < 1e-12
    assert ref['crossing_left_frame'] == 13 and ref['crossing_right_frame'] == 14
    assert abs(ref['reference_fraction']-13.125) < 1e-12
    _, missed = deadline_reference(corners(box([.2, -.1, 1.2], [.3, .26, 1.3])), poses)
    assert not missed['covered'] and missed['censor_reason'] == 'right_censored'
    _, started_inside = deadline_reference(corners(b), np.repeat(np.eye(4)[None], 13, axis=0))
    assert not started_inside['covered'] and started_inside['censor_reason'] == 'left_censored'
    # Nonzero yaw interpolation remains rigid and finds the geometric crossing.
    for i, yaw in enumerate(np.linspace(-16., 0., 13)):
        poses[i, :3, :3] = S.ry(yaw)
    _, turned = deadline_reference(corners(b), poses)
    assert turned['covered'] and abs(target_front(corners(b), turned['reference_pose'])-.9) < 1e-10
    print('PASS boundary bins, y/z interior, physical surfaces, rigid yaw transforms, crossing/censoring and interpolated deadline; no scores read')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('check', 'build'), required=True)
    args = parser.parse_args()
    check() if args.stage == 'check' else build()
