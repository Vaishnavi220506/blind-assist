"""Shared constant heading-bias truth on retained observed sequence geometry.

Only evaluator truth changes: local future centre x=z*tan(delta). The same
scene draw is shared by every object, both queries, 13 frames and reference.
Nominal travel, deadline coverage, ranges, observations and scores stay fixed.
This is not a physically replayed alternative global trajectory.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_sequence_observed_geometry as G
import cnh_rgb_scene_heading_truth as H

OUT = G.MC.SS.WORK / 'cnh-sequence-heading-20261002'
DRAWS = 400
SEED = 2026100214
SIGMAS = np.array([0., 1.])
CATEGORIES = np.array(['contact0-2cm', 'contact2-5cm', 'contact>5cm', 'pass0-10cm', 'clear'])
# Numerical endpoint handling only: at 3 m this is 3e-10 m, below the retained
# 1e-8 m interior convention. Exact/tiny-area tangencies use full mesh clipping.
ENDPOINT_TOL = 1e-10
DOMAINS = {
    'source': dict(folder=G.OUT, manifest=G.MC.OUT / 'scene_manifest.json', units=list(range(96000, 96096))),
    'target': dict(folder=G.MC.SS.WORK / 'cnh-sequence-transfer-20261002',
                   manifest=G.MC.SS.WORK / 'cnh-sequence-transfer-20261002/scene_manifest.json',
                   units=list(range(94000, 94096))),
}


def scene_slopes(unit, config):
    z = np.random.default_rng(np.random.SeedSequence([SEED, int(unit), int(config)])).standard_normal(DRAWS)
    return np.r_[0., np.tan(np.deg2rad(z))]


def slope_intervals(local, query):
    """One y/z clipping per physical triangle; no per-draw mesh clipping."""
    tri = G.clipped_yz_triangles(local, query)
    intervals, endpoints = {}, []
    for width in G.WIDTHS:
        h = width-(G.EPS if width != .40 else 0.)
        raw = np.column_stack((((tri[:, :, 0]-h)/tri[:, :, 2]).min(1),
                               ((tri[:, :, 0]+h)/tri[:, :, 2]).max(1))) if len(tri) else np.empty((0, 2))
        assert not len(raw) or (np.isfinite(raw).all() and np.all(raw[:, 0] < raw[:, 1]))
        intervals[width] = H.merge_intervals(raw.tolist())
        endpoints.extend(raw.ravel().tolist())
    return intervals, np.asarray(endpoints)


def category_hits(category):
    order = {'clear': 0, 'pass0-10cm': 1, 'contact0-2cm': 2, 'contact2-5cm': 3, 'contact>5cm': 4}
    rank = order[category]
    return {.40: rank >= 1, .30: rank >= 2, .28: rank >= 3, .25: rank >= 4}


def labels_from_hits(hits):
    return np.select([hits[.25], hits[.28], hits[.30], hits[.40]],
                     ['contact>5cm', 'contact2-5cm', 'contact0-2cm', 'pass0-10cm'], default='clear')


def sample_hits(local, query, intervals, endpoints, slopes):
    slopes = np.asarray(slopes, float)
    hits = {w: ((slopes[:, None] > spans[:, 0]) & (slopes[:, None] < spans[:, 1])).any(1)
            for w, spans in intervals.items()}
    if len(endpoints):
        ordered = np.unique(endpoints)
        at = np.searchsorted(ordered, slopes)
        left, right = np.maximum(0, at-1), np.minimum(len(ordered)-1, at)
        near = np.minimum(abs(slopes-ordered[left]), abs(slopes-ordered[right])) <= ENDPOINT_TOL
    else:
        near = np.zeros(len(slopes), bool)
    for index in np.flatnonzero(near):
        moved = local.copy()
        moved[:, :, 0] -= moved[:, :, 2]*slopes[index]
        exact = category_hits(G.surface_category(moved, query))
        for width in G.WIDTHS:
            hits[width][index] = exact[width]
    for narrow, broad in ((.25, .28), (.28, .30), (.30, .40)):
        assert np.all(~hits[narrow] | hits[broad])
    return hits, int(near.sum())


def reference_membership(hits):
    return np.column_stack([hits[.30] & ~hits[.28], hits[.28] & ~hits[.25],
                            hits[.25], hits[.40] & ~hits[.30]])


def analytic_reference(intervals):
    p = {w: H.interval_probability(spans, 1.) for w, spans in intervals.items()}
    for narrow, broad in ((.25, .28), (.28, .30), (.30, .40)):
        assert p[narrow] <= p[broad]+1e-12
    return np.maximum(0., [p[.30]-p[.28], p[.28]-p[.25], p[.25], p[.40]-p[.30]])


def input_data(domain):
    config = DOMAINS[domain]
    folder = config['folder']
    receipt_path = folder / 'geometry_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt['status'] != 'COMPLETE':
        raise ValueError(f'{domain}: incomplete retained geometry')
    paths = {'geometry.npz': folder / 'geometry.npz', 'rows.json': folder / 'rows.json',
             'geometry_receipt.json': receipt_path, 'scene_manifest.json': config['manifest']}
    hashes = {name: G.sha(p) for name, p in paths.items()}
    if hashes['geometry.npz'] != receipt['geometry_sha256'] or hashes['rows.json'] != receipt['rows_sha256']:
        raise ValueError(f'{domain}: retained geometry input hash mismatch')
    expected_manifest = (receipt['inputs']['scene_manifest_sha256'] if domain == 'source'
                         else receipt['scene_manifest_sha256'])
    if hashes['scene_manifest.json'] != expected_manifest:
        raise ValueError(f'{domain}: scene manifest hash mismatch')
    if domain == 'source' and receipt['source_sha256'] != G.sha(G.__file__):
        raise ValueError('Observed geometry implementation changed')
    if domain == 'target' and receipt['inputs']['observed_geometry_sha256'] != G.sha(G.__file__):
        raise ValueError('Transfer and current observed geometry differ')
    rows = json.loads(paths['rows.json'].read_text(encoding='utf8'))
    manifest = json.loads(config['manifest'].read_text(encoding='utf8'))
    with np.load(paths['geometry.npz'], allow_pickle=False) as z:
        data = {key: z[key] for key in ('unit', 'config', 'query', 'split', 'frames', 'covered',
                'frame_poses', 'reference_pose', 'frame_category', 'ref_category', 'clear_all')}
    assert np.array_equal(data.pop('frames'), G.FRAMES)
    mask = data['split'] == 'evaluation'
    all_rows = len(mask)
    assert len(rows) == all_rows
    rows = [row for row, keep in zip(rows, mask) if keep]
    data = {key: value[mask] for key, value in data.items()}
    expected = [(u, c, q) for u in config['units'] for c in range(40) for q in (0, 1)]
    actual = list(zip(data['unit'].tolist(), data['config'].tolist(), data['query'].tolist()))
    assert actual == expected and len(rows) == 7680
    assert [(r['unit'], r['config'], r['query']) for r in rows] == expected
    scenes = {(m['unit'], m['config']): m for m in manifest if m['split'] == 'evaluation'}
    assert set(scenes) == {(u, c) for u in config['units'] for c in range(40)}
    assert np.array_equal(data['covered'], data['ref_category'] != 'censored')
    assert np.array_equal(data['clear_all'], np.all(data['frame_category'] == 'clear', axis=1))
    return data, scenes, hashes


def domain_truth(domain, started):
    data, scenes, input_hashes = input_data(domain)
    weights = np.zeros((2, 7680, 5), float)
    analytic = np.zeros_like(weights)
    overlap = np.zeros((2, 7680), float)
    fallback = 0
    frame_matches, reference_matches = 0, 0
    for pair in range(3840):
        first = 2*pair
        u, c = int(data['unit'][first]), int(data['config'][first])
        scene = scenes[(u, c)]
        mesh = np.concatenate([G.S.box_mesh(b['lo'], b['hi']) for b in scene['boxes']])
        slopes = scene_slopes(u, c)
        assert np.array_equal(data['frame_poses'][first], data['frame_poses'][first+1])
        assert data['covered'][first] == data['covered'][first+1]
        local_frames = [G.transform(mesh, pose) for pose in data['frame_poses'][first]]
        local_reference = G.transform(mesh, data['reference_pose'][first]) if data['covered'][first] else None
        if local_reference is not None:
            assert np.array_equal(data['reference_pose'][first], data['reference_pose'][first+1])
        for q in (0, 1):
            index = first+q
            pass_by_frame, all_frame_intervals = [], []
            for f, local in enumerate(local_frames):
                intervals, endpoints = slope_intervals(local, q)
                hits, used = sample_hits(local, q, intervals, endpoints, slopes)
                fallback += used
                category = str(labels_from_hits(hits)[0])
                if category != str(data['frame_category'][index, f]):
                    raise ValueError(f'Sigma0 frame regression: {domain}/{u}/{c}/{q}/{f+3}: {category}')
                frame_matches += 1
                pass_by_frame.append(hits[.40])
                all_frame_intervals.extend(intervals[.40].tolist())
            clear = ~np.asarray(pass_by_frame).any(0)
            if bool(clear[0]) != bool(data['clear_all'][index]):
                raise ValueError('Sigma0 whole-window clear changed')
            weights[0, index, 4] = clear[0]
            weights[1, index, 4] = clear[1:].mean()
            analytic[0, index, 4] = clear[0]
            analytic[1, index, 4] = 1-H.interval_probability(H.merge_intervals(all_frame_intervals), 1.)
            if local_reference is not None:
                intervals, endpoints = slope_intervals(local_reference, q)
                hits, used = sample_hits(local_reference, q, intervals, endpoints, slopes)
                fallback += used
                if str(labels_from_hits(hits)[0]) != str(data['ref_category'][index]):
                    raise ValueError(f'Sigma0 reference regression: {domain}/{u}/{c}/{q}')
                reference_matches += 1
                members = reference_membership(hits)
                weights[0, index, :4] = members[0]
                weights[1, index, :4] = members[1:].mean(0)
                analytic[0, index, :4] = members[0]
                analytic[1, index, :4] = analytic_reference(intervals)
                overlap[0, index] = clear[0] & hits[.30][0]
                overlap[1, index] = np.mean(clear[1:] & hits[.30][1:])
        if c == 39:
            G.save(OUT / 'progress_truth.json', dict(domain=domain, unit=u, query_rows=first+2,
                elapsed_s=time.monotonic()-started, endpoint_full_clip_fallbacks=fallback))
            print('heading truth', domain, u, round(time.monotonic()-started, 1), flush=True)
    assert frame_matches == 7680*13 and reference_matches == int(data['covered'].sum())
    assert np.isfinite(weights).all() and np.isfinite(analytic).all()
    assert np.all((weights >= 0) & (weights <= 1)) and np.all((analytic >= -1e-12) & (analytic <= 1+1e-12))
    assert np.array_equal(weights[0], analytic[0])
    assert not np.any(weights[:, ~data['covered'], :4])
    # Five categories intentionally do not partition episodes: reference truth
    # and sampled-window clear are defined at different temporal scopes.
    diff = weights[1]-analytic[1]
    metadata = dict(input_sha256=input_hashes, sigma0_frame_matches=frame_matches,
        sigma0_reference_matches=reference_matches, sigma0_clear_episode_matches=7680,
        covered_query_rows=int(data['covered'].sum()), endpoint_full_clip_fallbacks=fallback,
        mc_minus_analytic={str(name): dict(max_abs=float(abs(diff[:, i]).max()),
            mean_abs=float(abs(diff[:, i]).mean()), mean_signed=float(diff[:, i].mean()),
            expected_count_difference=float(diff[:, i].sum())) for i, name in enumerate(CATEGORIES)},
        expected_memberships={str(sig): {str(name): float(weights[s, :, i].sum())
            for i, name in enumerate(CATEGORIES)} for s, sig in enumerate(SIGMAS)},
        analytic_expected_memberships={str(sig): {str(name): float(analytic[s, :, i].sum())
            for i, name in enumerate(CATEGORIES)} for s, sig in enumerate(SIGMAS)},
        sampled_clear_reference_contact_overlap_expected_episodes=overlap.sum(1).tolist())
    result = dict(unit=data['unit'], config=data['config'], query=data['query'], sigmas=SIGMAS,
                  categories=CATEGORIES, weights=weights, analytic_weights=analytic,
                  covered=data['covered'], sampled_clear_reference_contact_overlap=overlap)
    return result, metadata


def build():
    if not (OUT / 'PLAN.json').is_file():
        raise RuntimeError('Write PLAN.json and RUNS predeclaration before truth build')
    if any((OUT / name).exists() for name in ('truth_source.npz', 'truth_target.npz', 'truth_receipt.json')):
        raise FileExistsError('Existing truth output requires inspection, no silent overwrite')
    started = time.monotonic()
    source_hashes = {Path(m.__file__).name: G.sha(m.__file__) for m in (G, H)}
    source_hashes[Path(__file__).name] = G.sha(__file__)
    plan_hash = G.sha(OUT / 'PLAN.json')
    payload, domains = {}, {}
    for domain in ('source', 'target'):
        payload[domain], domains[domain] = domain_truth(domain, started)
    if G.sha(OUT / 'PLAN.json') != plan_hash:
        raise ValueError('Plan changed during geometry truth build')
    for domain in domains:
        _, _, current = input_data(domain)
        assert current == domains[domain]['input_sha256']
    for name, digest in source_hashes.items():
        assert G.sha(G.HERE / name) == digest
    output_hashes = {}
    for domain, data in payload.items():
        temp = OUT / f'truth_{domain}.partial.npz'
        np.savez_compressed(temp, **data)
        target = OUT / f'truth_{domain}.npz'
        temp.replace(target)
        output_hashes[domain] = G.sha(target)
    receipt = dict(status='COMPLETE', plan_sha256=plan_hash, output_sha256=output_hashes,
        input_sha256={domain: value['input_sha256'] for domain, value in domains.items()},
        source_sha256=source_hashes, domains=domains, draws=DRAWS, seed=SEED,
        draw_key='SeedSequence([2026100214, unit, config]); same draw for objects/HEAD/BODY/frames/reference',
        sigmas_deg=SIGMAS.tolist(), endpoint_tolerance_slope=ENDPOINT_TOL,
        elapsed_s=time.monotonic()-started, scores_read=False,
        geometry='First original rigid world-to-local transform; future centre x=z*tan(delta); original local y/z interior unchanged',
        analytic='Gaussian CDF of union of positive-area surface slope intervals; 13-frame union for all-clear; numerical area endpoints fall back to retained mesh predicate for sampled draws',
        limits=['Counterfactual constant local future-heading bias relative to each recorded body yaw; not a single replayed alternative global trajectory',
            'Nominal observed poses, target-front distances, .9m coverage/reference and score/threshold timing remain fixed',
            '400 draws integrate assumed Gaussian sensitivity, not extra independent examples or measured user/device uncertainty',
            'Monte Carlo is primary; analytic integration is a numerical cross-check and cannot replace a less favorable primary verdict',
            'All-clear is only the 13 sampled poses; reference lies between samples and may contact while sampled frames are all clear',
            'First four categories use covered reference poses; clear uses whole-window all-clear, so five memberships are not a partition',
            'Existing consumed Development; no rendering, new sensor observations or hardware claim',
            'Target near-range offset support lacks same-height targets farther than 10cm outside; not equal-distribution transfer'])
    G.save(OUT / 'truth_receipt.json', receipt)
    print('COMPLETE heading truth', json.dumps(output_hashes), flush=True)


def check():
    box = lambda lo, hi: dict(lo=list(lo), hi=list(hi))
    boxes = [box([.29,-.1,1.], [.4,.26,1.2]), box([.4,-.1,1.], [.5,.26,1.2]),
             box([-4.,-4.,-4.], [4.,4.,4.]), box([1.85,-.1,1.], [1.90,.26,5.]),
             box([-.5,.5,.6], [-.35,.84,.8])]
    slopes = np.tan(np.deg2rad([-2.3, -1., 0., 1., 2.3]))
    matched, fallbacks = 0, 0
    for b in boxes:
        for yaw in (-17., 0., 11.):
            pose = np.eye(4); pose[:3, :3] = G.S.ry(yaw)
            local = G.transform(G.S.box_mesh(b['lo'], b['hi']), pose)
            for q in (0, 1):
                intervals, endpoints = slope_intervals(local, q)
                actual, used = sample_hits(local, q, intervals, endpoints, slopes)
                fallbacks += used
                for i, m in enumerate(slopes):
                    moved = local.copy(); moved[:, :, 0] -= moved[:, :, 2]*m
                    for width in G.WIDTHS:
                        h = width-(G.EPS if width != .40 else 0.)
                        expected = bool(len(G.S.clip_triangles(moved,
                            [-h, G.Y_BOUNDS[q][0]+G.EPS, .3+G.EPS],
                            [h, G.Y_BOUNDS[q][1]-G.EPS, 3.-G.EPS])))
                        assert actual[width][i] == expected
                        matched += 1
    # Same draw over frames: mutually exclusive dangerous heading halves make
    # all-clear almost impossible, not the product of two half-clear rates.
    union = H.merge_intervals([[-1., 0.], [0., 1.]])
    assert 1-H.interval_probability(union, 1.) < 1e-12
    independent_wrong = (1-H.interval_probability(np.array([[-1., 0.]]), 1.))**2
    assert independent_wrong > .24
    np.testing.assert_array_equal(scene_slopes(123, 4), scene_slopes(123, 4))
    assert not np.array_equal(scene_slopes(123, 4), scene_slopes(123, 5))
    assert len(scene_slopes(123, 4)) == 401 and scene_slopes(123, 4)[0] == 0
    print('PASS', matched, 'synthetic interval/full-mesh predicates;', fallbacks,
          'endpoint fallbacks; shared-frame union and deterministic scene draws; no real datasets read')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('check', 'build'), required=True)
    args = parser.parse_args()
    check() if args.stage == 'check' else build()
