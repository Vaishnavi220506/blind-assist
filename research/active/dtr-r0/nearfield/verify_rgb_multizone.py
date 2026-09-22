"""Independent frozen RGB/multizone saved-record check; no alternate recipe.

Reuses OpenCV's prescribed Otsu and eight-connectivity primitives, but does not
import the experiment runner. Recounts footprints, components, anchors, queries,
full interval envelopes, paired results and private contributor coverage.
"""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
AXIS = ROOT / 'artifacts.local/work/ba-axis-evidence-20260921'
OUT = ROOT / 'artifacts.local/work/ba-rgb-multizone-20260921'
PAIRS = {'selection': [(702, 678), (711, 687)],
         'evaluation': [(486, 462), (487, 463), (494, 470), (1063, 1039), (1070, 1046)]}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same(a, b, path='root'):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + '.' + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + '[' + str(i) + ']')
    elif isinstance(a, float):
        assert b is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


def relation(left, right, lower, upper):
    focal = 640 / (2 * np.tan(np.deg2rad(50)))
    amin = (max(0, left - 1) - 320) / focal
    amax = (min(640, right + 1) - 320) / focal
    values = [a * z for a in (amin, amax) for z in (lower, upper)]
    if max(values) < -.3 - 1e-9 or min(values) > .3 + 1e-9:
        return 'OUTSIDE'
    if min(values) > -.3 + 1e-9 and max(values) < .3 - 1e-9:
        return 'INSIDE'
    return 'CROSSING'


def recount_image(image, boxes, zones):
    assert image.shape == (360, 640, 3)
    # Enumerate native pixel centres; avoid reusing the runner's ceil transform.
    centres_x, centres_y = np.arange(640) + .5, np.arange(360) + .5
    rects = []
    for y0, x0, y1, x1 in boxes:
        xs = np.flatnonzero((centres_x >= x0 * 640 / 256) & (centres_x < x1 * 640 / 256))
        ys = np.flatnonzero((centres_y >= y0 * 360 / 192) & (centres_y < y1 * 360 / 192))
        rects.append((int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1))
    left, top = min(r[0] for r in rects), min(r[1] for r in rects)
    right, bottom = max(r[2] for r in rects), max(r[3] for r in rects)
    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    full = np.zeros((360, 640), np.int32)
    components, offset = [], 0
    for polarity in (0, 255):
        count, labels = cv2.connectedComponents((binary == polarity).astype(np.uint8), connectivity=8)
        for label in range(1, count):
            ys, xs = np.nonzero(labels == label)
            cid = offset + label
            full[ys + top, xs + left] = cid
            x0, x1 = int(xs.min()) + left, int(xs.max()) + left + 1
            y0, y1 = int(ys.min()) + top, int(ys.max()) + top + 1
            clipped = x0 == left or x1 == right or y0 == top or y1 == bottom
            components.append(dict(component=cid, polarity=polarity, area=len(xs),
                                   bbox=[x0, y0, x1, y1], clipped=clipped))
        offset += count - 1
    for c in components:
        anchors = []
        if not c['clipped']:
            for z, (x0, y0, x1, y1) in enumerate(rects):
                members = set(np.unique(full[y0:y1, x0:x1]).tolist())
                if zones[z]['valid'] and members == {c['component']}:
                    anchors.append(z)
        intervals = [zones[z]['interval_m'] for z in anchors]
        common = [max(i[0] for i in intervals), min(i[1] for i in intervals)] if intervals else None
        adjacent = any(abs(a // 8 - b // 8) + abs(a % 8 - b % 8) == 1
                       for j, a in enumerate(anchors) for b in anchors[j + 1:])
        eligible = bool(len(anchors) >= 2 and adjacent and common[0] <= common[1])
        c.update(anchors=anchors, common_interval=common, eligible=eligible)
        if eligible:
            c['envelope'] = [min(i[0] for i in intervals), max(i[1] for i in intervals)]
    sy = [int((i + .5) * 360 / 192) for i in range(192)]
    sx = [int((i + .5) * 640 / 256) for i in range(256)]
    sampled = full[np.ix_(sy, sx)]
    queries = []
    for z, record in enumerate(zones):
        if not record['valid']:
            continue
        if not (record['depth_state'] == 'CONTAINED' and record['possible'] and record['horizontal_relation'] == 'CROSSING'):
            continue
        y0, x0, y1, x1 = boxes[z]
        counts = Counter(sampled[y0:y1, x0:x1].ravel().tolist())
        candidates = []
        for c in components:
            if not c['eligible'] or counts[c['component']] < 4:
                continue
            lo, hi = record['interval_m']
            if hi < c['common_interval'][0] or lo > c['common_interval'][1]:
                continue
            # Use envelope, not shared overlap, for geometry.
            envelope = [min(lo, c['envelope'][0]), max(hi, c['envelope'][1])]
            candidates.append(dict(component=c['component'], sampled_overlap=counts[c['component']],
                interval_envelope=envelope, relation=relation(c['bbox'][0], c['bbox'][2], *envelope)))
        queries.append(dict(zone=z, candidates=candidates, relation=candidates[0]['relation'] if len(candidates) == 1 else 'UNKNOWN'))
    return dict(otsu_threshold=float(threshold), footprint=[left, top, right, bottom],
                components=components, queries=queries), full


def main():
    assert not (OUT / 'independent-audit.json').exists(), 'Do not overwrite completed audit'
    checked, seals = 0, []
    for directory, names in ((SOURCE, ('observation-seal.json', 'prediction-seal.json', 'evaluation-seal.json')),
                             (AXIS, ('public-seal.json', 'analysis-seal.json')),
                             (OUT, ('public-seal.json', 'evaluation-seal.json'))):
        for name in names:
            seal = read(directory / name)
            assert seal['protocol_sha256'] == sha(directory / 'protocol.json')
            for file, digest in seal['hashes'].items():
                assert sha(directory / file) == digest, file
                checked += 1
            seals.append(str((directory / name).relative_to(ROOT)))
    protocol = read(OUT / 'protocol.json')
    for file, digest in protocol['inputs'].items():
        assert sha(ROOT / file) == digest, file
        checked += 1
    same({k: [list(p) for p in v] for k, v in PAIRS.items()}, protocol['pairs'])
    assert (OUT / 'protocol-before-run.md').read_bytes() == (HERE / 'RGB_MULTIZONE_PROTOCOL_20260921.md').read_bytes()
    receipt = read(OUT / 'extraction-receipt.json')
    same(dict(evaluator_values_read=False, private_lineage_hashed_only=True, raw_rgb_hashes_checked=14,
              frames=14, segmentation_recipes=1, parameter_retries=0), receipt)
    tree = ast.parse((HERE / 'rgb_multizone_probe.py').read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'probe')
    assert [a.arg for a in function.args.args] == ['rgb', 'boxes', 'zone_records']
    assert not {n.id for n in ast.walk(function) if isinstance(n, ast.Name)} & {
        'read', 'truth', 'lineage', 'SOURCE', 'AXIS', 'PAIRS'}
    ids = read(SOURCE / 'identities.json')
    public_zones = {r['index']: r for r in read(AXIS / 'public-descriptors.json')}
    saved = read(OUT / 'public-results.json')
    indices = sorted(i for pairs in PAIRS.values() for pair in pairs for i in pair)
    assert len(indices) == len(set(indices)) == len(saved) == 14
    assert [r['index'] for r in saved] == indices
    with np.load(SOURCE / 'observations.npz', allow_pickle=False) as obs:
        boxes = obs['boxes']
    masks, rebuilt = {}, {}
    for i, record in zip(indices, saved):
        assert sha(SOURCE / ids[i]['rgb_path']) == ids[i]['rgb_sha256']
        image = cv2.imread(str(SOURCE / ids[i]['rgb_path']))
        descriptor, mask = recount_image(image, boxes, public_zones[i]['zones'])
        actual = dict(id='transfer:' + ids[i]['id'], index=i, **descriptor)
        same(actual, record, 'public.' + str(i))
        with np.load(OUT / f'regions-{i:04d}.npz', allow_pickle=False) as payload:
            assert payload.files == ['labels']
            assert np.array_equal(mask, payload['labels']), ('mask', i)
        rebuilt[i], masks[i] = actual, mask
    truth = {r['index']: r for r in read(AXIS / 'joined-frame-results.json')}
    # Verify this is exactly the existing seven-error subset, not selected winners.
    for role, pairs in PAIRS.items():
        errors = sorted(r['index'] for r in truth.values() if r['role'] == role
            and not r['a'] and r['scores']['original'] >= 7.6612162590026855
            and r['axes']['category'] == 'LATERAL_NEGATIVE' and r['public']['witnesses']['contained_depth'])
        assert errors == sorted(n for n, _ in pairs)
    lineage = read(SOURCE / 'private-lineage.json')
    frames, contributor_total, assigned_total = [], 0, 0
    for role, pairs in PAIRS.items():
        for negative, positive in pairs:
            n, p = truth[negative], truth[positive]
            assert n['base_group_id'] == p['base_group_id'] and n['time_s'] == p['time_s']
            assert n['frame_in_clip'] == p['frame_in_clip'] and not n['truth'] and p['truth']
            assert n['layout_relation'] == 'OUTSIDE' and p['layout_relation'] == 'BOUNDARY'
            for i in (negative, positive):
                record, r = rebuilt[i], truth[i]
                assert r['role'] == role
                coverage = []
                for query in record['queries']:
                    if len(query['candidates']) != 1:
                        continue
                    trace = lineage[i]['traces'][query['zone']]
                    assert trace['zone_id'] == query['zone'] and trace['observed']
                    assert trace['distance_m'] == public_zones[i]['zones'][query['zone']]['range_m']
                    component = query['candidates'][0]['component']
                    kept = 0
                    for pixel in trace['pixel_indices']:
                        y, x = divmod(pixel, 256)
                        native_y, native_x = int((y + .5) * 360 / 192), int((x + .5) * 640 / 256)
                        kept += int(masks[i][native_y, native_x] == component)
                    total = len(trace['pixel_indices'])
                    coverage.append(dict(zone=query['zone'], contributors=total, covered=kept, complete=kept == total))
                    contributor_total += total
                    assigned_total += 1
                q = record['queries']
                frames.append(dict(id=record['id'], role=role, truth=r['truth'], group=r['base_group_id'],
                    components=len(record['components']), unclipped=sum(not c['clipped'] for c in record['components']),
                    maximum_anchor_count=max((len(c['anchors']) for c in record['components']), default=0),
                    eligible_components=sum(c['eligible'] for c in record['components']), queries=len(q),
                    assigned=sum(len(x['candidates']) == 1 for x in q), unknown=sum(x['relation'] == 'UNKNOWN' for x in q),
                    outside=sum(x['relation'] == 'OUTSIDE' for x in q), native_corridor_samples=r['native_target_corridor_samples'],
                    assignment_coverage=coverage))
    useful = (all(any(r['role'] == role and not r['truth'] and r['queries'] > 0 and r['outside'] == r['queries']
                      for r in frames) for role in PAIRS)
              and not any(r['truth'] and r['outside'] for r in frames)
              and all(c['complete'] for r in frames for c in r['assignment_coverage']))
    expected = dict(status='COMPLETE', scope='CONSUMED_DIAGNOSTIC', frames=frames,
        useful_local_feasibility=useful, assignments=assigned_total, private_lineage_values_read=bool(assigned_total),
        alerts_changed=False, original_test_activated=False, automatic_successor=False)
    same(expected, read(OUT / 'result.json'), 'evaluation')
    old = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
    assert not any((old / n).exists() for n in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'))
    audit = dict(status='PASS', audit_script_sha256=sha(Path(__file__)), verified_hash_entries=checked,
        verified_seals=seals, rgb_hashes_checked=14, reconstructed_component_masks=14,
        all_queries_and_anchors_exact=True, cohort_complete_and_pairs_valid=True, original_test_closed=True,
        source_alert_files_unchanged=True, assignments=assigned_total, contributor_incidences=contributor_total,
        all_assigned_contributors_covered=all(c['complete'] for r in frames for c in r['assignment_coverage']),
        useful_local_feasibility=useful, frames=frames,
        limits=['Otsu and eight-connected labeling use the prescribed OpenCV primitives; all component statistics, full-zone anchors and query logic are independently recounted.',
                'Contributor coverage concerns all observed winning-return sampled points in assigned queries, not an assertion that all 307 are target-corridor samples.',
                'No assignments exist on lateral-negative queries; UNKNOWN is unresolved ownership, not successful rejection.',
                'Static extraction boundary and saved seals support dataflow separation, not independently observable historical process access.',
                'Fourteen consumed selected diagnostic frames in three layouts; no new alert policy, training, parameter search or general transfer claim.'])
    with (OUT / 'independent-audit.json').open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', verified_hash_entries=checked, masks=14, assignments=assigned_total,
        contributor_incidences=contributor_total, useful_local_feasibility=useful, frames=frames), indent=2))


if __name__ == '__main__':
    main()
