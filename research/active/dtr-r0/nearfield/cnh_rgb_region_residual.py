"""Fixed SAM regions plus one residual patch per public coarse zone.

Only replacing the observation-derived label -1 with 32 changes prediction.
Reference arrays enter evaluation only. Original experiment outputs stay intact.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import cnh_rgb_region_partition as R

P, S, A, V, G, ROOT = R.P, R.S, R.A, R.V, R.G, R.ROOT
OUT = ROOT/'artifacts.local/work/cnh-rgb-region-residual-20261002'
RUN_ID = 'CNH_RGB_REGION_RESIDUAL_20261002'
ARM = 'sam_residual'


def residual_labels(labels):
    """RGB-only rule; zone separation happens in the existing region x zone scorer."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('integer HxW observed labels required')
    if np.any((labels < -1) | (labels > 31)):
        raise ValueError('original SAM labels must be -1 or 0..31')
    return np.where(labels == -1, 32, labels).astype(np.int32)


def artifacts():
    return {'old_plan': R.OUT/'PLAN.json', 'old_result': R.OUT/'result.json',
            'old_frame_ledger': R.OUT/'frame-ledger.json',
            'old_inference_audit': R.OUT/'inference-audit.json',
            'support_frame_ledger': S.OUT/'frame-ledger.json'}


def prepare():
    assert not (OUT/'PLAN.json').exists(), 'Preserve the frozen plan'
    old = P.read(R.OUT/'PLAN.json')
    audit = P.read(R.OUT/'inference-audit.json')
    assert audit['status'] == 'PASS' and audit['audited_frames'] == len(old['inputs']) == 343
    for name, digest in old['source_sha256'].items():
        assert P.sha(ROOT/name) == digest, name
    assert P.sha(S.OUT/'frame-ledger.json') == old['reference_ledger_sha256']
    prereg = (f'| 2026-10-02 | {RUN_ID} | PRE_RUN; same343 consumed Development; preserve original SAM 0..31 and replace uncovered -1 with fixed32; region x public coarse zone makes separate residual per zone; no new inference or parameter search; full FOV coverage, exact old KD per-zone patch budget, unchanged object supports; retain coarse/SAM/KD and add sam_residual, fixed purity90%/coverage50% | NOT_RUN | Descriptive partition diagnostic only; not assignment, metric range or alarm performance | `artifacts.local/work/cnh-rgb-region-residual-20261002/result.json` |')
    text = P.RUNS.read_text(encoding='utf8')
    assert RUN_ID not in text, 'Run row already exists; inspect before preparing again'
    paths = [Path(__file__), Path(R.__file__), Path(S.__file__), Path(A.__file__),
             Path(V.__file__), Path(G.__file__), Path(P.__file__)]
    plan = dict(run_id=RUN_ID, frozen_at=datetime.now(timezone.utc).isoformat(),
        role='Consumed Development; same 343 previously evaluated frames, not confirmation',
        inputs=old['inputs'], source_sha256={str(p.relative_to(ROOT)): P.sha(p) for p in paths},
        artifacts={k: dict(path=str(p), sha256=P.sha(p)) for k, p in artifacts().items()},
        rules=dict(observation='Only original RGB SAM labels; -1 to fixed32, all 0..31 unchanged',
            grouping='Existing R.score_partition region x zone; residual distinct in each zone',
            purity=.9, coverage=.5, workers=3, parameters_searched=False,
            baseline='All original ledger arms and object/query supports preserved verbatim',
            guards=['All in-FOV pixels covered', 'Exact old KD per-zone patch budget',
                    'All object contact and expanded query supports unchanged',
                    'Original SAM pure pixels and hits cannot decrease'],
            primary='Same instance-view contact/pass coverage summary and paired sam_residual versus coarse/SAM/KD',
            claim_ceiling='Partition opportunity diagnostic only; not deployable object association, range or alarm'),
        preregistration=prereg)
    OUT.mkdir(parents=True, exist_ok=True)
    P.RUNS.write_text(text.rstrip()+'\n'+prereg+'\n', encoding='utf8')
    (OUT/'prerun-row.txt').write_text(prereg+'\n', encoding='utf8')
    P.save(OUT/'PLAN.json', plan)
    print('Prepared residual-only fixed comparison')


def counts_per_zone(patches):
    return dict(Counter(p['zone'] for p in patches))


def old_pure_map(labels, zone, inst, patches):
    """Read old scoring ledger, without recomputing a baseline or supplying it to prediction."""
    dominant = np.full(32*64, -999, dtype=np.int64)
    eligible = np.zeros(32*64, dtype=bool)
    for p in patches:
        key = p['region']*64+p['zone']
        assert 0 <= key < len(dominant)
        if p['evaluator_dominant_instance'] is not None:
            dominant[key] = p['evaluator_dominant_instance']
            eligible[key] = p['instance_purity_lower'] >= .9
    included = (labels >= 0) & (zone >= 0)
    keys = labels[included].astype(np.int64)*64+zone[included]
    result = np.zeros(zone.shape, dtype=bool)
    result[included] = eligible[keys] & (inst[included] == dominant[keys])
    return result


def evaluate_frame(args):
    row, old, support, audit, old_plan = args
    path = R.OUT/'predictions'/(row['id']+'.npz')
    receipt_path = path.with_suffix('.json'); receipt = P.read(receipt_path)
    assert P.sha(path) == receipt['output_sha256'] == audit['output_sha256']
    assert P.sha(receipt_path) == audit['receipt_sha256']
    assert receipt['status'] == audit['status'] == 'PASS'
    assert receipt['rgb_sha256'] == row['rgb_sha256'] == P.sha(row['rgb_path'])
    identity = receipt['identity']
    assert set(identity) == {'id', 'rgb_path', 'rgb_sha256'}
    assert identity == dict(id=row['id'], rgb_path=str(Path(row['rgb_path']).resolve()), rgb_sha256=row['rgb_sha256'])
    binding = receipt['binding']
    infer_name = str(Path(R.__file__).with_name('cnh_rgb_region_infer.py').relative_to(ROOT))
    assert binding['adapter_sha256'] == old_plan['source_sha256'][infer_name]
    recovery = old_plan['model_recovery']
    assert binding['model_revision'] == recovery['revision']
    assert binding['model_files'] == {f['name']: f['sha256'] for f in recovery['files']}
    with np.load(path, allow_pickle=False) as data:
        labels = data['labels'].copy()
        assert np.array_equal(labels, data['labelmap'])
    assert labels.shape == (768, 1024)
    # Prediction finishes before any evaluator reference arrays are loaded.
    residual = residual_labels(labels)
    assert np.array_equal(residual[labels >= 0], labels[labels >= 0])
    for name in ('depth', 'semantic', 'instance'):
        assert P.sha(row[name+'_path']) == row[name+'_sha256']
    radial = S.hdf(row['depth_path'])
    inst = S.hdf(row['instance_path']).astype(np.int32)
    sem = S.hdf(row['semantic_path']).astype(np.int32)
    geo = A.whole_geometry(V.ray_geometry(radial.shape, row['camera_matrix']))
    z = radial/geo['radial_factor']; valid = np.isfinite(z) & (z > 0)
    x = np.abs(z*geo['fx']); y = z*geo['fy']; fov = geo['fov_mask']; zone = geo['zone_id']
    assert np.array_equal(fov, zone >= 0)
    contacts, expanded = [], []
    for q in geo['queries']:
        near = fov & valid & (z >= .6) & (z < 2.1) & (y >= q['y_low']) & (y <= q['y_high'])
        contacts.append(near & (x < .3)); expanded.append(near & (x <= .4))
    contact = np.logical_or.reduce(contacts); expansion = np.logical_or.reduce(expanded)
    for obj in support['objects']:
        target = inst == obj['instance_id']
        assert len(obj['query_support']) == len(contacts)
        for i, q in enumerate(obj['query_support']):
            assert int((target & contacts[i]).sum()) == q['contact_pixels']
            assert int((target & expanded[i]).sum()) == q['expanded_pixels']
    truth = A.visible_truth(z, geo)
    queries = [dict(name=q['name'], category=str(truth['category'][i]), abstain=bool(truth['abstain'][i]),
                    contact_pixels=int(contacts[i].sum()), expanded_pixels=int(expanded[i].sum()))
               for i, q in enumerate(geo['queries'])]
    assert queries == old['queries']
    result = deepcopy(old)
    assert ARM not in result['arms']
    pure, semantic_pure, covered, patches = R.score_partition(residual, zone, inst, sem, radial)
    assert np.array_equal(covered, fov)
    per_zone = counts_per_zone(patches)
    assert per_zone == counts_per_zone(old['arms']['matched_spatial']['patches'])
    for zid in np.unique(zone[fov]):
        assert per_zone[int(zid)] == len(np.unique(labels[zone == zid]))
    assert len(patches) == old['arms']['sam']['budget_patches_including_uncovered']
    old_pure = old_pure_map(labels, zone, inst, old['arms']['sam']['patches'])
    assert np.array_equal(pure[labels >= 0], old_pure[labels >= 0])
    assert not np.any(old_pure & ~pure)
    surfaces = []
    for klass in (1, 2, 22):
        target = contact & (sem == klass)
        surfaces.append(dict(nyu40=klass, pixels=int(target.sum()),
                             semantic_pure_pixels=int((target & semantic_pure).sum())))
    result['arms'][ARM] = dict(patches=patches, patch_count=len(patches), fov_pixels=int(fov.sum()),
        budget_patches_including_uncovered=len(patches), covered_pixels=int(covered.sum()),
        uncovered_pixels=0, semantic_surfaces=surfaces,
        queries=[dict(covered_contact_pixels=int((m & covered).sum()),
                      pure_contact_instance_pixels=int((m & pure).sum())) for m in contacts])
    support_ids = {o['instance_id']: o for o in support['objects']}
    for obj in result['objects']:
        qsupport = support_ids[obj['instance_id']]['query_support']
        is_contact = any(q['contact_pixels'] >= 16 for q in qsupport)
        assert obj['stratum'] == ('contact' if is_contact else 'pass_adjacent')
        target = (inst == obj['instance_id']) & (contact if is_contact else expansion)
        denominator = int(target.sum()); assert denominator == obj['support_pixels'] and denominator > 0
        assert int((target & old_pure).sum()) == obj['arms']['sam']['pure_pixels']
        good = int((target & pure).sum())
        obj['arms'][ARM] = dict(pure_pixels=good, covered_pixels=int((target & covered).sum()),
                               fraction=good/denominator, hit=bool(good >= .5*denominator))
        assert good >= obj['arms']['sam']['pure_pixels']
        assert obj['arms'][ARM]['hit'] or not obj['arms']['sam']['hit']
    result['residual_guards'] = dict(full_fov_coverage=True, exact_per_zone_old_kd_budget=True,
        all_query_support_unchanged=True, original_sam_pixels_unchanged=True,
        sam_purity_and_hits_nondecreasing=True)
    return result


def summarize(frames):
    output = R.summarize(frames)  # Original arms and paired summaries stay unchanged.
    groups = {}
    for stratum in ('contact', 'pass_adjacent'):
        for group in ('all', 'mapped', 'unsupported', 'UNKNOWN_CLASS'):
            objects = [o for f in frames for o in f['objects'] if o['stratum'] == stratum and
                (group == 'all' or (group == 'mapped' and o['mapped']) or
                 (group == 'unsupported' and not o['mapped'] and o['nyu40'] is not None) or
                 (group == 'UNKNOWN_CLASS' and o['nyu40'] is None))]
            groups[stratum+'/'+group] = dict(n=len(objects), hits=sum(o['arms'][ARM]['hit'] for o in objects),
                pixels=sum(o['support_pixels'] for o in objects),
                pure_pixels=sum(o['arms'][ARM]['pure_pixels'] for o in objects))
    output['arms'][ARM] = dict(groups=groups, patches=sum(f['arms'][ARM]['patch_count'] for f in frames),
        budget_patches_including_uncovered=sum(f['arms'][ARM]['budget_patches_including_uncovered'] for f in frames),
        fov_pixels=sum(f['arms'][ARM]['fov_pixels'] for f in frames), uncovered_pixels=0)
    output['residual_paired'] = {}
    for baseline in R.ARMS:
        by = {}
        for stratum in ('contact', 'pass_adjacent'):
            objects = [o for f in frames for o in f['objects'] if o['stratum'] == stratum]
            by[stratum] = dict(n=len(objects),
                residual_only=sum(o['arms'][ARM]['hit'] and not o['arms'][baseline]['hit'] for o in objects),
                baseline_only=sum(not o['arms'][ARM]['hit'] and o['arms'][baseline]['hit'] for o in objects),
                both=sum(o['arms'][ARM]['hit'] and o['arms'][baseline]['hit'] for o in objects),
                neither=sum(not o['arms'][ARM]['hit'] and not o['arms'][baseline]['hit'] for o in objects),
                pure_pixel_delta=sum(o['arms'][ARM]['pure_pixels']-o['arms'][baseline]['pure_pixels'] for o in objects))
        output['residual_paired'][baseline] = by
    return output


def evaluate():
    assert not (OUT/'result.json').exists() and not (OUT/'frame-ledger.json').exists()
    plan = P.read(OUT/'PLAN.json')
    for name, digest in plan['source_sha256'].items():
        assert P.sha(ROOT/name) == digest, name
    for item in plan['artifacts'].values():
        assert P.sha(item['path']) == item['sha256'], item['path']
    old_plan = P.read(R.OUT/'PLAN.json')
    old_frames = P.read(R.OUT/'frame-ledger.json'); old = {f['id']: f for f in old_frames}
    support = {f['id']: f for f in P.read(S.OUT/'frame-ledger.json')}
    audit = {f['id']: f for f in P.read(R.OUT/'inference-audit.json')['frames']}
    ids = [r['id'] for r in plan['inputs']]
    assert len(ids) == len(set(ids)) == 343 and set(ids) == set(old) == set(audit)
    args = [(r, old[r['id']], support[r['id']], audit[r['id']], old_plan) for r in plan['inputs']]
    with ThreadPoolExecutor(max_workers=3) as pool:
        frames = list(pool.map(evaluate_frame, args))
    for frame in frames:
        original = deepcopy(frame)
        del original['arms'][ARM]; del original['residual_guards']
        for obj in original['objects']:
            del obj['arms'][ARM]
        assert original == old[frame['id']], 'Original ledger content changed'
    groups = {split: summarize([f for f in frames if split == 'all' or f['split'] == split])
              for split in ('cal', 'eval', 'all')}
    original_result = P.read(R.OUT/'result.json')
    for split, summary in groups.items():
        unchanged = deepcopy(summary); del unchanged['arms'][ARM]; del unchanged['residual_paired']
        assert unchanged == original_result['groups'][split], 'Original summary changed'
    P.save(OUT/'frame-ledger.json', frames)
    P.save(OUT/'result.json', dict(run_id=RUN_ID, role=plan['role'], groups=groups,
        guards=dict(all_343_original_ledgers_preserved=True, all_343_original_supports_preserved=True,
                    full_fov_coverage=True, per_zone_old_kd_budget_equal=True, sam_nondecreasing=True)))
    print(json.dumps(groups))


def selftest():
    zone = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]], np.int32)
    labels = np.array([[0, 0, -1, -1, 1, 1, -1, -1], [0, 0, -1, -1, 1, 1, -1, -1]], np.int32)
    inst = np.array([[0, 0, 2, 2, 1, 1, 3, 3], [0, 0, 2, 2, 1, 1, 3, 3]], np.int32)
    sem = np.ones_like(inst); radial = np.ones(inst.shape)
    filled = residual_labels(labels)
    assert np.array_equal(filled[labels >= 0], labels[labels >= 0])
    old_pure, _, _, old_patches = R.score_partition(labels, zone, inst, sem, radial)
    pure, _, covered, patches = R.score_partition(filled, zone, inst, sem, radial)
    assert covered.all() and pure.all() and not np.any(old_pure & ~pure)
    assert np.array_equal(old_pure, old_pure_map(labels, zone, inst, old_patches))
    assert len([p for p in patches if p['region'] == 32]) == 2
    assert {p['evaluator_dominant_instance'] for p in patches if p['region'] == 32} == {2, 3}
    yy, xx = np.indices(zone.shape)
    kd = G.matched_geometry_partition(labels, zone, xx/10., yy/10.)
    _, _, _, kd_patches = R.score_partition(kd, zone, inst, sem, radial)
    assert counts_per_zone(patches) == counts_per_zone(kd_patches) == {0: 2, 1: 2}
    # Unknown reference support stays impure; assigning residual never supplies an identity.
    unknown = inst.copy(); unknown[labels < 0] = -1
    unknown_pure, _, full, _ = R.score_partition(filled, zone, unknown, sem, radial)
    assert full.all() and not unknown_pure[labels < 0].any()
    assert np.array_equal(residual_labels(np.zeros((2, 2), np.int32)), np.zeros((2, 2)))
    assert np.all(residual_labels(np.full((2, 2), -1, np.int32)) == 32)
    for bad in (np.array([[32]]), np.array([[-2]])):
        try:
            residual_labels(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid original label accepted')
    print('PASS residual preservation, separate per-zone remainder, full coverage, exact KD budget, unknown denominator')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'selftest', 'evaluate'))
    globals()[parser.parse_args().action]()
