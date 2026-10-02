"""Fixed observation-only whole-zone range anchors propagated along RGB regions.

Simulator returns only 64 object-blind q10 radial values. Prediction has exactly
three inputs: labels, public geometry, and those returns. Identity is evaluator-only.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import numpy as np

import cnh_rgb_region_partition as R

P, S, A, V, G, ROOT = R.P, R.S, R.A, R.V, R.G, R.ROOT
OUT = ROOT/'artifacts.local/work/cnh-rgb-region-range-20261002'
RUN_ID = 'CNH_RGB_REGION_RANGE_20261002'
ARMS = ('coarse_q10', 'coarse_constant_z', 'sam_anchor', 'matched_spatial_anchor')
THRESHOLDS = (('05', .05), ('10', .10))


def anchor_prediction(labels, geometry, returns):
    """Only labels/public rays/64 returns; unknown zones never create anchors."""
    labels = np.asarray(labels)
    zone = np.asarray(geometry['zone_id'])
    factor = np.asarray(geometry['radial_factor'], dtype=float)
    values = np.asarray(returns, dtype=float).reshape(-1)
    if labels.shape != zone.shape or labels.shape != factor.shape or values.shape != (64,):
        raise ValueError('Aligned labels/public geometry and exactly64 radial returns required')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < -1):
        raise ValueError('Integer labels >=-1 required; -1 never anchors')
    fov = zone >= 0
    if not np.array_equal(fov, geometry['fov_mask']):
        raise ValueError('FOV and zone membership disagree')
    if np.any(zone > 63) or not np.all(np.isfinite(factor[fov]) & (factor[fov] > 0)):
        raise ValueError('Invalid public rays')
    # Uses every public ray, never reference validity or foreground selection.
    coefficient = V.coarse_returns(factor, geometry, .1)['zone_return_radial'].reshape(-1)
    valid = np.isfinite(values) & (values > 0)
    values = np.where(valid, values, np.nan)
    constant_z = values/coefficient
    broadcast = np.full(zone.shape, np.nan)
    broadcast[fov] = values[zone[fov]]/factor[fov]
    plane = np.full(zone.shape, np.nan); plane[fov] = constant_z[zone[fov]]
    by_region = {}
    for zid in np.unique(zone[fov]):
        observed = np.unique(labels[zone == zid])
        if len(observed) == 1 and observed[0] >= 0 and valid[zid] and np.isfinite(constant_z[zid]):
            by_region.setdefault(int(observed[0]), []).append(dict(zone=int(zid), z=float(constant_z[zid])))
    predicted = broadcast.copy()
    anchor_map = np.full(zone.shape, -1, dtype=np.int32)
    regions = []
    for region, anchors in sorted(by_region.items()):
        zvalue = float(np.median([a['z'] for a in anchors]))
        take = fov & (labels == region)
        predicted[take] = zvalue; anchor_map[take] = region
        regions.append(dict(region=region, anchors=anchors, predicted_z=zvalue, propagated_pixels=int(take.sum())))
    return dict(predicted_z=predicted, coarse_q10=broadcast, coarse_constant_z=plane,
                anchor_region_map=anchor_map, regions=regions, anchor_zone_count=sum(len(r['anchors']) for r in regions),
                anchored_region_count=len(regions), propagated_pixels=int((anchor_map >= 0).sum()),
                zone_coefficient=coefficient, zone_constant_z=constant_z)


def predict(labels, geometry, returns):
    """The complete four-arm predictor; no evaluator parameters or globals."""
    sam = anchor_prediction(labels, geometry, returns)
    spatial = G.matched_geometry_partition(labels, geometry['zone_id'], geometry['fx'], geometry['fy'])
    kd = anchor_prediction(spatial, geometry, returns)
    return dict(maps=dict(coarse_q10=sam['coarse_q10'], coarse_constant_z=sam['coarse_constant_z'],
                          sam_anchor=sam['predicted_z'], matched_spatial_anchor=kd['predicted_z']),
                anchor_maps=dict(sam_anchor=sam['anchor_region_map'], matched_spatial_anchor=kd['anchor_region_map']),
                diagnostics={name: {k: item[k] for k in ('regions', 'anchor_zone_count', 'anchored_region_count',
                                'propagated_pixels', 'zone_coefficient', 'zone_constant_z')}
                             for name, item in (('sam_anchor', sam), ('matched_spatial_anchor', kd))})


def simulator_returns(depth_path, geometry):
    radial = S.hdf(depth_path)
    return V.coarse_returns(radial, geometry, .1)['zone_return_radial'].reshape(64).copy()


def artifacts():
    return {'old_plan': R.OUT/'PLAN.json', 'old_ledger': R.OUT/'frame-ledger.json',
            'old_result': R.OUT/'result.json', 'inference_audit': R.OUT/'inference-audit.json',
            'support_ledger': S.OUT/'frame-ledger.json'}


def prepare():
    assert not (OUT/'PLAN.json').exists(), 'Preserve frozen experiment'
    old = P.read(R.OUT/'PLAN.json'); audit = P.read(R.OUT/'inference-audit.json')
    assert audit['status'] == 'PASS' and audit['audited_frames'] == len(old['inputs']) == 343
    for name, digest in old['source_sha256'].items():
        assert P.sha(ROOT/name) == digest
    assert P.sha(S.OUT/'frame-ledger.json') == old['reference_ledger_sha256']
    prereg = (f'| 2026-10-02 | {RUN_ID} | PRE_RUN; same343 consumed Development, fixed135 eval contact views; 64 object-blind radial r^-2 q10; anchor iff complete public zone has one observed label>=0 and finite return; public-factor q10 normalizes constant Z, median source-anchor Z per region, otherwise original q10 spherical broadcast; coarse_q10/coarse_constant_z/SAM/KD-anchor; fixed absZ<=.05/.10 and >=50% ALL support, missing fails; all-source attribution evaluator-only | NOT_RUN | Source evidence not proof of prediction error or assignment; preserve every denominator, UNKNOWN and old purehit | `artifacts.local/work/cnh-rgb-region-range-20261002/result.json` |')
    text = P.RUNS.read_text(encoding='utf8'); assert RUN_ID not in text
    sources = [Path(__file__), Path(R.__file__), Path(S.__file__), Path(A.__file__), Path(V.__file__), Path(G.__file__), Path(P.__file__)]
    plan = dict(run_id=RUN_ID, frozen_at=datetime.now(timezone.utc).isoformat(), inputs=old['inputs'],
        role='Same consumed Development; no new confirmation, phone or alarm claim',
        source_sha256={str(p.relative_to(ROOT)): P.sha(p) for p in sources},
        artifacts={k: dict(path=str(p), sha256=P.sha(p)) for k, p in artifacts().items()},
        rules=dict(arms=list(ARMS), quantile=.1, weight_power=2, anchor='All public zone pixels same observed region>=0 and finite positive sensor return',
            normalization='V.coarse_returns(public radial_factor, geometry, .1); no reference-validity mask',
            propagation='Median of all eligible anchor Z values for region, entire region FOV; otherwise q10 spherical broadcast',
            input_contract='Prediction only labels, geometry,64returns; simulator cannot return reference maps to predictor',
            thresholds_m=[.05,.10], hit='>=50% all original support pixels within threshold; missing prediction pixels fail',
            source_evidence='All radial==q10 pixels per anchor zone, all tied IDs retained; known iff exactly one nonnegative ID, else UNKNOWN',
            source_groups=['source_consistent','has_other_source_evidence','unknown'],
            source_consistency='All contributing source anchors known and same target; known-other evidence has priority over unknown',
            workers=3, fixed_eval_contact_views=135),
        limits=['GT identity is evaluated after prediction and never changes anchors or predictions.',
            'KD labels are unique across zones: no cross-zone identity propagation.',
            'KD K1 can arise from an all-uncovered SAM zone; SAM -1 cannot anchor, so anchor sets differ. This compares total methods, not a pure cross-zone propagation ablation.',
            'SAM versus coarse_constant_z also changes fallback scope; differences cannot all be attributed to association.',
            'Public coefficient uses every zone ray; missing reference samples may bias sensor quantiles and are not compensated.',
            'has_other_source_evidence does not prove prediction error: median may ignore/cancel a source.'], preregistration=prereg)
    OUT.mkdir(parents=True, exist_ok=True)
    P.RUNS.write_text(text.rstrip()+'\n'+prereg+'\n', encoding='utf8')
    (OUT/'prerun-row.txt').write_text(prereg+'\n', encoding='utf8')
    P.save(OUT/'PLAN.json', plan)
    print('Prepared fixed range-anchor comparison')


def error_stats(predicted, reference, target):
    support = int(target.sum())
    valid = target & np.isfinite(predicted) & (predicted > 0)
    errors = np.abs(predicted[valid]-reference[valid])
    n = len(errors)
    out = dict(support_pixels=support, finite_pixels=n, missing_pixels=support-n,
               abs_z_error_sum_m=float(errors.sum()), abs_z_mae_m=float(errors.mean()) if n else None,
               abs_z_median_m=float(np.median(errors)) if n else None,
               abs_z_p90_m=float(np.quantile(errors,.9)) if n else None)
    for tag, threshold in THRESHOLDS:
        count = int((errors <= threshold).sum())
        out['within_'+tag+'_pixels'] = count
        out['hit_'+tag] = bool(support > 0 and count >= .5*support)
    return out


def provider_evidence(radial, inst, zone, returns, regions):
    """Evaluation only: exact q10 ties, including unknown and mixed identities."""
    answer = []
    for region in regions:
        sources = []
        for anchor in region['anchors']:
            zid = anchor['zone']; tied = (zone == zid) & (radial == returns[zid])
            ids = np.unique(inst[tied]).astype(int).tolist()
            assert ids, 'Empirical q10 must equal at least one source radial pixel'
            known = ids[0] if len(ids) == 1 and ids[0] >= 0 else None
            sources.append(dict(zone=zid, anchor_z=anchor['z'], tied_source_pixels=int(tied.sum()),
                                source_instance_ids=ids, known_instance_id=known,
                                disposition='known' if known is not None else 'UNKNOWN'))
        answer.append(dict(region=region['region'], predicted_z=region['predicted_z'], sources=sources))
    return answer


def source_class(sources, instance_id):
    known = [s['known_instance_id'] for s in sources]
    if any(k is not None and k != instance_id for k in known):
        return 'has_other_source_evidence'
    if known and all(k == instance_id for k in known):
        return 'source_consistent'
    return 'unknown'


def evaluate_frame(args):
    row, old, support, audited, old_plan = args
    path = R.OUT/'predictions'/(row['id']+'.npz'); receiptpath = path.with_suffix('.json')
    receipt = P.read(receiptpath)
    assert P.sha(path) == receipt['output_sha256'] == audited['output_sha256']
    assert P.sha(receiptpath) == audited['receipt_sha256']
    assert receipt['status'] == audited['status'] == 'PASS'
    assert receipt['rgb_sha256'] == row['rgb_sha256'] == P.sha(row['rgb_path'])
    assert receipt['identity'] == dict(id=row['id'], rgb_path=str(Path(row['rgb_path']).resolve()), rgb_sha256=row['rgb_sha256'])
    infer_name = str(Path(R.__file__).with_name('cnh_rgb_region_infer.py').relative_to(ROOT))
    assert receipt['binding']['adapter_sha256'] == old_plan['source_sha256'][infer_name]
    assert receipt['binding']['model_files'] == {f['name']: f['sha256'] for f in old_plan['model_recovery']['files']}
    assert receipt['binding']['model_revision'] == old_plan['model_recovery']['revision']
    with np.load(path, allow_pickle=False) as data:
        labels = data['labels'].copy()
    assert labels.shape == (768,1024) and np.all((labels >= -1) & (labels <= 31))
    geo = A.whole_geometry(V.ray_geometry(labels.shape, row['camera_matrix']))
    assert P.sha(row['depth_path']) == row['depth_sha256']
    returns = simulator_returns(row['depth_path'], geo)
    prediction = predict(labels, geo, returns)  # No reference arrays exist in this scope yet.
    for name in ('depth','instance','semantic'):
        assert P.sha(row[name+'_path']) == row[name+'_sha256']
    radial = S.hdf(row['depth_path']); inst = S.hdf(row['instance_path']).astype(np.int32)
    z = radial/geo['radial_factor']; valid = np.isfinite(z) & (z > 0)
    x = np.abs(z*geo['fx']); y = z*geo['fy']; fov = geo['fov_mask']; zone = geo['zone_id']
    contacts, expanded = [], []
    for q in geo['queries']:
        take = fov & valid & (z >= .6) & (z < 2.1) & (y >= q['y_low']) & (y <= q['y_high'])
        contacts.append(take & (x < .3)); expanded.append(take & (x <= .4))
    contact = np.logical_or.reduce(contacts); expansion = np.logical_or.reduce(expanded)
    for obj in support['objects']:
        target = inst == obj['instance_id']
        for i, q in enumerate(obj['query_support']):
            assert int((target & contacts[i]).sum()) == q['contact_pixels']
            assert int((target & expanded[i]).sum()) == q['expanded_pixels']
    truth = A.visible_truth(z, geo)
    assert old['queries'] == [dict(name=q['name'], category=str(truth['category'][i]), abstain=bool(truth['abstain'][i]),
        contact_pixels=int(contacts[i].sum()), expanded_pixels=int(expanded[i].sum())) for i,q in enumerate(geo['queries'])]
    result = deepcopy(old)
    result['range_prediction'] = dict(sensor_q10_radial=returns, finite_return_zones=int(np.isfinite(returns).sum()),
                                      diagnostics=prediction['diagnostics'])
    evidence = provider_evidence(radial, inst, zone, returns, prediction['diagnostics']['sam_anchor']['regions'])
    result['anchor_provider_evidence'] = evidence
    for obj in result['objects']:
        target = (inst == obj['instance_id']) & (contact if obj['stratum'] == 'contact' else expansion)
        assert int(target.sum()) == obj['support_pixels']
        obj['range_arms'] = {arm: error_stats(prediction['maps'][arm], z, target) for arm in ARMS}
        for arm in ('sam_anchor','matched_spatial_anchor'):
            amap = prediction['anchor_maps'][arm]
            used = target & (amap >= 0)
            obj['range_arms'][arm].update(anchor_covered_pixels=int(used.sum()),
                anchor_covered_regions=len(np.unique(amap[used])))
        # Attribution denominator is contact support, also retained for pass-adjacent objects.
        obj_contact = (inst == obj['instance_id']) & contact
        amap = prediction['anchor_maps']['sam_anchor']; groups = {k: np.zeros(zone.shape,bool)
            for k in ('source_consistent','has_other_source_evidence','unknown')}
        details = []
        for entry in evidence:
            take = obj_contact & (amap == entry['region'])
            if not take.any():
                continue
            category = source_class(entry['sources'], obj['instance_id'])
            groups[category] |= take
            details.append(dict(region=entry['region'], category=category, sources=entry['sources'],
                                error=error_stats(prediction['maps']['sam_anchor'], z, take)))
        covered = obj_contact & (amap >= 0)
        assert sum(int(m.sum()) for m in groups.values()) == int(covered.sum())
        obj['sam_source_evidence'] = dict(contact_pixels=int(obj_contact.sum()),
            anchor_covered_contact_pixels=int(covered.sum()), regions=details,
            groups={name: error_stats(prediction['maps']['sam_anchor'], z, mask) for name,mask in groups.items()})
    return result


def summarize(frames):
    out = dict(frames=len(frames), queries=dict(Counter(q['category'] for f in frames for q in f['queries'])), strata={})
    for stratum in ('contact','pass_adjacent'):
        objects = [o for f in frames for o in f['objects'] if o['stratum'] == stratum]
        item = dict(objects=len(objects), support_pixels=sum(o['support_pixels'] for o in objects), arms={}, paired={})
        for arm in ARMS:
            metrics = [o['range_arms'][arm] for o in objects]
            finite = sum(m['finite_pixels'] for m in metrics)
            total = sum(m['abs_z_error_sum_m'] for m in metrics)
            a = dict(finite_pixels=finite, missing_pixels=sum(m['missing_pixels'] for m in metrics),
                     abs_z_error_sum_m=total, pooled_abs_z_mae_m=total/finite if finite else None,
                     objects_with_finite_prediction=sum(m['finite_pixels'] > 0 for m in metrics))
            for key in ('abs_z_mae_m','abs_z_median_m','abs_z_p90_m'):
                values = [m[key] for m in metrics if m[key] is not None]
                a['macro_mean_object_'+key] = float(np.mean(values)) if values else None
            for tag, _ in THRESHOLDS:
                a['within_'+tag+'_pixels'] = sum(m['within_'+tag+'_pixels'] for m in metrics)
                a['hits_'+tag] = sum(m['hit_'+tag] for m in metrics)
            if arm in ('sam_anchor','matched_spatial_anchor'):
                a['anchor_covered_pixels'] = sum(m['anchor_covered_pixels'] for m in metrics)
                a['object_region_intersections'] = sum(m['anchor_covered_regions'] for m in metrics)
            item['arms'][arm] = a
        for baseline in ('coarse_q10','coarse_constant_z','matched_spatial_anchor'):
            pair = dict(n=len(objects))
            for tag, _ in THRESHOLDS:
                pair[tag] = dict(sam_only=sum(o['range_arms']['sam_anchor']['hit_'+tag] and not o['range_arms'][baseline]['hit_'+tag] for o in objects),
                    baseline_only=sum(not o['range_arms']['sam_anchor']['hit_'+tag] and o['range_arms'][baseline]['hit_'+tag] for o in objects),
                    both=sum(o['range_arms']['sam_anchor']['hit_'+tag] and o['range_arms'][baseline]['hit_'+tag] for o in objects),
                    neither=sum(not o['range_arms']['sam_anchor']['hit_'+tag] and not o['range_arms'][baseline]['hit_'+tag] for o in objects))
            delta = [o['range_arms']['sam_anchor']['abs_z_mae_m']-o['range_arms'][baseline]['abs_z_mae_m'] for o in objects
                     if o['range_arms']['sam_anchor']['abs_z_mae_m'] is not None and o['range_arms'][baseline]['abs_z_mae_m'] is not None]
            pair.update(mae_pairs_with_both_finite=len(delta), mean_object_mae_delta_m=float(np.mean(delta)) if delta else None)
            item['paired'][baseline] = pair
        out['strata'][stratum] = item
    out['source_evidence_by_stratum'] = {}
    for stratum in ('contact','pass_adjacent'):
        objects = [o for f in frames for o in f['objects'] if o['stratum'] == stratum]
        out['source_evidence_by_stratum'][stratum] = {
            name: {output_key: sum(o['sam_source_evidence']['groups'][name][metric_key] for o in objects)
                   for output_key, metric_key in (
                       ('contact_pixels','support_pixels'), ('finite_pixels','finite_pixels'),
                       ('within_05_pixels','within_05_pixels'), ('within_10_pixels','within_10_pixels'),
                       ('abs_z_error_sum_m','abs_z_error_sum_m'))}
            for name in ('source_consistent','has_other_source_evidence','unknown')}
    return out


def evaluate():
    assert not (OUT/'result.json').exists() and not (OUT/'frame-ledger.json').exists()
    plan = P.read(OUT/'PLAN.json')
    for path,digest in plan['source_sha256'].items():
        assert P.sha(ROOT/path) == digest
    for a in plan['artifacts'].values():
        assert P.sha(a['path']) == a['sha256']
    old_plan = P.read(R.OUT/'PLAN.json')
    old = {f['id']:f for f in P.read(R.OUT/'frame-ledger.json')}
    support = {f['id']:f for f in P.read(S.OUT/'frame-ledger.json')}
    audit = {f['id']:f for f in P.read(R.OUT/'inference-audit.json')['frames']}
    ids = [r['id'] for r in plan['inputs']]
    assert len(ids) == len(set(ids)) == 343 and set(ids) == set(old) == set(audit)
    args = [(r, old[r['id']], support[r['id']], audit[r['id']], old_plan) for r in plan['inputs']]
    with ThreadPoolExecutor(max_workers=3) as pool:
        frames = list(pool.map(evaluate_frame, args))
    for frame in frames:
        original = deepcopy(frame); del original['range_prediction']; del original['anchor_provider_evidence']
        for obj in original['objects']:
            del obj['range_arms']; del obj['sam_source_evidence']
        assert original == old[frame['id']]
    groups = {split:summarize([f for f in frames if split == 'all' or f['split'] == split]) for split in ('cal','eval','all')}
    assert groups['eval']['strata']['contact']['objects'] == 135
    families = {split:{family:summarize([f for f in frames if f['family'] == family and (split == 'all' or f['split'] == split)])
                       for family in sorted({f['family'] for f in frames if split == 'all' or f['split'] == split})}
                for split in ('cal','eval','all')}
    P.save(OUT/'frame-ledger.json', frames)
    P.save(OUT/'result.json', dict(run_id=RUN_ID, role=plan['role'], groups=groups, families=families,
                                 original_objects_queries_and_purehit_preserved=True, limits=plan['limits']))
    print(json.dumps(P.plain(groups)))


def selftest():
    geometry = V.ray_geometry((32,32), np.diag([.4,.4,-1.]))
    factor = geometry['radial_factor']; zone = geometry['zone_id']; fov = geometry['fov_mask']
    labels = np.zeros(zone.shape, np.int32)
    radial = 1.7*factor
    returns = V.coarse_returns(radial,geometry,.1)['zone_return_radial'].reshape(64)
    prediction = predict(labels,geometry,returns)
    np.testing.assert_allclose(prediction['maps']['coarse_constant_z'][fov],1.7,atol=1e-12)
    np.testing.assert_allclose(prediction['maps']['sam_anchor'][fov],1.7,atol=1e-12)
    assert prediction['diagnostics']['sam_anchor']['anchor_zone_count'] == 64
    none = predict(np.full(zone.shape,-1,np.int32),geometry,returns)
    np.testing.assert_allclose(none['maps']['sam_anchor'],none['maps']['coarse_q10'],equal_nan=True)
    assert none['diagnostics']['sam_anchor']['anchor_zone_count'] == 0
    missing = predict(labels,geometry,np.full(64,np.nan))
    assert np.isnan(missing['maps']['sam_anchor']).all() and missing['diagnostics']['sam_anchor']['anchor_zone_count'] == 0
    coefficient = V.coarse_returns(factor,geometry,.1)['zone_return_radial'].reshape(64)
    few = np.full(64,np.nan); few[[0,1,2]] = coefficient[[0,1,2]]*np.array([1.,2.,9.])
    mixed = anchor_prediction(labels,geometry,few)
    np.testing.assert_allclose(mixed['predicted_z'][fov],2.)
    assert [a['zone'] for a in mixed['regions'][0]['anchors']] == [0,1,2]
    split_labels = labels.copy(); split_labels.flat[0] = 1
    split = anchor_prediction(split_labels,geometry,returns)
    assert split['anchor_zone_count'] == 63
    assert list(inspect.signature(predict).parameters) == ['labels','geometry','returns']
    assert list(inspect.signature(anchor_prediction).parameters) == ['labels','geometry','returns']
    # Ties cannot silently choose an object: mixed/unknown IDs make source UNKNOWN.
    tied_radial = np.ones(zone.shape); identities = np.zeros(zone.shape,np.int32)
    indices = np.flatnonzero(zone == 0); identities.flat[indices[0]] = 1
    ev = provider_evidence(tied_radial,identities,zone,np.ones(64),[dict(region=0,predicted_z=1.,anchors=[dict(zone=0,z=1.)])])
    assert ev[0]['sources'][0]['source_instance_ids'] == [0,1] and ev[0]['sources'][0]['known_instance_id'] is None
    assert source_class([dict(known_instance_id=2),dict(known_instance_id=2)],2) == 'source_consistent'
    assert source_class([dict(known_instance_id=2),dict(known_instance_id=None)],2) == 'unknown'
    assert source_class([dict(known_instance_id=3),dict(known_instance_id=None)],2) == 'has_other_source_evidence'
    e = error_stats(np.array([[1.,np.nan,np.nan]]), np.ones((1,3)), np.ones((1,3),bool))
    assert e['finite_pixels'] == 1 and not e['hit_05'] and e['support_pixels'] == 3
    print('PASS constant-Z plane, no-anchor fallback, missing returns, fixed median, exact ties, missing-denominator failure and three-input predictor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','evaluate','selftest'))
    globals()[parser.parse_args().action]()
