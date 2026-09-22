"""Frozen-readout transfer measurements; no training or cutoff selection."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ba_camera_corridor_metrics import evaluate_rows

ARMS = ('A_current', 'raw', 'local', 'raw_standalone', 'local_standalone')
LEARNED = ('raw', 'local')
CENTRE = [1, 4]
AXES = ('appearance', 'base_group_id', 'type_id', 'layer', 'layout_relation')
GATES = dict(bootstrap_seed=202609225, bootstrap_resamples=1000,
    minimum_mean_normalized_rgb_L1=1/255,
    component_recall_delta=.05, component_FP_delta=1, component_segment_delta=1,
    component_improved_groups=4, layer_max_recall_loss=.05,
    appearance_max_recall_loss=.05, appearance_FP_delta=1, appearance_segment_delta=1,
    minimum_base_rescues=8, rescue_retention=.75, geometry_ordering_fraction=.75,
    strict_recall_delta=.10, strict_FP_delta=2, strict_segment_delta=1)


def make_rows(identities, baseline, probabilities, labels, cutoffs):
    """Join already sealed predictions to evaluator labels, preserving UNKNOWN."""
    n = len(identities)
    if not np.array_equal(labels['indices'], np.arange(n)):
        raise ValueError('Evaluation label indices must exactly match all predictions')
    if labels['classes'].shape != (n, 6) or labels['valid'].shape != (n, 6):
        raise ValueError('Expected six query labels per frame')
    if not np.isin(labels['classes'], np.arange(7)).all():
        raise ValueError('Unexpected occupied/empty class')
    truth = (labels['classes'][:, CENTRE] < 6).any(1)
    valid = labels['valid'][:, CENTRE].all(1)
    rows = []
    for i, (meta, a) in enumerate(zip(identities, baseline)):
        scores = {arm: float(probabilities[arm][i, CENTRE].max()) for arm in LEARNED}
        flags = {'A_current': bool(a['alert'])}
        for arm in LEARNED:
            flags[arm+'_standalone'] = scores[arm] >= cutoffs[arm]
            flags[arm] = flags['A_current'] or flags[arm+'_standalone']
        rows.append(dict(**{key: meta[key] for key in ('id', 'clip_id', 'frame_in_clip',
            'time_s', 'appearance_pair_id', *AXES)},
            index=i, truth=bool(truth[i]) if valid[i] else None,
            boundary=meta['layout_relation'] == 'BOUNDARY',
            frame_scores=scores,
            query_probabilities={arm: probabilities[arm][i].tolist() for arm in LEARNED},
            query_truth=(labels['classes'][i] < 6).tolist(),
            query_valid=labels['valid'][i].astype(bool).tolist(),
            predictions={arm: dict(alert=bool(flag), unknown=bool(a['unknown']),
                                   ambiguous=bool(flag and a['unknown'])) for arm, flag in flags.items()}))
    return rows


def query_counts(rows, arm, cutoff):
    y = np.array([r['query_truth'] for r in rows], bool)
    v = np.array([r['query_valid'] for r in rows], bool)
    p = np.array([r['query_probabilities'][arm] for r in rows]) >= cutoff
    def one(mask):
        keep = v & mask
        tp, fp, fn, tn = [int(x.sum()) for x in (p & y & keep, p & ~y & keep,
                                               ~p & y & keep, ~p & ~y & keep)]
        return dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=int(keep.sum()),
            recall=tp/(tp+fn) if tp+fn else None,
            precision=tp/(tp+fp) if tp+fp else None,
            FPR=fp/(fp+tn) if fp+tn else None)
    return dict(all=one(True), by_query={str(q): one(np.arange(6)[None] == q) for q in range(6)},
                negative_query_is_not_observed_free_space=True)


def comparison(rows, metrics, arm, ref):
    a, b = (metrics['arms'][name] for name in (arm, ref))
    ac, bc = a['frames']['all_known'], b['frames']['all_known']
    before = {(e['clip_id'], e['start_frame']): e for e in b['events']}
    changes = []
    for event in a['events']:
        old = before[(event['clip_id'], event['start_frame'])]
        t0, t1 = old['first_alert_time_s'], event['first_alert_time_s']
        changes.append(dict(clip_id=event['clip_id'], start_frame=event['start_frame'],
            reference_first_s=t0, candidate_first_s=t1,
            delay_s=None if t0 is None or t1 is None else t1-t0,
            lost=old['detected'] and not event['detected'],
            gained=event['detected'] and not old['detected']))
    groups = sorted({r['base_group_id'] for r in rows})
    group_deltas = {}
    for group in groups:
        subset = [r for r in rows if r['base_group_id'] == group and r['truth'] is True]
        group_deltas[group] = sum(int(r['predictions'][arm]['alert'])-int(r['predictions'][ref]['alert']) for r in subset)
    return dict(TP_delta=ac['TP']-bc['TP'], FP_delta=ac['FP']-bc['FP'],
        recall_delta=None if ac['recall'] is None else ac['recall']-bc['recall'],
        false_segment_delta=a['false_alert_segment_count']-b['false_alert_segment_count'],
        lost_true_frames=[r['id'] for r in rows if r['truth'] is True and r['predictions'][ref]['alert'] and not r['predictions'][arm]['alert']],
        gained_true_frames=[r['id'] for r in rows if r['truth'] is True and not r['predictions'][ref]['alert'] and r['predictions'][arm]['alert']],
        added_false_frames=[r['id'] for r in rows if r['truth'] is False and not r['predictions'][ref]['alert'] and r['predictions'][arm]['alert']],
        removed_false_frames=[r['id'] for r in rows if r['truth'] is False and r['predictions'][ref]['alert'] and not r['predictions'][arm]['alert']],
        group_TP_delta=group_deltas, improved_groups=sum(v > 0 for v in group_deltas.values()),
        event_differences=changes)


def appearance_pairs(rows):
    grouped = defaultdict(dict)
    issues = []
    for row in rows:
        if row['appearance'] in grouped[row['appearance_pair_id']]:
            issues.append('Duplicate appearance member: '+row['id'])
        grouped[row['appearance_pair_id']][row['appearance']] = row
    pairs = []
    for key, members in sorted(grouped.items()):
        if set(members) != {'base', 'changed'}:
            issues.append('Missing appearance member: '+key)
            continue
        b, c = members['base'], members['changed']
        for field in ('base_group_id', 'type_id', 'layer', 'layout_relation', 'frame_in_clip',
                      'time_s', 'truth', 'query_truth', 'query_valid'):
            if b[field] != c[field]:
                issues.append('Appearance pair mismatch: '+key+'/'+field)
        pairs.append(dict(pair_id=key, base_id=b['id'], changed_id=c['id'],
            base_group_id=b['base_group_id'], layout_relation=b['layout_relation'],
            truth=b['truth'], arms={arm: dict(base_alert=b['predictions'][arm]['alert'],
            changed_alert=c['predictions'][arm]['alert']) for arm in ARMS},
            score_delta={arm: c['frame_scores'][arm]-b['frame_scores'][arm] for arm in LEARNED},
            query_score_delta={arm: (np.array(c['query_probabilities'][arm])-b['query_probabilities'][arm]).tolist() for arm in LEARNED}))
    summary = {}
    for arm in ARMS:
        summary[arm] = dict(pairs=len(pairs),
            alert_flips=sum(p['arms'][arm]['base_alert'] != p['arms'][arm]['changed_alert'] for p in pairs),
            positive_lost=sum(p['truth'] is True and p['arms'][arm]['base_alert'] and not p['arms'][arm]['changed_alert'] for p in pairs),
            positive_gained=sum(p['truth'] is True and not p['arms'][arm]['base_alert'] and p['arms'][arm]['changed_alert'] for p in pairs),
            negative_added=sum(p['truth'] is False and not p['arms'][arm]['base_alert'] and p['arms'][arm]['changed_alert'] for p in pairs),
            negative_removed=sum(p['truth'] is False and p['arms'][arm]['base_alert'] and not p['arms'][arm]['changed_alert'] for p in pairs))
        if arm in LEARNED:
            d = np.array([p['score_delta'][arm] for p in pairs])
            q = np.array([p['query_score_delta'][arm] for p in pairs])
            summary[arm].update(mean_signed_score_change=float(d.mean()) if len(d) else None,
                mean_absolute_score_change=float(np.abs(d).mean()) if len(d) else None,
                max_absolute_score_change=float(np.abs(d).max()) if len(d) else None,
                mean_absolute_query_score_change=float(np.abs(q).mean()) if len(q) else None)
    return dict(summary=summary, pairs=pairs, issues=issues)


def geometry_ordering(rows, cutoffs):
    grouped = defaultdict(dict)
    issues = []
    for row in rows:
        key = (row['base_group_id'], row['appearance'], row['frame_in_clip'])
        if row['layout_relation'] in grouped[key]:
            issues.append('Duplicate relation: '+str(key))
        grouped[key][row['layout_relation']] = row
    pairs = []
    for key, members in sorted(grouped.items()):
        if set(members) != {'INSIDE', 'BOUNDARY', 'OUTSIDE'}:
            issues.append('Missing geometry relation: '+str(key))
            continue
        b, o = members['BOUNDARY'], members['OUTSIDE']
        q = 1 if b['layer'] == 'BODY' else 4
        eligible = b['truth'] is True and o['truth'] is False
        arms = {}
        for arm in LEARNED:
            bs, os = b['query_probabilities'][arm][q], o['query_probabilities'][arm][q]
            arms[arm] = dict(boundary_score=bs, outside_score=os, difference=bs-os,
                ordered=bs > os, tied=bs == os, reversed=bs < os,
                boundary_frame_score=b['frame_scores'][arm], outside_frame_score=o['frame_scores'][arm],
                frame_ordered=b['frame_scores'][arm] > o['frame_scores'][arm],
                query_boundary_hit=bs >= cutoffs[arm], query_outside_false=os >= cutoffs[arm],
                actual_boundary_alert=b['predictions'][arm]['alert'],
                actual_outside_alert=o['predictions'][arm]['alert'],
                actual_separates=b['predictions'][arm]['alert'] and not o['predictions'][arm]['alert'])
        pairs.append(dict(base_group_id=key[0], appearance=key[1], frame_in_clip=key[2],
            boundary_id=b['id'], outside_id=o['id'], target_query=q, eligible=bool(eligible), arms=arms))
    def summarize(subset):
        valid = [p for p in subset if p['eligible']]
        return {arm: dict(total_pairs=len(subset), eligible_pairs=len(valid),
            ordered=sum(p['arms'][arm]['ordered'] for p in valid),
            frame_ordered=sum(p['arms'][arm]['frame_ordered'] for p in valid),
            frame_ordering_fraction=sum(p['arms'][arm]['frame_ordered'] for p in valid)/len(valid) if valid else None,
            tied=sum(p['arms'][arm]['tied'] for p in valid),
            reversed=sum(p['arms'][arm]['reversed'] for p in valid),
            mean_score_margin=float(np.mean([p['arms'][arm]['difference'] for p in valid])) if valid else None,
            query_boundary_hits=sum(p['arms'][arm]['query_boundary_hit'] for p in valid),
            query_outside_false=sum(p['arms'][arm]['query_outside_false'] for p in valid),
            actual_separates=sum(p['arms'][arm]['actual_separates'] for p in valid),
            actual_outside_alerts=sum(p['arms'][arm]['actual_outside_alert'] for p in valid)) for arm in LEARNED}
    return dict(summary=summarize(pairs), by_appearance={a: summarize([p for p in pairs if p['appearance'] == a])
        for a in ('base', 'changed')}, pairs=pairs, issues=issues,
        meaning='Target-query Boundary-over-Outside ranking and fixed-cutoff actual A-union alerts are distinct')


def source_admissibility(rows, labels, public_pairs, paired):
    issues = list(public_pairs['issues']) + list(paired['issues'])
    if len(public_pairs['pairs']) != 288 or any(not p['tof_exactly_equal'] or not p['rgb_changed'] for p in public_pairs['pairs']):
        issues.append('Need 288 exact-ToF, changed-RGB public appearance pairs')
    if public_pairs['mean_normalized_rgb_L1'] is None or public_pairs['mean_normalized_rgb_L1'] < GATES['minimum_mean_normalized_rgb_L1']:
        issues.append('Mean normalized RGB L1 is below 1/255')
    if len(rows) != 576 or not np.asarray(labels['valid'], bool).all():
        issues.append('All 576 frames and all six labels must be evaluable')
    for appearance in ('base', 'changed'):
        sub = [r for r in rows if r['appearance'] == appearance]
        if (len(sub), sum(r['truth'] is True for r in sub), sum(r['truth'] is False for r in sub)) != (288, 128, 160):
            issues.append('Unexpected appearance frame/positive/negative denominator: '+appearance)
    if len(paired['pairs']) != 288:
        issues.append('Need 288 complete appearance pairs')
    by_id = {r['id']: r['index'] for r in rows}
    for pair in paired['pairs']:
        b, c = by_id[pair['base_id']], by_id[pair['changed_id']]
        for key in ('classes', 'distances', 'valid'):
            if not np.array_equal(labels[key][b], labels[key][c], equal_nan=True):
                issues.append('Full-extent query label mismatch: '+pair['pair_id']+'/'+key)
    return dict(status='PASS' if not issues else 'NOT_EVALUABLE', issues=issues,
        public_pairs=public_pairs, all_rows_retained=len(rows), truth='Full-extent six-query classes and distances')


def gate_decisions(rows, measured, admissibility):
    def state(passed, evaluable=True):
        return 'PASS' if passed and evaluable else 'FAIL' if evaluable else 'NOT_EVALUABLE'
    admitted = admissibility['status'] == 'PASS'
    groups = sorted({r['base_group_id'] for r in rows})
    totals = np.array([[sum(r['truth'] is True for r in rows if r['base_group_id'] == g)] +
        [sum(r['truth'] is True and r['predictions'][a]['alert'] for r in rows if r['base_group_id'] == g)
         for a in ('local', 'raw')] for g in groups])
    rng = np.random.default_rng(GATES['bootstrap_seed'])
    boot = []
    for _ in range(GATES['bootstrap_resamples']):
        t = totals[rng.integers(0, len(groups), len(groups))].sum(0)
        boot.append((t[1]-t[2])/max(1, t[0]))
    ci = np.quantile(boot, [.025, .975]).tolist()
    component = {}
    for appearance in ('base', 'changed'):
        d = measured['comparisons_by_appearance'][appearance]['local_vs_raw']
        layer_deltas = {}
        for layer in ('BODY', 'HEAD'):
            sub = [r for r in rows if r['appearance'] == appearance and r['layer'] == layer and r['truth'] is True]
            layer_deltas[layer] = sum(int(r['predictions']['local']['alert'])-int(r['predictions']['raw']['alert']) for r in sub)/len(sub) if sub else None
        passed = (d['recall_delta'] is not None and d['recall_delta'] >= GATES['component_recall_delta']-1e-12
            and d['FP_delta'] <= GATES['component_FP_delta'] and d['false_segment_delta'] <= GATES['component_segment_delta']
            and d['improved_groups'] >= GATES['component_improved_groups']
            and all(v is not None and v >= -GATES['layer_max_recall_loss']-1e-12 for v in layer_deltas.values()))
        component[appearance] = dict(status=state(passed, admitted), layer_recall_deltas=layer_deltas)
    component_status = state(ci[0] > 0 and all(v['status'] == 'PASS' for v in component.values()), admitted)
    b, c = [measured['strata']['appearance'][a]['arms']['local'] for a in ('base', 'changed')]
    bc, cc = b['frames']['all_known'], c['frames']['all_known']
    recall_change = None if bc['recall'] is None or cc['recall'] is None else cc['recall']-bc['recall']
    fp_delta, segment_delta = cc['FP']-bc['FP'], c['false_alert_segment_count']-b['false_alert_segment_count']
    base_rescues = [p for p in measured['appearance_pairs']['pairs'] if p['truth'] is True
        and p['arms']['local']['base_alert'] and not p['arms']['A_current']['base_alert']]
    retained = [p['pair_id'] for p in base_rescues if p['arms']['local']['changed_alert']]
    fraction = len(retained)/len(base_rescues) if base_rescues else None
    enough = len(base_rescues) >= GATES['minimum_base_rescues']
    stable = (recall_change is not None and recall_change >= -GATES['appearance_max_recall_loss']-1e-12
        and fp_delta <= GATES['appearance_FP_delta'] and segment_delta <= GATES['appearance_segment_delta']
        and fraction is not None and fraction >= GATES['rescue_retention'])
    appearance = dict(status=state(stable, admitted and enough), recall_change=recall_change,
        FP_delta=fp_delta, false_segment_delta=segment_delta, base_rescues=len(base_rescues),
        retained_rescues=len(retained), retention_fraction=fraction,
        rescue_status=state(fraction is not None and fraction >= GATES['rescue_retention'], admitted and enough),
        base_rescue_pair_ids=[p['pair_id'] for p in base_rescues], retained_rescue_pair_ids=retained,
        lost_rescue_pair_ids=[p['pair_id'] for p in base_rescues if p['pair_id'] not in retained])
    ordering = measured['geometry_ordering']
    geometry = {a: dict(**ordering['by_appearance'][a]['local']) for a in ('base', 'changed')}
    for a, item in geometry.items():
        item['status'] = state(item['frame_ordering_fraction'] is not None and item['frame_ordering_fraction'] >= GATES['geometry_ordering_fraction'],
                               admitted and not ordering['issues'] and item['eligible_pairs'] > 0)
    geometry_status = ('NOT_EVALUABLE' if any(v['status'] == 'NOT_EVALUABLE' for v in geometry.values())
                       else state(all(v['status'] == 'PASS' for v in geometry.values())))
    d = measured['comparisons']['local_vs_A_current']
    strict = (d['recall_delta'] is not None and d['recall_delta'] >= GATES['strict_recall_delta']-1e-12
        and d['FP_delta'] <= GATES['strict_FP_delta'] and d['false_segment_delta'] <= GATES['strict_segment_delta']
        and not d['lost_true_frames'] and not any(e['lost'] or (e['delay_s'] is not None and e['delay_s'] > 1e-9) for e in d['event_differences']))
    statuses = [component_status, appearance['status'], geometry_status]
    retained_status = ('FAIL' if 'FAIL' in statuses else 'NOT_EVALUABLE'
                       if 'NOT_EVALUABLE' in statuses else 'PASS')
    return dict(configuration=GATES, source_status=admissibility['status'],
        component=dict(status=component_status, by_appearance=component,
            bootstrap=dict(unit='whole base geometry with both appearances and all relations', groups=len(groups),
                resamples=GATES['bootstrap_resamples'], seed=GATES['bootstrap_seed'], recall_delta_95=ci)),
        appearance_stability=appearance, geometry_sensitivity=dict(status=geometry_status, by_appearance=geometry),
        retained_transfer_component=retained_status, strict_system=dict(status=state(strict, admitted),
            scope='This controlled source only; no default App promotion'))


def report(rows, cutoffs, labels, public_pairs):
    metrics = evaluate_rows(rows, arms=ARMS)
    strata = {key: {str(value): evaluate_rows([r for r in rows if r[key] == value], arms=ARMS)
        for value in sorted({r[key] for r in rows})} for key in AXES}
    by_appearance = {appearance: {arm+'_vs_'+ref: comparison(
        [r for r in rows if r['appearance'] == appearance], strata['appearance'][appearance], arm, ref)
        for arm, ref in [('raw', 'A_current'), ('local', 'A_current'), ('local', 'raw')]}
        for appearance in ('base', 'changed')}
    measured = dict(status='PASS', scope='NEW_GEOMETRIES_SAME_GENERATOR_CONTROLLED_DEVELOPMENT',
        metrics=metrics, strata=strata,
        comparisons={arm+'_vs_'+ref: comparison(rows, metrics, arm, ref)
            for arm, ref in [('raw', 'A_current'), ('local', 'A_current'), ('local', 'raw')]},
        comparisons_by_appearance=by_appearance,
        query_metrics={arm: query_counts(rows, arm, cutoffs[arm]) for arm in LEARNED},
        appearance_pairs=appearance_pairs(rows), geometry_ordering=geometry_ordering(rows, cutoffs),
        thresholds=cutoffs, physical_return_ownership='NOT_ESTABLISHED',
        mask_IoU='NOT_APPLICABLE_SCALAR_CLASSIFIER', range_estimation='NOT_OUTPUT')
    measured['source_admissibility'] = source_admissibility(rows, labels, public_pairs, measured['appearance_pairs'])
    measured['gates'] = gate_decisions(rows, measured, measured['source_admissibility'])
    return measured
