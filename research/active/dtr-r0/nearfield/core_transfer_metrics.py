"""Frozen-core transfer metrics; evaluator-only labels and sampled timing.

No fitting, threshold selection, geometry relabelling, or model calls occur here.
Layout strata always retain whole clips; relation masks retain the timeline so
removing a middle frame cannot merge two false-alert segments.
"""
from ba_camera_corridor_metrics import evaluate_rows

ARMS = ('raw', 'calibrated', 'rgb', 'no_rgb')
RELATIONS = ('INSIDE', 'BOUNDARY', 'OUTSIDE')
ZONE_RELATIONS = ('INSIDE', 'OUTSIDE', 'CROSSING', 'UNKNOWN')
STRATA = ('layer', 'type_id', 'background', 'layout_relation')


def _ratio(a, b):
    return a / b if b else None


def _events(metrics):
    return {(e['clip_id'], e['start_frame'], e['end_frame']): e
            for e in metrics['events']}


def _retention(rows, metrics, candidate, reference):
    current, baseline = _events(metrics['arms'][candidate]), _events(metrics['arms'][reference])
    lost = [list(key) for key, event in baseline.items()
            if event['detected'] and not current[key]['detected']]
    delayed = [list(key) for key, event in baseline.items()
               if event['detected'] and current[key]['detected']
               and current[key]['first_alert_time_s'] > event['first_alert_time_s'] + 1e-8]
    tp_loss = [r['id'] for r in rows if r['truth'] and r['predictions'][reference]['alert']
               and not r['predictions'][candidate]['alert']]
    return dict(reference=reference, positive_alert_frames_lost=tp_loss,
                detected_events_lost=lost, detected_events_delayed=delayed,
                passed=not (tp_loss or lost or delayed))


def _zone_audit(audits):
    result = dict(samples=len(audits), truth_relation_counts={
        label: sum(z['truth_relation'] == label for z in audits) for label in ZONE_RELATIONS}, arms={})
    for arm in ('rgb', 'no_rgb'):
        confusion = {truth: {pred: 0 for pred in ZONE_RELATIONS} for truth in ZONE_RELATIONS}
        for zone in audits:
            confusion[zone['truth_relation']][zone[arm + '_relation']] += 1
        outside = [z for z in audits if z['truth_relation'] == 'OUTSIDE']
        nonoutside = [z for z in audits if z['truth_relation'] in ('INSIDE', 'CROSSING')]
        suppressed = [z for z in audits if z[arm + '_suppress']]
        native = [z for z in suppressed if z['native_corridor_contributors'] > 0]
        result['arms'][arm] = dict(confusion=confusion,
            eligible_pure_accuracy=_ratio(sum(z[arm + '_relation'] == z['truth_relation'] for z in audits if z['truth_relation'] != 'UNKNOWN'), sum(z['truth_relation'] != 'UNKNOWN' for z in audits)),
            outside_recall=_ratio(sum(z[arm + '_relation'] == 'OUTSIDE' for z in outside), len(outside)),
            outside_suppression_recall=_ratio(sum(z[arm + '_suppress'] for z in outside), len(outside)),
            nonoutside_suppression_rate=_ratio(sum(z[arm + '_suppress'] for z in nonoutside), len(nonoutside)),
            suppressed_samples=len(suppressed),
            suppressed_unknown_samples=sum(z['truth_relation'] == 'UNKNOWN' for z in suppressed),
            native_contributor_suppressed_samples=len(native),
            native_corridor_contributions_suppressed=sum(z['native_corridor_contributors'] for z in native),
            native_contributor_suppressed_sample_ids=[z['sample_id'] for z in native])
    return result


def evaluate(rows, zone_audits, dt_s=.2):
    """Evaluate four sealed arms. Missing opportunities cannot produce a pass.

    ``relation`` is the authenticated full-volume geometric relation; planned
    ``layout_relation`` is only a clip-level cohort tag. Boundary may include
    either sign of the tolerance band; strict truth remains independently supplied.
    ``sample_id`` uniquely identifies an eligible zone, ``frame_id`` its row ID.
    """
    rows, zone_audits = list(rows), list(zone_audits)
    ids = {r['id'] for r in rows}
    if len(ids) != len(rows):
        raise ValueError('Duplicate frame identity')
    clip_tags = {}
    for row in rows:
        if type(row['truth']) is not bool:
            raise ValueError('Transfer scoring requires admitted boolean truth')
        if row['relation'] not in RELATIONS or row['layout_relation'] not in RELATIONS:
            raise ValueError('Unknown frame or layout relation')
        if row['layer'] not in ('BODY', 'HEAD'):
            raise ValueError('Unknown layer')
        if (row['relation'] == 'INSIDE' and not row['truth']) or (row['relation'] == 'OUTSIDE' and row['truth']):
            raise ValueError('Full-volume relation contradicts strict truth')
        if (row['relation'] == 'BOUNDARY') != row['boundary']:
            raise ValueError('Full-volume relation contradicts boundary flag')
        tags = tuple(row[key] for key in STRATA)
        if clip_tags.setdefault(row['clip_id'], tags) != tags:
            raise ValueError('Stratum changes within clip')
    sample_ids = set()
    for zone in zone_audits:
        if zone['sample_id'] in sample_ids or zone['frame_id'] not in ids:
            raise ValueError('Duplicate zone identity or unknown parent frame')
        sample_ids.add(zone['sample_id'])
        if any(zone[key] not in ZONE_RELATIONS for key in ('truth_relation', 'rgb_relation', 'no_rgb_relation')):
            raise ValueError('Unknown zone relation')
        if any(type(zone[key]) is not bool for key in ('rgb_suppress', 'no_rgb_suppress')):
            raise ValueError('Zone suppression flags must be bool')
        count = zone['native_corridor_contributors']
        if type(count) is not int or count < 0:
            raise ValueError('Invalid native contributor count')
    metrics = evaluate_rows(rows, dt_s=dt_s, arms=ARMS)
    strata = {key: {value: evaluate_rows([r for r in rows if r[key] == value], dt_s=dt_s, arms=ARMS)
                    for value in sorted({r[key] for r in rows})} for key in STRATA}
    outside_rows = [{**r, 'truth': r['truth'] if r['relation'] == 'OUTSIDE' else None} for r in rows]
    outside_metrics = evaluate_rows(outside_rows, dt_s=dt_s, arms=ARMS)
    metrics_by_relation = {relation: evaluate_rows([{**r, 'truth': r['truth'] if r['relation'] == relation else None} for r in rows], dt_s=dt_s, arms=ARMS) for relation in RELATIONS}
    outside = {arm: dict(frames=outside_metrics['known_truth_frames'],
        FP=m['frames']['all_known']['FP'],
        FPR=m['frames']['all_known']['false_alert_rate_known_negative'],
        false_alert_segment_count=m['false_alert_segment_count'],
        false_alert_sampled_duration_s=m['false_alert_sampled_duration_s'],
        false_alert_segments=m['false_alert_segments']) for arm, m in outside_metrics['arms'].items()}
    core = evaluate_rows([r for r in rows if r['layout_relation'] == 'INSIDE'], dt_s=dt_s, arms=ARMS)
    boundary = evaluate_rows([r for r in rows if r['layout_relation'] == 'BOUNDARY'], dt_s=dt_s, arms=ARMS)
    audit = _zone_audit(zone_audits)
    opportunities = dict(positive_frames=sum(r['truth'] for r in rows),
        outside_frames=outside_metrics['known_truth_frames'],
        core_events=core['arms']['raw']['event_count'],
        audited_outside_samples=audit['truth_relation_counts']['OUTSIDE'])
    opportunity_checks = {key: value > 0 for key, value in opportunities.items()}
    retentions = {ref: _retention(rows, metrics, 'rgb', ref) for ref in ('raw', 'calibrated', 'no_rgb')}
    checks = dict(nonempty_opportunities=all(opportunity_checks.values()),
        fewer_fp_than_calibrated=metrics['arms']['rgb']['frames']['all_known']['FP'] < metrics['arms']['calibrated']['frames']['all_known']['FP'],
        fewer_fp_than_no_rgb=metrics['arms']['rgb']['frames']['all_known']['FP'] < metrics['arms']['no_rgb']['frames']['all_known']['FP'],
        no_tp_event_or_onset_loss=all(v['passed'] for v in retentions.values()),
        no_native_contributor_suppression=audit['arms']['rgb']['native_contributor_suppressed_samples'] == 0)
    # Primary OUTSIDE means independently constructed lateral OUTSIDE clips.
    # Full-volume OUTSIDE above also includes pre-entry depth negatives; retain
    # both denominators, but depth-only improvements cannot establish attribution.
    lateral_outside = strata['layout_relation'].get('OUTSIDE')
    for ref in ('calibrated', 'no_rgb'):
        checks['fewer_outside_layout_fp_than_' + ref] = bool(lateral_outside and
            lateral_outside['arms']['rgb']['frames']['all_known']['FP'] < lateral_outside['arms'][ref]['frames']['all_known']['FP'])
    clip_first = {arm: {clip: next((r['time_s'] for r in sorted(rows, key=lambda r:r['frame_in_clip']) if r['clip_id'] == clip and r['predictions'][arm]['alert']), None) for clip in sorted(clip_tags)} for arm in ARMS}
    calibrated = _retention(rows, metrics, 'calibrated', 'raw')
    calibration_checks = dict(nonempty_opportunities=all(opportunity_checks[k] for k in ('positive_frames', 'outside_frames', 'core_events')),
        fewer_fp_than_raw=metrics['arms']['calibrated']['frames']['all_known']['FP'] < metrics['arms']['raw']['frames']['all_known']['FP'],
        no_tp_event_or_onset_loss=calibrated['passed'])
    return dict(metrics=metrics, strata=strata, metrics_by_relation=metrics_by_relation,
        per_frame_relation_counts={label: sum(r['relation'] == label for r in rows) for label in RELATIONS},
        outside_only=outside, outside_layout=lateral_outside, core_events=core, boundary_events=boundary, zone_attribution=audit,
        clip_first_alert_time_s=clip_first,
        opportunities=opportunities,
        primary_gate=dict(passed=all(checks.values()), checks=checks,
            opportunity_checks=opportunity_checks, retention=retentions),
        calibration_gate=dict(passed=all(calibration_checks.values()), checks=calibration_checks, retention=calibrated),
        limitations=['Posed nominal sampled timing, not real-time latency.',
                     'Zone labels are evaluator-only pure-contributor attribution, not hardware truth.',
                     'UNKNOWN non-alerts remain abstentions, never true negatives.'])
