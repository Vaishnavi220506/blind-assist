"""Pure saved-output MZ161 reducer; no I/O, fitting or prediction invocation."""
from collections import Counter
import math

import numpy as np

from evaluate_mz136_corridor_pair import pair_metrics, score
from mz161_dense_task import ray_query
from run_mz107_four_sensor import truth
from run_mz143_corridor_evidence import native_account


ARMS = ('frame', 'dense')
FAMILIES = ('suspended_head', 'substantial_body', 'near_rod_farwall', 'shallow_boundary_stress')


def _ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def _rates(result):
    m = result['metrics']
    result.update(precision=_ratio(m['TP'], m['TP']+m['FP']),
                  recall=_ratio(m['TP'], m['TP']+m['FN']),
                  accuracy=_ratio(m['TP']+m['TN'], sum(m[k] for k in ('TP', 'FP', 'FN', 'TN'))))
    return result


def _comparison(rows, gt, flags, reference, indices, candidate_report, reference_report):
    changes = dict(lost_true=[], gained_true=[], new_fp=[], removed_fp=[])
    for i in indices:
        if flags[i] == reference[i]: continue
        name = ('gained_true' if gt[i] else 'new_fp') if flags[i] else ('lost_true' if gt[i] else 'removed_fp')
        changes[name].append(rows[i]['id'])
    delta = []
    a, b = candidate_report['event_details'], reference_report['event_details']
    if len(a) != len(b): raise ValueError('Event inventory differs')
    for candidate, old in zip(a, b):
        if (candidate['episode'], candidate['onset_s']) != (old['episode'], old['onset_s']):
            raise ValueError('Event identity differs')
        if old['first_alert_s'] is not None:
            difference = None if candidate['first_alert_s'] is None else candidate['first_alert_s']-old['first_alert_s']
            delta.append(dict(episode=old['episode'], onset_s=old['onset_s'], extra_delay_s=difference))
    return dict(**changes, event_deltas=delta,
        all_reference_true_frames_retained=not changes['lost_true'],
        all_reference_events_without_extra_delay=all(d['extra_delay_s'] is not None and d['extra_delay_s'] <= 1e-9 for d in delta))


def _spatial_counts(label, logits, public_mask):
    known = np.asarray(label['known'], bool); target = np.asarray(label['target'], bool)
    visible = np.asarray(label['target_visible'], bool); risky = np.asarray(label['target_corridor'], bool)
    if any(x.shape != logits.shape for x in (known, target, visible, risky, public_mask)):
        raise ValueError('Dense map/label shape mismatch')
    if np.any(target & ~known) or np.any(visible & ~known) or not np.array_equal(risky, target & visible):
        raise ValueError('Inconsistent dense label masks')
    pred = logits >= 0
    columns = risky.sum(axis=0); detected = (pred & risky).sum(axis=0)
    offcorridor = visible & ~risky
    return dict(TP=int((known & target & pred).sum()), FP=int((known & ~target & pred).sum()),
        FN=int((known & target & ~pred).sum()), TN=int((known & ~target & ~pred).sum()),
        known_pixels=int(known.sum()), target_corridor_pixels=int(risky.sum()),
        target_risk_columns=int((columns > 0).sum()),
        target_risk_half_detected_columns=int(((columns > 0) & (2*detected >= columns)).sum()),
        target_visible_offcorridor_pixels=int(offcorridor.sum()),
        target_visible_offcorridor_false_positive_pixels=int((offcorridor & pred).sum()),
        unknown_pixels=int((~known).sum()), unknown_predicted_positive_pixels=int((~known & pred).sum()),
        all_map_pixels=int(logits.size), finite_map_pixels=int(np.isfinite(logits).sum()),
        public_query_pixels=int(public_mask.sum()),
        public_query_known_pixels=int((public_mask & known).sum()),
        public_query_target_corridor_pixels=int((public_mask & risky).sum()),
        predicted_positive_outside_public_query_pixels=int((~public_mask & pred).sum()))


def _spatial_report(cases, arm, indices):
    counts = Counter()
    for i in indices: counts.update(cases[i]['spatial'][arm])
    value = dict(counts)
    value.update(frames=len(indices), precision=_ratio(counts['TP'], counts['TP']+counts['FP']),
        recall=_ratio(counts['TP'], counts['TP']+counts['FN']),
        target_risk_half_column_recall=_ratio(counts['target_risk_half_detected_columns'], counts['target_risk_columns']),
        target_visible_offcorridor_false_positive_fraction=_ratio(counts['target_visible_offcorridor_false_positive_pixels'], counts['target_visible_offcorridor_pixels']),
        unknown_predicted_positive_fraction=_ratio(counts['unknown_predicted_positive_pixels'], counts['unknown_pixels']),
        public_map_coverage_fraction=_ratio(counts['public_query_pixels'], counts['all_map_pixels']),
        public_query_target_corridor_coverage=_ratio(counts['public_query_target_corridor_pixels'], counts['target_corridor_pixels']),
        undefined_rate_semantics='NULL_IS_NOT_APPLICABLE_NOT_PERFECT_SCORE')
    return value


def evaluate(rows, es, labels, maps, predictions, baseline, spec, metadata):
    """Reduce the sealed 192-frame original TRAIN cohort, without reading files.

    The caller admits hashes and TRAIN selection before calling. Frame logits
    use public ray eligibility; dense spatial metrics use all known reference
    pixels. Baseline has no comparable dense output. Missing labels never count
    as negative pixels, and any nonalert remains UNKNOWN.
    """
    n = len(rows)
    if n != 192 or len(es) != n or len(labels) != n:
        raise ValueError('MZ161 requires all 192 selected original TRAIN rows')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != n or [e['id'] for e in es] != ids or set(metadata) != set(ids):
        raise ValueError('Cohort identity mismatch')
    if set(maps) != set(ARMS) or set(predictions) != set(ARMS):
        raise ValueError('Expected fixed frame and dense arms')
    baseline = np.asarray(baseline, bool)
    if baseline.shape != (n,): raise ValueError('Baseline must have 192 flags')
    for arm in ARMS:
        if [p['id'] for p in predictions[arm]] != ids:
            raise ValueError('Prediction ordering mismatch')
        if maps[arm].ndim != 3 or maps[arm].shape[0] != n or maps[arm].dtype != np.float32:
            raise ValueError('Saved maps must be N x H x W float32')
        if not np.isfinite(maps[arm]).all(): raise ValueError('Nonfinite saved pixel logits')
    gt = np.array([truth(e) for e in es], bool)
    values = {'baseline':baseline.astype(float)}
    flags = {'baseline':baseline}
    for arm in ARMS:
        values[arm] = np.array([p['frame_logit'] for p in predictions[arm]], float)
        if not np.isfinite(values[arm]).all(): raise ValueError('Nonfinite frame logits')
        flags[arm] = values[arm] >= 0
    partitions = {part:[i for i, r in enumerate(rows) if metadata[r['id']]['partition'] == part] for part in ('fit', 'heldout')}
    if sum(map(len, partitions.values())) != n or not all(partitions.values()):
        raise ValueError('Expected nonempty fit/heldout partitions')
    group_parts, episode_parts, episode_indices = {}, {}, {}
    for i, row in enumerate(rows):
        m = metadata[row['id']]
        if m['family'] != es[i]['family']: raise ValueError('Family metadata mismatch')
        for key, table in ((m['scene_group'], group_parts), (row['episode_id'], episode_parts)):
            table.setdefault(key, set()).add(m['partition'])
        episode_indices.setdefault(row['episode_id'], []).append(i)
    if any(len(parts) != 1 for parts in [*group_parts.values(), *episode_parts.values()]):
        raise ValueError('Scene group or episode crosses partitions')
    pairs = {part:[] for part in partitions}
    for pair in spec['pairs']:
        a, b = pair['episodes']
        if a not in episode_indices and b not in episode_indices: continue
        if a not in episode_indices or b not in episode_indices: raise ValueError('Incomplete selected paired episode')
        aa, bb = episode_indices[a], episode_indices[b]
        if len(aa) != len(bb): raise ValueError('Paired episode lengths differ')
        pa, pb = metadata[rows[aa[0]]['id']]['partition'], metadata[rows[bb[0]]['id']]['partition']
        if pa != pb: continue
        for i, j in zip(aa, bb):
            if rows[i]['time_s'] != rows[j]['time_s']: raise ValueError('Paired frame times differ')
            pairs[pa].append(dict(a=i, b=j))
    cases = []; yaw = 0.; previous_episode = None
    for i, (row, label) in enumerate(zip(rows, labels)):
        if row['episode_id'] != previous_episode: yaw = 0.
        if row['imu_valid']: yaw += row['delta_yaw']
        previous_episode = row['episode_id']
        public_mask = ray_query(row, yaw)[1][2] > 0
        case = dict(id=row['id'], episode_id=row['episode_id'], time_s=row['time_s'], **metadata[row['id']],
            truth=bool(gt[i]), baseline=bool(baseline[i]), arms={}, spatial={},
            native_volume_vs_visible_mismatch=bool(gt[i] != bool(np.asarray(label['target']).any())))
        for arm in ARMS:
            p = predictions[arm][i]
            pm, tm = float(p['pixel_max']), float(p['token_max'])
            expected_pm = float(np.where(public_mask, maps[arm][i], -30.).max())
            if not np.isfinite([pm, tm]).all() or not math.isclose(pm, expected_pm, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError('Saved pixel maximum differs from public query-masked map')
            if not math.isclose(float(p['frame_logit']), max(pm, tm), rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError('Frame maximum disagrees with saved branches')
            winner = p['winner']
            valid_winner = (winner == 'pixel' and pm >= tm) or (winner == 'token' and tm >= pm) or (winner == 'tie' and pm == tm)
            if not valid_winner: raise ValueError('Saved branch winner mismatch')
            case['arms'][arm] = dict(candidate=bool(flags[arm][i]), frame_logit=float(p['frame_logit']),
                pixel_max=pm, token_max=tm, winner=winner, pixel_only=pm >= 0 and tm < 0,
                token_only=tm >= 0 and pm < 0, both=pm >= 0 and tm >= 0, neither=pm < 0 and tm < 0)
            case['spatial'][arm] = _spatial_counts(label, maps[arm][i], public_mask)
        cases.append(case)
    partition_reports, spatial_reports = {}, {}
    for part, ix in partitions.items():
        reports = {arm:_rates(score(rows, es, gt, flag, ix, baseline)) for arm, flag in flags.items()}
        for arm, report in reports.items():
            report['pairs'] = pair_metrics(gt, values[arm], flags[arm], pairs[part])
            report['strata'] = {name:_rates(score(rows, es, gt, flags[arm],
                [i for i in ix if (es[i]['family'] == 'shallow_boundary_stress') == pressure], baseline))
                for name, pressure in (('ordinary', False), ('boundary_pressure', True))}
            for family, metrics in list(report['families'].items()):
                report['families'][family] = _rates(dict(metrics=metrics))
            native = native_account([rows[i] for i in ix], [es[i] for i in ix], flags[arm][ix], baseline[ix])
            report['native'] = {k:v for k, v in native.items() if k != 'cases'}
            report['native']['radar_native_lineage'] = 'NOT_EVALUABLE'
            for i, record in zip(ix, native['cases']):
                cases[i].setdefault('native', {})[arm] = record
            if arm != 'baseline':
                report['vs_baseline'] = _comparison(rows, gt, flags[arm], baseline, ix, report, reports['baseline'])
                report['vs_control'] = _comparison(rows, gt, flags[arm], flags['frame'], ix, report, reports['frame'])
                report['winners'] = dict(Counter(predictions[arm][i]['winner'] for i in ix))
                report['branch_decisions'] = {k:sum(cases[i]['arms'][arm][k] for i in ix) for k in ('pixel_only', 'token_only', 'both', 'neither')}
        partition_reports[part] = reports
        spatial_reports[part] = {arm:dict(all=_spatial_report(cases, arm, ix),
            families={family:_spatial_report(cases, arm, [i for i in ix if es[i]['family'] == family])
                      for family in sorted({es[i]['family'] for i in ix})}) for arm in ARMS}
    held = partition_reports['heldout']; candidate = held['dense']; base = held['baseline']; control = held['frame']
    fit_accuracy = partition_reports['fit']['dense']['accuracy']
    alert_checks = dict(all_baseline_true_frames_retained=candidate['vs_baseline']['all_reference_true_frames_retained'],
        all_baseline_events_without_extra_delay=candidate['vs_baseline']['all_reference_events_without_extra_delay'],
        fp_at_most_floor_75pct_baseline=candidate['metrics']['FP'] <= math.floor(.75*base['metrics']['FP']),
        no_family_fp_increase=all(candidate['families'][f]['metrics']['FP'] <= base['families'][f]['metrics']['FP'] for f in base['families']),
        zero_native_supported_nonalerts=not candidate['native']['nonalert_with_native_corridor_contributors'],
        fewer_fp_than_frame_control=candidate['metrics']['FP'] < control['metrics']['FP'],
        all_control_true_frames_retained=candidate['vs_control']['all_reference_true_frames_retained'])
    family_spatial = spatial_reports['heldout']['dense']['families']; spatial_checks = {}
    for family in FAMILIES:
        value = family_spatial.get(family, {})
        spatial_checks[family] = {name:None if value.get(metric) is None else bool(value[metric] >= .9)
            for name, metric in (('precision_at_least_90pct', 'precision'), ('recall_at_least_90pct', 'recall'),
                                 ('target_risk_half_column_recall_at_least_90pct', 'target_risk_half_column_recall'))}
    primary_spatial = all(v['precision_at_least_90pct'] is True and v['recall_at_least_90pct'] is True for v in spatial_checks.values())
    column_applicable = [f for f in FAMILIES if spatial_checks[f]['target_risk_half_column_recall_at_least_90pct'] is not None]
    columns_pass = all(spatial_checks[f]['target_risk_half_column_recall_at_least_90pct'] for f in FAMILIES) if len(column_applicable) == len(FAMILIES) else None
    fit_pass = fit_accuracy is not None and fit_accuracy >= .95
    alert_pass = all(alert_checks.values())
    spatial_pass = primary_spatial and columns_pass is True
    summary = dict(authority='CONSUMED_ORIGINAL_TRAIN_SCENE_GROUP_HELDOUT_NOT_DEV_TEST_OR_PROMOTION', frames=n,
        threshold_logit=0., partitions=partition_reports, spatial=spatial_reports,
        label_audit={part:dict(frames=len(ix), native_volume_vs_visible_mismatches=sum(cases[i]['native_volume_vs_visible_mismatch'] for i in ix),
            mismatch_ids=[cases[i]['id'] for i in ix if cases[i]['native_volume_vs_visible_mismatch']]) for part, ix in partitions.items()},
        gates=dict(fit_accuracy=fit_accuracy, fit_accuracy_at_least_95pct=bool(fit_pass), heldout_alert_checks=alert_checks,
            heldout_alert_pass=alert_pass, heldout_family_spatial_checks=spatial_checks,
            target_risk_column_applicable_families=column_applicable, all_applicable_target_risk_columns_pass=columns_pass,
            heldout_spatial_pass=bool(spatial_pass), overall_pass=bool(fit_pass and alert_pass and spatial_pass)),
        decision='MZ161_TRAIN_HELDOUT_COMPONENT_PASS_NOT_PROMOTION' if fit_pass and alert_pass and spatial_pass else 'MZ161_FIXED_DENSE_TASK_GATE_NOT_MET',
        limits=['Pixel risk scores are not metric depth, certified extent or clear space.',
                'Native AABB first hits use source-commanded camera pose, not rendered labels.',
                'All known pixels enter spatial metrics; public ray eligibility is reported separately.',
                'No baseline dense map exists; no spatial precision is invented for MZ129.',
                'Unknown-pixel positives are reported without asserting false-positive truth.',
                'Slot input retention alone does not guarantee learned alert retention.'])
    return dict(summary=summary, cases=cases)
