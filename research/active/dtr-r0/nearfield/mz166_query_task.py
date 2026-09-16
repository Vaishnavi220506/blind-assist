"""Pure MZ166 saved-output reductions; no fitting, inference or file access."""
import math

import numpy as np

from mz161_dense_task import ray_query
from mz165_pretrained_task import evaluate_context


ARMS = ('central', 'queries')
OFFSETS = (-.45, -.30, -.15, 0., .15, .30, .45)
ZERO = OFFSETS.index(0.)
METHOD = dict(schema='MZ166_DENSE_VIRTUAL_QUERY_SUPERVISION', arms=list(ARMS),
    offsets_m=list(OFFSETS), threshold_logit=0.,
    primary='CENTRAL_QUERY_ALERTS_ON_CONSUMED_TRAIN_SCENE_HOLDOUT',
    diagnostic='FIT_ONLY_CHANGED_QUERY_FRAME_AND_PIXEL_PAIRS',
    query_evidence='SEPARATE_FROM_CENTRAL_ALERT_ADMISSION',
    unknown='NO_SAVED_REFERENCE_IS_UNKNOWN; NONALERT_IS_NOT_CLEAR')


def _ratio(a, b):
    return a/b if b else None


def _rename(value, aliases):
    if isinstance(value, dict):
        return {aliases.get(k, k): _rename(v, aliases) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v, aliases) for v in value]
    return value


def _query_diagnostic(rows, labels, maps, predictions, cases, query_maps,
                      query_predictions, query_indices, query_targets, query_frames):
    indices = np.asarray(query_indices)
    if (indices.shape != (24,) or indices.dtype.kind not in 'iu'
            or len(set(indices.tolist())) != 24
            or np.any(indices < 0) or np.any(indices >= len(rows))):
        raise ValueError('Expected 24 distinct integer global query indices')
    families = sorted({c['family'] for c in cases})
    expected = {i for family in families for i in
                [j for j, c in enumerate(cases)
                 if c['partition'] == 'fit' and c['family'] == family][:6]}
    if len(families) != 4 or len(expected) != 24 or set(indices.tolist()) != expected:
        raise ValueError('Queries require the first six FIT frames of each family in source order')
    targets, frame_targets = np.asarray(query_targets), np.asarray(query_frames)
    shape = (24, len(OFFSETS), *np.asarray(labels[int(indices[0])]['target']).shape)
    if targets.shape != shape or targets.dtype != np.bool_:
        raise ValueError('Query targets must be bool [24,7,H,W]')
    if frame_targets.shape != (24, len(OFFSETS)) or frame_targets.dtype != np.bool_:
        raise ValueError('Query frame truth must be bool [24,7]')
    if set(query_maps) != set(ARMS) or set(query_predictions) != set(ARMS):
        raise ValueError('Expected central and queries diagnostic arms')
    for arm in ARMS:
        array = np.asarray(query_maps[arm])
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Query maps must be finite float32 [24,7,H,W]')
        if len(query_predictions[arm]) != 24 or any(len(p) != 7 for p in query_predictions[arm]):
            raise ValueError('Query prediction records must have shape [24][7]')

    yaws = []; yaw = 0.; episode = None
    for row in rows:
        if row['episode_id'] != episode:
            yaw = 0.
        if row['imu_valid']:
            yaw += row['delta_yaw']
        yaws.append(yaw); episode = row['episode_id']
    records = []
    for local, index in enumerate(indices.tolist()):
        label = labels[index]; known = np.asarray(label['known'], bool)
        if known.shape != shape[-2:] or np.any(targets[local] & ~known[None]):
            raise ValueError('Query positives must have the unchanged known first-hit reference')
        if (not np.array_equal(targets[local, ZERO], label['target'])
                or bool(frame_targets[local, ZERO]) != cases[index]['truth']):
            raise ValueError('Query0 labels disagree with central reference')
        flags = {arm: [] for arm in ARMS}
        for q, offset in enumerate(OFFSETS):
            shifted = dict(rows[index])
            shifted['camera_in_body_m'] = list(rows[index]['camera_in_body_m'])
            shifted['camera_in_body_m'][1] -= offset
            eligible = ray_query(shifted, yaws[index])[1][2] > 0
            for arm in ARMS:
                p = query_predictions[arm][local][q]
                if not all(math.isfinite(float(p[k])) for k in ('frame_logit', 'pixel_max', 'token_max')):
                    raise ValueError('Nonfinite query prediction record')
                maximum = float(np.where(eligible, query_maps[arm][local, q], -30.).max())
                if (not math.isclose(maximum, p['pixel_max'], rel_tol=1e-6, abs_tol=1e-6)
                        or not math.isclose(max(p['pixel_max'], p['token_max']), p['frame_logit'], rel_tol=1e-6, abs_tol=1e-6)):
                    raise ValueError('Query record differs from masked map or branch maximum')
                flags[arm].append(bool(p['frame_logit'] >= 0))
        for arm in ARMS:
            if flags[arm][ZERO] != cases[index]['arms'][arm]['candidate']:
                raise ValueError('Query0 alert differs from central inference')
            if not np.allclose(query_maps[arm][local, ZERO], maps[arm][index], rtol=1e-5, atol=1e-5):
                raise ValueError('Query0 map differs from central inference')
        for q in range(len(OFFSETS)):
            if q == ZERO:
                continue
            changed = known & (targets[local, q] != targets[local, ZERO])
            frame_changed = bool(frame_targets[local, q] != frame_targets[local, ZERO])
            arm_counts = {}
            for arm in ARMS:
                zero_correct = (query_maps[arm][local, ZERO] >= 0) == targets[local, ZERO]
                shifted_correct = (query_maps[arm][local, q] >= 0) == targets[local, q]
                arm_counts[arm] = dict(
                    frame_both_correct=int(frame_changed
                        and flags[arm][ZERO] == bool(frame_targets[local, ZERO])
                        and flags[arm][q] == bool(frame_targets[local, q])),
                    pixel_both_correct=int((changed & zero_correct & shifted_correct).sum()))
            records.append(dict(id=rows[index]['id'], global_index=index,
                family=cases[index]['family'], query_offset_m=OFFSETS[q],
                frame_label_changed=frame_changed, pixel_label_changed_pairs=int(changed.sum()), arms=arm_counts))

    def reduce(selected):
        frame_n = sum(r['frame_label_changed'] for r in selected)
        pixel_n = sum(r['pixel_label_changed_pairs'] for r in selected)
        result = dict(images=len({r['id'] for r in selected}), query_pairs=len(selected),
            frame_label_changed_pairs=frame_n, pixel_label_changed_pairs=pixel_n, arms={})
        for arm in ARMS:
            f = sum(r['arms'][arm]['frame_both_correct'] for r in selected)
            p = sum(r['arms'][arm]['pixel_both_correct'] for r in selected)
            result['arms'][arm] = dict(frame_both_correct=f, frame_both_correct_rate=_ratio(f, frame_n),
                pixel_both_correct=p, pixel_both_correct_rate=_ratio(p, pixel_n))
        return result

    all_counts = reduce(records)
    checks = dict(frame_denominator_nonempty=all_counts['frame_label_changed_pairs'] > 0,
        pixel_denominator_nonempty=all_counts['pixel_label_changed_pairs'] > 0,
        candidate_frame_both_correct_strictly_better=all_counts['arms']['queries']['frame_both_correct'] > all_counts['arms']['central']['frame_both_correct'],
        candidate_pixel_both_correct_strictly_better=all_counts['arms']['queries']['pixel_both_correct'] > all_counts['arms']['central']['pixel_both_correct'])
    return dict(authority='FIXED_FIT24_QUERY_DIAGNOSTIC_NOT_HELDOUT_OR_ALERT_ADMISSION',
        offsets_m=list(OFFSETS), zero_query_index=ZERO, query_indices=indices.tolist(),
        ids=[rows[i]['id'] for i in indices], all=all_counts,
        families={family: reduce([r for r in records if r['family'] == family]) for family in families},
        checks=checks, strong_query_evidence=all(checks.values()), cases=records,
        limits=['Each nonzero query is paired with query0; repeated pixels across offsets remain repeated pixel-pairs.',
                'Pixel rates pool changed known pixel-pairs, not per-image averages; unchanged and UNKNOWN pixels are excluded.',
                'Frame changes use native volume truth; visible pixel truth need not agree with frame volume truth.',
                'Query0 maps use 1e-5 numerical tolerance, while their alert flags must agree exactly.',
                'This diagnostic is on FIT examples and does not establish held-scene query generalization.'])


def evaluate_queries(rows, es, labels, maps, predictions, baseline, spec, metadata,
                     prior_predictions, query_maps, query_predictions, query_indices,
                     query_targets, query_frames):
    """Evaluate sealed central outputs and a separate fixed FIT query diagnostic.

    Callers authenticate files and row identities. No arguments are modified.
    ``prior_predictions`` contains the original MZ165 pretrained zero-cutoff
    records. Query rows may use any order if ``query_indices`` explicitly binds
    that order to the fixed first-six-FIT-per-family inventory.
    """
    if set(maps) != set(ARMS) or set(predictions) != set(ARMS):
        raise ValueError('Expected central and queries main arms')
    aliases = {'random': 'central', 'pretrained': 'queries'}
    result = evaluate_context(rows, es, labels, {k: maps[v] for k, v in aliases.items()},
        {k: predictions[v] for k, v in aliases.items()}, baseline, spec, metadata)
    result = _rename(result, aliases)
    summary = result['summary']; gates = summary['gates']; cases = result['cases']
    ids = [row['id'] for row in rows]
    if len(prior_predictions) != len(rows) or [p['id'] for p in prior_predictions] != ids:
        raise ValueError('MZ165 prior prediction IDs/order differ')
    if not all(math.isfinite(float(p['frame_logit'])) for p in prior_predictions):
        raise ValueError('Nonfinite MZ165 prior score')
    prior_flags = [bool(p['frame_logit'] >= 0) for p in prior_predictions]
    held_indices = [i for i, c in enumerate(cases) if c['partition'] == 'heldout']
    prior_true = [i for i in held_indices if cases[i]['truth'] and prior_flags[i]]
    for arm in ARMS:
        lost = [ids[i] for i in prior_true if not cases[i]['arms'][arm]['candidate']]
        summary['partitions']['heldout'][arm]['vs_mz165_pretrained'] = dict(
            prior_true_frames=len(prior_true), retained_true_frames=len(prior_true)-len(lost),
            lost_true=lost, all_prior_true_frames_retained=not lost)
    checks = gates['heldout_alert_checks']
    checks['fewer_fp_than_central_control'] = checks.pop('fewer_fp_than_random_control')
    checks['all_mz165_pretrained_true_frames_retained'] = summary['partitions']['heldout']['queries']['vs_mz165_pretrained']['all_prior_true_frames_retained']
    gates['heldout_alert_pass'] = all(checks.values())
    gates['overall_pass'] = bool(gates['fit_accuracy_at_least_95pct']
        and gates['fit_native_supported_nonalerts_zero'] and gates['heldout_alert_pass'])
    summary['legacy_evaluator_arm_aliases'] = dict(mz161={'frame': 'central', 'dense': 'queries'}, mz165=aliases)
    summary['method'] = METHOD
    summary['decision'] = ('MZ166_VIRTUAL_QUERY_ALERT_COMPONENT_PASS' if gates['overall_pass']
                           else 'MZ166_VIRTUAL_QUERY_ALERT_GAIN_NOT_MET')
    diagnostic = _query_diagnostic(rows, labels, maps, predictions, cases, query_maps,
        query_predictions, query_indices, query_targets, query_frames)
    summary['query_conditioning'] = dict(strong_query_evidence=diagnostic['strong_query_evidence'],
        role='SEPARATE_FIT_DIAGNOSTIC_NOT_AN_ALERT_GATE', checks=diagnostic['checks'])
    return dict(cases=cases, summary=summary, query_diagnostic=diagnostic)
