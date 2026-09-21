"""Independent path-based verification of the frozen conditional decoder.

No runner import. Uses the independent current-only audit's event/count recount,
not the production metrics. Only selection thresholds are enumerated.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import audit_current_only as independent_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'artifacts.local/work/ba-last-layer-20260921'
CURRENT = ROOT / 'artifacts.local/work/ba-current-only-20260921'
TRANSFER = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
OUT = ROOT / 'artifacts.local/work/ba-conditional-hold-20260921'
ROLES = ('selection', 'evaluation')
ARMS = ('A_current', 'A_hold', 'original_current', 'original_hold', 'conditional')
independent_metrics.ARMS = ARMS
same = independent_metrics.same
clips = independent_metrics.clips


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def path_predictions(rows, threshold, entry):
    """Alert iff a trigger reaches this frame through an unbroken >=keep path."""
    flags = {}
    for c in clips(rows):
        assert len(c) == 24 and [r['frame_in_clip'] for r in c] == list(range(24))
        assert all(math.isfinite(r['score']) and math.isclose(r['time_s'], i * .2, rel_tol=0, abs_tol=1e-8)
                   for i, r in enumerate(c))
        triggers = [r['a'] or r['score'] >= entry for r in c]
        for end, r in enumerate(c):
            reachable = any(triggers[start] and all(c[k]['score'] >= threshold for k in range(start + 1, end + 1))
                            for start in range(end + 1))
            flags[r['id']] = dict(A_current=r['a'], A_hold=r['a'] or (end > 0 and c[end - 1]['a']),
                original_current=triggers[end], original_hold=triggers[end] or (end > 0 and triggers[end - 1]),
                conditional=reachable)
    return [dict(id=r['id'], flags=flags[r['id']], current_unknown={a: r['unknown'] for a in ARMS}) for r in rows]


def curve_record(rows, labels, threshold, entry):
    predictions = path_predictions(rows, threshold, entry)
    added_false, gains = [], {'Core': [], 'Boundary': []}
    for r, p, y in zip(rows, predictions, labels):
        f = p['flags']
        if not f['conditional'] or f['original_current']:
            continue
        if not y['truth']:
            added_false.append(r['id'])
        else:
            gains['Boundary' if r['layout_relation'] == 'BOUNDARY' else 'Core'].append(r['id'])
    return dict(tau_keep=threshold, feasible=not added_false, added_FP_ids=added_false,
                gains=gains, boundary_gain=len(gains['Boundary']), core_gain=len(gains['Core']))


def opportunities(rows):
    results = []
    for c in clips(rows):
        positive = [r for r in c if r['truth']]
        if not positive:
            continue
        seeds = [r for r in c if r['flags']['original_current']]
        detected = [r for r in positive if r['flags']['original_current']]
        first = seeds[0]['frame_in_clip'] if seeds else math.inf
        before = [r['id'] for r in positive if r['frame_in_clip'] < first]
        reachable = [r for r in positive if r['frame_in_clip'] >= first]
        added = [r for r in positive if r['flags']['conditional'] and not r['flags']['original_current']]
        counts = Counter()
        for r in added:
            if not detected:
                counts['previously_missed'] += 1
            elif r['frame_in_clip'] < detected[0]['frame_in_clip']:
                counts['initial'] += 1
            elif r['frame_in_clip'] > detected[-1]['frame_in_clip']:
                counts['terminal'] += 1
            else:
                counts['internal'] += 1
        results.append(dict(clip_id=c[0]['clip_id'], group=c[0]['base_group_id'], relation=c[0]['layout_relation'],
            positive_count=len(positive), first_trigger_id=seeds[0]['id'] if seeds else None,
            first_trigger_truth=seeds[0]['truth'] if seeds else None,
            positives_before_any_trigger=before, reachable_true_suffix=[r['id'] for r in reachable],
            no_release_additional_TP_ceiling=[r['id'] for r in reachable if not r['flags']['original_current']],
            current_detected=bool(detected), added_TP_ids=[r['id'] for r in added], added_TP_kinds=dict(counts)))
    return results


def summary(report):
    return {part: {a: dict(counts=[m['frames'][k] for k in ('TP', 'FP', 'FN')],
        precision=m['frames']['precision'], FPR=m['frames']['FPR'], events=[m['detected_events'], m['event_count']],
        false_segments=m['false_alert_segment_count'], false_sampled_s=m['false_alert_sampled_s'], unknown=m['current_unknown'],
        internal_silent_frames=sum(e['internal_silent_frames'] for e in m['events']),
        terminal_silent_frames=sum(e['terminal_silent_frames'] for e in m['events']),
        exit_carryover_frames=sum(e['postexit_carryover_alert_frames'] or 0 for e in m['events']))
        for a, m in report[part]['arms'].items()} for part in ('Core', 'Boundary')}


def main():
    assert not (OUT / 'independent-audit.json').exists(), 'Do not overwrite completed audit'
    count, seals = 0, []
    for directory, names in ((SOURCE, ('role-seal.json', 'prediction-seal.json')),
                             (CURRENT, ('prediction-seal.json', 'evaluation-seal.json')),
                             (OUT, ('prediction-seal.json', 'evaluation-seal.json'))):
        for name in names:
            seal = read(directory / name)
            assert seal['protocol_sha256'] == sha(directory / 'protocol.json')
            for file, digest in seal['hashes'].items():
                assert sha(directory / file) == digest, file
                count += 1
            seals.append(str((directory / name).relative_to(ROOT)))
    protocol = read(OUT / 'protocol.json')
    for file, digest in protocol['inputs'].items():
        assert sha(ROOT / file) == digest, file
        count += 1
    assert (OUT / 'protocol-before-run.md').read_bytes() == (HERE / 'CONDITIONAL_HOLD_PROTOCOL_20260921.md').read_bytes()
    start = read(OUT / 'evaluation-start.json')
    assert start['prediction_seal_sha256'] == sha(OUT / 'prediction-seal.json')
    assert datetime.fromisoformat(protocol['frozen_at_utc']) < datetime.fromisoformat(start['time_utc'])
    same(dict(evaluation_label_values_read=False, evaluation_label_file_hashed=True, native_values_read=False,
        source_scores_reused=True, training=False, model_inference=False, entry_threshold_changed=False,
        threshold_selected_on='selection only', recursive_continuation=True), read(OUT / 'prediction-receipt.json'))
    entry = read(CURRENT / 'selection.json')['original']['threshold']
    assert entry == 15.769264221191408
    saved_selection = read(OUT / 'selection.json')
    result = read(OUT / 'result.json')
    rows_by_role, groups, reports, gates, changes_by_role, opportunity_summary = {}, {}, {}, {}, {}, {}
    native = {r['id']: r['returned_target_corridor_samples'] for r in read(TRANSFER / 'source-admission.json')['frames']}
    curve, chosen = None, None
    for role in ROLES:
        metadata = read(SOURCE / (role + '-rows.json'))
        scores = read(SOURCE / (role + '-predictions.json'))
        labels = read(SOURCE / 'labels' / (role + '.json'))
        assert len(metadata) == len(scores) == len(labels) == 576
        ids = [r['id'] for r in metadata]
        assert ids == [r['id'] for r in scores] == [r['id'] for r in labels] and len(set(ids)) == 576
        rows = [dict(m, score=s['scores']['original']) for m, s in zip(metadata, scores)]
        groups[role] = {r['base_group_id'] for r in rows}
        assert len(groups[role]) == 8 and len(clips(rows)) == 24
        assert all(g.endswith(('g00', 'g01') if role == 'selection' else ('g02', 'g03')) for g in groups[role])
        assert all(type(y['truth']) is bool for y in labels)
        if role == 'selection':
            thresholds = sorted(set([entry] + [r['score'] for r in rows if r['score'] < entry]))
            curve = [curve_record(rows, labels, threshold, entry) for threshold in thresholds]
            same(curve, read(OUT / 'selection-curve.json'), 'selection_curve')
            feasible = [r for r in curve if r['feasible']]
            chosen = sorted(feasible, key=lambda r: (-r['boundary_gain'], -r['core_gain'], -r['tau_keep']))[0]
            same(dict(tau_on=entry, selected=chosen, candidates=len(curve),
                selection_rule='zero added negative flags; maximize Boundary gain, Core gain, tau_keep'), saved_selection)
        # Only the already frozen selection optimum is applied to evaluation.
        predictions = path_predictions(rows, chosen['tau_keep'], entry)
        same(predictions, read(OUT / (role + '-predictions.json')), role + '.predictions')
        comparators = read(CURRENT / (role + '-predictions.json'))
        assert [p['id'] for p in predictions] == [p['id'] for p in comparators]
        for p, old in zip(predictions, comparators):
            assert all(p['flags'][a] == old['flags'][a] for a in ARMS[:-1])
            assert all(p['current_unknown'][a] == old['current_unknown'][a] for a in ARMS[:-1])
        rows = [dict(r, truth=y['truth'], flags=p['flags'], current_unknown=p['current_unknown'],
                     native_target_corridor_samples=native[r['source_id']]) for r, y, p in zip(rows, labels, predictions)]
        same(rows, read(OUT / (role + '-frame-results.json')), role + '.rows')
        report = independent_metrics.recount(rows)
        same(report, read(OUT / (role + '-metrics.json')), role + '.metrics')
        reports[role] = summary(report)
        same(reports[role], result['metrics'][role], role + '.summary')
        opportunity = opportunities(rows)
        same(opportunity, read(OUT / (role + '-opportunity.json')), role + '.opportunity')
        changes = {key: [] for key in ('added_TP', 'added_FP', 'lost_current', 'held_A_TP_missing')}
        for r in rows:
            f = r['flags']
            item = dict(id=r['id'], group=r['base_group_id'], relation=r['layout_relation'], time_s=r['time_s'],
                        score=r['score'], native=r['native_target_corridor_samples'])
            if f['conditional'] and not f['original_current']:
                changes['added_TP' if r['truth'] else 'added_FP'].append(item)
            if f['original_current'] and not f['conditional']:
                changes['lost_current'].append(item)
            if r['truth'] and f['A_hold'] and not f['conditional']:
                changes['held_A_TP_missing'].append(item)
        same(changes, read(OUT / (role + '-changes.json')), role + '.changes')
        gaining = [r for r in changes['added_TP'] if r['relation'] == 'BOUNDARY']
        gaining_groups = sorted({r['group'] for r in gaining})
        gates[role] = dict(all_current_flags_retained=not changes['lost_current'], no_added_negative_frames=not changes['added_FP'],
            boundary_gain=len(gaining), boundary_gaining_groups=gaining_groups,
            useful=bool(not changes['lost_current'] and not changes['added_FP'] and len(gaining) >= 3 and len(gaining_groups) >= 2))
        assert report['Core']['arms']['conditional']['frames']['TP'] >= report['Core']['arms']['original_current']['frames']['TP']
        same(gates[role], result['gates'][role], role + '.gate')
        opportunity_summary[role] = {}
        for part in ('Core', 'Boundary'):
            subset = [o for o in opportunity if (o['relation'] == 'BOUNDARY') == (part == 'Boundary')]
            opportunity_summary[role][part] = dict(events=len(subset), no_clip_trigger=sum(o['first_trigger_id'] is None for o in subset),
                current_detected=sum(o['current_detected'] for o in subset),
                negative_first_trigger=sum(o['first_trigger_truth'] is False for o in subset),
                positives_before_any_trigger=sum(len(o['positives_before_any_trigger']) for o in subset),
                reachable_true_suffix=sum(len(o['reachable_true_suffix']) for o in subset),
                additional_no_release_ceiling=sum(len(o['no_release_additional_TP_ceiling']) for o in subset),
                clips=subset)
        changes_by_role[role], rows_by_role[role] = changes, rows
    assert not groups['selection'] & groups['evaluation']
    assert not {r['id'] for r in rows_by_role['selection']} & {r['id'] for r in rows_by_role['evaluation']}
    same(saved_selection, result['selection'])
    assert result['candidate_supported'] == gates['evaluation']['useful']
    assert result['status'] == 'COMPLETE' and result['scope'] == 'CONSUMED_DEVELOPMENT'
    assert result['original_test_activated'] is False and result['automatic_successor'] is False
    # Explain the adjacent selection partition only; no evaluation counterfactual.
    position = next(i for i, r in enumerate(curve) if r['tau_keep'] == chosen['tau_keep'])
    lower = curve[position - 1] if position else None
    selected_index = {r['id']: r for r in rows_by_role['selection']}
    frontier = dict(feasible_partitions=sum(r['feasible'] for r in curve), selected=chosen, immediately_lower=lower,
        immediately_lower_added_FP_details=[dict(id=i, score=selected_index[i]['score'], group=selected_index[i]['base_group_id'],
            relation=selected_index[i]['layout_relation'], time_s=selected_index[i]['time_s']) for i in lower['added_FP_ids']] if lower else [])
    old = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
    assert not any((old / file).exists() for file in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'))
    audit = dict(status='PASS', audit_script_sha256=sha(__file__), independent_metric_script_sha256=sha(Path(independent_metrics.__file__)),
        verified_hash_entries=count, verified_seals=seals, prediction_recount='existence of trigger-to-frame >=keep path within each clip',
        selection_partitions=len(curve), selection_frontier=frontier, tau_on=entry, tau_keep=chosen['tau_keep'],
        evaluation_thresholds_applied=1, no_evaluation_threshold_sweep=True, no_fit_values_read=True, original_test_closed=True,
        role_groups={r: sorted(g) for r, g in groups.items()}, metrics=reports, gates=gates,
        changes=changes_by_role, opportunities=opportunity_summary, candidate_supported=gates['evaluation']['useful'],
        limits=['Frame/event/segment metrics reuse the independent audit_current_only implementation with this experiment arm list; production runner/metrics are not imported.',
                'Source FIT entries in seals are hashed only; FIT values and protected-test contents are not parsed.',
                'Static runner review and saved receipts establish the recorded selection/evaluation boundary, not independently observable historical process access.',
                'Never-release suffix is a seed-capacity upper bound ignoring false alerts, not an evaluated candidate.',
                'Consumed same-simulator Development; no model fitting, operating-point retry or runtime modification.'])
    with (OUT / 'independent-audit.json').open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', verified_hash_entries=count, selection_partitions=len(curve),
        selection_frontier=frontier, evaluation_metrics=reports['evaluation'], gates=gates,
        opportunities={r: {p: {k: v for k, v in s.items() if k != 'clips'} for p, s in parts.items()}
                       for r, parts in opportunity_summary.items()}), indent=2))


if __name__ == '__main__':
    main()
