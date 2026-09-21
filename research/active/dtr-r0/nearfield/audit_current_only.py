"""Independent standard-library saved-record recount; no fit or threshold search.

Reconstructs predictions from frozen scores and the specified selection statistic.
Does not import the runner or its metrics implementation. Source FIT payloads are
hashed only when named by source seals; no FIT values or protected test are read.
"""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts.local/work/ba-current-only-20260921'
SOURCE = ROOT / 'artifacts.local/work/ba-last-layer-20260921'
TRANSFER = ROOT / 'artifacts.local/work/ba-spatial-complement-transfer-20260921'
OLD = ROOT / 'artifacts.local/work/ba-spatial-bce-20260920'
HEADS = ('original', 'uniform', 'balanced')
ARMS = tuple(h + '_' + s for h in ('A',) + HEADS for s in ('current', 'hold'))
ROLES = ('selection', 'evaluation')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same(actual, saved, context='root'):
    if isinstance(actual, dict):
        assert actual.keys() == saved.keys(), (context, actual.keys(), saved.keys())
        for k, v in actual.items():
            same(v, saved[k], context + '.' + str(k))
    elif isinstance(actual, list):
        assert len(actual) == len(saved), (context, len(actual), len(saved))
        for i, (a, b) in enumerate(zip(actual, saved)):
            same(a, b, context + '[' + str(i) + ']')
    elif isinstance(actual, float):
        assert saved is not None and math.isclose(actual, saved, rel_tol=0, abs_tol=1e-12), (context, actual, saved)
    else:
        assert actual == saved, (context, actual, saved)


def clips(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['clip_id']].append(row)
    return [sorted(grouped[k], key=lambda r: r['frame_in_clip']) for k in sorted(grouped)]


def runs(values):
    starts = [i for i, v in enumerate(values) if v and (i == 0 or not values[i - 1])]
    ends = [i + 1 for i, v in enumerate(values) if v and (i + 1 == len(values) or not values[i + 1])]
    return list(zip(starts, ends))


def segment(c, start, end):
    return dict(clip_id=c[0]['clip_id'], start_frame=start, end_frame=end - 1,
                start_time_s=c[start]['time_s'], end_time_s=c[end - 1]['time_s'],
                samples=end - start, sampled_s=.2 * (end - start))


def ratio(a, b):
    return a / b if b else None


def clip_metric(c, arm):
    alert = [r['flags'][arm] for r in c]
    episodes = runs(alert)
    first_clip = episodes[0][0] if episodes else None
    output = dict(clip_id=c[0]['clip_id'],
                  whole_clip_first_alert_time_s=c[first_clip]['time_s'] if first_clip is not None else None,
                  alert_episode_count=len(episodes),
                  alert_episode_start_times_s=[c[a]['time_s'] for a, _ in episodes], event=None)
    positive = [i for i, r in enumerate(c) if r['truth']]
    if not positive:
        assert c[0]['layout_relation'] == 'OUTSIDE'
        return output
    start, end = min(positive), max(positive) + 1
    assert positive == list(range(start, end))
    found = [i for i in positive if alert[i]]
    first, last = (found[0], found[-1]) if found else (None, None)
    initial = first - start if found else end - start
    terminal = end - last - 1 if found else 0
    silent_runs = runs([start <= i < end and not alert[i] for i in range(len(c))])
    interruptions = [segment(c, a, b) for a, b in silent_runs if found and a > first and b <= last]
    post = alert[end:]
    silent = next((i for i, value in enumerate(post) if not value), None)
    prefix = len(post) if silent is None else silent
    carry = prefix if alert[end - 1] else 0
    new_starts = [a for a, _ in episodes if a >= end]
    entry = c[start]['time_s']
    inside_starts = [a for a, _ in episodes if start <= a < end]
    output['event'] = dict(
        entry_frame=start, last_positive_frame=end - 1, entry_time_s=entry,
        positive_frames=len(positive), positive_alert_frames=len(found),
        positive_coverage=len(found) / len(positive), detected=bool(found), entry_left_censored=start == 0,
        first_in_event_alert_time_s=c[first]['time_s'] if found else None,
        first_in_event_alert_delay_s=(first - start) * .2 if found else None,
        whole_clip_first_alert_relative_to_entry_s=c[first_clip]['time_s'] - entry if first_clip is not None else None,
        preentry_false_alert_frames=sum(alert[:start]),
        preexisting_alert_at_entry=bool(start and alert[start - 1] and alert[start]),
        in_event_new_alert_episode_count=len(inside_starts),
        in_event_new_alert_episode_start_times_s=[c[i]['time_s'] for i in inside_starts],
        initial_silent_frames=initial, initial_silent_sampled_s=initial * .2,
        internal_interruption_count=len(interruptions), internal_interruptions=interruptions,
        internal_silent_frames=sum(s['samples'] for s in interruptions),
        internal_silent_sampled_s=sum(s['samples'] for s in interruptions) * .2,
        terminal_silent_frames=terminal, terminal_silent_sampled_s=terminal * .2,
        total_silent_frames=len(positive) - len(found), exit_observed=bool(post),
        first_negative_exit_time_s=c[end]['time_s'] if post else None,
        postexit_observed_frames=len(post), postexit_leading_alert_frames=prefix if post else None,
        postexit_carryover_alert_frames=carry if post else None,
        postexit_carryover_sampled_s=carry * .2 if post else None,
        first_silent_relative_to_exit_s=silent * .2 if silent is not None else None,
        release_right_censored=silent is None,
        postexit_false_alert_frames=sum(post), postexit_false_alert_sampled_s=sum(post) * .2,
        postexit_false_alert_episode_count=len(runs(post)),
        postexit_false_alert_episodes=[segment(c, end + a, end + b) for a, b in runs(post)],
        postexit_new_alert_episode_count=len(new_starts),
        postexit_new_alert_episode_start_times_s=[c[i]['time_s'] for i in new_starts])
    return output


def group_metric(rows):
    cs = clips(rows)
    result = dict(frames=len(rows), clips=len(cs), arms={})
    for arm in ARMS:
        pos = sum(r['truth'] for r in rows)
        tp = sum(r['truth'] and r['flags'][arm] for r in rows)
        fp = sum(not r['truth'] and r['flags'][arm] for r in rows)
        unknown = sum(r['unknown'] for r in rows)
        f = dict(TP=tp, FP=fp, FN=pos - tp,
                 TN=sum(not r['truth'] and not r['flags'][arm] and not r['unknown'] for r in rows),
                 abstained_positive=sum(r['truth'] and not r['flags'][arm] and r['unknown'] for r in rows),
                 abstained_negative=sum(not r['truth'] and not r['flags'][arm] and r['unknown'] for r in rows),
                 current_unknown=unknown,
                 current_unknown_positive=sum(r['truth'] and r['unknown'] for r in rows),
                 current_unknown_negative=sum(not r['truth'] and r['unknown'] for r in rows),
                 frames=len(rows), positive_frames=pos, negative_frames=len(rows) - pos,
                 precision=ratio(tp, tp + fp), recall=ratio(tp, pos), FPR=ratio(fp, len(rows) - pos),
                 F1=ratio(2 * tp, tp + fp + pos))
        false = [segment(c, a, b) for c in cs for a, b in runs([not r['truth'] and r['flags'][arm] for r in c])]
        cm = [clip_metric(c, arm) for c in cs]
        events = [dict(clip_id=c['clip_id'], **c['event']) for c in cm if c['event'] is not None]
        result['arms'][arm] = dict(frames=f, current_unknown=unknown,
            alert_episode_count=sum(c['alert_episode_count'] for c in cm),
            false_alert_segments=false, false_alert_segment_count=len(false), false_alert_sampled_s=fp * .2,
            clips=cm, events=events, event_count=len(events), detected_events=sum(e['detected'] for e in events),
            event_recall=ratio(sum(e['detected'] for e in events), len(events)))
    return result


def recount(rows):
    return dict(dt_s=.2, duration_convention='sample count * dt_s; not wall-clock latency',
        Core_definition='Complete INSIDE and OUTSIDE clips', overall=group_metric(rows),
        Core=group_metric([r for r in rows if r['layout_relation'] != 'BOUNDARY']),
        Boundary=group_metric([r for r in rows if r['layout_relation'] == 'BOUNDARY']),
        subgroups={key: {v: group_metric([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
                   for key in ('layout_relation', 'layer', 'background')})


def summarize(report):
    return {part: {arm: dict(counts=[m['frames'][k] for k in ('TP', 'FP', 'FN')],
        precision=m['frames']['precision'], FPR=m['frames']['FPR'], events=[m['detected_events'], m['event_count']],
        false_segments=m['false_alert_segment_count'], false_sampled_s=m['false_alert_sampled_s'], unknown=m['current_unknown'])
        for arm, m in report[part]['arms'].items()} for part in ('Core', 'Boundary')}


def main():
    assert not (OUT / 'independent-audit.json').exists(), 'Do not overwrite a completed audit'
    checked = 0
    seal_names = []
    for directory, names in ((SOURCE, ('role-seal.json', 'prediction-seal.json')),
                             (OUT, ('prediction-seal.json', 'evaluation-seal.json'))):
        for name in names:
            seal = read(directory / name)
            assert seal['protocol_sha256'] == sha(directory / 'protocol.json')
            for file, digest in seal['hashes'].items():
                assert sha(directory / file) == digest, file
                checked += 1
            seal_names.append(str((directory / name).relative_to(ROOT)))
    protocol = read(OUT / 'protocol.json')
    for file, digest in protocol['inputs'].items():
        assert sha(ROOT / file) == digest, file
        checked += 1
    assert (OUT / 'protocol-before-run.md').read_bytes() == (HERE / 'CURRENT_ONLY_PROTOCOL_20260921.md').read_bytes()
    start = read(OUT / 'evaluation-start.json')
    assert start['prediction_seal_sha256'] == sha(OUT / 'prediction-seal.json')
    assert datetime.fromisoformat(protocol['frozen_at_utc']) < datetime.fromisoformat(start['time_utc'])
    receipt = read(OUT / 'prediction-receipt.json')
    assert receipt == dict(evaluation_label_values_read=False, evaluation_label_file_hashed=True,
        native_values_read=False, source_scores_reused=True, training=False, model_inference=False,
        thresholds_selected_on='selection only', hold_used_for_selection=False)
    native = {r['id']: r['returned_target_corridor_samples'] for r in read(TRANSFER / 'source-admission.json')['frames']}
    cases = read(TRANSFER / 'spec.json')['cases']
    thresholds = read(OUT / 'selection.json')
    original_selection = read(SOURCE / 'selection.json')
    result = read(OUT / 'result.json')
    all_rows, groups, reports, details, gates, cutoffs = {}, {}, {}, {}, {}, {}
    for role in ROLES:
        metadata = read(SOURCE / (role + '-rows.json'))
        labels = read(SOURCE / 'labels' / (role + '.json'))
        old_pred = read(SOURCE / (role + '-predictions.json'))
        pred = read(OUT / (role + '-predictions.json'))
        rows = read(OUT / (role + '-frame-results.json'))
        assert len(metadata) == len(labels) == len(old_pred) == len(pred) == len(rows) == 576
        ids = [r['id'] for r in metadata]
        assert len(set(ids)) == 576
        for data in (labels, old_pred, pred, rows):
            assert ids == [r['id'] for r in data]
        groups[role] = {r['base_group_id'] for r in metadata}
        assert len(groups[role]) == 8
        assert all(g.endswith(('g00', 'g01') if role == 'selection' else ('g02', 'g03')) for g in groups[role])
        assert set(Counter(r['type_id'] for r in metadata).values()) == {144}
        for m, y, old, p, r in zip(metadata, labels, old_pred, pred, rows):
            assert type(y['truth']) is bool and m['source'] == 'transfer'
            assert all(type(m[k]) is bool for k in ('a', 'unknown'))
            assert old['flags']['A_current'] == m['a']
            assert p['scores'] == old['scores'] and all(math.isfinite(v) for v in p['scores'].values())
            assert p['current_unknown'] == {arm: m['unknown'] for arm in ARMS}
            expected = dict(**m, truth=y['truth'], scores=p['scores'], flags=p['flags'],
                            current_unknown=p['current_unknown'], native_target_corridor_samples=native[m['source_id']])
            same(expected, r, role + '.row.' + m['id'])
        if role == 'selection':
            forbidden = [r for r in rows if not r['truth'] and not r['a']]
            for head in HEADS:
                maximum = max(r['scores'][head] for r in forbidden)
                threshold = math.nextafter(maximum, math.inf)
                expected = dict(threshold=threshold, forbidden_max=maximum,
                    forbidden_ids=[r['id'] for r in forbidden],
                    blockers=[dict(id=r['id'], group=r['base_group_id'], relation=r['layout_relation'],
                                   time_s=r['time_s'], score=r['scores'][head]) for r in forbidden if r['scores'][head] == maximum])
                if head != 'original':
                    expected['prior_current_plus_hold_threshold'] = original_selection[head]['threshold']
                # Threshold identity is exact, not approximate: >= must reject all atomic maximum ties.
                assert threshold == thresholds[head]['threshold']
                same(expected, thresholds[head], 'cutoff.' + head)
                cutoffs[head] = dict(threshold=threshold, forbidden_max=maximum,
                    forbidden_count=len(forbidden), blockers=expected['blockers'])
        cs = clips(rows)
        assert len(cs) == 24
        for c in cs:
            assert [r['frame_in_clip'] for r in c] == list(range(24))
            for i, r in enumerate(c):
                assert math.isclose(r['time_s'], i * .2, rel_tol=0, abs_tol=1e-8)
                expected = {'A_current': r['a']}
                expected.update({h + '_current': r['a'] or r['scores'][h] >= cutoffs[h]['threshold'] for h in HEADS})
                for h in ('A',) + HEADS:
                    expected[h + '_hold'] = expected[h + '_current'] or (i > 0 and c[i - 1]['flags'][h + '_current'])
                assert all(type(v) is bool for v in r['flags'].values())
                same(expected, r['flags'], role + '.flags.' + r['id'])
        for g in groups[role]:
            subset = [r for r in rows if r['base_group_id'] == g]
            assert Counter(r['layout_relation'] for r in subset) == {'INSIDE': 24, 'BOUNDARY': 24, 'OUTSIDE': 24}
        report = recount(rows)
        same(report, read(OUT / (role + '-metrics.json')), role + '.metrics')
        group_report = {g: recount([r for r in rows if r['base_group_id'] == g]) for g in sorted(groups[role])}
        same(group_report, read(OUT / (role + '-group-metrics.json')), role + '.group_metrics')
        same(summarize(report), result['metrics'][role], role + '.summary')
        changes, geometry = {}, {}
        for arm in ARMS:
            counts = Counter()
            for r in rows:
                if r['truth'] or not r['flags'][arm]:
                    continue
                case = cases[r['index']]
                camera = case['camera']
                assert all(camera[k] == 0 for k in ('pitch', 'yaw', 'roll'))
                target = next(o for o in case['objects'] if o['name'] == case['target_name'])
                near = target['center_m'][0] - target['size_m'][0] / 2 - camera['x']
                far = target['center_m'][0] + target['size_m'][0] / 2 - camera['x']
                tag = 'axial_depth_outside_contract' if near > 3 or far < .3 else 'within_depth_outside_corridor'
                counts[r['layout_relation'] + ':' + tag] += 1
            geometry[arm] = dict(counts)
            if arm.startswith('A_'):
                continue
            base = 'A_' + arm.rsplit('_', 1)[1]
            chosen_sets = dict(
                rescued_A_TP=[r for r in rows if r['truth'] and r['flags'][arm] and not r['flags'][base]],
                added_A_FP=[r for r in rows if not r['truth'] and r['flags'][arm] and not r['flags'][base]],
                lost_A_TP=[r for r in rows if r['truth'] and r['flags'][base] and not r['flags'][arm]],
                A_hold_TP_missing=[r for r in rows if r['truth'] and r['flags']['A_hold'] and not r['flags'][arm]])
            changes[arm] = {key: dict(ids=[r['id'] for r in chosen], count=len(chosen),
                native_backed=sum(r['native_target_corridor_samples'] > 0 for r in chosen),
                by_relation=dict(Counter(r['layout_relation'] for r in chosen)),
                by_group=dict(Counter(r['base_group_id'] for r in chosen))) for key, chosen in chosen_sets.items()}
        same(changes, read(OUT / (role + '-changes.json')), role + '.changes')
        same(geometry, read(OUT / (role + '-fp-geometry.json')), role + '.geometry')
        gates[role] = {}
        for head in HEADS:
            arm = head + '_current'
            rescued = [r for r in rows if r['truth'] and r['layout_relation'] == 'BOUNDARY' and r['flags'][arm] and not r['a']]
            retaining = all(not r['a'] or r['flags'][arm] for r in rows)
            no_extra = not any(not r['truth'] and r['flags'][arm] and not r['a'] for r in rows)
            no_extra &= all(report[p]['arms'][arm]['false_alert_segment_count'] <= report[p]['arms']['A_current']['false_alert_segment_count'] for p in ('Core', 'Boundary'))
            gain = len(rescued) / sum(r['truth'] and r['layout_relation'] == 'BOUNDARY' for r in rows)
            gaining = sorted({r['base_group_id'] for r in rescued})
            gates[role][head] = dict(all_A_current_retained=retaining, no_added_FP_or_segments=no_extra,
                boundary_recall_gain=gain, boundary_rescued=len(rescued), gaining_groups=gaining,
                supported=bool(retaining and no_extra and gain >= .1 and len(gaining) >= 4))
        same(gates[role], result['gates'][role], role + '.gates')
        details[role] = dict(counts=summarize(report), changes=changes, false_positive_geometry=geometry,
            event_timing={part: {arm: v['events'] for arm, v in report[part]['arms'].items()} for part in ('Core', 'Boundary')})
        all_rows[role], reports[role] = rows, report
    assert not groups['selection'] & groups['evaluation']
    assert not {r['id'] for r in all_rows['selection']} & {r['id'] for r in all_rows['evaluation']}
    support = {h: all(gates[role][h]['supported'] for role in ROLES) for h in HEADS}
    same(support, result['current_only_support'])
    assert result['status'] == 'COMPLETE' and result['scope'] == 'CONSUMED_DEVELOPMENT'
    assert result['original_test_activated'] is False and result['automatic_successor'] is False
    assert not any((OLD / name).exists() for name in ('test-start.json', 'test-logits.npy', 'test-prediction-seal.json', 'test-metrics.json'))
    audit = dict(status='PASS', scope='Independent consumed-Development saved-record recount',
        audit_script_sha256=sha(__file__), verified_hash_entries=checked, verified_seals=seal_names,
        backend='CPU', backend_reason='TASK_NOT_GPU_SUITABLE', original_test_closed=True,
        no_training=True, no_threshold_search=True, no_fit_values_read=True,
        thresholds_from_selection_only=True, hold_clip_local_nonrecursive=True,
        role_groups={r: sorted(v) for r, v in groups.items()}, cutoff_recount=cutoffs,
        current_only_support=support, gates=gates, roles=details,
        limits=['Source FIT payloads named in seals were hashed only; no FIT values were parsed.',
                'Prediction-before-evaluation ordering reconciles saved receipts and code; historical process access is not independently observable.',
                'Saved frozen scores are authenticated and reused, not re-inferred or retrained.',
                'FP geometry uses complete target axial extent with zero camera rotation; no new geometry labels enter prediction.',
                'Consumed same-simulator Development; no hardware, deployment or safety claim.'])
    with (OUT / 'independent-audit.json').open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', verified_hash_entries=checked, current_only_support=support,
                         evaluation_counts=details['evaluation']['counts'], gates=gates), indent=2))


if __name__ == '__main__':
    main()
