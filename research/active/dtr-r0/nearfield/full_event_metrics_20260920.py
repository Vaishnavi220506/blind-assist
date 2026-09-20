"""Evaluator-only sampled complete-event metrics for the frozen paired pilot.

Durations count samples, not wall time. Current UNKNOWN remains explicit when
a frozen hold produces an alert (the alert still scores TP/FP; UNKNOWN silence
never becomes TN). Core contains complete INSIDE and OUTSIDE clips;
Boundary is reported separately. No thresholds, fitting, or prospective gates.

First in-event delay starts at the first positive sample, independent of an
earlier false alert. Silent positives partition into initial, internal and
terminal silence; a wholly missed event is assigned entirely to initial silence.
Exit is the first negative sample after the event. Carryover counts the alert
prefix after exit only when the final positive sample was alerting. First silence
is relative to exit (zero means immediately silent); absence of silence is right
censored, including when no exit was observed. Later false reactivations count
toward post-exit false samples and new episode starts, never carryover. A leading
post-exit continuation is a false-alert episode but not a new alert episode.
No-truth OUTSIDE clips keep their full negative denominator and event=None.
"""

from collections import defaultdict
from math import isclose, isfinite

ARMS = ('strongest_hold', 'closest_hold')
STRATA = ('layout_relation', 'layer', 'background')


def _ratio(a, b):
    return a / b if b else None


def _runs(flags):
    start = None
    for index in range(len(flags) + 1):
        active = index < len(flags) and flags[index]
        if active and start is None:
            start = index
        elif not active and start is not None:
            yield start, index
            start = None


def _segment(clip, start, end, dt_s):
    return dict(clip_id=clip[0]['clip_id'], start_frame=start, end_frame=end - 1,
                start_time_s=clip[start]['time_s'], end_time_s=clip[end - 1]['time_s'],
                samples=end - start, sampled_s=(end - start) * dt_s)


def _clip_metrics(clip, arm, dt_s):
    alerts = [r['flags'][arm] for r in clip]
    positives = [i for i, row in enumerate(clip) if row['truth']]
    first_clip = next((i for i, alert in enumerate(alerts) if alert), None)
    episode_starts = [i for i, alert in enumerate(alerts) if alert and (i == 0 or not alerts[i - 1])]
    result = dict(clip_id=clip[0]['clip_id'],
                  whole_clip_first_alert_time_s=clip[first_clip]['time_s'] if first_clip is not None else None,
                  alert_episode_count=len(episode_starts),
                  alert_episode_start_times_s=[clip[i]['time_s'] for i in episode_starts],
                  event=None)
    if not positives:
        return result
    start, end = positives[0], positives[-1] + 1
    event_alerts = alerts[start:end]
    detected_indices = [i for i, alert in enumerate(event_alerts) if alert]
    first = detected_indices[0] if detected_indices else None
    last = detected_indices[-1] if detected_indices else None
    # A wholly missed event is initial silence only: the partition never counts
    # its samples a second time as terminal silence or internal interruptions.
    initial = first if first is not None else len(event_alerts)
    terminal = len(event_alerts) - last - 1 if last is not None else 0
    interruptions = [_segment(clip, start + a, start + b, dt_s)
                     for a, b in _runs([not a for a in event_alerts])
                     if first is not None and a > first and b <= last]
    post = alerts[end:]
    first_silent = next((i for i, alert in enumerate(post) if not alert), None)
    prefix = first_silent if first_silent is not None else len(post)
    # An alert newly starting after a silent final positive is a post-exit FP,
    # not a hold tail. Preserve the raw prefix as a separate auditable quantity.
    carryover = prefix if event_alerts[-1] else 0
    post_segments = [_segment(clip, end + a, end + b, dt_s) for a, b in _runs(post)]
    new_starts = [end + i for i, alert in enumerate(post)
                  if alert and not alerts[end + i - 1]]
    entry_time = clip[start]['time_s']
    result['event'] = dict(
        entry_frame=start, last_positive_frame=end - 1, entry_time_s=entry_time,
        positive_frames=len(event_alerts), positive_alert_frames=sum(event_alerts),
        positive_coverage=_ratio(sum(event_alerts), len(event_alerts)),
        detected=first is not None, entry_left_censored=start == 0,
        first_in_event_alert_time_s=clip[start + first]['time_s'] if first is not None else None,
        first_in_event_alert_delay_s=first * dt_s if first is not None else None,
        whole_clip_first_alert_relative_to_entry_s=(clip[first_clip]['time_s'] - entry_time
                                                   if first_clip is not None else None),
        preentry_false_alert_frames=sum(alerts[:start]),
        preexisting_alert_at_entry=bool(start and alerts[start - 1] and alerts[start]),
        in_event_new_alert_episode_count=sum(start <= i < end for i in episode_starts),
        in_event_new_alert_episode_start_times_s=[clip[i]['time_s'] for i in episode_starts if start <= i < end],
        initial_silent_frames=initial, initial_silent_sampled_s=initial * dt_s,
        internal_interruption_count=len(interruptions), internal_interruptions=interruptions,
        internal_silent_frames=sum(s['samples'] for s in interruptions),
        internal_silent_sampled_s=sum(s['samples'] for s in interruptions) * dt_s,
        terminal_silent_frames=terminal, terminal_silent_sampled_s=terminal * dt_s,
        total_silent_frames=len(event_alerts) - sum(event_alerts),
        exit_observed=bool(post), first_negative_exit_time_s=clip[end]['time_s'] if post else None,
        postexit_observed_frames=len(post),
        postexit_leading_alert_frames=prefix if post else None,
        postexit_carryover_alert_frames=carryover if post else None,
        postexit_carryover_sampled_s=carryover * dt_s if post else None,
        first_silent_relative_to_exit_s=first_silent * dt_s if first_silent is not None else None,
        release_right_censored=first_silent is None,
        postexit_false_alert_frames=sum(post), postexit_false_alert_sampled_s=sum(post) * dt_s,
        postexit_false_alert_episode_count=len(post_segments), postexit_false_alert_episodes=post_segments,
        postexit_new_alert_episode_count=len(new_starts),
        postexit_new_alert_episode_start_times_s=[clip[i]['time_s'] for i in new_starts],
    )
    return result


def _group(clips, dt_s):
    rows = [row for clip in clips for row in clip]
    result = dict(frames=len(rows), clips=len(clips), arms={})
    for arm in ARMS:
        counts = dict.fromkeys(('TP', 'FP', 'FN', 'TN', 'abstained_positive',
                                'abstained_negative', 'current_unknown',
                                'current_unknown_positive', 'current_unknown_negative'), 0)
        for row in rows:
            truth, alert, unknown = row['truth'], row['flags'][arm], row['current_unknown'][arm]
            counts['current_unknown'] += unknown
            counts['current_unknown_positive' if truth else 'current_unknown_negative'] += unknown
            if alert:
                counts['TP' if truth else 'FP'] += 1
            elif truth:
                counts['FN'] += 1
            elif not unknown:
                counts['TN'] += 1
            if unknown and not alert:
                counts['abstained_positive' if truth else 'abstained_negative'] += 1
        positive = sum(row['truth'] for row in rows)
        counts.update(frames=len(rows), positive_frames=positive,
                      negative_frames=len(rows) - positive,
                      precision=_ratio(counts['TP'], counts['TP'] + counts['FP']),
                      recall=_ratio(counts['TP'], positive),
                      FPR=_ratio(counts['FP'], len(rows) - positive),
                      F1=_ratio(2 * counts['TP'], 2 * counts['TP'] + counts['FP'] + counts['FN']))
        false_segments = [_segment(clip, a, b, dt_s) for clip in clips
                          for a, b in _runs([not row['truth'] and row['flags'][arm] for row in clip])]
        clip_metrics = [_clip_metrics(clip, arm, dt_s) for clip in clips]
        events = [dict(clip_id=c['clip_id'], **c['event']) for c in clip_metrics if c['event'] is not None]
        result['arms'][arm] = dict(frames=counts, current_unknown=counts['current_unknown'],
            alert_episode_count=sum(c['alert_episode_count'] for c in clip_metrics),
            false_alert_segments=false_segments, false_alert_segment_count=len(false_segments),
            false_alert_sampled_s=counts['FP'] * dt_s, clips=clip_metrics, events=events,
            event_count=len(events), detected_events=sum(event['detected'] for event in events),
            event_recall=_ratio(sum(event['detected'] for event in events), len(events)))
    return result


def evaluate(rows, dt_s=.2):
    """Evaluate arbitrary input order; one contiguous event per non-OUTSIDE clip.

    This evaluator accepts smaller synthetic fixtures. The experiment runner owns
    the 24-clip/24-frame completeness check. Complete clip strata preserve all
    pre-entry and post-exit negatives in each frame metric denominator.
    """
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError('dt_s must be finite and positive')
    rows = list(rows)
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate frame identity')
    by_clip = defaultdict(list)
    for row in rows:
        if type(row['truth']) is not bool:
            raise ValueError('truth must be admitted boolean')
        if row['layout_relation'] not in ('INSIDE', 'BOUNDARY', 'OUTSIDE'):
            raise ValueError('Unknown layout_relation')
        if row['layer'] not in ('BODY', 'HEAD') or row['phase'] not in ('approach', 'dwell', 'depart'):
            raise ValueError('Unknown layer or phase')
        if any(type(row[key][arm]) is not bool for key in ('flags', 'current_unknown') for arm in ARMS):
            raise ValueError('Prediction flags must be boolean')
        if not isfinite(row['time_s']):
            raise ValueError('Timestamp must be finite')
        by_clip[row['clip_id']].append(row)
    clips = []
    for _, clip in sorted(by_clip.items()):
        clip.sort(key=lambda row: row['time_s'])
        tags = tuple(clip[0][key] for key in STRATA)
        for index, row in enumerate(clip):
            if tuple(row[key] for key in STRATA) != tags:
                raise ValueError('Clip stratum changes')
            if not isclose(row['time_s'], index * dt_s, rel_tol=0, abs_tol=1e-8):
                raise ValueError('Timestamps must be contiguous samples starting at zero')
        event_count = len(list(_runs([row['truth'] for row in clip])))
        if event_count != (0 if clip[0]['layout_relation'] == 'OUTSIDE' else 1):
            raise ValueError('Expected one contiguous event per INSIDE/BOUNDARY clip and none for OUTSIDE')
        clips.append(clip)
    return dict(dt_s=dt_s, duration_convention='sample count * dt_s; not wall-clock latency',
        Core_definition='Complete INSIDE and OUTSIDE clips',
        overall=_group(clips, dt_s),
        Core=_group([c for c in clips if c[0]['layout_relation'] != 'BOUNDARY'], dt_s),
        Boundary=_group([c for c in clips if c[0]['layout_relation'] == 'BOUNDARY'], dt_s),
        subgroups={key: {value: _group([c for c in clips if c[0][key] == value], dt_s)
                         for value in sorted({c[0][key] for c in clips})} for key in STRATA})
