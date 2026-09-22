"""CPU metrics for controlled, sampled camera-forward trajectories; no safety claim."""

from collections import defaultdict
from math import isclose, isfinite

ARMS = ("raw_tof", "nfo", "depthpro_global")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _frame_counts(rows, arm):
    out = dict.fromkeys(("frames", "TP", "FP", "FN", "TN", "prediction_unknown",
                         "prediction_unknown_positive", "prediction_unknown_negative",
                         "abstained_positive", "abstained_negative", "ambiguous",
                         "ambiguous_alerts"), 0)
    for row in rows:
        truth, pred = row["truth"], row["predictions"][arm]
        out["frames"] += 1
        out["prediction_unknown"] += pred["unknown"]
        out["prediction_unknown_positive" if truth else "prediction_unknown_negative"] += pred["unknown"]
        out["ambiguous"] += pred["ambiguous"]
        out["ambiguous_alerts"] += pred["ambiguous"] and pred["alert"]
        if pred["alert"]:
            out["TP" if truth else "FP"] += 1
        elif truth:
            out["FN"] += 1
        elif not pred["unknown"]:
            out["TN"] += 1
        if pred["unknown"] and not pred["alert"]:
            out["abstained_positive" if truth else "abstained_negative"] += 1
    out["recall"] = _ratio(out["TP"], out["TP"] + out["FN"])
    out["precision"] = _ratio(out["TP"], out["TP"] + out["FP"])
    negatives = out["FP"] + out["TN"] + out["abstained_negative"]
    out["false_alert_rate_known_negative"] = _ratio(out["FP"], negatives)
    return out


def _runs(rows, predicate):
    start = None
    for index in range(len(rows) + 1):
        active = index < len(rows) and predicate(rows[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            yield start, index
            start = None


def evaluate_rows(rows, dt_s=0.2, arms=ARMS):
    """Return JSON-ready counts; UNKNOWN truth splits runs and never scores negative.

    Clips contain one intended obstacle and contiguous frame indices starting at 0.
    Sample timestamps must equal frame index * dt_s. Duration = frame count * dt_s,
    an inclusive sampled duration, not measured runtime or wall-clock latency.
    Boundary truth is supplied by the evaluator; contact is never relabelled here.
    """
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    rows, arms = list(rows), tuple(arms)
    clips = defaultdict(list)
    for row in rows:
        if row["truth"] is not None and type(row["truth"]) is not bool:
            raise ValueError("truth must be True, False or None")
        if type(row["boundary"]) is not bool:
            raise ValueError("boundary must be bool")
        for arm in arms:
            if any(type(row["predictions"][arm][key]) is not bool
                   for key in ("alert", "unknown", "ambiguous")):
                raise ValueError("prediction flags must be bool")
        clips[row["clip_id"]].append(row)
    for clip in clips.values():
        clip.sort(key=lambda row: row["frame_in_clip"])
        for index, row in enumerate(clip):
            if type(row["frame_in_clip"]) is not int or row["frame_in_clip"] != index:
                raise ValueError("clip frame indices must be contiguous from zero")
            if not isfinite(row["time_s"]) or not isclose(row["time_s"], index * dt_s, abs_tol=1e-8):
                raise ValueError("timestamp does not match sampled frame index")
    known = [row for row in rows if row["truth"] is not None]
    out = {"scope": "controlled_development_sampled_trajectory", "dt_s": dt_s,
           "duration_convention": "inclusive sampled frames * dt_s; not wall latency",
           "frames": len(rows), "clips": len(clips), "known_truth_frames": len(known),
           "unknown_truth_frames": len(rows) - len(known),
           "truth_coverage": _ratio(len(known), len(rows)),
           "boundary_frames": sum(row["boundary"] for row in rows), "arms": {}}
    for arm in arms:
        strata = {"all_known": known, "interior": [r for r in known if not r["boundary"]],
                  "boundary": [r for r in known if r["boundary"]]}
        metrics = {"frames": {name: _frame_counts(group, arm) for name, group in strata.items()},
                   "prediction_unknown_frames": sum(r["predictions"][arm]["unknown"] for r in rows),
                   "alerts_on_unknown_truth": sum(r["truth"] is None and r["predictions"][arm]["alert"] for r in rows),
                   "events": [], "false_alert_segments": [], "interior_false_alert_segments": []}
        for clip_id, clip in sorted(clips.items()):
            for start, end in _runs(clip, lambda r: r["truth"] is True):
                event = clip[start:end]
                first = next((r for r in event if r["predictions"][arm]["alert"]), None)
                metrics["events"].append({"clip_id": clip_id, "start_frame": start,
                    "end_frame": end - 1, "entry_time_s": event[0]["time_s"],
                    "sampled_duration_s": len(event) * dt_s, "detected": first is not None,
                    "first_alert_time_s": first["time_s"] if first else None,
                    "first_alert_relative_to_entry_s": first["time_s"] - event[0]["time_s"] if first else None,
                    "left_censored": start == 0,
                    "preceding_truth_unknown": start > 0 and clip[start - 1]["truth"] is None,
                    "preexisting_alert_at_entry": (clip[start - 1]["predictions"][arm]["alert"]
                        and event[0]["predictions"][arm]["alert"]) if start else None,
                    "abstained_frames": sum(r["predictions"][arm]["unknown"] and not r["predictions"][arm]["alert"] for r in event),
                    "interior_frames": sum(not r["boundary"] for r in event),
                    "interior_detected": any(not r["boundary"] and r["predictions"][arm]["alert"] for r in event),
                    "boundary_frames": sum(r["boundary"] for r in event)})
            for start, end in _runs(clip, lambda r: r["truth"] is False and r["predictions"][arm]["alert"]):
                metrics["false_alert_segments"].append({"clip_id": clip_id,
                    "start_frame": start, "end_frame": end - 1,
                    "start_time_s": clip[start]["time_s"], "end_time_s": clip[end - 1]["time_s"],
                    "sampled_duration_s": (end - start) * dt_s, "left_censored": start == 0})
            for start, end in _runs(clip, lambda r: r["truth"] is False and not r["boundary"] and r["predictions"][arm]["alert"]):
                metrics["interior_false_alert_segments"].append({"clip_id": clip_id,
                    "start_frame": start, "end_frame": end - 1,
                    "sampled_duration_s": (end - start) * dt_s})
        events = metrics["events"]
        metrics["event_count"] = len(events)
        metrics["detected_events"] = sum(event["detected"] for event in events)
        metrics["event_recall"] = _ratio(metrics["detected_events"], len(events))
        metrics["left_censored_events"] = sum(event["left_censored"] for event in events)
        metrics["interior_event_count"] = sum(event["interior_frames"] > 0 for event in events)
        metrics["interior_detected_events"] = sum(event["interior_detected"] for event in events)
        metrics["interior_event_recall"] = _ratio(metrics["interior_detected_events"], metrics["interior_event_count"])
        metrics["false_alert_segment_count"] = len(metrics["false_alert_segments"])
        metrics["false_alert_sampled_duration_s"] = sum(s["end_frame"] - s["start_frame"] + 1 for s in metrics["false_alert_segments"]) * dt_s
        metrics["interior_false_alert_segment_count"] = len(metrics["interior_false_alert_segments"])
        metrics["interior_false_alert_sampled_duration_s"] = sum(s["end_frame"] - s["start_frame"] + 1 for s in metrics["interior_false_alert_segments"]) * dt_s
        out["arms"][arm] = metrics
    return out
