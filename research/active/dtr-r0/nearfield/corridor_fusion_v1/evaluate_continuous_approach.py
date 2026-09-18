"""Offline continuous-approach evaluator; never feeds truth into prediction.

Risk is native corridor OR body-envelope contact. Target ToF support requires
an actual valid public returned slot joined to evaluator-only returned lineage.
Private ray hits, projection visibility, stationary occupancy and future
contact timing remain separate measurements. All time values are nominal
source timestamps, not measured end-to-end latency or transport cadence.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
ARTIFACTS = (REPO / "artifacts.local").resolve()
METHODS = ("astar", "raw_hgb", "astar_hysteresis")
VALID_TOF = ("SIM_VALID", "SIM_MERGED")
CORRIDOR = {"forward": [0.2, 3.6], "lateral": [-0.3, 0.3], "height": [0.4, 2.05]}
CONTACT = {"forward": [0.0, 0.75], "lateral": [-0.22, 0.22], "height": [0.45, 1.95]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_sealed_inputs(capture, predictions_path):
    """Verify completed public predictions before touching evaluator payloads."""
    capture, predictions_path = Path(capture).resolve(), Path(predictions_path).resolve()
    prediction_root = predictions_path.parent

    def load(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def verify_outputs(root, values, required):
        if not isinstance(values, dict) or not required.issubset(values):
            raise ValueError(f"Incomplete sealed outputs in {root}")
        for name, expected in values.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
                raise ValueError(f"Sealed output hash mismatch or unsafe path: {name}")

    # Ordering is deliberate: no capture evaluator payload is read or hashed
    # until a PASS receipt authenticates the already saved public predictions.
    completed = load(prediction_root / "completion.json")
    if completed.get("status") != "PASS" or completed.get("evaluator_read") is not False:
        raise ValueError("Public prediction completion must be PASS with evaluator_read=false")
    verify_outputs(prediction_root, completed.get("outputs"),
                   {predictions_path.name, "summary.json", "input-seal.json"})
    prediction_summary = load(prediction_root / "summary.json")
    prediction_inputs = load(prediction_root / "input-seal.json")
    if (prediction_summary.get("status") != "PASS" or prediction_summary.get("evaluator_read") is not False
            or prediction_summary.get("source_labels_used") is not False
            or prediction_inputs.get("evaluator_read") is not False):
        raise ValueError("Public prediction boundary is not sealed")

    receipt = load(capture / "receipt.json")
    if receipt.get("status") != "PASS":
        raise ValueError("Capture receipt must be PASS before evaluation")
    verify_outputs(capture, receipt.get("hashes"),
                   {"raw.jsonl", "evaluator.jsonl", "provenance.jsonl", "manifest.json"})
    if receipt.get("spec_sha256") != sha(capture / "spec.json"):
        raise ValueError("Capture source specification hash mismatch")
    spec = load(capture / "spec.json")
    if receipt.get("frames") != spec.get("frame_count") or prediction_summary.get("frames") != spec.get("frame_count"):
        raise ValueError("Sealed capture/prediction/source frame counts differ")
    raw_hash = sha(capture / "raw.jsonl")
    matching = [value for name, value in prediction_inputs.get("inputs", {}).items()
                if Path(name).resolve() == (capture / "raw.jsonl").resolve()]
    if matching != [raw_hash]:
        raise ValueError("Predictions were not sealed against this exact raw input")
    return dict(status="PASS", public_prediction_seal_checked_before_evaluator=True,
                prediction_completion_sha256=sha(prediction_root / "completion.json"),
                capture_receipt_sha256=sha(capture / "receipt.json"),
                capture_payloads_verified=len(receipt["hashes"]), prediction_payloads_verified=len(completed["outputs"]))


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def vector3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    result = [finite(v) for v in value]
    return None if any(v is None for v in result) else result


def keyed(rows, key, label):
    if not isinstance(rows, list) or any(key not in row for row in rows):
        raise ValueError(f"{label} must be a list with {key}")
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} in {label}")
    return result


def native_target(row):
    matches = [bound for bound in row.get("native_bounds", []) if bound.get("name") == "target"]
    if len(matches) > 1:
        raise ValueError(f"Duplicate native target: {row.get('id')}")
    return matches[0] if matches else None


def is_target(ray):
    actor = ray.get("actor_id")
    return isinstance(actor, str) and actor.rsplit("/", 1)[-1] == "target"


def target_tof_ranges(public, evaluator):
    """Keep latent target hits separate from valid returned target slots.

    A private hit outside selected returned_lineage never creates public
    support. Missing/invalid attribution for a usable public slot is UNKNOWN.
    A packet with no target return describes sensor support, not clear space.
    """
    native_rows = evaluator.get("zonal_tof_native")
    native = keyed(native_rows, "zone_id", "native ToF zones") if native_rows is not None else {}
    private_ranges = [
        value for zone in native.values() for ray in zone.get("private_rays", [])
        if is_target(ray) and (value := finite(ray.get("range_m"))) is not None and value > 0
    ]
    packet = public.get("tof_packet_received")
    zones = public.get("tof_zones")
    known_slots, unknown_slots, target_slots = [], [], []
    public_slots = 0
    if packet is True and isinstance(zones, list):
        keyed(zones, "zone_id", "public ToF zones")
        for zone in zones:
            zid = zone["zone_id"]
            targets = zone.get("targets", [])
            if not isinstance(targets, list):
                raise ValueError(f"Public targets must be a list: zone {zid}")
            nz = native.get(zid)
            rays = keyed(nz.get("private_rays", []), "subray", "private ToF rays") if nz else {}
            lineage = nz.get("returned_lineage") if nz else None
            linked = keyed(lineage, "target_index", "returned ToF lineage") if lineage is not None else {}
            if nz and "packet_received" in nz and nz["packet_received"] is not packet:
                raise ValueError(f"Public/native packet mismatch: zone {zid}")
            if any(type(i) is not int or not 0 <= i < len(targets) for i in linked):
                raise ValueError(f"Lineage does not address a public slot: zone {zid}")
            for slot, target in enumerate(targets):
                distance = finite(target.get("distance_m"))
                sigma = finite(target.get("range_noise_sigma_m"))
                strength = finite(target.get("signal_strength_proxy"))
                usable = (target.get("status") in VALID_TOF and distance is not None and distance > 0
                          and sigma is not None and sigma >= 0 and strength is not None and strength >= 0)
                if not usable:
                    continue
                public_slots += 1
                item = dict(zone_id=zid, target_index=slot, public_distance_m=distance,
                            public_status=target["status"])
                indices = linked.get(slot, {}).get("hit_indices")
                if not isinstance(indices, list) or not indices:
                    unknown_slots.append(dict(item, reason="MISSING_RETURNED_LINEAGE"))
                    continue
                if len(set(indices)) != len(indices):
                    raise ValueError(f"Duplicate hit index in returned lineage: zone {zid}, slot {slot}")
                hits = [rays.get(index) for index in indices]
                if any(hit is None or "actor_id" not in hit or finite(hit.get("range_m")) is None
                       or finite(hit.get("range_m")) <= 0 for hit in hits):
                    unknown_slots.append(dict(item, reason="UNRESOLVED_RETURNED_HIT"))
                    continue
                target_hits = [hit for hit in hits if is_target(hit)]
                item.update(target_hit_indices=[hit["subray"] for hit in target_hits],
                            target_private_ranges_m=[float(hit["range_m"]) for hit in target_hits],
                            target_attributed=bool(target_hits))
                known_slots.append(item)
                if target_hits:
                    target_slots.append(item)
    if packet is not True:
        support = False if packet is False else None
        status = "NO_PUBLIC_PACKET" if packet is False else "UNKNOWN_PACKET_STATE"
    elif not isinstance(zones, list):
        support, status = None, "UNKNOWN_PUBLIC_ZONES"
    elif target_slots:
        support, status = True, "TARGET_RETURNED_PARTIAL_ATTRIBUTION" if unknown_slots else "TARGET_RETURNED"
    elif unknown_slots:
        support, status = None, "UNKNOWN_RETURNED_ATTRIBUTION"
    else:
        support, status = False, "NO_TARGET_IN_VALID_PUBLIC_RETURNS"
    values = [slot["public_distance_m"] for slot in target_slots]
    return dict(
        packet_received=packet, private_target_hit_count=len(private_ranges) if native_rows is not None else None,
        private_target_hit_min_m=min(private_ranges) if private_ranges else None,
        valid_public_return_slots=public_slots, target_returned=support, status=status,
        valid_returned_target_slot_count=len(target_slots),
        target_public_return_min_m=min(values) if values else None,
        target_slots=target_slots, unknown_returned_slots=unknown_slots,
        attribution_unknown_slot_count=len(unknown_slots),
        absence_is_clear_space=False,
    )


def interval_gap(a_lo, a_hi, b_lo, b_hi):
    return max(0.0, b_lo - a_hi, a_lo - b_hi)


def geometry(bound, body_origin, corridor=CORRIDOR, contact_box=CONTACT):
    if bound is None:
        return None
    center, extent, body = vector3(bound.get("center_m")), vector3(bound.get("extent_m")), vector3(body_origin)
    if center is None or extent is None or body is None or min(extent) < 0:
        return None
    lo = [center[i] - extent[i] for i in range(3)]
    hi = [center[i] + extent[i] for i in range(3)]

    def box(values):
        intervals = [values[key] for key in ("forward", "lateral", "height")]
        if any(len(pair) != 2 or any(finite(v) is None for v in pair) or pair[0] > pair[1] for pair in intervals):
            raise ValueError("Invalid evaluator geometry box")
        return [body[i] + intervals[i][0] for i in range(3)], [body[i] + intervals[i][1] for i in range(3)]

    corridor_lo, corridor_hi = box(corridor)
    contact_lo, contact_hi = box(contact_box)
    in_corridor = all(lo[i] <= corridor_hi[i] and hi[i] >= corridor_lo[i] for i in range(3))
    contact = all(lo[i] <= contact_hi[i] and hi[i] >= contact_lo[i] for i in range(3))
    gap = math.sqrt(sum(interval_gap(lo[i], hi[i], contact_lo[i], contact_hi[i]) ** 2 for i in range(3)))
    return dict(corridor=in_corridor, contact=contact, risk=in_corridor or contact,
                nearest_gap_m=gap, forward_clearance_m=lo[0] - body[0])


def basis(camera):
    p, y, r = (math.radians(float(camera.get(key, 0.0))) for key in ("pitch", "yaw", "roll"))
    cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    return ((cp * cy, cp * sy, sp),
            (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp),
            (-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cr * cp))


def visible_proxy(bound, camera, rig):
    """Conservative native-AABB/frustum projection; does not test RGB occlusion."""
    if bound is None or not isinstance(camera, dict):
        return None
    center, extent = vector3(bound.get("center_m")), vector3(bound.get("extent_m"))
    origin = vector3([camera.get(key) for key in ("x", "y", "z")])
    hfov, width, height = (finite(rig.get(key)) for key in ("hfov_deg", "width", "height"))
    if center is None or extent is None or origin is None or min(extent) < 0:
        return None
    if any(finite(camera.get(key, 0.0)) is None for key in ("pitch", "yaw", "roll")):
        return None
    if hfov is None or not 0 < hfov < 180 or width is None or height is None or min(width, height) <= 0:
        return None
    axes = basis(camera)
    projected = []
    for signs in itertools.product((-1, 1), repeat=3):
        delta = [center[i] + signs[i] * extent[i] - origin[i] for i in range(3)]
        projected.append([sum(axis[i] * delta[i] for i in range(3)) for axis in axes])
    th = math.tan(math.radians(hfov / 2))
    tv = th * height / width
    # A bounding box rejected by any one frustum plane cannot be visible;
    # surviving boxes are only projection opportunities, not observed pixels.
    planes = [lambda p: p[0], lambda p: p[0] * th - p[1], lambda p: p[0] * th + p[1],
              lambda p: p[0] * tv - p[2], lambda p: p[0] * tv + p[2]]
    return not any(all(plane(point) < 0 for point in projected) for plane in planes)


def intervals(flags, times, dt):
    result = []
    start = None
    for i, active in enumerate(list(flags) + [False]):
        if active and start is None:
            start = i
        if not active and start is not None:
            last = i - 1
            result.append(dict(start_index=start, end_index=last, start_s=times[start],
                               last_sample_s=times[last], duration_s=times[last] - times[start] + dt,
                               left_censored=start == 0, right_censored=last == len(times) - 1))
            start = None
    return result


def count_metrics(frames, method):
    counts = {key: 0 for key in ("TP", "FP", "FN", "TN")}
    unknown = 0
    for frame in frames:
        risk, alert = frame["risk_truth"], frame["predictions"][method]["alert"]
        if risk is None:
            unknown += 1
        else:
            counts["TP" if risk and alert else "FN" if risk else "FP" if alert else "TN"] += 1
    tp, fp, fn, tn = (counts[key] for key in ("TP", "FP", "FN", "TN"))
    return dict(counts, frames=len(frames), evaluated_frames=tp + fp + fn + tn, unknown_truth_frames=unknown,
                precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def onset(frames, key, absent_status):
    first = next((i for i, frame in enumerate(frames) if frame[key] is True), None)
    if first is None:
        return dict(time_s=None, status="UNKNOWN_INPUT" if any(f[key] is None for f in frames) else absent_status,
                    frame_id=None, left_censored=False, unknown_before_onset=False)
    unknown_before = any(frame[key] is None for frame in frames[:first])
    return dict(time_s=frames[first]["time_s"], frame_id=frames[first]["id"], left_censored=first == 0,
                unknown_before_onset=unknown_before,
                status="LEFT_CENSORED_AT_FIRST_SAMPLE" if first == 0 else "UNKNOWN_EARLIER_INPUT" if unknown_before else "OBSERVED_TRANSITION")


def release_metric(frames, alerts, process):
    result = dict(release_delay_s=None, release_status="NOT_APPLICABLE", release_start_s=None,
                  release_first_nonrisk_id=None, release_frame_id=None)
    if process != "stop_back":
        return result
    risk_indices = [i for i, f in enumerate(frames) if f["risk_truth"] is True]
    if not risk_indices:
        result["release_status"] = "UNKNOWN_NO_RISK_INTERVAL" if any(f["risk_truth"] is None for f in frames) else "NO_RISK_INTERVAL"
        return result
    last = risk_indices[-1]
    # Use the last contiguous risk interval, never an earlier alert/interval.
    begin = last
    while begin > 0 and frames[begin - 1]["risk_truth"] is True:
        begin -= 1
    if not any(alerts[begin:last + 1]):
        result["release_status"] = "NO_PRIOR_ALERT_IN_LAST_RISK_INTERVAL"
        return result
    if last == len(frames) - 1:
        result["release_status"] = "UNKNOWN_RIGHT_CENSORED_RISK"
        return result
    if any(f["risk_truth"] is None for f in frames[last + 1:]):
        result["release_status"] = "UNKNOWN_EXIT_OR_LATER_RISK"
        return result
    start = last + 1
    result.update(release_start_s=frames[start]["time_s"], release_first_nonrisk_id=frames[start]["id"])
    end = next((i for i in range(start, len(frames)) if not alerts[i]), None)
    if end is None:
        result["release_status"] = "UNKNOWN_RIGHT_CENSORED_ALERT"
    else:
        result.update(release_delay_s=frames[end]["time_s"] - frames[start]["time_s"],
                      release_status="OBSERVED", release_frame_id=frames[end]["id"])
    return result


def metric_for_method(frames, method, process, expected=None, dt=0.1):
    alerts = [frame["predictions"][method]["alert"] for frame in frames]
    times = [frame["time_s"] for frame in frames]
    risk_indices = [i for i, f in enumerate(frames) if f["risk_truth"] is True]
    unknown_risk = any(f["risk_truth"] is None for f in frames)
    expected = True if risk_indices else None if unknown_risk else False
    correct = [i for i in risk_indices if alerts[i]]
    first_alert = next((i for i, active in enumerate(alerts) if active), None)
    first_correct = correct[0] if correct else None
    missed = None if expected is None or (expected and not correct and unknown_risk) else bool(expected and not correct)
    contact = onset(frames, "contact_truth", "NO_CONTACT_OBSERVED")
    first_contact = next((i for i, f in enumerate(frames) if f["contact_truth"] is True), None)
    precontact = next((i for i in correct if first_contact is not None and i < first_contact), None)
    if process == "head_turn":
        lead_status = "NOT_APPLICABLE_STATIONARY_OCCUPANCY"
    elif first_contact is None:
        lead_status = "UNKNOWN_CONTACT_GEOMETRY" if any(f["contact_truth"] is None for f in frames) else "NO_CONTACT_OBSERVED"
    elif contact["left_censored"] or contact["unknown_before_onset"]:
        lead_status = "UNKNOWN_CONTACT_ONSET_CENSORED"
    elif precontact is None and any(f["risk_truth"] is None for f in frames[:first_contact]):
        lead_status = "UNKNOWN_PRECONTACT_RISK"
    elif precontact is not None:
        lead_status = "OBSERVED"
    else:
        lead_status = "NO_CORRECT_ALERT_BEFORE_CONTACT"
    lead = times[first_contact] - times[precontact] if lead_status == "OBSERVED" else None
    false_flags = [a and f["risk_truth"] is False for a, f in zip(alerts, frames)]
    false_segments = intervals(false_flags, times, dt)
    support = [i for i in risk_indices if frames[i]["target_returned"] is True]
    native_fn = [frames[i]["id"] for i in support if not alerts[i]]
    phase_metrics = {}
    for phase in dict.fromkeys(f["phase"] for f in frames):
        subset = [f for f in frames if f["phase"] == phase]
        phase_metrics[phase] = dict(confusion=count_metrics(subset, method),
                                    alert_frames=sum(f["predictions"][method]["alert"] for f in subset),
                                    nominal_sampled_duration_s=len(subset) * dt)
    return dict(
        confusion=count_metrics(frames, method),
        first_alert_s=None if first_alert is None else times[first_alert],
        first_alert_distance_m=None if first_alert is None else frames[first_alert]["forward_clearance_m"],
        first_alert_status="NO_ALERT_OBSERVED" if first_alert is None else "LEFT_CENSORED_AT_FIRST_SAMPLE" if first_alert == 0 else "OBSERVED",
        first_correct_alert_s=None if first_correct is None else times[first_correct],
        first_correct_alert_frame_id=None if first_correct is None else frames[first_correct]["id"],
        first_correct_alert_distance_m=None if first_correct is None else frames[first_correct]["forward_clearance_m"],
        first_correct_alert_status="NO_CORRECT_ALERT_OBSERVED" if first_correct is None else "LEFT_CENSORED_AT_FIRST_SAMPLE" if first_correct == 0 else "OBSERVED",
        first_correct_alert_lead_before_contact_s=lead, contact_warning_status=lead_status,
        contact_warning_lead_authority="NOMINAL_SAMPLED_CONTACT_ONSET_MINUS_FIRST_CORRECT_ALERT; first-sample alert is left-censored",
        correct_alert_before_contact=None if process == "head_turn" or first_contact is None or lead_status.startswith("UNKNOWN") else precontact is not None,
        first_correct_alert_to_contact_signed_s=(times[first_contact] - times[first_correct]
            if first_contact is not None and first_correct is not None and lead_status not in ("NOT_APPLICABLE_STATIONARY_OCCUPANCY", "UNKNOWN_CONTACT_ONSET_CENSORED") else None),
        alert_frames=sum(alerts), alert_duration_s=sum(alerts) * dt,
        alert_segments=intervals(alerts, times, dt), false_alert_segments=false_segments,
        complete_event_kind="STATIONARY_CORRIDOR_OCCUPANCY" if process == "head_turn" else "CORRIDOR_OR_CONTACT",
        complete_event_expected=expected, complete_event_detected=bool(correct) if expected is True and missed is not None else None,
        missed_complete_event=missed,
        native_supported_risk_frames=len(support), native_supported_TP=len(support) - len(native_fn),
        native_supported_FN=len(native_fn), native_supported_FN_ids=native_fn,
        native_supported_recall=(len(support) - len(native_fn)) / len(support) if support else None,
        risk_FN_ids=[frames[i]["id"] for i in risk_indices if not alerts[i]],
        risk_frames_with_unknown_target_support=sum(f["risk_truth"] is True and f["target_returned"] is None for f in frames),
        side_pass_false_alert_segments=len(false_segments) if process == "side_pass" else 0,
        side_pass_false_alert_duration_s=sum(s["duration_s"] for s in false_segments) if process == "side_pass" else 0.0,
        side_pass_false_alert_intervals=false_segments if process == "side_pass" else [],
        phase_metrics=phase_metrics, **release_metric(frames, alerts, process),
    )


def validate_inputs(spec, raw_rows, eval_rows, pred_rows):
    frames = spec.get("frames", [])
    if not frames or len(frames) != spec.get("frame_count"):
        raise ValueError("Source frame count does not match frames")
    maps = [keyed(rows, "id", label) for rows, label in
            ((frames, "spec"), (raw_rows, "raw"), (eval_rows, "evaluator"), (pred_rows, "predictions"))]
    if any(set(mapping) != set(maps[0]) for mapping in maps[1:]):
        raise ValueError("Spec/raw/evaluator/predictions must contain identical unique IDs")
    dt = finite(spec.get("sampling", {}).get("dt_s"))
    if dt is None or dt <= 0:
        raise ValueError("A positive source sampling dt_s is required")
    episodes = keyed(spec.get("episodes", []), "id", "source episodes")
    episode_ids = {frame["episode"] for frame in frames}
    if set(episodes) != episode_ids or len(episodes) != spec.get("episode_count"):
        raise ValueError("Source episode coverage is incomplete")
    nested = [frame for episode in episodes.values() for frame in episode.get("frames", [])]
    if keyed(nested, "id", "nested source frames") != maps[0]:
        raise ValueError("Nested and flattened source frames disagree")
    previous = {}
    metadata = {}
    for frame in frames:
        frame_id, episode, t = frame["id"], frame["episode"], finite(frame.get("time_s"))
        if t is None or (episode in previous and not math.isclose(t - previous[episode], dt, abs_tol=1e-7)):
            raise ValueError(f"Nonmonotonic or irregular source time: {frame_id}")
        previous[episode] = t
        properties = tuple(frame.get(key) for key in ("family", "process", "layout"))
        if episode in metadata and metadata[episode] != properties:
            raise ValueError(f"Sequence metadata changes within {episode}")
        metadata[episode] = properties
        for label, mapping in zip(("raw", "evaluator", "predictions"), maps[1:]):
            row = mapping[frame_id]
            value = finite(row.get("time_s"))
            if row.get("episode_id") != episode or value is None or not math.isclose(value, t, abs_tol=1e-7):
                raise ValueError(f"{label} episode/time mismatch: {frame_id}")
        public, evaluator, prediction = (mapping[frame_id] for mapping in maps[1:])
        if type(public.get("tof_packet_received")) is not bool:
            raise ValueError(f"Invalid public packet flag: {frame_id}")
        if "tof_packet_received" in prediction and prediction["tof_packet_received"] is not public["tof_packet_received"]:
            raise ValueError(f"Prediction packet flag differs from raw: {frame_id}")
        for name in METHODS:
            if type(prediction.get(f"{name}_alert")) is not bool:
                raise ValueError(f"Invalid {name} alert: {frame_id}")
        for name in ("astar", "raw_hgb"):
            score, threshold = finite(prediction.get(f"{name}_score")), finite(prediction.get(f"{name}_threshold"))
            if score is None or not 0 <= score <= 1 or threshold is None or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid {name} probability/threshold: {frame_id}")
            if prediction[f"{name}_alert"] != (score >= threshold):
                raise ValueError(f"Inconsistent saved {name} score/alert: {frame_id}")
        # Missing geometry is reportable UNKNOWN; contradictory source/native
        # identity or timestamps is a corrupt join and is rejected above.
        if evaluator.get("family", frame["family"]) != frame["family"]:
            raise ValueError(f"Evaluator family mismatch: {frame_id}")
    threshold_fields = ("astar_threshold", "raw_hgb_threshold", "astar_hysteresis_on_threshold", "astar_hysteresis_off_threshold")
    thresholds = {}
    for field in threshold_fields:
        values = [finite(row.get(field)) for row in pred_rows]
        if any(value is None for value in values) or len(set(values)) != 1:
            raise ValueError(f"Missing or changing frozen threshold: {field}")
        thresholds[field] = values[0]
    on, off = thresholds["astar_hysteresis_on_threshold"], thresholds["astar_hysteresis_off_threshold"]
    if on != thresholds["astar_threshold"] or not 0 <= off < on <= 1:
        raise ValueError("Invalid frozen hysteresis threshold pair")
    states = {}
    for frame in frames:
        p, episode = maps[3][frame["id"]], frame["episode"]
        active = states.get(episode, False)
        if not active and p["astar_score"] >= on:
            active = True
        elif active and p["astar_score"] < off:
            active = False
        if p["astar_hysteresis_alert"] is not active:
            raise ValueError(f"Saved hysteresis state/reset mismatch: {frame['id']}")
        states[episode] = active
    return maps[1:], dt, thresholds


def motion_phases(frames, process):
    phases = []
    directions = []
    for i, frame in enumerate(frames):
        before = frames[max(0, i - 1)]["body_origin_m"]
        after = frames[min(len(frames) - 1, i + 1)]["body_origin_m"] if i == 0 else frame["body_origin_m"]
        if before is None or after is None:
            directions.append("UNKNOWN_MOTION")
        else:
            dx, dy = after[0] - before[0], after[1] - before[1]
            directions.append("STATIONARY" if math.hypot(dx, dy) < 1e-7 else "BACK" if dx < -1e-7 else "FORWARD")
    for frame, direction in zip(frames, directions):
        phase = "STATIONARY_HEAD_TURN" if process == "head_turn" and direction == "STATIONARY" else direction
        frame["phase"] = phase
    if process == "head_turn" and any(value in ("FORWARD", "BACK") for value in directions):
        raise ValueError("Head-turn source contains wearer translation")
    for phase, group in itertools.groupby(enumerate(frames), key=lambda item: item[1]["phase"]):
        items = list(group)
        phases.append(dict(phase=phase, first_frame_id=items[0][1]["id"], last_frame_id=items[-1][1]["id"],
                           start_s=items[0][1]["time_s"], last_sample_s=items[-1][1]["time_s"], frames=len(items)))
    return phases


def aggregate(sequences, method):
    metrics = [row["methods"][method] for row in sequences]
    all_frames = [frame for row in sequences for frame in row["frames"]]
    expected = [row for row in sequences if row["methods"][method]["complete_event_expected"] is True]
    missed = [row["sequence_id"] for row in sequences if row["methods"][method]["missed_complete_event"] is True]
    unknown_events = [row["sequence_id"] for row in sequences if row["methods"][method]["missed_complete_event"] is None]
    evaluable_expected = [row for row in expected if row["methods"][method]["missed_complete_event"] is not None]
    releases = [row["methods"][method] for row in sequences if row["process"] == "stop_back"]
    observed_release = [m["release_delay_s"] for m in releases if m["release_status"] == "OBSERVED"]
    collision = [row for row in sequences if row["process"] != "head_turn" and row["contact_first_s"] is not None]
    warned = [row for row in collision if row["methods"][method]["correct_alert_before_contact"] is True]
    lead = [row["methods"][method]["first_correct_alert_lead_before_contact_s"] for row in warned]
    support_count = sum(m["native_supported_risk_frames"] for m in metrics)
    support_tp = sum(m["native_supported_TP"] for m in metrics)
    return dict(
        sequence_count=len(sequences), confusion=count_metrics(all_frames, method),
        expected_complete_events=len(expected), complete_events_detected=sum(m["complete_event_detected"] is True for m in metrics),
        missed_complete_events=len(missed), missed_sequence_ids=missed,
        complete_event_miss_rate=len(missed) / len(evaluable_expected) if evaluable_expected else None,
        complete_event_miss_rate_denominator=len(evaluable_expected),
        unknown_complete_events=len(unknown_events), unknown_complete_event_sequence_ids=unknown_events,
        no_alert_sequence_ids=[row["sequence_id"] for row in sequences if row["methods"][method]["alert_frames"] == 0],
        stationary_occupancy_sequences=sum(row["process"] == "head_turn" for row in sequences),
        stationary_occupancy_detected=sum(row["process"] == "head_turn" and row["methods"][method]["complete_event_detected"] is True for row in sequences),
        stationary_occupancy_missed_sequence_ids=[row["sequence_id"] for row in sequences if row["process"] == "head_turn" and row["methods"][method]["missed_complete_event"] is True],
        moving_risk_expected_sequences=sum(row["process"] != "head_turn" and row["methods"][method]["complete_event_expected"] is True for row in sequences),
        moving_risk_detected_sequences=sum(row["process"] != "head_turn" and row["methods"][method]["complete_event_detected"] is True for row in sequences),
        moving_risk_missed_sequence_ids=[row["sequence_id"] for row in sequences if row["process"] != "head_turn" and row["methods"][method]["missed_complete_event"] is True],
        moving_contact_sequences=len(collision), correct_alert_before_contact_sequences=len(warned),
        contact_without_advance_alert_sequence_ids=[row["sequence_id"] for row in collision if row["methods"][method]["correct_alert_before_contact"] is False],
        contact_warning_unknown_sequence_ids=[row["sequence_id"] for row in collision if row["methods"][method]["correct_alert_before_contact"] is None],
        contact_lead_mean_observed_s=sum(lead) / len(lead) if lead else None,
        contact_lead_observed_count=len(lead),
        native_supported_risk_frames=support_count, native_supported_TP=support_tp,
        native_supported_recall=support_tp / support_count if support_count else None,
        native_supported_FN=sum(m["native_supported_FN"] for m in metrics),
        native_supported_FN_ids=[identifier for m in metrics for identifier in m["native_supported_FN_ids"]],
        risk_FN_ids=[identifier for m in metrics for identifier in m["risk_FN_ids"]],
        side_pass_sequences=sum(row["process"] == "side_pass" for row in sequences),
        side_pass_sequences_with_alert=sum(m["side_pass_false_alert_segments"] > 0 for m in metrics),
        side_pass_false_alert_segments=sum(m["side_pass_false_alert_segments"] for m in metrics),
        side_pass_false_alert_duration_s=sum(m["side_pass_false_alert_duration_s"] for m in metrics),
        stop_back_release_status_counts=dict(Counter(m["release_status"] for m in releases)),
        stop_back_release_observed=len(observed_release),
        stop_back_release_delay_mean_observed_s=sum(observed_release) / len(observed_release) if observed_release else None,
        stop_back_release_denominator=len(releases),
    )


def evaluate(spec, raw_rows, eval_rows, pred_rows):
    (raw_by_id, eval_by_id, pred_by_id), dt, thresholds = validate_inputs(spec, raw_rows, eval_rows, pred_rows)
    timelines = {}
    for src in spec["frames"]:
        frame_id = src["id"]
        raw, ev, pred = raw_by_id[frame_id], eval_by_id[frame_id], pred_by_id[frame_id]
        bound = native_target(ev)
        geo = geometry(bound, ev.get("body_origin_m"), spec.get("corridor_m", CORRIDOR), spec.get("contact_proxy", CONTACT))
        tof = target_tof_ranges(raw, ev)
        episode = src["episode"]
        rec = timelines.setdefault(episode, dict(sequence_id=episode, family=src["family"], process=src["process"], layout=src["layout"], frames=[]))
        rec["frames"].append(dict(
            id=frame_id, time_s=float(src["time_s"]), body_origin_m=vector3(ev.get("body_origin_m")),
            target_visible_projection_proxy=visible_proxy(bound, ev.get("camera"), spec["rig"]),
            projection_authority="NATIVE_AABB_FRUSTUM_PROXY_NO_RGB_OCCLUSION_TEST",
            target_returned=tof["target_returned"], target_tof=tof,
            corridor_truth=None if geo is None else geo["corridor"], contact_truth=None if geo is None else geo["contact"],
            risk_truth=None if geo is None else geo["risk"],
            nearest_gap_m=None if geo is None else geo["nearest_gap_m"], forward_clearance_m=None if geo is None else geo["forward_clearance_m"],
            predictions={name: dict(score=pred["raw_hgb_score" if name == "raw_hgb" else "astar_score"],
                                    alert=pred[f"{name}_alert"]) for name in METHODS},
        ))
    sequences = []
    for rec in timelines.values():
        frames = rec["frames"]
        rec["motion_phases"] = motion_phases(frames, rec["process"])
        times = [f["time_s"] for f in frames]
        projection = onset(frames, "target_visible_projection_proxy", "NO_PROJECTION_OPPORTUNITY_OBSERVED")
        returned = onset(frames, "target_returned", "NO_VALID_TARGET_RETURN_OBSERVED")
        contact = onset(frames, "contact_truth", "NO_CONTACT_OBSERVED")
        nearest = min((f for f in frames if f["nearest_gap_m"] is not None), key=lambda f: f["nearest_gap_m"], default=None)
        rec.update(
            target_projection_first=projection, first_valid_target_return=returned, contact_first=contact,
            target_first_visible_s=projection["time_s"], target_first_visible_status=projection["status"],
            target_first_visible_authority="PROJECTION_PROXY_NOT_ACTUAL_RGB_APPEARANCE",
            first_valid_tof_s=returned["time_s"], first_valid_tof_status=returned["status"],
            contact_first_s=contact["time_s"], nearest_pass_s=None if nearest is None else nearest["time_s"],
            nearest_gap_m=None if nearest is None else nearest["nearest_gap_m"],
            event_semantics="STATIONARY_OCCUPANCY_NOT_FUTURE_COLLISION" if rec["process"] == "head_turn" else "MOVING_CORRIDOR_OR_CONTACT",
            risk_intervals=intervals([f["risk_truth"] is True for f in frames], times, dt),
            contact_intervals=intervals([f["contact_truth"] is True for f in frames], times, dt),
            coverage=dict(frames=len(frames), geometry_unknown_frames=sum(f["risk_truth"] is None for f in frames),
                          projection_unknown_frames=sum(f["target_visible_projection_proxy"] is None for f in frames),
                          target_return_unknown_frames=sum(f["target_returned"] is None for f in frames),
                          target_return_present_frames=sum(f["target_returned"] is True for f in frames),
                          no_public_packet_frames=sum(f["target_tof"]["packet_received"] is False for f in frames),
                          private_target_hit_frames=sum((f["target_tof"]["private_target_hit_count"] or 0) > 0 for f in frames)),
            methods={name: metric_for_method(frames, name, rec["process"], dt=dt) for name in METHODS},
        )
        rec["expected_complete_event"] = rec["methods"][METHODS[0]]["complete_event_expected"]
        sequences.append(rec)
    by_family_process = {}
    for family, process in sorted({(r["family"], r["process"]) for r in sequences}):
        rows = [r for r in sequences if (r["family"], r["process"]) == (family, process)]
        by_family_process[f"{family}/{process}"] = {method: aggregate(rows, method) for method in METHODS}
    summary = dict(
        schema="DTR_CONTINUOUS_APPROACH_EVALUATION_V2", status="PASS", sequences=len(sequences), frames=len(raw_rows),
        sequence_ids=[r["sequence_id"] for r in sequences], methods={method: aggregate(sequences, method) for method in METHODS},
        by_family_process=by_family_process, frozen_thresholds_verified=thresholds,
        coverage={key: sum(row["coverage"][key] for row in sequences) for key in sequences[0]["coverage"]},
        integrity=dict(unique_ids=True, raw_evaluator_prediction_spec_join=True, nested_source_parity=True,
                       episode_time_parity=True, constant_thresholds=True, score_alert_consistency=True, hysteresis_reset_verified=True),
        authority="CONTROLLED_UE_DEVELOPMENT_EVALUATOR_ONLY_NO_HARDWARE_OR_SAFETY_CLAIM",
        risk_definition="native corridor OR native body-envelope contact",
        target_support_definition="received packet + finite valid public slot + target in its returned_lineage hit_indices",
        visibility_definition="native AABB projection proxy; no RGB appearance/occlusion claim; onset may be left-censored",
        nominal_dt_s=dt, timing_authority="SOURCE_TIMESTAMPS_NOT_MEASURED_TRANSPORT_OR_LATENCY",
        unknown_policy="Unknown geometry/lineage and censored event timing are explicit; no observations does not mean clear space",
    )
    return sequences, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture, predictions_path, out = args.capture.resolve(), args.predictions.resolve(), args.output.resolve()
    if not out.is_relative_to(ARTIFACTS) or out == ARTIFACTS or out.exists():
        raise ValueError("Fresh evaluation output must be strictly under artifacts.local")
    seal_verification = verify_sealed_inputs(capture, predictions_path)
    spec = json.loads((capture / "spec.json").read_text(encoding="utf-8-sig"))
    sequences, summary = evaluate(spec, read_jsonl(capture / "raw.jsonl"), read_jsonl(capture / "evaluator.jsonl"), read_jsonl(predictions_path))
    out.mkdir(parents=True)
    with (out / "timelines.jsonl").open("x", encoding="utf-8") as handle:
        for row in sequences:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    summary["sealed_input_verification"] = seal_verification
    write(out / "summary.json", summary)
    paths = (capture / "spec.json", capture / "raw.jsonl", capture / "evaluator.jsonl", predictions_path)
    write(out / "input-seal.json", dict(inputs={str(path): sha(path) for path in paths}, code_sha256=sha(HERE), evaluator_read=True, predictor_not_rerun=True))
    write(out / "completion.json", dict(status="PASS", outputs={name: sha(out / name) for name in ("timelines.jsonl", "summary.json", "input-seal.json")}, evaluator_read=True))
    print(json.dumps(dict(status="PASS", sequences=len(sequences), frames=summary["frames"], methods=summary["methods"], output=str(out)), indent=2))


if __name__ == "__main__":
    main()
