"""Independent post-seal audit; no fitting, threshold selection, or result edits.

Only stdlib and NumPy are used. Evaluation arrays and captured geometry are
opened after the prediction/checkpoint/selection/source hash gate has passed.
World-coordinate box intersection and sampled-run metrics are recomputed here;
the label builder and production metric implementation are never imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

LEARNED = ("classifier", "occupancy")
ARMS = ("A_current", "A_hold", *LEARNED)
EDGES = np.asarray([.3, .75, 1.25, 1.75, 2.25, 2.75, 3.], np.float32)
QUERIES = np.asarray([[-.6, 0., .42, .9], [-.3, .3, .42, .9], [0., .6, .42, .9],
                      [-.6, 0., -.2, .42], [-.3, .3, -.2, .42], [0., .6, -.2, .42]], np.float32)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] / "artifacts.local/evidence/ba-query-occupancy-20260922"


class AuditFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def stage(root, name):
    root = Path(root)
    return root.with_name(root.name + "-" + name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_matches(path, expected):
    require(isinstance(expected, str) and len(expected) == 64, f"Missing SHA256: {path}")
    require(digest(path) == expected, f"SHA256 mismatch: {path}")


def verify_prediction_gate(root, source_dir=HERE):
    """This function must complete before any evaluation truth is read."""
    fit, pred = stage(root, "fit"), stage(root, "predictions")
    seal = read_json(pred / "prediction-seal.json")
    require(seal.get("status") == "PASS" and seal.get("held_labels_opened") is False,
            "Prediction seal is absent, incomplete, or declares prior held access")
    hash_matches(fit / "selection.json", seal["selection_sha256"])
    hash_matches(fit / "learning-source-seal.json", seal["source_seal_sha256"])
    selection = read_json(fit / "selection.json")
    source = read_json(fit / "learning-source-seal.json")
    require(source.get("held_labels_opened") is False, "Source seal declares held access")
    required_code = {f"query_occupancy_{part}.py" for part in ("data", "learning", "model", "evaluate")}
    required_code.update(("run_query_occupancy.py", "ba_camera_corridor_metrics.py", "../../../../tools/research_backend.py"))
    require(required_code <= set(source["code"]), "Incomplete learning/evaluator source seal")
    for name, expected in source["code"].items():
        require(Path(name).name == name or name == "../../../../tools/research_backend.py",
                "Unexpected source path in learning seal")
        hash_matches(Path(source_dir) / name, expected)
    hash_matches(Path(root) / "plan/learning-protocol.md", source["protocol_sha256"])
    require(set(seal["models"]) == set(LEARNED), "Prediction seal must contain exactly two learned arms")
    for arm in LEARNED:
        model = seal["models"][arm]
        hash_matches(pred / f"{arm}.npz", model["predictions_sha256"])
        hash_matches(fit / f"{arm}.pt", model["checkpoint_sha256"])
        require(model["checkpoint_sha256"] == selection[arm]["checkpoint_sha256"],
                f"Selected checkpoint mismatch: {arm}")
        require(math.isfinite(float(selection[arm]["selection"]["threshold"])),
                f"Nonfinite fixed dev threshold: {arm}")
    prepared = stage(root, "prepared")
    hash_matches(prepared / "materialization.json", source["materialization_sha256"])
    required_inputs = {"observations/rgb.npy", "observations/tof.npy", "observations/identities.json",
                       "labels/train.npz", "labels/dev.npz"}
    require(set(source["used_inputs"]) == required_inputs, "Learning source seal input boundary differs")
    for name, expected in source["used_inputs"].items():
        hash_matches(prepared / name, expected)
    return seal, selection, source


def world_box_labels(objects, camera, queries=QUERIES, edges=EDGES):
    """Intersect world AABBs with world query prisms, including all objects.

    In this level camera pose, world X is optical Z, world Y is camera X,
    and camera Y is minus world Z. Closed box contact counts as occupancy.
    Distances are rounded to the stored float32 representation before binning.
    """
    require(all(abs(float(camera[k])) <= 1e-8 for k in ("pitch", "yaw", "roll")),
            "World-prism audit requires the frozen level camera")
    bounds = []
    for obj in objects:
        center = np.asarray(obj["render_bounds_center_m"], np.float64)
        half = np.asarray(obj["render_bounds_extent_m"], np.float64)
        require(center.shape == half.shape == (3,) and np.isfinite(center).all()
                and np.isfinite(half).all() and (half > 0).all(), "Invalid declared render AABB")
        bounds.append((center - half, center + half))
    classes, distances = [], []
    for xmin, xmax, ymin, ymax in queries:
        query_low = np.asarray([camera["x"] + float(edges[0]), camera["y"] + float(xmin),
                                camera["z"] - float(ymax)])
        query_high = np.asarray([camera["x"] + float(edges[-1]), camera["y"] + float(xmax),
                                 camera["z"] - float(ymin)])
        entry_depths = []
        for box_low, box_high in bounds:
            entry, leave = np.maximum(box_low, query_low), np.minimum(box_high, query_high)
            if np.all(entry <= leave):
                entry_depths.append(entry[0] - camera["x"])
        if entry_depths:
            distance = np.float32(min(entry_depths))
            # Explicit upper-bound tests, independent of searchsorted labels.
            bin_index = next(i for i, upper in enumerate(edges[1:]) if distance <= upper)
        else:
            distance, bin_index = np.float32(np.nan), len(edges) - 1
        distances.append(distance)
        classes.append(bin_index)
    return np.asarray(classes, np.int64), np.asarray(distances, np.float32)


def ratio(num, den):
    return num / den if den else None


def contiguous_spans(flags):
    """Boolean transition indices; each returned span is half-open."""
    padded = np.r_[False, np.asarray(flags, bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def frame_counts(rows, arm):
    y = np.asarray([r["truth"] for r in rows], bool)
    alert, unknown, ambiguous = (np.asarray([r["predictions"][arm][key] for r in rows], bool)
                                  for key in ("alert", "unknown", "ambiguous"))
    count = lambda x: int(np.count_nonzero(x))
    result = dict(frames=len(rows), TP=count(y & alert), FP=count(~y & alert),
        FN=count(y & ~alert), TN=count(~y & ~alert & ~unknown),
        prediction_unknown=count(unknown), prediction_unknown_positive=count(unknown & y),
        prediction_unknown_negative=count(unknown & ~y), abstained_positive=count(unknown & ~alert & y),
        abstained_negative=count(unknown & ~alert & ~y), ambiguous=count(ambiguous),
        ambiguous_alerts=count(ambiguous & alert))
    result.update(recall=ratio(result["TP"], count(y)), precision=ratio(result["TP"], count(alert)),
                  false_alert_rate_known_negative=ratio(result["FP"], count(~y)))
    return result


def recompute_metrics(rows, dt=.2):
    clips = defaultdict(list)
    for row in rows:
        require(row["truth"] is None or type(row["truth"]) is bool, "Invalid truth state")
        clips[row["clip_id"]].append(row)
    for clip in clips.values():
        clip.sort(key=lambda r: r["frame_in_clip"])
        require([r["frame_in_clip"] for r in clip] == list(range(len(clip))), "Noncontiguous clip")
        require(all(math.isclose(r["time_s"], i * dt, abs_tol=1e-8) for i, r in enumerate(clip)),
                "Sample time differs from frozen posed trajectory")
    known = [r for r in rows if r["truth"] is not None]
    result = dict(scope="controlled_development_sampled_trajectory", dt_s=dt,
        duration_convention="inclusive sampled frames * dt_s; not wall latency",
        frames=len(rows), clips=len(clips), known_truth_frames=len(known),
        unknown_truth_frames=len(rows) - len(known), truth_coverage=ratio(len(known), len(rows)),
        boundary_frames=sum(r["boundary"] for r in rows), arms={})
    for arm in ARMS:
        metrics = dict(frames={name: frame_counts(subset, arm) for name, subset in (
            ("all_known", known), ("interior", [r for r in known if not r["boundary"]]),
            ("boundary", [r for r in known if r["boundary"]]))},
            prediction_unknown_frames=sum(r["predictions"][arm]["unknown"] for r in rows),
            alerts_on_unknown_truth=sum(r["truth"] is None and r["predictions"][arm]["alert"] for r in rows),
            events=[], false_alert_segments=[], interior_false_alert_segments=[])
        for clip_id in sorted(clips):
            clip = clips[clip_id]
            alerts = np.asarray([r["predictions"][arm]["alert"] for r in clip], bool)
            unknown = np.asarray([r["predictions"][arm]["unknown"] for r in clip], bool)
            interior = np.asarray([not r["boundary"] for r in clip], bool)
            positive = np.asarray([r["truth"] is True for r in clip], bool)
            negative = np.asarray([r["truth"] is False for r in clip], bool)
            for start, end in contiguous_spans(positive):
                hits = np.flatnonzero(alerts[start:end])
                first_time = clip[start + int(hits[0])]["time_s"] if len(hits) else None
                metrics["events"].append(dict(clip_id=clip_id, start_frame=start, end_frame=end - 1,
                    entry_time_s=clip[start]["time_s"], sampled_duration_s=(end - start) * dt,
                    detected=bool(len(hits)), first_alert_time_s=first_time,
                    first_alert_relative_to_entry_s=None if first_time is None else first_time - clip[start]["time_s"],
                    left_censored=start == 0, preceding_truth_unknown=start > 0 and clip[start - 1]["truth"] is None,
                    preexisting_alert_at_entry=bool(alerts[start - 1] and alerts[start]) if start else None,
                    abstained_frames=int(np.count_nonzero(unknown[start:end] & ~alerts[start:end])),
                    interior_frames=int(interior[start:end].sum()),
                    interior_detected=bool(np.any(interior[start:end] & alerts[start:end])),
                    boundary_frames=int((~interior[start:end]).sum())))
            for inner in (False, True):
                active = negative & alerts & (interior if inner else True)
                key = "interior_false_alert_segments" if inner else "false_alert_segments"
                for start, end in contiguous_spans(active):
                    segment = dict(clip_id=clip_id, start_frame=start, end_frame=end - 1,
                                   sampled_duration_s=(end - start) * dt)
                    if not inner:
                        segment.update(start_time_s=clip[start]["time_s"], end_time_s=clip[end - 1]["time_s"],
                                       left_censored=start == 0)
                    metrics[key].append(segment)
        events = metrics["events"]
        metrics.update(event_count=len(events), detected_events=sum(e["detected"] for e in events),
            left_censored_events=sum(e["left_censored"] for e in events),
            interior_event_count=sum(e["interior_frames"] > 0 for e in events),
            interior_detected_events=sum(e["interior_detected"] for e in events))
        metrics["event_recall"] = ratio(metrics["detected_events"], metrics["event_count"])
        metrics["interior_event_recall"] = ratio(metrics["interior_detected_events"], metrics["interior_event_count"])
        for prefix in ("", "interior_"):
            segments = metrics[prefix + "false_alert_segments"]
            metrics[prefix + "false_alert_segment_count"] = len(segments)
            metrics[prefix + "false_alert_sampled_duration_s"] = sum(s["end_frame"] - s["start_frame"] + 1 for s in segments) * dt
        result["arms"][arm] = metrics
    return result


def rebuild_rows(identities, labels, predictions, selection):
    rows, previous = [], {}
    for k, item in enumerate(identities):
        base = item["baseline"]
        known = bool(labels["valid"][k, 1] and labels["valid"][k, 4])
        truth = bool(labels["classes"][k, 1] < 6 or labels["classes"][k, 4] < 6) if known else None
        flags = {"A_current": bool(base["alert"]),
                 "A_hold": bool(base["alert"] or previous.get(item["clip_id"], False))}
        previous[item["clip_id"]] = bool(base["alert"])
        for arm in LEARNED:
            # Python floats preserve a nextafter(max, +inf) no-alert cutoff.
            score = max(float(predictions[arm]["probability"][k, j]) for j in (1, 4))
            flags[arm] = score >= float(selection[arm]["selection"]["threshold"])
        row = {key: item[key] for key in ("id", "clip_id", "frame_in_clip", "time_s", "base_group_id",
                                         "type_id", "layer", "layout_relation")}
        row.update(truth=truth, boundary=item["layout_relation"] == "BOUNDARY", predictions={
            arm: dict(alert=flag, unknown=bool(base["unknown"]), ambiguous=bool(base["unknown"] and flag))
            for arm, flag in flags.items()})
        rows.append(row)
    return rows


def same_tree(actual, expected, location="value", tolerance=1e-9):
    """Compare every field, with small float tolerance but exact types for flags."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Keys differ: {location}")
        for key in expected:
            same_tree(actual[key], expected[key], location + "." + key, tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"List differs: {location}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same_tree(a, b, f"{location}[{i}]", tolerance)
    elif type(expected) is float:
        require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
                f"Number differs: {location}: {actual!r} != {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"Value differs: {location}: {actual!r} != {expected!r}")


def prediction_checks(predictions, indices):
    n = len(indices)
    for arm, pred in predictions.items():
        require(np.array_equal(pred["indices"], indices), f"Prediction index/order mismatch: {arm}")
        p = pred["probability"]
        require(p.shape == (n, 6) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1.000002).all(),
                f"Invalid query probabilities: {arm}")
    occ = predictions["occupancy"]
    distribution = occ["distribution"]
    require(distribution.shape == (n, 6, 7) and np.isfinite(distribution).all() and (distribution >= 0).all(),
            "Invalid distance distribution")
    require(np.allclose(distribution.sum(-1), 1., atol=2e-6, rtol=0), "Distance distribution not normalized")
    cdf = np.cumsum(distribution[..., :6], axis=-1)
    require((np.diff(cdf, axis=-1) >= 0).all(), "Query CDF is not monotone")
    require(np.allclose(cdf[..., -1], occ["probability"], atol=2e-6, rtol=0), "CDF and occupancy probability disagree")
    mask = occ["mask"]
    require(mask.shape == (n, 6, 45, 80) and np.isfinite(mask).all() and (mask >= 0).all() and (mask <= 1).all(),
            "Invalid predicted visible mask")
    return dict(frames=n, queries=n * 6, normalized=True, monotone_cdf=True, cdf_probability_consistent=True)


def audit(root, source_dir=HERE):
    root = Path(root)
    # Do not move ANY held read, including checksums, before this gate.
    seal, selection, source = verify_prediction_gate(root, source_dir)
    cap, prepared, evaluated = (stage(root, s) for s in ("capture", "prepared", "evaluated"))
    report = read_json(evaluated / "metrics.json")
    require(report.get("status") == "PASS", "Evaluation is not complete")
    hash_matches(stage(root, "predictions") / "prediction-seal.json", report["held_predictions_sha256"])
    same_tree(report["selection"], selection, "evaluation.selection")
    material = read_json(prepared / "materialization.json")
    require(material["status"] == "PASS", "Materialization failed")
    hash_matches(root / "plan/spec.json", material["source_spec_sha256"])
    hashes = {key.replace("\\", "/"): value for key, value in material["hashes"].items()}
    required = ["observations/identities.json", "labels/visibility-audit.json",
                "labels/train.npz", "labels/dev.npz", "labels/evaluation.npz"]
    for relative in required:
        hash_matches(prepared / relative, hashes[relative])
    require(np.array_equal(material["queries"], QUERIES), "Frozen query bounds differ")
    require(np.array_equal(material["bin_edges_m"], EDGES), "Frozen distance bins differ")
    with np.load(prepared / "labels/evaluation.npz", allow_pickle=False) as archive:
        labels = dict(archive)
    all_ids = read_json(prepared / "observations/identities.json")
    spec = read_json(root / "plan/spec.json")
    geometry = read_json(cap / "evaluator/geometry.json")
    visibility = read_json(prepared / "labels/visibility-audit.json")
    require(len(all_ids) == len(spec["cases"]) == len(geometry) == len(visibility) == spec["frames"],
            "Declared frames were dropped or duplicated")
    groups, counts = defaultdict(set), Counter()
    for i, (case, item) in enumerate(zip(spec["cases"], all_ids)):
        require(item["index"] == i and item["id"] == case["name"], f"Identity differs at {i}")
        for key in ("split", "clip_id", "frame_in_clip", "time_s", "base_group_id", "type_id", "layer", "layout_relation"):
            require(item[key] == case[key], f"Spec identity mismatch at {i}: {key}")
        groups[item["base_group_id"]].add(item["split"])
        counts[item["split"]] += 1
    require(all(len(value) == 1 for value in groups.values()), "A base group leaked across splits")
    require(dict(counts) == material["counts"], "Split frame count mismatch")
    for split in ("train", "dev", "evaluation"):
        expected_indices = np.asarray([i for i, r in enumerate(all_ids) if r["split"] == split])
        with np.load(prepared / f"labels/{split}.npz", allow_pickle=False) as archive:
            require(np.array_equal(archive["indices"], expected_indices), f"Label split/index mismatch: {split}")
    indices = labels["indices"]
    n = len(indices)
    require(n == seal["frames"] and labels["classes"].shape == labels["valid"].shape == labels["distances"].shape == (n, 6),
            "Evaluation label/query count mismatch")
    require(labels["valid"].dtype == bool and ((labels["classes"] >= 0) & (labels["classes"] <= 6)).all(), "Invalid label types")
    require(labels["mask"].shape == (n, 6, 45, 80) and labels["coverage"].shape == (n, 45, 80), "Invalid visible-label shapes")
    require(np.isfinite(labels["mask"]).all() and np.isfinite(labels["coverage"]).all()
            and (labels["mask"] >= 0).all() and (labels["coverage"] >= 0).all() and (labels["coverage"] <= 1).all()
            and (labels["mask"] <= labels["coverage"][:, None] + 1e-7).all(), "Mask/coverage contract failed")
    geo_positive = 0
    for k, index in enumerate(indices):
        i = int(index)
        case, geo, vis = spec["cases"][i], geometry[i], visibility[i]
        require(geo["id"] == case["name"] and geo["sample_index"] == i and vis["index"] == i, "Geometry index mismatch")
        same_tree(geo["declared_camera"], case["camera"], "declared camera")
        require(np.max(np.abs(np.asarray(geo["actual_camera_location_m"]) - [case["camera"][a] for a in ("x", "y", "z")])) < .002,
                f"Camera location mismatch: {i}")
        require(np.max(np.abs(geo["actual_camera_rotation"])) < .002, f"Camera rotation mismatch: {i}")
        require(len(case["objects"]) == len(geo["objects"]) and len({o["name"] for o in geo["objects"]}) == len(geo["objects"]),
                f"Declared controlled objects missing/duplicated: {i}")
        for planned, actual in zip(case["objects"], geo["objects"]):
            require(planned["name"] == actual["name"], f"Object identity differs: {i}")
            require(np.allclose(planned["center_m"], actual["render_bounds_center_m"], atol=.002, rtol=0)
                    and np.allclose(planned["size_m"], 2 * np.asarray(actual["render_bounds_extent_m"]), atol=.002, rtol=0),
                    f"Rendered geometry differs from spec: {i}")
        classes, distances = world_box_labels(geo["objects"], case["camera"])
        require(np.array_equal(classes, labels["classes"][k]), f"Independent all-object geometry class mismatch: {i}")
        require(np.allclose(distances, labels["distances"][k], atol=1e-6, rtol=0, equal_nan=True), f"First-hit distance mismatch: {i}")
        validity = np.asarray(vis["unexplained_pixels"]) == 0
        require(np.array_equal(validity, labels["valid"][k]) and np.array_equal(validity, vis["query_label_valid"]),
                f"Invalid query was relabelled or discarded: {i}")
        require(vis["source_native_sha256"] == geo["native_sha256"], f"Visibility provenance mismatch: {i}")
        require(np.allclose(labels["mask"][k].sum((1, 2)) * 64, vis["visible_pixels"], atol=.01, rtol=0),
                f"Visible occupied area lost during pooling: {i}")
        geo_positive += int((classes < 6).sum())
    predictions = {}
    for arm in LEARNED:
        with np.load(stage(root, "predictions") / f"{arm}.npz", allow_pickle=False) as archive:
            predictions[arm] = dict(archive)
    prediction_contract = prediction_checks(predictions, indices)
    identities = [all_ids[int(i)] for i in indices]
    rows = rebuild_rows(identities, labels, predictions, selection)
    same_tree(read_json(evaluated / "frame-results.json"), rows, "frame-results")
    metrics = recompute_metrics(rows, float(spec["dt_s"]))
    same_tree(report["metrics"], metrics, "metrics")
    strata = {key: {value: recompute_metrics([r for r in rows if r[key] == value], float(spec["dt_s"]))
                   for value in sorted({r[key] for r in rows})} for key in ("type_id", "layer", "layout_relation")}
    same_tree(report["strata"], strata, "strata")
    hash_paths = {"spec": root / "plan/spec.json", "geometry": cap / "evaluator/geometry.json",
                  "materialization": prepared / "materialization.json", "evaluation_labels": prepared / "labels/evaluation.npz",
                  "prediction_seal": stage(root, "predictions") / "prediction-seal.json",
                  "metrics": evaluated / "metrics.json", "frame_results": evaluated / "frame-results.json", "audit_source": Path(__file__)}
    return dict(status="PASS", audited_at_utc=datetime.now(timezone.utc).isoformat(),
        backend="TASK_NOT_GPU_SUITABLE", prediction_gate_passed_before_held_reads=True,
        fixed_dev_thresholds={arm: selection[arm]["selection"]["threshold"] for arm in LEARNED},
        input_hashes={name: digest(path) for name, path in hash_paths.items()},
        split_check=dict(unit="base_group", groups=len(groups), frames=dict(counts), disjoint=True),
        geometry_check=dict(frames=n, queries=n * 6, positive_queries=geo_positive,
            invalid_queries_retained=int((~labels["valid"]).sum()), all_declared_objects=True,
            algorithm="world-AABB intersection with query prisms; no production label imports"),
        prediction_check=prediction_contract, metrics=metrics, strata=strata,
        source_seal_check=dict(all_listed_files_match=True, files=sorted(source["code"]),
            source_directory=str(source_dir),
            coverage="Checks exact bytes in the specified source directory against the execution seal; does not prove historic process imports"),
        limits=["Visibility validity and pooled area checked against captured visibility receipt; native pixels not reprojected",
                "Capture geometry JSON authenticated against declared spec; no independent renderer reconstruction",
                "Auxiliary localization components, descriptive curves, bootstrap and scientific decision not recomputed",
                "Controlled same-generator Development only; no hardware, natural-distribution or safety claim"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-dir", type=Path, default=HERE,
                        help="Exact execution source snapshot, if repository files later received mechanical formatting")
    parser.add_argument("--output", type=Path, required=True, help="New audit receipt; existing results are never overwritten")
    args = parser.parse_args()
    output = args.output.resolve()
    artifacts = (HERE.parents[3] / "artifacts.local").resolve()
    require(output.is_relative_to(artifacts), "Audit output must use canonical artifacts.local")
    require(not output.exists(), "Refusing to overwrite an existing audit receipt")
    require(not any(output.is_relative_to(stage(args.root, name).resolve())
                    for name in ("capture", "prepared", "fit", "predictions", "evaluated")),
            "Audit output must not modify an existing result stage")
    try:
        result = audit(args.root, args.source_dir)
    except (AuditFailure, KeyError, ValueError, OSError) as exc:
        result = dict(status="FAIL", error=f"{type(exc).__name__}: {exc}", backend="TASK_NOT_GPU_SUITABLE")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(status=result["status"], output=str(output), error=result.get("error")), ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
