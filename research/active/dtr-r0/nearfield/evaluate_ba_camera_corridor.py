"""Evaluate sealed camera-corridor predictions; CPU metadata/geometry, zero models."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ba_camera_corridor_metrics import ARMS, evaluate_rows

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT / "artifacts.local/work/ba-camera-corridor-20260919"
EPSILON_M = 1e-9


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(out, relative):
    path = (out / relative).resolve()
    require(path.is_relative_to(out.resolve()), f"Artifact escapes experiment: {relative}")
    return path


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def frozen_code_hash(out, protocol, name):
    current, original = sha(Path(__file__).with_name(name)), protocol["hashes"][name]
    if current == original:
        return current
    require(name in ("ba_camera_corridor_capture.py", "launch_ba_camera_corridor.py"), f"Frozen code changed: {name}")
    repairs = read(out / "mechanical-repairs.json")["repairs"]
    allowed = [r for r in repairs if r["file"] == name and r["original_sha256"] == original
               and r["repaired_sha256"] == current and r["model_calls_before_repair"] == 0
               and r["unchanged_spec_sha256"] == protocol["spec_sha256"] and r["reason"]
               and isinstance(r["frames_before_repair"], int) and 0 <= r["frames_before_repair"] <= 96
               and (r["frames_before_repair"] == 0 or r.get("kind") == "SOURCE_READINESS")]
    require(len(allowed) == 1, "Capture hash lacks an exact pre-model mechanical repair receipt")
    return current


def validate_readiness(row):
    require(row["status"] == "READY" and row["error"] is None, "Capture readiness failed")
    require(row["scope"] == "CURRENT_REQUESTED_ASSETS_NOT_ALL_FUTURE_CAMERA_VIEWS"
            and row["settling_unchanged"] is True, "Readiness scope or settling changed")
    require(type(row["poll_count"]) is int and row["poll_count"] >= 1, "Missing native readiness poll")
    require(row["timeout_s"] == 900 and 0 <= row["elapsed_s"] <= row["timeout_s"], "Invalid readiness timing")
    observation = row["observation"]
    require(all(type(observation[k]) is int and observation[k] == 0 for k in
                ("asset_compilation_remaining", "shader_jobs_remaining", "pending_render_assets")),
            "Native asset/shader/render readiness counters are not zero")
    require(observation["asset_registry_loading"] is False and observation["streaming_update_completed"] is True,
            "Native asset registry or streaming is not ready")


def validate_capture_dependencies(receipt, launch):
    readiness = receipt["view_readiness"]
    require(len(readiness) == 96 and [r["sample_index"] for r in readiness] == list(range(96)),
            "Readiness does not cover the unchanged96-frame sequence")
    for row in readiness:
        validate_readiness(row)
    validate_readiness(receipt["readiness_terminal"])
    require(receipt["readiness_terminal"] == {k: v for k, v in readiness[-1].items() if k != "sample_index"},
            "Readiness terminal does not match the last frame")
    helper = sha(Path(__file__).with_name("ue_capture_readiness.py"))
    require(helper == launch["readiness_helper_sha256"] == receipt["readiness_helper_sha256"],
            "Capture readiness helper hash changed")
    plugin = Path(launch["plugin_path"])
    require(plugin.is_absolute() and plugin.name == "BlindAssistCapture.uplugin", "Unexpected native plugin identity")
    require(sha(plugin) == launch["plugin_sha256"], "Capture plugin descriptor hash changed")
    require(sha(plugin.parent / "Binaries/Win64/UnrealEditor-BlindAssistCapture.dll") == launch["plugin_binary_sha256"],
            "Capture native plugin binary hash changed")
    return dict(ready_frames=96, native_counters_zero=True, settling_unchanged=True,
                readiness_helper_sha256=helper, plugin_sha256=launch["plugin_sha256"],
                plugin_binary_sha256=launch["plugin_binary_sha256"])


def primitive_truth(source, profile):
    """Closed authenticated primitive bounds; signed penetration into volume faces.

    Thin width itself is not the boundary distance. These axis-aligned controlled
    cylinders have either their axis inside the X slab or their whole width outside;
    therefore their bounds give the same intersection label as the cylinder.
    """
    case, geometry = source["case"], source["geometry"]
    target = next(obj for obj in case["objects"] if obj["name"] == case["target_name"])
    actual = next(obj for obj in geometry["objects"] if obj["name"] == case["target_name"])
    center, size = np.asarray(target["center_m"]), np.asarray(target["size_m"])
    require(np.max(np.abs(np.asarray(actual["render_bounds_center_m"]) - center)) <= .002 + EPSILON_M,
            "Native target center does not corroborate frozen primitive")
    require(np.max(np.abs(2 * np.asarray(actual["render_bounds_extent_m"]) - size)) <= .002 + EPSILON_M,
            "Native target size does not corroborate frozen primitive")
    pose = case["camera"]
    require(all(abs(pose[key]) <= EPSILON_M for key in ("pitch", "yaw", "roll")), "Nonlevel pose")
    camera_center = np.array([center[1] - pose["y"], pose["z"] - center[2], center[0] - pose["x"]])
    half = size[[1, 2, 0]] / 2
    lower, upper = camera_center - half, camera_center + half
    volume = np.asarray([profile["volume"][axis] for axis in ("x", "y", "z")])
    penetration = np.minimum(upper - volume[:, 0], volume[:, 1] - lower)
    signed_margin = float(penetration.min())
    if target["kind"] == "cylinder":
        # Guard the bounds equivalence rather than silently generalizing to corner contact.
        require(volume[0, 0] <= camera_center[0] <= volume[0, 1] or penetration[0] < -EPSILON_M,
                "Cylinder lateral corner needs exact curved-primitive evaluator")
    else:
        require(target["kind"] == "cube", "Unsupported primitive")
    return dict(truth=bool(signed_margin >= -EPSILON_M),
                boundary=bool(abs(signed_margin) <= profile["boundary_band_m"] + EPSILON_M),
                signed_boundary_margin_m=signed_margin, axis_penetration_m=penetration.tolist(),
                target_camera_bounds_m=dict(lower=lower.tolist(), upper=upper.tolist()),
                exact_contact=bool(abs(signed_margin) <= EPSILON_M))


def validate_predictions(out):
    """Validate every observable/prediction payload before opening private truth."""
    protocol, seal = read(out / "protocol.json"), read(out / "prediction-seal.json")
    observation_seal, launch = read(out / "observation-seal.json"), read(out / "inference-launch.json")
    digest = sha(out / "protocol.json")
    count = protocol["budget"]["total_frames"]
    require(count == 96 and protocol["budget"]["clips"] == 8, "Frozen cohort budget changed")
    for value in (seal, observation_seal):
        require(value["status"] == "COMPLETE" and value["frames"] == count, "Incomplete seal")
        require(value["protocol_sha256"] == digest, "Protocol seal mismatch")
    require(launch["protocol_sha256"] == digest, "Inference launch protocol mismatch")
    require(seal["launch_sha256"] == sha(out / "inference-launch.json"), "Inference launch seal mismatch")
    require(launch["observation_seal_sha256"] == sha(out / "observation-seal.json"), "Observation seal mismatch")
    require(observation_seal["observations_sha256"] == sha(out / "observations.json"), "Observation index mismatch")
    require(observation_seal["source_admission_sha256"] == sha(out / "source-admission.json"), "Admission hash mismatch")
    require(observation_seal["evaluator_source_sha256"] == sha(out / "evaluator-source.json"), "Private source hash mismatch")
    require(observation_seal["capture_geometry_sha256"] == sha(out / "capture/evaluator/geometry.json"), "Geometry hash mismatch")
    require(protocol["spec_sha256"] == sha(out / "spec.json"), "Frozen spec hash mismatch")
    for name in protocol["hashes"]:
        frozen_code_hash(out, protocol, name)
    runner = sha(Path(__file__).with_name("run_ba_camera_corridor.py"))
    require(launch["runner_sha256"] == observation_seal["materializer_sha256"] == runner, "Runner changed")
    require(observation_seal["sensor_source_sha256"] == protocol["hashes"]["ba_nfo_data.py"], "Sensor source changed")
    observed, predicted = read(out / "observations.json"), seal["outputs"]
    require(len(observed) == len(predicted) == count, "Missing or extra frame")
    require([row["id"] for row in observed] == [f"f{i:04d}" for i in range(count)], "Observation identity changed")
    for observation, prediction in zip(observed, predicted):
        for key in ("id", "clip_id", "frame_in_clip", "time_s"):
            require(observation[key] == prediction[key], f"Prediction identity mismatch: {key}")
        require(prediction["protocol_sha256"] == seal["protocol_sha256"], "Per-frame protocol mismatch")
        require(prediction["observation_sha256"] == observation["prepared_sha256"], "Per-frame input mismatch")
        for key in ("prepared", "rgb_path"):
            hash_key = "prepared_sha256" if key == "prepared" else "rgb_sha256"
            require(sha(artifact(out, observation[key])) == observation[hash_key], f"Input payload mismatch: {key}")
        path = out / "predictions" / (prediction["id"] + ".npz")
        require(sha(path) == prediction["sha256"], "Prediction NPZ mismatch")
        require(read(path.with_suffix(".json")) == prediction, "Prediction sidecar mismatch")
        require(set(prediction["predictions"]) == set(ARMS), "Prediction arms changed")
    return protocol, observation_seal, launch, observed, predicted


def summarize(rows, dt_s):
    report = evaluate_rows(rows, dt_s=dt_s)
    by_clip = {clip: [r for r in rows if r["clip_id"] == clip] for clip in {r["clip_id"] for r in rows}}
    for arm, metrics in report["arms"].items():
        for event in metrics["events"]:
            event["boundary_detected"] = any(r["boundary"] and r["predictions"][arm]["alert"]
                for r in by_clip[event["clip_id"]] if event["start_frame"] <= r["frame_in_clip"] <= event["end_frame"])
        count = sum(e["boundary_frames"] > 0 for e in metrics["events"])
        hits = sum(e["boundary_detected"] for e in metrics["events"])
        metrics.update(boundary_event_count=count, boundary_detected_events=hits,
                       boundary_event_recall=hits / count if count else None)
    return report


def source_rows(out, protocol, observations, predictions):
    """Private evaluator authority starts only after validate_predictions returns."""
    import ba_camera_corridor as corridor
    sources, spec = read(out / "evaluator-source.json"), read(out / "spec.json")
    geometry = read(out / "capture/evaluator/geometry.json")
    manifest = read(out / "capture/observations/manifest.json")
    receipt, release = read(out / "capture/receipt.json"), read(out / "capture/process-release.json")
    capture_launch = read(out / "capture/launch-receipt.json")
    admission = read(out / "source-admission.json")
    require(receipt["status"] == "PASS" and receipt["frame_count"] == 96 and release["released"], "Capture incomplete")
    require(receipt["task_actors_released"] and receipt["source_unchanged"], "Capture lifecycle/source changed")
    require(receipt["protocol_sha256"] == capture_launch["protocol_sha256"] == sha(out / "protocol.json"), "Capture protocol mismatch")
    require(receipt["spec_sha256"] == capture_launch["spec_sha256"] == protocol["spec_sha256"], "Capture spec mismatch")
    require(receipt["map_sha256_before"] == receipt["map_sha256_after"] == spec["expected_map_sha256"], "Map changed")
    require(capture_launch["capture_script_sha256"] == receipt["script_sha256"] == frozen_code_hash(out, protocol, "ba_camera_corridor_capture.py"), "Capture script mismatch")
    require(capture_launch["launcher_sha256"] == frozen_code_hash(out, protocol, "launch_ba_camera_corridor.py"), "Capture launcher mismatch")
    readiness_audit = validate_capture_dependencies(receipt, capture_launch)
    require(len(sources) == len(geometry) == len(spec["cases"]) == len(manifest["frames"]) == len(admission["checks"]) == 96, "Source count mismatch")
    require(admission["frames"] == 96 and admission["source_unchanged"] and admission["process_released"], "Source admission incomplete")
    rows, failures = [], []
    calibration = manifest["calibration"]
    yy, xx = np.mgrid[:calibration["height"], :calibration["width"]]
    a, b = (xx - calibration["cx"]) / calibration["fx"], (yy - calibration["cy"]) / calibration["fy"]
    for index, (source, observation, prediction) in enumerate(zip(sources, observations, predictions)):
        case, geo, original, check = source["case"], source["geometry"], source["original"], source["check"]
        require(source["id"] == observation["id"] and case == spec["cases"][index], "Private identity/spec mismatch")
        require(geo == geometry[index] and original == manifest["frames"][index] and check == admission["checks"][index], "Private source index mismatch")
        require(original["id"] == geo["id"] == case["name"] and geo["sample_index"] == original["sample_index"] == index, "Native capture identity mismatch")
        require(geo["readiness"] == {k: v for k, v in receipt["view_readiness"][index].items() if k != "sample_index"},
                "Frame geometry readiness differs from capture receipt")
        settling = 32 if case["frame_in_clip"] == 0 else 16
        require(geo["unchanged_warmup_ticks"] == settling and geo["post_ready_rgb_render_calls"] == settling + 1,
                "Per-frame post-readiness settling changed")
        require(observation["frame_in_clip"] == case["frame_in_clip"] and observation["time_s"] == case["time_s"], "Source time mismatch")
        require(observation["calibration"] == calibration, "Calibration changed")
        require(geo["declared_camera"] == case["camera"], "Declared camera changed")
        pose = case["camera"]
        require(np.max(np.abs(np.asarray(geo["actual_camera_location_m"]) - [pose[k] for k in ("x", "y", "z")])) <= .002 + EPSILON_M, "Camera position mismatch")
        require(max(abs(v) for v in geo["actual_camera_rotation"]) <= 1e-6, "Camera rotation mismatch")
        target = next(obj for obj in geo["objects"] if obj["name"] == case["target_name"])
        require(target["trace"]["blocking"] and target["trace"]["hit_expected_actor"] and target["trace"]["hit_actor_path"] == target["actor_path"], "Target authentication missing")
        declared = next(obj for obj in case["objects"] if obj["name"] == case["target_name"])
        mesh = {"cube": "Cube", "cylinder": "Cylinder"}[declared["kind"]]
        require(target["mesh_path"] == f"/Engine/BasicShapes/{mesh}.{mesh}", "Target mesh mismatch")
        require(target["material_path"] == declared["material"] + "." + declared["material"].split("/")[-1], "Target material mismatch")
        native_path = artifact(out / "capture/evaluator", geo["native_path"])
        require(sha(native_path) == geo["native_sha256"] == source["sensor_native_sha256"], "Native depth mismatch")
        require(original["rgb_sha256"] == geo["rgb_sha256"] == observation["rgb_sha256"], "Native/RGB pairing mismatch")
        native = np.load(native_path, allow_pickle=False)
        require(native.shape == a.shape, "Native depth shape mismatch")
        known = np.isfinite(native) & (native > 0)
        world = np.stack([native + pose["x"], a * native + pose["y"], pose["z"] - b * native], axis=-1)
        center, extent = np.asarray(target["render_bounds_center_m"]), np.asarray(target["render_bounds_extent_m"])
        on_target = known & np.all((world >= center - extent - .02) & (world <= center + extent + .02), axis=-1)
        volume = protocol["profile"]["volume"]
        inside = known & (native >= volume["z"][0]) & (native <= volume["z"][1]) & (a * native >= volume["x"][0]) & (a * native <= volume["x"][1]) & (b * native >= volume["y"][0]) & (b * native <= volume["y"][1])
        reconstructed = dict(native_valid_pixels=int(known.sum()), visible_target_pixels=int(on_target.sum()),
                             competing_corridor_pixels=int((inside & ~on_target).sum()))
        require(all(check[k] == v for k, v in reconstructed.items()), "Native admission recount mismatch")
        if reconstructed["visible_target_pixels"] == 0 or reconstructed["competing_corridor_pixels"] >= 4:
            failures.append(dict(id=source["id"], **reconstructed))
        require(case["pair_id"] != "g4_same_zone" or check["g4_same_zone"] is True, "G4 same-zone admission failed")
        label = primitive_truth(source, protocol["profile"])
        if reconstructed["visible_target_pixels"] == 0 or reconstructed["competing_corridor_pixels"] >= 4:
            label["primitive_intersection"] = label["truth"]
            label["truth"] = None
        with np.load(artifact(out, observation["prepared"]), allow_pickle=False) as inputs, np.load(out / "predictions" / (source["id"] + ".npz"), allow_pickle=False) as saved:
            raw, raw_support = corridor.tof_readout(inputs["boxes"], inputs["values"])
            nf, np_possible, np_definite = corridor.nfo_readout(saved["nfo_probabilities"])
            dp, dp_possible, dp_definite = corridor.depth_readout(saved["depthpro_global"])
            require(prediction["predictions"] == dict(raw_tof=raw, nfo=nf, depthpro_global=dp), "Saved alert/readout mismatch")
            require(prediction["raw_support"] == raw_support, "Raw support mismatch")
            for key, value in (("nfo_possible", np_possible), ("nfo_definite", np_definite), ("depthpro_possible", dp_possible), ("depthpro_definite", dp_definite)):
                require(np.array_equal(saved[key], value), f"Saved support mismatch: {key}")
        rows.append(dict(id=source["id"], clip_id=prediction["clip_id"], source_clip_id=case["clip_id"],
            pair_id=case["pair_id"], frame_in_clip=case["frame_in_clip"], time_s=case["time_s"],
            predictions=prediction["predictions"], native_admission=reconstructed, **label))
    return rows, failures, admission["status"], readiness_audit


def evaluate(out=DEFAULT_OUT):
    out = Path(out).resolve()
    require(not (out / "results.json").exists() and not (out / "frame-results.json").exists(), "Evaluation outputs already exist")
    began = time.perf_counter()
    protocol, _, launch, observations, predictions = validate_predictions(out)
    rows, failures, admission_status, readiness_audit = source_rows(out, protocol, observations, predictions)
    evidence_paths = ("protocol.json", "spec.json", "observation-seal.json", "observations.json", "evaluator-source.json",
        "source-admission.json", "prediction-seal.json", "inference-launch.json", "capture/evaluator/geometry.json",
        "capture/observations/manifest.json", "capture/receipt.json", "capture/launch-receipt.json", "capture/process-release.json")
    result = dict(status="COMPLETE", scope=protocol["phase"], frames=len(rows),
        evidence_hashes={name: sha(out / name) for name in evidence_paths}, evaluator_sha256=sha(__file__),
        evaluator_backend=dict(device="CPU", reason="TASK_NOT_GPU_SUITABLE", model_calls=0, training_updates=0),
        inference_backend=launch, limitations=protocol["limitations"], source_admission_failures=failures,
        capture_readiness_verification=readiness_audit,
        timing_semantics="Posed sampled trajectory; first hit relative to labelled entry, not runtime latency or pre-entry lead")
    if (out / "mechanical-repairs.json").exists():
        result["evidence_hashes"]["mechanical-repairs.json"] = sha(out / "mechanical-repairs.json")
        result["mechanical_repairs"] = read(out / "mechanical-repairs.json")["repairs"]
    if admission_status != "PASS" or failures:
        result.update(status="NOT_EVALUABLE", decision="SOURCE_ADMISSION_FAILED_NO_ALGORITHM_CONCLUSION")
    else:
        dt = protocol["budget"]["nominal_dt_s"]
        result["metrics"] = summarize(rows, dt)
        result["by_clip"] = {clip: summarize([r for r in rows if r["clip_id"] == clip], dt) for clip in sorted({r["clip_id"] for r in rows})}
        result["by_pair"] = {pair: summarize([r for r in rows if r["pair_id"] == pair], dt) for pair in sorted({r["pair_id"] for r in rows})}
        result["clip_identity"] = {r["clip_id"]: dict(source_clip_id=r["source_clip_id"], pair_id=r["pair_id"]) for r in rows}
        base, candidate = [result["metrics"]["arms"][arm] for arm in ("nfo", "depthpro_global")]
        gate = dict(more_interior_events=candidate["interior_detected_events"] > base["interior_detected_events"],
            no_more_false_alert_segments=candidate["false_alert_segment_count"] <= base["false_alert_segment_count"],
            no_more_false_alert_duration=candidate["false_alert_sampled_duration_s"] <= base["false_alert_sampled_duration_s"],
            no_more_positive_unknown=candidate["frames"]["all_known"]["prediction_unknown_positive"] <= base["frames"]["all_known"]["prediction_unknown_positive"],
            same_truth_denominator=all(m["frames"]["all_known"]["frames"] == len(rows) and m["event_count"] == base["event_count"] and m["interior_event_count"] == base["interior_event_count"] for m in result["metrics"]["arms"].values()))
        result["retain_gate"] = dict(passed=all(gate.values()), checks=gate)
        result["decision"] = "RETAIN_CONTROLLED_DEVELOPMENT_CHALLENGER" if all(gate.values()) else "EXACT_CORRIDOR_READOUT_GAIN_NOT_MET"
        result["raw_comparison"] = {arm: dict(
            raw_positive_alerts_lost=sum(r["truth"] and r["predictions"]["raw_tof"]["alert"] and not r["predictions"][arm]["alert"] for r in rows),
            raw_positive_misses_rescued=sum(r["truth"] and not r["predictions"]["raw_tof"]["alert"] and r["predictions"][arm]["alert"] for r in rows),
            raw_false_alerts_removed=sum(not r["truth"] and r["predictions"]["raw_tof"]["alert"] and not r["predictions"][arm]["alert"] for r in rows),
            raw_false_alerts_added=sum(not r["truth"] and not r["predictions"]["raw_tof"]["alert"] and r["predictions"][arm]["alert"] for r in rows)) for arm in ("nfo", "depthpro_global")}
    write_new(out / "frame-results.json", rows)
    result["frame_results_sha256"] = sha(out / "frame-results.json")
    result["evaluator_elapsed_seconds"] = time.perf_counter() - began
    write_new(out / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = evaluate(args.output)
    print(json.dumps({key: result[key] for key in ("status", "decision", "frames")}), flush=True)
