"""Independent CPU evaluator for sealed sparse-contour NF-G6 predictions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT / "artifacts.local/work/ba-contour-parallax-20260919"
OLD = ROOT / "artifacts.local/nearfield/structure-information-20260907-v1"
MISS_CLIPS = ("clip_01", "clip_02", "clip_05", "clip_06")
FAR_MARK_CLIPS = ("clip_03", "clip_07", "clip_10")
YAW_CLIPS = ("clip_08", "clip_09")
TARGET_NAMES = ("bar", "flat_mark", "wall_mark", "mark", "flush_wall_mark")
CALIBRATION = dict(width=640, height=360, horizontal_fov_degrees=100.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def frozen_code_hash(out, protocol, name):
    current, original = sha(Path(__file__).with_name(name)), protocol["code_hashes"][name]
    if current == original:
        return current
    require(name == "run_ba_contour_parallax.py", "Frozen scientific code changed: "+name)
    repairs = read(out/"mechanical-repairs.json")["repairs"]
    allowed = [r for r in repairs if r["file"] == name and r["original_sha256"] == original
        and r["repaired_sha256"] == current and r["kind"] == "RESULT_SERIALIZATION"
        and r["scientific_settings_unchanged"] is True and r["completed_predictions_before_repair"] == 0
        and r["model_calls_before_repair"] == 0 and r["matcher_calls_before_repair"] == 1]
    require(len(allowed) == 1, "Runner lacks exact zero-completed-prediction serialization repair receipt")
    return current


def pose_matrix(pose):
    """Camera right/down/forward columns in UE XYZ, matching the original G6."""
    require(pose.get("roll", 0.) == 0, "Only original roll-zero G6 poses are admitted")
    pitch, yaw = math.radians(pose["pitch"]), math.radians(pose["yaw"])
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[-sy, sp*cy, cp*cy], [cy, sp*sy, cp*sy], [0., -cp, sp]]), np.array([pose[k] for k in ("x", "y", "z")])


def points_world(pixels, depth, pose, calibration=CALIBRATION):
    pixels, depth = np.asarray(pixels, dtype=float).reshape(-1, 2), np.asarray(depth, dtype=float).reshape(-1)
    require(len(pixels) == len(depth), "Point/depth count mismatch")
    width, height = calibration["width"], calibration["height"]
    focal = width / (2 * math.tan(math.radians(calibration["horizontal_fov_degrees"] / 2)))
    rays = np.column_stack(((pixels[:, 0]-(width-1)/2)/focal, (pixels[:, 1]-(height-1)/2)/focal, np.ones(len(pixels))))
    rotation, translation = pose_matrix(pose)
    return (rays * depth[:, None]) @ rotation.T + translation


def heading_height(world, pose):
    yaw = math.radians(pose["yaw"])
    origin = np.array([pose[k] for k in ("x", "y", "z")])
    return (world-origin) @ np.array([math.cos(yaw), math.sin(yaw), 0.]), world[:, 2]-.12


def interval_near(pixels, intervals, pose, calibration=CALIBRATION):
    """Full optical-Z intervals must fit heading (.08,3] and height [.65,1.85)."""
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    require(len(pixels) == len(intervals), "Point/interval count mismatch")
    world = points_world(np.repeat(pixels, 2, axis=0), intervals.ravel(), pose, calibration)
    ranges, heights = heading_height(world, pose)
    ranges, heights = ranges.reshape(-1, 2), heights.reshape(-1, 2)
    valid = np.isfinite(intervals).all(axis=1) & (intervals[:, 0] > 0) & (intervals[:, 1] >= intervals[:, 0])
    near = valid & (ranges.min(axis=1) > .08) & (ranges.max(axis=1) <= 3.)
    near &= (heights.min(axis=1) >= .65) & (heights.max(axis=1) < 1.85)
    return near, ranges, heights


def native_support(pixels, native, pose, objects, calibration=CALIBRATION):
    """Nearest native pixel; native bounds +5mm only, no nearby-depth selection."""
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    finite = np.isfinite(pixels).all(axis=1)
    sampled = np.zeros_like(pixels, dtype=int)
    sampled[finite] = np.floor(pixels[finite]+.5).astype(int)
    width, height = calibration["width"], calibration["height"]
    require(native.shape == (height, width), "Native image shape mismatch")
    in_image = finite & (sampled[:, 0] >= 0) & (sampled[:, 0] < width) & (sampled[:, 1] >= 0) & (sampled[:, 1] < height)
    depth = np.full(len(pixels), np.nan)
    depth[in_image] = native[sampled[in_image, 1], sampled[in_image, 0]]
    valid = in_image & np.isfinite(depth) & (depth > .08) & (depth < 100.)
    world = points_world(sampled, depth, pose, calibration)
    target = np.zeros(len(pixels), bool)
    for obj in objects:
        if obj["name"] in TARGET_NAMES:
            target |= (np.abs(world-np.asarray(obj["center_m"])) <= np.asarray(obj["size_m"])/2 + .005).all(axis=1)
    ranges, heights = heading_height(world, pose)
    native_near = valid & (ranges > .08) & (ranges <= 3.)
    native_body_head = native_near & (heights >= .65) & (heights < 1.85)
    return dict(sampled_pixels=sampled, depth_m=depth, valid=valid, target=target & valid,
                heading_m=ranges, height_m=heights, true_near=native_near, true_body_head=native_body_head)


def distribution(values):
    values = np.asarray(values, float).reshape(-1)
    values = values[np.isfinite(values)]
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def finite_list(values):
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values).reshape(-1)]


def evaluate_line(line, native, pose, objects, calibration=CALIBRATION):
    """Score every proposed point; never trim proposals using reference geometry."""
    pixels = np.asarray(line["points"], float).reshape(-1, 2)
    supported = np.asarray(line["point_supported"], bool)
    require(len(pixels) == len(supported) == 17, "Frozen17-point line support changed")
    require(np.isfinite(pixels).all(), "Nonfinite proposed pixel")
    require(type(line["accepted"]) is bool and all(type(v) is bool for v in line["point_supported"]), "Prediction flags must be bool")
    require(np.allclose(pixels[[0,-1]], line["endpoints"], rtol=0, atol=1e-6), "Line endpoint/sample mismatch")
    interval = np.asarray(line["interval_m"], float).reshape(2)
    depth = float(line["depth_m"]) if line["depth_m"] is not None else float("nan")
    if line["accepted"]:
        require(np.isfinite(interval).all() and interval[0] > 0 and interval[0] <= depth <= interval[1], "Accepted distance lacks valid interval")
        require(supported.sum() >= 13, "Accepted line lacks frozen13 observation supports")
    point_near, _, _ = interval_near(pixels, np.tile(interval, (17,1)), pose, calibration)
    near = line["accepted"] and bool(point_near.all())
    reference = native_support(pixels, native, pose, objects, calibration)
    valid, target = reference["valid"], reference["target"]
    non_target = valid & ~target
    consistent = valid & (reference["depth_m"] >= interval[0]-1e-9) & (reference["depth_m"] <= interval[1]+1e-9)
    supported_target_consistent = supported & target & consistent
    accepted_valid = valid & line["accepted"]
    estimated = np.full(17, depth)
    return dict(id=line["id"], accepted=line["accepted"], near=near,
        whole_line_interval_inside_region=bool(point_near.all()), candidate_points=17,
        reference_unknown_points=int((~valid).sum()), reference_valid_points=int(valid.sum()),
        target_candidate_points=int(target.sum()), target_supported_points=int((target & supported).sum()),
        target_consistent_supported_points=int(supported_target_consistent.sum()),
        target_recovered=near and int(supported_target_consistent.sum()) >= 13,
        target_false_near_points=int((target & ~reference["true_near"]).sum()) if near else 0,
        target_false_supported_near_points=int((target & supported & ~reference["true_near"]).sum()) if near else 0,
        non_target_candidate_points=int(non_target.sum()), non_target_reference_far_points=int((non_target & ~reference["true_near"]).sum()),
        non_target_false_near_points=int((non_target & ~reference["true_near"]).sum()) if near else 0,
        non_target_false_supported_near_points=int((non_target & supported & ~reference["true_near"]).sum()) if near else 0,
        non_target_wrong_heading_or_height_points=int((non_target & ~reference["true_body_head"]).sum()) if near else 0,
        accepted_reference_points=int(accepted_valid.sum()), accepted_reference_in_interval_points=int((accepted_valid & consistent).sum()),
        accepted_absolute_depth_error_m=distribution(abs(estimated[accepted_valid]-reference["depth_m"][accepted_valid])),
        accepted_relative_depth_error=distribution(abs(estimated[accepted_valid]-reference["depth_m"][accepted_valid])/reference["depth_m"][accepted_valid]),
        point_reference=dict(sampled_pixels=reference["sampled_pixels"].tolist(), valid=valid.tolist(), target=target.tolist(),
            native_optical_z_m=finite_list(reference["depth_m"]), native_heading_m=finite_list(reference["heading_m"]),
            native_height_m=finite_list(reference["height_m"]), native_in_interval=consistent.tolist(),
            point_supported=supported.tolist(), point_interval_near=point_near.tolist()))


def replay_old_clip(row, native, pose, objects, calibration=CALIBRATION):
    """Exact old G6 semantics, including old best-height and upper-range checks."""
    sparse = row["sparse"]
    pixels = np.asarray(sparse["points"], int).reshape(-1,2)
    reference = native_support(pixels,native,pose,objects,calibration)
    depth, accepted = np.asarray(sparse["depth_m"],float), np.asarray(sparse["accepted"],bool)
    predicted = points_world(pixels,depth,pose,calibration)
    heading, height = heading_height(predicted,pose)
    upper = points_world(pixels,np.asarray(sparse["interval_m"],float).reshape(-1,2)[:,1],pose,calibration)
    upper_heading, _ = heading_height(upper,pose)
    near = accepted & (heading > .08) & (heading <= 3.) & (upper_heading <= 3.)
    body = near & (height >= .65) & (height < 1.85)
    groups = {}
    for name, mask in (("target",reference["target"]),("non_target",reference["valid"] & ~reference["target"])):
        groups[name] = dict(candidates=int(mask.sum()),accepted=int((mask & accepted).sum()),
            near=int((mask & near).sum()),body_head_near=int((mask & body).sum()),unknown=int((mask & ~accepted).sum()),
            true_near=int((mask & reference["true_near"]).sum()),false_near=int((mask & near & ~reference["true_near"]).sum()),
            false_body_head_near=int((mask & body & ~reference["true_near"]).sum()))
    yy,xx = np.indices(native.shape)
    full = native_support(np.column_stack([xx.ravel(),yy.ravel()]),native,pose,objects,calibration)
    dense_path = Path(row["dense_path"])
    require(sha(dense_path) == row["dense_sha256"], "Old dense prediction hash mismatch")
    dense = np.load(dense_path,allow_pickle=False).ravel()
    return dict(clip_id=row["clip_id"],all_candidate_count=len(pixels),groups=groups,
        sparse_target_body_head_recovered=bool((reference["target"] & body).any()),
        native_target_pixels=int(full["target"].sum()),
        dense_target_near_pixels=int((full["target"] & (dense > .08) & (dense <= 3.)).sum()))


def validate_old_replay(replayed, reported, enriched=False):
    """Original reports omit the later false-body/head attribution, never zero it."""
    for key in ("clip_id","all_candidate_count","sparse_target_body_head_recovered","native_target_pixels","dense_target_near_pixels"):
        require(replayed[key] == reported[key], "Old baseline replay mismatch: "+replayed["clip_id"]+"/"+key)
    for group,counts in replayed["groups"].items():
        reference = reported["groups"][group]
        for key,value in counts.items():
            if key not in reference:
                require(not enriched and key == "false_body_head_near", "Missing required old baseline count: "+group+"/"+key)
                continue
            require(reference[key] == value, "Old group count mismatch: "+replayed["clip_id"]+"/"+group+"/"+key)


def summarize_clip(identifier, lines, native_target_pixels):
    keys = ("candidate_points","reference_unknown_points","reference_valid_points","target_candidate_points",
            "target_supported_points","target_consistent_supported_points","target_false_near_points",
            "target_false_supported_near_points","non_target_candidate_points","non_target_reference_far_points",
            "non_target_false_near_points","non_target_false_supported_near_points",
            "non_target_wrong_heading_or_height_points","accepted_reference_points","accepted_reference_in_interval_points")
    row = dict(id=identifier,lines=lines,line_candidates=len(lines),native_target_pixels=native_target_pixels,
               accepted_lines=sum(line["accepted"] for line in lines),unknown_lines=sum(not line["accepted"] for line in lines),
               near_lines=sum(line["near"] for line in lines),target_recovered=any(line["target_recovered"] for line in lines))
    row.update({key:sum(line[key] for line in lines) for key in keys})
    unique_target = {tuple(pixel) for line in lines for pixel,target in zip(line["point_reference"]["sampled_pixels"],line["point_reference"]["target"]) if target}
    row["unique_target_candidate_pixels"] = len(unique_target)
    row["visible_target_pixel_coverage"] = len(unique_target)/native_target_pixels if native_target_pixels else None
    row["non_target_false_near_rate_all_candidates"] = row["non_target_false_near_points"]/row["non_target_candidate_points"] if row["non_target_candidate_points"] else None
    row["non_target_false_near_rate_reference_far"] = row["non_target_false_near_points"]/row["non_target_reference_far_points"] if row["non_target_reference_far_points"] else None
    row["accepted_reference_interval_coverage"] = row["accepted_reference_in_interval_points"]/row["accepted_reference_points"] if row["accepted_reference_points"] else None
    return row


def validate_seals(out):
    """No source geometry/native read before all twelve new prediction rows seal."""
    protocol, predictions, seal = [read(out/name) for name in ("protocol.json","predictions.json","prediction-seal.json")]
    observations, launch, release = [read(out/name) for name in ("observations.json","launch.json","inference-release.json")]
    evaluator_seal = read(out/"evaluator-code-seal.json")
    protocol_hash, observation_hash = sha(out/"protocol.json"), sha(out/"observations.json")
    require(evaluator_seal["protocol_sha256"] == protocol_hash, "Evaluator seal protocol mismatch")
    for name,digest in evaluator_seal["code_hashes"].items():
        require(sha(Path(__file__).with_name(name)) == digest, "Evaluator code changed after seal: "+name)
    require(seal["status"] == predictions["status"] == "COMPLETE" and seal["clips"] == 12 and seal["frames"] == 36, "Incomplete prediction cohort")
    require(seal["predictions_sha256"] == sha(out/"predictions.json"), "Prediction seal mismatch")
    require(seal["protocol_sha256"] == predictions["protocol_sha256"] == launch["protocol_sha256"] == protocol_hash, "Prediction protocol mismatch")
    require(seal["observations_sha256"] == predictions["observations_sha256"] == launch["observations_sha256"] == protocol["observations_sha256"] == observation_hash, "Observation binding mismatch")
    require(predictions["launch_sha256"] == sha(out/"launch.json"), "Launch binding mismatch")
    require(sha(out/"source-audit.json") == protocol["source_audit_sha256"], "Source audit changed")
    for name in protocol["code_hashes"]:
        frozen_code_hash(out,protocol,name)
    require(launch["runner_sha256"] == seal["runner_sha256"] == frozen_code_hash(out,protocol,"run_ba_contour_parallax.py"), "Runner seal mismatch")
    require(len(observations) == len(predictions["rows"]) == predictions["matcher_calls"] == 12, "Twelve clip denominator changed")
    require(predictions["model_calls"] == predictions["training_updates"] == launch["model_calls"] == launch["training_updates"] == 0, "Unexpected model/training work")
    require(predictions["uses_native_depth"] is False and predictions["uses_ideal_metric_poses"] is True, "Observation authority changed")
    require(predictions["cpu"]["affinity"] == launch["affinity"] == [0]
            and predictions["cpu"]["opencv_threads"] == launch["opencv_threads"] == 1
            and launch["opencv_opencl"] is False, "Frozen CPU placement changed")
    require(release["affinity_restored"] and release["background_workers_started"] == release["model_allocations"] == 0, "Inference resource release incomplete")
    require(len(seal["outputs"]) == 12, "Per-clip seal denominator changed")
    for index,(row,observation) in enumerate(zip(predictions["rows"],observations)):
        identifier = f"s{index:02d}"
        require(row["id"] == observation["id"] == identifier and row["endpoint_index"] == observation["endpoint_index"] == index*3+2, "Clip order/identity changed")
        require(read(out/(identifier+".json")) == row and sha(out/(identifier+".json")) == seal["outputs"][identifier+".json"], "Per-clip output changed")
        require(len(observation["frames"]) == 3, "Three-view window changed")
        for frame in observation["frames"]:
            require(sha(frame["path"]) == frame["sha256"], "Observed RGB changed")
        require(row["matcher"]["candidate_count"] == len(row["matcher"]["lines"]) <= protocol["configuration"]["max_lines"], "Candidate budget/count mismatch")
        require(len({line["id"] for line in row["matcher"]["lines"]}) == len(row["matcher"]["lines"]), "Duplicate line identifier")
    return protocol,predictions,observations


def evaluate(out=DEFAULT_OUT):
    out = Path(out).resolve()
    require(not (out/"results.json").exists() and not (out/"line-results.json").exists(), "Evaluation outputs already exist")
    began = time.perf_counter()
    protocol,predictions,observations = validate_seals(out)
    # Evaluator-only private authority begins after the complete new prediction seal.
    source = read(out/"source-audit.json")
    require(Path(source["source_root"]).resolve() == OLD.resolve(), "Original G6 source root changed")
    for name,digest in source["source_receipt_hashes"].items():
        require(sha(OLD/name) == digest, "Old source receipt changed: "+name)
    require(len(source["source_payloads"]) == 36 and len(source["existing_dense_payloads"]) == 12, "Old source denominator changed")
    for payload in source["source_payloads"]:
        for name in ("rgb","native"):
            require(sha(payload[name]["path"]) == payload[name]["sha256"], "Old source payload changed")
    spec, old_predictions = read(OLD/"capture/evaluator/spec.json"),read(OLD/"predictions/result.json")
    old_reports = [read(OLD/name) for name in ("evaluation/result.json","evaluation-attribution/result.json")]
    require(len(spec["cases"]) == 36 and len(spec["clips"]) == len(old_predictions["rows"]) == 12, "Old clip denominator changed")
    rows, baseline = [],[]
    for index,(prediction,observation) in enumerate(zip(predictions["rows"],observations)):
        endpoint = prediction["endpoint_index"]
        case, old = spec["cases"][endpoint],old_predictions["rows"][index]
        pose,calibration = observation["frames"][-1]["pose"],observation["calibration"]
        require(case["camera"] == pose and case["clip_id"] == old["clip_id"] == f"clip_{index:02d}", "Original pose/clip mismatch")
        require(old["endpoint_index"] == endpoint and old["poses"] == [f["pose"] for f in observation["frames"]], "Old three-view pose binding changed")
        for frame,old_rgb in zip(observation["frames"],old["rgb"]):
            require(frame["sha256"] == old_rgb["sha256"], "Old/new RGB pairing differs")
        native = np.load(OLD/"capture/evaluator/native"/f"{endpoint:04d}.npy",allow_pickle=False)
        exact = replay_old_clip(old,native,pose,case["objects"],calibration)
        for report_index,report in enumerate(old_reports):
            reported = report["rows"][index]
            validate_old_replay(exact,reported,enriched=report_index == 1)
        baseline.append(exact)
        lines = [evaluate_line(line,native,pose,case["objects"],calibration) for line in prediction["matcher"]["lines"]]
        row = summarize_clip(prediction["id"],lines,exact["native_target_pixels"])
        clip_spec = spec["clips"][index]
        row.update(original_clip_id=case["clip_id"],endpoint_index=endpoint,motion=clip_spec["motion"],
                   initial_distance_m=clip_spec["distance_m"],control=clip_spec["control"],timing=prediction["timing"],
                   native_sha256=sha(OLD/"capture/evaluator/native"/f"{endpoint:04d}.npy"))
        rows.append(row)
    by_id = {r["original_clip_id"]:r for r in rows}
    require(all(next(b for b in baseline if b["clip_id"] == name)["dense_target_near_pixels"] == 0 for name in MISS_CLIPS), "Old four-clip dense-miss denominator changed")
    require(protocol["retention"]["native_consistent_supported_target_points_per_line"] == protocol["configuration"]["minimum_supported_points"] == 13, "Frozen native support gate changed")
    timing = {key:distribution([r["timing"][key] for r in rows]) for key in ("decode_ms","algorithm_ms","total_ms","cpu_algorithm_ms")}
    peak_mib = predictions["peak_rss_bytes"]/2**20
    require(predictions["peak_rss_bytes"] >= max(r["rss_bytes"] for r in predictions["rows"]), "Peak RSS below observed process RSS")
    recovered = [name for name in MISS_CLIPS if by_id[name]["target_recovered"]]
    control_coverage = {name:by_id[name]["target_candidate_points"] > 0 for name in FAR_MARK_CLIPS}
    yaw_unknown = {name:by_id[name]["accepted_lines"] == 0 and by_id[name]["near_lines"] == 0 for name in YAW_CLIPS}
    gate = dict(at_least_one_of_four_dense_misses_recovered=len(recovered) >= 1,
                all_three_far_mark_controls_have_native_target_candidates=all(control_coverage.values()),
                no_far_mark_target_false_near=all(by_id[name]["target_false_near_points"] == 0 for name in FAR_MARK_CLIPS),
                both_yaw_clips_all_unknown=all(yaw_unknown.values()),
                algorithm_p95_within_200ms=timing["algorithm_ms"]["p95"] <= protocol["budget"]["algorithm_p95_ms"],
                peak_rss_within_256mib=peak_mib <= protocol["budget"]["peak_rss_mib"])
    aggregate_keys = ("line_candidates","accepted_lines","unknown_lines","near_lines","candidate_points","reference_unknown_points",
        "reference_valid_points","target_candidate_points","target_supported_points","target_consistent_supported_points",
        "target_false_near_points","non_target_candidate_points","non_target_reference_far_points","non_target_false_near_points",
        "non_target_false_supported_near_points","non_target_wrong_heading_or_height_points","accepted_reference_points","accepted_reference_in_interval_points")
    totals = {key:sum(row[key] for row in rows) for key in aggregate_keys}
    totals["non_target_false_near_rate_all_candidates"] = totals["non_target_false_near_points"]/totals["non_target_candidate_points"] if totals["non_target_candidate_points"] else None
    totals["non_target_false_near_rate_reference_far"] = totals["non_target_false_near_points"]/totals["non_target_reference_far_points"] if totals["non_target_reference_far_points"] else None
    totals["accepted_reference_interval_coverage"] = totals["accepted_reference_in_interval_points"]/totals["accepted_reference_points"] if totals["accepted_reference_points"] else None
    result = dict(status="COMPLETE",schema="ba-contour-parallax-evaluation-v1",phase=protocol["phase"],clips=12,rgb_frames=36,
        retain_gate=dict(passed=all(gate.values()),checks=gate),decision="RETAIN_IDEAL_POSE_GEOMETRIC_CANDIDATE" if all(gate.values()) else "CONTOUR_PARALLAX_FROZEN_GATE_NOT_MET",
        recovered_dense_miss_clips=recovered,recovery_denominator=list(MISS_CLIPS),recovered_count=len(recovered),recovery_denominator_count=4,
        far_mark_target_coverage=control_coverage,pure_yaw_all_unknown=yaw_unknown,totals=totals,
        rows=[{k:v for k,v in row.items() if k != "lines"} for row in rows],old_baseline=dict(exact_replay_match=True,rows=baseline,
            total_target_candidates=sum(r["groups"]["target"]["candidates"] for r in baseline),
            total_target_false_near=sum(r["groups"]["target"]["false_near"] for r in baseline),
            total_non_target_false_near=sum(r["groups"]["non_target"]["false_near"] for r in baseline),
            total_non_target_false_body_head_near=sum(r["groups"]["non_target"]["false_body_head_near"] for r in baseline),
            historical_timing=old_predictions["timing"],note="Exact unchanged old cached predictions; no old matcher or dense model rerun; different proposal populations prevent a paired false-positive improvement claim"),
        cpu=dict(timing_ms=timing,peak_rss_mib=peak_mib,affinity=[0],opencv_threads=1,development_proxy_only=True),
        evaluator_backend=dict(device="CPU",reason="TASK_NOT_GPU_SUITABLE",model_calls=0,training_updates=0),
        limitations=protocol["limitations"]+[
            "Line near requires its full optical-depth interval at all17 points inside original G6 heading and height bounds",
            "Point counts include repeated samples across lines; unique native target pixel coverage is reported separately",
            "One target candidate establishes presence only; it does not prove robust far rejection or broad coverage",
            "Yaw UNKNOWN with zero candidates remains a missing-coverage case, never proof of correct distance rejection"],
        source_provenance_gaps=source["provenance_gaps"],evidence_hashes={name:sha(out/name) for name in
            ("protocol.json","observations.json","source-audit.json","prediction-seal.json","predictions.json","launch.json","inference-release.json","evaluator-code-seal.json")})
    repairs = read(out/"mechanical-repairs.json")["repairs"] if (out/"mechanical-repairs.json").exists() else []
    result["execution"] = dict(sealed_matcher_calls=12,unsealed_matcher_calls_before_repair=sum(r["matcher_calls_before_repair"] for r in repairs),
        total_matcher_calls_including_unsealed_attempts=12+sum(r["matcher_calls_before_repair"] for r in repairs),model_calls=0,training_updates=0,
        timing_scope="Twelve successfully sealed windows only; prior failed serialization work is separately retained and counted")
    result["mechanical_repairs"] = repairs
    if repairs:
        result["evidence_hashes"]["mechanical-repairs.json"] = sha(out/"mechanical-repairs.json")
    if (out/"evaluator-mechanical-repairs.json").exists():
        result["evaluator_mechanical_repairs"] = read(out/"evaluator-mechanical-repairs.json")
        result["evidence_hashes"]["evaluator-mechanical-repairs.json"] = sha(out/"evaluator-mechanical-repairs.json")
    with (out/"line-results.json").open("x",encoding="utf-8") as stream:
        json.dump(rows,stream,indent=2,allow_nan=False)
    result["line_results_sha256"] = sha(out/"line-results.json")
    result["evaluator_elapsed_seconds"] = time.perf_counter()-began
    with (out/"results.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    result = evaluate(parser.parse_args().output)
    print(json.dumps({key:result[key] for key in ("status","decision","recovered_count","retain_gate")}),flush=True)
