"""Governed consumed UE Core ToF regression, using the retained scalar readout."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import asset_catalog as catalog
import asset_runtime as runtime

REPO = Path(__file__).resolve().parents[2]
SOURCE = "work/ba-core-transfer-20260920"
CALIBRATION = "work/ba-tof-corridor-calibration-20260920"


def predict(args):
    journal_name = os.environ.get("BLINDASSIST_ASSET_RUN_JOURNAL")
    if not journal_name:
        raise ValueError("Use ba.ps1 run research-ue; direct native replay is not admitted")
    journal = catalog.read_json(Path(journal_name))
    if journal.get("state") != "running" or journal.get("reuse_preflight", {}).get("status") != "PASS":
        raise ValueError("UE replay needs a running admitted lifecycle journal")
    declared = set(journal["command"])
    if any(str(getattr(args, name)) not in declared for name in ("manifest", "observations", "baseline", "calibration", "predictions", "seal", "result")):
        raise ValueError("Native replay paths differ from the admitted command")
    import numpy as np
    sys.path.insert(0, str(REPO / "research/active/dtr-r0/nearfield"))
    from tof_corridor_calibration import decide, score_frame

    started = time.perf_counter()
    rows = catalog.read_json(args.manifest)
    threshold = catalog.read_json(args.calibration)["threshold"]
    identities = [row["id"] for row in rows]
    if not rows or len(set(identities)) != len(identities):
        raise ValueError("Empty or duplicate frame identities")
    predictions = []
    observation_root = args.observations.resolve()
    for row in rows:
        relative = Path(row["path"])
        if relative.parts[0] != "observations":
            raise ValueError("Manifest points outside admitted observations")
        path = observation_root.joinpath(*relative.parts[1:]).resolve()
        if not path.is_relative_to(observation_root):
            raise ValueError("Observation path escapes admitted scope")
        if catalog.sha256_file(path) != row["sha256"]:
            raise ValueError(f"Observation hash mismatch: {row['id']}")
        with np.load(path, allow_pickle=False) as observation:
            decision = decide(score_frame(observation["boxes"], observation["values"]), threshold)
        predictions.append({**{key: row[key] for key in ("id", "clip_id", "frame_in_clip", "time_s")},
                            "observation_sha256": row["sha256"], "calibrated": decision})
    for path, value in ((args.predictions, predictions), (args.seal, dict(
        predictions_sha256=None, frames=len(predictions), manifest_sha256=catalog.sha256_file(args.manifest),
        calibration_sha256=catalog.sha256_file(args.calibration), threshold=threshold,
        baseline_opened=False, evaluator_truth_opened=False))):
        if path == args.seal:
            value["predictions_sha256"] = catalog.sha256_file(args.predictions)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    # Read the previous predictions only after sealing this replay's predictions.
    baseline = catalog.read_json(args.baseline)
    baseline_by_id = {row["id"]: row for row in baseline}
    if len(baseline_by_id) != len(baseline) or set(baseline_by_id) != set(identities):
        raise ValueError("Baseline frame identities do not match observation identities")
    mismatches = []
    for row in predictions:
        previous = baseline_by_id[row["id"]]
        if (row["calibrated"] != previous["predictions"]["calibrated"] or
                any(row[key] != previous[key] for key in ("clip_id", "frame_in_clip", "time_s", "observation_sha256"))):
            mismatches.append(row["id"])
    result = dict(status="PASS" if not mismatches else "FAIL", frames=len(predictions),
                  matching_frames=len(predictions)-len(mismatches), mismatches=mismatches,
                  threshold=threshold, baseline_sha256=catalog.sha256_file(args.baseline),
                  prediction_sha256=catalog.sha256_file(args.predictions),
                  backend="CPU", backend_reason="TASK_NOT_GPU_SUITABLE", elapsed_s=time.perf_counter()-started,
                  evidence="Consumed synthetic Development engineering regression; no new performance or fresh validation",
                  evaluator_truth_opened=False, baseline_opened_after_prediction_seal=True)
    with args.result.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result))
    return 0 if not mismatches else 1


def run(args):
    root = REPO / "artifacts.local"
    run_id = args.run_id or "ue-core-reuse-" + uuid.uuid4().hex[:12]
    # A separate catalog unit keeps source data and different runs disjoint.
    output = root / "evidence" / ("ue-reuse-" + runtime.fabric.slug(run_id))
    output.mkdir(parents=True, exist_ok=False)
    inputs = [dict(alias=alias, asset=SOURCE, relative_path=relative, role=role, purpose="consumed-core-regression")
              for alias, relative, role in (("manifest", "observations.json", "observation"),
                   ("observations", "observations", "observation"), ("baseline", "predictions.json", "evaluator"))]
    inputs.append(dict(alias="calibration", asset=CALIBRATION, relative_path="operating-point.json",
                       role="configuration", purpose="frozen-operating-point"))
    outputs = [dict(alias=name, path=str(output / filename), role="result" if name == "result" else "evaluator", required=True)
               for name, filename in (("predictions", "predictions.json"), ("seal", "prediction-seal.json"), ("result", "result.json"))]
    command = [sys.executable, str(Path(__file__).resolve()), "predict"]
    for name in ("manifest", "observations", "baseline", "calibration"):
        command += ["--" + name, "{{input:" + name + "}}"]
    for name in ("predictions", "seal", "result"):
        command += ["--" + name, "{{output:" + name + "}}"]
    spec = dict(schema=runtime.RUN_SCHEMA, id=run_id, route="ue-reuse",
                question="Does the retained Core scalar readout reproduce its sealed decisions through governed reuse?",
                evaluator="tools/data/ue_reuse_run.py", evidence_boundary="Consumed Development engineering regression only",
                reuse=dict(mode="regression", query="core transfer observations tof calibration"),
                command=command, inputs=inputs, outputs=outputs, result_output="result",
                parameters=dict(backend_reason="TASK_NOT_GPU_SUITABLE", code_sha256={
                    name: catalog.sha256_file(REPO / name) for name in (
                        "tools/data/ue_reuse_run.py", "research/active/dtr-r0/nearfield/tof_corridor_calibration.py",
                        "research/active/dtr-r0/nearfield/ba_camera_corridor.py")}))
    path = output / "run-spec.json"
    catalog.atomic_write_json(path, spec)
    result, code = runtime.run_spec(path, repo_root=REPO, artifact_root=root,
                                    policy_path=catalog.DEFAULT_POLICY_PATH, require_ue_reuse=True)
    catalog.atomic_write_json((root / result["journal"]).with_name(run_id + "-receipt.json"), result)
    print(json.dumps(dict(state=result["state"], journal=result["journal"], output=str(output))))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    entry = commands.add_parser("run")
    entry.add_argument("--run-id")
    native = commands.add_parser("predict")
    for name in ("manifest", "observations", "baseline", "calibration", "predictions", "seal", "result"):
        native.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
