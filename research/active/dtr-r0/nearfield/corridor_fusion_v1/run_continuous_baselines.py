"""Replay frozen A*, raw HGB and a predeclared A* hysteresis on public data.

The evaluator file and source specification are intentionally not opened here.
Only ``raw.jsonl`` and the RGB paths referenced by it reach the feature/model
code; target identity, family, process and labels stay in the separate
evaluator step.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
ARTIFACTS = (REPO / "artifacts.local").resolve()
DEFAULT_BUNDLE = ARTIFACTS / "work" / "corridor-astar-effect-20260917" / "bundle"
DEFAULT_RAW_MODEL = ARTIFACTS / "work" / "corridor-representation-20260918" / "raw-final.pkl"
DEFAULT_RAW_SEAL = ARTIFACTS / "work" / "corridor-representation-20260918" / "model-seal.json"

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from astar_inference import AStarSystem  # noqa: E402
from representation_features import extract, raw_indices  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def image_path(root, row):
    root = root.resolve()
    path = (root / row["rgb_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"RGB path escapes capture root: {row['id']}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--raw-model", type=Path, default=DEFAULT_RAW_MODEL)
    parser.add_argument("--raw-seal", type=Path, default=DEFAULT_RAW_SEAL)
    parser.add_argument("--hysteresis-gap", type=float, default=0.02)
    args = parser.parse_args()
    capture = args.capture.resolve()
    raw_path = capture / "raw.jsonl"
    rgb_root = capture
    out = args.output.resolve()
    if not out.is_relative_to(ARTIFACTS) or out == ARTIFACTS or out.exists():
        raise ValueError("Fresh baseline output must be strictly under artifacts.local")
    if args.hysteresis_gap <= 0 or not np.isfinite(args.hysteresis_gap):
        raise ValueError("A positive finite predeclared hysteresis gap is required")
    rows = read_rows(raw_path)
    receipt_path = capture / 'receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['status'] != 'PASS' or receipt['frames'] != len(rows):
        raise ValueError('Complete authenticated capture required')
    if receipt['hashes']['raw.jsonl'] != sha(raw_path):
        raise ValueError('Public raw capture hash mismatch')
    # Validate only public RGB here. Native evaluator payloads remain unopened.
    for row in rows:
        if receipt['hashes'][row['rgb_path']] != sha(image_path(capture, row)):
            raise ValueError('Public RGB capture hash mismatch: '+row['id'])
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Capture must contain nonempty unique public frame IDs")
    if any("family" in row or "process" in row or "source_corridor_truth" in row for row in rows):
        raise AssertionError("Evaluator/source labels must not be in the public replay input")
    config = json.loads((args.bundle / "config.json").read_text(encoding="utf-8"))
    raw_seal = json.loads(args.raw_seal.read_text(encoding="utf-8"))
    raw_meta = raw_seal["models"]["raw"]
    raw_threshold = float(raw_meta["threshold"])
    if sha(args.raw_model) != raw_meta["sha256"]:
        raise ValueError("Raw HGB checkpoint identity mismatch")
    raw_model = pickle.loads(args.raw_model.read_bytes())
    if getattr(raw_model, "n_features_in_", None) != int(raw_meta["features"]):
        raise ValueError("Raw HGB feature count mismatch")

    out.mkdir(parents=True)
    nearfield = HERE.parent.parent
    sources = [HERE, HERE.parent / "astar_inference.py", HERE.parent / "representation_features.py",
               nearfield / "mz143_corridor_features.py", nearfield / "mz115_spatial_allocation.py",
               nearfield / "mz125_observable_correction.py", nearfield / "mz136_boundary_geometry.py"]
    source_hashes = {str(path.resolve()): sha(path) for path in sources if path.exists()}
    inputs = {str(raw_path): sha(raw_path), str(receipt_path): sha(receipt_path), str((args.bundle / "config.json").resolve()): sha(args.bundle / "config.json"),
              str((args.bundle / config["model_file"]).resolve()): sha(args.bundle / config["model_file"]),
              str(args.raw_model): sha(args.raw_model), str(args.raw_seal): sha(args.raw_seal)}
    astar = AStarSystem(args.bundle)
    if float(astar.threshold) != float(config["threshold"]):
        raise AssertionError("A* runtime threshold differs from sealed config")
    on_threshold = float(astar.threshold)
    off_threshold = on_threshold - float(args.hysteresis_gap)
    index_checked = False
    predictions = []
    previous_episode = None
    yaw = 0.0
    hysteresis = False
    started = time.perf_counter()
    with threadpool_limits(4):
        for row in rows:
            if row["episode_id"] != previous_episode:
                yaw = 0.0
                hysteresis = False
                previous_episode = row["episode_id"]
            if row.get("imu_valid"):
                yaw += float(row.get("delta_yaw", 0.0))
            rgb = cv2.imread(str(image_path(rgb_root, row)), cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError(f"RGB decode failed: {row['id']}")
            a = astar.predict(row, rgb, yaw)
            feat = extract(row, rgb, yaw, association="multi")
            names = feat["sensor_names"] + feat["geometry_names"]
            vector = np.r_[feat["sensor"], feat["geometry"]]
            keep = raw_indices(names)
            if not index_checked:
                if len(keep) != int(raw_meta["features"]):
                    raise AssertionError(f"Raw feature count {len(keep)} != sealed {raw_meta['features']}")
                index_checked = True
            raw_score = float(raw_model.predict_proba(vector[keep][None])[0, 1])
            raw_alert = bool(raw_score >= raw_threshold)
            if not hysteresis and a["score"] >= on_threshold:
                hysteresis = True
            elif hysteresis and a["score"] < off_threshold:
                hysteresis = False
            predictions.append(dict(
                id=row["id"], episode_id=row["episode_id"], time_s=row["time_s"], yaw=yaw,
                astar_score=a["score"], astar_threshold=on_threshold, astar_alert=a["alert"],
                raw_hgb_score=raw_score, raw_hgb_threshold=raw_threshold, raw_hgb_alert=raw_alert,
                astar_hysteresis_on_threshold=on_threshold,
                astar_hysteresis_off_threshold=off_threshold,
                astar_hysteresis_alert=hysteresis,
                tof_packet_received=bool(row["tof_packet_received"]),
                usable_tof_slots=sum(1 for zone in row["tof_zones"] for target in zone["targets"]
                                     if target.get("status") in ("SIM_VALID", "SIM_MERGED") and
                                     target.get("distance_m") is not None),
                rgb_path=row["rgb_path"],
            ))
    with (out / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    methods = ("astar", "raw_hgb", "astar_hysteresis")
    summary = dict(
        schema="DTR_CONTINUOUS_PUBLIC_BASELINES_V1", status="PASS", frames=len(predictions),
        episodes=len({row["episode_id"] for row in predictions}),
        alerts={name: sum(bool(row[f"{name}_alert"]) for row in predictions) for name in methods},
        model_ids={"astar": config["model_id"], "raw_hgb": "raw_hgb_frozen", "astar_hysteresis": config["model_id"]},
        thresholds=dict(astar=on_threshold, raw_hgb=raw_threshold,
                        astar_hysteresis_on=on_threshold, astar_hysteresis_off=off_threshold),
        model_count=2, training_steps=0, evaluator_read=False, source_labels_used=False,
        predictor_inputs="public raw.jsonl plus referenced RGB only",
        timing_seconds=time.perf_counter() - started,
    )
    write(out / "input-seal.json", dict(inputs=inputs, sources=source_hashes,
        raw_feature_count=int(raw_meta["features"]), raw_feature_indices_sha256=hashlib.sha256(keep.tobytes()).hexdigest(),
        raw_association="multi", evaluator_read=False, source_labels_used=False,
        hysteresis=dict(on_threshold=on_threshold, off_threshold=off_threshold, gap=args.hysteresis_gap),
        backend="CPU_NUMPY_OPENCV_SKLEARN"))
    write(out / "summary.json", summary)
    if not all(sha(path) == digest for path, digest in {**inputs, **source_hashes}.items()):
        raise AssertionError('Replay input/source changed during execution')
    write(out / "completion.json", dict(status="PASS", outputs={name: sha(out / name) for name in ("predictions.jsonl", "input-seal.json", "summary.json")},
        resources="Local model objects released at process exit", evaluator_read=False, source_labels_used=False))
    print(json.dumps(dict(status="PASS", frames=len(predictions), alerts=summary["alerts"], output=str(out)), indent=2))


if __name__ == "__main__":
    main()
