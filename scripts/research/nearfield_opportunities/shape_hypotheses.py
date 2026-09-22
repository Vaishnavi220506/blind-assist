"""One fixed, synthetic finite-shape-set experiment. No learned model or old data."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research/active/dtr-r0/nearfield/SHAPE_HYPOTHESES_PROTOCOL_20260921.md"
CONFIG = {
    "image_wh": [64, 48], "horizontal_fov_deg": 45.0,
    "tof_zones_wh": [8, 8], "subrays_per_axis": 3,
    "max_radial_range_m": 4.0, "range_tolerance_m": 0.01,
    "mask_xor_tolerance_pixels": 0,
    "corridor_min": [-0.30, -0.30, 0.30],
    "corridor_max": [0.30, 0.60, 3.00],
    "bank_x_centres": [-0.60, -0.42, 0.0, 0.42, 0.60],
    "bank_widths": [0.08, 0.24, 0.48],
    "bank_front_z": [0.25, 0.60, 1.50, 2.50, 3.10],
    "bank_thicknesses": [0.04, 0.30],
    "backend": "CPU NumPy", "backend_reason": "TASK_NOT_GPU_SUITABLE",
}
OBSERVATION_FIELDS = {"case_id", "mask", "zone_ranges_m", "zone_valid"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def box(x: float, width: float, front: float, thickness: float,
        y_min: float = -0.15, y_max: float = 0.15) -> list[list[float]]:
    return [[x - width / 2, y_min, front], [x + width / 2, y_max, front + thickness]]


def intersects_corridor(solids) -> bool:
    """Full closed-solid intersection, including exact contact."""
    lo = np.array(CONFIG["corridor_min"])
    hi = np.array(CONFIG["corridor_max"])
    return any(np.all(np.asarray(b[1]) >= lo - 1e-10)
               and np.all(np.asarray(b[0]) <= hi + 1e-10) for b in solids)


def bank_shapes() -> list[dict]:
    shapes = []
    for x, width, front, thickness in itertools.product(
        CONFIG["bank_x_centres"], CONFIG["bank_widths"],
        CONFIG["bank_front_z"], CONFIG["bank_thicknesses"],
    ):
        shapes.append({"shape_id": f"bank_{len(shapes):03d}",
                       "solids": [box(x, width, front, thickness)]})
    return shapes


def source_cases(bank: list[dict]) -> list[dict]:
    cases = [{"case_id": f"case_{i:03d}", "group": "closed_world",
              "slice": "closed_near_0p30" if s["solids"][0][0][2] == 0.25 else "closed_other",
              "solids": s["solids"]} for i, s in enumerate(bank)]
    for sign, front, half_y in itertools.product([-1, 1], [0.60, 1.50, 2.50], [0.002, 0.025]):
        arm_x = sorted([sign * 0.28, sign * 0.40])
        cases.append({"case_id": f"case_{len(cases):03d}",
                      "group": "appendage_thin" if half_y == 0.002 else "appendage_visible",
                      "solids": [box(sign * 0.60, 0.48, front, 0.04),
                                 [[arm_x[0], -half_y, front - 0.01],
                                  [arm_x[1], half_y, front + 0.01]]]})
    for x, width, front in itertools.product([0.0, 0.60], [0.18, 0.32], [0.60, 1.50, 2.50]):
        cases.append({"case_id": f"case_{len(cases):03d}", "group": "off_grid",
                      "solids": [box(x, width, front * 1.07, 0.11, -0.12, 0.18)]})
    return cases


def pixel_rays(u, v) -> np.ndarray:
    width, height = CONFIG["image_wh"]
    focal = width / (2 * math.tan(math.radians(CONFIG["horizontal_fov_deg"]) / 2))
    u, v = np.broadcast_arrays(u, v)
    rays = np.stack([(u - width / 2) / focal, (v - height / 2) / focal, np.ones_like(u)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def image_rays() -> np.ndarray:
    width, height = CONFIG["image_wh"]
    return pixel_rays(np.arange(width)[None, :] + 0.5, np.arange(height)[:, None] + 0.5)


def tof_rays() -> np.ndarray:
    width, height = CONFIG["image_wh"]
    # Layout: zone_y, zone_x, subray_y, subray_x, xyz.
    u = (np.arange(8)[None, :, None, None] + (np.arange(3)[None, None, None, :] + 0.5) / 3) * width / 8
    v = (np.arange(8)[:, None, None, None] + (np.arange(3)[None, None, :, None] + 0.5) / 3) * height / 8
    return pixel_rays(u, v)


def centre_rays() -> np.ndarray:
    width, height = CONFIG["image_wh"]
    return pixel_rays((np.arange(8)[None, :] + 0.5) * width / 8,
                      (np.arange(8)[:, None] + 0.5) * height / 8)


def raycast(solids, rays: np.ndarray) -> np.ndarray:
    """Nearest forward surface radial range from the camera origin."""
    result = np.full(rays.shape[:-1], np.inf)
    for lower, upper in solids:
        lower, upper = np.asarray(lower), np.asarray(upper)
        parallel = np.abs(rays) < 1e-14
        safe = np.where(parallel, 1.0, rays)
        t0, t1 = lower / safe, upper / safe
        near = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(axis=-1)
        far = np.where(parallel, np.inf, np.maximum(t0, t1)).min(axis=-1)
        invalid_parallel = np.any(parallel & ((lower > 0) | (upper < 0)), axis=-1)
        hit = (~invalid_parallel) & (far >= np.maximum(near, 0.0))
        candidate = np.where(near >= 0, near, far)
        result = np.minimum(result, np.where(hit, candidate, np.inf))
    return result


def observe(case_id: str, solids) -> dict:
    mask = np.isfinite(raycast(solids, image_rays()))
    ranges = raycast(solids, tof_rays())
    valid_hits = np.isfinite(ranges) & (ranges <= CONFIG["max_radial_range_m"])
    counts = valid_hits.sum(axis=(-2, -1))
    means = np.where(valid_hits, ranges, 0.0).sum(axis=(-2, -1)) / np.maximum(counts, 1)
    return {"case_id": case_id, "mask": mask.astype(int).ravel().tolist(),
            "zone_ranges_m": means.ravel().tolist(), "zone_valid": (counts > 0).ravel().tolist()}


def compile_bank(bank: list[dict]) -> dict:
    obs = [observe(s["shape_id"], s["solids"]) for s in bank]
    return {"ids": [s["shape_id"] for s in bank],
            "masks": np.array([o["mask"] for o in obs], dtype=bool),
            "ranges": np.array([o["zone_ranges_m"] for o in obs]),
            "valid": np.array([o["zone_valid"] for o in obs], dtype=bool),
            "inside": np.array([intersects_corridor(s["solids"]) for s in bank], dtype=bool)}


def predict(observation: dict, prior: dict) -> dict:
    if set(observation) != OBSERVATION_FIELDS:
        raise ValueError("Predictor observation contains forbidden or missing fields")
    mask = np.asarray(observation["mask"], dtype=bool)
    measured = np.asarray(observation["zone_ranges_m"])
    valid = np.asarray(observation["zone_valid"], dtype=bool)
    xor_pixels = np.count_nonzero(prior["masks"] != mask[None, :], axis=1)
    residual = np.abs(prior["ranges"] - measured[None, :])
    range_ok = np.all((~valid[None, :]) | (prior["valid"] & (residual <= CONFIG["range_tolerance_m"])), axis=1)
    indices = np.flatnonzero((xor_pixels <= CONFIG["mask_xor_tolerance_pixels"]) & range_ok)
    states = prior["inside"][indices]
    if not np.any(valid):
        consensus = single = "UNKNOWN"
        reason, selected = "NO_OBSERVED_RANGE", None
    elif len(indices) == 0:
        consensus = single = "UNKNOWN"
        reason, selected = "NO_CONSISTENT_PRIOR", None
    else:
        costs = ((residual[indices] ** 2) * valid[None, :]).sum(axis=1)
        selected = int(indices[int(np.argmin(costs))])
        single = "IN" if prior["inside"][selected] else "OUT_MODEL_PRIOR"
        if np.all(states):
            consensus, reason = "IN", "UNANIMOUS_IN"
        elif np.all(~states):
            consensus, reason = "OUT_MODEL_PRIOR", "UNANIMOUS_OUT_WITHIN_PRIOR"
        else:
            consensus, reason = "UNKNOWN", "DISAGREEING_SHAPES"
    points = centre_rays().reshape(-1, 3) * measured[:, None]
    within = np.all(points >= np.array(CONFIG["corridor_min"]) - 1e-10, axis=1) & np.all(points <= np.array(CONFIG["corridor_max"]) + 1e-10, axis=1)
    point_state = "IN" if np.any(valid & within) else "UNKNOWN"
    return {"case_id": observation["case_id"], "consensus": consensus,
            "single_best": single, "point_proxy": point_state, "reason": reason,
            "feasible_count": len(indices), "feasible_inside_count": int(states.sum()),
            "feasible_ids": [prior["ids"][i] for i in indices],
            "selected_id": None if selected is None else prior["ids"][selected],
            "observed_zones": int(valid.sum())}


def metrics(rows: list[dict], policy: str) -> dict:
    tp = sum(r["truth_inside"] and r[policy] == "IN" for r in rows)
    fp = sum(not r["truth_inside"] and r[policy] == "IN" for r in rows)
    fn_out = sum(r["truth_inside"] and r[policy] == "OUT_MODEL_PRIOR" for r in rows)
    tn_out = sum(not r["truth_inside"] and r[policy] == "OUT_MODEL_PRIOR" for r in rows)
    up = sum(r["truth_inside"] and r[policy] == "UNKNOWN" for r in rows)
    un = sum(not r["truth_inside"] and r[policy] == "UNKNOWN" for r in rows)
    p, n = tp + fn_out + up, fp + tn_out + un
    assert tp + fp + fn_out + tn_out + up + un == len(rows)
    return {"n": len(rows), "positive": p, "negative": n, "TP": tp, "FP": fp,
            "FN_alert": fn_out + up, "false_OUT": fn_out, "correct_OUT": tn_out,
            "UNKNOWN_positive": up, "UNKNOWN_negative": un, "UNKNOWN": up + un,
            "coverage": (len(rows) - up - un) / len(rows),
            "precision": tp / (tp + fp) if tp + fp else None,
            "alert_recall": tp / p if p else None, "FPR": fp / n if n else None,
            "wrong_commits": fp + fn_out}


def evaluate(observations: list[dict], predictions: list[dict], cases: list[dict]) -> dict:
    by_id = {c["case_id"]: c for c in cases}
    rows = [{**p, "group": by_id[p["case_id"]]["group"],
             "slice": by_id[p["case_id"]].get("slice"),
             "truth_inside": intersects_corridor(by_id[p["case_id"]]["solids"])} for p in predictions]
    groups = ["all"] + sorted({r["group"] for r in rows}) + ["closed_near_0p30", "closed_other"]
    result = {"metrics": {g: {policy: metrics([r for r in rows if g == "all" or r["group"] == g or r["slice"] == g], policy)
                                   for policy in ["consensus", "single_best", "point_proxy"]} for g in groups},
              "cases": rows,
              "disagreeing_cases": [r["case_id"] for r in rows if r["reason"] == "DISAGREEING_SHAPES"],
              "no_fit_cases": [r["case_id"] for r in rows if r["feasible_count"] == 0],
              "wrong_out_cases": [r["case_id"] for r in rows if r["truth_inside"] and r["consensus"] == "OUT_MODEL_PRIOR"]}
    # Exact byte-equivalent observations, with IDs removed, establish ambiguity
    # independently of the .01m consistency tolerance.
    equivalence: dict[str, list[str]] = {}
    for obs in observations:
        key = json.dumps({k: v for k, v in obs.items() if k != "case_id"}, sort_keys=True)
        equivalence.setdefault(key, []).append(obs["case_id"])
    truth_by_id = {r["case_id"]: r["truth_inside"] for r in rows}
    result["exact_opposite_truth_observation_classes"] = [ids for ids in equivalence.values() if len({truth_by_id[i] for i in ids}) > 1]
    closed = result["metrics"]["closed_world"]
    result["decision"] = ("COMPONENT_SYNTHETIC_AMBIGUITY_RETAINED_PRIOR_COVERAGE_REQUIRED"
                          if result["exact_opposite_truth_observation_classes"]
                          and result["disagreeing_cases"]
                          and closed["consensus"]["wrong_commits"] < closed["single_best"]["wrong_commits"]
                          and closed["consensus"]["coverage"] > 0
                          else "NOT_EVALUABLE_AMBIGUITY_OR_NO_USEFUL_SET_GAIN")
    result["unqualified_clearance_claim"] = "FALSIFIED" if result["wrong_out_cases"] else "NOT_ESTABLISHED"
    return result


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior evidence: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        source = Path(__file__).resolve()
        shutil.copy2(source, output / "source.py")
        shutil.copy2(PROTOCOL, output / "protocol.md")
        bank = bank_shapes()
        cases = source_cases(bank)
        save_json(output / "config.json", CONFIG)
        save_json(output / "prior-bank.json", bank)
        save_json(output / "source-truth.json", cases)
        save_json(output / "pre-execution-freeze.json", {
            "protocol_sha256": digest(PROTOCOL), "source_sha256": digest(source),
            "config_sha256": digest(output / "config.json"),
            "bank_sha256": digest(output / "prior-bank.json"),
            "source_truth_sha256": digest(output / "source-truth.json"),
            "case_count": len(cases), "bank_count": len(bank),
            "phase": "BEFORE_OBSERVATION_AND_PREDICTION", "python": sys.executable,
            "numpy": np.__version__})
        observations = [observe(c["case_id"], c["solids"]) for c in cases]
        save_json(output / "observations.json", observations)
        # Remove source geometry from the predictor phase; reload truth only
        # after writing and hashing all prediction outputs.
        del cases
        observation_hash = digest(output / "observations.json")
        prior = compile_bank(bank)
        inference_start = time.perf_counter()
        predictions = [predict(o, prior) for o in observations]
        inference_seconds = time.perf_counter() - inference_start
        save_json(output / "predictions.json", predictions)
        save_json(output / "prediction-seal.json", {
            "phase": "SEALED_BEFORE_EVALUATOR_TRUTH_JOIN",
            "predictions_sha256": digest(output / "predictions.json"),
            "observations_sha256": observation_hash,
            "source_sha256": digest(source), "protocol_sha256": digest(PROTOCOL),
            "prior_bank_sha256": digest(output / "prior-bank.json")})
        cases = json.loads((output / "source-truth.json").read_text(encoding="utf-8"))
        result = evaluate(observations, predictions, cases)
        result["execution"] = {"seconds_total": time.perf_counter() - started,
                               "seconds_prediction_only": inference_seconds,
                               "backend": CONFIG["backend"], "backend_reason": CONFIG["backend_reason"],
                               "persistent_resources": []}
        save_json(output / "results.json", result)
        save_json(output / "local-disposition.json", {
            "terminal_id": "ba-shape-hypotheses-synthetic-20260921",
            "role": "COMPONENT_OR_CHALLENGER" if result["decision"].startswith("COMPONENT") else "NEGATIVE_CONTROL",
            "mode": "COMPONENT", "decision": result["decision"],
            "scope": "Fixed finite-bank, ideal-observation synthetic set-valued geometry; no runtime admission",
            "unqualified_clearance_claim": result["unqualified_clearance_claim"],
            "global_registration": "PENDING_ROOT_SUPPORTED_COMMAND"})
        save_json(output / "completion.json", {"status": "COMPLETED", "results_sha256": digest(output / "results.json"),
                                               "predictions_sha256": digest(output / "predictions.json"),
                                               "observations_unchanged": digest(output / "observations.json") == observation_hash})
        print(json.dumps({"decision": result["decision"], "metrics": result["metrics"],
                          "disagreeing_cases": len(result["disagreeing_cases"]),
                          "exact_ambiguous_classes": len(result["exact_opposite_truth_observation_classes"]),
                          "wrong_out_cases": result["wrong_out_cases"], "execution": result["execution"]}, indent=2))
    except BaseException as exc:
        save_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc), "seconds": time.perf_counter() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = (ROOT / "artifacts.local/work/ba-shape-hypotheses-20260921").resolve()
    resolved = args.output.resolve()
    if resolved == canonical or canonical not in resolved.parents:
        parser.error("--output must be a new child directory under canonical artifacts.local/work/ba-shape-hypotheses-20260921")
    run(args.output)


if __name__ == "__main__":
    main()
