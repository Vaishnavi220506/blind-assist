"""One frozen analytic active-view pilot; no physical-sensor claims."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

ANGLES = tuple(math.radians(-22.5 + (i + 0.5) * 45 / 8) for i in range(8))
RANGE_STEP = 0.10
ACTIONS = ((0.12, 0.0), (-0.12, 0.0), (0.0, 0.12), (0.0, -0.12))
ORIGIN = (0.0, 0.0)
QUERY = (-0.30, 0.30, 0.30, 3.00)
EPS = 1e-12


@dataclass(frozen=True)
class Box:
    x: float
    z: float
    width: float
    thickness: float = 0.04


@dataclass(frozen=True)
class Scene:
    boxes: tuple[Box, ...]
    wall_z: float = 4.20


@dataclass(frozen=True)
class ForecastBank:
    initial: tuple[tuple[int, ...], ...]
    forecasts: tuple[tuple[tuple[int, ...], ...], ...]
    labels: tuple[bool, ...]


def hit_distance(box: Box, camera: tuple[float, float], angle: float) -> float:
    """Nearest nonnegative radial intersection with the full target rectangle."""
    lo, hi = 0.0, math.inf
    for origin, direction, minimum, maximum in (
        (camera[0], math.sin(angle), box.x-box.width/2, box.x+box.width/2),
        (camera[1], math.cos(angle), box.z-box.thickness/2, box.z+box.thickness/2),
    ):
        if abs(direction) < EPS:
            if origin < minimum-EPS or origin > maximum+EPS:
                return math.inf
            continue
        t0, t1 = (minimum-origin)/direction, (maximum-origin)/direction
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if lo > hi+EPS:
            return math.inf
    return lo if hi >= 0 else math.inf


def raycast(scene: Scene, camera: tuple[float, float]) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    ranges, target_hits = [], []
    for angle in ANGLES:
        wall = (scene.wall_z-camera[1])/math.cos(angle)
        target = min((hit_distance(b, camera, angle) for b in scene.boxes), default=math.inf)
        distance = min(wall, target)
        if distance <= 0 or not math.isfinite(distance):
            raise ValueError("Camera lies beyond the modeled observation domain")
        ranges.append(math.floor(distance/RANGE_STEP+0.5))
        target_hits.append(target < wall)
    return tuple(ranges), tuple(target_hits)


def observe(scene: Scene, camera: tuple[float, float]) -> tuple[int, ...]:
    return raycast(scene, camera)[0]


def intersects_query(scene: Scene) -> bool:
    """Always the reference query, independent of the current observation pose."""
    left, right, near, far = QUERY
    return any(b.x+b.width/2 >= left-EPS and b.x-b.width/2 <= right+EPS
               and b.z+b.thickness/2 >= near-EPS and b.z-b.thickness/2 <= far+EPS
               for b in scene.boxes)


def hypothesis_scenes() -> tuple[Scene, ...]:
    return (Scene(()),) + tuple(
        Scene((Box(round(i*0.04, 8), z, w),))
        for i in range(-15, 16)
        for z in (1.0, 1.5, 2.0, 2.5, 3.3)
        for w in (0.04, 0.12, 0.28, 0.48)
    )


def make_bank(hypotheses: tuple[Scene, ...]) -> ForecastBank:
    return ForecastBank(
        tuple(observe(s, ORIGIN) for s in hypotheses),
        tuple(tuple(observe(s, a) for s in hypotheses) for a in ACTIONS),
        tuple(intersects_query(s) for s in hypotheses),
    )


def make_cohort(hypotheses: tuple[Scene, ...]) -> list[dict]:
    rows = [{"id": f"in_{i:04d}", "stratum": "in_prior", "scene": s}
            for i, s in enumerate(hypotheses)]
    offgrid = [Scene((Box(x, z, w),))
               for x in (-0.54, -0.38, -0.22, 0.22, 0.38, 0.54)
               for z in (1.25, 2.25) for w in (0.08, 0.20)]
    mismatch = [Scene((Box(x, z, 0.12),), wall_z=4.80)
                for x in (-0.40, -0.24, 0.24, 0.40) for z in (1.5, 2.5)]
    for stratum, scenes in (("off_grid", offgrid), ("wall_mismatch", mismatch)):
        rows.extend({"id": f"{stratum}_{i:03d}", "stratum": stratum, "scene": s}
                    for i, s in enumerate(scenes))
    return rows


def initial_candidates(initial: tuple[int, ...], bank: ForecastBank) -> tuple[int, ...]:
    return tuple(i for i, o in enumerate(bank.initial) if o == initial)


def choose_action(initial: tuple[int, ...], bank: ForecastBank) -> dict:
    """No true scene, ID, truth label, or actual future observation is accepted."""
    candidates = initial_candidates(initial, bank)
    costs = []
    for action_i, forecast in enumerate(bank.forecasts):
        groups = defaultdict(list)
        for i in candidates:
            groups[forecast[i]].append(i)
        cross_label, same_observation_pairs = 0, 0
        for indices in groups.values():
            positive = sum(bank.labels[i] for i in indices)
            cross_label += positive*(len(indices)-positive)
            same_observation_pairs += len(indices)**2
        costs.append((cross_label, same_observation_pairs, action_i))
    selected = min(costs)[2]
    labels = {bank.labels[i] for i in candidates}
    return {"action_index": selected, "initial_candidates": list(candidates),
            "initial_ambiguous": len(labels) == 2,
            "initial_no_match": not candidates,
            "forecast_costs": [list(c) for c in costs]}


def posterior(initial: tuple[int, ...], next_observation: tuple[int, ...],
              camera: tuple[float, float], bank: ForecastBank) -> tuple[int, ...]:
    predicted = bank.initial if camera == ORIGIN else bank.forecasts[ACTIONS.index(camera)]
    return tuple(i for i in initial_candidates(initial, bank) if predicted[i] == next_observation)


def decide(indices: tuple[int, ...], bank: ForecastBank) -> str:
    labels = {bank.labels[i] for i in indices}
    if labels == {True}:
        return "INTERSECTS"
    if labels == {False}:
        return "NONINTERSECTING_HYPOTHESES"
    return "UNKNOWN"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def write_json(path: Path, value) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path: Path, dependencies: dict[str, Path]) -> None:
    write_json(path, {"files": {name: {"path": str(p.resolve()), "sha256": file_hash(p)}
                                for name, p in dependencies.items()}})


def verify_seal(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    for item in value["files"].values():
        if file_hash(Path(item["path"])) != item["sha256"]:
            raise ValueError("Sealed input changed")
    return value


def observe_after_choice(scene: Scene, case_id: str, arm: str,
                         choices_path: Path, choices_seal: Path) -> tuple[int, ...]:
    sealed = verify_seal(choices_seal)
    if sealed["files"]["choices"]["path"] != str(choices_path.resolve()):
        raise ValueError("Choice file was not bound to this seal")
    choices = json.loads(choices_path.read_text(encoding="utf-8"))
    choice = next(row for row in choices if row["id"] == case_id)
    camera = tuple(choice["arms"][arm]["camera"])
    return observe(scene, camera)


def summarize(rows: list[dict]) -> dict:
    result = {}
    strata = ("in_prior", "off_grid", "wall_mismatch")
    for stratum in strata:
        selected = [r for r in rows if r["stratum"] == stratum]
        for subset, cases in (("all", selected),
                              ("initially_ambiguous", [r for r in selected if r["initial_ambiguous"]])):
            arms = {}
            for arm in ("passive", "fixed", "adaptive"):
                counts = Counter()
                remaining = initial_count = hit_rays = new_rays = 0
                positive_truth = sum(r["truth"] for r in cases)
                action_counts = Counter()
                for row in cases:
                    r = row["arms"][arm]
                    d, truth = r["decision"], row["truth"]
                    if d == "UNKNOWN":
                        counts["unknown"] += 1
                        counts["unknown_positive" if truth else "unknown_negative"] += 1
                    else:
                        predicted = d == "INTERSECTS"
                        counts["decisive"] += 1
                        counts["correct" if predicted == truth else "wrong"] += 1
                        counts[("tp" if truth else "fp") if predicted else ("fn" if truth else "tn")] += 1
                    initial_count += row["initial_candidate_count"]
                    remaining += r["remaining_count"]
                    hit_rays += r["next_target_hit_rays"]
                    new_rays += r["new_target_hit_ray_indices"]
                    counts["newly_seen_target_cases"] += int(row["initial_target_hit_rays"] == 0 and r["next_target_hit_rays"] > 0)
                    counts["target_seen_in_either_view"] += int(row["initial_target_hit_rays"] > 0 or r["next_target_hit_rays"] > 0)
                    counts["no_match_final"] += int(r["remaining_count"] == 0)
                    action_counts[str(r["camera"])] += 1
                for key in ("decisive", "correct", "wrong", "unknown", "unknown_positive", "unknown_negative", "tp", "fp", "fn", "tn", "newly_seen_target_cases", "target_seen_in_either_view", "no_match_final"):
                    counts.setdefault(key, 0)
                arms[arm] = dict(counts, cases=len(cases), positive_truth=positive_truth,
                                 initial_hypothesis_count_sum=initial_count,
                                 final_hypothesis_count_sum=remaining,
                                 hypotheses_eliminated=initial_count-remaining,
                                 next_target_hit_rays=hit_rays, new_target_hit_ray_indices=new_rays,
                                 measurements_per_case=2,
                                 motion_metres_per_case=0 if arm == "passive" else 0.12,
                                 action_counts=dict(action_counts))
            result[f"{stratum}/{subset}"] = arms
    primary = result["in_prior/initially_ambiguous"]
    adaptive = primary["adaptive"]
    gain = all(adaptive["correct"] > primary[a]["correct"] and adaptive["wrong"] <= primary[a]["wrong"]
               for a in ("passive", "fixed"))
    result["decision"] = "SCOPED_IN_PRIOR_COMPONENT" if gain else "NO_ADAPTIVE_TASK_GAIN"
    return result


def run(output: Path, protocol: Path) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite the fixed pilot output")
    output.mkdir(parents=True)
    started = time.perf_counter()
    events = []
    def event(name):
        events.append({"stage": name, "elapsed_seconds": time.perf_counter()-started})

    # Source exists only in the simulator/evaluator. The selector gets ForecastBank.
    hypotheses = hypothesis_scenes()
    cohort = make_cohort(hypotheses)
    source_path, bank_path = output/"source.json", output/"hypothesis-bank.json"
    source = [{"id": r["id"], "stratum": r["stratum"], "scene": asdict(r["scene"])} for r in cohort]
    write_json(source_path, source)
    write_json(bank_path, [asdict(s) for s in hypotheses])
    protocol_copy = output/"protocol-before-observation.md"
    protocol_copy.write_bytes(protocol.read_bytes())
    code = Path(__file__).resolve()
    test_code = code.with_name("test_active_view.py")
    source_seal = output/"source-seal.json"
    seal(source_seal, {"source": source_path, "hypothesis_bank": bank_path,
                       "protocol": protocol_copy, "predictor": code, "tests": test_code})
    event("source_and_protocol_sealed")
    bank = make_bank(hypotheses)
    forecasts_path = output/"public-model-forecasts.json"
    write_json(forecasts_path, asdict(bank))
    initial = [{"id": r["id"], "ranges": list(observe(r["scene"], ORIGIN))} for r in cohort]
    initial_path = output/"initial-observations.json"
    write_json(initial_path, initial)
    initial_by_id = {r["id"]: tuple(r["ranges"]) for r in initial}
    initial_seal = output/"initial-observation-seal.json"
    seal(initial_seal, {"observations": initial_path, "forecasts": forecasts_path, "source_seal": source_seal})
    event("initial_observations_sealed")
    choices = []
    for row in initial:
        choice = choose_action(tuple(row["ranges"]), bank)
        choices.append({"id": row["id"], **choice,
                        "arms": {"passive": {"camera": list(ORIGIN)},
                                 "fixed": {"camera": list(ACTIONS[0])},
                                 "adaptive": {"camera": list(ACTIONS[choice["action_index"]])}}})
    choices_path, choices_seal = output/"sealed-choices.json", output/"choice-seal.json"
    write_json(choices_path, choices)
    seal(choices_seal, {"choices": choices_path, "initial_observations": initial_path,
                        "forecasts": forecasts_path, "predictor": code, "source": source_path,
                        "protocol": protocol_copy})
    event("all_choices_sealed_before_actual_next_observations")
    observations = []
    # Verify the seal before the simulator produces any true-scene second view.
    verify_seal(choices_seal)
    # Re-read the bound plans, so actual poses cannot diverge from sealed choices.
    sealed_choices = json.loads(choices_path.read_text(encoding="utf-8"))
    for source_row, choice in zip(cohort, sealed_choices, strict=True):
        if source_row["id"] != choice["id"]:
            raise ValueError("Source and sealed choice identity mismatch")
        for arm, plan in choice["arms"].items():
            camera = tuple(plan["camera"])
            value = observe(source_row["scene"], camera)
            observations.append({"id": source_row["id"], "arm": arm,
                                 "camera": list(camera), "ranges": list(value)})
    next_path = output/"selected-next-observations.json"
    write_json(next_path, observations)
    seal(output/"next-observation-seal.json", {"next_observations": next_path, "choices": choices_path})
    event("actual_selected_next_observations_sealed")
    predictions = []
    for row in observations:
        candidates = posterior(initial_by_id[row["id"]], tuple(row["ranges"]), tuple(row["camera"]), bank)
        predictions.append({"id": row["id"], "arm": row["arm"],
                            "camera": row["camera"], "remaining": list(candidates),
                            "decision": decide(candidates, bank)})
    predictions_path, prediction_seal = output/"predictions.json", output/"prediction-seal.json"
    write_json(predictions_path, predictions)
    seal(prediction_seal, {"predictions": predictions_path, "initial": initial_path,
                           "next": next_path, "choices": choices_path, "source": source_path,
                           "bank": bank_path, "protocol": protocol_copy, "predictor": code})
    event("public_predictions_sealed_before_evaluation")
    verify_seal(prediction_seal)
    by_key = {(r["id"], r["arm"]): r for r in predictions}
    evaluated = []
    for source_row, choice in zip(cohort, choices, strict=True):
        scene = source_row["scene"]
        initial_hits = raycast(scene, ORIGIN)[1]
        record = {"id": source_row["id"], "stratum": source_row["stratum"],
                  "truth": intersects_query(scene), "initial_ambiguous": choice["initial_ambiguous"],
                  "initial_no_match": choice["initial_no_match"],
                  "initial_candidate_count": len(choice["initial_candidates"]),
                  "initial_target_hit_rays": sum(initial_hits), "arms": {}}
        for arm in ("passive", "fixed", "adaptive"):
            p = by_key[source_row["id"], arm]
            hits = raycast(scene, tuple(p["camera"]))[1]
            record["arms"][arm] = {"camera": p["camera"], "decision": p["decision"],
                                    "remaining_count": len(p["remaining"]),
                                    "next_target_hit_rays": sum(hits),
                                    "new_target_hit_ray_indices": sum(b and not a for a, b in zip(initial_hits, hits, strict=True))}
        evaluated.append(record)
    event("evaluator_truth_and_target_return_audit_completed")
    summary = summarize(evaluated)
    summary["cohort_counts"] = dict(Counter(r["stratum"] for r in source))
    summary["initial_no_match"] = dict(Counter(r["stratum"] for r in evaluated if r["initial_no_match"]))
    summary["backend"] = {"device": "CPU", "reason": "TASK_NOT_GPU_SUITABLE",
                           "description": "small scalar analytic geometry; no sensor/device latency claim"}
    write_json(output/"evaluation.json", evaluated)
    write_json(output/"summary.json", summary)
    write_json(output/"stage-order.json", events)
    write_json(output/"local-disposition.json", {"id": "ba-active-view-20260921",
                "role": "COMPONENT_OR_CHALLENGER" if summary["decision"] == "SCOPED_IN_PRIOR_COMPONENT" else "NEGATIVE_CONTROL",
                "mode": "COMPONENT" if summary["decision"] == "SCOPED_IN_PRIOR_COMPONENT" else "NEGATIVE_CONTROL",
                "terminal": summary["decision"],
                "scope": "finite known-model in-prior synthetic fixed-reference observation selection only",
                "global_registry_status": "PENDING_ROOT_SUPPORTED_REGISTRATION"})
    for upstream in (source_seal, initial_seal, choices_seal, prediction_seal):
        verify_seal(upstream)
    seal(output/"completion-seal.json", {p.name: p for p in output.iterdir() if p.is_file()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output, args.protocol)
    print(json.dumps({"decision": summary["decision"], "cohort_counts": summary["cohort_counts"],
                      "primary": summary["in_prior/initially_ambiguous"]}, indent=2))
