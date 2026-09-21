"""One posthoc consumed-Development public positive-priority selector replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

import active_view as base


def select_positive_priority(initial: tuple[int, ...], bank: base.ForecastBank) -> dict:
    """Public initial observation and declared prior only; no truth or case ID."""
    candidates = tuple(i for i, value in enumerate(bank.initial) if value == initial)
    costs = []
    for action_i, forecast in enumerate(bank.forecasts):
        groups = defaultdict(list)
        for i in candidates:
            groups[forecast[i]].append(i)
        unresolved_positive, unresolved_total = 0, 0
        for members in groups.values():
            p = sum(bank.labels[i] for i in members)
            if 0 < p < len(members):
                unresolved_positive += p
                unresolved_total += len(members)
        costs.append((unresolved_positive, unresolved_total, action_i))
    chosen = min(costs)[2]
    return {"action_index": chosen, "camera": list(base.ACTIONS[chosen]),
            "initial_candidates": list(candidates), "costs": [list(v) for v in costs],
            "initial_ambiguous": len({bank.labels[i] for i in candidates}) == 2,
            "initial_no_match": not candidates}


def load_true_source_after_choices(source_path: Path, choice_seal: Path) -> list[dict]:
    bound = base.verify_seal(choice_seal)
    if bound["files"]["true_source_hash_only_before_choices"]["path"] != str(source_path.resolve()):
        raise ValueError("Wrong true source for sealed choices")
    return json.loads(source_path.read_text(encoding="utf-8"))


def source_scene(record: dict) -> base.Scene:
    scene = record["scene"]
    return base.Scene(tuple(base.Box(**b) for b in scene["boxes"]), wall_z=scene["wall_z"])


def metrics(rows: list[dict], policy: str) -> dict:
    c = Counter()
    for row in rows:
        d, truth = row["policies"][policy]["decision"], row["truth"]
        if d == "UNKNOWN":
            c["unknown_positive" if truth else "unknown_negative"] += 1
        elif d == "INTERSECTS":
            c["tp" if truth else "fp"] += 1
        else:
            c["explicit_false_negative" if truth else "correct_negative"] += 1
    for key in ("tp", "fp", "explicit_false_negative", "correct_negative", "unknown_positive", "unknown_negative"):
        c.setdefault(key, 0)
    assert sum(c.values()) == len(rows)
    positive = sum(row["truth"] for row in rows)
    negative = len(rows)-positive
    c["cases"] = len(rows)
    c["positive_truth"] = positive
    c["negative_truth"] = negative
    c["fn_no_including_unknown"] = c["explicit_false_negative"]+c["unknown_positive"]
    c["unknown"] = c["unknown_positive"]+c["unknown_negative"]
    c["correct_decisive"] = c["tp"]+c["correct_negative"]
    c["wrong_decisive"] = c["fp"]+c["explicit_false_negative"]
    c["precision"] = c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else None
    c["recall"] = c["tp"]/positive if positive else None
    c["fpr"] = c["fp"]/negative if negative else None
    c["measurements_per_case"] = 2
    c["translation_metres_per_case"] = 0.12
    c["no_bank_match_final"] = sum(row["policies"][policy]["remaining_count"] == 0 for row in rows)
    return dict(c)


def differences(rows: list[dict], baseline: str) -> dict:
    result = {}
    for name, decision in (("positive", "INTERSECTS"), ("negative", "NONINTERSECTING_HYPOTHESES")):
        old = {r["id"] for r in rows if r["policies"][baseline]["decision"] == decision}
        new = {r["id"] for r in rows if r["policies"]["positive_priority"]["decision"] == decision}
        result[name+"_gained_ids"] = sorted(new-old)
        result[name+"_lost_ids"] = sorted(old-new)
        result[name+"_retained_count"] = len(old & new)
    return result


def run(source: Path, protocol: Path, output: Path):
    if output.exists():
        raise FileExistsError("Refusing to overwrite consumed-replay evidence")
    original_files = sorted(p for p in source.iterdir() if p.is_file())
    before = {p.name: base.file_hash(p) for p in original_files}
    for seal_name in ("source-seal.json", "choice-seal.json", "prediction-seal.json", "completion-seal.json"):
        base.verify_seal(source/seal_name)
    output.mkdir(parents=True)
    start = time.perf_counter()
    stages = []
    def stage(label):
        stages.append({"stage": label, "elapsed_seconds": time.perf_counter()-start})
    try:
        code, tests = Path(__file__).resolve(), Path(__file__).with_name("test_active_view_positive.py").resolve()
        shutil.copyfile(code, output/"source.py")
        shutil.copyfile(tests, output/"tests.py")
        shutil.copyfile(protocol, output/"protocol-before-run.md")
        base.write_json(output/"consumed-input-hashes.json", before)
        base.seal(output/"pre-run-seal.json", {"code": output/"source.py", "tests": output/"tests.py",
                    "protocol": output/"protocol-before-run.md", "base_geometry": Path(base.__file__),
                    "old_input_hashes": output/"consumed-input-hashes.json"})
        # Only saved public initial observations and declared forecasts are parsed.
        values = json.loads((source/"public-model-forecasts.json").read_text())
        bank = base.ForecastBank(tuple(tuple(v) for v in values["initial"]),
               tuple(tuple(tuple(v) for v in action) for action in values["forecasts"]),
               tuple(values["labels"]))
        initial = json.loads((source/"initial-observations.json").read_text())
        initial_by_id = {r["id"]: tuple(r["ranges"]) for r in initial}
        stage("public_initial_prior_and_rule_frozen")
        choices = [{"id": row["id"], **select_positive_priority(tuple(row["ranges"]), bank)} for row in initial]
        choices_path, choice_seal = output/"choices.json", output/"choice-seal.json"
        base.write_json(choices_path, choices)
        base.seal(choice_seal, {"choices": choices_path, "initial": source/"initial-observations.json",
                  "prior_forecasts": source/"public-model-forecasts.json", "code": output/"source.py",
                  "protocol": output/"protocol-before-run.md",
                  "true_source_hash_only_before_choices": source/"source.json"})
        stage("all_653_choices_sealed_before_true_scene_parse_and_future_observation")
        true_source = load_true_source_after_choices(source/"source.json", choice_seal)
        sealed_choices = json.loads(choices_path.read_text())
        observations = []
        for row, choice in zip(true_source, sealed_choices, strict=True):
            assert row["id"] == choice["id"]
            camera = tuple(choice["camera"])
            observations.append({"id": row["id"], "camera": list(camera),
                                 "ranges": list(base.observe(source_scene(row), camera))})
        base.write_json(output/"selected-observations.json", observations)
        base.seal(output/"selected-observation-seal.json", {"observations": output/"selected-observations.json",
                  "choices": choices_path, "true_source": source/"source.json"})
        stage("selected_actual_second_observations_sealed")
        predictions = []
        for row in observations:
            remain = base.posterior(initial_by_id[row["id"]], tuple(row["ranges"]), tuple(row["camera"]), bank)
            predictions.append({"id": row["id"], "camera": row["camera"],
                                "remaining": list(remain), "decision": base.decide(remain, bank)})
        base.write_json(output/"predictions.json", predictions)
        base.seal(output/"prediction-seal.json", {"predictions": output/"predictions.json",
                 "observations": output/"selected-observations.json", "choices": choices_path,
                 "source": source/"source.json", "prior": source/"public-model-forecasts.json",
                 "code": output/"source.py", "protocol": output/"protocol-before-run.md"})
        stage("public_predictions_sealed_before_truth_and_baseline_join")
        # Evaluator truth and actual old-policy outcomes enter only after prediction seal.
        base.verify_seal(output/"prediction-seal.json")
        old_evaluation = {r["id"]: r for r in json.loads((source/"evaluation.json").read_text())}
        evaluated = []
        for row, choice, prediction in zip(true_source, sealed_choices, predictions, strict=True):
            truth = base.intersects_query(source_scene(row))
            old = old_evaluation[row["id"]]
            assert old["truth"] == truth and old["initial_ambiguous"] == choice["initial_ambiguous"]
            evaluated.append({"id": row["id"], "stratum": row["stratum"], "truth": truth,
                 "initial_ambiguous": choice["initial_ambiguous"], "initial_no_match": choice["initial_no_match"],
                 "policies": {"fixed": old["arms"]["fixed"], "old_adaptive": old["arms"]["adaptive"],
                    "positive_priority": {"decision": prediction["decision"], "camera": prediction["camera"],
                                          "remaining_count": len(prediction["remaining"])}}})
        selections = {"all": evaluated}
        selections.update({s: [r for r in evaluated if r["stratum"] == s]
                           for s in ("in_prior", "off_grid", "wall_mismatch")})
        selections["in_prior_initial_ambiguous"] = [r for r in selections["in_prior"] if r["initial_ambiguous"]]
        summary = {"kind": "POSTHOC_CONSUMED_DEVELOPMENT_IMPLEMENTATION_REPLAY_NOT_CONFIRMATION",
                   "metrics": {name: {p: metrics(rows, p) for p in ("fixed", "old_adaptive", "positive_priority")}
                               for name, rows in selections.items()},
                   "paired": {name: {p: differences(rows, p) for p in ("fixed", "old_adaptive")}
                              for name, rows in selections.items()},
                   "action_counts": dict(Counter(str(r["camera"]) for r in sealed_choices)),
                   "backend": "CPU / TASK_NOT_GPU_SUITABLE / ideal scalar simulation"}
        primary = summary["metrics"]["in_prior_initial_ambiguous"]["positive_priority"]
        fixed = summary["metrics"]["in_prior_initial_ambiguous"]["fixed"]
        mechanism_parity = (primary["tp"], primary["correct_negative"], primary["unknown"]) == (56, 62, 155)
        retain = (mechanism_parity and primary["tp"] > fixed["tp"]
                  and primary["correct_decisive"] > fixed["correct_decisive"]
                  and primary["wrong_decisive"] == 0
                  and not summary["paired"]["in_prior"]["fixed"]["positive_lost_ids"])
        summary["audited_expected_mechanism_matches_actual_selected_views"] = mechanism_parity
        summary["decision"] = "RETAIN_CONSUMED_PUBLIC_SELECTOR_COMPONENT" if retain else "IMPLEMENTATION_OR_MECHANISM_GATE_NOT_MET"
        stage("truth_evaluation_and_paired_baseline_accounting_completed")
        base.write_json(output/"evaluation.json", evaluated)
        base.write_json(output/"summary.json", summary)
        base.write_json(output/"stage-order.json", stages)
        after = {p.name: base.file_hash(p) for p in original_files}
        assert after == before, "Old sealed evidence mutated"
        base.write_json(output/"local-disposition.json", {"terminal_id": "ba-active-view-positive-20260921",
                "role": "COMPONENT_OR_CHALLENGER" if retain else "NEGATIVE_CONTROL", "mode": "COMPONENT" if retain else "NEGATIVE_CONTROL",
                "terminal": summary["decision"], "scope": "Consumed finite-prior public positive-priority observation selector only",
                "not_claimed": ["independent confirmation", "OOD coverage", "free-space certification", "hardware", "no-loss dominance over old adaptive"],
                "global_registration": "PENDING_SUPPORTED_ATTEMPT"})
        base.write_json(output/"old-input-integrity.json", {"unchanged": before == after, "before": before, "after": after})
        base.seal(output/"completion-seal.json", {p.name: p for p in output.iterdir() if p.is_file()})
        print(json.dumps({"decision": summary["decision"], "primary": summary["metrics"]["in_prior_initial_ambiguous"],
                          "all": summary["metrics"]["all"], "stage_seconds": stages[-1]["elapsed_seconds"]}, indent=2))
    except BaseException as exc:
        base.write_json(output/"failure.json", {"type": type(exc).__name__, "message": str(exc),
                        "stage_log": stages, "seconds": time.perf_counter()-start})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = Path(__file__).resolve().parents[3]/"artifacts.local/work/ba-active-view-positive-20260921"
    if canonical.resolve() not in args.output.resolve().parents:
        parser.error("Output must be a new child of canonical active-view-positive artifact root")
    run(args.source, args.protocol, args.output)
