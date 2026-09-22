"""One frozen +/-1cm geometry transfer; unchanged finite-bank predictors."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import active_view as av
import active_view_positive as pos
import shape_hypotheses as sh
import shape_support_union as union

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research/active/dtr-r0/nearfield/OPPORTUNITY_TRANSFER_PROTOCOL_20260921.md"
SHAPE = ROOT / "artifacts.local/work/ba-shape-hypotheses-20260921/run-v1"
ACTIVE = ROOT / "artifacts.local/work/ba-active-view-20260921/run-v1"
SHIFTS = (-0.01, 0.01)
POLICIES = ("fixed", "old_adaptive", "positive_priority")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def geometry_key(value):
    if isinstance(value, dict):
        return tuple((k, geometry_key(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(geometry_key(v) for v in value)
    return round(value, 10) if isinstance(value, (float, int)) else value


def shifted_sources(old_shape, old_active):
    shape, active = [], []
    for parent in old_shape:
        for dx in SHIFTS:
            solids = copy.deepcopy(parent["solids"])
            for solid in solids:
                for vertex in solid:
                    vertex[0] = round(vertex[0] + dx, 10)
            shape.append(dict(case_id=f"{parent['case_id']}_{'L' if dx < 0 else 'R'}",
                              parent=parent["case_id"], group=parent["group"], dx=dx, solids=solids))
    for parent in old_active:
        if not parent["scene"]["boxes"]:
            continue
        for dx in SHIFTS:
            scene = copy.deepcopy(parent["scene"])
            for box in scene["boxes"]:
                box["x"] = round(box["x"] + dx, 10)
            active.append(dict(id=f"{parent['id']}_{'L' if dx < 0 else 'R'}",
                               parent=parent["id"], stratum=parent["stratum"], dx=dx, scene=scene))
    for old, new, field in ((old_shape, shape, "solids"), (old_active, active, "scene")):
        old_keys = {geometry_key(r[field]) for r in old}
        new_keys = [geometry_key(r[field]) for r in new]
        if len(set(new_keys)) != len(new_keys) or old_keys.intersection(new_keys):
            raise ValueError("Geometry admission failed: duplicated or consumed source")
    return shape, active


def choose_all(initial, bank):
    choices = []
    for row in initial:
        obs = tuple(row["ranges"])
        old = av.choose_action(obs, bank)
        new = pos.select_positive_priority(obs, bank)
        choices.append(dict(id=row["id"], initial_ambiguous=new["initial_ambiguous"],
            initial_no_match=new["initial_no_match"], old_costs=old["forecast_costs"],
            positive_costs=new["costs"], actions={"fixed": 0,
                "old_adaptive": old["action_index"], "positive_priority": new["action_index"]}))
    return choices


def active_contrast(rows, baseline):
    result = {}
    for label, decision, truth in (("TP", "INTERSECTS", True),
            ("correct_negative", "NONINTERSECTING_HYPOTHESES", False),
            ("FP", "INTERSECTS", False),
            ("false_OUT", "NONINTERSECTING_HYPOTHESES", True)):
        old = {r["id"] for r in rows if r["truth"] == truth and r["policies"][baseline]["decision"] == decision}
        new = {r["id"] for r in rows if r["truth"] == truth and r["policies"]["positive_priority"]["decision"] == decision}
        result[label] = dict(gained=sorted(new-old), lost=sorted(old-new), retained=len(old & new))
    return result


def grouped(rows, kind):
    group_key = "group" if kind == "shape" else "stratum"
    subsets = {"all": rows}
    subsets.update({"parent_"+g: [r for r in rows if r[group_key] == g] for g in sorted({r[group_key] for r in rows})})
    subsets.update({"shift_"+str(dx): [r for r in rows if r["dx"] == dx] for dx in SHIFTS})
    if kind == "active":
        subsets["initial_ambiguous"] = [r for r in rows if r["initial_ambiguous"]]
        return {key: dict(metrics={p: pos.metrics(part, p) for p in POLICIES},
                         contrasts={p: active_contrast(part, p) for p in POLICIES[:2]}) for key, part in subsets.items()}
    return {key: dict(metrics={p: union.summarize(part, p) for p in ("point_proxy", "consensus", "union")},
                     contrasts={p: union.contrast(part, p) for p in ("point_proxy", "consensus")}) for key, part in subsets.items()}


def decision_checks(shape, active):
    s, a = shape["all"], active["all"]
    delta, pair = s["contrasts"]["point_proxy"], a["contrasts"]["fixed"]
    candidate, fixed = a["metrics"]["positive_priority"], a["metrics"]["fixed"]
    return dict(shape=dict(added_TP=delta["added_TP"] > 0, no_added_FP=delta["added_FP"] == 0,
        retains_point_IN=delta["lost_IN"] == 0,
        no_extra_false_OUT=s["metrics"]["union"]["false_OUT"] <= s["metrics"]["consensus"]["false_OUT"]),
        active=dict(net_TP_gain=candidate["tp"] > fixed["tp"], retains_fixed_TP=not pair["TP"]["lost"],
            no_new_wrong_commit=not pair["FP"]["gained"] and not pair["false_OUT"]["gained"],
            correct_decisive_not_decreased=candidate["correct_decisive"] >= fixed["correct_decisive"]))


def run(output):
    if output.exists():
        raise FileExistsError("Refusing to overwrite evidence")
    output.mkdir(parents=True)
    started, stages = time.perf_counter(), []
    def stage(name):
        stages.append(dict(stage=name, seconds=time.perf_counter()-started))
    try:
        files = [p for folder in (SHAPE, ACTIVE) for p in folder.iterdir() if p.is_file()]
        old_hashes = {str(p.resolve()): av.file_hash(p) for p in files}
        old_shape_seal = read(SHAPE/"prediction-seal.json")
        old_shape_freeze = read(SHAPE/"pre-execution-freeze.json")
        assert av.file_hash(SHAPE/"source-truth.json") == old_shape_freeze["source_truth_sha256"]
        assert av.file_hash(SHAPE/"predictions.json") == old_shape_seal["predictions_sha256"]
        assert av.file_hash(SHAPE/"results.json") == read(SHAPE/"completion.json")["results_sha256"]
        assert av.file_hash(SHAPE/"prior-bank.json") == old_shape_seal["prior_bank_sha256"]
        for name in ("source-seal.json", "choice-seal.json", "prediction-seal.json", "completion-seal.json"):
            av.verify_seal(ACTIVE/name)
        # Exact old implementation bindings; extra two component modules use their
        # own original execution snapshots, not a newly selected variant.
        positive_run = ROOT/"artifacts.local/work/ba-active-view-positive-20260921/run-v1"
        union_run = ROOT/"artifacts.local/work/ba-shape-support-union-20260921/run-v1"
        av.verify_seal(positive_run/"pre-run-seal.json")
        assert av.file_hash(Path(pos.__file__)) == av.file_hash(positive_run/"source.py")
        assert av.file_hash(Path(union.__file__)) == read(union_run/"freeze.json")["source_sha256"]
        assert av.file_hash(Path(sh.__file__)) == old_shape_freeze["source_sha256"]
        for path in (Path(__file__), Path(__file__).with_name("test_opportunity_transfer.py"),
                     Path(av.__file__), Path(pos.__file__), Path(sh.__file__), Path(union.__file__), PROTOCOL):
            shutil.copyfile(path, output/path.name)
        av.write_json(output/"old-input-hashes.json", old_hashes)
        old_shape, old_active = read(SHAPE/"source-truth.json"), read(ACTIVE/"source.json")
        shape_source, active_source = shifted_sources(old_shape, old_active)
        assert len(shape_source) == 348 and len(active_source) == 1304
        av.write_json(output/"shape-source.json", shape_source)
        av.write_json(output/"active-source.json", active_source)
        av.write_json(output/"admission.json", dict(shape_cases=348, active_cases=1304,
            excluded_empty_parent="in_0000", geometry_overlap_with_old=0, new_geometry_duplicates=0,
            shifts_m=SHIFTS, kind="CORRELATED_SAME_GENERATOR_LOCAL_PERTURBATION_DEVELOPMENT",
            backend="CPU NumPy / TASK_NOT_GPU_SUITABLE", python=sys.executable))
        av.seal(output/"pre-observation-seal.json", {p.name: p for p in output.iterdir() if p.is_file()})
        stage("source_code_protocol_frozen_before_observation")
        shape_obs = [sh.observe(r["case_id"], r["solids"]) for r in shape_source]
        initial = [dict(id=r["id"], ranges=list(av.observe(pos.source_scene(r), av.ORIGIN))) for r in active_source]
        av.write_json(output/"shape-observations.json", shape_obs)
        av.write_json(output/"active-initial.json", initial)
        del shape_source, active_source, old_shape, old_active
        values = read(ACTIVE/"public-model-forecasts.json")
        bank = av.ForecastBank(tuple(map(tuple, values["initial"])),
            tuple(tuple(map(tuple, action)) for action in values["forecasts"]), tuple(values["labels"]))
        choices = choose_all(initial, bank)
        av.write_json(output/"choices.json", choices)
        av.seal(output/"choice-seal.json", {"choices": output/"choices.json", "initial": output/"active-initial.json",
            "source": output/"active-source.json", "forecasts": ACTIVE/"public-model-forecasts.json",
            "pre_observation": output/"pre-observation-seal.json"})
        stage("all_choices_sealed_before_actual_second_observations")
        av.verify_seal(output/"choice-seal.json")
        choices = read(output/"choices.json")
        active_source = read(output/"active-source.json")
        future = []
        for r, c in zip(active_source, choices, strict=True):
            assert r["id"] == c["id"]
            future.append(dict(id=r["id"], policies={p: list(av.observe(pos.source_scene(r), av.ACTIONS[c["actions"][p]])) for p in POLICIES}))
        av.write_json(output/"active-selected-observations.json", future)
        del active_source
        shape_bank = sh.compile_bank(read(SHAPE/"prior-bank.json"))
        shape_preds = [sh.predict(o, shape_bank) for o in shape_obs]
        for r in shape_preds:
            r["union"] = union.union_state(r["point_proxy"], r["consensus"])
        active_preds = []
        for o, c, f in zip(initial, choices, future, strict=True):
            assert o["id"] == c["id"] == f["id"]
            policies = {}
            for p in POLICIES:
                camera = av.ACTIONS[c["actions"][p]]
                posterior = av.posterior(tuple(o["ranges"]), tuple(f["policies"][p]), camera, bank)
                policies[p] = dict(decision=av.decide(posterior, bank), remaining_count=len(posterior),
                                   remaining=list(posterior), camera=list(camera))
            active_preds.append(dict(id=o["id"], initial_ambiguous=c["initial_ambiguous"],
                                     initial_no_match=c["initial_no_match"], policies=policies))
        av.write_json(output/"shape-predictions.json", shape_preds)
        av.write_json(output/"active-predictions.json", active_preds)
        av.seal(output/"prediction-seal.json", {p.name: p for p in output.iterdir() if p.is_file()})
        stage("all_predictions_sealed_before_truth_join")
        av.verify_seal(output/"prediction-seal.json")
        shape_source, active_source = read(output/"shape-source.json"), read(output/"active-source.json")
        shape_rows = [dict(**p, parent=s["parent"], group=s["group"], dx=s["dx"], truth_inside=sh.intersects_corridor(s["solids"]))
                      for p, s in zip(shape_preds, shape_source, strict=True)]
        active_rows = [dict(**p, parent=s["parent"], stratum=s["stratum"], dx=s["dx"], truth=av.intersects_query(pos.source_scene(s)))
                       for p, s in zip(active_preds, active_source, strict=True)]
        shape_groups, active_groups = grouped(shape_rows, "shape"), grouped(active_rows, "active")
        checks = decision_checks(shape_groups, active_groups)
        old_shape_obs = {av.canonical({k: v for k, v in o.items() if k != "case_id"}) for o in read(SHAPE/"observations.json")}
        old_active_obs = {tuple(o["ranges"]) for o in read(ACTIVE/"initial-observations.json")}
        old_wrong = [r for r in read(SHAPE/"results.json")["cases"] if r["truth_inside"] and r["consensus"] == "OUT_MODEL_PRIOR"]
        assert {r["case_id"] for r in old_wrong} == {"case_152", "case_154", "case_158", "case_160"}
        summary = dict(shape=shape_groups, active=active_groups, checks=checks,
            decision={k: "LOCAL_TRANSFER_INCREMENT_SUPPORTED" if all(v.values()) else "LOCAL_TRANSFER_INCREMENT_NOT_SUPPORTED" for k, v in checks.items()},
            initial_observation_overlap_with_old=dict(shape=sum(av.canonical({k: v for k, v in o.items() if k != "case_id"}) in old_shape_obs for o in shape_obs),
                                                     active=sum(tuple(o["ranges"]) in old_active_obs for o in initial)),
            old_four_wrong_OUT_saved_regression=[dict(id=r["case_id"], union=union.union_state(r["point_proxy"], r["consensus"])) for r in old_wrong],
            shifted_wrong_OUT_descendants=[r for r in shape_rows if r["parent"] in {v["case_id"] for v in old_wrong}],
            active_action_counts={p: dict(Counter(str(c["actions"][p]) for c in choices)) for p in POLICIES})
        av.write_json(output/"shape-evaluation.json", shape_rows)
        av.write_json(output/"active-evaluation.json", active_rows)
        av.write_json(output/"summary.json", summary)
        stage("evaluator_accounting_completed")
        av.write_json(output/"stage-order.json", stages)
        assert all(av.file_hash(Path(p)) == h for p, h in old_hashes.items())
        av.write_json(output/"old-input-integrity.json", dict(unchanged=True, files=len(old_hashes)))
        av.write_json(output/"local-disposition.json", dict(terminal_id="ba-opportunity-transfer-20260921",
            role="COMPONENT_OR_CHALLENGER", mode="COMPONENT", scope="Local perturbation transfer diagnostic; original gains preserved",
            decisions=summary["decision"], global_registration="PENDING_SUPPORTED_COMMAND", persistent_resources=[]))
        av.seal(output/"completion-seal.json", {p.name: p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(decision=summary["decision"], checks=checks, shape=shape_groups["all"]["metrics"],
                             active=active_groups["all"]["metrics"], seconds=stages[-1]["seconds"]), indent=2))
    except BaseException as exc:
        av.write_json(output/"failure.json", dict(type=type(exc).__name__, message=str(exc), stages=stages))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = (ROOT/"artifacts.local/work/ba-opportunity-transfer-20260921").resolve()
    if canonical not in args.output.resolve().parents:
        parser.error("Output must be a new child of the canonical transfer artifact root")
    run(args.output)
