"""Frozen paired boundary information diagnostic; never an oracle policy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time

import active_view as av
import active_view_positive as pos

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/"artifacts.local/work/ba-active-view-20260921/run-v1"
TRANSFER = ROOT/"artifacts.local/work/ba-opportunity-transfer-20260921/run-v1"
PROTOCOL = ROOT/"research/active/dtr-r0/nearfield/BOUNDARY_SEPARABILITY_PROTOCOL_20260922.md"
CONFIG = dict(sides=[-1, 1], widths=[.06, .18, .36], depths=[.85, 1.35, 1.85, 2.35, 2.85],
              margins=[.005, .015, .030], thickness=.04, wall_z=4.20,
              raw_equality_tolerance_m=1e-12, backend="CPU / TASK_NOT_GPU_SUITABLE")
POLICIES = ("fixed", "old_adaptive", "positive_priority")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source():
    pairs, scenes = [], []
    for side, width, depth, margin in itertools.product(CONFIG["sides"], CONFIG["widths"], CONFIG["depths"], CONFIG["margins"]):
        pair_id = f"pair_{len(pairs):03d}"
        ids = []
        for offset, name in ((-margin, "inside"), (margin, "outside")):
            scene = av.Scene((av.Box(round(side*(.30+width/2+offset), 10), depth, width, CONFIG["thickness"]),), CONFIG["wall_z"])
            ident = f"scene_{len(scenes):03d}"
            scenes.append(dict(id=ident, scene=asdict(scene)))
            ids.append(ident)
        pairs.append(dict(id=pair_id, members=ids, side=side, width=width, depth=depth, margin=margin))
    return pairs, scenes


def geometry_key(scene):
    return (round(scene["wall_z"], 10), tuple(tuple(round(b[k], 10) for k in ("x", "z", "width", "thickness")) for b in scene["boxes"]))


def observe_with_raw(scene, camera):
    raw = [min((scene.wall_z-camera[1])/math.cos(angle),
               min((av.hit_distance(b, camera, angle) for b in scene.boxes), default=math.inf)) for angle in av.ANGLES]
    bins = av.observe(scene, camera)
    assert tuple(math.floor(v/av.RANGE_STEP+.5) for v in raw) == bins
    return dict(bins=list(bins), raw_radial_m=raw)


def choose(initial, bank):
    result = []
    for row in initial:
        bins = tuple(row["bins"])
        old, new = av.choose_action(bins, bank), pos.select_positive_priority(bins, bank)
        result.append(dict(id=row["id"], initial_candidates=new["initial_candidates"],
            initial_no_match=new["initial_no_match"], actions=dict(fixed=0,
                old_adaptive=old["action_index"], positive_priority=new["action_index"])))
    return result


def pair_rows(pairs, scenes, observations, choices):
    source_by_id, obs, plans = ({r["id"]: r for r in rows} for rows in (scenes, observations, choices))
    rows = []
    for pair in pairs:
        a, b = pair["members"]
        truths = [av.intersects_query(pos.source_scene(source_by_id[i])) for i in (a, b)]
        assert truths == [True, False], "Full-extent source does not cross the fixed boundary"
        initial_equal = obs[a]["views"][0]["bins"] == obs[b]["views"][0]["bins"]
        separating = [i for i in range(4) if obs[a]["views"][i+1]["bins"] != obs[b]["views"][i+1]["bins"]]
        raw_delta = [max(abs(x-y) for x, y in zip(va["raw_radial_m"], vb["raw_radial_m"], strict=True))
                     for va, vb in zip(obs[a]["views"], obs[b]["views"], strict=True)]
        if initial_equal:
            assert plans[a] == {**plans[b], "id": a}, "Identical public inputs cannot get hidden-ID choices"
        status = ("INITIALLY_SEPARATED" if not initial_equal else "ACTION_SEPARABLE" if separating
                  else "QUANTIZATION_ALIAS" if max(raw_delta) > CONFIG["raw_equality_tolerance_m"] else "RAW_RAY_ALIAS")
        rows.append(dict(**pair, truth=truths, initial_equal=initial_equal,
            initial_bins=obs[a]["views"][0]["bins"] if initial_equal else None,
            separating_actions=separating, raw_max_difference_by_view_m=raw_delta, status=status,
            initial_no_match=plans[a]["initial_no_match"] if initial_equal else None,
            policies={p: dict(actions=[plans[a]["actions"][p], plans[b]["actions"][p]],
                pair_separated=(plans[a]["actions"][p] in separating) if initial_equal else None) for p in POLICIES}))
    return rows


def summarize(rows):
    eligible = [r for r in rows if r["initial_equal"]]
    by_initial = defaultdict(list)
    for r in eligible:
        by_initial[tuple(r["initial_bins"])].append(r)
    classes = []
    for bins, group in sorted(by_initial.items()):
        counts = [sum(i in r["separating_actions"] for r in group) for i in range(4)]
        best = max(range(4), key=lambda i: (counts[i], -i))
        classes.append(dict(initial_bins=bins, pairs=[r["id"] for r in group],
            separating_counts=counts, oracle_common_action=best, oracle_common_separated=counts[best],
            separates_every_pair=counts[best] == len(group)))
    metrics = dict(pairs=len(rows), initial_different=len(rows)-len(eligible), initial_equal=len(eligible),
        status_counts=dict(Counter(r["status"] for r in rows)),
        per_action_separated=[sum(i in r["separating_actions"] for r in eligible) for i in range(4)],
        pair_oracle_separated=sum(bool(r["separating_actions"]) for r in eligible),
        class_common_oracle_separated=sum(c["oracle_common_separated"] for c in classes),
        initial_classes=len(classes), universal_action_classes=sum(c["separates_every_pair"] for c in classes),
        policies={p: dict(separated=sum(r["policies"][p]["pair_separated"] for r in eligible),
            missed_available=[r["id"] for r in eligible if r["separating_actions"] and not r["policies"][p]["pair_separated"]],
            missed_with_no_initial_bank_match=[r["id"] for r in eligible if r["separating_actions"] and not r["policies"][p]["pair_separated"] and r["initial_no_match"]]) for p in POLICIES})
    metrics["decision"] = ("NOT_EVALUABLE_NO_INITIAL_ALIAS" if not eligible else
        "NO_ALLOWED_ACTION_SEPARATES_INITIAL_ALIASES" if not metrics["pair_oracle_separated"] else
        "OBSERVATION_CLASS_OPPORTUNITY_CURRENT_SELECTOR_MISSES" if metrics["class_common_oracle_separated"] > metrics["policies"]["positive_priority"]["separated"] else
        "PAIR_INFORMATION_EXISTS_NO_CLASS_AGGREGATE_GAIN_OVER_SELECTOR")
    return dict(metrics=metrics, classes=classes)


def cohort_ceiling(scenes, observations):
    obs = {r["id"]: r for r in observations}
    truth = {r["id"]: av.intersects_query(pos.source_scene(r)) for r in scenes}
    groups = defaultdict(list)
    for ident, row in obs.items():
        groups[tuple(row["views"][0]["bins"])].append(ident)
    classes = []
    for bins, ids in sorted(groups.items()):
        initially_mixed = len({truth[i] for i in ids}) > 1
        resolvable = []
        for action in range(4):
            buckets = defaultdict(list)
            for ident in ids:
                buckets[tuple(obs[ident]["views"][action+1]["bins"])].append(ident)
            resolvable.append(sorted(i for members in buckets.values() if len({truth[i] for i in members}) == 1 for i in members))
        best = max(range(4), key=lambda i: (len(resolvable[i]), -i))
        classes.append(dict(initial_bins=bins, ids=ids, initially_mixed=initially_mixed,
            resolved_ids_by_action=resolvable, oracle_common_action=best,
            oracle_common_resolved=len(resolvable[best]), per_case_oracle_resolved=len(set().union(*map(set, resolvable)))))
    mixed = [c for c in classes if c["initially_mixed"]]
    return dict(total_scenes=len(scenes), initial_mixed_scenes=sum(len(c["ids"]) for c in mixed),
        initially_resolved_scenes=sum(len(c["ids"]) for c in classes if not c["initially_mixed"]),
        mixed_per_action_resolved=[sum(len(c["resolved_ids_by_action"][i]) for c in mixed) for i in range(4)],
        mixed_common_action_ceiling=sum(c["oracle_common_resolved"] for c in mixed),
        mixed_per_case_action_ceiling=sum(c["per_case_oracle_resolved"] for c in mixed), classes=classes)


def run(output):
    if output.exists():
        raise FileExistsError("Refusing existing evidence directory")
    output.mkdir(parents=True)
    start, stages = time.perf_counter(), []
    def stage(name):
        stages.append(dict(stage=name, seconds=time.perf_counter()-start))
    try:
        av.verify_seal(OLD/"completion-seal.json")
        av.verify_seal(TRANSFER/"completion-seal.json")
        positive = ROOT/"artifacts.local/work/ba-active-view-positive-20260921/run-v1"
        av.verify_seal(positive/"pre-run-seal.json")
        assert av.file_hash(Path(pos.__file__)) == av.file_hash(positive/"source.py")
        old_hashes = {str(p.resolve()): av.file_hash(p) for folder in (OLD, TRANSFER) for p in folder.iterdir() if p.is_file()}
        pairs, scenes = source()
        old_keys = {geometry_key(r["scene"]) for path in (OLD/"source.json", TRANSFER/"active-source.json") for r in read(path)}
        keys = [geometry_key(r["scene"]) for r in scenes]
        assert len(pairs) == 90 and len(keys) == len(set(keys)) == 180 and not old_keys.intersection(keys)
        av.write_json(output/"pairs.json", pairs)
        av.write_json(output/"source.json", scenes)
        av.write_json(output/"config.json", dict(**CONFIG, actions=av.ACTIONS, angles=av.ANGLES, range_step=av.RANGE_STEP, query=av.QUERY, python=sys.executable))
        av.write_json(output/"admission.json", dict(unique_scenes=180, old_geometry_overlap=0, pairs=90,
            evidence="NEW_PAIRED_SAME_GENERATOR_SYNTHETIC_DEVELOPMENT", original_hashes=old_hashes))
        for path in (Path(__file__), Path(__file__).with_name("test_boundary_separability.py"), Path(av.__file__), Path(pos.__file__), PROTOCOL):
            shutil.copyfile(path, output/path.name)
        av.seal(output/"source-seal.json", {p.name: p for p in output.iterdir() if p.is_file()})
        stage("source_and_protocol_frozen_before_observations")
        initial = [dict(id=r["id"], **observe_with_raw(pos.source_scene(r), av.ORIGIN)) for r in scenes]
        av.write_json(output/"initial-observations.json", initial)
        del scenes, pairs
        public = [dict(id=r["id"], bins=r["bins"]) for r in initial]
        values = read(OLD/"public-model-forecasts.json")
        bank = av.ForecastBank(tuple(map(tuple, values["initial"])), tuple(tuple(map(tuple, a)) for a in values["forecasts"]), tuple(values["labels"]))
        choices = choose(public, bank)
        av.write_json(output/"choices.json", choices)
        av.seal(output/"choice-seal.json", {"choices": output/"choices.json", "initial": output/"initial-observations.json",
            "source_seal": output/"source-seal.json", "bank": OLD/"public-model-forecasts.json"})
        stage("public_policy_choices_sealed_before_all_future_views")
        av.verify_seal(output/"choice-seal.json")
        scenes = read(output/"source.json")
        observations = []
        for s, init in zip(scenes, initial, strict=True):
            assert s["id"] == init["id"]
            observations.append(dict(id=s["id"], views=[{k: v for k, v in init.items() if k != "id"}]+
                [observe_with_raw(pos.source_scene(s), a) for a in av.ACTIONS]))
        av.write_json(output/"all-view-observations.json", observations)
        av.seal(output/"observation-seal.json", {p.name: p for p in output.iterdir() if p.is_file()})
        stage("all_evaluator_views_sealed_before_truth_and_pair_analysis")
        av.verify_seal(output/"observation-seal.json")
        rows = pair_rows(read(output/"pairs.json"), scenes, observations, read(output/"choices.json"))
        summary = summarize(rows)
        summary["by_margin"] = {str(v): summarize([r for r in rows if r["margin"] == v])["metrics"] for v in CONFIG["margins"]}
        summary["by_depth"] = {str(v): summarize([r for r in rows if r["depth"] == v])["metrics"] for v in CONFIG["depths"]}
        summary["by_width"] = {str(v): summarize([r for r in rows if r["width"] == v])["metrics"] for v in CONFIG["widths"]}
        summary["by_side"] = {str(v): summarize([r for r in rows if r["side"] == v])["metrics"] for v in CONFIG["sides"]}
        summary["cohort_ceiling"] = cohort_ceiling(scenes, observations)
        summary["view_cost"] = dict(scenes=180, evaluator_views_per_scene=5, total_evaluator_views=900,
                                    policy_views_per_scene=2, policy_translation_m=.12)
        av.write_json(output/"pair-evaluation.json", rows)
        av.write_json(output/"summary.json", summary)
        assert all(av.file_hash(Path(p)) == h for p, h in old_hashes.items())
        stage("accounting_complete_old_inputs_unchanged")
        av.write_json(output/"stage-order.json", stages)
        av.write_json(output/"local-disposition.json", dict(terminal_id="ba-boundary-separability-20260922",
            role="COMPONENT_OR_CHALLENGER", mode="COMPONENT", decision=summary["metrics"]["decision"],
            scope="Finite boundary-pair information diagnostic only; no oracle policy admission",
            registration="PENDING_SUPPORTED_COMMAND", persistent_resources=[]))
        av.seal(output/"completion-seal.json", {p.name: p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(metrics=summary["metrics"], cohort={k:v for k,v in summary["cohort_ceiling"].items() if k != "classes"},
                             seconds=stages[-1]["seconds"]), indent=2))
    except BaseException as exc:
        av.write_json(output/"failure.json", dict(type=type(exc).__name__, message=str(exc), stages=stages))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    canonical = (ROOT/"artifacts.local/work/ba-boundary-separability-20260922").resolve()
    if canonical not in args.output.resolve().parents:
        parser.error("Output must be a new child under canonical boundary-separability artifacts")
    run(args.output)
