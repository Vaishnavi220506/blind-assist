"""Read-only finite-prior action opportunity audit of sealed active-view forecasts.

No renderer, active_view import, new future observation or strategy execution.
All counterfactual figures are enumeration over the old public model bank.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ACTION_NAMES = ("+X", "-X", "+Z", "-Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def action_metrics(indices, forecast, labels):
    groups = defaultdict(list)
    for i in indices:
        groups[tuple(forecast[i])].append(i)
    positive, negative, unknown_positive, unknown_negative, pairs, remaining = 0, 0, 0, 0, 0, 0
    decisive_positive_ids, decisive_negative_ids = [], []
    for members in groups.values():
        p = sum(labels[i] for i in members)
        n = len(members)-p
        pairs += p*n
        remaining += len(members)**2
        if p and n:
            unknown_positive += p
            unknown_negative += n
        elif p:
            positive += p
            decisive_positive_ids.extend(members)
        else:
            negative += n
            decisive_negative_ids.extend(members)
    assert positive+negative+unknown_positive+unknown_negative == len(indices)
    return {"positive_decisive": positive, "negative_decisive": negative,
            "unknown_positive": unknown_positive, "unknown_negative": unknown_negative,
            "unknown_total": unknown_positive+unknown_negative,
            "decisive_total": positive+negative, "opposite_pair_cost": pairs,
            "remaining_hypotheses_sum": remaining,
            "decisive_positive_bank_ids": sorted(decisive_positive_ids),
            "decisive_negative_bank_ids": sorted(decisive_negative_ids)}


def dominates(a, b):
    return (a[0] >= b[0] and a[1] >= b[1] and a != b)


def pareto(points):
    unique = set(points)
    return sorted(p for p in unique if not any(dominates(q, p) for q in unique))


def summed(class_rows, selector):
    fields = ("positive_decisive", "negative_decisive", "unknown_positive", "unknown_negative",
              "unknown_total", "decisive_total", "opposite_pair_cost", "remaining_hypotheses_sum")
    total = {field: 0 for field in fields}
    total["chosen_actions_by_class"] = {}
    for row in class_rows:
        action = selector(row)
        total["chosen_actions_by_class"][row["class_id"]] = ACTION_NAMES[action]
        for field in fields:
            total[field] += row["actions"][action][field]
    return total


def public_positive_objective(row):
    # Algebraic inspection only; no runner receives these actions.
    return min(range(4), key=lambda i: (row["actions"][i]["unknown_positive"],
                                        row["actions"][i]["unknown_total"],
                                        row["actions"][i]["opposite_pair_cost"], i))


def guarded_public_positive_objective(row):
    fixed = row["actions"][0]
    eligible = [i for i in range(4)
                if row["actions"][i]["positive_decisive"] >= fixed["positive_decisive"]
                and row["actions"][i]["decisive_total"] >= fixed["decisive_total"]]
    return min(eligible, key=lambda i: (row["actions"][i]["unknown_positive"],
                                       row["actions"][i]["unknown_total"],
                                       row["actions"][i]["opposite_pair_cost"], i))


def audit(source: Path, output: Path):
    if output.exists():
        raise FileExistsError("Refusing to overwrite audit evidence")
    names = ("public-model-forecasts.json", "sealed-choices.json", "evaluation.json", "summary.json",
             "initial-observations.json", "source.json", "hypothesis-bank.json", "predictions.json",
             "choice-seal.json", "prediction-seal.json", "completion-seal.json")
    before = {name: digest(source/name) for name in names}
    load = lambda name: json.loads((source/name).read_text(encoding="utf-8"))
    for name in ("choice-seal.json", "prediction-seal.json", "completion-seal.json"):
        for item in load(name)["files"].values():
            assert digest(Path(item["path"])) == item["sha256"], item["path"]
    data = load("public-model-forecasts.json")
    choices, evaluation = load("sealed-choices.json"), load("evaluation.json")
    labels, initial, forecasts = data["labels"], data["initial"], data["forecasts"]
    assert len(labels) == len(initial) == 621 and len(forecasts) == 4
    saved_choices = {r["id"]: r for r in choices}
    saved_evaluation = {r["id"]: r for r in evaluation}
    groups = defaultdict(list)
    for i, obs in enumerate(initial):
        groups[tuple(obs)].append(i)
    rows = []
    for obs, indices in sorted(groups.items()):
        if len({labels[i] for i in indices}) != 2:
            continue
        old_actions = {saved_choices[f"in_{i:04d}"]["action_index"] for i in indices}
        assert len(old_actions) == 1, "Same public observation had different old actions"
        old_action = old_actions.pop()
        metrics = [dict(action_index=j, action=ACTION_NAMES[j], **action_metrics(indices, f, labels))
                   for j, f in enumerate(forecasts)]
        expected_old = min(range(4), key=lambda j: (metrics[j]["opposite_pair_cost"],
                                                   metrics[j]["remaining_hypotheses_sum"], j))
        assert old_action == expected_old
        row = {"class_id": f"mixed_{len(rows):03d}", "initial_range_bins": list(obs),
               "bank_ids": indices, "case_count": len(indices),
               "positive_count": sum(labels[i] for i in indices),
               "negative_count": sum(not labels[i] for i in indices),
               "old_adaptive_action_index": old_action,
               "actions": metrics}
        old, fixed = metrics[old_action], metrics[0]
        row["maximum_positive_bound"] = max(a["positive_decisive"] for a in metrics)
        row["minimum_unknown_bound"] = min(a["unknown_total"] for a in metrics)
        row["joint_max_positive_min_unknown_actions"] = [a["action"] for a in metrics
              if a["positive_decisive"] == row["maximum_positive_bound"]
              and a["unknown_total"] == row["minimum_unknown_bound"]]
        row["count_pareto_frontier_positive_negative"] = pareto(
            (a["positive_decisive"], a["negative_decisive"]) for a in metrics)
        row["actions_weakly_dominate_fixed_positive_total"] = [a["action"] for a in metrics
              if a["positive_decisive"] >= fixed["positive_decisive"]
              and a["decisive_total"] >= fixed["decisive_total"]]
        row["actions_strictly_dominate_old_positive_negative"] = [a["action"] for a in metrics
              if dominates((a["positive_decisive"], a["negative_decisive"]),
                           (old["positive_decisive"], old["negative_decisive"]))]
        row["actions_preserve_all_fixed_positive_ids"] = [a["action"] for a in metrics
              if set(fixed["decisive_positive_bank_ids"]) <= set(a["decisive_positive_bank_ids"])]
        row["positive_mass_objective_action"] = ACTION_NAMES[public_positive_objective(row)]
        row["guarded_positive_mass_objective_action"] = ACTION_NAMES[guarded_public_positive_objective(row)]
        row["net_old_positive_change"] = old["positive_decisive"]-fixed["positive_decisive"]
        row["net_old_negative_change"] = old["negative_decisive"]-fixed["negative_decisive"]
        rows.append(row)
    assert sum(r["case_count"] for r in rows) == 273
    totals = {"fixed_actual_forecast_parity": summed(rows, lambda r: 0),
              "old_adaptive_actual_forecast_parity": summed(rows, lambda r: r["old_adaptive_action_index"]),
              "posthoc_public_positive_objective_forecast_only": summed(rows, public_positive_objective),
              "posthoc_guarded_positive_objective_forecast_only": summed(rows, guarded_public_positive_objective),
              "posthoc_min_unknown_oracle": summed(rows, lambda r: min(range(4), key=lambda i: (
                   r["actions"][i]["unknown_total"], -r["actions"][i]["positive_decisive"], i))),
              "posthoc_positive_oracle": summed(rows, lambda r: min(range(4), key=lambda i: (
                   -r["actions"][i]["positive_decisive"], r["actions"][i]["unknown_total"], i)))}
    saved_summary = load("summary.json")["in_prior/initially_ambiguous"]
    for arm, key in (("fixed", "fixed_actual_forecast_parity"), ("adaptive", "old_adaptive_actual_forecast_parity")):
        assert totals[key]["positive_decisive"] == saved_summary[arm]["tp"]
        assert totals[key]["negative_decisive"] == saved_summary[arm]["tn"]
        assert totals[key]["unknown_total"] == saved_summary[arm]["unknown"]
        # Actual evaluator truth is consulted only to authenticate prior/source identity.
        for row in rows:
            chosen = row["actions"][0 if arm == "fixed" else row["old_adaptive_action_index"]]
            for i in row["bank_ids"]:
                actual = saved_evaluation[f"in_{i:04d}"]
                assert actual["truth"] == labels[i]
                expected = "INTERSECTS" if i in chosen["decisive_positive_bank_ids"] else (
                    "NONINTERSECTING_HYPOTHESES" if i in chosen["decisive_negative_bank_ids"] else "UNKNOWN")
                assert actual["arms"][arm]["decision"] == expected
    # Exact aggregate frontier with a single action per indistinguishable initial class.
    frontier = {(0, 0)}
    for row in rows:
        frontier = set(pareto((p+a["positive_decisive"], n+a["negative_decisive"])
                              for p, n in frontier for a in row["actions"]))
    max_positive = sum(r["maximum_positive_bound"] for r in rows)
    min_unknown = sum(r["minimum_unknown_bound"] for r in rows)
    # Truth-conditioned per-case action bound is intentionally marked non-policy.
    pointwise_positive, pointwise_negative = set(), set()
    for row in rows:
        for a in row["actions"]:
            pointwise_positive.update(a["decisive_positive_bank_ids"])
            pointwise_negative.update(a["decisive_negative_bank_ids"])
    payload = {
        "kind": "POSTHOC_FINITE_PRIOR_OPPORTUNITY_AUDIT_NOT_A_NEW_POLICY_RESULT",
        "source_root": str(source), "source_sha256_before": before,
        "read_boundary": "Saved public forecasts only for counterfactual actions; saved choices/evaluation only authenticate old actual parity; no renderer, new strategy run, new future observations, or OOD counterfactual views",
        "initial_mixed_class_count": len(rows), "mixed_cases": 273, "mixed_positive": 113, "mixed_negative": 160,
        "per_class": rows, "totals": totals,
        "class_consistent_oracle_bounds": {"max_positive_decisive": max_positive,
                                          "minimum_unknown_total": min_unknown,
                                          "jointly_attainable": all(r["joint_max_positive_min_unknown_actions"] for r in rows),
                                          "aggregate_positive_negative_frontier": sorted(frontier)},
        "truth_conditioned_per_case_upper_bound_NOT_A_POLICY": {
            "positive_decisive": len(pointwise_positive), "negative_decisive": len(pointwise_negative),
            "unknown_total": 273-len(pointwise_positive)-len(pointwise_negative)},
        "old_adaptive_positive_loss_classes": [r["class_id"] for r in rows if r["net_old_positive_change"] < 0],
        "objective_different_action_classes": [r["class_id"] for r in rows if r["positive_mass_objective_action"] != ACTION_NAMES[r["old_adaptive_action_index"]]],
        "backend": "CPU scalar saved-set arithmetic / TASK_NOT_GPU_SUITABLE",
    }
    after = {name: digest(source/name) for name in names}
    assert after == before
    payload["source_sha256_after"] = after
    output.mkdir(parents=True)
    write(output/"opportunity-audit.json", payload)
    write(output/"completion.json", {"status": "PASS", "scientific_runs": 0,
         "source_files_unchanged": before == after, "auditor_sha256": digest(Path(__file__)),
         "audit_sha256": digest(output/"opportunity-audit.json")})
    print(json.dumps({"mixed_classes": len(rows), "bounds": payload["class_consistent_oracle_bounds"],
                      "pointwise_nonpolicy": payload["truth_conditioned_per_case_upper_bound_NOT_A_POLICY"],
                      "totals": {k: {x: v for x,v in t.items() if x != "chosen_actions_by_class"} for k,t in totals.items()},
                      "loss_classes": payload["old_adaptive_positive_loss_classes"],
                      "objective_differences": payload["objective_different_action_classes"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source, args.output)
