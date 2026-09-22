"""Read-only independent audit; refuses access until the pilot result exists."""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]/"artifacts.local/work/ba-sparse-path-20260922/run-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_counts(rows, choices, budget):
    """No production imports: independently rebuild the equivalence relation."""
    identities = {r["id"] for r in rows}
    require(len(identities) == len(rows) and identities == set(choices), "ID accounting")
    classes, pairs, routes, signatures = defaultdict(list), defaultdict(list), {}, {}
    for row in rows:
        ident = row["id"]
        require(type(row["truth"]) is bool and len(row["bins"]) == 13, "Row shape/truth")
        require(all(len(v) == 8 and all(type(b) is int for b in v) for v in row["bins"]), "Eight integer bins")
        schedule = choices[ident]
        require(len(schedule) == budget and all(type(i) is int for i in schedule), "Budget/type")
        require(schedule == sorted(set(schedule)) and schedule[0] == 0 and schedule[-1] == 12,
                "Ordered endpoint-retaining schedule")
        initial = tuple(row["bins"][0])
        if initial in routes:
            require(routes[initial] == schedule, "Equal initial observation routed differently")
        routes[initial] = schedule
        signature = tuple((i, tuple(row["bins"][i])) for i in schedule)
        signatures[ident] = signature
        classes[signature].append(row)
        pairs[row["pair_id"]].append(row)
    pure = [row for group in classes.values() if len({r["truth"] for r in group}) == 1 for row in group]
    separated, aliased, gained = [], [], []
    for name, group in pairs.items():
        require(len(group) == 2 and {r["truth"] for r in group} == {True, False}, "Complete opposite-label pairs")
        a, b = group
        split = signatures[a["id"]] != signatures[b["id"]]
        if split:
            separated.append(name)
        if a["bins"][0] == b["bins"][0]:
            aliased.append(name)
            if split:
                gained.append(name)
    return dict(scenes=len(rows), pairs=len(pairs), resolved_scenes=len(pure),
                resolved_ids=sorted(r["id"] for r in pure),
                resolved_positive=sum(r["truth"] for r in pure),
                resolved_negative=sum(not r["truth"] for r in pure),
                unresolved_ids=sorted(identities-{r["id"] for r in pure}),
                total_classes=len(classes), mixed_classes=sum(len({r["truth"] for r in g}) > 1 for g in classes.values()),
                separated_pairs=len(separated), separated_pair_ids=sorted(separated),
                initially_aliased_pairs=len(aliased), initially_aliased_pair_ids=sorted(aliased),
                initially_aliased_separated_pairs=len(gained))


def compare(expected, actual, context):
    for field, value in actual.items():
        require(expected[field] == value, f"{context}: {field}")


def literal_policy(path):
    statements = ast.parse(path.read_text(encoding="utf-8")).body
    if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant) and isinstance(statements[0].value.value, str):
        statements = statements[1:]
    require(len(statements) == 1 and isinstance(statements[0], ast.Assign), "Literal policy only")
    assignment = statements[0]
    require(len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name)
            and assignment.targets[0].id == "POLICY", "Literal POLICY target")
    return ast.literal_eval(assignment.value)


def independently_choose(name, definition, rows, budget):
    result = {}
    for row in rows:
        bins = row["bins"][0]
        if name in ("direct", "sky"):
            features = {"min": min(bins), "max": max(bins), "left_min": min(bins[:4]),
                        "right_min": min(bins[4:]), "spread": max(bins)-min(bins),
                        "argmin": bins.index(min(bins)), "near_count": sum(b <= 30 for b in bins)}
            arm = definition[str(budget)]
            require(len(arm["rules"]) <= 3, "Compact rule budget")
            schedule = next((r["schedule"] for r in arm["rules"] if features[r["feature"]] <= r["value"]), arm["default"])
        else:
            arm = definition["budgets"][str(budget)]
            schedule = (arm["lookup"].get(",".join(map(str, bins)), arm["fallback"])
                        if definition["kind"] == "exact_initial_lookup" else arm["schedule"])
        result[row["id"]] = schedule
    return result


def audit_native(arm, selected_hash):
    folder = ROOT/arm
    terminal, ledger = read(folder/"terminal.json"), read(folder/"budget.json")
    require(terminal["status"] == "COMPLETED", "Native arm incomplete")
    require(terminal["budget"] == ledger, "Native terminal ledger mismatch")
    require(terminal["selected_sha256"] == selected_hash == digest(folder/"selected.py"), "Selected hash")
    require((folder/"selected.py").read_text(encoding="utf-8") == terminal["best"]["solution"], "Selected best solution")
    usage = ledger["usage"]
    require(usage["generation_calls"] <= 3 and usage["evaluator_attempts"] <= 4, "Native call/evaluation ceiling")
    require(usage["total_tokens"] == usage["input_tokens"]+usage["output_tokens"], "Token accounting")
    require(usage["total_tokens"] <= 120000 or ledger["stop_reason"] == "token_ceiling_crossed", "Token ceiling")
    calls = sorted(folder.glob("call_[0-9][0-9]"))
    require(len(calls) == usage["generation_calls"], "Dispatch count")
    generations = [e for e in ledger["events"] if e["kind"] == "generation"]
    evaluations = [e for e in ledger["events"] if e["kind"] == "evaluation"]
    require(len(generations) == len(calls) and len(evaluations) == usage["evaluator_attempts"], "Ledger events")
    for call in calls:
        dispatch = read(call/"dispatch.json")
        require(dispatch["prompt_sha256"] == digest(call/"prompt.txt"), "Prompt identity")
        require(read(call/"receipt.json")["process_released"] is True, "Unreleased call")
        events_path = call/"events.jsonl"
        require(events_path.exists(), "Missing model event evidence")
        for line in events_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("type") in ("item.started", "item.updated", "item.completed"):
                require(event.get("item", {}).get("type") in ("reasoning", "agent_message"), "Model used a tool")
    checkpoints = list((folder/"search/checkpoints").glob("checkpoint_*"))
    require(bool(checkpoints), "Missing native checkpoint")
    latest = max(checkpoints, key=lambda p: int(p.name.rsplit("_", 1)[1]))
    programs = {r["id"]: r for r in (read(p) for p in (latest/"programs").glob("*.json"))}
    seeds = [r for r in programs.values() if r["iteration_found"] == 0]
    require(len(seeds) == 1, "Expected exactly one native seed")
    seed = seeds[0]
    require(seed["solution"] == (ROOT/"seed.py").read_text(encoding="utf-8"), "Frozen parent seed")
    admitted = {e["program_id"] for e in evaluations}
    require(seed["id"] in admitted, "Seed lacks evaluation admission")
    children = [r for r in programs.values() if r["iteration_found"] > 0]
    for child in children:
        require(child["id"] in admitted, "Child lacks evaluation admission")
        parent_id = child["parent_id"]
        require(parent_id in programs and parent_id in admitted, "Unevaluated native parent")
        require(programs[parent_id]["iteration_found"] < child["iteration_found"], "Noncausal parent")
        contexts = child.get("other_context_ids") or []
        if arm == "direct":
            require(parent_id == seed["id"] and not contexts, "Direct candidate not independent seed proposal")
        else:
            require(len(contexts) <= 1, "Too many Sky contexts")
            earlier = [p for p in programs.values() if p["iteration_found"] < child["iteration_found"]]
            require(programs[parent_id]["metrics"]["combined_score"] == max(p["metrics"]["combined_score"] for p in earlier), "Sky parent not evaluated best")
            for ident in contexts:
                require(ident in admitted and programs[ident]["iteration_found"] < child["iteration_found"], "Noncausal context")
    require(terminal["best"]["id"] in programs, "Selected program missing checkpoint")
    return dict(calls=len(calls), evaluations=len(evaluations), checkpoint_children=len(children),
                provenance="ALL_GENERATED_CANDIDATES" if len(children) == len(calls) else "CHECKPOINTED_CHILDREN_ONLY")


def main():
    # No held source or observations are opened before a completed result exists.
    require((ROOT/"result.json").is_file(), "Result absent: held access forbidden")
    result, manifest = read(ROOT/"result.json"), read(ROOT/"manifest.json")
    for relative, expected in manifest["files"].items():
        require(digest(ROOT/relative) == expected, "Frozen input changed: "+relative)
    for path, expected in manifest["old_files"].items():
        require(digest(path) == expected, "Old evidence changed: "+path)
    for relative, expected in read(ROOT/"completion_seal.json").items():
        require(digest(ROOT/relative) == expected, "Completion seal changed: "+relative)
    selections = read(ROOT/"selection_seal.json")["selected"]
    require(selections == result["selection_seal"] == read(ROOT/"held_start.json")["selected"], "Held selection seal")
    native = {arm: audit_native(arm, selections[arm]) for arm in ("direct", "sky")}
    held, dev = read(ROOT/"held_observations.json"), read(ROOT/"development.json")
    require(len(held) == result["held_scenes"] == 192, "Held denominator")
    definitions = read(ROOT/"baselines.json")
    definitions.update({arm: literal_policy(ROOT/arm/"selected.py") for arm in ("direct", "sky")})
    checked = 0
    for split, rows in (("development", dev), ("held", held), ("joint", dev+held)):
        saved = result["results"][split]
        endpoint = independent_counts(rows, {r["id"]: [0, 12] for r in rows}, 2)
        full = independent_counts(rows, {r["id"]: list(range(13)) for r in rows}, 13)
        compare(saved["endpoint"], endpoint, split+" endpoint")
        compare(saved["full"], full, split+" full")
        for name, definition in definitions.items():
            for budget in (3, 4):
                choices = independently_choose(name, definition, rows, budget)
                if split == "held":
                    saved_choices = read(ROOT/f"held_choices_{name}_{budget}.json")
                    require(choices == saved_choices, "Saved held choices differ from frozen rule")
                    choices = saved_choices
                actual = independent_counts(rows, choices, budget)
                expected = saved["methods"][name][str(budget)]
                compare(expected, actual, f"{split} {name} {budget}")
                for field in ("resolved_ids", "separated_pair_ids"):
                    require(set(endpoint[field]) <= set(actual[field]) <= set(full[field]), "Endpoint/sparse/full subset")
                require(expected["lost_vs_full_ids"] == sorted(set(full["resolved_ids"])-set(actual["resolved_ids"])), "Lost-ID accounting")
                require(expected["cost"]["views_total"] == len(rows)*budget and expected["cost"]["rays_total"] == 8*len(rows)*budget, "Readout cost")
                if split == "joint":
                    require(expected["held_resolved_in_joint"] == sum(i.startswith("held_") for i in actual["resolved_ids"]), "Joint held purity")
                checked += 1
    print(json.dumps(dict(status="PASS",held_scenes=len(held),split_method_budgets_checked=checked,
                          native=native,old_files_unchanged=len(manifest["old_files"])), sort_keys=True))


if __name__ == "__main__":
    main()
