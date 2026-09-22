"""Public-initial sparse schedules and finite-cohort information diagnostics."""
from __future__ import annotations

import ast
from collections import defaultdict
import itertools
import math
from pathlib import Path

FEATURES = frozenset(("min", "max", "left_min", "right_min", "spread", "argmin", "near_count"))


def _initial(value):
    if not isinstance(value, (list, tuple)) or len(value) != 8 or any(type(x) is not int for x in value):
        raise ValueError("Expected eight integer public bins")
    return tuple(value)


def _schedule(value, budget):
    if (not isinstance(value, (list, tuple)) or len(value) != budget or budget < 2
            or any(type(i) is not int for i in value)
            or list(value) != sorted(set(value)) or value[0] != 0 or value[-1] != 12
            or any(i < 0 or i > 12 for i in value)):
        raise ValueError("Schedule must retain 0/12 and have budget unique increasing integer indices")
    return list(value)


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {"3", "4"}:
        raise ValueError("Policy requires exactly budgets 3 and 4")
    for budget in (3, 4):
        arm = policy[str(budget)]
        if not isinstance(arm, dict) or set(arm) != {"default", "rules"}:
            raise ValueError("Each budget requires default and rules")
        _schedule(arm["default"], budget)
        if not isinstance(arm["rules"], list) or len(arm["rules"]) > 3:
            raise ValueError("At most three ordered rules per budget")
        for rule in arm["rules"]:
            if not isinstance(rule, dict) or set(rule) != {"feature", "value", "schedule"}:
                raise ValueError("Rule requires feature, value and schedule")
            if not isinstance(rule["feature"], str) or rule["feature"] not in FEATURES:
                raise ValueError("Unsupported public feature")
            if (type(rule["value"]) not in (int, float)
                    or (type(rule["value"]) is float and not math.isfinite(rule["value"]))):
                raise ValueError("Rule threshold must be finite numeric")
            _schedule(rule["schedule"], budget)
    return policy


def load_policy(path):
    """Parse, never execute: optional docstring followed by POLICY literal only."""
    source = Path(path).read_text(encoding="utf-8")
    if len(source) > 32000:
        raise ValueError("Policy file exceeds bounded literal size")
    try:
        body = ast.parse(source).body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if (len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1
                or not isinstance(body[0].targets[0], ast.Name) or body[0].targets[0].id != "POLICY"):
            raise ValueError("Only a docstring and one POLICY literal assignment are allowed")
        policy = ast.literal_eval(body[0].value)
    except (SyntaxError, TypeError, RecursionError) as exc:
        raise ValueError("Invalid policy literal") from exc
    return validate_policy(policy)


def public_features(initial):
    bins = _initial(initial)
    return dict(min=min(bins), max=max(bins), left_min=min(bins[:4]), right_min=min(bins[4:]),
                spread=max(bins)-min(bins), argmin=bins.index(min(bins)),
                near_count=sum(x <= 30 for x in bins))


def select(policy, initial, budget):
    if budget not in (3, 4):
        raise ValueError("Budget must be 3 or 4")
    validate_policy(policy)
    features = public_features(initial)
    arm = policy[str(budget)]
    for rule in arm["rules"]:
        if features[rule["feature"]] <= rule["value"]:
            return list(rule["schedule"])
    return list(arm["default"])


def summarize(rows, schedules):
    """Evaluator-only cohort purity, not a classifier or continuous-domain proof."""
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(schedules):
        raise ValueError("Unique row IDs and matching schedule IDs required")
    buckets, pairs, signatures, initial = defaultdict(list), defaultdict(list), {}, {}
    counts = []
    for row in rows:
        ident = row["id"]
        if type(row["truth"]) is not bool or len(row["bins"]) != 13:
            raise ValueError("Expected boolean truth and thirteen views")
        bins = tuple(_initial(view) for view in row["bins"])
        schedule = _schedule(schedules[ident], len(schedules[ident]))
        signature = tuple((index, bins[index]) for index in schedule)
        signatures[ident], initial[ident] = signature, bins[0]
        buckets[signature].append(row)
        pairs[row["pair_id"]].append(row)
        counts.append(len(schedule))
    resolved = [r for group in buckets.values() if len({r["truth"] for r in group}) == 1 for r in group]
    separated, aliased, aliased_separated = [], [], []
    for pair_id, members in pairs.items():
        if len(members) > 2 or (len(members) == 2 and len({r["truth"] for r in members}) != 2):
            raise ValueError("Pairs must contain at most two opposite-label members")
        # Singleton pairs occur inside initial-signature groups during development fit.
        if len(members) != 2:
            continue
        a, b = (r["id"] for r in members)
        different = signatures[a] != signatures[b]
        if different:
            separated.append(pair_id)
        if initial[a] == initial[b]:
            aliased.append(pair_id)
            if different:
                aliased_separated.append(pair_id)
    resolved_ids = sorted(r["id"] for r in resolved)
    return dict(scenes=len(rows), pairs=sum(len(g) == 2 for g in pairs.values()),
                resolved_scenes=len(resolved), resolved_ids=resolved_ids,
                resolved_positive=sum(r["truth"] for r in resolved),
                resolved_negative=sum(not r["truth"] for r in resolved),
                unresolved_ids=sorted(set(ids)-set(resolved_ids)),
                mixed_classes=sum(len({r["truth"] for r in g}) > 1 for g in buckets.values()),
                total_classes=len(buckets), separated_pairs=len(separated), separated_pair_ids=sorted(separated),
                initially_aliased_pairs=len(aliased), initially_aliased_separated_pairs=len(aliased_separated),
                initially_aliased_pair_ids=sorted(aliased),
                cost=dict(path_length_m=.12, views_total=sum(counts), rays_total=8*sum(counts),
                          views_per_scene=(counts[0] if counts and len(set(counts)) == 1 else None),
                          rays_per_view=8))


def schedules_for(budget):
    if budget not in (3, 4):
        raise ValueError("Budget must be 3 or 4")
    return [(0, *middle, 12) for middle in itertools.combinations(range(1, 12), budget-2)]


def _objective(metrics):
    return metrics["resolved_scenes"], metrics["separated_pairs"]


def score(metrics_by_budget):
    """Bounded search score; exact baseline optimizers use the integer tuple."""
    metrics = [metrics_by_budget[str(b)] for b in (3, 4)]
    pairs = sum(m["separated_pairs"] for m in metrics)
    if pairs >= 1000:
        raise ValueError("Scalar score requires fewer than 1000 separated pairs across budgets")
    return sum(m["resolved_scenes"] for m in metrics) + pairs / 1000


def _best(rows, candidates):
    # Iterate lexicographically and retain first tied schedule.
    ranked = [(tuple(s), _objective(summarize(rows, {r["id"]: s for r in rows}))) for s in sorted(candidates)]
    return list(max(ranked, key=lambda item: item[1])[0])


def _key(initial):
    return ",".join(map(str, _initial(initial)))


def build_baselines(rows):
    """Fit only on supplied Development; never call with held outcome rows."""
    if not rows:
        raise ValueError("Development rows cannot be empty")
    result = {name: dict(kind=name, budgets={}) for name in ("uniform", "greedy", "exact_global", "exact_initial_lookup")}
    groups = defaultdict(list)
    for row in rows:
        groups[_key(row["bins"][0])].append(row)
    greedy = [0, 12]
    for budget in (3, 4):
        key = str(budget)
        global_schedule = _best(rows, schedules_for(budget))
        greedy = _best(rows, [sorted([*greedy, i]) for i in range(1, 12) if i not in greedy])
        result["uniform"]["budgets"][key] = dict(schedule=[0, 6, 12] if budget == 3 else [0, 4, 8, 12])
        result["greedy"]["budgets"][key] = dict(schedule=greedy)
        result["exact_global"]["budgets"][key] = dict(schedule=global_schedule)
        result["exact_initial_lookup"]["budgets"][key] = dict(
            fallback=global_schedule,
            lookup={initial: _best(group, schedules_for(budget)) for initial, group in sorted(groups.items())})
    return result


def apply_baseline(definition, rows, budget):
    if budget not in (3, 4):
        raise ValueError("Budget must be 3 or 4")
    arm = definition["budgets"][str(budget)]
    if definition["kind"] == "exact_initial_lookup":
        return {r["id"]: _schedule(arm["lookup"].get(_key(r["bins"][0]), arm["fallback"]), budget) for r in rows}
    return {r["id"]: _schedule(arm["schedule"], budget) for r in rows}
