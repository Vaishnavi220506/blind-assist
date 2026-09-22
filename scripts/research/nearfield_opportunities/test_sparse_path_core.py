"""Focused invariants for public sparse schedules and information accounting."""
import itertools
from pathlib import Path
import tempfile
import unittest

import sparse_path_core as core


def policy():
    return {"3": {"default": [0, 6, 12], "rules": []},
            "4": {"default": [0, 4, 8, 12], "rules": []}}


def row(ident, truth, pair, changes=None):
    bins = [[40]*8 for _ in range(13)]
    for index, value in (changes or {}).items():
        bins[index] = [value]*8
    return dict(id=ident, truth=truth, pair_id=pair, bins=bins)


class SparsePathTests(unittest.TestCase):
    def test_literal_loader_and_reject_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"candidate.py"
            path.write_text('"""safe"""\nPOLICY = '+repr(policy()), encoding="utf-8")
            self.assertEqual(core.load_policy(path), policy())
            for source in ("import os\nPOLICY = {}", "POLICY = dict()", "POLICY = {}\nprint('unsafe')",
                           "x = POLICY = {}", "POLICY: dict = {}", "POLICY = __import__('os').environ"):
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(ValueError):
                    core.load_policy(path)

    def test_schema_budget_and_ordered_features(self):
        p = policy()
        p["3"]["rules"] = [dict(feature="min", value=20, schedule=[0, 4, 12]),
                           dict(feature="near_count", value=8, schedule=[0, 8, 12])]
        self.assertEqual(core.select(p, [20]*8, 3), [0, 4, 12])
        self.assertEqual(core.select(p, [40]*8, 3), [0, 8, 12])
        self.assertEqual(core.public_features([40, 20, 35, 50, 10, 60, 70, 80]),
                         dict(min=10, max=80, left_min=20, right_min=10, spread=70, argmin=4, near_count=2))
        for schedule in ([0, 6, 6], [0, 12, 6], [0, 4, 8, 12], [0, True, 12], [0, 13, 12]):
            bad = policy()
            bad["3"]["default"] = schedule
            with self.assertRaises(ValueError):
                core.validate_policy(bad)
        with self.assertRaises(ValueError):
            core.select(policy(), [True]*8, 3)
        with self.assertRaises(ValueError):
            core.select(policy(), [40]*8, 2)
        for feature, value in (([], 20), ("truth", 20), ("min", float("inf")), ("min", True)):
            bad = policy()
            bad["3"]["rules"] = [dict(feature=feature, value=value, schedule=[0, 6, 12])]
            with self.assertRaises(ValueError):
                core.validate_policy(bad)
        bad = policy()
        bad["3"]["rules"] = [dict(feature="min", value=20, schedule=[0, 6, 12])]*4
        with self.assertRaises(ValueError):
            core.validate_policy(bad)

    def test_pair_separation_is_not_purity(self):
        rows = [row("a", True, "p", {6: 10}), row("b", False, "p", {6: 20}),
                row("c", False, "q", {6: 10}), row("d", True, "q", {6: 20})]
        result = core.summarize(rows, {r["id"]: [0, 6, 12] for r in rows})
        self.assertEqual(result["separated_pairs"], 2)
        self.assertEqual(result["resolved_scenes"], 0)
        self.assertEqual(result["initially_aliased_pairs"], 2)

    def test_subset_monotonic_and_cost(self):
        rows = [row("a", True, "p", {4: 10}), row("b", False, "p"),
                row("c", True, "q", {8: 20}), row("d", False, "q")]
        previous = set()
        for schedule in ([0, 12], [0, 4, 12], [0, 4, 8, 12], list(range(13))):
            result = core.summarize(rows, {r["id"]: schedule for r in rows})
            self.assertLessEqual(previous, set(result["resolved_ids"]))
            previous = set(result["resolved_ids"])
            self.assertEqual(result["cost"]["rays_total"], 8*len(schedule)*len(rows))

    def test_exhaustive_and_lookup_match_brute(self):
        rows = [row("a", True, "p", {4: 10}), row("b", False, "p"),
                row("c", True, "q", {0: 30, 8: 20}), row("d", False, "q", {0: 30})]
        definitions = core.build_baselines(rows)
        for budget in (3, 4):
            candidates = [(0, *middle, 12) for middle in itertools.combinations(range(1, 12), budget-2)]
            self.assertEqual(len(candidates), 11 if budget == 3 else 55)
            def value(schedules):
                m = core.summarize(rows, schedules)
                return m["resolved_scenes"], m["separated_pairs"]
            brute = max(value({r["id"]: s for r in rows}) for s in candidates)
            self.assertEqual(value(core.apply_baseline(definitions["exact_global"], rows, budget)), brute)
            chosen = core.apply_baseline(definitions["exact_initial_lookup"], rows, budget)
            self.assertEqual(chosen["a"], chosen["b"])
            self.assertEqual(chosen["c"], chosen["d"])
            lookup_brute = max(value({"a": s, "b": s, "c": t, "d": t}) for s in candidates for t in candidates)
            self.assertEqual(value(chosen), lookup_brute)
            unknown = [row("new", True, "new", {0: 99})]
            fallback = core.apply_baseline(definitions["exact_initial_lookup"], unknown, budget)["new"]
            self.assertEqual(fallback, definitions["exact_global"]["budgets"][str(budget)]["schedule"])

    def test_pose_is_part_of_signature(self):
        rows = [row("a", True, "p"), row("b", False, "p")]
        result = core.summarize(rows, {"a": [0, 4, 12], "b": [0, 6, 12]})
        self.assertEqual(result["resolved_scenes"], 2)
        # Such routing is not a valid equal-initial policy; select enforces that boundary.
        self.assertEqual(core.select(policy(), rows[0]["bins"][0], 3), core.select(policy(), rows[1]["bins"][0], 3))


if __name__ == "__main__":
    unittest.main()
