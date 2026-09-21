import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import active_view as base
import active_view_positive as positive


class PositivePriorityTests(unittest.TestCase):
    @staticmethod
    def example():
        # +X identifies two positives; -X one negative and fewer opposite pairs.
        return base.ForecastBank(((1,),)*7,
            (((0,), (0,), (1,), (1,), (1,), (1,), (1,)),
             ((0,), (0,), (1,), (1,), (0,), (1,), (2,)),
             ((0,),)*7, ((0,),)*7), (True, True, True, True, False, False, False))

    def test_positive_mass_is_not_pair_split(self):
        bank = self.example()
        self.assertEqual(base.choose_action((1,), bank)["action_index"], 1)
        chosen = positive.select_positive_priority((1,), bank)
        self.assertEqual(chosen["action_index"], 0)
        self.assertEqual(chosen["costs"][0], [2, 5, 0])
        self.assertEqual(chosen["costs"][1], [4, 6, 1])

    def test_truth_future_and_case_identity_unavailable(self):
        self.assertEqual(list(inspect.signature(positive.select_positive_priority).parameters), ["initial", "bank"])
        with patch.object(base, "observe", side_effect=AssertionError("Future")), \
             patch.object(base, "raycast", side_effect=AssertionError("Future")), \
             patch.object(base, "intersects_query", side_effect=AssertionError("Truth")), \
             patch.object(positive, "source_scene", side_effect=AssertionError("True source")), \
             patch.object(positive, "load_true_source_after_choices", side_effect=AssertionError("True source")):
            self.assertEqual(positive.select_positive_priority((1,), self.example())["action_index"], 0)

    def test_ties_and_absent_prior_are_deterministic_unknown(self):
        bank = base.ForecastBank(((1,), (1,)), (((2,), (2,)),)*4, (True, False))
        self.assertEqual(positive.select_positive_priority((1,), bank)["action_index"], 0)
        missing = positive.select_positive_priority((9,), bank)
        self.assertTrue(missing["initial_no_match"])
        self.assertEqual(missing["action_index"], 0)
        self.assertEqual(base.decide((), bank), "UNKNOWN")

    def test_true_source_cannot_be_parsed_without_choice_seal(self):
        root = Path(__file__).resolve().parents[3]/"artifacts.local/work/ba-active-view-positive-20260921/test-receipts"/str(uuid4())
        root.mkdir(parents=True)
        source, choices, seal = root/"source.json", root/"choices.json", root/"choice-seal.json"
        source.write_text('[{"secret": true}]', encoding="utf-8")
        choices.write_text('[]', encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            positive.load_true_source_after_choices(source, seal)
        base.seal(seal, {"true_source_hash_only_before_choices": source, "choices": choices})
        self.assertEqual(positive.load_true_source_after_choices(source, seal), [{"secret": True}])
        choices.write_text('[1]', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Sealed input changed"):
            positive.load_true_source_after_choices(source, seal)

    def test_fn_includes_unknown_and_negative_decisions_are_separate(self):
        rows = [{"truth": t, "policies": {"p": {"decision": d, "remaining_count": 2}}}
                for t,d in ((True,"UNKNOWN"),(True,"NONINTERSECTING_HYPOTHESES"),
                            (True,"INTERSECTS"),(False,"NONINTERSECTING_HYPOTHESES"))]
        m = positive.metrics(rows, "p")
        self.assertEqual((m["tp"],m["fn_no_including_unknown"],m["explicit_false_negative"],m["unknown_positive"],m["correct_negative"]), (1,2,1,1,1))


if __name__ == "__main__":
    unittest.main()
