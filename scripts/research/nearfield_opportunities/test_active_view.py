"""Boundary tests, not repeated fits or main-source evaluation."""
import inspect
import math
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import active_view as av


class ActiveViewTests(unittest.TestCase):
    def test_selector_has_no_truth_or_future_parameter(self):
        self.assertEqual(list(inspect.signature(av.choose_action).parameters), ["initial", "bank"])
        bank = av.ForecastBank(((1,), (1,)), (((2,), (3,)), ((2,), (2,)),
                                            ((2,), (2,)), ((2,), (2,))), (False, True))
        before = av.choose_action((1,), bank)
        self.assertEqual(before["action_index"], 0)
        # During selection all true-scene observation/truth APIs are poisoned.
        # Only the already declared generic-bank forecasts may be consulted.
        with patch.object(av, "observe", side_effect=AssertionError("Future access")), \
             patch.object(av, "raycast", side_effect=AssertionError("Future access")), \
             patch.object(av, "intersects_query", side_effect=AssertionError("Truth access")), \
             patch.object(av, "make_cohort", side_effect=AssertionError("Source access")):
            self.assertEqual(av.choose_action((1,), bank), before)

    def test_selected_action_ignores_case_identity(self):
        bank = av.ForecastBank(((9,), (9,)), (((7,), (7,)), ((7,), (8,)),
                                            ((7,), (8,)), ((7,), (7,))), (False, True))
        a = av.choose_action((9,), bank)
        self.assertEqual(a["action_index"], 1)  # fixed tie order, no private scene selector
        self.assertEqual(a, av.choose_action((9,), bank))

    def test_extent_and_fixed_query(self):
        scene = av.Scene((av.Box(0.40, 1.5, 0.24),))
        self.assertTrue(av.intersects_query(scene))  # center outside, full extent intrudes
        outside = av.Scene((av.Box(0.40, 1.5, 0.10),))
        self.assertFalse(av.intersects_query(outside))
        # Moving to X=.12 must not recenter the reference query.
        shifted_wrongly = av.Scene((av.Box(0.28, 1.5, 0.10),))
        self.assertTrue(av.intersects_query(shifted_wrongly))
        bank = av.make_bank((outside,))
        initial, future = av.observe(outside, av.ORIGIN), av.observe(outside, av.ACTIONS[0])
        self.assertEqual(av.decide(av.posterior(initial, future, av.ACTIONS[0], bank), bank),
                         "NONINTERSECTING_HYPOTHESES")

    def test_radial_range_not_optical_z(self):
        wall = av.Scene(())
        ranges = av.observe(wall, av.ORIGIN)
        theta = av.ANGLES[0]
        expected = math.floor((4.20 / math.cos(theta))/av.RANGE_STEP+0.5)
        self.assertEqual(ranges[0], expected)
        self.assertGreater(ranges[0], round(4.20/av.RANGE_STEP))
        self.assertAlmostEqual(av.hit_distance(av.Box(0, 2, 2), av.ORIGIN, 0), 1.98)

    def test_unknown_and_no_match_never_negative(self):
        bank = av.ForecastBank(((1,), (1,)), (((2,), (2,)),)*4, (False, True))
        self.assertEqual(av.decide((), bank), "UNKNOWN")
        self.assertEqual(av.decide((0, 1), bank), "UNKNOWN")
        self.assertTrue(av.choose_action((99,), bank)["initial_no_match"])

    def test_future_measurement_requires_intact_choice_seal(self):
        root = Path(__file__).resolve().parents[3]/"artifacts.local/work/ba-active-view-20260921/test-receipts"/str(uuid4())
        root.mkdir(parents=True)
        choices, sealed = root/"choices.json", root/"seal.json"
        av.write_json(choices, [{"id": "s", "arms": {"adaptive": {"camera": [0.12, 0.0]}}}])
        with self.assertRaises(FileNotFoundError):
            av.observe_after_choice(av.Scene(()), "s", "adaptive", choices, sealed)
        av.seal(sealed, {"choices": choices})
        self.assertEqual(av.observe_after_choice(av.Scene(()), "s", "adaptive", choices, sealed),
                         av.observe(av.Scene(()), av.ACTIONS[0]))
        # Keep the intentionally mutated tiny files as boundary-test evidence.
        choices.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Sealed input changed"):
            av.observe_after_choice(av.Scene(()), "s", "adaptive", choices, sealed)


if __name__ == "__main__":
    unittest.main()
