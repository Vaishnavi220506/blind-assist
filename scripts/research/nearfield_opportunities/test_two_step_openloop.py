"""Public-only hand fixtures; never score the consumed cohort comparator."""
import unittest
from unittest.mock import patch

import active_view as av
import two_step_observation as two
import two_step_openloop as loop


class OpenLoopTests(unittest.TestCase):
    def test_initial_only_and_fixed_budget(self):
        forecasts = {p: ((0,), (0,)) for p in two.tree_poses()}
        forecasts[(.12, 0.)] = ((1,), (2,))
        with patch.object(av, "observe", side_effect=AssertionError("actual future")), \
             patch.object(av, "intersects_query", side_effect=AssertionError("source truth")):
            result = loop.choose((0,), forecasts, (True, False))
        self.assertEqual((result["first_action"], result["second_action"]), (0, 0))
        self.assertEqual(len(result["costs"]), 16)
        self.assertEqual(result["candidates"], [0, 1])
        for _, _, a, b in result["costs"]:
            self.assertAlmostEqual(sum(map(abs, two.STEPS[a]))+sum(map(abs, two.STEPS[b])), .12)

    def test_joint_signature_preserves_midpoint_evidence(self):
        forecasts = {p: ((0,), (0,)) for p in two.tree_poses()}
        forecasts[two.STEPS[0]] = ((1,), (2,))
        result = loop.choose((0,), forecasts, (True, False))
        self.assertEqual(result["costs"][0], (0, 0, 0, 0))
        self.assertEqual(result["costs"][4], (1, 2, 1, 0))

    def test_empty_support_and_ties_are_deterministic(self):
        forecasts = {p: ((0,), (0,)) for p in two.tree_poses()}
        for initial, expected in (((0,), [0, 1]), ((999,), [])):
            first = loop.choose(initial, forecasts, (True, False))
            second = loop.choose(list(initial), forecasts, (True, False))
            self.assertEqual(first, second)
            self.assertEqual(first["candidates"], expected)
            self.assertEqual((first["first_action"], first["second_action"]), (0, 0))

    def test_comparator_cannot_branch_after_midpoint(self):
        forecasts = {p: ((0,),)*4 for p in two.tree_poses()}
        forecasts[(.06, 0.)] = ((1,), (1,), (2,), (2,))
        forecasts[(.12, 0.)] = ((1,), (2,), (3,), (3,))
        forecasts[(.06, .06)] = ((1,), (1,), (2,), (3,))
        labels = (True, False, True, False)
        adaptive = two.first_choice((0,), forecasts, labels)
        openloop = loop.choose((0,), forecasts, labels)
        self.assertEqual(min(adaptive["costs"])[:2], (0, 0))
        self.assertEqual(min(openloop["costs"])[:2], (1, 2))


if __name__ == "__main__":
    unittest.main()
