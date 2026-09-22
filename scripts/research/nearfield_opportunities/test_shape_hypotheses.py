"""Focused geometry, ambiguity, and observation-boundary checks."""
import unittest

import numpy as np

import shape_hypotheses as sh


class ShapeHypothesesTests(unittest.TestCase):
    def test_raycast_range_and_parallel_axis(self):
        rays = np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
        values = sh.raycast([sh.box(0, 0.4, 1.2, 0.3)], rays)
        self.assertAlmostEqual(values[0], 1.2)
        self.assertTrue(np.isinf(values[1]))
        diagonal = np.array([[1, 0, 2]], dtype=float)
        diagonal /= np.linalg.norm(diagonal)
        distance = sh.raycast([sh.box(0.6, 0.4, 1.2, 0.3)], diagonal)[0]
        self.assertAlmostEqual(distance, np.sqrt(1.2 ** 2 + 0.6 ** 2))

    def test_full_extent_contact_not_centre(self):
        self.assertTrue(sh.intersects_corridor([sh.box(0.42, 0.24, 1, 0.1)]))
        self.assertFalse(sh.intersects_corridor([sh.box(0.43, 0.24, 1, 0.1)]))
        self.assertFalse(sh.intersects_corridor([sh.box(0, 0.24, 0.25, 0.04)]))
        self.assertTrue(sh.intersects_corridor([sh.box(0, 0.24, 0.25, 0.30)]))

    def test_hidden_depth_ambiguity_is_retained(self):
        bank = [{"shape_id": "thin", "solids": [sh.box(0, 0.24, 0.25, 0.04)]},
                {"shape_id": "thick", "solids": [sh.box(0, 0.24, 0.25, 0.30)]}]
        a = sh.observe("a", bank[0]["solids"])
        b = sh.observe("b", bank[1]["solids"])
        self.assertEqual({k: v for k, v in a.items() if k != "case_id"},
                         {k: v for k, v in b.items() if k != "case_id"})
        result = sh.predict(a, sh.compile_bank(bank))
        self.assertEqual(result["consensus"], "UNKNOWN")
        self.assertEqual(result["reason"], "DISAGREEING_SHAPES")
        self.assertEqual(result["single_best"], "OUT_MODEL_PRIOR")

    def test_observation_whitelist_rejects_truth(self):
        shape = {"shape_id": "one", "solids": [sh.box(0, 0.2, 1, 0.2)]}
        observation = sh.observe("a", shape["solids"])
        observation["truth_inside"] = True
        with self.assertRaises(ValueError):
            sh.predict(observation, sh.compile_bank([shape]))

    def test_metrics_unknown_is_not_wrong_out_or_removed_case(self):
        rows = [{"truth_inside": True, "policy": "UNKNOWN"},
                {"truth_inside": True, "policy": "OUT_MODEL_PRIOR"},
                {"truth_inside": False, "policy": "UNKNOWN"}]
        result = sh.metrics(rows, "policy")
        self.assertEqual(result["FN_alert"], 2)
        self.assertEqual(result["false_OUT"], 1)
        self.assertEqual(result["UNKNOWN"], 2)
        self.assertAlmostEqual(result["coverage"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
