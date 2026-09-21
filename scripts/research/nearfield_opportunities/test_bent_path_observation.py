"""Hand fixtures only; no generation/scoring of the consumed180-scene cohort."""
import math
import unittest
from unittest.mock import patch

import active_view as av
import bent_path_observation as bent


class BentObservationTests(unittest.TestCase):
    def test_thirteen_views_equal_euclidean_motion_and_shared_bend_endpoint(self):
        paths = bent.paths()
        self.assertEqual(set(paths), {"straight_x", "x_then_z", "z_then_x"})
        for poses in paths.values():
            self.assertEqual(len(poses), 13)
            self.assertEqual(poses[0], (0., 0.))
            lengths = [math.dist(a, b) for a, b in zip(poses, poses[1:])]
            self.assertTrue(all(abs(d-.01) < 1e-12 for d in lengths))
            self.assertAlmostEqual(sum(lengths), .12)
        self.assertEqual(paths["x_then_z"][-1], paths["z_then_x"][-1])
        self.assertEqual(paths["x_then_z"][-1], (.06, .06))
        self.assertEqual(paths["x_then_z"][6], (.06, 0.))
        self.assertEqual(paths["z_then_x"][6], (0., .06))

    def test_radial_bias_precedes_quantization_in_both_directions(self):
        scene = av.Scene((av.Box(0., 1., .5),))
        for raw, condition, expected in ((1.049, "range_plus2mm", 11),
                                          (1.051, "range_minus2mm", 10)):
            with patch.object(av, "hit_distance", return_value=raw):
                value = bent.observe(scene, (0., 0.), condition)
            self.assertEqual(value["raw_ranges"], [raw]*8)
            self.assertEqual(value["bins"], [expected]*8)
            self.assertEqual(value["actual_camera"], value["camera"])
            self.assertEqual(value["biased_ranges"], [raw+value["range_bias_m"]]*8)

    def test_pose_bias_is_constant_global_and_keeps_actual_path_cost(self):
        scene = av.Scene(())
        for condition, delta in (("pose_plus1mm", .001), ("pose_minus1mm", -.001)):
            for poses in bent.paths().values():
                observed = [bent.observe(scene, p, condition) for p in poses]
                for nominal, row in zip(poses, observed):
                    self.assertEqual(row["camera"], list(nominal))
                    self.assertEqual(row["range_bias_m"], 0.)
                    self.assertEqual(row["raw_ranges"], row["biased_ranges"])
                    for actual, reported in zip(row["actual_camera"], nominal):
                        self.assertAlmostEqual(actual-reported, delta)
                self.assertAlmostEqual(sum(math.dist(a["actual_camera"], b["actual_camera"])
                    for a, b in zip(observed, observed[1:])), .12)

    def test_nominal_original_observer_identity_without_truth(self):
        scene = av.Scene((av.Box(.34, 1.4, .08),))
        with patch.object(av, "intersects_query", side_effect=AssertionError("No truth in observation")):
            for poses in bent.paths().values():
                for p in poses:
                    row = bent.observe(scene, p, "nominal")
                    self.assertEqual(row["bins"], list(av.observe(scene, p)))
                    self.assertEqual(row["camera"], row["actual_camera"])
        settings = bent.conditions()
        settings["nominal"]["range_bias_m"] = 999
        self.assertEqual(bent.conditions()["nominal"]["range_bias_m"], 0.)
        with self.assertRaises(ValueError):
            bent.observe(scene, (0., 0.), "unfrozen_condition")


if __name__ == "__main__":
    unittest.main()
